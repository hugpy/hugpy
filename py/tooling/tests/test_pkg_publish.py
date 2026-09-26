"""pkg_publish: the GitHub commit+push step, exercised with a real temp git repo
and a bare temp remote (no network, no DB). Covers: config off => no-op; the
secret gate blocks a push and unstages; a diverged remote is skipped and never
force-pushed; the token is scrubbed from recorded error text."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pkg_publish as P  # noqa: E402

IDENT = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]


def g(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *IDENT, *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def make_repo(tmp: Path, name: str, branch: str = "main") -> Path:
    repo = tmp / name
    repo.mkdir()
    g(repo, "init", "-q", "-b", branch)
    (repo / "README.md").write_text("hello\n")
    g(repo, "add", "-A")
    g(repo, "commit", "-q", "-m", "init")
    return repo


def make_bare(tmp: Path, name: str) -> Path:
    bare = tmp / name
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    return bare


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    """Capture pkg_src.publishes rows instead of writing them; no DB in these tests."""
    rows = []
    monkeypatch.setattr(P, "record", lambda conn, row: (rows.append(dict(row)), len(rows))[1])
    return rows


def _repo_cfg(repo: Path, url: Path | None, branch: str = "main") -> dict:
    return {"path": str(repo), "url": str(url) if url else None, "branch": branch,
            "include_untracked": True, "extra_gitignore": []}


# --------------------------------------------------------------------------- off

def test_config_off_is_a_noop(tmp_path, _no_db):
    cfg = tmp_path / "pkg_publish.toml"
    cfg.write_text('enabled = false\nmode = "push"\n')
    out = P.publish_after_promotion(None, 7, config_path=cfg)
    assert out == {"enabled": False}
    assert _no_db == []                                   # nothing recorded, no git touched


def test_absent_config_is_disabled(tmp_path):
    out = P.publish(None, promotion_id=1, config_path=tmp_path / "nope.toml")
    assert out["enabled"] is False and out["repos"] == []


# ------------------------------------------------------------------ secret gate

def test_secret_gate_blocks_push_and_unstages(tmp_path, _no_db):
    repo = make_repo(tmp_path, "src")
    bare = make_bare(tmp_path, "remote.git")
    g(repo, "remote", "add", "origin", str(bare))
    g(repo, "push", "-q", "origin", "main")
    remote_before = g(repo, "rev-parse", "origin/main")

    (repo / "leak.py").write_text('GITHUB_TOKEN = "ghp_' + "a" * 36 + '"\n')
    row = P.publish_repo(None, _repo_cfg(repo, bare), config="c", release="0.2.1.post1",
                         promotion_id=5, mode="push", token="SECRET", author={},
                         message="c -> 0.2.1.post1")
    assert row["status"] == "blocked_secret" and row["pushed"] is False
    types = {h["type"] for h in row["secret_scan"]["hits"]}
    assert "github_token" in types or "assigned_secret" in types
    assert all("ghp_" not in (h.get("type") or "") for h in row["secret_scan"]["hits"])  # no value
    # nothing was staged after the block, and the remote never moved
    assert subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"]).returncode == 0
    g(repo, "fetch", "-q", "origin")
    assert g(repo, "rev-parse", "origin/main") == remote_before


# --------------------------------------------------------------------- diverged

def test_diverged_remote_is_skipped_never_forced(tmp_path, _no_db):
    repo = make_repo(tmp_path, "src")
    bare = make_bare(tmp_path, "remote.git")
    g(repo, "remote", "add", "origin", str(bare))
    g(repo, "push", "-q", "origin", "main")

    # a second clone advances the remote past local
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", "-b", "main", str(bare), str(other)], check=True)
    (other / "extra.txt").write_text("remote-only\n")
    g(other, "add", "-A"); g(other, "commit", "-q", "-m", "remote commit")
    g(other, "push", "-q", "origin", "main")
    remote_head = g(other, "rev-parse", "HEAD")

    (repo / "local.txt").write_text("local change\n")     # local has its own work
    row = P.publish_repo(None, _repo_cfg(repo, bare), config="c", release="0.2.1.post2",
                         promotion_id=6, mode="push", token="SECRET", author={},
                         message="c -> 0.2.1.post2")
    assert row["status"] == "diverged" and row["pushed"] is False
    g(repo, "fetch", "-q", "origin")
    assert g(repo, "rev-parse", "origin/main") == remote_head  # remote untouched (never forced)


# ------------------------------------------------------------------ clean / dry-run

def test_dry_run_commits_locally_but_never_pushes(tmp_path, _no_db):
    repo = make_repo(tmp_path, "src")
    bare = make_bare(tmp_path, "remote.git")
    g(repo, "remote", "add", "origin", str(bare))
    g(repo, "push", "-q", "origin", "main")
    remote_before = g(repo, "rev-parse", "origin/main")

    (repo / "feature.py").write_text("x = 1\n")
    row = P.publish_repo(None, _repo_cfg(repo, bare), config="c", release="0.2.1.post3",
                         promotion_id=8, mode="dry-run", token=None,
                         author={"name": "putkoff", "email": "j@x"},
                         message="c -> 0.2.1.post3\n\nPublished-By: pkg_src promotion 8")
    assert row["status"] == "staged" and row["pushed"] is False and row["commit_sha"]
    g(repo, "fetch", "-q", "origin")
    assert g(repo, "rev-parse", "origin/main") == remote_before   # nothing pushed


def test_nothing_to_commit_is_clean(tmp_path, _no_db):
    repo = make_repo(tmp_path, "src")
    row = P.publish_repo(None, _repo_cfg(repo, None), config="c", release=None,
                         promotion_id=None, mode="dry-run", token=None, author={},
                         message="c -> (unreleased)")
    assert row["status"] == "clean" and row["commit_sha"] is None


# ------------------------------------------------------------------ token scrub

def test_token_is_scrubbed_from_error_text(tmp_path, _no_db, monkeypatch):
    repo = make_repo(tmp_path, "src")
    bare = make_bare(tmp_path, "remote.git")
    g(repo, "remote", "add", "origin", str(bare))
    g(repo, "push", "-q", "origin", "main")
    token = "ghp_secretvalue000000000000000000000"

    real_git = P.git

    def fake_git(rp, *args, **kw):
        if args and args[0] == "push":                    # a push that leaks the token
            return (128, "", f"fatal: authentication failed for {token}")
        return real_git(rp, *args, **kw)

    monkeypatch.setattr(P, "git", fake_git)
    (repo / "feature.py").write_text("y = 2\n")
    row = P.publish_repo(None, _repo_cfg(repo, bare), config="c", release="0.2.1.post4",
                         promotion_id=9, mode="push", token=token, author={},
                         message="c -> 0.2.1.post4")
    assert row["status"] == "committed" and row["pushed"] is False
    assert token not in (row["error"] or "") and "***" in (row["error"] or "")


def test_scrub_replaces_every_secret():
    assert P.scrub("a TOK b TOK c", ["TOK"]) == "a *** b *** c"
    assert P.scrub(None, ["TOK"]) is None
    assert P.scrub("clean", []) == "clean"
