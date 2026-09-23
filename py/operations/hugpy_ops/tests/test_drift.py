"""hugpy-drift-check: every section against fakes that are as real as they
can be — git is a real temporary repository (with a real clone as origin),
HTTP is the module's ``fetch_json`` monkeypatched, installed metadata is the
module's ``distribution_record`` / ``workspace_distributions`` monkeypatched.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import urllib.error

import pytest

from hugpy_ops import drift

pytestmark = pytest.mark.skipif(not drift.shutil.which("git"), reason="git not on PATH")

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "drift test", "GIT_AUTHOR_EMAIL": "drift@test",
    "GIT_COMMITTER_NAME": "drift test", "GIT_COMMITTER_EMAIL": "drift@test",
    "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
}


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True,
                          text=True, env=_GIT_ENV).stdout.strip()


def _commit(path, name, text):
    (path / name).write_text(text)
    _git(path, "add", name)
    _git(path, "commit", "-q", "-m", f"add {name}")
    return _git(path, "rev-parse", "HEAD")


@pytest.fixture
def repos(tmp_path):
    """``upstream`` (one commit, tag v1.0.0) and ``work`` (its clone, so
    ``origin/main`` exists and tracks it)."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _commit(upstream, "README", "one\n")
    _git(upstream, "tag", "v1.0.0")
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(upstream), str(work)], check=True, env=_GIT_ENV)
    return upstream, work


def _rows(report, section):
    return {r.subject: r for r in report.rows if r.section == section}


# --------------------------------------------------------------------------- #
# A — checkout
# --------------------------------------------------------------------------- #
def test_clean_clone_is_in_sync(repos):
    _, work = repos
    report = drift.run("A", workspace=str(work))
    a = _rows(report, "A")
    assert a["tree"].status == drift.OK
    assert a["origin/main"].status == drift.OK
    assert report.exit_code == 0 and report.ok


def test_dirty_tree_is_drift_unless_allowed(repos):
    _, work = repos
    (work / "README").write_text("edited\n")
    report = drift.run("A", workspace=str(work))
    tree = _rows(report, "A")["tree"]
    assert tree.status == drift.DRIFT and "README" in tree.detail
    assert report.exit_code == 1
    allowed = drift.run("A", workspace=str(work), allow_dirty=True)
    assert _rows(allowed, "A")["tree"].status == drift.INFO
    assert allowed.exit_code == 0


def test_untracked_files_are_not_drift(repos):
    _, work = repos
    (work / "scratch.txt").write_text("not tracked\n")
    assert _rows(drift.run("A", workspace=str(work)), "A")["tree"].status == drift.OK


def test_ahead_of_origin_is_drift(repos):
    _, work = repos
    _commit(work, "local.txt", "unpushed\n")
    row = _rows(drift.run("A", workspace=str(work)), "A")["origin/main"]
    assert row.status == drift.DRIFT and "1 unpushed commit" in row.detail


def test_behind_origin_is_drift_after_fetch(repos):
    upstream, work = repos
    _commit(upstream, "remote.txt", "new upstream\n")
    stale = _rows(drift.run("A", workspace=str(work)), "A")["origin/main"]
    assert stale.status == drift.OK and "vs last fetch" in stale.detail
    fresh = _rows(drift.run("A", workspace=str(work), fetch=True), "A")["origin/main"]
    assert fresh.status == drift.DRIFT and "behind origin" in fresh.detail


def test_detached_head_and_missing_origin_are_info(repos, tmp_path):
    _, work = repos
    _git(work, "checkout", "-q", "--detach")
    assert _rows(drift.run("A", workspace=str(work)), "A")["origin"].status == drift.INFO
    lone = tmp_path / "lone"
    lone.mkdir()
    _git(lone, "init", "-q", "-b", "solo")
    _commit(lone, "f", "x\n")
    row = _rows(drift.run("A", workspace=str(lone)), "A")["origin/solo"]
    assert row.status == drift.INFO and "no origin/solo ref" in row.detail


def test_no_checkout_is_info_not_error(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    rows = drift.check_checkout(str(plain))
    assert rows[0].status == drift.ERROR  # a directory that is not a repo
    assert drift.check_checkout(None)[0].status == drift.INFO


# --------------------------------------------------------------------------- #
# B — installed
# --------------------------------------------------------------------------- #
def _fake_records(monkeypatch, records):
    monkeypatch.setattr(drift, "workspace_distributions", lambda: tuple(records))
    monkeypatch.setattr(drift, "distribution_record", lambda name: records.get(name))


def test_lockstep_ok_for_editable_sources_at_workspace_head(repos, monkeypatch):
    _, work = repos
    head = _git(work, "rev-parse", "HEAD")
    v = f"1.0.1.dev1+g{head[:9]}"
    _fake_records(monkeypatch, {
        "hugpy-platform": {"name": "hugpy-platform", "version": v, "editable": True, "source": str(work / "py/a")},
        "hugpy-ops": {"name": "hugpy-ops", "version": v, "editable": True, "source": str(work / "py/b")},
    })
    (work / "py/a").mkdir(parents=True)
    (work / "py/b").mkdir(parents=True)
    b = _rows(drift.run("B", workspace=str(work)), "B")
    assert b["lockstep"].status == drift.OK and "2 distribution(s)" in b["lockstep"].detail
    assert b["hugpy-platform"].status == drift.OK and b["hugpy-ops"].status == drift.OK


def test_mixed_versions_are_drift(repos, monkeypatch):
    _, work = repos
    _fake_records(monkeypatch, {
        "hugpy-platform": {"name": "hugpy-platform", "version": "1.0.0", "editable": False, "source": None},
        "hugpy-ops": {"name": "hugpy-ops", "version": "1.0.1", "editable": False, "source": None},
        "hugpy": None,
    })
    b = _rows(drift.run("B", workspace=str(work)), "B")
    assert b["lockstep"].status == drift.DRIFT and "mixed" in b["lockstep"].detail
    assert b["hugpy"].status == drift.INFO and "not installed" in b["hugpy"].detail


def test_editable_source_at_other_head_is_drift(repos, monkeypatch, tmp_path):
    _, work = repos
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", str(work), str(other)], check=True, env=_GIT_ENV)
    _commit(other, "diverge.txt", "x\n")
    head = _git(other, "rev-parse", "HEAD")
    _fake_records(monkeypatch, {
        "hugpy-ops": {"name": "hugpy-ops", "version": f"1.0.1.dev2+g{head[:9]}", "editable": True,
                      "source": str(other)},
    })
    row = _rows(drift.run("B", workspace=str(work)), "B")["hugpy-ops"]
    assert row.status == drift.DRIFT and "!= workspace HEAD" in row.detail


def test_unknown_build_identity_is_drift_but_untagged_sha_is_not(repos, monkeypatch):
    _, work = repos
    head = _git(work, "rev-parse", "HEAD")
    _fake_records(monkeypatch, {
        "hugpy-platform": {"name": "hugpy-platform", "version": "0.0.0+unknown", "editable": False, "source": None},
        "hugpy-ops": {"name": "hugpy-ops", "version": f"0.0.1.dev11+unknown.g{head[:9]}",
                      "editable": True, "source": str(work)},
    })
    b = _rows(drift.run("B", workspace=str(work)), "B")
    assert b["hugpy-platform"].status == drift.DRIFT
    assert "no git identity at build time" in b["hugpy-platform"].detail
    assert b["hugpy-ops"].status == drift.OK


def test_wheel_built_from_another_commit_is_drift(repos, monkeypatch):
    _, work = repos
    _fake_records(monkeypatch, {
        "hugpy-ops": {"name": "hugpy-ops", "version": "1.0.1.dev3+gdeadbeef1", "editable": False, "source": None},
        "hugpy": {"name": "hugpy", "version": "1.0.1.dev3+gdeadbeef1", "editable": False, "source": None},
    })
    b = _rows(drift.run("B", workspace=str(work)), "B")
    assert b["hugpy-ops"].status == drift.DRIFT and "built from gdeadbee" in b["hugpy-ops"].detail
    assert b["lockstep"].status == drift.OK


def test_tagged_release_without_sha_is_ok(monkeypatch):
    _fake_records(monkeypatch, {
        "hugpy-ops": {"name": "hugpy-ops", "version": "1.2.0", "editable": False, "source": None},
    })
    rows = drift.check_installed(None)
    assert [r.status for r in rows] == [drift.OK, drift.OK]


def test_buildinfo_absent_degrades_to_metadata(monkeypatch):
    monkeypatch.setattr(drift, "_buildinfo", lambda: None)
    assert drift.workspace_distributions() == drift.FALLBACK_DISTRIBUTIONS
    assert drift.local_build_identity() is None
    assert drift.distribution_record("definitely-not-a-distribution") is None
    rec = drift.distribution_record("hugpy-ops")
    if rec is not None:  # installed in this interpreter
        assert set(rec) == {"name", "version", "editable", "source"}


def test_buildinfo_present_is_used(monkeypatch):
    import types

    fake = types.ModuleType("hugpy_platform.buildinfo")
    fake.WORKSPACE_DISTRIBUTIONS = ("hugpy-x",)
    fake.distribution_record = lambda name: {"name": name, "version": "9.9.9", "editable": False, "source": None}
    fake.build_identity = lambda: {"version": "9.9.9", "sha": "abc", "dirty": False,
                                   "editable": False, "source": None, "distribution": "hugpy-x"}
    monkeypatch.setattr(drift, "_buildinfo", lambda: fake)
    assert drift.workspace_distributions() == ("hugpy-x",)
    assert drift.distribution_record("hugpy-x")["version"] == "9.9.9"
    assert drift.local_build_identity()["sha"] == "abc"


# --------------------------------------------------------------------------- #
# C — fleet
# --------------------------------------------------------------------------- #
CENTRAL = "http://central.test:7002"
CBUILD = {"version": "1.0.1.dev5+gaaaaaaa", "sha": "aaaaaaa1234", "dirty": False}


def _fake_fleet(monkeypatch, health, workers):
    def fetch(url, token=None, timeout=None):
        if url == CENTRAL + "/api/health":
            return health
        if url == CENTRAL + "/api/llm/workers":
            return workers
        raise AssertionError(url)
    monkeypatch.setattr(drift, "fetch_json", fetch)
    monkeypatch.setattr(drift, "local_build_identity", lambda: dict(CBUILD))


def _worker(name, **kw):
    return {"name": name, "status": "online", "unreachable": False, **kw}


def test_worker_monolith_version_is_drift(monkeypatch):
    _fake_fleet(monkeypatch, {"ok": True, "build": CBUILD}, [
        _worker("aeb", pkg_version="0.1.266", version_ok=False, required_pkg_version="0.1.0",
                environment_digest={"digest": "x"}),
        _worker("nobuild", pkg_version=None, version_ok=None),
    ])
    c = _rows(drift.run("C", central=CENTRAL), "C")
    assert c["central"].status == drift.OK
    assert c["worker aeb"].status == drift.DRIFT and "monolith" in c["worker aeb"].detail
    assert c["worker nobuild"].status == drift.DRIFT


def test_offline_worker_is_info_not_drift(monkeypatch):
    """A powered-off box (op, 2026-09-23) must not keep the verdict red: its
    stale version is reported, not compared. Online monolith workers still drift."""
    _fake_fleet(monkeypatch, {"ok": True, "build": CBUILD}, [
        _worker("op", status="offline", pkg_version="0.1.266", version_ok=False,
                required_pkg_version="1.0.1"),
        _worker("aeb", pkg_version="0.1.266", version_ok=False, required_pkg_version="1.0.1"),
    ])
    report = drift.run("C", central=CENTRAL)
    c = _rows(report, "C")
    assert c["worker op"].status == drift.INFO and "offline" in c["worker op"].detail \
        and "0.1.266" in c["worker op"].detail
    assert c["worker aeb"].status == drift.DRIFT
    # with only the offline box drifting, the fleet is in sync
    _fake_fleet(monkeypatch, {"ok": True, "build": CBUILD}, [
        _worker("op", status="offline", pkg_version="0.1.266"),
        _worker("same", environment_digest={"build": {"version": CBUILD["version"], "sha": "aaaaaaa"}}),
    ])
    assert drift.run("C", central=CENTRAL).exit_code == 0


def test_worker_sha_mismatch_is_drift_and_match_is_ok(monkeypatch):
    _fake_fleet(monkeypatch, {"ok": True, "build": CBUILD}, [
        _worker("same", environment_digest={"build": {"version": CBUILD["version"], "sha": "aaaaaaa"}}),
        _worker("other", environment_digest={"build": {"version": CBUILD["version"], "sha": "bbbbbbb"}}),
        _worker("oldver", environment_digest={"build": {"version": "1.0.0", "sha": "aaaaaaa"}}),
    ])
    c = _rows(drift.run("C", central=CENTRAL), "C")
    assert c["worker same"].status == drift.OK
    assert c["worker other"].status == drift.DRIFT and "!= central" in c["worker other"].detail
    assert c["worker oldver"].status == drift.DRIFT


def test_worker_without_build_compares_pkg_version(monkeypatch):
    _fake_fleet(monkeypatch, {"ok": True, "build": CBUILD}, [
        _worker("pinned-ok", pkg_version="1.0.1", version_ok=True, required_pkg_version="1.0.1"),
        _worker("pinned-bad", pkg_version="1.0.0", version_ok=False, required_pkg_version="1.0.1"),
        _worker("unpinned", pkg_version="1.0.1", version_ok=None, required_pkg_version=None),
    ])
    c = _rows(drift.run("C", central=CENTRAL), "C")
    assert c["worker pinned-ok"].status == drift.OK
    assert c["worker pinned-bad"].status == drift.DRIFT
    assert c["worker unpinned"].status == drift.INFO and "pins nothing" in c["worker unpinned"].detail


def test_central_without_build_is_error_and_local_mismatch_is_drift(monkeypatch):
    _fake_fleet(monkeypatch, {"ok": True}, [])
    c = _rows(drift.run("C", central=CENTRAL), "C")
    assert c["central"].status == drift.ERROR and "predates buildinfo" in c["central"].detail
    assert c["workers"].status == drift.INFO
    _fake_fleet(monkeypatch, {"ok": True, "build": {**CBUILD, "sha": "cccccc"}}, [])
    assert _rows(drift.run("C", central=CENTRAL), "C")["central"].status == drift.DRIFT


def test_central_unreachable_is_error_rows_not_a_crash(monkeypatch):
    def fetch(url, token=None, timeout=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(drift, "fetch_json", fetch)
    report = drift.run("C", central=CENTRAL)
    assert [r.status for r in report.rows] == [drift.ERROR]
    assert "unreachable" in report.rows[0].detail
    assert report.exit_code == 2 and not report.ok


def test_central_calls_use_the_central_timeout(monkeypatch):
    seen = []

    def fetch(url, token=None, timeout=None):
        seen.append((url.rsplit("/api", 1)[1], timeout))
        return {"ok": True} if url.endswith("/api/health") else []
    monkeypatch.setattr(drift, "fetch_json", fetch)
    drift.run("C", central=CENTRAL)
    assert seen == [("/health", drift.CENTRAL_TIMEOUT), ("/llm/workers", drift.CENTRAL_TIMEOUT)]
    seen.clear()
    drift.run("C", central=CENTRAL, timeout=123.0)
    assert [t for _, t in seen] == [123.0, 123.0]
    assert drift.CENTRAL_TIMEOUT > drift.HTTP_TIMEOUT   # the fleet is slow right after a restart


def test_timer_records_a_non_default_timeout(tmp_path, monkeypatch):
    args = drift.build_parser().parse_args(["--install-timer", "--dry-run", "--timeout", "90",
                                            "--central", CENTRAL])
    monkeypatch.setattr(drift, "console_script_path", lambda: "/x/hugpy-drift-check")
    import io, contextlib
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert drift.install_timer(args) == 0
    assert "--timeout 90.0" in out.getvalue()


def test_central_url_and_token_follow_env(monkeypatch):
    for name in drift.CENTRAL_ENV_VARS + drift.TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert drift.central_url() == drift.DEFAULT_CENTRAL
    monkeypatch.setenv("HUGPY_BASE_URL", "http://x:1/")
    assert drift.central_url() == "http://x:1"
    assert drift.central_url("http://y:2/") == "http://y:2"
    assert drift.central_token() is None
    monkeypatch.setenv("HUGPY_API_KEY", "hp_key")
    assert drift.central_token() == "hp_key"
    monkeypatch.setenv("HUGPY_TOKEN", "hpp_tok")
    assert drift.central_token() == "hpp_tok"
    assert drift.central_token("explicit") == "explicit"


# --------------------------------------------------------------------------- #
# D — pypi
# --------------------------------------------------------------------------- #
def _fake_pypi(monkeypatch, latest: dict):
    def fetch(url, token=None, timeout=None):
        name = url.split("/pypi/")[1].split("/")[0]
        if name not in latest:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
        return {"info": {"version": latest[name]}}
    monkeypatch.setattr(drift, "fetch_json", fetch)
    monkeypatch.setattr(drift, "workspace_distributions", lambda: ("hugpy-ops", "hugpy", "hugpy-new"))
    _fake_central_index(monkeypatch, {})


def _fake_central_index(monkeypatch, served: dict):
    """Central's ``/api/llm/pip/simple/<name>/`` pages: ``served`` maps a
    distribution name to the versions its wheels are listed at; a name absent
    from it 404s (the route lists nothing) and an empty dict means no central
    (connection refused)."""
    def fetch_text(url, token=None, timeout=None):
        if not served:
            raise urllib.error.URLError("connection refused")
        name = url.rstrip("/").rsplit("/", 1)[1]
        stem = name.replace("-", "_")
        return "".join(f'<a href="{stem}-{v}-py3-none-any.whl">x</a>' for v in served.get(name, ()))
    monkeypatch.setattr(drift, "fetch_text", fetch_text)


def test_unpublished_tag_is_drift(repos, monkeypatch):
    _, work = repos
    _fake_pypi(monkeypatch, {"hugpy-ops": "0.9.0", "hugpy": "1.0.0"})
    d = _rows(drift.run("D", workspace=str(work)), "D")
    assert "1.0.0" in d["tag"].detail
    assert d["hugpy-ops"].status == drift.DRIFT and "unpublished release" in d["hugpy-ops"].detail
    assert d["hugpy"].status == drift.OK
    assert d["hugpy-new"].status == drift.INFO and "not on PyPI" in d["hugpy-new"].detail


def test_tag_served_by_central_index_is_published(repos, monkeypatch):
    """A release published on central's own index (py/build_wheels.py --publish)
    is not 'unpublished' just because PyPI lacks it — the fleet converges from
    central. PyPI's version is still reported alongside."""
    _, work = repos
    _fake_pypi(monkeypatch, {"hugpy-ops": "0.9.0", "hugpy": "1.0.0"})
    _fake_central_index(monkeypatch, {"hugpy-ops": ["1.0.0"], "hugpy-new": ["1.0.0"],
                                      "hugpy": ["0.9.0"]})
    d = _rows(drift.run("D", workspace=str(work), central="http://c:1"), "D")
    assert d["hugpy-ops"].status == drift.OK and "central's index" in d["hugpy-ops"].detail \
        and "PyPI 0.9.0" in d["hugpy-ops"].detail
    assert d["hugpy-new"].status == drift.OK and "not on PyPI" in d["hugpy-new"].detail
    assert d["hugpy"].status == drift.OK and "= PyPI" in d["hugpy"].detail


def test_central_index_probe_stops_after_first_unreachable(repos, monkeypatch):
    _, work = repos
    _fake_pypi(monkeypatch, {"hugpy-ops": "0.9.0"})
    calls = []

    def fetch_text(url, token=None, timeout=None):
        calls.append(url)
        raise urllib.error.URLError("timed out")
    monkeypatch.setattr(drift, "fetch_text", fetch_text)
    monkeypatch.setattr(drift, "workspace_distributions", lambda: ("hugpy-ops", "hugpy-new", "hugpy"))
    d = _rows(drift.run("D", workspace=str(work), central="http://c:1"), "D")
    assert d["hugpy-ops"].status == drift.DRIFT and "unpublished release" in d["hugpy-ops"].detail
    assert d["hugpy-new"].status == drift.INFO
    assert len(calls) == 1   # one timeout, not one per distribution


def test_central_index_has_parses_the_simple_page(monkeypatch):
    monkeypatch.setattr(drift, "fetch_text", lambda url, token=None, timeout=None:
                        '<a href="hugpy_fleet-0.2.0-py3-none-any.whl#sha256=x">a</a>'
                        '<a href="hugpy_fleet-0.2.0.tar.gz">b</a>')
    assert drift.central_index_has("http://c:1/", "hugpy-fleet", "0.2.0") is True
    assert drift.central_index_has("http://c:1/", "hugpy-fleet", "0.2.0a0") is False
    assert drift.central_index_has(None, "hugpy-fleet", "0.2.0") is None


def test_checkout_behind_release_is_drift(repos, monkeypatch):
    _, work = repos
    _fake_pypi(monkeypatch, {"hugpy-ops": "1.1.0"})
    row = _rows(drift.run("D", workspace=str(work)), "D")["hugpy-ops"]
    assert row.status == drift.DRIFT and "checkout behind release" in row.detail


def test_no_tag_is_info(repos, monkeypatch):
    _, work = repos
    _git(work, "tag", "-d", "v1.0.0")
    _fake_pypi(monkeypatch, {"hugpy-ops": "1.1.0"})
    d = _rows(drift.run("D", workspace=str(work)), "D")
    assert d["tag"].status == drift.INFO and "no release tag" in d["tag"].detail
    assert d["hugpy-ops"].status == drift.INFO and "1.1.0" in d["hugpy-ops"].detail


def test_pypi_outage_is_error_and_no_pypi_skips(repos, monkeypatch):
    _, work = repos

    def fetch(url, token=None, timeout=None):
        raise urllib.error.URLError("timed out")
    monkeypatch.setattr(drift, "fetch_json", fetch)
    monkeypatch.setattr(drift, "workspace_distributions", lambda: ("hugpy-ops",))
    report = drift.run("D", workspace=str(work))
    assert _rows(report, "D")["hugpy-ops"].status == drift.ERROR and report.exit_code == 2
    skipped = drift.run("D", workspace=str(work), pypi=False)
    assert [r.status for r in skipped.rows] == [drift.INFO] and skipped.exit_code == 0


@pytest.mark.parametrize("a, b, expect", [
    ("1.0.0", "1.0.0", 0), ("v1.0.1", "1.0.0", 1), ("1.0.0", "1.0.1", -1),
    ("1.0.0", "1.0.0rc1", 1), ("1.0.0.dev3", "1.0.0", -1), ("1.0.0.post1", "1.0.0", 1),
    ("1.0.1.dev3+gabc", "1.0.0", 1), ("1.0", "1.0.0", 0),
])
def test_version_ordering(a, b, expect):
    assert drift.version_cmp(a, b) == expect


def test_version_cmp_rejects_garbage():
    assert drift.version_cmp("not-a-version", "1.0") is None
    assert drift.strip_v("v2.0") == "2.0" and drift.strip_v("vibes") == "vibes"


# --------------------------------------------------------------------------- #
# report, exit codes, CLI
# --------------------------------------------------------------------------- #
def test_exit_codes_and_json():
    r = drift.Report()
    r.add("A", "x", drift.OK)
    r.add("A", "y", drift.INFO)
    assert r.exit_code == 0 and r.ok
    r.add("C", "z", drift.ERROR, "unreachable")
    assert r.exit_code == 2 and not r.ok
    r.add("B", "w", drift.DRIFT, "mixed")
    assert r.exit_code == 1
    doc = json.loads(r.to_json())
    assert doc["exit_code"] == 1 and doc["verdict"] == "DRIFT"
    assert doc["counts"] == {"ok": 1, "drift": 1, "error": 1, "info": 1}
    assert doc["rows"][3] == {"section": "B", "subject": "w", "status": "drift", "detail": "mixed"}


def test_table_rendering_is_aligned():
    r = drift.Report()
    r.add("A", "HEAD", drift.INFO, "abc on main")
    r.add("B", "hugpy-platform", drift.DRIFT, "mixed versions")
    r.add("C", "worker computron", drift.OK, "1.0.0 gabc1234 [online]")
    text = r.table()
    lines = text.splitlines()
    assert lines[0].startswith("SECTION") and "SUBJECT" in lines[0] and "DETAIL" in lines[0]
    assert set(lines[1]) <= {"-", " "}
    body = lines[2:5]
    # the STATUS column starts at the same offset on every row and drift shouts
    offsets = {ln.index(word) for ln, word in zip(body, ("info", "DRIFT", "ok"))}
    assert len(offsets) == 1
    assert "B installed  hugpy-platform" in body[1]
    assert lines[-1].startswith("drift-check: 1 ok, 1 drift, 0 error, 1 info")
    assert lines[-1].endswith("DRIFT  (exit 1)")


def test_parse_sections():
    assert drift.parse_sections(None) == ["A", "B", "C", "D"]
    assert drift.parse_sections("c, a") == ["C", "A"]
    assert drift.parse_sections("fleet,checkout,A") == ["C", "A"]
    with pytest.raises(ValueError):
        drift.parse_sections("Z")


def test_main_exit_codes_quiet_and_json(monkeypatch, capsys):
    def fake_run(sections, **kw):
        r = drift.Report()
        r.add("A", "tree", drift.OK, "clean")
        if "C" in sections:
            r.add("C", "worker x", drift.DRIFT, "monolith")
        return r
    monkeypatch.setattr(drift, "run", fake_run)
    assert drift.main(["--sections", "A", "--quiet"]) == 0
    assert capsys.readouterr().out == ""
    assert drift.main(["--sections", "A,C", "--quiet"]) == 1
    assert "worker x" in capsys.readouterr().out
    assert drift.main(["--sections", "A,C", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["exit_code"] == 1
    assert drift.main(["--sections", "Q"]) == 2
    assert "unknown section" in capsys.readouterr().err


def test_notify_url_posts_the_report_only_when_not_ok(monkeypatch, capsys):
    posted = []

    def fake_run(sections, **kw):
        r = drift.Report()
        r.add("B", "lockstep", drift.DRIFT if "B" in sections else drift.OK, "x")
        return r
    monkeypatch.setattr(drift, "run", fake_run)
    monkeypatch.setattr(drift, "post_json", lambda url, payload, timeout=None: posted.append((url, payload)) or 200)
    assert drift.main(["--sections", "A", "--notify-url", "http://n/hook", "--quiet"]) == 0
    assert posted == []
    assert drift.main(["--sections", "B", "--notify-url", "http://n/hook", "--quiet"]) == 1
    assert posted[0][0] == "http://n/hook" and posted[0][1]["exit_code"] == 1
    assert "notified http://n/hook" in capsys.readouterr().err


def test_install_timer_dry_run_prints_units(monkeypatch, capsys, tmp_path):
    script = tmp_path / "bin" / "hugpy-drift-check"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\n")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "bin" / "python"))
    rc = drift.main(["--install-timer", "--dry-run", "--user", "--on-calendar", "hourly",
                     "--central", "http://c:7002", "--notify-url", "http://n/hook",
                     "--env-file", str(tmp_path / "drift.env"), "--sections", "A,B",
                     "--token", "hpp_secret"])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"ExecStart={script} --quiet --fetch --central http://c:7002 --notify-url http://n/hook --sections A,B" in out
    assert f"EnvironmentFile=-{tmp_path / 'drift.env'}" in out
    assert "Environment=HUGPY_TOKEN=hpp_secret" in out
    assert "OnCalendar=hourly" in out
    assert "WantedBy=default.target" in out and "WantedBy=timers.target" in out
    assert ".config/systemd/user/hugpy-drift-check.service" in out
    assert "#   systemctl --user daemon-reload" in out
    assert "#   systemctl --user enable --now hugpy-drift-check.timer" in out
    assert not (tmp_path / ".config").exists()


def test_install_timer_falls_back_to_python_m_when_script_missing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "nobin" / "python"))
    assert drift.main(["--install-timer", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"ExecStart={tmp_path / 'nobin' / 'python'} -m hugpy_ops.drift --quiet --fetch" in out
    assert "--sections" not in out
    assert "/etc/systemd/system/hugpy-drift-check.timer" in out


def test_install_timer_writes_units_and_enables(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []
    args = drift.build_parser().parse_args(["--install-timer", "--user"])
    rc = drift.install_timer(args, runner=lambda cmd: calls.append(cmd) or 0)
    assert rc == 0
    unit_dir = tmp_path / ".config" / "systemd" / "user"
    assert (unit_dir / "hugpy-drift-check.service").exists()
    assert "OnCalendar=daily" in (unit_dir / "hugpy-drift-check.timer").read_text()
    assert calls == [["systemctl", "--user", "daemon-reload"],
                     ["systemctl", "--user", "enable", "--now", "hugpy-drift-check.timer"]]


def test_help_mentions_sections():
    with pytest.raises(SystemExit) as info:
        drift.main(["--help"])
    assert info.value.code == 0


def test_render_units_run_as_writes_user_for_system_units_only():
    units = drift.render_units(exec_start="/x/hugpy-drift-check", run_as="hugpy")
    assert "User=hugpy\n" in units["hugpy-drift-check.service"]
    units = drift.render_units(exec_start="/x/hugpy-drift-check", run_as="hugpy", user_mode=True)
    assert "User=" not in units["hugpy-drift-check.service"]
    units = drift.render_units(exec_start="/x/hugpy-drift-check")
    assert "User=" not in units["hugpy-drift-check.service"]
