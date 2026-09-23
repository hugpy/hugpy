"""hugpy_platform.buildinfo — which build is this process.

* ``git_identity`` against a throwaway repository: sha/short/branch, a clean
  tree, then a dirty tree, then a tag (nearest-tag + describe), then not-a-repo
  and a missing ``git`` binary both answer None.
* ``parse_version`` recovers sha/dirty from setuptools-scm local segments and
  says "unknown" when a build had no metadata.
* ``distribution_record`` / ``build_identity`` / ``build_info`` over a
  monkeypatched ``importlib.metadata``: the precedence order, editable vs.
  wheel sha sourcing, the lockstep verdict, absent externals as None.
* Nothing raises even when the metadata layer blows up.
* The lazy package export costs nothing at import.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hugpy_platform import buildinfo

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]


def _git(repo, *args):
    return subprocess.run([*GIT, "-C", str(repo), *args], check=True,
                          capture_output=True, text=True, timeout=30).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / "a.txt").write_text("one\n")
    _git(r, "add", "a.txt")
    _git(r, "commit", "-q", "-m", "first")
    return r


# -- git_identity --------------------------------------------------------------


def test_git_identity_clean_tree(repo):
    ident = buildinfo.git_identity(str(repo))
    assert ident is not None
    assert ident["sha"] == _git(repo, "rev-parse", "HEAD")
    assert len(ident["sha"]) == 40
    assert ident["sha"].startswith(ident["short"])
    assert ident["branch"] == "main"
    assert ident["dirty"] is False
    assert ident["tag"] is None, "untagged repo: no nearest tag"
    assert ident["describe"] == _git(repo, "describe", "--tags", "--always", "--dirty")
    assert os.path.samefile(ident["root"], repo)


def test_git_identity_file_path_and_subdir_resolve_to_repo(repo):
    sub = repo / "pkg"
    sub.mkdir()
    (sub / "m.py").write_text("x = 1\n")  # untracked: must NOT count as dirty
    by_dir = buildinfo.git_identity(str(sub))
    by_file = buildinfo.git_identity(str(sub / "m.py"))
    assert by_dir and by_file
    assert by_dir["sha"] == by_file["sha"]
    assert by_dir["dirty"] is False


def test_git_identity_dirty_tracked_change(repo):
    (repo / "a.txt").write_text("two\n")
    ident = buildinfo.git_identity(str(repo))
    assert ident["dirty"] is True
    assert ident["describe"].endswith("-dirty")


def test_git_identity_tag_and_describe(repo):
    _git(repo, "tag", "v1.2.3")
    at_tag = buildinfo.git_identity(str(repo))
    assert at_tag["tag"] == "v1.2.3"
    assert at_tag["describe"] == "v1.2.3"
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-q", "-m", "second")
    later = buildinfo.git_identity(str(repo))
    assert later["tag"] == "v1.2.3", "nearest reachable tag"
    assert later["describe"].startswith("v1.2.3-1-g")
    assert later["sha"] != at_tag["sha"]


def test_git_identity_detached_head_has_no_branch(repo):
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach", sha)
    assert buildinfo.git_identity(str(repo))["branch"] is None


def test_git_identity_not_a_repo_and_missing_paths(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    # tmp_path may itself sit under some repository; force a non-repo answer by
    # pointing GIT_CEILING_DIRECTORIES at the parent.
    os.environ["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    try:
        assert buildinfo.git_identity(str(plain)) is None
    finally:
        os.environ.pop("GIT_CEILING_DIRECTORIES", None)
    assert buildinfo.git_identity(None) is None
    assert buildinfo.git_identity("") is None
    assert buildinfo.git_identity(str(tmp_path / "does" / "not" / "exist")) is None


def test_git_identity_survives_missing_git_binary(repo, monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(buildinfo.subprocess, "run", boom)
    assert buildinfo.git_identity(str(repo)) is None


def test_git_calls_carry_the_timeout(repo, monkeypatch):
    seen = []
    real = subprocess.run

    def spy(argv, **kw):
        seen.append(kw.get("timeout"))
        return real(argv, **kw)
    monkeypatch.setattr(buildinfo.subprocess, "run", spy)
    buildinfo.git_identity(str(repo))
    assert seen and all(t == buildinfo.GIT_TIMEOUT_S for t in seen)


# -- parse_version ---------------------------------------------------------------


@pytest.mark.parametrize("version, expected", [
    ("0.2.1.dev3+g952c2d8", {"sha": "952c2d8", "dirty": False, "unknown": False}),
    ("0.2.1.dev3+g952c2d8.d20260922", {"sha": "952c2d8", "dirty": True, "unknown": False}),
    ("0.0.1.dev11+unknown.g952c2d8aa", {"sha": "952c2d8aa", "dirty": False, "unknown": True}),
    ("0.0.0+unknown", {"sha": None, "dirty": None, "unknown": True}),
    ("1.0.0", {"sha": None, "dirty": None, "unknown": False}),
    (None, {"sha": None, "dirty": None, "unknown": False}),
    ("", {"sha": None, "dirty": None, "unknown": False}),
    ("1.0.0+d20260922", {"sha": None, "dirty": True, "unknown": False}),
])
def test_parse_version(version, expected):
    assert buildinfo.parse_version(version) == expected


# -- importlib.metadata seam ----------------------------------------------------


class _FakeDist:
    def __init__(self, version, direct_url=None):
        self.version = version
        self._direct = direct_url

    def read_text(self, name):
        if name == "direct_url.json" and self._direct is not None:
            return json.dumps(self._direct)
        return None


def _install(monkeypatch, dists):
    """Replace ``importlib.metadata.distribution`` with a lookup over ``dists``
    (keys already normalised)."""
    import importlib.metadata as md

    def fake_distribution(name):
        key = buildinfo.normalize_dist(name)
        if key not in dists:
            raise md.PackageNotFoundError(name)
        return dists[key]
    monkeypatch.setattr(md, "distribution", fake_distribution)


def _editable(version, path):
    return _FakeDist(version, {"dir_info": {"editable": True}, "url": f"file://{path}"})


def _wheel(version):
    return _FakeDist(version)


def test_distribution_record_editable_wheel_and_absent(monkeypatch, tmp_path):
    _install(monkeypatch, {
        "hugpy-platform": _editable("1.0.0+gabc1234", tmp_path / "platform"),
        "hugpy-agent": _wheel("0.9.0"),
        "abstract-hugpy-dev": _FakeDist("0.1.267.dev0", {"url": "https://example/x.whl", "archive_info": {}}),
    })
    rec = buildinfo.distribution_record("hugpy-platform")
    assert rec == {"name": "hugpy-platform", "version": "1.0.0+gabc1234",
                   "editable": True, "source": str(tmp_path / "platform")}
    assert buildinfo.distribution_record("hugpy-agent") == {
        "name": "hugpy-agent", "version": "0.9.0", "editable": False, "source": None}
    # the compat shell under its importlib-normalised name, from a non-file direct url
    shell = buildinfo.distribution_record("abstract_hugpy_dev")
    assert shell["name"] == "abstract-hugpy-dev" and shell["editable"] is False and shell["source"] is None
    assert buildinfo.distribution_record("hugpy-fleet") is None


def test_build_identity_editable_reads_live_git(monkeypatch, repo):
    _install(monkeypatch, {
        "hugpy-platform": _editable("0.0.1.dev1+gdeadbee", repo / "py" / "platform"),
    })
    (repo / "py" / "platform").mkdir(parents=True)
    ident = buildinfo.build_identity()
    assert ident["distribution"] == "hugpy-platform"
    assert ident["version"] == "0.0.1.dev1+gdeadbee"
    assert ident["editable"] is True
    assert ident["source"] == str(repo / "py" / "platform")
    assert ident["sha"] == _git(repo, "rev-parse", "HEAD"), "live checkout wins over the stamped sha"
    assert ident["dirty"] is False
    (repo / "a.txt").write_text("changed\n")
    assert buildinfo.build_identity()["dirty"] is True


def test_build_identity_precedence_and_wheel_parsing(monkeypatch):
    _install(monkeypatch, {
        "hugpy-platform": _wheel("0.2.1.dev3+g952c2d8.d20260922"),
        "hugpy-server": _wheel("0.2.1.dev3+g952c2d8.d20260922"),
        "hugpy-fleet": _wheel("0.2.1.dev3+g952c2d8"),
    })
    ident = buildinfo.build_identity()
    assert ident["distribution"] == "hugpy-fleet", "fleet before server before platform"
    assert ident["editable"] is False and ident["source"] is None
    assert ident["sha"] == "952c2d8" and ident["dirty"] is False
    _install(monkeypatch, {"hugpy-server": _wheel("0.2.1.dev3+g952c2d8.d20260922")})
    ident = buildinfo.build_identity()
    assert ident["distribution"] == "hugpy-server" and ident["dirty"] is True
    _install(monkeypatch, {"hugpy-platform": _wheel("0.0.0+unknown")})
    ident = buildinfo.build_identity()
    assert ident["distribution"] == "hugpy-platform"
    assert ident["sha"] is None and ident["dirty"] is None


def test_build_identity_nothing_installed(monkeypatch):
    _install(monkeypatch, {})
    assert buildinfo.build_identity() == {
        "version": None, "sha": None, "dirty": None, "editable": False,
        "source": None, "distribution": None}
    assert buildinfo.identity_line() == "hugpy unknown"


def test_build_info_document_and_lockstep(monkeypatch, repo):
    src = repo / "py"
    src.mkdir()
    dists = {name: _editable("0.3.0", src) for name in buildinfo.WORKSPACE_DISTRIBUTIONS}
    dists["hugpy-agent"] = _wheel("0.9.0")
    _install(monkeypatch, dists)
    doc = buildinfo.build_info()
    assert set(doc) >= {"identity", "distributions", "lockstep", "workspace",
                        "python", "executable", "hostname", "generated_at"}
    assert set(doc["distributions"]) == set(buildinfo.WORKSPACE_DISTRIBUTIONS + buildinfo.EXTERNAL_DISTRIBUTIONS)
    assert doc["distributions"]["abstract-identity"] is None
    assert doc["distributions"]["hugpy-agent"]["version"] == "0.9.0"
    assert doc["lockstep"] == {"ok": True, "versions": {"0.3.0": list(buildinfo.WORKSPACE_DISTRIBUTIONS)}}
    assert doc["workspace"]["sha"] == _git(repo, "rev-parse", "HEAD")
    assert doc["identity"]["sha"] == doc["workspace"]["sha"]
    assert doc["python"] == sys.version.split()[0]
    assert doc["executable"] == sys.executable
    assert doc["generated_at"].endswith("Z")
    json.dumps(doc)  # wire-safe

    # break lockstep: one distribution lags
    dists["hugpy-ops"] = _wheel("0.2.9")
    doc = buildinfo.build_info()
    assert doc["lockstep"]["ok"] is False
    assert doc["lockstep"]["versions"]["0.2.9"] == ["hugpy-ops"]
    assert "hugpy-ops" not in doc["lockstep"]["versions"]["0.3.0"]


def test_lockstep_ignores_absent_and_is_ok_when_empty():
    assert buildinfo.lockstep({}) == {"ok": True, "versions": {}}
    assert buildinfo.lockstep({"hugpy-platform": None, "hugpy-fleet": {"version": "1"}}) == {
        "ok": True, "versions": {"1": ["hugpy-fleet"]}}


def test_identity_line_shapes():
    assert buildinfo.identity_line({"distribution": "hugpy-fleet", "version": "0.2.1.dev3+g952c2d8",
                                    "editable": True, "source": "/p", "dirty": True,
                                    "sha": "952c2d8"}) == "hugpy-fleet 0.2.1.dev3+g952c2d8 (editable /p, dirty)"
    assert buildinfo.identity_line({"distribution": "hugpy-server", "version": "1.0.0",
                                    "editable": False, "source": None, "dirty": False,
                                    "sha": None}) == "hugpy-server 1.0.0"
    assert buildinfo.identity_line({"distribution": "hugpy-server", "version": "1.0.0",
                                    "editable": False, "source": None, "dirty": False,
                                    "sha": "abcdef0123"}) == "hugpy-server 1.0.0 (gabcdef0)"


def test_never_raises_when_metadata_layer_explodes(monkeypatch):
    import importlib.metadata as md

    def boom(_name):
        raise RuntimeError("corrupt dist-info")
    monkeypatch.setattr(md, "distribution", boom)
    assert buildinfo.distribution_record("hugpy-platform") is None
    ident = buildinfo.build_identity()
    assert ident["version"] is None and ident["distribution"] is None
    doc = buildinfo.build_info()
    assert doc["lockstep"] == {"ok": True, "versions": {}}
    assert all(v is None for v in doc["distributions"].values())
    monkeypatch.setattr(buildinfo, "_identity_record", lambda: (_ for _ in ()).throw(ValueError("x")))
    assert "error" in buildinfo.build_identity()
    assert "error" in buildinfo.build_info()


def test_real_interpreter_answers_something():
    """No monkeypatch: whatever this interpreter holds, the calls answer."""
    ident = buildinfo.build_identity()
    assert set(ident) >= {"version", "sha", "dirty", "editable", "source", "distribution"}
    assert isinstance(buildinfo.identity_line(), str) and buildinfo.identity_line()
    assert isinstance(buildinfo.build_info()["lockstep"]["ok"], bool)


def test_module_main_prints_identity_and_json(capsys):
    assert buildinfo.main([]) == 0
    assert capsys.readouterr().out.strip() == buildinfo.identity_line()
    assert buildinfo.main(["--json"]) == 0
    assert "identity" in json.loads(capsys.readouterr().out)


# -- lazy package export --------------------------------------------------------


def test_package_exports_buildinfo_lazily():
    code = (
        "import sys; import hugpy_platform; "
        "before = 'hugpy_platform.buildinfo' in sys.modules; "
        "mod = hugpy_platform.buildinfo; "
        "print(before, mod.__name__, mod is sys.modules['hugpy_platform.buildinfo'])"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.split() == ["False", "hugpy_platform.buildinfo", "True"]
    import hugpy_platform
    with pytest.raises(AttributeError):
        hugpy_platform.no_such_submodule_x9
