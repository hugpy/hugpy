"""WP5 worker convergence: the self-update converges the WHOLE lockstep set.

After the 0.2 partition a worker runs hugpy-fleet plus its siblings
(hugpy-platform, -control, -storage, -engine, -media, ...) at ONE git-derived
version. ``pip install -U --no-deps hugpy-fleet==X`` alone leaves the siblings
behind — the silent skew of the 2026-07-20 incident class. The converge now
fetches central's ``/api/llm/workers/constraints.txt`` and runs pip under
those pins; when the pins cannot be had it falls back to the old single-package
command and says so LOUDLY.

Covers:
  * ``_pip_converge_command``: the constrained form (``--upgrade-strategy
    only-if-needed -c <file>``, no ``--no-deps``, ``--index-url`` kept, installed
    siblings appended) and the fallback form (``--no-deps``, one spec).
  * ``_prepare_converge``: fetch → temp constraints file → constrained command;
    fetch raising / returning no lines → fallback + "fleet may be SKEWED".
  * ``_constraints_url``: reply hint wins; else derived from ``--central`` under
    the ``/api`` prefix; None without a central.
  * ``_fetch_constraints``: parses lines, drops comments/blanks, 204 → [].
  * ``_self_update_if_needed`` end to end (pip + restart faked): runs the
    constrained command, cleans up the temp file, records ``constrained`` in the
    update state and restarts with the WorkerState (not the update dict).
  * ``lockstep_constraints()``: buildinfo import forced to fail → the 13-name
    fallback; forced to succeed → the platform's list; no required → [].
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as A  # noqa: E402
from hugpy_fleet.central import workers as W  # noqa: E402


class _Args:
    def __init__(self, **kw):
        self.pkg_name = "hugpy-fleet"
        self.pkg_index = None
        self.central = "https://central.example"
        self.id_file = None
        self.__dict__.update(kw)


# ── _pip_converge_command ────────────────────────────────────────────────────
def test_constrained_command_shape():
    cmd = A._pip_converge_command(_Args(), "0.2.0", "/tmp/c.txt")
    assert cmd == [sys.executable, "-m", "pip", "install", "-U",
                   "--upgrade-strategy", "only-if-needed", "-c", "/tmp/c.txt",
                   "hugpy-fleet==0.2.0"]
    assert "--no-deps" not in cmd


def test_constrained_command_keeps_index_url_and_appends_siblings():
    cmd = A._pip_converge_command(
        _Args(pkg_index="https://central.example/api/llm/pip/simple"), "0.2.0",
        "/tmp/c.txt", extra_specs=["hugpy-platform==0.2.0", "hugpy-media==0.2.0"])
    assert cmd == [sys.executable, "-m", "pip", "install", "-U",
                   "--upgrade-strategy", "only-if-needed", "-c", "/tmp/c.txt",
                   "--index-url", "https://central.example/api/llm/pip/simple",
                   "hugpy-fleet==0.2.0", "hugpy-platform==0.2.0", "hugpy-media==0.2.0"]


def test_fallback_command_is_the_pre_partition_no_deps_form():
    cmd = A._pip_converge_command(_Args(), "0.2.0", None,
                                  extra_specs=["hugpy-platform==0.2.0"])
    assert cmd == [sys.executable, "-m", "pip", "install", "-U", "--no-deps",
                   "hugpy-fleet==0.2.0"]
    # siblings are never named on the fallback path (no pins to hold them)
    assert "hugpy-platform==0.2.0" not in cmd


def test_fallback_command_keeps_index_url():
    cmd = A._pip_converge_command(_Args(pkg_index="https://idx/simple"), "0.2.0", None)
    assert cmd[-3:] == ["--index-url", "https://idx/simple", "hugpy-fleet==0.2.0"]


# ── _constraints_url ─────────────────────────────────────────────────────────
def test_constraints_url_prefers_reply_hint_then_derives_from_central():
    assert A._constraints_url(_Args(), "https://other/api/llm/workers/constraints.txt") \
        == "https://other/api/llm/workers/constraints.txt"
    assert A._constraints_url(_Args(central="https://central.example/")) \
        == "https://central.example/api/llm/workers/constraints.txt"
    assert A._constraints_url(_Args(central=None)) is None


# ── _fetch_constraints ───────────────────────────────────────────────────────
class _Resp(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_fetch_constraints_parses_lines_and_drops_comments(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return _Resp(b"# central pins\nhugpy-platform==0.2.0\n\n  hugpy-fleet==0.2.0 \n")

    monkeypatch.setattr(A.urllib.request, "urlopen", fake_urlopen)
    lines = A._fetch_constraints("https://c/api/llm/workers/constraints.txt")
    assert lines == ["hugpy-platform==0.2.0", "hugpy-fleet==0.2.0"]
    assert seen["url"] == "https://c/api/llm/workers/constraints.txt"


def test_fetch_constraints_204_means_no_pins(monkeypatch):
    monkeypatch.setattr(A.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(b"", status=204))
    assert A._fetch_constraints("https://c/x") == []


# ── _prepare_converge ────────────────────────────────────────────────────────
def test_prepare_converge_constrained_path(monkeypatch):
    lines = ["hugpy-platform==0.2.0", "hugpy-fleet==0.2.0", "hugpy-media==0.2.0"]
    monkeypatch.setattr(A, "_fetch_constraints", lambda url, timeout=None: list(lines))
    # deterministic sibling set (don't depend on what this venv has installed)
    monkeypatch.setattr(A, "_installed_lockstep_siblings",
                        lambda ls, pkg: ["hugpy-platform"])
    cmd, path = A._prepare_converge(_Args(), "0.2.0")
    try:
        assert path and os.path.isfile(path)
        assert Path(path).read_text(encoding="utf-8") == "\n".join(lines) + "\n"
        assert cmd == [sys.executable, "-m", "pip", "install", "-U",
                       "--upgrade-strategy", "only-if-needed", "-c", path,
                       "hugpy-fleet==0.2.0", "hugpy-platform==0.2.0"]
    finally:
        A._discard_constraints(path)
    assert not os.path.exists(path)


@pytest.mark.parametrize("fetch", [
    pytest.param(lambda url, timeout=None: (_ for _ in ()).throw(OSError("unreachable")),
                 id="fetch-raises"),
    pytest.param(lambda url, timeout=None: [], id="fetch-empty"),
])
def test_prepare_converge_falls_back_loudly(monkeypatch, caplog, fetch):
    monkeypatch.setattr(A, "_fetch_constraints", fetch)
    with caplog.at_level(logging.WARNING, logger=A.logger.name):
        cmd, path = A._prepare_converge(_Args(), "0.2.0")
    assert path is None
    assert cmd == [sys.executable, "-m", "pip", "install", "-U", "--no-deps",
                   "hugpy-fleet==0.2.0"]
    assert any("SKEWED" in r.getMessage() for r in caplog.records)


def test_prepare_converge_without_central_url_falls_back(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(A, "_fetch_constraints",
                        lambda url, timeout=None: calls.append(url) or [])
    with caplog.at_level(logging.WARNING, logger=A.logger.name):
        cmd, path = A._prepare_converge(_Args(central=None), "0.2.0")
    assert calls == [] and path is None and "--no-deps" in cmd


# ── _installed_lockstep_siblings ─────────────────────────────────────────────
def test_installed_siblings_excludes_tracked_and_absent(monkeypatch):
    from importlib import metadata
    present = {"hugpy-platform": "0.1.0", "hugpy-fleet": "0.1.0"}

    def fake_version(name):
        if name in present:
            return present[name]
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", fake_version)
    got = A._installed_lockstep_siblings(
        ["hugpy-platform==0.2.0", "hugpy-fleet==0.2.0", "hugpy-media==0.2.0",
         "# comment", "garbage"], "hugpy_fleet")
    assert got == ["hugpy-platform"]


# ── _self_update_if_needed end to end ────────────────────────────────────────
def test_self_update_runs_constrained_pip_then_restarts_with_worker_state(
        monkeypatch, tmp_path):
    args = _Args(id_file=str(tmp_path / "worker.id"))
    lines = ["hugpy-platform==0.2.0", "hugpy-fleet==0.2.0"]
    monkeypatch.setattr(A, "_fetch_constraints", lambda url, timeout=None: list(lines))
    monkeypatch.setattr(A, "_installed_lockstep_siblings", lambda ls, pkg: ["hugpy-platform"])
    monkeypatch.setattr(A, "_installed_pkg_version", lambda name: "0.1.0")

    ran = {}

    def fake_call(cmd):
        ran["cmd"] = list(cmd)
        cpath = cmd[cmd.index("-c") + 1]
        ran["constraints_body"] = Path(cpath).read_text(encoding="utf-8")
        ran["cpath"] = cpath
        return 0

    monkeypatch.setattr(A.subprocess, "call", fake_call)
    restarted = {}
    monkeypatch.setattr(A, "_restart",
                        lambda state, **kw: restarted.update(state=state, **kw))

    ws = A.WorkerState(name="t", url=None, worker_id="w-converge")
    A._self_update_if_needed("0.2.0", args, ws,
                             constraints_url="https://c/api/llm/workers/constraints.txt")

    assert ran["cmd"][:9] == [sys.executable, "-m", "pip", "install", "-U",
                              "--upgrade-strategy", "only-if-needed", "-c", ran["cpath"]]
    assert ran["cmd"][9:] == ["hugpy-fleet==0.2.0", "hugpy-platform==0.2.0"]
    assert ran["constraints_body"] == "hugpy-platform==0.2.0\nhugpy-fleet==0.2.0\n"
    assert not os.path.exists(ran["cpath"])          # temp pins cleaned up
    # the existing safeguards: restart on the fresh code, slots torn down, and
    # the restart gets the WorkerState (the update-state dict used to shadow it)
    assert restarted["state"] is ws
    assert restarted["kill_slots"] is True and restarted["reason"] == "self-update"
    saved = json.loads((tmp_path / "worker.id.update.json").read_text())
    assert saved["target"] == "0.2.0" and saved["rc"] == 0 and saved["constrained"] is True


def test_self_update_fallback_still_restarts_and_records_unconstrained(
        monkeypatch, tmp_path):
    args = _Args(id_file=str(tmp_path / "worker.id"))
    monkeypatch.setattr(A, "_fetch_constraints",
                        lambda url, timeout=None: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setattr(A, "_installed_pkg_version", lambda name: "0.1.0")
    ran = {}
    monkeypatch.setattr(A.subprocess, "call", lambda cmd: ran.setdefault("cmd", list(cmd)) and 0)
    restarted = {}
    monkeypatch.setattr(A, "_restart", lambda state, **kw: restarted.update(kw))
    A._self_update_if_needed("0.2.0", args, A.WorkerState(name="t", url=None, worker_id="w"))
    assert ran["cmd"] == [sys.executable, "-m", "pip", "install", "-U", "--no-deps",
                          "hugpy-fleet==0.2.0"]
    assert restarted["kill_slots"] is True
    saved = json.loads((tmp_path / "worker.id.update.json").read_text())
    assert saved["constrained"] is False


def test_self_update_backoff_skips_recent_same_target(monkeypatch, tmp_path):
    args = _Args(id_file=str(tmp_path / "worker.id"))
    monkeypatch.setattr(A, "_installed_pkg_version", lambda name: "0.1.0")
    A._save_update_state(args, {"target": "0.2.0", "at": A.time.time(), "rc": 1})
    monkeypatch.setattr(A, "_prepare_converge",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    A._self_update_if_needed("0.2.0", args, None)   # backoff → returns before pip


# ── lockstep_constraints (central side) ──────────────────────────────────────
def test_lockstep_constraints_fallback_when_buildinfo_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "hugpy_platform.buildinfo", None)  # import fails
    lines = W.lockstep_constraints("0.2.0")
    assert len(lines) == 13
    assert lines == [f"{n}==0.2.0" for n in W.WORKSPACE_DISTRIBUTIONS_FALLBACK]
    assert "hugpy-fleet==0.2.0" in lines and "hugpy==0.2.0" in lines


def test_lockstep_constraints_uses_buildinfo_when_present(monkeypatch):
    fake = types.ModuleType("hugpy_platform.buildinfo")
    fake.WORKSPACE_DISTRIBUTIONS = ("hugpy-platform", "hugpy-fleet")
    monkeypatch.setitem(sys.modules, "hugpy_platform.buildinfo", fake)
    assert W.lockstep_constraints("0.2.0") == ["hugpy-platform==0.2.0", "hugpy-fleet==0.2.0"]


def test_lockstep_constraints_reads_required_and_is_empty_when_unpinned(monkeypatch):
    monkeypatch.setitem(sys.modules, "hugpy_platform.buildinfo", None)
    monkeypatch.setattr(W, "required_pkg_version", lambda: None)
    assert W.lockstep_constraints() == []
    monkeypatch.setattr(W, "required_pkg_version", lambda: "0.3.1")
    assert W.lockstep_constraints()[0] == "hugpy-platform==0.3.1"


def test_constraints_url_helper():
    assert W.constraints_url("https://c.example/") == \
        "https://c.example/api/llm/workers/constraints.txt"


# ── central's own index as an EXTRA source (pkg_index_url hint) ──────────────
def test_extra_index_joins_pypi_on_constrained_and_fallback_commands():
    hint = "https://central.example/api/llm/pip/simple"
    cmd = A._pip_converge_command(_Args(), "0.2.0", "/tmp/c.txt", extra_index=hint)
    assert "--index-url" not in cmd                      # PyPI stays primary
    assert cmd[cmd.index("--extra-index-url") + 1] == hint
    assert cmd[-1] == "hugpy-fleet==0.2.0"
    cmd = A._pip_converge_command(_Args(), "0.2.0", None, extra_index=hint)
    assert cmd == [sys.executable, "-m", "pip", "install", "-U", "--no-deps",
                   "--extra-index-url", hint, "hugpy-fleet==0.2.0"]


def test_extra_index_is_dropped_when_it_equals_the_explicit_pkg_index():
    idx = "https://central.example/api/llm/pip/simple"
    cmd = A._pip_converge_command(_Args(pkg_index=idx), "0.2.0", None, extra_index=idx + "/")
    assert cmd.count(idx) == 1 and "--extra-index-url" not in cmd
    assert cmd[cmd.index("--index-url") + 1] == idx


def test_prepare_converge_threads_the_index_hint(monkeypatch):
    monkeypatch.setattr(A, "_fetch_constraints",
                        lambda url, timeout=None: ["hugpy-platform==0.2.0", "hugpy-fleet==0.2.0"])
    monkeypatch.setattr(A, "_installed_lockstep_siblings", lambda lines, pkg: ["hugpy-platform"])
    cmd, path = A._prepare_converge(_Args(), "0.2.0",
                                    pkg_index_url="https://c/api/llm/pip/simple")
    try:
        assert cmd[cmd.index("--extra-index-url") + 1] == "https://c/api/llm/pip/simple"
        assert "-c" in cmd and cmd[-2:] == ["hugpy-fleet==0.2.0", "hugpy-platform==0.2.0"]
    finally:
        A._discard_constraints(path)


def test_self_update_passes_the_reply_hint_to_pip(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(A, "_installed_pkg_version", lambda pkg: "0.1.9")
    monkeypatch.setattr(A, "_load_update_state", lambda args: {})
    monkeypatch.setattr(A, "_save_update_state", lambda args, st: seen.setdefault("state", st))
    monkeypatch.setattr(A, "_fetch_constraints", lambda url, timeout=None: [])
    monkeypatch.setattr(A.subprocess, "call", lambda cmd: seen.setdefault("cmd", cmd) and 1)
    A._self_update_if_needed("0.2.0", _Args(id_file=str(tmp_path / "id")), state=None,
                             constraints_url=None,
                             pkg_index_url="https://c/api/llm/pip/simple")
    assert seen["cmd"][-3:] == ["--extra-index-url", "https://c/api/llm/pip/simple",
                                "hugpy-fleet==0.2.0"]


# ── central side: pkg_index_has / pkg_index_url ─────────────────────────────
def test_pkg_index_url_is_under_the_api_mount():
    assert W.pkg_index_url("https://central.example/") == \
        "https://central.example/api/llm/pip/simple"


def test_pkg_index_has_requires_every_lockstep_wheel_at_that_version(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "pkg_index_dir", lambda: str(tmp_path))
    monkeypatch.setattr(W, "workspace_distributions", lambda: ("hugpy-platform", "hugpy-fleet"))
    assert W.pkg_index_has("0.2.0") is False                      # empty dir
    (tmp_path / "hugpy_platform-0.2.0-py3-none-any.whl").write_bytes(b"")
    (tmp_path / "hugpy_fleet-0.2.0.tar.gz").write_bytes(b"")     # sdist alone does not count
    assert W.pkg_index_has("0.2.0") is False
    (tmp_path / "hugpy_fleet-0.2.0-py3-none-any.whl").write_bytes(b"")
    assert W.pkg_index_has("0.2.0") is True
    assert W.pkg_index_has("0.2.1") is False
    assert W.pkg_index_has(None) is False
    # a subset is enough when the caller names it
    assert W.pkg_index_has("0.2.0", names=["hugpy-fleet"]) is True


def test_pkg_index_has_is_false_for_a_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "pkg_index_dir", lambda: str(tmp_path / "nope"))
    assert W.pkg_index_has("0.2.0") is False
