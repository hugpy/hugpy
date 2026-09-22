"""k94 — the install-engine "not taking" fix, unit-level.

Three seams under test (no network, no GPU, no real llama.cpp):

  * PERSISTENCE + RESOLUTION ORDER — `hugpy install-engine` records where it
    landed in the box's persisted settings file (the /ops/config store:
    ``<WORKER_ID_FILE>.settings.json``); engine.resolve consults it between the
    env overrides (which still win absolutely) and the per-user data dir, and a
    stale record is ignored-with-reason, never fatal.
  * LIB-DIR DERIVATION — a binary resolved from ANY hugpy-managed engine dir
    (env-pinned, persisted, or the default data dir) gets its sibling lib dirs
    onto a child's LD_LIBRARY_PATH; a system/PATH binary stays untouched.
  * HEALTH SHAPE — native_engine_status() reports
    {found, path, source, spawn_ok, error} with the spawn probe monkeypatched
    and TTL-cached, plus the agent-side wrappers that carry it to /health.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_engine_persisted_resolve.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest

# (module-level logging.disable removed: it leaked into every later test under pytest)

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_engine.native import resolve


def _mk_bin(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho 'version: 1 (test)'\n")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def iso(monkeypatch, tmp_path):
    """A hermetic box: no engine env overrides, controlled data dir / id file /
    PATH. env_value normally prefers the box's file-based secrets store (which
    on this dev box really pins HUGPY_ENGINE_DIR), so it is stubbed to plain
    os.environ at both consumer modules — the resolution logic under test is
    ours, not the store's."""
    from hugpy_platform import app_dirs as _paths
    _env = lambda name: os.environ.get(name) or None  # noqa: E731
    monkeypatch.setattr(resolve, "env_value", _env)
    monkeypatch.setattr(_paths, "env_value", _env)
    monkeypatch.chdir(tmp_path)
    for key in ("HUGPY_ENGINE_DIR", "LLAMA_CPP_DIR", "LLAMA_SERVER_BIN",
                "WORKER_RPC_BIN", "LLAMA_RPC_BIN", "LLAMA_CLI_BIN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HUGPY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WORKER_ID_FILE", str(tmp_path / "worker.json"))
    pathbin = tmp_path / "pathbin"
    pathbin.mkdir()
    monkeypatch.setenv("PATH", str(pathbin))
    resolve._stale_warned.clear()
    resolve._spawn_probe_cache.clear()
    return tmp_path


# ---------------------------------------------------------------------------
# Persistence: settings-file round-trip
# ---------------------------------------------------------------------------
def test_persist_roundtrip_preserves_other_settings(iso):
    settings = iso / "worker.json.settings.json"
    settings.write_text(json.dumps({"slot_count": 2}))
    wrote = resolve.persist_install(str(iso / "eng"), str(iso / "eng" / "llama-server"))
    assert wrote == str(settings)
    data = json.loads(settings.read_text())
    assert data["slot_count"] == 2                      # /ops/config keys survive
    assert data["engine"] == {"dir": str(iso / "eng"),
                              "server_bin": str(iso / "eng" / "llama-server")}
    assert resolve.persisted_engine() == data["engine"]


def test_persisted_engine_absent_or_garbage_is_empty(iso):
    assert resolve.persisted_engine() == {}
    (iso / "worker.json.settings.json").write_text("not json {")
    assert resolve.persisted_engine() == {}


# ---------------------------------------------------------------------------
# Resolution order: env bin -> env dir -> persisted -> data dir -> PATH
# ---------------------------------------------------------------------------
def test_env_bin_override_wins_over_everything(iso, monkeypatch):
    envbin = _mk_bin(iso / "envpin" / "llama-server")
    databin = _mk_bin(iso / "data" / "engine" / "llama-server")
    resolve.persist_install(str(iso / "eng"),
                            _mk_bin(iso / "eng" / "llama-server"))
    monkeypatch.setenv("LLAMA_SERVER_BIN", envbin)
    assert resolve.server_bin() == envbin
    status = resolve.native_engine_status(probe=False)
    assert (status["path"], status["source"]) == (envbin, "env")
    assert databin  # data-dir install present but outranked


def test_env_pinned_dir_wins_over_persisted(iso, monkeypatch):
    pinned = _mk_bin(iso / "pinned" / "bin" / "llama-server")
    resolve.persist_install(str(iso / "eng"),
                            _mk_bin(iso / "eng" / "llama-server"))
    monkeypatch.setenv("HUGPY_ENGINE_DIR", str(iso / "pinned"))
    assert resolve.server_bin() == pinned
    assert resolve.native_engine_status(probe=False)["source"] == "env"


def test_persisted_config_beats_data_dir_and_path(iso):
    engbin = _mk_bin(iso / "eng" / "llama-server")
    _mk_bin(iso / "data" / "engine" / "llama-server")
    _mk_bin(iso / "pathbin" / "llama-server")
    resolve.persist_install(str(iso / "eng"), engbin)
    assert resolve.server_bin() == engbin
    assert resolve.native_engine_status(probe=False)["source"] == "config"


def test_persisted_dir_resolves_sibling_binaries(iso):
    # rpc-server has no exact recorded path — it resolves via the persisted dir.
    rpc = _mk_bin(iso / "eng" / "build" / "bin" / "rpc-server")
    resolve.persist_install(str(iso / "eng"),
                            _mk_bin(iso / "eng" / "build" / "bin" / "llama-server"))
    assert resolve.rpc_bin() == rpc


def test_stale_persisted_record_falls_through_with_reason(iso):
    databin = _mk_bin(iso / "data" / "engine" / "llama-server")
    resolve.persist_install(str(iso / "gone"), str(iso / "gone" / "llama-server"))
    status = resolve.native_engine_status(probe=False)
    assert (status["path"], status["source"]) == (databin, "data_dir")
    # both stale fields ignored-with-reason, warned once per process
    assert str(iso / "gone" / "llama-server") in resolve._stale_warned
    assert str(iso / "gone") in resolve._stale_warned
    before = set(resolve._stale_warned)
    resolve.server_bin()
    assert resolve._stale_warned == before


def test_data_dir_then_path_fallback(iso):
    pathbin = _mk_bin(iso / "pathbin" / "llama-server")
    assert resolve.native_engine_status(probe=False)["source"] == "path"
    assert resolve.server_bin() == pathbin
    databin = _mk_bin(iso / "data" / "engine" / "llama-server")
    assert resolve.server_bin() == databin
    assert resolve.native_engine_status(probe=False)["source"] == "data_dir"


def test_absent_everywhere_is_none(iso):
    status = resolve.native_engine_status(probe=False)
    assert status == {"found": False, "path": None, "source": None,
                      "spawn_ok": None, "error": None}


# ---------------------------------------------------------------------------
# Lib-dir derivation: managed vs system binaries
# ---------------------------------------------------------------------------
def test_lib_dirs_for_managed_data_dir_install(iso):
    databin = _mk_bin(iso / "data" / "engine" / "bin" / "llama-server")
    (iso / "data" / "engine" / "lib").mkdir()
    dirs = resolve.engine_lib_dirs(databin)
    assert str(iso / "data" / "engine" / "lib") in dirs
    assert str(iso / "data" / "engine" / "bin") in dirs
    # no env pin -> the no-arg (historical) form still adds nothing
    assert resolve.engine_lib_dirs() == []
    ld = resolve.ld_library_path_with_engine("/usr/lib", bin_path=databin)
    assert ld.split(os.pathsep)[-1] == "/usr/lib"
    assert str(iso / "data" / "engine" / "lib") in ld.split(os.pathsep)


def test_lib_dirs_for_persisted_install(iso):
    engbin = _mk_bin(iso / "eng" / "build" / "bin" / "llama-server")
    resolve.persist_install(str(iso / "eng"), engbin)
    assert str(iso / "eng" / "build" / "bin") in resolve.engine_lib_dirs(engbin)


def test_system_path_binary_left_untouched(iso):
    pathbin = _mk_bin(iso / "pathbin" / "llama-server")
    assert resolve.engine_lib_dirs(pathbin) == []
    assert resolve.ld_library_path_with_engine("/usr/lib", bin_path=pathbin) == "/usr/lib"
    assert resolve.ld_library_path_with_engine(None, bin_path=pathbin) is None


def test_env_pinned_lib_dirs_unchanged(iso, monkeypatch):
    # A hand-pinned box behaves exactly as today: dirs derive from the pin,
    # with or without a bin_path.
    _mk_bin(iso / "pinned" / "bin" / "llama-server")
    monkeypatch.setenv("HUGPY_ENGINE_DIR", str(iso / "pinned"))
    dirs = resolve.engine_lib_dirs()
    assert str(iso / "pinned" / "bin") in dirs
    ld = resolve.ld_library_path_with_engine(None)
    assert str(iso / "pinned" / "bin") in ld.split(os.pathsep)


# ---------------------------------------------------------------------------
# native_engine_status: probe wiring + TTL cache
# ---------------------------------------------------------------------------
def test_native_engine_status_probe_shape_and_ttl(iso, monkeypatch):
    databin = _mk_bin(iso / "data" / "engine" / "llama-server")
    calls = []
    monkeypatch.setattr(resolve, "_spawn_probe",
                        lambda p: calls.append(p) or (True, None))
    status = resolve.native_engine_status()
    assert set(status) == {"found", "path", "source", "spawn_ok", "error"}
    assert status == {"found": True, "path": databin, "source": "data_dir",
                      "spawn_ok": True, "error": None}
    resolve.native_engine_status()
    assert calls == [databin]                       # second hit served from TTL cache


def test_native_engine_status_reports_spawn_failure(iso, monkeypatch):
    _mk_bin(iso / "data" / "engine" / "llama-server")
    monkeypatch.setattr(
        resolve, "_spawn_probe",
        lambda p: (False, "exit 127: libggml.so: cannot open shared object file"))
    status = resolve.native_engine_status()
    assert status["found"] is True and status["spawn_ok"] is False
    assert "libggml.so" in status["error"]


def test_real_spawn_probe_runs_the_binary(iso):
    databin = _mk_bin(iso / "data" / "engine" / "llama-server")   # echoes version:
    ok, err = resolve._spawn_probe(databin)
    assert (ok, err) == (True, None)
    broken = iso / "data" / "engine" / "broken"
    broken.write_text("#!/nonexistent-interpreter\n")
    broken.chmod(0o755)
    ok, err = resolve._spawn_probe(str(broken))
    assert ok is False and err


# ---------------------------------------------------------------------------
# install persistence seam (fetch._resolved) — managed recorded, PATH not
# ---------------------------------------------------------------------------
def test_install_result_persists_managed_binary(iso):
    from hugpy_engine.native import fetch
    _mk_bin(iso / "data" / "engine" / "llama-server")
    info = fetch._resolved(note="test")
    assert info["persisted_to"] == str(iso / "worker.json.settings.json")
    assert resolve.persisted_engine()["server_bin"] == info["server_bin"]


def test_install_result_skips_system_path_binary(iso):
    from hugpy_engine.native import fetch
    _mk_bin(iso / "pathbin" / "llama-server")
    info = fetch._resolved(note="test")
    assert info["persisted_to"] is None
    assert resolve.persisted_engine() == {}


# ---------------------------------------------------------------------------
# agent-side carriers: /health + the "engine" dict central snapshots
# ---------------------------------------------------------------------------
def test_worker_agent_health_carriers(iso, monkeypatch):
    import importlib
    agent = importlib.import_module("hugpy_fleet.worker.agent")
    sentinel = {"found": True, "path": "/x/llama-server", "source": "config",
                "spawn_ok": True, "error": None}
    monkeypatch.setattr(resolve, "native_engine_status", lambda probe=True: sentinel)
    assert agent._native_engine_status_safe() == sentinel
    monkeypatch.setattr(agent, "_probe_llama_cpp_subprocess",
                        lambda: {"installed": False, "error": "no llama_cpp"})
    monkeypatch.setattr(agent, "_LLAMA_PROBE_CACHE", None)
    engine = agent.llama_cpp_cuda_status()
    assert engine["native_engine"] == sentinel
    assert engine["installed"] is False              # python-binding fields intact


def test_worker_agent_safe_wrapper_never_raises(iso, monkeypatch):
    import importlib
    agent = importlib.import_module("hugpy_fleet.worker.agent")
    monkeypatch.setattr(resolve, "native_engine_status",
                        lambda probe=True: (_ for _ in ()).throw(RuntimeError("boom")))
    out = agent._native_engine_status_safe()
    assert out["found"] is False and "boom" in out["error"]


def test_gguf_worker_status_carries_native_engine(iso, monkeypatch):
    import importlib
    gguf = importlib.import_module("hugpy_fleet.gguf_worker.agent")
    sentinel = {"found": False, "path": None, "source": None,
                "spawn_ok": None, "error": None}
    monkeypatch.setattr(resolve, "native_engine_status", lambda probe=True: sentinel)
    assert gguf.llama_cpp_status()["native_engine"] == sentinel


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
