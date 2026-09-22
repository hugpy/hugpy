"""Uniform worker install — installer unit template + install-shape detector.

Covers the two slices of the make-worker-installs-uniform work:
  * Slice A: the canonical systemd user unit the installer writes carries the
    hardened bits (Wants=, Restart=on-failure, the full env block incl.
    HUGPY_ENGINE_DIR) — rendered without touching systemctl/loginctl.
  * Slice B: the heartbeat install-shape detector is fully defensive (a
    well-formed dict even when /proc/self/cgroup can't be read) and its
    canonical truth-table (both unit names accepted, wrong venv -> false).

Real pytest functions so `pytest tests/test_worker_install.py -q` reports a count.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import install as wi
from hugpy_fleet.worker import agent as wa


# --------------------------------------------------------------------------- #
# Slice A — the systemd user unit the installer writes                        #
# --------------------------------------------------------------------------- #
def _fake_opts():
    return SimpleNamespace(
        central="https://dev.hugpy.ai",
        name="testbox",
        port=9100,
        models=None,
        storage="/mnt/llm_storage",
        engine_dir=None,          # -> default %h/hugpy-worker/engine
        serve_mode="off",
        enroll_token="tok-123",
    )


def _render_unit(monkeypatch, tmp_path, opts):
    """Drive _install_systemd_user with systemctl/loginctl stubbed out and HOME
    pointed at a temp dir, then return the unit file's text."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USER", "tester")
    # Stub every subprocess call (daemon-reload, enable, loginctl enable-linger).
    monkeypatch.setattr(wi.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0))
    wi._install_systemd_user(opts)
    unit_path = tmp_path / ".config" / "systemd" / "user" / "hugpy-worker.service"
    return unit_path.read_text(encoding="utf-8")


def test_unit_has_wants_and_on_failure(monkeypatch, tmp_path):
    unit = _render_unit(monkeypatch, tmp_path, _fake_opts())
    assert "Wants=network-online.target" in unit
    assert "After=network-online.target" in unit
    assert "Restart=on-failure" in unit
    assert "RestartSec=5" in unit
    # The old template's Restart=always was WRONG for a deliberate block/revoke.
    assert "Restart=always" not in unit


def test_unit_env_block_is_complete(monkeypatch, tmp_path):
    unit = _render_unit(monkeypatch, tmp_path, _fake_opts())
    assert 'Environment="HUGPY_ENGINE_DIR=%h/hugpy-worker/engine"' in unit
    assert 'Environment="WORKER_CENTRAL_URL=https://dev.hugpy.ai"' in unit
    assert 'Environment="DEFAULT_SERVE_MODE=off"' in unit
    assert 'Environment="WORKER_ENROLL_TOKEN=tok-123"' in unit
    assert 'Environment="DEFAULT_ROOT=/mnt/llm_storage"' in unit
    # Description names the box.
    assert "Description=hugpy worker (testbox)" in unit


def test_engine_dir_flag_overrides_default(monkeypatch, tmp_path):
    opts = _fake_opts()
    opts.engine_dir = "/opt/llama-engine"
    unit = _render_unit(monkeypatch, tmp_path, opts)
    assert 'Environment="HUGPY_ENGINE_DIR=/opt/llama-engine"' in unit
    assert "%h/hugpy-worker/engine" not in unit


# --------------------------------------------------------------------------- #
# Slice B — install-shape detector                                            #
# --------------------------------------------------------------------------- #
def test_install_shape_wellformed_when_cgroup_unreadable(monkeypatch):
    """A /proc/self/cgroup read failure must NOT break the heartbeat: the
    detector still returns a well-formed dict with every key."""
    wa._INSTALL_SHAPE = None  # bust the once-per-process cache
    real_open = open

    def boom(path, *a, **k):
        if str(path) == "/proc/self/cgroup":
            raise OSError("simulated: cannot read cgroup")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", boom)
    shape = wa._install_shape()
    assert set(shape) == {"unit", "via_systemd", "venv", "python", "canonical"}
    assert shape["unit"] is None            # cgroup unreadable -> no unit
    assert shape["canonical"] is False      # can't be canonical without a unit
    assert shape["venv"] == sys.prefix
    assert shape["python"] == sys.executable
    wa._INSTALL_SHAPE = None                # don't leak the cached value


def test_detect_systemd_unit_parses_last_service(monkeypatch):
    import io
    cgroup = ("0::/user.slice/user-1000.slice/user@1000.service"
              "/app.slice/hugpy-worker.service\n")
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: (io.StringIO(cgroup)
                                            if str(p) == "/proc/self/cgroup"
                                            else real_open(p, *a, **k)))
    assert wa._detect_systemd_unit() == "hugpy-worker.service"


def test_detect_systemd_unit_none_when_no_service(monkeypatch):
    import io
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: io.StringIO("0::/user.slice\n"))
    assert wa._detect_systemd_unit() is None


def test_canonical_truth_table():
    good_venv = "/home/op/hugpy-worker/venv"
    good_py = "/home/op/hugpy-worker/venv/bin/python"

    def shape(**kw):
        base = dict(invocation_id="abc123", unit="hugpy-worker.service",
                    prefix=good_venv, executable=good_py)
        base.update(kw)
        return wa._compute_install_shape(**base)

    # Canonical: systemd + canonical unit + hugpy-worker/venv.
    assert shape()["canonical"] is True
    assert shape()["via_systemd"] is True
    # Legacy alias is ALSO canonical.
    assert shape(unit="abstract-hugpy-worker.service")["canonical"] is True
    # Trailing slash on the venv path still matches (rstrip).
    assert shape(prefix=good_venv + "/")["canonical"] is True
    # Wrong venv -> not canonical.
    assert shape(prefix="/usr")["canonical"] is False
    # Not under systemd (no INVOCATION_ID) -> not canonical, via_systemd False.
    s = shape(invocation_id=None)
    assert s["canonical"] is False and s["via_systemd"] is False
    # Unknown unit name -> not canonical (even under systemd, right venv).
    assert shape(unit="some-other.service")["canonical"] is False
    # No unit detected at all -> not canonical.
    assert shape(unit=None)["canonical"] is False
