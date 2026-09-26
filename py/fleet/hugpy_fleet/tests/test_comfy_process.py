"""The worker's comfy process manager (2026-09-24) — the worker OWNS comfy's
lifecycle: pluggable launcher, on-demand start with a bounded readiness wait,
stop, and checkpoints advertised while stopped.

Every box-touching seam (systemctl, /system_stats readiness, Popen, the clock)
is injected — no systemd, no ComfyUI, no GPU.
"""
import pytest

from hugpy_fleet.worker import comfy_process as cp
from hugpy_fleet.worker.comfy_process import ComfyManager, parse_launch


# ── launcher selection / defaults ───────────────────────────────────────────
def test_parse_launch_default_is_external():
    assert parse_launch(None) == {"kind": "external"}
    assert parse_launch("") == {"kind": "external"}
    assert parse_launch("  ") == {"kind": "external"}
    assert parse_launch("external") == {"kind": "external"}
    assert parse_launch("EXTERNAL") == {"kind": "external"}


def test_parse_launch_systemd_user():
    assert parse_launch("systemd-user:comfyui.service") == {
        "kind": "systemd-user", "unit": "comfyui.service"}


def test_parse_launch_spawn_splits_python_main_args():
    spec = parse_launch("spawn:/v/bin/python /app/main.py --listen 0.0.0.0 --port 8188")
    assert spec == {"kind": "spawn", "python": "/v/bin/python",
                    "main": "/app/main.py",
                    "args": ["--listen", "0.0.0.0", "--port", "8188"]}


def test_parse_launch_malformed_degrades_to_external_with_reason():
    for bad in ("systemd-user:", "spawn:onlyone", "banana:foo"):
        spec = parse_launch(bad)
        assert spec["kind"] == "external"
        assert spec.get("error")            # WHY it was ignored is recorded


def test_external_launcher_is_not_managed_and_never_starts_or_stops():
    mgr = ComfyManager({"kind": "external"}, "http://x:8188",
                       readiness=lambda url, t=2.0: False)
    assert mgr.managed is False
    started = mgr.start()
    assert started["ok"] is False
    assert started["launcher"] == "external"
    stopped = mgr.stop()
    assert stopped["ok"] is False
    assert "does not stop" in stopped["note"]


def test_systemd_and_spawn_are_managed():
    assert ComfyManager({"kind": "systemd-user", "unit": "c.service"}, "u").managed
    assert ComfyManager({"kind": "spawn", "python": "p", "main": "m", "args": []},
                        "u").managed


# ── on-demand start: readiness + failure envelopes ──────────────────────────
class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_start_success_when_comfy_becomes_ready():
    clock = Clock()
    ready = {"v": False}
    calls = []

    def runner(cmd, timeout=30.0, env=None):
        calls.append(cmd)
        return 0, "", ""              # systemctl start accepted

    def readiness(url, t=2.0):
        return ready["v"]

    def sleep(_s):
        clock.advance(1.0)
        ready["v"] = True             # comfy comes up on the next poll

    mgr = ComfyManager({"kind": "systemd-user", "unit": "c.service"},
                       "http://x:8188", runner=runner, readiness=readiness,
                       clock=clock, sleep=sleep)
    res = mgr.start(ready_timeout=30.0)
    assert res["ok"] is True
    assert calls and calls[0][:3] == ["systemctl", "--user", "start"]


def test_start_launch_failure_is_a_recorded_load_failure_with_real_reason():
    def runner(cmd, timeout=30.0, env=None):
        return 1, "", "Failed to start c.service: Unit not found."

    mgr = ComfyManager({"kind": "systemd-user", "unit": "c.service"},
                       "http://x:8188", runner=runner,
                       readiness=lambda url, t=2.0: False)
    res = mgr.start(ready_timeout=5.0)
    assert res["ok"] is False
    assert res["load_class"] == "engine_unavailable"
    # the loader_stderr is the launcher's OWN words, verbatim
    assert "Unit not found" in res["loader_stderr"]


def test_start_readiness_timeout_yields_unreachable_load_failure_with_journal():
    clock = Clock()

    def runner(cmd, timeout=30.0, env=None):
        if cmd[:2] == ["journalctl", "--user"] or "journalctl" in cmd[0]:
            return 0, "traceback: CUDA out of memory during comfy init", ""
        if "journalctl" in " ".join(cmd):
            return 0, "traceback: CUDA out of memory during comfy init", ""
        return 0, "", ""             # start accepted, but never ready

    def sleep(_s):
        clock.advance(10.0)          # burn the readiness ceiling fast

    mgr = ComfyManager({"kind": "systemd-user", "unit": "c.service"},
                       "http://x:8188", runner=runner,
                       readiness=lambda url, t=2.0: False, clock=clock, sleep=sleep)
    res = mgr.start(ready_timeout=30.0)
    assert res["ok"] is False
    assert res["load_class"] == "unreachable"
    assert "CUDA out of memory" in (res.get("loader_stderr") or "")


def test_start_already_running_is_a_noop_ok():
    mgr = ComfyManager({"kind": "systemd-user", "unit": "c.service"},
                       "http://x:8188",
                       runner=lambda *a, **k: (0, "", ""),
                       readiness=lambda url, t=2.0: True)
    res = mgr.start()
    assert res["ok"] is True
    assert res["note"] == "comfy already running"


def test_spawn_child_exits_before_readiness_is_a_hard_failure():
    class DeadChild:
        pid = 4321

        def __init__(self):
            import io
            self.stderr = io.BytesIO(b"ImportError: No module named comfy")

        def poll(self):
            return 1                  # exited immediately

    clock = Clock()
    mgr = ComfyManager({"kind": "spawn", "python": "/v/py", "main": "/a/main.py",
                        "args": []}, "http://x:8188",
                       popen=lambda *a, **k: DeadChild(),
                       readiness=lambda url, t=2.0: False,
                       clock=clock, sleep=lambda s: clock.advance(1.0))
    res = mgr.start(ready_timeout=30.0)
    assert res["ok"] is False
    assert res["load_class"] == "engine_unavailable"
    assert "ImportError" in res["loader_stderr"]


# ── checkpoints advertised while stopped ────────────────────────────────────
def test_checkpoints_on_disk_scans_configured_dirs(tmp_path):
    (tmp_path / "sd15.safetensors").write_bytes(b"x")
    (tmp_path / "sdxl.ckpt").write_bytes(b"y")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "turbo.safetensors").write_bytes(b"z")
    (tmp_path / "notes.txt").write_bytes(b"nope")

    mgr = ComfyManager({"kind": "systemd-user", "unit": "c.service"}, "u",
                       checkpoint_dirs=lambda: [str(tmp_path)])
    ckpts = mgr.checkpoints_on_disk()
    assert set(ckpts) == {"sd15.safetensors", "sdxl.ckpt", "turbo.safetensors"}


def test_state_reports_stopped_running_and_failed():
    up = {"v": True}
    mgr = ComfyManager({"kind": "systemd-user", "unit": "c.service"}, "u",
                       readiness=lambda url, t=2.0: up["v"])
    assert mgr.state() == "running"
    up["v"] = False
    assert mgr.state() == "stopped"
    mgr._last_error = "boom"
    assert mgr.state() == "failed"
