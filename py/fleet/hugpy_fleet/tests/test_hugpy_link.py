"""hugpy-link daemon: the per-tick worker_state classification."""
from hugpy_fleet.link.hugpy_link import classify


def test_unit_transitions_are_restarting():
    for st in ("activating", "deactivating", "reloading"):
        assert classify(st, 5.0, "ok", 10.0) == "restarting"


def test_unit_failed_or_inactive_is_down():
    assert classify("failed", None, "refused", 1.0) == "down"
    assert classify("inactive", None, "ok", 1.0) == "down"


def test_fast_health_up_slow_health_busy():
    assert classify("active", 600.0, "ok", 150.0, busy_s=2.0) == "up"
    assert classify("active", 600.0, "ok", 2500.0, busy_s=2.0) == "busy"
    assert classify("active", 600.0, "timeout", 4000.0) == "busy"


def test_refused_inside_grace_is_restarting_after_is_down():
    assert classify("active", 10.0, "refused", 1.0, grace_s=60.0) == "restarting"
    assert classify("active", 120.0, "refused", 1.0, grace_s=60.0) == "down"
    assert classify("active", None, "error", 1.0) == "down"


def test_daemon_is_stdlib_only():
    import ast
    import pathlib
    import sys
    import hugpy_fleet.link.hugpy_link as m
    tree = ast.parse(pathlib.Path(m.__file__).read_text())
    mods = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods


def test_config_inherits_worker_unit_env():
    from hugpy_fleet.link.hugpy_link import config
    wenv = {"WORKER_CENTRAL_URL": "http://192.168.1.100:7002/", "WORKER_PORT": "9100",
            "WORKER_ENROLL_TOKEN": "tok"}
    assert config({}, wenv) == ("http://192.168.1.100:7002", "http://127.0.0.1:9100", "tok")
    assert config({"HUGPY_LINK_CENTRAL": "http://c:1"}, wenv)[0] == "http://c:1"
    assert config({}, {}) == ("http://127.0.0.1:7002", "http://127.0.0.1:9200", "")
