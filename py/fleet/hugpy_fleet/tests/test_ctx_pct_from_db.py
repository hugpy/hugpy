"""GUARD (2026-10-02): the worker's context % comes from the DB pair row that
central overlays into ``spill_by_model`` on the heartbeat reply. Before this,
Qwythos-9B (ctx_pct 1 = 2,048 tokens in the console) was refused at
"KV 64.0 GB at ctx 262,144 (loader default)" because the fit gate read only
the local settings file the legacy /assign relay filled."""
from hugpy_fleet.worker import agent as A


class _State:
    pass


def _reset():
    A._RUNTIME_SETTINGS.pop("ctx_pct_db", None)
    A._RUNTIME_SETTINGS.pop("ctx_pct", None)


def test_db_ctx_pct_from_heartbeat_wins_over_local_settings():
    _reset()
    A._RUNTIME_SETTINGS["ctx_pct"] = {"M": 50, "Stale": 40}
    assert A._ctx_pct("M") == 50                      # pre-DB central: local still applies
    A._adopt_storage_inputs(_State(), {"spill_by_model": {"M": {"ctx_pct": 1}, "Auto": {"gguf_file": "x"}}})
    assert A._ctx_pct("M") == 1                       # the DB value
    assert A._ctx_pct("Auto") is None                 # in the map without ctx_pct = auto
    assert A._ctx_pct("Stale") is None                # a local-only value no longer leaks in
    _reset()


def test_flex_floor_still_wins():
    _reset()
    A._adopt_storage_inputs(_State(), {"spill_by_model": {"M": {"ctx_pct": 1}}})
    A._FLEX_CTX_FLOOR["M"] = 3
    try:
        assert A._ctx_pct("M") == 3
    finally:
        A._FLEX_CTX_FLOOR.pop("M", None)
        _reset()
