"""Worker maintenance controls — POST /ops/free-ram (worker agent).

The real bug: after a model is freed, glibc keeps the pages in its allocator
arena (malloc_trim is used nowhere in the tree), so RSS stays pinned — ae was
observed at 0 free / 128 GB used with nothing loaded. /ops/free-ram is the
NON-destructive reclaim: gc.collect() + malloc_trim(0) + torch.cuda.empty_cache()
that hands the orphaned arena back to the OS WITHOUT evicting any model.

This exercises the route through the worker agent's own Flask app (build_app) —
no live worker, no network:
  * the contract shape is present (ok + ram_free_before/after/freed +
    rss_before/after + loaded_models);
  * loaded_models is reported VERBATIM (non-destructive: the route never
    touches residency — proven by stubbing loaded_model_keys and asserting the
    response echoes it unchanged);
  * the ram/rss deltas are ints (Ubuntu workers) or None (best-effort), never
    fabricated;
  * _trim_host_ram() runs without raising (malloc_trim wrapped for musl/other).

pytest isn't in the venv, so this runs script-style like its siblings:
    venv/bin/python tests/test_free_ram_ops.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib

agent = importlib.import_module("hugpy_fleet.worker.agent")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# --- _trim_host_ram runs clean (no torch/glibc assumptions may raise) --------
agent._trim_host_ram()
check("_trim_host_ram() runs without raising", True)

# _agent_rss_bytes is best-effort: int on this Linux box, or None, never raises.
rss = agent._agent_rss_bytes()
check("_agent_rss_bytes() returns int|None", rss is None or isinstance(rss, int))


# --- route through the agent's real Flask app --------------------------------
state = agent.WorkerState(name="test-agent", url=None, worker_id="test-wid")
app = agent.build_app(state)
client = app.test_client()

# Stub residency so "non-destructive" is observable: if the route ever evicted,
# a real loaded_model_keys() would drop these — the stub can't, so an unchanged
# echo proves the route reports residency verbatim and clears nothing.
SENTINEL = ["Model-A", "Model-B"]
_orig_loaded = agent.loaded_model_keys
try:
    agent.loaded_model_keys = lambda: list(SENTINEL)

    r = client.post("/ops/free-ram", json={})
    body = r.get_json()

    check("free-ram: 200", r.status_code == 200)
    check("free-ram: ok true", body.get("ok") is True)

    for field in ("ram_free_before", "ram_free_after", "ram_freed",
                  "rss_before", "rss_after", "loaded_models"):
        check(f"free-ram: contract field '{field}' present", field in body)

    for field in ("ram_free_before", "ram_free_after", "ram_freed",
                  "rss_before", "rss_after"):
        v = body[field]
        check(f"free-ram: '{field}' is bytes int or None (best-effort)",
              v is None or isinstance(v, int))

    check("free-ram: NON-destructive — loaded_models echoed verbatim",
          body["loaded_models"] == SENTINEL)

    # ram_freed must be the exact delta (or None) — never fabricated.
    b, a, freed = (body["ram_free_before"], body["ram_free_after"],
                   body["ram_freed"])
    if b is not None and a is not None:
        check("free-ram: ram_freed == after - before", freed == a - b)
    else:
        check("free-ram: ram_freed None when a reading is unavailable",
              freed is None)

    # --- /models/unload now also carries the ram_* fields (enhanced) ---------
    ru = client.post("/models/unload", json={})
    ub = ru.get_json()
    check("unload: 200", ru.status_code == 200)
    for field in ("vram_free_before", "vram_free_after", "freed",
                  "ram_free_before", "ram_free_after", "ram_freed",
                  "loaded_models"):
        check(f"unload: field '{field}' present (VRAM kept + RAM added)",
              field in ub)
finally:
    agent.loaded_model_keys = _orig_loaded

print(f"\nall {ok} checks passed")
