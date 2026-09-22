"""_ram_summary / _vram_summary — central honest-bar wiring + legacy degrade
(t13/t14). These pure functions turn the worker's raw heartbeat fields into the
spec bar (spill.budget_bar), and MUST degrade gracefully for pre-slice workers
(absent fields → today's numbers + bar_semantics="legacy" so the UI labels
honestly instead of guessing).

Runs like the other tests here: venv/bin/python tests/test_bar_summary_central.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-bar-summary-test-")

import importlib
wk = importlib.import_module(
    "hugpy_fleet.central.workers")

G = 2 ** 30
ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# ── RAM: honest bar from the new fields, against a central ceiling ────────────
# 128 physical, limit 90, worker 20, external 10 → bar_used 20, remaining 70.
w = {
    "ram_total": 128 * G,
    "free_ram": 5 * G,                 # the OLD (clamped) field — must NOT drive the bar now
    "ram_worker_bytes": 20 * G,
    "ram_external_bytes": 10 * G,
    "limits": {"ram_max_gib": 90},
}
r = wk._ram_summary(w)
check("RAM: semantics central (prefixed key, collision-free)",
      r["ram_bar_semantics"] == "central")
check("RAM: generic bar_semantics also written (wire-compat)",
      r["bar_semantics"] == "central")
check("RAM: ram_used is the SPEC bar_used (20), not physical−free artifact",
      r["ram_used"] == 20 * G)
check("RAM: bar_total = limit 90", r["bar_total"] == 90 * G)
check("RAM: bar_remaining 70", r["bar_remaining"] == 70 * G)
check("RAM: encroachment 0", r["bar_encroachment"] == 0)
check("RAM: not over limit", r["bar_over_limit"] is False)
check("RAM: ram_total preserved", r["ram_total"] == 128 * G)

# ── RAM: over-limit box surfaces the flag + clamps, keeps raw ─────────────────
w2 = {"ram_total": 128 * G, "ram_worker_bytes": 100 * G, "ram_external_bytes": 0,
      "limits": {"ram_max_gib": 90}}
r2 = wk._ram_summary(w2)
check("RAM over: ram_used clamped to limit 90", r2["ram_used"] == 90 * G)
check("RAM over: bar_remaining 0", r2["bar_remaining"] == 0)
check("RAM over: over_limit true", r2["bar_over_limit"] is True)
check("RAM over: raw_used kept (100 > 90)", r2["bar_raw_used"] == 100 * G)
check("RAM over: over_by 10", r2["bar_over_by"] == 10 * G)

# ── RAM: no ceiling → physical semantics ──────────────────────────────────────
w3 = {"ram_total": 128 * G, "ram_worker_bytes": 20 * G, "ram_external_bytes": 10 * G,
      "limits": {}}
r3 = wk._ram_summary(w3)
check("RAM no-limit: semantics physical", r3["bar_semantics"] == "physical")
check("RAM no-limit: ram_used = worker+external (plain)", r3["ram_used"] == 30 * G)
check("RAM no-limit: bar_total = physical", r3["bar_total"] == 128 * G)

# ── RAM: LEGACY degrade — pre-slice worker (no ram_worker_bytes) ──────────────
wl = {"ram_total": 124 * G, "free_ram": 96 * G, "limits": {"ram_max_gib": 90}}
rl = wk._ram_summary(wl)
check("RAM legacy: flagged legacy", rl["ram_bar_semantics"] == "legacy")
check("RAM legacy: keeps OLD ram_used = total−free (the acknowledged artifact)",
      rl["ram_used"] == (124 - 96) * G)
check("RAM legacy: no bar_* numbers fabricated", "bar_used" not in rl)

# ── VRAM: honest bar from pid_registry attributed split ───────────────────────
# driver total 24G, free 6G → used 18G; attributed models 12G; limit 20G.
# external = used − attributed = 6G; headroom = 24−20 = 4G → encroach 2G →
# bar_used = 12+2 = 14, remaining = 20−14 = 6.
wv = {
    "gpus": [{"name": "RTX 3090", "memory_total": 24 * G, "memory_free": 6 * G}],
    "vram_attributed_bytes": 12 * G,
    "vram_unattributed_bytes": 3 * G,
    "limits": {"gpu_mem_gib": 20},
}
rv = wk._vram_summary(wv)
check("VRAM: driver totals preserved (physical truth)",
      rv["vram_total"] == 24 * G and rv["vram_used"] == 18 * G)
check("VRAM: semantics central (prefixed)", rv["vram_bar_semantics"] == "central")
check("VRAM: worker_usage = attributed 12", rv["vram_bar_worker_usage"] == 12 * G)
check("VRAM: external = used−attributed = 6", rv["vram_bar_external_usage"] == 6 * G)
check("VRAM: encroachment = 6−4 = 2", rv["vram_bar_encroachment"] == 2 * G)
check("VRAM: bar_used = 14", rv["vram_bar_used"] == 14 * G)
check("VRAM: bar_remaining = 6", rv["vram_bar_remaining"] == 6 * G)

# ── VRAM: LEGACY degrade — no attributed split reported ───────────────────────
wvl = {"gpus": [{"name": "GPU", "memory_total": 24 * G, "memory_free": 6 * G}]}
rvl = wk._vram_summary(wvl)
check("VRAM legacy: flagged legacy", rvl["vram_bar_semantics"] == "legacy")
check("VRAM legacy: driver figures intact", rvl["vram_used"] == 18 * G)
check("VRAM legacy: no vram_bar_* numbers", "vram_bar_used" not in rvl)

# ── VRAM: no GPU at all ───────────────────────────────────────────────────────
rvn = wk._vram_summary({"gpus": []})
check("VRAM no-gpu: legacy + all None", rvn["vram_bar_semantics"] == "legacy"
      and rvn["vram_total"] is None)

# ── _public_view: RAM+VRAM spread together — prefixed keys DON'T collide ───────
# Both summaries write generic bar_* (RAM wins, applied second), but their
# PREFIXED keys must both survive so the UI can read each independently.
full = dict(w)              # the RAM honest case (limit 90, worker 20, ext 10)
full.update({
    "gpus": [{"name": "RTX 3090", "memory_total": 24 * G, "memory_free": 6 * G}],
    "vram_attributed_bytes": 12 * G,
    "vram_unattributed_bytes": 3 * G,
    "limits": {"ram_max_gib": 90, "gpu_mem_gib": 20},
})
pv = wk._public_view(full)
check("public_view: RAM prefixed bar survives (ram_used = 20)",
      pv["ram_bar_used"] == 20 * G and pv["ram_bar_semantics"] == "central")
check("public_view: VRAM prefixed bar survives (14)",
      pv["vram_bar_used"] == 14 * G and pv["vram_bar_semantics"] == "central")
check("public_view: the two prefixed bars are independent (20 ≠ 14)",
      pv["ram_bar_used"] != pv["vram_bar_used"])

print(f"\nall {ok} checks passed")
