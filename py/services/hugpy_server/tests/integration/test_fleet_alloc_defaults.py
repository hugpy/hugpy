"""k37 CAPABILITY-AWARE DEFAULT ALLOC MODE (operator ruling 2026-07-24).

Two halves of the operator ask:
  1. the BLANK default is derived per (model x worker) by FEASIBILITY, engine-
     aware: fits-the-card-whole -> gpu-only (operator default order 2026-07-31:
     prefer full-card residency); oversized/unknown keep their spill-capable
     leaves (GGUF degrade -> max-gpu; a transformers model too big for the
     box's GPU but fitting RAM -> ram-only ON THAT WORKER — the only feasible
     option, so it IS the default, and it must SERVE — defaults-are-promises);
  2. the SELECTABLE set is bounded to what's feasible (feasible_modes) so an
     infeasible mode is never offered — enforced at /assign (and the bulk path)
     with an honest numbers-naming 409/skip, surfaced on the serving row.

Covers:
  * feasible_default_mode matrix (oversized gguf -> max-gpu, fits-whole ->
    gpu-only, transformers 68/24/124 -> ram-only, 5/24 -> gpu-only,
    200/24/124 -> max-gpu (fits neither), unknown -> max-gpu);
  * feasible_modes matrix per engine (incl. the 68/24/124 case -> exactly
    (ram-only, max-gpu, max-ram); only gpu-only is eliminated for an oversized
    transformers model — max-gpu is a spill PREFERENCE, not a whole-card
    requirement (operator ruling 2026-07-29); unknown
    data -> coarse trio + max-ram feasible; max-ram is engine-AGNOSTIC as of
    2026-07-24, only explicit stays engine-gated for non-GGUF);
  * an explicit persisted mode wins over derivation;
  * the emission seam: spill_for emits {"n_gpu_layers":"off"} for the oversized-
    transformers blank case, {} for the GGUF/fitting blank case, and leaves a
    persisted spill untouched;
  * the /assign feasibility gate refuses an infeasible mode (max-gpu on the
    68/24 case) with a numbers-naming reason; the bulk path skips it.

Run:  venv/bin/python tests/test_alloc_defaults.py
"""
import os
import sys
import tempfile

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-alloc-defaults-test-")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

GIB = 2 ** 30

from hugpy_engine.alloc_modes import (
    feasible_default_mode,
    feasible_modes,
    is_gguf_engine,
    ALLOC_MODES,
    derive_alloc_mode,
    _GPU_FIT_HEADROOM,
)

# ── engine classification ────────────────────────────────────────────────────
check("gguf / llama_cpp are the GGUF family",
      is_gguf_engine("gguf") and is_gguf_engine("llama_cpp")
      and is_gguf_engine("GGUF"))
check("transformers / comfy / None are NOT GGUF",
      not is_gguf_engine("transformers") and not is_gguf_engine("comfy")
      and not is_gguf_engine(None))

# ── feasible_default_mode matrix ─────────────────────────────────────────────
check("gguf any size -> max-gpu (partial offload universal), even oversized",
      feasible_default_mode("gguf", 200 * GIB, 24 * GIB, 124 * GIB) == "max-gpu"
      and feasible_default_mode("llama_cpp", 500 * GIB, 8 * GIB, 16 * GIB)
      == "max-gpu")
# the operator's headline case: 68 GB transformers, 24 GB GPU, 124 GB RAM
check("transformers 68GB / 24GB GPU / 124GB RAM -> ram-only (only feasible)",
      feasible_default_mode("transformers", 68 * GIB, 24 * GIB, 124 * GIB)
      == "ram-only")
check("transformers 5GB / 24GB GPU -> gpu-only (fits whole; operator default "
      "order 2026-07-31: prefer full-card residency)",
      feasible_default_mode("transformers", 5 * GIB, 24 * GIB, 124 * GIB)
      == "gpu-only")
check("gguf 5GB / 24GB GPU -> gpu-only (dense fits-whole leaf, same order)",
      feasible_default_mode("gguf", 5 * GIB, 24 * GIB, 124 * GIB)
      == "gpu-only")
check("transformers 200GB / 24GB GPU / 124GB RAM -> max-gpu (fits NEITHER, "
      "honest refusal downstream — no invented fourth state)",
      feasible_default_mode("transformers", 200 * GIB, 24 * GIB, 124 * GIB)
      == "max-gpu")
# headroom boundary: exactly at 0.9x GPU still fits; just over falls to ram-only
check("transformers at exactly 0.9x GPU total -> gpu-only (within headroom)",
      feasible_default_mode("transformers", int(_GPU_FIT_HEADROOM * 24 * GIB),
                            24 * GIB, 124 * GIB) == "gpu-only")
check("transformers just over 0.9x GPU total (but fits RAM) -> ram-only",
      feasible_default_mode("transformers", int(_GPU_FIT_HEADROOM * 24 * GIB) + 1,
                            24 * GIB, 124 * GIB) == "ram-only")
# degrade-not-guess: any missing input -> max-gpu (today's behavior)
check("unknown size -> max-gpu (never derive from a guess)",
      feasible_default_mode("transformers", None, 24 * GIB, 124 * GIB) == "max-gpu")
check("unknown gpu total -> max-gpu",
      feasible_default_mode("transformers", 68 * GIB, None, 124 * GIB) == "max-gpu")
check("oversized-for-GPU but unknown RAM -> max-gpu (can't justify ram-only "
      "on a guess)",
      feasible_default_mode("transformers", 68 * GIB, 24 * GIB, None) == "max-gpu")

# ── feasible_modes matrix ────────────────────────────────────────────────────
# gguf: max-gpu/max-ram/explicit universal; gpu-only bounded by GPU fit;
# ram-only bounded by RAM fit.
fg = feasible_modes("gguf", 68 * GIB, 24 * GIB, 124 * GIB)
check("gguf 68/24/124: max-gpu, max-ram, explicit, ram-only feasible; gpu-only "
      "eliminated (won't fit GPU alone)",
      "max-gpu" in fg and "max-ram" in fg and "explicit" in fg
      and "ram-only" in fg and "gpu-only" not in fg)
fg2 = feasible_modes("gguf", 5 * GIB, 24 * GIB, 124 * GIB)
check("gguf 5/24: every mode feasible (fits GPU, RAM, and combined)",
      fg2 == ALLOC_MODES)
# THE headline: transformers 68/24/124 -> (ram-only, max-ram). max-ram is now
# engine-agnostic (2026-07-24): 68 <= 24+124 GiB combined, so it is feasible for
# a non-GGUF model too. explicit stays engine-gated off (banded leniency floor
# has no transformers analogue).
ft = feasible_modes("transformers", 68 * GIB, 24 * GIB, 124 * GIB)
check("transformers 68/24/124 -> feasible EXACTLY (ram-only, max-gpu, max-ram) — "
      "only gpu-only eliminated (it alone means GPU-or-nothing); max-gpu and "
      "max-ram both fit combined; explicit engine-gated off",
      ft == ("ram-only", "max-gpu", "max-ram"))
# OPERATOR RULING 2026-07-29, verbatim: "max-gpu just means it prefers
# maximizing the gpu for spill. gpu only dictates that it ONLY is on the gpu or
# nothing." So the fits-the-card-WHOLE test belongs to gpu-only ALONE; max-gpu
# is a PREFERENCE and is feasible whenever the model fits GPU+RAM combined —
# the same rule max-ram already used, on the same non-GGUF spill loaders.
# Previously max-gpu was eliminated here, which stripped THE DEFAULT mode from
# every oversized non-GGUF model and made 8 of ae's designations unassignable
# (/assign 409 "not feasible") after the in-VM migration.
check("=> max-gpu IS offered for the 68/24 transformers case (preference, not "
      "a whole-card requirement)",
      "max-gpu" in ft)
check("=> explicit is NEVER offered for a non-GGUF model (stays engine-gated)",
      "explicit" not in ft)
ft2 = feasible_modes("transformers", 5 * GIB, 24 * GIB, 124 * GIB)
check("transformers 5/24: gpu-only, ram-only, max-gpu, max-ram feasible; "
      "explicit engine-gated off",
      set(ft2) == {"gpu-only", "ram-only", "max-gpu", "max-ram"})
# fits neither GPU nor RAM nor combined: transformers 200/24/124
ft3 = feasible_modes("transformers", 200 * GIB, 24 * GIB, 124 * GIB)
check("transformers 200/24/124 (fits neither, nor combined 148): nothing "
      "physically lands, so the set falls back to (max-gpu,) — honest refusal "
      "downstream, never empty",
      ft3 == ("max-gpu",))
# unknown data -> never eliminate on missing data. max-ram no longer engine-gated
# so it rides through on unknown size; only explicit is dropped.
check("unknown size -> coarse trio + max-ram feasible (fail-open); only explicit "
      "engine-gated off",
      feasible_modes("transformers", None, 24 * GIB, 124 * GIB) ==
      ("gpu-only", "ram-only", "max-gpu", "max-ram"))
check("unknown size on GGUF -> literally every mode",
      feasible_modes("gguf", None, None, None) == ALLOC_MODES)
check("unknown totals on transformers -> coarse trio + max-ram feasible",
      feasible_modes("transformers", 68 * GIB, None, None) ==
      ("gpu-only", "ram-only", "max-gpu", "max-ram"))
# explicit is ENGINE-GATED regardless of numbers (a capability fact); max-ram is
# NOT — it is offered for non-GGUF whenever the numbers fit.
check("non-GGUF never offers explicit (engine gate) even with room; max-ram IS "
      "offered when it fits combined",
      "explicit" not in feasible_modes("transformers", 1 * GIB, 24 * GIB, 124 * GIB)
      and "max-ram" in feasible_modes("transformers", 1 * GIB, 24 * GIB, 124 * GIB))
check("the default is the BEST feasible member (default in feasible set)",
      feasible_default_mode("transformers", 68 * GIB, 24 * GIB, 124 * GIB)
      in feasible_modes("transformers", 68 * GIB, 24 * GIB, 124 * GIB))

# ── explicit persisted mode wins over derivation ─────────────────────────────
check("a persisted alloc_mode is NOT blank -> derivation never overrides it",
      derive_alloc_mode({"alloc_mode": "gpu-only"}) == "gpu-only")

# ── the emission seam (spill_for) ────────────────────────────────────────────
from hugpy_fleet.central import workers as W
from hugpy_fleet.central.workers import WorkerStore

# Stub central's engine + size resolution so the seam is deterministic without a
# real registry/manifest. "tf-big" = 68 GiB transformers; "tf-small" = 5 GiB
# transformers; "g-big" = 200 GiB gguf; "unk" = unknown size.
_SIZES = {"tf-big": 68 * GIB, "tf-small": 5 * GIB, "g-big": 200 * GIB, "unk": None}
_ENGINES = {"tf-big": "transformers", "tf-small": "transformers",
            "g-big": "gguf", "unk": "transformers"}
_REAL_MODEL_SIZE_BYTES = W._model_size_bytes
_REAL_MODEL_ENGINE = W._model_engine
W._model_size_bytes = lambda mk: _SIZES.get(mk)
W._model_engine = lambda mk: _ENGINES.get(mk)


@pytest.fixture(autouse=True, scope="module")
def _restore_workers_module_stubs():
    """Put W's real functions back when this MODULE is done.

    These are module-SCOPE monkeypatches with no teardown, so without this they
    outlive the file and poison every later test in the same pytest process.
    Concretely: test_storage_budget_fifo's
    ``test_model_size_bytes_really_resolves_against_the_real_manifest`` passes
    alone (54/54) and fails whenever this file runs first, because
    _model_size_bytes still returns the stub's None for every real manifest key.

    That cost real debugging time twice on 2026-07-25 — a leaked stub makes a
    wide run report failures that are not real, which is exactly how a GENUINE
    regression gets waved through as "just the usual pollution".
    """
    yield
    W._model_size_bytes = _REAL_MODEL_SIZE_BYTES
    W._model_engine = _REAL_MODEL_ENGINE

# Point the MODULE-GLOBAL store at an ISOLATED tmp registry so this test never
# touches the real workers.json (the default path sits next to the manifest) and
# is deterministic run-to-run. The wrappers (derived_default_for,
# feasible_modes_for, feasibility_context) and the route gate all read
# W.worker_store, so reassigning it here routes them at our isolated store.
W.worker_store = WorkerStore(
    path=os.path.join(os.environ["PROJECTS_HOME"], "wk.json"))
store = W.worker_store
# Isolate the assignment-memory sidecar too (durable hardware totals ride it) —
# its default path sits next to the real manifest.
# The sidecar derives from the fleet state dir (hugpy_fleet.central.config):
# redirect the settings rather than rebinding the function (a rebound lambda
# would outlive this module and break other tests' isolation).
W.settings.manifest_path = os.path.join(os.environ["PROJECTS_HOME"], "model_manifest.json")
_MEM_PATH = W._assign_memory_path()
# a 24 GiB GPU / 124 GiB RAM worker on a mode-honoring version
w = store.register(name="box24", url="http://z:9100", pkg_version="0.1.203")
# supply the totals the way a heartbeat does (raw record), via a transaction.
with store._transaction() as wk:
    wk[w["id"]]["gpus"] = [{"name": "RTX 3090", "memory_total": 24 * GIB,
                            "memory_free": 20 * GIB}]
    wk[w["id"]]["ram_total"] = 124 * GIB

for mk in ("tf-big", "tf-small", "g-big", "unk"):
    store.assign_model(w["id"], mk)          # blank spill (no alloc contract)

check("seam: oversized transformers blank default SERVES as ram-only "
      "({'n_gpu_layers':'off'})",
      store.spill_for(w["id"], "tf-big") == {"n_gpu_layers": "off"})
check("seam: fitting transformers blank default -> {'n_gpu_layers': -1} "
      "(gpu-only; operator default order 2026-07-31)",
      store.spill_for(w["id"], "tf-small") == {"n_gpu_layers": -1})
check("seam: GGUF blank default -> {} (max-gpu always, any size)",
      store.spill_for(w["id"], "g-big") == {})
check("seam: unknown-size transformers blank default -> {} (max-gpu, degrade)",
      store.spill_for(w["id"], "unk") == {})

# an explicit persisted spill is left untouched (the blank derivation only ever
# fires when NOTHING placement-affecting is persisted).
store.assign_model(w["id"], "tf-big", spill={"n_gpu_layers": -1})
check("seam: a persisted placement spill wins — derivation never overrides it",
      store.spill_for(w["id"], "tf-big") == {"n_gpu_layers": -1})
store.assign_model(w["id"], "tf-big", spill={})   # back to blank
check("seam: clearing back to blank restores the ram-only feasible default",
      store.spill_for(w["id"], "tf-big") == {"n_gpu_layers": "off"})

# a worker with NO reported totals -> fail-open (max-gpu blank), even oversized
w2 = store.register(name="box-fresh", url="http://q:9100", pkg_version="0.1.203")
store.assign_model(w2["id"], "tf-big")
check("seam: a worker missing totals fails open to {} (max-gpu) — never a "
      "ram-only guess",
      store.spill_for(w2["id"], "tf-big") == {})

# module wrappers used by the surface
check("derived_default_for reads the RAW record -> ram-only for the 68/24 case",
      W.derived_default_for(w["id"], "tf-big") == "ram-only")
check("feasible_modes_for -> (ram-only, max-gpu, max-ram) for the 68/24 "
      "transformers case (max-ram opened for non-GGUF 2026-07-24; max-gpu "
      "joined it 2026-07-29 — a spill PREFERENCE, not a whole-card test; both "
      "fit combined 148 GiB)",
      W.feasible_modes_for(w["id"], "tf-big") == ("ram-only", "max-gpu", "max-ram"))
check("feasibility_context surfaces the raw numbers for an honest 409",
      W.feasibility_context(w["id"], "tf-big") ==
      {"engine": "transformers", "model_bytes": 68 * GIB,
       # Per-device (2026-09-25): gpu_total_bytes is the ENGINE-AWARE GPU ceiling
       # (largest single card for a non-splittable engine) — here a single-GPU
       # box so it equals the box sum; gpu_box_total_bytes carries the naive sum.
       "gpu_total_bytes": 24 * GIB, "gpu_box_total_bytes": 24 * GIB,
       "ram_total_bytes": 124 * GIB,
       # MoE (2026-07-24): the expert-split GPU need; None for non-MoE.
       "moe_split_gpu_bytes": None,
       # bnb (2026-07-29): which size the decision priced — a 409 quoting fp16
       # bytes while 4-bit is on would read as a bug in the numbers.
       "bnb": False})

# ── 4-bit re-prices FEASIBILITY, not just the allocation (2026-07-29) ────────
# Operator: "its auto should change upon 4-bit delegation". default_allocation
# has priced the 4-bit footprint since 2026-07-26; feasible_modes did not, so
# the allocator planned ~15 GiB for a 51.7 GiB model while the gate refused the
# mode on the fp16 size. sdxl-turbo on ae's 24 GiB card is exactly that case.
SDXL = 52 * GIB                       # ~ the real stabilityai~sdxl-turbo dir
f_fp16 = feasible_modes("transformers", SDXL, 24 * GIB, 88 * GIB, bnb=False)
f_4bit = feasible_modes("transformers", SDXL, 24 * GIB, 88 * GIB, bnb=True)
check("4-bit OFF: a 52 GiB transformers model cannot hold the 24 GiB card whole "
      "-> gpu-only eliminated",
      "gpu-only" not in f_fp16)
check("4-bit ON: the same model prices at ~15.6 GiB (0.30 ratio) and gpu-only "
      "BECOMES feasible — the gate now moves with the lever",
      "gpu-only" in f_4bit)
check("4-bit never eliminates a mode that was feasible at full size "
      "(re-pricing can only widen the set)",
      set(f_fp16) <= set(f_4bit))
check("GGUF ignores the bnb lever entirely (llama.cpp carries its own quant)",
      feasible_modes("gguf", SDXL, 24 * GIB, 88 * GIB, bnb=True)
      == feasible_modes("gguf", SDXL, 24 * GIB, 88 * GIB, bnb=False))

# ── the /assign feasibility gate + bulk skip ─────────────────────────────────
import importlib
wr = importlib.import_module(
    "hugpy_server.app.routes.worker_routes")

# max-gpu on the 68/24 transformers case is FEASIBLE (2026-07-29): it means
# "prefer the GPU, spill the rest", so it only needs GPU+RAM combined. The gate
# must therefore ALLOW it — refusing it stripped the default mode from every
# oversized non-GGUF model and made them unassignable.
okf, reason = wr._alloc_mode_feasible_for_worker({}, w["id"], "tf-big")
check("gate: max-gpu (blank derive) on 68/24 transformers -> ALLOWED (spill "
      "preference, fits combined)", okf is True and reason is None)
# gpu-only is the mode that genuinely cannot fit -> still refused, numbers named
okg, reasong = wr._alloc_mode_feasible_for_worker(
    {"n_gpu_layers": -1}, w["id"], "tf-big")
check("gate: gpu-only on the same case -> refused (GPU-or-nothing, 68 > 24)",
      okg is False)
check("gate: the refusal NAMES the mode, the numbers, and the feasible set",
      "gpu-only" in reasong and "68.0GiB" in reasong and "24.0GiB" in reasong
      and "ram-only" in reasong)
# the feasible pick is allowed
okf2, reason2 = wr._alloc_mode_feasible_for_worker(
    {"n_gpu_layers": "off"}, w["id"], "tf-big")
check("gate: ram-only on the same case -> allowed", okf2 is True and reason2 is None)
# a fitting model allows max-gpu
okf3, _ = wr._alloc_mode_feasible_for_worker({}, w["id"], "tf-small")
check("gate: max-gpu on a fitting transformers model -> allowed", okf3 is True)
# unknown worker / unknown data fails open (allow)
okf4, _ = wr._alloc_mode_feasible_for_worker({}, "no-such-worker", "tf-big")
check("gate: unknown worker fails open (allow — assign 404s later)", okf4 is True)
okf5, _ = wr._alloc_mode_feasible_for_worker({}, w["id"], "unk")
check("gate: unknown model size fails open (allow — never eliminate on a guess)",
      okf5 is True)

# ── durable hardware totals (operator addendum 2026-07-24) ───────────────────
# A box's GPU/RAM CAPACITY is a physical FACT, not session state. A re-register
# (or a transient empty gpu probe) must NOT erase totals central already learned,
# so feasibility keeps working across the re-register window.
GPUS_24 = [{"name": "RTX 3090", "memory_total": 24 * GIB, "memory_free": 20 * GIB}]

# fresh box: first register WITH totals learns them and persists them durably.
d1 = store.register(name="durable-box", url="http://d:9100",
                    pkg_version="0.1.203", gpus=GPUS_24, ram_total=124 * GIB)
did = d1["id"]
store.assign_model(did, "tf-big")            # designate the oversized transformers
check("durable: first register with totals -> feasibility resolves "
      "(ram-only, max-gpu, max-ram)",
      W.feasible_modes_for(did, "tf-big") == ("ram-only", "max-gpu", "max-ram"))
rec = store._load()[did]
check("durable: register persisted the last-known GPU total as a durable fact",
      rec.get("gpu_total_bytes_known") == 24 * GIB
      and rec.get("ram_total_bytes_known") == 124 * GIB)

# RE-REGISTER with an EMPTY gpu probe (driver not ready) + no ram_total: the live
# gpus[] is wiped, but the durable fact must survive so feasibility still works.
store.register(name="durable-box", url="http://d:9100", pkg_version="0.1.203",
               gpus=[], ram_total=None)
rec2 = store._load()[did]
check("durable: an empty-probe re-register did NOT erase the durable totals",
      rec2.get("gpu_total_bytes_known") == 24 * GIB
      and rec2.get("ram_total_bytes_known") == 124 * GIB)
check("durable: live gpus[] IS now empty (the transient reading), proving the "
      "fallback — not stale live data — carries feasibility",
      not rec2.get("gpus"))
check("durable: feasibility STILL resolves (ram-only, max-gpu, max-ram) across the re-register window",
      W.feasible_modes_for(did, "tf-big") == ("ram-only", "max-gpu", "max-ram"))
check("durable: the derived default STILL serves as ram-only across the window",
      W.derived_default_for(did, "tf-big") == "ram-only"
      and store.spill_for(did, "tf-big") == {"n_gpu_layers": "off"})

# a beat that carries fresh totals confirms/updates them (advance path).
store.heartbeat(did, gpus=GPUS_24, ram_total=124 * GIB)
check("durable: a heartbeat re-confirms the totals (advance-only update)",
      store._load()[did].get("gpu_total_bytes_known") == 24 * GIB)

# survive a FULL registry loss: the durable totals ride the assignment-memory
# sidecar, so a returning worker id inherits them on a brand-new row.
_remembered = W._load_assign_memory().get(did) or {}
check("durable: the totals were snapshotted into the assignment-memory sidecar",
      _remembered.get("gpu_total_bytes_known") == 24 * GIB
      and _remembered.get("ram_total_bytes_known") == 124 * GIB)
store.remove(did)                            # registry row gone (row swept)
d3 = store.register(name="durable-box", url="http://d2:9100",
                    pkg_version="0.1.203", gpus=[], ram_total=None,
                    worker_id=did)           # returning id, no live totals yet
check("durable: a returning id with a LOST row inherits totals from memory",
      store._load()[did].get("gpu_total_bytes_known") == 24 * GIB)
check("durable: feasibility works immediately on the restored row (pre-first-beat)",
      W.feasible_modes_for(did, "tf-big") == ("ram-only", "max-gpu", "max-ram"))

# a GENUINELY-NEW worker id (never seen, no memory) still fails open + self-logs.
import logging as _logging
_caplog = []
class _H(_logging.Handler):
    def emit(self, r):
        _caplog.append(r.getMessage())
_h = _H()
W.logger.addHandler(_h)
try:
    dn = store.register(name="brand-new", url="http://new:9100",
                        pkg_version="0.1.203", gpus=[], ram_total=None)
    store.assign_model(dn["id"], "tf-big")
    modes = W.feasible_modes_for(dn["id"], "tf-big")
finally:
    W.logger.removeHandler(_h)
check("first-contact: a brand-new worker with NO totals fails open (all coarse "
      "modes + max-ram feasible — never eliminate on missing data; only explicit "
      "stays engine-gated)",
      set(modes) == {"gpu-only", "ram-only", "max-gpu", "max-ram"})
check("first-contact: the self-policing fail-open WARNING fired (drift signal), "
      "naming the missing data",
      any("fail-open" in m and "no gpu/ram totals" in m for m in _caplog))
# rate-limited: a second identical resolution does NOT re-log.
_caplog.clear()
W.logger.addHandler(_h)
try:
    W.feasible_modes_for(dn["id"], "tf-big")
finally:
    W.logger.removeHandler(_h)
check("first-contact: the fail-open log is once-per-(model,worker) (no spam)",
      not any("fail-open" in m for m in _caplog))

# ── STRUCTURE-DERIVED DEFAULTS: the MoE split at the emission seam ───────────
# The operator's decision tree (2026-07-25) end-to-end through central: a blank
# MoE allocation must emit its own DERIVED split, not the blanket stamp. The
# tree's own leaf matrix is proven in test_moe_placement.py against the measured
# files; this covers the CENTRAL WIRING — engine/size/MoE/totals resolution, the
# version gate, and "an operator's choice always wins".
_CODER_NEXT_MOE = {"is_moe": True, "expert_count": 512, "expert_used_count": 10,
                   "expert_bytes": int(43.59 * GIB),
                   "non_expert_bytes": int(1.49 * GIB),
                   "expert_bytes_by_layer": {}, "files": 4}
_SIZES["moe-big"] = int(45.08 * GIB)     # coder-next Q4_K_M, measured
_ENGINES["moe-big"] = "gguf"
_SIZES["gguf-dense"] = 5 * GIB
_ENGINES["gguf-dense"] = "gguf"
_MOE = {"moe-big": _CODER_NEXT_MOE}
W._model_moe_detail = lambda mk: _MOE.get(mk)

for mk in ("moe-big", "gguf-dense"):
    store.assign_model(w["id"], mk)      # blank spill (no alloc contract)

_derived = store.spill_for(w["id"], "moe-big")
check("seam: a blank MoE allocation emits its DERIVED split, not a blanket stamp",
      _derived.get("alloc_mode") == "explicit"
      and _derived.get("n_cpu_moe") == 999
      and _derived.get("n_gpu_layers") == -1)
check("seam: the derived split DECLARES both sides — 1.49 GiB of non-expert "
      "tensors on the card, 43.59 GiB of experts in RAM",
      abs(_derived.get("gpu_mem_gib", 0) - 1.49) < 0.01
      and abs(_derived.get("cpu_mem_gib", 0) - 43.59) < 0.01)
check("seam: THE LIVE BUG — a MoE never receives the bare {'n_gpu_layers': -1} "
      "stamp that would disable the load-time auto split",
      _derived != {"n_gpu_layers": -1})
check("seam: a fits-whole DENSE gguf derives gpu-only -> {'n_gpu_layers': -1} "
      "(operator default order 2026-07-31; the MoE split branch above is "
      "untouched by it)",
      store.spill_for(w["id"], "gguf-dense") == {"n_gpu_layers": -1})

# an operator's explicit choice ALWAYS wins — derivation is blank-only. A
# gpu-only pin ({"n_gpu_layers": -1}) on a MoE is honored AS gpu-only: the mode
# is not overridden, but the explicit n_cpu_moe:0 that "every tensor on the
# card" MEANS for a MoE is folded in, so the worker's auto-MoE policy cannot
# offload experts to CPU behind the pin. Before this the bare -1 reached the
# worker's auto branch and it split anyway (the live Qwen3.6-27B on aeb: pinned
# gpu-only, fits VRAM, yet expert-split). gpu-only is only ever chosen/derived
# when the model fits the card whole, so n_cpu_moe:0 is fit-safe here.
store.assign_model(w["id"], "moe-big", spill={"n_gpu_layers": -1})
check("seam: an operator's gpu-only pin on a MoE is honored AS gpu-only — the "
      "mode is untouched and n_cpu_moe:0 is folded in so experts stay on the "
      "card (no auto-split behind the pin)",
      store.spill_for(w["id"], "moe-big") == {"n_gpu_layers": -1, "n_cpu_moe": 0})
store.assign_model(w["id"], "moe-big", spill={})
check("seam: clearing back to blank restores the derived split",
      store.spill_for(w["id"], "moe-big").get("alloc_mode") == "explicit")

# the version gate: the derived split carries NEW keys, so a pre-0.1.203 worker
# gets {} — whose own load-time auto MoE policy reaches the right placement.
w_old = store.register(name="old-box", url="http://o:9100", pkg_version="0.1.150")
with store._transaction() as wk:
    wk[w_old["id"]]["gpus"] = [{"name": "RTX 3090", "memory_total": 24 * GIB,
                                "memory_free": 20 * GIB}]
    wk[w_old["id"]]["ram_total"] = 128 * GIB
store.assign_model(w_old["id"], "moe-big")
check("seam: a pre-0.1.203 worker gets {} (max-gpu) — never a silent dead knob; "
      "its own auto MoE policy still lands the split",
      store.spill_for(w_old["id"], "moe-big") == {})

# degrade-not-guess at the seam: no measured totals -> no derived split.
store.assign_model(w2["id"], "moe-big")          # w2 has no gpu/ram totals
check("seam: a worker with NO measured totals never derives a split -> {} "
      "(degrade-not-guess)",
      store.spill_for(w2["id"], "moe-big") == {})

# the mode-name view agrees with the allocation view at the wrapper level.
check("derived_default_for names the MoE leaf 'explicit'",
      W.derived_default_for(w["id"], "moe-big") == "explicit")
_alloc = W.derived_allocation_for(w["id"], "moe-big")
check("derived_allocation_for returns mode+spill+why, and the why names the "
      "numbers that chose it",
      _alloc["mode"] == "explicit" and "MoE split" in _alloc["why"]
      and "1.49" in _alloc["why"] and "43.59" in _alloc["why"])
check("derived_allocation_for on an unknown worker -> None",
      W.derived_allocation_for("no-such-worker", "moe-big") is None)

# the local mirror of NEW_SPILL_KEYS must not drift from the real one.
from hugpy_engine.alloc_modes import NEW_SPILL_KEYS as _NSK
check("the workers.py NEW_SPILL_KEYS mirror matches alloc_modes (no drift)",
      W._NEW_SPILL_KEYS_LOCAL == _NSK)

print(f"\nALL {ok} capability-aware-default checks passed")


# ─────────────────────────────────────────────────────────────────────────────
# THE DROPDOWN DISAGREEMENT (spec assets/evictionflow.html, open item 3).
#
# "feasible_default_mode and default_allocation still disagree for a large dense
#  GGUF. Once preference drives eviction too, a wrong label mispredicts not just
#  placement but which model dies."
#
# The two functions are now ONE tree with two views (feasible_default_mode is a
# pure projection of default_allocation["mode"]). These assert they cannot drift
# apart again — the drift was two hand-maintained copies of the same decision.
# ─────────────────────────────────────────────────────────────────────────────
def test_the_two_default_views_agree_for_a_large_dense_gguf():
    """THE reported disagreement, pinned. A dense GGUF that overflows the card
    but fits RAM: the name-view used to say max-gpu (its blanket "a GGUF is
    ALWAYS max-gpu" rule) while the allocation-view persisted max-ram (the
    preference-not-prohibition ruling). Under the eviction flow that label IS
    the eviction preference, so the mismatch mispredicted which model dies."""
    from hugpy_engine.alloc_modes import feasible_default_mode, default_allocation
    GIB = 1 << 30
    args = ("gguf", 40 * GIB, 24 * GIB, 128 * GIB)
    assert feasible_default_mode(*args) == default_allocation(*args)["mode"]
    assert feasible_default_mode(*args) == "max-ram", (
        "default_allocation is the side that won: it is what gets PERSISTED, "
        "it carries the measured ~5x-cliff rationale, and it is the only one "
        "that can express the MoE leaf")


def test_the_two_default_views_agree_across_the_whole_tree():
    """Not just the reported case — every leaf, so a future edit to one side
    cannot silently re-open the gap."""
    from hugpy_engine.alloc_modes import feasible_default_mode, default_allocation
    GIB = 1 << 30
    moe = {"is_moe": True, "non_expert_bytes": 2 * GIB,
           "expert_bytes": 44 * GIB}
    cases = [
        ("gguf", 10 * GIB, 24 * GIB, 128 * GIB, None),      # small dense
        ("gguf", 40 * GIB, 24 * GIB, 128 * GIB, None),      # large dense
        ("gguf", 400 * GIB, 24 * GIB, 128 * GIB, None),     # fits neither
        ("gguf", 46 * GIB, 24 * GIB, 128 * GIB, moe),       # the MoE split
        ("gguf", 46 * GIB, 24 * GIB, 8 * GIB, moe),         # experts > RAM
        ("transformers", 10 * GIB, 24 * GIB, 128 * GIB, None),
        ("transformers", 40 * GIB, 24 * GIB, 128 * GIB, None),
        ("transformers", 400 * GIB, 24 * GIB, 128 * GIB, None),
        ("gguf", None, 24 * GIB, 128 * GIB, None),          # unknown size
        ("gguf", 40 * GIB, None, 128 * GIB, None),          # unknown GPU
        ("gguf", 40 * GIB, 24 * GIB, None, None),           # unknown RAM
    ]
    for engine, size, gpu, ram, moe_detail in cases:
        name = feasible_default_mode(engine, size, gpu, ram, moe_detail)
        alloc = default_allocation(engine, size, gpu, ram, moe=moe_detail)["mode"]
        assert name == alloc, (
            f"drift for ({engine}, size={size}, gpu={gpu}, ram={ram}, "
            f"moe={bool(moe_detail)}): name-view={name}, alloc-view={alloc}")
