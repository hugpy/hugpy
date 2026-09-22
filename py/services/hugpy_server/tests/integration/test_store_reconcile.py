"""Store-flattening + catalog-reconcile regression (operator-locked 2026-07-11).

Fabricates a temp store exercising every migration shape and asserts the
dry-run/apply/re-run/read-through/presence/atomicity contract end to end.

  * task-twin (partial + complete)      -> complete wins, .part twin archived
  * legacy-misc stranding (empty twin)  -> misc weights win, empty twin archived
  * flat-complete entry                 -> no-op
  * .part-only orphan (no good copy)    -> archived, no winner
  * gguf with a NON-pinned complete quant -> installed + pin WARNING (not partial)
  * vision gguf: quant here, mmproj in a twin -> MERGE mmproj, then move flat

Asserts: dry-run plan exact + touches nothing; apply -> flat layout + archives +
registry/marker updates; re-run is a no-op; read-through resolves every legacy
path BEFORE and AFTER; presence verdicts correct; an atomic provision leaves NO
resolvable partial on simulated failure.

Every test gets a FRESH fabricated store (the ``store`` fixture), so the phases
are independent; the apply-dependent phases run their own apply first.
"""
import importlib
import json
import logging
import os
from types import SimpleNamespace

import pytest

logging.disable(logging.CRITICAL)

paths = importlib.import_module("hugpy_storage.model_paths")
constants = importlib.import_module("hugpy_platform.constants")
reconcile = importlib.import_module("hugpy_engine.apis.reconcile")
main = importlib.import_module("hugpy_engine.config.main")

MB = 1024 * 1024


def wfile(path, mb=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * (mb * MB + 7))


def marker(directory, hub_id, framework, primary_task=None, tasks=None, filename=None):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "hugpy.json"), "w") as fh:
        json.dump({"hub_id": hub_id, "framework": framework,
                   "primary_task": primary_task, "tasks": tasks,
                   "filename": filename, "source": "test"}, fh)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Fabricate the store + catalog under a temp root and point the reconcile's
    catalog paths at it (restored by monkeypatch)."""
    root = str(tmp_path)
    models = os.path.join(root, "models")

    # ---- fabricate the store -----------------------------------------------
    # 1. task-twin: text-generation has ONLY a .part; image-text-to-text is complete
    tg_twin = os.path.join(models, "gguf", "text-generation", "own", "twin-gguf")
    wc_twin = os.path.join(models, "gguf", "image-text-to-text", "own", "twin-gguf")
    wfile(os.path.join(tg_twin, "twin-Q4_K_M.gguf.part"), 3)          # crashed pull
    wfile(os.path.join(wc_twin, "twin-Q4_K_M.gguf"), 3)              # complete
    marker(wc_twin, "own/twin-gguf", "gguf", "text-generation", ["text-generation"])

    # 2. legacy-misc stranding: weights in misc, empty text-generation twin
    misc_lora = os.path.join(models, "transformers", "misc", "own", "lora-adapter")
    empty_lora = os.path.join(models, "transformers", "text-generation", "own", "lora-adapter")
    wfile(os.path.join(misc_lora, "adapter_model.safetensors"), 2)
    with open(os.path.join(misc_lora, "adapter_config.json"), "w") as fh:
        fh.write("{}")
    os.makedirs(empty_lora, exist_ok=True)                           # empty twin

    # 3. flat-complete entry (already migrated)
    flat_ok = os.path.join(models, "transformers", "own", "already-flat")
    wfile(os.path.join(flat_ok, "model.safetensors"), 2)
    with open(os.path.join(flat_ok, "config.json"), "w") as fh:
        fh.write("{}")
    marker(flat_ok, "own/already-flat", "transformers", "text-generation", ["text-generation"])

    # 4. .part-only orphan (no complete copy anywhere)
    orphan = os.path.join(models, "gguf", "text-generation", "own", "orphan-gguf")
    wfile(os.path.join(orphan, "orphan.gguf.part"), 3)

    # 5. gguf pinned to a quant that is NOT the one on disk (flat-complete)
    pinned = os.path.join(models, "gguf", "own", "pinned-gguf")
    wfile(os.path.join(pinned, "pinned-Q4_K_M.gguf"), 3)
    marker(pinned, "own/pinned-gguf", "gguf", "text-generation", ["text-generation"],
           filename="pinned-Q8_0.gguf")

    # 6. vision gguf: quant in image-text-to-text, mmproj sidecar in a twin
    vq = os.path.join(models, "gguf", "image-text-to-text", "own", "vision-gguf")
    vm = os.path.join(models, "gguf", "text-generation", "own", "vision-gguf")
    wfile(os.path.join(vq, "vision-Q4_K_M.gguf"), 3)                 # quant, no mmproj
    wfile(os.path.join(vm, "mmproj-vision.gguf"), 2)                 # projector twin

    # ---- catalog (discovery report + manifest) ----------------------------
    discovery = {
        "twin-gguf": {"hub_id": "own/twin-gguf", "framework": "gguf",
                      "primary_task": "text-generation", "tasks": ["text-generation"],
                      "dir": wc_twin, "folder": "gguf/image-text-to-text/own/twin-gguf"},
        "lora-adapter": {"hub_id": "own/lora-adapter", "framework": "transformers",
                         "primary_task": "text-generation", "tasks": ["text-generation"],
                         "dir": misc_lora, "folder": "transformers/misc/own/lora-adapter"},
        "already-flat": {"hub_id": "own/already-flat", "framework": "transformers",
                         "primary_task": "text-generation", "tasks": ["text-generation"],
                         "dir": flat_ok, "folder": "transformers/own/already-flat"},
        "orphan-gguf": {"hub_id": "own/orphan-gguf", "framework": "gguf",
                        "primary_task": "text-generation", "tasks": ["text-generation"]},
        "pinned-gguf": {"hub_id": "own/pinned-gguf", "framework": "gguf",
                        "primary_task": "text-generation", "tasks": ["text-generation"],
                        "filename": "pinned-Q8_0.gguf", "dir": pinned,
                        "folder": "gguf/own/pinned-gguf"},
        "vision-gguf": {"hub_id": "own/vision-gguf", "framework": "gguf",
                        "primary_task": "image-text-to-text",
                        "tasks": ["image-text-to-text"], "dir": vq,
                        "folder": "gguf/image-text-to-text/own/vision-gguf"},
    }
    disc_path = os.path.join(root, "model_discovery.json")
    mani_path = os.path.join(root, "model_manifest.json")
    with open(disc_path, "w") as fh:
        json.dump(discovery, fh)
    with open(mani_path, "w") as fh:
        json.dump({"lora-adapter": dict(discovery["lora-adapter"])}, fh)  # in both

    # point the reconcile's catalog at our temp artifacts
    monkeypatch.setattr(constants, "MODELS_DISCOVERY_PATH", disc_path)
    monkeypatch.setattr(constants, "MODELS_DICT_PATH", mani_path)

    return SimpleNamespace(
        root=root, models=models, discovery=discovery,
        disc_path=disc_path, mani_path=mani_path,
        tg_twin=tg_twin, wc_twin=wc_twin, misc_lora=misc_lora, empty_lora=empty_lora,
        flat_ok=flat_ok, orphan=orphan, pinned=pinned, vq=vq, vm=vm,
        flat_twin=os.path.join(models, "gguf", "own", "twin-gguf"),
        flat_lora=os.path.join(models, "transformers", "own", "lora-adapter"),
        flat_vis=os.path.join(models, "gguf", "own", "vision-gguf"),
    )


def cfg_of(entry):
    return paths._routing_as_cfg(entry)


def plan_for(report, hub_id):
    for pl in report["plans"]:
        if pl.get("hub_id") == hub_id:
            return pl
    raise AssertionError(f"no plan for {hub_id}")


def acts(pl, op):
    return [a for a in pl["actions"] if a.get("op") == op]


def _live_dirs(s):
    return (s.tg_twin, s.wc_twin, s.misc_lora, s.empty_lora, s.flat_ok,
            s.orphan, s.pinned, s.vq, s.vm)


# ===========================================================================
# READ-THROUGH before migration — every legacy path resolves to the complete dir
# ===========================================================================
def test_read_through_before_migration(store):
    s, d = store, store.discovery
    # twin: route_destination -> complete image-text-to-text copy
    assert paths.route_destination(d["twin-gguf"], s.root) == s.wc_twin
    # lora: route_destination -> misc weights (not the empty twin)
    assert paths.route_destination(d["lora-adapter"], s.root) == s.misc_lora
    # vision: resolve(require_complete) is None (mmproj missing pre-merge)
    assert paths.resolve_model_dir(d["vision-gguf"], s.root) is None


# ===========================================================================
# DRY RUN — plan exact, touches nothing
# ===========================================================================
def test_dry_run_plan_is_exact_and_touches_nothing(store):
    s = store
    models = s.models
    before = {p: sorted(os.listdir(p)) for p in _live_dirs(s)}
    rep = reconcile.reconcile_store(root=s.root, apply=False)

    pl_twin = plan_for(rep, "own/twin-gguf")
    moves = acts(pl_twin, "move")
    assert moves and moves[0]["dst"] == os.path.join(models, "gguf", "own", "twin-gguf") \
        and moves[0]["src"] == s.wc_twin, "twin plan: moves complete copy to flat"
    assert any(a["src"] == s.tg_twin and a["reason"] == "part-orphan"
               for a in acts(pl_twin, "archive")), "twin plan: .part twin archived as part-orphan"
    assert not acts(pl_twin, "merge"), "twin plan: NO merge of the .part file"

    pl_lora = plan_for(rep, "own/lora-adapter")
    lmoves = acts(pl_lora, "move")
    assert lmoves and lmoves[0]["src"] == s.misc_lora, "lora plan: misc weights move to flat"
    assert any(a["src"] == s.empty_lora and a["reason"] == "empty-twin"
               for a in acts(pl_lora, "archive")), "lora plan: empty twin archived"

    pl_flat = plan_for(rep, "own/already-flat")
    assert not acts(pl_flat, "move") and not acts(pl_flat, "archive") \
        and pl_flat["status"] in ("already-flat", "noop"), "already-flat plan: no-op"

    pl_orphan = plan_for(rep, "own/orphan-gguf")
    assert pl_orphan["status"] == "incomplete-no-winner" \
        and any(a["reason"] == "part-orphan" for a in acts(pl_orphan, "archive")), \
        "orphan plan: no winner, part-orphan archived"

    pl_pin = plan_for(rep, "own/pinned-gguf")
    assert any("pinned filename" in w for w in pl_pin["warnings"]), \
        "pinned plan: pin-mismatch WARNING present"
    assert not acts(pl_pin, "move"), "pinned plan: no move (already flat, quant present)"

    pl_vis = plan_for(rep, "own/vision-gguf")
    assert any(a["file"] == "mmproj-vision.gguf" and a["from"] == s.vm
               for a in acts(pl_vis, "merge")), "vision plan: MERGE mmproj from the twin"
    vmoves = acts(pl_vis, "move")
    assert vmoves and vmoves[0]["src"] == s.vq, "vision plan: winner (quant dir) moves to flat"

    assert rep["summary"]["moves"] == 3 and rep["summary"]["merges"] == 1 \
        and rep["summary"]["archives"] >= 3, "dry-run summary counts moves/merges/archives"

    after = {p: sorted(os.listdir(p)) for p in before}
    assert before == after, "dry-run touched NOTHING on disk"
    assert not os.path.exists(rep["archive_dir"]), "dry-run did not write the archive dir"
    with open(s.disc_path) as fh:
        assert json.load(fh)["twin-gguf"]["dir"] == s.wc_twin, \
            "dry-run did not mutate the discovery report"


# ===========================================================================
# APPLY — flat layout, archives, registry + markers
# ===========================================================================
def test_apply_flattens_archives_and_updates_catalog(store):
    s, d0 = store, store.discovery
    models = s.models
    rep2 = reconcile.reconcile_store(root=s.root, apply=True)

    assert os.path.isfile(os.path.join(s.flat_twin, "twin-Q4_K_M.gguf")), \
        "apply: twin complete copy now at flat"
    assert not os.path.isdir(s.wc_twin), "apply: twin legacy image-text-to-text dir gone"
    assert os.path.isdir(os.path.join(models, "_archive", rep2["timestamp"],
                                      "gguf", "text-generation", "own", "twin-gguf")), \
        "apply: .part twin archived (never deleted)"
    assert os.path.isfile(os.path.join(s.flat_lora, "adapter_model.safetensors")), \
        "apply: lora weights now at flat"
    assert os.path.isfile(os.path.join(s.flat_vis, "vision-Q4_K_M.gguf")) \
        and os.path.isfile(os.path.join(s.flat_vis, "mmproj-vision.gguf")), \
        "apply: vision quant + MERGED mmproj both at flat"
    assert main.model_looks_downloaded(s.flat_vis, cfg_of(d0["vision-gguf"])), \
        "apply: vision flat dir is COMPLETE (mmproj satisfied the vision gate)"
    assert os.path.isfile(os.path.join(s.flat_twin, "hugpy.json")), \
        "apply: fresh hugpy.json marker at the flat winner"
    with open(s.disc_path) as fh:
        d = json.load(fh)
    assert d["twin-gguf"]["dir"] == s.flat_twin \
        and d["twin-gguf"]["folder"] == "gguf/own/twin-gguf", \
        "apply: registry dir/folder updated to flat"
    with open(s.mani_path) as fh:
        assert json.load(fh)["lora-adapter"]["folder"] == "transformers/own/lora-adapter", \
            "apply: manifest entry (lora) also updated"
    assert os.path.isdir(os.path.join(models, "_archive", rep2["timestamp"])), \
        "apply: NOTHING under _archive was deleted (orphan preserved)"


# ===========================================================================
# READ-THROUGH after migration — flat resolves; legacy dict still resolves
# ===========================================================================
def test_read_through_after_migration(store):
    s, d0 = store, store.discovery
    reconcile.reconcile_store(root=s.root, apply=True)
    with open(s.disc_path) as fh:
        d = json.load(fh)
    assert paths.route_destination(d["twin-gguf"], s.root) == s.flat_twin, \
        "twin: route_destination now flat"
    # a stale routing dict that still names the OLD task path must not 404
    stale = {"hub_id": "own/twin-gguf", "framework": "gguf",
             "primary_task": "text-generation", "tasks": ["text-generation"]}
    assert paths.route_destination(stale, s.root) == s.flat_twin, \
        "stale legacy routing still resolves to the flat copy"
    assert paths.resolve_model_dir(d0["vision-gguf"], s.root) == s.flat_vis, \
        "vision: resolve now finds the completed flat copy"


# ===========================================================================
# RE-RUN — idempotent no-op (no ping-pong)
# ===========================================================================
def test_rerun_is_idempotent(store):
    s = store
    reconcile.reconcile_store(root=s.root, apply=True)
    rep3 = reconcile.reconcile_store(root=s.root, apply=True)
    assert rep3["summary"]["moves"] == 0, "re-run: zero moves"
    assert rep3["summary"]["merges"] == 0, "re-run: zero merges"
    assert all(pl["status"] in ("already-flat", "noop", "incomplete-no-winner", "absent")
               for pl in rep3["plans"]), "re-run: zero NEW archives of live dirs"


# ===========================================================================
# PRESENCE HONESTY verdicts
# ===========================================================================
def test_presence_verdicts(store):
    s, d = store, store.discovery
    assert main.model_looks_downloaded(s.pinned, cfg_of(d["pinned-gguf"])), \
        "gguf any-quant: pinned-gguf installed despite pin mismatch"
    assert reconcile.pinned_filename_present(s.pinned, cfg_of(d["pinned-gguf"])) is False, \
        "pin mismatch is a WARNING signal, not incomplete"
    assert not main.model_looks_downloaded(
        os.path.join(s.models, "gguf", "own", "no-such"), cfg_of(d["vision-gguf"])), \
        "vision with no mmproj reads INCOMPLETE (gate preserved)"


# ===========================================================================
# ATOMIC PROVISION — a simulated failure leaves NO resolvable partial
# ===========================================================================
def test_atomic_provision_leaves_no_resolvable_partial(store):
    s = store
    atomic = importlib.import_module("hugpy_storage.download_models")
    # stage dir naming: the download must land in <dest>.tmp-<pid>, never at <dest>
    fresh = {"hub_id": "own/fresh-model", "framework": "gguf",
             "primary_task": "text-generation", "tasks": ["text-generation"]}
    dest = paths.flat_destination(fresh, s.root)
    staged = atomic._staging_dir(dest)
    assert staged != dest and staged.startswith(dest + ".tmp-"), \
        "staging dir is a sibling temp of the final dest"
    # simulate a crash: staged dir with a half-file exists, final does not
    os.makedirs(staged, exist_ok=True)
    wfile(os.path.join(staged, "half.gguf.part"), 1)
    assert paths.resolve_model_dir(fresh, s.root) is None and not os.path.isdir(dest), \
        "a crashed provision's staged dir does NOT resolve as the model"
