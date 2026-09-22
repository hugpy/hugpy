"""Regression: MODEL_REGISTRY and MODEL_REGISTRY_DICT must stay SYMMETRIC.

2026-07-29 defect. ``model_resolver.validate_registry()`` ran at import and
POPPED any discovered model whose (framework, task) pair had no runner/builder
registered — but it popped ``MODEL_REGISTRY`` ONLY. ``MODEL_REGISTRY_DICT`` was
never touched, and ``refresh_registry()`` re-adds the row to BOTH anyway. The
two registries therefore disagreed:

    Wan2.1-T2V-1.3B   MODEL_REGISTRY=False   MODEL_REGISTRY_DICT=True
    get_model_config(k, dict_return=True)  -> OK, tasks=['text-to-video']
    get_model_config(k, dict_return=False) -> KeyError: Unknown model

Consequence: the model was listed in /models, counted in storage and present on
disk, but UNDESIGNATABLE — ``POST /llm/workers/<id>/assign`` passed its dict-form
gate, then hit the object form, caught the KeyError, and refused by blaming the
DISK ("central does not have X on disk — download it on the Models tab first").
That message was a lie; the real cause was a missing runner row.

Operator standing order: a model may be inefficient, never SILENTLY unavailable.
So validate_registry() now REPORTS instead of mutating, and refusal lives at the
point of use — resolve() already refuses an unservable pair loudly and precisely
("No request builder / No runner for ('transformers','text-to-video')").

Two invariants are locked here:
  1. validate_registry() does not mutate the registry; an unservable discovered
     row stays in BOTH dicts and is fetchable in OBJECT form.
  2. It still fails HARD (RuntimeError) for a CURATED staple — a code bug must
     brick import; a data row must never.

Runs like the other tests here:
    venv/bin/python tests/test_registry_symmetry.py
"""
import copy
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib

mc = importlib.import_module("hugpy_engine.config.models.models_config")
md = importlib.import_module("hugpy_engine.config.models.models_default")
cfg_main = importlib.import_module("hugpy_engine.config.main")
MR = importlib.import_module("hugpy_engine.resolvers.model_resolver")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def test_registry_symmetry():
    """Converted from the script-style suite: every `check` is an assert."""


    # --- the live registries agree, right now, after import-time validation ------
    check("MODEL_REGISTRY and MODEL_REGISTRY_DICT hold the same keys",
          set(mc.MODEL_REGISTRY) == set(mc.MODEL_REGISTRY_DICT))

    check("every live key resolves in OBJECT form (no dict-only ghosts)",
          all(cfg_main.get_model_config(k) is not None for k in mc.MODEL_REGISTRY))


    # --- a DISCOVERED row with no runner survives validation --------------------
    FAKE_KEY = "zz-symmetry-probe-no-runner"
    FAKE_FW = "zz-nonexistent-framework"
    FAKE_TASK = "zz-nonexistent-task"

    _template = next(iter(mc.MODEL_REGISTRY.values()))
    probe = copy.deepcopy(_template)
    for _field, _value in (("model_key", FAKE_KEY), ("name", FAKE_KEY),
                           ("folder", FAKE_KEY), ("hub_id", f"zz/{FAKE_KEY}"),
                           ("framework", FAKE_FW), ("primary_task", FAKE_TASK),
                           ("tasks", [FAKE_TASK])):
        if any(f.name == _field for f in dataclasses.fields(probe)):
            object.__setattr__(probe, _field, _value)

    check("probe pair really has no runner",
          (FAKE_FW, FAKE_TASK) not in MR.FRAMEWORK_RUNNERS)
    check("probe pair really has no request builder",
          (FAKE_FW, FAKE_TASK) not in MR.MODEL_REQUEST_BUILDERS)

    mc.MODEL_REGISTRY[FAKE_KEY] = probe
    mc.MODEL_REGISTRY_DICT[FAKE_KEY] = (probe.to_dict() if hasattr(probe, "to_dict")
                                        else dataclasses.asdict(probe))
    try:
        # THE regression: this used to pop FAKE_KEY out of MODEL_REGISTRY only.
        MR.validate_registry()

        check("unservable discovered row is KEPT in MODEL_REGISTRY",
              FAKE_KEY in mc.MODEL_REGISTRY)
        check("unservable discovered row is KEPT in MODEL_REGISTRY_DICT",
              FAKE_KEY in mc.MODEL_REGISTRY_DICT)
        check("registries stay symmetric across validate_registry()",
              set(mc.MODEL_REGISTRY) == set(mc.MODEL_REGISTRY_DICT))

        check("object form fetches it (this is what /assign's reason check calls)",
              cfg_main.get_model_config(FAKE_KEY) is not None)
        check("dict form fetches it too — the two forms agree",
              cfg_main.get_model_config(FAKE_KEY, dict_return=True) is not None)

        # ...and refusal moved to the POINT OF USE, naming the real cause.
        refused = None
        try:
            MR.resolve({"model_key": FAKE_KEY})
        except KeyError as exc:          # builder/runner lookups raise KeyError
            refused = str(exc)
        check("resolve() still REFUSES the unservable pair", refused is not None)
        check("the refusal names the (framework, task) pair, not the disk",
              FAKE_FW in refused and FAKE_TASK in refused
              and "on disk" not in refused.lower())
        check("the refusal names the missing runner or builder",
              "runner" in refused.lower() or "builder" in refused.lower())

        # An unservable row must not leak into the task-filtered serving registries
        # (chat/vision/whisper/embed) — its task matches none of their task lists.
        md.refresh_task_registries()
        for _name in ("CHAT_MODELS_REGISTRY", "VISION_MODELS_REGISTRY",
                      "WHISPER_MODELS_REGISTRY", "EMBED_MODELS_REGISTRY"):
            check(f"{_name} does not pick up the unservable row",
                  FAKE_KEY not in getattr(md, _name))
    finally:
        mc.MODEL_REGISTRY.pop(FAKE_KEY, None)
        mc.MODEL_REGISTRY_DICT.pop(FAKE_KEY, None)
        md.refresh_task_registries()

    check("probe cleaned up; registries symmetric again",
          FAKE_KEY not in mc.MODEL_REGISTRY
          and set(mc.MODEL_REGISTRY) == set(mc.MODEL_REGISTRY_DICT))


    # --- a CURATED staple with no runner still fails HARD -----------------------
    # Keeping discovered rows must NOT soften the code-bug check: MODELS (staples,
    # declared in code) still brick the import so the bug is caught at build time.
    STAPLE_KEY = "zz-symmetry-probe-broken-staple"
    staple = copy.deepcopy(probe)
    object.__setattr__(staple, "model_key", STAPLE_KEY)
    object.__setattr__(staple, "name", STAPLE_KEY)

    mc.MODELS[STAPLE_KEY] = {"hub_id": f"zz/{STAPLE_KEY}", "framework": FAKE_FW,
                             "primary_task": FAKE_TASK, "tasks": [FAKE_TASK]}
    mc.MODEL_REGISTRY[STAPLE_KEY] = staple
    mc.MODEL_REGISTRY_DICT[STAPLE_KEY] = (staple.to_dict() if hasattr(staple, "to_dict")
                                          else dataclasses.asdict(staple))
    try:
        raised = None
        try:
            MR.validate_registry()
        except RuntimeError as exc:
            raised = str(exc)
        check("a broken CURATED staple still raises RuntimeError", raised is not None)
        check("the RuntimeError names the offending staple", STAPLE_KEY in raised)
        check("the broken staple is NOT popped either (fail loud, don't hide)",
              STAPLE_KEY in mc.MODEL_REGISTRY)
    finally:
        mc.MODELS.pop(STAPLE_KEY, None)
        mc.MODEL_REGISTRY.pop(STAPLE_KEY, None)
        mc.MODEL_REGISTRY_DICT.pop(STAPLE_KEY, None)
        md.refresh_task_registries()

    # The registry must be back exactly as import left it — validate_registry()
    # is a REPORT, and this suite must not leave the process's registry dirty.
    MR.validate_registry()
    check("final state: no probes left, registries symmetric",
          STAPLE_KEY not in mc.MODEL_REGISTRY
          and FAKE_KEY not in mc.MODEL_REGISTRY
          and set(mc.MODEL_REGISTRY) == set(mc.MODEL_REGISTRY_DICT))

    print(f"\nall {ok} checks passed")
