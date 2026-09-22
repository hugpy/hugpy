"""id_lock must never silently render the WRONG PERSON.

Found by the model-routing trace, 2026-07-27. The worst failure shape in the
product: a plausible-looking clip, no error, wrong identity.

THE PATH:
  CAPABILITY_TASKS[ID_LOCK] = (VACE_CONTROL, I2V). Wan-VACE is the only runner
  that consumes reference images, and it maxes at 480p. Ask for id_lock at any
  larger geometry — INCLUDING the studio routes' own former 512x512 default — and
  no VACE model fits, so the router falls through to Task.I2V and binds
  wan2.1-i2v-14b-720p. ``run_wan_i2v`` contains ZERO references to
  ``reference_images`` (grep it), so the identity was simply dropped.

models_seed carried a comment asserting this could not happen ("in practice VACE
always wins ... so id_lock never silently routes to a runner that would ignore the
references"). It was true only inside the 480p envelope.

TWO FIXES, both asserted here:
  1. ``run_wan_i2v`` REFUSES a manifest carrying reference_images — at the point of
     harm, so it holds no matter how routing later changes.
  2. The studio routes default to 832x480 (R_480P) instead of 512x512, which was
     outside every id-capable and v2v-capable envelope on this fleet — a
     guaranteed-fail default (defaults-are-promises).

The route-default half of fix 2 is asserted against hugpy_server in
py/services/hugpy_server/tests/integration/test_id_lock_route_default_geometry.py.
Run: venv/bin/python -m pytest tests/test_id_lock_never_silently_drops.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_video.intel.studio.enums import Capability, DeterminismClass, Framework, Precision, Task
from hugpy_video.intel.studio.errors import ErrorCode
from hugpy_video.intel.studio.runners import wan_i2v as R
from hugpy_video.intel.studio.schemas import RenderManifest, Resolution, SamplerConfig, SeedBundle


def _manifest(reference_images=()):
    """A minimal VALID i2v manifest, varying only the identity references — the
    same construction the studio's own tests use (no router/env needed)."""
    return RenderManifest(
        render_id="idlock-guard",
        capability=Capability.ID_LOCK if reference_images else Capability.I2V,
        model_id="wan2.1-i2v-14b-720p",
        weight_hash=None,
        framework=Framework.WAN,
        task=Task.I2V,
        precision=Precision.BF16,
        seeds=SeedBundle(global_seed=7, stage_seeds=(("base", 7),)),
        sampler=SamplerConfig(sampler="euler", scheduler="normal", steps=30, cfg=6.0),
        resolution_ladder=(Resolution(832, 480, 24),),
        determinism_class=DeterminismClass.SEEDED_APPROX,
        env_snapshot=(),
        reference_images=tuple(reference_images),
    )


def test_the_i2v_runner_refuses_reference_images():
    """THE FIX. It cannot consume them, so it must not pretend to."""
    m = _manifest(("/tmp/subject.png",))
    res = R.run_wan_i2v(m, "/tmp/out")
    assert res.is_err(), "an id_lock manifest on the i2v runner MUST refuse"
    err = res.error
    assert err.code == ErrorCode.NO_CAPABLE_MODEL, err.code
    # The message has to be actionable: name the real path and the ceiling.
    msg = (err.message or "").lower()
    assert "vace" in msg and "480" in msg, err.message


def test_the_refusal_happens_before_any_work():
    """It must fire on the manifest alone — no weights, no GPU, no out_root
    needed. A guard that only trips after a 14 GB load is not a guard."""
    m = _manifest(("/nonexistent/a.png", "/nonexistent/b.png"))
    res = R.run_wan_i2v(m, "/nonexistent/out/root")
    assert res.is_err() and res.error.code == ErrorCode.NO_CAPABLE_MODEL
    assert "2 reference image" in (res.error.message or "")


def test_a_plain_i2v_render_is_untouched():
    """No references = not an id_lock request. This must fall through to the
    normal preflight (DEPS/NO_GPU/WEIGHTS on this box), NOT to the new refusal —
    or the guard would break every ordinary i2v render."""
    res = R.run_wan_i2v(_manifest(()), "/tmp/out")
    assert res.is_err(), "GPU-less box: preflight still refuses"
    assert res.error.code != ErrorCode.NO_CAPABLE_MODEL, (
        "a plain i2v render must not hit the identity guard")


def test_the_vace_runner_is_the_one_that_reads_references():
    """The asymmetry the guard exists for, asserted from source so it cannot
    silently invert: VACE consumes reference_images, i2v does not."""
    i2v_src = Path(R.__file__).read_text()
    vace_src = (Path(R.__file__).parent / "wan_vace.py").read_text()
    body_i2v = i2v_src.split("def run_wan_i2v", 1)[1]
    # the ONLY mentions in the i2v runner are the guard + its explanation
    assert "reference_images" in body_i2v, "the guard reads it"
    assert "reference_images" in vace_src, "VACE consumes it"
    assert vace_src.count("reference_images") > 5, (
        "VACE should genuinely thread references through, not just mention them")
