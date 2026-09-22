"""Backend test for the HONEST 'start image required' early error.

Closes the silent-t2i-fallback gap: an image-to-image-ONLY model (a native edit
model like Qwen-Image-Edit — cfg.tasks lists "image-to-image" but NOT
"text-to-image") run through generate_image / generate_scene with NO start image
used to fall through to task="text-to-image" and die LATE inside the plane. It
now refuses UP FRONT with code="start_image_required" (non-retryable).

GPU-free (script-style with a __main__ guard, like the sibling tests):

  1. text_to_image_available: False for an i2i-ONLY catalog model
     (a3527183~Qwen-Image-Edit-2509), True for a text-to-image model (sd-turbo).
  2. start_image_required predicate — all four (model x has_image) combos.
  3. run_generate_image on an i2i-ONLY model with NO image returns
     code="start_image_required" — a REAL early return, before any plane drive.
  4. The guard does NOT fire when it shouldn't: run_generate_image on a
     text-to-image model with no image, and on the i2i-ONLY model WITH an image,
     both pass the guard and reach the (stubbed) inference plane instead.

The inference plane is STUBBED (guard_gpu_worker -> no refusal; execute_prompt ->
raises a sentinel) so the negative cases prove the guard was passed and the plane
reached, with no GPU. media_bus.DB_PATH is repointed to a private temp db so the
best-effort set_progress call never touches the real job bus.

Run:
  /home/ubuntu/station/dev/abstract_hugpy_dev/venv/bin/python \
      tests/test_video_start_image_required.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# A confirmed image-to-image-ONLY catalog key (tasks == ['image-to-image']); and
# a plain text-to-image staple (tasks include text-to-image + image-to-image).
I2I_ONLY_MODEL = "a3527183~Qwen-Image-Edit-2509"
T2I_MODEL = "sd-turbo"


def _require_catalog_fixtures():
    """These checks are about the PREDICATE, not the catalog: skip (never
    fail) when this box's model catalog lacks the fixture rows."""
    import pytest
    from hugpy_engine.resolvers.model_resolver import resolve_model_key
    for key, task in ((I2I_ONLY_MODEL, "image-to-image"), (T2I_MODEL, "text-to-image")):
        try:
            resolve_model_key(model_key=key, task=task)
        except Exception as exc:  # noqa: BLE001 - unknown model on this box
            pytest.skip(f"catalog fixture {key!r} unavailable here: {exc}")

_STUB_SENTINEL = "STUB-PLANE-REACHED"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _private_bus():
    """Repoint media_bus.DB_PATH to a private temp db (best-effort set_progress
    in run_generate_image must never touch the real bus)."""
    from hugpy_video.intel import media_bus
    tmpdir = tempfile.mkdtemp(prefix="hugpy_test_startimg_")
    media_bus.DB_PATH = os.path.join(tmpdir, "media_jobs.db")
    media_bus._initialized = False
    return tmpdir


def _image_spec(model_id, with_image=False):
    """A minimal GenerateImageSpec with a text part and, optionally, a real
    on-disk image part (so the img2img path is reachable)."""
    from hugpy_video.intel.gen_schema import make_generate_image, text_part, image_part
    from hugpy_video.intel.media_schema import make_media_ref
    parts = [text_part("make the sky orange")]
    if with_image:
        from PIL import Image
        d = tempfile.mkdtemp(prefix="hugpy_test_startimg_init_")
        png = os.path.join(d, "init.png")
        Image.new("RGB", (128, 128), (10, 20, 30)).save(png)
        ref = make_media_ref(asset_id="t", kind="image", uri=png, mime="image/png",
                             width=128, height=128)
        parts.append(image_part(ref))
    return make_generate_image(
        parts=tuple(parts), model_id=model_id,
        width=256, height=256, steps=2, guidance=0.0,
    )


def _stub_plane():
    """Neutralize the GPU guard and make execute_prompt raise a sentinel, so a
    request that passes the start-image guard deterministically reaches — and is
    caught at — the plane drive (no GPU, no real generation). Monkeypatch persists
    for the run.

    NOTE the patch target: the dispatch package re-exports execute_prompt via
    `from .dispatch import *`, and the package dir also holds a submodule named
    `dispatch`. The runner binds `from abstract_hugpy_dev.managers.dispatch import
    execute_prompt`, which reads the PACKAGE object in
    sys.modules['hugpy_engine.dispatch'] — NOT the submodule that a
    plain `from abstract_hugpy_dev.managers import dispatch` yields. So we patch
    the sys.modules entry the runner actually reads."""
    import sys
    from hugpy_video.intel.runners import _gpu_guard
    _gpu_guard.guard_gpu_worker = lambda model_id, job_id: None

    def _boom(**kwargs):
        raise RuntimeError(_STUB_SENTINEL)

    from hugpy_video.intel import plane
    plane.execute_prompt = _boom


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def test_text_to_image_available():
    _require_catalog_fixtures()
    from hugpy_video.intel.runners._img2img import text_to_image_available, img2img_available
    assert text_to_image_available(I2I_ONLY_MODEL) is False, \
        "i2i-only model must NOT report text-to-image available"
    assert img2img_available(I2I_ONLY_MODEL) is True, \
        "i2i-only model must report image-to-image available"
    assert text_to_image_available(T2I_MODEL) is True, \
        "text-to-image model must report text-to-image available"
    print("ok  text_to_image_available: i2i-only=False, t2i=True")


def test_start_image_required_predicate():
    _require_catalog_fixtures()
    from hugpy_video.intel.runners._img2img import start_image_required
    # fires ONLY for i2i-only + no image
    assert start_image_required(I2I_ONLY_MODEL, has_start_image=False) is True
    assert start_image_required(I2I_ONLY_MODEL, has_start_image=True) is False
    assert start_image_required(T2I_MODEL, has_start_image=False) is False
    assert start_image_required(T2I_MODEL, has_start_image=True) is False
    print("ok  start_image_required predicate: fires only for (i2i-only, no image)")


def test_generate_image_i2i_only_no_image_refuses():
    """The critical positive case — a REAL early return, no plane drive/GPU."""
    _require_catalog_fixtures()
    _private_bus()
    from hugpy_video.intel.runners.imagegen import run_generate_image
    res = run_generate_image(_image_spec(I2I_ONLY_MODEL, with_image=False), "job-i2i-noimg")
    assert res.ok is False, "must fail"
    assert res.error is not None and res.error.code == "start_image_required", \
        f"expected start_image_required, got {res.error and res.error.code!r}"
    assert res.error.retryable is False, "start_image_required must be non-retryable"
    print(f"ok  generate_image(i2i-only, no image) -> {res.error.code} "
          f"(retryable={res.error.retryable}) msg={res.error.message!r}")


def test_generate_image_t2i_no_image_does_not_fire():
    """A text-to-image model with no image must PASS the guard and reach the
    (stubbed) plane — never start_image_required."""
    _private_bus()
    _stub_plane()
    from hugpy_video.intel.runners.imagegen import run_generate_image
    res = run_generate_image(_image_spec(T2I_MODEL, with_image=False), "job-t2i-noimg")
    assert res.ok is False
    assert res.error.code != "start_image_required", \
        f"guard wrongly fired for a t2i model: {res.error.code!r}"
    assert res.error.code == "generation_failed" and _STUB_SENTINEL in res.error.message, \
        f"expected to reach the stubbed plane; got {res.error.code}: {res.error.message}"
    print(f"ok  generate_image(t2i, no image) passed guard -> reached plane "
          f"({res.error.code})")


def test_generate_image_i2i_only_with_image_does_not_fire():
    """The i2i-only model WITH a start image must PASS the guard and reach the
    (stubbed) img2img plane — never start_image_required."""
    _private_bus()
    _stub_plane()
    from hugpy_video.intel.runners.imagegen import run_generate_image
    res = run_generate_image(_image_spec(I2I_ONLY_MODEL, with_image=True), "job-i2i-img")
    assert res.ok is False
    assert res.error.code != "start_image_required", \
        f"guard wrongly fired despite a start image: {res.error.code!r}"
    assert res.error.code == "generation_failed" and _STUB_SENTINEL in res.error.message, \
        f"expected to reach the stubbed plane; got {res.error.code}: {res.error.message}"
    print(f"ok  generate_image(i2i-only, WITH image) passed guard -> reached plane "
          f"({res.error.code})")


def main():
    test_text_to_image_available()
    test_start_image_required_predicate()
    test_generate_image_i2i_only_no_image_refuses()
    test_generate_image_t2i_no_image_does_not_fire()
    test_generate_image_i2i_only_with_image_does_not_fire()
    print("\nALL PASSED")


if __name__ == "__main__":
    main()
