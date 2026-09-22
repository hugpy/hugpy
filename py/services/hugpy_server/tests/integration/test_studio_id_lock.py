"""IDENTITY LOCK (id_lock) — Wan VACE reference-to-video, the studio's flagship
identity thread. Conformance in the studio script style (plain python, ``__main__``
guard, numbered ``[n] PASS`` / ``[n] FAIL``, nonzero exit iff any check FAILED,
every check independent).

What is under test:
  * SPEC/FACTORY: reference_images coerce to an ordered tuple; >4 rejected; the VACE
    control channel (control_image + control_kind) is both-or-neither with a valid
    kind; asdict->from_dict round-trip preserves them.
  * HASH: reference images are CANONICAL and ORDER-PRESERVED — different refs (or a
    reorder) -> a different content_hash; control inputs are canonical when present.
  * ROUTER/REGISTRY: id_lock is routeable (CAPABILITY_TASKS prefers VACE_CONTROL); the
    vace rows claim ID_LOCK; validate_registry passes; id_lock@480p budget 9 binds
    wan2.1-vace-1.3b; portrait rejects (the VACE envelope is landscape-only).
  * RUNNER PREFLIGHT (GPU-less box, errors-as-data): id_lock with no refs ->
    REFERENCE_MISSING; a ghost ref -> REFERENCE_MISSING; a real ref -> DEPS_MISSING
    (the real VACE path is reached, then degrades on this bitsandbytes-less box).
  * ROUTE: reference_images[] jail-resolved + image-classified; id_lock requires >=1
    (400 otherwise); non-image / jail-escape / >4 rejected; control_* only with
    id_lock; a valid id_lock POST (+ control) -> 200 {job_id}.
  * PRESET: identity-lock-1.3b (id_lock, requires_reference, valid request_body).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_id_lock.py
"""
from __future__ import annotations
import pytest

import importlib
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import asdict

logging.disable(logging.INFO)

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from flask import Flask  # noqa: E402

from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_video.intel import media_bus
from hugpy_video.intel.studio.enums import Capability, Task
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.errors import Ok
from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.job import make_studio_i2v, studio_i2v_from_dict
from hugpy_video.intel.studio import produce as produce_mod
from hugpy_video.intel.studio.produce import produce_clip
from hugpy_video.intel.studio.registry import CAPABILITY_TASKS, MODEL_REGISTRY, validate_registry
from hugpy_video.intel.studio.router import CapabilityRouter
from hugpy_video.intel.studio.schemas import CapabilityRequest, Resolution
from hugpy_video.intel.studio_presets import get_studio_preset

_FFMPEG = shutil.which("ffmpeg") is not None
_FFPROBE = shutil.which("ffprobe") is not None
try:
    from PIL import Image
    _PIL = True
except Exception:  # noqa: BLE001
    _PIL = False

R_480 = Resolution(832, 480, 16)

# --- isolation: point the bus at a temp DB; fixtures live inside the storage jail ---
_TMP_DB = tempfile.mkstemp(prefix="idlock-bus-", suffix=".db")[1]
media_bus.DB_PATH = _TMP_DB
media_bus._initialized = False
with sqlite3.connect(_TMP_DB) as _c:
    _c.execute(
        "CREATE TABLE IF NOT EXISTS media_jobs ("
        "job_id TEXT PRIMARY KEY, name TEXT, status TEXT, spec_json TEXT, "
        "result_json TEXT, claim_token TEXT, created REAL, updated REAL, "
        "progress_json TEXT)")

_WORK = tempfile.mkdtemp(prefix="studio-idlock-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
_REF_PNG = os.path.join(_WORK, "subject.png")
_REF_PNG2 = os.path.join(_WORK, "subject2.png")
_CTRL_PNG = os.path.join(_WORK, "pose.png")
_NOT_IMAGE_MP4 = os.path.join(_WORK, "clip.mp4")

vr = importlib.import_module("hugpy_server.app.routes.video_routes")
app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()


def _make_png(path, color=(200, 120, 60)):
    Image.new("RGB", (96, 96), color).save(path, "PNG")


def _make_mp4(path):
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i",
         "testsrc=duration=1:size=160x120:rate=8", "-pix_fmt", "yuv420p", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def _setup_fixtures():
    if _PIL:
        _make_png(_REF_PNG, (200, 120, 60))
        _make_png(_REF_PNG2, (60, 120, 200))
        _make_png(_CTRL_PNG, (30, 30, 30))
    if _FFMPEG:
        _make_mp4(_NOT_IMAGE_MP4)


def _teardown_fixtures():
    shutil.rmtree(_WORK, ignore_errors=True)
    try:
        os.remove(_TMP_DB)
    except OSError:
        pass


@pytest.fixture(scope="module", autouse=True)
def _module_fixtures():
    """Under pytest, build and tear down the same fixture files main() does under __main__."""
    _setup_fixtures()
    yield
    _teardown_fixtures()


def _env():
    return StudioEnv(
        output_root="/out", weights_root="/weights", manifest_root="/manifests",
        master_colorspace="rec709", master_fps=16, max_vram_gb=24.0,
        loudness_target_lufs=-14.0, allow_unpinned=True)


# --------------------------------------------------------------------------- #
# (1) SPEC/FACTORY validation.
# --------------------------------------------------------------------------- #
def test_spec_factory_validation():
    # refs coerce to an ordered tuple; round-trip through asdict/from_dict.
    s = make_studio_i2v(capability="id_lock", width=832, height=480, fps=16,
                        vram_budget_gb=9.0, reference_images=["/a.png", "/b.png"])
    assert s.reference_images == ("/a.png", "/b.png"), s.reference_images
    r = studio_i2v_from_dict(asdict(s))
    assert r.reference_images == ("/a.png", "/b.png"), r.reference_images

    # >4 rejected.
    try:
        make_studio_i2v(capability="id_lock", width=832, height=480, fps=16,
                        reference_images=[f"/{i}.png" for i in range(5)])
        raise AssertionError(">4 reference_images must be rejected")
    except ValueError:
        pass

    # control both-or-neither + valid kind.
    for ci, ck in (("/c.png", None), (None, "pose")):
        try:
            make_studio_i2v(capability="id_lock", width=832, height=480, fps=16,
                            reference_images=["/a.png"], control_image=ci, control_kind=ck)
            raise AssertionError(f"control both-or-neither must reject ({ci},{ck})")
        except ValueError:
            pass
    try:
        make_studio_i2v(capability="id_lock", width=832, height=480, fps=16,
                        reference_images=["/a.png"], control_image="/c.png",
                        control_kind="bogus")
        raise AssertionError("bad control_kind must be rejected")
    except ValueError:
        pass
    # a valid control pair round-trips.
    sc = make_studio_i2v(capability="id_lock", width=832, height=480, fps=16,
                         reference_images=["/a.png"], control_image="/c.png",
                         control_kind="depth")
    rc = studio_i2v_from_dict(asdict(sc))
    assert rc.control_image == "/c.png" and rc.control_kind == "depth", rc


# --------------------------------------------------------------------------- #
# (2) HASH: references are canonical + order-preserved; control is canonical.
# --------------------------------------------------------------------------- #
def _hash_for(reference_images=(), control_image=None, control_kind=None):
    cap = {}

    def _stub(manifest, out_root, start_image=None, should_cancel=None):
        cap["h"] = manifest.content_hash()
        return Ok(Artifact(path="/x", content_hash=manifest.content_hash(),
                           frames=1, width=1, height=1, duration_s=1.0, resumed=False))

    key = (produce_mod.Framework.WAN, Task.VACE_CONTROL) if hasattr(produce_mod, "Framework") \
        else None
    from hugpy_video.intel.studio.enums import Framework
    key = (Framework.WAN, Task.VACE_CONTROL)
    orig = produce_mod._DISPATCH.get(key)
    produce_mod._DISPATCH[key] = _stub
    try:
        produce_clip(
            CapabilityRequest(capability=Capability.ID_LOCK,
                              target_resolution=R_480, vram_budget_gb=9.0),
            env=_env(), out_root="/tmp/none",
            reference_images=reference_images,
            control_image=control_image, control_kind=control_kind)
    finally:
        if orig is not None:
            produce_mod._DISPATCH[key] = orig
        else:
            produce_mod._DISPATCH.pop(key, None)
    return cap["h"]


def test_hash_keys_on_references_and_control():
    h_ab = _hash_for(reference_images=("/a.png", "/b.png"))
    h_ac = _hash_for(reference_images=("/a.png", "/c.png"))
    h_ba = _hash_for(reference_images=("/b.png", "/a.png"))
    h_none = _hash_for(reference_images=())
    assert h_ab != h_ac, "different reference sets must hash differently"
    assert h_ab != h_ba, "reference ORDER must be part of the hash"
    assert h_ab != h_none, "adding references must change the hash"
    # control is canonical when present.
    h_ctrl = _hash_for(reference_images=("/a.png",), control_image="/c.png",
                       control_kind="pose")
    h_ctrl2 = _hash_for(reference_images=("/a.png",), control_image="/c.png",
                        control_kind="depth")
    h_noctrl = _hash_for(reference_images=("/a.png",))
    assert h_ctrl != h_noctrl, "adding a control image must change the hash"
    assert h_ctrl != h_ctrl2, "control_kind must be part of the hash"


# --------------------------------------------------------------------------- #
# (3) REGISTRY + ROUTER: id_lock is routeable to VACE; portrait rejects.
# --------------------------------------------------------------------------- #
def test_registry_and_router_id_lock():
    assert not validate_registry(), validate_registry()
    tasks = CAPABILITY_TASKS.get(Capability.ID_LOCK)
    assert tasks and tasks[0] == Task.VACE_CONTROL, (
        f"id_lock must prefer VACE_CONTROL; got {tasks}")
    # both vace rows claim ID_LOCK.
    for mid in ("wan2.1-vace-1.3b", "wan2.1-vace-14b"):
        assert Capability.ID_LOCK in MODEL_REGISTRY.get(mid).capabilities, mid
    # id_lock @480p budget 9 -> vace-1.3b VACE_CONTROL.
    b = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.ID_LOCK, target_resolution=R_480, vram_budget_gb=9.0)).unwrap()
    assert b.model_id == "wan2.1-vace-1.3b" and b.task == Task.VACE_CONTROL, b
    # budget 6 also binds vace-1.3b (INT8 floor); portrait rejects.
    b6 = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.ID_LOCK, target_resolution=R_480, vram_budget_gb=6.0)).unwrap()
    assert b6.model_id == "wan2.1-vace-1.3b", b6
    rp = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.ID_LOCK, target_resolution=Resolution(480, 832, 16),
        vram_budget_gb=9.0))
    assert rp.is_err(), "portrait id_lock must reject (VACE envelope is landscape-only)"


# --------------------------------------------------------------------------- #
# (4) RUNNER PREFLIGHT (GPU-less box, errors-as-data).
# --------------------------------------------------------------------------- #
def _produce_id_lock(reference_images=(), out=None):
    return produce_clip(
        CapabilityRequest(capability=Capability.ID_LOCK, target_resolution=R_480,
                          vram_budget_gb=9.0),
        env=_env(), out_root=(out or tempfile.gettempdir()),
        reference_images=reference_images)


def test_runner_preflight_reference_missing():
    # id_lock with NO references -> REFERENCE_MISSING (spec error, before deps/GPU).
    res = _produce_id_lock(reference_images=())
    assert res.is_err() and res.error.code.value == "reference_missing", (
        f"no-ref id_lock must be reference_missing; got {res}")
    # a ghost reference (path that doesn't exist) -> REFERENCE_MISSING.
    res2 = _produce_id_lock(reference_images=("/nope/ghost.png",))
    assert res2.is_err() and res2.error.code.value == "reference_missing", res2


def test_runner_preflight_real_ref_degrades_deps_missing():
    # a REAL reference reaches the real VACE path, which degrades on this
    # bitsandbytes-less box -> DEPS_MISSING (proving refs pass the spec check).
    if not _PIL:
        print("      (PIL unavailable — skipping real-ref preflight)")
        return
    res = _produce_id_lock(reference_images=(_REF_PNG,))
    assert res.is_err(), f"must be Err on this GPU-less box; got {res}"
    assert res.error.code.value in ("deps_missing", "no_gpu", "weights_missing"), (
        f"a real-ref id_lock must degrade at a preflight code, not reference_missing; "
        f"got {res.error.code.value}")


# --------------------------------------------------------------------------- #
# (5) ROUTE validation.
# --------------------------------------------------------------------------- #
def test_route_id_lock_requires_reference():
    r = client.post("/video/studio/i2v", json={
        "capability": "id_lock",
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 9.0})
    assert r.status_code == 400, (r.status_code, r.get_json())
    assert "reference" in (r.get_json().get("error", "").lower()), r.get_json()


def test_route_id_lock_valid_200():
    if not _PIL:
        print("      (PIL unavailable — skipping route id_lock 200)")
        return
    r = client.post("/video/studio/i2v", json={
        "capability": "id_lock",
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 9.0,
        "reference_images": [_REF_PNG, _REF_PNG2],
        "prompt": "the subject walks forward"})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert isinstance(r.get_json().get("job_id"), str), r.get_json()


def test_route_id_lock_non_image_rejected():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping non-image reject)")
        return
    r = client.post("/video/studio/i2v", json={
        "capability": "id_lock",
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 9.0,
        "reference_images": [_NOT_IMAGE_MP4]})
    assert r.status_code == 400, (r.status_code, r.get_json())
    assert "image" in r.get_json().get("error", "").lower(), r.get_json()


def test_route_id_lock_jail_escape_rejected():
    r = client.post("/video/studio/i2v", json={
        "capability": "id_lock",
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 9.0,
        "reference_images": ["/etc/passwd"]})
    assert r.status_code in (400, 404), (r.status_code, r.get_json())


def test_route_too_many_refs_rejected():
    if not _PIL:
        print("      (PIL unavailable — skipping >4 refs reject)")
        return
    r = client.post("/video/studio/i2v", json={
        "capability": "id_lock",
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 9.0,
        "reference_images": [_REF_PNG] * 5})
    assert r.status_code == 400, (r.status_code, r.get_json())


def test_route_refs_require_vace_capability():
    # reference_images on a non-VACE capability (i2v) -> 400.
    if not _PIL:
        print("      (PIL unavailable — skipping refs-capability reject)")
        return
    r = client.post("/video/studio/i2v", json={
        "capability": "i2v",
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 9.0,
        "reference_images": [_REF_PNG]})
    assert r.status_code == 400, (r.status_code, r.get_json())


def test_route_control_only_with_id_lock():
    # control_* on a non-id_lock capability -> 400.
    if not _PIL:
        print("      (PIL unavailable — skipping control-capability reject)")
        return
    r = client.post("/video/studio/i2v", json={
        "capability": "v2v",
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 6.0,
        "control_image": _CTRL_PNG, "control_kind": "pose"})
    assert r.status_code == 400, (r.status_code, r.get_json())
    # bad control_kind on id_lock -> 400.
    r2 = client.post("/video/studio/i2v", json={
        "capability": "id_lock",
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 9.0,
        "reference_images": [_REF_PNG],
        "control_image": _CTRL_PNG, "control_kind": "bogus"})
    assert r2.status_code == 400, (r2.status_code, r2.get_json())


def test_route_id_lock_with_control_200():
    if not _PIL:
        print("      (PIL unavailable — skipping id_lock+control 200)")
        return
    r = client.post("/video/studio/i2v", json={
        "capability": "id_lock",
        "resolution": {"width": 832, "height": 480, "fps": 16},
        "vram_budget_gb": 9.0,
        "reference_images": [_REF_PNG],
        "control_image": _CTRL_PNG, "control_kind": "pose",
        "prompt": "match this pose"})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert isinstance(r.get_json().get("job_id"), str), r.get_json()


# --------------------------------------------------------------------------- #
# (6) PRESET.
# --------------------------------------------------------------------------- #
def test_preset_identity_lock():
    p = get_studio_preset("identity-lock-1.3b")
    assert p is not None and p.capability == "id_lock", p
    assert p.requires_reference is True, "id_lock preset must signal requires_reference"
    assert p.to_dict()["requires_reference"] is True and p.apply()["requires_reference"] is True
    # request_body() is a valid make_studio_i2v body (references threaded by the route).
    make_studio_i2v(**p.request_body())
    # binds vace-1.3b at its budget.
    b = CapabilityRouter().resolve(CapabilityRequest(
        capability=Capability.ID_LOCK,
        target_resolution=Resolution(p.width, p.height, p.fps),
        vram_budget_gb=p.vram_budget_gb)).unwrap()
    assert b.model_id == "wan2.1-vace-1.3b", b


CHECKS = [
    ("spec/factory: refs->ordered tuple, >4 rejected, control both-or-neither + kind",
     test_spec_factory_validation),
    ("hash: references canonical + ORDER-preserved; control canonical",
     test_hash_keys_on_references_and_control),
    ("registry/router: id_lock prefers VACE, binds vace-1.3b@480p, portrait rejects",
     test_registry_and_router_id_lock),
    ("runner preflight: no-ref / ghost-ref -> REFERENCE_MISSING (spec error)",
     test_runner_preflight_reference_missing),
    ("runner preflight: real ref reaches VACE path -> DEPS_MISSING on this box",
     test_runner_preflight_real_ref_degrades_deps_missing),
    ("route: id_lock without reference_images -> 400", test_route_id_lock_requires_reference),
    ("route: id_lock with real image refs -> 200 {job_id}", test_route_id_lock_valid_200),
    ("route: reference_images pointing at a non-image -> 400",
     test_route_id_lock_non_image_rejected),
    ("route: jail-escaping reference_image -> 4xx", test_route_id_lock_jail_escape_rejected),
    ("route: >4 reference_images -> 400", test_route_too_many_refs_rejected),
    ("route: reference_images require id_lock/v2v capability -> 400",
     test_route_refs_require_vace_capability),
    ("route: control_* only valid with id_lock; bad control_kind -> 400",
     test_route_control_only_with_id_lock),
    ("route: id_lock + control_image (pose) -> 200 {job_id}",
     test_route_id_lock_with_control_200),
    ("preset: identity-lock-1.3b (id_lock, requires_reference, valid body, binds vace-1.3b)",
     test_preset_identity_lock),
]


def main() -> int:
    _setup_fixtures()
    passed = failed = 0
    try:
        for i, (name, fn) in enumerate(CHECKS, 1):
            try:
                fn()
            except Exception as exc:  # surface EVERY divergence
                failed += 1
                import traceback
                print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"[{i}] PASS  {name}")
    finally:
        _teardown_fixtures()
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
