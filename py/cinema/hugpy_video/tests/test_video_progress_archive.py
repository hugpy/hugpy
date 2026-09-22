"""Backend tests for the video "live progress + auto-archive" feature.

Covers (GPU-free, script-style with a __main__ guard like the sibling tests):

  1. media_bus.set_progress + get() round-trip the live progress blob, and the
     `progress` key is None at enqueue / nulled at terminal.
  2. The idempotent `ALTER TABLE ... ADD COLUMN progress_json` migration works on
     an EXISTING (pre-feature) DB and is safe to run repeatedly.
  3. The optional `project` NAME survives serialize -> deserialize through the
     bus (the enqueue path re-validates the spec).
  4. scene._write_bundle produces assets/<projectmeta>/{frames, video.mp4,
     project.json} with the pinned project.json keys (stubbed frame set, no GPU).
  5. imagegen._write_image_bundle produces the single-image bundle + project.json.

Isolation: media_bus.DB_PATH is repointed to a PRIVATE temp sqlite db (the
_selftest_scene idiom) so the real job bus is never touched; the bundle writers'
module-level DEFAULT_ROOT is monkeypatched to a temp dir so nothing lands under
the shared storage root.

Run:
  /home/ubuntu/station/dev/abstract_hugpy_dev/venv/bin/python \
      tests/test_video_progress_archive.py
"""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _private_bus():
    """Repoint media_bus.DB_PATH to a private temp db + reset the one-time init
    flag, so these checks drain their OWN store in isolation."""
    from hugpy_video.intel import media_bus
    tmpdir = tempfile.mkdtemp(prefix="hugpy_test_progress_")
    media_bus.DB_PATH = os.path.join(tmpdir, "media_jobs.db")
    media_bus._initialized = False
    return media_bus, tmpdir


def _scene_spec(project=None, seed=1000):
    from hugpy_video.intel.scene_schema import make_generate_scene
    from hugpy_video.intel.gen_schema import text_part
    return make_generate_scene(
        parts=(text_part("a serene landscape, slowly panning"),),
        model_id="sd-turbo",
        width=256, height=256, steps=2, guidance=0.0,
        n_frames=3, fps=8, assemble=True,
        seed=seed, negative="blurry", motion="pan {i}/{n}",
        chain=False, project=project,
    )


# --------------------------------------------------------------------------- #
# 1) set_progress + get() round-trip; None at enqueue / terminal
# --------------------------------------------------------------------------- #
def test_progress_roundtrip():
    media_bus, _ = _private_bus()
    job_id = media_bus.enqueue("generate_scene", _scene_spec(project="Round Trip"))

    view = media_bus.get(job_id)
    assert "progress" in view, view
    assert view["progress"] is None, f"progress should be None at enqueue: {view}"

    blob = {
        "done": 2, "total": 3, "stage": "generating",
        "label": "frame 2/3 — a serene landscape",
        "model": "sd-turbo",
        "frames": [
            {"asset_id": "abc", "kind": "image", "uri": "/x/frame_00000.png",
             "mime": "image/png", "width": 256, "height": 256},
        ],
        "started_at": 1234.5, "eta_s": 0.75,
    }
    media_bus.set_progress(job_id, blob)

    got = media_bus.get(job_id)["progress"]
    assert got == blob, f"progress blob did not round-trip:\n got={got}\n exp={blob}"

    # Simulate a terminal write nulling the progress (run_claimed does this).
    from hugpy_video.intel.result_schema import JobResult
    conn = media_bus._connect()
    try:
        conn.execute(
            "UPDATE media_jobs SET status='done', result_json=?, progress_json=NULL, "
            "updated=? WHERE job_id=?",
            (media_bus.serialize_result(JobResult(job_id=job_id, ok=True)),
             1.0, job_id),
        )
    finally:
        conn.close()
    term = media_bus.get(job_id)
    assert term["status"] == "done", term
    assert term["progress"] is None, f"progress must be null at terminal: {term}"
    print("[1] PASS  set_progress/get round-trip + None at enqueue/terminal")


# --------------------------------------------------------------------------- #
# 2) idempotent ALTER migration on a pre-existing (pre-feature) DB
# --------------------------------------------------------------------------- #
def test_migration_idempotent():
    from hugpy_video.intel import media_bus
    tmpdir = tempfile.mkdtemp(prefix="hugpy_test_migrate_")
    db = os.path.join(tmpdir, "media_jobs.db")

    # Build an OLD-schema DB (no progress_json) with a row already present.
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE media_jobs ("
        " job_id TEXT PRIMARY KEY, name TEXT, status TEXT, spec_json TEXT,"
        " result_json TEXT, claim_token TEXT, created REAL, updated REAL)"
    )
    conn.execute(
        "INSERT INTO media_jobs (job_id, name, status, created, updated) "
        "VALUES ('old-job', 'generate_scene', 'running', 1.0, 1.0)"
    )
    conn.commit()
    conn.close()

    def _cols():
        c = sqlite3.connect(db)
        try:
            return {r[1] for r in c.execute("PRAGMA table_info(media_jobs)")}
        finally:
            c.close()

    assert "progress_json" not in _cols(), "precondition: old DB lacks the column"

    media_bus.DB_PATH = db
    media_bus._initialized = False
    media_bus._ensure_db()                      # ALTER adds the column
    assert "progress_json" in _cols(), "migration did not add progress_json"

    # Idempotent: re-running _ensure_db (duplicate-column ALTER) must not raise.
    media_bus._initialized = False
    media_bus._ensure_db()
    media_bus._initialized = False
    media_bus._ensure_db()
    assert "progress_json" in _cols()

    # The pre-existing row now carries a usable progress column.
    media_bus.set_progress("old-job", {"done": 1, "total": 2, "stage": "generating"})
    assert media_bus.get("old-job")["progress"] == {
        "done": 1, "total": 2, "stage": "generating"}
    print("[2] PASS  ALTER migration adds progress_json + idempotent on re-run")


# --------------------------------------------------------------------------- #
# 3) the project NAME survives serialize -> deserialize through the bus
# --------------------------------------------------------------------------- #
def test_project_field_roundtrips():
    from hugpy_video.intel import media_bus
    import json

    # scene
    spec = _scene_spec(project="My Cool Scene!")
    spec_json = media_bus.serialize_spec("generate_scene", spec)
    back = media_bus.deserialize_spec("generate_scene", json.loads(spec_json))
    assert back.project == "My Cool Scene!", back.project

    # image
    from hugpy_video.intel.gen_schema import make_generate_image, text_part
    img = make_generate_image(
        parts=(text_part("a red cube"),),
        model_id="sd-turbo", width=256, height=256, steps=2, guidance=0.0,
        project="Img Proj",
    )
    img_json = media_bus.serialize_spec("generate_image", img)
    img_back = media_bus.deserialize_spec("generate_image", json.loads(img_json))
    assert img_back.project == "Img Proj", img_back.project

    # empty/blank project -> None (factory normalizes)
    assert _scene_spec(project="").project is None
    assert _scene_spec(project=None).project is None
    print("[3] PASS  project NAME survives serialize->deserialize (scene + image)")


# --------------------------------------------------------------------------- #
# 4) scene._write_bundle -> assets/<projectmeta>/{frames, video.mp4, project.json}
# --------------------------------------------------------------------------- #
_SCENE_MANIFEST_KEYS = {
    "project_name", "project_uuid", "model_key", "prompt", "negative", "chain",
    "width", "height", "steps", "guidance", "n_frames", "fps", "strength",
    "seeds", "frames", "mp4", "started_at", "finished_at", "per_frame_secs",
    # oracle directive (or-k2/or-p1): a chained scene DECLARES itself
    "legacy_pixel_chain", "limitations", "label",
}


def test_scene_bundle_writer():
    import json
    from hugpy_video.intel.runners import scene

    workroot = tempfile.mkdtemp(prefix="hugpy_test_scenebundle_")
    # frames live in a synthetic out_dir; the bundle lands under a temp DEFAULT_ROOT.
    out_dir = os.path.join(workroot, "scene_out")
    os.makedirs(out_dir, exist_ok=True)
    frame_paths = []
    for i in range(3):
        fp = os.path.join(out_dir, f"frame_{i:05d}.png")
        with open(fp, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + bytes([i]) * 32)  # stub PNG bytes
        frame_paths.append(fp)
    mp4_path = os.path.join(out_dir, "scene.mp4")
    with open(mp4_path, "wb") as fh:
        fh.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)

    spec = _scene_spec(project="Sunset Pan", seed=1000)
    projectmeta = "sunset_pan"

    # Point the writer's DEFAULT_ROOT at our temp dir (module-level import).
    saved = scene.DEFAULT_ROOT
    scene.DEFAULT_ROOT = workroot
    try:
        bundle_dir = scene._write_bundle(
            spec=spec, job_id="job-uuid-xyz", projectmeta=projectmeta,
            frame_paths=frame_paths, mp4_path=mp4_path,
            base_prompt="a serene landscape, slowly panning",
            started_at=100.0, finished_at=112.5,
            per_frame_secs=[4.0, 4.1, 3.9],
        )
    finally:
        scene.DEFAULT_ROOT = saved

    assert bundle_dir == os.path.join(workroot, "assets", projectmeta), bundle_dir
    # frames copied
    for i in range(3):
        assert os.path.isfile(os.path.join(bundle_dir, f"frame_{i:05d}.png")), i
    # mp4 copied as video.mp4
    assert os.path.isfile(os.path.join(bundle_dir, "video.mp4")), "video.mp4 missing"
    # project.json present with the pinned keys + expected values
    pj = os.path.join(bundle_dir, "project.json")
    assert os.path.isfile(pj), "project.json missing"
    with open(pj) as fh:
        manifest = json.load(fh)
    assert set(manifest) == _SCENE_MANIFEST_KEYS, (
        f"key mismatch:\n got={sorted(manifest)}\n exp={sorted(_SCENE_MANIFEST_KEYS)}")
    assert manifest["project_name"] == "Sunset Pan"
    assert manifest["project_uuid"] == "job-uuid-xyz"
    assert manifest["model_key"] == "sd-turbo"
    assert manifest["chain"] is False
    assert manifest["legacy_pixel_chain"] is False
    assert manifest["label"] == "independent"
    assert manifest["limitations"] == []
    assert manifest["n_frames"] == 3
    assert manifest["seeds"] == [1000, 1001, 1002], manifest["seeds"]
    assert manifest["frames"] == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    assert manifest["mp4"] == "video.mp4"
    assert manifest["per_frame_secs"] == [4.0, 4.1, 3.9]
    print("[4] PASS  scene _write_bundle -> assets/<meta>/{frames,video.mp4,project.json}")

    # random-seed variant: seeds -> "random", no mp4 -> mp4 None
    spec2 = _scene_spec(project=None, seed=None)
    scene.DEFAULT_ROOT = workroot
    try:
        bundle2 = scene._write_bundle(
            spec=spec2, job_id="job-noseed", projectmeta="job-noseed",
            frame_paths=frame_paths[:1], mp4_path=None,
            base_prompt="p", started_at=1.0, finished_at=2.0, per_frame_secs=[1.0],
        )
    finally:
        scene.DEFAULT_ROOT = saved
    with open(os.path.join(bundle2, "project.json")) as fh:
        m2 = json.load(fh)
    assert m2["seeds"] == "random", m2["seeds"]
    assert m2["mp4"] is None, m2["mp4"]
    assert m2["project_name"] is None
    print("[4b] PASS scene _write_bundle seeds='random' + mp4=None when unassembled")


# --------------------------------------------------------------------------- #
# 5) imagegen._write_image_bundle -> assets/<projectmeta>/{image, project.json}
# --------------------------------------------------------------------------- #
_IMAGE_MANIFEST_KEYS = {
    "project_name", "project_uuid", "model_key", "prompt", "negative",
    "width", "height", "steps", "guidance", "n_frames", "strength", "seed",
    "frames", "image", "started_at", "finished_at",
}


def test_image_bundle_writer():
    import json
    from hugpy_video.intel.runners import imagegen
    from hugpy_video.intel.gen_schema import make_generate_image, text_part

    workroot = tempfile.mkdtemp(prefix="hugpy_test_imgbundle_")
    img_path = os.path.join(workroot, "gen_0.png")
    with open(img_path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    spec = make_generate_image(
        parts=(text_part("a red cube"),),
        model_id="sd-turbo", width=256, height=256, steps=2, guidance=0.0,
        seed=7, negative="blurry", project="Cube Study",
    )

    saved = imagegen.DEFAULT_ROOT
    imagegen.DEFAULT_ROOT = workroot
    try:
        bundle_dir = imagegen._write_image_bundle(
            spec, "img-job-uuid", "cube_study", img_path, "a red cube",
            started_at=5.0, finished_at=6.0,
        )
    finally:
        imagegen.DEFAULT_ROOT = saved

    assert os.path.isfile(os.path.join(bundle_dir, "gen_0.png")), "image not copied"
    with open(os.path.join(bundle_dir, "project.json")) as fh:
        m = json.load(fh)
    assert set(m) == _IMAGE_MANIFEST_KEYS, (
        f"key mismatch:\n got={sorted(m)}\n exp={sorted(_IMAGE_MANIFEST_KEYS)}")
    assert m["project_name"] == "Cube Study"
    assert m["project_uuid"] == "img-job-uuid"
    assert m["n_frames"] == 1
    assert m["seed"] == 7
    assert m["frames"] == ["gen_0.png"] and m["image"] == "gen_0.png"
    print("[5] PASS  imagegen _write_image_bundle -> assets/<meta>/{image,project.json}")


# --------------------------------------------------------------------------- #
def _run_all():
    test_progress_roundtrip()
    test_migration_idempotent()
    test_project_field_roundtrips()
    test_scene_bundle_writer()
    test_image_bundle_writer()
    print("\nALL video progress+archive backend checks passed")


if __name__ == "__main__":
    _run_all()
