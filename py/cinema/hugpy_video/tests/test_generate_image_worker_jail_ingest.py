"""Backend test: a worker-produced generate_image output is ingested through
central's storage jail regardless of the worker's local storage root.

Live bug (central 0.2.1.post21, 2026-09-24): text-to-image job on worker
ae-worker (unix user `aeb`, CO-LOCATED on the SAME host `ae` as central) died at
ingest with:

    ValueError: ingest: path escapes storage jail:
      /mnt/nvmes/2T_samsung_990/hot990/aeb/uploads/generated/req-..._0.png

The SAME flow on the genuinely-remote worker computron (192.168.1.128) succeeds.

Root cause: run_generate_image used the plane result's `path` DIRECTLY whenever
os.path.isfile(path) was True. A co-located worker writes its output to its OWN
uploads root; that path is visible on central's disk (same host) so isfile() is
True, but it escapes central's jail and media_store.ingest rejects it. A truly
remote worker's path is NOT a local file, so the runner already materialized the
inline b64 bytes into UPLOADS_HOME/generated (jailed) — which is why computron
worked. The fix: a returned path is only usable directly when it is a local file
AND within central's jail (UPLOADS_HOME / DEFAULT_ROOT); otherwise materialize
from the b64 bytes the result rides (return_b64=True), exactly as the remote path
already does.

GPU/ffprobe-free: the inference plane is stubbed to return a fixed result, and
media_store.ingest is stubbed to CAPTURE the path handed to it (so the assertions
are about WHICH path central would ingest — no real probe, no real generation).
The storage roots are repointed to a private temp jail so no real storage tree is
touched.

Run:
  <venv>/bin/python tests/test_generate_image_worker_jail_ingest.py
"""
import base64
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

T2I_MODEL = "sd-turbo"


@pytest.fixture(autouse=True)
def _restore_patched_globals():
    """The harness below patches module globals (storage roots, bus path, plane,
    GPU guard, ingest, bundle writer). Restore them after every test so the stubs
    never leak into other test modules sharing the process."""
    from hugpy_platform import constants
    from hugpy_video.intel import media_bus, plane
    from hugpy_video.intel.runners import _gpu_guard, imagegen
    patched = [
        (constants, "UPLOADS_HOME"), (constants, "DEFAULT_ROOT"),
        (media_bus, "DB_PATH"), (media_bus, "_initialized"),
        (plane, "execute_prompt"), (_gpu_guard, "guard_gpu_worker"),
        (imagegen, "DEFAULT_ROOT"), (imagegen, "ingest"),
        (imagegen, "_write_image_bundle"),
    ]
    saved = [(mod, name, getattr(mod, name)) for mod, name in patched]
    yield
    for mod, name, value in saved:
        setattr(mod, name, value)

# A minimal but real 1x1 PNG (so b64 decode writes genuine image bytes to disk).
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9"
    "awAAAABJRU5ErkJggg=="
)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
def _private_bus():
    """Repoint media_bus.DB_PATH to a private temp db (best-effort set_progress
    in run_generate_image must never touch the real bus)."""
    from hugpy_video.intel import media_bus
    tmpdir = tempfile.mkdtemp(prefix="hugpy_test_jail_bus_")
    media_bus.DB_PATH = os.path.join(tmpdir, "media_jobs.db")
    media_bus._initialized = False
    return tmpdir


def _repoint_jail():
    """Repoint UPLOADS_HOME and DEFAULT_ROOT to a private temp jail. Returns the
    jail root. Patches the constants module (run_generate_image reads UPLOADS_HOME
    at call time) AND the imagegen module global DEFAULT_ROOT (bound at import)."""
    from hugpy_platform import constants
    from hugpy_video.intel.runners import imagegen
    jail = tempfile.mkdtemp(prefix="hugpy_test_jail_root_")
    constants.UPLOADS_HOME = jail
    constants.DEFAULT_ROOT = jail
    imagegen.DEFAULT_ROOT = jail
    return jail


def _stub_plane(result):
    """Neutralize the GPU guard and make execute_prompt return a FIXED result
    (plain value; async_runtime.run tolerates non-coroutines)."""
    from hugpy_video.intel.runners import _gpu_guard
    _gpu_guard.guard_gpu_worker = lambda model_id, job_id: None
    from hugpy_video.intel import plane
    plane.execute_prompt = lambda **kwargs: result


def _capture_ingest():
    """Replace media_store.ingest (as imported into imagegen) with a capture stub
    that records the path handed to it and returns a minimal ref. Returns the
    list that receives the captured paths."""
    from hugpy_video.intel.runners import imagegen
    seen = []

    def _fake_ingest(path, kind_hint=None, sid=None, owner=None, private=False):
        seen.append(path)
        return SimpleNamespace(uri=path, kind="image")

    imagegen.ingest = _fake_ingest
    # the auto-archive bundle copies ref.uri; neutralize it so the test asserts
    # only the ingest decision (bundle IO is already best-effort in the runner).
    imagegen._write_image_bundle = lambda *a, **k: "/dev/null"
    return seen


def _spec():
    from hugpy_video.intel.gen_schema import make_generate_image, text_part
    return make_generate_image(
        parts=(text_part("a red apple"),), model_id=T2I_MODEL,
        width=256, height=256, steps=2, guidance=0.0,
    )


def _result(path, b64):
    img = SimpleNamespace(path=path, b64=b64)
    return SimpleNamespace(ok=True, images=[img], error=None)


def _b64png():
    return base64.b64encode(_PNG_1x1).decode("ascii")


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def test_colocated_worker_path_escapes_jail_is_rematerialized():
    """THE LIVE BUG. A co-located worker returns a path that EXISTS on central's
    disk (same host) but is OUTSIDE the jail, plus b64. The runner must ignore
    the escaping path and materialize the b64 into <UPLOADS_HOME>/generated —
    so ingest receives a JAILED path, not the worker's local one."""
    _private_bus()
    jail = _repoint_jail()
    seen = _capture_ingest()

    # a real on-disk file OUTSIDE the jail (the co-located worker's own uploads)
    worker_dir = tempfile.mkdtemp(prefix="hugpy_test_workerlocal_")
    worker_path = os.path.join(worker_dir, "req-abc_0.png")
    with open(worker_path, "wb") as fh:
        fh.write(_PNG_1x1)
    assert os.path.isfile(worker_path)  # visible to central (same host)

    _stub_plane(_result(worker_path, _b64png()))
    from hugpy_video.intel.runners.imagegen import run_generate_image
    res = run_generate_image(_spec(), "job-colo")

    assert res.ok is True, f"must succeed, got {res.error and res.error.message!r}"
    assert len(seen) == 1, f"ingest called once, got {seen}"
    ingested = seen[0]
    assert ingested != worker_path, \
        f"runner ingested the worker's ESCAPING path directly: {ingested}"
    assert os.path.realpath(ingested).startswith(os.path.realpath(jail)), \
        f"ingested path is not jailed: {ingested} (jail {jail})"
    assert ingested == os.path.join(jail, "generated", "job-colo_0.png"), \
        f"unexpected materialized path: {ingested}"
    assert os.path.isfile(ingested), "materialized image was not written"
    print(f"ok  co-located escaping path re-materialized -> {ingested}")


def test_remote_worker_path_absent_is_rematerialized():
    """The genuinely-remote case (computron): the worker path is NOT a local
    file. Existing behavior — materialize the b64 into the jail — is preserved."""
    _private_bus()
    jail = _repoint_jail()
    seen = _capture_ingest()

    _stub_plane(_result("/nonexistent/remote/req-xyz_0.png", _b64png()))
    from hugpy_video.intel.runners.imagegen import run_generate_image
    res = run_generate_image(_spec(), "job-remote")

    assert res.ok is True, f"must succeed, got {res.error and res.error.message!r}"
    assert seen and seen[0] == os.path.join(jail, "generated", "job-remote_0.png"), \
        f"remote path not materialized into jail: {seen}"
    assert os.path.isfile(seen[0])
    print(f"ok  remote absent path re-materialized -> {seen[0]}")


def test_jailed_local_path_used_directly():
    """REGRESSION: a path that IS a local file AND already inside the jail must be
    ingested DIRECTLY (no needless re-materialize) — the fast path central relies
    on when it serves in-process / a worker shares central's store."""
    _private_bus()
    jail = _repoint_jail()
    seen = _capture_ingest()

    gen_dir = os.path.join(jail, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    jailed_path = os.path.join(gen_dir, "already_here_0.png")
    with open(jailed_path, "wb") as fh:
        fh.write(_PNG_1x1)

    _stub_plane(_result(jailed_path, _b64png()))
    from hugpy_video.intel.runners.imagegen import run_generate_image
    res = run_generate_image(_spec(), "job-jailed")

    assert res.ok is True, f"must succeed, got {res.error and res.error.message!r}"
    assert seen and seen[0] == jailed_path, \
        f"a jailed local path must be used directly, got {seen}"
    print(f"ok  jailed local path used directly -> {seen[0]}")


def test_colocated_escaping_path_no_bytes_returns_data_error():
    """DEFENSIVE: a co-located escaping path with NO b64 must return a clean DATA
    error (generation_no_output), never ingest the escaping path (which would
    raise the jail ValueError across the job boundary as the live bug did)."""
    _private_bus()
    _repoint_jail()
    seen = _capture_ingest()

    worker_dir = tempfile.mkdtemp(prefix="hugpy_test_workerlocal_nobytes_")
    worker_path = os.path.join(worker_dir, "req-nob_0.png")
    with open(worker_path, "wb") as fh:
        fh.write(_PNG_1x1)

    _stub_plane(_result(worker_path, None))
    from hugpy_video.intel.runners.imagegen import run_generate_image
    res = run_generate_image(_spec(), "job-nobytes")

    assert res.ok is False, "must fail as DATA, not raise"
    assert res.error is not None and res.error.code == "generation_no_output", \
        f"expected generation_no_output, got {res.error and res.error.code!r}"
    assert not seen, f"ingest must NOT be called on an escaping path: {seen}"
    print(f"ok  escaping path + no bytes -> {res.error.code} (no ingest)")


def main():
    test_colocated_worker_path_escapes_jail_is_rematerialized()
    test_remote_worker_path_absent_is_rematerialized()
    test_jailed_local_path_used_directly()
    test_colocated_escaping_path_no_bytes_returns_data_error()
    print("\nALL PASSED")


if __name__ == "__main__":
    main()
