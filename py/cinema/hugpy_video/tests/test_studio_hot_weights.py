"""HOT weights root (item 5) — a per-box NVMe copy that loads faster than the shared
/mnt/llm_storage mount. Conformance in the same script style as the other studio
tests (plain python, ``__main__`` guard, numbered ``[n] PASS`` / ``[n] FAIL`` lines,
nonzero exit iff any check FAILED, every check independently run).

What is under test (video_intel/studio/runners/wan_i2v resolution helpers, reused by
wan_vace + ltx_upscale):
  * _resolve_model_dir order: hot NVMe root (STUDIO_WEIGHTS_HOT_ROOT, box-local,
    process-env ONLY) wins WHEN it holds the model (model_index.json present); else
    the shared/snapshot root, unchanged.
  * hot set but the model is NOT on the hot copy -> transparently falls back to shared.
  * neither root holds the model -> WEIGHTS_MISSING message names BOTH roots tried.
  * env UNSET -> byte-identical to the historical shared-root resolution.
  * the hot var is NEVER captured into the manifest env_snapshot (the only weights-
    location input to content_hash), so the hot path cannot change a clip's hash.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_hot_weights.py
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile

logging.disable(logging.INFO)

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.runners.wan_i2v import (
    _hot_weights_root,
    _local_model_dir,
    _resolve_model_dir,
    _weights_missing_msg,
)

_HOT_ENV = "STUDIO_WEIGHTS_HOT_ROOT"
_WEIGHT_URI = "Wan-AI/Wan2.1-T2V-1.3B"


class _M:
    """Minimal manifest stand-in: _resolve_model_dir/_weights_root only read
    ``env_snapshot`` (the shared root the manifest captured on central)."""

    def __init__(self, shared_root: str | None):
        self.env_snapshot = (
            (("STUDIO_WEIGHTS_ROOT", shared_root),) if shared_root else ())


def _stage_model(root: str, weight_uri: str = _WEIGHT_URI) -> str:
    """Create ``<root>/<org>/<name>/model_index.json`` (the completeness gate) and
    return the model dir."""
    d = _local_model_dir(root, weight_uri)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "model_index.json"), "w") as fh:
        fh.write("{}")
    return d


def _clear_hot() -> None:
    os.environ.pop(_HOT_ENV, None)


# --------------------------------------------------------------------------- #
# (1) hot-root HIT: hot holds the model -> resolves to the hot dir ("hot").
# --------------------------------------------------------------------------- #
def test_hot_root_hit():
    _clear_hot()
    tmp = tempfile.mkdtemp(prefix="hotw_hit_")
    try:
        hot, shared = os.path.join(tmp, "hot"), os.path.join(tmp, "shared")
        hot_dir = _stage_model(hot)
        _stage_model(shared)                       # shared has it too, but hot wins
        os.environ[_HOT_ENV] = hot
        d, tag = _resolve_model_dir(_M(shared), _WEIGHT_URI)
        assert tag == "hot", f"hot copy present must serve; got tag {tag!r}"
        assert d == hot_dir, f"must resolve the hot dir; got {d}"
        assert _hot_weights_root() == hot
    finally:
        _clear_hot()
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (2) hot-root MISS: hot set but lacks the model -> falls back to the shared root.
# --------------------------------------------------------------------------- #
def test_hot_root_miss_falls_back_to_shared():
    _clear_hot()
    tmp = tempfile.mkdtemp(prefix="hotw_miss_")
    try:
        hot, shared = os.path.join(tmp, "hot"), os.path.join(tmp, "shared")
        os.makedirs(hot, exist_ok=True)           # hot exists but has NO model
        shared_dir = _stage_model(shared)
        os.environ[_HOT_ENV] = hot
        d, tag = _resolve_model_dir(_M(shared), _WEIGHT_URI)
        # Operator tiering doctrine 2026-08-13: a hot root WITHOUT the model
        # warms it (copy-on-first-use, atomic) and serves the hot copy; only a
        # failed warm falls back to the shared root.
        assert tag == "hot", f"a hot miss must warm and serve the hot copy; tag {tag!r}"
        assert d != shared_dir and d.startswith(hot), f"must resolve under the hot root; got {d}"
        assert os.path.isfile(os.path.join(d, "model_index.json")), "warm must be complete"
    finally:
        _clear_hot()
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (3) hot set but model ABSENT everywhere -> shared dir (no model), and the
#     WEIGHTS_MISSING message names BOTH roots tried.
# --------------------------------------------------------------------------- #
def test_hot_set_but_model_absent_names_both_roots():
    _clear_hot()
    tmp = tempfile.mkdtemp(prefix="hotw_absent_")
    try:
        hot, shared = os.path.join(tmp, "hot"), os.path.join(tmp, "shared")
        os.environ[_HOT_ENV] = hot
        d, tag = _resolve_model_dir(_M(shared), _WEIGHT_URI)
        # Falls through to the shared dir (which also lacks the model), so the caller's
        # model_index.json check will report WEIGHTS_MISSING.
        assert tag == "shared" and d == _local_model_dir(shared, _WEIGHT_URI), (tag, d)
        assert not os.path.isfile(os.path.join(d, "model_index.json"))
        msg = _weights_missing_msg(_WEIGHT_URI, hot, shared)
        assert "hot NVMe" in msg and "shared" in msg, f"message must name both roots: {msg}"
        assert _WEIGHT_URI in msg and _local_model_dir(hot, _WEIGHT_URI) in msg, msg
    finally:
        _clear_hot()
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (4) env UNSET -> byte-identical to the historical shared-root resolution; and no
#     shared root configured -> (None, "shared").
# --------------------------------------------------------------------------- #
def test_env_unset_is_todays_behavior():
    _clear_hot()
    tmp = tempfile.mkdtemp(prefix="hotw_unset_")
    try:
        shared = os.path.join(tmp, "shared")
        shared_dir = _stage_model(shared)
        assert _hot_weights_root() is None, "unset hot root must be None"
        d, tag = _resolve_model_dir(_M(shared), _WEIGHT_URI)
        assert tag == "shared" and d == shared_dir, (tag, d)
        assert d == _local_model_dir(shared, _WEIGHT_URI), "must equal the old resolution"
        # No shared root configured at all -> (None, "shared") (caller reports MISSING).
        d2, tag2 = _resolve_model_dir(_M(None), _WEIGHT_URI)
        assert d2 is None and tag2 == "shared", (d2, tag2)
    finally:
        _clear_hot()
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (5) the hot var NEVER enters the manifest env_snapshot -> cannot change content_hash
#     (content_hash's only weights-location input is env_snapshot.STUDIO_WEIGHTS_ROOT).
# --------------------------------------------------------------------------- #
def test_hot_root_never_in_env_snapshot():
    _clear_hot()
    env = StudioEnv(
        output_root="/out", weights_root="/shared/weights", manifest_root="/manifests",
        master_colorspace="rec709", master_fps=16, max_vram_gb=24.0,
        loudness_target_lufs=-14.0, allow_unpinned=True)
    os.environ[_HOT_ENV] = "/mnt/root_800/hugpy-ae-hot-drive"
    try:
        snap = dict(env.to_snapshot())
        assert _HOT_ENV not in snap, f"hot root must NOT be captured into env_snapshot: {snap}"
        assert snap.get("STUDIO_WEIGHTS_ROOT") == "/shared/weights", (
            f"only the shared root belongs in the snapshot; got {snap.get('STUDIO_WEIGHTS_ROOT')}")
    finally:
        _clear_hot()


CHECKS = [
    ("hot-root HIT: hot copy present -> resolves the hot dir ('hot')", test_hot_root_hit),
    ("hot-root MISS: hot lacks the model -> falls back to shared ('shared')",
     test_hot_root_miss_falls_back_to_shared),
    ("hot set but model absent -> WEIGHTS_MISSING message names BOTH roots",
     test_hot_set_but_model_absent_names_both_roots),
    ("env UNSET -> byte-identical shared resolution; no shared root -> (None,'shared')",
     test_env_unset_is_todays_behavior),
    ("hot var NEVER in env_snapshot -> cannot change content_hash",
     test_hot_root_never_in_env_snapshot),
]


def main() -> int:
    passed = failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:  # surface EVERY divergence, not just the first
            failed += 1
            import traceback
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
