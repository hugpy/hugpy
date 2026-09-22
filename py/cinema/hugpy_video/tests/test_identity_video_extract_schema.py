"""IDENTITY VIDEO-EXTRACT (char360) SPEC — the central-side, JSON-safe, validate-at-
construction currency of an ``identity_video_extract`` relay job (CHAR360-FEATURE-PLAN S3).

Exercises video_intel/identity_video_extract_schema.py in isolation (a pure-data module —
no store, no bus, no network):
  * make_identity_video_extract validates its fields (rejects a non-video source, a blank
    target; accepts both a "create" target and a slug target; filters char360_params to the
    keys the service understands; normalizes a blank identity_id to None);
  * identity_video_extract_from_dict round-trips an asdict() form back through the factory
    (deserialize-then-revalidate, like every other bus spec), and rejects a missing source.

Mirrors the style of the existing schema-shaped checks (a CHECKS list + a plain main()),
so it runs standalone AND is pytest-collectable (each ``test_*`` is independent).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/test_identity_video_extract_schema.py
  # or: venv/bin/python -m pytest tests/test_identity_video_extract_schema.py -q
"""
from __future__ import annotations

import dataclasses
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_video.intel.identity_video_extract_schema import (
    CHAR360_PARAM_KEYS,
    REVIEW_TARGET,
    IdentityVideoExtractSpec,
    identity_video_extract_from_dict,
    make_identity_video_extract,
)
from hugpy_video.intel.media_schema import make_media_ref


# --- fixtures (pure data — MediaRef only requires an absolute uri) ------------------ #
def _video() -> "object":
    return make_media_ref(asset_id="v1", kind="video",
                          uri="/mnt/llm_storage/uploads/clip.mp4", mime="video/mp4")


def _image() -> "object":
    return make_media_ref(asset_id="i1", kind="image",
                          uri="/mnt/llm_storage/uploads/still.png", mime="image/png")


# --------------------------------------------------------------------------- #
# [1] factory: a "create" target builds; char360_params is filtered to the known
#     keys; a blank identity_id normalizes to None.
# --------------------------------------------------------------------------- #
def test_factory_create_target_and_param_filter():
    spec = make_identity_video_extract(
        source=_video(), target="create",
        char360_params={"stride": 4, "min_faces": 2, "not_a_real_knob": 99},
        identity_id="   ")
    assert isinstance(spec, IdentityVideoExtractSpec), spec
    assert spec.target == "create", spec
    assert spec.source.kind == "video", spec
    # unknown keys dropped; only service-understood knobs survive.
    assert spec.char360_params == {"stride": 4, "min_faces": 2}, spec.char360_params
    for k in spec.char360_params:
        assert k in CHAR360_PARAM_KEYS, k
    # a blank identity_id -> None (the runner synthesizes a fallback correlation id).
    assert spec.identity_id is None, spec


# --------------------------------------------------------------------------- #
# [2] factory: a SLUG target builds and keeps an explicit identity_id.
# --------------------------------------------------------------------------- #
def test_factory_slug_target():
    spec = make_identity_video_extract(source=_video(), target="luigi", identity_id="luigi")
    assert spec.target == "luigi" and spec.identity_id == "luigi", spec
    # char360_params defaults to an EMPTY dict (never None) so asdict/json is stable.
    assert spec.char360_params == {}, spec.char360_params
    # A surrounding-whitespace target is stripped (structure-only normalization).
    spec2 = make_identity_video_extract(source=_video(), target="  mira  ")
    assert spec2.target == "mira", spec2


# --------------------------------------------------------------------------- #
# [3] factory REJECTS a non-video source (mirrors frame_schema's video guard).
# --------------------------------------------------------------------------- #
def test_factory_rejects_non_video_source():
    try:
        make_identity_video_extract(source=_image(), target="create")
    except ValueError as exc:
        assert "video" in str(exc).lower(), exc
    else:
        raise AssertionError("expected a non-video source to be rejected")

    # a non-MediaRef source is also a clean structural reject.
    try:
        make_identity_video_extract(source={"kind": "video"}, target="create")
    except ValueError:
        pass
    else:
        raise AssertionError("expected a raw-dict source to be rejected")


# --------------------------------------------------------------------------- #
# [4] factory REJECTS a blank/missing target and a non-dict char360_params.
# --------------------------------------------------------------------------- #
def test_factory_rejects_blank_target_and_bad_params():
    for bad in ("", "   ", None):
        try:
            make_identity_video_extract(source=_video(), target=bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"expected target={bad!r} to be rejected")

    try:
        make_identity_video_extract(source=_video(), target="create",
                                    char360_params=["stride", 4])  # a list, not a dict
    except ValueError:
        pass
    else:
        raise AssertionError("expected a non-dict char360_params to be rejected")


# --------------------------------------------------------------------------- #
# [5] from_dict round-trips an asdict() form back through the validating factory
#     (deserialize-then-revalidate), preserving source + target + params + id.
# --------------------------------------------------------------------------- #
def test_from_dict_round_trip():
    spec = make_identity_video_extract(
        source=_video(), target="luigi",
        char360_params={"stride": 6, "cluster_dist": 0.5}, identity_id="luigi")
    d = dataclasses.asdict(spec)                # asdict turns MediaRef into a plain dict
    spec2 = identity_video_extract_from_dict(d)
    assert spec2.target == spec.target, (spec2, spec)
    assert spec2.source.uri == spec.source.uri and spec2.source.kind == "video", spec2
    assert spec2.char360_params == spec.char360_params, spec2.char360_params
    assert spec2.identity_id == spec.identity_id, spec2

    # a create-target, param-less spec round-trips too (the common one-shot shape).
    bare = make_identity_video_extract(source=_video(), target="create")
    bare2 = identity_video_extract_from_dict(dataclasses.asdict(bare))
    assert bare2.target == "create" and bare2.char360_params == {} and bare2.identity_id is None


# --------------------------------------------------------------------------- #
# [6] from_dict REJECTS a payload with no source MediaRef (a construction-local raise,
#     never a silent None) — mirrors the bus's deserialize-then-revalidate discipline.
# --------------------------------------------------------------------------- #
def test_from_dict_rejects_missing_source():
    try:
        identity_video_extract_from_dict({"target": "create"})
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError("expected from_dict to reject a payload with no source")


# --------------------------------------------------------------------------- #
# [7] factory: the REVIEW target sentinel (CHARACTER-GROUPS-PLAN S1) builds like any
#     non-empty target and round-trips; blank identity_id stays None (the runner
#     synthesizes a correlation id for review, exactly as for create).
# --------------------------------------------------------------------------- #
def test_factory_review_target():
    spec = make_identity_video_extract(source=_video(), target=REVIEW_TARGET)
    assert spec.target == "review" and spec.identity_id is None, spec
    spec2 = identity_video_extract_from_dict(dataclasses.asdict(spec))
    assert spec2.target == "review" and spec2.source.kind == "video", spec2


CHECKS = [
    ("factory: create target + char360_params filtered + blank id -> None",
     test_factory_create_target_and_param_filter),
    ("factory: review target builds + round-trips (S1 curation sentinel)",
     test_factory_review_target),
    ("factory: slug target keeps identity_id; blank params default {}", test_factory_slug_target),
    ("factory: rejects a non-video source (and a raw-dict source)", test_factory_rejects_non_video_source),
    ("factory: rejects a blank target + a non-dict char360_params",
     test_factory_rejects_blank_target_and_bad_params),
    ("from_dict: asdict round-trip preserves source/target/params/id", test_from_dict_round_trip),
    ("from_dict: rejects a payload with no source MediaRef", test_from_dict_rejects_missing_source),
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
