"""IDENTITY CLEANUP-PROMPT (C1) — the additive render-steer channels on the identity
specs (operator-requested 2026-07-15).

Two knobs, both additive, both default ``""`` -> byte-identical to today
(defaults-are-promises):
  * ``IdentityMeshSpec.cleanup_prompt``  — a positive-worded avoid instruction woven into
    the T-pose front render prompt (C2 wires it into ``_TPOSE_PROMPT``).
  * ``IdentityMeshSpec.negative_prompt`` — a true negative forwarded to the studio render.
  * ``IdentityReconstructionSpec.negative_prompt`` — the reconstruction/turntable render's
    true negative (base_prompt already covers the positive channel there, so ONLY the
    negative is added to the reconstruction spec).

This is a PURE-SCHEMA test (no store / no bus / no GPU): it exercises both
validating factories and their serialization round trips.

Run (both as pytest and as a script; run ALONE — the identity test family cross-pollutes):
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest tests/test_identity_cleanup_prompt_schema.py -q
  venv/bin/python tests/test_identity_cleanup_prompt_schema.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_video.intel.identity_reconstruction_schema import (
    make_identity_mesh,
    identity_mesh_from_dict,
    identity_reconstruction_from_dict,
    IdentityMeshSpec,
    IdentityReconstructionSpec,
)


def _mesh(**kw) -> IdentityMeshSpec:
    base = dict(slug="s", recon_id="r", view_sources=[("front", "/a.png")])
    base.update(kw)
    return make_identity_mesh(**base)


# --------------------------------------------------------------------------- #
# MESH SPEC — cleanup_prompt / negative_prompt default "" and round-trip.
# --------------------------------------------------------------------------- #
def test_mesh_defaults_empty_byte_identical():
    m = _mesh()
    assert m.cleanup_prompt == "", m.cleanup_prompt
    assert m.negative_prompt == "", m.negative_prompt


def test_mesh_set_and_roundtrip():
    m = _mesh(cleanup_prompt="no object on her back, clean bare back",
              negative_prompt="backpack, symbols, prop")
    d = asdict(m)
    assert d["cleanup_prompt"] == "no object on her back, clean bare back"
    assert d["negative_prompt"] == "backpack, symbols, prop"
    m2 = identity_mesh_from_dict(d)
    assert m2.cleanup_prompt == m.cleanup_prompt
    assert m2.negative_prompt == m.negative_prompt


def test_mesh_old_dict_no_keys_defaults_empty():
    # An OLD serialized spec (pre-cleanup) has neither key -> both default "" (byte-
    # identical render), so deserialization stays backward-compat.
    old = {"slug": "s", "recon_id": "r", "view_sources": [["front", "/a.png"]]}
    m = identity_mesh_from_dict(old)
    assert m.cleanup_prompt == "" and m.negative_prompt == ""


def test_mesh_none_coerces_to_empty():
    m = _mesh(cleanup_prompt=None, negative_prompt=None)
    assert m.cleanup_prompt == "" and m.negative_prompt == ""


def test_mesh_non_str_raises():
    # make_identity_mesh is SINGLY defined -> its validation is live.
    for bad in (dict(cleanup_prompt=5), dict(cleanup_prompt=["x"]),
                dict(negative_prompt=5), dict(negative_prompt={"a": 1})):
        try:
            _mesh(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


# --------------------------------------------------------------------------- #
# RECONSTRUCTION SPEC — negative_prompt default "" and validated round-trip.
# --------------------------------------------------------------------------- #
def test_reconstruction_default_empty():
    spec = IdentityReconstructionSpec(slug="s", recon_id="r", source_images=("/a.png",))
    assert spec.negative_prompt == "", spec.negative_prompt


def test_reconstruction_negative_roundtrip_live_path():
    d = {"slug": "s", "recon_id": "r", "source_images": ["/a.png"],
         "negative_prompt": "backpack, symbols"}
    rc = identity_reconstruction_from_dict(d)
    assert rc.negative_prompt == "backpack, symbols", rc.negative_prompt
    # re-serialize -> re-hydrate is stable
    rc2 = identity_reconstruction_from_dict(asdict(rc))
    assert rc2.negative_prompt == "backpack, symbols"


def test_reconstruction_old_dict_defaults_empty():
    rc = identity_reconstruction_from_dict(
        {"slug": "s", "recon_id": "r", "source_images": ["/a.png"]})
    assert rc.negative_prompt == ""


def test_reconstruction_factory_validates_public_calls_and_deserialization():
    from hugpy_video.intel.identity_reconstruction_schema import make_identity_reconstruction

    base = {"slug": "s", "recon_id": "r", "source_images": ["/a.png"]}
    for bad in ({"source_images": []}, {"negative_prompt": 5},
                {"seed": True}, {"width": 0}):
        values = {**base, **bad}
        for factory in (make_identity_reconstruction, identity_reconstruction_from_dict):
            try:
                factory(**values) if factory is make_identity_reconstruction else factory(values)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError for {bad!r}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} CLEANUP-PROMPT SCHEMA CHECKS PASSED")
