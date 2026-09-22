"""GET /video/presets + POST /video/presets/<id>/apply — route contract.

Verifies the video-preset HTTP surface WITHOUT a live worker/catalog touch:
  * GET  /video/presets            -> {"presets":[...]} carrying (at least) the
                                       known seed presets, each in the pinned
                                       wire shape (incl. the advisory sampler);
  * POST /video/presets/<bad>/apply-> 404 (get_preset -> None short-circuits
                                       before any catalog/worker work).

Also covers the FOURTH preset surface on this blueprint, GET /video/render/presets
(added 2026-07-27) — the RENDER PRESETS from video_intel/studio/presets.py, the only
preset table grounded in measurement rather than curation. Those checks live at the
bottom of this file and are deliberately written against the REGISTRY, not against a
fixture copy of it: see the section header there.

And, at the very bottom, THE BOUNDARY REFUSAL: POST /video/studio/i2v must 400 a
capability no preset covers BEFORE minting a job_id. That is a different kind of
check from everything above it — the others read a table, this one drives the real
enqueue path with the real test client and asserts on the status code a caller
actually receives. It exists because the table and the route had DRIFTED: the
registry knew eight capabilities were dead and the route accepted all sixteen.

The known-id checks are subset assertions (``<=``) so that adding presets to the
registry does not break this contract test — only removing/renaming a known one
or dropping a pinned wire key does.

Mirrors test_reap_approve_route.py's idiom (temp PROJECTS_HOME, a minimal Flask
app with the blueprint mounted, a test_client), but exposed as pytest tests so
`python -m pytest tests/test_video_presets_route.py` reports a clean pass.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Keep any bus/audit writes out of the real projects tree.
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-video-presets-test-"))

import importlib

from flask import Flask

vr = importlib.import_module(
    "hugpy_server.app.routes.video_routes")

app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()

# The pinned per-preset wire shape.
_TOP_KEYS = {"id", "name", "description", "mode", "model_key", "defaults", "recommended"}
_DEFAULT_KEYS = {"strength", "steps", "guidance", "width", "height",
                 "n_frames", "fps", "negative"}
# The known preset ids -> catalog model_key (the 3 seeds + the 5 ComfyUI Field
# Guide presets). Subset-checked, so future additions won't break this test.
_EXPECTED = {
    "realistic-edit-chain": "a3527183~Qwen-Image-Edit-2509",
    "realistic-img2img": "comfy-juggernautxl-ragnarok",
    "fast-draft": "sdxl-turbo",
    "photoreal-portrait-sd15": "comfy-epicrealism-naturalsinrc1vae",
    "photoreal-sdxl": "comfy-juggernautxl-ragnarok",
    "anime-stylized": "comfy-neverendingdreamned-v122bakedvae",
    "painterly-art": "comfy-dreamshaper-8",
    "sdxl-lightning": "comfy-dreamshaperxl-lightningdpmsde",
}
# The known modes (subset-checked alongside _EXPECTED).
_EXPECTED_MODES = {
    "realistic-edit-chain": "edit-chain",
    "realistic-img2img": "img2img",
    "fast-draft": "text-to-image",
    "photoreal-portrait-sd15": "text-to-image",
    "photoreal-sdxl": "text-to-image",
    "anime-stylized": "text-to-image",
    "painterly-art": "text-to-image",
    "sdxl-lightning": "text-to-image",
}


def test_get_presets_contract_shape():
    r = client.get("/video/presets")
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert isinstance(body, dict) and "presets" in body
    presets = body["presets"]
    assert isinstance(presets, list) and presets, presets

    by_id = {p["id"]: p for p in presets}
    # every known preset id is present (subset — additions are fine)
    assert set(_EXPECTED) <= set(by_id), set(by_id)

    for pid, model_key in _EXPECTED.items():
        p = by_id[pid]
        # every pinned top-level key present
        assert _TOP_KEYS <= set(p), (pid, set(p))
        assert p["model_key"] == model_key, (pid, p["model_key"])
        assert p["recommended"] == "gpu", (pid, p["recommended"])
        # defaults sub-object carries exactly the pinned keys
        assert _DEFAULT_KEYS <= set(p["defaults"]), (pid, set(p["defaults"]))

    # the advisory sampler field is present in every preset's wire shape
    for p in presets:
        assert "sampler" in p, (p.get("id"), set(p))


def test_get_presets_modes():
    presets = client.get("/video/presets").get_json()["presets"]
    modes = {p["id"]: p["mode"] for p in presets}
    for pid, mode in _EXPECTED_MODES.items():
        assert modes.get(pid) == mode, (pid, modes.get(pid))


def test_apply_unknown_preset_404():
    r = client.post("/video/presets/does-not-exist/apply")
    assert r.status_code == 404, r.status_code
    body = r.get_json()
    assert body.get("ok") is False, body
    assert body.get("error", {}).get("code") == "UnknownPreset", body


# --------------------------------------------------------------------------- #
# MOVIE templates — the img2img-drift "street-walk" preset (strength+negative)
# --------------------------------------------------------------------------- #
# The empirically-tuned street-walk template is the first movie preset to carry a
# strength + negative; assert both reach the directly-POSTable generate_movie body
# (apply().request) and that its 4-goal timeline tiles [0, 12). storm-front is the
# regression: an existing preset still constructs + applies unchanged.
_STREET_WALK_NEGATIVE = ("different person, face change, identity change, "
                         "deformed face, extra limbs, warped body, morphing, blurry")


def _assert_tiles(goals, total):
    """The goal timeline is contiguous, non-overlapping and tiles [0, total)."""
    assert goals and goals[0]["start_frame"] == 0, goals
    cursor = 0
    for g in goals:
        assert g["start_frame"] == cursor, (cursor, g)
        assert g["end_frame"] > g["start_frame"], g
        cursor = g["end_frame"]
    assert cursor == total, (cursor, total)


def test_movie_presets_list_includes_street_walk():
    r = client.get("/movie/presets")
    assert r.status_code == 200, r.status_code
    presets = r.get_json()["presets"]
    by_id = {p["id"]: p for p in presets}
    assert "street-walk" in by_id, sorted(by_id)
    sw = by_id["street-walk"]
    assert sw["model_key"] == "comfy-juggernautxl-ragnarok", sw["model_key"]
    # strength + negative are surfaced at the top level AND in the settings bundle
    assert sw["strength"] == 0.45, sw["strength"]
    assert sw["negative"] == _STREET_WALK_NEGATIVE, sw["negative"]
    assert sw["settings"]["strength"] == 0.45, sw["settings"]
    assert sw["settings"]["negative"] == _STREET_WALK_NEGATIVE, sw["settings"]
    _assert_tiles(sw["goals"], 12)


def test_movie_preset_street_walk_apply_carries_strength_negative():
    r = client.post("/movie/presets/street-walk/apply")
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert body.get("ok") is True, body
    assert body.get("id") == "street-walk", body
    req = body["request"]
    # the POSTable generate_movie body MUST carry strength + negative
    assert req["strength"] == 0.45, req["strength"]
    assert req["negative"] == _STREET_WALK_NEGATIVE, req["negative"]
    assert req["model_id"] == "comfy-juggernautxl-ragnarok", req["model_id"]
    # 4 contiguous goals tiling [0, 12)
    assert len(req["goals"]) == 4, req["goals"]
    _assert_tiles(req["goals"], 12)


def test_movie_preset_storm_front_regression():
    """An existing preset (no strength/negative set) still applies — defaults ride
    through: strength None, negative "" — proving the 6 seeds are unaffected."""
    r = client.post("/movie/presets/storm-front/apply")
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert body.get("ok") is True, body
    req = body["request"]
    assert req["model_id"] == "comfy-juggernautxl-ragnarok", req["model_id"]
    assert req["strength"] is None, req["strength"]
    assert req["negative"] == "", req["negative"]
    _assert_tiles(req["goals"], 12)


# --------------------------------------------------------------------------- #
# RENDER PRESETS — GET /video/render/presets (2026-07-27)
# --------------------------------------------------------------------------- #
# The other suites above subset-check against hardcoded expectations, which is the
# right shape for a CURATED table that grows freely. The render presets are a
# different animal: the table is the deliverable ("adding a ninth means proving a
# ninth"), and the whole failure this endpoint exists to prevent is a route layer
# that drifts from the measured truth. So these checks:
#
#   * compare the payload FIELD BY FIELD against ``studio.presets.all_presets()``
#     rather than against a copy of it — a fixture copy here would only prove that
#     two hardcoded lists agree, which is exactly the thing that rots;
#   * demand ALL EIGHT rows (equality, not subset), so a preset silently dropped
#     from the wire is a failure and a ninth is a deliberate, visible edit;
#   * prove DERIVATION rather than assert it, by swapping the registry accessor and
#     re-requesting: if the route ever caches or hardcodes, the response will not
#     follow and this fails. The swap is restored in a finally so the module-level
#     client stays usable by the script-style __main__ run below.
from hugpy_video.intel.studio import presets as _render_presets

# The pinned per-preset wire shape. EXACT (==), not subset: this is a fresh contract
# with no live callers to protect yet, so a stray key is caught now rather than
# frozen in by the first console that reads it.
_RENDER_KEYS = {
    "id", "title", "description", "capability", "capabilities", "model",
    "framework", "task", "precision", "geometry", "width", "height", "fps",
    "default_frames", "max_frames", "inputs", "proven", "evidence",
    "composes", "joints",
    # PLACEMENT (added 2026-07-29 for the compatibility-aware studio console). Derived,
    # never restated — see test_render_presets_publish_derived_placement below.
    "vram_envelope_gb", "vram_need_gib", "fits_render_box",
}
_RENDER_ENVELOPE_KEYS = {
    "presets", "unavailable", "menu", "frame_cadence", "render_box",
    "render_box_vram_gib",
}


def test_render_presets_returns_all_eight():
    r = client.get("/video/render/presets")
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert set(body) == _RENDER_ENVELOPE_KEYS, set(body)
    rows = body["presets"]
    # EIGHT rows, in the registry's ratified order (clips -> movie -> enhance), which
    # is not alphabetical and is meaningful — so order is asserted, not just membership.
    assert len(rows) == 8, [p["id"] for p in rows]
    assert [p["id"] for p in rows] == [p.preset_id for p in _render_presets.all_presets()]


def test_render_presets_payload_matches_the_registry():
    """Every field of every row, read off the dataclass — no fixture copy anywhere."""
    rows = {p["id"]: p for p in client.get("/video/render/presets").get_json()["presets"]}
    for p in _render_presets.all_presets():
        row = rows[p.preset_id]
        assert set(row) == _RENDER_KEYS, (p.preset_id, set(row) ^ _RENDER_KEYS)
        assert row["title"] == p.title
        assert row["description"] == p.description
        # primary capability + the full set. Every ratified row serves exactly one
        # today; the plural key survives the retirement of clip-control-480p (which
        # listed four and rendered a fifth thing) because the shape should keep that
        # lesson visible rather than collapse it away.
        assert row["capability"] == p.capability.value
        assert row["capabilities"] == [c.value for c in p.capabilities]
        assert row["model"] == p.model_id
        assert row["framework"] == p.framework.value
        assert row["task"] == p.task.value
        assert row["precision"] == p.precision.value
        # geometry is the registry's own string — "source" on the two ffmpeg enhance
        # presets, whose width/height/fps/frames are all None by design.
        assert row["geometry"] == p.geometry
        assert (row["width"], row["height"], row["fps"]) == (p.width, p.height, p.fps)
        assert row["default_frames"] == p.default_frames
        assert row["max_frames"] == p.max_frames
        assert row["inputs"] == list(p.inputs)
        assert row["proven"] is p.proven
        assert row["evidence"] == p.evidence
        assert row["composes"] == list(p.composes)
        assert row["joints"] == list(p.joints)


def test_render_presets_publish_derived_placement():
    """The three placement fields must be DERIVED from the registry + the runner's own
    arithmetic, not typed into the route.

    They exist because the studio console was mirroring these numbers by hand and had
    drifted: `STUDIO_REAL_FLOOR_GB = 6` in the SPA against a cheapest real envelope of
    8.2, so a budget of 7 bound the synthetic prover while the honesty banner stayed
    silent. Same lesson as WAN_DEFAULT_FRAMES — one literal, two importers, no second
    place to disagree — so this test recomputes both numbers from source and demands the
    wire follow, rather than pinning the values.

    NULLS ARE PART OF THE CONTRACT: the two ffmpeg enhance rows have no geometry of their
    own, so there is nothing to price and `vram_need_gib` / `fits_render_box` are null. A
    0 there would be a lie with a shape.
    """
    from hugpy_video.intel.studio.registry import MODEL_REGISTRY
    from hugpy_video.intel.studio.runners.wan_i2v import _placement_need_gib

    body = client.get("/video/render/presets").get_json()
    assert body["render_box_vram_gib"] == _render_presets.RENDER_BOX_VRAM_GIB
    rows = {p["id"]: p for p in body["presets"]}
    priced = 0
    for p in _render_presets.all_presets():
        row = rows[p.preset_id]
        cfg = MODEL_REGISTRY.get(p.model_id)
        assert cfg is not None, p.preset_id  # a preset naming an absent model is a defect
        assert row["vram_envelope_gb"] == cfg.vram.as_map().get(p.precision), p.preset_id
        # The envelope is what a caller's vram_budget_gb is compared against, so it must
        # be a usable positive number on every row — a null here would leave a console
        # with no floor to advise and it would go back to mirroring a constant.
        assert isinstance(row["vram_envelope_gb"], (int, float)), p.preset_id
        assert row["vram_envelope_gb"] > 0, p.preset_id

        if p.width and p.height and p.default_frames:
            need = _placement_need_gib(
                p.model_id, p.precision, p.width, p.height, p.default_frames,
            )
            assert row["vram_need_gib"] == need, p.preset_id
            if need is None:
                assert row["fits_render_box"] is None, p.preset_id
            else:
                priced += 1
                assert row["fits_render_box"] is (
                    need <= _render_presets.RENDER_BOX_VRAM_GIB
                ), p.preset_id
        else:
            # enhance-upres / enhance-interp: geometry comes from the source clip.
            assert row["geometry"] == "source", p.preset_id
            assert row["vram_need_gib"] is None, p.preset_id
            assert row["fits_render_box"] is None, p.preset_id
    # The six Wan rows must all price; a silent None everywhere would make this test
    # vacuous and hand the console back its guesses.
    assert priced == 6, priced
    # And the table's own honesty must survive the round trip: the 14B i2v row is the
    # one that does NOT fit ae, which is exactly why it is proven=False.
    assert rows["clip-i2v-480p"]["fits_render_box"] is False
    assert rows["clip-t2v-480p"]["fits_render_box"] is True


def test_render_presets_frame_budgets_are_wan_cadence():
    """A frame count off Wan's 4k+1 latent cadence would advertise a length the
    runner then silently snaps down from — so the WIRE numbers, not just the table's,
    must satisfy it. The enhance presets carry None (no frames of their own)."""
    for row in client.get("/video/render/presets").get_json()["presets"]:
        if row["default_frames"] is None:
            assert row["max_frames"] is None and row["geometry"] == "source", row["id"]
            continue
        assert _render_presets.is_wan_cadence(row["default_frames"]), row
        assert _render_presets.is_wan_cadence(row["max_frames"]), row
        assert row["default_frames"] <= row["max_frames"], row


def test_render_presets_report_what_is_refused():
    """The refusals are the other half of the answer, and they are the registry's own
    prose (capability_verdict) — the route must not re-word a blocker."""
    body = client.get("/video/render/presets").get_json()
    unavailable = {u["capability"]: u for u in body["unavailable"]}
    expected = {c.value for c in _render_presets.unservable_capabilities()}
    assert set(unavailable) == expected, set(unavailable) ^ expected
    assert unavailable, "a fleet that refuses nothing is not being honest"
    for cap in _render_presets.unservable_capabilities():
        verdict = _render_presets.capability_verdict(cap)
        row = unavailable[cap.value]
        assert row["reason"] == verdict.reason, cap.value
        assert row["refusal"] == verdict.refusal, cap.value
        assert row["refusal"], cap.value
    # a capability is served or refused, never both and never neither
    served = {c for p in _render_presets.all_presets() for c in p.capabilities}
    assert not ({c.value for c in served} & set(unavailable))
    assert body["menu"] == _render_presets.available_menu()
    assert body["frame_cadence"] == _render_presets.WAN_FRAME_CADENCE
    assert body["render_box"] == _render_presets.RENDER_BOX


def test_render_presets_are_derived_at_request_time():
    """PROOF, not assertion, that the route reads the registry per request: swap the
    accessor and the response must follow. A hardcoded or cached copy fails here."""
    real = _render_presets.all_presets
    one = real()[0]
    try:
        _render_presets.all_presets = lambda: (one,)
        rows = client.get("/video/render/presets").get_json()["presets"]
        assert [p["id"] for p in rows] == [one.preset_id], rows
    finally:
        _render_presets.all_presets = real
    # and the swap really was reversible — the full table is back on the wire
    assert len(client.get("/video/render/presets").get_json()["presets"]) == 8


def test_render_presets_name_no_zero_byte_model():
    """No row may bind one of the twelve registry entries holding 0 bytes on the
    shared store — the disqualifying class the table exists to keep out."""
    for row in client.get("/video/render/presets").get_json()["presets"]:
        assert row["model"] not in _render_presets.ZERO_BYTE_MODELS, row["id"]
        for seg in row["composes"]:
            assert _render_presets.preset(seg) is not None, (row["id"], seg)


# --------------------------------------------------------------------------- #
# THE BOUNDARY REFUSAL — POST /video/studio/i2v (2026-07-27)
# --------------------------------------------------------------------------- #
# Everything above reads a table. These drive the REAL enqueue path with the real
# test client, because the defect they pin was precisely a gap between the two:
# ``presets.capability_verdict`` existed, its own docstring said the route layer
# consumed it to "refuse at the BOUNDARY ... instead of enqueuing a job that dies in
# a runner three layers down", and NO route called it. ``studio/job.py::
# _VALID_CAPABILITIES`` was (and still is) every member of the Capability enum.
#
# MEASURED on this exact client before the gate landed — all 200, all with a job_id,
# each one a burned media-bus queue slot:
#     audio 200 · lipsync 200 · restore 200 · stream 200 · keyframe 200
#     inpaint 200 · outpaint 200 · retake 200
# The four VACE-shaped ones were the worse half: they did not fail, they SUCCEEDED at
# rendering a plain full restyle under the requested capability's name.
#
# No bus/catalog/worker is touched by a 400 (the refusal returns before enqueue), and
# the 200 half of the check enqueues into the temp PROJECTS_HOME this module already
# points at, exactly like the sibling suites.
from hugpy_video.intel.studio.enums import Capability as _Capability


def _post_capability(capability, **extra):
    body = {"capability": capability, "prompt": "boundary gate probe"}
    body.update(extra)
    return client.post("/video/studio/i2v", json=body)


def test_dead_capabilities_are_refused_before_a_job_exists():
    """Every capability NO preset covers must 400 with the registry's own refusal —
    and must not hand back a job_id, which is the whole point: a caller who cannot be
    served should not be given something to poll."""
    dead = sorted(_render_presets.unservable_capabilities(), key=lambda c: c.value)
    assert dead, "a fleet that refuses nothing is not being honest"
    for cap in dead:
        r = _post_capability(cap.value)
        assert r.status_code == 400, (cap.value, r.status_code, r.get_json())
        body = r.get_json()
        assert "job_id" not in body, (
            f"{cap.value}: refused but still minted a job — the queue slot is the "
            f"cost this gate exists to avoid: {body}")
        # The registry's wording, not a paraphrase: one explanation for console + log.
        verdict = _render_presets.capability_verdict(cap)
        assert body["error"] == verdict.refusal, (cap.value, body["error"])
        assert body["reason"] == verdict.reason, cap.value
        # NAMES AN ALTERNATIVE. A refusal that only says "no" teaches nothing; every
        # one of ours has to point at something the fleet CAN render.
        live_ids = {p.preset_id for p in _render_presets.all_presets()}
        named = {pid for pid in live_ids if pid in body["error"]}
        assert named, (
            f"{cap.value}: the refusal names no preset the caller could have "
            f"instead: {body['error']}")
        assert body["available"] == _render_presets.available_menu(), cap.value


def test_the_five_measured_200s_are_now_400s():
    """The regression, named one capability at a time rather than derived, so that a
    future edit which quietly re-admits one of them fails HERE with its name in the
    output. These five were measured returning 200 + job_id on 2026-07-27."""
    for cap_value in ("audio", "lipsync", "restore", "stream", "keyframe"):
        r = _post_capability(cap_value)
        assert r.status_code == 400, (cap_value, r.status_code, r.get_json())
        assert cap_value in r.get_json()["error"], cap_value


def test_the_three_phantom_vace_capabilities_are_now_400s():
    """The other half, and the more dangerous half: these three did not fail after
    being admitted — they rendered a full restyle and called it an inpaint / outpaint
    / retake. The refusal has to name the restyle AND name v2v, so a caller learns
    what they were actually getting."""
    for cap_value in ("inpaint", "outpaint", "retake"):
        r = _post_capability(cap_value)
        assert r.status_code == 400, (cap_value, r.status_code, r.get_json())
        error = r.get_json()["error"]
        assert "restyle" in error.lower(), (cap_value, error)
        assert "v2v" in error, (cap_value, error)


def test_every_working_capability_still_enqueues():
    """THE OTHER DIRECTION, and the one that makes the gate safe rather than merely
    strict: nothing that worked was narrowed. Each servable capability is posted with
    the MINIMUM body its own route rules require and must still come back 200 with a
    job_id, on exactly the path it used before the gate existed.

    id_lock and motion are excluded here and checked separately below — both have a
    REQUIRED input (references / a control still) that a minimal body cannot satisfy,
    so a 200 for them would need a real image inside the storage jail. Their gate
    behaviour is asserted instead: they must fail on their OWN rule, never on the
    capability gate."""
    for cap in sorted(_render_presets.servable_capabilities(), key=lambda c: c.value):
        if cap.value in ("id_lock", "motion"):
            continue
        r = _post_capability(cap.value)
        assert r.status_code == 200, (cap.value, r.status_code, r.get_json())
        assert isinstance(r.get_json().get("job_id"), str), (cap.value, r.get_json())


def test_capabilities_with_required_inputs_fail_on_their_own_rule():
    """id_lock and motion must NOT be caught by the capability gate — they are
    servable, and their 400 has to be the specific, actionable one about the input
    they are missing. This is the check that would catch a future over-broad gate
    that swallowed a working capability into "this fleet cannot render it"."""
    r = _post_capability("id_lock")
    assert r.status_code == 400, r.get_json()
    assert "reference_image" in r.get_json()["error"], r.get_json()

    r = _post_capability("motion")
    assert r.status_code == 400, r.get_json()
    error = r.get_json()["error"]
    assert "control_image" in error, error
    # ...and it must point at what to use instead for the two things a caller might
    # have MEANT by "motion" without a control.
    assert "t2v" in error and "v2v" in error, error


def test_motion_accepts_a_control_image_that_id_lock_only_used_to():
    """THE FIX THAT MADE `motion` REAL. Until 2026-07-27 the route rejected
    control_image/control_kind for every capability except id_lock, so the one input
    that distinguishes a motion render from a restyle was unreachable — which is what
    made the old clip-control-480p preset a phantom. Widening that gate is what let
    motion survive the split as its own preset.

    Uses a real PNG inside the storage jail (the same idiom as
    tests/studio/test_studio_id_lock.py) because the route jail-resolves and
    ffprobe/PIL-classifies the file — a fake path would be rejected for the wrong
    reason and prove nothing."""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 — imaging stack is optional on some boxes
        import pytest
        pytest.skip("PIL unavailable on this box — cannot build a jailed control "
                    "image. SKIP, not a pass.")
    from hugpy_platform.constants import DEFAULT_ROOT

    work = tempfile.mkdtemp(prefix="studio-motion-gate-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    try:
        pose = os.path.join(work, "pose.png")
        Image.new("RGB", (96, 96), (30, 30, 30)).save(pose, "PNG")

        r = client.post("/video/studio/i2v", json={
            "capability": "motion", "prompt": "hold this pose",
            "resolution": {"width": 832, "height": 480, "fps": 16},
            "vram_budget_gb": 9.0,
            "control_image": pose, "control_kind": "pose"})
        assert r.status_code == 200, (r.status_code, r.get_json())
        assert isinstance(r.get_json().get("job_id"), str), r.get_json()

        # ...and the combination the runner would resolve AGAINST the control still
        # is refused by name rather than silently rendering a restyle: wan_vace picks
        # exactly one channel and source_video wins.
        r2 = client.post("/video/studio/i2v", json={
            "capability": "motion", "prompt": "hold this pose",
            "control_image": pose, "control_kind": "pose",
            "source_video": pose})
        assert r2.status_code == 400, (r2.status_code, r2.get_json())
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_an_unknown_capability_is_refused_with_the_menu():
    """A typo gets the same SHAPE of answer as a real-but-dead capability: a 400 that
    names what this fleet renders. Previously this fell through to make_studio_i2v's
    ValueError, which listed all 16 enum values — including the eight that cannot be
    served, i.e. it advertised the phantoms as the fix for the typo."""
    r = _post_capability("t2vv")
    assert r.status_code == 400, r.get_json()
    body = r.get_json()
    assert "t2vv" in body["error"]
    assert body["available"] == _render_presets.available_menu()
    assert "clip-t2v-480p" in body["error"]


if __name__ == "__main__":  # allow the script-style run the sibling tests use
    test_get_presets_contract_shape()
    test_get_presets_modes()
    test_apply_unknown_preset_404()
    test_movie_presets_list_includes_street_walk()
    test_movie_preset_street_walk_apply_carries_strength_negative()
    test_movie_preset_storm_front_regression()
    test_render_presets_returns_all_eight()
    test_render_presets_payload_matches_the_registry()
    test_render_presets_frame_budgets_are_wan_cadence()
    test_render_presets_report_what_is_refused()
    test_render_presets_are_derived_at_request_time()
    test_render_presets_name_no_zero_byte_model()
    test_dead_capabilities_are_refused_before_a_job_exists()
    test_the_five_measured_200s_are_now_400s()
    test_the_three_phantom_vace_capabilities_are_now_400s()
    test_every_working_capability_still_enqueues()
    test_capabilities_with_required_inputs_fail_on_their_own_rule()
    test_motion_accepts_a_control_image_that_id_lock_only_used_to()
    test_an_unknown_capability_is_refused_with_the_menu()
    print("all video-preset route checks passed")
