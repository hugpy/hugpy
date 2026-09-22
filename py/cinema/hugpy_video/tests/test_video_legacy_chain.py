"""Legacy pixel chain gate (board or-k2 / proposal or-p1).

Chaining (frame i+1 conditioning on frame i; segment N+1 starting from segment
N's last frame) is the LEGACY behaviour the oracle directive prohibits. It must
be OFF by default and only run when the operator opts in with
HUGPY_LEGACY_CHAIN=1 — read through ONE shared helper
(movie_schema.legacy_chain_enabled) so movie.py and scene.py can never disagree.
When it actually runs, every manifest/progress blob says
'legacy (pixel-chained)'. Hermetic: no GPU, no bus, stubbed render core.
"""
import json
import os
import tempfile

import pytest

LABEL = "legacy (pixel-chained)"


def _scene_spec(chain, with_start=False):
    from hugpy_video.intel.scene_schema import make_generate_scene
    from hugpy_video.intel.gen_schema import text_part, image_part
    from hugpy_video.intel.media_schema import make_media_ref
    parts = [text_part("a serene landscape, slowly panning")]
    if with_start:
        parts.append(image_part(make_media_ref("start", "image", "/x/start.png", "image/png", 64, 64)))
    return make_generate_scene(
        parts=tuple(parts), model_id="sd-turbo", width=64, height=64, steps=2,
        guidance=0.0, n_frames=2, fps=8, assemble=False, seed=7, chain=chain,
        project="Chain Gate",
    )


def _stub_frames(out_dir, n):
    paths = []
    for i in range(n):
        fp = os.path.join(out_dir, f"frame_{i:05d}.png")
        with open(fp, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + bytes([i]) * 16)
        paths.append(fp)
    return paths


# --------------------------------------------------------------------------- #
# 1) the shared helper: unset/0/false/off -> False; 1/true/yes/on -> True
# --------------------------------------------------------------------------- #
def test_helper_default_off(monkeypatch):
    from hugpy_video.intel.movie_schema import LEGACY_CHAIN_LABEL, effective_chain, legacy_chain_enabled
    assert LEGACY_CHAIN_LABEL == LABEL
    monkeypatch.delenv("HUGPY_LEGACY_CHAIN", raising=False)
    assert legacy_chain_enabled() is False
    assert effective_chain(True) is False and effective_chain(False) is False
    for off in ("0", "false", "no", "off", "", "garbage"):
        monkeypatch.setenv("HUGPY_LEGACY_CHAIN", off)
        assert legacy_chain_enabled() is False, off
    for on in ("1", "true", "YES", " on "):
        monkeypatch.setenv("HUGPY_LEGACY_CHAIN", on)
        assert legacy_chain_enabled() is True, on
        assert effective_chain(True) is True and effective_chain(False) is False


# --------------------------------------------------------------------------- #
# 2) schema + presets default to chain=False
# --------------------------------------------------------------------------- #
def test_schema_and_presets_default_chain_off():
    from hugpy_video.intel.movie_schema import GoalInterval, MovieSpec, make_movie
    from hugpy_video.intel import presets
    assert MovieSpec.__dataclass_fields__["chain"].default is False
    m = make_movie(goals=(GoalInterval(0, 2, "a"),), model_id="sd-turbo", width=64,
                   height=64, steps=1, guidance=0.0, fps=8, assemble=False)
    assert m.chain is False
    movie_presets = presets.available_movie_presets()
    assert movie_presets, "expected registered movie presets"
    bad = [p.id for p in movie_presets if p.chain]
    assert not bad, f"movie presets must ship chain=False (legacy chain is opt-in): {bad}"


# --------------------------------------------------------------------------- #
# 3) run_generate_scene: chain=True requested, env unset -> render core gets
#    chain=False and the live progress blob is labelled 'independent'
# --------------------------------------------------------------------------- #
def test_scene_runner_forces_chain_off(monkeypatch):
    from hugpy_video.intel.runners import scene
    from hugpy_video.intel import media_bus
    monkeypatch.delenv("HUGPY_LEGACY_CHAIN", raising=False)
    tmp = tempfile.mkdtemp(prefix="hugpy_test_chain_gate_")
    monkeypatch.setattr(scene, "DEFAULT_ROOT", tmp)
    calls, blobs = [], []

    def _fake_render(**kw):
        calls.append(kw)
        for i, fp in enumerate(_stub_frames(kw["out_dir"], kw["n_frames"])):
            kw["on_frame_done"](fp, i)
        return None

    monkeypatch.setattr(scene, "render_scene_frames", _fake_render)
    from hugpy_video.intel.media_schema import make_media_ref
    monkeypatch.setattr(scene, "ingest", lambda path, *a, **k: make_media_ref(
        os.path.basename(path), "image", path, "image/png", 64, 64))
    monkeypatch.setattr(media_bus, "set_progress", lambda jid, blob: blobs.append(dict(blob)))
    monkeypatch.setattr(media_bus, "is_cancelling", lambda jid: False)

    res = scene.run_generate_scene(_scene_spec(chain=True, with_start=True), "job-gate-1")
    assert calls, "render core must have been called"
    assert calls[0]["chain"] is False, "chain must be forced OFF without HUGPY_LEGACY_CHAIN=1"
    assert blobs and all(b["legacy_pixel_chain"] is False and b["mode"] == "independent"
                         for b in blobs), blobs[:2]
    # the sidecar spec on disk records the EFFECTIVE chain (False), not the request
    side = os.path.join(tmp, "video_intel", "scenes", "job-gate-1")
    sidecars = [f for f in os.listdir(side) if f.endswith(".json")]
    assert sidecars, os.listdir(side)
    with open(os.path.join(side, sidecars[0])) as fh:
        assert json.dumps(json.load(fh)).count('"chain": false') >= 1

    # opted in: the request is honoured and the progress blob says so
    monkeypatch.setenv("HUGPY_LEGACY_CHAIN", "1")
    calls.clear(); blobs.clear()
    scene.run_generate_scene(_scene_spec(chain=True, with_start=True), "job-gate-2")
    assert calls and calls[0]["chain"] is True
    assert blobs and all(b["legacy_pixel_chain"] is True and b["mode"] == LABEL for b in blobs)
    # opted in but NO start frame -> v1 path, nothing is chained, no legacy label
    calls.clear(); blobs.clear()
    scene.run_generate_scene(_scene_spec(chain=True, with_start=False), "job-gate-3")
    assert blobs and all(b["legacy_pixel_chain"] is False and b["mode"] == "independent" for b in blobs)


# --------------------------------------------------------------------------- #
# 4) scene bundle manifest: legacy_pixel_chain true ONLY when actually chained,
#    with the label + limitations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env,chain,with_start,expect", [
    (None, True, True, False),     # default: declared request, but NOT chained
    ("0", True, True, False),
    ("1", True, True, True),       # opted in + start frame -> chained
    ("1", True, False, False),     # opted in, no start frame -> v1, not chained
    ("1", False, True, False),     # opted in, chain=false -> not chained
])
def test_scene_manifest_declares_actual_chain(monkeypatch, env, chain, with_start, expect):
    from hugpy_video.intel.runners import scene
    if env is None:
        monkeypatch.delenv("HUGPY_LEGACY_CHAIN", raising=False)
    else:
        monkeypatch.setenv("HUGPY_LEGACY_CHAIN", env)
    tmp = tempfile.mkdtemp(prefix="hugpy_test_chain_manifest_")
    monkeypatch.setattr(scene, "DEFAULT_ROOT", tmp)
    out = os.path.join(tmp, "out"); os.makedirs(out)
    frames = _stub_frames(out, 2)
    bundle = scene._write_bundle(
        spec=_scene_spec(chain=chain, with_start=with_start), job_id="j", projectmeta="pm",
        frame_paths=frames, mp4_path=None, base_prompt="p", started_at=1.0,
        finished_at=2.0, per_frame_secs=[0.5, 0.5])
    with open(os.path.join(bundle, "project.json")) as fh:
        m = json.load(fh)
    assert m["legacy_pixel_chain"] is expect, m
    assert m["chain"] is expect, "manifest chain reports the EFFECTIVE chain"
    assert m["label"] == (LABEL if expect else "independent"), m["label"]
    assert bool(m["limitations"]) is expect, m["limitations"]
