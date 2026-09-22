"""ID-locked STILL image generation via ComfyUI IP-Adapter (2026-07-12).

The STILL sibling of the studio VIDEO arm's id_lock (Wan-VACE reference-to-video):
give reference image(s) of a subject and generate NEW stills that HOLD the
identity, through the comfy delegation path with an IP-Adapter graph. Zero-train
reference-embedding guidance (video_intel.studio.enums.AttentionMethod.IP_ADAPTER).

Five halves, each an invariant of the slice:

  * REQUEST — ImageGenRequest gains reference_images (jailed abs paths) + id_strength
    (0..1) + the reference_images_b64 offload transport; the builder jail-resolves +
    image-classifies + count-checks the paths; ABSENT everything => byte-for-byte the
    legacy request.
  * BUILDER — a request with references composes the IPAdapter chain (loaders ->
    LoadImage(s) -> ImageBatch -> IPAdapterAdvanced) wired onto the EXISTING KSampler,
    with sampler/scheduler/cfg/seed all carried through and the id_strength on the
    apply node's weight. Family match (SD1.5 vs SDXL) picks the adapter weight.
  * TRANSPORT — a comfy worker (127.0.0.1) can't see central's paths, so
    remote._inline_reference_images base64s them into reference_images_b64 and DROPS
    the paths; the worker rebuilds the request + the comfy runner uploads the bytes.
  * DETECTION — the IPAdapter node pack is PROBED (object_info), never assumed: present
    -> id_lock True; absent -> False AND an id_lock request fails as data with the
    install pointer (NEVER a silent non-locked image).
  * ROUTING — an id_lock request only lands on a box whose comfy advertises id_lock
    (STRICT), skipping comfy-less / nodeless workers; a plain request is untouched.

The imagegen builder now lives in ``hugpy_media.plugin_builders`` (public name
``build_imagegen_request``); the storage jail is redirected to a tmp dir so no
test writes under the operator's real UPLOADS_HOME.
"""
import asyncio
import importlib
import os
import types

import pytest
from PIL import Image

from worker_store_isolation import swap_worker_store

builders = importlib.import_module("hugpy_media.plugin_builders")
comfy_runner = importlib.import_module("hugpy_media.comfy.comfy_runner")
remote = importlib.import_module("hugpy_engine.resolvers.remote")
schemas = importlib.import_module("hugpy_media.imagegen.schemas")

ImageGenRequest = schemas.ImageGenRequest
build_request = builders.build_imagegen_request


class _Cfg:
    model_key = "m"
    filename = "sd15.safetensors"


@pytest.fixture
def jail(tmp_path, monkeypatch):
    """A pair of real PNGs (and one non-image) UNDER a tmp storage jail; the
    builder's jail roots are pointed at it so /etc/passwd is truly outside."""
    root = tmp_path / "uploads"
    root.mkdir()
    monkeypatch.setattr(builders, "UPLOADS_HOME", str(root))
    monkeypatch.setattr(builders, "DEFAULT_ROOT", str(root))
    ref1 = str(root / "_idlock_ut_ref1.png")
    ref2 = str(root / "_idlock_ut_ref2.png")
    Image.new("RGB", (8, 8), (200, 10, 10)).save(ref1)
    Image.new("RGB", (8, 8), (10, 10, 200)).save(ref2)
    txt = str(root / "_idlock_ut_notimg.txt")
    with open(txt, "w") as fh:
        fh.write("not an image")
    return types.SimpleNamespace(root=str(root), ref1=ref1, ref2=ref2, txt=txt)


# ── REQUEST validation (jail / bounds / legacy) ──────────────────────────────

def test_request_jailed_paths_accepted_in_order(jail):
    req = build_request(
        {"prompt": "a hero", "reference_images": [jail.ref1, jail.ref2],
         "id_strength": 0.42}, "comfy-model")
    assert req.reference_images == [jail.ref1, jail.ref2]
    assert req.id_strength == 0.42


def test_request_jail_escape_rejected_as_data(jail):
    with pytest.raises(ValueError, match="escapes the storage jail"):
        build_request({"prompt": "x", "reference_images": ["/etc/passwd"]}, "m")


def test_request_non_image_reference_rejected(jail):
    with pytest.raises(ValueError, match="is not an image"):
        build_request({"prompt": "x", "reference_images": [jail.txt]}, "m")


def test_request_over_count_rejected(jail):
    with pytest.raises(ValueError, match="at most"):
        build_request({"prompt": "x", "reference_images": [jail.ref1] * 5}, "m")


def test_request_id_strength_out_of_bounds_rejected():
    with pytest.raises(Exception) as ei:
        ImageGenRequest(request_id="r", model_key="m", prompt="p", id_strength=1.5)
    assert "ValidationError" in type(ei.value).__name__


def test_request_absent_references_is_legacy(jail):
    legacy = build_request({"prompt": "plain"}, "m")
    assert legacy.reference_images is None
    assert legacy.id_strength is None
    assert legacy.reference_images_b64 is None


# ── BUILDER graph structure (IPAdapter chain wired to the sampler) ──────────

def _graph_req():
    return ImageGenRequest(
        request_id="g", model_key="m", prompt="a knight", negative_prompt="blurry",
        sampler_name="dpmpp_2m", scheduler="karras", guidance_scale=6.5, seed=1234,
        id_strength=0.55)


def test_builder_two_reference_graph():
    greq = _graph_req()
    base = comfy_runner._t2i_workflow("sd15-model.safetensors", greq, seed=1234)
    wf = comfy_runner._ipadapter_workflow("sd15-model.safetensors", greq, base,
                                          ["ref0.png", "ref1.png"])
    # loader + clip-vision + 2 LoadImage + batch + apply nodes present
    assert {"20", "21", "30", "31", "40", "50"} <= set(wf)
    assert wf["50"]["class_type"] == comfy_runner.IPADAPTER_APPLY_NODE
    assert wf["20"]["class_type"] == comfy_runner.IPADAPTER_LOADER_NODE
    # KSampler.model rewired onto the IPAdapter-patched model
    assert wf["5"]["inputs"]["model"] == ["50", 0]
    # apply.model reads the checkpoint MODEL (node 1)
    assert wf["50"]["inputs"]["model"] == ["1", 0]
    # apply.image reads the batched references (node 40)
    assert wf["50"]["inputs"]["image"] == ["40", 0]
    assert wf["50"]["inputs"]["ipadapter"] == ["20", 0]
    assert wf["50"]["inputs"]["clip_vision"] == ["21", 0]
    # ImageBatch folds the two references
    assert wf["40"]["inputs"]["image1"] == ["30", 0]
    assert wf["40"]["inputs"]["image2"] == ["31", 0]
    # id_strength -> apply weight
    assert wf["50"]["inputs"]["weight"] == 0.55
    # sampler/scheduler/cfg/seed carried through UNTOUCHED
    assert wf["5"]["inputs"]["sampler_name"] == "dpmpp_2m"
    assert wf["5"]["inputs"]["scheduler"] == "karras"
    assert wf["5"]["inputs"]["cfg"] == 6.5
    assert wf["5"]["inputs"]["seed"] == 1234


def test_builder_single_reference_has_no_batch_node():
    greq = _graph_req()
    wf1 = comfy_runner._ipadapter_workflow(
        "sd15-model.safetensors", greq,
        comfy_runner._t2i_workflow("sd15-model.safetensors", greq, 1), ["only.png"])
    assert "40" not in wf1
    assert wf1["50"]["inputs"]["image"] == ["30", 0]


@pytest.mark.parametrize("ckpt, adapter", [
    ("juggernautXL.safetensors", "ip-adapter_sdxl_vit-h.safetensors"),
    ("sd15-pruned.safetensors", "ip-adapter_sd15.safetensors"),
])
def test_builder_family_match_picks_adapter_weight(ckpt, adapter):
    greq = _graph_req()
    wf = comfy_runner._ipadapter_workflow(
        ckpt, greq, comfy_runner._t2i_workflow(ckpt, greq, 1), ["r.png"])
    assert wf["20"]["inputs"]["ipadapter_file"] == adapter


# ── TRANSPORT (central inline -> worker rebuild -> reference bytes) ─────────

def test_transport_inline_rebuild_and_reference_bytes(jail):
    payload = build_request(
        {"prompt": "hero", "reference_images": [jail.ref1, jail.ref2],
         "id_strength": 0.5}, "m").model_dump()
    inlined = remote._inline_reference_images(payload)
    # inline succeeds and drops the unreachable paths
    assert inlined
    assert payload.get("reference_images") is None
    assert len(payload.get("reference_images_b64") or []) == 2
    # worker rebuilds via the same builder (paths gone; b64 present)
    wreq = build_request(payload, "m")
    assert wreq.reference_images is None
    assert len(wreq.reference_images_b64) == 2
    # reference bytes round-trip to the original files
    pays = comfy_runner.ComfyRunner(_Cfg())._reference_payloads(wreq)
    with open(jail.ref1, "rb") as f1, open(jail.ref2, "rb") as f2:
        assert [d for _, d in pays] == [f1.read(), f2.read()]


# ── DETECTION (object_info probe -> capability + errors-as-data) ────────────

class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


class _FakeClient:
    """A comfy whose object_info advertises exactly ``present`` node classes."""

    def __init__(self, present):
        self.present = set(present)

    def get(self, url):
        cls = url.rsplit("/", 1)[-1]
        return _Resp(200, {cls: {"input": {}}} if cls in self.present else {})


def test_detection_probe_with_both_nodes_is_capable():
    both = comfy_runner.IPADAPTER_REQUIRED_NODES
    assert comfy_runner._probe_ipadapter("http://x", client=_FakeClient(both)) is True


def test_detection_probe_missing_apply_node_not_capable():
    assert comfy_runner._probe_ipadapter(
        "http://x", client=_FakeClient([comfy_runner.IPADAPTER_LOADER_NODE])) is False


def test_detection_probe_no_nodes_not_capable():
    assert comfy_runner._probe_ipadapter("http://x", client=_FakeClient([])) is False


def test_detection_absent_nodes_id_lock_request_fails_as_data(monkeypatch):
    """An id_lock request on a comfy WITHOUT the IPAdapter pack fails as data with
    the install pointer — never a silent non-locked image."""
    monkeypatch.setattr(comfy_runner, "comfy_has_ipadapter",
                        lambda url, client=None: False)
    idreq = ImageGenRequest(request_id="e", model_key="m", prompt="hero",
                            reference_images_b64=["aGVsbG8="], id_strength=0.6)
    result = asyncio.run(comfy_runner.ComfyRunner(_Cfg()).run(idreq))
    assert result.ok is False
    assert "IPAdapter" in (result.error or "")
    assert "WORKER-SETUP" in (result.error or "")


# ── ROUTING (id_lock lands only on a comfy-with-nodes worker) ───────────────

MODEL = "org/Comfy-Ckpt"


@pytest.fixture
def routing_store():
    """Three approved workers advertising MODEL: comfy+nodes, comfy-without-nodes,
    and no comfy at all (isolated, tmpdir-backed registry)."""
    with swap_worker_store(prefix="hugpy-idlock-") as store:
        def _add(wid, comfy):
            store.register(name=wid, url=f"http://{wid}:9100", worker_id=wid,
                           models=[MODEL])
            store.set_admission(wid, "approved")
            if comfy is not None:
                store.heartbeat(wid, comfy=comfy)
        _add("has_nodes", {"available": True, "id_lock": True})
        _add("no_nodes", {"available": True, "id_lock": False})
        _add("no_comfy", None)
        yield store


def test_routing_id_lock_keeps_only_capable_worker(routing_store):
    ids_locked = {w["id"] for w in routing_store.workers_for_model(
        MODEL, online_only=False, require_comfy_id_lock=True)}
    assert ids_locked == {"has_nodes"}
    assert (routing_store.pick_for_model(MODEL, require_comfy_id_lock=True) or {}
            ).get("id") == "has_nodes"


def test_routing_plain_request_is_untouched(routing_store):
    ids_plain = {w["id"] for w in routing_store.workers_for_model(
        MODEL, online_only=False)}
    assert ids_plain == {"has_nodes", "no_nodes", "no_comfy"}


def test_routing_remote_filter_agrees_with_registry_gate():
    assert remote._worker_comfy_id_lock_capable(
        {"comfy": {"available": True, "id_lock": True}}) is True
    assert remote._worker_comfy_id_lock_capable(
        {"comfy": {"available": True, "id_lock": False}}) is False
    assert remote._worker_comfy_id_lock_capable({}) is False
