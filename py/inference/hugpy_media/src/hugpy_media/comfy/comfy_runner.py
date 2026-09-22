"""ComfyRunner — drive a LOCAL ComfyUI instance as a hugpy engine (slice B).

The operator installs ComfyUI on the worker box (slice A adopts + advertises
it); this runner turns it into a registry engine: `("comfy", task)` rows route
here, a VANILLA-nodes workflow template is built per request (no custom nodes
in the base templates), submitted to ComfyUI's HTTP API, and the artifacts are
returned as ImageGenResult — so the remote/delegation factories, the b64
artifact seam, the video arm, and the scene fan-out all work unchanged.

ID-LOCK (identity-locked STILLs) is the ONE variant that needs CUSTOM nodes: a
request carrying reference_images composes an IP-Adapter graph on top of the
base template (reference still(s) -> CLIP-Vision -> IPAdapter apply -> the same
sampler chain). Those nodes are NOT assumed — they're PROBED via /object_info at
request time and the request fails as data when the pack is absent. See the
id_lock section below and WORKER-SETUP §5b.

Request/result REUSE ImageGenRequest/ImageGenResult (the Img2ImgRunner
precedent): the checkpoint is the registry row's ``filename`` (a file inside
ComfyUI's own models/checkpoints — hugpy holds no files for comfy models, by
design).

API surface used (all vanilla ComfyUI):
    POST /prompt {"prompt": workflow, "client_id"}   -> {"prompt_id"}
    GET  /history/<prompt_id>                        -> outputs when done
    GET  /view?filename&subfolder&type=output        -> image bytes
    POST /upload/image (multipart)                   -> init image for img2img
    POST /interrupt (+ /queue delete)                -> cancel before the ONE
                                                        VRAM-class retry (k71)
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from typing import Any, Dict

from hugpy_media.imagegen.schemas import GeneratedImage, ImageGenRequest, ImageGenResult
from hugpy_media.imagegen.vram_retry import is_retryable_vram_failure, settle_delay_s
from hugpy_platform.constants import UPLOADS_HOME

from hugpy_media import hooks

logger = logging.getLogger(__name__)

_POLL_S = 1.0
_TIMEOUT_S = float(os.environ.get("COMFY_TIMEOUT_S", "600"))


# ── call-time attribution (2026-07-14) ──────────────────────────────────────
# ComfyUI is an EXTERNAL GPU process the worker drives; in nvidia-smi it is
# anonymous (not a slot child, holds no torch weights we can split), so its VRAM
# fell into pid_registry.unattributed even though THIS call knows the checkpoint.
# Record the (model_key, job_id) at dispatch so the worker heartbeat's reconcile
# can stamp the comfy PID with the model it was called for. Both helpers are
# STRICTLY best-effort and go through ``hugpy_media.hooks`` (no-op unless the
# fleet worker installed its PID-registry callbacks), so they can NEVER perturb
# a generation and this runner never imports the worker agent stack.
def _reg_comfy_call(model_key, job_id) -> None:
    hooks.on_foreign_call_started("comfy", model_key, job_id)


def _end_comfy_call(job_id) -> None:
    hooks.on_foreign_call_ended("comfy", job_id)


# ── ensure-comfy-headroom hook (Fix B, 2026-07-15) ──────────────────────────
# When hugpy drives a ComfyUI gen, ComfyUI (a SEPARATE process) silently grabs
# VRAM or waits — it is never a first-class queue entry, so a card already full
# of managed on-demand models leaves ComfyUI to under-allocate or stall. This
# hook lets the WORKER agent register a routine that evicts on-demand managed
# models (LRU, via the same eviction mechanism Fix A uses) until real free VRAM
# reaches a target, BEFORE every comfy gen commits. The setter/None-default
# pattern mirrors slots.set_eviction_policy / dispatch.set_fit_check: comfy_runner
# is package-shared (central imports it too), so it must NOT import worker/GPU
# internals directly — the worker registers the hook at boot and comfy_runner
# calls it if present, else no-op. Central's import stays clean (no GPU deps),
# and a no-GPU / unregistered box is byte-identical to today (the call is a
# no-op). Strictly best-effort: any failure is swallowed so it can NEVER break a
# generation (honest-degrade — never block/hang the comfy gen).
_COMFY_HEADROOM_HOOK = None


def set_comfy_headroom_hook(fn) -> None:
    """Register the worker-side 'ensure comfy headroom' routine:
    ``fn(model_key, job_id) -> None`` (evicts on-demand managed models until real
    free VRAM >= target). None (bare central / no-GPU) => the pre-gen headroom
    step is a no-op, byte-identical to before."""
    global _COMFY_HEADROOM_HOOK
    _COMFY_HEADROOM_HOOK = fn


def _ensure_comfy_headroom(model_key, job_id) -> None:
    """Call the registered ensure-comfy-headroom hook (Fix B) if present, else a
    no-op. Best-effort — never raises into the generation path."""
    hook = _COMFY_HEADROOM_HOOK
    if hook is None:
        return
    try:
        hook(model_key, job_id)
    except Exception:  # noqa: BLE001 — headroom prep never breaks generation
        logger.warning("ensure-comfy-headroom hook failed (proceeding with the "
                       "gen anyway)", exc_info=True)


def _comfy_url() -> str:
    return (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")


def _t2i_workflow(ckpt: str, req: ImageGenRequest, seed: int) -> Dict[str, Any]:
    """Vanilla text-to-image graph: checkpoint -> CLIP pos/neg -> KSampler ->
    VAEDecode -> SaveImage."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": req.prompt or ""}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1],
                         "text": req.negative_prompt or ""}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": req.width or 512, "height": req.height or 512,
                         "batch_size": req.num_images or 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0],
                         "negative": ["3", 0], "latent_image": ["4", 0],
                         "seed": seed,
                         "steps": req.num_inference_steps or 20,
                         "cfg": req.guidance_scale if req.guidance_scale is not None else 7.0,
                         "sampler_name": req.sampler_name or "euler",
                         "scheduler": req.scheduler or "normal",
                         "denoise": 1.0}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": "hugpy"}},
    }


def _i2i_workflow(ckpt: str, req: ImageGenRequest, seed: int,
                  uploaded_name: str) -> Dict[str, Any]:
    """Vanilla img2img graph: LoadImage -> VAEEncode -> KSampler(denoise=
    strength) — same tail as t2i."""
    wf = _t2i_workflow(ckpt, req, seed)
    strength = req.strength if req.strength is not None else 0.6
    wf["8"] = {"class_type": "LoadImage", "inputs": {"image": uploaded_name}}
    wf["9"] = {"class_type": "VAEEncode",
               "inputs": {"pixels": ["8", 0], "vae": ["1", 2]}}
    wf["5"]["inputs"]["latent_image"] = ["9", 0]
    wf["5"]["inputs"]["denoise"] = max(0.05, min(1.0, float(strength)))
    del wf["4"]  # EmptyLatentImage replaced by the encoded init image
    return wf


# ── ID-LOCK (identity-locked STILLs) via ComfyUI IP-Adapter ─────────────────
# The STILL sibling of the studio VIDEO arm's id_lock (Wan-VACE reference-to-
# video): reference still(s) -> CLIP-Vision embedding -> IPAdapter apply -> the
# SAME checkpoint/sampler chain, so a NEW pose/scene HOLDS the subject identity
# with ZERO training (video_intel.studio.enums.AttentionMethod.IP_ADAPTER —
# "ID-2 (b): zero-train reference embedding"; we reuse that vocabulary, not a
# parallel one). The physics: this is EMBEDDING guidance, not a weight edit —
# quality depends on the checkpoint <-> adapter-weight family match (SD1.5 vs
# SDXL vs flux want DIFFERENT ip-adapter weights + CLIP-Vision), encoded in
# _checkpoint_family below.
#
# The pieces live in ComfyUI, NOT hugpy: the ComfyUI_IPAdapter_plus custom-node
# pack (the loader + apply classes), the ip-adapter weight file
# (models/ipadapter/…), and a CLIP-Vision model (models/clip_vision/…). NONE are
# assumed — we PROBE the target comfy's /object_info for the node classes at
# request time and FAIL AS DATA when the pack is absent. We NEVER silently drop
# the reference and generate a non-locked image. Install recipe: WORKER-SETUP §5b.
IPADAPTER_LOADER_NODE = "IPAdapterModelLoader"
IPADAPTER_APPLY_NODE = "IPAdapterAdvanced"
IPADAPTER_REQUIRED_NODES = (IPADAPTER_LOADER_NODE, IPADAPTER_APPLY_NODE)

_OBJECT_INFO_TTL_S = 60.0
_IPADAPTER_PROBE_CACHE: Dict[str, Any] = {}   # url -> {"at": ts, "present": bool}


def _probe_ipadapter(url: str, client=None) -> bool:
    """Whether ``url``'s ComfyUI has the IPAdapter node pack — probes
    ``/object_info/<Class>`` for EACH required class (cheap: one small JSON per
    class, exactly how _comfy_status probes CheckpointLoaderSimple). Any miss (a
    non-200 or the class absent from the response) => the pack isn't installed.
    A network/parse error => False: we can't PROVE it's there, and id_lock must
    never route on a maybe."""
    import httpx
    own = client is None
    if own:
        client = httpx.Client(timeout=3.0)
    try:
        for cls in IPADAPTER_REQUIRED_NODES:
            r = client.get(url.rstrip("/") + f"/object_info/{cls}")
            if r.status_code != 200 or cls not in (r.json() or {}):
                return False
        return True
    except Exception:  # noqa: BLE001 — unreachable/parse error: treat as absent
        return False
    finally:
        if own:
            client.close()


def comfy_has_ipadapter(url: str, client=None) -> bool:
    """TTL-cached (~60s) IPAdapter-presence probe. ONE source of truth for "can
    this comfy do id_lock", shared by the request-time detection here AND the
    worker heartbeat's ``comfy.id_lock`` advertisement (worker_agent.agent imports
    this, so the node-class contract never forks)."""
    now = time.time()
    hit = _IPADAPTER_PROBE_CACHE.get(url)
    if hit and now - hit["at"] < _OBJECT_INFO_TTL_S:
        return hit["present"]
    present = _probe_ipadapter(url, client=client)
    _IPADAPTER_PROBE_CACHE[url] = {"at": now, "present": present}
    return present


def _checkpoint_family(ckpt: str) -> str:
    """Best-effort model family from the checkpoint FILE NAME — all a comfy row
    tells us. Governs which ip-adapter weight + CLIP-Vision pairing to wire: an
    SD1.5 adapter's cross-attention dims don't match an SDXL model (and neither
    matches flux), so the wrong family is a REAL failure — caught as a ComfyUI
    execution error (errors-as-data), never a silent bad result. Unknowable from
    the name -> 'unknown' (we still try the SD1.5 default + log a note)."""
    n = (ckpt or "").lower()
    if "flux" in n:
        return "flux"
    if "sdxl" in n or "xl" in n:            # sdxl, *_xl, juggernautXL, …
        return "sdxl"
    if any(t in n for t in ("sd15", "sd_15", "sd-1.5", "sd1.5", "v1-5", "v1.5")):
        return "sd15"
    return "unknown"


# family -> (ip-adapter weight file, CLIP-Vision model file). These MUST match
# the files WORKER-SETUP §5b places under ComfyUI's models/ipadapter and
# models/clip_vision. Operators who chose different filenames override per box
# via COMFY_IPADAPTER_WEIGHT / COMFY_CLIPVISION_MODEL.
_IPADAPTER_FILES: Dict[str, tuple] = {
    "sd15": ("ip-adapter_sd15.safetensors",
             "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"),
    "sdxl": ("ip-adapter_sdxl_vit-h.safetensors",
             "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"),
}


def _ipadapter_files(family: str) -> tuple:
    """(weight_file, clip_vision_file) for a family, env-overridable. flux/unknown
    fall back to the SD1.5 pairing (the caller logs the family-mismatch note)."""
    default = _IPADAPTER_FILES.get(family) or _IPADAPTER_FILES["sd15"]
    return (os.environ.get("COMFY_IPADAPTER_WEIGHT") or default[0],
            os.environ.get("COMFY_CLIPVISION_MODEL") or default[1])


def _ipadapter_workflow(ckpt: str, req: ImageGenRequest, base_wf: Dict[str, Any],
                        uploaded_names: "list[str]") -> Dict[str, Any]:
    """Wrap a base graph (t2i node 1-7 OR i2i +8,9) with the IPAdapter chain and
    return it. The reference embedding patches the checkpoint MODEL; the base
    graph's KSampler (node 5) is rewired onto the patched model and is otherwise
    UNTOUCHED — sampler/scheduler/cfg/seed/denoise all carry through, so an
    id-locked render honours every knob a plain render does.

    Node ids 20/21 (loaders), 30+ (one LoadImage per reference), 40+ (pairwise
    ImageBatch to fold N references into one batch), 50 (apply) sit ABOVE the
    base graph's 1-9 so they never collide."""
    wf = base_wf
    family = _checkpoint_family(ckpt)
    ipa_file, clip_file = _ipadapter_files(family)
    if family in ("flux", "unknown"):
        logger.warning(
            "ComfyRunner id_lock: checkpoint %r family=%s — the SD/SDXL pairing "
            "(%s + %s) may not match; a mismatch surfaces as a ComfyUI execution "
            "error (errors-as-data), never a silent non-locked image. Override "
            "with COMFY_IPADAPTER_WEIGHT / COMFY_CLIPVISION_MODEL.",
            ckpt, family, ipa_file, clip_file)
    wf["20"] = {"class_type": IPADAPTER_LOADER_NODE,
                "inputs": {"ipadapter_file": ipa_file}}
    wf["21"] = {"class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": clip_file}}
    load_ids: list[str] = []
    for i, name in enumerate(uploaded_names):
        nid = str(30 + i)
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        load_ids.append(nid)
    # Fold N reference LoadImages into ONE image input (core ImageBatch takes two
    # at a time, so chain them). One reference needs no batching.
    ref_image = [load_ids[0], 0]
    batch_seq = 40
    for nid in load_ids[1:]:
        bid = str(batch_seq)
        batch_seq += 1
        wf[bid] = {"class_type": "ImageBatch",
                   "inputs": {"image1": ref_image, "image2": [nid, 0]}}
        ref_image = [bid, 0]
    weight = req.id_strength if req.id_strength is not None else 0.6
    wf["50"] = {"class_type": IPADAPTER_APPLY_NODE,
                "inputs": {"model": ["1", 0], "ipadapter": ["20", 0],
                           "image": ref_image, "clip_vision": ["21", 0],
                           "weight": float(weight), "weight_type": "linear",
                           "combine_embeds": "concat", "start_at": 0.0,
                           "end_at": 1.0, "embeds_scaling": "V only"}}
    wf["5"]["inputs"]["model"] = ["50", 0]      # sampler runs on the patched model
    return wf


class ComfyRunner:
    """Runner for ComfyUI-backed image tasks — text-to-image + image-to-image."""

    request_type = ImageGenRequest
    result_type = ImageGenResult

    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        self.model_key = cfg.model_key
        # The checkpoint FILE NAME inside ComfyUI's models/checkpoints — the
        # registry row's `filename` is the designation.
        self.checkpoint = getattr(cfg, "filename", None)
        if not self.checkpoint:
            raise ValueError(
                f"{self.model_key}: comfy model rows must set `filename` to the "
                "checkpoint name inside ComfyUI's models/checkpoints")

    # -- HTTP helpers --------------------------------------------------------
    def _upload_image_bytes(self, client, name: str, data: bytes) -> str:
        """Upload raw image bytes to ComfyUI, returning the stored name (the
        handle a LoadImage node references)."""
        files = {"image": (name, data, "image/png")}
        r = client.post(_comfy_url() + "/upload/image",
                        files=files, data={"overwrite": "true"})
        r.raise_for_status()
        return r.json()["name"]

    def _upload_init_image(self, client, path: str) -> str:
        with open(path, "rb") as fh:
            return self._upload_image_bytes(client, os.path.basename(path), fh.read())

    def _reference_payloads(self, req: ImageGenRequest) -> "list[tuple[str, bytes]]":
        """(filename, bytes) for the id_lock reference stills — from the b64
        OFFLOAD transport (a worker can't see central's paths) when present, else
        straight from the jailed paths (the in-process / worker-local case)."""
        out: "list[tuple[str, bytes]]" = []
        if req.reference_images_b64:
            for i, b in enumerate(req.reference_images_b64):
                out.append((f"{req.request_id}_ref{i}.png", base64.b64decode(b)))
        elif req.reference_images:
            for i, p in enumerate(req.reference_images):
                ext = os.path.splitext(p)[1] or ".png"
                with open(p, "rb") as fh:
                    out.append((f"{req.request_id}_ref{i}{ext}", fh.read()))
        return out

    def _generate(self, req: ImageGenRequest) -> "list[GeneratedImage]":
        import httpx

        # Fix B: ensure headroom before the comfy gen commits. Evict on-demand
        # managed models (LRU) until real free VRAM reaches the target, so
        # ComfyUI's demand is a first-class queue entry instead of a silent
        # squatter. Runs UNCONDITIONALLY (a no-op when already above target, when
        # no hook is registered, or on a no-GPU box) and honest-degrades: if
        # eviction can't reach the target it proceeds anyway (logged), never
        # blocking the gen. Single choke point: every comfy gen (t2i/i2i/id_lock)
        # passes through _generate before the /prompt POST below.
        _ensure_comfy_headroom(self.model_key, getattr(req, "request_id", None))

        seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(4), "big")
        url = _comfy_url()
        references = self._reference_payloads(req)
        with httpx.Client(timeout=30.0) as client:
            # ID-LOCK gate: an identity-locked request is DEFINED by its reference
            # images (the studio errors.py REFERENCE_MISSING invariant, mirrored).
            # Detect the IPAdapter pack on THIS comfy before composing the graph —
            # if it's absent, fail as data with the install pointer rather than
            # degrade to a non-locked image.
            if references and not comfy_has_ipadapter(url, client=client):
                raise RuntimeError(
                    f"ComfyUI at {url} lacks the IPAdapter nodes — install "
                    "ComfyUI_IPAdapter_plus + weights (see WORKER-SETUP §5b)")
            if req.image_path:
                uploaded = self._upload_init_image(client, req.image_path)
                workflow = _i2i_workflow(self.checkpoint, req, seed, uploaded)
            else:
                workflow = _t2i_workflow(self.checkpoint, req, seed)
            if references:
                names = [self._upload_image_bytes(client, n, d) for n, d in references]
                workflow = _ipadapter_workflow(self.checkpoint, req, workflow, names)

            r = client.post(_comfy_url() + "/prompt", json={
                "prompt": workflow, "client_id": f"hugpy-{uuid.uuid4().hex[:8]}"})
            if r.status_code != 200:
                raise RuntimeError(
                    f"ComfyUI rejected the workflow ({r.status_code}): "
                    f"{r.text[:400]}")
            prompt_id = r.json()["prompt_id"]
            mode = "img2img" if req.image_path else "text2img"
            if references:
                mode += f"+id_lock({len(references)}ref)"
            logger.info("ComfyRunner: %s submitted prompt %s (ckpt=%s, %s)",
                        self.model_key, prompt_id, self.checkpoint, mode)

            try:
                # Poll history until outputs arrive (generation runs server-side).
                deadline = time.time() + _TIMEOUT_S
                outputs = None
                while time.time() < deadline:
                    h = client.get(_comfy_url() + f"/history/{prompt_id}").json()
                    entry = h.get(prompt_id)
                    if entry:
                        status = (entry.get("status") or {})
                        if status.get("status_str") == "error":
                            msgs = [m for m in (status.get("messages") or [])
                                    if m and m[0] == "execution_error"]
                            detail = json.dumps(msgs[-1][1] if msgs else status)[:400]
                            raise RuntimeError(f"ComfyUI execution error: {detail}")
                        if entry.get("outputs"):
                            outputs = entry["outputs"]
                            break
                    time.sleep(_POLL_S)
                if outputs is None:
                    raise TimeoutError(
                        f"ComfyUI did not finish prompt {prompt_id} within "
                        f"{_TIMEOUT_S:.0f}s")

                out_dir = os.path.join(UPLOADS_HOME, "generated")
                os.makedirs(out_dir, exist_ok=True)
                images: list[GeneratedImage] = []
                index = 0
                for node_out in outputs.values():
                    for im in node_out.get("images", []):
                        r = client.get(_comfy_url() + "/view", params={
                            "filename": im["filename"],
                            "subfolder": im.get("subfolder", ""),
                            "type": im.get("type", "output")})
                        r.raise_for_status()
                        data = r.content
                        path = os.path.join(out_dir, f"{req.request_id}_{index}.png")
                        with open(path, "wb") as fh:
                            fh.write(data)
                        b64 = (base64.b64encode(data).decode("ascii")
                               if req.return_b64 else None)
                        images.append(GeneratedImage(
                            path=path, b64=b64, width=req.width, height=req.height,
                            seed=seed))
                        index += 1
                if not images:
                    raise RuntimeError("ComfyUI finished but produced no images")
                return images
            except BaseException as exc:
                # Tag the failure with the submission it belongs to, so the
                # one-shot VRAM retry seam (run) can interrupt/clean exactly
                # this prompt before re-submitting. Best-effort: a tag that
                # can't stick just means the cancel falls back to a bare
                # /interrupt.
                try:
                    exc.comfy_prompt_id = prompt_id
                except Exception:  # noqa: BLE001
                    pass
                raise

    # -- one-shot VRAM retry (k71) -------------------------------------------
    def _cancel_prompt(self, prompt_id) -> None:
        """Best-effort cancel/clean of a submitted comfy prompt before the ONE
        retry, so the retry cannot duplicate: drop it from the queue if it is
        still pending (vanilla ``POST /queue {"delete": [id]}``) and interrupt
        it if it is the one running (``POST /interrupt``). Never raises — the
        first attempt usually already terminated server-side (that is why we
        are here at all), and a cancel failure must not block the retry."""
        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                if prompt_id:
                    try:
                        client.post(_comfy_url() + "/queue",
                                    json={"delete": [prompt_id]})
                    except Exception:  # noqa: BLE001 — best-effort clean
                        pass
                client.post(_comfy_url() + "/interrupt")
        except Exception:  # noqa: BLE001 — best-effort cancel, never blocks
            pass

    def _settle_for_retry(self, req: ImageGenRequest, prompt_id) -> None:
        """Between a retryable VRAM-class first failure and the ONE retry:
        cancel/clean the first submission, re-drive the SAME ensure-comfy-
        headroom eviction hook every attempt gets (so the pool state the first
        attempt tripped over is actually re-driven, not just waited out), then
        give the pool a bounded few seconds to settle. Every step is
        best-effort — the retry proceeds regardless."""
        self._cancel_prompt(prompt_id)
        _ensure_comfy_headroom(self.model_key, getattr(req, "request_id", None))
        delay = settle_delay_s()
        if delay > 0:
            time.sleep(delay)

    # -- public API (mirrors Img2ImgRunner) ----------------------------------
    async def run(self, req: ImageGenRequest) -> ImageGenResult:
        # Call-time attribution: record which model this comfy dispatch is FOR (job
        # id = the hugpy request id) for the whole generation window, cleared in the
        # finally. Best-effort — a telemetry failure never touches the result path.
        _job_id = getattr(req, "request_id", None)
        _reg_comfy_call(self.model_key, _job_id)
        try:
            try:
                images = await asyncio.to_thread(self._generate, req)
            except Exception as first_exc:
                # ONE retry, and only for the VRAM/eviction/OOM/allocation
                # class (stale pre-eviction pool state: the identical second
                # try succeeds once the eviction settles). Everything else —
                # workflow validation, missing checkpoint, timeout, no-images —
                # surfaces exactly as before. A second failure takes the outer
                # except, i.e. exactly today's error path.
                if not is_retryable_vram_failure(first_exc):
                    raise
                logger.warning(
                    "ComfyRunner %s: retrying ONCE — first attempt failed with "
                    "a VRAM/allocation-class error (%s: %s); cancelling the "
                    "first submission and re-driving comfy headroom eviction "
                    "before the retry", self.model_key,
                    type(first_exc).__name__, first_exc)
                await asyncio.to_thread(
                    self._settle_for_retry, req,
                    getattr(first_exc, "comfy_prompt_id", None))
                images = await asyncio.to_thread(self._generate, req)
            return ImageGenResult(
                request_id=req.request_id, model_key=req.model_key,
                ok=True, images=images,
                text=f"generated {len(images)} image(s) via ComfyUI")
        except Exception as exc:  # noqa: BLE001 — errors are data
            logger.warning("ComfyRunner %s failed: %s", self.model_key, exc)
            return ImageGenResult(
                request_id=req.request_id, model_key=req.model_key,
                ok=False, images=[], error=f"{type(exc).__name__}: {exc}")
        finally:
            _end_comfy_call(_job_id)
