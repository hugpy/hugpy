# video_intel/runners/identity_mesh.py
import json
import logging
import os
import time
import urllib.request
from typing import Any
from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_video.intel.identity_reconstruction_schema import IdentityMeshSpec
from hugpy_video.intel.identity_profiles import get_profile

logger = logging.getLogger(__name__)

COMFY_API_URL = "http://127.0.0.1:8188"

def _resolve_view_uris(slug: str, recon_id: str, view_ids: tuple[str, ...]) -> dict[str, str]:
    """Extracts the physical disk paths for the selected approved angle views."""
    profile = get_profile(slug)
    recons = profile.get("reconstructions", [])
    target_recon = next((r for r in recons if r.get("recon_id") == recon_id), None)

    if not target_recon:
        raise RuntimeError(f"Reconstruction {recon_id} not found for {slug}")

    views = target_recon.get("views", [])
    uri_map = {}

    for vid in view_ids:
        view_data = next((v for v in views if v.get("viewId") == vid), None)
        if not view_data or not view_data.get("imageUri"):
            raise RuntimeError(f"Approved view {vid} missing or has no image data")
        uri_map[vid] = view_data["imageUri"]

    return uri_map

def build_hunyuan3d_payload(uri_map: dict[str, str]) -> dict:
    """
    Constructs the ComfyUI API JSON graph for Hunyuan3D-2mv.
    The graph explicitly isolates the Shape and Paint nodes to allow the ComfyUI
    execution engine to unload the Shape model from VRAM before loading the Texture model.
    """
    # Note: This is a structural skeleton of the ComfyUI API payload.
    # The actual node IDs and class types depend on your specific ComfyUI custom nodes
    # (e.g., 'Hunyuan3D_Multiview_Input', 'Hunyuan3D_ShapeGen', 'Hunyuan3D_PaintGen').

    prompt = {
        "1": {
            "class_type": "LoadImageList",
            "inputs": {
                # Map the 8 anchor paths (0, 45, 90, 135, 180, 225, 270, 315)
                "image_paths": list(uri_map.values())
            }
        },
        "2": {
            "class_type": "Hunyuan3DShapeGenerator",
            "inputs": {
                "images": ["1", 0],
                "force_vram_unload_after": True # Custom flag telling the node to release VRAM
            }
        },
        "3": {
            "class_type": "EmptyCUDACache", # Explicit barrier node
            "inputs": {
                "dependency": ["2", 0]
            }
        },
        "4": {
            "class_type": "Hunyuan3DPBRTextureGenerator",
            "inputs": {
                "mesh": ["2", 0],
                "primary_image": ["1", 0], # Usually the 0-degree front view
                "dependency": ["3", 0] # Ensure cache clears before paint loads
            }
        },
        "5": {
            "class_type": "SaveGLB",
            "inputs": {
                "mesh": ["4", 0],
                "filename_prefix": "identity_mesh"
            }
        }
    }
    return prompt

def _submit_and_wait(payload: dict) -> str | None:
    """Submit the graph to the local ComfyUI and poll /history until it
    completes; returns the output GLB name (or None if node 5 reported no
    files, which the caller surfaces as before).

    A history entry whose status says "error" now RAISES a RuntimeError naming
    the ComfyUI execution error (previously it read as a silent completion with
    no outputs) — so the k71 one-shot VRAM retry in run() can classify it. The
    exception is tagged with the prompt id so the retry can interrupt/clean
    exactly this submission.
    """
    req = urllib.request.Request(
        f"{COMFY_API_URL}/prompt",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read())
        prompt_id = res_data["prompt_id"]

    try:
        # Poll ComfyUI /history endpoint until complete
        # (In a production runner, you would use a websocket for realtime progress)
        while True:
            time.sleep(2)
            hist_req = urllib.request.Request(f"{COMFY_API_URL}/history/{prompt_id}")
            with urllib.request.urlopen(hist_req) as hist_res:
                history = json.loads(hist_res.read())
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        msgs = [m for m in (status.get("messages") or [])
                                if m and m[0] == "execution_error"]
                        detail = json.dumps(msgs[-1][1] if msgs else status)[:400]
                        raise RuntimeError(f"ComfyUI execution error: {detail}")
                    # Parse output node (Node 5) for the generated filename
                    outputs = entry.get("outputs", {})
                    if "5" in outputs and "files" in outputs["5"]:
                        return outputs["5"]["files"][0]
                    return None
    except BaseException as exc:
        try:
            exc.comfy_prompt_id = prompt_id
        except Exception:
            pass
        raise

def _cancel_and_settle(prompt_id, job_id) -> None:
    """Between a retryable VRAM-class first failure and the ONE retry:
    best-effort cancel/clean the first submission (queue delete + /interrupt,
    so the retry cannot duplicate), re-drive the ensure-comfy-headroom eviction
    hook every comfy gen gets, then a bounded settle sleep. Every step is
    best-effort — the retry proceeds regardless."""
    try:
        if prompt_id:
            q = urllib.request.Request(
                f"{COMFY_API_URL}/queue",
                data=json.dumps({"delete": [prompt_id]}).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(q, timeout=5).close()
    except Exception:
        pass
    try:
        i = urllib.request.Request(f"{COMFY_API_URL}/interrupt", data=b"")
        urllib.request.urlopen(i, timeout=5).close()
    except Exception:
        pass
    try:
        from hugpy_media.comfy.comfy_runner import _ensure_comfy_headroom
        _ensure_comfy_headroom("identity-mesh", job_id)
    except Exception:
        pass
    try:
        from hugpy_media.imagegen.vram_retry import settle_delay_s
        delay = settle_delay_s()
    except Exception:
        delay = 3.0
    if delay > 0:
        time.sleep(delay)

def run(spec: IdentityMeshSpec) -> dict[str, Any]:
    """
    Executes the 3D mesh building pipeline.
    """
    try:
        # 1. Gather the approved anchor images
        uri_map = _resolve_view_uris(spec.slug, spec.recon_id, spec.view_ids)

        # 2. Build the ComfyUI graph
        payload = {"prompt": build_hunyuan3d_payload(uri_map)}

        # 3. Submit to local ComfyUI and wait — with ONE retry for the
        # VRAM/eviction/OOM/allocation class only (k71): stale pre-eviction
        # pool state fails the first attempt; an identical second try succeeds
        # once the eviction settles. Every other failure class (validation,
        # missing model, timeout, …) surfaces first-attempt as before, and a
        # second failure surfaces exactly as today.
        from hugpy_media.imagegen.vram_retry import is_retryable_vram_failure
        try:
            output_glb_name = _submit_and_wait(payload)
        except Exception as first_exc:
            if not is_retryable_vram_failure(first_exc):
                raise
            logger.warning(
                "identity_mesh %s/%s: retrying ONCE — first attempt failed "
                "with a VRAM/allocation-class error (%s: %s); cancelling the "
                "first submission and re-driving comfy headroom eviction "
                "before the retry", spec.slug, spec.recon_id,
                type(first_exc).__name__, first_exc)
            _cancel_and_settle(getattr(first_exc, "comfy_prompt_id", None),
                               f"identity-mesh:{spec.slug}:{spec.recon_id}")
            output_glb_name = _submit_and_wait(payload)

        # 5. Move the GLB from ComfyUI output dir to the durable identity profile dir
        comfy_output_path = os.path.join(DEFAULT_ROOT, "comfy_output", output_glb_name)
        durable_path = os.path.join(DEFAULT_ROOT, "identities", spec.slug, "mesh", f"{spec.recon_id}.glb")

        os.makedirs(os.path.dirname(durable_path), exist_ok=True)
        os.rename(comfy_output_path, durable_path)

        # 6. Update the identity_profiles.json mesh status to "completed" (done via bus result watcher usually)

        return {
            "ok": True,
            "outputs": [{
                "kind": "mesh",
                "uri": durable_path,
                "mime": "model/gltf-binary"
            }]
        }

    except Exception as e:
        return {
            "ok": False,
            "error": {"code": "MeshBuildFailed", "message": str(e)}
        }
