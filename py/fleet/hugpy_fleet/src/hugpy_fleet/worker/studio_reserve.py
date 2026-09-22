"""Studio render VRAM RESERVE — template evict-to-fit + a non-evictable hold.

Operator ask (2026-08-13, model battery): "specific reserves for video
generation — a template evict-to-fit-and-reserve rather than depending on the
standard evict methods." The failure this deletes, observed live during the
battery: a wan2.2-a14b render (peak ~22.1 GiB) OOM'd because an LLM slot
demand-loaded 3.2 GiB mid-render — the render was INVISIBLE to the worker's
residency economy, so nothing stopped the card from being re-filled under it.

Mechanism (all via the worker's OWN public ops seams, self-HTTP on localhost —
no state plumbing, and the registry stays the single authority):

  acquire(job_id, model_id):
    1. TEMPLATE: resolve the model's reserve target (GiB) — operator override
       env ``HUGPY_STUDIO_RESERVE_TEMPLATES`` (JSON map, fnmatch keys) over the
       built-in defaults below (battery-measured envelopes + margin).
    2. CLAIM: POST /ops/external/claim {target_free_gib} — the same
       idle-guarded, unforced evict-to-target loop gpu_lease batch jobs use
       (feasibility pre-check, in-flight guard, comfy reclaimed last).
    3. HOLD: POST /ops/external/register {model_key: "studio:<job_id>",
       vram_gib: target, evictable: false} — the render becomes a visible,
       NON-evictable resident: headroom candidates skip it, the console's
       leases strip shows it, and demand paths that consult the registry see
       the card as spoken for.
  release(): POST /ops/external/unregister — in the render's ``finally``, so
       the hold can never outlive the render (crash included: the agent's own
       process exit clears the in-memory registry anyway).

Best-effort BY DESIGN: a reserve that cannot be taken (no template, claim
falls short, ops route down) logs and lets the render proceed exactly as
today — this module must only ever ADD safety, never a new way to fail a
render. ``reached=false`` is forwarded so the log says what the render is
risking.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# Battery-measured envelopes (2026-08-13, ae 3090 24GB, 480x480x16fps 12 steps)
# + headroom margin. Keys are fnmatch patterns against the spec's model_id.
# wan2.2-a14b: peaks ~22.1 GiB quantized — effectively the whole card; the
# claim will also reclaim idle comfy, which is exactly what it needs.
# 14B-class wan2.1: nf4-quantized load ~19 GiB class (refined by the battery).
# 1.3B class: ~7 GiB observed; 9 leaves margin for the fp32 VAE decode tail.
_DEFAULT_TEMPLATES: dict[str, float] = {
    # (wan2.2-a14b had a 23.0 row here; removed 2026-08-13 with its registry
    # rows — the model cannot complete on this card, see models_seed tombstone.)
    "wan2.1-i2v-14b*": 21.0,
    "wan2.1-vace-14b*": 21.0,
    "wan2.1-*1.3b*": 9.0,
    "ltx-*": 12.0,
    "hunyuanvideo*": 21.0,
    # UNPINNED render (the router picks the model at render time inside
    # produce_clip, so the reserve cannot know the exact envelope): hold the
    # id-movie working-set class. The router's own budget still refuses what
    # truly cannot fit; this just keeps demand loads off the card meanwhile.
    "auto": 12.0,
}
_TEMPLATES_ENV = "HUGPY_STUDIO_RESERVE_TEMPLATES"
# A model with NO matching pattern gets no reserve (today's behavior); an
# UNPINNED spec matches the literal "auto" key above.

_BASE = "http://127.0.0.1:{port}".format(
    port=os.environ.get("WORKER_PORT", "9100"))
_HTTP_TIMEOUT_S = 20.0
# The claim itself can evict several models (each a subprocess kill + registry
# settle) — give it a real budget, it is one-shot per render.
_CLAIM_TIMEOUT_S = 120.0


def _post(path: str, payload: dict, timeout: float = _HTTP_TIMEOUT_S) -> dict:
    req = urllib.request.Request(
        _BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def reserve_gib_for(model_id: str) -> "float | None":
    """The reserve target for ``model_id``: operator env map first (fnmatch,
    first match wins in insertion order), then the built-in defaults. None =
    no template = no reserve."""
    mid = (model_id or "").strip().lower() or "auto"
    raw = os.environ.get(_TEMPLATES_ENV, "")
    if raw:
        try:
            overrides = json.loads(raw)
            for pat, gib in overrides.items():
                if fnmatch.fnmatch(mid, str(pat).lower()):
                    val = float(gib)
                    return val if val > 0 else None
        except Exception:  # noqa: BLE001 — a bad env map must not kill renders
            logger.warning("studio reserve: unparseable %s (ignored)",
                           _TEMPLATES_ENV, exc_info=True)
    for pat, gib in _DEFAULT_TEMPLATES.items():
        if fnmatch.fnmatch(mid, pat):
            return gib
    return None


def acquire(job_id: str, model_id: "str | None"):
    """Take the reserve for one render. Returns a zero-arg ``release``
    callable (always safe to call, including when nothing was reserved)."""
    key = f"studio:{job_id}"
    target = reserve_gib_for(model_id or "")
    if target is None:
        return lambda: None

    # 1. evict-to-fit (idle-guarded, unforced, feasibility-checked worker-side).
    try:
        claim = _post("/ops/external/claim",
                      {"model_key": key, "target_free_gib": target,
                       "min_idle_s": 0,
                       # DEMAND CLASS (2026-08-13): a studio render is a USER
                       # waiting, not a batch peer — evictable external leases
                       # (bluebook OCR) are last-resort candidates for THIS
                       # claim; their supervisors park and resume afterwards.
                       "include_external": True},
                      timeout=_CLAIM_TIMEOUT_S)
        if not claim.get("reached"):
            logger.warning(
                "studio reserve %s (%s): claim fell short of %.1fGiB "
                "(free_after=%s evicted=%s) — rendering anyway at risk",
                key, model_id, target, claim.get("free_after"),
                claim.get("evicted"))
        else:
            logger.info("studio reserve %s (%s): claimed %.1fGiB (evicted=%s)",
                        key, model_id, target, claim.get("evicted"))
    except Exception:  # noqa: BLE001
        logger.warning("studio reserve %s: claim call failed — rendering "
                       "without a reserve", key, exc_info=True)

    # 2. hold: a NON-evictable registry row for the render's lifetime.
    held = False
    try:
        _post("/ops/external/register",
              {"model_key": key, "pid": os.getpid(), "vram_gib": target,
               "evictable": False, "resume": "disabled",
               "note": f"studio render hold: {model_id}"})
        held = True
    except Exception:  # noqa: BLE001
        logger.warning("studio reserve %s: register failed — no hold",
                       key, exc_info=True)

    def release() -> None:
        if not held:
            return
        try:
            _post("/ops/external/unregister", {"model_key": key})
        except Exception:  # noqa: BLE001
            logger.warning("studio reserve %s: unregister failed (registry "
                           "clears on agent restart)", key, exc_info=True)

    return release
