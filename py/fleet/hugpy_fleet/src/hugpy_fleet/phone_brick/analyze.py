"""Reason over phone-detected image analysis with an LLM (pipeline path "b").

The phone fleet does the *seeing* — :class:`OnnxYoloDetector` returns boxes and
class labels (see :mod:`abstract_hugpy_dev.phone_brick.detector`). This module does the
*reasoning*: it renders those detections as text and hands them to a chat model
via :func:`abstract_hugpy_dev.managers.dispatch.execute_prompt`, which produces a natural-
language read of the scene ("worker at left is missing a helmet", etc.).

The reasoning model defaults to ``DEFAULT_CHAT_MODEL`` — a **GGUF** model, so
its inference can itself be sharded across the fleet, and a phone running the
RPC shard backend (:mod:`abstract_hugpy_dev.phone_brick.rpc_backend`) can lend compute to
that step. So a phone contributes to image analysis twice over: as the detector
and as a shard backend for the model that interprets the detections.

The pure formatting helpers (:func:`detections_to_text`,
:func:`build_analysis_prompt`) import nothing from hugpy, so they stay testable
and usable on a phone; :func:`analyze_detections` lazily imports the dispatch
stack and is meant to run on the control box, not the phone.
"""
from __future__ import annotations

from typing import Optional, Sequence

from hugpy_fleet.phone_brick.schemas import ChainResult, Detection
from hugpy_fleet.phone_brick.protocol import parse_detections

_DEFAULT_QUESTION = (
    "Describe what is happening in this scene based on the detected objects, "
    "and flag any safety concerns (e.g. missing PPE such as helmet, vest, "
    "gloves, goggles, mask, or safety shoes)."
)


# ---------------------------------------------------------------------------
# Pure formatting (no hugpy deps — testable, phone-safe)
# ---------------------------------------------------------------------------
def detections_to_text(detections: Sequence[Detection]) -> str:
    """Render detections as a compact, model-friendly bullet list."""
    if not detections:
        return "(no objects detected)"
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        box = f" at [{x1},{y1},{x2},{y2}]" if any((x1, y1, x2, y2)) else ""
        lines.append(f"- {det.cls} ({det.conf_pct}% confidence){box}")
    return "\n".join(lines)


def chain_result_to_text(result: ChainResult) -> str:
    """Summarise a whole phone-chain verdict: each phone's call + the detections.

    Includes the per-phone consensus flag (AGR/DIS/NOD) so the model can weigh
    agreement across the fleet, then the union of detected objects.
    """
    blocks = [f"Image: {result.image}", "", "Per-phone verdicts:"]
    union: list[Detection] = []
    seen: set = set()
    for phase in result.phases:
        blocks.append(
            f"- {phase.phone}: top={phase.top_cls} "
            f"({phase.top_conf_pct}%) consensus={phase.consensus} "
            f"({len(phase.detections)} detection(s))")
        for det in phase.detections:
            key = (det.cls, det.bbox)
            if key not in seen:
                seen.add(key)
                union.append(det)
    blocks += ["", "Detected objects (union across phones):",
               detections_to_text(union)]
    return "\n".join(blocks)


def build_analysis_prompt(scene_text: str, question: Optional[str] = None) -> str:
    """Compose the prompt sent to the reasoning model."""
    return (
        f"{question or _DEFAULT_QUESTION}\n\n"
        f"Object-detection results from a fleet of phone cameras:\n"
        f"{scene_text}\n"
    )


# ---------------------------------------------------------------------------
# Reasoning step (runs on the control box; lazy-imports the dispatch stack)
# ---------------------------------------------------------------------------
def analyze_scene_text(
    scene_text: str,
    *,
    model_key: Optional[str] = None,
    question: Optional[str] = None,
    max_new_tokens: int = 512,
) -> str:
    """Send pre-rendered detection text to a chat model and return its analysis.

    ``model_key`` defaults to ``DEFAULT_CHAT_MODEL`` (a GGUF model, so the step
    is shardable). The dispatch import is deferred so importing this module on a
    phone stays dependency-light.
    """
    from hugpy_engine.dispatch.dispatch import execute_prompt
    if model_key is None:
        from hugpy_platform.constants import DEFAULT_CHAT_MODEL
        model_key = DEFAULT_CHAT_MODEL

    # NO-THINK (utils/no_think.py). The contract here is display-as-prose: the
    # return value is handed straight to the operator/orchestrator as THE
    # analysis. DEFAULT_CHAT_MODEL is non-reasoning today, so this is latent —
    # but it fires the moment the chat default moves to a reasoning model, which
    # is exactly what happened to /video/prompt/assist.
    from hugpy_engine.utils.no_think import with_no_think, strip_think
    prompt = with_no_think(build_analysis_prompt(scene_text, question))
    result = execute_prompt(
        model_key=model_key, prompt=prompt, max_new_tokens=max_new_tokens)
    # execute_prompt returns a result object (ChatResult-like) or a dict.
    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    if text is None:
        return str(result)
    analysis, _reasoning = strip_think(text)
    return analysis


def analyze_detections(
    detections: Sequence[Detection], **kwargs
) -> str:
    """Convenience: render then reason over a flat list of detections."""
    return analyze_scene_text(detections_to_text(detections), **kwargs)


def analyze_chain_result(result: ChainResult, **kwargs) -> str:
    """Convenience: reason over a whole phone-chain :class:`ChainResult`."""
    return analyze_scene_text(chain_result_to_text(result), **kwargs)


def analyze_worker_output(text: str, **kwargs) -> str:
    """Convenience: parse a single worker's raw ``yolo`` output, then reason."""
    return analyze_detections(parse_detections(text), **kwargs)
