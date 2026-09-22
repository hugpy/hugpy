"""Oracle scorecard builder (k90b): deterministic technical checks, every response.

Operator ruling (KEEPER-ASSESSMENT Decision, 2026-08-05): the Scorecard is
load-bearing from day one. This module builds it from evidence OUTSIDE the
generator — the receipt's failure class and cheap direct inspection of the
produced artifacts. Checks implemented here:

  execution         — the receipt carries no failure class (TIMEOUT /
                      WORKER_UNAVAILABLE / RUNNER_ERROR surface here)
  empty_output      — >=1 artifact with substance (text: non-blank; inline
                      data: truthy; file: exists and >0 bytes; AUDIO
                      additionally carries sound — a measurable wav whose peak
                      is below SILENT_AUDIO_PEAK_FLOOR is empty output, and the
                      measured peak/RMS is named on the check)
  format            — every artifact kind is one the capability declares in
                      ``produces`` (catalog IO table)
  decode            — file artifacts are readable; images additionally open
                      via PIL with nonzero dimensions (size-only when PIL is
                      not installed — the degradation is named in the check)

``hard_pass`` is the conjunction. ``judge_results`` is PRESENT AND EMPTY on
every card built here — that tuple is the k90c seam: the evaluator kernel
(judges, heterogeneous evidence, repair loop) fills it; k90b deliberately does
not. ``build_gap_scorecard`` / ``build_deferred_scorecard`` cover the two
non-executed response shapes so the scorecard is mandatory on EVERY response.
"""

from __future__ import annotations

import array
import math
import os
import sys
import wave
from typing import Any

from hugpy_oracle.contracts import (
    ArtifactKind,
    Check,
    CheckKind,
    ExecutionReceipt,
    FailureClass,
    GoalSpec,
    RepairCode,
    Scorecard,
)
from hugpy_oracle.router import RouteDecision

# Receipt failure class -> the repair code named on the card. RUNNER_ERROR /
# REFUSED / CANCELLED / UNKNOWN have no regeneration verb in RepairCode — they
# fail the card with a diagnosis and no repair_code (k90c may widen this).
_FAILURE_REPAIR: dict[FailureClass, RepairCode] = {
    FailureClass.TIMEOUT:            RepairCode.TIMEOUT,
    FailureClass.WORKER_UNAVAILABLE: RepairCode.WORKER_UNAVAILABLE,
    FailureClass.DECODE_FAILED:      RepairCode.DECODE_FAILED,
    FailureClass.EMPTY_OUTPUT:       RepairCode.EMPTY_OUTPUT,
    FailureClass.FORMAT_MISMATCH:    RepairCode.FORMAT_MISMATCH,
    FailureClass.CAPABILITY_GAP:     RepairCode.CAPABILITY_GAP,
}


#: Peak PCM amplitude, in int16 units, below which a produced audio artifact is
#: DIGITAL SILENCE rather than quiet speech. 500/32767 is -36 dBFS peak: live
#: chatterbox speech on this fleet peaks at 21 000-32 000 (about -3 dBFS), and
#: the silent-wav failure this guard exists for peaks at 1 (-90 dBFS). Nothing
#: legitimate lands in between, so the floor is deliberately far from both.
SILENT_AUDIO_PEAK_FLOOR: int = 500


def _wav_levels(uri: str) -> tuple[int, float] | None:
    """(peak, rms) in int16 units for a 16-bit PCM wav — or None when the file
    is not one we can measure (compressed audio, another sample width, an
    unreadable file). Stdlib only, on purpose: deciding "did anything come out
    of the speaker" must not acquire a numpy/soundfile dependency, and a format
    this cannot measure is never called silent."""
    try:
        with wave.open(uri, "rb") as fh:
            if fh.getsampwidth() != 2:
                return None
            frames = fh.readframes(fh.getnframes())
    except Exception:  # noqa: BLE001 — unmeasurable is not the same as silent
        return None
    data = array.array("h")
    data.frombytes(frames[:len(frames) - (len(frames) % 2)])
    if sys.byteorder == "big":                      # wav is little-endian
        data.byteswap()
    if not data:
        return 0, 0.0
    peak = max(max(data), -min(data))
    rms = math.sqrt(sum(float(v) * v for v in data) / len(data))
    return peak, rms


def _audio_substance(uri: str) -> tuple[bool, str]:
    """(carries sound, detail) for an audio artifact — the CONTENT guard.

    A wav can be a valid, non-zero-byte, exactly-right-duration file and still
    contain nothing: the tts-silence fault (2026-08-21) wrote 2.32 s of PCM16
    whose peak amplitude was 1 and whose RMS was -117 dBFS, and every existing
    technical check passed it hard_pass. Existence is not substance for audio,
    so this measures the samples. The measured level is named in the detail on
    BOTH branches, so an operator reads the number, not a verdict."""
    levels = _wav_levels(uri)
    if levels is None:
        return True, f"audio level unmeasurable (not 16-bit PCM wav): {uri}"
    peak, rms = levels
    dbfs = 20 * math.log10(rms / 32768.0) if rms > 0 else float("-inf")
    return peak >= SILENT_AUDIO_PEAK_FLOOR, (
        f"audio peak {peak}/32767, RMS {dbfs:.1f} dBFS "
        f"(silence floor: peak {SILENT_AUDIO_PEAK_FLOOR})")


def _has_substance(art: dict[str, Any]) -> bool:
    if "text" in art:
        return bool(str(art["text"]).strip())
    if "data" in art:
        return bool(art["data"])
    uri = art.get("uri", "")
    try:
        if not (os.path.isfile(uri) and os.path.getsize(uri) > 0):
            return False
    except OSError:
        return False
    if art.get("kind") == ArtifactKind.AUDIO.value:
        return _audio_substance(uri)[0]
    return True


def _decode_file(art: dict[str, Any]) -> tuple[bool, str]:
    """(readable, detail) for a file-backed artifact. Images get a real decode
    via PIL when available; everything else is open-and-read-a-byte."""
    uri = art.get("uri", "")
    if not os.path.isfile(uri):
        return False, f"file missing: {uri}"
    try:
        with open(uri, "rb") as fh:
            if not fh.read(1):
                return False, f"zero-byte file: {uri}"
    except OSError as exc:
        return False, f"unreadable ({exc}): {uri}"
    if art.get("kind") == ArtifactKind.IMAGE.value:
        try:
            from PIL import Image  # optional dep — degrade, don't require
        except ImportError:
            return True, "readable (PIL unavailable — size-only image check)"
        try:
            with Image.open(uri) as img:
                w, h = img.size
            if w <= 0 or h <= 0:
                return False, f"image decodes to zero dimensions: {uri}"
            return True, f"image decodes ({w}x{h})"
        except Exception as exc:  # noqa: BLE001 — undecodable is the finding
            return False, f"image undecodable ({type(exc).__name__}: {exc})"
    return True, "readable"


def build_technical_scorecard(goal: GoalSpec, route: RouteDecision,
                              artifacts: list[dict[str, Any]],
                              receipt: ExecutionReceipt) -> Scorecard:
    """The mandatory deterministic card for an executed route."""
    checks: list[Check] = []
    repair: RepairCode | None = None
    diagnoses: list[str] = []

    exec_ok = receipt.failure is None
    checks.append(Check(
        name="execution", kind=CheckKind.TECHNICAL,
        value="ok" if exec_ok else receipt.failure.value, threshold=None,
        passed=exec_ok,
        detail="" if exec_ok else "; ".join(receipt.log_excerpt)[:500]))
    if not exec_ok:
        repair = _FAILURE_REPAIR.get(receipt.failure)
        diagnoses.append(f"execution failed: {receipt.failure.value}")

    substantive = [a for a in artifacts if _has_substance(a)]
    empty_ok = bool(substantive)
    # Audio levels are MEASURED and reported whether or not they fail: a card
    # that says "3 s of audio, peak 1/32767" is the difference between an
    # operator seeing the fault and shipping it.
    audio_levels = [_audio_substance(a["uri"]) for a in artifacts
                    if a.get("kind") == ArtifactKind.AUDIO.value and a.get("uri")
                    and os.path.isfile(a["uri"])]
    detail = (f"{len(substantive)}/{len(artifacts)} artifacts carry substance"
              if artifacts else "no artifacts produced")
    if audio_levels:
        detail += "; " + "; ".join(d for _ok, d in audio_levels)
    checks.append(Check(
        name="empty_output", kind=CheckKind.TECHNICAL,
        value=len(substantive), threshold=1, passed=empty_ok,
        detail=detail))
    if not empty_ok:
        repair = repair or RepairCode.EMPTY_OUTPUT
        silent = [d for ok, d in audio_levels if not ok]
        diagnoses.append(
            f"no substantive artifact — synthesized audio is digital silence: "
            f"{'; '.join(silent)}" if silent
            else "no substantive artifact (blank/zero-byte output)")

    declared = {k.value for k in route.produces}
    observed = [a.get("kind", "?") for a in artifacts]
    fmt_ok = all(k in declared for k in observed) if declared else True
    checks.append(Check(
        name="format", kind=CheckKind.TECHNICAL,
        value=",".join(sorted(set(observed))) or "(none)",
        threshold=",".join(sorted(declared)), passed=fmt_ok,
        detail="artifact kinds vs the capability's declared produces"))
    if not fmt_ok:
        repair = repair or RepairCode.FORMAT_MISMATCH
        diagnoses.append(
            f"produced kind(s) {sorted(set(observed) - declared)} not in "
            f"{route.capability}'s declared produces")

    file_arts = [a for a in artifacts if "text" not in a and "data" not in a]
    decode_details: list[str] = []
    decode_ok = True
    for art in file_arts:
        ok, detail = _decode_file(art)
        decode_ok = decode_ok and ok
        decode_details.append(detail)
    checks.append(Check(
        name="decode", kind=CheckKind.TECHNICAL,
        value=len(file_arts), threshold=None, passed=decode_ok,
        detail="; ".join(decode_details) or "no file-backed artifacts"))
    if not decode_ok:
        repair = repair or RepairCode.DECODE_FAILED
        diagnoses.append("a produced file is missing/unreadable/undecodable")

    hard_pass = all(c.passed for c in checks)
    return Scorecard(
        hard_pass=hard_pass,
        checks=tuple(checks),
        judge_results=(),   # k90c seam: the evaluator kernel fills this
        confidence=1.0,     # deterministic checks — no judge disagreement yet
        diagnosis="; ".join(diagnoses) or None,
        repair_code=None if hard_pass else repair,
        recommended_repair=None if hard_pass else (
            "re-route or regenerate per repair_code (bounded repair loop "
            "lands in k90c)"))


def build_gap_scorecard(route: RouteDecision) -> Scorecard:
    """The card for a CAPABILITY_GAP response — nothing executed; the catalog's
    eligibility reasons are the evidence."""
    return Scorecard(
        hard_pass=False,
        checks=(Check(
            name="route.eligibility", kind=CheckKind.TECHNICAL,
            value="capability_gap", threshold=None, passed=False,
            detail="; ".join(route.reasons) or route.capability),),
        judge_results=(),   # k90c seam
        diagnosis=f"no eligible route for {route.capability!r}",
        repair_code=RepairCode.CAPABILITY_GAP,
        recommended_repair=("register/unblock a model serving this capability, "
                            "or pick one from GET /oracle/capabilities"))


def build_deferred_scorecard(route: RouteDecision) -> Scorecard:
    """The card for a deferred (video.*) response: routing succeeded, execution
    did not happen here — hard_pass must not claim it did."""
    return Scorecard(
        hard_pass=False,
        checks=(Check(
            name="execution.deferred", kind=CheckKind.TECHNICAL,
            value="deferred", threshold=None, passed=False,
            detail=(f"routed to {route.model_id or 'the studio router'}; "
                    "video capabilities execute through the studio job "
                    "pipeline, not POST /oracle/route")),),
        judge_results=(),   # k90c seam
        diagnosis="video execution is deferred by k90b scope",
        recommended_repair=None)


__all__ = ["SILENT_AUDIO_PEAK_FLOOR", "build_technical_scorecard",
           "build_gap_scorecard", "build_deferred_scorecard"]
