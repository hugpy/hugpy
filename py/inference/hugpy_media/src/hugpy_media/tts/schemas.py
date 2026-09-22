"""Schemas for the ``text-to-speech`` task (chatterbox seat).

Shaped like the imagegen pair on purpose: the RESULT carries both the on-disk
``path`` (wherever the synthesis actually ran — a worker, usually) and the
``b64`` bytes, because a relayed task's path is on the WORKER's disk and the
caller cannot read it. The oracle's artifact extraction materializes the bytes
into the shared store so the artifact URI is a real, hashable file rather than
a path that only existed on another box (the same reason
``ImageGenRequest.return_b64`` exists).

``duration_s``/``sample_rate`` are MEASURED off the written wav by the runner,
never taken from the backend's claim (doc invariant 11 / k102 rule 1).
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SynthesizedAudio(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str                        # where the wav was written (worker-local when relayed)
    b64: Optional[str] = None        # base64 wav bytes, omitted when return_b64=False
    sample_rate: int
    duration_s: float
    seed: Optional[int] = None
    reference_used: bool = False
    #: The k98 sidecar verbatim (model_id, weights_source, device, backend, …) —
    #: the provenance record travels WITH the bytes instead of being stranded on
    #: the box that made them.
    sidecar: dict = Field(default_factory=dict)


class TtsRequest(BaseModel):
    """One line of speech to synthesize. Field names line up 1:1 with the k98
    adapter's ``TtsSpec`` (this schema is the transport, that dataclass is the
    contract) — plus the transport-only ``request_id``/``model_key``/``pool``/
    ``return_b64`` every other request in this tree carries."""
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    pool: Optional[str] = None
    text: str = Field(min_length=1)

    #: An AUTHORIZED reference voice. ``authorized`` is not "the route is fine",
    #: it is "k97's VOICE gate demanded a grant for this request and got one";
    #: the adapter refuses a reference without it rather than downgrading to the
    #: default voice (doc invariant 12). Rides remote._PATH_KEYS-style inlining
    #: only when a caller supplies a path the worker can read.
    reference_audio: Optional[str] = None
    authorized: bool = False
    voice_style: Optional[str] = None
    seed: Optional[int] = Field(default=None, ge=0)
    language: Optional[str] = None
    device: Optional[str] = None
    return_b64: bool = True


class TtsResult(BaseModel):
    request_id: str
    model_key: str
    ok: bool = True
    audio: List[SynthesizedAudio] = Field(default_factory=list)
    #: Human-readable summary (the text that was spoken + how long it came out),
    #: so a chat-shaped consumer degrades to something sensible.
    text: str = ""
    error: Optional[str] = None
    #: Which failure this was, in the runner's own vocabulary
    #: (``missing_consent`` / ``deps_missing`` / ``bad_spec`` / …). The bus and
    #: the oracle both already understand these codes.
    error_code: Optional[str] = None
    #: MEASURED peak VRAM of the synthesis process, in bytes, when the backend
    #: could report it. None means "not measured here" — never a guess.
    vram_peak_bytes: Optional[int] = None
