import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional
from abstract_essentials import safe_dump_to_json
from pydantic import BaseModel, Field
class TranscribeWord(BaseModel):
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    probability: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class TranscribeSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str

    seek: Optional[int] = None
    tokens: list[int] = Field(default_factory=list)
    temperature: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None

    words: list[TranscribeWord] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class FrameContext(BaseModel):
    frame_path: str
    timestamp: float
    reason: str
    segment_index: int
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class TranscribeRequest(BaseModel):
    request_id: str
    file_path: str = Field(..., description="Path to an audio or video file.")

    model_key: Optional[str] = None
    pool: Optional[str] = None   # dedicated worker pool (routing); None = general
    # `base` is the fast-cold-start CPU default and is cached as base.pt in the
    # openai-whisper .pt store; overridable per-request via this field.
    model_size: str = "base"
    language: Optional[str] = "english"
    task: Literal["transcribe", "translate"] = "transcribe"
    whisper_model_path: Optional[str] = None

    output_root: Optional[str] = None
    output_audio_path: Optional[str] = None
    copy_source: bool = False

    capture_frames: bool = False
    min_gap_seconds: float = 2.0
    long_segment_seconds: float = 8.0

    cleanup_extracted_audio: bool = False

    # Word-level timestamps (oracle k98: audio.transcribe.word_timestamps).
    # False by default (unchanged existing behavior); catalog._word_timestamps_
    # wired() probes this exact field to know whether the passthrough chain is
    # closed end to end (builders._build_whisper_request -> runner -> execute).
    word_timestamps: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class TranscribeResult(BaseModel):
    ok: bool = True

    request_id: str
    file_path: str
    media_type: Literal["audio", "video"]

    model_key: Optional[str] = None
    model_size: str
    task: Literal["transcribe", "translate"] = "transcribe"

    text: str = ""
    language: Optional[str] = None
    duration: Optional[float] = None

    audio_path: Optional[str] = None
    workspace_dir: Optional[str] = None
    transcript_json_path: Optional[str] = None
    transcript_text_path: Optional[str] = None
    manifest_path: Optional[str] = None

    frames: list[FrameContext] = Field(default_factory=list)
    segments: list[TranscribeSegment] = Field(default_factory=list)

    error: Optional[str] = None

    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Original Whisper/workspace result for debugging or future fields.",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
@dataclass(frozen=True)
class FrameCandidate:
    timestamp: float
    reason: str
    segment_index: int
    text: str

@dataclass
class MediaArtifactManifest:
    source_path: str
    workspace_dir: str
    created_at: str
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_file(self, key: str, path: str | None) -> None:
        if path is not None:
            self.files[key] = str(path)

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def save(self) -> str:
        manifest_path = os.path.join(self.workspace_dir,"manifest.json")
        safe_dump_to_json(data=asdict(self), file_path=manifest_path,indent=2, ensure_ascii=False,encoding="utf-8")
        return str(manifest_path)


@dataclass
class MediaWorkspace:
    source_path: str
    root_dir: str
    manifest: MediaArtifactManifest

    @property
    def audio_path(self) -> str:
        return os.path.join(self.root_dir,"audio.wav")

    @property
    def transcript_json_path(self) -> str:
        return os.path.join(self.root_dir,"transcript.json")

    @property
    def transcript_text_path(self) -> str:
        return os.path.join(self.root_dir,"transcript.txt")

    @property
    def frames_dir(self) -> str:
        path = os.path.join(self.root_dir,"frames")
        os.makedirs(path,exist_ok=True)
        return path

    @property
    def frame_context_path(self) -> str:
        return os.path.join(self.root_dir,"frame_context.json")

    @property
    def frame_analysis_path(self) -> str:
        return os.path.join(self.root_dir,"frame_analysis.json")
    
    def save_manifest(self) -> str:
        return self.manifest.save()


