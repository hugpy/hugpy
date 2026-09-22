"""Job envelope + registry — map §5.

Registry over globals: JOB_REGISTRY maps a job `name` to its frozen JobSpec
(spec_type, runner_key = (framework, task), queue, timeout_s). The worker looks
up runner_key here and dispatches through the runner DISPATCH table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple, Type

from hugpy_video.intel.audio_schema import AudioExtractSpec
from hugpy_video.intel.crop_schema import CropSpec
from hugpy_video.intel.frame_schema import FrameExtractSpec
from hugpy_video.intel.gen_schema import GenerateImageSpec
from hugpy_video.intel.movie_schema import MovieSpec
from hugpy_video.intel.scene_schema import GenerateSceneSpec
from hugpy_video.intel.studio.job import StudioI2VSpec
from hugpy_video.intel.studio.tester import StudioTesterSpec
from hugpy_video.intel.studio_movie_schema import StudioMovieSpec
from hugpy_video.intel.identity_reconstruction_schema import (
    IdentityReconstructionSpec,
    IdentityMeshSpec,
)
from hugpy_video.intel.identity_video_extract_schema import IdentityVideoExtractSpec
from hugpy_video.intel.identity_from_video_schema import IdentityFromVideoSpec
from hugpy_media.tts.chatterbox_runner import TtsSpec
from hugpy_video.intel.mlt_render_schema import MltRenderSpec


@dataclass(frozen=True)
class JobSpec:
    name: str
    spec_type: Type                       # one of the frozen specs
    runner_key: Tuple[str, str]           # (framework, task)
    queue: str
    timeout_s: int


def register_job(name: str, *, spec_type: Type, runner_key: Tuple[str, str],
                 queue: str, timeout_s: int,
                 from_dict: Callable[[dict], object],
                 runner: Callable[[Any, str], Any],
                 replace: bool = True) -> JobSpec:
    """Register a bus job owned by an UPPER layer (the oracle's
    ``video_performance``, a plugin's extra runner) without video importing it.

    Wires the three registry tables the bus reads: ``JOB_REGISTRY`` (envelope),
    ``media_bus.SPEC_DESERIALIZERS`` (json -> frozen spec) and
    ``runners.DISPATCH`` (``(spec, job_id) -> JobResult``). ``replace=False``
    keeps an existing registration and returns it.
    """
    from hugpy_video.intel import media_bus
    from hugpy_video.intel.runners import DISPATCH

    if not replace and name in JOB_REGISTRY:
        return JOB_REGISTRY[name]
    spec = JobSpec(name, spec_type, tuple(runner_key), queue, int(timeout_s))
    JOB_REGISTRY[name] = spec
    media_bus.SPEC_DESERIALIZERS[name] = from_dict
    DISPATCH[spec.runner_key] = runner
    return spec


def unregister_job(name: str) -> None:
    """Forget a job registered with :func:`register_job` (tests, plugin unload)."""
    from hugpy_video.intel import media_bus
    from hugpy_video.intel.runners import DISPATCH

    spec = JOB_REGISTRY.pop(name, None)
    media_bus.SPEC_DESERIALIZERS.pop(name, None)
    if spec is not None:
        DISPATCH.pop(spec.runner_key, None)


# Registry, not globals. Phase 1/2 registered "crop"; Phase 4 landed
# frame_extract + generate_image; Phase 5 landed audio_extract. Every job's
# spec type must exist as an import above before it can be registered here.
JOB_REGISTRY = {
    "crop": JobSpec("crop", CropSpec, ("ffmpeg", "crop"), "media", 300),  # runner branches spatial/temporal
    "frame_extract": JobSpec("frame_extract", FrameExtractSpec, ("ffmpeg", "frame_extract"), "media", 600),
    "audio_extract": JobSpec("audio_extract", AudioExtractSpec, ("ffmpeg", "audio_extract"), "media", 300),
    "generate_image": JobSpec("generate_image", GenerateImageSpec, ("diffusers", "generate_image"), "gpu", 900),
    "generate_scene": JobSpec("generate_scene", GenerateSceneSpec, ("diffusers", "generate_scene"), "gpu", 3600),
    # Movie = a SEQUENCE of scene segments; the fat orchestrator sequences them
    # inline, so its wall-clock is (segments × per-scene) — a longer timeout.
    "generate_movie": JobSpec("generate_movie", MovieSpec, ("diffusers", "generate_movie"), "gpu", 14400),
    # B2: studio i2v — the first job routed to the studio spine (produce_clip). Its
    # runner_key ("studio","i2v") is DISTINCT from the other frameworks; the runner
    # (runners/studio_i2v.py) resolves a StudioEnv and delegates to produce_clip.
    "studio_i2v": JobSpec("studio_i2v", StudioI2VSpec, ("studio", "i2v"), "gpu", 3600),
    # Studio movie = an ordered strip of studio clips conjoined at splice points (an
    # NLE row). The fat orchestrator (runners/studio_movie.py) renders each segment
    # INLINE through the same produce_clip spine as studio_i2v, so its wall-clock is
    # (segments × per-clip) — a longer timeout, mirroring generate_movie.
    "generate_studio_movie": JobSpec(
        "generate_studio_movie", StudioMovieSpec, ("studio", "movie"), "gpu", 14400),
    # Identity reconstruction (studio stage (b)) = an ORCHESTRATOR that renders one
    # id_lock still per view INLINE through the same render_clip spine as studio_i2v,
    # so its wall-clock is (views × per-clip) — a long timeout, mirroring the movies.
    "identity_reconstruction": JobSpec(
        "identity_reconstruction", IdentityReconstructionSpec,
        ("identity", "reconstruction"), "gpu", 14400),
    # Identity 3D MESH build (+ optional turntable) = a RELAY job: central has no GPU,
    # so its runner (runners/identity_render_relay.py) forwards the spec over HTTP to the
    # IDENTITY_RENDER_URL service and polls it. runner_key ("identity","mesh_build");
    # "gpu" queue (it consumes a remote GPU); long timeout — a mesh + Blender orbit is
    # minutes-to-longer, mirroring the movie/reconstruction budgets.
    "identity_mesh_build": JobSpec(
        "identity_mesh_build", IdentityMeshSpec,
        ("identity", "mesh_build"), "gpu", 14400),
    # Identity VIDEO-EXTRACT (char360) = a RELAY job (like identity_mesh_build): central has
    # no GPU + never runs char360, so its runner (runners/identity_video_extract_relay.py)
    # forwards the source video over HTTP to the IDENTITY_RENDER_URL service, polls it,
    # downloads the per-character view-sets, and writes them back into identity profiles.
    # runner_key ("identity","video_extract"); "gpu" queue (it consumes a remote GPU);
    # long timeout — scene-detect + YOLO track + insightface over a whole clip is
    # minutes-to-longer, mirroring the mesh/movie/reconstruction budgets.
    "identity_video_extract": JobSpec(
        "identity_video_extract", IdentityVideoExtractSpec,
        ("identity", "video_extract"), "gpu", 14400),
    # Identity FROM-VIDEO (k94) = ONE chained RELAY job (char360 + one Hunyuan3D GLB per
    # detected character — the service's ``video_characters_glb`` kind, clownworld's MO).
    # runner_key ("identity","from_video"); "gpu" queue (remote GPU); the longest identity
    # budget — extraction plus minutes-per-character meshing (+ texture bake).
    "identity_from_video": JobSpec(
        "identity_from_video", IdentityFromVideoSpec,
        ("identity", "from_video"), "gpu", 14400),
    # TTS (Chatterbox) = voice vertical I (k98). runner_key/queue/timeout mirror the
    # runner's own RUNNER_KEY/JOB_QUEUE/JOB_TIMEOUT_S constants (read, not retyped).
    "tts_chatterbox": JobSpec("tts_chatterbox", TtsSpec, ("chatterbox", "tts"), "gpu", 1800),
    # video.performance (k106) — the oracle's audio-first orchestrator — is NOT
    # a built-in row any more: the oracle registers it at composition time via
    # ``hugpy_video.jobs.register_job`` (video never imports the oracle).
    # MLT/Kdenlive headless render (k22) = a CPU-only LOCAL subprocess job (melt): the
    # operator's Kdenlive project (saved into the Samba studio share) is path-mapped +
    # rendered server-side, output written back under edits/renders/. runner_key
    # ("mlt","render"); "media" (CPU) queue — NO GPU reservation template; a long timeout
    # since a real NLE timeline can be minutes.
    "mlt_render": JobSpec("mlt_render", MltRenderSpec, ("mlt", "render"), "media", 14400),
    # Studio TESTER = a cross-model SWEEP: one prompt iterated across every servable
    # model of a category's type, one battery row per model. It is a CPU ORCHESTRATOR
    # (each inner generation manages its own GPU via the plane / the studio spine), so
    # it runs on the "media" queue with NO GPU-reservation template — mirroring
    # mlt_render. A full sweep is many renders, so its wall-clock is (models × per-gen)
    # — a very long timeout (24h), the longest budget in the registry.
    "studio_tester": JobSpec(
        "studio_tester", StudioTesterSpec, ("studio", "tester"), "media", 86400),
}
