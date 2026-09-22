"""Runner dispatch table — map §6/§8: (framework, task) -> pure runner fn.

Registries over globals. The worker looks up JOB_REGISTRY[name].runner_key and
calls DISPATCH[runner_key](spec, job_id). Runners are pure `spec -> JobResult`
(they also take the job_id so the JobResult can carry it).

Phase 4 landed frame_extract + generate_image; Phase 5 landed audio_extract;
generate_scene (one query -> N consecutive frames + optional mp4) is wired below.
All runner keys are wired below:
    ("ffmpeg", "crop"): run_crop,
    ("ffmpeg", "frame_extract"): run_frame_extract,
    ("ffmpeg", "audio_extract"): run_audio_extract,
    ("diffusers", "generate_image"): run_generate_image,
    ("diffusers", "generate_scene"): run_generate_scene,
"""
from __future__ import annotations

from hugpy_video.intel.runners.ffmpeg_audio import run_audio_extract
from hugpy_video.intel.runners.ffmpeg_crop import run_crop
from hugpy_video.intel.runners.ffmpeg_frames import run_frame_extract
from hugpy_video.intel.runners.imagegen import run_generate_image
from hugpy_video.intel.runners.movie import run_generate_movie
from hugpy_video.intel.runners.scene import run_generate_scene
# B2: studio i2v — the media bus's seam to the studio spine (produce_clip). Its
# module top is dependency-light (studio/numpy imports are lazy inside the runner),
# so this import can never break app boot.
from hugpy_video.intel.runners.studio_i2v import run_studio_i2v
# Studio movie — the fat orchestrator that renders an ordered strip of studio clips
# INLINE through the produce_clip spine. Import-safe like studio_i2v (studio/numpy
# imports stay lazy inside the runner), so this never breaks app boot.
from hugpy_video.intel.runners.studio_movie import run_generate_studio_movie
# Studio TESTER — the cross-model sweep runner. Import-safe like the studio runners
# (plane/studio-spine imports stay lazy inside studio.tester), so this top-level
# import can never break app boot.
from hugpy_video.intel.runners.studio_tester import run_studio_tester
# Identity reconstruction (studio stage (b)) — the orchestrator that renders an
# identity-locked turnaround set from a profile + description. Import-safe like the
# studio runners (studio/media_store imports stay lazy inside the runner).
from hugpy_video.intel.runners.identity_reconstruction import run_identity_reconstruction
# Identity 3D MESH build (+ turntable) — a RELAY to a remote GPU render service (central
# has no GPU). Import-safe like the other identity runners: requests + the store imports
# stay lazy INSIDE the runner, so this top-level import can never break app boot.
from hugpy_video.intel.runners.identity_render_relay import run_identity_mesh_build
# Identity VIDEO-EXTRACT (char360) — a RELAY to the SAME remote GPU render service (central
# has no GPU and NEVER runs char360/cv2/insightface). Import-safe like the mesh relay:
# requests + the store imports stay lazy INSIDE the runner, so this top-level import can
# never break app boot AND never pulls a char360 dependency onto the central side.
from hugpy_video.intel.runners.identity_video_extract_relay import run_identity_video_extract
# Identity FROM-VIDEO (k94) — ONE chained char360 + Hunyuan3D GLB relay to the SAME remote
# render service (clownworld's ``video_characters_glb`` MO). Import-safe like the other
# relays: requests + the store imports stay lazy INSIDE the runner.
from hugpy_video.intel.runners.identity_from_video import run_identity_from_video
# TTS (Chatterbox) — voice vertical I (k98). Import-safe like the relays: torch/
# chatterbox/torchaudio imports stay lazy INSIDE the runner, so this top-level
# import can never break app boot on a worker without the backend installed.
from hugpy_video.intel.runners.tts_chatterbox import run_tts_chatterbox
# MLT/Kdenlive headless render (k22) — a CPU-only LOCAL subprocess runner (melt). Import-safe:
# its module top is pure stdlib (subprocess/xml/re) so this boot-time import can never break
# app boot, and it pulls no GPU/char360 dependency onto the central side.
from hugpy_video.intel.runners.mlt_render import run_mlt_render
# video.performance (k106) — the oracle's audio-first orchestrator — plugs in
# at composition time through ``hugpy_video.jobs.register_job`` (which fills
# this table); video never imports the oracle.

DISPATCH = {
    ("ffmpeg", "crop"): run_crop,
    ("ffmpeg", "frame_extract"): run_frame_extract,
    ("ffmpeg", "audio_extract"): run_audio_extract,
    ("diffusers", "generate_image"): run_generate_image,
    ("diffusers", "generate_scene"): run_generate_scene,
    ("diffusers", "generate_movie"): run_generate_movie,
    ("studio", "i2v"): run_studio_i2v,
    ("studio", "movie"): run_generate_studio_movie,
    ("studio", "tester"): run_studio_tester,
    ("identity", "reconstruction"): run_identity_reconstruction,
    ("identity", "mesh_build"): run_identity_mesh_build,
    ("identity", "video_extract"): run_identity_video_extract,
    ("identity", "from_video"): run_identity_from_video,
    ("chatterbox", "tts"): run_tts_chatterbox,
    ("mlt", "render"): run_mlt_render,
}
