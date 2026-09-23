"""hugpy-video: execution and artifacts for Hugpy video work.

Owns the media library jail, the sqlite media job bus and its lifecycle
bridge, the frozen job specs and presets, the GPU reservation engine, the
ffmpeg/diffusers/studio/identity runners and the studio spine
(``intel``), the chat-side video analyzer (``chat_video``), the engine
task runner for text-/image-to-video (``video_gen``) and the studio assist
log. Decisions and plans belong to ``hugpy_oracle``, which plugs in through
``hugpy_video.hooks`` (prompt coordination) and ``hugpy_video.jobs``
(``register_job``) — this package never imports the oracle, the fleet or
the server.

Import-light: nothing here pulls torch/ffmpeg/studio stacks. Sub-modules are
imported explicitly by callers (``hugpy_video.jobs``, ``hugpy_video.hooks``,
``hugpy_video.plugin``, ``hugpy_video.state``, ``hugpy_video.config``).

Version is single-sourced from pyproject.toml via importlib.metadata.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hugpy-video")
except PackageNotFoundError:  # running from a checkout without install
    __version__ = "0.0.0+unknown"

from hugpy_video.config import Config, load_config  # stdlib-only, cheap

__all__ = [
    "__version__",
    "Config",
    "load_config",
]
