"""Studio runners — the executors a ModelBinding dispatches to (INV-8).

Each ``RunnerSpec.entrypoint`` in the zoo is a dotted path into this subpackage.
Wired today: synthetic (P0-B1), wan_i2v/wan_t2v/wan_vace (P0-6/B-3), ltx_upscale,
rife_interpolate, ffmpeg_enhance (slice b) — see ``produce.py``'s ``_DISPATCH``.
Several other model bets are declared in the registry but their runner module has
not landed (hunyuan, cog, mochi, opensora, skyreels, animatediff, framepack,
codeformer, and LTX's t2v/i2v/av — only ``ltx_upscale`` is wired). Those stay
INTENTIONALLY UNWIRED until built: ``validate_registry()`` only checks a runner is
*registered* (structural totality), not importable, and the k1 servability gate
(``registry.runner_available`` / ``runner_gate_reason``) keeps the router from
binding to one of them — a not-yet-built runner module is a routing/dispatch
concern, not an import-time failure.

Importing this package is cheap: it does NOT eagerly import the runner modules
(which pull PIL/numpy/torch). Callers import a concrete runner by name, e.g.
``from ...studio.runners.synthetic import run_synthetic_i2v``.
"""

from __future__ import annotations
