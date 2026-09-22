"""Sentinel settings: env-driven, injectable for tests.

The systemd unit passes these as Environment= lines; nothing here reads
central's config — the sentinel stays a plain HTTP client.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from hugpy_platform.app_dirs import hugpy_state_dir
from hugpy_platform.central import DEFAULT_CENTRAL, central_base_url

_TRUE = ("1", "true", "yes", "on")


def _as_bool(v: str | None) -> bool:
    return (v or "").strip().lower() in _TRUE


def default_state_dir() -> str:
    """The sentinel's state root: ``<hugpy state dir>/sentinel`` (platform
    owns the base — ``HUGPY_HOME``, default ``~/.hugpy/state``). The systemd
    unit pins ``HUGPY_SENTINEL_DIR`` explicitly; this is the bare default."""
    return os.path.join(hugpy_state_dir(), "sentinel")


@dataclass
class SentinelSettings:
    central: str = DEFAULT_CENTRAL
    # State root: case db, per-case dirs, scorecard/capability history,
    # cases.md ledger all live under here.
    state_dir: str = field(default_factory=default_state_dir)
    # THE remedy gate. Default OFF: the sentinel documents and escalates
    # only. The keeper enables deliberately (HUGPY_SENTINEL_REMEDIES=1)
    # after reviewing documented cases.
    remedies_enabled: bool = False
    # k97 DOWNLOADS gate — deliberately SEPARATE from remedies_enabled and
    # default ON (operator ruling 2026-08-06: "autonomous downloads should
    # be greenlit — especially for ComfyUI"). Enqueueing a download only
    # creates a queued job on central's transfer plane — reversible (cancel),
    # touches no worker — so it does not ride the mutate-a-worker gate.
    # HUGPY_SENTINEL_DOWNLOADS=0 turns it off.
    downloads_enabled: bool = True
    # Detection thresholds.
    stalled_grace_s: float = 120.0      # stalled must persist this long
    hard_fail_streak: int = 3           # consecutive hard_pass=False to case
    # Agent spawn bounds.
    agent_cmd: str = "hugpy-agent"      # PATH lookup; unit pins an absolute
    agent_timeout_s: float = 900.0      # one case run may not outlive this
    agent_max_steps: int = 20
    agent_extra_args: list = field(default_factory=list)
    # k96 pilot light: the model key that should ALWAYS be warm on some worker
    # (the last entry of the agent's brain ladder — the brain a case run falls
    # back to when everything else is cold). Empty = the check is off.
    # Resolved from HUGPY_SENTINEL_PILOT_LIGHT, else derived from the last
    # entry of HUGPY_AGENT_BRAINS (the same env the spawned case agents
    # inherit), so one env line configures both the ladder and its watchdog.
    pilot_light: str = ""

    @property
    def db_path(self) -> str:
        return os.path.join(self.state_dir, "cases.db")

    @property
    def cases_dir(self) -> str:
        return os.path.join(self.state_dir, "cases")

    @property
    def ledger_path(self) -> str:
        return os.path.join(self.state_dir, "cases.md")

    @property
    def history_path(self) -> str:
        return os.path.join(self.state_dir, "history.jsonl")


def load_settings(environ: dict | None = None) -> SentinelSettings:
    env = os.environ if environ is None else environ
    s = SentinelSettings()
    # Explicit sentinel override first, else the canonical resolver
    # (HUGPY_BASE_URL and its legacy aliases, then localhost central).
    explicit = env.get("HUGPY_SENTINEL_CENTRAL")
    s.central = (explicit.rstrip("/") if explicit
                 else central_base_url() if environ is None
                 else DEFAULT_CENTRAL)
    s.state_dir = env.get("HUGPY_SENTINEL_DIR", s.state_dir)
    s.remedies_enabled = _as_bool(env.get("HUGPY_SENTINEL_REMEDIES"))
    # Downloads default ON — only an explicit falsy value turns them off.
    raw_downloads = env.get("HUGPY_SENTINEL_DOWNLOADS")
    s.downloads_enabled = True if raw_downloads is None \
        else _as_bool(raw_downloads)
    if env.get("HUGPY_SENTINEL_STALLED_GRACE_S"):
        s.stalled_grace_s = float(env["HUGPY_SENTINEL_STALLED_GRACE_S"])
    if env.get("HUGPY_SENTINEL_HARD_FAIL_STREAK"):
        s.hard_fail_streak = int(env["HUGPY_SENTINEL_HARD_FAIL_STREAK"])
    s.agent_cmd = env.get("HUGPY_SENTINEL_AGENT", s.agent_cmd)
    if env.get("HUGPY_SENTINEL_AGENT_TIMEOUT_S"):
        s.agent_timeout_s = float(env["HUGPY_SENTINEL_AGENT_TIMEOUT_S"])
    if env.get("HUGPY_SENTINEL_AGENT_MAX_STEPS"):
        s.agent_max_steps = int(env["HUGPY_SENTINEL_AGENT_MAX_STEPS"])
    explicit_pilot = (env.get("HUGPY_SENTINEL_PILOT_LIGHT") or "").strip()
    if explicit_pilot:
        s.pilot_light = explicit_pilot
    else:
        # Derive from the agent ladder the sentinel's case runs inherit: the
        # LAST csv entry is the pilot light by the k96 convention.
        brains = [b.strip() for b in
                  (env.get("HUGPY_AGENT_BRAINS") or "").split(",") if b.strip()]
        if brains:
            s.pilot_light = brains[-1]
    return s
