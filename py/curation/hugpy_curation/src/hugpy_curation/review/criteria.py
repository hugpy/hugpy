"""review/criteria.py — what a model is being reviewed *for*.

A ReviewCriteria is a saved, re-runnable question: "find me text-generation
models that fit the 3090 at 16k context, are a real capability step over what
I already run, and actually generate at a usable speed." It drives both the
cheap metadata screen (screen.py) and, for survivors, the load test (smoke.py).

Deliberately plain dataclasses + dicts: this module is imported by the Flask
app, the CLI and a systemd timer, and must never drag in pydantic version
skew (see _compat_pydantic.py for why that hurts here).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

# One 24 GB card on `ae` today. Overridable so the same criteria file can be
# evaluated for a different box in the fleet without editing it.
DEFAULT_VRAM_BYTES = int(os.environ.get("REVIEW_VRAM_BYTES") or 24 * 1024**3)

# Leave the GPU room for the desktop/compositor, CUDA context and fragmentation.
# A model that "just fits" at 100% never actually loads.
VRAM_HEADROOM_FRACTION = 0.90

# k120: how deep a card's trial goes, cheapest first. Each is a SUPERSET of the
# one before it. Duplicated deliberately from
# ``discovery_dossier.dossier.TRIAL_DEPTHS`` rather than imported: this module
# is loaded by the Flask app, the CLI and a systemd timer and must not pull in
# the dossier package (which reaches the oracle, which reaches the fleet) just
# to validate a string. A test asserts the two lists are identical, so the
# duplication cannot drift.
TRIAL_DEPTHS = ("screen-only", "load-test", "full-samples")


@dataclass
class ReviewCriteria:
    """The question. `name` identifies it in the store and on the timer."""

    name: str
    query: str = ""                          # HF search text
    task: str | None = "text-generation"     # pipeline_tag filter
    library: str | None = None               # HF `filter` (e.g. "gguf")
    author: str | None = None

    # ── fit / runtime cost ────────────────────────────────────────────────
    vram_bytes: int = DEFAULT_VRAM_BYTES
    target_context: int = 16384              # context to size the KV cache for
    min_context: int = 8192                  # reject models that can't reach it
    max_total_bytes: int | None = None       # hard disk cap for one download
    require_gguf: bool = True                # llama.cpp is the fleet's runtime
    allowed_quants: list[str] = field(default_factory=lambda: [
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "IQ4_XS"])
    min_tokens_per_sec: float = 8.0          # smoke-test floor for "usable"

    # ── capability / quality ──────────────────────────────────────────────
    min_downloads: int = 500
    min_trust_tier: int = 0                  # 0 any, 1 community+, 2 first-party
    required_tags: list[str] = field(default_factory=list)
    excluded_tags: list[str] = field(default_factory=list)
    max_age_days: int | None = None          # only recently-updated repos
    min_params: int | None = None            # reject toys
    max_params: int | None = None
    # Models already in the fleet. A candidate that is a fine-tune of one of
    # these is flagged as lineage — usually the interesting kind of candidate.
    incumbents: list[str] = field(default_factory=list)

    # ── pipeline behaviour ────────────────────────────────────────────────
    # `enabled` (2026-08-13) is the console's off switch: pipeline.run()
    # no-ops a disabled criteria unless forced, so the nightly timer can keep
    # firing while the operator flips discovery on/off per-criteria from the
    # settings UI (no systemctl, no root). Pre-existing files lack the field
    # and read as True — nothing changes until someone turns it off.
    enabled: bool = True
    pool_limit: int = 60                     # candidates pulled from HF search
    max_downloads_per_run: int = 2           # disk/bandwidth guard for the timer
    smoke_test: bool = True
    judge: bool = True                       # ask a hugpy-agent for a read

    # ── k120: how much DOSSIER this card wants ────────────────────────────
    # Every knob below defaults to the pre-k120 behaviour of an existing card,
    # so a criteria file written in July screens, downloads and load-tests
    # EXACTLY as it did before and simply gains the sections that cost nothing.
    # The operator's ask was that "complexity to be dictated" per question —
    # a cheap nightly sweep and a deep look at one family are the same machine
    # with different numbers in it.
    dossier: bool = True                     # build a ModelDossier at all
    # screen-only | load-test | full-samples. `load-test` is what the reviewer
    # already did (download + llama.cpp load); `full-samples` adds k109b's
    # stationary battery and the comparison against the routing matrix's
    # incumbent. Deliberately NOT the default: it costs GPU time, and a card
    # that never asked for it must not silently start spending it.
    trial_depth: str = "load-test"
    sample_count: int = 2                    # stationary points per candidate
    # Routing-matrix operation names the candidate is compared against, e.g.
    # ["plot.construct"]. Empty = compare on whatever the battery sampled.
    compare_against: list[str] = field(default_factory=list)
    # Screen knobs, answered from tags + repo name with no extra fetch
    # (discovery_dossier/screening.py). Empty = no rule, as before.
    required_specializations: list[str] = field(default_factory=list)
    licenses_allowed: list[str] = field(default_factory=list)
    # The two network-facing sections, each switchable per card.
    external_research: bool = True           # card README + linked papers
    community: bool = True                   # reddit / HN / HF discussions
    community_sources: list[str] = field(default_factory=list)   # [] = defaults
    subreddits: list[str] = field(default_factory=list)          # [] = defaults
    # GEM RADAR: a second pass over the SAME cached community pulls looking for
    # models no card is asking about. Off by default — it publishes tips, and a
    # tip nobody asked for is noise.
    radar: bool = False

    def __post_init__(self) -> None:
        # Fail at LOAD, not mid-screen — the same rule the numeric coercion in
        # from_dict follows. A card asking for "full_samples" (underscore) or
        # "deep" would otherwise silently fall through to screen-only and the
        # operator would spend a week wondering why no samples ever ran.
        if self.trial_depth not in TRIAL_DEPTHS:
            raise ValueError(
                f"ReviewCriteria({self.name!r}).trial_depth "
                f"{self.trial_depth!r} is not one of {list(TRIAL_DEPTHS)}")
        if int(self.sample_count) < 0:
            raise ValueError(
                f"ReviewCriteria({self.name!r}).sample_count must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReviewCriteria":
        known = {f for f in cls.__dataclass_fields__}          # ignore extras
        vals = {k: v for k, v in (d or {}).items() if k in known}
        # `criteria set --set max_age_days=120` stores the raw CLI string; a
        # str landing in an int field detonated EVERY screen comparison
        # ("'>' not supported between int and str" — 60/60 candidates
        # rejected, 2026-08-06). Coerce scalars back to the field's declared
        # numeric/bool type; a value that won't parse raises here, at load,
        # instead of mid-screen.
        for k, v in vals.items():
            if not isinstance(v, str):
                continue
            decl = str(cls.__dataclass_fields__[k].type)
            if "bool" in decl:
                vals[k] = v.strip().lower() in ("1", "true", "yes", "on")
            elif "int" in decl:
                vals[k] = int(v)
            elif "float" in decl:
                vals[k] = float(v)
        return cls(**vals)

    @property
    def usable_vram_bytes(self) -> int:
        return int(self.vram_bytes * VRAM_HEADROOM_FRACTION)


def criteria_dir() -> str:
    """``REVIEW_CRITERIA_DIR``, else ``<config_dir>/review`` — see
    :mod:`hugpy_curation.config`."""
    from hugpy_curation.config import criteria_dir as _criteria_dir
    d = _criteria_dir()
    os.makedirs(d, exist_ok=True)
    return d


def save_criteria(c: ReviewCriteria) -> str:
    path = os.path.join(criteria_dir(), f"{c.name}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(c.to_dict(), fh, indent=2)
    os.replace(tmp, path)                       # atomic: the timer may be reading
    return path


def load_criteria(name: str) -> ReviewCriteria:
    path = os.path.join(criteria_dir(), f"{name}.json")
    with open(path, "r", encoding="utf-8") as fh:
        return ReviewCriteria.from_dict(json.load(fh))


def list_criteria() -> list[str]:
    try:
        return sorted(f[:-5] for f in os.listdir(criteria_dir())
                      if f.endswith(".json"))
    except OSError:
        return []
