"""The studio's VRAM ceiling must come from the DEVICE, not a constant.

``resolve_studio_env`` took ``max_vram_gb: float = 24.0`` and **neither caller ever
overrode it** — ``runners/studio_i2v.py::run_produce_clip`` is the path on BOTH the
central and the GPU-worker side. So every studio placement decision the fleet has
ever made was measured against a hardcoded 24.0, *including on computron, an 8 GiB
RTX 4060*. The studio believed every box was a 3090.

That number decides whether a pipeline goes whole-on-GPU or streams from the CPU
(``wan_i2v._should_place_whole_on_gpu``), so a wrong ceiling is not cosmetic.

MEASURED 2026-07-27: ``total_vram_bytes()`` on ae returns 25,298,141,184 B =
**23.56 GB**, not 24.0 — the true usable device total. Pinned below, because it is
the reason B1 alone makes ae *slightly* more conservative (23.56 < 24.0) while
making computron *hugely* more honest (8.0 << 24.0). The derived-margin slice is
what unlocks whole-GPU placement; this slice only makes the input true.

Run: venv/bin/python -m pytest tests/test_studio_device_budget.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_video.intel.studio import job as J
from hugpy_video.intel.studio.runners import wan_i2v as W


class _Manifest:
    """Only what _max_vram_gb reads."""
    def __init__(self, snap=()):
        self.env_snapshot = tuple(snap)


def _budget(monkeypatch, detected, snap=()):
    """The RUNNER's placement budget with the device probe stubbed."""
    monkeypatch.setattr(W, "total_vram_bytes",
                        lambda *_a, **_k: (None if detected is None
                                           else int(detected * 1024 ** 3)),
                        raising=False)
    import hugpy_platform.hardware as HW
    monkeypatch.setattr(HW, "total_vram_bytes",
                        lambda *_a, **_k: (None if detected is None
                                           else int(detected * 1024 ** 3)))
    return W._max_vram_gb(_Manifest(snap))


# ── ADDRESSING MUST NOT MOVE ────────────────────────────────────────────────
def test_the_snapshot_value_is_a_stable_constant():
    """THE TRAP THIS SLICE ALMOST FELL INTO.

    StudioEnv.to_snapshot() emits STUDIO_MAX_VRAM_GB, and schemas.py puts
    env_snapshot INSIDE the clip content_hash. A device-derived SNAPSHOT would
    re-address every clip the moment two boxes disagree (ae 23.56 vs computron
    8.0) — an identical spec would re-render instead of resuming. The placement
    decision reads the card; the manifest must not.
    """
    assert J.resolve_studio_env("/tmp/x", master_fps=24).max_vram_gb == 24.0
    assert J._FALLBACK_MAX_VRAM_GB == 24.0


def test_two_boxes_produce_the_same_env_snapshot():
    """Directly: the snapshot cannot vary by card, or addressing forks per box."""
    a = J.resolve_studio_env("/tmp/x", master_fps=24).to_snapshot()
    b = J.resolve_studio_env("/tmp/x", master_fps=24).to_snapshot()
    assert dict(a).get("STUDIO_MAX_VRAM_GB") == dict(b).get("STUDIO_MAX_VRAM_GB")


# ── BUT THE PLACEMENT DECISION MUST SEE THE REAL CARD ───────────────────────
def test_the_runner_reads_the_live_device(monkeypatch):
    """A 3090 must give the placement decision ~23.56, not the 24.0 constant."""
    assert abs(_budget(monkeypatch, 23.56072998046875) - 23.56072998046875) < 1e-6


def test_an_8gb_card_is_no_longer_told_it_is_a_3090(monkeypatch):
    """THE BUG THIS SLICE EXISTS FOR. computron is an 8 GiB 4060 and every
    placement call was handed 24.0 — a 3x over-statement of the card."""
    assert abs(_budget(monkeypatch, 8.0) - 8.0) < 1e-6


def test_the_device_beats_the_snapshot(monkeypatch):
    """A stale/foreign snapshot value must not override the card in front of us."""
    got = _budget(monkeypatch, 8.0, snap=(("STUDIO_MAX_VRAM_GB", "24.0"),))
    assert abs(got - 8.0) < 1e-6


def test_unmeasurable_falls_back_to_the_snapshot_not_to_zero(monkeypatch):
    """Degrade to the recorded value, NEVER to 0 — total_vram_bytes returns None
    (not 0) so "unmeasurable" stays distinguishable from "no headroom".
    Collapsing to 0 would make every placement decision refuse."""
    got = _budget(monkeypatch, None, snap=(("STUDIO_MAX_VRAM_GB", "24.0"),))
    # Round-2 review (2026-07-27): a DEAF PROBE FAILS CLOSED — the declared
    # 24.0 is almost never a declaration (resolve_studio_env's default), so an
    # unmeasurable card yields None (conservative offload), never 24.0 and
    # never 0. See wan_i2v._max_vram_gb.
    assert got is None


def test_unmeasurable_and_unrecorded_is_None_not_zero(monkeypatch):
    """None keeps the conservative offload branch; 0 would be a lie."""
    monkeypatch.delenv("STUDIO_MAX_VRAM_GB", raising=False)
    assert _budget(monkeypatch, None) is None


def test_a_probe_explosion_never_fails_a_render(monkeypatch):
    """A driver hiccup must degrade, not raise into the render path."""
    import hugpy_platform.hardware as HW
    def _boom(*_a, **_k):
        raise RuntimeError("driver gone")
    monkeypatch.setattr(HW, "total_vram_bytes", _boom)
    got = W._max_vram_gb(_Manifest((("STUDIO_MAX_VRAM_GB", "24.0"),)))
    assert got is None      # degraded (fail-closed), not raised — see above
