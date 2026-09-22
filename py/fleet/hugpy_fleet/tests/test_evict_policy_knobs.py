"""The eviction policy knob, on a real switch — and the one that was RETIRED.

Two knobs used to live here (operator, 2026-07-25). One remains:

  1. ANTI-THRASH FLOOR  ``evict_min_residency_s``  — **RETIRED 2026-07-27.**
  2. LEAST REAPING      ``evict_least_reaping``    — FLEET-WIDE.

**Why the floor is gone** (operator: _"is there still some timeblock on a model
being evicted? if so eliminate it"_): it vetoed eviction of any model resident
for less than 300s. That is a clock-driven THIRD protection class, and the
standing ruling (2026-07-23) is that exactly two exist — 🔒static residency and
actively-answering. It also contradicted the minimize-loading doctrine directly:
_"if the answer is a timer, it's wrong by default."_

Section 2 is now the REGRESSION SUITE for that removal: a fresh model must be
evictable, nothing may veto on age, and the settings key must be rejected rather
than silently accepted. Freshness survives as RANK (``sort_key`` orders on
calls, then last_call) — chosen last, never unchoosable.

The scope split for what remains is asserted, not just documented: least-reaping
gates the DROP PASS that central's ``storage_proposal`` runs too, so a per-worker
value would break Parity. ``test_parity_holds_under_the_fleet_switch`` guards it.

Run: venv/bin/python -m pytest tests/test_evict_policy_knobs.py -v
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_engine import eviction as ev
from hugpy_fleet.worker import agent, budget


# The reserve is carved OUT of disk_cache_gib since 2026-08-22 (effective =
# allocation - reserve). These tests reason in pure-allocation numbers, so pin
# the reserve to 0 here; the carve-out itself is covered by
# tests/test_budget_reserve_carveout.py.
@pytest.fixture(autouse=True)
def _zero_disk_reserve(monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_DISK_RESERVE_GIB", "0")


GIB = 1 << 30
NOW = 1_000_000.0

# The retired floor's env name. Kept ONLY so the regression tests can prove that
# setting it has no effect whatsoever.
_ENV_FLOOR = "HUGPY_EVICT_MIN_RESIDENCY_S"
_ENV_REAP = "HUGPY_EVICT_LEAST_REAPING"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """The knob reads the ENV, so every test starts from a known-absent one."""
    monkeypatch.delenv(_ENV_FLOOR, raising=False)
    monkeypatch.delenv(_ENV_REAP, raising=False)
    agent._SETTINGS_SOURCE.clear()
    agent._RUNTIME_SETTINGS.clear()
    yield


# ─────────────────────────────────────────────────────────────────────────────
# 1. THE DROP PASS — the switch must actually change it, and OFF must be
#    byte-identical to the pre-drop-pass greedy walk.
# ─────────────────────────────────────────────────────────────────────────────
# Sized so the drop pass BITES: the walk takes small+big, but big alone covers
# the need, so least-reaping spares the small one. Without a fixture where the
# two answers differ, a broken switch would pass silently.
_DROP_RESIDENTS = [
    ev.EvictUnit(model_key="small", bytes=5 * GIB, last_call=NOW - 9000, calls=0),
    ev.EvictUnit(model_key="big", bytes=35 * GIB, last_call=NOW - 8000, calls=0),
]
_DROP_NEED = 15 * GIB


def test_least_reaping_on_drops_the_covered_victim():
    """ON (today's default): one 35 GiB unload satisfies a 15 GiB need."""
    plan = ev.evict_plan(ev.VRAM, _DROP_NEED, _DROP_RESIDENTS,
                         now=NOW, least_reaping=True)
    assert plan.victims == ["big"]
    assert plan.spared == ["small"]
    assert plan.freed == 35 * GIB


def test_least_reaping_off_keeps_the_whole_walk():
    """OFF: the greedy walk, nothing spared — MORE headroom, more unloads."""
    plan = ev.evict_plan(ev.VRAM, _DROP_NEED, _DROP_RESIDENTS,
                         now=NOW, least_reaping=False)
    assert plan.victims == ["small", "big"]
    assert plan.spared == []
    assert plan.freed == 40 * GIB          # the extra headroom the operator wanted


def test_off_is_byte_identical_to_the_pre_drop_pass_walk():
    """The OFF state must be the OLD behaviour EXACTLY, not merely similar.

    Recomputes the pre-drop-pass algorithm independently (sort by the spec key,
    walk until covered, keep everything walked) and asserts the switch reproduces
    it — victims, order, and freed bytes. This is the regression that would catch
    the switch accidentally changing the WALK or the SORT rather than only
    skipping the drop.
    """
    walkable = sorted(_DROP_RESIDENTS, key=lambda r: ev.sort_key(r, ev.VRAM, NOW))
    expected, freed = [], 0
    for r in walkable:
        if freed >= _DROP_NEED:
            break
        expected.append(r.model_key)
        freed += r.bytes

    plan = ev.evict_plan(ev.VRAM, _DROP_NEED, _DROP_RESIDENTS,
                         now=NOW, least_reaping=False)
    assert plan.victims == expected
    assert plan.freed == freed


def test_default_is_least_reaping_on():
    """Absent argument == today's behaviour. Defaults must not change behaviour."""
    assert ev.DEFAULT_LEAST_REAPING is True
    plan = ev.evict_plan(ev.VRAM, _DROP_NEED, _DROP_RESIDENTS, now=NOW)
    assert plan.victims == ["big"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. THE RETIRED FLOOR — regression suite for its REMOVAL (2026-07-27).
#
#    Each test here fails if the time-based veto comes back in any form:
#    as a parameter, as a module default, as an env, or as a settings key.
# ─────────────────────────────────────────────────────────────────────────────
def _fresh_resident():
    """One model, resident 10s — inside any floor that might be reintroduced."""
    return [ev.EvictUnit(model_key="fresh", bytes=10 * GIB, last_call=None,
                        calls=0, resident_since=NOW - 10)]


def test_a_freshly_loaded_model_is_evictable():
    """THE ruling, asserted directly: no timeblock on a model being evicted.

    A model resident for 10 seconds must be a legal victim. Under the retired
    floor this plan returned no victims and the admission REFUSED.
    """
    plan = ev.evict_plan(ev.VRAM, 5 * GIB, _fresh_resident(), now=NOW)
    assert plan.victims == ["fresh"]
    assert plan.enough
    assert plan.blocking == []


def test_nothing_blocks_on_age():
    """No blocking entry may cite residency time, whatever the resident's age.

    Sweeps ages from 0s to a day. Any age-based veto shows up as a blocking row,
    which is exactly what the operator asked to eliminate.
    """
    for age in (0, 1, 10, 60, 299, 300, 301, 86400):
        rows = [ev.EvictUnit(model_key="m", bytes=10 * GIB, last_call=None,
                            calls=0, resident_since=NOW - age)]
        plan = ev.evict_plan(ev.VRAM, 5 * GIB, rows, now=NOW)
        assert plan.victims == ["m"], f"age={age}s was vetoed"
        assert not any("residency" in b["why"] for b in plan.blocking), age


def test_the_floor_parameter_no_longer_exists():
    """``min_residency_s`` must be gone from the signature, not merely defaulted
    to 0 — a defaulted parameter is a veto one caller away from returning."""
    import inspect
    for fn in (ev.evict_plan, ev.plan_admission, ev.plan_admission_split):
        assert "min_residency_s" not in inspect.signature(fn).parameters, fn.__name__
    with pytest.raises(TypeError):
        ev.evict_plan(ev.VRAM, 5 * GIB, _fresh_resident(),
                      now=NOW, min_residency_s=300.0)


def test_the_module_default_is_gone():
    assert not hasattr(ev, "DEFAULT_MIN_RESIDENCY_S")
    assert not hasattr(agent, "_evict_min_residency_s")


def test_the_retired_env_has_no_effect(monkeypatch):
    """An old systemd drop-in still carrying the env must be INERT, not honoured.

    This is the realistic upgrade path: a box with the drop-in from 2026-07-25
    still on disk. It must not resurrect the veto.
    """
    monkeypatch.setenv(_ENV_FLOOR, "3600")
    plan = ev.evict_plan(ev.VRAM, 5 * GIB, _fresh_resident(), now=NOW)
    assert plan.victims == ["fresh"]


def test_only_two_classes_can_block_eviction():
    """The standing ruling (2026-07-23) as an executable assertion.

    static and in-flight block; a plain idle resident does not — and neither
    does a brand-new one. 'unmeasurable' is not a protection class but a
    degrade-not-guess refusal, so it is asserted separately below.
    """
    rows = [
        ev.EvictUnit(model_key="static", bytes=10 * GIB, calls=0, static=True),
        ev.EvictUnit(model_key="busy", bytes=10 * GIB, calls=0, mid_generation=True),
        ev.EvictUnit(model_key="fresh", bytes=10 * GIB, calls=0,
                    resident_since=NOW - 1),
    ]
    plan = ev.evict_plan(ev.VRAM, 5 * GIB, rows, now=NOW)
    assert plan.victims == ["fresh"]
    blocked = {b["model_key"] for b in plan.blocking}
    assert blocked == {"static", "busy"}


def test_unmeasurable_is_still_not_walked():
    """Degrade-not-guess survives the floor's removal: an occupant we cannot
    size is still never evicted, because the plan could not be verified."""
    rows = [ev.EvictUnit(model_key="unknown", bytes=None, calls=0)]
    plan = ev.evict_plan(ev.VRAM, 5 * GIB, rows, now=NOW)
    assert plan.victims == []
    assert any("unmeasurable" in b["why"] for b in plan.blocking)


def test_least_reaping_reader_env_and_default():
    assert agent._evict_least_reaping() is True
    for falsy in ("0", "false", "FALSE", "no", "off"):
        os.environ[_ENV_REAP] = falsy
        assert agent._evict_least_reaping() is False, falsy
    for truthy in ("1", "true", "yes", "on"):
        os.environ[_ENV_REAP] = truthy
        assert agent._evict_least_reaping() is True, truthy


def test_budget_reader_matches_the_agent_reader():
    """budget.py must parse the shared env EXACTLY as agent.py does — they are
    two readers of one contract, and a divergence would split the fleet's
    storage half from its VRAM half."""
    for val in ("0", "false", "off", "1", "true", ""):
        os.environ[_ENV_REAP] = val
        assert budget._least_reaping() == agent._evict_least_reaping(), val
    del os.environ[_ENV_REAP]
    assert budget._least_reaping() == agent._evict_least_reaping()


# ─────────────────────────────────────────────────────────────────────────────
# 3. SETTINGS ROUND-TRIP + SOURCE PRECEDENCE (settings > env > default).
# ─────────────────────────────────────────────────────────────────────────────
class _Args:
    def __init__(self, path):
        self.settings_file = str(path)


def _apply(tmp_path, settings, monkeypatch, env=None):
    """Write a settings file, project it, and return least_reaping."""
    import json
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(settings))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    # The base-sentinels are captured once per boot; clear them so each call is
    # an independent "boot" rather than inheriting the previous test's base.
    for k in list(os.environ):
        if k.startswith("_HUGPY_BASE_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(agent, "_load_settings", lambda _a: dict(settings))
    agent._apply_settings_env(_Args(p))
    return agent._evict_least_reaping()


def test_settings_roundtrip_through_apply(tmp_path, monkeypatch):
    reap = _apply(tmp_path, {"evict_least_reaping": False}, monkeypatch)
    assert reap is False
    assert agent._SETTINGS_SOURCE["evict_least_reaping"] == "settings"


def test_setting_false_survives_projection(tmp_path, monkeypatch):
    """The escape hatch must survive the projector.

    ``False`` is a legitimately-FALSY projected value. A truthiness guard in
    _apply_settings_env would drop exactly the state the operator needs to
    reach — hence this test.
    """
    reap = _apply(tmp_path, {"evict_least_reaping": False}, monkeypatch)
    assert reap is False
    assert agent._SETTINGS_SOURCE["evict_least_reaping"] == "settings"


def test_settings_beat_the_env_dropin(tmp_path, monkeypatch):
    reap = _apply(tmp_path, {"evict_least_reaping": False},
                  monkeypatch, env={_ENV_REAP: "1"})
    assert reap is False


def test_default_when_neither(tmp_path, monkeypatch):
    reap = _apply(tmp_path, {}, monkeypatch)
    assert reap is ev.DEFAULT_LEAST_REAPING
    assert agent._SETTINGS_SOURCE["evict_least_reaping"] == "default"


def test_retired_floor_setting_is_not_silently_stored(tmp_path, monkeypatch):
    """A settings file still carrying the retired key must not resurrect it.

    Storing-and-ignoring is the exact 'setting you can write, that reads back
    correct, and that silently stops mattering' shape this codebase rejects
    elsewhere. Projection must simply not know the key.
    """
    _apply(tmp_path, {"evict_min_residency_s": 900.0}, monkeypatch)
    assert "evict_min_residency_s" not in agent._SETTINGS_SOURCE
    assert _ENV_FLOOR not in os.environ


def test_neither_eviction_key_is_a_per_worker_setting():
    """RETIRED + FLEET-ONLY: neither key may be accepted by /ops/config.

    ``evict_least_reaping`` is fleet policy (operator ruling 2026-07-25, "yes
    reject it"); ``evict_min_residency_s`` no longer exists at all (2026-07-27).
    """
    assert "evict_min_residency_s" not in agent._SETTINGS_KEYS
    assert "evict_least_reaping" not in agent._SETTINGS_KEYS


def test_unknown_key_is_still_rejected():
    """The strict whitelist must not have been loosened."""
    assert "evict_min_residency" not in agent._SETTINGS_KEYS   # near-miss typo
    assert "totally_made_up" not in agent._SETTINGS_KEYS


def test_retired_key_rejection_says_what_happened_to_it():
    """A removal must ANSWER, not just refuse.

    Same principle as the fleet-only rejection naming the right door: an
    operator (or an old console build) asking for a lever that no longer exists
    gets told it was retired and why, instead of a bare "unsupported" that reads
    like a typo.
    """
    assert "evict_min_residency_s" in agent._RETIRED_SETTINGS
    why = agent._RETIRED_SETTINGS["evict_min_residency_s"]
    assert "static" in why and "answering" in why, (
        "the rejection must name the two classes that DO block eviction")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FLEET ADOPTION — the heartbeat path for the fleet-wide knob.
# ─────────────────────────────────────────────────────────────────────────────
def test_adopt_least_reaping_from_the_heartbeat():
    agent._adopt_least_reaping({"evict_least_reaping": False})
    assert agent._evict_least_reaping() is False
    agent._adopt_least_reaping({"evict_least_reaping": True})
    assert agent._evict_least_reaping() is True


def test_absent_key_does_not_clobber_a_local_dropin(monkeypatch):
    """Central with NO opinion must leave a local drop-in alone — absence means
    'no ruling', never 'force the default'."""
    monkeypatch.setenv(_ENV_REAP, "0")
    monkeypatch.setenv(f"_HUGPY_BASE_{_ENV_REAP}", "0")
    agent._adopt_least_reaping({})                     # no key on the reply
    assert agent._evict_least_reaping() is False       # drop-in survived


def test_adoption_never_raises_on_junk():
    """A heartbeat must never die on policy adoption."""
    for junk in (None, {}, {"evict_least_reaping": object()}):
        agent._adopt_least_reaping(junk)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PARITY — the invariant the scope decision exists to protect.
# ─────────────────────────────────────────────────────────────────────────────
def test_parity_holds_under_the_fleet_switch(monkeypatch):
    """Central's preview and the worker's storage auto-evict must name the SAME
    victims in BOTH switch states.

    This is the assertion that justifies making least-reaping fleet-wide: the
    two sites are driven over ONE fixture with the policy flipped, and the
    victim lists must match each time. If the knob were per-worker, this is the
    test that would fail the moment two boxes disagreed.
    """
    from test_eviction_parity import _central_victims, _worker_victims
    from hugpy_fleet.central import workers as central

    for state in ("1", "0"):
        monkeypatch.setenv(_ENV_REAP, state)
        # Patch the module OBJECT, not a dotted string: the package has a
        # `functions.functions` name collision that breaks string resolution.
        monkeypatch.setattr(central, "_fleet_least_reaping",
                            (lambda s=state: s == "1"))
        assert _worker_victims() == _central_victims(), (
            f"victim sets diverged with least_reaping={state}")


def test_no_site_passes_a_residency_floor():
    """The removal, asserted at every call site rather than assumed.

    If someone later reintroduces a floor at any eviction site, this fails.
    Checks the two storage sites plus the worker's VRAM auto-evict — the one
    that actually carried the 300s veto.
    """
    import inspect
    from hugpy_fleet.central import workers
    for src in (inspect.getsource(budget.fit_plan),
                inspect.getsource(workers.storage_proposal),
                inspect.getsource(agent)):
        assert "min_residency_s=" not in src


# ── the operator's ruling: a fleet key is REJECTED, and says where to go ─────
def test_fleet_only_key_is_rejected_by_ops_config_with_the_right_door():
    """evict_least_reaping must NOT be a per-worker setting.

    Operator ruling 2026-07-25 ("yes reject it... best to have these decisions
    explicit in proof of action"). It was briefly accepted so the relay stayed
    uniform — but it is FLEET policy, so a worker would store it, report it back
    as saved, and have central's heartbeat overwrite it on the next beat: a
    setting you can write, that reads back correct, and that silently stops
    mattering.

    That is the same shape as the max-gpu bug found earlier the SAME DAY, where
    /assign returned "admission: approved" for a value it never persisted. So
    the rejection is not a bare "unsupported" — it names the route that DOES own
    the knob, because the operator asked for a real thing at the wrong door.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hugpy_fleet.worker import agent as A

    assert "evict_least_reaping" not in A._SETTINGS_KEYS, (
        "fleet policy must not be a per-worker setting key")

    # The variable is the declaration — one place, extensible, no toggle.
    assert "evict_least_reaping" in A._FLEET_ONLY_SETTINGS
    where = A._FLEET_ONLY_SETTINGS["evict_least_reaping"]
    assert "/llm/evict-policy" in where, (
        "the rejection must name the route that owns the knob, not just refuse")
