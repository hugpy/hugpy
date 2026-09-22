"""CASE A auto-block + CASE B manifest-orphan surfacing (operator ruling 2026-07-25).

  "models that aren't in the manifest and models that simply will not fit on a
   worker no matter what, if allocated, should be blocked. the user will be
   forced to acknowledge it and act or not"

The approved shape treats the two as DIFFERENT problems:

CASE A — fits NO worker in ANY mode -> AUTO-BLOCK (arithmetic, not judgement),
reversible via the existing /unblock. Covered here:
  * alloc_modes.worker_fit_verdict — three-valued (True/False/None);
  * alloc_modes.fleet_fit_verdict — blockable IFF >=1 confident refusal and
    ZERO "fits"; never on missing data; never when ONE worker can hold it;
  * blocklist.auto_block — by="auto", declines on an operator unblock and on an
    operator-authored block; refreshes its own record;
  * blocklist.unblock stickiness — an operator unblock leaves the
    operator_unblocked tombstone; an auto unblock deletes without one;
  * workers.fleet_fit_for_model / maybe_auto_block — online-only, degrade-safe;
  * the /assign hook — an infeasible model is auto-blocked and refused with the
    machine's own reasoning (blocked_by:"auto"), and an operator unblock is NOT
    undone by a subsequent /assign.

CASE B — NOT in central's manifest -> a FAULT, never a block (/block 404s on a
non-manifest key by design). Covered here:
  * the /assign clear path: an empty spill on an ALREADY-DESIGNATED orphan key
    clears the row instead of 404ing, while a genuinely unknown key still 404s
    and a NON-empty spill on an orphan key still 404s.

(The ``bin/hugpy-fleet-triage`` read-only collector that surfaced the orphan
FAULT is not part of this workspace — those checks lived with the script.)

The blocklist persists in the F4 settings store; ``HUGPY_SETTINGS_PATH`` is
pointed at a tmp file per test so nothing touches the operator's settings.json.
The /assign route reaches the REAL module-level assign_model(), so the worker
registry is swapped for a tmpdir-backed store (see worker_store_isolation.py).
"""
import importlib

import pytest
from flask import Flask

from worker_store_isolation import swap_worker_store

GIB = 2 ** 30
BIG, SMALL = 68 * GIB, 4 * GIB

am = importlib.import_module("hugpy_engine.alloc_modes")
bl = importlib.import_module("hugpy_fleet.central.blocklist")
W = importlib.import_module("hugpy_fleet.central.workers")
wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
cr = importlib.import_module("hugpy_server.app.routes.comms_routes")


def _reset():
    for k in list(bl.blocked_keys()):
        bl.unblock(k, by="auto")          # auto-unblock leaves no tombstone
    # also clear any tombstones a prior test left
    for k in list((bl.settings_store.all(bl.NS) or {}).keys()):
        bl.settings_store.delete(bl.NS, k)


@pytest.fixture
def blocklist(tmp_path, monkeypatch):
    """An empty blocklist backed by a tmp settings.json (path is read at call
    time; the TTL cache is dropped so no stale read leaks across paths)."""
    monkeypatch.setenv("HUGPY_SETTINGS_PATH", str(tmp_path / "settings.json"))
    bl.settings_store._cache = None
    _reset()
    yield bl
    _reset()
    bl.settings_store._cache = None


# ── 1) worker_fit_verdict: three-valued, loose-by-design ─────────────────────

def test_fit_verdict_fits_gpu_outright():
    assert am.worker_fit_verdict("gguf", 10 * GIB, 24 * GIB, 15 * GIB) is True


def test_fit_verdict_too_big_for_gpu_but_fits_ram():
    assert am.worker_fit_verdict("gguf", 14 * GIB, 8 * GIB, 15 * GIB) is True


def test_fit_verdict_combined_gpu_plus_ram_is_the_real_ceiling():
    """max-gpu/max-ram spill across both; combined is the real physical ceiling."""
    assert am.worker_fit_verdict("gguf", 30 * GIB, 24 * GIB, 15 * GIB) is True


def test_fit_verdict_exceeds_everything_is_confident_refusal():
    assert am.worker_fit_verdict("gguf", 68 * GIB, 24 * GIB, 15 * GIB) is False


def test_fit_verdict_exactly_combined_capacity_is_true():
    """<=, never a fencepost block."""
    assert am.worker_fit_verdict("gguf", 39 * GIB, 24 * GIB, 15 * GIB) is True


def test_fit_verdict_no_headroom_factor_applied():
    """A fudge is right for picking a default that must succeed, wrong for taking
    a model OUT of the pool."""
    assert am.worker_fit_verdict("gguf", int(0.99 * 24 * GIB), 24 * GIB, None) is True


def test_fit_verdict_transformers_gets_same_combined_ceiling():
    assert am.worker_fit_verdict("transformers", 30 * GIB, 24 * GIB, 15 * GIB) is True


def test_fit_verdict_degrade_not_guess():
    # unknown model size -> None (no vote)
    assert am.worker_fit_verdict("gguf", None, 24 * GIB, 15 * GIB) is None
    # zero/garbage model size -> None
    assert am.worker_fit_verdict("gguf", 0, 24 * GIB, 15 * GIB) is None
    assert am.worker_fit_verdict("gguf", "big", 24 * GIB, 15 * GIB) is None
    # BOTH totals unknown -> None (unmeasured box has no opinion)
    assert am.worker_fit_verdict("gguf", 68 * GIB, None, None) is None


def test_fit_verdict_one_known_total_is_enough_to_vote():
    assert am.worker_fit_verdict("gguf", 68 * GIB, 24 * GIB, None) is False
    assert am.worker_fit_verdict("gguf", 10 * GIB, 24 * GIB, None) is True


# ── 2) fleet_fit_verdict: the roll-up, and the asymmetry that keeps it safe ──

def _box(name, gpu, ram, size=BIG, engine="gguf"):
    return {"name": name, "engine": engine, "model_bytes": size,
            "gpu_total_bytes": gpu, "ram_total_bytes": ram}


def test_fleet_verdict_fits_no_worker_is_blockable():
    v = am.fleet_fit_verdict([_box("ae", 24 * GIB, 15 * GIB),
                              _box("op", None, 30 * GIB),
                              _box("computron", 8 * GIB, 16 * GIB)])
    assert v["blockable"] is True
    assert v["fits_somewhere"] is False
    assert set(v["refused_by"]) == {"ae", "op", "computron"}
    # the why is the operator-facing reasoning (size, biggest GPU/RAM, escape hatch)
    assert "68.0 GiB exceeds every worker" in v["why"]
    assert "24.0" in v["why"] and "unblock to override" in v["why"]
    assert v["why"].startswith("auto:")


def test_fleet_verdict_one_worker_that_fits_is_never_blockable():
    """'no matter what' means the whole fleet, not one box."""
    v = am.fleet_fit_verdict([_box("ae", 24 * GIB, 15 * GIB),
                              _box("bigbox", 80 * GIB, 256 * GIB)])
    assert v["blockable"] is False and v["fits_somewhere"] is True
    assert v["fits_on"] == ["bigbox"] and v["refused_by"] == ["ae"]


def test_fleet_verdict_all_unknown_is_not_blockable():
    v = am.fleet_fit_verdict([_box("ae", None, None), _box("op", None, None)])
    assert v["blockable"] is False and v["fits_somewhere"] is None
    assert set(v["unknown"]) == {"ae", "op"}
    assert "degrade-not-guess" in v["why"]


def test_fleet_verdict_one_refusal_plus_one_unknown_still_blockable():
    v = am.fleet_fit_verdict([_box("ae", 24 * GIB, 15 * GIB), _box("op", None, None)])
    assert v["blockable"] is True
    assert "1 worker(s) had no data" in v["why"]


def test_fleet_verdict_unsizable_model_nobody_votes():
    v = am.fleet_fit_verdict([_box("ae", 24 * GIB, 15 * GIB, size=None)])
    assert v["blockable"] is False and v["fits_somewhere"] is None


def test_fleet_verdict_empty_or_junk_fleet_never_blocks():
    assert am.fleet_fit_verdict([])["blockable"] is False
    assert am.fleet_fit_verdict(None)["blockable"] is False
    assert am.fleet_fit_verdict(["nope", None, 7])["blockable"] is False


# ── 3) blocklist: auto authorship + operator-unblock STICKINESS ──────────────

def test_auto_block_is_machine_authored(blocklist):
    rec = bl.auto_block("A~1", "auto: 68.0 GiB exceeds every worker — unblock to override")
    assert rec is not None and rec["by"] == "auto" and rec["blocked"] is True
    assert "unblock to override" in rec["note"]
    assert bl.is_blocked("A~1") is True
    # re-stamping its OWN record is allowed (refreshes numbers)
    assert (bl.auto_block("A~1", "auto: refreshed") or {}).get("note") == "auto: refreshed"


def test_operator_unblock_is_sticky_against_auto_block(blocklist):
    bl.auto_block("A~1", "auto: too big")
    assert bl.operator_unblocked("A~1") is False
    assert bl.unblock("A~1", by="operator") is True
    assert bl.is_blocked("A~1") is False
    # leaves the sticky operator_unblocked marker, which is NOT in blocked_keys
    assert bl.operator_unblocked("A~1") is True
    assert "A~1" not in bl.blocked_keys()
    # THE POINT — auto_block DECLINES a key the operator released
    assert bl.auto_block("A~1", "auto: still too big") is None
    assert bl.is_blocked("A~1") is False
    # an OPERATOR block still works on that key (only the MACHINE is suppressed)
    assert bl.block("A~1", by="operator")["by"] == "operator"
    assert bl.is_blocked("A~1") is True
    # an operator block CLEARS the tombstone (whole-record write)
    assert bl.operator_unblocked("A~1") is False


def test_auto_block_declines_to_overwrite_operator_block(blocklist):
    bl.block("A~2", by="operator", note="operator's own reason")
    assert bl.auto_block("A~2", "auto: whatever") is None
    assert (bl.block_info("A~2") or {}).get("note") == "operator's own reason"


def test_auto_unblock_leaves_no_tombstone(blocklist):
    bl.auto_block("A~3", "auto: too big")
    assert bl.unblock("A~3", by="auto") is True
    assert bl.operator_unblocked("A~3") is False
    # so a LATER, genuinely-correct auto-block still lands
    assert bl.auto_block("A~3", "auto: too big again") is not None


# ── 4) workers.py glue: online-only, degrade-safe ────────────────────────────

ROWS = [{"name": "ae", "status": "online", "gpu_total": 24 * GIB, "ram_total": 15 * GIB},
        {"name": "op", "status": "online", "gpu_total": None, "ram_total": 30 * GIB}]
ROWS_WITH_BIGBOX = ROWS + [{"name": "bigbox", "status": "online",
                            "gpu_total": 80 * GIB, "ram_total": 256 * GIB}]


@pytest.fixture
def glue(monkeypatch):
    monkeypatch.setattr(W, "_model_engine", lambda mk: "gguf")
    monkeypatch.setattr(W, "_model_size_bytes", lambda mk: BIG)
    return monkeypatch


def test_glue_blocks_fleet_wide_misfit(glue):
    r = W.fleet_fit_for_model("M~big", ROWS)
    assert r["blockable"] is True and set(r["refused_by"]) == {"ae", "op"}


def test_glue_one_box_that_can_hold_it_is_not_blockable(glue):
    assert W.fleet_fit_for_model("M~big", ROWS_WITH_BIGBOX)["blockable"] is False


def test_glue_offline_box_is_not_a_candidate(glue):
    """A rebooting box must not vote a model out (neither refuses nor rescues)."""
    rows3 = [dict(ROWS[0]), {"name": "bigbox", "status": "offline",
                             "gpu_total": 80 * GIB, "ram_total": 256 * GIB}]
    r3 = W.fleet_fit_for_model("M~big", rows3)
    assert "bigbox" not in r3["refused_by"]
    assert "bigbox" not in r3["fits_on"]
    assert "bigbox" not in r3["unknown"]
    # an ALL-offline fleet is unblockable
    assert W.fleet_fit_for_model("M~big", [rows3[1]])["blockable"] is False


def test_glue_unsizable_model_not_blockable(glue):
    glue.setattr(W, "_model_size_bytes", lambda mk: None)
    assert W.fleet_fit_for_model("M~big", ROWS)["blockable"] is False


def test_glue_exception_never_manufactures_a_block(glue):
    def _boom(mk):
        raise RuntimeError("manifest exploded")
    glue.setattr(W, "_model_size_bytes", _boom)
    assert W.fleet_fit_for_model("M~big", ROWS)["blockable"] is False


def test_maybe_auto_block_composes_the_two(glue, blocklist):
    assert W.maybe_auto_block("M~big", ROWS) is not None
    assert bl.is_blocked("M~big") is True
    _reset()
    assert W.maybe_auto_block("M~big", ROWS_WITH_BIGBOX) is None
    assert bl.is_blocked("M~big") is False


def test_maybe_auto_block_honors_operator_unblock_tombstone(glue, blocklist):
    bl.auto_block("M~big", "auto: prior")
    bl.unblock("M~big", by="operator")
    assert W.maybe_auto_block("M~big", ROWS) is None
    assert bl.is_blocked("M~big") is False


# ── 5) the /assign hook + the CASE-B clear path ──────────────────────────────

@pytest.fixture
def assign_client(monkeypatch, blocklist):
    """worker_bp on a bare app, gate helpers stubbed so the ONLY variable is the
    fit arithmetic; 'ae' (w1) designates an ORPHAN key (not in the stubbed
    manifest) with a spill — exactly the live ae rows."""
    monkeypatch.setattr(wr, "get_models_dict",
                        lambda dict_return=False: {"M~big": {}, "M~ok": {}})
    monkeypatch.setattr(cr, "audit", lambda *a, **k: None)
    monkeypatch.setattr(wr, "_central_missing_reason", lambda mk: None)
    monkeypatch.setattr(wr, "_disk_preflight_reason", lambda w, mk: None)
    monkeypatch.setattr(W, "_model_engine", lambda mk: "gguf")
    monkeypatch.setattr(W, "_model_size_bytes",
                        lambda mk: BIG if mk == "M~big" else SMALL)
    monkeypatch.setattr(W, "list_workers", lambda: [
        {"name": "ae", "status": "online", "gpu_total": 24 * GIB,
         "ram_total": 15 * GIB}])
    with swap_worker_store() as store:
        store.register(worker_id="w1", name="ae", url="http://x")
        W.assign_model("w1", "orphan~row", spill={"n_gpu_layers": 5})
        monkeypatch.setattr(wr, "get_worker", lambda wid: W.worker_store._load().get(wid))
        app = Flask(__name__)
        app.register_blueprint(wr.worker_bp)
        yield app.test_client(), store


def test_route_assign_auto_blocks_infeasible_model(assign_client):
    """CASE A at the route: the operator's 'if allocated' trigger."""
    client, _ = assign_client
    ra = client.post("/llm/workers/w1/assign", json={"model_key": "M~big"})
    rj = ra.get_json() or {}
    assert bl.is_blocked("M~big") is True
    assert ra.status_code == 409
    # HONEST about authorship (blocked_by 'auto', never 'by the operator')
    assert rj.get("blocked_by") == "auto"
    assert "by the operator" not in rj.get("error", "")
    # carries the machine's own numbers so the operator can act
    assert "exceeds every worker" in rj.get("error", "")
    assert "unblock to override" in rj.get("error", "")
    assert (bl.block_info("M~big") or {}).get("by") == "auto"


def test_route_reassign_does_not_undo_operator_unblock(assign_client):
    """THE STICKINESS PROOF — otherwise the unblock button reads as broken."""
    client, _ = assign_client
    client.post("/llm/workers/w1/assign", json={"model_key": "M~big"})
    assert bl.is_blocked("M~big") is True
    ru = client.post("/llm/models/M~big/unblock", json={})
    assert ru.status_code == 200 and bl.is_blocked("M~big") is False
    ra2 = client.post("/llm/workers/w1/assign", json={"model_key": "M~big"})
    assert bl.is_blocked("M~big") is False
    assert ra2.status_code != 409


def test_route_assign_never_auto_blocks_a_model_that_fits(assign_client):
    client, _ = assign_client
    rok = client.post("/llm/workers/w1/assign", json={"model_key": "M~ok"})
    assert bl.is_blocked("M~ok") is False and rok.status_code == 200


def test_route_assign_empty_spill_clears_orphan_row(assign_client):
    """CASE B: cleanup, never a new assignment."""
    client, store = assign_client
    before = (store._load()["w1"].get("spill_by_model") or {})
    assert "orphan~row" in before
    assert "orphan~row" not in wr.get_models_dict(dict_return=True)
    rc = client.post("/llm/workers/w1/assign",
                     json={"model_key": "orphan~row", "spill": {}})
    after = (store._load()["w1"].get("spill_by_model") or {})
    assert rc.status_code == 200 and "orphan~row" not in after


def test_route_assign_unknown_key_still_404s(assign_client):
    """The name-vs-key slip this gate exists for is unaffected."""
    client, _ = assign_client
    assert client.post("/llm/workers/w1/assign",
                       json={"model_key": "never~seen", "spill": {}}).status_code == 404


def test_route_assign_non_empty_spill_on_orphan_still_404s(assign_client):
    """The relaxation can only remove a row, never write a contract."""
    client, _ = assign_client
    assert client.post("/llm/workers/w1/assign",
                       json={"model_key": "orphan~row2",
                             "spill": {"n_gpu_layers": 3}}).status_code == 404


def test_route_unassign_clears_an_orphan_too(assign_client):
    """The full-removal path has no manifest gate — the inconsistency the
    relaxation fixed."""
    client, store = assign_client
    assert client.post("/llm/workers/w1/unassign",
                       json={"model_key": "orphan~row"}).status_code == 200
    assert "orphan~row" not in (store._load()["w1"].get("models") or [])
