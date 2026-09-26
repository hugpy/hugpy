"""STALE WORKER-NAME REFERENCES (operator incident 2026-09-25).

Placement is stored by NAME — per-model ``worker_prefs`` in serve_overrides.json
and the priority-group ``workers`` order in settings.json — so renaming a worker
("aeb" -> "ae-worker", same worker_id) stranded every token written under the old
name, and central logged "ordered worker preference ['aeb'] but NONE of them is
an eligible candidate" every few minutes while the model failed to route.

This locks: (1) register() now records rename history; (2) _pref_index resolves a
former name; (3) the startup migration rewrites the stale tokens to the worker's
stable id in BOTH stores — including the pre-history 'aeb' rename via the
one-time KNOWN_WORKER_RENAMES backfill.

Run:  venv/bin/python -m pytest tests/test_worker_rename_migration.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Isolate the settings.json / serve_overrides.json stores in a temp home BEFORE
# importing the modules that resolve their paths at import time.
_HOME = tempfile.mkdtemp(prefix="hugpy-rename-mig-")
os.environ["PROJECTS_HOME"] = _HOME
os.environ["SERVE_OVERRIDES_PATH"] = os.path.join(_HOME, "serve_overrides.json")
os.environ.setdefault("HUGPY_COMMS_DB", "off")

import pytest  # noqa: E402

from hugpy_fleet.central import workers as W  # noqa: E402
from hugpy_fleet.central.workers import WorkerStore  # noqa: E402
from hugpy_fleet.central import priority_groups as PG  # noqa: E402
from hugpy_engine.serve import overrides as OV  # noqa: E402

AE_ID = "688ca48f0e5445f2aa2594f54bafb6be"
MK = "Qwen3-Coder-Next-GGUF"


# ── register() records rename history ─────────────────────────────────────────

def test_register_records_prev_names_on_rename(tmp_path):
    s = WorkerStore(path=str(tmp_path / "wk.json"))
    w = s.register(name="aeb", url="http://192.168.1.100:9200")
    wid = w["id"]
    # rename: same URL, new name
    w2 = s.register(name="ae-worker", url="http://192.168.1.100:9200", worker_id=wid)
    assert w2["name"] == "ae-worker"
    assert "aeb" in (w2.get("prev_names") or [])
    # re-registering under the SAME name does not duplicate history
    w3 = s.register(name="ae-worker", url="http://192.168.1.100:9200", worker_id=wid)
    assert (w3.get("prev_names") or []).count("aeb") == 1


def test_pref_index_resolves_a_former_name():
    worker = {"id": AE_ID, "name": "ae-worker", "prev_names": ["aeb"]}
    assert W._pref_index(worker, ["aeb"]) == 0
    assert W._pref_index(worker, ["ae-worker"]) == 0
    assert W._pref_index(worker, [AE_ID]) == 0
    assert W._pref_index(worker, ["nope"]) is None


def test_seed_prev_names_backfills_known_rename(tmp_path):
    s = WorkerStore(path=str(tmp_path / "wk.json"))
    s.register(name="ae-worker", url="http://192.168.1.100:9200", worker_id=AE_ID)
    n = s.seed_prev_names({AE_ID: ["aeb"]})
    assert n == 1
    w = next(x for x in s.all() if x["id"] == AE_ID)
    assert "aeb" in w["prev_names"]
    # idempotent
    assert s.seed_prev_names({AE_ID: ["aeb"]}) == 0


# ── the two store migrations ──────────────────────────────────────────────────

def test_overrides_migration_rewrites_stale_token(monkeypatch, tmp_path):
    path = str(tmp_path / "ov.json")
    monkeypatch.setattr(OV, "_OVERRIDES_PATH", path)
    OV.set_override(MK, {"worker_prefs": ["aeb"]})
    OV.set_override("Some-Other", {"worker_prefs": ["computron"],
                                   "no_evict_by_worker": {"aeb": True}})

    def resolve(tok):
        return AE_ID if str(tok).lower() == "aeb" else None

    changed = OV.migrate_worker_tokens(resolve)
    assert MK in changed
    prefs, _polite, by_worker = OV.placement_policy(MK)
    assert prefs == [AE_ID]
    # a token that already resolves (computron) is untouched; the aeb key is healed
    _p, _po, bw = OV.placement_policy("Some-Other")
    assert AE_ID in bw and "aeb" not in bw
    # idempotent: a second run rewrites nothing (the token is now an id)
    assert OV.migrate_worker_tokens(resolve) == {}


def test_priority_group_migration_rewrites_stale_token(monkeypatch):
    # Fresh settings store isolated in the temp home.
    for g in PG.all_groups():
        PG.delete_group(g["id"])
    PG.put_group("brains", name="brains",
                 members=[MK], enabled=True,
                 workers=["computron", "aeb"])

    def resolve(tok):
        return AE_ID if str(tok).lower() == "aeb" else None

    changed = PG.migrate_worker_tokens(resolve)
    assert "brains" in changed
    g = PG.get_group("brains")
    assert AE_ID in g["workers"] and "aeb" not in g["workers"]
    assert "computron" in g["workers"]              # already-valid token untouched
    assert PG.migrate_worker_tokens(resolve) == {}  # idempotent


# ── the orchestrator: seed + resolve + rewrite both stores ───────────────────

def test_migrate_worker_name_references_end_to_end(monkeypatch, tmp_path):
    s = WorkerStore(path=str(tmp_path / "wk.json"))
    s.register(name="ae-worker", url="http://192.168.1.100:9200", worker_id=AE_ID)
    monkeypatch.setattr(W, "worker_store", s)
    ovpath = str(tmp_path / "ov.json")
    monkeypatch.setattr(OV, "_OVERRIDES_PATH", ovpath)
    OV.set_override(MK, {"worker_prefs": ["aeb"]})
    for g in PG.all_groups():
        PG.delete_group(g["id"])
    PG.put_group("brains", name="brains", members=[MK], enabled=True,
                 workers=["aeb"])

    summary = W.migrate_worker_name_references()
    assert summary["seeded"] == 1                    # backfilled aeb via KNOWN map
    # prev_names now carries the historical name
    w = next(x for x in s.all() if x["id"] == AE_ID)
    assert "aeb" in (w.get("prev_names") or [])
    # both stores rewritten to the stable id
    prefs, _polite, _bw = OV.placement_policy(MK)
    assert prefs == [AE_ID]
    assert PG.get_group("brains")["workers"] == [AE_ID]


def test_orchestrator_migrates_a_runtime_rename_without_the_seed(monkeypatch, tmp_path):
    """A worker renamed at runtime (register twice) carries prev_names, so the
    migration heals its stale tokens with NO entry in KNOWN_WORKER_RENAMES."""
    s = WorkerStore(path=str(tmp_path / "wk.json"))
    w = s.register(name="oldbox", url="http://10.0.0.9:9200")
    wid = w["id"]
    s.register(name="newbox", url="http://10.0.0.9:9200", worker_id=wid)
    monkeypatch.setattr(W, "worker_store", s)
    monkeypatch.setattr(OV, "_OVERRIDES_PATH", str(tmp_path / "ov.json"))
    OV.set_override(MK, {"worker_prefs": ["oldbox"]})

    W.migrate_worker_name_references()
    prefs, _polite, _bw = OV.placement_policy(MK)
    assert prefs == [wid]
