"""SUPERSEDE A STALE SAME-NAME WORKER RECORD (operator follow-up 2026-09-25).

WorkerStore.register dedupes by worker_id then url. When a box re-registers with
the SAME name but a NEW worker_id AND url (a-brain came back on id
240559cc… @ http://10.99.0.1:9101 while the old id 96f6cfbb… @
http://192.168.1.100:9100 stayed offline+approved), neither dedup key matched and
the stale row lingered as a console duplicate 'a-brain'.

This locks: a NEW registration whose name matches an existing record that is
OFFLINE beyond STALE_SUPERSEDE_SECONDS + a different id RETIRES the ghost and
carries the operator state that follows the box (admission gate, designations, and
name/old-id placement refs rewritten to the new id). A still-ONLINE or not-yet-
stale same-name record is a genuine collision — never superseded, logged instead.

Run:  venv/bin/python -m pytest tests/test_worker_supersede.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_HOME = tempfile.mkdtemp(prefix="hugpy-supersede-")
os.environ["PROJECTS_HOME"] = _HOME
os.environ["SERVE_OVERRIDES_PATH"] = os.path.join(_HOME, "serve_overrides.json")
os.environ.setdefault("HUGPY_COMMS_DB", "off")

import pytest  # noqa: E402

from hugpy_fleet.central import workers as W  # noqa: E402
from hugpy_fleet.central.workers import WorkerStore  # noqa: E402
from hugpy_fleet.central import priority_groups as PG  # noqa: E402
from hugpy_engine.serve import overrides as OV  # noqa: E402

OLD_ID = "96f6cfbb7ebf40a2a1a616c4c95cd71b"
NEW_ID = "240559cc11904b368e214ddec3070614"
OLD_URL = "http://192.168.1.100:9100"
NEW_URL = "http://10.99.0.1:9101"
NAME = "a-brain"
MK = "Qwen2.5-VL-7B-Instruct-GGUF"


@pytest.fixture(autouse=True)
def _isolate_assign_memory(monkeypatch, tmp_path):
    # Assignment memory defaults under PROJECTS_HOME, which every test in this file
    # shares — isolate it per test so a carried designation in one test can't be
    # restored (by id) into another.
    monkeypatch.setattr(W, "_assign_memory_path",
                        lambda: str(tmp_path / "assign.json"))


def _store(tmp_path):
    return WorkerStore(path=str(tmp_path / "wk.json"))


def _age(store, wid, seconds):
    """Push a worker's last_seen into the past so it reads offline/stale."""
    with store._transaction() as workers:
        workers[wid]["last_seen"] = W._now() - seconds


def _names(store):
    return sorted(w["name"] for w in store.all())


def _ids(store):
    return sorted(w["id"] for w in store.all())


# ── the core supersede ────────────────────────────────────────────────────────

def test_stale_same_name_ghost_is_superseded(tmp_path):
    s = _store(tmp_path)
    s.register(name=NAME, url=OLD_URL, worker_id=OLD_ID)
    s.set_admission(OLD_ID, "approved")
    _age(s, OLD_ID, W.STALE_SUPERSEDE_SECONDS + 60)     # a genuine ghost

    new = s.register(name=NAME, url=NEW_URL, worker_id=NEW_ID)
    assert new["id"] == NEW_ID
    # exactly one a-brain, and it's the new id
    assert _names(s) == [NAME]
    assert _ids(s) == [NEW_ID]
    assert OLD_ID not in _ids(s)


def test_superseding_box_inherits_approved_admission(tmp_path):
    s = _store(tmp_path)
    s.register(name=NAME, url=OLD_URL, worker_id=OLD_ID)
    s.set_admission(OLD_ID, "approved")
    _age(s, OLD_ID, W.STALE_SUPERSEDE_SECONDS + 60)

    new = s.register(name=NAME, url=NEW_URL, worker_id=NEW_ID)
    # already-approved box is NOT reset to pending
    assert new["admission"] == "approved"


def test_blocked_ghost_is_not_revived_to_approved(tmp_path):
    s = _store(tmp_path)
    s.register(name=NAME, url=OLD_URL, worker_id=OLD_ID)
    s.set_admission(OLD_ID, "blocked")
    _age(s, OLD_ID, W.STALE_SUPERSEDE_SECONDS + 60)

    new = s.register(name=NAME, url=NEW_URL, worker_id=NEW_ID)
    assert new["admission"] == "blocked"          # a block follows the box


def test_pending_ghost_leaves_new_box_pending(tmp_path):
    s = _store(tmp_path)
    s.register(name=NAME, url=OLD_URL, worker_id=OLD_ID)   # lands pending
    _age(s, OLD_ID, W.STALE_SUPERSEDE_SECONDS + 60)

    new = s.register(name=NAME, url=NEW_URL, worker_id=NEW_ID)
    assert new["admission"] == "pending"


def test_designations_follow_the_box(tmp_path):
    s = _store(tmp_path)
    s.register(name=NAME, url=OLD_URL, worker_id=OLD_ID, models=[MK])
    s.set_admission(OLD_ID, "approved")
    _age(s, OLD_ID, W.STALE_SUPERSEDE_SECONDS + 60)

    new = s.register(name=NAME, url=NEW_URL, worker_id=NEW_ID)
    assert MK in new["models"]


def test_transient_automated_designation_is_not_carried(tmp_path):
    s = _store(tmp_path)
    s.register(name=NAME, url=OLD_URL, worker_id=OLD_ID)
    # an automated, non-pinned designation on the ghost
    with s._transaction() as workers:
        workers[OLD_ID]["models"] = [MK]
        src = sorted(W.AUTOMATED_DESIGNATION_SOURCES)[0]
        workers[OLD_ID]["designation_meta"] = {MK: {"source": src, "pinned": False}}
    s.set_admission(OLD_ID, "approved")
    _age(s, OLD_ID, W.STALE_SUPERSEDE_SECONDS + 60)

    new = s.register(name=NAME, url=NEW_URL, worker_id=NEW_ID)
    assert MK not in new["models"]                # automation is transient


# ── never supersede a live/recent same-name record ────────────────────────────

def test_online_same_name_is_a_collision_not_superseded(tmp_path):
    s = _store(tmp_path)
    s.register(name=NAME, url=OLD_URL, worker_id=OLD_ID)
    s.set_admission(OLD_ID, "approved")
    s.heartbeat(OLD_ID)                            # fresh last_seen -> online

    s.register(name=NAME, url=NEW_URL, worker_id=NEW_ID)
    # both rows kept — a genuine name collision, not a stale ghost
    assert set(_ids(s)) == {OLD_ID, NEW_ID}
    assert _names(s) == [NAME, NAME]


def test_recently_offline_within_window_is_not_superseded(tmp_path):
    s = _store(tmp_path)
    s.register(name=NAME, url=OLD_URL, worker_id=OLD_ID)
    s.set_admission(OLD_ID, "approved")
    _age(s, OLD_ID, W.HEARTBEAT_TIMEOUT_SECONDS + 5)   # offline, but not stale yet

    s.register(name=NAME, url=NEW_URL, worker_id=NEW_ID)
    assert set(_ids(s)) == {OLD_ID, NEW_ID}            # tolerate the transient dup


def test_different_name_never_supersedes(tmp_path):
    s = _store(tmp_path)
    s.register(name="other-box", url=OLD_URL, worker_id=OLD_ID)
    _age(s, OLD_ID, W.STALE_SUPERSEDE_SECONDS + 60)

    s.register(name=NAME, url=NEW_URL, worker_id=NEW_ID)
    assert set(_ids(s)) == {OLD_ID, NEW_ID}


# ── name/old-id placement refs are rewritten to the new id ────────────────────

def test_placement_refs_follow_to_the_new_id(monkeypatch, tmp_path):
    s = _store(tmp_path)
    monkeypatch.setattr(W, "worker_store", s)
    ovpath = str(tmp_path / "ov.json")
    monkeypatch.setattr(OV, "_OVERRIDES_PATH", ovpath)
    # a per-model preference written under the box NAME, and a group under old id
    OV.set_override(MK, {"worker_prefs": [NAME]})
    for g in PG.all_groups():
        PG.delete_group(g["id"])
    PG.put_group("brains", name="brains", members=[MK], enabled=True,
                 workers=[OLD_ID])

    s.register(name=NAME, url=OLD_URL, worker_id=OLD_ID)
    s.set_admission(OLD_ID, "approved")
    _age(s, OLD_ID, W.STALE_SUPERSEDE_SECONDS + 60)

    s.register(name=NAME, url=NEW_URL, worker_id=NEW_ID)

    prefs, _polite, _bw = OV.placement_policy(MK)
    assert prefs == [NEW_ID]                       # name ref -> new id
    assert PG.get_group("brains")["workers"] == [NEW_ID]   # old id -> new id
