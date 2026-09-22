"""Eviction telemetry: emitter, ring, relay, store, and the ingest/read roundtrip.

The invariant every one of these guards is the same one: telemetry is
OBSERVATION ONLY. A broken sink, a full ring, a dead relay or a wedged store
must be invisible to the eviction path — so most of these tests assert that
something DOESN'T happen (no raise, no block, no changed return value) rather
than that a pretty event was produced.
"""
import json
import os
import threading
import time

import pytest

from hugpy_fleet.central import evictions as ev


@pytest.fixture(autouse=True)
def _clean():
    ev.reset_for_tests()
    yield
    ev.reset_for_tests()
    ev.set_store(None)


@pytest.fixture
def store(tmp_path):
    s = ev.EvictionStore(path=str(tmp_path / "comms.db"))
    ev.set_store(s)
    return s


# --------------------------------------------------------------------------- #
# emitter
# --------------------------------------------------------------------------- #

def test_event_carries_the_mandatory_stamps():
    got = ev.emit_eviction_event("evict.done", model_key="m1", tier="in-process",
                                 freed_bytes=123)
    assert got["stage"] == "evict.done"
    assert got["model_key"] == "m1"
    assert got["tier"] == "in-process"
    assert isinstance(got["ts"], float) and got["ts"] > 0
    assert got["seq"] == 1
    assert got["worker_id"]                      # never empty — see worker_id()


def test_seq_is_monotonic_per_process():
    seqs = [ev.emit_eviction_event("reclaim.done")["seq"] for _ in range(5)]
    assert seqs == [1, 2, 3, 4, 5]


def test_none_fields_are_omitted_not_nulled():
    got = ev.emit_eviction_event("headroom.start", incoming_model="m1",
                                 need_bytes=None, trigger="load")
    assert "need_bytes" not in got
    assert got["trigger"] == "load"


def test_run_scope_stamps_and_restores():
    with ev.run_scope("RUN-A"):
        a = ev.emit_eviction_event("headroom.start", incoming_model="m1")
        with ev.run_scope("RUN-B"):
            b = ev.emit_eviction_event("evict.start", model_key="m2")
        c = ev.emit_eviction_event("headroom.done", outcome="fit")
    d = ev.emit_eviction_event("reclaim.done")
    assert a["run_id"] == "RUN-A"
    assert b["run_id"] == "RUN-B"          # nested pass gets its own id
    assert c["run_id"] == "RUN-A"          # …and the parent's is restored
    assert "run_id" not in d               # outside any scope -> uncorrelated


def test_explicit_run_id_beats_the_ambient_one():
    with ev.run_scope("AMBIENT"):
        got = ev.emit_eviction_event("evict.done", run_id="EXPLICIT")
    assert got["run_id"] == "EXPLICIT"


def test_run_scope_is_thread_local():
    seen = {}

    def other():
        seen["ev"] = ev.emit_eviction_event("reclaim.done")

    with ev.run_scope("MAIN"):
        t = threading.Thread(target=other)
        t.start()
        t.join()
    # The other thread never entered a scope, so it must NOT inherit MAIN's id —
    # otherwise two concurrent loads would render as one card.
    assert "run_id" not in seen["ev"]


def test_worker_id_override():
    ev.set_worker_id("ae-3090")
    assert ev.emit_eviction_event("reclaim.done")["worker_id"] == "ae-3090"


# --------------------------------------------------------------------------- #
# ring
# --------------------------------------------------------------------------- #

def test_ring_keeps_the_newest_and_drops_the_oldest():
    for i in range(ev.RING_MAX + 50):
        ev.emit_eviction_event("evict.done", model_key=f"m{i}")
    got = ev.recent(limit=ev.RING_MAX + 100)
    assert len(got) == ev.RING_MAX
    # drop-OLDEST: the last event emitted must still be there.
    assert got[-1]["model_key"] == f"m{ev.RING_MAX + 49}"
    assert got[0]["model_key"] != "m0"


def test_recent_limit_returns_the_latest():
    for i in range(10):
        ev.emit_eviction_event("evict.done", model_key=f"m{i}")
    got = ev.recent(limit=3)
    assert [e["model_key"] for e in got] == ["m7", "m8", "m9"]


# --------------------------------------------------------------------------- #
# THE load-path contract: nothing here may ever raise
# --------------------------------------------------------------------------- #

def test_a_throwing_sink_does_not_break_the_emit():
    def boom(_ev):
        raise RuntimeError("sink is broken")

    good = []
    ev.register_sink(boom)
    ev.register_sink(good.append)
    got = ev.emit_eviction_event("evict.done", model_key="m1")
    assert got is not None                 # the emit still succeeded
    assert len(good) == 1                  # …and the healthy sink still ran


def test_unserializable_field_does_not_raise():
    class Weird:
        def __repr__(self):
            raise RuntimeError("even repr is broken")

    # Must not propagate: a caller passing junk is a telemetry bug, not a reason
    # to fail a model load.
    ev.emit_eviction_event("evict.done", model_key="m1", junk=Weird())


def test_emit_survives_a_disabled_store_sink(tmp_path):
    s = ev.EvictionStore(path="/proc/definitely/not/writable/x.db")
    ev.set_store(s)
    ev.install_store_sink()
    for _ in range(10):
        assert ev.emit_eviction_event("evict.done", model_key="m1") is not None


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #

def test_store_roundtrip_preserves_the_whole_event(store):
    src = ev.build_event("makeroom.verdict", run_id="R1", action="refuse",
                         reason={"reason": "won't fit", "deficit": 42},
                         evicted=["a", "b"])
    assert store.append([src]) == 1
    got = store.recent(limit=10)
    assert len(got) == 1
    assert got[0]["action"] == "refuse"
    assert got[0]["reason"] == {"reason": "won't fit", "deficit": 42}
    assert got[0]["evicted"] == ["a", "b"]
    assert got[0]["_id"] > 0               # the stream cursor


def test_store_rejects_rows_without_a_stage(store):
    assert store.append([{"no": "stage"}, {"stage": "evict.done"}]) == 1


def test_after_id_is_a_working_stream_cursor(store):
    store.append([ev.build_event("evict.done", model_key=f"m{i}")
                  for i in range(5)])
    first = store.recent(limit=100)
    cursor = first[2]["_id"]
    tail = store.recent(limit=100, after_id=cursor)
    assert [e["model_key"] for e in tail] == ["m3", "m4"]


def test_since_ts_filters_by_time(store):
    old = ev.build_event("evict.done", model_key="old")
    old["ts"] = time.time() - 3600
    new = ev.build_event("evict.done", model_key="new")
    store.append([old, new])
    got = store.recent(limit=100, since_ts=time.time() - 60)
    assert [e["model_key"] for e in got] == ["new"]


def test_recent_returns_the_newest_when_limited(store):
    store.append([ev.build_event("evict.done", model_key=f"m{i}")
                  for i in range(20)])
    got = store.recent(limit=3)
    assert [e["model_key"] for e in got] == ["m17", "m18", "m19"]


def test_prune_bounds_the_history(store):
    store.max_rows = 10
    store.append([ev.build_event("evict.done", model_key=f"m{i}")
                  for i in range(40)])
    store.prune()
    got = store.recent(limit=1000)
    assert len(got) <= 10
    assert got[-1]["model_key"] == "m39"   # the newest survive


def test_max_id_tracks_the_head(store):
    assert store.max_id() == 0
    store.append([ev.build_event("evict.done", model_key="m1")])
    assert store.max_id() > 0


def test_store_self_disables_after_repeated_failure():
    s = ev.EvictionStore(path="/proc/definitely/not/writable/x.db")
    for _ in range(ev.MAX_FAILURES + 2):
        s.append([ev.build_event("evict.done", model_key="m1")])
    assert s._disabled is True
    # A disabled store still answers, it just answers empty.
    assert s.recent(limit=10) == []


def test_install_store_sink_persists_emitted_events(store):
    ev.install_store_sink()
    with ev.run_scope("R9"):
        ev.emit_eviction_event("headroom.start", incoming_model="m1",
                               trigger="reservation")
        ev.emit_eviction_event("headroom.done", outcome="fit", evicted=[])
    got = store.recent(limit=10)
    assert [e["stage"] for e in got] == ["headroom.start", "headroom.done"]
    assert {e["run_id"] for e in got} == {"R9"}


# --------------------------------------------------------------------------- #
# relay
# --------------------------------------------------------------------------- #

def test_relay_batches_and_posts():
    posted = []
    r = ev.EvictionRelay(posted.append)
    for i in range(5):
        r.offer(ev.build_event("evict.done", model_key=f"m{i}"))
    assert r.flush_once() == 5
    assert len(posted) == 1 and len(posted[0]) == 5
    assert r.sent == 5


def test_relay_flush_on_empty_is_a_noop():
    posted = []
    r = ev.EvictionRelay(posted.append)
    assert r.flush_once() == 0
    assert posted == []


def test_relay_drops_oldest_when_full():
    r = ev.EvictionRelay(lambda b: None, buffer_max=10)
    for i in range(25):
        r.offer(ev.build_event("evict.done", model_key=f"m{i}"))
    assert r.dropped == 15
    batch = r.drain()
    assert len(batch) == 10
    assert batch[-1]["model_key"] == "m24"     # newest kept


def test_relay_failure_does_not_requeue_or_raise():
    def dead(_b):
        raise OSError("central is down")

    r = ev.EvictionRelay(dead)
    r.offer(ev.build_event("evict.done", model_key="m1"))
    assert r.flush_once() == 0
    assert r.failures == 1
    # Dropped, NOT re-queued: a sustained outage must not turn the buffer into a
    # replay loop that never drains.
    assert r.drain() == []


def test_relay_offer_never_blocks_the_caller():
    r = ev.EvictionRelay(lambda b: time.sleep(10), buffer_max=50)
    t0 = time.time()
    for i in range(1000):
        r.offer(ev.build_event("evict.done", model_key=f"m{i}"))
    assert time.time() - t0 < 1.0


# --------------------------------------------------------------------------- #
# ingest / read roundtrip through the HTTP surface
# --------------------------------------------------------------------------- #

@pytest.fixture
def client(store, monkeypatch):
    flask = pytest.importorskip("flask")
    from hugpy_server.app.routes import eviction_routes as er

    monkeypatch.setattr(er, "_worker_authorized", lambda: True)
    monkeypatch.setattr(er, "_operator_or_worker", lambda: True)
    app = flask.Flask(__name__)
    app.register_blueprint(er.eviction_bp)
    return app.test_client()


def test_ingest_then_read_roundtrip(client):
    batch = [ev.build_event("headroom.start", run_id="R1", incoming_model="m1",
                            trigger="load"),
             ev.build_event("candidate.skip", run_id="R1", model_key="m2",
                            reason="static", tier="in-process"),
             ev.build_event("headroom.done", run_id="R1", outcome="refused",
                            evicted=[])]
    r = client.post("/llm/evictions/ingest", json={"events": batch})
    assert r.status_code == 200
    assert r.get_json()["stored"] == 3

    r = client.get("/llm/evictions?limit=50")
    assert r.status_code == 200
    body = r.get_json()
    assert [e["stage"] for e in body["events"]] == [
        "headroom.start", "candidate.skip", "headroom.done"]
    assert body["events"][1]["reason"] == "static"
    assert body["cursor"] > 0


def test_ingest_rejects_a_malformed_body(client):
    assert client.post("/llm/evictions/ingest", json={"events": "nope"}).status_code == 400


def test_ingest_skips_rows_without_a_stage(client):
    r = client.post("/llm/evictions/ingest",
                    json={"events": [{"stage": "evict.done"}, {"x": 1}, "junk"]})
    assert r.get_json()["received"] == 1


def test_ingest_requires_auth(store, monkeypatch):
    flask = pytest.importorskip("flask")
    from hugpy_server.app.routes import eviction_routes as er
    monkeypatch.setattr(er, "_worker_authorized", lambda: False)
    app = flask.Flask(__name__)
    app.register_blueprint(er.eviction_bp)
    c = app.test_client()
    assert c.post("/llm/evictions/ingest", json={"events": []}).status_code == 401


def test_read_honors_after_id(client):
    client.post("/llm/evictions/ingest", json={"events": [
        ev.build_event("evict.done", model_key=f"m{i}") for i in range(4)]})
    first = client.get("/llm/evictions?limit=50").get_json()
    cursor = first["events"][1]["_id"]
    tail = client.get(f"/llm/evictions?after_id={cursor}").get_json()
    assert [e["model_key"] for e in tail["events"]] == ["m2", "m3"]


def test_stream_replays_history_then_signals_ready(client):
    client.post("/llm/evictions/ingest", json={"events": [
        ev.build_event("evict.done", run_id="R1", model_key="m1",
                       tier="slot-child", freed_bytes=42)]})
    r = client.get("/llm/evictions/stream")
    assert r.status_code == 200
    assert r.mimetype == "text/event-stream"
    # Pull just the replay + the ready marker, then abandon the generator (it
    # would otherwise poll for an hour).
    frames = []
    it = r.response.__iter__()
    for _ in range(2):
        frames.append(next(it).decode())
    r.response.close()
    first = json.loads(frames[0].split("data: ", 1)[1])
    assert first["stage"] == "evict.done"
    assert first["model_key"] == "m1"
    assert first["freed_bytes"] == 42
    ready = json.loads(frames[1].split("data: ", 1)[1])
    assert ready["stage"] == "stream.ready"
    assert ready["cursor"] >= first["_id"]


# --------------------------------------------------------------------------- #
# SERVE PIPELINE — provision / resolve / load
#
# The 2026-07-28 incident: computron's disk was 100% full, provisioning died
# with ENOSPC before eviction ever ran, and the feed said only "trigger load /
# no candidates walked". The feed was not wrong, it was SHORT — it began after
# the point of failure. These guard that a request now says WHERE it died and
# WHY, in the same card, without ssh'ing into the box.
# --------------------------------------------------------------------------- #

import errno as _errno


def test_serve_stages_are_in_the_vocabulary():
    for s in ("provision.start", "provision.fail", "provision.done",
              "resolve.fail", "load.start", "load.done", "load.fail"):
        assert s in ev.STAGES
    # The eviction vocabulary is unchanged and still fully present.
    for s in ("headroom.start", "candidate.skip", "evict.done", "headroom.done"):
        assert s in ev.STAGES


def test_provision_start_names_the_source_and_destination():
    got = ev.emit_provision_start("Q", "central-transfer",
                                  dest_path="/mnt/storage/models/Q")
    assert got["stage"] == "provision.start"
    assert got["source"] == "central-transfer"
    assert got["dest_path"] == "/mnt/storage/models/Q"


def test_provision_fail_on_enospc_carries_the_disk_facts(tmp_path):
    """THE regression test for the incident. A full disk must be visible AS a
    full disk — errno name, free bytes, total bytes, and a human sentence — so
    nobody has to read journalctl to learn the drive filled up."""
    exc = OSError(_errno.ENOSPC, "No space left on device")
    got = ev.emit_provision_fail("Q", "central-transfer", exc=exc,
                                 dest_path=str(tmp_path / "does" / "not" / "exist"))
    assert got["stage"] == "provision.fail"
    assert got["errno_name"] == "ENOSPC"
    assert got["error_class"] == "OSError"
    assert "No space left on device" in got["detail"]
    # statvfs of the destination FILESYSTEM — resolved by walking up to the
    # nearest existing ancestor, because the dest dir usually doesn't exist yet.
    assert isinstance(got["disk_free_bytes"], int)
    assert isinstance(got["disk_total_bytes"], int)
    assert got["disk_total_bytes"] >= got["disk_free_bytes"]
    assert "disk full (ENOSPC)" in got["human"]
    assert "free of" in got["human"]


def test_provision_fail_without_an_exception_still_reports_a_reason():
    got = ev.emit_provision_fail("Q", "archive",
                                 detail="central cannot provide the files")
    assert got["stage"] == "provision.fail"
    assert got["detail"] == "central cannot provide the files"
    assert "errno_name" not in got          # absent beats a null on the wire


def test_provision_fail_on_a_non_os_error_omits_disk_noise(tmp_path):
    got = ev.emit_provision_fail("Q", "hf", exc=RuntimeError("404 from HF"),
                                 dest_path=str(tmp_path))
    assert got["error_class"] == "RuntimeError"
    assert "errno_name" not in got
    # A 404 is not a disk problem; don't imply one by attaching free space.
    assert "disk_free_bytes" not in got
    assert "human" not in got


def test_provision_done_reports_bytes_and_duration():
    got = ev.emit_provision_done("Q", "hf", bytes_=4_700_000_000,
                                 duration_ms=91_000)
    assert got["stage"] == "provision.done"
    assert got["bytes"] == 4_700_000_000
    assert got["duration_ms"] == 91_000


def test_resolve_fail_names_the_path_and_the_reason():
    got = ev.emit_resolve_fail("Q", "/mnt/storage/models/Q/x.gguf",
                               "resolved GGUF is 0 bytes")
    assert got["stage"] == "resolve.fail"
    assert got["resolved_path"] == "/mnt/storage/models/Q/x.gguf"
    assert got["reason"] == "resolved GGUF is 0 bytes"


def test_load_stages_carry_the_engine():
    a = ev.emit_load_start("Q", engine="LlamaCppChatRunner")
    b = ev.emit_load_done("Q", engine="LlamaCppChatRunner", duration_ms=4200)
    c = ev.emit_load_fail("Q", engine="LlamaCppChatRunner",
                          exc=RuntimeError("SIGILL"))
    assert a["stage"] == "load.start" and a["engine"] == "LlamaCppChatRunner"
    assert b["duration_ms"] == 4200
    assert c["error"] == "RuntimeError: SIGILL"


def test_errno_name_and_disk_stats_are_total():
    assert ev.errno_name(OSError(_errno.ENOSPC, "x")) == "ENOSPC"
    assert ev.errno_name(RuntimeError("no errno")) == ""
    assert ev.errno_name(OSError("no errno at all")) == ""
    st = ev.disk_stats("/")
    assert st["disk_total_bytes"] > 0
    # Never raises, whatever it is handed.
    assert ev.disk_stats(None) != {} or True
    assert ev.disk_stats("\0bogus") == {} or isinstance(ev.disk_stats("\0bogus"), dict)


def test_describe_disk_error_is_an_operator_sentence():
    msg = ev.describe_disk_error(OSError(_errno.ENOSPC, "x"), "/")
    assert msg.startswith("disk full (ENOSPC) on /")
    assert " free of " in msg
    # Nothing sharper to say than str(exc) -> "" so the caller can fall back.
    assert ev.describe_disk_error(RuntimeError("boom"), "/") == ""
    assert ev.describe_disk_error(OSError(_errno.EROFS, "x"), "/") \
        .startswith("cannot write to /")


def test_serve_emitters_never_raise_on_junk():
    """THE contract, extended to the serve path: a telemetry bug must never
    convert an honest ENOSPC into a mystery."""
    class Weird:
        def __repr__(self):
            raise RuntimeError("even repr is broken")

        def __str__(self):
            raise RuntimeError("and str too")

    ev.emit_provision_start(Weird(), Weird(), dest_path=Weird())
    ev.emit_provision_fail("Q", "hf", exc=Weird(), dest_path=Weird())
    ev.emit_provision_done("Q", "hf", bytes_=Weird())
    ev.emit_resolve_fail("Q", Weird(), Weird())
    ev.emit_load_fail("Q", exc=Weird())


def test_serve_scope_joins_an_open_pass_instead_of_splitting_it():
    """A serve that is already inside somebody's pass is part of that pass.

    This is the difference that puts provisioning and the eviction it provoked
    in ONE console card: run_scope always mints a fresh id, serve_scope adopts
    the ambient one when there is one."""
    with ev.run_scope("PASS-1"):
        with ev.serve_scope():
            inner = ev.emit_provision_start("Q", "local")
        after = ev.emit_eviction_event("headroom.start", incoming_model="Q")
    assert inner["run_id"] == "PASS-1"
    assert after["run_id"] == "PASS-1"


def test_serve_scope_opens_one_when_nothing_is_open():
    with ev.serve_scope() as rid:
        got = ev.emit_provision_start("Q", "local")
    assert rid and got["run_id"] == rid


def test_a_whole_failed_serve_is_one_correlated_card():
    """End to end, the shape the console renders: provisioning tried two
    sources, both died on a full disk, and every event shares one run_id."""
    seen = []
    ev.register_sink(seen.append)
    exc = OSError(_errno.ENOSPC, "No space left on device")
    with ev.serve_scope():
        for source in ("central-transfer", "archive"):
            ev.emit_provision_start("Q", source, dest_path="/")
            ev.emit_provision_fail("Q", source, exc=exc, dest_path="/")
        ev.emit_resolve_fail("Q", None, "no GGUF resolved for this model on disk")

    assert [e["stage"] for e in seen] == [
        "provision.start", "provision.fail",
        "provision.start", "provision.fail",
        "resolve.fail"]
    assert len({e["run_id"] for e in seen}) == 1       # ONE card
    assert [e["source"] for e in seen if e["stage"] == "provision.fail"] == [
        "central-transfer", "archive"]
    assert all(e["errno_name"] == "ENOSPC"
               for e in seen if e["stage"] == "provision.fail")


def test_serve_stages_survive_the_store_and_the_http_surface(client):
    """Central needs NO migration for a new stage — the full event rides in the
    row's JSON body — so a worker on a newer release can add fields freely."""
    batch = [ev.build_event("provision.start", run_id="R7", model_key="Q",
                            source="central-transfer", dest_path="/mnt/storage"),
             ev.build_event("provision.fail", run_id="R7", model_key="Q",
                            source="central-transfer", errno_name="ENOSPC",
                            error_class="OSError", disk_free_bytes=0,
                            disk_total_bytes=938_000_000_000,
                            human="disk full (ENOSPC) on /mnt/storage — "
                                  "0 B free of 938 GB")]
    r = client.post("/llm/evictions/ingest", json={"events": batch})
    assert r.status_code == 200 and r.get_json()["stored"] == 2

    body = client.get("/llm/evictions?limit=50").get_json()
    assert [e["stage"] for e in body["events"]] == ["provision.start",
                                                    "provision.fail"]
    fail = body["events"][1]
    assert fail["errno_name"] == "ENOSPC"
    assert fail["disk_free_bytes"] == 0
    assert fail["human"].startswith("disk full (ENOSPC)")
    assert fail["run_id"] == "R7"           # the UI groups the card on this


def test_serve_stages_stream_over_sse(client):
    client.post("/llm/evictions/ingest", json={"events": [
        ev.build_event("provision.fail", run_id="R8", model_key="Q",
                       source="archive", errno_name="ENOSPC",
                       human="disk full (ENOSPC) on /mnt/storage — 0 B free")]})
    r = client.get("/llm/evictions/stream")
    assert r.status_code == 200
    it = r.response.__iter__()
    first = json.loads(next(it).decode().split("data: ", 1)[1])
    r.response.close()
    assert first["stage"] == "provision.fail"
    assert first["errno_name"] == "ENOSPC"


def test_journal_line_leads_with_the_fields_an_operator_greps():
    """A box whose relay is down must still tell the whole story to journalctl."""
    line = ev._kv_line(ev.build_event(
        "provision.fail", run_id="R1", model_key="Q", source="archive",
        errno_name="ENOSPC", detail="[Errno 28] No space left on device"))
    assert line.startswith("stage=provision.fail ")
    assert "source=archive" in line
    assert "errno_name=ENOSPC" in line
    assert "model_key=Q" in line


# --------------------------------------------------------------------------- #
# dispatch wiring — the events a real headroom pass produces
# --------------------------------------------------------------------------- #

@pytest.fixture
def wired_placement():
    """The engine's dispatch emits eviction telemetry through the
    ``hugpy_engine.placement`` EvictionLedger seam, which the server wiring
    fills with ``hugpy_fleet.central.evictions``. Install it the way the server
    does so these tests do not depend on an earlier test having built an app
    (the conftest restores the placement registry afterwards)."""
    from hugpy_server.wiring import install_placement
    install_placement()


def test_headroom_pass_emits_a_correlated_story(monkeypatch, wired_placement):
    from hugpy_engine.dispatch import dispatch as d

    seen = []
    ev.register_sink(seen.append)

    # A resident that fits after one yield: "cold" is evictable, "hot" is not.
    fits = {"v": False}
    monkeypatch.setattr(d, "loaded_model_keys",
                        lambda: [("cold", "t"), ("hot", "t")])
    monkeypatch.setattr(d, "evict", lambda mk, task=None: fits.__setitem__("v", True))
    d.set_fit_check(lambda mk: fits["v"])
    d.set_evictable(lambda mk: mk == "cold")
    d.set_evict_reason(lambda mk: "actively-replying")
    d.set_post_evict_hook(lambda: None)
    d.set_make_room(None)
    try:
        evicted = d.ensure_headroom_for_load("incoming")
    finally:
        d.set_fit_check(None)
        d.set_evictable(None)
        d.set_evict_reason(None)
        d.set_post_evict_hook(None)

    assert evicted == ["cold"]
    stages = [e["stage"] for e in seen]
    assert stages[0] == "headroom.start"
    assert stages[-1] == "headroom.done"
    for s in ("fit.fail", "candidate.skip", "evict.start", "evict.done",
              "reclaim.done"):
        assert s in stages, f"{s} missing from {stages}"

    # ONE run_id for the whole pass — this is what groups the console card.
    run_ids = {e["run_id"] for e in seen}
    assert len(run_ids) == 1

    skip = next(e for e in seen if e["stage"] == "candidate.skip")
    assert skip["model_key"] == "hot"
    assert skip["reason"] == "actively-replying"     # WHY it was protected

    done = seen[-1]
    assert done["outcome"] == "fit"
    assert done["evicted"] == ["cold"]


def test_headroom_pass_is_unaffected_by_a_broken_emitter(monkeypatch, wired_placement):
    """THE contract: telemetry failure is invisible to the load path."""
    from hugpy_engine.dispatch import dispatch as d

    def boom(*_a, **_k):
        raise RuntimeError("telemetry exploded")

    monkeypatch.setattr(ev, "emit_eviction_event", boom)
    monkeypatch.setattr(ev, "new_run_id", boom)
    monkeypatch.setattr(ev, "run_scope", boom)

    fits = {"v": False}
    monkeypatch.setattr(d, "loaded_model_keys", lambda: [("cold", "t")])
    monkeypatch.setattr(d, "evict", lambda mk, task=None: fits.__setitem__("v", True))
    d.set_fit_check(lambda mk: fits["v"])
    d.set_evictable(lambda mk: True)
    d.set_post_evict_hook(lambda: None)
    d.set_make_room(None)
    try:
        assert d.ensure_headroom_for_load("incoming") == ["cold"]
    finally:
        d.set_fit_check(None)
        d.set_evictable(None)
        d.set_post_evict_hook(None)


def test_makeroom_refusal_emits_a_verdict_and_still_raises(monkeypatch, wired_placement):
    from hugpy_engine.dispatch import dispatch as d

    seen = []
    ev.register_sink(seen.append)
    monkeypatch.setattr(d, "loaded_model_keys", lambda: [])
    d.set_fit_check(lambda mk: True)
    d.set_evictable(lambda mk: False)
    d.set_make_room(lambda mk: {"action": "refuse",
                                "reason": {"reason": "won't fit on GPU"},
                                "evicted": []})
    try:
        with pytest.raises(d.LoadRefusal):
            d.ensure_headroom_for_load("incoming")
    finally:
        d.set_fit_check(None)
        d.set_evictable(None)
        d.set_make_room(None)

    stages = [e["stage"] for e in seen]
    assert "makeroom.verdict" in stages
    verdict = next(e for e in seen if e["stage"] == "makeroom.verdict")
    assert verdict["action"] == "refuse"
    assert seen[-1]["stage"] == "headroom.done"
    assert seen[-1]["outcome"] == "refused"
