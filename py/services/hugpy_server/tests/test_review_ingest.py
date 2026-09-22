"""Review results push: the worker -> central seam.

The pipeline runs where the GPU is and writes to that box's local sqlite;
central's /llm/review/* routes read CENTRAL's DB. These tests guard the three
properties that make the push safe to run unattended every night:

  1. the ingest endpoint is CLOSED to an unauthenticated caller,
  2. it is IDEMPOTENT — a retried or replayed push updates in place, so a
     worker that never sees our 200 can resend forever without duplicating,
  3. it NEVER 5xx's over data — a malformed row is counted, not raised, because
     a worker reading a 5xx as "central is broken" is how a relay storms.

Plus the round trip: the payload push.py builds from a real local DB is a
payload the route accepts.
"""
import importlib
import os
import sqlite3

import pytest

flask = pytest.importorskip("flask")


@pytest.fixture
def review_db(tmp_path, monkeypatch):
    """Point the store at a temp file. REVIEW_DB is the same knob the ae user
    unit sets, so this exercises the real configuration path."""
    path = tmp_path / "reviews.db"
    monkeypatch.setenv("REVIEW_DB", str(path))
    from hugpy_curation.review import store
    store._conn().close()    # create the file + schema so a 401 test can count 0
    return str(path)


@pytest.fixture
def routes(monkeypatch):
    from hugpy_server.app.routes import review_routes as rr
    return rr


def _mod(path):
    """importlib, not `import a.b.c as x`: abstract_hugpy_dev.flask_app exposes a
    name that shadows the `app` subpackage on attribute lookup, so the `as` form
    resolves to flask.app instead."""
    return importlib.import_module(path)


def _client(rr):
    app = flask.Flask(__name__)
    app.register_blueprint(rr.review_bp)
    return app.test_client()


@pytest.fixture
def client(review_db, routes, monkeypatch):
    monkeypatch.setattr(routes, "_ingest_authorized", lambda: True)
    return _client(routes)


def _batch(host="ae", run_id=7, hub="org/model-a"):
    return {
        "host": host,
        "criteria": "nightly",
        "run": {"run_id": run_id, "criteria": "nightly", "started_at": 100.0,
                "finished_at": 200.0, "screened": 3, "passed": 2,
                "downloaded": 1, "smoked": 1, "error": None},
        "results": [
            {"run_id": run_id, "criteria": "nightly", "hub_id": hub,
             "stage": "screened", "passed": 1, "score": 8.5, "verdict": None,
             "payload": {"hub_id": hub, "screen": {"passed": True}},
             "reviewed_at": 150.0},
            {"run_id": run_id, "criteria": "nightly", "hub_id": hub,
             "stage": "smoked", "passed": 1, "score": 8.5, "verdict": "trial",
             "payload": {"hub_id": hub, "smoke": {"ok": True}},
             "reviewed_at": 190.0},
        ],
    }


def _count(db, table, **where):
    con = sqlite3.connect(db)
    try:
        sql = f"SELECT COUNT(*) FROM {table}"
        args = []
        if where:
            sql += " WHERE " + " AND ".join(f"{k} IS ?" for k in where)
            args = list(where.values())
        return con.execute(sql, args).fetchone()[0]
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #

def test_ingest_rejects_a_tokenless_caller(review_db, routes, monkeypatch):
    """No credential -> 401, and nothing is written."""
    monkeypatch.setattr(routes, "_ingest_authorized", lambda: False)
    c = _client(routes)
    r = c.post("/llm/review/ingest", json=_batch())
    assert r.status_code == 401
    assert _count(review_db, "reviews") == 0


def test_ingest_accepts_the_operator_token(review_db, routes, monkeypatch):
    """The gate is operator-OR-worker: an operator token alone is enough, which
    is what makes `review push` from a laptop work."""
    oa = _mod("hugpy_server.app.operator_auth")
    monkeypatch.setattr(oa, "operator_authenticated", lambda: True)
    # The worker gate must NOT be consulted once the operator passes.
    wr = _mod("hugpy_server.app.routes.worker_routes")
    monkeypatch.setattr(wr, "_enrollment_ok", lambda: False)
    c = _client(routes)
    r = c.post("/llm/review/ingest", json=_batch())
    assert r.status_code == 200
    assert r.get_json()["accepted"] == 2


def test_ingest_accepts_a_valid_worker_enrollment_bearer(review_db, routes,
                                                         monkeypatch):
    """Same credential /llm/evictions/ingest uses — no operator session needed
    on a worker box. The gate verifies the PRESENTED bearer, not the rollout
    posture."""
    oa = _mod("hugpy_server.app.operator_auth")
    monkeypatch.setattr(oa, "operator_authenticated", lambda: False)
    et = _mod("hugpy_fleet.central.enrollment_tokens")
    monkeypatch.setattr(et, "verify_enrollment_token",
                        lambda tok: tok == "hpw_valid")
    c = _client(routes)
    r = c.post("/llm/review/ingest", json=_batch(),
               headers={"Authorization": "Bearer hpw_valid"})
    assert r.status_code == 200
    r = c.post("/llm/review/ingest", json=_batch(),
               headers={"Authorization": "Bearer hpw_wrong"})
    assert r.status_code == 401


def test_ingest_rejects_tokenless_even_when_rollout_would_allow(review_db,
                                                                routes,
                                                                monkeypatch):
    """STRICTNESS REGRESSION GUARD (2026-07-29): register/heartbeat's gradual-
    rollout allowance (no token -> allow while enrollment isn't required) must
    NEVER extend to this write endpoint — through the public origin that
    allowance made ingest writable by the whole internet (probed, confirmed).
    """
    oa = _mod("hugpy_server.app.operator_auth")
    monkeypatch.setattr(oa, "operator_authenticated", lambda: False)
    wr = _mod("hugpy_server.app.routes.worker_routes")
    monkeypatch.setattr(wr, "_enrollment_ok", lambda: True)
    r = _client(routes).post("/llm/review/ingest", json=_batch())
    assert r.status_code == 401


def test_ingest_fails_closed_when_both_gates_are_unavailable(review_db, routes,
                                                             monkeypatch):
    oa = _mod("hugpy_server.app.operator_auth")
    wr = _mod("hugpy_server.app.routes.worker_routes")

    def boom():
        raise RuntimeError("gate unavailable")

    monkeypatch.setattr(oa, "operator_authenticated", boom)
    monkeypatch.setattr(wr, "_enrollment_ok", boom)
    assert _client(routes).post("/llm/review/ingest",
                                json=_batch()).status_code == 401


# --------------------------------------------------------------------------- #
# idempotence
# --------------------------------------------------------------------------- #

def test_the_same_batch_twice_does_not_duplicate(client, review_db):
    first = client.post("/llm/review/ingest", json=_batch())
    second = client.post("/llm/review/ingest", json=_batch())
    assert first.status_code == second.status_code == 200
    assert first.get_json()["accepted"] == second.get_json()["accepted"] == 2
    assert _count(review_db, "reviews") == 2
    assert _count(review_db, "runs") == 1
    # …and the central row id is stable across the retry.
    assert first.get_json()["run_id"] == second.get_json()["run_id"]


def test_a_resend_updates_in_place(client, review_db):
    client.post("/llm/review/ingest", json=_batch())
    later = _batch()
    later["results"][1]["verdict"] = "adopt"
    later["run"]["smoked"] = 2
    client.post("/llm/review/ingest", json=later)
    con = sqlite3.connect(review_db)
    try:
        assert con.execute("SELECT verdict FROM reviews WHERE stage='smoked'"
                           ).fetchone()[0] == "adopt"
        assert con.execute("SELECT smoked FROM runs").fetchone()[0] == 2
    finally:
        con.close()


def test_two_hosts_with_the_same_local_run_id_stay_separate(client, review_db):
    """Run ids are per-box autoincrements — they collide across the fleet. The
    natural key is (host, run_id), so ae's run 7 must not clobber op's run 7."""
    client.post("/llm/review/ingest", json=_batch(host="ae", hub="org/a"))
    client.post("/llm/review/ingest", json=_batch(host="op", hub="org/b"))
    assert _count(review_db, "runs") == 2
    assert _count(review_db, "reviews") == 4


def test_ingest_does_not_constrain_locally_produced_rows(client, review_db):
    """source_host NULL = written on this box. SQLite treats NULLs as distinct
    in a UNIQUE index, so the ingest key never blocks a local write."""
    from hugpy_curation.review import store
    a = store.record("nightly", "org/local", "screened", {"x": 1}, passed=True)
    b = store.record("nightly", "org/local", "screened", {"x": 2}, passed=True)
    assert a != b
    assert _count(review_db, "reviews", source_host=None) == 2


# --------------------------------------------------------------------------- #
# malformed input is counted, never 5xx'd
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [
    {"hub_id": "org/x"},                                  # no stage
    {"stage": "screened"},                                # no hub_id
    {"hub_id": "org/x", "stage": "screened"},             # no criteria
    "not-a-dict",
    None,
    {"hub_id": "", "stage": "", "criteria": ""},          # empty strings
])
def test_a_malformed_row_is_rejected_not_raised(client, bad):
    body = _batch()
    body["results"] = [bad]
    r = client.post("/llm/review/ingest", json=body)
    assert r.status_code == 200
    assert r.get_json()["rejected"] == 1
    assert r.get_json()["accepted"] == 0


def test_good_rows_still_land_alongside_a_bad_one(client, review_db):
    body = _batch()
    body["results"].append({"stage": "screened"})
    r = client.post("/llm/review/ingest", json=body)
    assert r.get_json() == {**r.get_json(), "accepted": 2, "rejected": 1}
    assert _count(review_db, "reviews") == 2


def test_junk_types_in_the_numeric_columns_are_coerced_not_fatal(client):
    body = _batch()
    body["run"]["screened"] = "lots"
    body["results"][0]["score"] = "high"
    r = client.post("/llm/review/ingest", json=body)
    assert r.status_code == 200
    assert r.get_json()["accepted"] == 2


def test_an_unusable_run_header_does_not_sink_the_results(client, review_db):
    body = _batch()
    body["run"] = {"criteria": "nightly"}          # no run id at all
    r = client.post("/llm/review/ingest", json=body)
    assert r.status_code == 200
    assert r.get_json()["accepted"] == 2
    assert r.get_json()["rejected"] == 1           # the header itself
    assert _count(review_db, "runs") == 0


def test_a_store_fault_is_a_200_with_everything_counted_rejected(client, routes,
                                                                 monkeypatch):
    from hugpy_curation.review import store

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "ingest_results", boom)
    monkeypatch.setattr(store, "ingest_run", boom)
    r = client.post("/llm/review/ingest", json=_batch())
    assert r.status_code == 200                    # never 5xx at a worker
    assert r.get_json()["accepted"] == 0
    assert r.get_json()["rejected"] >= 2


def test_a_bad_envelope_is_a_400_not_a_500(client):
    assert client.post("/llm/review/ingest",
                       json={"results": []}).status_code == 400          # no host
    assert client.post("/llm/review/ingest",
                       json={"host": "ae", "results": "nope"}).status_code == 400
    assert client.post("/llm/review/ingest",
                       json={"host": "ae", "run": "nope"}).status_code == 400
    over = {"host": "ae", "results": [{"hub_id": "a", "stage": "s",
                                       "criteria": "c"}] * 1001}
    assert client.post("/llm/review/ingest", json=over).status_code == 400


def test_an_empty_batch_is_fine(client):
    r = client.post("/llm/review/ingest", json={"host": "ae", "results": []})
    assert r.status_code == 200
    assert r.get_json()["accepted"] == 0


# --------------------------------------------------------------------------- #
# round trip: what push.py builds is what the route accepts
# --------------------------------------------------------------------------- #

def test_push_builder_payload_round_trips_through_ingest(review_db, routes,
                                                         monkeypatch, tmp_path):
    """Write a run the way the pipeline does, build the payload the way push.py
    does, and feed it to the route — against a SECOND temp DB standing in for
    central, so a real worker->central hop is simulated end to end."""
    from hugpy_curation.review import push, store

    run_id = store.start_run("nightly")
    store.record("nightly", "org/a", "screened", {"screen": {"passed": True}},
                 passed=True, score=7.5, run_id=run_id)
    store.record("nightly", "org/a", "smoked", {"smoke": {"ok": True}},
                 passed=True, score=7.5, verdict="trial", run_id=run_id)
    store.record("nightly", "org/b", "screened", {"screen": {"passed": False}},
                 passed=False, score=1.0, run_id=run_id)
    store.finish_run(run_id, screened=2, passed=1, downloaded=1, smoked=1)

    monkeypatch.setenv("REVIEW_PUSH_HOST", "ae")
    payload = push.build_payload(run_id)
    assert payload["host"] == "ae"
    assert payload["run"]["run_id"] == run_id
    assert len(payload["results"]) == 3
    # payload column comes back out of sqlite as a decoded dict — it must
    # survive JSON serialisation to the wire.
    assert isinstance(payload["results"][0]["payload"], dict)

    # Now central: a different DB file.
    central = tmp_path / "central" / "reviews.db"
    monkeypatch.setenv("REVIEW_DB", str(central))
    monkeypatch.setattr(routes, "_ingest_authorized", lambda: True)
    r = _client(routes).post("/llm/review/ingest", json=payload)
    assert r.status_code == 200
    assert r.get_json()["accepted"] == 3
    assert r.get_json()["rejected"] == 0

    # …and the read routes central serves now see it.
    rows = store.recent("nightly", limit=10)
    assert {row["hub_id"] for row in rows} == {"org/a", "org/b"}
    assert store.runs("nightly")[0]["source_host"] == "ae"
    board = store.leaderboard("nightly", limit=5)
    assert board and board[0]["hub_id"] == "org/a"
    # The payload JSON survived the hop intact.
    smoked = [row for row in rows if row["stage"] == "smoked"][0]
    assert smoked["payload"]["smoke"]["ok"] is True
    assert smoked["verdict"] == "trial"


def test_push_is_off_and_silent_when_no_central_is_configured(review_db,
                                                              monkeypatch):
    """A local-only box must not log a warning every night or fail a run."""
    from hugpy_curation.review import push, store
    monkeypatch.delenv("REVIEW_CENTRAL_URL", raising=False)
    run_id = store.start_run("nightly")
    store.finish_run(run_id)
    assert push.push_run(run_id) == {"ok": False, "reason": "not_configured"}


def test_push_swallows_an_unreachable_central(review_db, monkeypatch):
    """The run already wrote its rows locally; a dead central costs one log
    line and NOTHING else — no raise, and no pushed_at stamp so the replay
    still knows about it."""
    from hugpy_curation.review import push, store
    monkeypatch.setenv("REVIEW_CENTRAL_URL", "http://127.0.0.1:1/api")
    run_id = store.start_run("nightly")
    store.finish_run(run_id)
    lines = []
    out = push.push_run(run_id, log=lines.append)
    assert out["ok"] is False
    assert len(lines) == 1
    assert store.get_run(run_id)["pushed_at"] is None
    assert [r["id"] for r in store.unpushed_runs()] == [run_id]


def test_a_pushed_run_stops_showing_up_as_pending(review_db, monkeypatch):
    from hugpy_curation.review import push, store
    run_id = store.start_run("nightly")
    store.finish_run(run_id)
    monkeypatch.setenv("REVIEW_CENTRAL_URL", "https://central.invalid/api")
    monkeypatch.setattr(push, "_post",
                        lambda *a, **k: {"accepted": 0, "rejected": 0})
    assert push.push_run(run_id)["ok"] is True
    assert store.unpushed_runs() == []


def test_the_pipeline_push_never_fails_the_run(review_db, monkeypatch):
    """_push is the guard the nightly timer depends on: even an exploding push
    module must not turn a completed run into a failure."""
    from hugpy_curation.review import pipeline
    lines = []
    monkeypatch.setenv("REVIEW_CENTRAL_URL", "https://central.invalid/api")

    import hugpy_curation.review.push as push_mod

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(push_mod, "push_run", boom)
    pipeline._push(1, lines.append)                # must not raise
    assert lines and "kaboom" in lines[0]


def test_the_token_prefers_review_specific_then_the_worker_enrollment_token(
        monkeypatch):
    from hugpy_curation.review import push
    for var in ("REVIEW_CENTRAL_TOKEN", "WORKER_ENROLL_TOKEN",
                "HUGPY_OPERATOR_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    assert push.central_token() is None
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "op")
    assert push.central_token() == "op"
    monkeypatch.setenv("WORKER_ENROLL_TOKEN", "hpw_live")
    assert push.central_token() == "hpw_live"      # the worker credential wins
    monkeypatch.setenv("REVIEW_CENTRAL_TOKEN", "explicit")
    assert push.central_token() == "explicit"


def test_the_bearer_actually_rides_the_request(review_db, monkeypatch):
    from hugpy_curation.review import push, store
    seen = {}

    def fake_post(url, body, token, timeout):
        seen.update(url=url, token=token, body=body)
        return {"accepted": len(body["results"]), "rejected": 0}

    monkeypatch.setenv("REVIEW_CENTRAL_URL", "https://dev.hugpy.ai/api")
    monkeypatch.setenv("REVIEW_CENTRAL_TOKEN", "hpw_secret")
    monkeypatch.setattr(push, "_post", fake_post)
    run_id = store.start_run("nightly")
    store.finish_run(run_id)
    assert push.push_run(run_id)["ok"] is True
    assert seen["url"] == "https://dev.hugpy.ai/api/llm/review/ingest"
    assert seen["token"] == "hpw_secret"


def test_a_missing_run_is_reported_not_raised(review_db, monkeypatch):
    from hugpy_curation.review import push
    monkeypatch.setenv("REVIEW_CENTRAL_URL", "https://central.invalid/api")
    assert push.push_run(999999)["reason"] == "no_such_run"


def test_old_dbs_gain_the_ingest_columns(tmp_path, monkeypatch):
    """A box that has been reviewing for weeks must not need a hand migration
    for the first push to work."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE reviews (id INTEGER PRIMARY KEY AUTOINCREMENT,
            criteria TEXT NOT NULL, hub_id TEXT NOT NULL, stage TEXT NOT NULL,
            passed INTEGER, score REAL, verdict TEXT, payload TEXT NOT NULL,
            reviewed_at REAL NOT NULL);
        CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT,
            criteria TEXT NOT NULL, started_at REAL NOT NULL, finished_at REAL,
            screened INTEGER DEFAULT 0, passed INTEGER DEFAULT 0,
            downloaded INTEGER DEFAULT 0, smoked INTEGER DEFAULT 0, error TEXT);
        INSERT INTO reviews (criteria, hub_id, stage, payload, reviewed_at)
            VALUES ('nightly','org/old','screened','{}', 1.0);
    """)
    con.commit()
    con.close()
    monkeypatch.setenv("REVIEW_DB", str(path))
    from hugpy_curation.review import store
    assert store.ingest_run("ae", {"run_id": 3, "criteria": "nightly"})
    assert store.ingest_results("ae", [{"hub_id": "org/n", "stage": "screened",
                                        "criteria": "nightly"}]) == (1, 0)
    # the pre-existing row is untouched
    assert _count(str(path), "reviews") == 2


def test_review_db_env_is_what_the_ae_unit_sets(tmp_path, monkeypatch):
    """The user unit hands the pipeline REVIEW_DB; if that knob ever stopped
    being honoured, the nightly run would silently write somewhere else."""
    target = tmp_path / "nested" / "reviews.db"
    monkeypatch.setenv("REVIEW_DB", str(target))
    from hugpy_curation.review import store
    assert store.db_path() == str(target)
    store.start_run("nightly")
    assert os.path.exists(target)
