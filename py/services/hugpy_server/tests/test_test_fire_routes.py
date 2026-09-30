"""Per-worker TEST FIRE (test_fire_routes): model selection + result bookkeeping.

Operator ask 2026-09-29: "a test button where it starts to randomly call every
model on that worker which does text gen". No network, no worker, no engine:
the call function is injected, the catalog / worker / blocklist collaborators
are faked on the module.

Pinned here:
  * SELECTION reads the catalog's ``tasks``/``primary_task``: text-generation /
    text2text-generation rows are targets, including a VL row that also lists
    text-generation; rows with only vision / image / video / audio / embedding
    tasks are NOT; operator-blocked, archive-marked and
    off-catalog keys are skipped with a reason; candidates come from the
    worker's designations + local files + storage rows, de-duplicated.
  * RUN LOOP: every model exactly once per round in a seeded random order (a
    different order per round), concurrency=1 is strictly sequential, rounds=0
    runs until stopped, stop lets the in-flight call finish, a raising call is
    a recorded failure (never a crashed loop), results ring holds 200.
  * ROUTES: POST 202 {job_id}; 409 while one is running on the worker; 404 for
    an unknown worker; 400 for a bad body / a worker with no text-gen model;
    GET status carries running/round/done_calls/total_planned/results/summary;
    POST stop requests a stop and relays the jobs cancel for in-flight ids.
"""
from __future__ import annotations

import importlib
import threading
import time

import pytest
from flask import Flask

tf = importlib.import_module("hugpy_server.app.routes.test_fire_routes")

CATALOG = {
    "tg-a": {"model_key": "tg-a", "framework": "gguf",
             "primary_task": "text-generation", "tasks": ["text-generation"]},
    "tg-b": {"model_key": "tg-b", "framework": "transformers",
             "primary_task": "text-generation", "tasks": ["text-generation"]},
    "t2t": {"model_key": "t2t", "framework": "transformers",
            "primary_task": "text-summarization",
            "tasks": ["text-summarization", "text2text-generation"]},
    "vl": {"model_key": "vl", "framework": "gguf",
           "primary_task": "image-text-to-text",
           "tasks": ["image-text-to-text", "text-generation"]},
    "vl2": {"model_key": "vl2", "framework": "transformers",
            "primary_task": "text-generation",
            "tasks": ["text-generation", "image-text-to-text"]},
    "img": {"model_key": "img", "framework": "transformers",
            "primary_task": "text-to-image", "tasks": ["text-to-image", "image-to-image"]},
    "asr": {"model_key": "asr", "framework": "transformers",
            "primary_task": "automatic-speech-recognition",
            "tasks": ["automatic-speech-recognition"]},
    "emb": {"model_key": "emb", "framework": "transformers",
            "primary_task": "feature-extraction",
            "tasks": ["feature-extraction", "sentence-similarity"]},
    "blocked-tg": {"model_key": "blocked-tg", "framework": "gguf",
                   "primary_task": "text-generation", "tasks": ["text-generation"],
                   "blocked": True},
    "owner~qual-tg": {"model_key": "owner~qual-tg", "framework": "gguf",
                      "primary_task": "text-generation", "tasks": ["text-generation"]},
    # PEFT/LoRA adapter rows (base_model set). The engine marks an adapter
    # whose base is not in CENTRAL's store extra.serveable=False with the
    # "PEFT adapter (base '…') — … NOT in this store" refusal.
    "sentiment-lora": {"model_key": "sentiment-lora", "framework": "transformers",
                       "primary_task": "text-generation", "tasks": ["text-generation"],
                       "base_model": "unsloth/Llama-3.2-3B-Instruct",
                       "extra": {"serveable": False,
                                 "unserveable_reason": "PEFT adapter (base 'unsloth/Llama-3.2-3B-Instruct') — the adapter is on disk but its base model is NOT in this store, and an adapter cannot be loaded without it. FIX: acquire it."}},
    # adapter whose base IS on the worker (hub id spelling vs bare key)
    "stage1-lora": {"model_key": "stage1-lora", "framework": "transformers",
                    "primary_task": "text-generation", "tasks": ["text-generation"],
                    "base_model": "Qwen/Qwen3.5-0.8B", "extra": {"serveable": True}},
    "Qwen3.5-0.8B": {"model_key": "Qwen3.5-0.8B", "framework": "transformers",
                     "primary_task": "text-generation", "tasks": ["text-generation"],
                     "extra": {"serveable": True}},
    # adapter central CAN serve (base in central's store) but the base is not
    # in THIS worker's store
    "med-qlora": {"model_key": "med-qlora", "framework": "transformers",
                  "primary_task": "text-generation", "tasks": ["text-generation"],
                  "base_model": "microsoft/biogpt-large", "extra": {"serveable": True}},
}

WORKER = {
    "id": "wid", "name": "box",
    "models": ["tg-a", "vl", "img", "blocked-tg", "qual-tg", "ghost"],
    "models_local": ["tg-b", "tg-a"],
    "storage": {"models": [{"model_key": "t2t"}, {"model_key": "asr"},
                           {"model_key": "emb"}, {"model_key": "vl2"},
                           {"model_key": "tg-b"}]},
}


# ── selection ────────────────────────────────────────────────────────────────
def test_is_text_gen_reads_tasks_and_primary_task():
    assert tf.is_text_gen(CATALOG["tg-a"])
    assert tf.is_text_gen(CATALOG["t2t"]), "text2text-generation counts as text gen"
    assert tf.is_text_gen(CATALOG["vl"]), "VL + text-generation is text-capable"
    assert tf.is_text_gen(CATALOG["vl2"]), "primary text-gen + VL task is text-capable"
    assert not tf.is_text_gen(CATALOG["img"])
    assert not tf.is_text_gen(CATALOG["asr"])
    assert not tf.is_text_gen(CATALOG["emb"])
    assert not tf.is_text_gen({"tasks": []})
    assert not tf.is_text_gen(None)
    # primary_task alone is enough when tasks is missing
    assert tf.is_text_gen({"primary_task": "text-generation"})


def test_worker_model_keys_unions_designations_local_and_storage_dedup():
    keys = tf.worker_model_keys(WORKER)
    assert keys == ["tg-a", "vl", "img", "blocked-tg", "qual-tg", "ghost",
                    "tg-b", "t2t", "asr", "emb", "vl2"]
    assert len(keys) == len(set(keys))


def test_select_text_gen_models_keeps_text_gen_and_explains_every_skip():
    selected, skipped = tf.select_text_gen_models(
        WORKER, CATALOG, blocked={"blocked-tg"}, archived={"tg-b"})
    assert [m["model_key"] for m in selected] == ["tg-a", "vl", "qual-tg", "t2t", "vl2"]
    assert selected[0] == {"model_key": "tg-a", "framework": "gguf",
                           "tasks": ["text-generation"]}
    reasons = {s["model_key"]: s["reason"] for s in skipped}
    assert reasons["blocked-tg"] == "operator-blocked"
    assert reasons["tg-b"] == "archive-marked"
    assert reasons["ghost"] == "not in catalog"
    assert reasons["img"].startswith("not text-gen:")
    assert reasons["asr"].startswith("not text-gen:")
    assert reasons["emb"].startswith("not text-gen:")
    # the catalog's own blocked flag also skips (no blocklist passed)
    sel2, sk2 = tf.select_text_gen_models(WORKER, CATALOG)
    assert "blocked-tg" not in [m["model_key"] for m in sel2]
    assert {s["model_key"]: s["reason"] for s in sk2}["blocked-tg"] == "operator-blocked"
    assert "tg-b" in [m["model_key"] for m in sel2]


def test_select_tolerates_qualified_vs_bare_spelling():
    # the roster says "qual-tg", the catalog holds "owner~qual-tg"
    selected, _ = tf.select_text_gen_models({"id": "w", "models": ["qual-tg"]}, CATALOG)
    assert [m["model_key"] for m in selected] == ["qual-tg"]
    selected, _ = tf.select_text_gen_models({"id": "w", "models": ["owner~tg-a"]}, CATALOG)
    assert [m["model_key"] for m in selected] == ["owner~tg-a"]


def test_adapter_rows_are_skipped_unless_their_base_is_in_this_workers_store():
    """Coordinator 2026-09-29: a PEFT/LoRA adapter whose base is missing raises
    on the worker ("PEFT adapter (base '…') — … NOT in this store") — a
    deterministic non-load, so it is SKIPPED with a reason, never fired."""
    worker = {"id": "w", "name": "box",
              "models": ["sentiment-lora", "stage1-lora", "med-qlora", "tg-a"],
              "storage": {"models": [{"model_key": "Qwen3.5-0.8B"}]}}
    selected, skipped = tf.select_text_gen_models(worker, CATALOG)
    assert [m["model_key"] for m in selected] == ["stage1-lora", "tg-a", "Qwen3.5-0.8B"], \
        "an adapter whose base is on the worker (Qwen/Qwen3.5-0.8B ~ Qwen3.5-0.8B) fires; the base itself too"
    sk = {s["model_key"]: s for s in skipped}
    # central's own unserveable verdict is carried verbatim
    assert sk["sentiment-lora"]["kind"] == "base missing"
    assert sk["sentiment-lora"]["reason"].startswith("unserveable: PEFT adapter (base 'unsloth/Llama-3.2-3B-Instruct')")
    # serveable centrally, but the base is not in THIS worker's store
    assert sk["med-qlora"]["kind"] == "base missing"
    assert "microsoft/biogpt-large" in sk["med-qlora"]["reason"]
    assert "not in this worker's store" in sk["med-qlora"]["reason"]
    # the same adapter fires on a worker that DOES hold the base
    worker2 = {"id": "w2", "name": "box2", "models": ["med-qlora"],
               "models_local": ["microsoft~biogpt-large"]}
    selected2, skipped2 = tf.select_text_gen_models(worker2, CATALOG)
    assert [m["model_key"] for m in selected2] == ["med-qlora"]
    assert skipped2 == [{"model_key": "microsoft~biogpt-large", "reason": "not in catalog"}], \
        "the base's own row is only a candidate when the catalog knows it"
    # helpers
    assert tf.adapter_base_on_worker("Qwen/Qwen3.5-0.8B", worker)
    assert tf.adapter_base_on_worker("qwen~Qwen3.5-0.8B", worker)
    assert not tf.adapter_base_on_worker("Qwen/Qwen3.5-9B", worker)
    assert not tf.adapter_base_on_worker(None, worker)
    assert tf.unserveable_reason(CATALOG["sentiment-lora"]).startswith("PEFT adapter")
    assert tf.unserveable_reason(CATALOG["tg-a"]) is None
    assert tf.unserveable_reason({"extra": {"serveable": False}}) == "marked unserveable"


def test_classify_error_names_the_refusal_kinds():
    assert tf.classify_error("LoadRefusal: computron 7.74 GiB vs 6.14 GiB") == "LoadRefusal"
    assert tf.classify_error("plan_fit: fit_failure need 9G free 6G") == "fit_failure"
    assert tf.classify_error("worker_busy: slot held 120s") == "worker_busy"
    assert tf.classify_error("requested worker 'box' refused for X: no file") == "requested_worker"
    assert tf.classify_error("something else") == "error"
    assert tf.classify_error(None) is None


# ── run loop / bookkeeping ───────────────────────────────────────────────────
def _fake_call(log, fail=(), raise_on=(), sleep=0.0):
    def call(model_key, prompt, max_tokens, job):
        log.append((job.round, model_key, prompt, max_tokens))
        if sleep:
            time.sleep(sleep)
        if model_key in raise_on:
            raise RuntimeError(f"LoadRefusal: {model_key} does not fit")
        if model_key in fail:
            return {"ok": False, "error": f"worker_busy: {model_key}"}
        return {"ok": True, "tokens": 7, "tok_s": 70.0, "content80": "four"}
    return call


def test_run_job_calls_every_model_once_per_round_in_seeded_random_order():
    log = []
    job = tf.TestFireJob("wid", "box", ["m1", "m2", "m3", "m4", "m5", "m6"],
                         rounds=3, concurrency=1, max_tokens=16, seed=7)
    assert job.total_planned == 18
    tf.run_job(job, _fake_call(log))
    assert job.running is False and job.finished is not None
    assert job.round == 3 and job.done_calls == 18
    orders = [[k for r, k, _, _ in log if r == n] for n in (1, 2, 3)]
    for order in orders:
        assert sorted(order) == ["m1", "m2", "m3", "m4", "m5", "m6"], "each model once per round"
    assert len({tuple(o) for o in orders}) > 1, "shuffled per round"
    assert all(mt == 16 for _, _, _, mt in log)
    assert all(p in tf.PROMPTS for _, _, p, _ in log)
    # deterministic under the same seed
    log2 = []
    job2 = tf.TestFireJob("wid", "box", ["m1", "m2", "m3", "m4", "m5", "m6"],
                          rounds=3, concurrency=1, max_tokens=16, seed=7)
    tf.run_job(job2, _fake_call(log2))
    assert [(r, k) for r, k, _, _ in log2] == [(r, k) for r, k, _, _ in log]
    assert job.stop_reason == "complete"


def test_run_job_records_ok_failed_and_raised_calls_per_model():
    log = []
    job = tf.TestFireJob("wid", "box", ["good", "busy", "boom"], rounds=2, seed=1)
    tf.run_job(job, _fake_call(log, fail={"busy"}, raise_on={"boom"}))
    snap = job.snapshot()
    assert snap["done_calls"] == 6 and snap["total_planned"] == 6
    assert snap["summary"]["ok"] == 2 and snap["summary"]["failed"] == 4
    pm = snap["summary"]["per_model"]
    assert pm["good"]["ok"] == 2 and pm["good"]["last_status"] == "ok"
    assert pm["good"]["last_tok_s"] == 70.0
    assert pm["busy"]["failed"] == 2 and pm["busy"]["last_status"] == "fail"
    assert pm["busy"]["last_error"] == "worker_busy: busy"
    assert pm["busy"]["last_error_kind"] == "worker_busy"
    assert pm["boom"]["failed"] == 2
    assert "LoadRefusal" in pm["boom"]["last_error"]
    assert pm["boom"]["last_error_kind"] == "LoadRefusal", "a raised call is classified too"
    rows = snap["results"]
    assert len(rows) == 6
    for r in rows:
        assert {"model_key", "prompt", "started", "latency_s", "ok", "round"} <= set(r)
        assert r["latency_s"] >= 0
    assert {r["round"] for r in rows} == {1, 2}


def test_results_ring_keeps_only_the_last_200():
    job = tf.TestFireJob("wid", "box", ["m"], rounds=250, seed=3)
    tf.run_job(job, _fake_call([]))
    snap = job.snapshot()
    assert job.done_calls == 250
    assert len(snap["results"]) == tf.RESULT_RING == 200
    assert snap["summary"]["ok"] == 250, "counters are not truncated with the ring"
    assert snap["results"][-1]["round"] == 250


def test_rounds_zero_runs_until_stopped_and_lets_in_flight_finish():
    log = []
    job = tf.TestFireJob("wid", "box", ["m1", "m2"], rounds=0, seed=5)
    assert job.total_planned is None
    started = threading.Event()

    def call(model_key, prompt, max_tokens, j):
        started.set()
        time.sleep(0.02)
        log.append(model_key)
        return {"ok": True}

    t = threading.Thread(target=tf.run_job, args=(job, call), daemon=True)
    t.start()
    assert started.wait(2)
    time.sleep(0.05)
    assert job.running and job.round >= 1
    job.request_stop("operator")
    t.join(3)
    assert not t.is_alive()
    assert job.running is False
    assert job.stop_reason == "operator"
    assert job.done_calls == len(log) >= 1, "the in-flight call finished and was recorded"


def test_concurrency_one_is_strictly_sequential():
    active = {"n": 0, "max": 0}
    lock = threading.Lock()

    def call(model_key, prompt, max_tokens, j):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.01)
        with lock:
            active["n"] -= 1
        return {"ok": True}

    job = tf.TestFireJob("wid", "box", ["a", "b", "c", "d"], rounds=2, concurrency=1)
    tf.run_job(job, call)
    assert active["max"] == 1
    job3 = tf.TestFireJob("wid", "box", ["a", "b", "c", "d"], rounds=2, concurrency=3)
    tf.run_job(job3, call)
    assert 1 < active["max"] <= 3


def test_run_job_with_no_models_ends_immediately():
    job = tf.TestFireJob("wid", "box", [], rounds=1)
    tf.run_job(job, _fake_call([]))
    assert job.running is False and job.done_calls == 0
    assert job.stop_reason == "no text-gen models"


def test_parse_start_body_defaults_and_bounds():
    assert tf.parse_start_body(None) == {"rounds": 1, "concurrency": 1,
                                         "max_tokens": tf.DEFAULT_MAX_TOKENS, "seed": None}
    assert tf.parse_start_body({"rounds": 0, "concurrency": 2, "max_tokens": 8, "seed": 42}) == {
        "rounds": 0, "concurrency": 2, "max_tokens": 8, "seed": 42}
    for bad in ({"rounds": -1}, {"concurrency": 0}, {"max_tokens": 0},
                {"max_tokens": 10_000}, {"rounds": "x"}, {"concurrency": True}):
        with pytest.raises(ValueError):
            tf.parse_start_body(bad)


# ── routes ───────────────────────────────────────────────────────────────────
class Harness:
    def __init__(self, monkeypatch):
        tf.reset_registry()
        self.mp = monkeypatch
        self.cancelled: list = []
        self.calls: list = []
        self.gate = threading.Event()      # released -> fake calls return
        self.gate.set()
        app = Flask(__name__)
        app.register_blueprint(tf.test_fire_bp)
        self.client = app.test_client()
        monkeypatch.setattr(tf, "_get_worker",
                            lambda wid: dict(WORKER) if wid == "wid" else None)
        monkeypatch.setattr(tf, "_catalog", lambda: CATALOG)
        monkeypatch.setattr(tf, "_blocked_keys", lambda: {"blocked-tg"})
        monkeypatch.setattr(tf, "_archived_keys", lambda: set())
        monkeypatch.setattr(tf, "_cancel_request",
                            lambda rid, reason: self.cancelled.append((rid, reason)) or True)
        h = self

        def fake_fire(model_key, prompt, max_tokens, job):
            rid = f"v1-{model_key}-{job.round}"
            job.mark_in_flight(rid, model_key)
            h.calls.append(model_key)
            h.gate.wait(5)
            return {"ok": model_key != "t2t", "request_id": rid, "tokens": 3, "tok_s": 30.0,
                    "error": None if model_key != "t2t" else "fit_failure: need 9G"}
        monkeypatch.setattr(tf, "fire_one", fake_fire)
        # start_job's default call_fn was bound at def time; route through the attr
        orig = tf.start_job
        monkeypatch.setattr(tf, "start_job",
                            lambda *a, **k: orig(*a, **{**k, "call_fn": tf.fire_one}))

    def start(self, wid="wid", **body):
        return self.client.post(f"/llm/workers/{wid}/test-fire", json=body)

    def status(self, job_id, wid="wid"):
        return self.client.get(f"/llm/workers/{wid}/test-fire/{job_id}")

    def stop(self, job_id, wid="wid"):
        return self.client.post(f"/llm/workers/{wid}/test-fire/{job_id}/stop")

    def wait_done(self, job_id, timeout=5):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.status(job_id).get_json()
            if not s["running"]:
                return s
            time.sleep(0.01)
        raise AssertionError("job did not finish")


@pytest.fixture
def h(monkeypatch):
    harness = Harness(monkeypatch)
    yield harness
    harness.gate.set()
    tf.reset_registry()


def test_start_selects_text_gen_models_and_returns_job_id(h):
    r = h.start(rounds=2, max_tokens=8, seed=9)
    assert r.status_code == 202, r.get_json()
    body = r.get_json()
    assert body["ok"] and body["job_id"]
    assert sorted(body["models"]) == ["qual-tg", "t2t", "tg-a", "tg-b", "vl", "vl2"]
    assert body["total_planned"] == 12 and body["rounds"] == 2 and body["max_tokens"] == 8
    skipped = {s["model_key"]: s["reason"] for s in body["skipped"]}
    assert skipped["blocked-tg"] == "operator-blocked"
    assert skipped["img"].startswith("not text-gen")
    s = h.wait_done(body["job_id"])
    assert s["done_calls"] == 12 and s["round"] == 2
    assert s["summary"]["ok"] == 10 and s["summary"]["failed"] == 2
    assert s["summary"]["per_model"]["t2t"]["last_status"] == "fail"
    assert s["summary"]["per_model"]["t2t"]["last_error_kind"] == "fit_failure"
    assert s["summary"]["per_model"]["tg-a"]["last_status"] == "ok"
    assert s["stop_reason"] == "complete" and s["in_flight"] == []
    assert sorted(h.calls) == sorted(["qual-tg", "t2t", "tg-a", "tg-b", "vl", "vl2"] * 2)
    # the worker's current/last job is readable without the id
    cur = h.client.get("/llm/workers/wid/test-fire").get_json()
    assert cur["job_id"] == body["job_id"]


def test_second_start_while_running_is_409_then_stop_cancels_in_flight(h):
    h.gate.clear()                      # first call blocks in flight
    r = h.start(rounds=0)
    assert r.status_code == 202
    jid = r.get_json()["job_id"]
    for _ in range(200):
        if h.calls:
            break
        time.sleep(0.01)
    assert h.calls, "first call is in flight"
    r2 = h.start()
    assert r2.status_code == 409
    assert r2.get_json()["job_id"] == jid
    s = h.status(jid).get_json()
    assert s["running"] and s["total_planned"] is None and len(s["in_flight"]) == 1
    r3 = h.stop(jid)
    assert r3.status_code == 200
    b3 = r3.get_json()
    assert b3["was_running"] and b3["in_flight"] == 1
    assert h.cancelled and h.cancelled[0][0] == s["in_flight"][0]["request_id"]
    h.gate.set()                        # let the in-flight call finish
    s = h.wait_done(jid)
    assert s["stop_reason"] == "operator"
    assert s["done_calls"] == 1, "in-flight finished and was recorded; nothing new started"
    # a new run may start once the previous one has ended
    assert h.start().status_code == 202


def test_start_errors_unknown_worker_bad_body_and_no_text_gen(h):
    assert h.start(wid="nope").status_code == 404
    r = h.start(rounds=-3)
    assert r.status_code == 400 and "rounds" in r.get_json()["error"]
    h.mp.setattr(tf, "_get_worker", lambda wid: {"id": "wid", "name": "box", "models": ["img", "asr"]})
    r = h.start()
    assert r.status_code == 400
    body = r.get_json()
    assert "no text-generation models" in body["error"]
    assert {s["model_key"] for s in body["skipped"]} == {"img", "asr"}


def test_status_and_stop_404_for_unknown_or_foreign_job(h):
    assert h.status("nope").status_code == 404
    assert h.stop("nope").status_code == 404
    r = h.start()
    jid = r.get_json()["job_id"]
    h.wait_done(jid)
    assert h.status(jid, wid="other").status_code == 404, "job ids are scoped to their worker"
