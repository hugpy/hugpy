"""GET /llm/models/status — the per-model "worth my time" row: worth rules,
explicit-missing states, filters. Pure builder + the route with every source
monkeypatched (no DB, no audit file, no workers)."""
from __future__ import annotations

import json

import pytest
from flask import Flask

from hugpy_server.app.routes import model_status_routes as ms

GB = 10 ** 9
AEB = {"name": "aeb", "status": "online", "gpu_total_bytes_known": 24 * GB,
       "ram_total_bytes_known": 128 * GB, "models_local": ["Hot-Text"], "loaded_models": ["Hot-Text"]}
TINY = {"name": "tiny", "status": "online", "gpu_total_bytes_known": 8 * GB,
        "ram_total_bytes_known": 16 * GB}


def cat(key, task="text-generation", **kw):
    row = {"model_key": key, "hub_id": f"org/{key}", "framework": "gguf", "primary_task": task,
           "tasks": [task], "status": "installed", "size_bytes": 4 * GB,
           "effective_gguf": f"{key.lower()}-q4_k_m.gguf",
           "admission": {"status": "pending", "reason": "static_ok; no aptitude grade yet",
                         "at": "2026-09-23T20:37:20+00:00", "integrity": "static_ok", "job": "seed"}}
    row.update(kw)
    return row


def audit(**verdicts):
    return {"generated_at": 1790000000.0,
            "models": {k: {"verdict": v, "why": f"why {k}", "log": [f"log {k}"],
                           "findings": [{"check": "x", "verdict": v, "detail": "d"}],
                           "suggested_fix": f"fix {k}" if v not in ("static_ok",) else None}
                       for k, v in verdicts.items()}}


def grade_row(name, grade, suite="hugpy-native-v2", at=1790000100.0, detail=None, worker="aeb"):
    return {"model_name": name, "grade": grade, "grade_suite": suite, "graded_at": str(at),
            "worker": worker, "quant": "Q4_K_M", "grade_detail": json.dumps(detail or {})}


FULL = {"tier": 3, "max": 3, "history": [{"tier": t, "pass": True} for t in ("easy", "medium", "hard")]}
TEXT_TASKS = ["math", "wordprob", "factual", "format_primes", "logic", "exact_instruction", "coding", "json", "letters"]
# 8 tasks full + coding 1 of 3 = 25 of 27 items
TIERS = {**{t: FULL for t in TEXT_TASKS},
         "coding": {"tier": 1, "max": 3, "history": [{"tier": "easy", "pass": True}, {"tier": "medium", "pass": False},
                                                     {"tier": "hard", "pass": False}]}}


def build(catalog, *, workers=(AEB,), metrics=(), audit_doc=None, failures=(), detail=False):
    return {r["model_key"]: r for r in ms.build_status_rows(
        catalog, workers=list(workers), metrics_rows=list(metrics), audit_doc=audit_doc,
        failures=list(failures), detail=detail)}


# ── worth rules, one per label, in rule order ───────────────────────────────

def test_ready_needs_verified_graded_and_fit():
    r = build([cat("Hot-Text")], audit_doc=audit(**{"Hot-Text": "static_ok"}),
              metrics=[grade_row("org~Hot-Text", 88, detail=TIERS)])["Hot-Text"]
    assert r["worth"]["label"] == "ready" and r["worth"]["bucket"] == "ready"
    assert r["servable"]["now"] is True and r["servable"]["where"] == ["aeb"]
    g = r["grade"]
    # score/max (pct) is the visible PASS count over the suite's own 27 items
    assert (g["suite"], g["score"], g["max"], g["worker"], g["quant"]) == ("hugpy-native-v2", 25, 27, "aeb", "Q4_K_M")
    assert g["text"] == "25/27 (93%)" and g["value"] == pytest.approx(92.6) and g["score_from_pct"] is False
    assert g["detail_summary"] == "25/27 tiers · 8/9 tasks full"
    assert r["quant_effective"] == "Q4_K_M"


def test_weak_below_threshold():
    r = build([cat("M")], audit_doc=audit(M="static_ok"), metrics=[grade_row("M", 20)])["M"]
    assert r["worth"]["label"] == "weak" and r["worth"]["bucket"] == "waste"
    # no per-item detail: derived from the percent and SAID so
    assert r["grade"]["text"] == "5/27 (20%) · from %, no per-item detail" and r["grade"]["score_from_pct"] is True


def test_ungraded_is_explicit_not_zero():
    r = build([cat("M")], audit_doc=audit(M="static_ok"))["M"]
    g = r["grade"]
    assert g["suite"] == "hugpy-native-v2" and g["value"] is None and g["max"] == 27 and g["score"] is None
    assert g["reason"].startswith(("not graded", "no hugpy-native-v2 grade row"))
    r = build([cat("Vid", task="text-to-video")], audit_doc=audit(Vid="suite_mismatch"),
              metrics=[grade_row("Vid", 0)])["Vid"]
    assert r["grade"]["suite"] is None and r["grade"]["value"] is None
    assert r["grade"]["reason"].startswith(("no grader for text-to-video", "no grading suite is registered for task 'text-to-video'"))
    assert r["grade"]["stray"][0]["suite"] == "hugpy-native-v2"      # shown, never counted
    assert r["worth"]["label"] == "no-grader" and r["worth"]["bucket"] == "unknown"


def test_unverified_when_no_source_at_all():
    r = build([cat("M", admission=None)])["M"]
    v = r["verification"]
    assert v["source"] == "none" and v["verdict"] is None and "never verified" in v["why"]
    assert r["admission"]["status"] == "none" and "no admission record" in r["admission"]["reason"]
    assert r["worth"]["label"] == "unverified"


def test_admission_integrity_is_a_verification_source_of_last_resort():
    r = build([cat("M")])["M"]
    assert r["verification"]["source"] == "admission" and r["verification"]["verdict"] == "static_ok"


def test_unservable_reasons():
    rows = build([cat("Blk", blocked=True), cat("Part", status="partial"),
                  cat("Huge", size_bytes=900 * GB), cat("Mis"),
                  cat("Adapter", extra={"serveable": False, "unserveable_reason": "LoRA adapter"})],
                 audit_doc=audit(Blk="static_ok", Part="static_ok", Huge="static_ok", Mis="misconfigured",
                                 Adapter="static_ok"))
    assert {k: r["worth"]["label"] for k, r in rows.items()} == dict.fromkeys(rows, "unservable")
    assert rows["Huge"]["servable"]["fits"] == [] and "fits no known worker" in rows["Huge"]["worth"]["why"]
    assert "LoRA adapter" in rows["Adapter"]["worth"]["why"]
    assert rows["Mis"]["worth"]["action"] == "fix Mis"               # the audit's suggested fix


def test_rule_order_broken_beats_held_beats_unservable():
    held = {"status": "held", "reason": "load failure on aeb", "at": 1}
    rows = build([cat("B", admission=held), cat("H", admission=held, blocked=True)],
                 audit_doc=audit(B="broken_download", H="static_ok"))
    assert rows["B"]["worth"]["label"] == "broken"
    assert rows["H"]["worth"]["label"] == "held" and rows["H"]["worth"]["why"] == "load failure on aeb"


def test_fit_offload_and_offline_worker():
    off = {**TINY, "name": "big", "status": "offline", "gpu_total_bytes_known": 80 * GB}
    r = build([cat("M", size_bytes=30 * GB)], workers=[AEB, off], audit_doc=audit(M="static_ok"))["M"]
    assert r["servable"]["fits"] == ["aeb"]                           # 24 GB GPU + RAM offload
    assert r["servable"]["fits_offline"] == ["big"]
    assert r["servable"]["now"] is False and "would fit on aeb" in r["servable"]["reason"]


def test_integrity_row_newer_than_audit_wins_but_keeps_audit_fix():
    irow = {"model_name": "M", "grade": 0, "grade_suite": "integrity", "graded_at": "1790000500",
            "grade_detail": json.dumps({"verdict": "faulty_model", "why": "bad dims"})}
    r = build([cat("M")], audit_doc=audit(M="misconfigured"), metrics=[irow], detail=True)["M"]
    assert r["verification"]["source"] == "integrity-grade" and r["verification"]["verdict"] == "faulty_model"
    assert r["worth"]["label"] == "broken"
    assert r["detail"]["verification"]["suggested_fix"] == "fix M"


def test_last_failure_first_line_and_refusal():
    fails = [{"model": "M", "ts": 10, "action": "load", "outcome": "fail", "worker_card": "aeb:0",
              "detail": {"class": "hard_load_failure",
                         "loader_stderr": "0.1 I loading\n0.2 E llama_model_load: error loading model: dims\n"}},
             {"model": "org~M", "ts": 20, "action": "call", "outcome": "refused",
              "detail": {"predicate": "admission_held", "request_id": "req-1", "message": "held"}}]
    r = build([cat("M")], audit_doc=audit(M="static_ok"), failures=fails)["M"]
    assert r["last_failure"]["kind"] == "routing refusal" and r["last_failure"]["request_id"] == "req-1"
    only_load = build([cat("M")], audit_doc=audit(M="static_ok"), failures=fails[:1])["M"]["last_failure"]
    assert only_load["first_line"].startswith("0.2 E llama_model_load") and only_load["worker"] == "aeb"
    assert build([cat("M")], audit_doc=audit(M="static_ok"))["M"]["last_failure"] is None


def test_provisioning_shows_progress_while_not_servable():
    w = {**AEB, "models_local": [], "provisioning": ["M"],
         "provision_progress": {"M": {"done_bytes": 12.4 * GB, "total_bytes": 21.6 * GB, "frac": 0.574}}}
    s = build([cat("M")], workers=[w], audit_doc=audit(M="static_ok"))["M"]["servable"]
    assert s["now"] is False and s["provisioning"][0]["worker"] == "aeb"
    assert s["provisioning"][0]["since"] is None
    assert s["reason"] == "provisioning on aeb 57% (12.4/21.6 GB)"


def test_no_blank_fields_for_any_label():
    rows = build([cat("A"), cat("B", admission=None), cat("V", task="text-to-video")],
                 audit_doc=audit(A="static_ok", V="static_ok"))
    for r in rows.values():
        assert r["worth"]["label"] in ms.LABELS and r["worth"]["why"] and r["worth"]["action"]
        assert r["verification"]["why"] or r["verification"]["verdict"]
        assert r["grade"].get("reason") or r["grade"].get("value") is not None
        assert r["servable"]["reason"]


# ── the route ───────────────────────────────────────────────────────────────

@pytest.fixture()
def client(monkeypatch):
    catalog = [cat("Hot-Text"), cat("Vid", task="text-to-video"), cat("Plain")]
    monkeypatch.setattr(ms, "load_catalog", lambda: [dict(c) for c in catalog])
    monkeypatch.setattr(ms, "load_audit", lambda: (audit(**{"Hot-Text": "static_ok", "Vid": "static_ok"}),
                                                   {"error": None}))
    monkeypatch.setattr(ms, "load_grade_rows", lambda: ([grade_row("Hot-Text", 90)], {"error": None}))
    monkeypatch.setattr(ms, "load_failures", lambda: ([], {"error": None}))
    monkeypatch.setattr(ms, "load_workers", lambda: ([AEB], {"error": None}))
    monkeypatch.setattr(ms, "load_events", lambda: ([], {"error": None}))
    monkeypatch.setattr(ms, "load_calls", lambda: ([], {"error": None}))
    monkeypatch.setattr(ms, "_inflight_of", lambda w, m: 0)
    monkeypatch.setattr(ms, "load_throughput", lambda: ([
        {"model_name": "Hot-Text", "worker": None, "n_calls": 3, "n_rated": 3, "mean_tok_s": 50.0, "p50": 48.0,
         "p90": 60.0, "min": 40.0, "max": 61.0, "first_at": 1.0, "last_at": 2.0},
        {"model_name": "Hot-Text", "worker": "aeb", "quant": "Q4_K_M", "alloc_mode": "gpu_only", "n_calls": 3,
         "n_rated": 3, "mean_tok_s": 50.0}], {"error": None}))
    ms.clear_cache()
    app = Flask(__name__)
    app.register_blueprint(ms.model_status_bp)
    yield app.test_client()
    ms.clear_cache()


def test_route_one_row_per_model_with_counts(client):
    d = client.get("/llm/models/status").get_json()
    assert d["count"] == 3 and {r["model_key"] for r in d["models"]} == {"Hot-Text", "Vid", "Plain"}
    assert d["counts"]["ready"] == 1 and d["counts"]["no-grader"] == 1
    assert d["buckets"] == {"ready": 1, "waste": 0, "unknown": 2}
    assert d["rules"]["ready_min_grade"] == ms.READY_MIN_GRADE


def test_route_model_filter_accepts_alias_forms(client):
    for q in ("Hot-Text", "org/Hot-Text", "org~Hot-Text"):
        d = client.get(f"/llm/models/status?model={q}").get_json()
        assert [r["model_key"] for r in d["models"]] == ["Hot-Text"], q
    assert client.get("/llm/models/status?model=nope").get_json()["models"] == []


def test_route_worth_filter_label_or_bucket(client):
    d = client.get("/llm/models/status?worth=ready").get_json()
    assert [r["model_key"] for r in d["models"]] == ["Hot-Text"]
    d = client.get("/llm/models/status?worth=unknown").get_json()
    assert {r["model_key"] for r in d["models"]} == {"Vid", "Plain"}
    assert d["counts"]["ready"] == 1                                  # counts are pre-filter


def test_route_detail(client):
    d = client.get("/llm/models/status?model=Hot-Text&detail=1").get_json()["models"][0]["detail"]
    assert d["verification"]["log"] == ["log Hot-Text"]
    assert d["grade_rows"][0]["grade"] == 90 and d["fit"][0]["fit"] == "gpu"


def test_route_never_500s_on_catalog_fault(client, monkeypatch):
    def boom():
        raise RuntimeError("store down")
    monkeypatch.setattr(ms, "load_catalog", boom)
    r = client.get("/llm/models/status")
    assert r.status_code == 200 and r.get_json()["sources"]["catalog"]["error"].startswith("RuntimeError")


# ── per-(model, worker) live state: every state + overlays ──────────────────

NOW = 1790000000.0
M = {"model_key": "M", "hub_id": "org/M"}


def W(**kw):
    base = {"id": "wid-aeb", "name": "aeb", "status": "online", "models_local": [], "loaded_models": [],
            "loading": [], "slots": [], "provisioning": [], "provision_progress": {}, "load_reports": {}}
    base.update(kw)
    return base


def st(w, **kw):
    kw.setdefault("now", NOW)
    return ms.model_worker_state(M, w, None, kw.pop("actions", ()), **kw)


def test_state_on_central_when_installed_or_has_bytes():
    # Not on THIS worker's drive, but the catalog row is installed on central (or
    # records bytes): the worker copies it on first call — NOT missing.
    s = ms.model_worker_state({**M, "status": "installed"}, W(), now=NOW)
    assert (s["state"], s["base"]) == ("on central", "on central")
    assert "central storage" in s["detail"] and "first call" in s["detail"] and "not missing" in s["detail"]
    assert ms.model_worker_state({**M, "dir_bytes": 4 * GB}, W(), now=NOW)["base"] == "on central"


def test_state_missing_only_when_nowhere_and_allocated():
    # On no drive (no status/dir_bytes anywhere) AND designated/assigned to this
    # worker: the one alarming state.
    s = ms.model_worker_state(M, W(models=["M"]), now=NOW)
    assert (s["state"], s["base"], s["label"]) == ("missing", "missing", "missing")
    assert "allocated to this worker" in s["detail"] and "NO drive" in s["detail"]
    # the unified allocations view and model_alloc_modes count as allocation too
    assert ms.model_worker_state(M, W(allocations=[{"model_key": "org~M"}]), now=NOW)["base"] == "missing"
    assert ms.model_worker_state(M, W(model_alloc_modes={"M": "gpu-only"}), now=NOW)["base"] == "missing"


def test_state_not_allocated_when_nowhere_and_unassigned():
    # On no drive and not allocated here: neutral, non-alarming.
    s = st(W())
    assert (s["state"], s["base"], s["label"]) == ("not allocated", "not allocated", "not allocated")
    assert "not allocated to this worker" in s["detail"] and "nothing wrong" in s["detail"]


def test_failed_overlay_applies_over_on_central():
    # A failed load attempt must still surface even when the model is otherwise
    # "on central" (operator: the overlay applies to the new state).
    reports = {"M": {"ok": False, "ts": NOW - 5, "error": "LoadRefusal: won't fit on GPU"}}
    s = ms.model_worker_state({**M, "status": "installed"}, W(load_reports=reports), now=NOW)
    assert s["state"] == "failed" and s["base"] == "on central" and s["label"] == "failed: LoadRefusal"


def test_state_cold_from_models_local_or_catalog_disk_row():
    assert st(W(models_local=["org~M"]))["state"] == "cold"
    row = {**M, "workers": [{"worker": "aeb", "on_disk_bytes": 4 * GB}]}
    s = ms.model_worker_state(row, W(), now=NOW)
    assert s["state"] == "cold" and "4.0 GB" in s["detail"]


def test_state_downloading_with_progress_and_elapsed():
    w = W(provisioning=["M"], provision_progress={"M": {"done_bytes": 12.4e9, "total_bytes": 21.6e9, "frac": 0.574}})
    ev = [{"stage": "provision.start", "model_key": "M", "worker_id": "wid-aeb", "ts": NOW - 80,
           "source": "central-transfer"}]
    s = st(w, events=ev)
    assert s["state"] == "downloading from central"
    assert s["detail"].startswith("12.4/21.6 GB 57% · 1m20s elapsed")
    assert s["progress"]["since"] == NOW - 80
    assert "elapsed unknown" in st(w)["detail"]                       # no event: said, not faked
    assert st(w, events=ev)["detail"].endswith("(worker heartbeat — no central transfer-ledger entry)")


def test_state_loading_three_ways():
    assert st(W(loading=["M"]))["state"] == "loading"
    slot = {"slot_id": "2", "model_key": "M", "healthy": False, "busy": False}
    assert st(W(slots=[slot]))["detail"] == "slot 2 starting (not healthy yet)"
    ev = [{"stage": "load.start", "model_key": "M", "worker_id": "wid-aeb", "ts": NOW - 12}]
    assert st(W(models_local=["M"]), events=ev)["state"] == "loading"
    done = ev + [{"stage": "load.done", "model_key": "M", "worker_id": "wid-aeb", "ts": NOW - 5}]
    assert st(W(models_local=["M"]), events=done)["state"] == "cold"


def test_state_hot_serving_answering():
    slot = {"slot_id": "1", "model_key": "M", "healthy": True, "busy": False, "ctx": 16384,
            "last_used": NOW - 300}
    hot = st(W(slots=[slot]))
    assert hot["state"] == "hot" and hot["detail"].startswith("loaded in slot 1, ctx 16384 · idle")
    assert st(W(loaded_models=["M"]))["detail"].startswith("loaded in-process")
    busy = {**slot, "busy": True}
    s = st(W(slots=[busy]))
    assert s["state"] == "serving" and "not distinguishable" in s["detail"]
    assert st(W(loaded_models=["M"]), inflight=2)["state"] == "serving"
    s = st(W(slots=[{**busy, "last_used": NOW - 0.5}]))
    assert s["state"] == "answering" and "0.5s ago" in s["detail"]
    call = [{"action": "call", "model": "M", "worker_card": "aeb:0", "ts": NOW - 1.0}]
    assert st(W(slots=[busy]), actions=call)["state"] == "answering"


def test_overlay_failed_only_over_cold_or_missing():
    reports = {"M": {"ok": False, "ts": NOW - 30,
                     "error": "LoadRefusal: won't fit on GPU: needs 22.8 GB, 21.1 GB free"}}
    s = st(W(models_local=["M"], load_reports=reports))
    assert s["state"] == "failed" and s["base"] == "cold" and s["label"] == "failed: LoadRefusal"
    assert "won't fit on GPU" in s["detail"]
    fail = [{"action": "load", "outcome": "fail", "model": "M", "worker_card": "aeb:0", "ts": NOW - 10,
             "detail": {"class": "hard_load_failure",
                        "loader_stderr": "x\nE llama_model_load: error loading model: dims"}}]
    s = st(W(models_local=["M"], load_reports=reports), actions=fail)
    assert s["label"] == "failed: hard_load_failure" and s["failed"]["first_line"].startswith("E llama_model_load")
    ok_later = {"M": {"ok": True, "ts": NOW - 1}}
    assert st(W(models_local=["M"], load_reports=ok_later), actions=fail)["state"] == "cold"
    assert st(W(loaded_models=["M"], load_reports=reports))["state"] == "hot"
    lazy = {"M": {"ok": False, "ts": NOW, "error": "not local — probe does not download (lazy doctrine)"}}
    assert st(W(load_reports=lazy))["state"] == "not allocated"       # a declined probe is no attempt
    odd = {"M": {"ok": False, "ts": NOW, "error": "vision model loaded in-process (text-only)"}}
    assert st(W(load_reports=odd))["label"] == "failed: load_failure"


def test_overlay_held_and_offline_worker():
    s = st(W(loaded_models=["M"]), held=True)
    assert s["state"] == "hot" and s["held"] is True and s["label"] == "hot · held"
    s = st(W(status="offline", models_local=["M"]))
    assert s["online"] is False and s["detail"].startswith("worker offline — last known: on disk")


def test_other_workers_rows_do_not_leak():
    fail = [{"action": "load", "outcome": "fail", "model": "M", "worker_card": "computron:0", "ts": NOW}]
    ev = [{"stage": "load.start", "model_key": "M", "worker_id": "wid-computron", "ts": NOW}]
    assert st(W(models_local=["M"]), actions=fail, events=ev)["state"] == "cold"


def test_status_rows_carry_worker_states_and_servable_uses_them():
    comp = {**TINY, "id": "c", "models_local": ["M"], "loaded_models": []}
    r = build([cat("M")], workers=[{**AEB, "loaded_models": ["M"]}, comp], audit_doc=audit(M="static_ok"))["M"]
    assert [(w["worker"], w["state"]) for w in r["workers"]] == [("aeb", "hot"), ("tiny", "cold")]
    assert r["servable"]["where"] == ["aeb"] and r["servable"]["cold"] == ["tiny"]


def test_waste_bucket_disk_absent_states_become_na():
    # A blocked model is in the "waste" bucket, not expected on any worker: its
    # disk-absent worker rows (on central / not allocated / missing) collapse to
    # n/a instead of an alarming state (operator 2026-09-24).
    r = build([cat("Blk", blocked=True)], audit_doc=audit(Blk="static_ok"))["Blk"]
    assert r["worth"]["bucket"] == "waste"
    aeb = next(w for w in r["workers"] if w["worker"] == "aeb")
    assert (aeb["state"], aeb["base"], aeb["label"]) == ("n/a", "n/a", "n/a")
    assert aeb["detail"].startswith("not expected on workers")


def test_route_live_poll_hint(client):
    d = client.get("/llm/models/status").get_json()
    assert d["live"] is False and d["poll_s"] == 15 and "answering" in d["worker_states"]


def test_route_throughput_is_mean_of_calls_or_explicit_none(client):
    d = {r["model_key"]: r for r in client.get("/llm/models/status").get_json()["models"]}
    tp = d["Hot-Text"]["throughput"]
    # mean-of-calls (Σtok/Σgen-s over n_rated of n_calls), never an EMA / last call.
    assert (tp["n_calls"], tp["n_rated"], tp["mean_tok_s"]) == (3, 3, 50.0)
    assert tp["label"] == "Σtok/Σgen-s over 3 of 3 calls"
    assert "tok_per_s_avg" not in tp and "ema" not in str(tp).lower()
    assert d["Plain"]["throughput"]["n_calls"] == 0 and d["Plain"]["throughput"]["mean_tok_s"] is None and d["Plain"]["throughput"]["reason"].startswith("no calls recorded")
    det = client.get("/llm/models/status?model=Hot-Text&detail=1").get_json()["models"][0]["detail"]
    assert det["throughput_cells"][0]["worker"] == "aeb" and det["throughput_cells"][0]["n_calls"] == 3


def test_comfy_checkpoint_on_central_is_not_missing():
    # A ComfyUI checkpoint absent from the worker's ComfyUI but present on
    # central follows the same disk-truth rule: "on central", never "missing".
    fw = next(iter(ms.CENTRAL_SERVED_FRAMEWORKS))
    row = {**M, "framework": fw, "filename": "x.safetensors",
           "central_file": {"exists": True, "path": "/c/x.safetensors", "bytes": 2 * GB}}
    s = ms.model_worker_state(row, W(comfy={"available": False, "url": "http://w:8188"}), now=NOW)
    assert s["base"] == "on central" and "checkpoint on central" in s["detail"]
    gone = {**row, "central_file": {"exists": False, "path": "/c/x.safetensors"}}
    assert ms.model_worker_state(gone, W(comfy={"available": False}), now=NOW)["base"] == "not allocated"
    assert ms.model_worker_state(gone, W(models=["M"], comfy={"available": False}), now=NOW)["base"] == "missing"
