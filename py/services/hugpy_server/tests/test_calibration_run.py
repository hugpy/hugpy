"""Calibrate run orchestration (2026-10-02): precheck, predict, load, measure,
unload, measure leftovers, verdict. Worker HTTP, the /v1 chat and the DB
record are faked; the decisions are the real code."""
import time

from hugpy_server.app import calibration_run as cr

G = 2 ** 30


class _Fake:
    def __init__(self, residents, used0, used_loaded, used_after, mine_bytes, gate_need, static=False):
        self.res = residents
        self.loaded = False
        self.used = [used0, used_loaded, used_after]
        self.mine = mine_bytes
        self.gate = gate_need
        self.evicted = []
        self.margins = []
        self.action = "proceed"

    def get(self, worker, path, params=None, timeout=20):
        if path == "/ops/residents":
            rs = list(self.res)
            if self.loaded:
                rs.append({"model_key": "M", "vram_bytes": self.mine, "host_mode": "in_process", "residency": "on-demand"})
            return {"residents": rs}
        if path == "/ops/vram-holders":
            u = self.used[0] if not self.loaded else self.used[1]
            if self.evicted and "M" in self.evicted:
                u = self.used[2]
            return {"vram_used_bytes": u, "vram_free_bytes": 24 * G - u, "vram_total_bytes": 24 * G, "cards": [], "holders": []}
        if path.startswith("/fit-preview/"):
            return {"need_bytes": self.gate, "weights_bytes": self.gate, "kv_bytes": 0, "action": self.action,
                    "ctx": {"resolved": 2048, "source": "ctx_pct"}, "need_detail": {"weights_file": "m.gguf"}}
        raise AssertionError(path)

    def post(self, worker, path, body, timeout=120):
        if path == "/ops/weights-margin":
            self.margins.append(body)
            return 200, {"ok": True, "record": {"margin": 0.96}}
        assert path == "/ops/evict"
        self.evicted.append(body["model_key"])
        self.res = [r for r in self.res if r["model_key"] != body["model_key"]]
        return 200, {"ok": True}


def _run(monkeypatch, fake, evict_others=False):
    monkeypatch.setattr(cr, "SETTLE_S", 0)
    monkeypatch.setattr(cr, "_worker_get", fake.get)
    monkeypatch.setattr(cr, "_worker_post", fake.post)
    monkeypatch.setattr(cr, "_record", lambda row: 1)

    def chat(mk, name):
        fake.loaded = True
        return {"choices": [{}], "_took_s": 1.0}
    monkeypatch.setattr(cr, "_chat", chat)
    job = cr.start({"id": "w1", "name": "ae-worker"}, "M", evict_others=evict_others)
    for _ in range(200):
        j = cr.get_job(job["job_id"])
        if j["status"] != "running":
            return j
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_busy_card_refuses_without_evict_others(monkeypatch):
    f = _Fake([{"model_key": "Other", "residency": "on-demand", "vram_bytes": 5 * G}], G, 10 * G, G, 9 * G, 9 * G)
    j = _run(monkeypatch, f)
    assert j["status"] == "error" and "card not idle" in j["error"] and not f.evicted


def test_static_resident_is_never_evicted(monkeypatch):
    f = _Fake([{"model_key": "Pinned", "residency": "static", "vram_bytes": 5 * G}], G, 10 * G, G, 9 * G, 9 * G)
    j = _run(monkeypatch, f, evict_others=True)
    assert j["status"] == "error" and "STATIC" in j["error"] and not f.evicted


def test_clean_run_agrees_and_clears_on_demand_neighbours(monkeypatch):
    f = _Fake([{"model_key": "Other", "residency": "on-demand", "vram_bytes": 5 * G}], G, 10 * G, G, 9 * G, int(9.3 * G))
    j = _run(monkeypatch, f, evict_others=True)
    assert j["status"] == "done", j
    assert f.evicted == ["Other", "M"]
    r = j["result"]
    assert r["verdict"] == "agree" and r["measured"]["leftover_after_unload_bytes"] == 0


def test_unload_that_leaves_memory_is_a_leak(monkeypatch):
    f = _Fake([], G, 18 * G, 17 * G, 17 * G, 17 * G)
    j = _run(monkeypatch, f)
    assert j["status"] == "done" and j["result"]["verdict"] == "leak"


def test_gate_far_from_measured_is_a_disagreement(monkeypatch):
    f = _Fake([], G, 10 * G, G, 9 * G, 30 * G)
    j = _run(monkeypatch, f)
    assert j["result"]["verdict"] == "disagree" and j["result"]["detail"]["gate_error_pct"] > 100


def test_full_load_records_the_measured_margin_before_unloading(monkeypatch):
    """2026-10-02: a planned-full load measured alone becomes the gate's measured
    margin — the delta goes to the worker BEFORE the eviction; a planned partial
    never records (only part of the weights were on the card)."""
    fake = _Fake([], 1 * G, 20 * G, 1 * G, 19 * G, 21 * G)
    j = _run(monkeypatch, fake)
    assert fake.margins == [{"model_key": "M", "delta_bytes": 19 * G, "ctx": 2048}]
    assert j["result"]["measured"]["margin"]["ok"] is True
    part = _Fake([], 1 * G, 15 * G, 1 * G, 14 * G, 25 * G)
    part.action = "partial"
    _run(monkeypatch, part)
    assert part.margins == []
