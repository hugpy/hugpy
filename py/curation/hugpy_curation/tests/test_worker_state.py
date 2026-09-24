"""Snapshot/restore of worker allocation state around a benchmark run
(operator ruling 2026-09-24), including disk removal-to-fit for prior models."""
import copy
import json

import pytest

from hugpy_curation.review import worker_state

GB = 10 ** 9


def _stor(rows, disk_free, disk_total=1000 * GB):
    """A worker storage survey: rows = [(model_key, bytes, store, protected)]."""
    return {"disk_free": disk_free, "disk_total": disk_total,
            "models": [{"model_key": mk, "bytes": b, "store": store,
                        "protected": prot, "pinned": False}
                       for (mk, b, store, prot) in rows]}


def _worker(wid, name, **over):
    w = {"id": wid, "name": name, "status": "online", "admission": "approved",
         "unreachable": False, "serve_mode": "on",
         "models": [], "designations": [], "designation_meta": {},
         "spill_by_model": {}, "moe_by_model": {}, "bnb_by_model": {},
         "limits": {}, "wildcard": False, "models_local": [], "storage": {}}
    w.update(over)
    return w


class FakeClient:
    """Serves /llm/workers and /llm/serving from mutable state; records POSTs,
    applies serving writes and answers /cache-evict with a freed-bytes contract
    (sized from the worker's storage) so a roundtrip is checkable.
    ``fail`` = path substrings whose POST returns a typed refusal (ok False);
    ``raise_on`` = substrings whose POST raises."""

    def __init__(self, workers, serving):
        self.workers = workers
        self.serving = serving
        self.posts = []
        self.fail = set()
        self.raise_on = set()
        self.unsupported = set()   # substrings whose POST answers an old-agent refusal

    def _worker(self, wid):
        return next((w for w in self.workers if (w.get("id") or w.get("name")) == wid), None)

    def request(self, path, method="GET", body=None, timeout=None):
        if method == "GET":
            if path == "/llm/workers":
                return copy.deepcopy(self.workers)
            if path == "/llm/serving":
                return [{"key": mk, "override": {"gguf_file_by_worker": copy.deepcopy(v)}}
                        for mk, v in self.serving.items()]
            if path.startswith("/models"):
                return {"models": []}
            return {}
        # POST
        self.posts.append((path, body))
        if any(s in path for s in self.raise_on):
            raise RuntimeError("network boom")
        if any(s in path for s in self.unsupported):
            return {"ok": False, "started": False, "unsupported": True,
                    "error": "this worker agent does not support fetch-to-disk "
                             "(POST /models/fetch); update the worker"}
        if any(s in path for s in self.fail):
            return {"ok": False, "error": "worker refused"}
        if path.startswith("/llm/serving/"):
            mk = path[len("/llm/serving/"):].replace("%2F", "/")
            self.serving[mk] = dict((body or {}).get("gguf_file_by_worker") or {})
            return {"ok": True}
        if path.endswith("/cache-evict"):
            wid = path[len("/llm/workers/"):-len("/cache-evict")]
            w = self._worker(wid) or {}
            mk = (body or {}).get("model_key")
            size = next((r.get("bytes") for r in (w.get("storage") or {}).get("models") or []
                         if r.get("model_key") == mk), 0)
            return {"ok": True, "freed_bytes": size, "removed": True}
        return {"ok": True}


def _fixture():
    w1 = _worker("w1", "aeb", models=["org/m"], models_local=["org/m", "org/keep"],
                 designations=[{"model_key": "org/m", "pinned": False, "source": "operator"}],
                 spill_by_model={"org/m": {"n_gpu_layers": 20}},
                 storage=_stor([("org/m", 4 * GB, "reapable", False),
                                ("org/keep", 2 * GB, "reapable", False)], disk_free=100 * GB))
    w2 = _worker("w2", "computron", models_local=["org/other"],
                 storage=_stor([("org/other", 3 * GB, "reapable", False)], disk_free=100 * GB))
    serving = {"org/m": {"w1": "Q4_K_M"}}
    return [w1, w2], serving


def test_snapshot_captures_the_three_state_classes_and_is_jsonable():
    workers, serving = _fixture()
    client = FakeClient(workers, serving)
    snap = worker_state.snapshot(client, worker_ids=None, model_ids=["org/m", "org/keep", "org/other"])
    json.dumps(snap)                                   # persistable with the run
    assert set(snap["worker_ids"]) == {"w1", "w2"}
    pw = snap["per_worker"]["w1"]
    assert pw["models"] == ["org/m"] and pw["spill_by_model"] == {"org/m": {"n_gpu_layers": 20}}
    assert pw["pins"] == {"org/m": False}
    assert snap["serving"]["org/m"] == {"w1": "Q4_K_M"}
    # (3) disk: the FULL prior set, per-model sizes, and reported free space.
    assert set(pw["on_disk"]) == {"org/m", "org/keep"}
    assert pw["model_sizes"]["org/m"] == 4 * GB and pw["disk_free"] == 100 * GB


def test_snapshot_disk_truth_is_not_scoped_to_the_test():
    # A model on the drive but OUTSIDE the requested model scope is still captured.
    workers, serving = _fixture()
    workers[0]["models_local"] = ["org/m", "offscope"]
    workers[0]["storage"] = _stor([("org/m", 4 * GB, "reapable", False),
                                   ("offscope", 6 * GB, "reapable", False)], disk_free=100 * GB)
    snap = worker_state.snapshot(FakeClient(workers, serving), worker_ids=None, model_ids=["org/m"])
    assert set(snap["per_worker"]["w1"]["on_disk"]) == {"org/m", "offscope"}


def test_restore_converges_quant_pin_and_reprovisions_evicted_disk():
    workers, serving = _fixture()
    client = FakeClient(workers, serving)
    snap = worker_state.snapshot(client, worker_ids=None, model_ids=["org/m", "org/keep", "org/other"])
    # The benchmark repinned the quant on w1 and cache-evicted org/m (ample free).
    client.serving["org/m"] = {"w1": "Q8_0"}
    client.workers[0]["models_local"] = ["org/keep"]
    rep = worker_state.restore(client, snap)
    assert ("/llm/serving/org%2Fm", {"gguf_file_by_worker": {"w1": "Q4_K_M"}}) in client.posts
    # Re-provision uses the FETCH-ONLY verb (disk, no VRAM), never /probe (load).
    assert ("/llm/workers/w1/fetch", {"model_key": "org/m"}) in client.posts
    assert not any(p.endswith("/probe") for p, _ in client.posts)
    w1 = next(r for r in rep["workers"] if r["worker_id"] == "w1")
    assert "org/m" in w1["redownloaded"] and not w1["removed"] and not w1["failures"]
    assert any("quant pin for org/m" in s for s in w1["restored"])
    # Ample free space => no extra was removed.
    assert not any(p.endswith("/cache-evict") for p, _ in client.posts)


def test_restore_is_a_noop_when_nothing_changed():
    workers, serving = _fixture()
    client = FakeClient(workers, serving)
    snap = worker_state.snapshot(client, worker_ids=None, model_ids=["org/m", "org/keep", "org/other"])
    rep = worker_state.restore(client, snap)
    assert client.posts == []
    for r in rep["workers"]:
        assert not r["restored"] and not r["redownloaded"] and not r["removed"] and not r["failures"]


def test_restore_records_failures_without_raising():
    workers, serving = _fixture()
    client = FakeClient(workers, serving)
    snap = worker_state.snapshot(client, worker_ids=None, model_ids=["org/m", "org/keep", "org/other"])
    client.serving["org/m"] = {"w1": "Q8_0"}
    client.workers[0]["models_local"] = ["org/keep"]
    client.fail = {"/fetch"}
    client.raise_on = {"/llm/serving/"}
    rep = worker_state.restore(client, snap)
    w1 = next(r for r in rep["workers"] if r["worker_id"] == "w1")
    errs = " ".join(f["error"] for f in w1["failures"])
    assert any(f["item"] == "re-provision org/m" for f in w1["failures"]) and "worker refused" in errs
    assert any(f["item"] == "gguf pin org/m" for f in w1["failures"]) and "network boom" in errs


def test_restore_fetch_unsupported_by_old_agent_is_a_failure_not_a_silent_load():
    # An old worker/central without the fetch verb answers with a refusal
    # (ok False / 501 unsupported). Restore must record it as a failure and must
    # NEVER fall back to a load (/probe).
    workers, serving = _fixture()
    client = FakeClient(workers, serving)
    snap = worker_state.snapshot(client, worker_ids=None, model_ids=["org/m", "org/keep", "org/other"])
    client.workers[0]["models_local"] = ["org/keep"]     # org/m evicted, must come back
    client.unsupported = {"/fetch"}
    rep = worker_state.restore(client, snap)
    w1 = next(r for r in rep["workers"] if r["worker_id"] == "w1")
    assert not w1["redownloaded"]
    assert not any(p.endswith("/probe") for p, _ in client.posts)   # no silent load
    fail = next(f for f in w1["failures"] if f["item"] == "re-provision org/m")
    assert "fetch-to-disk" in fail["error"]


def test_restore_handles_missing_snapshot():
    assert worker_state.restore(FakeClient([], {}), None)["error"]
    assert worker_state.restore(FakeClient([], {}), {})["error"]


def test_restore_runs_in_a_finally_after_the_run_raises():
    workers, serving = _fixture()
    client = FakeClient(workers, serving)
    snap = worker_state.snapshot(client, worker_ids=None, model_ids=["org/m", "org/keep", "org/other"])
    outcome = {}

    def run_with_restore():
        try:
            client.serving["org/m"] = {"w1": "Q8_0"}
            raise RuntimeError("lane exploded")
        finally:
            outcome["rep"] = worker_state.restore(client, snap)

    with pytest.raises(RuntimeError):
        run_with_restore()
    assert client.serving["org/m"] == {"w1": "Q4_K_M"}
    assert outcome["rep"]["workers"]


# ── disk removal-to-fit (operator ruling 2026-09-24, gap 2) ─────────────────

def _one_prior(disk_free, extras=(), prior=("prior_big", 50 * GB)):
    """A worker whose only prior model is ``prior``; ``extras`` = accumulated
    test models now on the drive. Snapshot is taken BEFORE, so its prior set is
    just the prior model; the live record is mutated to the post-test drive."""
    mk, size = prior
    before = _worker("w1", "aeb", models_local=[mk],
                     storage=_stor([(mk, size, "reapable", False)], disk_free=disk_free))
    return before, mk, size


def test_missing_prior_fits_so_no_extra_is_removed():
    before, mk, size = _one_prior(disk_free=60 * GB)
    client = FakeClient([before], {})
    snap = worker_state.snapshot(client, worker_ids=None, model_ids=[mk])
    # Post-test: prior evicted, one extra accumulated, plenty of room.
    client.workers[0]["models_local"] = ["extraA"]
    client.workers[0]["storage"] = _stor([("extraA", 20 * GB, "reapable", False)], disk_free=80 * GB)
    rep = worker_state.restore(client, snap)
    w1 = rep["workers"][0]
    assert w1["redownloaded"] == [mk] and not w1["removed"] and not w1["failures"]
    assert not any(p.endswith("/cache-evict") for p, _ in client.posts)


def test_full_drive_removes_the_fewest_largest_extras_to_fit_the_prior():
    before, mk, size = _one_prior(disk_free=60 * GB)          # prior_big = 50 GB
    client = FakeClient([before], {})
    snap = worker_state.snapshot(client, worker_ids=None, model_ids=[mk])
    # Post-test: prior gone; two extras; only 10 GB free (deficit 40 GB). The
    # 40 GB extra alone covers it — the 30 GB one must NOT be touched.
    client.workers[0]["models_local"] = ["extraA", "extraB"]
    client.workers[0]["storage"] = _stor([("extraA", 40 * GB, "reapable", False),
                                          ("extraB", 30 * GB, "reapable", False)], disk_free=10 * GB)
    rep = worker_state.restore(client, snap)
    w1 = rep["workers"][0]
    assert w1["removed"] == ["extraA"]                        # fewest, largest-first
    assert (mk in w1["redownloaded"]) and not w1["failures"]
    evicted = [b.get("model_key") for p, b in client.posts if p.endswith("/cache-evict")]
    assert evicted == ["extraA"] and "extraB" not in evicted
    assert any(s.startswith("removed extraA") and "to fit prior model prior_big" in s
               for s in w1["restored"])


def test_extras_cannot_cover_so_nothing_is_deleted_and_a_failure_is_reported():
    before, mk, size = _one_prior(disk_free=60 * GB)          # prior_big = 50 GB
    client = FakeClient([before], {})
    snap = worker_state.snapshot(client, worker_ids=None, model_ids=[mk])
    # Post-test: prior gone; 5 GB free; the only extra is 10 GB (deficit 45 GB).
    client.workers[0]["models_local"] = ["extraSmall"]
    client.workers[0]["storage"] = _stor([("extraSmall", 10 * GB, "reapable", False)], disk_free=5 * GB)
    rep = worker_state.restore(client, snap)
    w1 = rep["workers"][0]
    assert not w1["removed"] and not w1["redownloaded"]
    assert not any(p.endswith("/cache-evict") for p, _ in client.posts)   # deleted NOTHING
    fail = next(f for f in w1["failures"] if f["item"] == "re-provision prior_big")
    assert "not enough room" in fail["error"]
    assert "need 50.0 GB" in fail["error"] and "free 5.0 GB" in fail["error"] and "10.0 GB" in fail["error"]


def test_a_prior_model_is_never_removed_to_fit_another():
    # Two prior models; one still present, one evicted; free is tight. The only
    # thing on disk besides the survivor is the survivor itself (a prior) — it
    # must NOT be removed, so the missing one reports a shortfall, not a deletion.
    before = _worker("w1", "aeb", models_local=["priorA", "priorB"],
                     storage=_stor([("priorA", 50 * GB, "reapable", False),
                                    ("priorB", 50 * GB, "reapable", False)], disk_free=100 * GB))
    client = FakeClient([before], {})
    snap = worker_state.snapshot(client, worker_ids=None, model_ids=["priorA", "priorB"])
    # priorB evicted; almost no free space; priorA (a prior) is the only resident.
    client.workers[0]["models_local"] = ["priorA"]
    client.workers[0]["storage"] = _stor([("priorA", 50 * GB, "reapable", False)], disk_free=1 * GB)
    rep = worker_state.restore(client, snap)
    w1 = rep["workers"][0]
    assert not any(p.endswith("/cache-evict") for p, _ in client.posts)   # priorA never removed
    assert any(f["item"] == "re-provision priorB" for f in w1["failures"])
