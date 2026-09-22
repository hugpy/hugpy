"""Per-model install-status memo — the /v1/models availability fix (2026-07-27).

Live incident: central was unusable. ``/health`` answered in 0.0009s and
``/llm/workers`` in 2.0s, but ``/v1/models`` took 18.5s, then 25.7s, then 55.3s
— degrading — while established connections to :7002 climbed 47 -> 140 against
only 24 gunicorn slots (``--workers 3 --threads 8``). Thread wchans across all
three workers read ``request_wait_answer`` (FUSE/virtiofs), ``folio_wait_bit_common``
(disk) and ``locks_lock_inode_wait``: the API was not computing, it was queued
on I/O.

Cause: ``v1_models`` (and ``list_models``) loop the whole manifest and call
``update_model_status(model)`` per model. That delegates to ``model_status``,
which walks the store — ``route_destination`` globs four runtime families'
legacy task dirs and stats every candidate, then ``model_looks_downloaded``
globs the winner. ~10^2 filesystem calls per model, ~107 models, every call a
virtiofs round-trip. The manifest itself was already cached; the per-model
status stat was not. Installation status only changes on download / delete /
prune / reconcile / discovery, so re-walking per request was pure waste.

SCOPE (narrowed 2026-07-27, same day): the memo was the MITIGATION — it made a
redundant walk cheap instead of removing it. Install status is now DERIVED at the
events that change it and PERSISTED beside the registry
(``comms/model_physical.py``), so ``/models`` and ``/v1/models`` read a dict and
there is no memo in front of ``model_status`` at all. Those tests live in
``tests/test_model_physical_state.py``.

The memo survives owning exactly ONE question: **/llm/central-provisioning** —
"can central PROVIDE this model to a worker?" (scope ``central-holdings``). Same
underlying walk, different answer, no persisted home, and the console polls it
every 10s over the whole manifest.

This suite locks (``comms/model_status_cache.py`` + that call site):

  * same VALUES cached vs uncached — the content never changes shape;
  * a second fan-out does far fewer probes (counted stub, not wall time);
  * TTL expiry re-reads; scopes never collide; a routing change re-keys;
  * concurrent callers single-flight instead of stampeding the walk;
  * a failure in the cache machinery degrades to the LIVE probe, and an error
    raised by the live probe still propagates (never an invented status);
  * a sibling PROCESS's invalidation is picked up (the gunicorn --workers 3
    case) via the local epoch file;
  * every store event STILL flushes it — download completed / failed /
    cancelled, delete, prune, reconcile apply, refresh_registry — because the
    provisioning answer is derived from the same presence facts.

Runs under pytest:
    venv/bin/python -m pytest tests/test_model_status_cache.py -q
"""
import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The memo publishes a cross-process invalidation token to a file. Point it at a
# per-run temp path BEFORE anything imports/uses the module, so a test run can
# never stomp (or be stomped by) the epoch a live central is sharing between its
# gunicorn workers.
_EPOCH_DIR = tempfile.mkdtemp(prefix="hugpy-status-cache-test-")
os.environ["HUGPY_MODEL_STATUS_EPOCH_PATH"] = os.path.join(_EPOCH_DIR, "epoch")
os.environ.pop("HUGPY_MODEL_STATUS_TTL_S", None)
os.environ.pop("HUGPY_MODEL_STATUS_EPOCH_POLL_S", None)
# Same rule for the PERSISTED physical table these routes also write: the
# delete/prune/reconcile wiring tests below drive real routes, and a test run
# must never drop rows out of the live central's table.
os.environ["HUGPY_MODEL_PHYSICAL_PATH"] = os.path.join(_EPOCH_DIR, "physical.json")

cache = importlib.import_module("hugpy_storage.model_status_cache")
from hugpy_control.jobs import normalize_status


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def _model(key="repo-0", **over):
    m = {
        "model_key": key,
        "hub_id": f"org/{key}",
        "name": key,
        "framework": "gguf",
        "primary_task": "text-generation",
        "tasks": ["text-generation"],
    }
    m.update(over)
    return m


class CountingStat:
    """Stand-in for ``model_status`` that counts how often the store is read."""

    def __init__(self, status="installed", destination="/store/x"):
        self.calls = 0
        self.status = status
        self.destination = destination
        self.lock = threading.Lock()
        self.delay = 0.0

    def __call__(self, model):
        with self.lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return {"status": self.status,
                "destination": f"{self.destination}/{model.get('model_key')}",
                "installed_marker": f"{self.destination}/{model.get('model_key')}/hugpy.json"}


def _catalog_stamp():
    """(mtime_ns, size) of the registry's two persisted artifacts."""
    from hugpy_platform.constants import MODELS_DICT_PATH, MODELS_DISCOVERY_PATH
    out = {}
    for path in (str(MODELS_DISCOVERY_PATH), str(MODELS_DICT_PATH)):
        try:
            st = os.stat(path)
            out[path] = (st.st_mtime_ns, st.st_size)
        except OSError:
            out[path] = None
    return out


@pytest.fixture(autouse=True)
def _live_catalog_is_off_limits():
    """Fail LOUDLY if a test writes the live discovery report or manifest.

    Earned 2026-07-27: refresh_registry(run_discovery=True) runs
    discover_models(save_json=True), which walks the live 16TB store and
    OVERWRITES model_discovery.json — and with get_models_dict stubbed it wrote
    an empty one. No test here has any business touching a real catalog file."""
    before = _catalog_stamp()
    yield
    after = _catalog_stamp()
    changed = [p for p in before if before[p] != after[p]]
    assert not changed, f"a test modified live registry state: {changed}"


@pytest.fixture(autouse=True)
def _clean_cache():
    cache.reset_model_status_cache()
    for var in ("HUGPY_MODEL_STATUS_TTL_S", "HUGPY_MODEL_STATUS_EPOCH_POLL_S",
                "HUGPY_MODEL_STATUS_LOCK_WAIT_S"):
        os.environ.pop(var, None)
    yield
    cache.reset_model_status_cache()
    for var in ("HUGPY_MODEL_STATUS_TTL_S", "HUGPY_MODEL_STATUS_EPOCH_POLL_S",
                "HUGPY_MODEL_STATUS_LOCK_WAIT_S"):
        os.environ.pop(var, None)


# ──────────────────────────────────────────────────────────────────────────
# 1. the memo itself
# ──────────────────────────────────────────────────────────────────────────
def test_same_value_cached_and_uncached():
    """The cached answer is byte-identical to the live one."""
    live = CountingStat()
    m = _model()
    os.environ["HUGPY_MODEL_STATUS_TTL_S"] = "0"          # cache off
    uncached = cache.cached_model_status(m, live)
    os.environ["HUGPY_MODEL_STATUS_TTL_S"] = "30"
    cold = cache.cached_model_status(m, live)
    warm = cache.cached_model_status(m, live)
    assert uncached == cold == warm
    assert live.calls == 2                                 # off + cold, not warm


def test_second_listing_does_far_fewer_status_reads():
    """A whole-manifest walk pays once; the next one pays nothing."""
    live = CountingStat()
    manifest = [_model(f"repo-{i}") for i in range(107)]

    first = [cache.cached_model_status(m, live) for m in manifest]
    after_first = live.calls
    second = [cache.cached_model_status(m, live) for m in manifest]
    after_second = live.calls

    assert after_first == 107, "cold walk must read every model exactly once"
    assert after_second == 107, "a warm listing must read the store ZERO times"
    assert first == second


def test_returned_dict_is_a_copy_callers_cannot_poison():
    """Listings do ``model.update(status)`` and mutate rows in place."""
    live = CountingStat()
    m = _model()
    got = cache.cached_model_status(m, live)
    got["status"] = "not_installed"
    got["injected"] = True
    again = cache.cached_model_status(m, live)
    assert again["status"] == "installed"
    assert "injected" not in again


def test_ttl_expiry_re_reads():
    live = CountingStat()
    m = _model()
    os.environ["HUGPY_MODEL_STATUS_TTL_S"] = "0.25"
    cache.cached_model_status(m, live)
    cache.cached_model_status(m, live)
    assert live.calls == 1
    time.sleep(0.35)
    cache.cached_model_status(m, live)
    assert live.calls == 2, "an expired entry must be re-read from the store"


def test_ttl_zero_disables_the_cache_entirely():
    live = CountingStat()
    m = _model()
    os.environ["HUGPY_MODEL_STATUS_TTL_S"] = "0"
    for _ in range(5):
        cache.cached_model_status(m, live)
    assert live.calls == 5


def test_a_routing_change_is_a_different_entry():
    """Re-key a model (new hub_id/filename/framework) and the old memo can't
    answer for it — the key IS the routing identity the live stat reads."""
    live = CountingStat()
    base = _model()
    cache.cached_model_status(base, live)
    assert live.calls == 1
    for field, value in (("hub_id", "org/other"), ("filename", "x.gguf"),
                         ("framework", "transformers"), ("dir", "/elsewhere"),
                         ("include", ["*.gguf"]), ("tasks", ["image-to-image"])):
        before = live.calls
        cache.cached_model_status(_model(**{field: value}), live)
        assert live.calls == before + 1, f"{field} must change the cache key"


def test_scopes_do_not_collide():
    """Two questions about the SAME model must not share an entry."""
    status = CountingStat(status="installed")
    holdings = CountingStat(status="ready")
    m = _model()
    a = cache.cached_model_status(m, status)
    b = cache.cached_model_status(m, holdings, scope="central-holdings")
    assert a["status"] == "installed"
    assert b["status"] == "ready"
    assert status.calls == 1 and holdings.calls == 1
    assert cache.cached_model_status(m, status)["status"] == "installed"
    assert cache.cached_model_status(
        m, holdings, scope="central-holdings")["status"] == "ready"
    assert status.calls == 1 and holdings.calls == 1


def test_invalidation_forces_a_re_read():
    live = CountingStat()
    m = _model()
    cache.cached_model_status(m, live)
    cache.cached_model_status(m, live)
    assert live.calls == 1
    cache.invalidate_model_status("test")
    cache.cached_model_status(m, live)
    assert live.calls == 2


def test_refresh_always_reads_live_and_seeds_the_memo():
    live = CountingStat()
    m = _model()
    cache.cached_model_status(m, live)
    assert live.calls == 1
    cache.refresh_model_status(m, live)                    # explicit refresh
    assert live.calls == 2, "an explicit refresh must not be served from cache"
    cache.cached_model_status(m, live)
    assert live.calls == 2, "…and it must leave the memo repaired"


def test_concurrent_callers_do_not_stampede():
    """32 threads asking for the same model share ONE walk."""
    live = CountingStat()
    live.delay = 0.15                                      # a slow store read
    m = _model()
    results, errors = [], []

    def worker():
        try:
            results.append(cache.cached_model_status(dict(m), live))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not errors
    assert len(results) == 32
    assert all(r == results[0] for r in results)
    assert live.calls == 1, (
        f"single-flight broken: {live.calls} threads each walked the store")


def test_concurrent_callers_across_models_still_walk_each_model_once():
    live = CountingStat()
    live.delay = 0.01
    manifest = [_model(f"repo-{i}") for i in range(20)]
    errors = []

    def worker():
        try:
            for m in manifest:
                cache.cached_model_status(dict(m), live)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors
    assert live.calls == 20, f"expected one walk per model, got {live.calls}"


# ──────────────────────────────────────────────────────────────────────────
# 2. degrade-not-guess
# ──────────────────────────────────────────────────────────────────────────
def test_cache_machinery_failure_falls_back_to_the_live_stat(monkeypatch):
    live = CountingStat()
    m = _model()

    def boom(*_a, **_k):
        raise RuntimeError("key computation exploded")

    monkeypatch.setattr(cache, "status_key", boom)
    got = cache.cached_model_status(m, live)
    assert got["status"] == "installed"
    assert live.calls == 1
    # and it keeps working — never a poisoned entry, just no caching
    cache.cached_model_status(m, live)
    assert live.calls == 2
    assert cache.cache_stats()["errors"] >= 1


def test_a_store_read_error_propagates_exactly_as_today():
    """The live stat raising is TODAY's behaviour; the memo must not swallow it
    and must never invent a status in its place."""
    calls = {"n": 0}

    def exploding(_model):
        calls["n"] += 1
        raise OSError("virtiofs went away")

    with pytest.raises(OSError):
        cache.cached_model_status(_model(), exploding)
    assert calls["n"] == 1
    # nothing was memoized, so the next caller retries the store
    with pytest.raises(OSError):
        cache.cached_model_status(_model(), exploding)
    assert calls["n"] == 2


def test_a_non_dict_status_is_never_memoized():
    calls = {"n": 0}

    def weird(_model):
        calls["n"] += 1
        return None

    assert cache.cached_model_status(_model(), weird) is None
    assert cache.cached_model_status(_model(), weird) is None
    assert calls["n"] == 2


def test_unwritable_epoch_degrades_to_ttl_only(monkeypatch):
    """A /tmp we cannot write must not break invalidation locally."""
    monkeypatch.setenv("HUGPY_MODEL_STATUS_EPOCH_PATH",
                       "/proc/definitely/not/writable/epoch")
    live = CountingStat()
    m = _model()
    cache.cached_model_status(m, live)
    cache.invalidate_model_status("unwritable-epoch")       # must not raise
    cache.cached_model_status(m, live)
    assert live.calls == 2


# ──────────────────────────────────────────────────────────────────────────
# 3. cross-process (gunicorn --workers 3)
# ──────────────────────────────────────────────────────────────────────────
def test_a_sibling_process_invalidation_is_picked_up():
    """Process A memoizes; process B deletes a model and invalidates; A must
    stop serving the stale entry — inside the epoch poll, not the TTL."""
    os.environ["HUGPY_MODEL_STATUS_EPOCH_POLL_S"] = "0"     # read every lookup
    os.environ["HUGPY_MODEL_STATUS_TTL_S"] = "3600"         # TTL can't be the reason
    live = CountingStat()
    m = _model()
    cache.cached_model_status(m, live)
    cache.cached_model_status(m, live)
    assert live.calls == 1

    code = (
        "import os,sys;"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / 'src')!r});"
        "import importlib;"
        "c=importlib.import_module('hugpy_storage.model_status_cache');"
        "c.invalidate_model_status('sibling-process')"
    )
    env = dict(os.environ)
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr

    cache.cached_model_status(m, live)
    assert live.calls == 2, (
        "a sibling gunicorn worker's invalidation was not observed")


def test_first_epoch_read_does_not_spuriously_clear():
    """Adopting a token we have never seen must not throw away a warm memo for
    no reason (a fresh process starts empty anyway)."""
    os.environ["HUGPY_MODEL_STATUS_EPOCH_POLL_S"] = "0"
    cache.invalidate_model_status("seed")                   # publish a token
    cache.reset_model_status_cache()                        # forget we saw it
    live = CountingStat()
    m = _model()
    cache.cached_model_status(m, live)
    cache.cached_model_status(m, live)
    cache.cached_model_status(m, live)
    assert live.calls == 1
    assert cache.cache_stats()["epoch_clears"] == 0


# ──────────────────────────────────────────────────────────────────────────
# 4. wiring — the real call sites
# ──────────────────────────────────────────────────────────────────────────
# cancelable_downloads.py lives under flask_app.app.functions; importing it as a
# bare dotted path before the flask app has booted trips the known import-order
# landmine (flask_app's `app` attribute shadowed by flask's own `app` submodule).
# Booting once first populates sys.modules correctly — same preamble as
# tests/test_download_progress.py.
importlib.import_module("hugpy_server.wsgi_app").get_hugpy_flask()
dl = importlib.import_module(
    "hugpy_storage.console.downloader")
cd = importlib.import_module(
    "hugpy_storage.console.cancelable_downloads")
routes = importlib.import_module(
    "hugpy_server.app.routes.llm_storage_routes")
v1 = importlib.import_module("hugpy_server.app.routes.v1_routes")
mc = importlib.import_module("hugpy_engine.config.models.models_config")


def _manifest(n=25):
    return {f"repo-{i}": _model(f"repo-{i}") for i in range(n)}


def _flask_app():
    return importlib.import_module(
        "hugpy_server.wsgi_app").get_hugpy_flask()



# ── the one question this memo still owns ─────────────────────────────────
# INSTALL STATUS moved out (2026-07-27): /models + /v1/models now read the
# PERSISTED physical record (comms/model_physical.py) written at the events that
# change it, so there is no memo in front of `model_status` any more — a memo and
# a persisted record both answering "is this installed?" is exactly the
# two-mechanisms-one-question trap. Those tests live in
# tests/test_model_physical_state.py.
#
# What remains here is /llm/central-provisioning: "can central PROVIDE this model
# to a worker?". Same walk underneath, different answer, no persisted home — the
# console polls it every 10s over the whole manifest, so the fan-out is memoized
# and the single-model GUARD stays live.
def test_the_install_status_question_no_longer_reads_the_memo(monkeypatch):
    """update_model_status must not be a memo caller — it is a store lookup."""
    live = CountingStat()
    monkeypatch.setattr(dl, "model_status", live)
    assert not hasattr(dl, "cached_model_status"), (
        "downloader.cached_model_status is retired; the persisted record owns "
        "the install-status question now")
    assert not hasattr(dl, "refresh_model_status"), (
        "downloader.refresh_model_status is retired; "
        "model_physical.refresh_fields is the force-refresh")
    before = cache.cache_stats()
    cd.update_model_status(_model("memo-check"))
    after = cache.cache_stats()
    assert after["hits"] == before["hits"]
    assert after["misses"] == before["misses"], (
        "the listing path went through the install-status memo")


def test_central_provisioning_poll_shares_the_memo(monkeypatch):
    """/llm/central-provisioning is polled every 10s by the console and loops
    the WHOLE manifest through the same route_destination + model_looks_downloaded
    walk. The listing must memoize; the single-model GUARD must stay live."""
    wr = importlib.import_module(
        "hugpy_server.app.routes.worker_routes")
    manifest = _manifest(12)
    probes = {"n": 0}

    def fake_reason(model_key):
        probes["n"] += 1
        return None if model_key.endswith("0") else "no model directory"

    class _Cfg:
        def __init__(self, row):
            self._row = row

        def to_dict(self):
            return dict(self._row)

    monkeypatch.setattr(wr, "_central_missing_reason", fake_reason)
    monkeypatch.setattr(wr, "get_models_dict", lambda **_k: manifest)
    main = importlib.import_module("hugpy_engine.config.main")
    monkeypatch.setattr(main, "get_model_config",
                        lambda mk, **_k: _Cfg(manifest[mk]))
    app = _flask_app()

    with app.test_client() as client:
        first = client.get("/llm/central-provisioning")
        assert first.status_code == 200
        assert probes["n"] == len(manifest)
        second = client.get("/llm/central-provisioning").get_json()
        assert probes["n"] == len(manifest), (
            "a second poll re-walked the store")
    assert first.get_json() == second
    assert second["repo-0"] == {"state": "ready", "reason": None}
    assert second["repo-1"]["state"] == "absent"

    # the guard path is deliberately NOT memoized
    before = probes["n"]
    wr._central_missing_reason("repo-1")
    wr._central_missing_reason("repo-1")
    assert probes["n"] == before + 2


# ── the memo must still hear the store events ─────────────────────────────
# The central-holdings answer is derived from the same presence facts, so every
# event that moves a model between not_installed/partial/installed still has to
# flush it. (What each event does to the PERSISTED record — targeted vs
# whole-table — is asserted in tests/test_model_physical_state.py.)
def _spy_invalidation(monkeypatch):
    seen = []
    real = cache.invalidate_model_status

    def spy(reason=""):
        seen.append(reason)
        real(reason)

    monkeypatch.setattr(cache, "invalidate_model_status", spy)
    return seen


def _memo_is_flushed(live, model):
    """True if the central-holdings memo re-reads after the event."""
    before = live.calls
    cache.cached_model_status(model, live, scope="central-holdings")
    return live.calls == before + 1


def test_delete_route_flushes_the_memo(monkeypatch, tmp_path):
    live = CountingStat()
    dest = tmp_path / "models" / "gguf" / "org" / "repo-0"
    dest.mkdir(parents=True)
    (dest / "w.gguf").write_bytes(b"x")
    manifest = _manifest(3)
    monkeypatch.setattr(routes, "get_models_dict", lambda **_k: manifest)
    monkeypatch.setattr(routes, "route_destination", lambda _m: str(dest))
    seen = _spy_invalidation(monkeypatch)
    app = _flask_app()

    m = _model("repo-0")
    cache.cached_model_status(m, live, scope="central-holdings")
    assert live.calls == 1
    with app.test_client() as client:
        resp = client.delete("/models/repo-0")
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] is True
    assert any("deleted" in r for r in seen), seen
    assert _memo_is_flushed(live, m)


def test_prune_route_flushes_the_memo(monkeypatch, tmp_path):
    live = CountingStat()
    manifest = _manifest(3)
    monkeypatch.setattr(routes, "get_models_dict", lambda **_k: manifest)
    monkeypatch.setattr(routes, "route_destination",
                        lambda _m: str(tmp_path / "gone"))
    monkeypatch.setattr(routes, "prune_model", lambda k: {"pruned": True, "key": k})
    seen = _spy_invalidation(monkeypatch)
    app = _flask_app()

    m = _model("repo-0")
    cache.cached_model_status(m, live, scope="central-holdings")
    with app.test_client() as client:
        assert client.post("/models/repo-0/prune").status_code == 200
    assert any("pruned" in r for r in seen), seen
    assert _memo_is_flushed(live, m)


def test_reconcile_apply_flushes_but_a_dry_run_does_not(monkeypatch):
    live = CountingStat()
    reconcile = importlib.import_module("hugpy_engine.apis.reconcile")
    monkeypatch.setattr(reconcile, "reconcile_store",
                        lambda **_k: {"actions": [], "warnings": []})
    seen = _spy_invalidation(monkeypatch)
    app = _flask_app()

    m = _model("repo-0")
    cache.cached_model_status(m, live, scope="central-holdings")
    assert live.calls == 1
    # /models/reconcile is operator-token gated (operator_auth._SENSITIVE), so
    # drive the view function inside a request context — we are testing the
    # invalidation wiring, not the gate.
    with app.test_request_context("/models/reconcile", method="POST",
                                  json={"apply": False}):
        _body, status = routes.reconcile_store_route()
        assert status == 202
    assert not seen, "a dry run touches nothing and must not invalidate"
    cache.cached_model_status(m, live, scope="central-holdings")
    assert live.calls == 1

    with app.test_request_context("/models/reconcile", method="POST",
                                  json={"apply": True}):
        _body, status = routes.reconcile_store_route()
        assert status == 200
    assert any("reconcile" in r for r in seen), seen
    assert _memo_is_flushed(live, m)


def test_refresh_registry_flushes_the_memo(monkeypatch):
    """THE chokepoint: download completion, the discovery sweep and reconcile's
    registry write all land here, so one hook covers them."""
    live = CountingStat()
    md = importlib.import_module(
        "hugpy_engine.config.models.models_default")
    ov = importlib.import_module("hugpy_engine.serve.overrides")
    # refresh_registry(run_discovery=False) is idempotent on the live registry
    # (get_models_dict returns the cached object, so update+prune are no-ops);
    # stub only the two expensive follow-ups so this test never walks the store.
    monkeypatch.setattr(md, "refresh_task_registries", lambda *a, **k: None)
    monkeypatch.setattr(ov, "migrate_overrides", lambda *a, **k: [])
    seen = _spy_invalidation(monkeypatch)

    m = _model("repo-0")
    cache.cached_model_status(m, live, scope="central-holdings")
    assert live.calls == 1
    mc.refresh_registry(run_discovery=False)
    assert any("refresh_registry" in r for r in seen), seen
    assert _memo_is_flushed(live, m)


def test_download_cancel_flushes_the_memo(monkeypatch):
    live = CountingStat()
    seen = _spy_invalidation(monkeypatch)
    job = cd.job_store.create("repo-0", kind="download", transport="test")

    m = _model("repo-0")
    cache.cached_model_status(m, live, scope="central-holdings")
    assert live.calls == 1
    assert cd.cancel_download(job.id)["cancelled"] is True
    assert any("cancelled" in r for r in seen), seen
    assert _memo_is_flushed(live, m)
