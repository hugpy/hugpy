"""Persisted PHYSICAL model state — /models and /v1/models answer from what
central already knows (2026-07-27).

THE PROBLEM THIS LOCKS
----------------------
The registry persisted a model's IDENTITY but not its PHYSICAL state, so every
listing RE-DERIVED "is it installed, where, and how big" from the store on every
GET: ``model_status`` (~10^2 stats/model) for ``/v1/models``, and that plus
``gguf_variants_detail`` + a recursive ``walk_listing``/``dir_size_bytes`` for
``/models``. ~107 models over **virtiofs** to a spinning array measured 40.4s
for ``/models`` and 3.9s cold for ``/v1/models``, and under concurrency the
threads queued on ``request_wait_answer`` / ``locks_lock_inode_wait`` until the
site stopped answering.

Central DOWNLOADED these models. It knows. So the facts are derived at the
moments they CHANGE, persisted beside the registry (comms/model_physical.py),
and read as a dict lookup.

WHAT IS ASSERTED HERE
---------------------
  * a warm ``/models`` and ``/v1/models`` make ZERO per-model filesystem calls —
    counted os.* calls and a counted deriver, never wall time;
  * the response is byte-identical to deriving live for the same state;
  * every write event updates the persisted record — download completed /
    failed / cancelled, delete, prune, reconcile apply, refresh_registry, and a
    ``gguf_file`` override change;
  * ABSENT MEANS DERIVE: a row with no record is derived live, never reported as
    0 bytes or not_installed (a wrongly-persisted not_installed HIDES a model);
  * a deleted model stops reporting installed, and the delete is TARGETED — the
    other rows stay warm;
  * ``GET /models/<key>`` stays a live read and repairs the record;
  * the repair sweep (``/models/discover``) fixes an out-of-band change nothing
    told us about;
  * concurrent writers — three gunicorn workers + reconcile — never corrupt the
    table;
  * every failure mode degrades to deriving live, never to an invented value.

Runs under pytest:
    venv/bin/python -m pytest tests/test_model_physical_state.py -q
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

_SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, _SRC)

# Point BOTH persisted stores at a per-run temp dir BEFORE anything imports
# them: this suite deletes, prunes and reconciles, and none of that may touch a
# live central's table (or its cross-process epoch).
_TMP = tempfile.mkdtemp(prefix="hugpy-model-physical-test-")
os.environ["HUGPY_MODEL_PHYSICAL_PATH"] = os.path.join(_TMP, "physical.json")
os.environ["HUGPY_MODEL_STATUS_EPOCH_PATH"] = os.path.join(_TMP, "epoch")
for _var in ("HUGPY_MODEL_PHYSICAL_MAX_AGE_S", "HUGPY_MODEL_PHYSICAL_POLL_S"):
    os.environ.pop(_var, None)

store = importlib.import_module("hugpy_storage.model_physical")

# flask_app.app is shadowed by flask's own `app` submodule until the app has
# booted once — same preamble as tests/test_model_status_cache.py.
importlib.import_module("hugpy_server.wsgi_app").get_hugpy_flask()
dl = importlib.import_module(
    "hugpy_storage.console.downloader")
mp = importlib.import_module(
    "hugpy_storage.console.model_physical")
cd = importlib.import_module(
    "hugpy_storage.console.cancelable_downloads")
routes = importlib.import_module(
    "hugpy_server.app.routes.llm_storage_routes")
v1 = importlib.import_module("hugpy_server.app.routes.v1_routes")
mc = importlib.import_module("hugpy_engine.config.models.models_config")
cache = importlib.import_module("hugpy_storage.model_status_cache")

_job_store = importlib.import_module("hugpy_control.jobs").job_store

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
        "model_max_length": 4096,
    }
    m.update(over)
    return m


def _manifest(n=25):
    return {f"repo-{i}": _model(f"repo-{i}") for i in range(n)}


def _flask_app():
    return importlib.import_module(
        "hugpy_server.wsgi_app").get_hugpy_flask()


class CountingStat:
    """Stand-in for ``model_status`` that counts how often the store is read."""

    def __init__(self, status="installed", destination="/store/x"):
        self.calls = 0
        self.status = status
        self.destination = destination
        self.lock = threading.Lock()

    def __call__(self, model):
        with self.lock:
            self.calls += 1
        mk = model.get("model_key")
        return {"status": self.status,
                "destination": f"{self.destination}/{mk}",
                "installed_marker": f"{self.destination}/{mk}/hugpy.json"}


class CountingSizes:
    """Stand-in for the two size annotators; counts the (much heavier) walk."""

    def __init__(self, base=1_000):
        self.calls = 0
        self.base = base

    def gguf(self, model, mk):
        self.calls += 1
        model["effective_bytes"] = self.base
        model["effective_gguf"] = f"{mk}.Q4_K_M.gguf"
        model["gguf_variants"] = [{"filename": f"{mk}.Q4_K_M.gguf",
                                   "bytes": self.base, "is_effective": True}]
        model["mmproj_bytes"] = None

    def size(self, model, mk):
        # the real annotate_size short-circuits on effective_bytes
        model["size_bytes"] = model.get("effective_bytes") or 0
        model.setdefault("dir_bytes", model["size_bytes"])


def _install_stubs(monkeypatch, status="installed"):
    live = CountingStat(status=status)
    sizes = CountingSizes()
    monkeypatch.setattr(dl, "model_status", live)
    monkeypatch.setattr(mp, "annotate_gguf_size", sizes.gguf)
    monkeypatch.setattr(mp, "annotate_size", sizes.size)
    return live, sizes


def _wire_manifest(monkeypatch, manifest):
    monkeypatch.setattr(v1, "get_models_dict", lambda **_k: manifest)
    monkeypatch.setattr(v1, "api_key_required", lambda: False)
    monkeypatch.setattr(routes, "get_models_dict", lambda **_k: manifest)


class OsCounter:
    """Counts real filesystem syscalls across a block of code."""

    NAMES = ("stat", "lstat", "listdir", "scandir", "walk")

    def __init__(self):
        self.n = 0
        self._orig = {}

    def __enter__(self):
        for name in self.NAMES:
            fn = getattr(os, name)
            self._orig[name] = fn
            setattr(os, name, self._wrap(fn))
        return self

    def _wrap(self, fn):
        def inner(*a, **k):
            self.n += 1
            return fn(*a, **k)
        return inner

    def __exit__(self, *_exc):
        for name, fn in self._orig.items():
            setattr(os, name, fn)
        return False


def _catalog_stamp():
    """(mtime_ns, size) of the registry's two persisted artifacts."""
    from hugpy_platform.constants import MODELS_DICT_PATH, MODELS_DISCOVERY_PATH
    out = {}
    for p in (str(MODELS_DISCOVERY_PATH), str(MODELS_DICT_PATH)):
        try:
            st = os.stat(p)
            out[p] = (st.st_mtime_ns, st.st_size)
        except OSError:
            out[p] = None
    return out


@pytest.fixture(autouse=True)
def _live_catalog_is_off_limits():
    """Fail LOUDLY if a test writes the live discovery report or manifest.

    Earned 2026-07-27: a test called ``refresh_registry(run_discovery=True)``
    with ``get_models_dict`` stubbed, and ``discover_models(save_json=True)``
    overwrote ``/mnt/llm_storage/projects/model_discovery.json`` with ``{}``.
    Nothing in this suite has any business touching a real catalog file."""
    before = _catalog_stamp()
    yield
    after = _catalog_stamp()
    changed = [p for p in before if before[p] != after[p]]
    assert not changed, f"a test modified live registry state: {changed}"


@pytest.fixture(autouse=True)
def _no_async_catalog_refresh(monkeypatch):
    """Since the partition, every ``catalog.changed`` on the control bus makes
    the engine's catalog bridge re-derive the registry on a daemon thread
    (``hugpy_engine.catalog_bridge``). These tests stub the manifest
    (``get_models_dict``) but not the registry, so that background refresh
    would race them and drop rows the empty test registry does not back.
    Silence the listener; the synchronous paths under test are unchanged."""
    cb = importlib.import_module("hugpy_engine.catalog_bridge")
    monkeypatch.setattr(cb, "refresh_from_event", lambda *_a, **_k: None)


@pytest.fixture(autouse=True)
def _clean_store():
    """A fresh, empty persisted table for every test."""
    for var in ("HUGPY_MODEL_PHYSICAL_MAX_AGE_S", "HUGPY_MODEL_PHYSICAL_POLL_S",
                "HUGPY_MODEL_STATUS_TTL_S"):
        os.environ.pop(var, None)
    # poll=0 => revalidate on every lookup, so a test never depends on timing.
    os.environ["HUGPY_MODEL_PHYSICAL_POLL_S"] = "0"
    try:
        os.remove(store.physical_store.path())
    except OSError:
        pass
    store.reset_physical_store()
    cache.reset_model_status_cache()
    yield
    try:
        os.remove(store.physical_store.path())
    except OSError:
        pass
    store.reset_physical_store()
    cache.reset_model_status_cache()
    for var in ("HUGPY_MODEL_PHYSICAL_MAX_AGE_S", "HUGPY_MODEL_PHYSICAL_POLL_S",
                "HUGPY_MODEL_STATUS_TTL_S"):
        os.environ.pop(var, None)


# ──────────────────────────────────────────────────────────────────────────
# 1. the store itself
# ──────────────────────────────────────────────────────────────────────────
def test_absent_means_derive_never_a_default_record():
    fields, state = store.lookup_physical("nope", _model("nope"))
    assert fields is None
    assert state == "absent"
    assert store.physical_store.record("nope") is None


def test_a_record_carries_its_provenance():
    m = _model()
    store.record_physical("repo-0", m, {"status": "installed",
                                        "destination": "/store/x/repo-0"},
                          [store.ASPECT_STATUS], source="unit",
                          dir_mtime=1234.5)
    rec = store.physical_store.record("repo-0")
    assert rec["identity"] == store.identity_of(m)
    assert rec["aspects"] == [store.ASPECT_STATUS]
    assert rec["source"] == "unit"
    assert rec["dir_mtime"] == 1234.5
    assert abs(rec["derived_at"] - time.time()) < 30
    assert rec["fields"]["status"] == "installed"


def test_a_routing_change_reads_as_absent_not_as_the_old_answer():
    """Re-key / re-route a model and its predecessor's record must not answer."""
    base = _model()
    store.record_physical("repo-0", base, {"status": "installed"},
                          [store.ASPECT_STATUS])
    assert store.lookup_physical("repo-0", base)[1] == "fresh"
    for field, value in (("hub_id", "org/other"), ("filename", "x.gguf"),
                         ("framework", "transformers"), ("dir", "/elsewhere"),
                         ("include", ["*.gguf"]), ("tasks", ["image-to-image"])):
        moved = _model(**{field: value})
        fields, state = store.lookup_physical("repo-0", moved)
        assert fields is None and state == "identity-changed", field


def test_the_two_aspects_are_independent():
    m = _model()
    store.record_physical("repo-0", m, {"status": "installed"},
                          [store.ASPECT_STATUS])
    assert store.lookup_physical("repo-0", m, store.ASPECT_STATUS)[1] == "fresh"
    assert store.lookup_physical("repo-0", m, store.ASPECT_SIZE)[1] == "aspect-missing"
    store.record_physical("repo-0", m, {"size_bytes": 7}, [store.ASPECT_SIZE])
    # writing the size half must not lose the status half
    assert store.lookup_physical("repo-0", m, store.ASPECT_STATUS)[0]["status"] == "installed"
    assert store.lookup_physical("repo-0", m, store.ASPECT_SIZE)[0]["size_bytes"] == 7


def test_re_deriving_an_aspect_drops_keys_the_deriver_stopped_producing():
    m = _model()
    store.record_physical("repo-0", m,
                          {"status": "installed", "destination": "/d",
                           "filename_warning": "pinned quant missing"},
                          [store.ASPECT_STATUS])
    store.record_physical("repo-0", m, {"status": "installed", "destination": "/d"},
                          [store.ASPECT_STATUS])
    got = store.lookup_physical("repo-0", m)[0]
    assert "filename_warning" not in got, (
        "a stale warning would linger on the cached registry row forever")


def test_max_age_expires_a_record_and_the_expiry_is_jittered():
    os.environ["HUGPY_MODEL_PHYSICAL_MAX_AGE_S"] = "0.2"
    m = _model()
    store.record_physical("repo-0", m, {"status": "installed"},
                          [store.ASPECT_STATUS])
    assert store.lookup_physical("repo-0", m)[1] == "fresh"
    time.sleep(0.35)
    fields, state = store.lookup_physical("repo-0", m)
    assert state == "expired"
    assert fields["status"] == "installed", (
        "an expired record is dated, not invented — it is still real data")
    # jitter: a rebuild stamps ~107 rows in the same second; they must not all
    # come due on the same request.
    ages = {round(store.max_age_for(f"repo-{i}"), 6) for i in range(50)}
    assert len(ages) > 40, "expiry is not spread across keys"
    assert all(0.15 <= a <= 0.25 for a in ages), sorted(ages)[:3]


def test_max_age_zero_never_expires():
    os.environ["HUGPY_MODEL_PHYSICAL_MAX_AGE_S"] = "0"
    m = _model()
    store.record_physical("repo-0", m, {"status": "installed"},
                          [store.ASPECT_STATUS])
    assert store.max_age_for("repo-0") == 0.0
    assert store.lookup_physical("repo-0", m)[1] == "fresh"


def test_a_corrupt_table_degrades_to_derive_not_to_a_wrong_answer():
    m = _model()
    store.record_physical("repo-0", m, {"status": "installed"},
                          [store.ASPECT_STATUS])
    with open(store.physical_store.path(), "w", encoding="utf-8") as fh:
        fh.write("{not json at all")
    store.reset_physical_store()
    fields, state = store.lookup_physical("repo-0", m)
    assert fields is None and state == "absent"


def test_a_non_serializable_field_is_never_persisted():
    m = _model()
    assert store.record_physical("repo-0", m,
                                 {"status": "installed", "moe": {1, 2, 3}},
                                 [store.ASPECT_STATUS, store.ASPECT_SIZE]) is False
    assert store.physical_store.record("repo-0") is None


def test_an_unwritable_store_degrades_and_never_raises(monkeypatch):
    monkeypatch.setenv("HUGPY_MODEL_PHYSICAL_PATH",
                       "/proc/definitely/not/writable/physical.json")
    store.reset_physical_store()
    m = _model()
    assert store.record_physical("repo-0", m, {"status": "installed"},
                                 [store.ASPECT_STATUS]) is False
    assert store.lookup_physical("repo-0", m)[1] == "absent"


# ──────────────────────────────────────────────────────────────────────────
# 2. the read path — zero per-model filesystem calls when warm
# ──────────────────────────────────────────────────────────────────────────
def test_v1_models_warm_makes_zero_per_model_store_reads(monkeypatch):
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest()
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()

    with app.test_client() as client:
        cold = client.get("/v1/models")
        assert cold.status_code == 200
        cold_body = cold.get_json()
        assert live.calls == len(manifest)
        warm_body = client.get("/v1/models").get_json()

    assert live.calls == len(manifest), "a warm /v1/models re-read the store"
    assert cold_body == warm_body
    assert len(cold_body["data"]) == len(manifest)
    assert sizes.calls == 0, (
        "/v1/models never shows a size — it must not pay the size walk")


def test_models_warm_makes_zero_per_model_store_reads(monkeypatch):
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest()
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()

    with app.test_client() as client:
        cold_body = client.get("/models").get_json()
        assert live.calls == len(manifest)
        assert sizes.calls == len(manifest)
        warm_body = client.get("/models").get_json()

    assert live.calls == len(manifest), "a warm /models re-read the status"
    assert sizes.calls == len(manifest), "a warm /models re-walked the sizes"
    assert cold_body == warm_body


def test_the_two_listings_share_one_persisted_record(monkeypatch):
    """/models and /v1/models must not each keep their own idea of status."""
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(10)
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()
    with app.test_client() as client:
        client.get("/models")
        assert live.calls == len(manifest)
        client.get("/v1/models")
        assert live.calls == len(manifest), (
            "/v1/models re-derived what /models had already persisted")


def test_a_warm_listing_makes_no_per_model_filesystem_calls(monkeypatch,
                                                            tmp_path):
    """The end-to-end measurement, with the REAL derivation against a throwaway
    store — the thing that was killing central, not a stub."""
    paths = importlib.import_module(
        "hugpy_storage.model_paths")
    root = tmp_path / "store"
    n = 20
    for i in range(n):
        d = root / "models" / "gguf" / f"org{i % 4}" / f"repo-{i}"
        d.mkdir(parents=True)
        (d / f"repo-{i}.Q4_K_M.gguf").write_bytes(b"\0" * (2 * 1024 * 1024))
    for task in ("text-generation", "image-text-to-text"):
        (root / "models" / "gguf" / task).mkdir(parents=True, exist_ok=True)

    real_route = paths.route_destination
    monkeypatch.setattr(dl, "route_destination",
                        lambda m, _r=str(root): real_route(m, _r))
    manifest = {f"repo-{i}": _model(f"repo-{i}") for i in range(n)}
    _wire_manifest(monkeypatch, manifest)
    # Poll effectively off so the warm pass is a pure in-memory lookup and the
    # count cannot drift with how long the surrounding suite took (cross-process
    # revalidation is tested separately, and is O(1) not O(models)).
    monkeypatch.setenv("HUGPY_MODEL_PHYSICAL_POLL_S", "86400")
    meta = importlib.import_module(
        "hugpy_engine.config.models.model_meta")
    app = _flask_app()

    spill = importlib.import_module("hugpy_engine.spill")

    def _reset_everything_derived():
        """Back to a genuinely cold central: no record, no in-process size or
        MoE-header cache, and FRESH registry rows (a listing stamps
        `destination` onto the cached row in place, which itself saves the sizer
        a resolution). Every one of these is module-level state that another
        test in the session may have warmed."""
        store.physical_store.reset()
        try:
            os.remove(store.physical_store.path())
        except OSError:
            pass
        with meta._SIZE_LOCK:
            meta._SIZE_CACHE.clear()
        spill._MOE_DETAIL_CACHE.clear()
        manifest.clear()
        manifest.update({f"repo-{i}": _model(f"repo-{i}") for i in range(n)})

    with app.test_client() as client:
        client.get("/models")            # warm the kernel dentry cache first,
        _reset_everything_derived()      # so we measure OUR I/O, not first-touch
        with OsCounter() as cold:
            cold_body = client.get("/models").get_json()
        with OsCounter() as warm:
            warm_body = client.get("/models").get_json()
        # …and again over a QUARTER of the manifest. What is left is per-REQUEST
        # (the media allow-flag + media-default stores, read once each), not
        # per-MODEL: the count must not move with the number of models.
        quarter = {k: manifest[k] for k in list(manifest)[:n // 4]}
        manifest.clear()
        manifest.update(quarter)
        with OsCounter() as warm_small:
            client.get("/models")

    assert cold_body == warm_body, (
        "the persisted record must not change what the listing reports")
    assert cold.n > 20 * max(warm.n, 1), (
        f"expected a real walk on the cold pass, saw {cold.n} os calls "
        f"against {warm.n} warm")
    assert warm.n == warm_small.n, (
        f"a warm /models still scales with the manifest: {warm.n} filesystem "
        f"calls for {n} models vs {warm_small.n} for {n // 4}")
    assert warm.n < n, (
        f"a warm /models made {warm.n} filesystem calls for {n} models "
        f"(cold pass was {cold.n})")
    print(json.dumps({"models": n, "cold_os_calls": cold.n,
                      "warm_os_calls": warm.n,
                      "warm_os_calls_quarter_manifest": warm_small.n}))


def test_measured_filesystem_calls_before_and_after(tmp_path, monkeypatch,
                                                    capsys):
    """THE number, for both endpoints, against a real synthetic store.

    BEFORE = the shipped behaviour (derive on every request). AFTER = the
    persisted record. Counts real ``os.*`` calls, never wall time."""
    paths = importlib.import_module(
        "hugpy_storage.model_paths")
    meta = importlib.import_module(
        "hugpy_engine.config.models.model_meta")
    root = tmp_path / "store"
    n = 20
    for i in range(n):
        d = root / "models" / "gguf" / f"org{i % 4}" / f"repo-{i}"
        d.mkdir(parents=True)
        (d / f"repo-{i}.Q4_K_M.gguf").write_bytes(b"\0" * (1024 * 1024))
    real_route = paths.route_destination
    monkeypatch.setattr(dl, "route_destination",
                        lambda m, _r=str(root): real_route(m, _r))
    monkeypatch.setenv("HUGPY_MODEL_PHYSICAL_POLL_S", "86400")
    manifest = {}
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()

    def _cold():
        store.physical_store.reset()
        try:
            os.remove(store.physical_store.path())
        except OSError:
            pass
        with meta._SIZE_LOCK:
            meta._SIZE_CACHE.clear()
        manifest.clear()
        manifest.update({f"repo-{i}": _model(f"repo-{i}") for i in range(n)})

    def _measure(url, derive_every_time):
        _cold()
        with app.test_client() as client:
            client.get(url)                      # warm the kernel dentry cache
            _cold()
            if derive_every_time:
                # the shipped behaviour: nothing persisted, nothing looked up
                monkeypatch.setattr(store.physical_store, "lookup",
                                    lambda *_a, **_k: (None, "absent"))
                monkeypatch.setattr(store.physical_store, "put",
                                    lambda *_a, **_k: False)
            with OsCounter() as first:
                client.get(url)
            with OsCounter() as second:
                client.get(url)
            if derive_every_time:
                monkeypatch.undo()
                monkeypatch.setattr(dl, "route_destination",
                                    lambda m, _r=str(root): real_route(m, _r))
                monkeypatch.setenv("HUGPY_MODEL_PHYSICAL_POLL_S", "86400")
                _wire_manifest(monkeypatch, manifest)
        return first.n, second.n

    report = {"models": n}
    for url, label in (("/models", "models"), ("/v1/models", "v1_models")):
        before_1, before_2 = _measure(url, derive_every_time=True)
        after_1, after_2 = _measure(url, derive_every_time=False)
        report[label] = {
            "before_first_request": before_1,
            "before_second_request": before_2,
            "after_first_request": after_1,
            "after_second_request": after_2,
        }
        assert before_2 >= before_1 * 0.5 > 0, (
            f"{url}: BEFORE must re-derive on every request")
        # What is left after the change is per-REQUEST, not per-MODEL: the
        # media allow-flag store and the media-default store, read once each
        # (test_a_warm_listing_makes_no_per_model_filesystem_calls proves the
        # count does not move with the manifest size).
        assert after_2 <= 4, (
            f"{url}: a warm request made {after_2} filesystem calls for {n} "
            f"models — that is per-model work, not per-request")
        assert after_2 * 10 < before_2, (
            f"{url}: {before_2} -> {after_2} is not the order-of-magnitude win")
    with capsys.disabled():
        print("\nMEASURED os.* calls: " + json.dumps(report))


def test_response_is_byte_identical_to_deriving_live(monkeypatch):
    """Same state, same bytes — only the cost changes."""
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(12)
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()

    # "today": force every lookup to miss, so every field is derived live.
    monkeypatch.setattr(store.physical_store, "lookup",
                        lambda *_a, **_k: (None, "absent"))
    with app.test_client() as client:
        live_models = client.get("/models").get_data()
        live_v1 = client.get("/v1/models").get_data()
    monkeypatch.undo()

    live2, _ = _install_stubs(monkeypatch)
    _wire_manifest(monkeypatch, manifest)
    with app.test_client() as client:
        client.get("/models")                                   # populate
        persisted_models = client.get("/models").get_data()
        persisted_v1 = client.get("/v1/models").get_data()

    assert persisted_models == live_models
    assert persisted_v1 == live_v1


def test_re_deriving_an_already_stamped_row_keeps_every_size_field(monkeypatch):
    """Registry rows are cached and stamped IN PLACE, so a re-derive is handed a
    row that already carries last time's numbers. A field whose value did not
    change must survive — being right twice cannot cost you the key."""
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(2)
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()
    with app.test_client() as client:
        first = client.get("/models").get_json()
        assert first[0]["size_bytes"] == 1_000
        # every event drops the record; the row itself keeps its stamped keys
        store.forget_all_physical("test")
        second = client.get("/models").get_json()
    assert second == first, "a re-derive lost fields whose values were unchanged"
    rec = store.physical_store.record("repo-0")
    assert rec["fields"]["size_bytes"] == 1_000
    assert rec["fields"]["effective_bytes"] == 1_000


def test_a_row_with_no_record_derives_rather_than_reporting_zeros(monkeypatch):
    """The first run after this change, and every newly-added model."""
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(3)
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()
    assert store.physical_store.keys() == []

    with app.test_client() as client:
        body = client.get("/models").get_json()

    assert len(body) == 3
    for row in body:
        assert row["status"] == "installed", "an absent record reported absent"
        assert row["size_bytes"] == 1_000, "an absent record reported 0 bytes"
        assert row["effective_bytes"] == 1_000
    assert sorted(store.physical_store.keys()) == sorted(manifest)


def test_a_new_model_appearing_in_the_registry_is_derived(monkeypatch):
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(3)
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()
    with app.test_client() as client:
        client.get("/models")
        assert live.calls == 3
        manifest["repo-new"] = _model("repo-new")
        body = client.get("/models").get_json()
        assert live.calls == 4, "the new model was not derived"
    assert [r for r in body if r["model_key"] == "repo-new"][0]["status"] == "installed"


# ──────────────────────────────────────────────────────────────────────────
# 3. write events
# ──────────────────────────────────────────────────────────────────────────
def _warm(app, monkeypatch, manifest):
    with app.test_client() as client:
        client.get("/models")
    return set(store.physical_store.keys())


def test_delete_drops_only_that_row_and_stops_reporting_installed(monkeypatch,
                                                                  tmp_path):
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(4)
    _wire_manifest(monkeypatch, manifest)
    dest = tmp_path / "models" / "gguf" / "org" / "repo-0"
    dest.mkdir(parents=True)
    (dest / "w.gguf").write_bytes(b"x")
    monkeypatch.setattr(routes, "route_destination", lambda _m: str(dest))
    app = _flask_app()

    with app.test_client() as client:
        client.get("/models")
        assert live.calls == 4
        assert set(store.physical_store.keys()) == set(manifest)

        assert client.delete("/models/repo-0").get_json()["deleted"] is True
        assert store.physical_store.record("repo-0") is None
        assert store.physical_store.record("repo-1") is not None, (
            "delete must be TARGETED — the other rows stay warm")

        # the files are gone, so the live derive now says not_installed
        live.status = "not_installed"
        body = client.get("/models").get_json()
        assert live.calls == 5, "only the deleted model was re-derived"

    rows = {r["model_key"]: r for r in body}
    assert rows["repo-0"]["status"] == "not_installed"
    assert rows["repo-1"]["status"] == "installed"


def test_prune_drops_only_that_row(monkeypatch, tmp_path):
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(3)
    _wire_manifest(monkeypatch, manifest)
    monkeypatch.setattr(routes, "route_destination",
                        lambda _m: str(tmp_path / "gone"))
    monkeypatch.setattr(routes, "prune_model", lambda k: {"pruned": True, "key": k})
    app = _flask_app()
    with app.test_client() as client:
        client.get("/models")
        assert client.post("/models/repo-0/prune").status_code == 200
    assert store.physical_store.record("repo-0") is None
    assert store.physical_store.record("repo-1") is not None


def test_reconcile_apply_drops_the_whole_table_but_a_dry_run_does_not(monkeypatch):
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(3)
    _wire_manifest(monkeypatch, manifest)
    reconcile = importlib.import_module("hugpy_engine.apis.reconcile")
    monkeypatch.setattr(reconcile, "reconcile_store",
                        lambda **_k: {"actions": [], "warnings": []})
    app = _flask_app()
    with app.test_client() as client:
        client.get("/models")
    assert len(store.physical_store.keys()) == 3

    with app.test_request_context("/models/reconcile", method="POST",
                                  json={"apply": False}):
        _b, status = routes.reconcile_store_route()
        assert status == 202
    assert len(store.physical_store.keys()) == 3, "a dry run touches nothing"

    with app.test_request_context("/models/reconcile", method="POST",
                                  json={"apply": True}):
        _b, status = routes.reconcile_store_route()
        assert status == 200
    assert store.physical_store.keys() == [], (
        "an applied reconcile moves weights — every destination is suspect")


def test_download_terminal_states_drop_the_downloaded_row(monkeypatch, tmp_path):
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(3)
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()

    for exitcode, expect in ((0, "completed"), (1, "failed")):
        with app.test_client() as client:
            client.get("/models")
        assert store.physical_store.record("repo-0") is not None
        job = _drive_download(monkeypatch, tmp_path / expect, exitcode=exitcode)
        assert job.terminal
        assert store.physical_store.record("repo-0") is None, expect
        assert store.physical_store.record("repo-1") is not None, (
            f"{expect} must be TARGETED at the model that was downloading")


def test_download_cancel_drops_the_downloading_row(monkeypatch):
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(3)
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()
    with app.test_client() as client:
        client.get("/models")
    job = _job_store.create("repo-0", kind="download", transport="test")
    assert cd.cancel_download(job.id)["cancelled"] is True
    assert store.physical_store.record("repo-0") is None
    assert store.physical_store.record("repo-1") is not None


def _quarantine_refresh_registry(monkeypatch):
    """Make ``refresh_registry`` safe to call in a test.

    LANDMINE (learned the hard way 2026-07-27): ``refresh_registry`` is not a
    pure function. ``run_discovery=True`` runs ``discover_models(save_json=True)``,
    which WALKS THE LIVE 16TB STORE and OVERWRITES ``model_discovery.json`` —
    the registry's persisted half — and a stubbed ``get_models_dict`` makes it
    write an EMPTY report. Nothing here may reach the real report, so the walk
    is stubbed out and the two expensive follow-ups with it."""
    md = importlib.import_module(
        "hugpy_engine.config.models.models_default")
    ov = importlib.import_module("hugpy_engine.serve.overrides")
    gm = importlib.import_module("hugpy_engine.apis.get_module")
    monkeypatch.setattr(md, "refresh_task_registries", lambda *a, **k: None)
    monkeypatch.setattr(ov, "migrate_overrides", lambda *a, **k: [])
    monkeypatch.setattr(gm, "discover_models",
                        lambda **_k: dict(mc.MODEL_REGISTRY_DICT))


def test_refresh_registry_with_discovery_drops_everything(monkeypatch):
    """A real store walk happened — anything on disk may have moved."""
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(3)
    _wire_manifest(monkeypatch, manifest)
    _quarantine_refresh_registry(monkeypatch)
    app = _flask_app()
    with app.test_client() as client:
        client.get("/models")
    assert len(store.physical_store.keys()) == 3

    mc.refresh_registry(run_discovery=True)
    assert store.physical_store.keys() == [], (
        "a discovery re-walk must drop the whole table")


def test_refresh_registry_without_discovery_drops_re_keyed_rows(monkeypatch):
    """The identity reconcile: a row the registry no longer backs, or whose
    routing identity moved, cannot keep answering."""
    live, sizes = _install_stubs(monkeypatch)
    registry = {"repo-0": _model("repo-0"), "repo-1": _model("repo-1")}
    _wire_manifest(monkeypatch, registry)
    app = _flask_app()
    with app.test_client() as client:
        client.get("/models")
    assert sorted(store.physical_store.keys()) == ["repo-0", "repo-1"]

    # repo-1 gets re-routed; repo-2 was never in the registry
    store.record_physical("repo-2", _model("repo-2"), {"status": "installed"},
                          [store.ASPECT_STATUS])
    moved = {"repo-0": _model("repo-0"),
             "repo-1": _model("repo-1", framework="transformers")}
    dropped = store.reconcile_physical_identities(moved, "test")
    assert dropped == 2
    assert store.physical_store.keys() == ["repo-0"]


def test_a_gguf_file_override_drops_the_size_record(monkeypatch, tmp_path):
    """The effective quant is an operator CHOICE, not part of routing identity —
    nothing else would notice it changed."""
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(3)
    _wire_manifest(monkeypatch, manifest)
    ov = importlib.import_module("hugpy_engine.serve.overrides")
    monkeypatch.setattr(ov, "_OVERRIDES_PATH", str(tmp_path / "serve.json"))
    app = _flask_app()
    with app.test_client() as client:
        client.get("/models")
    assert store.physical_store.record("repo-0") is not None

    ov.set_override("repo-0", {"gguf_file": "repo-0.Q8_0.gguf"})
    assert store.physical_store.record("repo-0") is None
    assert store.physical_store.record("repo-1") is not None

    # an unrelated override must NOT churn the record
    ov.set_override("repo-1", {"threads": 4})
    assert store.physical_store.record("repo-1") is not None


# ──────────────────────────────────────────────────────────────────────────
# 4. force refresh + repair
# ──────────────────────────────────────────────────────────────────────────
def test_the_single_model_route_is_always_live_and_repairs_the_record(monkeypatch):
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(3)
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()
    with app.test_client() as client:
        client.get("/models")
        assert live.calls == 3
        client.get("/models/repo-0")
        client.get("/models/repo-0")
        assert live.calls == 5, "the detail route must always read the store"

        # out-of-band change nobody told us about -> opening the row fixes it
        live.status = "not_installed"
        detail = client.get("/models/repo-0").get_json()
        assert detail["status"] == "not_installed"
        listed = {r["model_key"]: r for r in client.get("/models").get_json()}
    assert listed["repo-0"]["status"] == "not_installed", (
        "opening the row must repair the record the listings read")
    assert listed["repo-1"]["status"] == "installed"


def test_the_repair_sweep_fixes_an_out_of_band_change(monkeypatch):
    """The store is SHARED: another box wrote, an operator moved a dir, the
    reaper deleted. No event fires — /models/discover is the repair path."""
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(4)
    _wire_manifest(monkeypatch, manifest)
    app = _flask_app()
    with app.test_client() as client:
        body = client.get("/models").get_json()
    assert all(r["status"] == "installed" for r in body)

    live.status = "not_installed"                 # out-of-band deletion
    with app.test_client() as client:
        stale = client.get("/models").get_json()
    assert all(r["status"] == "installed" for r in stale), (
        "documented behaviour: no event, no per-request stat — still persisted")

    result = mp.rebuild_physical(manifest, source="test-repair")
    assert result == {"written": 4, "failed": 0}
    with app.test_client() as client:
        repaired = client.get("/models").get_json()
    assert all(r["status"] == "not_installed" for r in repaired)


def test_the_discover_route_runs_the_repair_sweep(monkeypatch, tmp_path):
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(3)
    _wire_manifest(monkeypatch, manifest)
    monkeypatch.setattr(routes, "refresh_registry", lambda **_k: None)
    # _run_discovery persists its progress beside the LIVE discovery report —
    # send it somewhere disposable instead.
    monkeypatch.setattr(routes, "_discover_state_path",
                        lambda: str(tmp_path / "discover.state.json"))
    state = {"running": True, "error": None}
    routes._run_discovery(state)
    assert state["error"] is None, state["error"]
    assert state["running"] is False
    assert state["physical"] == {"written": 3, "failed": 0}
    assert sorted(store.physical_store.keys()) == sorted(manifest)


def test_the_repair_sweep_skips_a_model_it_cannot_derive(monkeypatch):
    """One bad model must not stop the sweep, and must not get a fake record."""
    manifest = _manifest(3)

    def exploding(model):
        if model.get("model_key") == "repo-1":
            raise OSError("virtiofs went away")
        return {"status": "installed", "destination": "/d"}

    monkeypatch.setattr(dl, "model_status", exploding)
    monkeypatch.setattr(mp, "annotate_gguf_size", lambda *_a: None)
    monkeypatch.setattr(mp, "annotate_size", lambda *_a: None)
    result = mp.rebuild_physical(manifest, source="test")
    assert result == {"written": 2, "failed": 1}
    assert store.physical_store.record("repo-1") is None


def test_a_live_derive_error_still_propagates(monkeypatch):
    """Today's behaviour: a store read that raises is not swallowed and is
    certainly not replaced by an invented status."""
    def exploding(_model):
        raise OSError("virtiofs went away")

    monkeypatch.setattr(dl, "model_status", exploding)
    with pytest.raises(OSError):
        mp.status_fields(_model())
    assert store.physical_store.record("repo-0") is None


# ──────────────────────────────────────────────────────────────────────────
# 4b. /llm/workers — the third endpoint with the same disease
# ──────────────────────────────────────────────────────────────────────────
# Building a view of THREE machines cost 11.8s cold / 5.3s warm, and 31.0s for
# GET /llm/workers under the console's continuous polling, because _public_view
# derives PHYSICAL state per DESIGNATED model — ae alone has 75 — through
# _model_size_bytes / _model_moe_detail / _model_moe_gpu_bytes /
# _model_marker_flag, each of which walked the store. They are lookups now.
wk = importlib.import_module(
    "hugpy_fleet.central.workers")


def _wire_workers_manifest(monkeypatch, manifest):
    mcm = importlib.import_module(
        "hugpy_engine.config.models.models_config")
    monkeypatch.setattr(mcm, "get_models_dict", lambda **_k: manifest)


def test_worker_view_sizing_is_a_lookup_not_a_store_walk(monkeypatch):
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(30)
    _wire_workers_manifest(monkeypatch, manifest)

    first = [wk._model_size_bytes(mk) for mk in manifest]
    assert all(v == 1_000 for v in first)
    assert sizes.calls == len(manifest), "the cold pass must derive each model once"

    # Every downstream helper now reads the SAME record — the whole point is
    # that four callers per model stop being four store walks per model.
    for mk in manifest:
        wk._model_size_bytes(mk)
        wk._model_moe_detail(mk)
        wk._model_moe_gpu_bytes(mk)
    assert sizes.calls == len(manifest), (
        f"the workers view re-walked the store: {sizes.calls} derivations for "
        f"{len(manifest)} models")


def test_worker_view_marker_reads_are_a_lookup(monkeypatch):
    manifest = _manifest(20)
    _wire_workers_manifest(monkeypatch, manifest)
    reads = {"n": 0}

    def fake_marker(model, mk):
        reads["n"] += 1
        return {"hugpy_marker": {"moe_capable": True, "bnb_capable": False}}

    monkeypatch.setattr(mp, "derive_marker", fake_marker)
    for mk in manifest:
        assert wk._model_marker_flag(mk, "moe_capable") is True
        assert wk._model_marker_flag(mk, "bnb_capable") is False
    assert reads["n"] == len(manifest), "two fields must share ONE marker read"
    for mk in manifest:
        wk._model_marker_flag(mk, "moe_capable")
    assert reads["n"] == len(manifest), "a warm marker read touched the store"


def test_an_unreadable_marker_stays_unknown_and_is_not_persisted(monkeypatch):
    """Degrade-not-guess: 'never determined' must not become a persisted False —
    that would silently withdraw the MoE/4-bit levers from a capable model."""
    manifest = _manifest(2)
    _wire_workers_manifest(monkeypatch, manifest)

    def boom(_model, _mk):
        raise OSError("virtiofs went away")

    monkeypatch.setattr(mp, "derive_marker", boom)
    assert wk._model_marker_flag("repo-0", "moe_capable") is None
    rec = store.physical_store.record("repo-0")
    assert rec is None or store.ASPECT_MARKER not in (rec.get("aspects") or [])


def test_a_model_central_does_not_know_reports_unknown_never_zero(monkeypatch):
    _wire_workers_manifest(monkeypatch, {})
    assert wk._model_size_bytes("who-is-this") is None
    assert wk._model_moe_detail("who-is-this") is None
    assert wk._model_marker_flag("who-is-this", "moe_capable") is None


def test_the_marker_aspect_never_leaks_into_the_listings(monkeypatch):
    """/models and /v1/models must not grow a `hugpy_marker` key."""
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(3)
    _wire_manifest(monkeypatch, manifest)
    _wire_workers_manifest(monkeypatch, manifest)
    monkeypatch.setattr(mp, "derive_marker",
                        lambda _m, _k: {"hugpy_marker": {"moe_capable": True}})
    for mk in manifest:
        wk._model_marker_flag(mk, "moe_capable")
    app = _flask_app()
    with app.test_client() as client:
        rows = client.get("/models").get_json()
        v1_rows = client.get("/v1/models").get_json()["data"]
    assert all("hugpy_marker" not in r for r in rows)
    assert all("hugpy_marker" not in r for r in v1_rows)
    # …and the record really does carry it, so this is not passing vacuously
    assert store.ASPECT_MARKER in store.physical_store.record("repo-0")["aspects"]


def test_a_cold_worker_view_is_time_bounded_and_fills_over_polls(monkeypatch):
    """_public_view is on the HEARTBEAT REPLY path. Deriving ~111 cold records
    inside one call is a multi-minute walk, and a heartbeat that slow blows
    HEARTBEAT_TIMEOUT_SECONDS and makes the whole fleet read offline — the
    "pushes off all of the workers" failure through the back door."""
    manifest = _manifest(40)
    _wire_workers_manifest(monkeypatch, manifest)
    derived = {"n": 0}

    def slow_status(model):
        derived["n"] += 1
        time.sleep(0.02)
        mk = model.get("model_key")
        return {"status": "installed", "destination": f"/store/{mk}",
                "installed_marker": f"/store/{mk}/hugpy.json"}

    monkeypatch.setattr(dl, "model_status", slow_status)
    monkeypatch.setattr(mp, "annotate_gguf_size",
                        lambda m, mk: m.__setitem__("effective_bytes", 1_000))
    monkeypatch.setattr(mp, "annotate_size",
                        lambda m, mk: m.__setitem__("size_bytes", 1_000))
    monkeypatch.setenv("HUGPY_WORKER_VIEW_FILL_BUDGET_S", "0.1")

    worker = {"id": "w1", "last_seen": time.time(), "models": list(manifest),
              "gpus": [], "limits": {}, "config": {}}
    t0 = time.time()
    wk._public_view(worker)
    elapsed = time.time() - t0
    assert derived["n"] < len(manifest), (
        "the cold fill was unbounded — a heartbeat would have blocked on it")
    assert elapsed < 2.0, f"a cold worker view took {elapsed:.2f}s"

    # …and it converges: successive polls finish the fill.
    for _ in range(60):
        wk._public_view(worker)
        if derived["n"] >= len(manifest):
            break
    assert derived["n"] == len(manifest), "the cold fill never converged"

    warm_before = derived["n"]
    wk._public_view(worker)
    assert derived["n"] == warm_before, "a warm worker view re-derived"


def test_an_unfilled_worker_row_reports_unknown_never_zero(monkeypatch):
    """Out of budget must mean UNKNOWN — a 0 would make an over-subscribed
    assignment set read as comfortably fitting."""
    manifest = _manifest(5)
    _wire_workers_manifest(monkeypatch, manifest)
    monkeypatch.setattr(wk, "_may_derive", lambda: False)
    assert wk._model_size_bytes("repo-0") is None
    assert wk._model_moe_detail("repo-0") is None
    assert wk._model_marker_flag("repo-0", "moe_capable") is None
    totals = wk.allocated_totals({"models": list(manifest)})
    assert totals["allocated_total_bytes"] == 0
    assert totals["allocated_unknown_count"] == 5, (
        "unknown sizes must be COUNTED and surfaced, never silently zeroed")


def test_a_worker_row_keeps_its_shape_and_its_live_fields(monkeypatch):
    """Only the DERIVED per-model physical facts became lookups. Everything the
    heartbeat reports — gpus, slots, status, last_seen — must stay live."""
    live, sizes = _install_stubs(monkeypatch)
    manifest = _manifest(4)
    _wire_workers_manifest(monkeypatch, manifest)
    worker = {
        "id": "w1", "name": "ae", "pkg_version": "0.1.220",
        "last_seen": time.time(),
        "gpus": [{"index": 0, "name": "RTX 3090",
                  "total_bytes": 24 * 2 ** 30, "free_bytes": 9 * 2 ** 30}],
        "slots": [{"id": "s0", "model_key": "repo-0", "pid": 4242,
                   "health": "ready"}],
        "models": list(manifest),
        "storage": {"cache_used_bytes": 10, "disk_free": 20, "models": []},
        "disk": {"free_bytes": 20, "total_bytes": 100},
        "ram_total": 64 * 2 ** 30,
        "limits": {}, "config": {"pinned": {"repo-0": True}},
        "spill_by_model": {}, "loaded_models": ["repo-0"],
    }
    view = wk._public_view(worker)
    for key in ("id", "name", "pkg_version", "gpus", "slots", "storage",
                "models", "spill_by_model", "config", "ram_total", "limits",
                "status", "last_seen"):
        assert key in view, f"/llm/workers row lost {key}"
    # LIVE, straight from the heartbeat — untouched by this change
    assert view["gpus"][0]["free_bytes"] == 9 * 2 ** 30
    assert view["slots"][0]["pid"] == 4242
    assert view["slots"][0]["health"] == "ready"
    assert view["status"] == "online"
    assert view["config"]["pinned"] == {"repo-0": True}
    # DERIVED per-model physical facts — now lookups, same values
    assert set(view["planned_split"]) == set(manifest)
    assert view["planned_split"]["repo-0"]["size_bytes"] == 1_000

    before = sizes.calls
    second = wk._public_view(worker)
    assert sizes.calls == before, (
        "a second worker view re-derived per-model physical state")
    assert second == view


# ──────────────────────────────────────────────────────────────────────────
# 5. cross-process (gunicorn --workers 3 + the reconcile path)
# ──────────────────────────────────────────────────────────────────────────
_CHILD = """
import os, sys, time
sys.path.insert(0, {src!r})
os.environ["HUGPY_MODEL_PHYSICAL_PATH"] = {path!r}
os.environ["HUGPY_MODEL_PHYSICAL_POLL_S"] = "0"
import importlib
s = importlib.import_module("hugpy_storage.model_physical")
lo, hi = int(sys.argv[1]), int(sys.argv[2])
for i in range(lo, hi):
    key = "repo-%d" % i
    s.record_physical(key, {{"model_key": key, "hub_id": "org/" + key}},
                      {{"status": "installed", "destination": "/d/" + key}},
                      [s.ASPECT_STATUS], source="child")
"""


def _run_children(nproc=4, per=25):
    path = store.physical_store.path()
    code = _CHILD.format(src=_SRC, path=path)
    procs = []
    for p in range(nproc):
        procs.append(subprocess.Popen(
            [sys.executable, "-c", code, str(p * per), str((p + 1) * per)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    for proc in procs:
        out, err = proc.communicate(timeout=300)
        assert proc.returncode == 0, err
    return nproc * per


def test_concurrent_writers_do_not_corrupt_the_table():
    """Three gunicorn workers plus the reconcile path all write here."""
    total = _run_children(nproc=4, per=25)
    with open(store.physical_store.path(), "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    assert raw["version"] == store.RECORD_VERSION
    assert len(raw["models"]) == total, (
        f"a concurrent write lost rows: {len(raw['models'])} of {total}")
    store.reset_physical_store()
    for i in range(total):
        key = f"repo-{i}"
        fields, state = store.lookup_physical(
            key, {"model_key": key, "hub_id": f"org/{key}"})
        assert state == "fresh", (key, state)
        assert fields["destination"] == f"/d/{key}"


def test_a_sibling_process_write_is_picked_up():
    """Process A serves a row; process B (a delete, another worker) drops it —
    A must stop serving the stale record."""
    m = _model()
    store.record_physical("repo-0", m, {"status": "installed"},
                          [store.ASPECT_STATUS])
    assert store.lookup_physical("repo-0", m)[1] == "fresh"

    code = (
        "import os,sys;"
        f"sys.path.insert(0, {_SRC!r});"
        f"os.environ['HUGPY_MODEL_PHYSICAL_PATH']={store.physical_store.path()!r};"
        "import importlib;"
        "s=importlib.import_module('hugpy_storage.model_physical');"
        "s.forget_physical('repo-0', 'sibling-process')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    fields, state = store.lookup_physical("repo-0", m)
    assert fields is None and state == "absent", (
        "a sibling gunicorn worker's drop was not observed")


def test_the_writing_process_is_never_stale_about_its_own_write(monkeypatch):
    monkeypatch.setenv("HUGPY_MODEL_PHYSICAL_POLL_S", "3600")
    store.reset_physical_store()
    m = _model()
    store.record_physical("repo-0", m, {"status": "installed"},
                          [store.ASPECT_STATUS])
    assert store.lookup_physical("repo-0", m)[1] == "fresh"
    store.forget_physical("repo-0", "self")
    assert store.lookup_physical("repo-0", m)[1] == "absent"


# ──────────────────────────────────────────────────────────────────────────
# download-monitor driver (shared with tests/test_model_status_cache.py's shape)
# ──────────────────────────────────────────────────────────────────────────
class _FakeProc:
    def __init__(self, exitcode=0):
        self.exitcode = exitcode
        self._alive = False

    def start(self):
        self._alive = False

    def is_alive(self):
        return self._alive

    def join(self, *_a, **_k):
        return None


class _FakeCtx:
    def __init__(self, exitcode=0):
        self.exitcode = exitcode

    def Process(self, *_a, **_k):
        return _FakeProc(self.exitcode)


def _drive_download(monkeypatch, dest_root, exitcode, max_attempts=1):
    """Run start_cancellable_download's monitor to a terminal state with no
    real subprocess, network or store walk."""
    dest = Path(dest_root) / "dest"
    dest.mkdir(parents=True, exist_ok=True)
    # The transfer lifecycle lives in downloader/engine.py now (it runs in the
    # hugpy-downloader daemon, not in gunicorn); cancelable_downloads only
    # re-exports it. Patch it where it is DEFINED — a patch on the re-export
    # would not be seen by the engine's own globals.
    eng = cd.engine
    monkeypatch.setattr(eng, "route_destination", lambda **_k: str(dest))
    monkeypatch.setattr(eng, "_estimate_total_bytes_bounded", lambda _m: None)
    monkeypatch.setattr(eng, "_watch", lambda *_a, **_k: False)
    monkeypatch.setattr(eng, "_read_error", lambda _j: "synthetic failure")
    monkeypatch.setattr(eng, "record_downloaded_model", lambda *a, **k: None)
    # The engine registry refresh reaches storage via the catalog seam now.
    monkeypatch.setattr(eng, "catalog_refresh", lambda *a, **k: None)
    monkeypatch.setattr(eng, "MAX_ATTEMPTS", max_attempts)
    monkeypatch.setattr(eng.mp, "get_context", lambda _n: _FakeCtx(exitcode))

    job = _job_store.create("repo-0", kind="download", transport="test")
    eng.start_cancellable_download(job, _model("repo-0"))
    deadline = time.time() + 30
    while time.time() < deadline:
        cur = _job_store.get(job.id)
        if cur is not None and cur.terminal:
            return cur
        time.sleep(0.05)
    raise AssertionError("download monitor never reached a terminal state")
