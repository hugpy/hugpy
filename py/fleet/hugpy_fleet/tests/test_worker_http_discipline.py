"""k59 — outbound worker-call discipline: split timeouts, breaker, ONE client.

The operator's complaint: "the API really needs to be more robust — it
constantly blips over just a few calls. The worker calls shouldn't fry the
endpoints." The dev central runs `gunicorn --workers 1 --threads 8`, so every
central→worker call spends one of EIGHT threads for as long as it takes, and
the historic call sites passed httpx a bare scalar timeout — which httpx applies
to connect as well as read. A powered-off worker therefore held a thread for the
full op budget (up to 900 s) on the CONNECT.

This file pins the three properties that fix it:

  1. every call class has a SHORT connect budget and a read budget that matches
     what the worker was asked to do;
  2. N consecutive transport failures open a per-worker breaker, so further
     calls fail fast (honestly, as WorkerUnreachable) for a cooldown — and a
     single half-open trial is what re-closes it;
  3. worker_http is the ONLY sanctioned client: no central module makes a
     worker-facing HTTP call directly, and nothing anywhere calls httpx /
     requests / urlopen without an explicit timeout.

Runs under pytest AND as a plain script:
    venv/bin/python -m pytest tests/test_worker_http_discipline.py -q
    venv/bin/python tests/test_worker_http_discipline.py
"""
import ast
import os
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# In-process only — no cross-process comms DB side effects during tests.
os.environ.setdefault("HUGPY_COMMS_DB", "off")

from hugpy_fleet.central import worker_http

PKG = SRC / "hugpy_fleet"
WORKER = {"id": "w-test", "url": "http://worker.invalid:9000"}


@pytest.fixture(autouse=True)
def _clean_breakers(monkeypatch):
    """Every test starts with no observed failures and the shipped defaults."""
    for var in list(os.environ):
        if var.startswith("HUGPY_WORKER_"):
            monkeypatch.delenv(var, raising=False)
    worker_http.reset_breakers()
    yield
    worker_http.reset_breakers()


# ── 1. split timeouts ──────────────────────────────────────────────────────

def test_connect_budget_is_short_for_every_call_class():
    """The defect in one assertion: a 900 s LOAD must still be a 3 s CONNECT.

    Before k59 `httpx.post(url, timeout=900.0)` gave the connect the same 900 s,
    so one click on a dead box cost a thread for fifteen minutes.
    """
    for call in worker_http.READ_TIMEOUTS:
        t = worker_http.timeout_for(call)
        assert t.connect == worker_http.CONNECT_TIMEOUT_S == 3.0, call
        assert t.pool is not None and t.pool <= 10.0, call


def test_read_budgets_match_what_the_worker_was_asked_to_do():
    """Probes/status are seconds; loads and transfers are minutes. A probe that
    inherited a load's budget is how a health check ate a thread."""
    t = worker_http.timeout_for
    assert t("probe").read <= 5.0
    assert t("status").read <= 15.0
    assert t("control").read <= 60.0
    # Relays are the documented exception: a generation legitimately runs long.
    assert t("load").read >= 300.0
    assert t("relay").read >= 300.0
    # ...but they are still BOUNDED. "Excepted from the bounded-read rule" must
    # not silently become "no timeout at all".
    assert t("relay_long").read is not None


def test_no_call_class_is_unbounded():
    for call, read in worker_http.READ_TIMEOUTS.items():
        assert read is not None and read > 0, call


def test_unknown_call_class_falls_back_conservatively():
    """A typo at a call site must not buy an unbounded wait."""
    assert (worker_http.timeout_for("nonsense").read
            == worker_http.timeout_for("control").read)


def test_read_budget_is_env_tunable_but_connect_is_not_lengthened_by_it(monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_READ_TIMEOUT_PROBE_S", "9")
    assert worker_http.timeout_for("probe").read == 9.0
    assert worker_http.timeout_for("probe").connect == 3.0


def test_read_timeout_override_never_touches_connect():
    """_relay_worker_op carries per-verb read budgets; that is the ONE override,
    and it must not be able to re-create the long-connect defect."""
    t = worker_http.timeout_for("control", 900.0)
    assert t.read == 900.0
    assert t.connect == 3.0


# ── 2. the circuit breaker ─────────────────────────────────────────────────

class _Boom:
    """A transport that always fails the way a powered-off box does."""

    def __init__(self, exc=None):
        self.exc = exc or httpx.ConnectTimeout("timed out")
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1
        raise self.exc


def test_transport_failure_raises_typed_unreachable(monkeypatch):
    monkeypatch.setattr(httpx, "request", _Boom())
    with pytest.raises(worker_http.WorkerUnreachable) as ei:
        worker_http.get(WORKER, "/health", call="probe")
    # Honest, never silent: the refusal names the box and the cause.
    assert "ConnectTimeout" in ei.value.reason
    assert ei.value.tripped is False       # a real attempt, not a refusal


def test_breaker_opens_after_n_consecutive_failures_and_then_fails_fast(monkeypatch):
    boom = _Boom()
    monkeypatch.setattr(httpx, "request", boom)
    threshold = worker_http._breaker_failures()
    for _ in range(threshold):
        with pytest.raises(worker_http.WorkerUnreachable):
            worker_http.get(WORKER, "/health", call="probe")
    assert boom.calls == threshold
    # The whole point: the NEXT call costs no socket and no thread-time.
    with pytest.raises(worker_http.WorkerUnreachable) as ei:
        worker_http.get(WORKER, "/health", call="probe")
    assert boom.calls == threshold, "breaker dialed anyway"
    assert ei.value.tripped is True
    assert ei.value.retry_after > 0
    assert worker_http.breaker_snapshot()[WORKER["id"]]["open"] is True


def test_open_breaker_refusal_is_honest_on_the_wire(monkeypatch):
    monkeypatch.setattr(httpx, "request", _Boom())
    for _ in range(worker_http._breaker_failures()):
        with pytest.raises(worker_http.WorkerUnreachable):
            worker_http.post(WORKER, "/models/unload", call="control")
    try:
        worker_http.post(WORKER, "/models/unload", call="control")
    except worker_http.WorkerUnreachable as exc:
        env = exc.as_error()
    assert env["ok"] is False
    assert env["error"]["code"] == "WorkerUnreachable"
    assert env["error"]["breaker_open"] is True
    assert env["error"]["retry_after_s"] > 0
    assert "unreachable" in env["error"]["message"]


def test_http_error_status_is_not_a_breaker_event(monkeypatch):
    """A worker that answers 500 is REACHABLE. Tripping on it would take a
    misbehaving-but-live box out of the fleet for no reason."""
    monkeypatch.setattr(
        httpx, "request",
        lambda *a, **kw: httpx.Response(500, request=httpx.Request("GET", "http://x")))
    for _ in range(worker_http._breaker_failures() + 2):
        assert worker_http.get(WORKER, "/health", call="probe").status_code == 500
    assert worker_http.breaker_snapshot() == {}


def test_success_closes_the_breaker(monkeypatch):
    boom = _Boom()
    monkeypatch.setattr(httpx, "request", boom)
    with pytest.raises(worker_http.WorkerUnreachable):
        worker_http.get(WORKER, "/health", call="probe")
    monkeypatch.setattr(
        httpx, "request",
        lambda *a, **kw: httpx.Response(200, request=httpx.Request("GET", "http://x")))
    worker_http.get(WORKER, "/health", call="probe")
    assert worker_http.breaker_snapshot() == {}


def test_cooldown_lets_exactly_one_trial_through(monkeypatch):
    """Half-open by SINGLE trial. Without it, every held call would pile onto a
    still-dead worker the instant the cooldown expired — the thread storm the
    breaker exists to prevent."""
    monkeypatch.setenv("HUGPY_WORKER_BREAKER_COOLDOWN_S", "0.01")
    boom = _Boom()
    monkeypatch.setattr(httpx, "request", boom)
    for _ in range(worker_http._breaker_failures()):
        with pytest.raises(worker_http.WorkerUnreachable):
            worker_http.get(WORKER, "/health", call="probe")
    dialed = boom.calls

    import time
    time.sleep(0.05)
    # Trial #1 dials...
    with pytest.raises(worker_http.WorkerUnreachable):
        worker_http.get(WORKER, "/health", call="probe")
    assert boom.calls == dialed + 1
    # ...and its failure re-arms the cooldown, so #2 does not.
    with pytest.raises(worker_http.WorkerUnreachable) as ei:
        worker_http.get(WORKER, "/health", call="probe")
    assert boom.calls == dialed + 1
    assert ei.value.tripped is True


def test_force_bypasses_the_open_breaker(monkeypatch):
    """GET /llm/workers/<id>/health IS the 'is it back yet' question — it must
    dial even while the breaker is open, and its success is what re-closes it."""
    boom = _Boom()
    monkeypatch.setattr(httpx, "request", boom)
    for _ in range(worker_http._breaker_failures()):
        with pytest.raises(worker_http.WorkerUnreachable):
            worker_http.get(WORKER, "/health", call="probe")
    dialed = boom.calls
    monkeypatch.setattr(
        httpx, "request",
        lambda *a, **kw: httpx.Response(200, request=httpx.Request("GET", "http://x")))
    assert worker_http.get(WORKER, "/health", call="probe",
                           force=True).status_code == 200
    assert boom.calls == dialed
    assert worker_http.breaker_snapshot() == {}


def test_breaker_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_BREAKER", "off")
    boom = _Boom()
    monkeypatch.setattr(httpx, "request", boom)
    for _ in range(worker_http._breaker_failures() + 3):
        with pytest.raises(worker_http.WorkerUnreachable):
            worker_http.get(WORKER, "/health", call="probe")
    assert boom.calls == worker_http._breaker_failures() + 3


def test_breaker_is_keyed_per_worker(monkeypatch):
    """One dead box must not fail-fast a healthy sibling."""
    other = {"id": "w-other", "url": "http://other.invalid:9000"}
    monkeypatch.setattr(httpx, "request", _Boom())
    for _ in range(worker_http._breaker_failures()):
        with pytest.raises(worker_http.WorkerUnreachable):
            worker_http.get(WORKER, "/health", call="probe")
    snap = worker_http.breaker_snapshot()
    assert snap[WORKER["id"]]["open"] is True
    assert other["id"] not in snap


def test_raw_timeout_kwarg_cannot_reach_httpx(monkeypatch):
    """Defence in depth: even a call site that passes the old scalar gets the
    split timeout, not its own number."""
    seen = {}

    def _capture(method, url, **kw):
        seen.update(kw)
        return httpx.Response(200, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", _capture)
    worker_http.post(WORKER, "/probe/x", call="load", timeout=900.0)
    assert isinstance(seen["timeout"], httpx.Timeout)
    assert seen["timeout"].connect == 3.0


def test_worker_without_url_refuses_instead_of_building_a_bad_call():
    with pytest.raises(worker_http.WorkerUnreachable):
        worker_http.get({"id": "w-nourl", "url": ""}, "/health", call="probe")


def test_breaker_scope_reraises_the_original_exception(monkeypatch):
    """The async relay's callers classify httpx errors themselves (cold-hold vs
    honest refusal), so the scope must not swap the exception type."""
    with pytest.raises(httpx.ConnectTimeout):
        with worker_http.breaker_scope(WORKER):
            raise httpx.ConnectTimeout("nope")
    assert worker_http.breaker_snapshot()[WORKER["id"]]["fails"] == 1


# ── 3. worker_http is the ONLY sanctioned client ───────────────────────────
#
# The task's regression test: "no worker-facing HTTP call is made without an
# explicit timeout — make the wrapper the only sanctioned client and test THAT."

# Modules that talk to WORKERS. These may not construct HTTP calls themselves.
# Fleet-owned worker-facing modules. The server's worker_routes and the
# engine's remote resolver / video reservation engine are asserted by their own
# packages' discipline tests.
WORKER_FACING = (
    "central/workers.py",
    "central/peers.py",
    "central/feeds.py",
)

# Everything reached by an HTTP call name, for the timeout sweep below.
_HTTP_FUNCS = {"get", "post", "put", "patch", "delete", "head", "request",
               "stream", "send"}
_CLIENT_CTORS = {"Client", "AsyncClient"}

# Trees that are NOT central: the worker/agent side, standalone runners and
# CLIs. They run one call per process, not eight threads shared by a console,
# so the k59 pool argument does not apply to them.
_SKIP_DIRS = ("worker_agent/", "gguf_worker/", "phone_brick/", "chaos/",
              "review/", "engine/", "bot/", "video_intel/runners/",
              "managers/llama/", "managers/serve/", "managers/comfy/",
              "managers/vision/", "managers/whisper_model/", "downloader/")


def _central_sources():
    """Every central-side .py: the flask app, comms, and the manager/video_intel
    code the flask app calls into on a request thread."""
    for base in ("central", "fleet_manager", "phone_brick_orchestrator", "doctrine"):
        for path in (PKG / base).rglob("*.py"):
            rel = path.relative_to(PKG).as_posix()
            if any(rel.startswith(d) for d in _SKIP_DIRS):
                continue
            if "__pycache__" in rel or rel.endswith(".bak"):
                continue
            yield rel, path


def _http_calls(tree):
    """(node, dotted-name) for every httpx./requests./urlopen call in a tree."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            mod, attr = f.value.id, f.attr
            if mod in ("httpx", "requests") and (attr in _HTTP_FUNCS
                                                 or attr in _CLIENT_CTORS):
                yield node, f"{mod}.{attr}"
        elif isinstance(f, ast.Name) and f.id == "urlopen":
            yield node, "urlopen"
        elif (isinstance(f, ast.Attribute) and f.attr == "urlopen"):
            yield node, "urlopen"


def test_worker_facing_modules_make_no_direct_http_calls():
    """The discipline itself. If this fails, a call site invented its own
    timeout again — route it through worker_http instead."""
    offenders = []
    for rel in WORKER_FACING:
        path = PKG / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, name in _http_calls(tree):
            offenders.append(f"{rel}:{node.lineno} {name}()")
    assert not offenders, (
        "worker-facing modules must call workers through worker_http, not "
        "directly:\n  " + "\n  ".join(offenders))


def test_no_central_http_call_without_an_explicit_timeout():
    """The broader sweep: nowhere in central may an outbound call be made with
    httpx's/requests' default (httpx: 5 s; requests: NONE — an unbounded wait
    that pins a gunicorn thread until the peer closes the socket)."""
    offenders = []
    for rel, path in _central_sources():
        if rel.endswith("worker_http.py"):
            continue        # the wrapper itself; its timeouts are asserted above
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, name in _http_calls(tree):
            if not any(kw.arg == "timeout" for kw in node.keywords):
                offenders.append(f"{rel}:{node.lineno} {name}()")
    assert not offenders, (
        "outbound HTTP without an explicit timeout:\n  " + "\n  ".join(offenders))


def test_the_wrapper_always_passes_a_split_timeout():
    """worker_http is exempted from the sweep above, so pin it directly: every
    httpx entry point it uses is handed a timeout_for(...) Timeout."""
    src = (PKG / "central/worker_http.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    calls = list(_http_calls(tree))
    assert calls, "worker_http should be the module that actually calls httpx"
    for node, name in calls:
        kw = {k.arg: k.value for k in node.keywords}
        assert "timeout" in kw, f"{name}() at line {node.lineno} has no timeout"
        # ...and it must be the split one, not a scalar.
        val = kw["timeout"]
        assert isinstance(val, ast.Call) and getattr(val.func, "id", "") == "timeout_for", (
            f"{name}() at line {node.lineno} must use timeout_for(), not a bare number")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
