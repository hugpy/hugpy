"""Cold-load HOLD — a feasible-but-cold call is a presumed success (t36).

The operator's rule: "the on-demand will initially fail if a model is called
that is not loaded and needs to swap, it will then generate as it gets loaded.
this should be a predispositioned success until the model actually fails to
load." So when a worker IS selected but the model's on-demand load/swap trips a
TRANSIENT failure, central must HOLD the call (surfacing load progress) and
dispatch the moment it is healthy — failing ONLY on an honest load failure, and
NEVER expiring a placed-and-loading job. Genuine infeasibility (a permanent load
error / no worker) still fails FAST (honest refusal unchanged).

This covers, purely central-side (no worker needed):
  * error classification (transient hold vs permanent honest-refusal);
  * the load-state seam wrapper (degrade / arg-count / raising provider);
  * DelegatingRunner.stream(): transient-then-success held as ONE call, honest
    refusal surfaced fast, cancel-while-held, and the N-calls-one-load coalescer;
  * DelegatingRunner.run(): the one-shot hold + honest passthrough;
  * workers.load_state_for_model() reading the heartbeat (healthy / in_progress /
    fresh-vs-stale honest error / alias match);
  * the job store: a placed/loading job is NOT orphan-expired, and progress feeds
    the honest stall clock (the pending->expired-while-loading defect).

Each check group below is its own test; ``remote`` (hugpy_engine.resolvers.remote)
and ``W`` (hugpy_fleet.central.workers) are fixtures.
"""
import asyncio
import importlib
import os
import time
import types

import pytest

# In-process only — no cross-process comms DB side effects during tests.
os.environ.setdefault("HUGPY_COMMS_DB", "off")


def check(name, cond):
    assert cond, name


@pytest.fixture
def remote():
    return importlib.import_module("hugpy_engine.resolvers.remote")


@pytest.fixture
def W():
    return importlib.import_module("hugpy_fleet.central.workers")


# A fixed worker the mocked _select always returns.
WORKER = {"id": "w1", "name": "computron", "url": "http://w1:9100"}


def _req(rid="rid-1"):
    return types.SimpleNamespace(
        request_id=rid, pool=None,
        reference_images=None, reference_images_b64=None,
        model_dump=lambda: {"messages": [{"role": "user", "content": "hi"}]},
    )


def _text_framework(remote):
    """A (framework, task) whose task is NOT vision, so the vision gate is off."""
    for (fw, tk) in remote.FRAMEWORK_RUNNERS:
        if tk != "image-text-to-text":
            return fw, tk
    return next(iter(remote.FRAMEWORK_RUNNERS))


async def _collect(agen):
    out = []
    async for ev in agen:
        out.append(ev)
    return out


def _etypes(evs):
    return [getattr(e, "type", None) for e in evs]


# ---------------------------------------------------------------------------
def test_error_classification(remote):
    perm = remote._is_permanent_load_error
    check("won't fit is permanent (honest refusal)", perm("LoadRefusal: won't fit on GPU"))
    check("out of memory is permanent", perm("CUDA out of memory"))
    check("unknown model is permanent", perm("Unknown model_key=foo"))
    check("no worker is permanent", perm("no capable worker is available"))
    check("vision-in-process capability refusal is permanent",
          perm("vision model loaded in-process (text-only — ...)"))
    check("RemoteProtocolError/server-disconnect is TRANSIENT (held)",
          not perm("RemoteProtocolError: Server disconnected without sending a response."))
    check("503 model_busy is TRANSIENT", not perm("model_busy: still warming"))
    check("bare 'produced no output' is TRANSIENT", not perm("worker computron produced no output"))
    check("permanent-classifier reads .message off an exception too",
          perm(remote._LoadFailed("won't fit on GPU")))


def test_load_state_wrapper(remote):
    orig = remote._load_state_provider
    try:
        remote.set_load_state_provider(None)
        check("unset load-state seam -> None", remote._load_state("m", "w1") is None)
        check("no worker_id -> None", remote._load_state("m", "") is None)

        remote.set_load_state_provider(
            lambda mk, wid, since=0.0: {"healthy": True, "in_progress": False})
        st = remote._load_state("m", "w1", 123.0)
        check("registered provider result passes through", st and st.get("healthy") is True)

        # arg-count degrade: a 2-arg provider (older) still works.
        remote.set_load_state_provider(lambda mk, wid: {"healthy": False, "in_progress": True})
        st = remote._load_state("m", "w1", 5.0)
        check("2-arg provider is called via degrade", st and st.get("in_progress") is True)

        def _boom(*a, **k):
            raise RuntimeError("nope")
        remote.set_load_state_provider(_boom)
        check("raising provider -> None (never breaks a request)",
              remote._load_state("m", "w1") is None)
    finally:
        remote.set_load_state_provider(orig)


def test_cold_progress(remote):
    orig = remote._load_state_provider
    try:
        remote.set_load_state_provider(lambda mk, wid, since=0.0: {"healthy": True})
        moved, prog, msg, honest = remote._cold_progress("m", WORKER, 0.0)
        check("healthy load-state counts as forward movement", moved and honest is None)

        remote.set_load_state_provider(
            lambda mk, wid, since=0.0: {"in_progress": True, "progress": 0.5, "message": "loading"})
        moved, prog, msg, honest = remote._cold_progress("m", WORKER, 0.0)
        check("in-progress counts as movement + carries progress",
              moved and prog == 0.5 and msg == "loading" and honest is None)

        remote.set_load_state_provider(
            lambda mk, wid, since=0.0: {"healthy": False, "in_progress": False,
                                        "error": "won't fit on GPU"})
        moved, prog, msg, honest = remote._cold_progress("m", WORKER, 0.0)
        check("a fresh PERMANENT load-state error is honest-fail", honest and "won't fit" in honest)

        remote.set_load_state_provider(
            lambda mk, wid, since=0.0: {"healthy": False, "in_progress": False,
                                        "error": "RemoteProtocolError: Server disconnected"})
        moved, prog, msg, honest = remote._cold_progress("m", WORKER, 0.0)
        check("a TRANSIENT load-state error is NOT honest-fail (keep holding)",
              honest is None and not moved)

        remote.set_load_state_provider(None)
        check("no provider -> no movement, no error",
              remote._cold_progress("m", WORKER, 0.0) == (False, None, None, None))
    finally:
        remote.set_load_state_provider(orig)


def test_delegating_runner_stream_hold(remote):
    fw, tk = _text_framework(remote)
    Runner = remote.make_delegating_runner(fw, tk)
    runner = Runner(types.SimpleNamespace(model_key="cold-model"))

    orig_select = remote._select
    orig_ws = remote._worker_stream
    orig_ls = remote._load_state_provider
    # No-op relay gate (return the primary slot immediately, no gate math).
    os.environ["HUGPY_CENTRAL_GATE"] = "off"
    os.environ["HUGPY_COLD_HOLD_POLL_S"] = "0.01"
    os.environ["HUGPY_COLD_HOLD_MAX_S"] = "5"
    os.environ["HUGPY_COLD_HOLD_STALL_S"] = "5"
    os.environ.pop("HUGPY_LOCAL_FALLBACK", None)
    os.environ.pop("HUGPY_NO_LOCAL_SERVING", None)
    remote._select = lambda mk, pool=None, task=None, **kw: (dict(WORKER), None)
    # Load-state says "loading" so the hold's stall clock stays fed.
    remote.set_load_state_provider(
        lambda mk, wid, since=0.0: {"healthy": False, "in_progress": True,
                                    "progress": 0.4, "message": "loading weights"})

    from hugpy_engine.schemas.event_schemas import TokenEvent, DoneEvent

    try:
        # -- transient-then-success: ONE held call, no retry surfaced ----------
        calls = {"n": 0}

        async def ws_transient(worker, payload, rid):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("RemoteProtocolError: Server disconnected without sending a response.")
            yield TokenEvent(request_id=rid, text="Hello")
            yield DoneEvent(request_id=rid, input_tokens=1, output_chunks=1, finish_reason="stop")

        remote._worker_stream = ws_transient
        evs = asyncio.run(_collect(runner.stream(_req())))
        types_ = _etypes(evs)
        check("cold call held then dispatched (token + done, no error)",
              "token" in types_ and "done" in types_ and "error" not in types_)
        check("the transient failures were retried, not surfaced (>=3 relay attempts)",
              calls["n"] == 3)
        loading = [e for e in evs if getattr(e, "stage", None) == "awaiting-load"]
        check("a loading status was surfaced while holding", len(loading) >= 1)
        check("loading status reuses the browser-rendered shape (message + progress)",
              getattr(loading[0], "message", None) and getattr(loading[0], "progress", None) == 0.4)

        # -- honest refusal: permanent error fails FAST, no hold ---------------
        calls2 = {"n": 0}

        async def ws_refuse(worker, payload, rid):
            calls2["n"] += 1
            raise RuntimeError("LoadRefusal: won't fit on GPU: needs 6.7 GB, 6.3 GB free")
            yield  # pragma: no cover — marks this a generator

        remote._worker_stream = ws_refuse
        evs = asyncio.run(_collect(runner.stream(_req())))
        types_ = _etypes(evs)
        check("honest 'won't fit' refusal surfaces an error FAST (unchanged)",
              types_.count("error") == 1 and "token" not in types_)
        check("honest refusal is NOT retried/held (one relay attempt)", calls2["n"] == 1)
        err = [e for e in evs if getattr(e, "type", None) == "error"][0]
        check("the refusal message is preserved (won't fit)", "won't fit" in err.message)
        check("no loading status was emitted for a fast refusal",
              not any(getattr(e, "stage", None) == "awaiting-load" for e in evs))
        # ...and it leaves NO cached verdict: "won't fit" is state-dependent —
        # the next eviction makes it false — so it must never answer a LATER
        # request from cache (that is what poisoned this model for a whole TTL,
        # turning the clean cancel below into an error event).
        check("a capacity refusal leaves NO cached load verdict (state-dependent)",
              remote._active_load_verdict("w1", "cold-model") is None)

        # -- cancel-while-held: clean stop, no error ---------------------------
        async def ws_always_transient(worker, payload, rid):
            raise RuntimeError("still loading, server disconnected")
            yield  # pragma: no cover

        remote._worker_stream = ws_always_transient

        async def _drive_cancel():
            cancel = asyncio.Event()
            gen = runner.stream(_req("rid-cancel"), cancel_event=cancel)
            seen = []
            async for ev in gen:
                seen.append(ev)
                if getattr(ev, "stage", None) == "awaiting-load":
                    cancel.set()          # cancel WHILE held
            return seen

        seen = asyncio.run(_drive_cancel())
        check("cancel-while-held stops cleanly with NO error event",
              "error" not in _etypes(seen) and "token" not in _etypes(seen))
        check("cancel-while-held did surface at least one loading status first",
              any(getattr(e, "stage", None) == "awaiting-load" for e in seen))
        # A user pulling out is NOT a load failure: a cancelled attempt must
        # leave the load-verdict cache untouched, or one cancel would make the
        # model answer "permanently failed" for every caller for a whole TTL.
        check("a CANCELLED attempt leaves NO cached load verdict",
              remote._active_load_verdict("w1", "cold-model") is None
              and ("w1", "cold-model") not in remote._LOAD_VERDICTS)

        # -- the verdict cache still does its job for the DETERMINISTIC class --
        # (a not-a-model directory: no retry conjures a base model into a LoRA
        # dir, so the second call must not reach the worker at all.)
        broken = Runner(types.SimpleNamespace(model_key="broken-model"))
        hits = {"n": 0}

        async def ws_broken(worker, payload, rid):
            hits["n"] += 1
            raise RuntimeError("ValueError: Unrecognized model in /store/lora-dir. "
                               "Should have a `model_type` key in its config.json")
            yield  # pragma: no cover

        remote._worker_stream = ws_broken
        try:
            e1 = asyncio.run(_collect(broken.stream(_req("brk-1"))))
            check("a deterministic load failure fails fast on the first call",
                  _etypes(e1).count("error") == 1 and hits["n"] == 1)
            e2 = asyncio.run(_collect(broken.stream(_req("brk-2"))))
            check("the recorded verdict answers the re-submit without a new attempt",
                  _etypes(e2).count("error") == 1 and hits["n"] == 1)
            check("the cached answer names the cache and repeats the real cause",
                  "load-verdict cache" in e2[-1].message
                  and "Unrecognized model" in e2[-1].message)
        finally:
            remote._clear_load_verdict("w1", "broken-model")

        # -- coalescing: N concurrent cold calls -> ONE *load* at a time -------
        # `loading` counts kicks INSIDE their load window (before a token, i.e.
        # before the model is warm). The guarantee is that this never exceeds 1 —
        # two callers never trigger two concurrent on-demand LOADS of one cold
        # model. Once a kick loads the model (produces a token) it leaves the
        # window, so the other caller then kicks the now-WARM model (fine).
        live = {"loading": 0, "loadmax": 0, "n": 0}

        async def ws_coalesce(worker, payload, rid):
            live["n"] += 1
            live["loading"] += 1
            live["loadmax"] = max(live["loadmax"], live["loading"])
            await asyncio.sleep(0.03)              # the on-demand LOAD window
            if live["n"] <= 2:
                live["loading"] -= 1               # load failed transiently
                raise RuntimeError("server disconnected (swapping)")
            live["loading"] -= 1                   # loaded — now warm
            yield TokenEvent(request_id=rid, text="ok")
            yield DoneEvent(request_id=rid, input_tokens=1, output_chunks=1, finish_reason="stop")

        remote._worker_stream = ws_coalesce

        async def _two():
            return await asyncio.gather(
                _collect(runner.stream(_req("c-a"))),
                _collect(runner.stream(_req("c-b"))),
            )

        a, b = asyncio.run(_two())
        check("coalescer: never two concurrent on-demand loads for one cold model",
              live["loadmax"] == 1)
        check("both coalesced callers still each dispatched a real reply",
              "token" in _etypes(a) and "token" in _etypes(b))

        # -- warm-release: the cold-kick key frees on the FIRST token, so waiters
        #    dispatch concurrently against the warm model (no over-serialization).
        async def ws_slow(worker, payload, rid):
            yield TokenEvent(request_id=rid, text="a")
            await asyncio.sleep(0.02)
            yield TokenEvent(request_id=rid, text="b")
            yield DoneEvent(request_id=rid, input_tokens=1, output_chunks=2, finish_reason="stop")

        remote._worker_stream = ws_slow
        key = ("w1", "cold-model")
        freed = {"on_first_token": None}

        async def _drive_warm():
            gen = runner.stream(_req("warm-1"))
            seen_token = False
            async for ev in gen:
                if getattr(ev, "type", None) == "token" and not seen_token:
                    seen_token = True
                    freed["on_first_token"] = key not in remote._COLD_KICKING

        asyncio.run(_drive_warm())
        check("cold-kick key is freed on the first token (warm) so waiters run concurrently",
              freed["on_first_token"] is True)
        check("cold-kick key is fully cleared after the stream ends",
              key not in remote._COLD_KICKING)
    finally:
        remote._select = orig_select
        remote._worker_stream = orig_ws
        remote.set_load_state_provider(orig_ls)
        os.environ.pop("HUGPY_CENTRAL_GATE", None)
        os.environ.pop("HUGPY_COLD_HOLD_POLL_S", None)
        os.environ.pop("HUGPY_COLD_HOLD_MAX_S", None)
        os.environ.pop("HUGPY_COLD_HOLD_STALL_S", None)


def test_delegating_runner_run_hold(remote):
    fw, tk = _text_framework(remote)
    Runner = remote.make_delegating_runner(fw, tk)
    runner = Runner(types.SimpleNamespace(model_key="cold-model"))

    orig_select = remote._select
    orig_run = remote._worker_run_once
    orig_ls = remote._load_state_provider
    os.environ["HUGPY_CENTRAL_GATE"] = "off"
    os.environ["HUGPY_COLD_HOLD_POLL_S"] = "0.01"
    os.environ["HUGPY_COLD_HOLD_MAX_S"] = "5"
    os.environ["HUGPY_COLD_HOLD_STALL_S"] = "5"
    os.environ.pop("HUGPY_LOCAL_FALLBACK", None)
    os.environ.pop("HUGPY_NO_LOCAL_SERVING", None)
    remote._select = lambda mk, pool=None, task=None, **kw: (dict(WORKER), None)
    remote.set_load_state_provider(
        lambda mk, wid, since=0.0: {"healthy": False, "in_progress": True})

    try:
        calls = {"n": 0}

        async def run_transient(worker, payload, result_type, request_id, model_key):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("RemoteProtocolError: Server disconnected")
            return {"ok": True, "text": "done", "request_id": request_id, "model_key": model_key}

        remote._worker_run_once = run_transient
        res = asyncio.run(runner.run(_req("run-1")))
        check("run(): transient cold failure held + retried to success", res.get("ok") is True)
        check("run(): retried, did not fail fast (>=3 attempts)", calls["n"] == 3)

        async def run_refuse(worker, payload, result_type, request_id, model_key):
            raise RuntimeError("LoadRefusal: won't fit on GPU")

        remote._worker_run_once = run_refuse
        raised = None
        try:
            asyncio.run(runner.run(_req("run-2")))
        except RuntimeError as exc:
            raised = str(exc)
        check("run(): honest 'won't fit' refusal fails FAST (raises, no hold)",
              raised is not None and "won't fit" in raised)
    finally:
        remote._select = orig_select
        remote._worker_run_once = orig_run
        remote.set_load_state_provider(orig_ls)
        os.environ.pop("HUGPY_CENTRAL_GATE", None)
        os.environ.pop("HUGPY_COLD_HOLD_POLL_S", None)
        os.environ.pop("HUGPY_COLD_HOLD_MAX_S", None)
        os.environ.pop("HUGPY_COLD_HOLD_STALL_S", None)


def test_load_state_for_model_reads_heartbeat(W):
    now = time.time()
    SYNTH = {
        "loaded_models": ["Qwen2.5-3B-Instruct-GGUF"],
        "loading": ["Big-Model-GGUF"],
        "provisioning": ["Downloading-GGUF"],
        # The shape a WORKER actually writes (agent.py's _prog callback):
        # done_bytes / total_bytes / frac. This fixture used to carry
        # {"progress": .., "message": ..} — keys no worker has ever sent — and
        # the reader asked for those same invented keys, so the two agreed with
        # each other and disagreed with reality: every held call reported
        # progress=None and showed a bare spinner. Corrected 2026-07-28 with the
        # progress-aware hold; the check below now asserts real bytes.
        "provision_progress": {"Downloading-GGUF": {"done_bytes": 1 * 2**30,
                                                    "total_bytes": 4 * 2**30,
                                                    "frac": 0.25}},
        "load_reports": {
            "Fresh-Fail-GGUF": {"ok": False, "error": "CUDA out of memory", "ts": now},
            "Stale-Fail-GGUF": {"ok": False, "error": "old blip", "ts": now - 10_000},
            "Good-GGUF": {"ok": True, "fit": False, "ts": now},
        },
        # A STANDING storage refusal (agent._refused_snapshot ships this). It is
        # NOT a load_report — the pull was refused before any load — so the hold
        # must surface it as an honest error rather than streaming "loading".
        "refused": {
            "Refused-GGUF": {"state": "refused",
                             "reason": "won't fit on volume: needs 3.2 GB, "
                                       "148.7 GB free, 150.0 GB reserve kept"},
        },
    }
    orig = W.worker_store
    try:
        W.worker_store = types.SimpleNamespace(get=lambda wid: SYNTH)
        f = W.load_state_for_model
        check("healthy: loaded model reads healthy",
              f("Qwen2.5-3B-Instruct-GGUF", "w1", 0)["healthy"] is True)
        check("healthy match is alias-tolerant (hub_id form)",
              f("Qwen/Qwen2.5-3B-Instruct-GGUF", "w1", 0)["healthy"] is True)
        _bm = f("Big-Model-GGUF", "w1", 0)
        check("loading model reads in_progress (not healthy)",
              _bm["healthy"] is False and _bm["in_progress"] is True
              and _bm["progress"] is None and _bm["message"] is None
              and _bm["error"] is None)
        d = f("Downloading-GGUF", "w1", 0)
        check("provisioning model carries download progress + BYTES",
              d["in_progress"] and d["progress"] == 0.25
              and "1.0 GB" in (d["message"] or "")
              and "4.0 GB" in (d["message"] or ""))
        check("FRESH honest load error is surfaced",
              f("Fresh-Fail-GGUF", "w1", now - 1)["error"] == "CUDA out of memory")
        check("STALE load error (ts < since_ts) is NOT surfaced (no false fresh-fail)",
              f("Stale-Fail-GGUF", "w1", now - 1)["error"] is None)
        check("ok:True load report is not an error",
              f("Good-GGUF", "w1", 0)["error"] is None)
        check("STANDING storage refusal is surfaced as an honest error (won't fit)",
              "won't fit" in (f("Refused-GGUF", "w1", 0)["error"] or ""))
        check("a refusal is a standing verdict — surfaced even with a fresh since_ts",
              "won't fit" in (f("Refused-GGUF", "w1", now + 1)["error"] or ""))
        _np = f("Nope-GGUF", "w1", 0)
        check("unknown model on the worker reads cold/idle",
              _np["healthy"] is False and _np["in_progress"] is False
              and _np["progress"] is None and _np["message"] is None
              and _np["error"] is None)
        W.worker_store = types.SimpleNamespace(get=lambda wid: None)
        check("unknown worker -> None", f("x", "ghost", 0) is None)
    finally:
        W.worker_store = orig


def test_placed_job_never_orphan_expired():
    jobs = importlib.import_module("hugpy_control.jobs")
    store = jobs.JobStore(mirror=None)   # pure in-process
    old = time.time() - 10_000           # well past any expiry window

    # A worker-LESS pending row that never progressed IS an orphan -> expired.
    j1 = store.create("m", id="orphan", kind="chat")
    store.update("orphan", progressed_at=old)
    expired = store.expire_pending_orphans()
    check("a placeless pending job past the window expires (unchanged)",
          "orphan" in expired and store.get("orphan").status == "expired")

    # A pending row WITH a worker is NOT an orphan (it is placed + loading).
    store.create("m", id="placed", kind="chat", worker="computron")
    store.update("placed", progressed_at=old)
    check("a PLACED pending job is never orphan-expired (t36)",
          "placed" not in store.expire_pending_orphans()
          and store.get("placed").status == "pending")

    # Feeding progress moves it off pending AND refreshes the honest clock.
    store.create("m", id="loading", kind="chat")
    store.update("loading", progressed_at=old)
    j = store.update("loading", status="processing", stage="awaiting-load", progress=0.3)
    check("a loading job is 'processing' (out of the pending-orphan class)",
          j.status == "processing")
    check("a stage/progress advance refreshes progressed_at (honest stall clock fed)",
          j.progressed_at > old + 1)
    check("a loading job is not expired even with an old original progressed_at",
          "loading" not in store.expire_pending_orphans())


def test_feed_job_from_status():
    """streaming._feed_job_from_status reflects dispatch + loading into the job."""
    streaming = pytest.importorskip("hugpy_server.app.functions.chat.streaming")
    from hugpy_control.jobs import job_store
    job_store.create("m", id="feed-1", kind="chat")

    disp = types.SimpleNamespace(type="status", served_by="worker",
                                 worker_name="computron", worker_id="w1")
    streaming._feed_job_from_status("feed-1", disp)
    check("dispatch status stamps the selected worker onto the job",
          job_store.get("feed-1").worker == "computron")

    loading = types.SimpleNamespace(type="status", served_by=None,
                                    stage="awaiting-load", message="loading weights",
                                    progress=0.6)
    streaming._feed_job_from_status("feed-1", loading)
    j = job_store.get("feed-1")
    check("loading status moves the job to processing/awaiting-load with progress",
          j.status == "processing" and j.stage == "awaiting-load" and j.progress == 0.6)
