"""k53 honesty pair (2026-07-31): never serve a model nobody asked for.

Two silent SUBSTITUTIONS, both of which look like a working answer at every
layer above the box that produced it:

  1. /infer (+ /infer/stream) with an ABSENT model_key fell through to
     resolve_model_key's last resort — the chat default — and the worker
     answered with a different model entirely; an UNKNOWN key raised deep in
     resolution and came back as an opaque 500. Both are now a 400 that names
     the key and says what to do.
  2. the slot proxy forwarded to whatever answered on the child port and
     labelled it with the model the slot CLAIMED. Observed live: a Qwen2.5-3B
     process had taken the port and answered coder-next requests. The child is
     now asked what it is serving before anything is proxied to it; a mismatch
     drops the stale mapping so the next request cold-loads properly.

No real inference, no subprocess, no GPU: the dispatch entry point is stubbed
(module-shadowing landmine — agent._run_once is patched on the AGENT module,
which is what the route actually calls) and the slot's child is a fake pid.

Run: venv/bin/python -m pytest tests/test_infer_key_and_slot_identity.py -q
"""
import importlib
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

agent = importlib.import_module("hugpy_fleet.worker.agent")
provision = importlib.import_module("hugpy_storage.provision")
sa = importlib.import_module("hugpy_engine.serve.slot_agent")


# ═══════════════════ 1. /infer never substitutes a default ══════════════════
@pytest.fixture
def wclient(monkeypatch):
    """A worker app whose serving path is stubbed at the LAST honest seam: if a
    request reaches _run_once/_stream_sync, the route decided to serve it."""
    state = agent.WorkerState(name="t", url="http://central", worker_id="w-k53")
    served = []

    def _fake_run_once(payload):
        served.append(dict(payload))
        return {"ok": True, "text": "hi", "model_key": payload.get("model_key")}

    def _fake_stream(payload, request_id=None):
        served.append(dict(payload))
        yield b"data: {}\n\n"

    monkeypatch.setattr(agent, "_run_once", _fake_run_once)
    monkeypatch.setattr(agent, "_stream_sync", _fake_stream)
    monkeypatch.setattr(agent, "_ensure_present", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_ensure_present_streaming",
                        lambda *a, **k: iter(()))
    # The worker's registry, as the refusal gate sees it: 'known' resolves,
    # 'alias' canonicalizes, everything else is unknown.
    monkeypatch.setattr(
        provision, "ensure_model_registered",
        lambda key, url: {"known": "known", "alias": "known"}.get(key))
    return type("Rig", (), {"client": agent.build_app(state).test_client(),
                            "served": served})()


def test_absent_model_key_is_a_400_not_the_default_model(wclient):
    r = wclient.client.post("/infer", json={"messages": [{"role": "user",
                                                          "content": "hi"}]})
    assert r.status_code == 400
    assert "model_key is required" in r.get_json()["error"]
    assert wclient.served == []          # nothing was served in its place


def test_unknown_model_key_is_a_400_that_names_the_key(wclient):
    r = wclient.client.post("/infer", json={"model_key": "not-a-model",
                                            "prompt": "hi"})
    assert r.status_code == 400
    body = r.get_json()
    assert "not-a-model" in body["error"] and "No default" in body["error"]
    assert body["worker"]["id"] == "w-k53"
    assert wclient.served == []


def test_known_model_key_serves_exactly_that_model(wclient):
    r = wclient.client.post("/infer", json={"model_key": "known",
                                            "prompt": "hi"})
    assert r.status_code == 200
    assert wclient.served[-1]["model_key"] == "known"


def test_an_alias_is_canonicalized_not_refused(wclient):
    """Learning a model from central under its canonical name is the EXISTING
    behaviour of _ensure_present; the gate must reuse it, not fight it."""
    r = wclient.client.post("/infer", json={"model_key": "alias",
                                            "prompt": "hi"})
    assert r.status_code == 200
    assert wclient.served[-1]["model_key"] == "known"


def test_a_stated_task_may_still_use_its_designated_default(wclient):
    """TASK_DEFAULTS is a per-task designation the caller opted into by naming
    the task — not the fallthrough this gate exists to kill."""
    r = wclient.client.post("/infer", json={"task": "embed", "prompt": "hi"})
    assert r.status_code == 200
    assert wclient.served[-1]["task"] == "embed"


def test_stream_refuses_BEFORE_the_response_begins(wclient):
    """A stream that has already started can only report this as a mid-body SSE
    surprise with a 200 status — so the refusal happens first."""
    r = wclient.client.post("/infer/stream", json={"prompt": "hi"})
    assert r.status_code == 400
    assert "model_key is required" in r.get_json()["error"]
    assert wclient.served == []
    r = wclient.client.post("/infer/stream", json={"model_key": "nope",
                                                   "prompt": "hi"})
    assert r.status_code == 400
    assert wclient.served == []


def test_stream_serves_a_known_key(wclient):
    r = wclient.client.post("/infer/stream", json={"model_key": "known",
                                                   "prompt": "hi"})
    assert r.status_code == 200
    r.get_data()                                  # drain the generator
    assert wclient.served[-1]["model_key"] == "known"


def test_an_unreachable_central_never_invents_a_refusal(wclient, monkeypatch):
    """Degrade-not-guess: when the registration lookup itself FAILS we cannot
    say the key is unknown, so the request proceeds and the load path speaks."""
    def _boom(key, url):
        raise RuntimeError("central unreachable")
    monkeypatch.setattr(provision, "ensure_model_registered", _boom)
    r = wclient.client.post("/infer", json={"model_key": "maybe-real",
                                            "prompt": "hi"})
    assert r.status_code == 200
    assert wclient.served[-1]["model_key"] == "maybe-real"


# ═══════════════════ 2. slot identity before proxying ═══════════════════════
class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid

    def poll(self):
        return None                               # alive


@pytest.fixture
def seated(monkeypatch):
    """A slot claiming coder-next with a live (fake) child on its port."""
    slot = sa.Slot()
    slot.model_key = "coder-next"
    slot.model_path = "/models/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf"
    slot.proc = _FakeProc()
    slot._identity = {"pid": 4242, "ok": True, "note": None, "at": 0.0}
    return slot


def _reports(monkeypatch, slot, value, seen=None):
    def _id():
        if seen is not None:
            seen.append(value)
        return value
    monkeypatch.setattr(slot, "_child_model_id", _id)


def test_matching_child_verifies_and_keeps_the_seat(monkeypatch, seated):
    _reports(monkeypatch, seated,
             "/models/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf")
    assert seated.verify_identity() == (True, None)
    assert seated.model_key == "coder-next"


def test_a_stranger_on_the_port_is_a_mismatch_and_drops_the_claim(monkeypatch,
                                                                  seated):
    """The live incident: a Qwen2.5-3B process took the port and answered
    coder-next requests. The claim must not survive that."""
    killed = []
    monkeypatch.setattr(seated, "_kill", lambda: killed.append(True))
    _reports(monkeypatch, seated, "/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    ok, note = seated.verify_identity()
    assert ok is False
    assert "Qwen2.5-3B" in note and "coder-next" in note
    assert seated.model_key is None and seated.model_path is None
    assert killed == [True]


def test_a_child_that_says_nothing_is_never_torn_down(monkeypatch, seated):
    """Can't tell != wrong. An unanswered/uninformative probe must never cost a
    live seat (the same degrade-not-guess rule the placement probes follow)."""
    _reports(monkeypatch, seated, None)
    assert seated.verify_identity() == (True, None)
    assert seated.model_key == "coder-next"


def test_an_alias_style_id_matches_by_model_key(monkeypatch, seated):
    """llama_cpp.server reports an alias, not a path — matching is lenient by
    design, so only a clearly-different name is a mismatch."""
    _reports(monkeypatch, seated, "coder-next")
    assert seated.verify_identity()[0] is True


def test_the_verdict_is_cached_per_pid_and_window(monkeypatch, seated):
    seen = []
    _reports(monkeypatch, seated, "coder-next", seen)
    seated.verify_identity()
    seated.verify_identity()
    seated.verify_identity()
    assert len(seen) == 1                         # one probe, not three
    seated.verify_identity(force=True)
    assert len(seen) == 2                         # force re-probes
    seated._identity["at"] = time.time() - (sa._IDENTITY_RECHECK_S + 1)
    seated.verify_identity()
    assert len(seen) == 3                         # window expired


def test_a_new_child_pid_invalidates_the_cached_verdict(monkeypatch, seated):
    seen = []
    _reports(monkeypatch, seated, "coder-next", seen)
    seated.verify_identity()
    seated.proc = _FakeProc(pid=99)               # respawned under a new pid
    seated.verify_identity()
    assert len(seen) == 2


def test_a_busy_child_is_not_probed_mid_generation(monkeypatch, seated):
    seen = []
    _reports(monkeypatch, seated, "coder-next", seen)
    seated.inflight = 1
    assert seated.verify_identity() == (True, None)
    assert seen == []


def test_a_dead_child_clears_the_claim_instead_of_verifying(seated):
    seated.proc = None
    ok, note = seated.verify_identity()
    assert ok is True and seated.model_key is None
    assert "gone" in note


def test_status_stops_advertising_a_substituted_model(monkeypatch, seated):
    monkeypatch.setattr(seated, "_kill", lambda: None)
    _reports(monkeypatch, seated, "/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    st = seated.status()
    assert st["model_key"] is None                # the scheduler routes on this
    assert st["identity_ok"] is False and st["identity_note"]


def test_proxy_refuses_a_substituted_child_with_503(monkeypatch):
    app, slot = sa.build_app()
    slot.model_key = "coder-next"
    slot.model_path = "/models/coder-next.gguf"
    slot.proc = _FakeProc()
    slot._identity = {"pid": 4242, "ok": True, "note": None, "at": 0.0}
    monkeypatch.setattr(slot, "healthy", lambda: True)
    monkeypatch.setattr(slot, "_kill", lambda: None)
    monkeypatch.setattr(slot, "_child_model_id",
                        lambda: "/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    r = app.test_client().post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 503
    assert "not serving the claimed model" in r.get_json()["error"]
    assert slot.model_key is None                 # dropped -> next call reloads


# ── the two pure helpers the verdict is built from ──────────────────────────
@pytest.mark.parametrize("doc,expected", [
    ({"model_path": "/m/a.gguf"}, "/m/a.gguf"),
    ({"model": "/m/a.gguf"}, "/m/a.gguf"),
    ({"default_generation_settings": {"model": "/m/a.gguf"}}, "/m/a.gguf"),
    ({"data": [{"id": "alias"}]}, "alias"),
    ({"data": []}, None),
    ({}, None),
    ("not a dict", None),
])
def test_reported_model_id_reads_every_child_shape(doc, expected):
    assert sa._reported_model_id(doc) == expected


@pytest.mark.parametrize("reported,ok", [
    ("/models/a.gguf", True),                     # same path
    ("a.gguf", True),                             # bare filename
    ("a", True),                                  # stem / alias
    ("my-model", True),                           # the model_key itself
    ("", True),                                   # said nothing -> can't tell
    (None, True),
    ("/models/other.gguf", False),                # a genuinely different model
])
def test_identity_matches_is_lenient_but_not_blind(reported, ok):
    assert sa._identity_matches("/models/a.gguf", "my-model", reported) is ok


def test_model_path_is_read_off_the_launched_argv():
    assert sa._model_path_of(["llama-server", "-m", "/m/a.gguf",
                              "--port", "9"]) == "/m/a.gguf"
    assert sa._model_path_of(["python", "-m", "llama_cpp.server",
                              "--model", "/m/b.gguf"]) == "/m/b.gguf"
    assert sa._model_path_of(["llama-server", "--port", "9"]) is None
    assert sa._model_path_of([]) is None
