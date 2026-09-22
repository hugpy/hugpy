"""k14 — POST /llm/workers/<id>/slots/<slot_id>/relaunch (central verb) + the
worker agent's /slots/<slot_id>/relaunch ops endpoint.

The central verb relays the offload-depth relaunch to a worker's slot child so the
k7 speed-cliff sweep can drive it. Regressed here WITHOUT a live worker:

  central route:
    * operator-gated (_SENSITIVE), same tier as the other worker ops;
    * unknown worker id -> 404;
    * offline worker -> 409 (no agent to relay to);
    * relays to /slots/<slot_id>/relaunch with a {n_gpu_layers?, ctx?} payload,
      dropping blanks, and returns the worker's typed result verbatim;
    * the worker's own 404 (unknown slot) / 409 (empty slot) propagate.

  worker endpoint (faked slot pool):
    * unknown slot id -> 404;
    * empty slot -> 409;
    * a seated slot -> relays to the slot control /relaunch and echoes the HONEST
      launched n_gpu_layers (what the fresh child launched with, not just asked).
"""
import httpx
import pytest
from flask import Flask, jsonify

from hugpy_server.app.routes import worker_routes as wr
from hugpy_server.app.routes import comms_routes as cr
from hugpy_server.app import operator_auth as oa

ONLINE = {"id": "wid", "name": "box", "status": "online",
          "url": "http://worker:9100"}
OFFLINE = {"id": "off", "name": "box2", "status": "offline",
           "url": "http://worker2:9100"}


# ══════════════════════ central route ══════════════════════════════════════
def _fake_relay_factory(calls):
    def _fake_relay(worker_id, op_path, body, timeout, action, retry_on_connect=False):
        calls.append({"worker_id": worker_id, "op_path": op_path,
                      "body": body, "action": action})
        # emulate the worker's honest result: launched with the requested depth.
        return jsonify({"ok": True, "slot_id": op_path.split("/")[2],
                        "n_gpu_layers": body.get("n_gpu_layers"),
                        "requested_n_gpu_layers": body.get("n_gpu_layers")}), 200
    return _fake_relay


@pytest.fixture
def relay_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("PROJECTS_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(wr, "get_worker",
                        lambda wid: (dict(ONLINE) if wid == "wid"
                                     else dict(OFFLINE) if wid == "off" else None))
    monkeypatch.setattr(wr, "_relay_worker_op", _fake_relay_factory(calls))
    monkeypatch.setattr(cr, "audit", lambda *a, **k: None)
    return calls


@pytest.fixture
def client(relay_calls):
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    return app.test_client()


def test_happy_path_relays_trimmed_payload(client, relay_calls):
    r = client.post("/llm/workers/wid/slots/2/relaunch",
                    json={"n_gpu_layers": 17, "ctx": 8192})
    body = r.get_json()
    assert r.status_code == 200
    assert len(relay_calls) == 1
    assert relay_calls[0]["op_path"] == "/slots/2/relaunch"
    assert relay_calls[0]["action"] == "slot-relaunch"
    assert relay_calls[0]["body"] == {"n_gpu_layers": 17, "ctx": 8192}
    assert body["n_gpu_layers"] == 17          # honest launched ngl echoed


def test_blanks_dropped_absent_means_autofit(client, relay_calls):
    client.post("/llm/workers/wid/slots/1/relaunch", json={"n_gpu_layers": ""})
    assert relay_calls[-1]["body"] == {}
    client.post("/llm/workers/wid/slots/1/relaunch", json={})
    assert relay_calls[-1]["body"] == {}       # slot re-autofits


def test_explicit_zero_passes_through(client, relay_calls):
    """CPU-only (0) is a real override, not a blank."""
    client.post("/llm/workers/wid/slots/1/relaunch", json={"n_gpu_layers": 0})
    assert relay_calls[0]["body"] == {"n_gpu_layers": 0}


def test_unknown_worker_404(client):
    r = client.post("/llm/workers/nope/slots/1/relaunch", json={"n_gpu_layers": 4})
    assert r.status_code == 404


def test_offline_worker_409_never_relayed(client, relay_calls):
    r = client.post("/llm/workers/off/slots/1/relaunch", json={"n_gpu_layers": 4})
    assert r.status_code == 409
    assert relay_calls == []
    assert (r.get_json() or {}).get("error", {}).get("code") == "WorkerOffline"


def test_worker_unknown_slot_404_propagates(client, monkeypatch):
    def _relay_worker_404(worker_id, op_path, body, timeout, action, **k):
        return jsonify({"ok": False, "error": {"code": "UnknownSlot"}}), 404
    monkeypatch.setattr(wr, "_relay_worker_op", _relay_worker_404)
    r = client.post("/llm/workers/wid/slots/9/relaunch", json={})
    assert r.status_code == 404


def _gated(path, method):
    return any(method in methods and rx.match(path) for methods, rx in oa._SENSITIVE)


def test_relaunch_is_operator_gated():
    assert _gated("/llm/workers/wid/slots/2/relaunch", "POST")
    # the rule is POST-only
    assert not _gated("/llm/workers/wid/slots/2/relaunch", "GET")


# ══════════════════════ worker agent endpoint ══════════════════════════════
# Build the worker Flask app with a FAKE slot pool so we exercise slot-id
# resolution + relay without spawning any child.
class _FakePool:
    """Two slots: id 1 seated with 'coder', id 2 empty."""
    def __init__(self, urls=None):
        pass

    def statuses(self):
        return [
            {"slot_id": "1", "model_key": "coder", "_control": "http://s1",
             "n_gpu_layers": -1, "healthy": True, "child_pid": 111},
            {"slot_id": "2", "model_key": None, "_control": "http://s2",
             "n_gpu_layers": None, "healthy": True, "child_pid": None},
        ]


class _State:
    worker_id = "wid"
    name = "box"


class _Resp:
    def __init__(self, payload, status):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


def _fake_httpx_post(url, json=None, timeout=None):
    """The slot control /relaunch: the fresh child launched at the requested ngl."""
    return _Resp({"model_key": "coder", "n_gpu_layers": json.get("n_gpu_layers"),
                  "requested_n_gpu_layers": json.get("n_gpu_layers"),
                  "ctx": json.get("ctx", 4096), "child_pid": 222,
                  "healthy": True, "relaunched": True}, 200)


@pytest.fixture
def wclient(monkeypatch):
    wa = pytest.importorskip("hugpy_fleet.worker.agent")
    slots_mod = pytest.importorskip("hugpy_engine.serve.slots")
    monkeypatch.setattr(slots_mod, "SlotPool", _FakePool)
    monkeypatch.setattr(httpx, "post", _fake_httpx_post)
    return wa.build_app(_State()).test_client()


def test_worker_seated_slot_relays_and_echoes_honest_ngl(wclient):
    r = wclient.post("/slots/1/relaunch", json={"n_gpu_layers": 17})
    body = r.get_json()
    assert r.status_code == 200
    assert body["n_gpu_layers"] == 17
    assert body["child_pid"] == 222          # fresh child pid (PID recycled)
    assert body["ok"] is True


def test_worker_empty_slot_409(wclient):
    r = wclient.post("/slots/2/relaunch", json={"n_gpu_layers": 4})
    assert r.status_code == 409
    assert (r.get_json() or {}).get("error", {}).get("code") == "EmptySlot"


def test_worker_unknown_slot_404(wclient):
    r = wclient.post("/slots/9/relaunch", json={"n_gpu_layers": 4})
    assert r.status_code == 404
    assert (r.get_json() or {}).get("error", {}).get("code") == "UnknownSlot"
