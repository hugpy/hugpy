"""ChatRequest.caller — the central-only harness-attribution field (2026-09-24).

The /v1 route's derived client identity rides on ChatRequest.caller so the
metrics record site (resolvers.remote) can attribute the call. It is CENTRAL
ONLY: it must never cross the worker relay wire (workers run extra="forbid"), so
model_dump() drops it UNCONDITIONALLY while the attribute stays readable on the
object. The wire for every other request stays byte-identical.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_engine.schemas.chat_schemas import ChatRequest

MSGS = [{"role": "user", "content": "hi"}]


def test_caller_is_readable_on_the_object():
    req = ChatRequest(messages=MSGS, caller="hermes")
    assert req.caller == "hermes"


def test_caller_never_reaches_the_worker_wire():
    # model_dump() is exactly what _relay_payload ships to the worker.
    req = ChatRequest(messages=MSGS, caller="hermes")
    assert "caller" not in req.model_dump()


def test_absent_caller_defaults_to_none_and_is_absent_on_wire():
    req = ChatRequest(messages=MSGS)
    assert req.caller is None
    assert "caller" not in req.model_dump()


def test_builder_forwards_caller_into_the_request():
    from hugpy_engine.resolvers.categories.builders import _build_chat_request
    req = _build_chat_request({"messages": MSGS, "caller": "aider"}, "some-model")
    assert req.caller == "aider"
    assert "caller" not in req.model_dump()
