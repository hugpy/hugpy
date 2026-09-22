"""The /v1 seam half of hugpy_engine/tests/test_chat_role_alternation.py.

Moved here because it imports the server's v1 route helpers (tool-message
rendering, request-shape classification); the engine-side checks stay in the
engine suite. Shares that file's fixtures by importing them.
"""
import copy
import importlib

ok = 0

def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

WORKER = {"id": "w1", "name": "ae", "url": "http://w1:9100"}

TEMPLATE_ERR = ("TemplateError: After the optional system message, conversation "
                "roles must alternate user/assistant/user/assistant/...")

_ECHO = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "get_weather",
                                 "arguments": '{"city": "Berlin"}'}}],
}

_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "72F, sunny"}
def _v1_seam_checks(schemas):
    import json as _json
    mod = importlib.import_module("hugpy_server.app.routes.v1_helpers")

    # The realistic tool loop after rendering: user / assistant / user — already
    # alternating, so the merge must be a no-op on it.
    rendered = mod._render_tool_messages([
        {"role": "user", "content": "weather in Berlin?"}, _ECHO, _RESULT])
    check("_render_tool_messages produces an ALTERNATING user/assistant/user loop",
          [m["role"] for m in rendered] == ["user", "assistant", "user"])
    req = schemas.ChatRequest(messages=copy.deepcopy(rendered))
    check("the rendered tool loop passes through normalisation byte-identically",
          req.model_dump()["messages"] == rendered)
    check("the <tool_response> wrapper is intact after normalisation",
          "<tool_response>" in req.messages[2].content)
    check("the <tool_call> block is intact after normalisation",
          "<tool_call>" in req.messages[1].content
          and _json.loads(req.messages[1].content.split("<tool_call>")[1]
                          .split("</tool_call>")[0].strip())["name"] == "get_weather")

    # A double-send through /v1 (two user turns) is normalised for the engine.
    rendered = mod._render_tool_messages([{"role": "user", "content": "a"},
                                          {"role": "user", "content": "b"}])
    req = schemas.ChatRequest(messages=rendered)
    check("a /v1 double-send is normalised into one alternating user turn",
          req.model_dump()["messages"] == [{"role": "user", "content": "a\n\nb"}])


def _v1_status_checks():
    try:
        v1 = importlib.import_module(
            "hugpy_server.app.routes.v1_routes")
    except Exception as exc:  # pragma: no cover — app import too heavy here
        print(f"  ~ skip _v1_status_checks import ({type(exc).__name__}: {exc})")
        return
    shape = v1._is_request_shape_message
    remote = importlib.import_module("hugpy_engine.resolvers.remote")
    msg = remote._request_shape_message("MN-GRAND-23.5B", WORKER, TEMPLATE_ERR)
    check("/v1 classifies the request-shape error message (-> 400)", shape(msg))
    check("/v1 classifies the raw TemplateError too", shape(TEMPLATE_ERR))
    check("/v1 does NOT reclassify a capacity refusal",
          not shape("LoadRefusal: won't fit on GPU"))
    check("/v1 does NOT reclassify a busy worker",
          not shape("worker_busy: at concurrency cap"))
    check("/v1 does NOT reclassify an unknown model",
          not shape("Unknown model_key=nope; known: [...]"))



def test_chat_role_alternation_v1_seam():
    schemas = importlib.import_module("hugpy_engine.schemas.chat_schemas")
    _v1_seam_checks(schemas)
    _v1_status_checks()
