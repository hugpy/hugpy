"""/v1/chat/completions carries OpenAI image parts to the model (2026-09-23).

_render_tool_messages used to ``str()`` a content-part list — the image was
flattened into the prompt as a Python repr and never reached the model. Image
parts now ride as ``images`` (the engine's one image path, same as /ml/vision)
and the text parts are joined; a malformed image part is a 400, not a drop.
"""
import pytest

from hugpy_server.app.routes.v1_helpers import _completion_kwargs, _render_tool_messages

RED = "data:image/png;base64,iVBORw0KGgo="


def _payload(content):
    return {"model": "Qwen2.5-VL-3B-Instruct-GGUF",
            "messages": [{"role": "user", "content": content}]}


def test_image_url_parts_become_images_and_text_is_clean():
    kw = _completion_kwargs(_payload([
        {"type": "text", "text": "what color is the square? one word"},
        {"type": "image_url", "image_url": {"url": RED}}]))
    assert kw["images"] == [RED]
    assert kw["messages"] == [{"role": "user",
                               "content": "what color is the square? one word"}]
    assert "base64" not in kw["messages"][0]["content"]


def test_engine_request_accepts_the_images():
    from hugpy_engine.schemas.chat_schemas import ChatRequest
    kw = _completion_kwargs(_payload([
        {"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": RED}}]))
    req = ChatRequest(**kw)
    assert req.images == [RED]


def test_text_only_payload_unchanged():
    kw = _completion_kwargs(_payload("hello"))
    assert "images" not in kw
    assert kw["messages"] == [{"role": "user", "content": "hello"}]
    assert _render_tool_messages([{"role": "user", "content": None}]) == [
        {"role": "user", "content": ""}]


def test_image_part_without_url_is_rejected_explicitly():
    with pytest.raises(ValueError, match="image content part without a url"):
        _completion_kwargs(_payload([{"type": "image_url", "image_url": {}}]))
