"""Text-only task selection is identical for every chat entry point."""
from types import SimpleNamespace

from hugpy_server.app.functions.chat.task_selection import text_chat_task


def _cfg(primary, tasks):
    return SimpleNamespace(primary_task=primary, tasks=tasks)


def test_plain_text_uses_secondary_text_runner():
    cfg = _cfg("image-text-to-text", ["image-text-to-text", "text-generation"])
    assert text_chat_task(cfg) == "text-generation"


def test_primary_text_needs_no_override():
    cfg = _cfg("text-generation", ["text-generation"])
    assert text_chat_task(cfg) is None


def test_media_and_explicit_task_are_preserved():
    cfg = _cfg("image-text-to-text", ["image-text-to-text", "text-generation"])
    assert text_chat_task(cfg, has_media=True) is None
    assert text_chat_task(cfg, requested_task="image-text-to-text") == "image-text-to-text"
