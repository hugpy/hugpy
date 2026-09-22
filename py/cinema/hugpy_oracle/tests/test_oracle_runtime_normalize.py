"""``runtime.normalize_kwargs`` — the /ml amenity normalization as a pure
function, so the oracle dispatches without the HTTP adapter."""

from __future__ import annotations

from hugpy_oracle import runtime


def test_model_and_vision_folds(monkeypatch):
    monkeypatch.delenv("HUGPY_ML_POOL", raising=False)
    kw = runtime.normalize_kwargs("image-text-to-text",
                                  {"model": "org/vl", "image_b64": "AAAA", "_internal": 1,
                                   "prompt": "what is this", "none": None})
    assert kw == {"task": "image-text-to-text", "model_key": "org/vl",
                  "images": ["AAAA"], "prompt": "what is this"}


def test_explicit_model_key_and_images_win():
    kw = runtime.normalize_kwargs("image-text-to-text",
                                  {"model": "x", "model_key": "y", "images": ["b"], "image_b64": "c"})
    assert kw["model_key"] == "y" and kw["images"] == ["b"] and "image_b64" in kw


def test_pool_defaults_follow_the_env(monkeypatch):
    monkeypatch.setenv("HUGPY_ML_POOL", "ml")
    monkeypatch.setenv("HUGPY_ML_GENERAL_ROUTE_TASKS", "image-text-to-text")
    assert runtime.normalize_kwargs("text-summarization", {"text": "t"})["pool"] == "ml"
    assert "pool" not in runtime.normalize_kwargs("image-text-to-text", {"prompt": "p"})
    assert runtime.normalize_kwargs("text-summarization", {"text": "t", "pool": "mine"})["pool"] == "mine"
    monkeypatch.setenv("HUGPY_ML_POOL", "")
    assert "pool" not in runtime.normalize_kwargs("text-summarization", {"text": "t"})


def test_await_sync_drives_awaitables_and_passes_values():
    async def coro():
        return 7

    assert runtime._await_sync(3) == 3
    assert runtime._await_sync(coro()) == 7
