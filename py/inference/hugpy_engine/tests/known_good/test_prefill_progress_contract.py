"""KNOWN-GOOD CONTRACT — the llama-server runner forwards prefill progress.

Source under test: hugpy_engine/llama/runners/src/base_runner.py
(LlamaCppBaseRunner._take_stream_progress / _prefill_status / stream_chat) and
ccp_runner._iter_stream's ``return_progress`` request key. Deterministic: a
stub runner whose _iter_stream replays the chunk pattern the live llama-server
emitted on 2026-09-29 (verified against ae:9201 with return_progress=true:
prompt_progress {total, cache, processed, time_ms} on content-less chunks).
"""
from __future__ import annotations

import asyncio
import inspect
import types

from hugpy_engine.llama.runners.src import base_runner as br
from hugpy_engine.llama.runners.src import ccp_runner


class _Stub(br.LlamaCppBaseRunner):
    def __init__(self, chunks):
        self.model_key = "stub"
        self._chunks = chunks

    async def _iter_stream(self, messages, max_tokens, temp, top_p, extras=None):
        for pp, text, fr in self._chunks:
            if pp is not None:
                self._stream_progress = pp          # what ccp_runner stashes
            yield text, fr

    def _chat_complete(self, *a, **k):  # pragma: no cover — abstract fill
        raise NotImplementedError

    def _raw_complete(self, *a, **k):   # pragma: no cover — abstract fill
        raise NotImplementedError

    def _log_done(self, *a, **k):
        pass

    def _attach_image(self, messages, req):
        return messages

    @staticmethod
    def _engine_extras(req):
        return {}


def _req():
    return types.SimpleNamespace(request_id="r1", max_new_tokens=16, temperature=0.0,
                                 do_sample=False, top_p=1.0, messages=[])


def _collect(agen):
    async def _run():
        return [e async for e in agen]
    return asyncio.run(_run())


def test_progress_chunks_become_prefill_status_events_deduped_then_tokens(monkeypatch):
    """INVARIANT: each prompt_progress advance yields ONE StatusEvent(stage=
    "prefill", n_prompt, n_past, progress, n_cache, prefill_ms); a repeated
    figure yields nothing; tokens and the terminal done are unchanged.
    Established: 2026-09-29."""
    monkeypatch.setattr(br, "messages_to_dicts", lambda m: list(m))
    chunks = [
        ({"total": 55, "cache": 42, "processed": 42, "time_ms": 12}, "", None),
        ({"total": 55, "cache": 42, "processed": 51, "time_ms": 42}, "", None),
        ({"total": 55, "cache": 42, "processed": 55, "time_ms": 78}, "", None),
        ({"total": 55, "cache": 42, "processed": 55, "time_ms": 78}, "", None),   # repeat
        (None, "ok", None),
        (None, "", "stop"),
    ]
    evs = _collect(_Stub(chunks).stream_chat(_req()))
    kinds = [(e.type, getattr(e, "stage", None)) for e in evs]
    assert kinds == [("status", "prefill")] * 3 + [("token", None), ("done", None)], kinds
    assert [e.n_past for e in evs[:3]] == [42, 51, 55]
    assert evs[0].n_prompt == 55 and evs[0].n_cache == 42 and evs[0].prefill_ms == 12
    assert evs[2].progress == 1.0 and "55/55" in evs[2].message
    assert evs[-1].finish_reason == "stop"


def test_unbounded_stream_forwards_progress_too(monkeypatch):
    """INVARIANT: the continuation (unbounded) driver forwards prefill
    progress exactly like stream_chat. Established: 2026-09-29."""
    monkeypatch.setattr(br, "messages_to_dicts", lambda m: list(m))
    chunks = [({"total": 10, "cache": 0, "processed": 10, "time_ms": 5}, "", None),
              (None, "x", "stop")]
    evs = _collect(_Stub(chunks).stream_chat_unbounded(_req(), chunk_tokens=4, max_chunks=1))
    assert [e.type for e in evs][:2] == ["status", "token"] and evs[0].stage == "prefill"


def test_llama_server_request_asks_for_progress():
    """GUARD (source pin): ccp_runner's streaming request body carries
    ``return_progress: True`` and stashes ``prompt_progress`` — without both,
    every layer above reads a long prefill as silence. Established: 2026-09-29."""
    src = inspect.getsource(ccp_runner.LlamaCppRunner._iter_stream) \
        if hasattr(ccp_runner, "LlamaCppRunner") else inspect.getsource(ccp_runner)
    assert '"return_progress": True' in src
    assert 'data.get("prompt_progress")' in src and "_stream_progress" in src
