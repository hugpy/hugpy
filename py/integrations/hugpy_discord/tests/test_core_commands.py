"""The transport-neutral command cores: result shapes + the /prompt fallback."""
from __future__ import annotations

import asyncio
import json

from hugpy_discord.core import commands as core
from hugpy_discord.hugpy_client import HugpyError
from _fakes import FakeBot


def run(coro):
    return asyncio.run(coro)


def test_embed_core_attaches_vectors():
    async def body():
        bot = FakeBot()
        bot.hugpy.prompt_result = {"embeddings": [[1.0, 2.0, 3.0]], "model_key": "alpha"}
        res = await core.embed_core(bot, texts=["a"])
        assert res.files and res.files[0].filename == "embeddings.json"
        assert "**1**" in res.text and "3**-dim" in res.text
        payload = json.loads(res.files[0].data)
        assert payload["embeddings"] == [[1.0, 2.0, 3.0]]
        assert bot.hugpy.last_prompt["task"] == "feature-extraction"
    run(body())


def test_similarity_core_ranks():
    async def body():
        bot = FakeBot()
        bot.hugpy.prompt_result = {"similarities": [[0.1, 0.9]]}
        res = await core.similarity_core(bot, text="q", candidates=["lo", "hi"])
        assert res.long and res.filename == "similarity.txt"
        # highest score first
        lines = res.text.splitlines()
        assert lines[1].endswith("hi") and lines[2].endswith("lo")
    run(body())


def test_imagine_core_decodes_images():
    async def body():
        import base64
        bot = FakeBot()
        bot.hugpy.prompt_result = {"images": [{"b64": base64.b64encode(b"PNG").decode()}]}
        res = await core.imagine_core(bot, prompt="a cat", seed=7)
        assert len(res.files) == 1 and res.files[0].data == b"PNG"
        assert "a cat" in res.text and "seed: 7" in res.text
    run(body())


def test_imagine_core_no_bytes_is_error():
    async def body():
        bot = FakeBot()
        bot.hugpy.prompt_result = {"text": "nope"}
        res = await core.imagine_core(bot, prompt="x")
        assert res.error and "no image bytes" in res.text
    run(body())


def test_status_core_sections():
    async def body():
        res = await core.status_core(FakeBot())
        names = [n for n, _ in res.sections]
        assert res.title == "hugpy central"
        assert names[0] == "health" and res.sections[0][1].startswith("✅")
        assert any(n.startswith("serving") for n in names)
        assert any(n.startswith("workers") for n in names)
        assert res.footer == "http://central.test"
    run(body())


def test_status_core_health_failure_is_red():
    async def body():
        bot = FakeBot()
        async def boom():
            raise HugpyError("down")
        bot.hugpy.health = boom
        res = await core.status_core(bot)
        assert res.color == "red" and len(res.sections) == 1
    run(body())


def test_models_core_titles_and_empty():
    async def body():
        res = await core.models_core(FakeBot())
        assert res.title == "models (2)"
        bot = FakeBot()
        async def none():
            return []
        bot.hugpy.list_models = none
        empty = await core.models_core(bot)
        assert empty.text == "no models found" and empty.title is None
    run(body())


def test_tasks_core_is_ephemeral():
    async def body():
        res = await core.tasks_core(FakeBot())
        assert res.ephemeral and "text-generation" in res.text
    run(body())


def test_hf_core_and_jobs_core():
    async def body():
        hf = await core.hf_core(FakeBot(), query="cat")
        assert hf.title == "hf search: cat" and "org/model" in hf.text
        jobs = await core.jobs_core(FakeBot())
        assert jobs.title == "download jobs" and "alpha" in jobs.text
    run(body())


def test_canceljob_core_ephemeral():
    async def body():
        res = await core.canceljob_core(FakeBot(), job_id="j1")
        assert res.ephemeral and "cancelled" in res.text
    run(body())


def test_summarize_fallback_when_no_prompt_route():
    async def body():
        bot = FakeBot()
        bot.hugpy.prompt_error = HugpyError(
            "central does not expose /prompt yet — deploy …")
        res = await core.summarize_core(bot, text="long text")
        assert res.fallback_prompt is not None and "long text" in res.fallback_prompt
    run(body())


def test_describe_strips_think():
    async def body():
        bot = FakeBot()
        bot.hugpy.prompt_result = {"text": "<think>hmm</think>A red car."}
        res = await core.describe_core(bot, file="/srv/x.png")
        assert res.text == "A red car."
    run(body())


def test_download_core_yields_initial_and_terminal():
    async def body():
        orig = core.JOB_POLL_SECONDS
        core.JOB_POLL_SECONDS = 0.001
        try:
            lines = [ln async for ln in core.download_core(FakeBot(), model="alpha")]
        finally:
            core.JOB_POLL_SECONDS = orig
        assert lines[0].startswith("⬇️ downloading `alpha`")
        assert lines[-1].startswith("✅")
    run(body())


def test_task_core_bad_params_json():
    async def body():
        res = await core.task_core(FakeBot(), task="text-generation",
                                   params="{not json}")
        assert res.error and res.ephemeral and "bad params JSON" in res.text
    run(body())
