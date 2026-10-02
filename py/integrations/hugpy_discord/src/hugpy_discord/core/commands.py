"""Transport-neutral command cores: one coroutine per slash command's behavior.

Each talks to hugpy central and returns a :class:`CommandResult` (or, for
``download``, yields status lines). The Discord cogs and the chatshare adapter both
call these; the transport-specific parts (deferring, uploading the user's
attachment to a server-side path, rendering the result) stay in the cogs/adapter.

Attachments arrive here already forwarded to a central ``file`` path — the cog does
the Discord upload, the adapter downloads the ``share_url`` and uploads it, exactly
as ``forward_attachment`` does. Everything else is shared.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from hugpy_discord.hugpy_client import HugpyError
from hugpy_discord.no_think import strip_think, with_no_think
from hugpy_discord.core.models import split_items, model_label
from hugpy_discord.core.results import CommandResult, OutFile

# ── static registries (shared with the command specs) ─────────────────────
# Static mirror of hugpy's KNOWN_TASKS_REGISTRY — used for command choices (which
# must be static) and as the /tasks fallback when central predates /prompt/tasks.
TASKS = (
    "text-generation",
    "image-text-to-text",
    "automatic-speech-recognition",
    "text-summarization",
    "text2text-generation",
    "feature-extraction",
    "sentence-similarity",
    "text-to-image",
    "keyword-extraction",
)

# How the generic /task input string maps onto each task's primary field.
_INPUT_FIELD = {
    "text-generation": "prompt",
    "image-text-to-text": "prompt",
    "automatic-speech-recognition": None,   # input is the attachment
    "text-summarization": "text",
    "text2text-generation": "text",
    "feature-extraction": "texts",
    "sentence-similarity": "texts",
    "text-to-image": "prompt",
    "keyword-extraction": "text",
}

KEYWORD_PRESETS = ("default", "seo", "metadata", "social", "long_tail", "article")
SUMMARY_MODES = ("auto", "short", "medium", "long")
WHISPER_SIZES = ("tiny", "small", "medium", "large")

SUMMARIZE_PROMPT = (
    "Summarize the following content concisely. Lead with a 1-2 sentence "
    "overview, then key points as a short bullet list.\n\n{content}"
)
KEYWORDS_PROMPT = (
    "Extract keywords from the following content for the '{preset}' use case. "
    "Return: primary keywords, secondary keywords, hashtags, and a url slug.\n\n{content}"
)
TRANSCRIBE_PROMPT = "Transcribe this audio/video file verbatim."
DESCRIBE_PROMPT = "Describe this file in detail."

# ── download-job polling (ops) ─────────────────────────────────────────────
JOB_POLL_SECONDS = 5
JOB_POLL_MAX_MINUTES = 30
TERMINAL_JOB_STATES = {"done", "complete", "completed", "error", "failed",
                       "cancelled", "canceled"}


def _trim(text: str, limit: int = 1024) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _no_prompt_route(exc: HugpyError) -> bool:
    return "does not expose /prompt" in str(exc)


# ── ML commands (embed / similarity / imagine / task / tasks) ──────────────
async def embed_core(bot, *, texts: list[str], normalize=None, batch_size=None,
                     model=None) -> CommandResult:
    try:
        result = await bot.hugpy.execute_prompt(
            task="feature-extraction", texts=texts, normalize=normalize,
            batch_size=batch_size, model_key=model)
    except HugpyError as exc:
        return CommandResult.fail(f"⚠️ {exc}")
    vectors = result.get("embeddings") or []
    dim = len(vectors[0]) if vectors else 0
    payload = json.dumps(
        {"model": result.get("model_key"), "texts": texts, "embeddings": vectors},
        indent=2)
    return CommandResult(
        text=(f"🧮 embedded **{len(vectors)}** text(s) → **{dim}**-dim vectors "
              f"(`{result.get('model_key')}`) — vectors attached"),
        files=[OutFile("embeddings.json", payload.encode("utf-8"), "application/json")])


async def similarity_core(bot, *, text: str, candidates: list[str], normalize=None,
                          model=None) -> CommandResult:
    try:
        result = await bot.hugpy.execute_prompt(
            task="sentence-similarity", texts=[text], other_texts=candidates,
            normalize=normalize, model_key=model)
    except HugpyError as exc:
        return CommandResult.fail(f"⚠️ {exc}")
    scores = (result.get("similarities") or [[]])[0]
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    lines = [f"**similarity to:** {text}"]
    lines += [f"`{score:+.4f}` {candidate}" for candidate, score in ranked]
    return CommandResult(text="\n".join(lines), long=True, filename="similarity.txt")


async def imagine_core(bot, *, prompt: str, negative=None, width=None, height=None,
                       steps=None, guidance=None, seed=None, count=1,
                       model=None) -> CommandResult:
    try:
        result = await bot.hugpy.execute_prompt(
            task="text-to-image", prompt=prompt, negative_prompt=negative,
            width=width, height=height, steps=steps, guidance_scale=guidance,
            seed=seed, num_images=count if count != 1 else None, model_key=model)
    except HugpyError as exc:
        return CommandResult.fail(f"⚠️ {exc}")
    import base64
    files = []
    for index, image in enumerate(result.get("images") or []):
        if image.get("b64"):
            files.append(OutFile(f"imagine_{index}.png",
                                 base64.b64decode(image["b64"]), "image/png"))
    if not files:
        return CommandResult.fail(
            f"⚠️ no image bytes returned: "
            f"{result.get('text') or result.get('error') or '?'}")
    caption = f"🎨 **{prompt}**" + (f"\n-# seed: {seed}" if seed is not None else "")
    return CommandResult(text=caption, files=files)


async def task_core(bot, *, task: str, input=None, file=None, model=None,
                    params=None, temperature=None, top_p=None, max_tokens=None,
                    do_sample=None) -> CommandResult:
    extra: dict = {}
    if params:
        try:
            extra = json.loads(params)
            if not isinstance(extra, dict):
                raise ValueError("params must be a JSON object")
        except ValueError as exc:
            return CommandResult.fail(f"⚠️ bad params JSON: {exc}", ephemeral=True)

    kwargs: dict = {
        "task": task, "model_key": model, "temperature": temperature,
        "top_p": top_p, "max_new_tokens": max_tokens, "do_sample": do_sample,
    }
    if file:
        kwargs["file"] = file
    if input:
        field = _INPUT_FIELD.get(task)
        if field == "texts":
            kwargs["texts"] = split_items(input)
        elif field:
            kwargs[field] = input
    kwargs.update(extra)  # explicit params JSON wins over the mapping

    try:
        result = await bot.hugpy.execute_prompt(**kwargs)
    except HugpyError as exc:
        return CommandResult.fail(f"⚠️ {exc}")
    return _render_task_result(task, result)


def _render_task_result(task: str, result: dict) -> CommandResult:
    """Shape-aware rendering: images attach, vectors summarize, text sends."""
    import base64
    files = []
    for index, image in enumerate(result.get("images") or []):
        if isinstance(image, dict) and image.get("b64"):
            files.append(OutFile(f"{task}_{index}.png",
                                 base64.b64decode(image["b64"]), "image/png"))
    if files:
        return CommandResult(text=(result.get("text") or "🎨 done"), files=files)

    if result.get("embeddings") is not None and not result.get("similarities"):
        vectors = result["embeddings"]
        dim = len(vectors[0]) if vectors else 0
        payload = json.dumps(result, indent=2)
        return CommandResult(
            text=f"🧮 {len(vectors)} vector(s) × {dim} dims — attached",
            files=[OutFile("result.json", payload.encode("utf-8"), "application/json")])

    if result.get("similarities") is not None:
        rows = result["similarities"]
        lines = [f"row {row_index}: " + ", ".join(f"{score:+.4f}" for score in row)
                 for row_index, row in enumerate(rows)]
        return CommandResult(text="\n".join(lines), long=True,
                             filename="similarities.txt")

    return CommandResult(text=(result.get("text")
                               or json.dumps(result, indent=2)[:4000]),
                         long=True, filename=f"{task}.txt")


async def tasks_core(bot) -> CommandResult:
    try:
        info = await bot.hugpy.supported_tasks()
        tasks, defaults = info.get("tasks") or [], info.get("defaults") or {}
        note = ""
    except HugpyError:
        tasks, defaults = list(TASKS), {}
        note = "\n-# central predates /prompt/tasks — showing the bot's static list"
    lines = ["**hugpy task categories**"]
    for key in tasks:
        default = defaults.get(key)
        lines.append(f"• `{key}`" + (f" → `{default}`" if default else ""))
    return CommandResult(text="\n".join(lines) + note, ephemeral=True)


# ── tool commands (summarize / keywords / transcribe / describe) ───────────
async def summarize_core(bot, *, text=None, file=None, mode=None, preset=None,
                         model=None) -> CommandResult:
    try:
        result = await bot.hugpy.execute_prompt(
            task="text-summarization", text=text, file=file, summary_mode=mode,
            preset=preset, model_key=model)
    except HugpyError as exc:
        if _no_prompt_route(exc):
            return CommandResult(
                fallback_prompt=SUMMARIZE_PROMPT.format(
                    content=text or "the attached file"),
                fallback_file=file)
        return CommandResult.fail(f"⚠️ {exc}")
    summary = result.get("text") or ""
    stats = []
    if result.get("input_word_count"):
        stats.append(f"{result['input_word_count']}→"
                     f"{result.get('output_word_count', '?')} words")
    if result.get("preset_used"):
        stats.append(f"preset: {result['preset_used']}")
    if result.get("input_warning"):
        stats.append(f"⚠️ {result['input_warning']}")
    if stats:
        summary += f"\n-# {' · '.join(stats)}"
    return CommandResult(text=summary, long=True, filename="summary.txt")


async def keywords_core(bot, *, text=None, file=None, preset=None, top_n=None,
                        diversity=None, model=None) -> CommandResult:
    try:
        result = await bot.hugpy.execute_prompt(
            task="keyword-extraction", text=text, file=file, preset=preset,
            top_n=top_n, diversity=diversity, model_key=model)
    except HugpyError as exc:
        if _no_prompt_route(exc):
            return CommandResult(
                fallback_prompt=KEYWORDS_PROMPT.format(
                    preset=preset or "seo", content=text or "the attached file"),
                fallback_file=file)
        return CommandResult.fail(f"⚠️ {exc}")
    lines = []
    if result.get("primary"):
        lines.append("**primary:** " + ", ".join(f"`{kw}`" for kw in result["primary"]))
    if result.get("secondary"):
        lines.append("**secondary:** " + ", ".join(f"`{kw}`" for kw in result["secondary"]))
    if result.get("hashtags"):
        lines.append("**hashtags:** " + " ".join(result["hashtags"]))
    if result.get("slug_candidates"):
        lines.append("**slugs:** " + ", ".join(f"`{slug}`" for slug in result["slug_candidates"]))
    if not lines:
        lines.append(result.get("text") or "*no keywords extracted*")
    footer = []
    if result.get("preset_used"):
        footer.append(f"preset: {result['preset_used']}")
    if result.get("backends_used"):
        footer.append(f"backends: {'+'.join(result['backends_used'])}")
    if footer:
        lines.append(f"-# {' · '.join(footer)}")
    return CommandResult(text="\n".join(lines), long=True, filename="keywords.txt")


async def transcribe_core(bot, *, file: str, language=None, size=None,
                          translate=False, timestamps=False,
                          model=None) -> CommandResult:
    try:
        result = await bot.hugpy.execute_prompt(
            task="automatic-speech-recognition", file=file, language=language,
            model_size=size, translate=translate or None, model_key=model)
    except HugpyError as exc:
        if _no_prompt_route(exc):
            return CommandResult(fallback_prompt=TRANSCRIBE_PROMPT, fallback_file=file)
        return CommandResult.fail(f"⚠️ {exc}")
    if timestamps and result.get("segments"):
        lines = [
            f"[{seg.get('start', 0):>7.2f} → {seg.get('end', 0):>7.2f}] "
            f"{seg.get('text', '').strip()}"
            for seg in result["segments"]]
        text = "\n".join(lines)
    else:
        text = result.get("text") or ""
    footer = []
    if result.get("language"):
        footer.append(f"language: {result['language']}")
    if result.get("duration"):
        footer.append(f"duration: {result['duration']:.0f}s")
    if footer:
        text += f"\n-# {' · '.join(footer)}"
    return CommandResult(text=text, long=True, filename="transcript.txt")


async def describe_core(bot, *, file: str, prompt=None, max_tokens=None,
                        model=None) -> CommandResult:
    try:
        # NO-THINK: the reply is posted as the description (display-as-prose). The
        # prompt is only rewritten when the caller gave one; leaving it None keeps
        # central's own default, and the strip below covers that case.
        result = await bot.hugpy.execute_prompt(
            task="image-text-to-text", file=file,
            prompt=with_no_think(prompt) if prompt else prompt,
            max_new_tokens=max_tokens, model_key=model)
    except HugpyError as exc:
        if _no_prompt_route(exc):
            return CommandResult(fallback_prompt=prompt or DESCRIBE_PROMPT,
                                 fallback_file=file)
        return CommandResult.fail(f"⚠️ {exc}")
    description, _reasoning = strip_think(result.get("text") or "")
    return CommandResult(text=description, long=True, filename="description.txt")


# ── ops commands (status / models / jobs / canceljob / hf / download) ──────
async def status_core(bot) -> CommandResult:
    res = CommandResult(title="hugpy central", footer=bot.hugpy.base_url)
    try:
        health = await bot.hugpy.health()
        res.sections.append(
            ("health", f"✅ ok — storage `{health.get('storage_root', '?')}`"))
    except HugpyError as exc:
        res.color = "red"
        res.sections.append(("health", f"❌ {_trim(str(exc), 200)}"))
        return res
    try:
        serving = await bot.hugpy.serving()
        keys = list(serving) if isinstance(serving, (list, dict)) else []
        res.sections.append(
            (f"serving ({len(keys)})",
             _trim(", ".join(f"`{k}`" for k in keys) or "nothing")))
    except HugpyError:
        pass
    try:
        workers = await bot.hugpy.list_workers()
        lines = [
            f"`{w.get('name') or w.get('id')}` — {w.get('status', '?')}"
            + (f" ({', '.join(w.get('models') or [])})" if w.get("models") else "")
            for w in workers]
        res.sections.append(
            (f"workers ({len(workers)})", _trim("\n".join(lines) or "none registered")))
    except HugpyError:
        pass
    return res


async def models_core(bot, *, installed_only=False) -> CommandResult:
    try:
        models = await bot.hugpy.list_models()
    except HugpyError as exc:
        return CommandResult.fail(f"⚠️ {exc}")
    if installed_only:
        models = [m for m in models
                  if m.get("installed")
                  or str(m.get("status", "")).lower() == "installed"]
    if not models:
        return CommandResult(text="no models found")
    lines = [f"• {model_label(m)}" for m in models]
    return CommandResult(title=f"models ({len(models)})",
                         text=_trim("\n".join(lines), 4000))


async def jobs_core(bot) -> CommandResult:
    try:
        jobs = await bot.hugpy.list_jobs()
    except HugpyError as exc:
        return CommandResult.fail(f"⚠️ {exc}")
    if not jobs:
        return CommandResult(
            text="no download jobs — chat generations are listed by /running")
    lines = [
        f"`{j.get('id') or j.get('job_id')}` — {j.get('model_key', '?')} "
        f"— {j.get('status', '?')}"
        for j in jobs[-20:]]
    return CommandResult(title="download jobs", text="\n".join(lines))


async def canceljob_core(bot, *, job_id: str) -> CommandResult:
    try:
        await bot.hugpy.cancel_job(job_id)
        return CommandResult(text=f"🛑 cancelled `{job_id}`", ephemeral=True)
    except HugpyError as exc:
        return CommandResult.fail(f"⚠️ {exc}", ephemeral=True)


async def hf_core(bot, *, query: str, task=None) -> CommandResult:
    try:
        results = await bot.hugpy.hf_search(query, limit=10, task=task)
    except HugpyError as exc:
        return CommandResult.fail(f"⚠️ {exc}")
    items = (results if isinstance(results, list)
             else results.get("results") or results.get("models") or [])
    if not items:
        return CommandResult(text="no results")
    lines = []
    for item in items[:10]:
        hub_id = item.get("hub_id") or item.get("id") or item.get("modelId") or "?"
        extras = []
        if item.get("downloads") is not None:
            extras.append(f"{item['downloads']:,} dl")
        if item.get("likes") is not None:
            extras.append(f"{item['likes']} ♥")
        if item.get("size_human") or item.get("size"):
            extras.append(str(item.get("size_human") or item.get("size")))
        lines.append(f"• `{hub_id}`" + (f" — {', '.join(extras)}" if extras else ""))
    return CommandResult(title=f"hf search: {query}",
                         text=_trim("\n".join(lines), 4000),
                         footer="install with /download <hub_id>")


async def download_core(bot, *, model: str) -> AsyncIterator[str]:
    """Kick off a download and yield status lines (first line = initial status,
    then one per poll until the job settles). The transport sends the first line
    and edits it with each subsequent one."""
    try:
        if "/" in model:
            job = await bot.hugpy.download_repo(model)
        else:
            job = await bot.hugpy.download_model(model)
    except HugpyError as exc:
        yield f"⚠️ {exc}"
        return

    job_id = job.get("id") or job.get("job_id")
    yield f"⬇️ downloading `{model}` — job `{job_id}`"
    if not job_id:
        return
    for _ in range(int(JOB_POLL_MAX_MINUTES * 60 / JOB_POLL_SECONDS)):
        await asyncio.sleep(JOB_POLL_SECONDS)
        try:
            job = await bot.hugpy.get_job(job_id)
        except HugpyError:
            continue
        status = str(job.get("status", "?")).lower()
        progress = job.get("progress") or job.get("percent")
        line = f"⬇️ `{model}` — {status}" + (
            f" ({progress}%)" if progress is not None else "")
        if status in TERMINAL_JOB_STATES:
            icon = "✅" if status in ("done", "complete", "completed") else "❌"
            yield f"{icon} `{model}` — {status}"
            return
        yield line
    yield f"⏱️ `{model}` — still running; check /jobs (job `{job_id}`)"
