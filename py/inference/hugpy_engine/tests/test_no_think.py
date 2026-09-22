"""NO-THINK — the package-wide seam (abstract_hugpy_dev/utils/no_think.py).

Operator ruling 2026-07-29: "the expectation of a model to adhere to no_think
should be circumvented for any execution that requires this stipulation, package
wide." Two halves, both required, neither trusting the model:

  1. SUPPRESS at generation — the /no_think directive rides the query.
  2. STRIP defensively — any <think> that comes back anyway is removed and
     surfaced under its own key.

The load-bearing case is the UNCLOSED <think>: when the token budget runs out
mid-thought there is no closing tag at all, and a strip that only matched closed
blocks would serve a truncated ramble as the answer. That is not hypothetical —
it is what the live fleet produced on 2026-07-27 (see the module docstring).

Runs under pytest AND as a plain script:
    venv/bin/python -m pytest tests/test_no_think.py -q
    venv/bin/python tests/test_no_think.py
"""
from __future__ import annotations

import sys

from hugpy_engine.utils.no_think import (
    NO_THINK_DIRECTIVE,
    StreamingThinkSplitter,
    apply_no_think,
    finalize_no_think,
    strip_think,
    with_no_think,
)


def _drive_splitter(tokens):
    """Feed tokens one at a time; return the accumulated (answer, reasoning)."""
    sp = StreamingThinkSplitter()
    a = r = ""
    for t in tokens:
        da, dr = sp.feed(t)
        a += da
        r += dr
    fa, fr = sp.flush()
    return a + fa, r + fr


def test_splitter_separates_think_from_answer():
    assert _drive_splitter(["<think>reasoning</think>The answer."]) == (
        "The answer.", "reasoning")


def test_splitter_handles_a_tag_split_across_tokens():
    # the whole point of the streaming splitter: '<thi'+'nk>' must never leak
    assert _drive_splitter(["<thi", "nk>rea", "soning</thi", "nk>ans", "wer"]) == (
        "answer", "reasoning")


def test_splitter_char_by_char_stream():
    assert _drive_splitter(list("<think>ab</think>cd")) == ("cd", "ab")


def test_splitter_unclosed_think_drains_as_reasoning_never_answer():
    assert _drive_splitter(["<think>ran out of budget mid-thou"]) == (
        "", "ran out of budget mid-thou")


def test_splitter_plain_answer_has_no_reasoning():
    assert _drive_splitter(["just a plain answer"]) == ("just a plain answer", "")


def test_splitter_a_literal_lt_is_not_a_tag():
    assert _drive_splitter(["2 < 3 is true"]) == ("2 < 3 is true", "")


def test_splitter_thinking_variant_and_surrounding_answer():
    assert _drive_splitter(["pre <thinking>mid</thinking> post"]) == (
        "pre  post", "mid")


# --------------------------------------------------------------------------- #
# strip_think
# --------------------------------------------------------------------------- #
def test_strip_closed_block():
    text, reasoning = strip_think("<think>weighing options</think>A red car.")
    assert text == "A red car."
    assert reasoning == "weighing options"


def test_strip_unclosed_block_at_budget_exhaustion():
    """The budget ran out mid-thought — no </think> ever arrived. Everything
    after the open tag is reasoning, so a truncated ramble can never be served."""
    text, reasoning = strip_think("<think>Okay, the user wants me to expand")
    assert text == ""
    assert reasoning == "Okay, the user wants me to expand"


def test_strip_unclosed_block_after_prose():
    text, reasoning = strip_think("A red car.\n<think>now let me reconsider")
    assert text == "A red car."
    assert reasoning == "now let me reconsider"


def test_strip_multiple_blocks():
    text, reasoning = strip_think("<think>one</think>A red car. <think>two</think>Wet street.")
    assert text == "A red car. Wet street."
    assert reasoning == "one\ntwo"


def test_strip_no_block_is_identity():
    text, reasoning = strip_think("A red car on a wet street.")
    assert text == "A red car on a wet street."
    assert reasoning == ""


def test_strip_only_thinking():
    """The exact live failure: the ENTIRE budget went to reasoning."""
    text, reasoning = strip_think("<think>Okay, the user wants me to expand...</think>")
    assert text == ""
    assert reasoning == "Okay, the user wants me to expand..."


def test_strip_empty_and_case_insensitive():
    assert strip_think("") == ("", "")
    assert strip_think(None) == ("", "")
    assert strip_think("<THINK>x</THINK>y") == ("y", "x")


def test_strip_an_empty_pre_opened_think_block_still_removes_the_tags():
    # A Qwen3-style template pre-opens an empty <think></think> when thinking is
    # disabled: no reasoning to surface, but the tags must NOT be left in the
    # answer (this leaked into /v1 content until the condition was fixed to
    # rewrite whenever the answer changed, not only when reasoning was present).
    answer, reasoning = strip_think("<think>\n</think>\n\n blue")
    assert answer == "blue"
    assert reasoning == ""


# --------------------------------------------------------------------------- #
# with_no_think / apply_no_think — the suppression half
# --------------------------------------------------------------------------- #
def test_with_no_think_propagates_the_query():
    out = with_no_think("a red car on a wet street")
    assert out.startswith("a red car on a wet street")   # never substituted
    assert NO_THINK_DIRECTIVE in out


def test_with_no_think_is_idempotent():
    once = with_no_think("draft")
    assert with_no_think(once) == once


def test_apply_no_think_targets_last_user_message():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    out = apply_no_think(msgs)
    assert out[0]["content"] == "sys"           # system untouched
    assert out[1]["content"] == "first"         # earlier user turn untouched
    assert NO_THINK_DIRECTIVE in out[3]["content"]
    assert out[3]["content"].startswith("second")
    # never mutates the caller's list or dicts
    assert msgs[3]["content"] == "second"


def test_apply_no_think_multimodal_content_leaves_images_alone():
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:..."}},
        {"type": "text", "text": "describe this"},
    ]}]
    out = apply_no_think(msgs)
    parts = out[0]["content"]
    assert parts[0] == {"type": "image_url", "image_url": {"url": "data:..."}}
    assert NO_THINK_DIRECTIVE in parts[1]["text"]


def test_apply_no_think_system_only_conversation_gets_a_directive_turn():
    out = apply_no_think([{"role": "system", "content": "sys"}])
    assert out[-1]["role"] == "user"
    assert NO_THINK_DIRECTIVE in out[-1]["content"]


# --------------------------------------------------------------------------- #
# finalize_no_think / execute_prompt_no_think — both halves at the seam
# --------------------------------------------------------------------------- #
def test_finalize_splits_and_preserves_reasoning():
    out = finalize_no_think({"ok": True, "text": "<think>hmm</think>A red car."})
    assert out["text"] == "A red car."
    assert out["reasoning"] == "hmm"
    assert out["thinking_suppressed"] is True
    assert out["ok"] is True
    assert out["raw"] == "<think>hmm</think>A red car."


def test_finalize_only_thinking_is_an_honest_failure():
    out = finalize_no_think({"ok": True, "text": "<think>rambling"})
    assert out["ok"] is False
    assert out["text"] == ""
    assert out["reasoning"] == "rambling"
    assert "no-think directive" in out["error"]


def test_finalize_accepts_an_object_result():
    class R:
        ok = True
        text = "<think>x</think>done"
        error = None
    out = finalize_no_think(R())
    assert (out["text"], out["reasoning"]) == ("done", "x")


def _patch_dispatch(monkeypatch, fake):
    """Stub dispatch.execute_prompt for the wrapper.

    LANDMINE: ``from abstract_hugpy_dev.managers import dispatch`` does NOT give
    you the dispatch PACKAGE — ``managers/dispatch/__init__.py`` rebinds the name
    to the inner ``dispatch.dispatch`` module, so patching that attribute patches
    the wrong object and the test quietly runs REAL inference (it loads a model).
    Patch the sys.modules entry, which is what ``from ..managers.dispatch import
    execute_prompt`` actually reads."""
    monkeypatch.setattr(
        sys.modules["hugpy_engine.dispatch"],
        "execute_prompt", fake)


def test_execute_prompt_no_think_sends_directive_and_strips(monkeypatch):
    """The wrapper does BOTH halves, plus the t74 HARD switch: the directive
    rides the messages (a field the wire always carried), and
    chat_template_kwargs {"enable_thinking": False} rides the request for the
    slot path's template to enforce — version-gated at the relay, so an old
    worker still sees only the frozen wire."""
    seen = {}

    def fake_execute_prompt(*args, **kwargs):
        seen.update(kwargs)
        return {"ok": True, "text": "<think>deliberating</think>A red car."}

    _patch_dispatch(monkeypatch, fake_execute_prompt)

    from hugpy_engine.utils.no_think import NO_THINK_CHAT_TEMPLATE_KWARGS, execute_prompt_no_think
    out = execute_prompt_no_think(
        model_key="m", task="text-generation",
        messages=[{"role": "user", "content": "draft"}],
    )

    # half 1: the directive rode the query
    assert NO_THINK_DIRECTIVE in seen["messages"][-1]["content"]
    assert seen["messages"][-1]["content"].startswith("draft")
    # the HARD half rides too (t74) — and nothing else new
    assert seen["chat_template_kwargs"] == NO_THINK_CHAT_TEMPLATE_KWARGS
    assert set(seen) == {"model_key", "task", "messages", "chat_template_kwargs"}
    # half 2: stripped, reasoning preserved under its own key
    assert out["text"] == "A red car."
    assert out["reasoning"] == "deliberating"
    assert out["thinking_suppressed"] is True


def test_execute_prompt_no_think_merges_caller_chat_template_kwargs(monkeypatch):
    """A caller's own chat_template_kwargs survive; enable_thinking is only
    defaulted in, never clobbered."""
    seen = {}

    def fake_execute_prompt(*args, **kwargs):
        seen.update(kwargs)
        return {"ok": True, "text": "clean"}

    _patch_dispatch(monkeypatch, fake_execute_prompt)
    from hugpy_engine.utils.no_think import execute_prompt_no_think
    execute_prompt_no_think(prompt="p",
                            chat_template_kwargs={"custom": 1,
                                                  "enable_thinking": True})
    assert seen["chat_template_kwargs"] == {"custom": 1,
                                            "enable_thinking": True}


def test_execute_prompt_no_think_prompt_shaped_call(monkeypatch):
    seen = {}

    def fake_execute_prompt(*args, **kwargs):
        seen.update(kwargs)
        return {"ok": True, "text": "clean"}

    _patch_dispatch(monkeypatch, fake_execute_prompt)
    from hugpy_engine.utils.no_think import execute_prompt_no_think
    out = execute_prompt_no_think(prompt="describe this")
    assert NO_THINK_DIRECTIVE in seen["prompt"]
    assert out["text"] == "clean"


# --------------------------------------------------------------------------- #
# the migrated route still uses the seam (no local copy left behind)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# STUDIO GENERATE reads a reasoning-only reply as content, not a failure
# (operator 2026-07-31: "no_think isn't a request. it's a function that strips
# the <think>…</think> and returns the think as a dict var"). A reasoning model
# that keeps its whole answer inside <think> — the wazimondo~Qwen3.6-35B case —
# was hard-erroring "returned only reasoning and no output"; studio generate now
# salvages that reasoning as the prompt. Kept LOCAL: the shared finalize_no_think
# stays strict for the judge/gate callers.
# --------------------------------------------------------------------------- #












def test_shared_finalize_stays_strict_for_judge_gate_callers():
    # The salvage is studio-LOCAL. finalize_no_think (discord DEFER gate, movie
    # keyframe verdict) must still treat reasoning-only as a failure, or a
    # verdict regex could match inside the monologue and invert the decision.
    out = finalize_no_think({"ok": True, "text": "<think>the answer is yes</think>"})
    assert out["ok"] is False
    assert out["text"] == ""
    assert out["reasoning"] == "the answer is yes"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
