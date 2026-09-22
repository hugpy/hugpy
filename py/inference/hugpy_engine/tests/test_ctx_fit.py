"""CTX-FIT GUARD — long chat sessions must not die when the window fills.

The operator's chat sessions tapped out after ~10 queries: the UI posts the
WHOLE history every round, so once the accumulated turns outgrew the model's
32k window every subsequent request was over-ctx and refused — the session
could never recover. chat_context.compact_chat_request existed for exactly
this but was never wired into the serving path (dead code).

The fix is the ctx-fit guard (chat_context.ctx_fit_chat_request), applied in
_build_chat_request — the one funnel every chat pass (console, /v1, and each
continuation pass) goes through. Contract:

  * estimate prompt tokens with the existing chars/4 counter
    (estimate_message_tokens); ctx_max is model meta model_max_length — the
    SAME figure /v1/models reports as context_length;
  * over budget -> DROP the OLDEST non-system turns until it fits, always
    keeping system message(s), the newest user turn, and as much recent tail
    as fits; one INFO log names the dropped-turn count;
  * DROP-ONLY: content is never rewritten. When even the minimal set
    (system + newest user turn) cannot fit, the request passes through
    UNTOUCHED and today's honest refusal stands;
  * FAIL-OPEN: any error inside the guard (or importing it) leaves the
    request untouched.

Central-side only (no worker / no GPU / no model). Runs under pytest AND as a
plain script:
    venv/bin/python -m pytest tests/test_ctx_fit.py -q
    venv/bin/python tests/test_ctx_fit.py
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# In-process only — no cross-process comms DB side effects during tests.
os.environ.setdefault("HUGPY_COMMS_DB", "off")

ok = 0


def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


CTX = 8192          # small window so histories overflow with modest turns
BIG = "x" * 4000    # ~1000 estimated tokens per turn at chars/4


def _history(turns, system="Be terse."):
    """system + `turns` alternating user/assistant BIG turns, ending on user."""
    out = [{"role": "system", "content": system}] if system else []
    for i in range(turns):
        # oldest first; parity chosen so the NEWEST turn is always user
        role = "user" if (turns - 1 - i) % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"turn{i} {BIG}"})
    return out


# ---------------------------------------------------------------------------
# 1. the pure fit — ctx_fit_keep_indices
# ---------------------------------------------------------------------------
def _pure_checks(cc):
    fit = cc.ctx_fit_keep_indices
    est = cc.estimate_message_tokens

    # -- already fits: untouched --------------------------------------------
    small = _history(3)
    check("a short history is left untouched",
          fit(small, ctx_max=CTX) == (None, 0))
    check("an unknown window (ctx_max=0) is left untouched",
          fit(_history(20), ctx_max=0) == (None, 0))
    check("an unknown window (ctx_max=None) is left untouched",
          fit(_history(20), ctx_max=None) == (None, 0))

    # -- overflow: oldest turns drop, the recent tail survives ---------------
    msgs = _history(10)
    keep, dropped = fit(msgs, ctx_max=CTX)
    check("an overgrown history drops turns", keep is not None and dropped > 0)
    check("the system message is always kept", 0 in keep)
    check("the newest user turn is always kept", (len(msgs) - 1) in keep)
    check("the kept dialogue is the CONTIGUOUS newest tail (oldest N dropped)",
          keep == [0] + list(range(len(msgs) - (len(keep) - 1), len(msgs))))
    check("dropped counts exactly the missing dialogue turns",
          dropped == (len(msgs) - 1) - (len(keep) - 1))
    check("the first kept dialogue turn is a USER turn (template alternation)",
          msgs[keep[1]]["role"] == "user")

    # -- and the survivor actually fits the budget it was cut to -------------
    reserved = min(32768, max(4096, CTX // 3))
    budget = CTX - reserved - 512
    check("the kept set fits the input budget",
          sum(est(msgs[i]) for i in keep) <= budget)

    # -- a smaller output reservation keeps MORE history ---------------------
    keep_s, dropped_s = fit(msgs, ctx_max=CTX, requested_output_tokens=256)
    check("a small explicit max_tokens reservation keeps more turns",
          dropped_s < dropped)

    # -- minimal set can't fit: untouched (today's honest refusal) -----------
    huge = [{"role": "system", "content": "S"},
            {"role": "user", "content": "y" * (CTX * 8)}]
    check("an oversized newest user turn is NEVER trimmed or dropped — "
          "the request passes through for the honest refusal",
          fit(huge, ctx_max=CTX) == (None, 0))
    huge_sys = [{"role": "system", "content": "y" * (CTX * 8)},
                {"role": "user", "content": "hi"}]
    check("an oversized system prompt likewise passes through untouched",
          fit(huge_sys, ctx_max=CTX) == (None, 0))

    # -- degenerate shapes ----------------------------------------------------
    check("a history with no user turn is left untouched",
          fit([{"role": "system", "content": "S"}] +
              [{"role": "assistant", "content": BIG}] * 10,
              ctx_max=CTX) == (None, 0))
    check("a system-only history is left untouched",
          fit([{"role": "system", "content": "S"}], ctx_max=CTX) == (None, 0))
    check("an empty history is left untouched", fit([], ctx_max=CTX) == (None, 0))

    # -- an orphaned assistant reply never leads the kept tail ---------------
    # Force a boundary where the fit lands on an assistant turn: all-assistant
    # history except one giant old user turn and the newest user turn.
    tricky = ([{"role": "user", "content": "z" * (CTX * 8)}] +
              [{"role": "assistant", "content": f"a{i} {BIG}"} for i in range(6)] +
              [{"role": "user", "content": "newest " + BIG}])
    keep_t, dropped_t = fit(tricky, ctx_max=CTX)
    check("assistant turns whose user turn was dropped are dropped too",
          keep_t is not None and tricky[keep_t[0]]["role"] == "user")


# ---------------------------------------------------------------------------
# 2. the request-level guard — ctx_fit_chat_request
# ---------------------------------------------------------------------------
class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _request_checks(cc, schemas):
    CR = schemas.ChatRequest
    orig_ctx = cc._ctx_max_for_model
    handler = _Records()
    cc.logger.addHandler(handler)
    orig_level = cc.logger.level
    cc.logger.setLevel(logging.INFO)  # default effective level is WARNING
    cc._ctx_max_for_model = lambda mk: CTX
    try:
        # -- an overgrown request comes back shortened, everything else intact
        req = CR(model_key="ctx-fit-test", messages=_history(10),
                 max_new_tokens=256, temperature=0.5)
        fitted = cc.ctx_fit_chat_request(req)
        check("an overgrown request comes back with fewer messages",
              len(fitted.messages) < len(req.messages))
        check("the system message survives",
              fitted.messages[0].role == "system"
              and fitted.messages[0].content == "Be terse.")
        check("the newest user turn survives verbatim",
              fitted.messages[-1].content == req.messages[-1].content)
        check("non-message fields ride through the copy untouched",
              fitted.request_id == req.request_id
              and fitted.max_new_tokens == 256
              and fitted.temperature == 0.5
              and fitted.model_key == "ctx-fit-test")
        dropped = len(req.messages) - len(fitted.messages)
        infos = [r for r in handler.records if r.levelno == logging.INFO
                 and "ctx_fit" in r.getMessage()]
        check("ONE INFO log names the dropped-turn count",
              len(infos) == 1 and str(dropped) in infos[0].getMessage())

        # -- a fitting request is the SAME object (zero-copy no-op) ----------
        handler.records.clear()
        req_small = CR(model_key="ctx-fit-test", messages=_history(2))
        check("a fitting request is returned as the SAME object",
              cc.ctx_fit_chat_request(req_small) is req_small)
        check("a no-op emits no ctx_fit INFO log",
              not any("ctx_fit" in r.getMessage() for r in handler.records
                      if r.levelno == logging.INFO))

        # -- minimal set can't fit: untouched (honest refusal preserved) -----
        req_huge = CR(model_key="ctx-fit-test",
                      messages=[{"role": "system", "content": "S"},
                                {"role": "user", "content": "y" * (CTX * 8)}])
        check("a request whose minimal set can't fit passes through untouched",
              cc.ctx_fit_chat_request(req_huge) is req_huge)

        # -- FAIL-OPEN: a guard error leaves the request untouched -----------
        def _boom(mk):
            raise RuntimeError("registry exploded")
        cc._ctx_max_for_model = _boom
        req_big = CR(model_key="ctx-fit-test", messages=_history(10))
        check("a guard error returns the request UNTOUCHED (fail-open)",
              cc.ctx_fit_chat_request(req_big) is req_big)

        # -- unknown model: no window, no guess, no change --------------------
        cc._ctx_max_for_model = orig_ctx
        req_unknown = CR(model_key="no-such-model-ctx-fit", messages=_history(10))
        check("an unknown model's window is never guessed — untouched",
              cc.ctx_fit_chat_request(req_unknown) is req_unknown)
    finally:
        cc._ctx_max_for_model = orig_ctx
        cc.logger.setLevel(orig_level)
        cc.logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# 3. ctx_max sourcing — the SAME figure /v1/models reports as context_length
# ---------------------------------------------------------------------------
def _sourcing_checks(cc):
    from hugpy_engine.config.models.models_default import DEFAULT_CONTEXT_TOKENS_BY_MODEL
    known = next((k for k, v in DEFAULT_CONTEXT_TOKENS_BY_MODEL.items() if v),
                 None)
    if known is None:  # registry empty on this box — nothing to source
        print("  ~ skip sourcing checks (no model with model_max_length)")
        return
    check("ctx_max for a registry model is its model_max_length",
          cc._ctx_max_for_model(known)
          == int(DEFAULT_CONTEXT_TOKENS_BY_MODEL[known]))
    check("ctx_max for an unknown model is 0 (guard skips, never guesses)",
          cc._ctx_max_for_model("no-such-model-ctx-fit") == 0)
    check("ctx_max for model_key=None is 0", cc._ctx_max_for_model(None) == 0)


# ---------------------------------------------------------------------------
# 4. the seam — _build_chat_request applies the guard, fail-open
# ---------------------------------------------------------------------------
def _builder_checks(cc, builders):
    orig_ctx = cc._ctx_max_for_model
    orig_guard = cc.ctx_fit_chat_request
    cc._ctx_max_for_model = lambda mk: CTX
    try:
        req = builders._build_chat_request(
            {"messages": _history(10)}, "ctx-fit-test")
        check("_build_chat_request ships a FITTED request",
              len(req.messages) < len(_history(10)))
        check("the builder's fitted request keeps system + newest user turn",
              req.messages[0].role == "system"
              and req.messages[-1].role == "user"
              and "turn9" in req.messages[-1].content)

        # a fitting request builds exactly as before
        req_small = builders._build_chat_request(
            {"messages": _history(3)}, "ctx-fit-test")
        check("a fitting request builds with its full history",
              len(req_small.messages) == 4)

        # FAIL-OPEN at the seam too: a guard that raises changes nothing.
        def _boom(req):
            raise RuntimeError("guard exploded")
        cc.ctx_fit_chat_request = _boom
        req_open = builders._build_chat_request(
            {"messages": _history(10)}, "ctx-fit-test")
        check("a raising guard leaves the built request untouched (fail-open)",
              len(req_open.messages) == len(_history(10)))
    finally:
        cc._ctx_max_for_model = orig_ctx
        cc.ctx_fit_chat_request = orig_guard


# ---------------------------------------------------------------------------
def test_ctx_fit():
    global ok
    ok = 0
    cc = importlib.import_module(
        "hugpy_engine.chat_context.chat_context")
    schemas = importlib.import_module(
        "hugpy_engine.schemas.chat_schemas")
    builders = importlib.import_module(
        "hugpy_engine.resolvers.categories.builders")
    _pure_checks(cc)
    _request_checks(cc, schemas)
    _sourcing_checks(cc)
    _builder_checks(cc, builders)
    print(f"\nall {ok} checks passed")


if __name__ == "__main__":
    test_ctx_fit()
