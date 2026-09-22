"""Tests for the todo-keeper agent node — the RECORDED CONTRACT is the spec.

The contract these assert against is the operator-recorded one in the station
brief ``KEEPER-TASK-todo-agent-node.md`` ("Proposed contract", 2026-07-16). The
host's console-api builds against exactly that shape, so these tests exist to
make a DRIFT from it fail loudly — they are a contract fixture, not a unit test
of our own conveniences.

Four things the brief demanded coverage of, plus the rails:
  * contract shape conformance for BOTH task kinds (todo.add / todo.tidy);
  * the additive-vs-proposal distinction (the rail that decides whether the
    console APPENDS or OFFERS — getting it backwards corrupts a queue);
  * a malformed/garbage LLM reply degrading to status:"error" rather than
    emitting junk items;
  * the 409-is-not-an-error path (first report wins; a re-post is "recorded").

Run:  python -m pytest tests/test_todo_keeper.py -q   (in py/operations/hugpy_ops)
"""
import json
import os
from pathlib import Path

import pytest

from hugpy_ops.todo_keeper import (
    CONTRACT_VERSION,
    KIND_ADD,
    KIND_TIDY,
    MAX_ADD_ITEMS,
    TodoContractError,
    build_messages,
    encode_result,
    handle_task,
    parse_items,
    parse_model_items,
    parse_task,
    result_envelope,
)


# ── helpers ─────────────────────────────────────────────────────────────────
def _items(*texts):
    return [{"type": "todo", "text": t, "note": ""} for t in texts]


def _reply(items):
    return json.dumps(items)


def _stub(reply):
    """An inference callable that returns a fixed reply."""
    def _c(_messages):
        return reply
    return _c


def _decode(outcome):
    """The host's side of the round-trip: result is a STRING to json.parse()."""
    assert isinstance(outcome["result"], str), \
        "result MUST be a string — the host json.parse()s it"
    return json.loads(outcome["result"])


# ── 1. contract shape conformance: todo.add ─────────────────────────────────
def test_add_result_matches_recorded_contract():
    task = {"kind": "todo.add", "v": 1, "vm": "hugpy",
            "instruction": "check the disk and ping the operator",
            "items": []}
    model_items = [
        {"type": "todo", "text": "Check disk usage", "note": "on /mnt"},
        {"type": "request", "text": "Ping the operator", "note": ""},
    ]
    out = handle_task(task, _stub(_reply(model_items)))

    assert out["status"] == "done"
    env = _decode(out)
    # Exactly the recorded envelope keys — no more, no less.
    assert set(env) == {"kind", "v", "items", "mode"}
    assert env["kind"] == "todo.add"
    assert env["v"] == CONTRACT_VERSION == 1
    assert env["mode"] == "additive"
    assert env["items"] == model_items


def test_tidy_result_matches_recorded_contract():
    existing = _items("write docs", "write docs", "ship it")
    task = {"kind": "todo.tidy", "v": 1, "vm": "hugpy", "items": existing}
    revised = [{"type": "todo", "text": "Write docs", "note": "",
                "status": "open"},
               {"type": "todo", "text": "Ship it", "note": "", "status": "open"}]
    out = handle_task(task, _stub(_reply(revised)))

    assert out["status"] == "done"
    env = _decode(out)
    assert set(env) == {"kind", "v", "items", "mode"}
    assert env["kind"] == "todo.tidy"
    assert env["mode"] == "proposal"
    assert env["items"] == revised


# ── 2. the additive-vs-proposal rail ────────────────────────────────────────
def test_mode_is_derived_from_kind_not_caller_supplied():
    """mode is the rail deciding APPEND vs OFFER. It must be a function of kind
    — a tidy marked 'additive' would append a full revised list onto the queue
    it was meant to replace."""
    assert result_envelope(KIND_ADD, [])["mode"] == "additive"
    assert result_envelope(KIND_TIDY, [])["mode"] == "proposal"
    with pytest.raises(TodoContractError):
        result_envelope("todo.nuke", [])


def test_add_never_echoes_the_existing_queue():
    """ADDITIVE means NEW items only. If the node echoed the queue back, the
    console's append would duplicate every existing item."""
    existing = _items("already here", "also here")
    task = {"kind": "todo.add", "v": 1, "vm": "hugpy",
            "instruction": "add one more thing", "items": existing}
    new = [{"type": "todo", "text": "One more thing", "note": ""}]
    out = handle_task(task, _stub(_reply(new)))
    env = _decode(out)
    assert env["items"] == new
    texts = {i["text"] for i in env["items"]}
    assert "already here" not in texts and "also here" not in texts


def test_node_never_mints_ids_by_or_ts():
    """The ~/todo.json writer owns id/by/ts. Even if the model emits them, they
    must not survive into the result."""
    task = {"kind": "todo.add", "v": 1, "vm": "hugpy",
            "instruction": "do a thing", "items": []}
    smuggled = [{"type": "todo", "text": "A thing", "note": "",
                 "id": "t99", "by": "model", "ts": 12345}]
    out = handle_task(task, _stub(_reply(smuggled)))
    env = _decode(out)
    assert env["items"] == [{"type": "todo", "text": "A thing", "note": ""}]
    for item in env["items"]:
        assert "id" not in item and "by" not in item and "ts" not in item


# ── 3. garbage in -> error out, never junk items ────────────────────────────
@pytest.mark.parametrize("garbage", [
    "I'm sorry, I can't help with that.",
    "",
    "   ",
    "{not json at all",
    "[{'text': 'single quotes are not json'}]",
    "[1, 2, 3]",                      # a list, but not of items
    '[{"note": "no text field"}]',    # item-shaped, missing required text
    '[{"type": "explode", "text": "bad type", "note": ""}]',
    "null",
    '"just a string"',
])
def test_malformed_model_reply_degrades_to_error(garbage):
    """A reply we cannot parse is an ERROR — the console falls back to direct
    completions. Emitting a guessed item would pollute the operator's queue."""
    task = {"kind": "todo.add", "v": 1, "vm": "hugpy",
            "instruction": "do something", "items": []}
    out = handle_task(task, _stub(garbage))
    assert out["status"] == "error"
    assert isinstance(out["result"], str) and out["result"]
    # An error result is a PLAIN string, never a JSON items envelope — the host
    # must never mistake a failure for a list of items to append.
    try:
        decoded = json.loads(out["result"])
    except Exception:
        return                      # plain prose: correct
    assert not (isinstance(decoded, dict) and "items" in decoded)


def test_inference_failure_degrades_to_error_not_a_raise():
    """A node that raises into its loop stops serving every later task."""
    def _boom(_messages):
        raise RuntimeError("worker unreachable")
    task = {"kind": "todo.add", "v": 1, "vm": "hugpy",
            "instruction": "x", "items": []}
    out = handle_task(task, _boom)
    assert out["status"] == "error"
    assert "worker unreachable" in out["result"]


def test_fenced_and_garnished_json_is_recovered():
    """Small models garnish JSON with prose/fences even when told not to. That
    is a WELL-FORMED array wearing a hat — recover it (not a tolerance for
    malformed content, which the test above pins as an error)."""
    items = [{"type": "todo", "text": "Recovered", "note": ""}]
    for reply in (f"```json\n{_reply(items)}\n```",
                  f"Sure! Here you go:\n{_reply(items)}\nHope that helps.",
                  f"```\n{_reply(items)}\n```"):
        out = handle_task({"kind": "todo.add", "v": 1, "instruction": "x",
                           "items": []}, _stub(reply))
        assert out["status"] == "done", reply
        assert _decode(out)["items"] == items


def test_add_items_are_capped():
    many = [{"type": "todo", "text": f"item {i}", "note": ""} for i in range(20)]
    out = handle_task({"kind": "todo.add", "v": 1, "instruction": "x",
                       "items": []}, _stub(_reply(many)))
    assert len(_decode(out)["items"]) == MAX_ADD_ITEMS


# ── 4. corrupt inbound payload: REFUSE, never clobber ───────────────────────
@pytest.mark.parametrize("bad_task", [
    None,
    "not an object",
    {"kind": "todo.explode", "v": 1},
    {"kind": "todo.add", "v": 1, "items": []},            # no instruction
    {"kind": "todo.tidy", "v": 1, "items": []},           # tidy w/ empty list
    {"kind": "todo.add", "v": 2, "instruction": "x"},     # unknown version
    {"kind": "todo.add", "v": 1, "instruction": "x",
     "items": "not-a-list"},
    {"kind": "todo.tidy", "v": 1, "items": [{"garbage": True}]},
    {"kind": "todo.tidy", "v": 1, "items": [{"text": ""}]},
])
def test_corrupt_payload_is_refused(bad_task):
    """A payload we cannot read is REFUSED. Tidying a queue we failed to parse
    would silently DELETE what we couldn't read — an error is recoverable, a
    clobbered queue is not."""
    def _must_not_run(_messages):
        raise AssertionError("inference ran on a payload that should be refused")
    out = handle_task(bad_task, _must_not_run)
    assert out["status"] == "error"
    assert "refused" in out["result"]


def test_tidy_with_unparseable_items_never_reaches_the_model():
    """The refusal must happen BEFORE inference — a model shown a half-read
    queue could 'helpfully' return it minus the parts we dropped."""
    calls = []

    def _spy(messages):
        calls.append(messages)
        return _reply([{"type": "todo", "text": "x", "note": ""}])
    out = handle_task({"kind": "todo.tidy", "v": 1,
                       "items": [{"type": "todo", "text": "ok", "note": ""},
                                 {"nonsense": 1}]}, _spy)
    assert out["status"] == "error"
    assert calls == []


# ── 5. inbound parsing details ──────────────────────────────────────────────
def test_parse_task_normalizes_and_defaults():
    norm = parse_task({"kind": "todo.add", "instruction": "  do it  ",
                       "items": None})
    assert norm["kind"] == "todo.add"
    assert norm["v"] == CONTRACT_VERSION
    assert norm["instruction"] == "do it"
    assert norm["items"] == []
    assert norm["vm"] == ""


def test_parse_items_accepts_the_todo_v1_shape():
    got = parse_items([
        {"type": "todo", "text": "a", "note": "n", "status": "open"},
        {"type": "bookmark", "text": "b"},
        {"type": "request", "text": "c", "note": None},
    ])
    assert got == [
        {"type": "todo", "text": "a", "note": "n", "status": "open"},
        {"type": "bookmark", "text": "b", "note": ""},
        {"type": "request", "text": "c", "note": ""},
    ]


def test_status_is_preserved_through_tidy():
    """A tidy proposal must not silently re-open a done item."""
    items = [{"type": "todo", "text": "done thing", "note": "",
              "status": "done"}]
    out = handle_task({"kind": "todo.tidy", "v": 1, "items": items},
                      _stub(_reply(items)))
    assert _decode(out)["items"][0]["status"] == "done"


def test_tidy_restores_a_done_item_the_model_dropped():
    """REGRESSION, from live testing 2026-07-16: a real Qwen2.5-3B tidy of a
    4-item list DROPPED the done item ("ship the release") entirely. A tidy
    result REPLACES the list, so an omitted item is a deleted item — the
    operator applies the proposal and their completed-work record is gone.
    The prompt asks; this asserts."""
    original = [
        {"type": "todo", "text": "check disk", "note": ""},
        {"type": "todo", "text": "ship the release", "note": "",
         "status": "done"},
    ]
    # The model returns ONLY the open item — exactly what it did live.
    dropped = [{"type": "todo", "text": "Check disk", "note": "",
                "status": "open"}]
    out = handle_task({"kind": "todo.tidy", "v": 1, "items": original},
                      _stub(_reply(dropped)))
    env = _decode(out)
    texts = {i["text"] for i in env["items"]}
    assert "ship the release" in texts, "a done item must never be dropped"
    done = [i for i in env["items"] if i.get("status") == "done"]
    assert len(done) == 1 and done[0]["text"] == "ship the release"


def test_tidy_does_not_duplicate_a_done_item_the_model_reworded():
    """The safety net matches on folded text, so a legitimately reworded or
    reordered done item is left alone — restoring it would DUPLICATE it."""
    original = [{"type": "todo", "text": "ship the release", "note": "",
                 "status": "done"}]
    reworded = [{"type": "todo", "text": "Ship the release!", "note": "",
                 "status": "done"}]
    out = handle_task({"kind": "todo.tidy", "v": 1, "items": original},
                      _stub(_reply(reworded)))
    env = _decode(out)
    assert len(env["items"]) == 1, "reworded done item must not be duplicated"


def test_tidy_may_still_merge_and_drop_open_items():
    """The net is narrow on purpose: merging/dropping OPEN items is the job.
    Only 'done' records are protected."""
    original = _items("check disk", "check disk space")
    merged = [{"type": "todo", "text": "Check disk space", "note": "",
               "status": "open"}]
    out = handle_task({"kind": "todo.tidy", "v": 1, "items": original},
                      _stub(_reply(merged)))
    assert len(_decode(out)["items"]) == 1


def test_build_messages_shape():
    for kind, task in (
        (KIND_ADD, {"kind": KIND_ADD, "v": 1, "vm": "hugpy",
                    "instruction": "x", "items": []}),
        (KIND_TIDY, {"kind": KIND_TIDY, "v": 1, "vm": "hugpy",
                     "instruction": "", "items": _items("a")}),
    ):
        msgs = build_messages(parse_task(task))
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert all(isinstance(m["content"], str) and m["content"] for m in msgs)


def test_encode_result_is_parseable_text():
    s = encode_result(result_envelope(KIND_ADD, _items("a")))
    assert isinstance(s, str)
    assert json.loads(s)["mode"] == "additive"


# ── 6. the 409-is-not-an-error path ─────────────────────────────────────────
class _FakeResponses:
    """Drives TodoKeeperNode.report through _request without a network."""

    def __init__(self, *codes):
        self.codes = list(codes)
        self.calls = []

    def __call__(self, method, url, *, body=None, headers=None,
                 timeout=None):
        self.calls.append((method, url, body))
        code = self.codes.pop(0)
        return code, {"seq": 1, "status": "done"}


@pytest.fixture
def node(tmp_path, monkeypatch):
    from hugpy_ops import todo_keeper_daemon as d
    state = tmp_path / "state.json"
    state.write_text(json.dumps(
        {"node_id": "agn_test", "token": "agt_test", "cursor": 0}))
    monkeypatch.setattr(d, "_state_path", lambda: str(state))
    return d.TodoKeeperNode(state_path=str(state))


def test_report_treats_200_as_recorded(node, monkeypatch):
    from hugpy_ops import todo_keeper_daemon as d
    fake = _FakeResponses(200)
    monkeypatch.setattr(d, "_request", fake)
    assert node.report(1, {"status": "done", "result": "{}"}) is True


def test_report_treats_409_as_recorded_not_an_error(node, monkeypatch):
    """409 = already finalized (first report wins). A node that crashed after
    reporting re-posts and gets 409; treating that as failure would wedge the
    node on that task forever."""
    from hugpy_ops import todo_keeper_daemon as d
    monkeypatch.setattr(d, "_request", _FakeResponses(409))
    assert node.report(1, {"status": "done", "result": "{}"}) is True


def test_report_treats_a_real_failure_as_not_recorded(node, monkeypatch):
    """500 must NOT advance the cursor — the answer was lost and the task must
    be re-pulled."""
    from hugpy_ops import todo_keeper_daemon as d
    monkeypatch.setattr(d, "_request", _FakeResponses(500))
    assert node.report(1, {"status": "done", "result": "{}"}) is False


def test_cursor_advances_only_after_a_recorded_report(node, monkeypatch):
    """At-least-once: a crash before the report re-pulls the task (safe, thanks
    to 409). A cursor advanced early would DROP the answer."""
    from hugpy_ops import todo_keeper_daemon as d

    def _req(method, url, *, body=None, headers=None, timeout=None):
        if method == "GET" and "/tasks?since=" in url:
            return 200, {"tasks": [{"seq": 7, "task": {
                "kind": "todo.add", "v": 1, "vm": "hugpy",
                "instruction": "do a thing", "items": []}}]}
        if method == "POST" and url.endswith("/result"):
            return 200, {"seq": 7}
        if method == "POST" and url.endswith("/heartbeat"):
            return 200, {}
        if url.endswith("/v1/chat/completions"):
            return 200, {"choices": [{"message": {"content": _reply(
                [{"type": "todo", "text": "A thing", "note": ""}])}}]}
        raise AssertionError(f"unexpected {method} {url}")

    monkeypatch.setattr(d, "_request", _req)
    assert node.cursor == 0
    assert node.poll_once() == 1
    assert node.cursor == 7           # advanced, and persisted
    assert json.loads(Path(node.state_path).read_text())["cursor"] == 7


def test_cursor_does_not_advance_when_reporting_fails(node, monkeypatch):
    from hugpy_ops import todo_keeper_daemon as d

    def _req(method, url, *, body=None, headers=None, timeout=None):
        if method == "GET" and "/tasks?since=" in url:
            return 200, {"tasks": [{"seq": 7, "task": {
                "kind": "todo.add", "v": 1, "instruction": "x", "items": []}}]}
        if method == "POST" and url.endswith("/result"):
            return 503, "central down"
        if method == "POST" and url.endswith("/heartbeat"):
            return 200, {}
        if url.endswith("/v1/chat/completions"):
            return 200, {"choices": [{"message": {"content": _reply(
                [{"type": "todo", "text": "x", "note": ""}])}}]}
        raise AssertionError(f"unexpected {method} {url}")

    monkeypatch.setattr(d, "_request", _req)
    assert node.poll_once() == 0
    assert node.cursor == 0           # NOT advanced — the task re-pulls


# ── 7. token durability ─────────────────────────────────────────────────────
def test_state_file_is_written_0600(tmp_path):
    from hugpy_ops.todo_keeper_daemon import load_state, save_state
    p = tmp_path / "sub" / "state.json"
    save_state({"node_id": "agn_x", "token": "agt_secret", "cursor": 3},
               str(p))
    assert load_state(str(p))["token"] == "agt_secret"
    assert oct(p.stat().st_mode & 0o777) == "0o600"


def test_missing_state_is_empty_but_corrupt_state_raises(tmp_path):
    """A missing file = never enrolled (enroll). A CORRUPT file must NOT be
    silently overwritten — re-enrolling would orphan the live node row and
    strand whatever the host is polling."""
    from hugpy_ops.todo_keeper_daemon import load_state
    assert load_state(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    with pytest.raises(RuntimeError, match="refusing to re-enroll"):
        load_state(str(bad))
