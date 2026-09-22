"""The todo-keeper agent node — the PURE core (no network, no clock, no I/O).

This is the brain of the standing todo-keeper node (the daemon that carries it
is ``todo_keeper_daemon.py`` / the ``hugpy-todo-keeper`` console script). Everything here is a pure function of its
inputs so the contract can be tested without a fleet, an API, or a model:
build a prompt -> parse a model reply -> shape a contract result. The daemon
owns the sockets; this module owns the MEANING.

The contract it implements is RECORDED AND BINDING (station brief
``KEEPER-TASK-todo-agent-node.md``, "Proposed contract", 2026-07-16). The host's
``console-api`` builds against exactly this shape, so the shapes below are not
ours to drift:

    dispatch  {"task": {"kind": "todo.add"|"todo.tidy", "v": 1, "vm", ...}}
    result    POST /agent/<id>/tasks/<seq>/result {"status", "result": <string>}
              -> result is the JSON *STRING* of
                 {"kind", "v": 1, "items": [...], "mode": "additive"|"proposal"}

THE RAILS (from the brief; these are behavioural, not stylistic):

  * ADDITIVE-OR-PROPOSE ONLY. ``todo.add`` returns ONLY new items, which the
    console APPENDS — the node never echoes the existing queue back, because
    an echo would duplicate the queue on append. ``todo.tidy`` returns the FULL
    revised list as a PROPOSAL the operator applies by hand. Neither path ever
    writes a queue. The node cannot destructively rewrite anything: it has no
    writer, only a reply.
  * THE NODE DOES NOT MINT IDENTITY. No ``id``, no ``by``, no ``ts`` — the
    ``~/todo.json`` writer owns those. We return CONTENT. This is what keeps the
    shared file the keeper interface rather than a thing the model stomps on.
  * A CORRUPT PAYLOAD IS REFUSED, NEVER CLOBBERED. If the inbound items don't
    parse as the todo.v1 shape, we raise ``TodoContractError`` and the daemon
    reports ``status:"error"``. Refusing is the only safe move: "tidy" a queue
    we failed to read and the proposal would silently DELETE what we couldn't
    parse. An error is recoverable; a clobbered queue is not.
  * GARBAGE IN, ERROR OUT — NEVER JUNK ITEMS. A model reply we cannot parse into
    valid items is an ``error``, not a best-effort guess. The console falls back
    to direct completions on error, so failing honestly costs the operator
    nothing; emitting junk items would cost them a polluted queue.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# The contract version. Bump ONLY on a breaking change to the wire shape, and
# only in lockstep with the host's console-api (the brief's contract is shared).
CONTRACT_VERSION = 1

KIND_ADD = "todo.add"
KIND_TIDY = "todo.tidy"
_KINDS = (KIND_ADD, KIND_TIDY)

# todo.v1 item vocabulary — the shapes the console's drawer renders.
ITEM_TYPES = ("todo", "request", "bookmark")
ITEM_STATUSES = ("open", "done")

# ``add`` is capped at 6 items, matching the console's current direct path (the
# fallback must not behave differently from the agent path — a mode switch the
# operator can FEEL is a bug).
MAX_ADD_ITEMS = 6
# A tidy proposal is bounded too: a model that decides to "expand" a queue into
# hundreds of items is malfunctioning, and the result column is size-capped
# (65536 B) upstream anyway — better to refuse than to have the store truncate
# a proposal into invalid JSON the host cannot parse.
MAX_TIDY_ITEMS = 200

_TEXT_MAX = 400
_NOTE_MAX = 800


class TodoContractError(Exception):
    """A payload/reply that cannot be honoured as the recorded contract.

    The daemon maps this to ``status:"error"`` + a human-readable string, which
    is exactly what the host's fallback path expects. Raised for BOTH a corrupt
    inbound payload (refuse, never clobber) and an unparseable model reply
    (error, never junk items)."""


# ── inbound: the dispatched task ────────────────────────────────────────────
def parse_item(raw: Any, *, where: str = "items") -> dict[str, Any]:
    """Coerce ONE inbound todo.v1 item, or refuse.

    Deliberately strict on TYPE (a wrong type means we are not looking at a
    todo.v1 item and must not guess) and forgiving on absence (``note`` missing
    is normal; ``status`` missing means "open"). Unknown keys are DROPPED, not
    preserved — the node returns content in the recorded shape only, and
    passing through an unknown key would let a model invent ``id``/``by``/``ts``
    and violate the no-minting rail."""
    if not isinstance(raw, dict):
        raise TodoContractError(
            f"{where}: expected a todo item object, got {type(raw).__name__}")
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        raise TodoContractError(f"{where}: item is missing a non-empty 'text'")
    itype = raw.get("type", "todo")
    if itype not in ITEM_TYPES:
        raise TodoContractError(
            f"{where}: item 'type' must be one of {ITEM_TYPES}, got {itype!r}")
    note = raw.get("note", "")
    if note is None:
        note = ""
    if not isinstance(note, str):
        raise TodoContractError(f"{where}: item 'note' must be a string")
    item = {"type": itype, "text": text.strip()[:_TEXT_MAX],
            "note": note.strip()[:_NOTE_MAX]}
    status = raw.get("status")
    if status is not None:
        if status not in ITEM_STATUSES:
            raise TodoContractError(
                f"{where}: item 'status' must be one of {ITEM_STATUSES}")
        item["status"] = status
    return item


def parse_items(raw: Any, *, where: str = "items") -> list[dict[str, Any]]:
    """Coerce an inbound item LIST, or refuse. ``None``/absent -> ``[]`` (an
    empty queue is a legitimate state); a non-list is a corrupt payload."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TodoContractError(
            f"{where}: expected a list of items, got {type(raw).__name__}")
    return [parse_item(x, where=f"{where}[{i}]") for i, x in enumerate(raw)]


def parse_task(task: Any) -> dict[str, Any]:
    """Validate a dispatched task against the recorded contract.

    Returns a normalized ``{kind, v, vm, instruction, items}``. Raises
    ``TodoContractError`` on anything we cannot honour — which the daemon turns
    into ``status:"error"``, leaving the host free to fall back. We validate
    here because the dispatch route does NOT: it requires only a top-level
    ``task`` key and is otherwise free-form (verified in agent_routes.py), so
    this is the contract's only enforcement point."""
    if not isinstance(task, dict):
        raise TodoContractError("task must be an object")
    kind = task.get("kind")
    if kind not in _KINDS:
        raise TodoContractError(
            f"unsupported task kind {kind!r}; expected one of {_KINDS}")

    # Version: we accept an ABSENT v (be liberal — an early host build may omit
    # it) but REFUSE a version we don't implement. Silently treating a v2 task
    # as v1 would answer a question that was never asked.
    v = task.get("v", CONTRACT_VERSION)
    if v is not None and v != CONTRACT_VERSION:
        raise TodoContractError(
            f"unsupported contract version {v!r}; this node implements v{CONTRACT_VERSION}")

    vm = task.get("vm")
    if vm is not None and not isinstance(vm, str):
        raise TodoContractError("'vm' must be a string when present")

    instruction = task.get("instruction")
    if instruction is not None and not isinstance(instruction, str):
        raise TodoContractError("'instruction' must be a string when present")
    instruction = (instruction or "").strip()

    # Per the contract: add REQUIRES an instruction (it is the ask); tidy
    # REQUIRES items (it is the list to tidy) and its instruction is an
    # optional steer.
    if kind == KIND_ADD and not instruction:
        raise TodoContractError("todo.add requires a non-empty 'instruction'")

    items = parse_items(task.get("items"), where="task.items")
    if kind == KIND_TIDY and not items:
        raise TodoContractError("todo.tidy requires a non-empty 'items' list")

    return {"kind": kind, "v": CONTRACT_VERSION, "vm": vm or "",
            "instruction": instruction, "items": items}


# ── outbound: the result envelope ───────────────────────────────────────────
def result_envelope(kind: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    """The ``status:"done"`` payload, pre-serialization.

    ``mode`` is derived from ``kind``, never passed in — it is the rail that
    tells the console whether to APPEND or to OFFER A PROPOSAL, and deriving it
    means a caller cannot accidentally mark a tidy 'additive' (which would
    append a whole revised list onto the queue it was meant to replace)."""
    if kind not in _KINDS:
        raise TodoContractError(f"cannot build a result for kind {kind!r}")
    return {
        "kind": kind,
        "v": CONTRACT_VERSION,
        "items": items,
        "mode": "additive" if kind == KIND_ADD else "proposal",
    }


def encode_result(envelope: dict[str, Any]) -> str:
    """Serialize the envelope to the STRING the result route stores.

    The route ``json.dumps()`` any non-string itself, so this is belt-and-braces
    — but doing it here makes the round-trip explicit and testable: the host
    MUST ``json.parse()`` the polled ``result`` field. It is text, not a nested
    object."""
    return json.dumps(envelope, ensure_ascii=False)


# ── the model conversation ──────────────────────────────────────────────────
_SYSTEM_ADD = """You turn a person's free-text note into structured to-do items.

Reply with ONLY a JSON array. No prose, no markdown fences.
Each element: {"type": "todo"|"request"|"bookmark", "text": "...", "note": "..."}
  - "todo"     something the VM's keeper should DO.
  - "request"  something being asked OF someone else (another session/operator).
  - "bookmark" a link/reference to keep, not an action.
Rules:
  - At most %(max)d items. Fewer is better; do not pad.
  - "text" is one short imperative line. "note" is optional context ("" if none).
  - Split a genuinely multi-part ask into separate items; do NOT invent work
    that was not asked for.
  - Do NOT include ids, timestamps, or authors. Content only.
  - Do NOT repeat an item that already exists in the current list.""" % {
    "max": MAX_ADD_ITEMS}

_SYSTEM_TIDY = """You tidy a person's to-do list and PROPOSE a revised version.

Reply with ONLY a JSON array — the FULL revised list. No prose, no fences.
Each element: {"type": "todo"|"request"|"bookmark", "text": "...", "note": "...",
"status": "open"|"done"}
Rules:
  - Your output REPLACES the list, so it must contain EVERY item that still has
    meaning. An item you omit is DELETED. When in doubt, KEEP it.
  - Merge only EXACT duplicates (two items that mean the same thing). Two items
    that merely look similar are DIFFERENT items — keep both, separately.
  - NEVER drop a "done" item. Completed work stays on the list with
    "status": "done" — it is a record, not clutter.
  - Copy each item's "status" through EXACTLY as given. If an item has no
    status, use "open". Never change "done" to "open" or the reverse.
  - Do NOT drop an item because it looks stale — this is a proposal a human
    reviews, not a purge.
  - Do NOT invent new work.
  - Do NOT include ids, timestamps, or authors. Content only."""


def _items_for_prompt(items: list[dict[str, Any]]) -> str:
    return json.dumps(items, ensure_ascii=False, indent=None)


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    """The chat messages for a normalized task (from ``parse_task``)."""
    kind = task["kind"]
    vm = task.get("vm") or "this VM"
    if kind == KIND_ADD:
        user = (
            f"VM: {vm}\n"
            f"Current list (for context — do NOT repeat these):\n"
            f"{_items_for_prompt(task['items'])}\n\n"
            f"The ask:\n{task['instruction']}\n\n"
            f"JSON array of the NEW items only:"
        )
        return [{"role": "system", "content": _SYSTEM_ADD},
                {"role": "user", "content": user}]
    steer = task.get("instruction") or "(no extra steer — general tidy)"
    user = (
        f"VM: {vm}\n"
        f"Steer: {steer}\n\n"
        f"The list to tidy:\n{_items_for_prompt(task['items'])}\n\n"
        f"JSON array of the FULL revised list:"
    )
    return [{"role": "system", "content": _SYSTEM_TIDY},
            {"role": "user", "content": user}]


# ── parsing the model's reply ───────────────────────────────────────────────
# (the fence regex moved to utils/json_scavenge with the candidate walk)


def _extract_json_array(text: str) -> Any:
    """Pull a JSON array out of a model reply, or raise.

    Small models wrap JSON in prose or fences even when told not to, so we try,
    in order: the whole string; a fenced block; the outermost [...] span. This
    is recovery of a WELL-FORMED array that arrived with garnish — NOT a
    tolerance for malformed content. If none of these yield a list, we raise and
    the caller degrades to status:"error" (never a guess).

    The candidate walk itself is now the package seam
    ``utils/json_scavenge.extract_json_array`` (``review/judge`` held a second
    copy; the studio spread path wanted a third). Behaviour is unchanged,
    including the lone-object near-miss tolerance — what stays HERE is the
    raise, because the contract error type is this module's."""
    if not isinstance(text, str) or not text.strip():
        raise TodoContractError("model returned an empty reply")
    from hugpy_engine.utils.json_scavenge import extract_json_array
    # A lone object is a near-miss we accept as a 1-item array; anything else
    # (a bare string/number) is not a list of items and is refused.
    parsed = extract_json_array(text, accept_lone_object=True)
    if parsed is not None:
        return parsed
    raise TodoContractError(
        "model reply did not contain a JSON array of items")


def _norm_text(s: str) -> str:
    """Loose key for 'is this the same item' — case/punctuation/space folded."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def restore_dropped_done(original: list[dict[str, Any]],
                         revised: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-append any ``status:"done"`` item the tidy silently dropped.

    WHY THIS EXISTS (found in live testing, 2026-07-16, not in a unit test):
    a real Qwen2.5-3B tidy of a 4-item list dropped the "done" item entirely.
    The prompt says to keep it; the model did not. A tidy result REPLACES the
    list, so an omitted item is a DELETED item — the operator clicks apply and
    their completed-work record is gone.

    The prompt is a request; this is the guarantee. A model may reword, merge,
    or reorder a done item (all fine — we match on folded text, and a reworded
    match is left alone), but it may not make one VANISH. Dropped items are
    appended back at the end, preserving their original content.

    Deliberately narrow: only ``done`` items are restored. Dropping/merging an
    OPEN item is legitimate tidying (that's the job); losing a completed-work
    record is data loss the operator cannot reconstruct from the proposal."""
    kept = {_norm_text(i.get("text", "")) for i in revised}
    out = list(revised)
    for item in original:
        if item.get("status") != "done":
            continue
        if _norm_text(item.get("text", "")) in kept:
            continue
        out.append(dict(item))
    return out


def parse_model_items(reply: str, *, kind: str) -> list[dict[str, Any]]:
    """Turn a raw model reply into validated todo.v1 items, or raise.

    Every item is run through the SAME ``parse_item`` the inbound payload uses —
    so a model cannot smuggle an ``id``/``by``/``ts`` (dropped as unknown keys)
    or an invented type (refused). Over-long replies are truncated to the
    contract cap rather than refused: an over-eager model is a quality problem,
    not a correctness one, and the cap is what the console expects."""
    raw = _extract_json_array(reply)
    items = [parse_item(x, where=f"reply[{i}]") for i, x in enumerate(raw)]
    if not items:
        raise TodoContractError("model returned zero items")
    cap = MAX_ADD_ITEMS if kind == KIND_ADD else MAX_TIDY_ITEMS
    return items[:cap]


def handle_task(task: Any, complete: Any) -> dict[str, Any]:
    """The whole node behaviour, minus I/O: task -> {status, result}.

    ``complete(messages) -> str`` is the inference callable (the daemon passes
    one that talks to hugpy; a test passes a stub). Every failure path — corrupt
    payload, model error, unparseable reply — becomes ``status:"error"`` with a
    human-readable string, because the host's contract says an error means
    "fall back to direct completions". Nothing here can raise into the daemon's
    loop: a node that dies on a bad task stops serving every later one."""
    try:
        norm = parse_task(task)
    except TodoContractError as e:
        return {"status": "error", "result": f"todo-keeper: refused task: {e}"}

    # NO-THINK (utils/no_think.py). This is a strict JSON-array contract at
    # temperature 0 — "we are extracting structure, not writing prose". The
    # fence/brace scavenger in parse_model_items exists precisely BECAUSE models
    # wrap output in prose; a <think> block is that failure in its worst form.
    # Suppress at generation, then strip defensively before the parser sees it.
    from hugpy_engine.utils.no_think import apply_no_think, strip_think
    try:
        reply = complete(apply_no_think(build_messages(norm)))
    except Exception as e:  # inference is a network call to the fleet
        return {"status": "error",
                "result": f"todo-keeper: inference failed: {e}"}
    reply, _reasoning = strip_think(reply or "")

    try:
        items = parse_model_items(reply, kind=norm["kind"])
    except TodoContractError as e:
        return {"status": "error",
                "result": f"todo-keeper: unusable model reply: {e}"}

    # A tidy REPLACES the list, so a silently dropped 'done' item is data loss.
    # The prompt asks the model to keep them; this makes it structural.
    if norm["kind"] == KIND_TIDY:
        items = restore_dropped_done(norm["items"], items)[:MAX_TIDY_ITEMS]

    return {"status": "done",
            "result": encode_result(result_envelope(norm["kind"], items))}
