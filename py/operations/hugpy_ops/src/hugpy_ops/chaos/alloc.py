"""Snapshot / apply / restore a model's per-worker spill (the alloc contract).

The runner changes ONE thing per trial: ``spill_by_model[model_key]`` on the
target workers, via the operator-gated /assign route. This module makes that
strictly reversible:

  1. snapshot the CURRENT spill on each target worker BEFORE the trial;
  2. apply the chaos spill;
  3. restore the exact prior spill AFTER the trial and VERIFY it took.

Safety rails honoured here:
  * only already-assigned (worker, model) pairs are touched, so restore is a
    write-back of the prior value — never an unassign (no pin-409 risk);
  * a non-200 apply (e.g. 409 engine-gate refusal) aborts the trial cleanly with
    nothing to restore (the registry was not written);
  * restore is best-effort-until-verified: it re-reads spill_by_model and
    reports per-worker matches, so the runner can surface an un-restored alloc
    loudly instead of leaving the fleet drifted."""
from __future__ import annotations


def _spill_of(worker_row: dict, model_key: str) -> dict | None:
    """The current spill for a model on a worker, or None if none set."""
    sbm = worker_row.get("spill_by_model") or {}
    return sbm.get(model_key)


def snapshot(client, model_key: str, worker_names: list[str],
             widx: dict) -> dict:
    """Read the current spill for the model on each named worker.

    ``widx`` maps name -> {id, ...} (from assortment.worker_index over a fresh
    /llm/workers). Returns {name: {"worker_id", "before": spill|None}}."""
    workers = {w.get("name"): w for w in client.workers()}
    snap = {}
    for name in worker_names:
        wid = widx.get(name, {}).get("id")
        row = workers.get(name) or {}
        snap[name] = {"worker_id": wid, "before": _spill_of(row, model_key)}
    return snap


def apply(client, model_key: str, spill: dict, snap: dict) -> dict:
    """Apply the chaos spill to every snapshotted worker. Returns
    {"ok": bool, "results": {name: {status, error}}}. On ANY non-200 the caller
    should NOT fire; ``restore`` will still cleanly put back whatever DID change
    (a partial apply is possible if there are multiple targets)."""
    results = {}
    all_ok = True
    for name, s in snap.items():
        wid = s.get("worker_id")
        if not wid:
            results[name] = {"status": None, "error": "unknown worker id"}
            all_ok = False
            continue
        code, body = client.assign(wid, model_key, spill)
        err = None if code == 200 else (
            body.get("error") if isinstance(body, dict) else str(body))
        results[name] = {"status": code, "error": err}
        if code != 200:
            all_ok = False
    return {"ok": all_ok, "results": results}


def restore(client, model_key: str, snap: dict) -> dict:
    """Write back each worker's prior spill and verify. A None ``before`` means
    autofit ({}); we restore {} (the /assign convention for 'clear override').
    Returns {"ok": bool, "per_worker": {name: {before, after, matches}}}."""
    for name, s in snap.items():
        wid = s.get("worker_id")
        if not wid:
            continue
        prior = s.get("before")
        try:
            client.assign(wid, model_key, prior if prior is not None else {})
        except Exception:  # noqa: BLE001 — verification below is the real check
            pass
    # verify against a fresh read
    workers = {w.get("name"): w for w in client.workers()}
    per = {}
    ok = True
    for name, s in snap.items():
        before = s.get("before")
        after = _spill_of(workers.get(name) or {}, model_key)
        matches = _equiv(before, after)
        per[name] = {"before": before, "after": after, "matches": matches}
        if not matches:
            ok = False
    return {"ok": ok, "per_worker": per}


def _equiv(a: dict | None, b: dict | None) -> bool:
    """Two spills are equivalent if they mean the same contract. None and {}
    both mean autofit; otherwise a plain dict compare."""
    a = a or {}
    b = b or {}
    return a == b
