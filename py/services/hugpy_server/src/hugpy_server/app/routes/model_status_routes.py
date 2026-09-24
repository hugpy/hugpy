"""MODEL STATUS — "is this model worth my time?" for EVERY catalog model.

    GET /llm/models/status            one row per catalog model
        ?model=<key>                  one model (any alias form: key, owner~name
                                      tail, hub_id)
        ?worth=<label>[,<label>...]   only rows with that worth label
        ?detail=1                     + the verification log/findings/fix, every
                                      aptitude grade row with its tier detail,
                                      the serve override (meant with ?model=)

Member-readable (a GET, not in operator_auth._SENSITIVE). Never 500: a source
that faults is reported under ``sources.<name>.error`` and its fields render as
the explicit missing state, never as a blank or a fake value.

Everything the console's Models table needs is computed HERE so the UI does not
stitch five calls: the catalog (+ admission marker, blocked, unserveable), the
model audit report (``$PROJECTS_HOME/model_audit.json``), the registry grade
rows (``model_metrics``: ``integrity`` + the aptitude suites), the failed-load /
routing-refusal log (compute_actions) and the worker registry (hot/loaded,
GPU/RAM totals). The three external reads are cached for ``CACHE_TTL_S``.

ROW SHAPE (every key always present; "missing" is a labelled state)::

  model_key, framework, task, size_bytes, quant_effective (None = not a GGUF),
  verification: {verdict, why, at, source: audit|integrity-grade|admission|none, ok}
  admission:    {status: admitted|held|pending|none, reason, at, job,
                 [failure_class, evidence, blocked_on, elapsed_s, failures, run_id]}
  archived:     {marked, at, by, reason}   the operator's archive mark (hugpy.json "archive")
  grade:        {suite, score, max, value (pct), text "score/max (pct)", at, worker,
                 quant, detail_summary}                                    graded
              | {suite: <expected>, value: None, reason: "not graded"}      never graded
              | {suite: None, value: None, reason: "no grader for <task>"}  no suite
                (+ ``stray``: grades recorded under a suite that does not
                match the model's task — shown, never counted)
  last_failure: {class, first_line (a heading only), text (the WHOLE log:
                 loader stderr / message / record, verbatim), text_source,
                 bytes, log_ref, action_id, worker, ts, kind, request_id} | None
  servable:     {now, where, loaded, fits, fits_offline, provisioning:
                 [{worker, done_bytes, total_bytes, frac}], need_bytes, reason}
  worth:        {label, bucket, why, action}
  throughput:   {n_calls, mean_tok_s, p50, p90, min, max, first_at, last_at, label}
              | {n_calls: 0, mean_tok_s: None, reason: "no calls recorded"}
                (mean over EVERY recorded call: sum tokens / sum seconds)
  workers:      [{worker, state, base, label, held, failed, detail, progress}]

WORTH — deterministic, FIRST MATCHING RULE WINS:

  0. unservable  the operator marked it for archive (hugpy.json "archive";
                 why = "marked for archive by <by> at <at>: <reason>") —
                 outranks every other rule: the resolver refuses it
  1. broken      verification verdict is broken_download or faulty_model
                 (the files are bad; nothing on this fleet will serve them)
  2. held        admission status is ``held`` (the admission gate refuses it;
                 the reason is the gate's)
  3. unservable  operator-blocked, catalog-unserveable (adapter/component),
                 not installed, audit verdict misconfigured / not_downloaded,
                 or its size is known and no known worker could hold it
                 (GPU, or GPU + 80% RAM offload)
  4. unverified  no verification at all (never audited, no integrity grade,
                 no admission integrity)
  5. no-grader   verified, but no grading suite exists for its task
                 (video, ASR, TTS, embeddings, detection, ...)
  6. ungraded    verified, a suite exists, but no grade row for that suite
  7. weak        graded below ``READY_MIN_GRADE`` (default 50/100;
                 env HUGPY_WORTH_MIN_GRADE)
  8. ready       verified + graded >= READY_MIN_GRADE + fits somewhere

Buckets (the console's one-click filters): ready = {ready};
waste ("wasting your time") = {broken, held, unservable, weak};
unknown = {unverified, ungraded, no-grader}.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

from abstract_flask import get_bp
from flask import jsonify, request

model_status_bp, logger = get_bp("model_status_bp", __name__)

CACHE_TTL_S = 8.0     # audit report, grade rows, failure log
WORKERS_TTL_S = 2.0   # worker registry + evictions events (live states; the table polls <= 3 s)
READY_MIN_GRADE = float(os.environ.get("HUGPY_WORTH_MIN_GRADE") or 50)
FIT_HEADROOM = 1.15          # weights + KV/context/runtime overhead
OFFLOAD_RAM_FRACTION = 0.8   # RAM a GPU+CPU offload may reasonably use

LABELS = ("ready", "weak", "no-grader", "ungraded", "unverified",
          "unservable", "held", "broken")
BUCKETS = {"ready": ("ready",),
           "waste": ("broken", "held", "unservable", "weak"),
           "unknown": ("unverified", "ungraded", "no-grader")}
BUCKET_OF = {lab: b for b, labs in BUCKETS.items() for lab in labs}

BROKEN_VERDICTS = ("broken_download", "faulty_model")
UNSERVABLE_VERDICTS = ("misconfigured", "not_downloaded")
OK_VERDICTS = ("static_ok", "working", "suite_mismatch", "mislabeled_task", "unsupported")
INTEGRITY_SUITE = "integrity"
# Pre-v2 text suites: counted for a text model only when no current-suite row
# exists, and flagged ``legacy`` so the UI says so.
LEGACY_TEXT_SUITES = ("fleet-capacity-v1", "text-aptitude")

# Fallback when hugpy_curation is not importable (mirrors review/suites SUITES).
_FALLBACK_SUITES = {
    "text-generation": "hugpy-native-v2", "text2text-generation": "hugpy-native-v2",
    "text-summarization": "hugpy-native-v2", "image-text-to-text": "hugpy-vision-v1",
    "text-to-image": "hugpy-imagegen-v1",
}
_SUITE_MAX_TIERS = {"hugpy-native-v2": 27}

_QUANT_TOKENS = (
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m", "iq3_xxs", "iq3_xs",
    "iq3_s", "iq3_m", "iq4_xs", "iq4_nl", "q2_k", "q3_k_l", "q3_k_m", "q3_k_s",
    "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "q4_0", "q4_1",
    "q5_0", "q5_1", "bf16", "f16", "f32",
)


# ── small pure helpers ──────────────────────────────────────────────────────

def _num(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _epoch(v) -> Optional[float]:
    """Epoch seconds from an epoch number/string or an ISO-8601 string."""
    n = _num(v)
    if n is not None:
        return n / 1000.0 if n > 1e12 else n
    if isinstance(v, str) and v:
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _parse_json(v):
    if isinstance(v, (dict, list)):
        return v
    if not v:
        return None
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return None


def quant_tag(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    low = str(filename).lower()
    for q in _QUANT_TOKENS:
        if q in low:
            return q.upper()
    return re.sub(r"\.gguf$", "", str(filename), flags=re.I)


def first_line(text: Any) -> str:
    """The first meaningful line of a loader stderr / message (same rule as the
    console's modelFilters.firstLine)."""
    lines = [s.strip() for s in str(text or "").split("\n") if s.strip()]
    for ln in lines:
        if re.search(r"\berror\b|failed|E llama", ln, re.I):
            return ln
    return lines[0] if lines else ""


def model_task(row: dict) -> str:
    return (row.get("primary_task") or row.get("task")
            or next(iter(row.get("tasks") or []), None) or "unknown")


def name_forms(row: dict) -> List[str]:
    """Every spelling a metrics/audit/failure row may name this model by."""
    mk = row.get("model_key") or row.get("key") or row.get("name") or ""
    hub = row.get("hub_id") or ""
    out = [mk, mk.split("~")[-1], hub, hub.replace("/", "~"), row.get("name") or ""]
    return [f for i, f in enumerate(out) if f and f not in out[:i]]


def expected_suite(row: dict) -> Optional[str]:
    """The grading suite name for this catalog row, or None (no grader)."""
    try:
        from hugpy_curation.review.suites import suite_for_model
        s = suite_for_model(row)
        return s.name if s is not None else None
    except Exception:  # noqa: BLE001 — curation absent/broken: the mirror table
        primary = row.get("primary_task")
        if primary:
            return _FALLBACK_SUITES.get(primary)
        for t in row.get("tasks") or []:
            if t in _FALLBACK_SUITES:
                return _FALLBACK_SUITES[t]
        return None


_SUITE_MAX_CACHE: Dict[str, Optional[int]] = {}


def suite_max(name: Optional[str]) -> Optional[int]:
    """The suite's own item count (tasks x tiers) off the registry, cached;
    the legacy text suite is 9 single-prompt items."""
    if not name:
        return None
    if name in _SUITE_MAX_CACHE:
        return _SUITE_MAX_CACHE[name]
    val = None
    if name in LEGACY_TEXT_SUITES:
        val = 9
    else:
        try:
            from hugpy_curation.review.suites import SUITES_BY_NAME
            s = SUITES_BY_NAME.get(name)
            val = int(s.max) if s is not None else None
        except Exception:  # noqa: BLE001
            val = {"hugpy-native-v2": 27, "hugpy-vision-v1": 18, "hugpy-imagegen-v1": 18}.get(name)
    _SUITE_MAX_CACHE[name] = val
    return val


def tier_summary(detail) -> Optional[dict]:
    """{passed, total, tasks, tasks_full, text} over a grade_detail of the
    current shape ``{task: {tier, max, history:[{tier, pass}]}}`` (legacy
    bool/0-1 cells count as one tier each). None when there is no detail."""
    d = _parse_json(detail)
    if not isinstance(d, dict) or not d:
        return None
    passed = total = tasks = full = 0
    for cell in d.values():
        if isinstance(cell, bool) or isinstance(cell, (int, float)):
            tasks += 1
            total += 1
            ok = bool(cell)
            passed += ok
            full += ok
            continue
        if not isinstance(cell, dict):
            continue
        tasks += 1
        mx = int(_num(cell.get("max")) or 3)
        hist = [h for h in (cell.get("history") or []) if isinstance(h, dict)]
        p = sum(1 for h in hist if h.get("pass") is True) if hist else int(_num(cell.get("tier")) or 0)
        total += mx
        passed += min(p, mx)
        full += p >= mx
    if not tasks:
        return None
    return {"passed": passed, "total": total, "tasks": tasks, "tasks_full": full,
            "text": f"{passed}/{total} tiers · {full}/{tasks} tasks full"}


# ── the per-model computation (pure: data in, row out) ──────────────────────

def _verification(row, forms, audit_doc, integrity_rows, admission) -> dict:
    cands = []
    models = (audit_doc or {}).get("models") or {}
    arep = next((models[f] for f in forms if f in models), None)
    if isinstance(arep, dict) and arep.get("verdict"):
        cands.append({"verdict": arep.get("verdict"), "why": arep.get("why") or "",
                      "at": _epoch((audit_doc or {}).get("generated_at")),
                      "source": "audit", "_rep": arep})
    for r in integrity_rows:
        d = _parse_json(r.get("grade_detail")) or {}
        verdict = d.get("verdict") if isinstance(d, dict) else None
        if not verdict:
            verdict = "static_ok" if (_num(r.get("grade")) or 0) > 0 else "integrity_fail"
        cands.append({"verdict": verdict, "why": (d.get("why") if isinstance(d, dict) else "") or "",
                      "at": _epoch(r.get("graded_at")), "source": "integrity-grade",
                      "_rep": d if isinstance(d, dict) else {}})
    if cands:
        best = dict(max(cands, key=lambda c: (c["at"] or 0, c["source"] == "integrity-grade")))
        # The audit report carries findings/log/fix; keep them even when the
        # integrity row is the newer verdict source (its non-empty keys win).
        audit_rep = next((c["_rep"] for c in cands if c["source"] == "audit"), {}) or {}
        own = best.pop("_rep") or {}
        best["_rep"] = {**audit_rep, **{k: v for k, v in own.items() if v}}
        best["ok"] = best["verdict"] in OK_VERDICTS
        return best
    if admission and admission.get("integrity"):
        v = admission["integrity"]
        # A static integrity string ("suite_mismatch", "static_ok", …) is only a
        # VERIFICATION when the admission that produced it actually reached the
        # model. When that SAME admission recorded a functional ``failure_class``
        # (e.g. the cold-load was unreachable: the comfy backend refused the
        # connection) the failure is the CURRENT evidence, and a stale "verified
        # (suite_mismatch)" label would be a false positive — derive the label from
        # the failure instead. A plain ``pending`` with no failure_class (admitted,
        # integrity recorded, simply not graded yet) is still the legitimate
        # last-resort verification source it has always been.
        adm_status = str(admission.get("status") or "").lower()
        failed = bool(admission.get("failure_class"))
        if not failed:
            return {"verdict": v, "why": admission.get("reason") or "", "at": _epoch(admission.get("at")),
                    "source": "admission", "ok": v in OK_VERDICTS, "_rep": {}}
        cls = admission.get("failure_class")
        return {"verdict": None,
                "why": (f"not verified — admission {adm_status or 'attempt'} did not reach the model"
                        + (f" ({cls})" if cls else "") + ": "
                        + (admission.get("reason") or "no reason recorded")
                        + f" (static integrity was {v!r} at {_epoch(admission.get('at'))})"),
                "at": _epoch(admission.get("at")), "source": "none", "ok": False, "_rep": {}}
    return {"verdict": None, "why": "never verified: no entry in the audit report, no "
                                    f"{INTEGRITY_SUITE} grade row, no admission record for this model",
            "at": None, "source": "none", "ok": False, "_rep": {}}


ADMISSION_OPTIONAL = ("failure_class", "evidence", "blocked_on", "elapsed_s", "failures", "run_id")


def _admission(block) -> dict:
    if not isinstance(block, dict) or not block.get("status"):
        return {"status": "none", "at": None, "job": None,
                "reason": "no admission record on the catalog row "
                          f"(admission field = {block!r})"}
    out = {"status": str(block.get("status")), "reason": block.get("reason") or "",
           "at": _epoch(block.get("at")), "job": block.get("job")}
    # Optional recorded facts (hugpy_ops.admission): passed through verbatim.
    out.update({k: block[k] for k in ADMISSION_OPTIONAL if block.get(k) is not None})
    return out


def _grade(row, task, suite, aptitude_rows) -> dict:
    matching = [r for r in aptitude_rows if r.get("grade_suite") == suite] if suite else []
    legacy = False
    if suite == "hugpy-native-v2" and not matching:
        matching = [r for r in aptitude_rows if r.get("grade_suite") in LEGACY_TEXT_SUITES]
        legacy = bool(matching)
    used = {id(r) for r in matching}
    stray = [{"suite": r.get("grade_suite"), "value": _num(r.get("grade")),
              "at": _epoch(r.get("graded_at")), "worker": r.get("worker") or ""}
             for r in aptitude_rows if id(r) not in used]
    if suite is None:
        return {"suite": None, "value": None, "max": None,
                "reason": f"no grading suite is registered for task {task!r}",
                "stray": stray}
    smax = suite_max(suite)
    if not matching:
        return {"suite": suite, "value": None, "max": smax, "score": None,
                "reason": (f"no {suite} grade row in model_metrics for this model"
                           + (f" ({len(stray)} row(s) for other suites: "
                              + ", ".join(sorted({str(x['suite']) for x in stray})) + ")" if stray else
                              " (0 grade rows of any suite)")),
                "stray": stray}
    best = max(matching, key=lambda r: (_epoch(r.get("graded_at")) or 0, _num(r.get("grade")) or 0))
    summ = tier_summary(best.get("grade_detail"))
    bsuite = best.get("grade_suite")
    mx = suite_max(bsuite) or (summ or {}).get("total") or 100
    # score/max (pct): the visible PASS count over the suite's own max when
    # per-item markers exist; else derived from the stored percent (flagged).
    pct = _num(best.get("grade"))
    if summ:
        score, from_pct = summ["passed"], False
        pct = round(100.0 * score / mx, 1) if mx else pct
    else:
        score, from_pct = (round((pct or 0) * mx / 100.0) if pct is not None else None), True
    text = (f"{score}/{mx} ({round(pct or 0)}%)" + (" · from %, no per-item detail" if from_pct else "")
            if score is not None else f"{bsuite} row graded_at {best.get('graded_at')} has no grade value")
    return {"suite": bsuite, "value": pct, "max": mx, "score": score, "text": text,
            "score_from_pct": from_pct,
            "at": _epoch(best.get("graded_at")), "worker": best.get("worker") or "",
            "quant": best.get("quant") or "",
            "detail_summary": summ["text"] if summ else "no per-task detail recorded",
            "tiers": summ, "best": max(_num(r.get("grade")) or 0 for r in matching),
            "rows": len(matching), "legacy": legacy, "reason": None, "stray": stray}


def _last_failure(fails: List[dict]) -> Optional[dict]:
    if not fails:
        return None
    f = max(fails, key=lambda r: (_num(r.get("ts")) or 0, _num(r.get("id")) or 0))
    d = f.get("detail") if isinstance(f.get("detail"), dict) else (_parse_json(f.get("detail_json")) or {})
    refusal = f.get("action") == "call"
    src = next((k for k in ("loader_stderr", "message", "reason", "error") if d.get(k)), None)
    text = str(d.get(src)) if src else ""
    if not text and refusal and d:
        # a refusal's log IS its structured record
        src, text = "detail", json.dumps(d, indent=2, default=str)
    ref = d.get("log_ref") or (f"compute_actions#{f.get('id')}" if f.get("id") is not None else None)
    alloc = d.get("alloc") if isinstance(d.get("alloc"), dict) else {}
    return {"kind": "routing refusal" if refusal else "load failure",
            "benchmark": bool(alloc.get("benchmark")),
            # The routing GATE (routing_diagnostics record): ``no_worker`` means
            # routing refused BEFORE any worker was selected — a placement/capacity
            # fact (e.g. the benchmark's own unload/restore left the model unrouted
            # for a beat), never a model fault. Carried so worth() can tell a
            # model-quality refusal from a transient placement one.
            "gate": d.get("gate"),
            "class": d.get("class") or (d.get("predicate") if refusal else "") or f.get("outcome") or "",
            "first_line": first_line(text) or (f"(compute_actions row {f.get('id')} action={f.get('action')} "
                                               f"outcome={f.get('outcome')} carries no error text)"),
            # 2026-09-23: the log itself, whole; first_line is only its heading.
            "text": text,
            "text_source": (f"compute_actions#{f.get('id')} detail.{src}" if src else
                            f"compute_actions#{f.get('id')} detail (loader_stderr/message/reason/error all empty)"),
            "bytes": len(text.encode("utf-8")),
            "log_ref": ref, "action_id": f.get("id"),
            "worker": (f.get("worker_card") or d.get("worker") or "").split(":")[0],
            "ts": _num(f.get("ts")), "request_id": d.get("request_id") or f.get("request_id") or None}


def _worker_mem(w: dict) -> tuple:
    gpu = _num(w.get("gpu_total_bytes_known"))
    if gpu is None:
        gpu = sum(_num(g.get("memory_total")) or 0 for g in (w.get("gpus") or []) if isinstance(g, dict)) or None
    ram = _num(w.get("ram_total_bytes_known")) or _num(w.get("ram_total"))
    return gpu, ram


def fit_for(need: Optional[float], w: dict) -> dict:
    gpu, ram = _worker_mem(w)
    name = w.get("name") or w.get("id") or "?"
    out = {"worker": name, "online": w.get("status") == "online", "gpu_bytes": gpu,
           "ram_bytes": ram, "need_bytes": need}
    if not need:
        out["fit"] = "unknown"
        out["fit_reason"] = "model size_bytes/effective_bytes not recorded in the catalog"
    elif gpu and need * FIT_HEADROOM <= gpu:
        out["fit"] = "gpu"
    elif (gpu or ram) and need * FIT_HEADROOM <= (gpu or 0) + OFFLOAD_RAM_FRACTION * (ram or 0):
        out["fit"] = "offload"
    else:
        out["fit"] = "no"
    return out


def provisioning_on(w: dict, forms: set) -> Optional[dict]:
    """This worker's in-flight central->worker pull of the model, off its
    heartbeat (``provisioning: [keys]`` + ``provision_progress: {key:
    {done_bytes, total_bytes, frac}}``), or None."""
    progress = w.get("provision_progress") if isinstance(w.get("provision_progress"), dict) else {}
    keys = [k for k in list(w.get("provisioning") or []) + list(progress) if k in forms]
    if not keys:
        return None
    pp = progress.get(keys[0]) if isinstance(progress.get(keys[0]), dict) else {}
    done, total = _num(pp.get("done_bytes")), _num(pp.get("total_bytes"))
    frac = _num(pp.get("frac"))
    if frac is None and done is not None and total:
        frac = done / total
    return {"worker": w.get("name") or w.get("id") or "?", "done_bytes": done,
            "total_bytes": total, "frac": frac}


def provision_text(p: dict) -> str:
    """'on aeb 57% (12.4/21.6 GB)' — or '(size unknown)' when not reported."""
    gb = lambda b: f"{b / 1e9:.1f}"  # noqa: E731
    pct = f" {round(p['frac'] * 100)}%" if p.get("frac") is not None else ""
    size = (f" ({gb(p['done_bytes'] or 0)}/{gb(p['total_bytes'])} GB)" if p.get("total_bytes")
            else " (progress not reported)")
    return f"on {p['worker']}{pct}{size}"


# ── per-(model, worker) live state ──────────────────────────────────────────
#
# ONE vocabulary for every model/worker pair the console shows (Models table
# chips, model detail, Metrics picker, Workers panel):
#
#   on central                not on THIS worker's drive, but the files exist on
#                             central (catalog status installed / dir_bytes > 0):
#                             central copies them to the worker on the first call
#                             (lazy-download doctrine). NOT alarming — the files
#                             exist somewhere real (operator ruling 2026-09-10,
#                             restated 2026-09-24). For a central-served (comfy)
#                             checkpoint this branch is the ``comfy`` block's job.
#   missing                   the files are on NO drive — not this worker, not
#                             central, nowhere — AND the model is allocated
#                             (designated/assigned) to this worker: the one state
#                             that needs an operator (download it or unassign).
#   not allocated             the files are on no drive AND the model is not
#                             allocated to this worker — nothing to serve here and
#                             nothing wrong; a neutral, non-alarming state.
#   cold                      on disk (models_local / storage row), not loaded
#   downloading from central  central's transfer ledger has an active/stalled
#                             pull for the pair (authoritative: central serves
#                             the bytes); heartbeat provisioning is the fallback
#   loading                   in the worker's loading list, a slot holding it
#                             that is not healthy yet (no load error), or a
#                             load.start event with no load.done/fail after it
#   hot                       loaded (loaded_models / a healthy slot), idle
#   serving                   a slot holding it is busy, or central's in-flight
#                             relay counter for (worker, model) is > 0
#   answering                 serving AND token activity within ANSWER_WINDOW_S
#                             (slot last_used or a call row for the pair)
#
# Overlays: ``held`` (admission held: routing refuses it, whatever the state)
# and ``failed: <class>`` (the last load attempt on that worker failed; only
# over the disk-absent states cold/missing/on central/not allocated — a hot/
# loading/downloading model has since moved on).
WORKER_STATES = ("on central", "missing", "not allocated", "cold",
                 "downloading from central", "loading", "hot",
                 "serving", "answering")
ANSWER_WINDOW_S = 2.0
LOAD_EVENT_WINDOW_S = 600.0
LIVE_STATES = ("downloading from central", "loading", "serving", "answering")


def _age(ts, now) -> str:
    if ts is None:
        return "(no timestamp recorded)"
    d = max(0.0, now - ts)
    if d < 60:
        return f"{d:.0f}s"
    if d < 600:
        return f"{int(d // 60)}m{int(d) % 60:02d}s"
    if d < 5400:
        return f"{int(d // 60)}m"
    return f"{d / 3600:.1f}h"


def _err_class(text: str) -> str:
    """``LoadRefusal`` from ``"LoadRefusal: won't fit ..."``; else load_failure."""
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", str(text or ""))
    return m.group(1) if m else "load_failure"


# A probe that declined to pull files (lazy-download doctrine) is not a load
# attempt; it must not paint the pair ``failed``.
_NOT_AN_ATTEMPT = re.compile(r"^\s*not local\b|probe does not download", re.I)


# Frameworks whose serving seat is NOT a worker's model store: the checkpoint
# stays on central's store and a worker's ComfyUI backend loads it by filename
# (its CheckpointLoaderSimple list + the worker's comfy ledger of residents).
# ``models_local`` never lists them, so the generic branch read "missing" on
# every worker even with the file on central.
CENTRAL_SERVED_FRAMEWORKS = ("comfy",)


def is_central_served(model_row: dict) -> bool:
    return str((model_row or {}).get("framework") or "").lower() in CENTRAL_SERVED_FRAMEWORKS


def central_file_text(cf: Optional[dict]) -> str:
    """The central-store checkpoint fact (``catalog row["central_file"]``, stamped
    by load_catalog) in words."""
    cf = cf if isinstance(cf, dict) else {}
    if cf.get("exists"):
        return (f"checkpoint on central {cf.get('path')} ({(cf.get('bytes') or 0) / 1e9:.2f} GB, "
                f"{cf.get('bytes')} bytes)")
    if cf.get("path"):
        return f"no file on central at {cf['path']}" + (f" ({cf['error']})" if cf.get("error") else "")
    return "central checkpoint path not resolved (catalog row has no destination/filename)"


def _comfy_worker_state(model_row: dict, w: dict, forms: set, inflight: int, now: float) -> tuple:
    """``(base, detail, comfy_facts)`` for a central-served (comfy) checkpoint on
    ONE worker, from that worker's heartbeat ``comfy`` block — ``available``,
    ``url``, ``checkpoints`` (ComfyUI's CheckpointLoaderSimple list) and
    ``resident`` (the worker's comfy ledger) — plus central's file fact."""
    comfy = w.get("comfy") if isinstance(w.get("comfy"), dict) else None
    fname = (model_row or {}).get("filename")
    cf = (model_row or {}).get("central_file") if isinstance((model_row or {}).get("central_file"), dict) else {}
    central = central_file_text(cf)
    name = w.get("name") or w.get("id") or "?"
    facts = {"available": bool(comfy and comfy.get("available")), "url": (comfy or {}).get("url"),
             "version": (comfy or {}).get("version"), "listed": False, "resident": None,
             "central_file": cf or None}
    if comfy is None:
        return ("missing",
                f"worker {name} heartbeat carries no comfy block; {central}", facts)
    url = comfy.get("url") or "(no url reported)"
    if not comfy.get("available"):
        return ("missing",
                f"ComfyUI on {name} not available at {url} (heartbeat comfy.available=false); {central}", facts)
    listing = comfy.get("checkpoints") if isinstance(comfy.get("checkpoints"), list) else None
    facts["listed"] = bool(fname and listing and fname in listing)
    res = next((r for r in comfy.get("resident") or [] if isinstance(r, dict)
                and (r.get("model_key") in forms or (fname and r.get("filename") == fname))), None)
    facts["resident"] = res
    if res is not None:
        held = (f"ComfyUI {url} holds {res.get('filename') or fname} (comfy ledger"
                + (f", {(_num(res.get('bytes')) or 0) / 1e9:.2f} GB" if _num(res.get("bytes")) else "")
                + (f", loaded {_age(_num(res.get('loaded_at')), now)} ago" if _num(res.get("loaded_at")) else "")
                + ")")
        if inflight > 0:
            return "serving", f"{held}; central in-flight {inflight}", facts
        last = _num(res.get("last_used"))
        return "hot", held + " · idle" + (f" (last used {_age(last, now)} ago)" if last else ""), facts
    if inflight > 0:
        return ("serving", f"central in-flight {inflight} to ComfyUI {url}; comfy ledger has no resident "
                           f"row for {fname} yet", facts)
    if facts["listed"]:
        return ("cold", f"listed by ComfyUI {url} (CheckpointLoaderSimple, {len(listing)} checkpoints), "
                        f"not resident; {central}", facts)
    seen = (f"not in ComfyUI {url} checkpoint list ({len(listing)} listed)" if listing is not None
            else f"ComfyUI {url} reported no checkpoint list")
    return ("missing", f"{seen}; {central}", facts)


def model_worker_state(model_row: dict, worker_row: dict, slots=None, actions=(), *,
                       events=(), held: bool = False, inflight: int = 0,
                       now: Optional[float] = None, transfers=()) -> dict:
    """The live state of ONE model on ONE worker (vocabulary above).

    ``slots``: the worker's slot rows (default ``worker_row["slots"]``);
    ``actions``: compute_actions rows (load fails / calls) — filtered here to
    this model + worker; ``events``: evictions-log events (provision.* /
    load.*), filtered the same way; ``inflight``: central's relay in-flight
    count for the pair; ``transfers``: central transfer-ledger rows (the
    AUTHORITATIVE "downloading from central" — central serves the bytes; the
    worker heartbeat's provision_progress is only the fallback). Pure: no I/O."""
    now = time.time() if now is None else now
    w = worker_row or {}
    forms = set(name_forms(model_row or {}))
    name = w.get("name") or w.get("id") or "?"
    wid = w.get("id")
    online = w.get("status") == "online"
    slots = [s for s in (w.get("slots") if slots is None else slots) or []
             if isinstance(s, dict) and s.get("model_key") in forms]

    def mine(r, key="worker_card"):
        card = str(r.get(key) or r.get("worker") or "")
        return (r.get("model") in forms or r.get("model_key") in forms) and (
            card.split(":")[0] in (name, wid) or (r.get("worker_id") and r.get("worker_id") == wid))

    acts = [a for a in actions or () if isinstance(a, dict) and mine(a)]
    evs = sorted((e for e in events or () if isinstance(e, dict) and e.get("model_key") in forms
                  and (e.get("worker_id") in (wid, name))), key=lambda e: _num(e.get("ts")) or 0)

    def last_ev(*stages):
        return next((e for e in reversed(evs) if e.get("stage") in stages), None)

    # downloading from central: central's own ledger first, heartbeat fallback
    from hugpy_server.app.transfer_ledger import ledger_for
    led = next((t for t in ledger_for(w, transfers) if t.get("model_key") in forms
                and t.get("status") in ("active", "stalled")), None)
    prov = provisioning_on(w, forms)
    # loading
    loading_why = None
    if forms & set(w.get("loading") or []):
        loading_why = "in the worker's loading list"
    else:
        starting = next((s for s in slots if not s.get("healthy") and not s.get("last_load_error")), None)
        if starting is not None:
            loading_why = f"slot {starting.get('slot_id')} starting (not healthy yet)"
        else:
            ls = last_ev("load.start", "load.done", "load.fail")
            if ls and ls.get("stage") == "load.start" and now - (_num(ls.get("ts")) or 0) < LOAD_EVENT_WINDOW_S:
                loading_why = f"load.start {_age(_num(ls.get('ts')), now)} ago, no load.done yet"
    healthy = [s for s in slots if s.get("healthy")]
    loaded = bool(forms & set(w.get("loaded_models") or [])) or bool(healthy)
    busy = next((s for s in healthy if s.get("busy")), None)
    recent_use = max([_num(s.get("last_used")) or 0 for s in healthy]
                     + [_num(a.get("ts")) or 0 for a in acts if a.get("action") == "call"] or [0])
    on_disk = bool(forms & set(w.get("models_local") or []))
    disk_bytes = None
    for jr in (model_row or {}).get("workers") or []:
        if isinstance(jr, dict) and jr.get("worker") == name and _num(jr.get("on_disk_bytes")):
            on_disk, disk_bytes = True, _num(jr.get("on_disk_bytes"))
    # "on central" vs "missing" vs "not allocated" for a model NOT on this
    # worker's drive (operator ruling 2026-09-10, restated 2026-09-24). The files
    # exist on central when the catalog row is installed (or has recorded bytes);
    # the model is allocated to THIS worker when it is designated/assigned there
    # (the designation list ``models``, ``designation_meta``, the unified
    # ``allocations`` view, ``model_alloc_modes``, or a legacy ``config.pinned``).
    central_has = (str((model_row or {}).get("status") or "").lower() == "installed"
                   or (_num((model_row or {}).get("dir_bytes")) or 0) > 0)
    alloc_names = (set(w.get("models") or [])
                   | set(w.get("designation_meta") or {})
                   | set(w.get("model_alloc_modes") or {})
                   | {a.get("model_key") for a in (w.get("allocations") or [])
                      if isinstance(a, dict)}
                   | set(((w.get("config") or {}).get("pinned") or {})))
    allocated = bool(forms & alloc_names)

    progress = None
    comfy_state = (_comfy_worker_state(model_row, w, forms, inflight, now)
                   if is_central_served(model_row) else None)
    if led is not None:
        base = "downloading from central"
        pct = led.get("pct")
        size = (f"{(led.get('bytes_served') or 0) / 1e9:.1f}/{(led.get('total_bytes') or 0) / 1e9:.1f} GB"
                if led.get("total_bytes") else f"{(led.get('bytes_served') or 0) / 1e9:.1f} GB (total unknown)")
        detail = (f"{size}{f' {round(pct)}%' if pct is not None else ''} · "
                  f"{led.get('mb_per_s') if led.get('mb_per_s') is not None else '?'} MB/s · "
                  f"{_age(led.get('started_at'), now)} elapsed · files {led.get('files_done')}/{led.get('files_total')}"
                  + (f" · STALLED: no request for {led.get('idle_s')}s" if led.get("status") == "stalled" else "")
                  + " (central transfer ledger)")
        progress = {"worker": name, "done_bytes": led.get("bytes_served"), "total_bytes": led.get("total_bytes"),
                    "frac": (pct / 100.0) if pct is not None else None, "since": led.get("started_at"),
                    "mb_per_s": led.get("mb_per_s"), "files_done": led.get("files_done"),
                    "files_total": led.get("files_total"), "status": led.get("status"),
                    "last_request_at": led.get("last_request_at"), "source": "central-ledger"}
    elif prov is not None:
        base = "downloading from central"
        st = last_ev("provision.start", "provision.done", "provision.fail")
        since = _num(st.get("ts")) if st and st.get("stage") == "provision.start" else None
        pct = f" {round(prov['frac'] * 100)}%" if prov.get("frac") is not None else ""
        size = (f"{(prov['done_bytes'] or 0) / 1e9:.1f}/{prov['total_bytes'] / 1e9:.1f} GB"
                if prov.get("total_bytes") else "size not reported yet")
        detail = f"{size}{pct} · " + (f"{_age(since, now)} elapsed" if since else
                                      "elapsed unknown (no provision.start event)") + (
            " (worker heartbeat — no central transfer-ledger entry)")
        progress = {**prov, "since": since, "source": "worker-heartbeat"}
    elif comfy_state is not None:
        base, detail = comfy_state[0], comfy_state[1]
        # Same disk-truth rule as every other model: a checkpoint that exists on
        # central is "on central", never "missing" (operator 2026-09-24).
        if base == "missing":
            if (comfy_state[2].get("central_file") or {}).get("exists") or central_has:
                base = "on central"
            elif not allocated:
                base = "not allocated"
    elif loading_why:
        base, detail = "loading", loading_why
    elif busy is not None or inflight > 0:
        base = "serving"
        who = f"slot {busy.get('slot_id')} busy" if busy is not None else f"central in-flight {inflight}"
        if busy is not None and recent_use and now - recent_use <= ANSWER_WINDOW_S:
            base, detail = "answering", f"{who}; token activity {now - recent_use:.1f}s ago"
        else:
            detail = (f"{who} (request in progress); streaming vs waiting is not "
                      "distinguishable from the heartbeat")
    elif loaded:
        base = "hot"
        s0 = healthy[0] if healthy else None
        detail = (f"loaded in slot {s0.get('slot_id')}" + (f", ctx {s0.get('ctx')}" if s0.get("ctx") else "")
                  if s0 else "loaded in-process") + " · idle" + (
            f" (last used {_age(recent_use, now)} ago)" if recent_use else "")
    elif on_disk:
        base = "cold"
        detail = "on disk" + (f" ({disk_bytes / 1e9:.1f} GB)" if disk_bytes else "") + ", not loaded"
    elif central_has:
        base = "on central"
        detail = ("on central storage (llm_storage), not on this worker's drive yet — "
                  "central copies it to this worker on the first call (lazy download); "
                  "the files exist, so this is not missing")
    elif allocated:
        base = "missing"
        detail = ("allocated to this worker but the files are on NO drive — not here and "
                  "not on central storage: download the model or unassign the key")
    else:
        base = "not allocated"
        detail = ("not on this worker's drive, not on central storage, and not allocated "
                  "to this worker — nothing to serve here and nothing wrong")

    # failed: the LAST load attempt on this worker failed
    failed = None
    if base in ("cold", "missing", "on central", "not allocated"):
        cands = []
        for f in forms:
            rep = (w.get("load_reports") or {}).get(f)
            if isinstance(rep, dict) and "ok" in rep and not _NOT_AN_ATTEMPT.search(str(rep.get("error") or "")):
                _lf = rep.get("load_failure") if isinstance(rep.get("load_failure"), dict) else {}
                cands.append((_num(rep.get("ts")) or 0, bool(rep.get("ok")), _lf.get("class"),
                              rep.get("error") or "",
                              _lf.get("log_ref"), f"worker {name} load_reports[{f}].error"))
        for a in acts:
            if a.get("action") == "load":
                d = a.get("detail") if isinstance(a.get("detail"), dict) else {}
                ok = a.get("outcome") not in ("fail", "failed", "error")
                cands.append((_num(a.get("ts")) or 0, ok, d.get("class"),
                              d.get("loader_stderr") or d.get("message") or "",
                              d.get("log_ref") or f"compute_actions#{a.get('id')}",
                              f"compute_actions#{a.get('id')} detail."
                              + ("loader_stderr" if d.get("loader_stderr") else "message")))
        for sl in slots:
            if sl.get("last_load_error"):
                # the whole loader stderr + the error wording around it
                _full = sl.get("last_load_stderr_full")
                cands.append((_num(sl.get("loaded_at")) or 0, False, None,
                              (f"{sl.get('last_load_error')}\n\n{_full}" if _full
                               else sl.get("last_load_error") or ""),
                              sl.get("last_load_log_ref"),
                              f"worker {name} slot {sl.get('slot_id')} last_load_error"
                              + (" + last_load_stderr_full" if _full else "")))
        if cands:
            ts, ok, cls, text, ref, src = max(cands, key=lambda c: c[0])
            if not ok:
                text = str(text or "")
                failed = {"class": cls or _err_class(text),
                          "first_line": first_line(text) or f"({src} was empty, bytes=0)",
                          # 2026-09-23: the whole log, not only its first line
                          "text": text, "text_source": src,
                          "bytes": len(text.encode("utf-8")), "log_ref": ref,
                          "ts": ts or None}
    # A load.done / served / call newer than the recorded failure is the CURRENT
    # state; the failure stays in history (failures endpoint), not the overlay.
    if failed and failed.get("ts"):
        done = last_ev("load.done")
        succ_ts = (_num(done.get("ts")) or 0) if done else 0
        for f in forms:
            rep = (w.get("load_reports") or {}).get(f)
            if isinstance(rep, dict) and rep.get("ok"):
                succ_ts = max(succ_ts, _num(rep.get("ts")) or 0)
        for a in acts:
            if a.get("action") == "call" or (a.get("action") == "load"
                    and a.get("outcome") not in ("fail", "failed", "error")):
                succ_ts = max(succ_ts, _num(a.get("ts")) or 0)
        for sl in slots:
            if sl.get("healthy"):
                succ_ts = max(succ_ts, _num(sl.get("loaded_at")) or 0)
        succ_ts = max(succ_ts, recent_use or 0)
        if succ_ts > failed["ts"]:
            failed = None
    state = "failed" if failed else base
    label = f"failed: {failed['class']}" if failed else base
    if failed:
        detail = f"last load attempt failed {_age(failed['ts'], now)} ago: {failed['first_line']}"
    alloc = None
    if not failed and base in ("hot", "serving", "answering"):
        alloc = _alloc_state(w, forms, healthy)
        if alloc and alloc.get("label_suffix"):
            label += " · " + alloc["label_suffix"]
            detail += " · " + alloc["label_suffix"]
    if held:
        label += " · held"
        detail += " · admission HELD: routing refuses it"
    if not online:
        detail = f"worker {w.get('status') or 'offline'} — last known: {detail}"
    out = {"worker": name, "online": online, "state": state, "base": base, "label": label,
           "held": bool(held), "failed": failed, "detail": detail, "progress": progress,
           "alloc": alloc}
    if comfy_state is not None:
        out["comfy"] = comfy_state[2]
    return out


def _effective_mode(ngl, total) -> Optional[str]:
    try:
        n = int(ngl)
    except (TypeError, ValueError):
        return None
    if n == 0:
        return "ram-only"
    if n < 0 or (total and n >= int(total)):
        return "gpu"
    return "partial"


def _alloc_state(w: dict, forms: set, healthy: list) -> Optional[dict]:
    """Requested vs effective allocation of the resident seat on this worker,
    and whether it matches the model's CONFIGURED mode there (2026-09-23: a
    per-request ram-only benchmark lane left a 0/64 seat that read as plain
    "hot"). ``label_suffix`` is set only when they differ or the seat was
    loaded by a per-request override. Pure; None when nothing is resident."""
    s0 = healthy[0] if healthy else None
    row = s0
    if row is None:
        row = next((a for a in (w.get("allocations") or []) if isinstance(a, dict)
                    and a.get("model_key") in forms), None)
    if row is None:
        return None
    eff = row.get("alloc_effective") if isinstance(row.get("alloc_effective"), dict) else {
        "n_gpu_layers": row.get("n_gpu_layers"), "total_layers": row.get("total_layers"),
        "device": row.get("device")}
    src = row.get("alloc_source") if isinstance(row.get("alloc_source"), dict) else None
    modes = w.get("model_alloc_modes") or {}
    configured = next((modes[f] for f in forms if f in modes), None)
    ngl, total = eff.get("n_gpu_layers"), eff.get("total_layers")
    em = _effective_mode(ngl, total)
    mismatch = False
    if configured and em:
        if configured == "gpu-only":
            mismatch = em != "gpu"
        elif configured == "ram-only":
            mismatch = em != "ram-only"
        elif configured == "max-gpu":
            mismatch = em == "ram-only"
    by_override = bool(src and src.get("kind") == "per-request")
    suffix = None
    if mismatch or by_override:
        layers = (f"{0 if em == 'ram-only' else ngl}/{total}" if total is not None else f"n_gpu_layers={ngl}")
        what = {"ram-only": f"RAM-only {layers} GPU layers", "gpu": f"GPU {layers} layers",
                "partial": f"partial offload {layers} GPU layers"}.get(em, f"n_gpu_layers={ngl}")
        who = ""
        if by_override:
            who = f" (loaded by per-request override {src.get('request_id') or '?'}" + (
                f", {src.get('mode')}" if src.get("mode") else "") + ")"
        elif src and src.get("kind"):
            who = f" (loaded by {src.get('kind')}" + (f" {src.get('mode')}" if src.get("mode") else "") + ")"
        suffix = what + who + (f" — configured {configured}" if configured else "")
    return {"requested": row.get("alloc_requested"), "effective": eff, "source": src,
            "reload_reason": row.get("reload_reason"), "configured_mode": configured,
            "effective_mode": em, "matches_config": (not mismatch) if configured and em else None,
            "label_suffix": suffix}


def _servable(row, forms, workers, admission, unserveable_reason, wstates=()) -> dict:
    need = (_num(row.get("size_bytes")) or _num(row.get("effective_bytes")) or 0) + (_num(row.get("mmproj_bytes")) or 0)
    need = need or None
    where, cold, fits, fits_off, detail, prov = [], [], [], [], [], []
    central_ready, central_down = [], []
    by_name = {ws["worker"]: ws for ws in wstates}
    for w in workers:
        name = w.get("name") or w.get("id")
        online = w.get("status") == "online"
        f = fit_for(need, w)
        detail.append(f)
        ws = by_name.get(name) or {}
        if online and ws.get("base") in ("hot", "serving", "answering"):
            where.append(name)
        elif online and ws.get("base") == "cold":
            cold.append(name)
        elif (online and is_central_served(row) and ws.get("base") == "missing"
              and (row.get("central_file") or {}).get("exists")):
            (central_ready if (ws.get("comfy") or {}).get("available") else central_down).append(name)
        if online and ws.get("progress"):
            prov.append(ws["progress"])
        if f["fit"] in ("gpu", "offload"):
            (fits if online else fits_off).append(name)
    blocked = bool(row.get("blocked"))
    installed = (row.get("status") or "missing") == "installed"
    held = admission.get("status") == "held"
    archived = _archive_why(row)
    now = (bool(where) and not blocked and installed and not held and not unserveable_reason
           and not archived)
    if archived:
        reason = archived + " — the resolver and every placement write refuse it"
    elif blocked:
        reason = "on the operator blocklist (central blocklist.blocked_keys)"
    elif unserveable_reason:
        reason = f"not servable on its own: {unserveable_reason}"
    elif not installed:
        reason = f"not installed (status {row.get('status') or 'missing'})"
    elif held:
        reason = ("held by admission: " + (admission.get("reason") or
                  f"(admission record at {admission.get('at')} job {admission.get('job')} carries no reason)")
                  + " — the resolver refuses it")
    elif now:
        reason = "loaded on " + ", ".join(where)
    elif prov:
        reason = "provisioning " + "; ".join(provision_text(p) for p in prov)
    elif cold and is_central_served(row):
        reason = ("cold (listed by ComfyUI, not resident) on " + ", ".join(cold) + "; "
                  + central_file_text(row.get("central_file")))
    elif cold:
        reason = "cold (on disk, not loaded) on " + ", ".join(cold) + " — the first call loads it"
    elif central_ready:
        reason = (central_file_text(row.get("central_file")) + "; ComfyUI available on "
                  + ", ".join(central_ready) + " (not in its checkpoint list / not resident)")
    elif is_central_served(row):
        reason = (central_file_text(row.get("central_file")) + "; no online worker reports an available ComfyUI"
                  + (f" (unavailable on {', '.join(central_down)})" if central_down else ""))
    elif fits:
        reason = "not on any online worker's disk; would fit on " + ", ".join(fits) + " (first call pulls + loads)"
    elif fits_off:
        reason = "would fit only on offline worker(s): " + ", ".join(fits_off)
    elif need is None:
        reason = (f"not on any online worker's disk and fit cannot be computed: size_bytes/effective_bytes "
                  f"not recorded in the catalog ({len(detail)} worker(s) known)")
    else:
        big = max((d for d in detail), key=lambda d: (d["gpu_bytes"] or 0) + (d["ram_bytes"] or 0), default=None)
        reason = (f"fits no known worker: needs ~{need * FIT_HEADROOM / 2**30:.1f} GiB"
                  + (f"; largest is {big['worker']} ({(big['gpu_bytes'] or 0) / 2**30:.1f} GiB GPU + "
                     f"{(big['ram_bytes'] or 0) / 2**30:.1f} GiB RAM)" if big else "; no workers registered"))
    return {"now": now, "where": where, "cold": cold, "fits": fits, "fits_offline": fits_off,
            "provisioning": prov, "need_bytes": need, "reason": reason, "fit_detail": detail,
            **({"central": central_ready} if is_central_served(row) else {})}


def worth(row: dict) -> dict:
    """The worth label for one assembled status row (see the module docstring)."""
    v, adm, g, s = row["verification"], row["admission"], row["grade"], row["servable"]
    task = row.get("task") or "unknown"
    fix = row.get("_suggested_fix")

    def out(label, why, action):
        return {"label": label, "bucket": BUCKET_OF[label], "why": why, "action": action}

    arch = row.get("_archived")
    if arch:
        return out("unservable", arch,
                   "`hugpy-model-archive --apply` moves it to the ARCHIVE; "
                   "DELETE /llm/models/<key>/archive (↩ Unarchive) returns it to live")
    if v.get("verdict") in BROKEN_VERDICTS:
        why = v.get("why") or f"(the {v.get('source')} verdict at {v.get('at')} carries no why text)"
        return out("broken", f"{v['verdict']}: {why}",
                   fix or "re-download or drop the model")
    if adm.get("status") == "held":
        return out("held", adm.get("reason") or f"held by admission at {adm.get('at')} (job {adm.get('job')}); "
                                                 "the admission record carries no reason text",
                   fix or "fix the cause, then Verify + grade")
    unserv = row.get("_unservable")
    if unserv:
        return out("unservable", unserv, fix or "nothing on this fleet can serve it as-is")
    if v.get("source") == "none":
        return out("unverified", v.get("why") or "never verified (no audit, no integrity grade, no admission)",
                   "Verify + grade")
    if g.get("suite") is None:
        why = g.get("reason") or f"no grading suite is registered for task {task!r}"
        return out("no-grader", f"verified ({v.get('verdict')}); {why}",
                   "judge it by hand; no automated grader")
    if g.get("value") is None:
        why = g.get("reason") or f"no {g['suite']} grade row"
        return out("ungraded", f"verified ({v.get('verdict')}); {why}",
                   "Verify + grade")
    if g["value"] < READY_MIN_GRADE:
        return out("weak", f"graded {g.get('text') or g['value']} by {g['suite']} (< {READY_MIN_GRADE:g}%)"
                   + (f" — {g['detail_summary']}" if g.get("detail_summary") else ""),
                   "pick a better model or quant; re-grade after a fix")
    note = ""
    lf = row.get("last_failure")
    # A post-grade refusal only cautions the worth when it reflects the MODEL. A
    # benchmark's own call (``alloc.benchmark``) and a ``no_worker``-gate refusal
    # (routing refused before any worker was selected — a placement/capacity
    # transient, e.g. the benchmark's own unload/restore left the model unrouted
    # for a beat) are harness/placement facts, not model faults, so neither taints.
    # The worker is always named; when the refusal was unrouted the recorded gate
    # stands in for it — never the bare placeholder ``?``.
    if (lf and (lf.get("ts") or 0) > (g.get("at") or 0)
            and not lf.get("benchmark") and lf.get("gate") != "no_worker"):
        on = lf.get("worker") or (f"gate={lf['gate']}" if lf.get("gate") else "no worker recorded")
        note = f"; note: {lf['kind']} on {on} after the grade"
    where = ("hot on " + ", ".join(s["where"])) if s.get("now") else (
        "fits " + ", ".join(s["fits"]) if s.get("fits") else
        f"not hot: {s.get('reason')}" if s.get("cold") or s.get("central") else
        f"not hot, fits no online worker: {s.get('reason')}")
    return out("ready", f"verified ({v.get('verdict')}), graded {g.get('text') or g['value']} by {g['suite']}, {where}{note}",
               "use it" if s.get("now") else "call it (first call cold-loads)")


def _archive_why(row) -> Optional[str]:
    """``marked for archive by <by> at <at>: <reason>`` off the catalog row's
    ``archived`` projection, or None when not marked."""
    a = row.get("archived")
    if not isinstance(a, dict) or not a.get("marked"):
        return None
    try:
        from hugpy_storage.archive_mark import archive_text
        return archive_text(a)
    except Exception:  # noqa: BLE001 — same text, built here if the module is absent
        return (f"marked for archive by {a.get('by')} at {a.get('at')}"
                + (f": {a.get('reason')}" if a.get("reason") else
                   " (no reason was given when it was marked)"))


def _unservable_reason(row, verification, servable, unserveable_reason) -> Optional[str]:
    arch = _archive_why(row)
    if arch:
        return arch
    if row.get("blocked"):
        return "blocked by the operator"
    if unserveable_reason:
        return f"not servable on its own: {unserveable_reason}"
    status = row.get("status") or "missing"
    if status != "installed":
        return f"not installed (status {status})"
    if verification.get("verdict") in UNSERVABLE_VERDICTS:
        return f"{verification['verdict']}: {verification.get('why') or 'no detail'}"
    if (servable.get("need_bytes") and not servable["fits"] and not servable["fits_offline"]
            and not servable["where"] and not servable["cold"]):
        return servable["reason"]
    return None


def build_status_rows(catalog: Iterable[dict], *, workers: List[dict], metrics_rows: List[dict],
                      audit_doc: Optional[dict], failures: List[dict], detail: bool = False,
                      override_of: Optional[Callable[[str], Any]] = None, events=(), calls=(),
                      inflight_of: Optional[Callable[[str, str], int]] = None,
                      now: Optional[float] = None, throughput_stats=(), transfers=(),
                      throughput_error: Optional[str] = None) -> List[dict]:
    """One status row per catalog row. Pure: every input is plain data."""
    catalog = [c for c in catalog if isinstance(c, dict)]
    form_to_key: Dict[str, str] = {}
    for c in catalog:
        mk = c.get("model_key") or c.get("key") or c.get("name")
        if mk:
            form_to_key[mk] = mk
    for c in catalog:
        mk = c.get("model_key") or c.get("key") or c.get("name")
        for f in name_forms(c):
            form_to_key.setdefault(f, mk)
    integ: Dict[str, list] = {}
    apt: Dict[str, list] = {}
    for r in metrics_rows or []:
        if not isinstance(r, dict) or _num(r.get("grade")) is None:
            continue
        mk = form_to_key.get(r.get("model_name") or "")
        if not mk:
            continue
        (integ if r.get("grade_suite") == INTEGRITY_SUITE else apt).setdefault(mk, []).append(r)
    fails: Dict[str, list] = {}
    for f in failures or []:
        mk = form_to_key.get((f or {}).get("model") or "")
        if mk:
            fails.setdefault(mk, []).append(f)
    acts: Dict[str, list] = {mk: list(v) for mk, v in fails.items()}
    for a in calls or []:
        mk = form_to_key.get((a or {}).get("model") or "")
        if mk:
            acts.setdefault(mk, []).append(a)
    evs: Dict[str, list] = {}
    for e in events or []:
        mk = form_to_key.get((e or {}).get("model_key") or "")
        if mk:
            evs.setdefault(mk, []).append(e)
    now = time.time() if now is None else now
    tp_model: Dict[str, dict] = {}
    tp_cells: Dict[str, list] = {}
    for st in throughput_stats or []:
        mk = form_to_key.get((st or {}).get("model_name") or "")
        if not mk:
            continue
        if st.get("worker") is None:
            prev = tp_model.get(mk)
            if prev is None or (st.get("n_calls") or 0) > (prev.get("n_calls") or 0):
                tp_model[mk] = st
        else:
            tp_cells.setdefault(mk, []).append(st)

    out = []
    for c in catalog:
        mk = c.get("model_key") or c.get("key") or c.get("name")
        if not mk:
            continue
        forms = name_forms(c)
        task = model_task(c)
        extra = c.get("extra") if isinstance(c.get("extra"), dict) else {}
        serveable = c.get("serveable", extra.get("serveable", True))
        unserveable_reason = None
        if serveable is False:
            unserveable_reason = (c.get("unserveable_reason") or extra.get("unserveable_reason")
                                  or "adapter / pipeline component")
        admission = _admission(c.get("admission"))
        ver = _verification(c, forms, audit_doc, integ.get(mk, []), c.get("admission"))
        rep = ver.pop("_rep", {}) or {}
        suite = expected_suite(c)
        grade = _grade(c, task, suite, apt.get(mk, []))
        held = admission.get("status") == "held"
        wstates = [model_worker_state(c, w, None, acts.get(mk, ()), events=evs.get(mk, ()), held=held,
                                      inflight=_safe_inflight(inflight_of, w, mk), now=now,
                                      transfers=transfers)
                   for w in workers or []]
        serv = _servable(c, forms, workers or [], admission, unserveable_reason, wstates)
        row = {
            "model_key": mk, "hub_id": c.get("hub_id") or "", "framework": c.get("framework") or "",
            "task": task, "tasks": list(c.get("tasks") or []), "status": c.get("status") or "missing",
            "size_bytes": _num(c.get("size_bytes")) or _num(c.get("effective_bytes")),
            "quant_effective": quant_tag(c.get("effective_gguf")),
            "blocked": bool(c.get("blocked")),
            "archived": (c.get("archived") if isinstance(c.get("archived"), dict)
                         else {"marked": False, "at": None, "by": None, "reason": None}),
            "verification": ver, "admission": admission, "grade": grade,
            "last_failure": _last_failure(fails.get(mk, [])),
            "servable": {k: v for k, v in serv.items() if k != "fit_detail"},
            "workers": wstates,
            "suggested_fix": rep.get("suggested_fix") or None,
            "throughput": _throughput(tp_model.get(mk), model=mk, err=throughput_error),
        }
        row["_suggested_fix"] = row["suggested_fix"]
        row["_unservable"] = _unservable_reason(c, ver, serv, unserveable_reason)
        row["_archived"] = _archive_why(c)
        row["worth"] = worth(row)
        # A model in the "wasting your time" bucket is not expected on any
        # worker, so the disk-absent states (missing / on central / not
        # allocated) are not a gap worth flagging (operator 2026-09-24). Say why
        # it is n/a instead. A failed/held overlay still wins — that IS worth
        # showing.
        if row["worth"]["bucket"] == "waste":
            for ws in wstates:
                if (ws.get("base") in ("missing", "on central", "not allocated")
                        and not ws.get("failed") and not ws.get("held")):
                    ws.update(state="n/a", base="n/a", label="n/a",
                              detail=f"not expected on workers — {row['worth']['label']}: "
                                     f"{row['worth']['why']} ({ws.get('detail') or 'not on this worker'})")
        row.pop("_suggested_fix")
        row.pop("_unservable")
        row.pop("_archived")
        if detail:
            row["detail"] = {
                "verification": {"log": list(rep.get("log") or []), "findings": list(rep.get("findings") or []),
                                 "suggested_fix": rep.get("suggested_fix"), "eliminate": rep.get("eliminate"),
                                 "detected_task": rep.get("detected_task"), "tags": rep.get("tags") or []},
                "grade_rows": [{**{k: r.get(k) for k in ("grade_suite", "grade", "worker", "quant",
                                                         "alloc_mode", "tok_per_s", "cold_load_s")},
                                "graded_at": _epoch(r.get("graded_at")),
                                "grade_detail": _parse_json(r.get("grade_detail"))}
                               for r in sorted(apt.get(mk, []), key=lambda r: -(_epoch(r.get("graded_at")) or 0))],
                "fit": serv["fit_detail"],
                "override": (override_of(mk) if override_of else None),
                "throughput_cells": [{**_throughput(st), "worker": st.get("worker"), "quant": st.get("quant"),
                                      "alloc_mode": st.get("alloc_mode")}
                                     for st in sorted(tp_cells.get(mk, []), key=lambda x: -(x.get("n_calls") or 0))],
                "fit_headroom": FIT_HEADROOM, "offload_ram_fraction": OFFLOAD_RAM_FRACTION,
            }
        out.append(row)
    return out


def _throughput(st: Optional[dict], model: Optional[str] = None, worker: Optional[str] = None,
                err: Optional[str] = None) -> dict:
    """The model's tok/s = MEAN over every recorded call (sum tokens / sum
    seconds) with n and spread; the explicit no-calls state otherwise. Never
    an EMA, never the last call."""
    if not st or not st.get("n_calls"):
        if err:
            return {"n_calls": 0, "mean_tok_s": None, "error": err,
                    "reason": f"call ledger (model_calls) read failed: {err}"}
        scope = " on ".join(x for x in (str(model or "this model"), str(worker or "")) if x)
        return {"n_calls": 0, "mean_tok_s": None,
                "reason": f"no calls recorded in model_calls for {scope}"}
    # One shape everywhere (metrics_routes._tp): Σtok/Σgen-s with n / n_rated,
    # spread, basis counts, the unstamped bucket and by_worker, and — when no
    # mean — the reason from the rows' own counts.
    from hugpy_server.app.routes.metrics_routes import _tp
    return _tp(st)


def _safe_inflight(fn, w, mk) -> int:
    if fn is None:
        return 0
    try:
        return int(fn(w.get("id") or w.get("name") or "", mk) or 0)
    except Exception:  # noqa: BLE001
        return 0


def summarize(rows: List[dict]) -> dict:
    counts = {lab: 0 for lab in LABELS}
    for r in rows:
        counts[r["worth"]["label"]] = counts.get(r["worth"]["label"], 0) + 1
    buckets = {b: sum(counts.get(lab, 0) for lab in labs) for b, labs in BUCKETS.items()}
    return {"counts": counts, "buckets": buckets}


# ── cached source reads ─────────────────────────────────────────────────────

_cache_lock = threading.Lock()
_cache: Dict[str, tuple] = {}


def _cached(name: str, loader: Callable[[], Any], ttl: Optional[float] = None):
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(name)
        if hit and now - hit[0] < (CACHE_TTL_S if ttl is None else ttl):
            return hit[1]
    val = loader()
    with _cache_lock:
        _cache[name] = (time.monotonic(), val)
    return val


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _projects_home() -> str:
    env = os.environ.get("PROJECTS_HOME")
    if env:
        return env
    try:
        from hugpy_platform.constants import PROJECTS_HOME
        return str(PROJECTS_HOME)
    except Exception:  # noqa: BLE001
        return "/mnt/16T_toshiba/llm_storage/projects"


def audit_path() -> str:
    return os.environ.get("HUGPY_MODEL_AUDIT_PATH") or os.path.join(_projects_home(), "model_audit.json")


def load_audit() -> tuple:
    path = audit_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc, {"path": path, "generated_at": _epoch(doc.get("generated_at")),
                     "models": len(doc.get("models") or {}), "error": None}
    except FileNotFoundError:
        return None, {"path": path, "error": f"no audit report file at {path} (FileNotFoundError)"}
    except (OSError, ValueError) as exc:
        return None, {"path": path, "error": f"{type(exc).__name__}: {exc}"}


def load_grade_rows() -> tuple:
    try:
        from hugpy_engine.model_index.client import enabled
        if not enabled():
            return [], {"error": "registry DB disabled (HUGPY_REGISTRY_DB != pg)"}
        from hugpy_server.app.routes.metrics_routes import _live_db
        with _live_db().cursor() as cur:
            cur.execute("""SELECT model_name, quant, alloc_mode, worker, tok_per_s, cold_load_s,
                                  grade, grade_suite, grade_detail, extract(epoch from graded_at)
                           FROM model_metrics WHERE grade IS NOT NULL
                           ORDER BY graded_at DESC NULLS LAST LIMIT %s""", (5000,))
            cols = ("model_name", "quant", "alloc_mode", "worker", "tok_per_s", "cold_load_s",
                    "grade", "grade_suite", "grade_detail", "graded_at")
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        return rows, {"rows": len(rows), "error": None}
    except Exception as exc:  # noqa: BLE001 — a read surface must not 500
        logger.warning("model status: grade rows unavailable: %s", exc)
        return [], {"error": f"{type(exc).__name__}: {exc}"}


def load_failures() -> tuple:
    try:
        from hugpy_fleet.central import model_metrics as _mm
        loads = _mm.recent_actions(_mm.model_metrics_store, limit=2000, action="load", outcome="fail")
        refusals = _mm.recent_actions(_mm.model_metrics_store, limit=500, action="call", outcome="refused")
        rows = list(loads or []) + list(refusals or [])
        return rows, {"rows": len(rows), "error": None}
    except Exception as exc:  # noqa: BLE001
        logger.warning("model status: failure log unavailable: %s", exc)
        return [], {"error": f"{type(exc).__name__}: {exc}"}


def load_workers() -> tuple:
    try:
        from hugpy_fleet.central.workers import list_workers
        ws = list(list_workers() or [])
        return ws, {"workers": len(ws), "error": None}
    except Exception as exc:  # noqa: BLE001
        return [], {"error": f"{type(exc).__name__}: {exc}"}


def load_events() -> tuple:
    """provision.* / load.* events from the evictions log (last hour)."""
    try:
        from hugpy_fleet.central import evictions as evictions_mod
        evs = evictions_mod.get_store().recent(limit=2000, since_ts=time.time() - 3600)
        keep = ("provision.start", "provision.done", "provision.fail", "load.start", "load.done", "load.fail")
        evs = [e for e in evs or [] if isinstance(e, dict) and e.get("stage") in keep]
        return evs, {"rows": len(evs), "error": None}
    except Exception as exc:  # noqa: BLE001
        return [], {"error": f"{type(exc).__name__}: {exc}"}


def load_calls() -> tuple:
    """Call rows of the last few seconds (the ``answering`` signal)."""
    try:
        from hugpy_fleet.central import model_metrics as _mm
        rows = _mm.recent_actions(_mm.model_metrics_store, limit=500, action="call",
                                  since_ts=time.time() - 10)
        return list(rows or []), {"rows": len(rows or []), "error": None}
    except Exception as exc:  # noqa: BLE001
        return [], {"error": f"{type(exc).__name__}: {exc}"}


def load_transfers() -> tuple:
    try:
        from hugpy_server.app.transfer_ledger import ledger
        rows = ledger.snapshot(include_done=False)
        return rows, {"rows": len(rows), "error": None}
    except Exception as exc:  # noqa: BLE001
        return [], {"error": f"{type(exc).__name__}: {exc}"}


@model_status_bp.route("/llm/transfers", methods=["GET"])
def transfers_route():
    """Central's own accounting of every central->worker weight transfer:
    active / stalled first, then the bounded history (complete / aborted).
    ``worker`` is resolved from the puller's id to its registry name."""
    from hugpy_server.app.transfer_ledger import ledger, ledger_for
    rows = ledger.snapshot(include_done=True)
    workers, _src = load_workers()
    for w in workers:
        for t in ledger_for(w, rows):
            t["worker"] = w.get("name") or w.get("id")
    for t in rows:
        t.setdefault("worker", t.get("worker_id"))
    active = [t for t in rows if t.get("status") in ("active", "stalled")]
    return jsonify({"transfers": rows, "active": len(active), "generated_at": time.time(),
                    "stall_s": ledger.stall_s, "abort_s": ledger.abort_s})


def load_throughput() -> tuple:
    try:
        from hugpy_server.app.routes.metrics_routes import call_stats, call_stats_error
        st = call_stats("")
        return st, {"rows": len(st), "error": call_stats_error("")}
    except Exception as exc:  # noqa: BLE001
        return [], {"error": f"{type(exc).__name__}: {exc}"}


def _inflight_of(worker_id: str, model_key: str) -> int:
    """Central's relay in-flight count for (worker, model) — THIS API process's
    counter (gunicorn workers each hold their own), so a positive is real and a
    zero may miss a call relayed by a sibling process."""
    from hugpy_engine.resolvers.remote import _inflight_count
    return _inflight_count(worker_id, model_key)


def load_catalog() -> List[dict]:
    """The catalog rows as /models?verbose=1 stamps them (status, sizes,
    blocked, admission marker) — without the per-worker join."""
    from hugpy_engine.config.models.models_config import get_models_dict
    from hugpy_storage.console.cancelable_downloads import update_model_sizes, update_model_status
    from hugpy_server.app.routes.llm_storage_routes import _admission_of, _archived_of
    try:
        from hugpy_fleet.central.blocklist import blocked_keys
        blocked = blocked_keys()
    except Exception:  # noqa: BLE001
        blocked = set()
    out = []
    for key, model in get_models_dict(dict_return=True).items():
        m = dict(update_model_status(model))
        mk = m.get("model_key") or key
        m["model_key"] = mk
        try:
            update_model_sizes(m, mk)
        except Exception:  # noqa: BLE001 — sizes stay unknown (rendered as such)
            pass
        if is_central_served(m):
            m["central_file"] = _central_file(m)
        m["blocked"] = mk in blocked or key in blocked
        m["admission"] = _admission_of(m)
        m["archived"] = _archived_of(m, mk)
        out.append(m)
    return out


def _central_file(m: dict) -> dict:
    """``{path, exists, bytes, real_path, error?}`` for a central-served row's
    checkpoint on central's store (destination/filename; symlinks followed)."""
    dest = m.get("destination") or m.get("dir")
    fname = m.get("filename")
    if not dest or not fname:
        return {"path": None, "exists": False, "bytes": None,
                "error": f"destination={dest!r} filename={fname!r}"}
    path = os.path.join(dest, fname)
    try:
        st = os.stat(path)
        return {"path": path, "exists": True, "bytes": int(st.st_size), "real_path": os.path.realpath(path)}
    except OSError as exc:
        return {"path": path, "exists": False, "bytes": None, "error": f"{type(exc).__name__}: {exc.strerror}"}


def _override_of(mk: str):
    try:
        from hugpy_engine.serve.overrides import get_override
        return get_override(mk) or None
    except Exception:  # noqa: BLE001
        return None


@model_status_bp.route("/llm/models/status", methods=["GET"])
def models_status():
    """One "worth my time" row per catalog model (see the module docstring)."""
    want = (request.args.get("model") or "").strip()
    labels = {s.strip() for s in (request.args.get("worth") or "").split(",") if s.strip()}
    detail = request.args.get("detail") in ("1", "true", "yes")
    sources: Dict[str, Any] = {}
    try:
        catalog = load_catalog()
        sources["catalog"] = {"models": len(catalog), "error": None}
    except Exception as exc:  # noqa: BLE001
        logger.warning("model status: catalog unavailable: %s", exc, exc_info=True)
        catalog, sources["catalog"] = [], {"error": f"{type(exc).__name__}: {exc}"}
    if want:
        catalog = [c for c in catalog if want in name_forms(c)]
    audit_doc, sources["audit"] = _cached("audit", load_audit)
    grade_rows, sources["metrics"] = _cached("metrics", load_grade_rows)
    failures, sources["failures"] = _cached("failures", load_failures)
    workers, sources["workers"] = _cached("workers", load_workers, WORKERS_TTL_S)
    events, sources["events"] = _cached("events", load_events, WORKERS_TTL_S)
    calls, sources["calls"] = _cached("calls", load_calls, 1.0)
    tstats, sources["throughput"] = _cached("throughput", load_throughput)
    transfers, sources["transfers"] = load_transfers()
    rows = build_status_rows(catalog, workers=workers, metrics_rows=grade_rows, audit_doc=audit_doc,
                             failures=failures, detail=detail, override_of=_override_of if detail else None,
                             events=events, calls=calls, inflight_of=_inflight_of,
                             throughput_stats=tstats, transfers=transfers,
                             throughput_error=(sources.get("throughput") or {}).get("error"))
    summary = summarize(rows)
    live = any(ws["base"] in LIVE_STATES for r in rows for ws in r["workers"])
    if labels:
        rows = [r for r in rows if r["worth"]["label"] in labels or r["worth"]["bucket"] in labels]
    return jsonify({"models": rows, "count": len(rows), **summary, "sources": sources,
                    "live": live, "poll_s": 3 if live else 15, "worker_states": list(WORKER_STATES),
                    "rules": {"ready_min_grade": READY_MIN_GRADE, "labels": list(LABELS),
                              "buckets": {b: list(v) for b, v in BUCKETS.items()},
                              "fit_headroom": FIT_HEADROOM, "offload_ram_fraction": OFFLOAD_RAM_FRACTION},
                    "generated_at": time.time()})
