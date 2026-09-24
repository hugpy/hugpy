"""Post-download ADMISSION gate — the job runner (and ``hugpy-admission-seed``).

Operator ruling 2026-09-23: "upon quant download, run the grader, ideally with
its added methods of model and config validity".

A download that completes on central enqueues one admission job
(``hugpy_storage.admission.on_install_complete``, called by every path that
stamps a ``source: download`` marker). The server process runs ONE runner per
host (flock-elected, :func:`start_admission_runner`) that claims jobs and runs,
in order, each step appending to the job's log:

  a. STATIC — ``model_audit.audit_model`` for that model (files vs the install
     manifest, GGUF/safetensors structure, config fit) and the ``integrity``
     grade recorded through ``POST /llm/model-grade``. broken_download /
     faulty_model / misconfigured / not_downloaded stop here: HELD.
     suite_mismatch / mislabeled_task / unsupported (comfy): ADMITTED with a
     note — non-text, not gated by the text suite.
  b. BENCHMARK — the console's HugPy-native benchmark for THIS model only
     (``POST /llm/benchmark/run``). The single-run lock is respected: a 409
     means another run holds it, and the job WAITS behind it (bounded), it
     does not give up.
  c. COLLECT — the aptitude grade from the run's results, plus every failed
     load of the model recorded since the job started
     (``/llm/compute-actions?action=load&outcome=fail``).

Verdict: ``admitted`` = static_ok + the benchmark answered with a grade;
``held`` = a real defect only (static fail, a recorded load failure, a
benchmark row of class hard_load_failure / held); ``pending`` = the benchmark
could not reach the model for a transient or placement reason (no_lane,
timeout, unreachable, cold_load_capacity, vram_fit, refused, http_error,
error, cancelled, lock never freed). Every non-admitted record carries the
benchmark row's OWN ``reason`` + ``evidence`` + ``failure_class`` verbatim,
``elapsed_s`` of the admission attempt, and — for no_lane — ``blocked_on``.
Admission never forces placement: a model on no worker stays pending. Written to the model's
``hugpy.json["admission"]`` (atomic) — the record central's resolver reads to
refuse a held model (hugpy_fleet.central.admission_gate).

Everything goes through central's own HTTP API on localhost (the audit already
does), so the runner needs no server internals and a test needs no server.
No Hugging Face call anywhere in this module.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.parse
from typing import Any, Callable, Optional

from hugpy_ops import model_audit as ma

logger = logging.getLogger(__name__)

STATIC_FAIL = (ma.NOT_DOWNLOADED, ma.BROKEN, ma.FAULTY, ma.MISCONFIG)
NON_TEXT_OK = (ma.SUITE_MISMATCH, ma.MISLABELED, ma.UNSUPPORTED)
TEXT_SUITE = "hugpy-native-v2"
BENCHMARK_TOKENS = 128
BENCHMARK_TIMEOUT_S = float(os.environ.get("HUGPY_ADMISSION_BENCHMARK_TIMEOUT", 3 * 3600))
POLL_S = 10.0
CATALOG_ATTEMPTS = 6
RUNNER_ENV = "HUGPY_ADMISSION_RUNNER"     # "off" = never start the runner

# Benchmark row failure classes (fleet_grading.run_capacity_benchmark).
DEFECT_CLASSES = ("hard_load_failure", "held")          # -> held
NO_SUITE_CLASSES = ("no_suite",)                        # -> admitted, not text-gated
# Every other class (no_lane, timeout, unreachable, cold_load_capacity,
# vram_fit, refused, http_error, error, cancelled, ...) -> pending.
# Run states after which a run's rows are final (review_routes._BENCHMARK_ACTIVE
# = running/resuming are the live ones).
RUN_TERMINAL = ("complete", "failed", "cancelled", "interrupted")
CLASS_ORDER = ("hard_load_failure", "held", "no_lane", "vram_fit", "refused",
               "cold_load_capacity", "unreachable", "timeout", "http_error",
               "error", "cancelled")


# ── central over HTTP ────────────────────────────────────────────────────────

def operator_tokens() -> list:
    """The server's own operator token first (the runner runs inside it),
    then the operator keys the audit already knows about."""
    out = []
    t = (os.environ.get("HUGPY_OPERATOR_TOKEN") or "").strip()
    if t:
        out.append(t)
    for k in ma.api_key_candidates():
        if k not in out:
            out.append(k)
    return out


class Central:
    """The handful of central endpoints admission uses. Every method returns
    parsed JSON (or ``(status, body)`` where the status matters)."""

    def __init__(self, base: Optional[str] = None, tokens: Optional[list] = None,
                 timeout: float = ma.CENTRAL_TIMEOUT) -> None:
        self.base = ma.central_url(base)
        self.tokens = tokens if tokens is not None else operator_tokens()
        self.timeout = timeout

    def _get(self, path: str) -> Any:
        return ma.fetch_json(self.base + path, token=(self.tokens or [None])[0], timeout=self.timeout)

    def catalog(self) -> list:
        body = self._get("/api/models?verbose=1")
        return body if isinstance(body, list) else (body or {}).get("models") or []

    def workers(self) -> list:
        body = self._get("/llm/workers")
        return body if isinstance(body, list) else (body or {}).get("workers") or []

    def grade_rows(self) -> dict:
        return ma.grade_rows(self.base)

    def record_integrity(self, rep, ctx) -> Optional[dict]:
        return ma.record(rep, ctx, self.tokens)

    def benchmark_start(self, model_key: str, tokens: int = BENCHMARK_TOKENS):
        return ma._authed("POST", self.base + "/llm/benchmark/run",
                          {"executor": "hugpy-central", "tokens": tokens, "models": [model_key]},
                          self.tokens, self.timeout)

    def benchmark_status(self) -> dict:
        return self._get("/llm/benchmark/status") or {}

    def benchmark_run(self, run_id: str) -> Optional[dict]:
        """The archived run, whole (``GET /llm/benchmark/runs/<run_id>``), or
        None when central has no archive for it."""
        try:
            body = self._get("/llm/benchmark/runs/" + urllib.parse.quote(str(run_id), safe=""))
        except Exception:  # noqa: BLE001 — 404 / older central: status is the fallback
            return None
        return body if isinstance(body, dict) and body.get("run_id") == run_id else None

    def benchmark_cancel(self) -> None:
        ma._authed("POST", self.base + "/llm/benchmark/cancel", {"scope": "execution"},
                   self.tokens, self.timeout)

    def load_failures(self, model_key: str, since: float) -> list:
        q = urllib.parse.urlencode({"action": "load", "outcome": "fail", "model": model_key,
                                    "since": f"{since:.3f}", "limit": 50})
        body = self._get("/llm/compute-actions?" + q) or {}
        return body.get("actions") or []


# ── the job ──────────────────────────────────────────────────────────────────

def _same_dir(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except OSError:
        return a.rstrip("/") == b.rstrip("/")


def resolve_row(catalog: list, model_key: str, directory: Optional[str]) -> Optional[dict]:
    """The catalog row this job is about: by directory first (the install
    hook knows exactly where the bytes landed), then the key, then the
    ``owner~name`` spelling discovery mints on a name collision."""
    if directory:
        for r in catalog:
            if _same_dir(r.get("destination"), directory):
                return r
    for r in catalog:
        if r.get("model_key") == model_key:
            return r
    for r in catalog:
        mk = r.get("model_key") or ""
        if "~" in mk and mk.split("~", 1)[1] == model_key:
            return r
    return None


def _grade_of(result: dict) -> Optional[float]:
    """Percent grade of a graded row. A row the suite scored counts even when
    the lane's speed probe errored afterwards (``status: error`` with a
    score/max) — the grade is recorded data; a failure row never counts."""
    if result.get("failure_class") or result.get("status") not in ("complete", "error"):
        return None
    try:
        score, top = float(result.get("score")), float(result.get("max"))
    except (TypeError, ValueError):
        return None
    return round(100.0 * score / top, 2) if top else None


def _detail(row: dict) -> dict:
    d = row.get("detail")
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except ValueError:
            d = {"raw": d}
    return d if isinstance(d, dict) else {}


def load_failure_reason(rows: list) -> Optional[str]:
    """Per failed load: worker, class, log_ref and the loader stderr WHOLE
    (2026-09-23: no tail, no cap — the log itself, not a paraphrase)."""
    parts = []
    for r in rows:
        d = _detail(r)
        stderr = (d.get("loader_stderr") or d.get("error") or d.get("message") or "").strip("\n")
        ref = d.get("log_ref") or (f"compute_actions#{r.get('id')}" if r.get("id") is not None else None)
        body = stderr if stderr.strip() else (
            f"(loader stderr empty: read detail.loader_stderr/error/message of "
            f"{ref or 'the row'}, bytes=0)")
        cls = d.get("class") or d.get("reason") or "load_fail"
        parts.append(f"{r.get('worker_card') or r.get('worker') or '?'} [{cls}]"
                     f"{' log_ref=' + str(ref) if ref else ''} loader_stderr:\n{body}")
    return "\n".join(parts) if parts else None


def key_forms(row: dict, catalog: list, job_key: Optional[str] = None) -> set:
    """Every spelling the catalog exposes for this model (``model_key`` /
    ``id`` / ``name``), plus the bare name of an ``Owner~name`` key and the
    job's key — a bare form only when no OTHER catalog row owns it (the
    ``~`` spelling exists precisely because the bare name collided)."""
    own = {str(v) for v in (row.get("model_key"), row.get("id"), row.get("name"), job_key) if v}
    others = {str(r.get("model_key")) for r in catalog
              if r is not row and r.get("model_key") and r.get("model_key") != row.get("model_key")}
    forms = set(own)
    for k in own:
        if "~" in k:
            bare = k.split("~", 1)[1]
            if bare not in others:
                forms.add(bare)
    return forms - others


def expected_suite(row: dict) -> tuple:
    """``(suite name or None, known)`` — the suite the benchmark grades this
    catalog row with (hugpy_curation.review.suites.suite_for_model, by task).
    ``known`` is False when hugpy_curation is not importable here."""
    try:
        from hugpy_curation.review.suites import suite_for_model
    except Exception:  # noqa: BLE001
        return None, False
    chosen = suite_for_model(row)
    return (chosen.name if chosen is not None else None), True


def row_suite(result: dict) -> str:
    """A result row's suite (text rows omit ``grade_suite``)."""
    return result.get("grade_suite") or TEXT_SUITE


def failure_class_of(row: dict) -> Optional[str]:
    """A benchmark row's class; legacy rows (no ``failure_class``) fall back
    to their non-complete status (e.g. ``hardware_constraint_failed``)."""
    cls = row.get("failure_class")
    if cls:
        return str(cls)
    st = row.get("status")
    if _grade_of(row) is not None:
        return None
    if st == "complete":
        return "error" if row.get("error") not in (None, "", "N/A") else None
    return str(st) if st else None


def row_reason(row: dict) -> str:
    """The row's own recorded reason (``reason`` → ``constraint_reason`` →
    ``error``), verbatim."""
    for k in ("reason", "constraint_reason", "error"):
        v = row.get(k)
        if v not in (None, "", "N/A"):
            return str(v)
    return f"status={row.get('status')} (row recorded no reason)"


def primary_failure(rows: list) -> Optional[dict]:
    """The row that decides the verdict: defects first, then CLASS_ORDER."""
    failed = [r for r in rows if failure_class_of(r)]
    if not failed:
        return None
    rank = lambda r: (CLASS_ORDER.index(failure_class_of(r)) if failure_class_of(r) in CLASS_ORDER
                      else len(CLASS_ORDER))
    return min(failed, key=rank)


def blocked_on(cls: Optional[str], evidence: Any) -> Optional[str]:
    """For no_lane: what the lane waits on, read off the row's evidence —
    ``placement`` (no designation), ``eligible_worker`` (none eligible, or
    every designated worker ineligible), else ``lane``."""
    if cls != "no_lane":
        return None
    ev = evidence if isinstance(evidence, dict) else {}
    eligible = ev.get("eligible_workers")
    designated = ev.get("designated_workers") or []
    if eligible == []:
        return "eligible_worker"
    if not designated:
        return "placement"
    if eligible is not None and not set(designated) & set(eligible):
        return "eligible_worker"
    return "lane"


def run_error_for(error: Any, model_key: str) -> tuple:
    """``(reason, failure_class, evidence)`` for ``model_key`` from a run's
    ``error`` — the structured dict (headline + failure_classes) or a legacy
    string — verbatim; ``(None, None, None)`` when the run recorded none."""
    if error in (None, "", "N/A"):
        return None, None, None
    if not isinstance(error, dict):
        return str(error), None, None
    headline = error.get("headline")
    for cls, info in (error.get("failure_classes") or {}).items():
        if isinstance(info, dict) and model_key in (info.get("models") or []):
            reason = info.get("first_reason") or headline
            return str(reason), cls, info.get("evidence")
    counts = {k: error.get(k) for k in ("planned", "runnable", "requested_models", "elapsed_s")
              if error.get(k) is not None}
    return (str(headline) if headline else json.dumps(error, default=str)), None, counts or None


def run_job(job: dict, central: Any, *, log: Callable[[str], None],
            audit: Callable = None, finalize: Callable = None,
            sleep: Callable[[float], None] = time.sleep,
            clock: Callable[[], float] = time.time,
            benchmark_timeout: float = BENCHMARK_TIMEOUT_S,
            poll: float = POLL_S) -> dict:
    """Run one admission job end to end; returns the admission record (not
    yet written). ``audit`` / ``finalize`` default to the audit module's."""
    from hugpy_storage.admission import ADMITTED, HELD, PENDING, admission_record
    audit = audit or ma.audit_model
    finalize = finalize or ma.finalize
    key, jid = job["model_key"], job.get("id")
    t0 = clock()

    def verdict(status, reason, *, integrity=None, grade=None, **extra):
        extra.setdefault("elapsed_s", round(clock() - t0, 1))
        rec = admission_record(status, reason=reason, integrity=integrity, grade=grade,
                               job=jid, **extra)
        log(f"admission: {status} after {rec.get('elapsed_s')}s"
            + (f" [{rec['failure_class']}]" if rec.get("failure_class") else "")
            + (f" blocked_on={rec['blocked_on']}" if rec.get("blocked_on") else "")
            + f" — {reason}")
        return rec

    # (a) static audit + integrity grade
    # A just-landed file can take one registry read to appear (the comfy sweep
    # registers /checkpoints drops on the next read) — look a few times first.
    row, catalog = None, []
    for attempt in range(CATALOG_ATTEMPTS):
        catalog = central.catalog()
        row = resolve_row(catalog, key, job.get("directory"))
        if row is not None:
            break
        if attempt + 1 < CATALOG_ATTEMPTS:
            log(f"catalog: {key!r} not listed yet (attempt {attempt + 1}/{CATALOG_ATTEMPTS})")
            sleep(poll)
    if row is None:
        return verdict(HELD, f"not in central's catalog after {CATALOG_ATTEMPTS} reads "
                       f"(key {key!r}, dir {job.get('directory')!r})")
    key = row.get("model_key") or key
    workers = central.workers()
    overrides, opath = ma.load_overrides()
    ctx = ma.Context(central.base, workers, overrides, opath, central.grade_rows(),
                     {r.get("model_key") for r in catalog})
    rep = audit(row, ctx)
    finalize(rep, row, ctx)
    for line in rep.log:
        log(f"audit: {line}")
    log(f"audit verdict: {rep.verdict} — {rep.why}")
    try:
        wrote = central.record_integrity(rep, ctx)
        log(f"integrity grade recorded: {json.dumps(wrote, default=str)}")
    except Exception as exc:  # noqa: BLE001 — the verdict stands without the grade row
        log(f"integrity grade NOT recorded: {type(exc).__name__}: {exc}")
    base = {"model_key": key, "destination": row.get("destination")}
    if rep.verdict in STATIC_FAIL:
        return verdict(HELD, f"{rep.verdict}: {rep.why}", integrity=rep.verdict, **base)
    suite_name, suite_known = expected_suite(row)
    forms = key_forms(row, catalog, job["model_key"])
    log(f"suite for task {row.get('primary_task')!r}: "
        + (repr(suite_name) if suite_known else "unknown here (hugpy_curation not importable)")
        + f"; result rows matched by key forms {sorted(forms)}")
    if rep.verdict in NON_TEXT_OK and suite_known and suite_name is None:
        return verdict(ADMITTED, f"{rep.verdict}: {rep.why}; no grading suite for task "
                       f"{row.get('primary_task')!r}", integrity=rep.verdict, **base)
    if rep.verdict in NON_TEXT_OK:
        # The audit's verdict can predate the suite for this task (e.g.
        # suite_mismatch from pre-imagegen grade rows): the benchmark decides.
        log(f"audit verdict {rep.verdict} not final: the benchmark grades this task "
            f"with {suite_name or 'the suite central picks'}")

    # (b) benchmark this model only, queued behind a running benchmark
    deadline = t0 + benchmark_timeout
    run_id, waited = None, False
    while run_id is None:
        st, body = central.benchmark_start(key)
        if st in (200, 201, 202) and isinstance(body, dict):
            run_id = body.get("run_id")
            log(f"benchmark started: run {run_id} (models=[{key}], tokens={BENCHMARK_TOKENS})")
            break
        if st == 409:
            if not waited:
                other = body.get("run_id") if isinstance(body, dict) else None
                log(f"benchmark lock held by run {other} — waiting behind it (poll {poll}s)")
                waited = True
            if clock() > deadline:
                other = body.get("run_id") if isinstance(body, dict) else None
                return verdict(PENDING, f"benchmark never started: the single-run lock (run {other}) "
                               f"stayed held for {int(clock() - t0)}s", integrity=rep.verdict,
                               failure_class="timeout", evidence={"lock_run_id": other}, **base)
            sleep(poll)
            continue
        return verdict(PENDING, f"benchmark start refused: HTTP {st}: {body}",
                       integrity=rep.verdict, failure_class="http_error",
                       evidence={"http_status": st}, **base)
    status: dict = {}
    fetch_run = getattr(central, "benchmark_run", None)

    def run_state() -> dict:
        """THIS run's state: the live status while it is the current run,
        else its archive (a newer run may already own the status slot)."""
        live = central.benchmark_status() or {}
        if live.get("run_id") == run_id:
            return live
        archived = fetch_run(run_id) if fetch_run else None
        return archived or {"run_id": run_id, "status": None,
                            "_seen": f"status slot holds run {live.get('run_id')} "
                                     f"({live.get('status')}); no archive for {run_id}"}

    timed_out = False
    while True:
        status = run_state()
        if status.get("status") in RUN_TERMINAL:
            break
        if clock() > deadline:
            timed_out = True
            break
        sleep(poll)
    if timed_out:
        cancelled = False
        if (central.benchmark_status() or {}).get("run_id") == run_id:
            central.benchmark_cancel()          # free the single-run lock (bounded, as before)
            cancelled = True
        log(f"benchmark run {run_id} still {status.get('status')} after {int(clock() - t0)}s "
            f"(budget {int(benchmark_timeout)}s){' — cancelled' if cancelled else ''}")
        return verdict(PENDING, f"benchmark run {run_id} still {status.get('status') or 'unseen'} "
                       f"after {int(clock() - t0)}s (admission budget {int(benchmark_timeout)}s)"
                       + (f": {status['_seen']}" if status.get("_seen") else "")
                       + ("; run cancelled" if cancelled else ""),
                       integrity=rep.verdict, failure_class="timeout", run_id=run_id,
                       evidence={"run_status": status.get("status"), "cancelled": cancelled,
                                 "progress": status.get("progress")}, **base)
    if fetch_run:                              # the whole run, not the live slot's view
        archived = fetch_run(run_id)
        if archived and archived.get("status") in RUN_TERMINAL:
            status = archived
    run_error = status.get("error")
    log(f"benchmark run {run_id} ended: status={status.get('status')} "
        f"error={json.dumps(run_error, default=str)[:2000]}")

    # (c) collect
    all_rows = [r for r in (status.get("results") or []) if isinstance(r, dict)]
    results = [r for r in all_rows if str(r.get("model")) in forms]
    log(f"benchmark rows: {len(results)} for this model of {len(all_rows)} in run {run_id}")
    for r in results:
        log(f"benchmark result: worker={r.get('worker')} quant={r.get('quant')} status={r.get('status')} "
            f"grade={r.get('grade')} failure_class={r.get('failure_class')} reason={r.get('reason')} "
            f"evidence={json.dumps(r.get('evidence'), default=str)}")
    failures = central.load_failures(key, t0)
    for r in failures:
        log(f"load failure: {json.dumps(r, default=str)}")
    graded = [r for r in results if _grade_of(r) is not None
              and (suite_name is None or row_suite(r) == suite_name)]
    grade = max(_grade_of(r) for r in graded) if graded else None
    rows_seen = [{"failure_class": failure_class_of(r), "worker": r.get("worker"), "quant": r.get("quant"),
                  "reason": row_reason(r)} for r in results if failure_class_of(r)]
    if failures:
        return verdict(HELD, f"load failed during admission benchmark: {load_failure_reason(failures)}",
                       integrity=rep.verdict, grade=grade, failure_class="hard_load_failure",
                       evidence={"load_failures": len(failures)}, failures=rows_seen or None, **base)
    if grade is not None:
        best = max(graded, key=_grade_of)
        return verdict(ADMITTED, f"static checks passed ({rep.verdict}); benchmark grade {grade} "
                       f"({best.get('grade') or ''} {row_suite(best)}) on {best.get('worker')} "
                       f"{best.get('quant') or ''}".rstrip()
                       + (f"; run {run_id} ended {status.get('status')}" if status.get("status") != "complete" else ""),
                       integrity=rep.verdict, grade=grade, run_id=run_id, **base)
    row = primary_failure(results)
    if row is not None:
        cls, reason = failure_class_of(row), row_reason(row)
        common = dict(integrity=rep.verdict, failure_class=cls,
                      evidence=row.get("evidence") or None, failures=rows_seen or None, **base)
        if cls in NO_SUITE_CLASSES:
            return verdict(ADMITTED, f"static checks passed ({rep.verdict}); benchmark row {cls}: {reason}",
                           **common)
        if cls == "resume":
            prior = text_grades(ctx.grades).get(key)
            if prior is not None:
                return verdict(ADMITTED, f"static checks passed ({rep.verdict}); benchmark row {cls}: "
                               f"{reason}; recorded grade {prior:g}", grade=prior, **common)
        if cls in DEFECT_CLASSES:
            return verdict(HELD, reason, **common)
        return verdict(PENDING, reason, blocked_on=blocked_on(cls, row.get("evidence")), **common)
    # No row for this model: the run's own recorded error (dict or legacy
    # string). A run-level error is not a model failure: pending, never held.
    reason, cls, evidence = None, None, None
    for k in sorted(forms):
        reason, cls, evidence = run_error_for(run_error, k)
        if cls:
            break
    if reason is not None and not cls:
        reason = f"run {run_id} {status.get('status')} with no row for this model: {reason}"
    if reason is None:
        reason = (f"benchmark run {run_id} ended status={status.get('status')} with "
                  f"{len(results)} result row(s) for {key}, none graded, and no recorded error")
    return verdict(PENDING, reason, integrity=rep.verdict, failure_class=cls or "error",
                   evidence=evidence, blocked_on=blocked_on(cls, evidence), run_id=run_id, **base)


def process(job: dict, *, queue=None, central: Any = None, **kw) -> dict:
    """Claim-side wrapper: run the job, write the record, finish the row."""
    from hugpy_storage.admission import (HELD, Q_DONE, Q_FAILED, admission_queue,
                                         admission_record, write_admission)
    q = queue or admission_queue
    central = central or Central()
    lines: list = []
    started = time.time()

    def log(line: str) -> None:
        stamped = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}"
        lines.append(stamped)
        try:
            q.append_log(job["id"], stamped)
        except Exception:  # noqa: BLE001
            pass

    try:
        rec = run_job(job, central, log=log, **kw)
        status = Q_DONE
    except Exception as exc:  # noqa: BLE001 — a crashed job is HELD with the reason
        logger.exception("admission job %s crashed", job.get("id"))
        rec = admission_record(HELD, reason=f"admission job crashed: {type(exc).__name__}: {exc}",
                               job=job.get("id"), failure_class="error",
                               elapsed_s=round(time.time() - started, 1))
        import traceback as _tb
        log(f"crash: {type(exc).__name__}: {exc}\n{_tb.format_exc()}")
        status = Q_FAILED
    directory = rec.pop("destination", None) or job.get("directory")
    mk = rec.pop("model_key", None) or job["model_key"]
    path = write_admission(directory, rec, model_key=mk)
    log(f"record: {'written to ' + path if path else 'NO hugpy.json at ' + str(directory) + ' (queue row only)'}")
    q.finish(job["id"], status, result={**rec, "model_key": mk, "directory": directory}, log=None)
    return rec


# ── the runner ───────────────────────────────────────────────────────────────

_RUNNER: Optional[threading.Thread] = None


def recover_orphans(queue, owner: str) -> dict:
    """First act of a newly elected runner. The election flock is held for the
    whole life of the process that runs jobs, so a job still ``running`` now
    was left by a runner that died — in practice a central restart (package
    promotion, crash). It goes back to ``queued`` with attempt+1 and a log line
    (bounded: ``hugpy_storage.admission.MAX_ATTEMPTS``, then ``failed`` with the
    reason). Queues without ``requeue_orphaned`` fall back to the age rule."""
    try:
        if hasattr(queue, "requeue_orphaned"):
            out = queue.requeue_orphaned(f"central restarted (runner re-elected as {owner})")
            if out.get("requeued") or out.get("failed"):
                logger.warning("admission: re-queued %s, failed %s after a central restart",
                               out.get("requeued"), out.get("failed"))
            return out
        return {"requeued": queue.requeue_stale(), "failed": []}
    except Exception:  # noqa: BLE001 — recovery must never stop the runner
        logger.warning("admission: orphaned-job recovery failed", exc_info=True)
        return {"requeued": [], "failed": [], "error": True}


def _runner_loop(queue, lock_path: str, idle: float) -> None:
    owner = f"{socket.gethostname()}:{os.getpid()}"
    while True:                       # elect: one runner per host, the rest wait
        try:
            fh = open(lock_path, "a")
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:                      # who holds it, for runner_state()
                fh.seek(0)
                fh.truncate()
                fh.write(json.dumps({"owner": owner, "host": socket.gethostname(),
                                     "pid": os.getpid(), "since": time.time()}))
                fh.flush()
            except OSError:
                pass
            break
        except OSError:
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(60)
    logger.info("admission runner elected (%s, queue %s)", owner, queue.path)
    recover_orphans(queue, owner)
    while True:
        try:
            job = queue.claim_next(owner)
        except Exception:  # noqa: BLE001
            logger.warning("admission: claim failed", exc_info=True)
            job = None
        if job is None:
            time.sleep(idle)
            continue
        logger.info("admission: running job %s for %s", job["id"], job["model_key"])
        process(job, queue=queue)


def runner_lock_path(queue=None) -> str:
    from hugpy_storage.admission import admission_queue
    q = queue or admission_queue
    return os.path.join(os.path.dirname(q.path) or ".", "admission_runner.lock")


def runner_state(queue=None) -> dict:
    """Who runs admission jobs on this host: ``{elected, owner, host, pid,
    since, in_this_process}`` read off the election lock (held = a runner is
    elected; its holder wrote host/pid into it)."""
    path = runner_lock_path(queue)
    out = {"elected": False, "owner": None, "host": None, "pid": None, "since": None,
           "in_this_process": bool(_RUNNER is not None and _RUNNER.is_alive()), "lock": path}
    try:
        fh = open(path, "r")
    except OSError:
        return out
    with fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)   # nobody holds it
        except OSError:
            out["elected"] = True
        if out["elected"] or out["in_this_process"]:
            try:
                info = json.loads(fh.read() or "{}")
            except ValueError:
                info = {}
            out.update({k: info.get(k) for k in ("owner", "host", "pid", "since")})
            out["elected"] = True
    return out


def start_admission_runner(queue=None, *, idle: float = 15.0) -> bool:
    """Start the (single, per-host) admission runner thread in this process.
    Idempotent; ``HUGPY_ADMISSION_RUNNER=off`` disables it. Returns whether a
    thread is (now) running here — the elected one may be in another process."""
    global _RUNNER
    if (os.environ.get(RUNNER_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return False
    if _RUNNER is not None and _RUNNER.is_alive():
        return True
    from hugpy_storage.admission import admission_queue
    q = queue or admission_queue
    lock_path = runner_lock_path(q)
    _RUNNER = threading.Thread(target=_runner_loop, args=(q, lock_path, idle),
                               name="hugpy-admission-runner", daemon=True)
    _RUNNER.start()
    return True


# ── hugpy-admission-seed ─────────────────────────────────────────────────────

def text_grades(central_grades: dict) -> dict:
    """``{model: best text-suite grade}`` from grade rows, integrity excluded."""
    out: dict = {}
    for mk, rows in (central_grades or {}).items():
        vals = [float(r["grade"]) for r in rows
                if r.get("grade") is not None and r.get("grade_suite") != ma.SUITE]
        if vals:
            out[mk] = max(vals)
    return out


def seed_decision(model: dict, grades: dict) -> tuple:
    """``(status, reason, grade)`` for one audit-report model, or ``(None, why, None)``."""
    v, why, key = model.get("verdict"), model.get("why") or "", model["model_key"]
    if v in STATIC_FAIL:
        return "held", f"{v}: {why}", None
    if v in NON_TEXT_OK:
        return "admitted", f"{v}: non-text model, not gated by the text suite", None
    pt, fw = model.get("primary_task"), model.get("framework")
    if fw == "comfy" or (pt and pt not in ma.TEXT_TASKS):
        return "admitted", f"{v}: non-text model ({pt or fw}), not gated by the text suite", None
    g = grades.get(key)
    if g is None and "~" in key:
        g = grades.get(key.split("~", 1)[1])
    if g is None:
        return "pending", f"{v}; no aptitude grade yet — POST /llm/admission/{key}/rerun", None
    if g <= 0:
        return "held", f"{v}; graded 0 by the text suite — the grader got no answer", g
    return "admitted", f"{v}; graded {g:g}", g


def seed(report_path: str, *, apply: bool = False, force: bool = False,
         grades: Optional[dict] = None) -> dict:
    from hugpy_storage.admission import admission_record, read_admission, write_admission
    from hugpy_storage.hugpy_marker import read_hugpy_marker
    with open(report_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    models = doc.get("models") or {}
    counts: dict = {}
    rows = []
    for key, m in sorted(models.items()):
        m = {**m, "model_key": m.get("model_key") or key}
        status, reason, grade = seed_decision(m, grades or {})
        dest = m.get("destination")
        action = "write"
        if not dest or read_hugpy_marker(dest) is None:
            action = "skip: no hugpy.json"
        elif read_admission(dest) and not force:
            action = "skip: already has admission"
        counts.setdefault(status, 0)
        counts[status] += 1
        counts.setdefault("actions", {}).setdefault(action.split(":")[0], 0)
        counts["actions"][action.split(":")[0]] += 1
        rows.append({"model_key": key, "status": status, "reason": reason, "grade": grade,
                     "integrity": m.get("verdict"), "action": action})
        if apply and action == "write":
            write_admission(dest, admission_record(status, reason=reason, integrity=m.get("verdict"),
                                                   grade=grade, job="seed"), model_key=key)
    return {"counts": counts, "rows": rows, "applied": apply}


def seed_main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        prog="hugpy-admission-seed",
        description="Set hugpy.json admission for every installed model from the latest audit report + "
                    "existing text-suite grades. Dry-run unless --apply.")
    p.add_argument("--report", default=None, help="audit JSON (default: $PROJECTS_HOME/model_audit.json)")
    p.add_argument("--central", default=None, help="central URL for the grade rows (read-only GET)")
    p.add_argument("--apply", action="store_true", help="write the admission blocks")
    p.add_argument("--force", action="store_true", help="overwrite an existing admission block")
    p.add_argument("--json", action="store_true", help="print every row as JSON")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    report = args.report or os.path.join(ma.projects_home(), "model_audit.json")
    try:
        grades = text_grades(ma.grade_rows(ma.central_url(args.central)))
    except Exception as exc:  # noqa: BLE001
        print(f"hugpy-admission-seed: grade rows unavailable ({exc}); every static_ok model reads 'pending'",
              file=sys.stderr)
        grades = {}
    out = seed(report, apply=args.apply, force=args.force, grades=grades)
    if args.json:
        print(json.dumps(out, indent=1, default=str))
    else:
        for r in out["rows"]:
            print(f"{r['status']:<9} {r['model_key'][:64]:<64} {r['action']:<28} {r['reason'][:100]}")
    c = dict(out["counts"])
    acts = c.pop("actions", {})
    print(f"\n{'APPLIED' if args.apply else 'DRY-RUN'}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
          + " | actions: " + ", ".join(f"{k}={v}" for k, v in sorted(acts.items())))
    return 0


def vl_reclassify_main(argv: Optional[list] = None) -> int:
    """``hugpy-vl-reclassify`` — re-stamp GGUF rows whose dir holds an mmproj
    projector as image-text-to-text (text-generation kept in tasks). The rule
    is hugpy_marker.vl_gguf_tasks (the one the marker writer applies at
    install); this corrects the rows stamped before it. Dry-run by default."""
    p = argparse.ArgumentParser(prog="hugpy-vl-reclassify", description=vl_reclassify_main.__doc__)
    p.add_argument("--report", help="discovery report (default: $MODELS_DISCOVERY_PATH or "
                                    "$PROJECTS_HOME/model_discovery.json)")
    p.add_argument("--apply", action="store_true", help="rewrite hugpy.json + the report rows")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    report = (args.report or os.environ.get("MODELS_DISCOVERY_PATH")
              or ma._env_file_value("MODELS_DISCOVERY_PATH")
              or os.path.join(ma.projects_home(), "model_discovery.json"))
    p_central = ma.central_url(None)
    try:                                   # the task the registry SERVES (read-only GET)
        cat = ma.fetch_json(p_central + "/api/models")
        served = {r.get("model_key"): {"primary_task": r.get("primary_task"), "tasks": r.get("tasks")}
                  for r in (cat if isinstance(cat, list) else [])}
    except Exception as exc:  # noqa: BLE001
        print(f"hugpy-vl-reclassify: catalog unavailable ({exc}); judging from markers/report only",
              file=sys.stderr)
        served = {}
    from hugpy_engine.apis.reclassify import reclassify_vl_gguf
    out = reclassify_vl_gguf(apply=args.apply, discovery_path=report, served=served)
    for c in out["changed"]:
        print(f"{c['model_key']:<64} {c['from']['primary_task']} -> {c['to']['primary_task']} "
              f"tasks={c['to']['tasks']}{'  (applied)' if c['applied'] else ''}")
    print(f"\n{'APPLIED' if args.apply else 'DRY-RUN'}: {len(out['changed'])} to reclassify "
          f"(scanned {out['scanned']} gguf rows, skipped {out['skipped']}) — report {out['report_path']}"
          + ("\nthe registry re-reads the report on its next refresh (console: Discover models)"
             if args.apply and out["changed"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(seed_main())
