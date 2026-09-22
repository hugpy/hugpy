"""Thin, dependency-free HTTP seam onto the hugpy central (dev.hugpy.ai API).

Stdlib only (urllib) — the runner must not add a dependency (sklearn etc. is the
learner's concern). This is the single object the runner/alloc/observe modules
talk to, and the single object tests substitute with an in-memory fake, so all
the network lives here and nowhere else.

Only ONE method mutates fleet state: ``assign`` (operator-gated). Every other
call is a read. ``chat_stream`` fires the keyless ``web`` transport — the same
public path the console uses — and returns a terminal record (never raises for
an in-band model error; it captures it verbatim)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from hugpy_platform.central import DEFAULT_CENTRAL, central_base_url

DEFAULT_BASE = DEFAULT_CENTRAL


def default_base_url() -> str:
    """The central to exercise: ``HUGPY_BASE_URL`` (and its legacy aliases)
    via the platform resolver, else loopback central."""
    return central_base_url(DEFAULT_BASE)


class CentralClient:
    def __init__(self, base_url: str = DEFAULT_BASE,
                 operator_token: str | None = None, timeout: float = 12.0):
        self.base = base_url.rstrip("/")
        self.operator_token = (operator_token or "").strip() or None
        self.timeout = timeout

    # ── low-level ────────────────────────────────────────────────────────────
    def _get(self, path: str, params: dict | None = None,
             timeout: float | None = None):
        url = self.base + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url += "?" + urllib.parse.urlencode(clean)
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))

    def _post(self, path: str, body: dict, operator: bool = False,
              timeout: float | None = None):
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if operator:
            if not self.operator_token:
                raise RuntimeError(
                    "operator token required for %s but none configured" % path)
            headers["X-Operator-Token"] = self.operator_token
        req = urllib.request.Request(self.base + path, data=data,
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                raw = r.read().decode("utf-8", "replace")
                try:
                    return r.status, json.loads(raw)
                except Exception:
                    return r.status, {"_raw": raw}
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(raw)
            except Exception:
                return e.code, {"_raw": raw}

    # ── reads ────────────────────────────────────────────────────────────────
    def health(self) -> int:
        try:
            code, _ = self._get("/health", timeout=6)
            return code
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:
            return 0

    def models(self) -> list[dict]:
        _, d = self._get("/models")
        return d if isinstance(d, list) else (d.get("models") or [])

    def workers(self) -> list[dict]:
        _, d = self._get("/llm/workers")
        return d if isinstance(d, list) else []

    def jobs(self) -> dict:
        _, d = self._get("/llm/jobs")
        return d if isinstance(d, dict) else {"jobs": d, "counts": {}}

    def reservations(self, include_terminal: bool = False) -> list[dict]:
        """GET /llm/reservations — active GPU-reservation claims (the heavy video
        runs that pre-claimed a card). Read-only; never 5xxes (degrades to []).
        The sweep backs off a card that has an active claim on it."""
        try:
            _, d = self._get("/llm/reservations",
                             {"all": 1 if include_terminal else None})
            return (d.get("reservations") or []) if isinstance(d, dict) else []
        except Exception:  # noqa: BLE001 — a read hiccup must not crash the sweep
            return []

    def model_meta(self, model_key: str, vram_gib: float | None = None,
                   ctx_pct: int | None = None) -> dict:
        try:
            _, d = self._get(f"/models/{urllib.parse.quote(model_key, safe='~')}/meta",
                             {"vram_gib": vram_gib, "ctx_pct": ctx_pct})
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    # ── the ONLY mutation ────────────────────────────────────────────────────
    def assign(self, worker_id: str, model_key: str, spill: dict) -> tuple[int, dict]:
        """POST /assign — operator-gated central registry write to
        spill_by_model[model_key]. Returns (status_code, body). A non-200 (e.g.
        409 engine-gate refusal) is returned, never raised."""
        return self._post(f"/llm/workers/{worker_id}/assign",
                          {"model_key": model_key, "spill": spill},
                          operator=True)

    def unload(self, worker_id: str, model_key: str) -> tuple[int, dict]:
        """POST /llm/workers/<id>/unload {"model_key": ...} — force a model COLD
        (drop it from the worker's live residency so the next load re-reads the
        spill). Operator-gated. The registry/designation is UNTOUCHED — this only
        evicts the live cache, so a subsequent generation reloads under the new
        spill. Returns (status, body); a non-200 is returned, never raised."""
        return self._post(f"/llm/workers/{worker_id}/unload",
                          {"model_key": model_key}, operator=True)

    def cancel(self, request_id: str) -> None:
        """Best-effort cancel a chat (polite-guest on a timeout / stop)."""
        try:
            self._post(f"/llm/chat/cancel/{urllib.parse.quote(request_id)}", {},
                       timeout=6)
        except Exception:  # noqa: BLE001
            pass

    # ── the generation fire (keyless web transport) ──────────────────────────
    def chat_stream(self, model_key: str, prompt: str, request_id: str,
                    max_new_tokens: int, ceiling_s: float,
                    read_timeout_s: float | None = None,
                    unbounded: bool | None = None,
                    max_collect_tokens: int | None = None) -> dict:
        """Fire one /chat/stream and drain the SSE. Returns a terminal record:
        {outcome, served_worker, error, finish_reason, ttft_s, load_duration_s,
         wall_s, tokens, stages}. Captures an in-band model error verbatim
         rather than raising.

        Bounds TOTAL wall time by ``ceiling_s`` (a broken model that streams
        'awaiting-load' progress forever would otherwise keep the SSE open
        indefinitely — the per-socket-read timeout never fires while data flows).
        A single silent read is bounded by ``read_timeout_s``. On a wall-clock
        timeout the job is cancelled."""
        read_timeout = read_timeout_s or min(45.0, ceiling_s)
        payload = {
            "model_key": model_key, "prompt": prompt, "request_id": request_id,
            "transport": "web", "max_new_tokens": max_new_tokens,
        }
        # NOTE (measured, k7): NEITHER max_new_tokens NOR unbounded=False actually
        # hard-caps total output on the web transport — the engine auto-continues
        # (chunk-by-chunk re-prompt) until natural stop or the wall ceiling, and
        # those per-chunk re-prefill gaps (~0.8s) wreck a naive whole-span tok/s.
        # So the sweep (a) bounds each fire CLIENT-SIDE via max_collect_tokens, and
        # (b) reports decode speed as 1/median(inter-token interval) — robust to
        # the continuation gaps and to ttft. unbounded/max_new_tokens are still
        # sent as hints.
        if unbounded is not None:
            payload["unbounded"] = bool(unbounded)
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.base + "/chat/stream", data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        worker = None
        outcome = None
        error = None
        finish_reason = None
        first_token_t = None
        last_token_t = None
        first_load_t = None
        tokens = 0
        token_times: list[float] = []
        collected_enough = False
        stages: list[list] = []
        timed_out = False
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=read_timeout) as r:
                for raw in r:
                    # HARD wall-clock ceiling — bounds a never-ending load stream.
                    if time.time() - t0 > ceiling_s:
                        timed_out = True
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                    except Exception:
                        continue
                    et = ev.get("type")
                    if et == "status":
                        wn = ev.get("worker_name") or ev.get("worker_id")
                        if wn:
                            worker = wn
                        stage = ev.get("stage")
                        stages.append([stage, wn, ev.get("progress")])
                        if stage and first_load_t is None and "load" in str(stage).lower():
                            first_load_t = time.time()
                    elif et == "token":
                        now = time.time()
                        tokens += 1
                        if first_token_t is None:
                            first_token_t = now
                        last_token_t = now
                        token_times.append(now)
                        # Client-side bound: once we've sampled enough tokens for a
                        # stable median decode rate, stop reading and cancel (the
                        # server won't self-cap). Keeps each fire short.
                        if (max_collect_tokens is not None
                                and tokens >= max_collect_tokens):
                            collected_enough = True
                            break
                    elif et == "error":
                        error = ev.get("message")
                        outcome = "error"
                    elif et == "done":
                        finish_reason = ev.get("finish_reason")
                        outcome = outcome or "done"
                    elif et in ("end", "final"):
                        outcome = outcome or "done"
        except urllib.error.HTTPError as e:
            body_txt = ""
            try:
                body_txt = e.read().decode("utf-8", "replace")
            except Exception:
                pass
            error = f"HTTP {e.code}: {body_txt[:400]}"
            outcome = "error"
        except (TimeoutError, OSError) as e:  # socket read timed out / dropped
            timed_out = True
            error = error or f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001 — a client fault is data, not a crash
            error = f"{type(e).__name__}: {e}"
            outcome = outcome or "client_exception"
        wall = time.time() - t0
        if collected_enough:
            # we bounded the fire ourselves — stop the server generating more.
            self.cancel(request_id)
            outcome = outcome or "done"
        if timed_out and outcome is None:
            outcome = "done" if first_token_t else "load-timeout"
            error = error or (f"chat exceeded {ceiling_s}s wall ceiling in stage "
                              f"'{stages[-1][0] if stages else 'connect'}'")
            self.cancel(request_id)  # don't leave it churning on the worker
        if outcome is None:
            outcome = "done" if first_token_t else "closed_no_token"
        # Classify a refusal from the verbatim error text (structured detail is
        # captured later from the worker row; this is the coarse label).
        if outcome == "error" and error and (
                "won't fit" in error or "refus" in error.lower()
                or '"refused"' in error):
            outcome = "refused"
        ttft = (first_token_t - t0) if first_token_t else None
        load_ref = first_load_t or t0
        load_dur = None
        if first_token_t:
            load_dur = first_token_t - load_ref
        # Decode rate = 1 / MEDIAN(inter-token interval). The median is robust to
        # the engine's continuation stalls (periodic ~0.8s re-prefill gaps) and to
        # ttft, so it reports the true STEADY-STATE decode speed — the thing that
        # collapses when weight layers spill to host RAM. A naive tokens/whole-span
        # rate is destroyed by those gaps (measured 2.4 vs the real 92 tok/s).
        gen_s = None
        tokens_per_s = None            # steady-state decode (1/median interval)
        tokens_per_s_span = None       # naive whole-span rate (kept for reference)
        last_token_s = (last_token_t - t0) if last_token_t else None
        deltas = [token_times[i] - token_times[i - 1]
                  for i in range(1, len(token_times))]
        if deltas:
            sd = sorted(deltas)
            n = len(sd)
            med = sd[n // 2] if n % 2 else (sd[n // 2 - 1] + sd[n // 2]) / 2.0
            if med > 0:
                tokens_per_s = 1.0 / med
        if (first_token_t and last_token_t and last_token_t > first_token_t
                and tokens > 1):
            gen_s = last_token_t - first_token_t
            tokens_per_s_span = (tokens - 1) / gen_s
        return {
            "outcome": outcome, "served_worker": worker, "error": error,
            "finish_reason": finish_reason,
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "load_duration_s": round(load_dur, 3) if load_dur is not None else None,
            "wall_s": round(wall, 3), "tokens": tokens, "stages": stages,
            "last_token_s": round(last_token_s, 3) if last_token_s is not None else None,
            "gen_s": round(gen_s, 3) if gen_s is not None else None,
            "tokens_per_s": round(tokens_per_s, 2) if tokens_per_s is not None else None,
            "tokens_per_s_span": (round(tokens_per_s_span, 2)
                                  if tokens_per_s_span is not None else None),
        }
