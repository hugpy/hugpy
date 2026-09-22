// LIVE tok/s telemetry for the generate workbench's Toks panel. Polls TWO
// same-origin `/api` endpoints together every 2s, MOUNT-GATED exactly like
// `useMediaJobs` (mount → immediate fetch + 2s interval; unmount → interval
// cleared and a mounted-ref flipped so a late response never setStates). A fetch
// error keeps the last-good data rather than flashing empty, and every field is
// guarded — the server side is being built in parallel, so missing/extra fields
// must never throw.
//
//   • GET /api/llm/toks/recent?limit=100 → { entries:[{ ts, worker_id,
//     worker_name, model_key, tok_s, ttft_s, completion_tokens, config_key, ok }] }
//     newest-first — the live tail + the per-worker strip.
//   • GET /api/llm/toks/report → { groups:[{ worker_name, model_key, config_key,
//     n, mean_tok_s, p50_tok_s, p95_tok_s }] } sorted best mean first — the
//     best-config leaderboard.
import { useCallback, useEffect, useRef, useState } from "react";
import { hugpyConfig } from "../config";
import { request, okValue } from "../transport/client";

const POLL_MS = 2000;
const RECENT_LIMIT = 100;

/** One recent inference sample (omit-when-unset tolerated on the wire). */
export interface TokEntry {
  ts: number | null; // epoch seconds when the sample was recorded
  workerId: string | null;
  workerName: string | null;
  modelKey: string | null;
  tokS: number | null; // tokens/second
  ttftS: number | null; // time-to-first-token, seconds
  completionTokens: number | null;
  configKey: string | null;
  ok: boolean | null;
}

/** One best-config rollup row from /report. */
export interface TokGroup {
  workerName: string | null;
  modelKey: string | null;
  configKey: string | null;
  n: number | null;
  meanTokS: number | null;
  p50TokS: number | null;
  p95TokS: number | null;
}

function str(v: unknown): string | null {
  return typeof v === "string" && v.trim() ? v : null;
}
function num(v: unknown): number | null {
  return typeof v === "number" && isFinite(v) ? v : null;
}
function bool(v: unknown): boolean | null {
  return typeof v === "boolean" ? v : null;
}

function parseEntries(raw: unknown): TokEntry[] {
  const entries = (raw as { entries?: unknown } | null)?.entries;
  if (!Array.isArray(entries)) return [];
  const out: TokEntry[] = [];
  for (const e of entries) {
    if (!e || typeof e !== "object") continue;
    const r = e as Record<string, unknown>;
    out.push({
      ts: num(r.ts),
      workerId: str(r.worker_id),
      workerName: str(r.worker_name),
      modelKey: str(r.model_key),
      tokS: num(r.tok_s),
      ttftS: num(r.ttft_s),
      completionTokens: num(r.completion_tokens),
      configKey: str(r.config_key),
      ok: bool(r.ok),
    });
  }
  return out;
}

function parseGroups(raw: unknown): TokGroup[] {
  const groups = (raw as { groups?: unknown } | null)?.groups;
  if (!Array.isArray(groups)) return [];
  const out: TokGroup[] = [];
  for (const g of groups) {
    if (!g || typeof g !== "object") continue;
    const r = g as Record<string, unknown>;
    out.push({
      workerName: str(r.worker_name),
      modelKey: str(r.model_key),
      configKey: str(r.config_key),
      n: num(r.n),
      meanTokS: num(r.mean_tok_s),
      p50TokS: num(r.p50_tok_s),
      p95TokS: num(r.p95_tok_s),
    });
  }
  return out;
}

/**
 * The live tok/s feed: recent samples (newest-first) + the best-config report,
 * refreshed together every 2s while a consumer is mounted. `live` reports whether
 * the last poll cycle succeeded, so the panel can show a live/paused dot.
 */
export function useToks(): {
  entries: TokEntry[];
  groups: TokGroup[];
  live: boolean;
} {
  const [entries, setEntries] = useState<TokEntry[]>([]);
  const [groups, setGroups] = useState<TokGroup[]>([]);
  const [live, setLive] = useState(false);
  const mounted = useRef(true);

  const load = useCallback(async () => {
    // Both endpoints in parallel; each is independent, so one failing keeps the
    // other's last-good data instead of blanking the whole panel.
    const [recent, report] = await Promise.all([
      request<unknown>(`${hugpyConfig.toksRecentUrl}?limit=${RECENT_LIMIT}`, {
        meta: { specKey: "llm", operation: "llm.toks.recent" },
      }),
      request<unknown>(hugpyConfig.toksReportUrl, {
        meta: { specKey: "llm", operation: "llm.toks.report" },
      }),
    ]);
    if (!mounted.current) return;
    if (recent.ok) setEntries(parseEntries(okValue(recent)));
    if (report.ok) setGroups(parseGroups(okValue(report)));
    setLive(recent.ok || report.ok);
  }, []);

  useEffect(() => {
    mounted.current = true;
    void load();
    const id = window.setInterval(() => void load(), POLL_MS);
    return () => {
      mounted.current = false;
      window.clearInterval(id);
    };
  }, [load]);

  return { entries, groups, live };
}

/** The poll cadence, exported so the panel header can label it ("every 2s"). */
export const TOKS_POLL_MS = POLL_MS;

/** "12s ago" / "3m ago" from an epoch-seconds ts, "" when unknown. */
export function tokAgo(ts: number | null, nowMs: number): string {
  if (ts == null) return "";
  const secs = Math.max(0, Math.floor(nowMs / 1000 - ts));
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  return `${Math.floor(secs / 3600)}h ago`;
}

/** Group recent entries by worker → { last tok/s, avg tok/s over the window }. */
export interface WorkerRollup {
  workerId: string;
  workerName: string;
  modelKey: string | null;
  lastTokS: number | null;
  avgTokS: number | null;
  n: number;
}

export function rollupByWorker(entries: TokEntry[]): WorkerRollup[] {
  const byWorker = new Map<string, TokEntry[]>();
  for (const e of entries) {
    const key = e.workerId ?? e.workerName;
    if (!key) continue;
    const list = byWorker.get(key);
    if (list) list.push(e);
    else byWorker.set(key, [e]);
  }
  const out: WorkerRollup[] = [];
  for (const [key, list] of byWorker) {
    // Entries arrive newest-first, so the first tok_s we see is the latest.
    const samples = list.map((e) => e.tokS).filter((v): v is number => v != null);
    const avg =
      samples.length > 0
        ? samples.reduce((a, b) => a + b, 0) / samples.length
        : null;
    const first = list[0];
    out.push({
      workerId: key,
      workerName: first?.workerName ?? key,
      modelKey: first?.modelKey ?? null,
      lastTokS: samples.length > 0 ? samples[0] : null,
      avgTokS: avg,
      n: list.length,
    });
  }
  return out;
}
