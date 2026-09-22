// STUDIO-ASSIST LIVE LOG — the client half (operator directive, 2026-07-31).
//
// One shared EventSource for the whole studio: the log panel (and any future
// mount) subscribes to THIS module rather than each opening its own stream.
// Structurally the video-arm twin of the console's evictionStream.js — a
// module-scope store, a reference-counted connection (opens on first subscriber,
// closes on last), a de-dup set spanning the backfill + the SSE replay window,
// and a rowid cursor tail so the stream is correct across gunicorn workers.
//
// It differs from evictionStream in one way: an assist attempt is logged as ONE
// terminal record (the generation choke point emits on failure; the mode handler
// emits served/parse_error on success — see comms/studio_assist_log.py), so the
// store is keyed by run_id and the newest record for a run_id wins. No folding of
// many stage-events into a card is needed; a row IS an attempt.
//
// Auth rides the same path every other /video call uses: a console operator's
// same-origin session cookie (credentials:"include" via the EventSource
// withCredentials flag) or a stored ?share= key appended to the stream URL (an
// EventSource cannot carry the X-Video-Share header the XHR transport uses).
import { useEffect, useState } from "react";
import { request, okValue } from "../transport/client";
import { hugpyConfig } from "../config";
import { videoShareKey } from "../share";

/** One stored generate attempt — the wire shape of comms/studio_assist_log.py. */
export interface AssistAttempt {
  _id?: number;
  ts?: number;
  seq?: number;
  worker_id?: string;
  run_id?: string;
  mode?: string;
  kind?: string;
  model_requested?: string;
  model_resolved?: string;
  /** The UNTRUNCATED model reply — the whole point of the log. */
  raw?: string;
  /** The think-stripped prose that would have been used as the prompt. */
  text?: string;
  /** The reasoning that was stripped out. */
  reasoning?: string;
  /** True when `text` was salvaged FROM the reasoning (no prose came back). */
  from_reasoning?: boolean;
  outcome?: "served" | "empty" | "parse_error" | "worker_error" | "resolve_error" | string;
  error?: string;
  elapsed_ms?: number;
  /** The backfill route bounds a huge reply on the page load only. */
  raw_truncated?: boolean;
  /** Transport marker ("stream.ready") — never an attempt; filtered out. */
  stage?: string;
}

export type ConnState = "idle" | "connecting" | "live" | "disconnected";

const MAX_ATTEMPTS = 300; // oldest attempts drop off the front
const BACKFILL_LIMIT = 300;
const RETRY_MS = 5000;

let attempts: AssistAttempt[] = []; // oldest-first
let conn: ConnState = "idle";
let error: string | null = null;
let version = 0;

const index = new Map<string, number>(); // run_id (or synthetic) -> position
const seenId = new Set<number>(); // de-dup by store rowid across backfill + replay
const listeners = new Set<() => void>();

let es: EventSource | null = null;
let refs = 0;
let retryTimer: ReturnType<typeof setTimeout> | null = null;
let backfilled = false;

function emit(): void {
  version++;
  for (const fn of listeners) {
    try {
      fn();
    } catch {
      /* one bad subscriber must not stall the rest */
    }
  }
}

function keyOf(rec: AssistAttempt): string {
  return rec.run_id || `solo:${rec._id ?? rec.seq ?? rec.ts ?? Math.random()}`;
}

function ingest(batch: AssistAttempt[]): void {
  let changed = false;
  for (const rec of batch) {
    if (!rec || typeof rec !== "object") continue;
    if (rec.stage === "stream.ready") continue; // transport marker, not an attempt
    if (typeof rec._id === "number") {
      if (seenId.has(rec._id)) continue;
      seenId.add(rec._id);
    }
    const k = keyOf(rec);
    const at = index.get(k);
    if (at == null) {
      index.set(k, attempts.length);
      attempts = attempts.concat(rec);
    } else {
      // Newest record for a run_id wins (a terminal record supersedes any
      // provisional one that arrived first).
      const next = attempts.slice();
      next[at] = { ...next[at], ...rec };
      attempts = next;
    }
    changed = true;
  }
  if (!changed) return;
  if (attempts.length > MAX_ATTEMPTS) {
    attempts = attempts.slice(attempts.length - MAX_ATTEMPTS);
    // Rebuild the index after trimming the front.
    index.clear();
    attempts.forEach((r, i) => index.set(keyOf(r), i));
  }
  emit();
}

function setConn(c: ConnState): void {
  if (conn === c) return;
  conn = c;
  emit();
}

function streamUrl(): string {
  const key = videoShareKey();
  if (!key) return hugpyConfig.promptAssistLogStreamUrl;
  const base = hugpyConfig.promptAssistLogStreamUrl;
  const sep = base.includes("?") ? "&" : "?";
  return `${base}${sep}share=${encodeURIComponent(key)}`;
}

function openStream(): void {
  if (es) return;
  setConn("connecting");
  let source: EventSource;
  try {
    source = new EventSource(streamUrl(), { withCredentials: true });
  } catch {
    setConn("disconnected");
    return;
  }
  es = source;
  source.onopen = () => setConn("live");
  source.onmessage = (e: MessageEvent) => {
    let rec: AssistAttempt;
    try {
      rec = JSON.parse(e.data);
    } catch {
      return;
    }
    ingest([rec]);
  };
  source.onerror = () => {
    source.close();
    if (es === source) es = null;
    setConn("disconnected");
    // Reconnect only while someone is still watching. The server caps a stream at
    // an hour and expects re-attach; the replay window + de-dup set mean no gap.
    if (refs > 0) {
      if (retryTimer) clearTimeout(retryTimer);
      retryTimer = setTimeout(() => {
        if (refs > 0) openStream();
      }, RETRY_MS);
    }
  };
}

function closeStream(): void {
  if (retryTimer) {
    clearTimeout(retryTimer);
    retryTimer = null;
  }
  if (es) {
    es.close();
    es = null;
  }
  setConn("idle");
}

function backfillOnce(): void {
  if (backfilled) return;
  backfilled = true;
  request<{ events?: AssistAttempt[] }>(
    `${hugpyConfig.promptAssistLogUrl}?limit=${BACKFILL_LIMIT}`,
    { method: "GET", meta: { specKey: "studio", operation: "assist-log-backfill" } },
  )
    .then((res) => {
      if (res.ok) {
        error = null;
        const value = okValue(res);
        ingest(Array.isArray(value?.events) ? value.events : []);
        emit();
      } else {
        error = "could not load the assist log";
        emit();
      }
    })
    .catch(() => {
      error = "could not load the assist log";
      emit();
    });
}

function acquire(): void {
  refs++;
  if (refs === 1) {
    backfillOnce();
    openStream();
  }
}

function release(): void {
  refs = Math.max(0, refs - 1);
  if (refs === 0) closeStream();
}

/** Force a reconnect (the panel's Reconnect button). */
export function reconnectAssistLog(): void {
  closeStream();
  if (refs > 0) openStream();
}

export interface UseStudioAssistLog {
  attempts: AssistAttempt[];
  conn: ConnState;
  error: string | null;
  reconnect: () => void;
}

/** Subscribe to the shared studio-assist log stream. */
export function useStudioAssistLog(): UseStudioAssistLog {
  const [snap, setSnap] = useState(() => ({ attempts, conn, error, version }));

  useEffect(() => {
    acquire();
    const onChange = () => setSnap({ attempts, conn, error, version });
    onChange();
    listeners.add(onChange);
    return () => {
      listeners.delete(onChange);
      release();
    };
  }, []);

  return {
    attempts: snap.attempts,
    conn: snap.conn,
    error: snap.error,
    reconnect: reconnectAssistLog,
  };
}

/** Test/diagnostic reset — drops everything and detaches. */
export function _resetForTests(): void {
  closeStream();
  attempts = [];
  index.clear();
  seenId.clear();
  listeners.clear();
  refs = 0;
  backfilled = false;
  error = null;
}
