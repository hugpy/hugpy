/**
 * or-k18 / k114 — every oracle call the station makes, in one place.
 *
 *   GET  /api/oracle/steward        → StewardReport (report only)
 *   POST /api/oracle/steward        → StewardReport (report AND apply rebalancing)
 *   POST /api/oracle/selection      → SelectionDecision (explain, never execute)
 *   GET  /api/oracle/producers?ref= → ProducerLookup (who made this artifact)
 *   GET  /api/oracle/ledger/summary → LedgerSummary
 *   GET  /api/oracle/capabilities   → CapabilitiesBody
 *   GET  /video/jobs?all=1          → the performance jobs (media bus, same feed
 *                                     Active Processes reads) — the run picker
 *   GET  /video/jobs/<id>           → JobView; result.movie is the run manifest
 *                                     performance_relay.py wrote (with `dag`)
 *
 * Defensive by design: the producers/ledger routes were landing while this was
 * written, and the selector can be disabled (503). A missing route (404) or a
 * disabled selector is surfaced as a `gap` string next to the data it would
 * have filled, never swallowed and never faked.
 *
 * URL note: like useScriptFirstRun.ts, the oracle URLs are built off
 * `hugpyConfig.apiBase` here rather than added to config.ts (a foreign dirty
 * file today). The config.ts one-liners that belong there are:
 *   oracleStewardUrl:  `${apiBase}/oracle/steward`
 *   oracleSelectionUrl:`${apiBase}/oracle/selection`
 *   oracleProducersUrl:`${apiBase}/oracle/producers`
 *   oracleLedgerUrl:   `${apiBase}/oracle/ledger/summary`
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { hugpyConfig, videoJobUrl } from "../../config";
import {
  describeAppError,
  errorOf,
  okValue,
  request,
  type AppError,
} from "../../transport/client";
import type {
  CapabilitiesBody,
  DagNodeRow,
  JobView,
  LedgerSummary,
  ManifestShot,
  ProducerLookup,
  ProducerRow,
  RunManifest,
  Scorecard,
  SelectionDecision,
  SelectionQuery,
  StewardReport,
} from "./oracleTypes";

const STATION = "oracle";
const BASE = hugpyConfig.apiBase;
export const ORACLE_URLS = {
  capabilities: `${BASE}/oracle/capabilities`,
  steward: `${BASE}/oracle/steward`,
  selection: `${BASE}/oracle/selection`,
  producers: `${BASE}/oracle/producers`,
  ledgerSummary: `${BASE}/oracle/ledger/summary`,
  route: `${BASE}/oracle/route`,
} as const;

// ---------------------------------------------------------------------------
// A "gap" is the honest description of why a panel has no data.
// ---------------------------------------------------------------------------

export interface Gap {
  /** Short code the UI can branch on: not_found | unavailable | error. */
  code: "not_found" | "unavailable" | "error";
  message: string;
  status?: number;
}

function gapOf(err: AppError, what: string): Gap {
  if (err.kind === "server" && err.status === 404) {
    return {
      code: "not_found",
      status: 404,
      message: `${what}: route not found (404) — this server does not expose it yet.`,
    };
  }
  if (err.kind === "server" && err.status === 503) {
    return {
      code: "unavailable",
      status: 503,
      message: `${what}: ${err.message} (503 — selection disabled or ledger unavailable).`,
    };
  }
  return {
    code: "error",
    status: err.kind === "server" ? err.status : undefined,
    message: `${what}: ${describeAppError(err)}`,
  };
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

// ---------------------------------------------------------------------------
// Steward
// ---------------------------------------------------------------------------

export interface StewardApi {
  report: StewardReport | null;
  gap: Gap | null;
  busy: "get" | "post" | null;
  /** When the report in hand was fetched (browser clock — labelled as such). */
  fetchedAt: number | null;
  refresh(): void;
  /** POST: re-run the audit AND apply bounded rebalancing to the live selector. */
  rerun(): void;
}

export function useSteward(): StewardApi {
  const [report, setReport] = useState<StewardReport | null>(null);
  const [gap, setGap] = useState<Gap | null>(null);
  const [busy, setBusy] = useState<"get" | "post" | null>(null);
  const [fetchedAt, setFetchedAt] = useState<number | null>(null);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const call = useCallback(async (method: "GET" | "POST") => {
    setBusy(method === "GET" ? "get" : "post");
    const r = await request<StewardReport>(ORACLE_URLS.steward, {
      method,
      headers: method === "POST" ? { "Content-Type": "application/json" } : undefined,
      body: method === "POST" ? "{}" : undefined,
      meta: { specKey: STATION, operation: `oracle.steward.${method.toLowerCase()}` },
    });
    if (!alive.current) return;
    setBusy(null);
    if (!r.ok) {
      setGap(gapOf(errorOf(r), "steward"));
      return;
    }
    setGap(null);
    setReport(okValue(r));
    setFetchedAt(Date.now());
  }, []);

  useEffect(() => {
    void call("GET");
  }, [call]);

  return {
    report,
    gap,
    busy,
    fetchedAt,
    refresh: () => void call("GET"),
    rerun: () => void call("POST"),
  };
}

// ---------------------------------------------------------------------------
// Ledger summary + capabilities (read-once, manual refresh)
// ---------------------------------------------------------------------------

export function useLedgerSummary(): { summary: LedgerSummary | null; gap: Gap | null; refresh(): void } {
  const [summary, setSummary] = useState<LedgerSummary | null>(null);
  const [gap, setGap] = useState<Gap | null>(null);
  const load = useCallback(async () => {
    const r = await request<LedgerSummary>(ORACLE_URLS.ledgerSummary, {
      meta: { specKey: STATION, operation: "oracle.ledger.summary" },
    });
    if (!r.ok) {
      setGap(gapOf(errorOf(r), "ledger summary"));
      return;
    }
    setGap(null);
    setSummary(okValue(r));
  }, []);
  useEffect(() => {
    void load();
  }, [load]);
  return { summary, gap, refresh: () => void load() };
}

export function useCapabilities(): { caps: CapabilitiesBody | null; gap: Gap | null } {
  const [caps, setCaps] = useState<CapabilitiesBody | null>(null);
  const [gap, setGap] = useState<Gap | null>(null);
  useEffect(() => {
    let on = true;
    void (async () => {
      const r = await request<CapabilitiesBody>(ORACLE_URLS.capabilities, {
        meta: { specKey: STATION, operation: "oracle.capabilities" },
      });
      if (!on) return;
      if (!r.ok) setGap(gapOf(errorOf(r), "capabilities"));
      else setCaps(okValue(r));
    })();
    return () => {
      on = false;
    };
  }, []);
  return { caps, gap };
}

// ---------------------------------------------------------------------------
// Selection explain (POST /oracle/selection) — one in-flight query at a time
// ---------------------------------------------------------------------------

export interface SelectionApi {
  decision: SelectionDecision | null;
  gap: Gap | null;
  busy: boolean;
  /** The query that produced `decision` (so the UI can print what it asked). */
  asked: SelectionQuery | null;
  explain(q: SelectionQuery): void;
  clear(): void;
}

export function useSelectionExplain(): SelectionApi {
  const [decision, setDecision] = useState<SelectionDecision | null>(null);
  const [asked, setAsked] = useState<SelectionQuery | null>(null);
  const [gap, setGap] = useState<Gap | null>(null);
  const [busy, setBusy] = useState(false);
  const seq = useRef(0);

  const explain = useCallback((q: SelectionQuery) => {
    const my = ++seq.current;
    setBusy(true);
    setAsked(q);
    void (async () => {
      const r = await request<SelectionDecision>(ORACLE_URLS.selection, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(q),
        meta: { specKey: STATION, operation: "oracle.selection.explain" },
      });
      if (my !== seq.current) return; // a newer query superseded this one
      setBusy(false);
      if (!r.ok) {
        setDecision(null);
        setGap(gapOf(errorOf(r), `selection for ${q.capability}`));
        return;
      }
      setGap(null);
      setDecision(okValue(r));
    })();
  }, []);

  const clear = useCallback(() => {
    seq.current += 1;
    setDecision(null);
    setAsked(null);
    setGap(null);
    setBusy(false);
  }, []);

  return { decision, gap, busy, asked, explain, clear };
}

// ---------------------------------------------------------------------------
// Producer lookup (GET /oracle/producers?ref=) — cached per ref for the session
// ---------------------------------------------------------------------------

export type ProducerState =
  | { status: "loading" }
  | { status: "hit"; producer: ProducerRow }
  | { status: "miss" } // the ledger has no row for this ref
  | { status: "gap"; gap: Gap };

export function useProducers(): {
  lookup(ref: string | null | undefined): ProducerState | null;
  states: Record<string, ProducerState>;
} {
  const [states, setStates] = useState<Record<string, ProducerState>>({});
  const inflight = useRef(new Set<string>());

  const lookup = useCallback(
    (ref: string | null | undefined): ProducerState | null => {
      if (!ref) return null;
      const have = states[ref];
      if (have) return have;
      if (!inflight.current.has(ref)) {
        inflight.current.add(ref);
        void (async () => {
          const url = `${ORACLE_URLS.producers}?ref=${encodeURIComponent(ref)}`;
          const r = await request<ProducerLookup>(url, {
            meta: { specKey: STATION, operation: "oracle.producers.lookup" },
          });
          let next: ProducerState;
          if (!r.ok) next = { status: "gap", gap: gapOf(errorOf(r), "producer") };
          else {
            const body = okValue(r);
            const p = isRecord(body) ? body.producer : null;
            next = isRecord(p) && typeof p.model_id === "string"
              ? { status: "hit", producer: p as unknown as ProducerRow }
              : { status: "miss" };
          }
          setStates((s) => ({ ...s, [ref]: next }));
        })();
      }
      return { status: "loading" };
    },
    [states],
  );

  return { lookup, states };
}

// ---------------------------------------------------------------------------
// Run picker + run manifest — reuses the media-bus job routes the studio and
// Active Processes already poll (GET /video/jobs, GET /video/jobs/<id>).
// ---------------------------------------------------------------------------

export interface PerformanceJobRow {
  id: string;
  name: string;
  status: string;
  createdAt: number | null;
  runId: string | null;
}

function parseJobRows(raw: unknown): PerformanceJobRow[] {
  const list = isRecord(raw) && Array.isArray(raw.jobs) ? raw.jobs : Array.isArray(raw) ? raw : [];
  const out: PerformanceJobRow[] = [];
  for (const row of list) {
    if (!isRecord(row)) continue;
    const name = String(row.name ?? row.kind ?? "");
    // The bus kind for the oracle's audio-first runner is ("oracle","performance")
    // → "performance" / "oracle.performance" / "video_performance" depending on
    // the bridge; match loosely on the word rather than on one spelling.
    if (!/performance/i.test(name)) continue;
    const id = String(row.job_id ?? row.id ?? "");
    if (!id) continue;
    const prog = isRecord(row.progress) ? row.progress : null;
    out.push({
      id,
      name,
      status: String(row.status ?? ""),
      createdAt: typeof row.created === "number" ? row.created : null,
      runId: prog && typeof prog.run_id === "string" ? prog.run_id : null,
    });
  }
  return out;
}

export function usePerformanceJobs(): { jobs: PerformanceJobRow[]; gap: Gap | null; refresh(): void } {
  const [jobs, setJobs] = useState<PerformanceJobRow[]>([]);
  const [gap, setGap] = useState<Gap | null>(null);
  const load = useCallback(async () => {
    const r = await request<unknown>(`${hugpyConfig.mediaJobsUrl}?all=1`, {
      meta: { specKey: STATION, operation: "oracle.runs.list" },
    });
    if (!r.ok) {
      setGap(gapOf(errorOf(r), "performance jobs"));
      return;
    }
    setGap(null);
    setJobs(parseJobRows(okValue(r)));
  }, []);
  useEffect(() => {
    void load();
  }, [load]);
  return { jobs, gap, refresh: () => void load() };
}

export interface RunManifestApi {
  jobId: string | null;
  job: JobView | null;
  manifest: RunManifest | null;
  gap: Gap | null;
  loading: boolean;
  select(jobId: string | null): void;
  refresh(): void;
}

export function useRunManifest(): RunManifestApi {
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobView | null>(null);
  const [gap, setGap] = useState<Gap | null>(null);
  const [loading, setLoading] = useState(false);
  const seq = useRef(0);

  const load = useCallback(async (id: string | null) => {
    const my = ++seq.current;
    if (!id) {
      setJob(null);
      setGap(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    const r = await request<JobView>(videoJobUrl(id), {
      meta: { specKey: STATION, operation: "oracle.run.manifest" },
    });
    if (my !== seq.current) return;
    setLoading(false);
    if (!r.ok) {
      setJob(null);
      setGap(gapOf(errorOf(r), `job ${id}`));
      return;
    }
    setGap(null);
    setJob(okValue(r));
  }, []);

  useEffect(() => {
    void load(jobId);
  }, [jobId, load]);

  const manifest = useMemo<RunManifest | null>(() => {
    const m = job?.result?.movie;
    return isRecord(m) ? (m as unknown as RunManifest) : null;
  }, [job]);

  return {
    jobId,
    job,
    manifest,
    gap,
    loading,
    select: (id) => setJobId(id),
    refresh: () => void load(jobId),
  };
}

// ---------------------------------------------------------------------------
// Manifest → DAG rows. Pure; every derived field names its source.
// ---------------------------------------------------------------------------

const NODE_CAPABILITY: Record<string, string> = {
  spatial: "spatial.normalize.scene",
  keyframe: "image.generate",
  clip: "video.generate.i2v",
};

/** "keyframe:seg_03" → { kind:"keyframe", segmentId:"seg_03" }. */
export function splitNodeId(nodeId: string): { kind: string; segmentId: string | null } {
  const i = nodeId.indexOf(":");
  if (i < 0) return { kind: nodeId, segmentId: null };
  return { kind: nodeId.slice(0, i), segmentId: nodeId.slice(i + 1) || null };
}

/** The capability the selector would be asked for this node, or null when the
 *  node kind is model-free (spatial validation, lock gate, assembly). */
export function capabilityForNode(kind: string): string | null {
  return NODE_CAPABILITY[kind] ?? null;
}

export function dagRows(manifest: RunManifest | null): DagNodeRow[] {
  const dag = manifest?.dag;
  if (!dag || !isRecord(dag.nodes)) return [];
  const shotsBySeg = new Map<string, ManifestShot>();
  for (const s of manifest?.shots ?? []) shotsBySeg.set(s.segment_id, s);
  const dagShotsBySeg = new Map<string, (typeof dag.shots)[number]>();
  for (const s of dag.shots ?? []) if (s && s.segment_id) dagShotsBySeg.set(s.segment_id, s);
  const repairsByNode = new Map<string, (typeof dag.repairs)[number]>();
  for (const r of dag.repairs ?? []) repairsByNode.set(r.failed_node, r);
  const failed = new Set(dag.failed_nodes ?? []);

  // receipts: artifact ref → model_id, when the receipt lists its artifacts
  const modelByRef = new Map<string, string>();
  for (const rc of manifest?.receipts ?? []) {
    for (const a of rc.artifacts ?? []) {
      const ref = typeof a.ref === "string" ? a.ref : typeof a.uri === "string" ? a.uri : null;
      if (ref && rc.model_id) modelByRef.set(ref, rc.model_id);
    }
  }

  const rows: DagNodeRow[] = [];
  for (const [nodeId, state] of Object.entries(dag.nodes)) {
    const { kind, segmentId } = splitNodeId(nodeId);
    const shot = segmentId ? shotsBySeg.get(segmentId) : undefined;
    const dshot = segmentId ? dagShotsBySeg.get(segmentId) : undefined;
    let producedRef: string | null = null;
    let scorecard: Scorecard | null = null;
    if (kind === "keyframe") {
      producedRef = shot?.keyframe_ref ?? dshot?.keyframe_ref ?? null;
      scorecard = shot?.keyframe_scorecard ?? null;
    } else if (kind === "clip") {
      producedRef = shot?.clip_ref ?? dshot?.clip_ref ?? null;
      scorecard = shot?.scorecard ?? dshot?.scorecard ?? null;
    } else if (kind === "assemble") {
      producedRef = dag.video_ref ?? manifest?.video_ref ?? null;
      scorecard = manifest?.scorecard ?? null;
    }
    const producerModel = producedRef ? modelByRef.get(producedRef) ?? null : null;
    rows.push({
      nodeId,
      kind,
      segmentId,
      state: String(state),
      producedRef,
      producerModel,
      producerSource: producerModel ? "receipt" : null,
      lease: null, // NodeRecord.lease_owner/lease_expires_at live in the journal only
      scorecard,
      repair: repairsByNode.get(nodeId) ?? null,
      failed: failed.has(nodeId) || String(state) === "failed",
    });
  }
  // Stable order: lock, then per-segment spatial→keyframe→clip, then assemble.
  const rank = (k: string) => ({ lock: 0, spatial: 1, keyframe: 2, clip: 3, assemble: 9 })[k] ?? 5;
  rows.sort((a, b) => {
    const sa = a.segmentId ?? "";
    const sb = b.segmentId ?? "";
    if (a.kind === "lock" || b.kind === "lock" || a.kind === "assemble" || b.kind === "assemble") {
      return rank(a.kind) - rank(b.kind);
    }
    return sa.localeCompare(sb) || rank(a.kind) - rank(b.kind);
  });
  return rows;
}

// ---------------------------------------------------------------------------
// Spatial discovery — the evaluator (oracle/spatial_eval.py) is not yet folded
// into the run manifest, so this LOOKS rather than assumes: anything shaped
// like a SpatialSceneManifest (schema_version + coordinate_system + entities)
// or a SpatialEvalReport (metrics + measured + codes) anywhere in the manifest.
// ---------------------------------------------------------------------------

export interface SpatialFind {
  path: string;
  kind: "manifest" | "eval" | "ref";
  value: unknown;
}

export function findSpatial(manifest: RunManifest | null, maxDepth = 6): SpatialFind[] {
  const out: SpatialFind[] = [];
  if (!manifest) return out;
  const seen = new Set<unknown>();
  const walk = (v: unknown, path: string, depth: number) => {
    if (depth > maxDepth || v == null) return;
    if (typeof v !== "object") return;
    if (seen.has(v)) return;
    seen.add(v);
    if (Array.isArray(v)) {
      v.forEach((x, i) => walk(x, `${path}[${i}]`, depth + 1));
      return;
    }
    const r = v as Record<string, unknown>;
    if ("coordinate_system" in r && "entities" in r && ("timebase" in r || "schema_version" in r)) {
      out.push({ path, kind: "manifest", value: r });
      return;
    }
    if (Array.isArray(r.metrics) && ("measured" in r || "codes" in r) && "thresholds" in r) {
      out.push({ path, kind: "eval", value: r });
      return;
    }
    for (const [k, x] of Object.entries(r)) {
      if (k === "spatial_manifest" && typeof x === "string" && x) {
        out.push({ path: `${path}.${k}`, kind: "ref", value: x });
        continue;
      }
      walk(x, path ? `${path}.${k}` : k, depth + 1);
    }
  };
  walk(manifest, "", 0);
  return out;
}
