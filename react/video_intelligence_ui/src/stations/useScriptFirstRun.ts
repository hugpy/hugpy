/**
 * k114 — the script-first run, over HTTP.
 *
 * One hook owns every call to `/video/script/*` and the run state they all
 * return. The backend already answers every mutation with the WHOLE run, so
 * this never has to re-fetch to stay honest and there is no second projection
 * of "is it locked" living in component state where it could disagree with the
 * server.
 *
 * Two deliberate shapes:
 *
 * 1. `busy` is the NAME of the operation in flight, not a boolean, and
 *    `segmentBusy` is the id of the one segment card that is regenerating.
 *    That is what lets a Regenerate button busy its own card while every
 *    sibling stays interactive — the doc's "allow segment regeneration without
 *    silently changing sibling segments" is a UI property as well as a
 *    backend one.
 *
 * 2. A refusal is kept STRUCTURED (`code` + every `errors` line), never
 *    flattened to one string. The console's shared transport keeps only
 *    `parsed.error` from a non-2xx body, so the backend puts the code and
 *    every validator error in there — this splits it back apart, and the run's
 *    own journalled `last_refusal` carries the machine-readable form
 *    (including an authoring gap's raw model reply) across a page reload.
 *
 * URL note: `hugpyConfig.apiBase` is imported read-only. The one-liner that
 * belongs in `config.ts` — the ONE module permitted to hold URL literals — is
 *   scriptRunsUrl: read("VITE_HUGPY_SCRIPT_RUNS_URL") ?? `${apiBase}/video/script/runs`,
 * and is recorded rather than made because config.ts is another task's dirty
 * file today.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { hugpyConfig } from "../config";
import {
  describeAppError,
  errorOf,
  okValue,
  request,
} from "../transport/client";

const STATION = "script-first";

/** Built off the same base every other station reads. See the URL note above. */
const RUNS_URL = `${hugpyConfig.apiBase}/video/script/runs`;
const SOURCES_URL = `${hugpyConfig.apiBase}/video/script/sources`;

function runUrl(id: string): string {
  return `${RUNS_URL}/${encodeURIComponent(id)}`;
}

// ---------------------------------------------------------------------------
// Wire types — the run state as the backend actually returns it
// ---------------------------------------------------------------------------

export interface SourceRow {
  prompt_id: string;
  text: string;
  digest: string;
  claimed_hash: string | null;
  persisted_at: string | null;
  included: boolean;
  exclusion_reason: string | null;
  origin: string;
}

export interface ArtifactEntry {
  stage: string;
  payload: Record<string, unknown> | null;
  digest: string | null;
  provenance: string;
  note?: string;
  at: string;
  gap: AuthoringGapBody | null;
}

export interface AuthoringGapBody {
  errors: string[];
  raw: string;
  stage: string;
  code: string;
  attempts: number;
  raw_attempts: string[];
}

export interface SegmentRow {
  segment_id: string;
  index: number;
  digest: string;
  spec: Record<string, unknown>;
  prompt: string;
  seed_base: number;
  parents: string[];
  lock_digest: string;
  scene_ref: string | null;
  joint_mode: string;
  window: [number, number, string[]];
  rubric: string[];
}

export interface AttemptRow {
  attempt: number;
  segment_id: string;
  at: string;
  spec_digest: string;
  lock_digest: string;
  parents: string[];
  registry_version: string | null;
  siblings_before: Record<string, string>;
  siblings_after: Record<string, string>;
  siblings_unchanged: boolean;
  ok: boolean;
  kind: string;
  capability: string;
  model_id: string | null;
  seed: number;
  prompt: string;
  params: Record<string, unknown>;
  artifacts: Array<Record<string, unknown>>;
  receipt: Record<string, unknown> | null;
  gap: {
    code: string;
    capability?: string;
    reasons?: string[];
    requirement?: string;
    execution?: string;
    failure?: string;
  } | null;
}

export interface SegmentsBlock {
  compiled_at: string;
  lock_digest: string;
  revision: number;
  parent_digests: string[];
  specs: SegmentRow[];
  graph: { graph_id: string; nodes: string[]; structure_digest: string | null };
  validation: {
    ok: boolean | null;
    errors?: Array<Record<string, unknown>>;
    warnings?: Array<Record<string, unknown>>;
    note?: string;
  };
  execution_order: { sequential: string[][]; parallel: string[][] };
  sibling_shape: {
    parent: string;
    parent_digest: string;
    children: string[];
    note: string;
  };
}

export interface RunState {
  run_id: string;
  created_at: string;
  updated_at: string;
  deliverable: string;
  requirements: string;
  settings: Record<string, unknown>;
  sources: SourceRow[];
  snapshot: Record<string, unknown>;
  snapshot_digest: string;
  models: {
    fleet: {
      registry_version: string | null;
      capabilities: Array<{
        name: string;
        eligible: boolean;
        model_ids: string[];
        reasons: string[];
      }>;
      hardware: Record<string, unknown>;
      errors?: string[];
    };
    authoring_route: {
      capability?: string;
      execution?: string;
      model_id?: string | null;
      model_rationale?: string;
      reasons?: string[];
    };
  };
  ledger: string[];
  artifacts: Record<string, ArtifactEntry>;
  lock: {
    payload: Record<string, unknown>;
    digest: string;
    at: string;
    parent_digests: string[];
  } | null;
  lock_history: Array<{
    revision: number;
    digest: string;
    reason: string;
    at: string;
    parent_revision?: number | null;
  }>;
  segments: SegmentsBlock | null;
  attempts: Record<string, AttemptRow[]>;
  promotions: PromotedSource[];
  events: Array<{ at: string; event: string; detail: Record<string, unknown> }>;
  last_refusal: RefusalBody | null;
  locked: boolean;
  digests: Record<string, unknown>;
  state_path: string;
  limitations: string[];
}

export interface RunSummary {
  run_id: string;
  created_at: string;
  updated_at: string;
  deliverable: string;
  snapshot_digest: string;
  registry_version: string | null;
  locked: boolean;
  lock_digest: string | null;
  revision: number | null;
  stages: string[];
  segments: number;
  attempts: number;
}

export interface PromotedSource {
  source_id: string;
  text: string;
  digest: string;
  promoted_at: string;
  note: string;
  origin: Record<string, unknown>;
  usable_in: string;
  refused_here: string;
}

export interface RefusalBody {
  code: string;
  message: string;
  errors: string[];
  detail: Record<string, unknown>;
  at?: string;
}

export interface CreateRunBody {
  deliverable: string;
  requirements?: string;
  sources?: Array<{
    prompt_id?: string;
    source_id?: string;
    text?: string;
    hash?: string;
    persisted_at?: string;
  }>;
  references?: Record<string, string[]>;
  settings?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * Split the transport's flattened server message back into a refusal.
 *
 * The backend writes `"<CODE>: <message>\n<error>\n<error>"` into the one field
 * the transport preserves. Anything that does not match that shape (a network
 * error, a timeout, a 500 from somewhere else) is passed through as a single
 * line rather than being forced into a code it does not have.
 */
function parseRefusal(message: string): RefusalBody {
  const lines = String(message ?? "").split("\n");
  const head = lines[0] ?? "";
  const match = /^([A-Z_]+):\s*(.*)$/.exec(head);
  if (!match) {
    return { code: "", message: head || "The request failed.", errors: lines.slice(1), detail: {} };
  }
  return {
    code: match[1],
    message: match[2],
    errors: lines.slice(1).filter((l) => l.trim()),
    detail: {},
  };
}

export interface ScriptFirstApi {
  runs: RunSummary[];
  sources: PromotedSource[];
  run: RunState | null;
  runId: string | null;
  busy: string | null;
  segmentBusy: string | null;
  refusal: RefusalBody | null;
  loading: boolean;
  select(id: string | null): void;
  refreshRuns(): void;
  createRun(body: CreateRunBody): Promise<string | null>;
  authorPlot(inputText: string): void;
  putPlot(json: unknown): void;
  authorScreenplay(): void;
  putScreenplay(json: unknown): void;
  putAudioMaster(json: unknown): void;
  buildPreproduction(): void;
  lockRun(audioMaster?: unknown): void;
  revise(reason: string): void;
  compileSegments(): void;
  regenerate(segmentId: string, kind: string): void;
  promote(segmentId: string | null, text: string, note: string): void;
  clearRefusal(): void;
}

export function useScriptFirstRun(): ScriptFirstApi {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sources, setSources] = useState<PromotedSource[]>([]);
  const [run, setRun] = useState<RunState | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [segmentBusy, setSegmentBusy] = useState<string | null>(null);
  const [refusal, setRefusal] = useState<RefusalBody | null>(null);
  const [loading, setLoading] = useState(false);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const refreshRuns = useCallback(() => {
    void request<unknown>(RUNS_URL, {
      meta: { specKey: STATION, operation: "runs.list" },
    }).then((r) => {
      if (!alive.current) return;
      if (!r.ok) {
        setRefusal(parseRefusal(describeAppError(errorOf(r))));
        return;
      }
      const body = okValue(r);
      if (!isRecord(body)) return;
      setRuns((body.runs as RunSummary[]) ?? []);
      setSources((body.promoted_sources as PromotedSource[]) ?? []);
    });
  }, []);

  const load = useCallback((id: string) => {
    setLoading(true);
    void request<unknown>(runUrl(id), {
      meta: { specKey: STATION, operation: "runs.get" },
    }).then((r) => {
      if (!alive.current) return;
      setLoading(false);
      if (!r.ok) {
        setRefusal(parseRefusal(describeAppError(errorOf(r))));
        return;
      }
      const body = okValue(r);
      if (isRecord(body) && isRecord(body.run)) {
        setRun(body.run as unknown as RunState);
      }
    });
  }, []);

  const select = useCallback(
    (id: string | null) => {
      setRunId(id);
      setRefusal(null);
      setRun(null);
      if (id) load(id);
    },
    [load],
  );

  useEffect(() => {
    refreshRuns();
  }, [refreshRuns]);

  /**
   * Every mutation goes through here. On success it adopts the run the server
   * returned (never a locally patched copy); on failure it keeps the run it
   * already had and shows the refusal — a refused edit must not blank the
   * screen the operator is editing from.
   */
  const call = useCallback(
    (
      op: string,
      url: string,
      method: string,
      body: unknown,
      segment?: string,
    ): Promise<Record<string, unknown> | null> => {
      setRefusal(null);
      if (segment) setSegmentBusy(segment);
      else setBusy(op);
      return request<unknown>(url, {
        method,
        body: JSON.stringify(body ?? {}),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: STATION, operation: op },
      }).then((r) => {
        if (!alive.current) return null;
        if (segment) setSegmentBusy(null);
        else setBusy(null);
        if (!r.ok) {
          setRefusal(parseRefusal(describeAppError(errorOf(r))));
          // The structured form (with an authoring gap's raw reply) is on the
          // run itself — re-read it so a reload-proof record reaches the panel.
          if (runId) load(runId);
          return null;
        }
        const payload = okValue(r);
        if (isRecord(payload)) {
          if (isRecord(payload.run)) setRun(payload.run as unknown as RunState);
          return payload;
        }
        return null;
      });
    },
    [load, runId],
  );

  const createRun = useCallback(
    (body: CreateRunBody) =>
      call("runs.create", RUNS_URL, "POST", body).then((payload) => {
        if (!payload) return null;
        const id = String(payload.run_id ?? "");
        if (id) {
          setRunId(id);
          refreshRuns();
        }
        return id || null;
      }),
    [call, refreshRuns],
  );

  const authorPlot = useCallback(
    (inputText: string) => {
      if (!runId) return;
      void call("plot.author", `${runUrl(runId)}/plot`, "POST", {
        input_text: inputText,
      });
    },
    [call, runId],
  );

  const putPlot = useCallback(
    (json: unknown) => {
      if (!runId) return;
      void call("plot.put", `${runUrl(runId)}/plot`, "PUT", json);
    },
    [call, runId],
  );

  const authorScreenplay = useCallback(() => {
    if (!runId) return;
    void call("screenplay.author", `${runUrl(runId)}/screenplay`, "POST", {});
  }, [call, runId]);

  const putScreenplay = useCallback(
    (json: unknown) => {
      if (!runId) return;
      void call("screenplay.put", `${runUrl(runId)}/screenplay`, "PUT", json);
    },
    [call, runId],
  );

  const putAudioMaster = useCallback(
    (json: unknown) => {
      if (!runId) return;
      void call("audio.put", `${runUrl(runId)}/audio_master`, "PUT", json);
    },
    [call, runId],
  );

  const buildPreproduction = useCallback(() => {
    if (!runId) return;
    void call("preproduction", `${runUrl(runId)}/preproduction`, "POST", {});
  }, [call, runId]);

  const lockRun = useCallback(
    (audioMaster?: unknown) => {
      if (!runId) return;
      void call(
        "lock",
        `${runUrl(runId)}/lock`,
        "POST",
        audioMaster ? { audio_master: audioMaster } : {},
      );
    },
    [call, runId],
  );

  const revise = useCallback(
    (reason: string) => {
      if (!runId) return;
      void call("revise", `${runUrl(runId)}/revise`, "POST", { reason });
    },
    [call, runId],
  );

  const compileSegments = useCallback(() => {
    if (!runId) return;
    void call("segments.compile", `${runUrl(runId)}/segments`, "POST", {});
  }, [call, runId]);

  const regenerate = useCallback(
    (segmentId: string, kind: string) => {
      if (!runId) return;
      void call(
        "segments.generate",
        `${runUrl(runId)}/segments/${encodeURIComponent(segmentId)}/generate`,
        "POST",
        { kind },
        segmentId,
      );
    },
    [call, runId],
  );

  const promote = useCallback(
    (segmentId: string | null, text: string, note: string) => {
      if (!runId) return;
      void call("promote", `${runUrl(runId)}/promote`, "POST", {
        segment_id: segmentId,
        text,
        note,
      }).then((payload) => {
        if (payload) refreshRuns();
        void request<unknown>(SOURCES_URL, {
          meta: { specKey: STATION, operation: "sources.list" },
        }).then((r) => {
          if (alive.current && r.ok) {
            const body = okValue(r);
            if (isRecord(body)) setSources((body.sources as PromotedSource[]) ?? []);
          }
        });
      });
    },
    [call, refreshRuns, runId],
  );

  const clearRefusal = useCallback(() => setRefusal(null), []);

  return useMemo(
    () => ({
      runs,
      sources,
      run,
      runId,
      busy,
      segmentBusy,
      refusal,
      loading,
      select,
      refreshRuns,
      createRun,
      authorPlot,
      putPlot,
      authorScreenplay,
      putScreenplay,
      putAudioMaster,
      buildPreproduction,
      lockRun,
      revise,
      compileSegments,
      regenerate,
      promote,
      clearRefusal,
    }),
    [
      runs,
      sources,
      run,
      runId,
      busy,
      segmentBusy,
      refusal,
      loading,
      select,
      refreshRuns,
      createRun,
      authorPlot,
      putPlot,
      authorScreenplay,
      putScreenplay,
      putAudioMaster,
      buildPreproduction,
      lockRun,
      revise,
      compileSegments,
      regenerate,
      promote,
      clearRefusal,
    ],
  );
}
