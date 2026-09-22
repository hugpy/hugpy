// k94 — the ONE-PATH "Create identity" job watcher.
//
// Both create routes (from-video → identity_from_video, from-images →
// identity_mesh_build) hand back a bus job_id; this hook polls the generic
// GET /video/jobs/<id> (`videoJobUrl`) the way the other identity flows in this
// station do, and reduces the reply to what the panel shows on ONE line:
//   stage · % · log tail
// `progress` is the relay's mirrored blob ({stage, progress 0..1, log_tail[]}
// — additive fields an older render service may omit), `progress_ratio` is the
// bus's numeric bar, `result` is the terminal JobResult ({ok, error?,
// identities?}). Terminal = `result.ok` is a boolean. Bounded (90 min — char360
// plus a textured Hunyuan3D mesh PER character runs minutes each; clownworld's
// own ceiling) so a wedged remote never spins the panel forever.
//
// Why not the session job tracker (video/jobTracker.ts)? Its JobKind/StationId
// unions + per-kind terminal parse are closed over the crop/frames/generate
// arms; widening them is a cross-station change outside this task's files. The
// poll shape here is the same one ExtractFromVideoPanel / generateFullIdentity
// already use in this station.
import { useCallback, useEffect, useRef, useState } from "react";
import { request, okValue } from "../../transport/client";
import { videoJobUrl } from "../../config";

export type CreateJobPhase = "idle" | "queued" | "running" | "done" | "error";

export interface CreatedIdentity {
  char?: string;
  slug: string;
  name?: string;
  glb?: boolean;
  n_views?: number;
  canonical?: number;
  error?: string;
}

export interface CreateJobView {
  phase: CreateJobPhase;
  jobId: string | null;
  /** The remote stage ("detect", "mesh", …) or the bus's current_stage. */
  stage: string | null;
  /** 0..100 when the job reports anything measurable, else null. */
  pct: number | null;
  /** The last line of the relay's mirrored log tail. */
  logLine: string | null;
  /** Human error text on failure (enqueue or job). */
  error: string | null;
  /** from-video: the profiles the job minted (empty for from-images). */
  identities: CreatedIdentity[];
}

const IDLE: CreateJobView = {
  phase: "idle", jobId: null, stage: null, pct: null, logLine: null, error: null, identities: [],
};

const POLL_MS = 4000;
const CAP_MS = 90 * 60 * 1000;

type JobReply = {
  status?: string | null;
  progress?: { stage?: unknown; progress?: unknown; log_tail?: unknown; message?: unknown } | null;
  progress_ratio?: number | null;
  current_stage?: string | null;
  result?: {
    ok?: boolean;
    error?: { message?: string } | null;
    identities?: { profiles?: CreatedIdentity[] } | null;
  } | null;
};

export interface CreateJobApi {
  view: CreateJobView;
  /** Start watching a job; `onDone(identities)` fires once on success. */
  watch: (jobId: string, onDone?: (ids: CreatedIdentity[]) => void) => void;
  /** Surface an enqueue-time failure on the same line. */
  fail: (message: string) => void;
  reset: () => void;
}

export function useIdentityCreateJob(): CreateJobApi {
  const [view, setView] = useState<CreateJobView>(IDLE);
  const timer = useRef<number | null>(null);
  const active = useRef(false);

  const stop = useCallback(() => {
    active.current = false;
    if (timer.current != null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  useEffect(() => stop, [stop]);

  const watch = useCallback<CreateJobApi["watch"]>(
    (jobId, onDone) => {
      stop();
      active.current = true;
      setView({ ...IDLE, phase: "queued", jobId });
      const deadline = Date.now() + CAP_MS;

      const pollOnce = async () => {
        if (!active.current) return;
        if (Date.now() > deadline) {
          stop();
          setView((v) => ({ ...v, phase: "error", error: "Timed out waiting for the identity build." }));
          return;
        }
        const res = await request<unknown>(videoJobUrl(jobId), {
          meta: { specKey: "studio", operation: "identity.create.status" },
        });
        if (!active.current) return;
        if (res.ok) {
          const job = okValue(res) as JobReply;
          const result = job.result;
          if (result && typeof result.ok === "boolean") {
            stop();
            if (result.ok) {
              const ids = (result.identities && Array.isArray(result.identities.profiles)
                ? result.identities.profiles
                : []).filter((p) => p && typeof p.slug === "string");
              setView((v) => ({ ...v, phase: "done", pct: 100, identities: ids, error: null }));
              if (onDone) onDone(ids);
            } else {
              setView((v) => ({
                ...v,
                phase: "error",
                error: (result.error && result.error.message) || "The identity build failed.",
              }));
            }
            return;
          }
          const blob = job.progress || {};
          const stage =
            (typeof blob.stage === "string" && blob.stage) ||
            (typeof blob.message === "string" && blob.message) ||
            job.current_stage ||
            null;
          const ratio =
            typeof job.progress_ratio === "number"
              ? job.progress_ratio
              : typeof blob.progress === "number"
                ? blob.progress
                : null;
          const tail = Array.isArray(blob.log_tail) ? blob.log_tail : [];
          const last = tail.length ? tail[tail.length - 1] : null;
          setView((v) => ({
            ...v,
            phase: job.status === "running" ? "running" : "queued",
            stage,
            pct: ratio == null ? null : Math.max(0, Math.min(100, Math.round(ratio * 100))),
            logLine: typeof last === "string" ? last : null,
          }));
        }
        // queued / running / transient poll failure → keep polling.
        timer.current = window.setTimeout(() => void pollOnce(), POLL_MS);
      };
      timer.current = window.setTimeout(() => void pollOnce(), POLL_MS);
    },
    [stop],
  );

  const fail = useCallback<CreateJobApi["fail"]>((message) => {
    stop();
    setView({ ...IDLE, phase: "error", error: message });
  }, [stop]);

  const reset = useCallback(() => {
    stop();
    setView(IDLE);
  }, [stop]);

  return { view, watch, fail, reset };
}

/** The one-line progress text: "stage · 42% · last log line". */
export function describeCreateJob(v: CreateJobView): string | null {
  if (v.phase === "idle") return null;
  if (v.phase === "error") return v.error || "failed";
  if (v.phase === "done") return "done";
  const parts: string[] = [v.phase === "queued" && !v.stage ? "queued" : (v.stage || "running")];
  if (v.pct != null) parts.push(`${v.pct}%`);
  if (v.logLine) parts.push(v.logLine);
  return parts.join(" · ");
}
