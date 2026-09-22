// src/stations/studio/useReconstruction.ts
import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { request, okValue, errorOf, describeAppError } from "../../transport/client";
import {
  jobStatusUrl,
  identityReconstructionUrl,
  identityCanonicalUrl,
} from "../../config";
import { jobRecordSchema, isTerminal } from "../../video/contract";
import type { JobStatus } from "../../video/contract";

export const DEFAULT_VIEWS = ["front", "three_quarter", "profile", "back"] as const;
export type ViewName = (typeof DEFAULT_VIEWS)[number];

export const VIEW_LABELS: Record<string, string> = {
  front: "Front",
  three_quarter: "¾ view",
  profile: "Profile",
  back: "Back",
};

const reconEnqueueSchema = z
  .object({
    recon_id: z.string().optional(),
    job_id: z.union([z.string(), z.array(z.string())]).optional(),
    job_ids: z.array(z.string()).optional(),
  })
  .passthrough();

export interface ReconstructionRun {
  prompt?: string;
  views?: string[];
  seed?: number;
  mode?: "sheet" | "turntable" | "angle-ring";
  angle_step_deg?: number;
  elevations_deg?: number[];
  reference_images?: string[];
}

export interface PromoteResult {
  ok: boolean;
  message?: string;
}

export interface ReconstructionApi {
  running: boolean;
  jobIds: string[];
  reconId: string | null;
  done: number;
  total: number;
  error: string | null;
  
  generate: (run: ReconstructionRun) => void;
  promote: (reconId: string, viewIndices: number[]) => Promise<PromoteResult>;
  reset: () => void;
  
  // Angle-ring specific operations
  setViewStatus: (reconId: string, viewId: string, status: "approved" | "rejected") => Promise<PromoteResult>;
  regenerateView: (reconId: string, viewId: string, prompt?: string, seed?: number, useNeighbors?: boolean) => void;
  buildMesh: (reconId: string, viewIds: string[]) => void;
}

export function useReconstruction(slug: string, onComplete: () => void): ReconstructionApi {
  const [running, setRunning] = useState(false);
  const [jobIds, setJobIds] = useState<string[]>([]);
  const [statuses, setStatuses] = useState<Record<string, JobStatus>>({});
  const [reconId, setReconId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mounted = useRef(true);
  const timer = useRef<number | null>(null);
  const idsRef = useRef<string[]>([]);
  const statusesRef = useRef<Record<string, JobStatus>>({});
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const stopPoll = useCallback(() => {
    if (timer.current != null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      stopPoll();
    };
  }, [stopPoll]);

  const poll = useCallback(async () => {
    const ids = idsRef.current;
    if (ids.length === 0) return;
    const records = await Promise.all(
      ids.map(async (id) => {
        const res = await request<unknown>(jobStatusUrl(id), {
          meta: { specKey: "studio", operation: "identity.reconstruction.status" },
        });
        if (!res.ok) return null;
        const parsed = jobRecordSchema.safeParse(okValue(res));
        return parsed.success ? parsed.data : null;
      }),
    );
    if (!mounted.current) return;
    const merged: Record<string, JobStatus> = { ...statusesRef.current };
    let firstError: string | null = null;
    records.forEach((jr, i) => {
      if (!jr) return;
      merged[ids[i]] = jr.status ?? null;
      if ((jr.status === "failed" || jr.status === "cancelled") && firstError == null) {
        firstError = jr.result?.error?.message ?? `Job ${jr.status}.`;
      }
    });
    statusesRef.current = merged;
    setStatuses(merged);
    if (firstError != null) setError(firstError);
    if (ids.every((id) => isTerminal(merged[id] ?? null))) {
      stopPoll();
      setRunning(false);
      onCompleteRef.current();
    }
  }, [stopPoll]);

  // Shared response handler for anything that enqueues jobs (generate, regenerate, buildMesh)
  const handleEnqueueResponse = useCallback((r: Awaited<ReturnType<typeof request<unknown>>>) => {
    if (!mounted.current) return;
    if (!r.ok) {
      setError(describeAppError(errorOf(r)));
      setRunning(false);
      return;
    }
    const parsed = reconEnqueueSchema.safeParse(okValue(r));
    if (!parsed.success) {
      setError("Malformed reconstruction response.");
      setRunning(false);
      return;
    }
    const d = parsed.data;
    if (d.recon_id) setReconId(d.recon_id);
    const ids: string[] = [];
    if (typeof d.job_id === "string") ids.push(d.job_id);
    else if (Array.isArray(d.job_id)) ids.push(...d.job_id);
    if (Array.isArray(d.job_ids)) ids.push(...d.job_ids);
    const uniq = Array.from(new Set(ids.filter((x) => typeof x === "string" && x !== "")));
    if (uniq.length === 0) {
      setRunning(false);
      onCompleteRef.current();
      return;
    }
    idsRef.current = uniq;
    setJobIds(uniq);
    const seeded: Record<string, JobStatus> = {};
    uniq.forEach((id) => (seeded[id] = "queued"));
    statusesRef.current = seeded;
    setStatuses(seeded);
    void poll();
    timer.current = window.setInterval(() => void poll(), 2000);
  }, [poll]);

  const generate = useCallback(
    (run: ReconstructionRun) => {
      stopPoll();
      setError(null);
      setReconId(null);
      setJobIds([]);
      setStatuses({});
      statusesRef.current = {};
      idsRef.current = [];
      setRunning(true);

      const body: Record<string, unknown> = {};
      if (run.prompt != null && run.prompt.trim() !== "") body.prompt = run.prompt.trim();
      if (run.views && run.views.length > 0) body.views = run.views;
      if (run.seed != null && Number.isFinite(run.seed)) body.seed = run.seed;
      if (run.mode) body.mode = run.mode;
      if (run.angle_step_deg != null) body.angle_step_deg = run.angle_step_deg;
      if (run.elevations_deg != null) body.elevations_deg = run.elevations_deg;
      if (run.reference_images != null) body.reference_images = run.reference_images;

      void request<unknown>(identityReconstructionUrl(slug), {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "identity.reconstruction.enqueue" },
      }).then(handleEnqueueResponse);
    },
    [slug, handleEnqueueResponse, stopPoll],
  );

  const setViewStatus = useCallback(
    async (rid: string, viewId: string, status: "approved" | "rejected"): Promise<PromoteResult> => {
      const base = identityReconstructionUrl(slug);
      const res = await request<unknown>(`${base}/${rid}/views/${viewId}`, {
        method: "PATCH",
        body: JSON.stringify({ status }),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "identity.view.status" },
      });
      if (!res.ok) return { ok: false, message: describeAppError(errorOf(res)) };
      onCompleteRef.current(); // reload profile so UI marks it approved immediately
      return { ok: true };
    },
    [slug]
  );

  const regenerateView = useCallback(
    (rid: string, viewId: string, prompt?: string, seed?: number, useNeighbors: boolean = true) => {
      stopPoll();
      setError(null);
      setJobIds([]);
      setStatuses({});
      statusesRef.current = {};
      idsRef.current = [];
      setRunning(true);

      const body: Record<string, unknown> = { use_nearest_approved_neighbors: useNeighbors };
      if (prompt) body.prompt = prompt.trim();
      if (seed != null) body.seed = seed;

      const base = identityReconstructionUrl(slug);
      void request<unknown>(`${base}/${rid}/views/${viewId}/regenerate`, {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "identity.view.regenerate" },
      }).then(handleEnqueueResponse);
    },
    [slug, handleEnqueueResponse, stopPoll]
  );

  const buildMesh = useCallback(
    (rid: string, viewIds: string[]) => {
      stopPoll();
      setError(null);
      setJobIds([]);
      setStatuses({});
      statusesRef.current = {};
      idsRef.current = [];
      setRunning(true);

      const base = identityReconstructionUrl(slug);
      void request<unknown>(`${base}/${rid}/mesh`, {
        method: "POST",
        body: JSON.stringify({ views: viewIds }),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "identity.mesh.build" },
      }).then(handleEnqueueResponse);
    },
    [slug, handleEnqueueResponse, stopPoll]
  );

  const promote = useCallback(
    async (rid: string, viewIndices: number[]): Promise<PromoteResult> => {
      const res = await request<unknown>(identityCanonicalUrl(slug), {
        method: "POST",
        body: JSON.stringify({ recon_id: rid, views: viewIndices }),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "identity.canonical.promote" },
      });
      if (!res.ok) return { ok: false, message: describeAppError(errorOf(res)) };
      return { ok: true };
    },
    [slug]
  );

  const reset = useCallback(() => {
    stopPoll();
    setRunning(false);
    setJobIds([]);
    setStatuses({});
    statusesRef.current = {};
    idsRef.current = [];
    setReconId(null);
    setError(null);
  }, [stopPoll]);

  const total = jobIds.length;
  const done = jobIds.filter((id) => isTerminal(statuses[id] ?? null)).length;

  return { running, jobIds, reconId, done, total, error, generate, setViewStatus, regenerateView, buildMesh, promote, reset };
}