// CINEMA SESSIONS — the durable movie sessions on the studio-movies root, plus the
// two actions on one (pause / resume).
//
// WHY A SESSION AND NOT A JOB. A studio movie is ONE bus job that can run for hours,
// and the job id was the only handle on it: when the job ended — cancelled, reaped,
// or simply lost with the browser tab — the rendered segments stayed on disk and
// became unreachable. GET /video/studio/movies lists the DIRS instead, so a session
// outlives the job that made it; its id is the dir leaf and therefore survives a
// resume (which mints a NEW job id). See video_routes.py's `studio/movies` block.
//
// Cadence + shape follow the studio arm's existing list hooks (useStudioClips'
// background poll, useMediaJobs' mounted-ref guard): the poll runs ONLY while a
// consumer asks for it (`active` — the sessions panel is collapsed by default, and a
// filesystem walk is not worth paying for behind a closed panel), a failed poll keeps
// the last-good list rather than flashing empty, and a late response never setStates
// after unmount.
import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import {
  hugpyConfig,
  mediaBytesUrl,
  studioMoviePauseUrl,
  studioMovieResumeUrl,
} from "../config";
import { request, okValue, errorOf, describeAppError } from "../transport/client";

// 6s — the same background cadence useStudioClips polls the clip catalog at. A movie
// segment takes minutes, so anything faster only costs a filesystem walk.
const POLL_MS = 6000;

// The bus statuses that mean "this session's job is still going to do something" —
// the client-side mirror of the route's _MOVIE_INFLIGHT_STATES, used to decide
// whether a row gets a Pause affordance.
const IN_FLIGHT = new Set(["queued", "claimed", "running", "cancelling"]);

// ── wire schemas ────────────────────────────────────────────────────────────
// Tolerant/passthrough like the movie composer's local schemas: every field beyond
// the ones we render is optional, so a drifted-forward payload never breaks the
// parse and one odd row never blanks the list.
const segmentErrorSchema = z
  .object({
    code: z.string().nullable().optional(),
    message: z.string().nullable().optional(),
  })
  .passthrough();

const movieSegmentSchema = z
  .object({
    index: z.number(),
    segment_id: z.string().nullable().optional(),
    prompt: z.string().nullable().optional(),
    status: z.string().nullable().optional(),
    resumed: z.boolean().nullable().optional(),
    frames: z.number().nullable().optional(),
    duration_s: z.number().nullable().optional(),
    error: segmentErrorSchema.nullable().optional(),
    /** Honest "can I watch this right now" — false for a failed segment AND for one
     *  whose record exists but whose bytes are gone (the url is then omitted). */
    clip_available: z.boolean().nullable().optional(),
    media: z.string().nullable().optional(),
  })
  .passthrough();

export type MovieSegment = z.infer<typeof movieSegmentSchema>;

const movieSessionSchema = z
  .object({
    movie_id: z.string(),
    job_id: z.string().nullable().optional(),
    job_status: z.string().nullable().optional(),
    title: z.string().nullable().optional(),
    project: z.string().nullable().optional(),
    status: z.string().nullable().optional(),
    segments_completed: z.number().nullable().optional(),
    segments_total: z.number().nullable().optional(),
    segments: z.array(movieSegmentSchema).optional(),
    /** Server-computed: there is a persisted spec to re-enqueue AND the work is
     *  neither finished nor already in flight. The Resume affordance keys on this
     *  rather than re-deriving the rule per client. */
    resumable: z.boolean().nullable().optional(),
    updated: z.number().nullable().optional(),
    width: z.number().nullable().optional(),
    height: z.number().nullable().optional(),
    fps: z.number().nullable().optional(),
    id_lock: z.boolean().nullable().optional(),
    /** The assembled movie.mp4, when one was stitched AND still exists on disk. */
    movie: z.string().nullable().optional(),
  })
  .passthrough();

export type MovieSession = z.infer<typeof movieSessionSchema>;

const moviesResponseSchema = z
  .object({ movies: z.array(movieSessionSchema) })
  .passthrough();

// ── derived reads (one definition each, shared by the panel) ────────────────

/**
 * A playable `<video src>` for a wire media url (a segment clip or the assembled
 * movie), or null when there is nothing to serve.
 *
 * The route hands over a ready `/video/media?handle=<path>` url, but every other
 * element-src in this console goes through `mediaBytesUrl` — which is where the
 * share-session credential is stamped into the query (an element load carries no
 * header) and where canned-demo handles pass through. So the HANDLE is unwrapped
 * and re-issued through that one seam rather than the url being used raw, and a
 * shape this parser doesn't recognise degrades to "not playable" instead of a
 * link that 404s.
 */
export function movieMediaSrc(wire: string | null | undefined): string | null {
  if (!wire) return null;
  const q = wire.indexOf("?");
  if (q < 0) return null;
  const handle = new URLSearchParams(wire.slice(q + 1)).get("handle");
  return handle ? mediaBytesUrl(handle) : null;
}

/** The segment's clip src — null unless the route says its bytes are on disk NOW. */
export function segmentClipSrc(seg: MovieSegment): string | null {
  if (!seg.clip_available) return null;
  return movieMediaSrc(seg.media);
}

/** True when this session's CURRENT job is still on the bus — the Pause affordance. */
export function isSessionLive(s: MovieSession): boolean {
  return IN_FLIGHT.has(s.job_status ?? "");
}

/**
 * WHY this session cannot be resumed right now, or null when it can. The rule is the
 * server's (`resumable`); this only names the reason, so a disabled button always says
 * what it is waiting for instead of being silently dead.
 */
export function resumeBlockedReason(s: MovieSession): string | null {
  if (s.resumable) return null;
  if (s.status === "done") return "This movie finished — there is nothing left to render.";
  if (isSessionLive(s) || s.status === "running") {
    return "This movie is already running — pause it first.";
  }
  return (
    "This movie has no persisted spec.json, so it cannot be re-enqueued (it predates " +
    "resumable sessions); its rendered segments are still listed and playable."
  );
}

export interface MovieSessionsState {
  sessions: MovieSession[];
  loading: boolean;
  /** List-level failure (the poll keeps the last-good list; this is the foreground read). */
  error: string | null;
  /** Force an immediate reload (fired right after a pause/resume lands). */
  reload: () => void;
  /** Movie ids with a pause/resume in flight — the per-row button's busy state. */
  busy: Set<string>;
  /** Per-movie action failure, kept VERBATIM from the route (the 409 bodies say why). */
  actionError: Record<string, string>;
  pause: (movieId: string) => void;
  resume: (movieId: string) => void;
}

/**
 * The Cinema session list + its two actions. `active` gates the poll: pass the
 * panel's own visibility so a collapsed panel costs nothing (the route walks the
 * movies root on every call).
 */
export function useMovieSessions(active: boolean): MovieSessionsState {
  const [sessions, setSessions] = useState<MovieSession[]>([]);
  // Starts TRUE so a freshly-opened panel reads "reading the movies root…" instead of
  // flashing "no sessions" for one round trip — to an operator looking for a render
  // they think they lost, those two say opposite things.
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<Set<string>>(new Set());
  const [actionError, setActionError] = useState<Record<string, string>>({});

  const mounted = useRef(true);
  const inFlight = useRef(false);
  // Only the NEWEST read may write the list. Without this a poll issued BEFORE a
  // pause/resume can land after it and restore the pre-action rows, making the action
  // look like it did nothing for a whole tick.
  const seq = useRef(0);
  // First load of a newly-opened panel shows a spinner; every poll after it is
  // silent, so an open panel never flickers between ticks.
  const loaded = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // `force` is for the reload that follows a pause/resume: it must not be swallowed by
  // the in-flight guard that keeps the 6s poll from stacking up on a slow route.
  const load = useCallback(async (force = false) => {
    if (inFlight.current && !force) return;
    inFlight.current = true;
    const mine = (seq.current += 1);
    if (!loaded.current) setLoading(true);
    try {
      const res = await request<unknown>(hugpyConfig.studioMoviesUrl, {
        meta: { specKey: "studio", operation: "studio.movies.list" },
      });
      if (!mounted.current || mine !== seq.current) return;
      if (!res.ok) {
        // A failed poll must not wipe a working list — only the FIRST read (nothing
        // on screen yet) is allowed to surface as an error.
        if (!loaded.current) setError(describeAppError(errorOf(res)));
        return;
      }
      const parsed = moviesResponseSchema.safeParse(okValue(res));
      if (!parsed.success) {
        if (!loaded.current) setError("Malformed movie-sessions response.");
        return;
      }
      loaded.current = true;
      setError(null);
      setSessions(parsed.data.movies);
    } finally {
      inFlight.current = false;
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!active) return;
    void load();
    const id = window.setInterval(() => void load(), POLL_MS);
    return () => window.clearInterval(id);
  }, [active, load]);

  // Pause and resume differ only in their url and their telemetry op, so they share
  // ONE runner: mark the row busy, clear its last error, POST, keep the route's own
  // message verbatim on failure (the 409s — "no persisted spec.json", "already
  // running" — ARE the explanation), then reload so the row's new status is the
  // server's answer rather than an optimistic guess.
  const act = useCallback(
    async (movieId: string, url: string, operation: string) => {
      setBusy((prev) => new Set(prev).add(movieId));
      setActionError((prev) => {
        const next = { ...prev };
        delete next[movieId];
        return next;
      });
      try {
        const res = await request<unknown>(url, {
          method: "POST",
          meta: { specKey: "studio", operation },
        });
        if (!mounted.current) return;
        if (!res.ok) {
          setActionError((prev) => ({
            ...prev,
            [movieId]: describeAppError(errorOf(res)),
          }));
          return;
        }
        await load(true);
      } finally {
        if (mounted.current) {
          setBusy((prev) => {
            const next = new Set(prev);
            next.delete(movieId);
            return next;
          });
        }
      }
    },
    [load],
  );

  const pause = useCallback(
    (movieId: string) => {
      void act(movieId, studioMoviePauseUrl(movieId), "studio.movie.pause");
    },
    [act],
  );

  const resume = useCallback(
    (movieId: string) => {
      void act(movieId, studioMovieResumeUrl(movieId), "studio.movie.resume");
    },
    [act],
  );

  const reload = useCallback(() => {
    void load();
  }, [load]);

  return { sessions, loading, error, reload, busy, actionError, pause, resume };
}
