// Tiny seam letting the shell-level session-library sidebar STAGE a source clip
// into the Studio Clips station's generate form while (or before) it is mounted —
// the B2 movie->studio chain's "Send to Studio". It mirrors composerBridge exactly
// but for a DIFFERENT station with a DIFFERENT staging contract: the Generate
// composer takes ordered text/media PARTS (composerBridge), whereas the studio
// generate form takes ONE prior-tier VIDEO to extend. Keeping them as two small,
// self-contained seams (each registered by its own station) is cleaner than
// overloading composerBridge's StageRequest with a studio mode — the handlers,
// payloads, and mounting stations share nothing.
//
// Single-handler by design: exactly one StudioClipsStation is ever mounted (the
// /studio-clips route). Because "Send to Studio" fires from the shell sidebar on
// ANY route — before Studio Clips has mounted — a stage request raised while no
// handler is registered is BUFFERED and drained the instant the station registers
// (the sidebar navigates to /studio-clips alongside the call). No deps; never
// touches server media.
import type { MediaRef } from "./contract";
import type { StudioItemProvenance } from "./mediaLibrary";

/**
 * How the studio should consume the staged clip (slice (a) / v2v):
 *   • "extend"  — i2v: continue the clip FROM ITS LAST FRAME (the original B2 flow).
 *   • "restyle" — v2v: repaint/transform the WHOLE clip via the VACE control model.
 * Capability-aware "Send to Studio": the sidebar can pre-pick a mode, and the
 * Studio Clips station also lets the user toggle it before generating.
 */
export type StudioMode = "extend" | "restyle";

/**
 * A staged studio request: the prior-tier clip (a movie/scene/studio video ref)
 * to extend or restyle. The station derives the POST body from it — `source_video`
 * = the ref's abs-path uri (jail-resolved by the route), falling back to
 * `source_asset_id` = the ref's asset_id (resolved via the media catalog). Every
 * field optional + additive so the shape can grow without breaking callers.
 */
export interface StudioStageRequest {
  sourceVideo?: MediaRef;
  /** Preferred consume mode; the station defaults to "extend" when omitted. */
  mode?: StudioMode;
  /** Optional provenance forwarded from the source (e.g. the library item behind a
   *  "Send to Studio" / "Restyle") so the receiving surface can mint a first-class
   *  StudioItem carrying it. Purely additive — existing callers that omit it are
   *  unaffected. */
  provenance?: StudioItemProvenance;
}

type StudioStageHandler = (req: StudioStageRequest) => void;
// A STACK of registered handlers (mount-aware). There are now TWO surfaces that can
// host the studio generate form — the standalone Studio station's Generate tab and
// the Generate station's "studio" mode — so a single overwriteable slot could let
// two mounts fight. The stack makes the seam robust: the MOST-RECENTLY registered
// mount is the active target, and on unmount its handler is removed (restoring the
// previous one as active) rather than blindly nulling a slot another mount may own.
// In practice the two surfaces live on different routes and are never mounted at the
// same time, but this keeps the contract correct regardless of mount order.
let handlers: StudioStageHandler[] = [];
let pending: StudioStageRequest[] = [];

/**
 * A studio surface registers its stager while mounted. On register we drain any
 * buffered request (raised from the sidebar before any surface mounted). Returns an
 * unregister that removes THIS handler (not whatever is currently on top).
 */
export function registerStudioStager(fn: StudioStageHandler): () => void {
  handlers.push(fn);
  if (pending.length > 0) {
    const queued = pending;
    pending = [];
    for (const req of queued) fn(req);
  }
  return () => {
    handlers = handlers.filter((h) => h !== fn);
  };
}

/** Ask the active mounted studio surface to apply `req`. Buffered if none is mounted
 *  yet (drained by the next surface that registers). */
export function requestStudioStage(req: StudioStageRequest): void {
  const active = handlers[handlers.length - 1];
  if (active) active(req);
  else pending.push(req);
}
