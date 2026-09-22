// Tiny seam letting the shell-level session-library sidebar STAGE a request into
// the Generate station's ordered composer while Generate is mounted — the same
// entry point the Run/Submit path already reads (model + mode + ordered parts),
// pre-filled. There is NO second generation pipeline: staging just drives the
// station's own setters, then the user reviews and hits Run.
//
// Single-handler by design: exactly one GenerateStation is ever mounted (the
// /generate route). Because Continue / Replicate fire from the shell sidebar on
// ANY route — before Generate has mounted — a stage request raised while no
// handler is registered is BUFFERED and drained the instant the station
// registers (the sidebar navigates to /generate alongside the call). No deps;
// never touches server media.
import type { MediaRef } from "./contract";

export interface PendingPart {
  ref: MediaRef;
  origin: string;
  label?: string;
}

/**
 * A staged composer request. Every field is optional and additive: `model` /
 * `mode` drive the station's knobs, `text` appends a text prompt part, `parts`
 * appends media parts (in order), and `replace` clears the composer first so a
 * Replicate / Continue stages a clean prompt rather than appending to whatever
 * was already there.
 */
export interface StageRequest {
  model?: string;
  mode?: "image" | "scene";
  text?: string;
  parts?: PendingPart[];
  replace?: boolean;
}

type StageHandler = (req: StageRequest) => void;
let handler: StageHandler | null = null;
let pending: StageRequest[] = [];

/**
 * GenerateStation registers its stager while mounted. On register we drain any
 * buffered request (raised from the sidebar before the station mounted). Returns
 * an unregister.
 */
export function registerComposer(fn: StageHandler): () => void {
  handler = fn;
  if (pending.length > 0) {
    const queued = pending;
    pending = [];
    for (const req of queued) fn(req);
  }
  return () => {
    if (handler === fn) handler = null;
  };
}

/** Ask the mounted composer to apply `req`. Buffered if Generate isn't mounted yet. */
export function requestStage(req: StageRequest): void {
  if (handler) handler(req);
  else pending.push(req);
}

/** Back-compat convenience: append a single media part to the composer. */
export function requestAddPart(part: PendingPart): void {
  requestStage({ parts: [part] });
}
