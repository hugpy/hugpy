// COMPATIBILITY DERIVATIONS over the measured render-preset table (useRenderPresets).
//
// THE RULE THIS MODULE IMPLEMENTS (operator ruling 2026-07-29): the studio must be
// COMPATIBILITY-AWARE of all user inputs — every input the user makes filters the
// options subsequently offered — and a preset must be a WORKFLOW guaranteed to render,
// not a canned scene, absorbing the session's inputs into their proper slots.
//
// Everything here is a PURE function of the payload + the current form state. No React,
// no fetch, no component coupling — the Clip surface and the Movie composer both derive
// their pickers from these, so the two surfaces can never disagree about what the fleet
// can do.
//
// ── THE COURTESY-LAYER BOUNDARY, STATED ONCE ────────────────────────────────────────
// These filters decide what to OFFER. They are NOT a gate and must never be treated as
// one: POST /video/studio/i2v's capability gate (video_routes.py) and the router's own
// refusal stay authoritative, and every consumer here falls back to its UNFILTERED list
// when the discovery fetch has not landed (`loaded === false`). A filter that empties a
// picker because a GET was slow is a worse failure than the one it prevents.
//
// ── WHY SOME RULES ARE NOT IN THE PRESET TABLE ──────────────────────────────────────
// A `RenderPreset.inputs[]` names what the caller MUST supply to reach that binding. The
// route additionally accepts several OPTIONAL inputs the table has no field for, and
// refuses some combinations the table cannot express. Both are transcribed below with
// the route line that owns them, because a UI that only read `inputs[]` would delete
// working features (v2v scene-continuity would lose its identity references) and offer
// broken ones (a control still beside a source clip, which the route refuses by name).
import type { RenderPreset, UnavailableCapability } from "./useRenderPresets";

/** The capability strings the studio's CLIP surface can author end to end. Every one is
 *  servable on the fleet today; `motion` is the newest (video_routes widened the
 *  control_image gate from id_lock-only on 2026-07-27, which is what turned it from a
 *  phantom into a route). Ordered as the picker shows them. */
export const CLIP_CAPABILITIES = ["t2v", "i2v", "v2v", "id_lock", "motion"] as const;
export type ClipCapability = (typeof CLIP_CAPABILITIES)[number];

/** The MOVIE composer's capability — the multi-segment orchestration stage. */
export const MOVIE_CAPABILITY = "assemble";

export const CAPABILITY_LABELS: Record<string, string> = {
  t2v: "Text → video (t2v)",
  i2v: "Image → video (i2v)",
  v2v: "Video → video (v2v restyle)",
  id_lock: "Identity lock (reference → video)",
  motion: "Pose / structure control (motion)",
  assemble: "Multi-segment movie (assemble)",
  upres: "Upscale a clip (upres)",
  interp: "Interpolate frames (interp)",
  keyframe: "Keyframe (start + end frame)",
  inpaint: "Inpaint a region",
  outpaint: "Outpaint the canvas",
  retake: "Re-take a frame range",
  audio: "Audio track",
  lipsync: "Lip sync",
  restore: "Restore / face fix",
  stream: "Streaming generation",
};

export function capabilityLabel(cap: string): string {
  return CAPABILITY_LABELS[cap] ?? cap;
}

// ── the four attachment slots the Clip surface owns ─────────────────────────────────
// Named for the ROUTE FIELD each one becomes, so the matrix below can be read against
// video_routes.py directly.
export interface Attachments {
  /** i2v `start_image` — a single first frame. */
  startImage: boolean;
  /** `source_video` — the clip to extend (i2v) or repaint (v2v). */
  sourceVideo: boolean;
  /** `reference_images` — 1–4 ordered stills (identity). */
  referenceImages: boolean;
  /** `control_image` (+ `control_kind`) — one pose/depth/sketch still. */
  controlImage: boolean;
}

export const NO_ATTACHMENTS: Attachments = {
  startImage: false,
  sourceVideo: false,
  referenceImages: false,
  controlImage: false,
};

// WHICH CAPABILITIES ACCEPT WHICH INPUT — transcribed from the route's own gates so the
// UI narrows to exactly what the server admits, and never one option wider.
//
//   start_image      i2v only.        t2v DELIBERATELY drops it (video_routes: "a t2v
//                                     clip is a pure function of prompt + seed +
//                                     geometry"); the VACE capabilities have no
//                                     start-frame channel.
//   source_video     i2v | v2v.       t2v drops it; refused alongside control_image.
//   reference_images id_lock | v2v.   _REF_CAPS — "rejected for any other capability so
//                                     a non-VACE runner can never silently ignore them".
//                                     REQUIRED on id_lock, OPTIONAL on v2v (the
//                                     scene-continuity identity carry).
//   control_image    id_lock | motion. _CONTROL_CAPS — OPTIONAL composition blocking on
//                                     id_lock, REQUIRED and definitional on motion.
const ACCEPTS: Record<keyof Attachments, readonly string[]> = {
  startImage: ["i2v"],
  sourceVideo: ["i2v", "v2v"],
  referenceImages: ["id_lock", "v2v"],
  controlImage: ["id_lock", "motion"],
};

/** Human copy for "why this capability is greyed out given what you have attached". */
const ATTACHMENT_NOUNS: Record<keyof Attachments, string> = {
  startImage: "a start image",
  sourceVideo: "a source clip",
  referenceImages: "reference images",
  controlImage: "a control still",
};

// ── the table, sliced ───────────────────────────────────────────────────────────────

/** Every capability at least one ratified preset covers — what the fleet can do today. */
export function servableCapabilities(presets: RenderPreset[]): string[] {
  const out = new Set<string>();
  for (const p of presets) {
    out.add(p.capability);
    for (const c of p.capabilities ?? []) out.add(c);
  }
  return [...out].sort();
}

/** Every ratified preset serving `cap`, in the registry's own (meaningful) order. */
export function presetsForCapability(presets: RenderPreset[], cap: string): RenderPreset[] {
  return presets.filter((p) => p.capability === cap || (p.capabilities ?? []).includes(cap));
}

export function isServable(presets: RenderPreset[], cap: string): boolean {
  return presetsForCapability(presets, cap).length > 0;
}

/**
 * PROVEN = this capability has a ratified preset that has produced pixels on this fleet.
 * `false` is not a defect and must not be hidden — it is the row telling the truth about
 * itself (clip-i2v-480p is 0 of 47 landed Wan clips; clip-motion-480p is a real branch
 * the route only unlocked on 2026-07-27). The UI badges it rather than suppressing it.
 * `null` = no preset at all, i.e. unservable — a different answer from "unproven".
 */
export function provenForCapability(presets: RenderPreset[], cap: string): boolean | null {
  const rows = presetsForCapability(presets, cap);
  if (rows.length === 0) return null;
  return rows.some((p) => p.proven === true);
}

/**
 * The model ids a pin may name for `cap` without leaving the ratified set. For the movie
 * capability the segment bindings count too — `composes` names the preset ids whose
 * models a multi-segment render actually binds, so pinning one of those is legitimate
 * even though the movie row itself declares a single `model`.
 */
export function modelsForCapability(presets: RenderPreset[], cap: string): string[] {
  const rows = presetsForCapability(presets, cap);
  const byId = new Map(presets.map((p) => [p.id, p]));
  const out = new Set<string>();
  for (const p of rows) {
    out.add(p.model);
    for (const composedId of p.composes ?? []) {
      const c = byId.get(composedId);
      if (c) out.add(c.model);
    }
  }
  return [...out].sort();
}

export interface Geometry {
  width: number;
  height: number;
  fps: number;
  /** The preset the envelope came from — named in the UI so a lock is never anonymous. */
  presetId: string;
  presetTitle: string;
}

/**
 * The geometry envelope a capability renders at, or null when its presets declare none
 * (the two ffmpeg enhance rows inherit the source clip's geometry — `width: null` is the
 * table's honest encoding of that, and a 0 would be a lie with a shape).
 *
 * Every video row on this fleet is 832x480@16 today, so "the first row that declares
 * one" is unambiguous; if a second envelope is ever ratified for the same capability
 * this returns the first and `withinGeometry` widens to accept either.
 */
export function geometryForCapability(presets: RenderPreset[], cap: string): Geometry | null {
  for (const p of presetsForCapability(presets, cap)) {
    if (p.width != null && p.height != null && p.fps != null) {
      return {
        width: p.width,
        height: p.height,
        fps: p.fps,
        presetId: p.id,
        presetTitle: p.title,
      };
    }
  }
  return null;
}

/** Is w×h inside SOME ratified envelope for this capability? A capability with no
 *  declared geometry accepts anything (the enhance rows take the source's). */
export function withinGeometry(
  presets: RenderPreset[],
  cap: string,
  width: number,
  height: number,
): boolean {
  const rows = presetsForCapability(presets, cap).filter(
    (p) => p.width != null && p.height != null,
  );
  if (rows.length === 0) return true;
  if (width <= 0 || height <= 0) return false;
  return rows.some((p) => width <= (p.width as number) && height <= (p.height as number));
}

/**
 * The SMALLEST `vram_budget_gb` that can bind a real model for this capability.
 *
 * This is the number the studio console used to mirror by hand as
 * `STUDIO_REAL_FLOOR_GB = 6` — a constant that had drifted BELOW every real row in the
 * table (the cheapest is 8.2), so the honesty banner it fed under-warned. It is derived
 * here from `vram_envelope_gb`, which the backend reads off the same
 * `ModelConfig.vram` map the router compares a budget against (`cfg.vram.fits(budget)`
 * is `gb <= budget`).
 *
 * Returns null when the payload has not landed or carries no envelope — callers then
 * keep their fallback constant rather than inventing a floor.
 */
export function minBudgetGbForCapability(
  presets: RenderPreset[],
  cap: string,
): number | null {
  const vals = presetsForCapability(presets, cap)
    .map((p) => p.vram_envelope_gb)
    .filter((v): v is number => typeof v === "number" && v > 0);
  return vals.length ? Math.min(...vals) : null;
}

/** The budget to PREFILL for a capability: its cheapest binding, rounded up to a whole
 *  GB so the field reads like something a person typed and can never land a hair under
 *  the envelope through float display. */
export function suggestedBudgetGb(presets: RenderPreset[], cap: string): number | null {
  const min = minBudgetGbForCapability(presets, cap);
  return min == null ? null : Math.ceil(min);
}

// ── input-driven filtering (the "every input filters what comes next" half) ─────────

export interface CapabilityOffer {
  capability: string;
  label: string;
  /** Servable on the fleet AND compatible with what is currently attached. */
  offered: boolean;
  /** Covered by a ratified preset at all. */
  servable: boolean;
  /** Has produced pixels on this fleet (null when unservable). */
  proven: boolean | null;
  /** Why it is not offered — the backend's refusal reason for an unservable
   *  capability, or this module's attachment explanation for a servable-but-
   *  incompatible one. Empty when offered. */
  reason: string;
}

/**
 * THE MATRIX, applied: which capabilities remain reachable given what the user has
 * already attached. A capability is offered iff it is servable AND every attachment the
 * user is holding is an input that capability accepts.
 *
 * This is the "filters the options subsequently offered" direction. It is deliberately
 * ADVISORY at the picker level: an incompatible capability is shown DISABLED with the
 * reason rather than vanishing, because a user who attached a start image and then
 * wonders where restyle went is worse off than one who can see it greyed out saying
 * "restyle does not take a start image".
 *
 * Attachments are never auto-dropped to make a capability fit — clobbering the user's
 * inputs is the defect class this whole slice deletes.
 */
export function capabilityOffers(
  presets: RenderPreset[],
  unavailable: UnavailableCapability[],
  attached: Attachments,
  candidates: readonly string[] = CLIP_CAPABILITIES,
): CapabilityOffer[] {
  const refusals = new Map(unavailable.map((u) => [u.capability, u]));
  return candidates.map((cap) => {
    const servable = isServable(presets, cap);
    if (!servable) {
      const u = refusals.get(cap);
      return {
        capability: cap,
        label: capabilityLabel(cap),
        offered: false,
        servable: false,
        proven: null,
        // The backend's own wording, verbatim — re-phrasing it here would give the
        // console and the log two different explanations of one fact.
        reason: u?.reason || u?.refusal || "no ratified preset covers it on this fleet",
      };
    }
    const blockers: string[] = [];
    (Object.keys(ACCEPTS) as Array<keyof Attachments>).forEach((slot) => {
      if (attached[slot] && !ACCEPTS[slot].includes(cap)) {
        blockers.push(ATTACHMENT_NOUNS[slot]);
      }
    });
    // The route refuses this pair by name (the VACE control channel takes ONE input and
    // source_video wins, so the control still would be silently dropped). Surface it as
    // an offer-level blocker rather than letting the user build it and get a 400.
    if (attached.controlImage && attached.sourceVideo && cap === "motion") {
      blockers.push("both a control still and a source clip (motion takes one control)");
    }
    return {
      capability: cap,
      label: capabilityLabel(cap),
      offered: blockers.length === 0,
      servable: true,
      proven: provenForCapability(presets, cap),
      reason: blockers.length
        ? `${capabilityLabel(cap)} does not take ${blockers.join(" or ")} — remove it to switch here.`
        : "",
    };
  });
}

/** The inverse direction, for the AUTO-PICK on an external stage: the best capability
 *  for what is attached right now, or null when the current one is already fine.
 *  Ordered by specificity — the most constrained attachment wins, because it is the one
 *  the user went to the most trouble to provide. */
export function capabilityForAttachments(
  presets: RenderPreset[],
  attached: Attachments,
  current: string,
): string | null {
  const offers = capabilityOffers(presets, [], attached);
  const ok = (cap: string) => offers.find((o) => o.capability === cap)?.offered === true;
  if (ok(current)) return null; // the user's choice already fits — never override it
  if (attached.controlImage && ok("motion")) return "motion";
  if (attached.referenceImages && ok("id_lock")) return "id_lock";
  if (attached.sourceVideo && ok("v2v")) return "v2v";
  if (attached.startImage && ok("i2v")) return "i2v";
  return offers.find((o) => o.offered)?.capability ?? null;
}

/** Which attachment slots the given capability accepts — drives whether an input widget
 *  is even OFFERED (the surface's existing accepts* predicates, now data-derived). */
export function acceptedSlots(cap: string): Attachments {
  return {
    startImage: ACCEPTS.startImage.includes(cap),
    sourceVideo: ACCEPTS.sourceVideo.includes(cap),
    referenceImages: ACCEPTS.referenceImages.includes(cap),
    controlImage: ACCEPTS.controlImage.includes(cap),
  };
}

/** Which inputs the ratified preset for this capability REQUIRES (its `inputs[]`, minus
 *  `prompt` which every row lists and no picker gates on). Used for the required-input
 *  callouts so "what this generator needs" comes from the measured table. */
export function requiredInputs(presets: RenderPreset[], cap: string): string[] {
  const rows = presetsForCapability(presets, cap);
  if (rows.length === 0) return [];
  return (rows[0].inputs ?? []).filter((i) => i !== "prompt" && i !== "goals");
}

// ── merge-not-overwrite: the preset-apply philosophy, in one place ──────────────────

/**
 * A preset supplies the WORKFLOW (capability, geometry, cadence, budget, model). The
 * user's SESSION INPUTS (prompt, negative, identity, attachments, a seed they chose) are
 * theirs and survive every apply.
 *
 * `fillIfEmpty` is the whole rule for text: a preset's example prompt is a scaffold for
 * an empty box, never a replacement for something the user wrote. Before 2026-07-29 the
 * studio's `onPickPreset` assigned `p.prompt ?? ""` — so picking a preset with no prompt
 * of its own BLANKED whatever the user had typed. That is the defect class this deletes.
 */
export function fillIfEmpty(current: string, offered: string | null | undefined): string {
  if (current.trim() !== "") return current;
  return offered ?? "";
}

/** Same rule for a numeric knob held as a string ("" = unset). A preset fills a blank
 *  field; it never overwrites a number the user typed. */
export function fillNumberIfEmpty(
  current: string,
  offered: number | null | undefined,
): string {
  if (current.trim() !== "") return current;
  return offered == null ? current : String(offered);
}

// ── generalized MOVIE workflow presets ──────────────────────────────────────────────

/** The joint modes the movie preset ratifies, with what each one MEANS and what it
 *  costs. Copy is derived from the movie row's own evidence, not invented here. */
const JOINT_WORKFLOWS: Record<
  string,
  { name: string; hint: string; segmentCapability: string }
> = {
  cut: {
    name: "Multi-shot movie — hard cuts",
    hint:
      "Each shot is a fresh render joined by a scene cut; no frames carry across. Every "
      + "segment stays on the proven 1.3B binding, so this is the cheapest and most "
      + "reliable way to chain shots. Attach an identity below and every shot locks to it.",
    segmentCapability: "t2v",
  },
  vace_extend: {
    name: "Continuous motion — VACE extend",
    hint:
      "Each shot continues the previous one's motion from a branch frame. Same proven "
      + "1.3B weights as a cut, reached through the restyle wiring — use it when the "
      + "camera or the action should carry across the joint.",
    segmentCapability: "v2v",
  },
  still: {
    name: "Still-carry chain",
    hint:
      "Each shot starts from the previous one's last frame as an image. Honest cost: this "
      + "joint binds the 14B image-to-video row per segment, which has never completed a "
      + "clip on this fleet and re-reads ~90 GB of weights per segment. Prefer cuts or "
      + "VACE extend unless you specifically need a still hand-off.",
    segmentCapability: "i2v",
  },
};

export interface MovieWorkflowPreset {
  id: string;
  name: string;
  hint: string;
  /** Applied to segments 1..N (segment 0 is the root and has no joint). */
  joint: string;
  /** The geometry the ratified movie preset renders at. */
  geometry: Geometry | null;
  /** The minimum budget for the binding THIS joint actually uses. */
  minBudgetGb: number | null;
  /** Has the segment binding this joint uses produced pixels on this fleet? */
  proven: boolean;
  /** The fewest segments this workflow means anything with. NEVER a replacement for
   *  the user's existing rows — the apply path grows to this and no further. */
  minSegments: number;
}

/**
 * The movie composer's preset list, DERIVED — one workflow per joint mode the ratified
 * `movie-480p` row declares, sized against the binding that joint actually uses.
 *
 * This replaces a hardcoded array of scene prose ("Beach & volleyball — character day
 * out", with a baked-in subject, a baked-in identity slug and two baked-in scene
 * prompts). A preset here names a WORKFLOW SHAPE and nothing about the content: the
 * user's own goal prompts, identity and start image are the content, and the apply path
 * slots the workflow around them without touching any of it.
 *
 * Returns [] when the payload has not landed — the caller shows its loading state rather
 * than a picker built on guesses.
 */
export function movieWorkflowPresets(presets: RenderPreset[]): MovieWorkflowPreset[] {
  const movieRows = presetsForCapability(presets, MOVIE_CAPABILITY);
  if (movieRows.length === 0) return [];
  const byId = new Map(presets.map((p) => [p.id, p]));
  const out: MovieWorkflowPreset[] = [];
  for (const row of movieRows) {
    const geometry = geometryForCapability(presets, MOVIE_CAPABILITY);
    for (const joint of row.joints ?? []) {
      const wf = JOINT_WORKFLOWS[joint];
      if (!wf) continue; // a joint the console does not model yet — never invent copy
      // The segment binding this joint really uses, found among the row's `composes`.
      const segRow = (row.composes ?? [])
        .map((id) => byId.get(id))
        .find((p): p is RenderPreset => !!p && p.capability === wf.segmentCapability);
      out.push({
        id: `${row.id}:${joint}`,
        name: wf.name,
        hint: wf.hint,
        joint,
        geometry,
        minBudgetGb: segRow?.vram_envelope_gb ?? row.vram_envelope_gb ?? null,
        // A composite can never be more proven than the weakest thing it composes —
        // the same rule the backend's test_presets.py enforces on the table itself.
        proven: (segRow?.proven ?? row.proven) === true,
        minSegments: 2,
      });
    }
  }
  return out;
}

/**
 * The budget a movie actually needs, given the joint AND whether an identity is attached.
 *
 * The identity case is the one a static number gets wrong: the moment a movie carries
 * references, EVERY segment renders id_lock (studio_movie.py) and binds the VACE row, so
 * the floor is that row's envelope regardless of which joint was chosen.
 */
export function movieBudgetGb(
  presets: RenderPreset[],
  opts: { joint: string; identity: boolean },
): number | null {
  if (opts.identity) return suggestedBudgetGb(presets, "id_lock");
  const wf = JOINT_WORKFLOWS[opts.joint];
  if (!wf) return suggestedBudgetGb(presets, MOVIE_CAPABILITY);
  const seg = suggestedBudgetGb(presets, wf.segmentCapability);
  const base = suggestedBudgetGb(presets, MOVIE_CAPABILITY);
  if (seg == null) return base;
  if (base == null) return seg;
  return Math.max(seg, base);
}

/** The Wan frame cadence check — `num_frames` must be 4k+1 or the runner snaps down.
 *  Exposed so a surface can say so BEFORE six minutes of denoise. */
export function isWanCadence(frames: number, cadence: number | null): boolean {
  if (!cadence || cadence < 1) return true;
  return frames >= 1 && (frames - 1) % cadence === 0;
}
