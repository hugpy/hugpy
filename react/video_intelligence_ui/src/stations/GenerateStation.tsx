// Generate station (Phases 6–7) — the sample image's bottom prompt-bar + center
// preview. Flow:
//   assemble an ORDERED multimodal prompt (text + library images + optional video)
//     → the composer's condensed settings strip, part of the prompt component
//       (model / width / height / steps / guidance / seed / negative — k63 moved these
//        out of the left rail for image+scene; movie still renders them in the rail)
//     → Run: POST /video/jobs/generate_image {parts, model_id, ...} → {job_id}
//     → poll GET /video/jobs/<id> → render result.outputs[0] as the generated image
//     → the result is pushed back into the library (by useGenerateJob) so it is
//       reusable as an image part.
// Every hop goes through request<T>() + config URLs; errors surface as human
// strings. Job state is read-only after enqueue (poll only). No silent defaults —
// every knob is explicit and visible, pre-filled with sd-turbo-sensible values.
import { useEffect, useMemo, useRef, useState } from "react";
// Sub-tab / main-tab input memory — see the useSessionState block at the top of
// GenerateStation and the rationale in video/sessionForm.ts.
import { useSessionState } from "../video/sessionForm";
import type { StationSpec } from "./types";
import { useGenerateJob } from "./useGenerateJob";
import {
  useGenerateSceneRowJobs,
  type SceneRowJobState,
} from "./useGenerateSceneRowJobs";
import { useGenerateSceneJob } from "./useGenerateSceneJob";
import { useGenerateMovieJob } from "./useGenerateMovieJob";
import { SectionTabs } from "./SectionTabs";
import { KnobInfo } from "./KnobInfo";
// The condensed settings strip + carry ticks — extracted to their own module (k88)
// so the k89 Scene/Clip prompt-card slice can consume them; behavior unchanged here.
import { CarryTick, CondensedKnobStrip, toNumberOrNull, type KnobValues } from "./CondensedKnobStrip";
// The SHARED prompt-card system (k88) — the Movie timeline renders the same
// toolbar/card/tabs/settings structure the Cinema composer does.
import {
  AboutExpander,
  PromptCard,
  PromptCardSettings,
  PromptFieldTabs,
  PromptListToolbar,
  STANDARD_NEGATIVE,
  applyStandardNegative,
  composeNegative,
  hasStandardNegative,
} from "./studio/promptCard";
// Movie's group "Generate prompts" rides the SAME one-call spread the Cinema
// composer uses (usePromptAssist.runSpread + the pure builders/readers).
import {
  buildSpreadBody,
  spreadNoticeLines,
  type AssistMediaWire,
  type SpreadGoalInput,
} from "../video/spreadAssist";
import { StudioGenerateMode } from "./studio/StudioGenerateMode";
import { useIdentityProfiles, type IdentityProfile } from "./studio/useIdentityProfiles";
import { GenIdentityBar, type BoundIdentity } from "./GenIdentityBar";
import { useSidebarRegistry } from "./SidebarRegistry";
import { useProjects } from "../video/useProjects";
import { cancelJob, trackJob, useJobTracker } from "../video/jobTracker";
import { STILL_RUNNING_MESSAGE } from "./pollCaps";
import { AddFileBar } from "./AddFileBar";
import { useImageModels, useModelsByTask, requiredInputsFor } from "../video/useModels";
import {
  usePresets,
  presetApplyResponseSchema,
  type Preset,
  type PresetDefaults,
} from "../video/usePresets";
import { useMoviePresets } from "../video/useMoviePresets";
import {
  addToLibrary,
  useMediaLibrary,
} from "../video/mediaLibrary";
import {
  request,
  okValue,
  errorOf,
  describeAppError,
  type AppError,
} from "../transport/client";
import { hugpyConfig, mediaBytesUrl, presetApplyUrl } from "../config";
import { getSessionId } from "../session";
import { isCanned } from "../demo/mode";
import { DEMO_GENERATE_PROMPT } from "../demo/seed";
import { usePromptAssist } from "../video/usePromptAssist";
import { useOperatorHistory } from "../video/useOperatorHistory";
import { UnnamedWarnPanel, WarnUnnamedToggle, useWarnUnnamedPref, deriveNameFrom } from "../video/UnnamedWarn";
import { LibraryGroups } from "../video/LibraryGroups";
import { PromptAssistButtons } from "../video/PromptAssistButtons";
import { registerComposer, type StageRequest } from "../video/composerBridge";
import {
  mediaRefSchema,
  uploadResultSchema,
  enqueueResultSchema,
  isMovieProgress,
  type FrameExtractRequest,
  type GenPromptPart,
  type GenerateImageRequest,
  type GenerateSceneRequest,
  type GenerateMovieRequest,
  type MovieGoal,
  type MoviePreset,
  type MediaRef,
  type JobProgress,
  type MovieProgress,
  type AnyProgress,
  type ImageTestRow,
  type CarrySettingKey,
  type CarryTicks,
} from "../video/contract";
import { useGenerateRowJobs } from "./useGenerateRowJobs";

const MODEL_KEY = "vi.generate.imageModel.v1";
/** The Movie director's judge-VLM select default + the task it filters the registry by. */
const JUDGE_TASK = "image-text-to-text";
const DEFAULT_JUDGE_MODEL = "Qwen2.5-VL-3B-Instruct-GGUF";
/** A fresh goal's default frame length, and the split floor (need ≥2 to halve). */
const DEFAULT_GOAL_LEN = 6;
/**
 * The negative prompt the field opens pre-filled with (operator 2026-07-13):
 * distilled/painterly checkpoints (sd-turbo &c.) reproduce whole "framed gallery
 * painting" objects from their training prior, so a plain prompt renders the
 * subject INSIDE an ornate picture frame. Suppressing the frame + the usual junk
 * by default makes the happy path a clean image (defaults-are-promises). Fully
 * visible, editable and clearable in the negative field, and any preset/draft
 * negative overrides it (see the `!= null` guards below).
 * The TEXT now lives in promptCard.tsx as STANDARD_NEGATIVE (k88 — the per-row
 * [standard negative] checkbox reads the same source); re-imported here so this
 * station's prefill behavior is byte-identical.
 */
const DEFAULT_NEGATIVE = STANDARD_NEGATIVE;

/**
 * The MODE DEFAULTS for the generation knobs — the sd-turbo-sensible values the
 * station opens with. Named (rather than five inline literals) because the carry
 * ticks (operator ask 2026-08-04, k63) need a second reader: an UNTICKED setting
 * is not copied into the row seeded from this one, it falls back to exactly these.
 * One source of truth so "reset to default" can never drift from "opens with".
 */
// General-purpose SD defaults (operator 2026-08-10). The old values (steps 4,
// guidance 0) were sd-turbo-only: at CFG 0 the NEGATIVE PROMPT has no effect, so
// the standard anti-frame negative was silently ignored and SD1.5/SDXL models came
// out as undercooked "framed painting" blobs. CFG ~7 + ~25 steps is right for
// SD1.5/SDXL (the majority); turbo/lightning models want CFG lowered to ~1 and
// steps ~4 — set those per-part or via the model's preset.
const DEFAULT_KNOBS = {
  width: "512",
  height: "512",
  steps: "25",
  guidance: "7",
  seed: "", // blank = null (random)
} as const;

/**
 * Carry ticks default to ALL ON (operator ask 2026-08-04, k63: a tick means
 * "bring this setting into the post created component"). ON-by-default preserves
 * the pre-tick behavior — a new row was always a full copy of its source — so the
 * ticks are a subtractive tool: untick what you want re-defaulted, and the sweep
 * varies exactly that. Frozen shape; spread it, never mutate it.
 */
const CARRY_ALL: CarryTicks = {
  model: true,
  size: true,
  steps: true,
  guidance: true,
  seed: true,
  negative: true,
};

// mm:ss clock for the running readout (elapsed since started_at, and the ETA).
function fmtClock(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, "0")}`;
}

// The DESCRIPTIVE running readout: "{stage} · image {done}/{total} · {label} ·
// {elapsed} · ETA {eta}" — each segment omitted when the backend didn't send it.
// started_at is epoch SECONDS; elapsed re-derives on every poll re-render.
function progressReadout(p: JobProgress): string {
  const bits: string[] = [];
  if (p.stage) bits.push(p.stage);
  bits.push(`image ${p.done}/${p.total}`);
  if (p.label) bits.push(p.label);
  if (p.started_at != null) bits.push(fmtClock(Date.now() / 1000 - p.started_at));
  if (p.eta_s != null) bits.push(`ETA ${fmtClock(p.eta_s)}`);
  return bits.join(" · ");
}

// The Movie-mode running readout — the NESTED progress's top line: "{stage} ·
// segment {done}/{total} · {current label} · {elapsed} · ETA {eta}". Sibling of
// progressReadout (which stays the per-SEGMENT frame readout, fed `current`).
function movieReadout(p: MovieProgress): string {
  const bits: string[] = [];
  if (p.stage) bits.push(p.stage);
  bits.push(`segment ${p.segment_done}/${p.segment_total}`);
  if (p.current?.label) bits.push(p.current.label);
  if (p.started_at != null) bits.push(fmtClock(Date.now() / 1000 - p.started_at));
  if (p.eta_s != null) bits.push(`ETA ${fmtClock(p.eta_s)}`);
  return bits.join(" · ");
}

// The active mode's running readout, over EITHER progress shape (flat scene/image
// via progressReadout, nested movie via movieReadout). Used by the shared footer.
function runningReadoutFor(p: AnyProgress | null): string | null {
  if (p == null) return null;
  return isMovieProgress(p) ? movieReadout(p) : progressReadout(p);
}

/** Generate station sub-mode: single image, an N-frame scene, or a goal-timeline movie. */
// The generate TIERS. The former single "studio" tier was promoted (operator 2026-08-05)
// to TWO top-level tabs: "clip" (the studio clip surface) and "cinema" (the studio movie
// composer). Both mount the shared StudioGenerateSurface via StudioPlane, LOCKED to the
// matching sub-mode. "movie" is the SEPARATE goal-timeline movie tier — unrelated to
// "cinema". "image" is retained in the union (the composer bridge's StageRequest still
// names it) but no tab selects it.
type GenMode = "image" | "scene" | "movie" | "clip" | "cinema";

// ---- Movie GOAL TIMELINE row (client-side editor shape) ----
// start/end are kept as STRINGS (like every other knob) so an in-progress edit
// isn't silently coerced; validation parses them. `ref` reuses the composer's
// PartMedia so the per-goal reference image goes through the SAME picker/upload path.
// `negative`/`stdNegative` (k88, shared prompt-card shape): the WIRE carries ONE
// movie-level negative (GenerateMovieRequest.negative — MovieGoal has no per-goal
// field), so today only row 0's tabbed Negative pane is live — it edits the shared
// station `negative` state (the wired one) and row 0's `stdNegative` decides
// whether the standard set is composed into the submit payload. Rows > 0 render
// the Negative tab disabled; their fields go live if the wire ever grows a
// per-goal negative.
type GoalRow = {
  id: string;
  start: string;
  end: string;
  prompt: string;
  ref?: PartMedia;
  negative?: string;
  stdNegative?: boolean;
  // k92 — a movie goal IS a prompt component (same architecture as a Scene part):
  // it may carry its OWN generation knobs. Held as the same live strings the station
  // setters use so a goal's override parses/validates exactly like the shared value
  // it defaults from. Empty = falls back to the movie-level value (see goalKnobValues /
  // goalExtra). `knobs` = the 4 CondensedKnobStrip knobs (size/steps/guidance/seed);
  // `extra` = model + the goal-appropriate scene knobs (strength/chain/motion).
  knobs?: Partial<KnobValues>;
  extra?: Partial<GoalExtra>;
};
// The non-KnobValues per-goal settings (k92). A movie's fps/assemble/project are
// MOVIE-level (one clip), and a goal's frame count is its window — so a goal's own
// knobs are model + strength + chain + motion only (mirrors PartExtra, minus the
// movie-level fields).
type GoalExtra = {
  model: string;
  strength: string;
  chain: boolean;
  motion: string;
};
// SHARE-tick keys for a movie goal's settings (k92). "share" = one value across all
// goals (the movie-level knob); unticked = each goal keeps its own. `size` covers
// width+height as one setting, matching the knob strip. Default ON so a movie with
// no per-goal overrides renders exactly as before.
type MovieShareKey =
  | "model"
  | "size"
  | "steps"
  | "guidance"
  | "seed"
  | "strength"
  | "chain"
  | "motion"
  | "negative";

let _goalSeq = 0;
function goalKey(): string {
  _goalSeq += 1;
  return `goal_${_goalSeq.toString(36)}_${Date.now().toString(36)}`;
}

/**
 * Re-tile an ordered goal list into a CONTIGUOUS [0,total) timeline, preserving
 * each row's LENGTH (end-start) and order — used after every STRUCTURAL edit
 * (reorder / delete) so the frame windows stay gap/overlap-free. Manual numeric
 * edits do NOT re-tile (the user is explicitly setting windows there), so they can
 * introduce gaps the validator then flags. A row with an unusable window falls back
 * to DEFAULT_GOAL_LEN.
 */
function retile(rows: GoalRow[]): GoalRow[] {
  let cursor = 0;
  return rows.map((r) => {
    const s = toNumberOrNull(r.start);
    const e = toNumberOrNull(r.end);
    const rawLen = s != null && e != null && e > s ? e - s : DEFAULT_GOAL_LEN;
    const len = Math.max(1, Math.floor(rawLen));
    const start = cursor;
    const end = cursor + len;
    cursor = end;
    return { ...r, start: String(start), end: String(end) };
  });
}

/** A short label for a movie segment cell — its prompt (trimmed) or `goal N`. */
function segmentLabel(prompt: string | undefined, index: number): string {
  const t = (prompt ?? "").trim();
  if (t === "") return `goal ${index + 1}`;
  return t.length > 42 ? `${t.slice(0, 42)}…` : t;
}

// Ordered prompt parts. Each part is a text+media PAIR: a media part carries an
// optional caption, and a text part may attach a library image/video. The pairing
// is composer-only — buildPromptParts() flattens each pair back to the frozen
// GenPromptPart wire shape (caption emitted as a text part immediately BEFORE its
// media, order preserved), so the backend contract is untouched.
type PartMedia = { ref: MediaRef; origin: string; label?: string };
// `identitySlug` MARKS a part as one this station minted from a bound identity
// profile's reference set (see bindIdentity). It is composer-local ONLY — a
// bookkeeping tag so unbind/rebind can find exactly the parts it owns and never
// touch a part the user attached or captioned by hand. buildPromptParts() reads
// only .text/.media, so this never reaches the frozen GenPromptPart wire shape.
// The non-KnobValues per-part settings (model + the scene knobs + project). Held
// as the same live strings/booleans the station setters use, so a part's override
// parses/validates exactly like the shared value it defaults from.
type PartExtra = {
  model: string;
  nFrames: string;
  motion: string;
  fps: string;
  assemble: boolean;
  strength: string;
  chain: boolean;
  project: string;
};
type PartEntry = {
  key: string;
  text: string;
  media?: PartMedia;
  identitySlug?: string;
  negative?: string;
  stdNegative?: boolean;
  // Per-part overrides for the 5 CondensedKnobStrip knobs (size/steps/guidance/seed).
  knobs?: Partial<KnobValues>;
  // Per-part overrides for model + scene knobs + project.
  extra?: Partial<PartExtra>;
  // Per-part PROMPT-ASSIST model (which text generator writes THIS prompt's
  // Enhance/Generate). Undefined = use the shared assist model / fleet default.
  assistModel?: string | null;
  // k93 — per-card `negatives: [on/off]`: whether THIS card's Enhance/Generate
  // also writes its Negative field. Undefined = on (today's behavior).
  assistNegatives?: boolean;
  // k93 — a frame PULLED from this part's attached video (the frame-extract
  // job's output). The part keeps its video AND the frame; buildPartPromptParts
  // emits the frame image BEFORE the video so the runner's image_paths[0] (the
  // start frame) is the pulled still.
  frame?: PartMedia;
};
// The share-tick keys for EVERY per-part setting. "share" = one value across all
// prompts; unticked = each prompt keeps its own. (Scene owns its own share state
// rather than the tester's `carry`, so the two surfaces never contaminate each
// other.) `size` covers width+height as one setting, matching the knob strip.
type SceneShareKey =
  | "model"
  | "size"
  | "steps"
  | "guidance"
  | "seed"
  | "frames"
  | "motion"
  | "fps"
  | "assemble"
  | "strength"
  | "chain"
  | "project";

/** Display kind, derived: a part IS whatever media it carries, else text. */
function partKind(p: PartEntry): "text" | "image" | "video" {
  if (!p.media) return "text";
  return p.media.ref.kind === "video" ? "video" : "image";
}

// The plain-words capability line under the model select, derived from the model's
// required-input contract (see requiredInputsFor). States what THIS model needs.
function capabilityHint(image: "required" | "optional" | "none"): string {
  if (image === "none") return "Text-to-image — prompt only.";
  if (image === "required") return "Image edit — start image + prompt required.";
  return "Text or image-to-image — add a start image to edit.";
}

type VideoPhase = "idle" | "uploading" | "ingesting";

// Number of canonical views ONE bound identity contributes as prompt parts.
//
// RETIRED — was `MAX_IDENTITY_REFERENCE_IMAGES = 4` (a cap on a multi-view bind).
// The cap is now meaningless: a bind mints EXACTLY ONE part, the single view the
// user picked in GenIdentityBar's canonical row, so there is no N to bound. Keeping
// a "max 4" would also be an active lie once the parallel slice grows canonical to
// 8 azimuth views — the row must render whatever `profile.canonical` holds, and the
// bind takes exactly one of them regardless of the set's size.
//
// WHY ONE, not "up to 4": the generate runner reads only the FIRST image part —
//   abstract_hugpy_dev/src/abstract_hugpy_dev/video_intel/runners/imagegen.py:117
//     start_frame = image_paths[0] if image_paths else None
//   (scene.py:539 likewise, via next(...)). Views 2..N were collected, logged, and
// discarded. Attaching them was decorative and misled the user into believing the
// whole turnaround conditioned the shot.
//
// Named + exported-in-place (not deleted — archive-in-place) so the intent stays
// greppable and the constant is the ONE seam to change if the runner ever learns
// real multi-view conditioning: raise this and make the row a multi-select.
const IDENTITY_CARRYING_VIEWS = 1;

let _seq = 0;
function partKey(): string {
  _seq += 1;
  return `part_${_seq.toString(36)}_${Date.now().toString(36)}`;
}

// A file extension derived from a MediaRef mime, used only for the suggested
// `download` filename on the result anchors (the bytes are whatever the server
// serves; this just names the saved file sensibly).
function extFromMime(mime: string): string {
  const m = mime.toLowerCase();
  if (m.includes("png")) return "png";
  if (m.includes("jpeg") || m.includes("jpg")) return "jpg";
  if (m.includes("webp")) return "webp";
  if (m.includes("gif")) return "gif";
  if (m.includes("mp4")) return "mp4";
  if (m.includes("webm")) return "webm";
  const sub = m.split("/")[1];
  return sub ? sub.split(";")[0] : "bin";
}

// (toNumberOrNull, KnobValues/knobFlags, CarryTick and CondensedKnobStrip moved
// verbatim to ./CondensedKnobStrip.tsx — k88 — and are imported above.)

// Human text for a failed preset pre-warm. The two documented failures (409 no GPU
// worker / won't fit, 404 unknown preset/model) get purpose-written copy that also
// reassures the user the knobs are still prefilled — because the shared transport's
// serverMessage() stringifies the contract's nested {error:{code,message}} object to
// "[object Object]" (it only understands a flat {error:"<string>"}), so the raw
// server text is unusable here. Any other error falls back to describeAppError.
function presetApplyErrorText(err: AppError): string {
  if (err.kind === "server") {
    if (err.status === 409) {
      return "No GPU worker is free right now (or the model won't fit). The knobs are prefilled — you can still hit Run.";
    }
    if (err.status === 404) {
      return "The server doesn't recognize this preset or its model. The knobs are prefilled — you can still hit Run.";
    }
  }
  return describeAppError(err);
}

export function GenerateStation({ spec }: { spec: StationSpec }) {
  // ---- sub-mode: Scene (default) · Movie · Clip · Cinema ----
  // The standalone Image tab FOLDED into Scene (operator 2026-08-05): Scene's
  // Output selector covers the single-still case (Output=Image → 1 frame, no
  // mp4), so the station lands on Scene with Image-output defaults and opens on
  // the familiar single-image experience. GenMode keeps "image" in the union
  // (the composer bridge's StageRequest still names it) but no tab selects it.
  // ── SUB-TAB / MAIN-TAB INPUT MEMORY (operator ask 2026-08-06) ──────────────
  // Every OPERATOR-AUTHORED field in this station — the sub-mode itself, the
  // prompt parts, the knobs, the goal timeline, the director settings — is
  // `useSessionState` rather than `useState`, so it survives BOTH kinds of
  // unmount: switching workbench tabs (WorkbenchStation renders exactly one
  // station body) and switching to the Clip/Cinema sub-tabs (which swap this
  // whole layout out for StudioGenerateMode). Same tuple as useState; the store
  // is per-tab and session-scoped — see video/sessionForm.ts for why that beats
  // keep-mounted-but-hidden here.
  //
  // Transient state stays on plain useState on purpose: open pickers, upload
  // phases, preset apply status/errors, tester rows and every in-flight job.
  // Restoring "uploading…" or a stale error banner would be its own bug.
  const [mode, setMode] = useSessionState<GenMode>("gen.mode", "scene");
  // The two studio-surface tabs (Clip / Cinema) mount the StudioPlane instead of the
  // image/scene SectionTabs + Options-rail layout — gate both the settings-rail effect and
  // the render seam on this so they behave exactly as the old single "studio" mode did.
  const isStudioSurface = mode === "clip" || mode === "cinema";
  // Item C: the composer-drain effect below registers ONCE (empty deps → a stable
  // stager), so its closure can never read the live `mode`. Mirror `mode` into a ref
  // the drain reads to route a staged media part to the ACTIVE sub-tab (image/scene →
  // parts, movie → the current goal's reference image) instead of always Image.
  const modeRef = useRef<GenMode>(mode);
  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  // Round 9: the image/scene/movie OPTIONS rail folds into the arm shell's ONE left
  // sidebar "Settings" tab (the studio mode registers its own via StudioPlane). Reveal
  // the guarded Settings tab while in a rail-bearing mode; SectionTabs then portals the
  // options node into `reg.settingsHost` and the centre goes full width (movie-mirror).
  const reg = useSidebarRegistry();
  useEffect(() => {
    if (isStudioSurface) return;
    return reg.registerSettings();
  }, [reg.registerSettings, isStudioSurface]);
  // Existing project names for the "Project name" combobox (choose-or-type).
  const { projects: knownProjects } = useProjects(true);

  // ---- ordered multimodal prompt (insertion order preserved) ----
  // Item A: seed ONE empty text part so Image/Scene open with a prompt-part input
  // visible immediately (no "Add text" click needed). It is empty, so the "at least
  // one non-empty text part" Run gate (hasText, below) still blocks Run until the
  // user types. The parts.length===0 empty-state only shows if the user deletes it.
  //
  // Showroom flavor (?demo=1 only): pre-fill that first part with a cinematic
  // example prompt (matches the sample gens) so the brochure opens with a coherent,
  // runnable story instead of a blank composer. The seed string lives in src/demo/
  // and is injected ONLY behind isCanned() — the live path stays an empty prompt.
  // k93: a new TEXT part opens with the standard negative VISIBLE in its Negative
  // field (operator: standard negatives are the default) — explicit text, sent
  // as-is; the std. tick reads its state from the field and can remove it.
  const [parts, setParts] = useSessionState<PartEntry[]>("gen.parts", () => [
    {
      key: partKey(),
      text: isCanned() ? DEMO_GENERATE_PROMPT : "",
      negative: STANDARD_NEGATIVE,
    },
  ]);
  const [showImagePicker, setShowImagePicker] = useState<boolean>(false);
  const [showVideoPicker, setShowVideoPicker] = useState<boolean>(false);
  // When set, an open picker ATTACHES its selection to this existing part
  // (text ↔ media pairing) instead of appending a new part.
  const [attachTarget, setAttachTarget] = useState<string | null>(null);

  // ---- Scene knobs (the Output selector in the rail drives nFrames + assemble) ----
  // Defaults are the IMAGE output (1 frame, no mp4): Scene is the landing surface
  // now that the Image tab folded in, so it opens on the single-still experience.
  // (Pre-fold scene defaults were nFrames "6" + assemble true.)
  const [nFrames, setNFrames] = useSessionState<string>("gen.nFrames", "1"); // frames per scene (1..24)
  const [motion, setMotion] = useSessionState<string>("gen.motion", ""); // per-frame template; blank = null
  const [fps, setFps] = useSessionState<string>("gen.fps", "6"); // assembled clip fps
  const [assemble, setAssemble] = useSessionState<boolean>("gen.assemble", false); // build an mp4 from frames
  const [strength, setStrength] = useSessionState<string>("gen.strength", "0.45"); // img2img strength (0..1)
  const [chain, setChain] = useSessionState<boolean>("gen.chain", true); // condition each frame on the last

  // Scene "Output = Image" — the SAME active-state math the Output selector uses
  // (1 frame, no assembled mp4). Gates the Tester block and flips the prompt
  // assistant into its image dialect below.
  const sceneOutputIsImage = toNumberOrNull(nFrames) === 1 && !assemble;

  // ---- LLM prompt-assist (Enhance / Generate) ----
  // The transport + one-at-a-time busy/error model now lives in the shared
  // usePromptAssist hook (POSTs promptAssistUrl); this station wires it to its
  // ordered-parts composer (draft = assembled text parts; apply = replace the
  // primary text part). `context.kind` tracks the active sub-mode so video modes
  // get motion/camera phrasing. A Scene in Image output is a single still, so it
  // asks for the "image" dialect — multi-frame Scene outputs keep "scene".
  const assistKind: "image" | "scene" | "movie" =
    mode === "movie"
      ? "movie"
      : mode === "scene" && !sceneOutputIsImage
        ? "scene"
        : "image";
  const assist = usePromptAssist({ kind: assistKind, specKey: "generate" });

  // Default the prompt-assist model to flux-klein-uncensored (operator) once the
  // assist-model list loads, UNLESS the user already has an explicit pick (which
  // usePromptAssist restores from localStorage). Best-effort: a case-insensitive
  // match, so it survives minor catalog-key variations; if it's not offered, we
  // leave the fleet default in place.
  const assistDefaultApplied = useRef(false);
  useEffect(() => {
    if (assistDefaultApplied.current) return;
    if (assist.assistModel) {
      assistDefaultApplied.current = true; // user/localStorage already chose
      return;
    }
    if (assist.assistModels.length === 0) return;
    // Prefer the exact operator-specified key; fall back to a case-insensitive
    // pattern so a minor catalog rename still resolves.
    const DEFAULT_ASSIST_MODEL = "flux2-klein-9b-uncensored-text-encoder";
    const klein =
      assist.assistModels.find((m) => m.model === DEFAULT_ASSIST_MODEL) ??
      assist.assistModels.find((m) =>
        /flux\.?2?[-_ ]?klein.*(9b|uncensor)/i.test(m.model),
      );
    if (klein) assist.setAssistModel(klein.model);
    assistDefaultApplied.current = true;
  }, [assist.assistModels, assist.assistModel, assist.setAssistModel]);

  // ---- explicit knobs, pre-filled with sd-turbo-sensible values (all editable) ----
  const [width, setWidth] = useSessionState<string>("gen.width", DEFAULT_KNOBS.width);
  const [height, setHeight] = useSessionState<string>("gen.height", DEFAULT_KNOBS.height);
  const [steps, setSteps] = useSessionState<string>("gen.steps", DEFAULT_KNOBS.steps);
  const [guidance, setGuidance] = useSessionState<string>("gen.guidance", DEFAULT_KNOBS.guidance);
  // blank = null (random)
  const [seed, setSeed] = useSessionState<string>("gen.seed", DEFAULT_KNOBS.seed);
  // k89: no longer SEEDED with the standard-negative TEXT — for MOVIE the standard
  // set rides goal #1's stdNegative tick (seeded ON below) and is COMPOSED in at
  // submit via composeNegative. For SCENE (k93) this shared value is NOT read at
  // all any more: each part sends exactly what its own visible Negative field
  // holds (blank = blank, nothing implied); presets WRITE into part #1's field.
  const [negative, setNegative] = useSessionState<string>("gen.negative", "");

  // ---- per-setting CARRY ticks on the base composer strip (operator ask
  // 2026-08-04, k63) ----
  // "Bring this setting into the post created component": every setting in the
  // condensed strip carries a tick, and the tick decides what a component SEEDED
  // from this one receives (today: a tester row via "+ Add row" / the per-model /
  // per-preset expanders; the same rule is what any future scene-carry flow reads).
  // Untick a setting and the new component gets the mode default instead — that is
  // how you build a sweep that varies exactly one thing. Purely client-side: no
  // request payload ever sees a tick.
  const [carry, setCarry] = useSessionState<CarryTicks>("gen.carry", () => ({
    ...CARRY_ALL,
  }));
  function setCarryKey(key: CarrySettingKey, value: boolean): void {
    setCarry((prev) => ({ ...prev, [key]: value }));
  }
  // SHARE ticks for the scene knobs + project (model/size/steps/guidance/seed
  // ride `carry`). Default ON so an unconfigured composer behaves as one shared
  // set until a knob is unticked to go per-prompt.
  const [sceneShare, setSceneShare] = useSessionState<Record<SceneShareKey, boolean>>(
    "gen.sceneShare",
    () => ({
      model: true,
      size: true,
      steps: true,
      guidance: true,
      seed: true,
      frames: true,
      motion: true,
      fps: true,
      assemble: true,
      strength: true,
      chain: true,
      project: true,
    }),
  );

  // SHARE ticks for a MOVIE GOAL's per-component knobs (k92). Same idea as
  // sceneShare, for the goal-appropriate knob set. Default ON so an unconfigured
  // Movie behaves as one shared knob set (byte-identical payload) until a knob is
  // unticked to go per-goal.
  const [movieShare, setMovieShare] = useSessionState<Record<MovieShareKey, boolean>>(
    "gen.movieShare",
    () => ({
      model: true,
      size: true,
      steps: true,
      guidance: true,
      seed: true,
      strength: true,
      chain: true,
      motion: true,
      negative: true,
    }),
  );

  // ---- optional auto-archive project name (blank = backend auto-names) ----
  const [projectName, setProjectName] = useSessionState<string>("gen.projectName", "");
  // UNNAMED WARNING (operator 2026-08-13): explicit PER-SECTION option, default
  // OFF, localStorage-remembered — an image that renders in 2 seconds should
  // never suffer a 30-second warning unless its operator opted in.
  const [warnImagePref, setWarnImagePref] = useWarnUnnamedPref("gen.image");
  const [warnScenePref, setWarnScenePref] = useWarnUnnamedPref("gen.scene");
  const [warnMoviePref, setWarnMoviePref] = useWarnUnnamedPref("gen.movie");
  const warnPref =
    mode === "image" ? warnImagePref : mode === "scene" ? warnScenePref : mode === "movie" ? warnMoviePref : false;
  const setWarnPref =
    mode === "image" ? setWarnImagePref : mode === "scene" ? setWarnScenePref : setWarnMoviePref;
  const [nameWarn, setNameWarn] = useState(false);
  const [pendingRun, setPendingRun] = useState(false);
  // Auto-derived names may be replaced by the next ✨ Generate; typed ones never.
  const autoProjectRef = useRef("");

  // ---- STUDIO TESTER cross-model battery (operator 2026-08-05) ----
  // The SAME across-models sweep the studio surfaces expose (POST studioTesterUrl),
  // added to the Scene and Movie run rows. It fans the current prompt across every
  // model of the tab's category and records a battery; per-model results stream to
  // the generation log. This is ADDITIVE — it never touches the scene/movie ENQUEUE
  // path (onRun) or the per-row image tester above.
  const [studioTesterBusy, setStudioTesterBusy] = useState<boolean>(false);
  const [studioTesterMsg, setStudioTesterMsg] = useState<string | null>(null);
  const studioTesterMounted = useRef(true);
  useEffect(() => {
    studioTesterMounted.current = true;
    return () => {
      studioTesterMounted.current = false;
    };
  }, []);

  // ---- IMAGE TESTER (operator ask 2026-08-04: "the settings be per prompt") ------
  // The knobs above belong to the STATION, which is fine for one-shot generation but
  // useless for comparison — you cannot sit two prompts side by side at different
  // steps/guidance/seed. Each tester row therefore carries its OWN complete knob set
  // and produces its OWN job (see useGenerateRowJobs); "Run all rows" fires them
  // together so the results land beside each other.
  //
  // Rows deliberately do NOT get their own media. They reuse the base composer's
  // image/video/identity parts, so a sweep holds the reference material fixed and
  // varies only the thing under test — otherwise the comparison proves nothing.
  const [testerOpen, setTesterOpen] = useState<boolean>(false);
  const [testRows, setTestRows] = useState<ImageTestRow[]>([]);
  const rowJobs = useGenerateRowJobs();
  // Scene per-part fan-out: one independent scene job per prompt component.
  const sceneRowJobs = useGenerateSceneRowJobs();
  // k89: Movie's bottom "[▸ Tester]" collapsible (it has no row-based tester — the
  // fold gives the cross-model sweep a home after the run row's ghost was removed).
  const [movieTesterOpen, setMovieTesterOpen] = useState<boolean>(false);

  // ---- Movie-only state (Movie mode) ----
  // The GOAL TIMELINE: ordered, contiguous goal rows. Seeded with one goal covering
  // [0, DEFAULT_GOAL_LEN) so the editor opens with something to edit.
  // Row 0's stdNegative seeds ON (k89): with `negative` no longer prefilled with
  // the standard TEXT, the tick is what keeps the movie's effective default
  // negative identical to before — composed in at submit.
  const [goals, setGoals] = useSessionState<GoalRow[]>("gen.movie.goals", () => [
    { id: goalKey(), start: "0", end: String(DEFAULT_GOAL_LEN), prompt: "", stdNegative: true },
  ]);
  // Rows parked by lowering the goal count — revived (keys intact) when it grows.
  const [goalStash, setGoalStash] = useSessionState<GoalRow[]>("gen.movie.goalStash", () => []);
  // When set, the image picker attaches its pick as THIS goal's reference (vs a part).
  const [goalAttachTarget, setGoalAttachTarget] = useState<string | null>(null);
  // ---- Movie prompt-list toolbar state (k88 — mirrors Cinema's group assist) ----
  // Which goals the group actions may rewrite (ticked rows and NOTHING else);
  // live-row math below, since a removed row can leave a stale id behind.
  const [movieSelectedIds, setMovieSelectedIds] = useState<ReadonlySet<string>>(() => new Set());
  // The spread's non-destructive result notice (what changed / didn't / warnings).
  const [movieSpreadNotice, setMovieSpreadNotice] = useState<string[] | null>(null);
  // The last spread's steering seed — re-sent so a follow-up spread re-rolls a
  // couple of goals INTO THE SAME WORLD instead of a new one (Cinema's pin, minus
  // the checkbox: Movie keeps it implicit).
  const [movieSteeringSeed, setMovieSteeringSeed] = useState<number | null>(null);
  // ---- Scene prompt-list toolbar state (k89 — the parts composer joins the shared
  // card system; exact mirrors of the Movie trio above, keyed by part key) ----
  const [sceneSelectedIds, setSceneSelectedIds] = useState<ReadonlySet<string>>(() => new Set());
  const [sceneSpreadNotice, setSceneSpreadNotice] = useState<string[] | null>(null);
  const [sceneSteeringSeed, setSceneSteeringSeed] = useState<number | null>(null);
  // ---- k93 Scene group toolbar state ----
  // `negatives: [on/off]` — when ON the group generate/enhance also write each
  // selected part's Negative field (the old separate "Generate negative" button
  // folded in here). Session-remembered like the other toolbar prefs.
  const [sceneGroupNegatives, setSceneGroupNegatives] = useSessionState<boolean>(
    "gen.sceneGroupNegatives",
    true,
  );
  // The last EXPLICIT range (1-based, over text parts) — what "select all" goes
  // back to when unticked. Null until the user has touched the range inputs /
  // per-card ticks.
  const sceneRangeMemo = useRef<{ start: number; end: number } | null>(null);
  // The toolbar-level `[+ video]` PRETEXT: ONE video (library pick or upload)
  // sent as context.media on the group generate/enhance so the backend describes
  // it and feeds that description to the generator. Not a prompt part.
  const [scenePretext, setScenePretext] = useState<PartMedia | null>(null);
  const [scenePretextOpen, setScenePretextOpen] = useState<boolean>(false);
  // Which card has its inline attach panel expanded (one at a time), and what.
  const [partAttachOpen, setPartAttachOpen] = useState<
    { key: string; kind: "image" | "video" | "library" } | null
  >(null);
  // Whether the inline `+ identity` expander is open per card key.
  const [identityOpen, setIdentityOpen] = useState<ReadonlySet<string>>(() => new Set());
  // Pull-frame: per part, the time (seconds, as typed) + the in-flight job id.
  const [pullFrameAt, setPullFrameAt] = useState<Record<string, string>>({});
  const [pullFrameJobs, setPullFrameJobs] = useState<Record<string, string>>({});
  const [pullFrameError, setPullFrameError] = useState<Record<string, string>>({});
  // Watch the tracker for the pull-frame jobs: on `done` the first output becomes
  // this part's `frame` (the tracker has ALREADY pushed it into the library as
  // "frames / frame 1"); on failure the message surfaces on the card.
  const trackedJobs = useJobTracker();
  useEffect(() => {
    const entries = Object.entries(pullFrameJobs);
    if (entries.length === 0) return;
    for (const [key, jobId] of entries) {
      const rec = trackedJobs.find((r) => r.jobId === jobId);
      if (!rec) continue;
      if (rec.status === "done") {
        const out = rec.outputs[0];
        setParts((prev) =>
          prev.map((p) =>
            p.key === key && out
              ? { ...p, frame: { ref: out, origin: "frames", label: rec.label.replace(/^pull /, "") } }
              : p,
          ),
        );
        if (!out) setPullFrameError((prev) => ({ ...prev, [key]: "The job produced no frame." }));
      } else if (rec.status === "failed" || rec.status === "cancelled" || rec.expired) {
        setPullFrameError((prev) => ({ ...prev, [key]: rec.error ?? "Frame pull failed." }));
      } else {
        continue; // still running
      }
      setPullFrameJobs((prev) => {
        const { [key]: _drop, ...rest } = prev;
        return rest;
      });
    }
  }, [trackedJobs, pullFrameJobs, setParts]);
  // ---- Director knobs (opt-in vision scoring / re-roll loop) ----
  const [visionEnabled, setVisionEnabled] = useSessionState<boolean>(
    "gen.visionEnabled",
    false,
  );
  // 0..100
  const [scoreThreshold, setScoreThreshold] = useSessionState<string>("gen.scoreThreshold", "70");
  // re-rolls per segment
  const [maxAttempts, setMaxAttempts] = useSessionState<string>("gen.maxAttempts", "3");
  // seconds; blank = none
  const [timeBudget, setTimeBudget] = useSessionState<string>("gen.timeBudget", "");
  const [judgeModelId, setJudgeModelId] = useSessionState<string>(
    "gen.judgeModelId",
    DEFAULT_JUDGE_MODEL,
  );

  // ---- shared image-model dropdown ----
  const { models, defaultId, loading: modelsLoading, error: modelsError, refresh: refreshModels } =
    useImageModels();
  const [modelId, setModelId] = useState<string>(() => {
    try {
      return sessionStorage.getItem(MODEL_KEY) ?? "";
    } catch {
      return "";
    }
  });
  useEffect(() => {
    if (!modelId && defaultId) setModelId(defaultId);
  }, [defaultId, modelId]);
  useEffect(() => {
    if (!modelId) return;
    try {
      sessionStorage.setItem(MODEL_KEY, modelId);
    } catch {
      /* ignore */
    }
  }, [modelId]);

  // ---- capability-driven required inputs ----
  // Resolve the selected model's ModelOption (same list backing the <select>) and
  // derive what it REQUIRES from its tasks[]. Empty/unknown tasks fall back to
  // text + optional-image (see requiredInputsFor). This drives the explicit prompt
  // + start-image affordances and folds into the Run gate — orthogonally to `mode`
  // and `chain`, which stay as-is.
  const selectedModel = models.find((m) => m.id === modelId) ?? null;
  const req = requiredInputsFor(selectedModel?.tasks ?? []);
  const startImageRequired = req.image === "required";

  // ---- curated preset dropdown (the "ideal default loads") ----
  // Selecting a preset (a) prefills every knob + the model from the preset, then
  // (b) POSTs the apply endpoint to pre-warm that model on a GPU worker. The prefill
  // happens regardless of the apply outcome (a failed pre-warm still leaves the form
  // filled so the user can retry Run — the model otherwise loads lazily on Run).
  const { presets, loading: presetsLoading, error: presetsError } = usePresets();
  const [presetId, setPresetId] = useSessionState<string>("gen.presetId", "");
  const [presetStatus, setPresetStatus] = useState<string | null>(null);
  const [presetError, setPresetError] = useState<string | null>(null);
  const selectedPreset = presets.find((p) => p.id === presetId) ?? null;

  // ---- curated MOVIE TEMPLATE dropdown (Movie tab only) ----
  // Unlike a video preset (a knob-only "load"), a movie template carries the WHOLE
  // shot list: picking one drops a ready goal timeline + the shared knobs into the
  // editor. The list only fetches once the Movie tab is entered (gated on `mode`).
  const {
    presets: moviePresets,
    loading: moviePresetsLoading,
    error: moviePresetsError,
  } = useMoviePresets(mode === "movie");
  const [moviePresetId, setMoviePresetId] = useSessionState<string>("gen.moviePresetId", "");
  const selectedMoviePreset = moviePresets.find((p) => p.id === moviePresetId) ?? null;

  // Prefill knobs + model + sub-mode from a preset (or the authoritative apply echo,
  // which shares the {model_key, mode, defaults} shape). String setters get the
  // numeric defaults stringified; a null/absent field leaves the current value.
  function applyPresetLoad(load: {
    model_key?: string | null;
    mode?: string | null;
    defaults?: PresetDefaults | null;
  }): void {
    if (load.model_key) setModelId(load.model_key);
    // Map the preset's declared mode onto Scene's Output selector (the Image tab
    // folded into Scene). "text-to-image" → Scene with Output=Image (1 still, no
    // mp4); "edit-chain"/"img2img" (and anything else that walks frames) → Scene
    // with multi-frame output, conditioning each frame (chain on). A preset's own
    // n_frames default (below) still overrides these seeds.
    if (load.mode) {
      setMode("scene");
      if (load.mode === "text-to-image") {
        setNFrames("1");
        setAssemble(false);
      } else {
        setChain(true);
        // Leave a multi-frame count alone; only bump OFF the 1-frame Image output.
        setNFrames((prev) => {
          const n = toNumberOrNull(prev);
          return n == null || n <= 1 ? "6" : prev;
        });
      }
    }
    const d = load.defaults;
    if (d) {
      if (d.width != null) setWidth(String(d.width));
      if (d.height != null) setHeight(String(d.height));
      if (d.steps != null) setSteps(String(d.steps));
      if (d.guidance != null) setGuidance(String(d.guidance));
      if (d.n_frames != null) setNFrames(String(d.n_frames));
      if (d.fps != null) setFps(String(d.fps));
      if (d.strength != null) setStrength(String(d.strength));
      if (d.negative != null) {
        // A preset's negative REPLACES the default outright (pre-k89 it replaced
        // the prefilled text) — so the standard-set tick clears too, keeping the
        // effective payload identical to the old override semantics.
        setNegative(d.negative);
        // k93 (Scene): presets WRITE the negative into part #1's visible field —
        // never a silent merge at submit. Replaces the whole field (override).
        const presetNeg = d.negative;
        setParts((prev) => {
          const i = prev.findIndex((p) => !p.media);
          return i < 0 ? prev : prev.map((p, j) => (j === i ? { ...p, negative: presetNeg } : p));
        });
        setGoals((prev) =>
          prev.map((g, i) => (i === 0 ? { ...g, stdNegative: false } : g)),
        );
      }
    }
  }

  async function onPickPreset(id: string): Promise<void> {
    setPresetId(id);
    setPresetError(null);
    if (!id) {
      setPresetStatus(null);
      return;
    }
    const preset: Preset | undefined = presets.find((p) => p.id === id);
    if (!preset) return;

    // Optimistic prefill from the selected preset FIRST — so a failed pre-warm
    // still leaves the knobs + model populated for a manual Run.
    applyPresetLoad(preset);
    setPresetStatus(`warming ${preset.name}…`);

    const res = await request<unknown>(presetApplyUrl(id), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
      meta: { specKey: "generate", operation: "preset.apply" },
    });
    if (!res.ok) {
      setPresetStatus(null);
      setPresetError(presetApplyErrorText(errorOf(res)));
      return;
    }
    const parsed = presetApplyResponseSchema.safeParse(okValue(res));
    if (!parsed.success || parsed.data.ok !== true) {
      setPresetStatus(null);
      setPresetError(
        (parsed.success && parsed.data.error?.message) ||
          "Couldn't warm this preset. The knobs are prefilled — you can still hit Run.",
      );
      return;
    }
    // Re-apply from the AUTHORITATIVE echo — the server may have adjusted the model
    // or defaults to fit the worker it actually warmed.
    applyPresetLoad(parsed.data);
    const worker = parsed.data.worker?.name || "a GPU worker";
    setPresetStatus(`warming ${preset.name} on ${worker}`);
  }

  // Drop a curated MOVIE TEMPLATE into the editor. Mirrors applyPresetLoad's idiom
  // (same setters, numeric knobs stringified, a null/absent field leaves the current
  // value) and adds the key behavior: REPLACE the goal timeline with the template's
  // goals mapped to editor rows (fresh ids; start/end stringified). Does NOT auto-run.
  function applyMoviePreset(preset: MoviePreset): void {
    // Shared knobs (string-typed — stringify the numeric wire values).
    if (preset.model_key) setModelId(preset.model_key);
    if (preset.width != null) setWidth(String(preset.width));
    if (preset.height != null) setHeight(String(preset.height));
    if (preset.steps != null) setSteps(String(preset.steps));
    if (preset.guidance != null) setGuidance(String(preset.guidance));
    if (preset.fps != null) setFps(String(preset.fps));
    if (preset.chain != null) setChain(preset.chain);
    // img2img knobs — the negative field is a SHARED left-rail knob (shown in
    // Movie mode too) and strength is a shared movie knob, so both prefill here.
    if (preset.strength != null) setStrength(String(preset.strength));
    if (preset.negative != null) {
      // Same override semantics as applyPresetLoad: a template's negative replaces
      // the default whole, so goal #1's standard-set tick clears with it.
      setNegative(preset.negative);
      setGoals((prev) =>
        prev.map((g, i) => (i === 0 ? { ...g, stdNegative: false } : g)),
      );
    }
    // Populate the goal timeline — the whole shot list drops in, ready to run/tweak.
    if (preset.goals && preset.goals.length > 0) {
      setGoals(
        preset.goals.map((g) => ({
          id: goalKey(),
          start: String(g.start_frame),
          end: String(g.end_frame),
          prompt: g.prompt,
        })),
      );
    }
    // Director defaults (opt-in) — mirror the timeline: set when present, else leave.
    if (preset.vision_enabled != null) setVisionEnabled(preset.vision_enabled);
    if (preset.score_threshold != null) setScoreThreshold(String(preset.score_threshold));
  }

  // Pick handler for the Movie-template <select>. Populates from the ALREADY-fetched
  // list object (the list endpoint carries the full shot list — no apply POST needed).
  // Leaves the select showing the chosen name (matches the video preset dropdown).
  function onPickMoviePreset(id: string): void {
    setMoviePresetId(id);
    if (!id) return;
    const preset = moviePresets.find((p) => p.id === id);
    if (!preset) return;
    applyMoviePreset(preset);
  }

  // Register a composer STAGER while this station is mounted, so the shell-level
  // session-library sidebar can pre-fill the ordered composer from any route:
  // "Add to prompt" injects a media part; Replicate stages a prompt+model;
  // Continue stages a scene's prompt+model+start-frame. All of it drives the SAME
  // setters the Run path reads — no parallel pipeline. Unregisters on unmount.
  useEffect(() => {
    return registerComposer((req: StageRequest) => {
      if (req.model) setModelId(req.model);
      // A staged "image" (Replicate on a single-still run) lands on Scene with
      // Output=Image — the folded home of the old Image tab.
      if (req.mode === "image") {
        setMode("scene");
        setNFrames("1");
        setAssemble(false);
      } else if (req.mode) {
        setMode(req.mode);
      }
      const staged: PartEntry[] = [];
      if (req.text != null && req.text.trim() !== "") {
        staged.push({ key: partKey(), text: req.text });
      }
      for (const p of req.parts ?? []) {
        staged.push({
          key: partKey(),
          text: "",
          media: {
            ref: p.ref,
            origin: p.origin,
            ...(p.label != null ? { label: p.label } : {}),
          },
        });
      }
      if (staged.length === 0 && !req.replace) return;

      // Item C — route a bare "Add to prompt" to the ACTIVE sub-tab. Replicate /
      // Continue PIN a mode + `replace` (they explicitly target the image/scene parts
      // composer), so they keep the parts path; only a mode-less, non-replace stage
      // (requestAddPart) follows the live tab.
      const routeToActive = req.mode == null && !req.replace;
      const live = modeRef.current;

      // MOVIE — attach the first staged IMAGE to the current/last goal's reference
      // image (movie goals already carry an optional "＋ add reference image"), instead
      // of dumping it into the parts bucket that Movie mode never renders. Prefer the
      // last goal still missing a reference; else the last goal.
      if (routeToActive && live === "movie") {
        const firstImage = staged.find(
          (s) => s.media && s.media.ref.kind === "image",
        );
        if (firstImage?.media) {
          const media = firstImage.media;
          setGoals((prev) => {
            if (prev.length === 0) return prev;
            let idx = prev.length - 1;
            for (let i = prev.length - 1; i >= 0; i--) {
              if (!prev[i].ref) {
                idx = i;
                break;
              }
            }
            return prev.map((g, i) => (i === idx ? { ...g, ref: media } : g));
          });
          return;
        }
        // No image to attach (e.g. a video-only stage) — fall through so it is still
        // preserved in the parts bucket rather than silently dropped.
      }

      // STUDIO — routing into the studio surface's image slots is DEFERRED (it lives in
      // a separate component with capability-dependent slots + no image seam yet; see
      // report). Fall through so the media is preserved in the parts bucket.
      setParts((prev) => (req.replace ? staged : [...prev, ...staged]));
    });
  }, []);

  // ---- library (images are pickable prompt parts; videos too) ----
  const library = useMediaLibrary();
  // OPERATOR HISTORY (2026-08-13): a superadmin session back-fills the library
  // from the server record once per session (complete visibility); see the hook.
  useOperatorHistory();
  // GROUPED LIBRARY (operator ask 2026-08-13, v2): the pickers render the
  // library as TYPE-CATEGORIZED collapsible sections (LibraryGroups) instead
  // of a flat, section-filtered grid — everything stays visible + pickable,
  // organized; the CURRENT section's own group (and Inputs) starts expanded.
  const MODE_GROUP: Record<GenMode, string> = {
    image: "images", scene: "scenes", movie: "movies",
    clip: "clips", cinema: "clips",
  };
  const libraryDefaultOpen: readonly string[] = ["inputs", MODE_GROUP[mode]];
  const libraryImages = library.filter((it) => it.ref.kind === "image");
  const libraryVideos = library.filter((it) => it.ref.kind === "video");

  // ---- video source (upload → ingest → video MediaRef), like FrameExtractStation ----
  const [videoPhase, setVideoPhase] = useState<VideoPhase>("idle");
  const [videoError, setVideoError] = useState<string | null>(null);
  const videoBusy = videoPhase === "uploading" || videoPhase === "ingesting";
  // Image browse mirrors the video one — attach was library-only for images.
  const [imagePhase, setImagePhase] = useState<VideoPhase>("idle");
  const [imageError, setImageError] = useState<string | null>(null);
  const imageBusy = imagePhase === "uploading" || imagePhase === "ingesting";

  // ---- job lifecycle ----
  // All hooks are always mounted (React rule-of-hooks); the mode switch just
  // selects which one drives the Run button, status, and preview. With the Image
  // tab folded into Scene, no tab reaches imageJob any more, but the hook and its
  // run path stay wired (operator 2026-08-05): the Tester rows submit the same
  // GenerateImageRequest shape (via useGenerateRowJobs), and a single-still flow
  // may re-adopt the one-shot image job.
  const imageJob = useGenerateJob();
  const sceneJob = useGenerateSceneJob();
  const movieJob = useGenerateMovieJob();

  // Active-mode job surface for the shared Run row (button / status / cap / error).
  // The three hooks share these scalar fields (same types); the mode-specific
  // OUTPUT fields (imageJob.output vs {scene,movie}Job.outputs) + the differently
  // shaped `progress` are read from the concrete hook inside each render arm.
  const activeJob = mode === "movie" ? movieJob : mode === "scene" ? sceneJob : imageJob;
  const status = activeJob.status;
  const jobError = activeJob.error;
  const capped = activeJob.capped;
  // Fan-out job states for parts that STILL EXIST — removing a part must not leave
  // a ghost job that locks Run, inflates the summary, or can't be cancelled.
  const scenePartStates: SceneRowJobState[] = parts
    .map((p) => sceneRowJobs.byRow[p.key])
    .filter((s): s is SceneRowJobState => !!s);
  const sceneAnyRunning = scenePartStates.some((s) => s.running);

  // Scene mode enqueues via the per-part fan-out (sceneRowJobs), not the single
  // sceneJob, so fold its running state in — otherwise the footer label, the
  // status readout, and GenIdentityBar's disabled gate would all read idle while
  // N part-jobs poll.
  const running =
    activeJob.running || (mode === "scene" && sceneAnyRunning);
  const resume = activeJob.resume;
  const activeJobId = activeJob.jobId;
  const project = activeJob.project;
  // Live progress (running) — flat for image/scene, NESTED for movie (a union).
  const progress: AnyProgress | null = activeJob.progress;
  const runningReadout = runningReadoutFor(progress);

  // Scene result partition (frames first, then the assembled clip LAST per contract).
  const sceneFrames = sceneJob.outputs.filter((o) => o.kind === "image");
  const sceneClip = sceneJob.outputs.find((o) => o.kind === "video") ?? null;

  // Movie result partition — image frames + the assembled movie.mp4 (LAST video).
  const movieFrames = movieJob.outputs.filter((o) => o.kind === "image");
  const movieClip =
    [...movieJob.outputs].reverse().find((o) => o.kind === "video") ?? null;
  // Nested movie progress (narrowed by the hook) — the running preview reads it.
  const movieProgress: MovieProgress | null = movieJob.progress;

  // ---- Movie director: judge-VLM registry list (image-text-to-text models) ----
  const {
    models: judgeModels,
    loading: judgeLoading,
    error: judgeError,
    refresh: refreshJudge,
  } = useModelsByTask(JUDGE_TASK);

  // ---- generation-wide IDENTITY (Image/Scene) ------------------------------
  // ONE identity per generation, bindable/clearable from ANY prompt part (a profile
  // IS the whole identity — the same replace-semantics Studio encodes). The hook is
  // mounted exactly ONCE here and its data/callbacks are passed DOWN to every
  // GenIdentityBar as props: useIdentityProfiles() has no cross-instance cache
  // (inFlight/mounted are per-instance refs), so mounting it per part would fire N
  // duplicate GET /video/identity-profiles and hold N independently-stale lists.
  const {
    profiles: identityProfiles,
    loading: identityLoading,
    error: identityError,
    resolveCanonicalView: resolveIdentityCanonicalView,
  } = useIdentityProfiles();
  const [boundIdentity, setBoundIdentity] = useSessionState<BoundIdentity | null>(
    "gen.boundIdentity",
    null,
  );

  /** Drop every part this station minted from a bound identity, keeping the user's
   *  own parts untouched. Shared by clearIdentity and the rebind path. */
  function stripIdentityParts(rows: PartEntry[]): PartEntry[] {
    return rows.filter((p) => p.identitySlug == null);
  }

  /** Bind a profile as THIS GENERATION's character, carried by the ONE canonical
   *  view at `viewIndex` (the view the user picked from GenIdentityBar's canonical
   *  row). Resolves that single CANONICAL turntable view — a render of the profile's
   *  3D model, the same id_lock DNA the resolver uses server-side (see `canonical`
   *  doc in useIdentityProfiles.ts) — back into a MediaRef and appends it as ONE
   *  ordinary IMAGE part, so the character reaches the model through the EXISTING
   *  parts/media wire (no contract change).
   *
   *  EXACTLY ONE, and why (operator 2026-07-16): the generate runner consumes only
   *  the FIRST image part —
   *    abstract_hugpy_dev/src/abstract_hugpy_dev/video_intel/runners/imagegen.py:117
   *      start_frame = image_paths[0] if image_paths else None
   *  (scene.py:539 the same via next(...)). The previous bind minted one part PER
   *  canonical view, so 3 of 4 were silently discarded downstream while the UI
   *  implied the whole turnaround conditioned the shot. The user now chooses which
   *  single view carries the identity; see IDENTITY_CARRYING_VIEWS.
   *
   *  Deliberately NOT `reference_images` (the raw, often-messy source uploads a
   *  profile was built from — anywhere from 1 to 12+ per profile): the picker must
   *  bring a canonical view the identity's own id_lock resolver treats as its DNA,
   *  not an arbitrary slice of source photos.
   *
   *  MERGE SEMANTICS (least-surprising of the options):
   *   • the identity REPLACES any previously-bound identity's part (a profile is the
   *     whole identity — two characters' references must never mix), and rebinding a
   *     DIFFERENT VIEW of the same identity likewise replaces, never accumulates; but
   *   • it NEVER touches parts the user typed or attached themselves: their text and
   *     hand-picked media survive a bind, a rebind and an unbind untouched. The
   *     identity part is APPENDED after the user's, so the typed prompt still leads.
   *  Returns a human error string on failure, or null on success — an identity with
   *  no canonical views (no 3D model rendered yet), an out-of-range view, or a view
   *  whose image won't resolve FAILS LOUDLY in the bar, never silently no-ops and
   *  NEVER falls back to reference_images (that would silently reintroduce the
   *  wrong-pixels bug this fixes, with no indication of which set they actually got). */
  async function bindIdentity(
    profile: IdentityProfile,
    viewIndex: number,
  ): Promise<string | null> {
    if (profile.canonical.length === 0) {
      return `“${profile.name}” has no canonical 3D-model views yet (no turntable render on its active version) — nothing was attached. Render its 3D model first, or pick a different identity.`;
    }
    const path = profile.canonical[viewIndex];
    if (path == null) {
      // Defensive: a stale row (list refreshed mid-pick) must not silently bind the
      // wrong view — it must say so.
      return `That view is no longer part of “${profile.name}”'s canonical set — nothing was attached. Reopen the identity picker and choose again.`;
    }
    // Resolve ONLY the picked view. resolveIdentityCanonical resolves the whole set,
    // so ingesting all N to keep 1 would be wasted work on an 8-view profile; the
    // hook's single-path resolver is the honest tool here.
    const ref = await resolveIdentityCanonicalView(profile, viewIndex);
    if (ref == null) {
      return `Could not load canonical view ${viewIndex + 1} for “${profile.name}” — it may be missing from the store. Nothing was attached.`;
    }
    const minted: PartEntry[] = [
      {
        key: partKey(),
        text: "",
        media: { ref, origin: "identity", label: profile.name },
        identitySlug: profile.slug,
      },
    ];
    // The cardinality contract, enforced rather than merely asserted in prose: a bind
    // mints exactly IDENTITY_CARRYING_VIEWS part(s). Slicing (rather than trusting the
    // literal above) keeps this honest if the constant is ever raised to match a
    // runner that learns real multi-view conditioning.
    setParts((prev) => [
      ...stripIdentityParts(prev),
      ...minted.slice(0, IDENTITY_CARRYING_VIEWS),
    ]);
    setBoundIdentity({ slug: profile.slug, name: profile.name, viewIndex });
    return null;
  }

  /** Unbind the identity: remove exactly the parts it minted, leave the user's alone. */
  function clearIdentity() {
    setParts((prev) => stripIdentityParts(prev));
    setBoundIdentity(null);
  }

  // TRUTHFULNESS GUARD: the identity chip must never claim a character the prompt no
  // longer carries. The existing per-part ✕ / "✕ media" verbs act on ANY part, so a
  // user can hand-remove the identity's last reference part; when that happens the
  // binding is stale and self-clears. (detachMedia already rebuilds the row as a bare
  // {key,text}, dropping the marker — so a detached reference correctly stops
  // counting as the identity's.)
  useEffect(() => {
    if (boundIdentity == null) return;
    const stillThere = parts.some((p) => p.identitySlug === boundIdentity.slug);
    if (!stillThere) setBoundIdentity(null);
  }, [parts, boundIdentity]);

  // ---- part mutations ----
  function addText() {
    setParts((prev) => [...prev, { key: partKey(), text: "", negative: STANDARD_NEGATIVE }]);
  }
  function editText(key: string, text: string) {
    setParts((prev) => prev.map((p) => (p.key === key ? { ...p, text } : p)));
  }
  /** Patch any per-part field(s) — the per-component negative/settings live here. */
  function patchPart(key: string, patch: Partial<PartEntry>) {
    setParts((prev) => prev.map((p) => (p.key === key ? { ...p, ...patch } : p)));
  }
  // Per-part knob resolution + routing for the per-component settings strip. A
  // setting whose SHARE tick (carry) is ON reads/writes the ONE station value
  // (shared across every part); OFF reads/writes this part's own value, falling
  // back to the station value as its initial default. (width+height share one
  // "size" tick, matching the strip.)
  function partKnobValues(p: PartEntry): KnobValues {
    return {
      width: sceneShare.size ? width : p.knobs?.width ?? width,
      height: sceneShare.size ? height : p.knobs?.height ?? height,
      steps: sceneShare.steps ? steps : p.knobs?.steps ?? steps,
      guidance: sceneShare.guidance ? guidance : p.knobs?.guidance ?? guidance,
      seed: sceneShare.seed ? seed : p.knobs?.seed ?? seed,
    };
  }
  function patchPartKnobs(p: PartEntry, patch: Partial<KnobValues>) {
    const partPatch: Partial<KnobValues> = {};
    for (const k of Object.keys(patch) as (keyof KnobValues)[]) {
      const v = patch[k] as string;
      const shared =
        k === "width" || k === "height"
          ? sceneShare.size
          : k === "steps"
            ? sceneShare.steps
            : k === "guidance"
              ? sceneShare.guidance
              : sceneShare.seed; // seed
      if (shared) {
        if (k === "width") setWidth(v);
        else if (k === "height") setHeight(v);
        else if (k === "steps") setSteps(v);
        else if (k === "guidance") setGuidance(v);
        else setSeed(v);
      } else {
        partPatch[k] = v;
      }
    }
    if (Object.keys(partPatch).length)
      patchPart(p.key, { knobs: { ...p.knobs, ...partPatch } });
  }

  // ---- per-part model + scene knobs + project, with SHARE ticks ----
  // Resolve a part's non-core settings: a SHARE-ticked setting reads the ONE
  // station value; unticked reads the part's own value, defaulting to the station
  // value.
  function partExtra(p: PartEntry): PartExtra {
    return {
      model: sceneShare.model ? modelId : p.extra?.model ?? modelId,
      nFrames: sceneShare.frames ? nFrames : p.extra?.nFrames ?? nFrames,
      motion: sceneShare.motion ? motion : p.extra?.motion ?? motion,
      fps: sceneShare.fps ? fps : p.extra?.fps ?? fps,
      assemble: sceneShare.assemble ? assemble : p.extra?.assemble ?? assemble,
      strength: sceneShare.strength ? strength : p.extra?.strength ?? strength,
      chain: sceneShare.chain ? chain : p.extra?.chain ?? chain,
      project: sceneShare.project ? projectName : p.extra?.project ?? projectName,
    };
  }
  // Map a UI setting key to whether it is shared, and route the edit to either the
  // station setter (shared) or the part's own `extra` override (unshared).
  function patchPartExtra(
    p: PartEntry,
    key: "model" | "frames" | "motion" | "fps" | "assemble" | "strength" | "chain" | "project",
    value: string | boolean,
  ): void {
    const shared = sceneShare[key];
    if (shared) {
      switch (key) {
        case "model":
          setModelId(value as string);
          break;
        case "frames":
          setNFrames(value as string);
          break;
        case "motion":
          setMotion(value as string);
          break;
        case "fps":
          setFps(value as string);
          break;
        case "assemble":
          setAssemble(value as boolean);
          break;
        case "strength":
          setStrength(value as string);
          break;
        case "chain":
          setChain(value as boolean);
          break;
        case "project":
          setProjectName(value as string);
          break;
      }
      return;
    }
    const field: keyof PartExtra =
      key === "frames" ? "nFrames" : (key as keyof PartExtra);
    patchPart(p.key, { extra: { ...p.extra, [field]: value } });
  }
  const isShared = (key: SceneShareKey): boolean => sceneShare[key];
  function setSharedKey(key: SceneShareKey, v: boolean): void {
    setSceneShare((prev) => ({ ...prev, [key]: v }));
    if (!v) return;
    // Sharing turned ON: drop every part's own override for this setting, so a
    // later un-share resolves from the CURRENT shared value (via `?? shared`)
    // rather than resurrecting a stale override the user can no longer see.
    setParts((prev) =>
      prev.map((p) => {
        if (key === "size" || key === "steps" || key === "guidance" || key === "seed") {
          if (!p.knobs) return p;
          const knobs = { ...p.knobs };
          if (key === "size") {
            delete knobs.width;
            delete knobs.height;
          } else {
            delete knobs[key];
          }
          return { ...p, knobs };
        }
        if (!p.extra) return p;
        const field: keyof PartExtra = key === "frames" ? "nFrames" : key;
        const extra = { ...p.extra };
        delete extra[field];
        return { ...p, extra };
      }),
    );
  }
  // A tiny share-checkbox for the per-part settings strip: ☑ = one value across
  // all prompts, unticked = this prompt keeps its own.
  function shareTick(key: SceneShareKey, label: string) {
    return (
      <input
        type="checkbox"
        className="vi-carry-tick"
        checked={isShared(key)}
        onChange={(e) => setSharedKey(key, e.target.checked)}
        aria-label={`Share ${label} across all prompts`}
        title={`Share ${label} across all prompts. Unticked = this prompt keeps its own ${label}.`}
      />
    );
  }

  // k93 §C (client half): the toolbar's `[+ video]` pretext as the typed
  // `context.media` the assist routes accept — {uri, mime, label}. Undefined
  // when nothing is attached so the body stays byte-identical to today.
  function scenePretextWire(): AssistMediaWire | undefined {
    if (!scenePretext) return undefined;
    const media: AssistMediaWire = {
      uri: scenePretext.ref.uri,
      mime: scenePretext.ref.mime,
    };
    if (scenePretext.label) media.label = scenePretext.label;
    return media;
  }

  // Per-part Enhance/Generate — writes the prompt AND (when this card's
  // `negatives` toggle is on — default) its negative. The negative is derived from
  // the NEW prompt (captured from runAssist's apply); Enhance refines the existing
  // negative, Generate writes a fresh one. Sequential because the assist hook
  // allows one in-flight call at a time. `opts.negatives` lets the GROUP enhance
  // override the per-card toggle with the toolbar's; `opts.media` is the group
  // pretext (k93 §C) — the per-card buttons pass none.
  async function runPartAssist(
    p: PartEntry,
    mode: "detail" | "generate",
    opts: { negatives?: boolean; media?: AssistMediaWire } = {},
  ) {
    // This part's OWN assist model (falls back to the shared pick / fleet default).
    const model = p.assistModel ?? assist.assistModel ?? null;
    const context = opts.media ? { media: opts.media } : undefined;
    let newText = p.text;
    await assist.runAssist(
      mode,
      p.text,
      (t) => {
        newText = t;
        editText(p.key, t);
      },
      { model, ...(context ? { context } : {}) },
    );
    if (newText.trim() === "") return;
    const writeNegative = opts.negatives ?? p.assistNegatives ?? true;
    if (!writeNegative) return;
    await assist.runNegative(
      {
        subject: newText,
        draft: mode === "detail" ? p.negative ?? "" : "",
        extra: { model, ...(context ? { context } : {}) },
      },
      // k93: everything the wire will carry is IN the field. If the field held
      // the standard set before this call, re-insert it (the std. tick stays
      // honest — it reads the field); nothing is composed at submit.
      (neg) => {
        const std = hasStandardNegative(p.negative ?? "");
        // APPEND, never clobber (operator 2026-08-13): a GENERATED negative adds
        // to whatever the operator already typed (Enhance still refines-in-place —
        // its draft was the existing text, so appending would double it).
        const cur = (p.negative ?? "").trim().replace(/,\s*$/, "");
        const mergedRaw = mode === "generate" && cur ? `${cur}, ${neg}` : neg;
        patchPart(p.key, { negative: composeNegative(mergedRaw, std) });
      },
    );
    // NAME RIDES ALONG (operator 2026-08-13): derive the project name from the
    // fresh prompt when blank — or when the current one was itself derived.
    if (mode === "generate" && newText.trim()) {
      const t = projectName.trim();
      if (t === "" || t === autoProjectRef.current) {
        const derived = deriveNameFrom(newText);
        if (derived) {
          autoProjectRef.current = derived;
          setProjectName(derived);
        }
      }
    }
  }

  // "Test all models" / "per preset" as COMPONENTS — the single reusable
  // architecture (a prompt component IS what the old tester row was). Clone the
  // base prompt into one component per model / preset, each carrying its own model
  // (un-shared) so Run fans them out to one output each in the viewer.
  function basePromptText(): string {
    const p = parts.find((x) => !x.media && x.text.trim() !== "");
    return p ? p.text : assembledPrompt;
  }
  /** The base part's VISIBLE negative — a clone sends what its source shows (k93). */
  function basePartNegative(): string {
    const p = parts.find((x) => !x.media && x.text.trim() !== "") ?? parts.find((x) => !x.media);
    return p?.negative ?? "";
  }
  function addPartPerModel(): void {
    const text = basePromptText();
    if (text.trim() === "") return;
    // Skip models that REQUIRE a start image when there's no shared media to give
    // them one — they'd only produce dead, un-runnable cards.
    const hasSharedImage = sceneSharedMedia().some((x) => x.kind === "image");
    const targets = models.filter(
      (m) =>
        !m.disabled &&
        (hasSharedImage || requiredInputsFor(m.tasks).image !== "required"),
    );
    if (targets.length === 0) return;
    setSceneShare((prev) => ({ ...prev, model: false }));
    setParts((prev) => [
      ...prev,
      ...targets.map((m) => ({
        key: partKey(),
        text,
        negative: basePartNegative(),
        extra: { model: m.id } as Partial<PartExtra>,
      })),
    ]);
  }
  function addPartPerPreset(): void {
    const text = basePromptText();
    if (text.trim() === "" || presets.length === 0) return;
    setSceneShare((prev) => ({
      ...prev,
      model: false,
      size: false,
      steps: false,
      guidance: false,
    }));
    setParts((prev) => [
      ...prev,
      ...presets.map((pre) => {
        const d = pre.defaults;
        const knobs: Partial<KnobValues> = {};
        if (d.width != null) knobs.width = String(d.width);
        if (d.height != null) knobs.height = String(d.height);
        if (d.steps != null) knobs.steps = String(d.steps);
        if (d.guidance != null) knobs.guidance = String(d.guidance);
        const part: PartEntry = {
          key: partKey(),
          text,
          negative: basePartNegative(),
          extra: { model: pre.model_key || modelId },
          knobs,
        };
        // A preset's negative REPLACES the field outright (visible, sent as-is —
        // k93: nothing is composed at submit).
        if (d.negative != null) part.negative = d.negative;
        return part;
      }),
    ]);
  }

  // GROUP settings (operator: like the group prompt-assist, but for settings).
  // Editing a value here applies it to the SELECTED text components: it un-shares
  // that setting (so per-part values take effect) and writes it as each selected
  // part's own override. Unselected parts have no override, so they keep the
  // current shared value as their default and are untouched.
  function applyGroupKnobs(patch: Partial<KnobValues>) {
    const keys = Object.keys(patch) as (keyof KnobValues)[];
    setSceneShare((prev) => {
      const next = { ...prev };
      for (const k of keys)
        next[k === "width" || k === "height" ? "size" : (k as SceneShareKey)] = false;
      return next;
    });
    setParts((prev) =>
      prev.map((p) =>
        sceneSelectedIds.has(p.key) && !p.media
          ? { ...p, knobs: { ...p.knobs, ...patch } }
          : p,
      ),
    );
  }
  function applyGroupExtra(
    key: "model" | "frames" | "motion" | "fps" | "assemble" | "strength" | "chain" | "project",
    value: string | boolean,
  ) {
    setSceneShare((prev) => ({ ...prev, [key]: false }));
    const field: keyof PartExtra = key === "frames" ? "nFrames" : key;
    setParts((prev) =>
      prev.map((p) =>
        sceneSelectedIds.has(p.key) && !p.media
          ? { ...p, extra: { ...p.extra, [field]: value } }
          : p,
      ),
    );
  }
  // A group settings strip in the toolbar — reuses the per-part strip's inputs, but
  // WITHOUT share ticks (it applies to the selected parts instead). Baseline values
  // shown are the shared station values.
  function renderGroupSettings() {
    if (sceneSelectedCount === 0) return null;
    return (
      <PromptCardSettings
        label={`settings → apply to ${sceneSelectedCount} selected prompt${sceneSelectedCount === 1 ? "" : "s"}`}
      >
        <div className="vi-knob-strip">
          <div className="vi-knob-strip-line">
            <span className="vi-knob-mini-group vi-knob-mini-model">
              <label className="vi-knob-mini-label" htmlFor="vi-gen-group-model">
                model
              </label>
              <select
                id="vi-gen-group-model"
                className="vi-knob-select vi-knob-select-mini"
                value={modelId}
                onChange={(e) => applyGroupExtra("model", e.target.value)}
                disabled={modelsLoading}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id} disabled={m.disabled} title={m.reason}>
                    {m.label}
                  </option>
                ))}
              </select>
            </span>
          </div>
        </div>
        <CondensedKnobStrip
          idPrefix="vi-gen-group"
          values={{ width, height, steps, guidance, seed }}
          onPatch={applyGroupKnobs}
        />
        <div className="vi-knob-strip">
          <div className="vi-knob-strip-line">
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor="vi-gen-group-frames">
                frames
              </label>
              <input
                id="vi-gen-group-frames"
                type="text"
                inputMode="numeric"
                className="vi-knob-input vi-knob-mini"
                value={nFrames}
                onChange={(e) => applyGroupExtra("frames", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor="vi-gen-group-fps">
                fps
              </label>
              <input
                id="vi-gen-group-fps"
                type="text"
                inputMode="numeric"
                className="vi-knob-input vi-knob-mini"
                value={fps}
                onChange={(e) => applyGroupExtra("fps", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor="vi-gen-group-strength">
                strength
              </label>
              <input
                id="vi-gen-group-strength"
                type="text"
                inputMode="decimal"
                className="vi-knob-input vi-knob-mini"
                value={strength}
                onChange={(e) => applyGroupExtra("strength", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor="vi-gen-group-motion">
                motion
              </label>
              <input
                id="vi-gen-group-motion"
                type="text"
                className="vi-knob-input vi-knob-mini"
                placeholder="none"
                value={motion}
                onChange={(e) => applyGroupExtra("motion", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor="vi-gen-group-assemble">
                assemble
              </label>
              <input
                id="vi-gen-group-assemble"
                type="checkbox"
                checked={assemble}
                onChange={(e) => applyGroupExtra("assemble", e.target.checked)}
              />
            </span>
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor="vi-gen-group-chain">
                chain
              </label>
              <input
                id="vi-gen-group-chain"
                type="checkbox"
                checked={chain}
                onChange={(e) => applyGroupExtra("chain", e.target.checked)}
              />
            </span>
          </div>
        </div>
      </PromptCardSettings>
    );
  }

  // The COMPLETE per-component settings expander (operator: each prompt owns its
  // full settings). Every setting carries a ☑ share tick — ticked reads/writes the
  // one shared value; unticked is this prompt's own. Model + the 5 core knobs +
  // the scene knobs (frames/fps/strength/motion/assemble/chain) + project.
  function renderPartSettings(p: PartEntry) {
    const ex = partExtra(p);
    const id = (s: string) => `vi-gen-part-${p.key}-${s}`;
    return (
      <PromptCardSettings label="settings">
        {/* model */}
        <div className="vi-knob-strip">
          <div className="vi-knob-strip-line">
            <span className="vi-knob-mini-group vi-knob-mini-model">
              {shareTick("model", "the model")}
              <label className="vi-knob-mini-label" htmlFor={id("model")}>
                model
              </label>
              <select
                id={id("model")}
                className="vi-knob-select vi-knob-select-mini"
                value={ex.model}
                onChange={(e) => patchPartExtra(p, "model", e.target.value)}
                disabled={modelsLoading}
              >
                {modelsLoading && <option value="">Loading models…</option>}
                {!modelsLoading && models.every((m) => m.disabled) && (
                  <option value="">No image models available</option>
                )}
                {models.map((m) => (
                  <option key={m.id} value={m.id} disabled={m.disabled} title={m.reason}>
                    {m.label}
                  </option>
                ))}
              </select>
            </span>
          </div>
        </div>

        {/* size / steps / guidance / seed — share ticks ride the strip */}
        <CondensedKnobStrip
          idPrefix={`vi-gen-part-${p.key}`}
          values={partKnobValues(p)}
          onPatch={(patch) => patchPartKnobs(p, patch)}
          carry={{
            model: sceneShare.model,
            size: sceneShare.size,
            steps: sceneShare.steps,
            guidance: sceneShare.guidance,
            seed: sceneShare.seed,
            negative: false,
          }}
          onCarry={(k, v) => setSharedKey(k as SceneShareKey, v)}
        />

        {/* scene knobs + project */}
        <div className="vi-knob-strip">
          <div className="vi-knob-strip-line">
            <span className="vi-knob-mini-group">
              {shareTick("frames", "frames")}
              <label className="vi-knob-mini-label" htmlFor={id("frames")}>
                frames
              </label>
              <input
                id={id("frames")}
                type="text"
                inputMode="numeric"
                className="vi-knob-input vi-knob-mini"
                value={ex.nFrames}
                onChange={(e) => patchPartExtra(p, "frames", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              {shareTick("fps", "fps")}
              <label className="vi-knob-mini-label" htmlFor={id("fps")}>
                fps
              </label>
              <input
                id={id("fps")}
                type="text"
                inputMode="numeric"
                className="vi-knob-input vi-knob-mini"
                value={ex.fps}
                onChange={(e) => patchPartExtra(p, "fps", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              {shareTick("strength", "strength")}
              <label className="vi-knob-mini-label" htmlFor={id("strength")}>
                strength
              </label>
              <input
                id={id("strength")}
                type="text"
                inputMode="decimal"
                className="vi-knob-input vi-knob-mini"
                value={ex.strength}
                onChange={(e) => patchPartExtra(p, "strength", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              {shareTick("motion", "motion")}
              <label className="vi-knob-mini-label" htmlFor={id("motion")}>
                motion
              </label>
              <input
                id={id("motion")}
                type="text"
                className="vi-knob-input vi-knob-mini"
                placeholder="none"
                value={ex.motion}
                onChange={(e) => patchPartExtra(p, "motion", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              {shareTick("assemble", "assemble mp4")}
              <label className="vi-knob-mini-label" htmlFor={id("assemble")}>
                assemble
              </label>
              <input
                id={id("assemble")}
                type="checkbox"
                checked={ex.assemble}
                onChange={(e) => patchPartExtra(p, "assemble", e.target.checked)}
              />
            </span>
            <span className="vi-knob-mini-group">
              {shareTick("chain", "chain frames")}
              <label className="vi-knob-mini-label" htmlFor={id("chain")}>
                chain
              </label>
              <input
                id={id("chain")}
                type="checkbox"
                checked={ex.chain}
                onChange={(e) => patchPartExtra(p, "chain", e.target.checked)}
              />
            </span>
            <span className="vi-knob-mini-group">
              {shareTick("project", "project")}
              <label className="vi-knob-mini-label" htmlFor={id("project")}>
                project
              </label>
              <input
                id={id("project")}
                type="text"
                className="vi-knob-input vi-knob-mini"
                placeholder="auto"
                value={ex.project}
                onChange={(e) => patchPartExtra(p, "project", e.target.value)}
              />
            </span>
          </div>
          {partSceneFlags(p).length > 0 && (
            <div className="vi-knob-strip-flags">
              {partSceneFlags(p).map((f) => (
                <span className="vi-knob-flag" key={f}>
                  {f}
                </span>
              ))}
            </div>
          )}
        </div>
      </PromptCardSettings>
    );
  }

  // One prompt component's compact STATUS under its card (running/cancel, capped
  // resume, error, or a done tick). The actual frames/clip live in the central
  // VIEWER (renderSceneViewer), not here — the card shows only where this prompt's
  // job is at.
  function renderPartResult(p: PartEntry) {
    const j = sceneRowJobs.byRow[p.key];
    if (!j) return null;
    const hasOutput = j.outputs.length > 0;
    return (
      <div className="vi-gen-part-result">
        {j.running && (
          <span className="vi-status vi-status-running">
            {j.progress ? progressReadout(j.progress) : "running…"}
            {j.jobId && (
              <button
                type="button"
                className="vi-btn vi-btn-sm vi-btn-ghost"
                onClick={() => cancelJob(j.jobId as string)}
                title="Cancel this prompt's generation"
              >
                ✕ cancel
              </button>
            )}
          </span>
        )}
        {!j.running && j.capped && (
          <span className="vi-status">
            Still running on the server —{" "}
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              onClick={() => sceneRowJobs.resume(p.key)}
              title="Re-anchor the poll clock and check this prompt's job again"
            >
              Check again
            </button>
          </span>
        )}
        {j.error && (
          <span className="vi-status vi-status-failed" title={j.error}>
            {j.error}
          </span>
        )}
        {!j.running && !j.capped && !j.error && (
          <span className="vi-status">
            {hasOutput ? "✓ done — see the viewer above" : j.status ? `${j.status}…` : "queued…"}
          </span>
        )}
      </div>
    );
  }

  // The VIEWER: every prompt's output housed centrally, labeled by its prompt,
  // instead of stacked under the prompt cards. Each item shows its clip and/or
  // frames (plus inline running/capped/error status).
  function renderSceneViewer() {
    const rows = parts
      .map((p) => ({ p, j: sceneRowJobs.byRow[p.key] }))
      .filter((r): r is { p: PartEntry; j: SceneRowJobState } => !!r.j);
    if (rows.length === 0)
      return (
        <div className="vi-gen-preview-pending">
          <span className="vi-status">
            Add prompts below, give each its own settings, then Run — each
            prompt&apos;s output appears here.
          </span>
        </div>
      );
    const running = rows.filter((r) => r.j.running).length;
    const done = rows.filter((r) => r.j.status === "done").length;
    const failed = rows.filter((r) => r.j.status === "failed").length;
    return (
      <div className="vi-gen-result vi-scene-result vi-scene-viewer">
        <div className="vi-gen-note">
          {rows.length} prompt{rows.length === 1 ? "" : "s"} — {running} running ·{" "}
          {done} done{failed ? ` · ${failed} failed` : ""}
        </div>
        {rows.map(({ p, j }, idx) => {
          const frames = j.outputs.filter((o) => o.kind === "image");
          const clip = j.outputs.find((o) => o.kind === "video") ?? null;
          const label = p.text.trim() || `prompt ${idx + 1}`;
          return (
            <figure key={p.key} className="vi-scene-viewer-item">
              <figcaption className="vi-scene-viewer-label" title={label}>
                {idx + 1}. {label.length > 70 ? `${label.slice(0, 70)}…` : label}
              </figcaption>
              {j.running && (
                <span className="vi-status vi-status-running">
                  {j.progress ? progressReadout(j.progress) : "running…"}
                </span>
              )}
              {!j.running && j.capped && (
                <span className="vi-status">
                  Still running on the server —{" "}
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm vi-btn-ghost"
                    onClick={() => sceneRowJobs.resume(p.key)}
                  >
                    Check again
                  </button>
                </span>
              )}
              {j.error && (
                <span className="vi-status vi-status-failed" title={j.error}>
                  {j.error}
                </span>
              )}
              {clip && (
                <div className="vi-scene-clip">
                  <div className="vi-video-clip">
                    <video
                      controls
                      className="vi-video-player"
                      src={mediaBytesUrl(clip.uri)}
                    />
                  </div>
                  <a
                    className="vi-btn vi-btn-sm"
                    href={mediaBytesUrl(clip.uri)}
                    download={`scene-${idx + 1}.${extFromMime(clip.mime)}`}
                  >
                    ↓ MP4
                  </a>
                </div>
              )}
              {frames.length > 0 && (
                <div className="vi-frames-grid vi-scene-frames vi-studio-strip">
                  {frames.map((f, i) => (
                    <span className="vi-scene-frame-cell" key={f.uri}>
                      <img
                        className="vi-scene-frame-img"
                        src={mediaBytesUrl(f.uri)}
                        alt={`prompt ${idx + 1} frame ${i + 1}`}
                        loading="lazy"
                      />
                      <span className="vi-frame-tag">#{i + 1}</span>
                      <a
                        className="vi-frame-dl"
                        href={mediaBytesUrl(f.uri)}
                        download={`scene-${idx + 1}-frame-${i + 1}.${extFromMime(f.mime)}`}
                        title={`Download frame ${i + 1}`}
                        aria-label={`Download prompt ${idx + 1} frame ${i + 1}`}
                      >
                        ↓
                      </a>
                    </span>
                  ))}
                </div>
              )}
            </figure>
          );
        })}
      </div>
    );
  }

  // Build the ordered prompt parts for ONE component (its own text + own media) —
  // the per-part analogue of buildPromptParts, used by the fan-out submit.
  function buildPartPromptParts(p: PartEntry): GenPromptPart[] {
    const out: GenPromptPart[] = [];
    if (!p.media || p.text.trim() !== "") out.push({ kind: "text", text: p.text });
    // k93: a frame pulled from the attached video rides as an IMAGE part placed
    // before the video, so it is the start frame (image_paths[0]).
    if (p.frame) out.push({ kind: "image", media: p.frame.ref });
    if (p.media)
      out.push(
        p.media.ref.kind === "video"
          ? { kind: "video", media: p.media.ref }
          : { kind: "image", media: p.media.ref },
      );
    return out;
  }

  // Media-only components (bound identity, a standalone start frame, staged media
  // with no caption) are SHARED references: they condition every prompt's scene
  // (restoring the old single-prompt start-frame semantics), so they lead each
  // fanned-out request. Returns just their image/video parts.
  function sceneSharedMedia(): GenPromptPart[] {
    return parts
      .filter((p) => p.text.trim() === "" && !!p.media)
      .flatMap((p) => buildPartPromptParts(p))
      .filter((x) => x.kind !== "text");
  }

  // Per-part validity for the scene knobs (same rules the station/strip use). The
  // core 5 (size/steps/guidance/seed) already flag via CondensedKnobStrip; these
  // cover frames/fps/strength so an invalid override is visible, not silently
  // dropped at submit.
  function partSceneFlags(p: PartEntry): string[] {
    const ex = partExtra(p);
    const out: string[] = [];
    const nf = toNumberOrNull(ex.nFrames);
    const fp = toNumberOrNull(ex.fps);
    const str = toNumberOrNull(ex.strength);
    if (!(nf != null && Number.isInteger(nf) && nf >= 1 && nf <= 24))
      out.push("frames: enter an integer 1–24.");
    if (!(fp != null && Number.isInteger(fp) && fp >= 1))
      out.push("fps: enter an integer ≥ 1.");
    if (!(str != null && str >= 0 && str <= 1))
      out.push("strength: enter a number 0–1.");
    return out;
  }

  // Build ONE scene request from a component, or null when any resolved knob is
  // invalid (skip, never submit coerced garbage). `lead` = the shared media that
  // conditions every prompt. Negative (k93): EXACTLY what this part's visible
  // Negative field holds — blank is blank; no shared fallback, nothing composed.
  function buildSceneRow(
    p: PartEntry,
    lead: GenPromptPart[],
  ): { rowId: string; req: GenerateSceneRequest } | null {
    const kv = partKnobValues(p);
    const ex = partExtra(p);
    const w = toNumberOrNull(kv.width);
    const h = toNumberOrNull(kv.height);
    const st = toNumberOrNull(kv.steps);
    const g = toNumberOrNull(kv.guidance);
    const nf = toNumberOrNull(ex.nFrames);
    const fp = toNumberOrNull(ex.fps);
    const str = toNumberOrNull(ex.strength);
    const seedBlank = kv.seed.trim() === "";
    const sd = seedBlank ? null : toNumberOrNull(kv.seed);
    const valid =
      ex.model !== "" &&
      w != null && Number.isInteger(w) && w > 0 &&
      h != null && Number.isInteger(h) && h > 0 &&
      st != null && Number.isInteger(st) && st > 0 &&
      g != null && g >= 0 &&
      nf != null && Number.isInteger(nf) && nf >= 1 && nf <= 24 &&
      fp != null && Number.isInteger(fp) && fp >= 1 &&
      str != null && str >= 0 && str <= 1 &&
      (seedBlank || (sd != null && Number.isInteger(sd) && sd >= 0));
    if (!valid) return null;
    // Order: the part's OWN text+media first, then the shared lead media. The
    // runner uses image_paths[0] as the start frame, so a part that carries its
    // own image keeps it as the start frame; a part with none falls back to the
    // shared lead (bound identity / shared start frame).
    const reqParts = [...buildPartPromptParts(p), ...lead];
    // This part's OWN model may require a start image (edit/inpaint). The station
    // gate can't know each part's model, so enforce it here: skip the row if its
    // model needs an image and none is present (its own media or the shared lead).
    const partModel = models.find((m) => m.id === ex.model) ?? null;
    const partNeedsImage =
      requiredInputsFor(partModel?.tasks ?? []).image === "required";
    if (partNeedsImage && !reqParts.some((x) => x.kind === "image")) return null;
    const neg = (p.negative ?? "").trim();
    return {
      rowId: p.key,
      req: {
        parts: reqParts,
        model_id: ex.model,
        width: w as number,
        height: h as number,
        steps: st as number,
        guidance: g as number,
        n_frames: nf as number,
        fps: fp as number,
        assemble: ex.assemble,
        strength: str as number,
        chain: ex.chain,
        seed: sd,
        motion: ex.motion.trim() === "" ? null : ex.motion,
        negative: neg === "" ? null : neg,
        project: ex.project.trim() || undefined,
      },
    };
  }

  // The fan-out: one runnable row per TEXT-BEARING, fully-valid component, each led
  // by the shared media. Used by both onRun (submit) and canRunScene (gate).
  function computeSceneRows(): { rowId: string; req: GenerateSceneRequest }[] {
    const lead = sceneSharedMedia();
    const rows: { rowId: string; req: GenerateSceneRequest }[] = [];
    for (const p of parts) {
      if (p.text.trim() === "") continue;
      const row = buildSceneRow(p, lead);
      if (row) rows.push(row);
    }
    return rows;
  }
  function removePart(key: string) {
    setParts((prev) => prev.filter((p) => p.key !== key));
    if (attachTarget === key) setAttachTarget(null);
  }
  /** Unpair a part's media (and any frame pulled from it), keeping its text. */
  function detachMedia(key: string) {
    setParts((prev) =>
      prev.map((p) => (p.key === key ? { ...p, media: undefined, frame: undefined } : p)),
    );
  }
  /** Drop only the pulled frame; the video stays attached. */
  function detachFrame(key: string) {
    setParts((prev) => prev.map((p) => (p.key === key ? { ...p, frame: undefined } : p)));
  }
  /**
   * k93 §B.5: open the INLINE attach panel on card `key` (toggles closed when the
   * same one is clicked again). image/video pair the pick onto this part
   * (attachTarget routes the shared upload/ingest handlers here); "library" is
   * the old "+ from library" APPEND path, rendered inline as well.
   */
  function openAttachPicker(key: string, kind: "image" | "video" | "library") {
    const same = partAttachOpen?.key === key && partAttachOpen.kind === kind;
    if (same) {
      setPartAttachOpen(null);
      setAttachTarget(null);
      return;
    }
    setPartAttachOpen({ key, kind });
    setAttachTarget(kind === "library" ? null : key);
    // The station-level pickers are the APPEND path; the card owns attach now.
    setShowImagePicker(false);
    setShowVideoPicker(false);
  }
  // Route a picked media: pair it onto the attach-target part when one is set,
  // else append it as a new part (empty caption, editable in place).
  function placeMedia(media: PartMedia) {
    setPartAttachOpen(null);
    if (attachTarget != null) {
      const target = attachTarget;
      setParts((prev) =>
        prev.map((p) => (p.key === target ? { ...p, media, frame: undefined } : p)),
      );
      setAttachTarget(null);
      return;
    }
    setParts((prev) => [...prev, { key: partKey(), text: "", media }]);
  }

  // ── k93 §B.6 pull-frame: ONE frame at `t` seconds out of this part's attached
  // video, through the existing frame-extract job (station "frames", so the
  // tracker's terminal push lands the frame in the library exactly like the
  // Frames station does). A 0.25 s window at fps 4 yields exactly one frame, and
  // max_frames: 1 keeps the backend's loud cap honest.
  function pullFrame(p: PartEntry) {
    if (!p.media || p.media.ref.kind !== "video") return;
    const raw = (pullFrameAt[p.key] ?? "0").trim();
    const t = toNumberOrNull(raw === "" ? "0" : raw);
    const dur = p.media.ref.duration_s;
    if (t == null || t < 0 || (dur != null && t >= dur)) {
      setPullFrameError((prev) => ({
        ...prev,
        [p.key]: dur != null ? `Enter a time between 0 and ${dur.toFixed(2)} s.` : "Enter a time ≥ 0 s.",
      }));
      return;
    }
    setPullFrameError((prev) => {
      const { [p.key]: _drop, ...rest } = prev;
      return rest;
    });
    const req: FrameExtractRequest = {
      source: p.media.ref,
      fps: 4,
      quality: 90,
      fmt: "jpg",
      window: { start_s: t, end_s: t + 0.25 },
      max_frames: 1,
    };
    const enqueuedAt = Date.now();
    void request<unknown>(hugpyConfig.frameExtractEnqueueUrl, {
      method: "POST",
      body: JSON.stringify(req),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "frames", operation: "frame_extract.enqueue" },
    }).then((r) => {
      if (!r.ok) {
        setPullFrameError((prev) => ({ ...prev, [p.key]: describeAppError(errorOf(r)) }));
        return;
      }
      const parsed = enqueueResultSchema.safeParse(okValue(r));
      if (!parsed.success) {
        setPullFrameError((prev) => ({ ...prev, [p.key]: "Malformed enqueue response." }));
        return;
      }
      trackJob({
        jobId: parsed.data.job_id,
        kind: "frame_extract",
        station: "frames",
        rowKey: p.key,
        label: `pull frame @ ${t}s`,
        enqueuedAt,
      });
      setPullFrameJobs((prev) => ({ ...prev, [p.key]: parsed.data.job_id }));
    });
  }
  function movePart(key: string, dir: -1 | 1) {
    setParts((prev) => {
      const i = prev.findIndex((p) => p.key === key);
      if (i < 0) return prev;
      const j = i + dir;
      if (j < 0 || j >= prev.length) return prev;
      const next = prev.slice();
      const [it] = next.splice(i, 1);
      next.splice(j, 0, it);
      return next;
    });
  }
  // ---- Movie goal-timeline mutations (ordered-parts idiom) ----
  function addGoal() {
    setGoals((prev) => {
      const last = prev[prev.length - 1];
      const start = last ? toNumberOrNull(last.end) ?? 0 : 0;
      return [
        ...prev,
        {
          id: goalKey(),
          start: String(start),
          end: String(start + DEFAULT_GOAL_LEN),
          prompt: "",
        },
      ];
    });
  }
  function removeGoal(id: string) {
    setGoals((prev) => retile(prev.filter((g) => g.id !== id)));
    if (goalAttachTarget === id) setGoalAttachTarget(null);
  }
  function moveGoal(id: string, dir: -1 | 1) {
    setGoals((prev) => {
      const i = prev.findIndex((g) => g.id === id);
      if (i < 0) return prev;
      const j = i + dir;
      if (j < 0 || j >= prev.length) return prev;
      const next = prev.slice();
      const [it] = next.splice(i, 1);
      next.splice(j, 0, it);
      // Re-tile so the reordered windows stay contiguous (lengths preserved).
      return retile(next);
    });
  }
  /** Split a goal at its midpoint into two contiguous goals (needs length ≥ 2). */
  function splitGoal(id: string) {
    setGoals((prev) => {
      const i = prev.findIndex((g) => g.id === id);
      if (i < 0) return prev;
      const g = prev[i];
      const s = toNumberOrNull(g.start);
      const e = toNumberOrNull(g.end);
      if (s == null || e == null || e - s < 2) return prev; // nothing to split
      const mid = Math.floor((s + e) / 2);
      const a: GoalRow = { ...g, end: String(mid) };
      const b: GoalRow = {
        id: goalKey(),
        start: String(mid),
        end: String(e),
        prompt: g.prompt,
        ...(g.ref ? { ref: g.ref } : {}),
      };
      const next = prev.slice();
      next.splice(i, 1, a, b);
      return next;
    });
  }
  function editGoal(id: string, field: "start" | "end" | "prompt", value: string) {
    setGoals((prev) => prev.map((g) => (g.id === id ? { ...g, [field]: value } : g)));
  }
  function detachGoalRef(id: string) {
    setGoals((prev) =>
      prev.map((g) => {
        if (g.id !== id) return g;
        const { ref: _drop, ...rest } = g;
        void _drop;
        return rest;
      }),
    );
  }
  /** Open the image picker to ATTACH a reference image onto goal `id`. */
  function openGoalRefPicker(id: string) {
    setGoalAttachTarget(id);
    setAttachTarget(null);
    setShowVideoPicker(false);
    setShowImagePicker(true);
  }

  // ---- k92: a movie goal's PER-COMPONENT knobs (mirror of the Scene part helpers) ----
  // A goal IS a prompt component; these resolve/route its own settings against the
  // movieShare ticks, exactly as partKnobValues/partExtra/patchPart* do for a part.
  function patchGoal(id: string, patch: Partial<GoalRow>): void {
    setGoals((prev) => prev.map((g) => (g.id === id ? { ...g, ...patch } : g)));
  }
  function goalKnobValues(g: GoalRow): KnobValues {
    return {
      width: movieShare.size ? width : g.knobs?.width ?? width,
      height: movieShare.size ? height : g.knobs?.height ?? height,
      steps: movieShare.steps ? steps : g.knobs?.steps ?? steps,
      guidance: movieShare.guidance ? guidance : g.knobs?.guidance ?? guidance,
      seed: movieShare.seed ? seed : g.knobs?.seed ?? seed,
    };
  }
  function patchGoalKnobs(g: GoalRow, patch: Partial<KnobValues>): void {
    const goalPatch: Partial<KnobValues> = {};
    for (const k of Object.keys(patch) as (keyof KnobValues)[]) {
      const v = patch[k] as string;
      const shared =
        k === "width" || k === "height"
          ? movieShare.size
          : k === "steps"
            ? movieShare.steps
            : k === "guidance"
              ? movieShare.guidance
              : movieShare.seed; // seed
      if (shared) {
        if (k === "width") setWidth(v);
        else if (k === "height") setHeight(v);
        else if (k === "steps") setSteps(v);
        else if (k === "guidance") setGuidance(v);
        else setSeed(v);
      } else {
        goalPatch[k] = v;
      }
    }
    if (Object.keys(goalPatch).length)
      patchGoal(g.id, { knobs: { ...g.knobs, ...goalPatch } });
  }
  function goalExtra(g: GoalRow): GoalExtra {
    return {
      model: movieShare.model ? modelId : g.extra?.model ?? modelId,
      strength: movieShare.strength ? strength : g.extra?.strength ?? strength,
      chain: movieShare.chain ? chain : g.extra?.chain ?? chain,
      motion: movieShare.motion ? motion : g.extra?.motion ?? motion,
    };
  }
  function patchGoalExtra(
    g: GoalRow,
    key: "model" | "strength" | "chain" | "motion",
    value: string | boolean,
  ): void {
    if (movieShare[key]) {
      switch (key) {
        case "model":
          setModelId(value as string);
          break;
        case "strength":
          setStrength(value as string);
          break;
        case "chain":
          setChain(value as boolean);
          break;
        case "motion":
          setMotion(value as string);
          break;
      }
      return;
    }
    patchGoal(g.id, { extra: { ...g.extra, [key]: value } });
  }
  const isMovieShared = (key: MovieShareKey): boolean => movieShare[key];
  function setMovieSharedKey(key: MovieShareKey, v: boolean): void {
    setMovieShare((prev) => ({ ...prev, [key]: v }));
    if (!v) return;
    // Sharing ON: drop each goal's own override for this setting so a later
    // un-share resolves from the CURRENT shared value (mirrors setSharedKey).
    setGoals((prev) =>
      prev.map((g) => {
        if (key === "size" || key === "steps" || key === "guidance" || key === "seed") {
          if (!g.knobs) return g;
          const knobs = { ...g.knobs };
          if (key === "size") {
            delete knobs.width;
            delete knobs.height;
          } else {
            delete knobs[key];
          }
          return { ...g, knobs };
        }
        if (key === "negative") return g; // negative lives on the goal row, not extra
        if (!g.extra) return g;
        const extra = { ...g.extra };
        delete extra[key as keyof GoalExtra];
        return { ...g, extra };
      }),
    );
  }
  function movieShareTick(key: MovieShareKey, label: string) {
    return (
      <input
        type="checkbox"
        className="vi-carry-tick"
        checked={isMovieShared(key)}
        onChange={(e) => setMovieSharedKey(key, e.target.checked)}
        aria-label={`Share ${label} across all goals`}
        title={`Share ${label} across all goals. Unticked = this goal keeps its own ${label}.`}
      />
    );
  }
  // The per-goal knob BODY (model + size/steps/guidance/seed + strength/motion/chain).
  // Folded INTO each goal card's settings expander, beside its frame window — so a
  // movie goal exposes the SAME full settings a Scene part does. fps/assemble/project
  // stay movie-level (one clip); frames are the goal's window.
  function renderGoalKnobs(g: GoalRow) {
    const ex = goalExtra(g);
    const gid = (s: string) => `vi-gen-goal-${g.id}-${s}`;
    return (
      <>
        {/* model */}
        <div className="vi-knob-strip">
          <div className="vi-knob-strip-line">
            <span className="vi-knob-mini-group vi-knob-mini-model">
              {movieShareTick("model", "the model")}
              <label className="vi-knob-mini-label" htmlFor={gid("model")}>
                model
              </label>
              <select
                id={gid("model")}
                className="vi-knob-select vi-knob-select-mini"
                value={ex.model}
                onChange={(e) => patchGoalExtra(g, "model", e.target.value)}
                disabled={modelsLoading}
              >
                {modelsLoading && <option value="">Loading models…</option>}
                {!modelsLoading && models.every((m) => m.disabled) && (
                  <option value="">No image models available</option>
                )}
                {models.map((m) => (
                  <option key={m.id} value={m.id} disabled={m.disabled} title={m.reason}>
                    {m.label}
                  </option>
                ))}
              </select>
            </span>
          </div>
        </div>
        {/* size / steps / guidance / seed — share ticks ride the strip */}
        <CondensedKnobStrip
          idPrefix={`vi-gen-goal-${g.id}`}
          values={goalKnobValues(g)}
          onPatch={(patch) => patchGoalKnobs(g, patch)}
          carry={{
            model: movieShare.model,
            size: movieShare.size,
            steps: movieShare.steps,
            guidance: movieShare.guidance,
            seed: movieShare.seed,
            negative: false,
          }}
          onCarry={(k, v) => setMovieSharedKey(k as MovieShareKey, v)}
        />
        {/* strength / motion / chain (goal-level scene knobs) */}
        <div className="vi-knob-strip">
          <div className="vi-knob-strip-line">
            <span className="vi-knob-mini-group">
              {movieShareTick("strength", "strength")}
              <label className="vi-knob-mini-label" htmlFor={gid("strength")}>
                strength
              </label>
              <input
                id={gid("strength")}
                type="text"
                inputMode="decimal"
                className="vi-knob-input vi-knob-mini"
                value={ex.strength}
                onChange={(e) => patchGoalExtra(g, "strength", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              {movieShareTick("motion", "motion")}
              <label className="vi-knob-mini-label" htmlFor={gid("motion")}>
                motion
              </label>
              <input
                id={gid("motion")}
                type="text"
                className="vi-knob-input vi-knob-mini"
                placeholder="none"
                value={ex.motion}
                onChange={(e) => patchGoalExtra(g, "motion", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              {movieShareTick("chain", "chain frames")}
              <label className="vi-knob-mini-label" htmlFor={gid("chain")}>
                chain
              </label>
              <input
                id={gid("chain")}
                type="checkbox"
                checked={ex.chain}
                onChange={(e) => patchGoalExtra(g, "chain", e.target.checked)}
              />
            </span>
            {/* negative share: ticked = ONE movie negative (edit on goal #1);
                unticked = each goal keeps its OWN negative in the [Negative] tab. */}
            <span className="vi-knob-mini-group">
              {movieShareTick("negative", "the negative")}
              <label className="vi-knob-mini-label" htmlFor={gid("neg")}>
                negative
              </label>
              <span id={gid("neg")} className="vi-knob-mini" style={{ opacity: 0.7 }}>
                {movieShare.negative ? "shared" : "per-goal"}
              </span>
            </span>
          </div>
        </div>
        {/* UNIFORM-CARD PLACEHOLDERS (operator 2026-08-13): the core knob set
            renders on EVERY prompt component. Slots this tier serves at the
            MOVIE level (fps, VRAM budget) — or expresses differently (a movie
            goal's LENGTH is its start/end times, frames = duration x fps) —
            appear GREYED with the reason, never hidden, so every card in every
            section reads identically. */}
        <div className="vi-knob-strip">
          <div className="vi-knob-strip-line">
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor={gid("na-fps")}>fps</label>
              <input id={gid("na-fps")} className="vi-knob-input" style={{ maxWidth: "4.5rem" }} disabled
                     value={fpsNum ?? ""}
                     title="Movie-level — one fps for the whole timeline; set it in the station knobs." />
            </span>
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor={gid("na-length")}>length</label>
              <input id={gid("na-length")} className="vi-knob-input" style={{ maxWidth: "6.5rem" }} disabled
                     value={`${g.start || "0"}–${g.end || "?"}s`}
                     title="A movie goal's length IS its start/end times (frames = duration x fps) — edit those on the card; a frames knob does not apply here." />
            </span>
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor={gid("na-vram")}>VRAM</label>
              <input id={gid("na-vram")} className="vi-knob-input" style={{ maxWidth: "5.5rem" }} disabled
                     value="autofit"
                     title="Movie-level — renders size to the serving worker's free VRAM (autofit); the movie tier has no per-goal budget." />
            </span>
          </div>
        </div>
      </>
    );
  }

  // k92 TEST OPTIONS — "test all models" / "per preset" as GOALS (the single reusable
  // architecture; a goal IS a prompt component). Append one goal per model / per preset,
  // each carrying its OWN model (un-shared), so the movie's segments become a model /
  // preset comparison reel. Retile keeps the timeline contiguous.
  function goalBasePrompt(): string {
    for (let i = goals.length - 1; i >= 0; i--) {
      if (goals[i].prompt.trim() !== "") return goals[i].prompt;
    }
    return goals[0]?.prompt ?? "";
  }
  function addGoalPerModel(): void {
    const text = goalBasePrompt();
    if (text.trim() === "") return;
    // Skip models that REQUIRE a start image — a freshly-appended test goal has no ref.
    const targets = models.filter(
      (m) => !m.disabled && requiredInputsFor(m.tasks).image !== "required",
    );
    if (targets.length === 0) return;
    setMovieShare((prev) => ({ ...prev, model: false }));
    setGoals((prev) =>
      retile([
        ...prev,
        ...targets.map((m) => ({
          id: goalKey(),
          start: "0",
          end: String(DEFAULT_GOAL_LEN),
          prompt: text,
          extra: { model: m.id } as Partial<GoalExtra>,
        })),
      ]),
    );
  }
  function addGoalPerPreset(): void {
    const text = goalBasePrompt();
    if (text.trim() === "" || presets.length === 0) return;
    setMovieShare((prev) => ({
      ...prev,
      model: false,
      size: false,
      steps: false,
      guidance: false,
    }));
    setGoals((prev) =>
      retile([
        ...prev,
        ...presets.map((pre) => {
          const d = pre.defaults;
          const knobs: Partial<KnobValues> = {};
          if (d.width != null) knobs.width = String(d.width);
          if (d.height != null) knobs.height = String(d.height);
          if (d.steps != null) knobs.steps = String(d.steps);
          if (d.guidance != null) knobs.guidance = String(d.guidance);
          const row: GoalRow = {
            id: goalKey(),
            start: "0",
            end: String(DEFAULT_GOAL_LEN),
            prompt: text,
            extra: { model: pre.model_key || modelId },
            knobs,
          };
          // A preset's negative REPLACES the standard set (mirrors addPartPerPreset).
          if (d.negative != null) {
            row.negative = d.negative;
            row.stdNegative = false;
          }
          return row;
        }),
      ]),
    );
  }

  // GROUP settings for the selected goals (mirror renderGroupSettings). Editing a
  // value un-shares that knob and writes it as each SELECTED goal's own override.
  function applyGoalGroupKnobs(patch: Partial<KnobValues>) {
    const keys = Object.keys(patch) as (keyof KnobValues)[];
    setMovieShare((prev) => {
      const next = { ...prev };
      for (const k of keys)
        next[k === "width" || k === "height" ? "size" : (k as MovieShareKey)] = false;
      return next;
    });
    setGoals((prev) =>
      prev.map((g) =>
        movieSelectedIds.has(g.id) ? { ...g, knobs: { ...g.knobs, ...patch } } : g,
      ),
    );
  }
  function applyGoalGroupExtra(
    key: "model" | "strength" | "chain" | "motion",
    value: string | boolean,
  ) {
    setMovieShare((prev) => ({ ...prev, [key]: false }));
    setGoals((prev) =>
      prev.map((g) =>
        movieSelectedIds.has(g.id) ? { ...g, extra: { ...g.extra, [key]: value } } : g,
      ),
    );
  }
  function renderMovieGroupSettings() {
    if (movieSelectedCount === 0) return null;
    return (
      <PromptCardSettings
        label={`settings → apply to ${movieSelectedCount} selected goal${movieSelectedCount === 1 ? "" : "s"}`}
      >
        <div className="vi-knob-strip">
          <div className="vi-knob-strip-line">
            <span className="vi-knob-mini-group vi-knob-mini-model">
              <label className="vi-knob-mini-label" htmlFor="vi-gen-movie-group-model">
                model
              </label>
              <select
                id="vi-gen-movie-group-model"
                className="vi-knob-select vi-knob-select-mini"
                value={modelId}
                onChange={(e) => applyGoalGroupExtra("model", e.target.value)}
                disabled={modelsLoading}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id} disabled={m.disabled} title={m.reason}>
                    {m.label}
                  </option>
                ))}
              </select>
            </span>
          </div>
        </div>
        <CondensedKnobStrip
          idPrefix="vi-gen-movie-group"
          values={{ width, height, steps, guidance, seed }}
          onPatch={applyGoalGroupKnobs}
        />
        <div className="vi-knob-strip">
          <div className="vi-knob-strip-line">
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor="vi-gen-movie-group-strength">
                strength
              </label>
              <input
                id="vi-gen-movie-group-strength"
                type="text"
                inputMode="decimal"
                className="vi-knob-input vi-knob-mini"
                value={strength}
                onChange={(e) => applyGoalGroupExtra("strength", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor="vi-gen-movie-group-motion">
                motion
              </label>
              <input
                id="vi-gen-movie-group-motion"
                type="text"
                className="vi-knob-input vi-knob-mini"
                placeholder="none"
                value={motion}
                onChange={(e) => applyGoalGroupExtra("motion", e.target.value)}
              />
            </span>
            <span className="vi-knob-mini-group">
              <label className="vi-knob-mini-label" htmlFor="vi-gen-movie-group-chain">
                chain
              </label>
              <input
                id="vi-gen-movie-group-chain"
                type="checkbox"
                checked={chain}
                onChange={(e) => applyGoalGroupExtra("chain", e.target.checked)}
              />
            </span>
          </div>
        </div>
      </PromptCardSettings>
    );
  }

  // ---- Movie prompt-list toolbar actions (k88 — the shared PromptListToolbar) ----
  /** Semantics mirror Cinema's Segments stepper (2026-08-27): shrinking PARKS the
      tail rows in a session stash instead of deleting them; growing revives parked
      rows (keys intact) before appending default-length windows after the last
      row. No upper cap. */
  function setGoalCount(n: number) {
    if (!Number.isInteger(n) || n < 1 || n === goals.length) return;
    if (n < goals.length) {
      const cut = goals.slice(n);
      setGoalStash((st) => [...cut, ...st]);
      setGoals(goals.slice(0, n));
      return;
    }
    const revived = goalStash.slice(0, n - goals.length);
    if (revived.length) setGoalStash((st) => st.slice(revived.length));
    const next = [...goals, ...revived];
    while (next.length < n) {
      const last = next[next.length - 1];
      const start = last ? toNumberOrNull(last.end) ?? 0 : 0;
      next.push({
        id: goalKey(),
        start: String(start),
        end: String(start + DEFAULT_GOAL_LEN),
        prompt: "",
      });
    }
    setGoals(next);
  }
  function toggleMovieSelected(id: string) {
    setMovieSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }
  // Keyed off the LIVE rows, not the set's size (stale ids after a remove).
  const movieSelectedCount = goals.reduce(
    (n, g) => (movieSelectedIds.has(g.id) ? n + 1 : n),
    0,
  );
  const movieAllSelected = goals.length > 0 && movieSelectedCount === goals.length;
  function toggleMovieSelectAll() {
    setMovieSelectedIds(movieAllSelected ? new Set() : new Set(goals.map((g) => g.id)));
  }

  // GROUP ACTION (k88): ONE coherent spread across the ticked goals — the same
  // one-call rule Cinema's onSpreadSelected enforces: selected rows are targets,
  // unselected rows ride as LOCKED context, and the reply may only land on rows
  // whose segment_id came back. Movie rows carry no seed/joint/branch knobs; each
  // goal continues the previous frame interval, so the joint reads as "still".
  async function onMovieSpreadSelected() {
    const wiredNegative = composeNegative(negative, goals[0]?.stdNegative ?? false);
    const inputs: SpreadGoalInput[] = goals.map((g, i) => ({
      key: g.id,
      prompt: g.prompt,
      negative: i === 0 ? wiredNegative : "",
      seed: "",
      branchFrame: "",
      joint: "still",
      index: i,
    }));
    const body = buildSpreadBody({
      goals: inputs,
      selectedKeys: movieSelectedIds,
      movieQuery: "",
      globalNegative: wiredNegative,
      steeringSeed: movieSteeringSeed,
    });
    if (!body) return;
    setMovieSpreadNotice(null);
    const res = await assist.runSpread(body);
    if (!res) return;
    const bySeg = new Map(res.segments.map((s) => [s.segment_id, s]));
    const applied = goals
      .filter((g) => movieSelectedIds.has(g.id) && bySeg.has(g.id))
      .map((g) => g.id);
    setGoals((prev) =>
      prev.map((g) => {
        if (!movieSelectedIds.has(g.id)) return g; // an unselected row is never touched
        const hit = bySeg.get(g.id);
        // The wire carries ONE movie negative, so a returned per-row negative is
        // dropped here rather than written somewhere it could never be sent.
        return hit ? { ...g, prompt: hit.prompt } : g;
      }),
    );
    if (res.steering_seed != null) setMovieSteeringSeed(res.steering_seed);
    setMovieSpreadNotice(spreadNoticeLines(res, applied));
  }

  // GROUP ACTION (k88): write the movie's ONE wired negative from the ticked
  // goals' prompts. GenerateMovieRequest carries a single `negative`, so this is
  // ONE runNegative call (subject = the ticked prompts) — not Cinema's sequential
  // per-row loop, which only makes sense with per-row negative fields on the wire.
  async function onMovieNegativesSelected() {
    const rows = goals.filter((g) => movieSelectedIds.has(g.id));
    if (rows.length === 0) return;
    setMovieSpreadNotice(null);
    const subject = rows
      .map((g) => g.prompt.trim())
      .filter((t) => t !== "")
      .join("; ");
    await assist.runNegative({ subject, draft: negative }, setNegative);
  }

  // Toolbar [standard negatives]: the movie has ONE wired negative (goal #1's
  // Negative tab), so this ticks that row's [standard negative] on.
  function onMovieStandardNegatives() {
    setGoals((prev) => prev.map((g, i) => (i === 0 ? { ...g, stdNegative: true } : g)));
  }

  // ---- Scene prompt-list toolbar actions (k89 — the parts composer's mirror of
  // the Movie trio above) --------------------------------------------------------
  /** Count-knob rule (stated per the task): growing APPENDS blank text parts;
      shrinking removes TEXT parts from the tail only — a media part (image / video /
      identity reference) is never auto-removed, so the reachable floor is the media
      count. Simplest correct rule: media survives, text truncates. */
  const MAX_SCENE_PARTS = 24;
  function setScenePartCount(n: number) {
    if (!Number.isInteger(n) || n < 1 || n > MAX_SCENE_PARTS) return;
    setParts((prev) => {
      if (n === prev.length) return prev;
      if (n > prev.length) {
        const next = prev.slice();
        while (next.length < n)
          next.push({ key: partKey(), text: "", negative: STANDARD_NEGATIVE });
        return next;
      }
      const next = prev.slice();
      for (let i = next.length - 1; i >= 0 && next.length > n; i--) {
        // Never truncate a media part, nor a part whose fan-out job is still
        // active (running OR capped-but-live) — the stepper is a second removal
        // path, so it must mirror canRemove exactly or a still-live job orphans.
        if (
          !next[i].media &&
          !sceneRowJobs.byRow[next[i].key]?.running &&
          !sceneRowJobs.byRow[next[i].key]?.capped
        )
          next.splice(i, 1);
      }
      return next;
    });
  }
  // Selection is over TEXT parts only — the group actions rewrite prompts, and a
  // media part's caption rides as locked context (its checkbox renders disabled).
  // Live-row math, same as Movie: a removed part can leave a stale key behind.
  const sceneTextParts = parts.filter((p) => !p.media);
  const sceneSelectedCount = sceneTextParts.reduce(
    (n, p) => (sceneSelectedIds.has(p.key) ? n + 1 : n),
    0,
  );
  const sceneAllSelected =
    sceneTextParts.length > 0 && sceneSelectedCount === sceneTextParts.length;
  // k93 §A.1/2 — the RANGE view of the selection set (1-based over text parts):
  // min/max of the ticked text parts, or null when nothing is ticked. The SET
  // stays the source of truth; the range inputs, select-all and the per-card
  // ticks all write into it.
  const sceneSelectedIdx = sceneTextParts
    .map((p, i) => (sceneSelectedIds.has(p.key) ? i + 1 : 0))
    .filter((n) => n > 0);
  const sceneRangeStart = sceneSelectedIdx.length ? Math.min(...sceneSelectedIdx) : null;
  const sceneRangeEnd = sceneSelectedIdx.length ? Math.max(...sceneSelectedIdx) : null;
  /** Select text parts #start … #end (1-based, inclusive, already clamped by the toolbar). */
  function selectSceneRange(start: number, end: number) {
    const s = Math.max(1, Math.min(start, sceneTextParts.length));
    const e = Math.max(s, Math.min(end, sceneTextParts.length));
    sceneRangeMemo.current = { start: s, end: e };
    setSceneSelectedIds(new Set(sceneTextParts.slice(s - 1, e).map((p) => p.key)));
  }
  function toggleSceneSelected(key: string) {
    setSceneSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      // A per-card tick un-checks select-all implicitly (the set is no longer
      // every text part) and the range inputs show the new min/max; remember
      // that span as the explicit range select-all returns to.
      const idx = sceneTextParts
        .map((p, i) => (next.has(p.key) ? i + 1 : 0))
        .filter((n) => n > 0);
      sceneRangeMemo.current = idx.length
        ? { start: Math.min(...idx), end: Math.max(...idx) }
        : null;
      return next;
    });
  }
  // select all: checked ⇒ #1 … #last; unchecked ⇒ back to the remembered
  // explicit [start, end] (or nothing when no range was ever set).
  function toggleSceneSelectAll() {
    if (sceneAllSelected) {
      const memo = sceneRangeMemo.current;
      if (memo && !(memo.start === 1 && memo.end === sceneTextParts.length)) {
        selectSceneRange(memo.start, memo.end);
      } else {
        setSceneSelectedIds(new Set());
      }
      return;
    }
    if (sceneRangeStart != null && sceneRangeEnd != null) {
      sceneRangeMemo.current = { start: sceneRangeStart, end: sceneRangeEnd };
    }
    setSceneSelectedIds(new Set(sceneTextParts.map((p) => p.key)));
  }
  // enhance (M selected): the selected text parts with a NON-BLANK prompt.
  const sceneSelectedParts = sceneTextParts.filter((p) => sceneSelectedIds.has(p.key));
  const sceneEnhanceCount = sceneSelectedParts.filter((p) => p.text.trim() !== "").length;
  // The toolbar std. tick reads the FIELDS: checked iff every selected part's
  // Negative already holds the standard set.
  const sceneStdAllOn =
    sceneSelectedParts.length > 0 &&
    sceneSelectedParts.every((p) => hasStandardNegative(p.negative ?? ""));

  // The scene's group-assist negative (k93): part #1's VISIBLE Negative field,
  // explicitly — the first text part's field, sent as `global_negative` (its tab
  // tooltip says so). No std composition, no shared fallback.
  function sceneGroupNegative(): string {
    return (sceneTextParts[0]?.negative ?? "").trim();
  }

  // GROUP ACTION (k89/k93): ONE coherent spread across the ticked text parts — the
  // same one-call rule Cinema's onSpreadSelected / Movie's onMovieSpreadSelected
  // enforce: selected parts are targets, everything else (unticked text, media
  // captions) rides as LOCKED context, and the reply may only land on parts whose
  // key came back. With `negatives: on` the returned per-part negative is WRITTEN
  // into that part's visible Negative field (the old separate "Generate negative"
  // button folded in); off ⇒ prompts only. context.media carries the `[+ video]`
  // pretext (§C) when one is attached.
  async function onSceneSpreadSelected() {
    const wiredNegative = sceneGroupNegative();
    const inputs: SpreadGoalInput[] = parts.map((p, i) => ({
      key: p.key,
      prompt: p.text,
      negative: p.negative ?? "",
      seed: "",
      branchFrame: "",
      joint: "still",
      index: i,
    }));
    const media = scenePretextWire();
    const body = buildSpreadBody({
      goals: inputs,
      selectedKeys: sceneSelectedIds,
      movieQuery: "",
      globalNegative: wiredNegative,
      steeringSeed: sceneSteeringSeed,
      context: { kind: "scene", ...(media ? { media } : {}) },
    });
    if (!body) return;
    setSceneSpreadNotice(null);
    const writeNegatives = sceneGroupNegatives;
    const res = await assist.runSpread(body);
    if (!res) return;
    const bySeg = new Map(res.segments.map((s) => [s.segment_id, s]));
    const applied = parts
      .filter((p) => sceneSelectedIds.has(p.key) && !p.media && bySeg.has(p.key))
      .map((p) => p.key);
    setParts((prev) =>
      prev.map((p) => {
        // Skip unselected parts, and any part that has since gained media (its key
        // may linger in the select set — never overwrite a media caption).
        if (!sceneSelectedIds.has(p.key) || p.media) return p;
        const hit = bySeg.get(p.key);
        if (!hit) return p;
        if (!writeNegatives || hit.negative.trim() === "") return { ...p, text: hit.prompt };
        // Keep the standard set if the field held it (the std. tick reads the field).
        const std = hasStandardNegative(p.negative ?? "");
        return { ...p, text: hit.prompt, negative: composeNegative(hit.negative, std) };
      }),
    );
    if (res.steering_seed != null) setSceneSteeringSeed(res.steering_seed);
    setSceneSpreadNotice(spreadNoticeLines(res, applied));
  }

  // GROUP ACTION (k93 §A.5): enhance (M selected) — the "detail" assist over every
  // selected text part with a non-blank prompt. The spread endpoint carries no
  // mode, so this runs SEQUENTIALLY per part through runPartAssist (one in-flight
  // assist at a time — the hook's guard), each call carrying the toolbar's
  // negatives toggle and the `[+ video]` pretext.
  async function onSceneEnhanceSelected() {
    const rows = sceneSelectedParts.filter((p) => p.text.trim() !== "");
    if (rows.length === 0) return;
    setSceneSpreadNotice(null);
    const media = scenePretextWire();
    for (const p of rows) {
      await runPartAssist(p, "detail", { negatives: sceneGroupNegatives, media });
    }
  }

  // Toolbar `[ ] std.` (k93 §A.4): tick ⇒ INSERT the standard set into every
  // selected part's Negative field (deduped, comma-joined); untick ⇒ remove
  // exactly those terms. Visible in the field — nothing composed at submit.
  function onSceneStdChange(on: boolean) {
    setParts((prev) =>
      prev.map((p) =>
        sceneSelectedIds.has(p.key) && !p.media
          ? { ...p, negative: applyStandardNegative(p.negative ?? "", on) }
          : p,
      ),
    );
  }
  /** Per-card std. tick (k93 §B.3): same explicit insert/remove on this part only. */
  function onPartStdChange(p: PartEntry, on: boolean) {
    patchPart(p.key, { negative: applyStandardNegative(p.negative ?? "", on) });
  }
  // Route a picked image: onto the attach-target GOAL (Movie mode) when set, else
  // through the parts placeMedia path (Image/Scene). Keeps the picker/upload code
  // single-sourced across all three modes.
  function placePickedImage(media: PartMedia) {
    if (goalAttachTarget != null) {
      const target = goalAttachTarget;
      setGoals((prev) => prev.map((g) => (g.id === target ? { ...g, ref: media } : g)));
      setGoalAttachTarget(null);
      return;
    }
    placeMedia(media);
  }

  function addImageMedia(media: PartMedia) {
    placePickedImage(media);
    setShowImagePicker(false);
  }
  function addVideoMedia(media: PartMedia) {
    placeMedia(media);
    setShowVideoPicker(false);
  }

  // Upload a raw video, ingest it into a MediaRef, guard kind==="video", then
  // append it as a video part (and add it to the library so it is reusable).
  async function onPickVideo(file: File, place: (media: PartMedia) => void = placeMedia) {
    setVideoError(null);
    setVideoPhase("uploading");

    const fd = new FormData();
    fd.append("file", file);
    fd.append("sid", getSessionId());
    const up = await request<unknown>(hugpyConfig.uploadUrl, {
      method: "POST",
      body: fd,
      meta: { specKey: "generate", operation: "upload" },
    });
    if (!up.ok) {
      setVideoError(describeAppError(errorOf(up)));
      setVideoPhase("idle");
      return;
    }
    const upParsed = uploadResultSchema.safeParse(okValue(up));
    if (!upParsed.success) {
      setVideoError("Malformed upload response.");
      setVideoPhase("idle");
      return;
    }

    setVideoPhase("ingesting");
    const ing = await request<unknown>(hugpyConfig.videoIngestUrl, {
      method: "POST",
      body: JSON.stringify({ path: upParsed.data.path }),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "generate", operation: "ingest" },
    });
    if (!ing.ok) {
      setVideoError(describeAppError(errorOf(ing)));
      setVideoPhase("idle");
      return;
    }
    const media = mediaRefSchema.safeParse(okValue(ing));
    if (!media.success) {
      setVideoError("Malformed ingest response.");
      setVideoPhase("idle");
      return;
    }
    if (media.data.kind !== "video") {
      setVideoError(
        `Ingested asset is "${media.data.kind}", not a video — pick an image from the library instead.`,
      );
      setVideoPhase("idle");
      return;
    }
    addToLibrary(media.data, "upload", media.data.mime);
    place({ ref: media.data, origin: "upload", label: media.data.mime });
    setVideoPhase("idle");
    setShowVideoPicker(false);
  }

  // Browse-from-disk for IMAGE parts — the exact mirror of onPickVideo (attach
  // was library-only for images, which forced a detour through another
  // station just to use a file that's sitting on disk). Upload → ingest →
  // guard kind==="image" → attach as the part + into the library for reuse.
  async function onPickImage(file: File, place: (media: PartMedia) => void = placePickedImage) {
    setImageError(null);
    setImagePhase("uploading");

    const fd = new FormData();
    fd.append("file", file);
    fd.append("sid", getSessionId());
    const up = await request<unknown>(hugpyConfig.uploadUrl, {
      method: "POST",
      body: fd,
      meta: { specKey: "generate", operation: "upload" },
    });
    if (!up.ok) {
      setImageError(describeAppError(errorOf(up)));
      setImagePhase("idle");
      return;
    }
    const upParsed = uploadResultSchema.safeParse(okValue(up));
    if (!upParsed.success) {
      setImageError("Malformed upload response.");
      setImagePhase("idle");
      return;
    }

    setImagePhase("ingesting");
    const ing = await request<unknown>(hugpyConfig.videoIngestUrl, {
      method: "POST",
      body: JSON.stringify({ path: upParsed.data.path }),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "generate", operation: "ingest" },
    });
    if (!ing.ok) {
      setImageError(describeAppError(errorOf(ing)));
      setImagePhase("idle");
      return;
    }
    const media = mediaRefSchema.safeParse(okValue(ing));
    if (!media.success) {
      setImageError("Malformed ingest response.");
      setImagePhase("idle");
      return;
    }
    if (media.data.kind !== "image") {
      setImageError(
        `Ingested asset is "${media.data.kind}", not an image — use the video receptacle for clips.`,
      );
      setImagePhase("idle");
      return;
    }
    addToLibrary(media.data, "upload", media.data.mime);
    place({ ref: media.data, origin: "upload", label: media.data.mime });
    setImagePhase("idle");
    setShowImagePicker(false);
  }

  // ── k93 §B.5 — the picker BODIES (browse-from-disk + library grid), factored so
  // the same markup renders inline inside a card, under the toolbar (the `[+ video]`
  // pretext) and in the station-level append pickers. `onPick` receives the chosen
  // media as a PartMedia; the caller decides where it lands.
  function renderImagePickerBody(onPick: (media: PartMedia) => void) {
    return (
      <>
        <div className="vi-gen-video-upload">
          <AddFileBar
            accept="image/*"
            onFile={(f) => onPickImage(f, onPick)}
            busy={imageBusy}
            buttonLabel="Choose image from disk"
          >
            {imageBusy && <span className="vi-crop-status">Uploading…</span>}
          </AddFileBar>
          {imageError && <span className="vi-error">{imageError}</span>}
        </div>
        {libraryImages.length === 0 ? (
          <p className="vi-crop-hint">
            Nothing in the session library yet — browse above, pick frames in the
            Frames station, or generate an image; results land here and become
            pickable.
          </p>
        ) : (
          <LibraryGroups
            seed={mode}
            items={libraryImages}
            defaultOpen={libraryDefaultOpen}
            renderItem={(it) => (
              <button
                type="button"
                key={it.ref.uri}
                className="vi-gen-pick-cell"
                onClick={() => onPick({ ref: it.ref, origin: it.origin, label: it.label })}
                title={it.ref.uri}
              >
                <img
                  className="vi-gen-pick-thumb"
                  src={mediaBytesUrl(it.ref.uri)}
                  alt={it.label ?? "library image"}
                  loading="lazy"
                />
                <span className="vi-gen-pick-meta">
                  {it.origin}
                  {it.label ? ` · ${it.label}` : ""}
                </span>
              </button>
            )}
          />
        )}
      </>
    );
  }
  function renderVideoPickerBody(onPick: (media: PartMedia) => void) {
    return (
      <>
        <div className="vi-gen-video-upload">
          <AddFileBar
            accept="video/*"
            onFile={(f) => onPickVideo(f, onPick)}
            busy={videoBusy}
            buttonLabel="Choose video from disk"
          >
            {videoBusy && <span className="vi-crop-status">Uploading…</span>}
          </AddFileBar>
          {videoError && <span className="vi-error">{videoError}</span>}
        </div>
        {libraryVideos.length === 0 ? (
          <p className="vi-crop-hint">
            No videos in the session library yet — browse above or ingest one in
            another station.
          </p>
        ) : (
          <LibraryGroups
            seed={mode}
            items={libraryVideos}
            defaultOpen={libraryDefaultOpen}
            renderItem={(it) => (
              <button
                type="button"
                key={it.ref.uri}
                className="vi-gen-pick-cell"
                onClick={() => onPick({ ref: it.ref, origin: it.origin, label: it.label })}
                title={it.ref.uri}
              >
                <span className="vi-gen-pick-videoicon" aria-hidden>
                  ▶
                </span>
                <span className="vi-gen-pick-meta">
                  <code>{it.ref.mime}</code> · {it.origin}
                </span>
              </button>
            )}
          />
        )}
      </>
    );
  }
  /** The whole library (images + videos) — the inline `+ library` APPEND panel. */
  function renderLibraryPickerBody(onPick: (media: PartMedia) => void) {
    return library.length === 0 ? (
      <p className="vi-crop-hint">Nothing in the session library yet.</p>
    ) : (
      <LibraryGroups
        seed={mode}
        items={library}
        defaultOpen={libraryDefaultOpen}
        renderItem={(it) => (
          <button
            type="button"
            key={it.ref.uri}
            className="vi-gen-pick-cell"
            onClick={() => onPick({ ref: it.ref, origin: it.origin, label: it.label })}
            title={it.ref.uri}
          >
            {it.ref.kind === "image" ? (
              <img
                className="vi-gen-pick-thumb"
                src={mediaBytesUrl(it.ref.uri)}
                alt={it.label ?? "library image"}
                loading="lazy"
              />
            ) : (
              <span className="vi-gen-pick-videoicon" aria-hidden>
                ▶
              </span>
            )}
            <span className="vi-gen-pick-meta">
              {it.ref.kind === "image" ? it.origin : <><code>{it.ref.mime}</code> · {it.origin}</>}
              {it.label ? ` · ${it.label}` : ""}
            </span>
          </button>
        )}
      />
    );
  }

  // ---- run validation (no silent defaults; every gate is visible) ----
  const widthNum = toNumberOrNull(width);
  const heightNum = toNumberOrNull(height);
  const stepsNum = toNumberOrNull(steps);
  const guidanceNum = toNumberOrNull(guidance);
  const seedNum = toNumberOrNull(seed);

  const widthValid = widthNum != null && Number.isInteger(widthNum) && widthNum > 0;
  const heightValid =
    heightNum != null && Number.isInteger(heightNum) && heightNum > 0;
  const stepsValid = stepsNum != null && Number.isInteger(stepsNum) && stepsNum > 0;
  const guidanceValid = guidanceNum != null && guidanceNum >= 0;
  const seedValid =
    seed.trim() === "" ||
    (seedNum != null && Number.isInteger(seedNum) && seedNum >= 0);

  const widthMult8 = widthValid && widthNum % 8 === 0;
  const heightMult8 = heightValid && heightNum % 8 === 0;

  // Any non-empty text counts — standalone text parts AND captions on media.
  const hasText = parts.some((p) => p.text.trim() !== "");
  const modelReady = modelId !== "";

  // The prompt-assist `draft`: the non-empty text fields joined in order (mirrors
  // buildPromptParts' text ordering; captions on media parts count too, matching
  // hasText). Enhance sends this (required); Generate sends it as a loose theme.
  const assembledPrompt = parts
    .map((p) => p.text)
    .filter((t) => t.trim() !== "")
    .join("\n");

  // Start-image presence — any image part (library image, uploaded, or attached).
  // Edit-only models (req.image === "required") gate Run on this; otherwise it is
  // informational (optional models still run without one).
  const hasStartImage = parts.some((p) => partKind(p) === "image");
  const startImageOk = !startImageRequired || hasStartImage;

  // Scene-only knob validation (n_frames capped at 24 client-side; the backend
  // also rejects over-cap with HTTP 400 "frame_cap_exceeded").
  const nFramesNum = toNumberOrNull(nFrames);
  const fpsNum = toNumberOrNull(fps);
  const strengthNum = toNumberOrNull(strength);
  const nFramesValid =
    nFramesNum != null &&
    Number.isInteger(nFramesNum) &&
    nFramesNum >= 1 &&
    nFramesNum <= 24;
  const fpsValid = fpsNum != null && Number.isInteger(fpsNum) && fpsNum >= 1;
  // img2img strength — a plain 0..1 number (mirrors guidanceValid's shape).
  const strengthValid =
    strengthNum != null && strengthNum >= 0 && strengthNum <= 1;

  // Shared knob gates (both modes) — model + dims + steps + guidance + seed + text,
  // plus a start image when the selected model requires one (edit-only). Both Run
  // paths derive from this, so the required-image gate applies to Image and Scene.
  // The shared GENERATION-KNOB validity (model + dims + steps + guidance + seed),
  // WITHOUT the parts-prompt/start-image gates — Movie mode has no parts composer
  // (its prompts live in the goal timeline), so it gates on this + the timeline.
  const sharedKnobsValid =
    modelReady &&
    widthValid &&
    heightValid &&
    stepsValid &&
    guidanceValid &&
    seedValid;

  const sharedValid = hasText && startImageOk && sharedKnobsValid;

  // ---- Movie goal-timeline validation (contiguous, tile [0,total), prompts) ----
  // Per-row flags + an overall gap/overlap check drive the inline .vi-knob-flag red
  // lines; `total` = the last row's end (the [0,total) tiling); `valid` gates Run.
  const timeline = useMemo(() => {
    const rows = goals.map((g) => {
      const s = toNumberOrNull(g.start);
      const e = toNumberOrNull(g.end);
      const startOk = s != null && Number.isInteger(s) && s >= 0;
      const endOk = e != null && Number.isInteger(e) && s != null && e > s;
      const promptOk = g.prompt.trim() !== "";
      return { s, e, startOk, endOk, promptOk };
    });
    const flags: (string | null)[] = rows.map(() => null);
    let contiguous = rows.length > 0;
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const expected = i === 0 ? 0 : rows[i - 1].e;
      if (r.s != null && expected != null && r.s !== expected) {
        contiguous = false;
        flags[i] =
          i === 0
            ? `First goal must start at frame 0 (got ${r.s}).`
            : `Starts at ${r.s}, but the previous goal ends at ${expected} — gap/overlap.`;
      }
    }
    const lastEnd = rows.length ? rows[rows.length - 1].e : null;
    const total = lastEnd ?? 0;
    const allRowsOk = rows.every((r) => r.startOk && r.endOk && r.promptOk);
    const valid =
      rows.length >= 1 && allRowsOk && contiguous && rows[0].s === 0 && total > 0;
    return { rows, flags, total, valid };
  }, [goals]);
  const timelineValid = timeline.valid;
  const movieTotalFrames = timeline.total;

  // ---- Director-knob validation (only enforced when vision is enabled) ----
  const scoreThresholdNum = toNumberOrNull(scoreThreshold);
  const maxAttemptsNum = toNumberOrNull(maxAttempts);
  const timeBudgetNum = toNumberOrNull(timeBudget);
  const scoreThresholdValid =
    scoreThresholdNum != null && scoreThresholdNum >= 0 && scoreThresholdNum <= 100;
  const maxAttemptsValid =
    maxAttemptsNum != null && Number.isInteger(maxAttemptsNum) && maxAttemptsNum >= 1;
  const timeBudgetValid =
    timeBudget.trim() === "" ||
    (timeBudgetNum != null && timeBudgetNum > 0);
  const directorValid =
    !visionEnabled ||
    (scoreThresholdValid &&
      maxAttemptsValid &&
      timeBudgetValid &&
      judgeModelId.trim() !== "");

  const canRunImage = sharedValid && !imageJob.running;
  // Scene fans out one job per part; Run is enabled when at least one component
  // is fully runnable (its OWN resolved+validated settings, not the shared station
  // values — which may be unused when knobs are unshared) and a start image is
  // present if the model requires one. This also prevents the silent no-op where
  // every part is invalid but the stale station values pass.
  // buildSceneRow already enforces each part's OWN model start-image requirement,
  // so the station-wide startImageOk is not folded in here (it would mis-gate when
  // models are unshared).
  const sceneRunnableCount = mode === "scene" ? computeSceneRows().length : 0;
  const canRunScene =
    sceneRunnableCount > 0 && !sceneJob.running && !sceneAnyRunning;
  const canRunMovie =
    sharedKnobsValid &&
    timelineValid &&
    fpsValid &&
    strengthValid &&
    directorValid &&
    !movieJob.running;
  const canRun =
    mode === "movie" ? canRunMovie : mode === "scene" ? canRunScene : canRunImage;

  // The ordered prompt is identical for both jobs — flatten each text+media pair
  // back to the frozen GenPromptPart shape: caption text (when non-empty) is
  // emitted as a text part immediately BEFORE its media, order preserved.
  function buildPromptParts(): GenPromptPart[] {
    const out: GenPromptPart[] = [];
    for (const p of parts) {
      if (!p.media || p.text.trim() !== "") {
        out.push({ kind: "text", text: p.text });
      }
      if (p.frame) out.push({ kind: "image", media: p.frame.ref });
      if (p.media) {
        out.push(
          p.media.ref.kind === "video"
            ? { kind: "video", media: p.media.ref }
            : { kind: "image", media: p.media.ref },
        );
      }
    }
    return out;
  }

  // The composer's MEDIA parts only (images / videos / bound identity views), with
  // every text segment dropped. Tester rows supply their own text and reuse these,
  // so the reference material is identical across a sweep.
  function buildMediaOnlyParts(): GenPromptPart[] {
    const out: GenPromptPart[] = [];
    for (const p of parts) {
      if (p.frame) out.push({ kind: "image", media: p.frame.ref });
      if (p.media) {
        out.push(
          p.media.ref.kind === "video"
            ? { kind: "video", media: p.media.ref }
            : { kind: "image", media: p.media.ref },
        );
      }
    }
    return out;
  }

  // A new row is SEEDED from the last row (so a sweep is built by tweaking one
  // field at a time rather than retyping seven), or from the base composer when
  // there is no row yet.
  //
  // WHAT THE CARRY TICKS DO HERE (operator ask 2026-08-04, k63): the seed source
  // supplies a setting ONLY when that setting's tick is on. An unticked setting
  // falls back to the mode default (DEFAULT_KNOBS / DEFAULT_NEGATIVE / the
  // registry's default model) instead of being copied. Prompt TEXT is not a
  // setting and has no tick — it always carries, because a row with no prompt is
  // not runnable and would just be a blank card.
  //
  // The new row copies the source's tick pattern too, so a sweep keeps sweeping:
  // untick `seed` once and every row spun off the chain keeps re-randomizing.
  function seedTestRow(index: number, from: ImageTestRow | undefined): ImageTestRow {
    const ticks = from ? from.carry : carry;
    return {
      id: `row-${Date.now()}-${index}`,
      text: from ? from.text : assembledPrompt,
      model_id: ticks.model
        ? from
          ? from.model_id
          : modelId
        : defaultId || modelId,
      width: ticks.size ? (from ? from.width : width) : DEFAULT_KNOBS.width,
      height: ticks.size ? (from ? from.height : height) : DEFAULT_KNOBS.height,
      steps: ticks.steps ? (from ? from.steps : steps) : DEFAULT_KNOBS.steps,
      guidance: ticks.guidance
        ? from
          ? from.guidance
          : guidance
        : DEFAULT_KNOBS.guidance,
      seed: ticks.seed ? (from ? from.seed : seed) : DEFAULT_KNOBS.seed,
      // k93: a row seeded from the BASE composer carries part #1's VISIBLE
      // negative — the same value the scene submit sends, so a sweep row
      // reproduces the base generation.
      negative: ticks.negative
        ? from
          ? from.negative
          : sceneGroupNegative()
        : DEFAULT_NEGATIVE,
      carry: { ...ticks },
    };
  }

  function addTestRow(): void {
    setTestRows((prev) => [...prev, seedTestRow(prev.length, prev[prev.length - 1])]);
  }

  // ---- TESTER SWEEP EXPANDERS (operator ask 2026-08-04, k63: "a generation from
  // each model and/or preset") ----------------------------------------------
  // Both materialize the sweep as ORDINARY, VISIBLE, EDITABLE rows — one per model
  // / per preset — seeded by the same carry rules as "+ Add row" and then overridden
  // by the thing being swept. Nothing runs: "Run all rows" stays the ONE trigger, so
  // an accidental click on a 20-model registry costs a scroll, not 20 GPU jobs. Any
  // row can still be edited or removed before running.

  /** One row per ENABLED model in the registry (adapters / pipeline components are
   *  skipped — they are not generation targets, and the base select greys them out). */
  function addRowPerModel(): void {
    setTestRows((prev) => {
      const from = prev[prev.length - 1];
      const targets = models.filter((m) => !m.disabled);
      return [
        ...prev,
        ...targets.map((m, i) => ({
          ...seedTestRow(prev.length + i, from),
          model_id: m.id,
        })),
      ];
    });
  }

  /** One row per curated preset: seeded by the carry rules, then overridden by the
   *  preset's own model + non-null knob defaults.
   *
   *  Deliberately does NOT fire the preset apply/pre-warm POST that onPickPreset
   *  does. That POST warms ONE model on ONE GPU worker; firing it N times would
   *  thrash the fleet (each apply evicting the last) for rows that may never run.
   *  The rows carry the preset's numbers and load their model lazily on Run —
   *  exactly what a failed pre-warm already falls back to. */
  function addRowPerPreset(): void {
    setTestRows((prev) => {
      const from = prev[prev.length - 1];
      return [
        ...prev,
        ...presets.map((p, i) => {
          const base = seedTestRow(prev.length + i, from);
          const d = p.defaults;
          return {
            ...base,
            model_id: p.model_key || base.model_id,
            width: d.width != null ? String(d.width) : base.width,
            height: d.height != null ? String(d.height) : base.height,
            steps: d.steps != null ? String(d.steps) : base.steps,
            guidance: d.guidance != null ? String(d.guidance) : base.guidance,
            negative: d.negative != null ? d.negative : base.negative,
          };
        }),
      ];
    });
  }

  function patchTestRow(id: string, patch: Partial<ImageTestRow>): void {
    setTestRows((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  }

  function removeTestRow(id: string): void {
    setTestRows((prev) => prev.filter((r) => r.id !== id));
  }

  /** Rows that parse into a valid request. A row with a bad number or no text is
   *  SKIPPED rather than submitted with silently coerced values. */
  function runnableTestRows(): { rowId: string; req: GenerateImageRequest }[] {
    const media = buildMediaOnlyParts();
    const out: { rowId: string; req: GenerateImageRequest }[] = [];
    for (const r of testRows) {
      if (r.text.trim() === "") continue;
      const w = Number(r.width);
      const h = Number(r.height);
      const st = Number(r.steps);
      const g = Number(r.guidance);
      if (!Number.isFinite(w) || !Number.isFinite(h)) continue;
      if (!Number.isFinite(st) || !Number.isFinite(g)) continue;
      const sd = r.seed.trim() === "" ? null : Number(r.seed);
      if (sd !== null && !Number.isFinite(sd)) continue;
      out.push({
        rowId: r.id,
        req: {
          parts: [{ kind: "text", text: r.text }, ...media],
          model_id: r.model_id,
          width: w,
          height: h,
          steps: st,
          guidance: g,
          seed: sd,
          negative: r.negative.trim() === "" ? null : r.negative,
          project: projectName.trim() || undefined,
        },
      });
    }
    return out;
  }

  function runTester(): void {
    const rows = runnableTestRows();
    if (rows.length === 0) return;
    rowJobs.runRows(rows);
  }

  // STUDIO TESTER cross-model battery for the Scene / Movie run rows (operator
  // 2026-08-05). Category per tab: Scene → "image" when the Output selector is on
  // Image (a single still, rendered by a text-to-image model), else "scene"; Movie
  // → "movie". The optional `models` subset scopes the sweep to exactly those model
  // keys (backend: run_tester enumerates the whole type roster when it is omitted).
  //
  // SCENE has a genuine client-side filter: `models` from useImageModels() is the
  // registry's image-generation set, the SAME text-to-image registry the tester's
  // image/scene categories enumerate (their keys ARE model_keys). We pass the
  // ENABLED text-to-image entries so the battery is scoped to the servable image
  // roster; an empty list (models still loading) omits the subset and sweeps all.
  //
  // MOVIE has no client-side per-value filter available here: this station's model
  // list is the text-to-image registry, NOT the studio VIDEO roster the "movie"
  // tester category enumerates (capability t2v/i2v against the studio's own
  // registry). Passing image ids under a video category would sweep the wrong
  // models, so we OMIT `models` and let the backend enumerate the movie roster.
  // (Follow-up if per-value movie filtering is ever wanted client-side: read the
  // measured render table via useRenderPresets + modelsForCapability(MOVIE_CAPABILITY),
  // as the studio Cinema surface does — deliberately not done here to keep this UI-only.)
  async function runStudioTesterSweep(): Promise<void> {
    const p = assembledPrompt.trim();
    if (!p) {
      setStudioTesterMsg("Enter a prompt to test across every model of this type.");
      return;
    }
    const category =
      mode === "movie" ? "movie" : sceneOutputIsImage ? "image" : "scene";
    // Scene/Image: scope to the enabled text-to-image registry entries. Movie:
    // omit (see note above).
    const sceneModels =
      category === "movie"
        ? []
        : models
            .filter(
              (m) =>
                !m.disabled &&
                (m.tasks?.includes("text-to-image") ?? true),
            )
            .map((m) => m.id);
    setStudioTesterBusy(true);
    setStudioTesterMsg(null);
    try {
      const body: Record<string, unknown> = { category, prompt: p };
      if (widthNum != null) body.width = widthNum;
      if (heightNum != null) body.height = heightNum;
      if (fpsNum != null) body.fps = fpsNum;
      if (seed.trim() !== "" && seedNum != null) body.seed = seedNum;
      if (sceneModels.length > 0) body.models = sceneModels;
      const res = await request<unknown>(hugpyConfig.studioTesterUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        meta: { specKey: "studio", operation: "studio.tester.enqueue" },
      });
      if (!studioTesterMounted.current) return;
      if (!res.ok) {
        setStudioTesterMsg(`Tester failed: ${describeAppError(errorOf(res))}`);
        return;
      }
      setStudioTesterMsg(
        sceneModels.length > 0
          ? `Testing this prompt across ${sceneModels.length} ${category} ` +
              `model${sceneModels.length === 1 ? "" : "s"} that fit the current filter — ` +
              `each result appears in the generation log as it finishes.`
          : `Testing this prompt across every ${category} model — each result ` +
              `appears in the generation log as it finishes.`,
      );
    } finally {
      if (studioTesterMounted.current) setStudioTesterBusy(false);
    }
  }

  // Drop an LLM-assisted prompt into the composer. Targets the FIRST PURE-TEXT part
  // (no media) and replaces its `.text` — media parts and their captions are left
  // untouched, so pairings survive. If every part carries media (no pure-text part),
  // a fresh text part is PREPENDED rather than overwriting a caption.
  function applyAssistedPrompt(text: string): void {
    setParts((prev) => {
      const idx = prev.findIndex((p) => !p.media);
      if (idx === -1) return [{ key: partKey(), text }, ...prev];
      return prev.map((p, i) => (i === idx ? { ...p, text } : p));
    });
  }

  // Enhance / Generate — the shared hook owns the transport, one-at-a-time guard, and
  // error handling; this station supplies the draft (the assembled text parts) and the
  // apply (replace the primary text part). On ANY failure the hook surfaces .error
  // inline and never calls apply, so the existing prompt stays intact.
  function runAssist(assistMode: "detail" | "generate"): void {
    void assist.runAssist(assistMode, assembledPrompt, applyAssistedPrompt);
  }

  useEffect(() => {
    // Post-commit re-run after the warn panel names the run — fresh closure,
    // settled state, so the submit sees the new project name.
    if (!pendingRun) return;
    setPendingRun(false);
    onRun(true);
  });

  function onRun(skipNameCheck = false) {
    if (
      !skipNameCheck && warnPref && projectName.trim() === "" &&
      (mode === "image" || mode === "scene" || mode === "movie")
    ) {
      setNameWarn(true);
      return;
    }
    setNameWarn(false);
    // This guard is for the STATION-level knobs (movie + the image fallback).
    // Scene validates every knob PER-PART in buildSceneRow (station values may be
    // blank/unused when knobs are unshared), so it must not be gated here — else a
    // Run that canRunScene enabled would silently no-op. The scene branch re-checks
    // via `rows.length === 0`.
    if (
      mode !== "scene" &&
      (widthNum == null ||
        heightNum == null ||
        stepsNum == null ||
        guidanceNum == null)
    ) {
      return;
    }
    if (mode === "movie") {
      if (!canRunMovie || fpsNum == null) return;
      const builtGoals: MovieGoal[] = goals.map((g) => {
        // k92 — per-goal PROMPT-COMPONENT overrides. A knob is sent per-goal ONLY
        // when its share tick is OFF (else the goal INHERITS the movie-level value
        // sent at the top of the request — a shared movie is byte-identical to the
        // old payload). Numeric overrides parse defensively: a blank/invalid own
        // value falls back to the (validated) movie-level number, never NaN.
        const kv = goalKnobValues(g);
        const ex = goalExtra(g);
        // fallback is the movie-level value (non-null here — canRunMovie guards it);
        // `?? 0` only satisfies the type checker and is never reached at runtime.
        const num = (s: string, fallback: number | null): number =>
          toNumberOrNull(s) ?? fallback ?? 0;
        const goalOut: MovieGoal = {
          start_frame: Number(g.start),
          end_frame: Number(g.end),
          prompt: g.prompt,
          ...(g.ref ? { ref: g.ref.ref } : {}),
        };
        if (!movieShare.model && ex.model) goalOut.model_id = ex.model;
        if (!movieShare.size) {
          goalOut.width = num(kv.width, widthNum);
          goalOut.height = num(kv.height, heightNum);
        }
        if (!movieShare.steps) goalOut.steps = num(kv.steps, stepsNum);
        if (!movieShare.guidance) goalOut.guidance = num(kv.guidance, guidanceNum);
        if (!movieShare.seed)
          goalOut.seed = kv.seed.trim() === "" ? null : toNumberOrNull(kv.seed);
        if (!movieShare.strength)
          goalOut.strength =
            ex.strength.trim() === "" ? null : toNumberOrNull(ex.strength);
        if (!movieShare.chain) goalOut.chain = ex.chain;
        if (!movieShare.motion)
          goalOut.motion = ex.motion.trim() === "" ? null : ex.motion;
        if (!movieShare.negative) {
          const gneg = composeNegative(g.negative ?? "", g.stdNegative ?? false);
          goalOut.negative = gneg === "" ? null : gneg;
        }
        return goalOut;
      });
      const req: GenerateMovieRequest = {
        goals: builtGoals,
        model_id: modelId,
        width: widthNum,
        height: heightNum,
        steps: stepsNum,
        guidance: guidanceNum,
        fps: fpsNum,
        assemble,
        seed: seed.trim() === "" ? null : seedNum,
        // k88: the wired negative is goal #1's tabbed pane (the shared station
        // field), with the standard set composed in when its checkbox is on.
        // Composition happens HERE, at payload build — the wire is unchanged.
        negative: (() => {
          const eff = composeNegative(negative, goals[0]?.stdNegative ?? false);
          return eff === "" ? null : eff;
        })(),
        strength: strengthNum,
        chain,
        project: projectName.trim() || undefined,
        ...(visionEnabled
          ? {
              vision_enabled: true,
              ...(scoreThresholdNum != null ? { score_threshold: scoreThresholdNum } : {}),
              ...(maxAttemptsNum != null
                ? { max_attempts_per_segment: maxAttemptsNum }
                : {}),
              ...(judgeModelId.trim() ? { judge_model_id: judgeModelId.trim() } : {}),
              ...(timeBudgetNum != null ? { time_budget_s: timeBudgetNum } : {}),
            }
          : { vision_enabled: false }),
      };
      movieJob.run(req);
      return;
    }
    if (mode === "scene") {
      // FAN OUT: each TEXT-BEARING component is an INDEPENDENT scene → its own job
      // → its own output ("2 prompts = 2 outputs"). computeSceneRows resolves each
      // part's settings (shared value when ☑, else its own), validates every knob
      // (invalid parts are skipped, not submitted with coerced values), folds in
      // the shared media (identity / start frame) as leading context, and composes
      // each part's negative (own value else the shared station negative).
      const rows = computeSceneRows();
      if (rows.length === 0) return;
      // One GPU job per prompt on a single-GPU box — a "test all models" scaffold
      // can be dozens. Confirm before flooding the queue.
      if (
        rows.length > 8 &&
        !window.confirm(
          `Run ${rows.length} generations? That's one GPU job per prompt; they queue on the GPU and run largely sequentially.`,
        )
      )
        return;
      sceneRowJobs.runRows(rows);
      return;
    }
    // Single-shot IMAGE fallback — no tab reaches mode "image" since the fold
    // (Scene's Output=Image runs through sceneJob above); kept as the retained
    // image-job wiring alongside the always-mounted useGenerateJob hook.
    if (!canRunImage) return;
    const req: GenerateImageRequest = {
      parts: buildPromptParts(),
      model_id: modelId,
      width: widthNum,
      height: heightNum,
      steps: stepsNum,
      guidance: guidanceNum,
      seed: seed.trim() === "" ? null : seedNum,
      // Same explicit value as the scene path (part #1's visible field) — the
      // retained image-job wiring must not drift from what the live submit sends.
      negative: (() => {
        const eff = sceneGroupNegative();
        return eff === "" ? null : eff;
      })(),
      project: projectName.trim() || undefined,
    };
    imageJob.run(req);
  }

  return (
    <section className="station-card station-wide">
      {/* MODE SWITCH — Scene (whose Output selector covers single Image · Carousel
          · Video) vs Movie (goal timeline) vs Clip / Cinema (the promoted studio
          surfaces — clip vs movie-composer). These are internal modes of this station,
          not separate registry entries. The old Image tab folded into Scene; the old
          single Studio tab split into Clip + Cinema (operator 2026-08-05). */}
      <div
        className="vi-gen-mode"
        role="tablist"
        aria-label="Generate mode"
      >
        <button
          type="button"
          role="tab"
          aria-selected={mode === "scene"}
          className={`vi-gen-mode-btn${
            mode === "scene" ? " vi-gen-mode-btn-active" : ""
          }`}
          onClick={() => setMode("scene")}
        >
          Scene
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "movie"}
          className={`vi-gen-mode-btn${
            mode === "movie" ? " vi-gen-mode-btn-active" : ""
          }`}
          onClick={() => setMode("movie")}
        >
          Movie
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "clip"}
          className={`vi-gen-mode-btn${
            mode === "clip" ? " vi-gen-mode-btn-active" : ""
          }`}
          onClick={() => setMode("clip")}
        >
          Clip
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "cinema"}
          className={`vi-gen-mode-btn${
            mode === "cinema" ? " vi-gen-mode-btn-active" : ""
          }`}
          onClick={() => setMode("cinema")}
        >
          Cinema
        </button>
      </div>

      {/* CLIP / CINEMA tabs render the shared studio generate surface (its own self-
          contained, preset-shape-morphing panel), NOT the image/scene/movie
          SectionTabs + Options-rail + Run-row layout. The surface is LOCKED to the
          matching sub-mode: Clip → "clip", Cinema → "movie" (the StudioMovieComposer). */}
      {isStudioSurface && (
        <StudioGenerateMode lockedSurfaceMode={mode === "cinema" ? "movie" : "clip"} />
      )}

      {!isStudioSurface && (
      <SectionTabs
        ariaLabel="Generate station"
        initialTab="components"
        optionsHost={reg.settingsHost}
        options={
        /* OPTIONS — model + explicit generation knobs (shared + scene-only). */
        <aside className="vi-frames-rail">
          {/* ABOUT — the condensed what-is-this / how-to prose (k89), collapsed at
              the top of the hosted Settings content per mode. Absorbs the former
              inline guidance (Scene's empty-composer prose + the "first image part
              is the start frame" hint; Movie already had none inline). */}
          {mode === "scene" && (
            <AboutExpander title="About Scene">
              <p>
                Builds one generation from an ORDERED list of prompt parts — text,
                images, or a video, sent in the order listed. Each part pairs text
                with optional media.
              </p>
              <p>
                Output (below) picks the shape: Image = one still, Carousel = the
                frames as separate stills, Video = the frames muxed into an mp4.
              </p>
              <p>
                The first image part is the start frame and conditions generation
                (image-to-image, set by the strength knob); video parts are resolved
                to frames server-side.
              </p>
              <p>
                Each prompt has its OWN Negative tab, sent exactly as written (blank
                = none; nothing is merged at submit). The std. tick inserts or
                removes the standard exclusion set in that field; part #1&apos;s
                Negative is also the group assist&apos;s shared negative.
              </p>
              <p>
                The Tester at the bottom compares prompts, settings and models side
                by side without touching the main Run.
              </p>
            </AboutExpander>
          )}
          {mode === "movie" && (
            <AboutExpander title="About Movie">
              <p>
                Renders a movie from a GOAL TIMELINE — contiguous goals tile frames
                [0, total), each goal a prompt over its own frame window.
              </p>
              <p>
                A goal can carry an optional reference image; one negative per movie
                (goal #1&apos;s Negative tab).
              </p>
              <p>
                The toolbar&apos;s ✨ actions write every ticked goal in ONE coherent
                pass — unticked goals ride along as locked context.
              </p>
              <p>
                The vision director (opt-in, below) has a judge VLM score each
                segment and re-roll it until it passes.
              </p>
            </AboutExpander>
          )}
          {/* PROJECT — optional auto-archive folder name. When set, the run is
              filed under assets/<project> and the finished result says where. */}
          <div className="vi-knob">
            <label htmlFor="vi-gen-project">Project name (optional)</label>
            <input
              id="vi-gen-project"
              type="text"
              className="vi-knob-input"
              list="vi-gen-project-list"
              placeholder="untitled — or pick / type a project"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
            />
            <datalist id="vi-gen-project-list">
              {knownProjects.map((name) => (
                <option key={name} value={name} />
              ))}
            </datalist>
            <span className="vi-knob-hint">
              Choose an existing project or type a new name. Saves this run&apos;s frames +
              clip to assets/&lt;project&gt;. Blank = an auto-named folder.
            </span>
            <WarnUnnamedToggle checked={warnPref} onChange={setWarnPref} />
          </div>

          {/* PRESET — a curated "ideal default load". Selecting one prefills every
              knob + the model and pre-warms that model on a GPU worker. */}
          <div className="vi-knob">
            <label htmlFor="vi-gen-preset">Preset</label>
            <select
              id="vi-gen-preset"
              className="vi-knob-select"
              value={presetId}
              onChange={(e) => void onPickPreset(e.target.value)}
              disabled={presetsLoading}
            >
              <option value="">
                {presetsLoading ? "Loading presets…" : "Choose a preset…"}
              </option>
              {presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            <span className="vi-knob-hint">
              A curated default load — prefills the knobs + model below and warms
              that model on a GPU worker. Tweak anything after.
            </span>
            {selectedPreset?.description && (
              <span className="vi-knob-hint">{selectedPreset.description}</span>
            )}
            {presetStatus && (
              <span className="vi-status vi-status-running">{presetStatus}</span>
            )}
            {presetError && <span className="vi-error">{presetError}</span>}
            {presetsError && <span className="vi-error">{presetsError}</span>}
          </div>

          {/* k63 (operator ask 2026-08-04): "the prompt component absorbs the
              settings". The GENERATION knobs — model select + width/height/steps/
              guidance/seed — are no longer rendered in this rail for IMAGE and SCENE:
              they moved INTO the centre composer as a condensed settings strip
              attached to the prompt parts, the same relocation the negative prompt
              already made (see Item A below and the `vi-knob-strip` in the
              `vi-gen-composer` block). State is untouched — it still lives in this
              station's useState and both surfaces drive the same setters; only the
              rendering moved.

              MOVIE keeps them here, unchanged: movie mode renders a goal timeline
              instead of the parts composer, so it has no prompt component to absorb
              them and would otherwise lose access to its own knobs. */}
          {mode === "movie" && (
          <>
          <div className="vi-knob">
            <label htmlFor="vi-gen-model">Image model</label>
            <select
              id="vi-gen-model"
              className="vi-knob-select"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={modelsLoading}
            >
              {modelsLoading && <option value="">Loading models…</option>}
              {!modelsLoading && models.every((m) => m.disabled) && (
                <option value="">No image models available</option>
              )}
              {/* Greyed rows are adapters / pipeline components / unclassified
                  models: present on disk, but not generation targets. Disabled,
                  with the backend's own reason as the tooltip (k61). */}
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={m.disabled} title={m.reason}>
                  {m.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              onClick={refreshModels}
              disabled={modelsLoading}
              title="Re-scan the registry for image models adopted this session"
            >
              ↻ Refresh models
            </button>
            {/* CAPABILITY LINE — what THIS model needs, from its tasks[]. */}
            <span className="vi-knob-hint vi-knob-range">
              {capabilityHint(req.image)}
            </span>
            <span className="vi-knob-hint">
              In Scene mode the first image part is the start frame and now
              conditions generation (image-to-image), set by the strength knob.
              Video parts are resolved to frames server-side.
            </span>
            {modelsError && <span className="vi-error">{modelsError}</span>}
          </div>

          {/* width + height share one row — the dead-space offender from stacking two
              plain number inputs full-width. Their explanations move into a hover
              (desktop) / tap (mobile) "ⓘ" — see KnobInfo — the inline red validation
              flags stay exactly as before, always visible. */}
          <div className="vi-knob-row">
            <div className="vi-knob">
              <div className="vi-knob-label-row">
                <label htmlFor="vi-gen-width">width</label>
                <KnobInfo
                  label="width"
                  text="Output width in pixels. Multiples of 8 avoid the model silently rounding it."
                />
              </div>
              <input
                id="vi-gen-width"
                type="number"
                min="8"
                step="8"
                inputMode="numeric"
                className="vi-knob-input"
                value={width}
                onChange={(e) => setWidth(e.target.value)}
              />
              {!widthValid ? (
                <span className="vi-knob-flag">Enter a positive integer.</span>
              ) : (
                !widthMult8 && (
                  <span className="vi-knob-flag">
                    Not a multiple of 8 — the model may round it.
                  </span>
                )
              )}
            </div>

            <div className="vi-knob">
              <div className="vi-knob-label-row">
                <label htmlFor="vi-gen-height">height</label>
                <KnobInfo
                  label="height"
                  text="Output height in pixels. Multiples of 8 avoid the model silently rounding it."
                />
              </div>
              <input
                id="vi-gen-height"
                type="number"
                min="8"
                step="8"
                inputMode="numeric"
                className="vi-knob-input"
                value={height}
                onChange={(e) => setHeight(e.target.value)}
              />
              {!heightValid ? (
                <span className="vi-knob-flag">Enter a positive integer.</span>
              ) : (
                !heightMult8 && (
                  <span className="vi-knob-flag">
                    Not a multiple of 8 — the model may round it.
                  </span>
                )
              )}
            </div>
          </div>

          {/* steps + guidance + seed are all plain numbers (seed is numbers-or-blank) —
              consolidated onto one compact row instead of two stacked full-width
              blocks. Same KnobInfo pattern for each explanation. */}
          <div className="vi-knob-row">
            <div className="vi-knob">
              <div className="vi-knob-label-row">
                <label htmlFor="vi-gen-steps">steps</label>
                <KnobInfo label="steps" text="Denoising steps. sd-turbo runs well at ~4." />
              </div>
              <input
                id="vi-gen-steps"
                type="number"
                min="1"
                step="1"
                inputMode="numeric"
                className="vi-knob-input"
                value={steps}
                onChange={(e) => setSteps(e.target.value)}
              />
              {!stepsValid && (
                <span className="vi-knob-flag">Enter a positive integer.</span>
              )}
            </div>

            <div className="vi-knob">
              <div className="vi-knob-label-row">
                <label htmlFor="vi-gen-guidance">guidance</label>
                <KnobInfo
                  label="guidance"
                  text="CFG scale. sd-turbo uses 0 (guidance-free)."
                />
              </div>
              <input
                id="vi-gen-guidance"
                type="number"
                min="0"
                step="0.5"
                inputMode="decimal"
                className="vi-knob-input"
                value={guidance}
                onChange={(e) => setGuidance(e.target.value)}
              />
              {!guidanceValid && (
                <span className="vi-knob-flag">Enter a number ≥ 0.</span>
              )}
            </div>
          </div>

          {/* seed is numbers-or-blank — compact, capped-width input instead of the
              old full-width block (it doesn't pair with steps/guidance: three across
              doesn't fit this rail without risking a long seed value clipping). */}
          <div className="vi-knob">
            <div className="vi-knob-label-row">
              <label htmlFor="vi-gen-seed">seed</label>
              <KnobInfo label="seed" text="Optional. Blank = random each run." />
            </div>
            <input
              id="vi-gen-seed"
              type="number"
              min="0"
              step="1"
              inputMode="numeric"
              className="vi-knob-input vi-knob-input-compact"
              placeholder="random"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
            />
            {!seedValid && (
              <span className="vi-knob-flag">
                Enter a non-negative integer or leave blank.
              </span>
            )}
          </div>
          </>
          )}

          {/* Item A: the NEGATIVE prompt was relocated OUT of this left rail and INTO
              the center composer (rendered in tandem with the prompt via <GoalComposer>,
              mirroring the studio surface). Its state / preset-prefill / submit wiring are
              unchanged — only the rendering moved. See the `vi-gen-composer` block below. */}

          {/* SCENE-ONLY KNOBS — surfaced only in Scene mode. */}
          {mode === "scene" && (
            <>
              {/* OUTPUT selector (operator 2026-08-05): the "mainframe" choice —
                  Image (1 still) · Carousel (N stills) · Video (N frames → mp4).
                  It drives the existing nFrames + assemble knobs below, so a scene
                  now covers the single-image case (Image = 1 frame, no mp4) that
                  the separate Image tab did. The knobs below still fine-tune it. */}
              <div className="vi-knob">
                <label>output</label>
                <div role="tablist" aria-label="Scene output"
                     style={{ display: "flex", gap: "0.3rem", flexWrap: "wrap" }}>
                  {([
                    ["image", "Image", "A single still (1 frame)."],
                    ["carousel", "Carousel", "The frames as separate stills."],
                    ["video", "Video", "The frames muxed into a playable mp4."],
                  ] as const).map(([key, label, hint]) => {
                    const active =
                      key === "image"
                        ? nFramesNum === 1 && !assemble
                        : key === "carousel"
                          ? nFramesNum != null && nFramesNum > 1 && !assemble
                          : assemble;
                    return (
                      <button
                        key={key}
                        type="button"
                        role="tab"
                        aria-selected={active}
                        className={active ? "vi-btn vi-btn-sm vi-btn-accent" : "vi-btn vi-btn-sm vi-btn-ghost"}
                        title={hint}
                        onClick={() => {
                          if (key === "image") {
                            setNFrames("1");
                            setAssemble(false);
                          } else if (key === "carousel") {
                            if (nFramesNum == null || nFramesNum <= 1) setNFrames("6");
                            setAssemble(false);
                          } else {
                            if (nFramesNum == null || nFramesNum <= 1) setNFrames("6");
                            setAssemble(true);
                          }
                        }}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
                <span className="vi-knob-hint">
                  Image = one still (1 frame). Carousel = the frames as stills.
                  Video = the frames muxed into an mp4.
                </span>
              </div>

              <div className="vi-knob">
                <label htmlFor="vi-gen-nframes">frames (n)</label>
                <input
                  id="vi-gen-nframes"
                  type="number"
                  min="1"
                  max="24"
                  step="1"
                  inputMode="numeric"
                  className="vi-knob-input"
                  value={nFrames}
                  onChange={(e) => setNFrames(e.target.value)}
                />
                <span className="vi-knob-hint">
                  Consecutive frames in the scene. Backend cap is 24.
                </span>
                {!nFramesValid && (
                  <span className="vi-knob-flag">
                    Enter an integer from 1 to 24.
                  </span>
                )}
              </div>

              <div className="vi-knob">
                <label htmlFor="vi-gen-motion">motion (optional)</label>
                <input
                  id="vi-gen-motion"
                  type="text"
                  className="vi-knob-input"
                  placeholder="e.g. slow pan, frame {i} of {n}"
                  value={motion}
                  onChange={(e) => setMotion(e.target.value)}
                />
                <span className="vi-knob-hint">
                  Per-frame progression template; may contain {"{i}"}/{"{n}"}.
                  Blank = none.
                </span>
              </div>

              <div className="vi-knob">
                <label htmlFor="vi-gen-fps">fps</label>
                <input
                  id="vi-gen-fps"
                  type="number"
                  min="1"
                  step="1"
                  inputMode="numeric"
                  className="vi-knob-input"
                  value={fps}
                  onChange={(e) => setFps(e.target.value)}
                />
                <span className="vi-knob-hint">
                  Playback rate of the assembled clip.
                </span>
                {!fpsValid && (
                  <span className="vi-knob-flag">Enter a positive integer.</span>
                )}
              </div>

              <div className="vi-knob">
                <label className="vi-knob-check" htmlFor="vi-gen-assemble">
                  <input
                    id="vi-gen-assemble"
                    type="checkbox"
                    checked={assemble}
                    onChange={(e) => setAssemble(e.target.checked)}
                  />
                  assemble mp4
                </label>
                <span className="vi-knob-hint">
                  Build a playable clip from the frames (added last).
                </span>
              </div>

              <div className="vi-knob">
                <label htmlFor="vi-gen-strength">strength</label>
                <input
                  id="vi-gen-strength"
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  inputMode="decimal"
                  className="vi-knob-input"
                  value={strength}
                  onChange={(e) => setStrength(e.target.value)}
                />
                <span className="vi-knob-hint">
                  Image-to-image denoising strength (0–1) applied to the start
                  frame. Lower keeps more of it; 0.45 is a balanced default.
                </span>
                {!strengthValid && (
                  <span className="vi-knob-flag">
                    Enter a number from 0 to 1.
                  </span>
                )}
              </div>

              <div className="vi-knob">
                <label className="vi-knob-check" htmlFor="vi-gen-chain">
                  <input
                    id="vi-gen-chain"
                    type="checkbox"
                    checked={chain}
                    onChange={(e) => setChain(e.target.checked)}
                  />
                  chain frames
                </label>
                <span className="vi-knob-hint">
                  Condition each frame on the previous one.
                </span>
              </div>
            </>
          )}

          {/* MOVIE-ONLY KNOBS — surfaced only in Movie mode. fps/assemble/strength/
              chain are SHARED with Scene (same setters); the goal timeline (not
              n_frames) defines frame ranges. Then the opt-in vision director panel. */}
          {mode === "movie" && (
            <>
              {/* MOVIE TEMPLATE — a curated shot list. Picking one drops a ready goal
                  timeline + the shared knobs (model/dims/steps/guidance/fps/chain) +
                  director defaults into the editor. Populates only; does not run. */}
              <div className="vi-knob">
                <label htmlFor="vi-gen-movie-template">Template</label>
                <select
                  id="vi-gen-movie-template"
                  className="vi-knob-select"
                  value={moviePresetId}
                  onChange={(e) => onPickMoviePreset(e.target.value)}
                  disabled={moviePresetsLoading}
                >
                  <option value="">
                    {moviePresetsLoading ? "Loading templates…" : "— Template —"}
                  </option>
                  {moviePresets.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
                <span className="vi-knob-hint">
                  A curated movie — drops a ready goal timeline + settings into the
                  editor below. Run it as-is or tweak any goal first.
                </span>
                {selectedMoviePreset?.description && (
                  <span className="vi-knob-hint">{selectedMoviePreset.description}</span>
                )}
                {moviePresetsError && (
                  <span className="vi-error">{moviePresetsError}</span>
                )}
              </div>

              <div className="vi-knob">
                <label htmlFor="vi-gen-movie-fps">fps</label>
                <input
                  id="vi-gen-movie-fps"
                  type="number"
                  min="1"
                  step="1"
                  inputMode="numeric"
                  className="vi-knob-input"
                  value={fps}
                  onChange={(e) => setFps(e.target.value)}
                />
                <span className="vi-knob-hint">
                  Playback rate of the assembled movie — sets the clip length
                  ({movieTotalFrames} frames ÷ fps).
                </span>
                {!fpsValid && (
                  <span className="vi-knob-flag">Enter a positive integer.</span>
                )}
              </div>

              <div className="vi-knob">
                <label className="vi-knob-check" htmlFor="vi-gen-movie-assemble">
                  <input
                    id="vi-gen-movie-assemble"
                    type="checkbox"
                    checked={assemble}
                    onChange={(e) => setAssemble(e.target.checked)}
                  />
                  assemble mp4
                </label>
                <span className="vi-knob-hint">
                  Build one playable movie.mp4 from all segment frames (added last).
                </span>
              </div>

              <div className="vi-knob">
                <label htmlFor="vi-gen-movie-strength">strength</label>
                <input
                  id="vi-gen-movie-strength"
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  inputMode="decimal"
                  className="vi-knob-input"
                  value={strength}
                  onChange={(e) => setStrength(e.target.value)}
                />
                <span className="vi-knob-hint">
                  Image-to-image denoising strength (0–1) for a goal&apos;s reference
                  image. 0.45 is a balanced default.
                </span>
                {!strengthValid && (
                  <span className="vi-knob-flag">Enter a number from 0 to 1.</span>
                )}
              </div>

              <div className="vi-knob">
                <label className="vi-knob-check" htmlFor="vi-gen-movie-chain">
                  <input
                    id="vi-gen-movie-chain"
                    type="checkbox"
                    checked={chain}
                    onChange={(e) => setChain(e.target.checked)}
                  />
                  chain frames
                </label>
                <span className="vi-knob-hint">
                  Condition each frame on the previous one (continuity across cuts).
                </span>
              </div>

              {/* DIRECTOR PANEL — collapsed by default (vision off). Toggling it on
                  reveals the scoring/re-roll knobs + the judge-VLM select. */}
              <div className="vi-knob vi-director">
                <label className="vi-knob-check" htmlFor="vi-gen-vision">
                  <input
                    id="vi-gen-vision"
                    type="checkbox"
                    checked={visionEnabled}
                    onChange={(e) => setVisionEnabled(e.target.checked)}
                  />
                  vision director
                </label>
                <span className="vi-knob-hint">
                  Off — segments are rendered once, no scoring. Turn on to have a
                  judge VLM score each segment and re-roll until it passes.
                </span>

                {visionEnabled && (
                  <div className="vi-director-body">
                    <div className="vi-knob">
                      <label htmlFor="vi-gen-score">score threshold</label>
                      <input
                        id="vi-gen-score"
                        type="number"
                        min="0"
                        max="100"
                        step="1"
                        inputMode="numeric"
                        className="vi-knob-input"
                        value={scoreThreshold}
                        onChange={(e) => setScoreThreshold(e.target.value)}
                      />
                      <span className="vi-knob-hint">
                        Pass mark (0–100). A segment re-rolls until it scores this or
                        higher.
                      </span>
                      {!scoreThresholdValid && (
                        <span className="vi-knob-flag">
                          Enter a number from 0 to 100.
                        </span>
                      )}
                    </div>

                    <div className="vi-knob">
                      <label htmlFor="vi-gen-attempts">max attempts / segment</label>
                      <input
                        id="vi-gen-attempts"
                        type="number"
                        min="1"
                        step="1"
                        inputMode="numeric"
                        className="vi-knob-input"
                        value={maxAttempts}
                        onChange={(e) => setMaxAttempts(e.target.value)}
                      />
                      <span className="vi-knob-hint">
                        Re-roll ceiling before the best take is kept regardless.
                      </span>
                      {!maxAttemptsValid && (
                        <span className="vi-knob-flag">
                          Enter a positive integer.
                        </span>
                      )}
                    </div>

                    <div className="vi-knob">
                      <label htmlFor="vi-gen-judge">judge model</label>
                      <select
                        id="vi-gen-judge"
                        className="vi-knob-select"
                        value={judgeModelId}
                        onChange={(e) => setJudgeModelId(e.target.value)}
                        disabled={judgeLoading}
                      >
                        {/* Keep the default selectable even when the registry list
                            is empty / still loading (the option may not be listed). */}
                        {!judgeModels.some((m) => m.id === judgeModelId) && (
                          <option value={judgeModelId}>{judgeModelId}</option>
                        )}
                        {judgeModels.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.label}
                          </option>
                        ))}
                      </select>
                      <button
                        type="button"
                        className="vi-btn vi-btn-sm vi-btn-ghost"
                        onClick={refreshJudge}
                        disabled={judgeLoading}
                        title="Re-scan the registry for image-text-to-text judge models"
                      >
                        ↻ Refresh judges
                      </button>
                      <span className="vi-knob-hint">
                        A vision-language model (image-text-to-text) that scores each
                        segment.
                      </span>
                      {judgeError && <span className="vi-error">{judgeError}</span>}
                    </div>

                    <div className="vi-knob">
                      <label htmlFor="vi-gen-budget">time budget (s, optional)</label>
                      <input
                        id="vi-gen-budget"
                        type="number"
                        min="1"
                        step="1"
                        inputMode="numeric"
                        className="vi-knob-input"
                        placeholder="none"
                        value={timeBudget}
                        onChange={(e) => setTimeBudget(e.target.value)}
                      />
                      <span className="vi-knob-hint">
                        Overall wall-clock cap for the re-roll loop. Blank = none.
                      </span>
                      {!timeBudgetValid && (
                        <span className="vi-knob-flag">
                          Enter a positive number or leave blank.
                        </span>
                      )}
                    </div>

                    <span className="vi-knob-hint vi-knob-range">
                      Scoring adds ~5s per segment.
                    </span>
                  </div>
                )}
              </div>
            </>
          )}

        </aside>
        }
        addFile={
          /* ADD MEDIA — the ordered-prompt part input (+ text / image / video)
             and its pickers. The Run action lives in the components footer. */
          <>
            {/* MOVIE MODE — the prompt lives per-goal in the timeline below, so the
                parts requirement banner + add-bar are replaced by a short hint. The
                shared image picker still renders (it attaches a goal's reference). */}
            {mode === "movie" && (
              <div className="vi-gen-req">
                <span className="vi-gen-req-line">
                  Build the GOAL TIMELINE below — contiguous goals tile the whole
                  clip. Each goal carries its own prompt and an optional reference
                  image; the vision director (left) is opt-in.
                </span>
              </div>
            )}
            {mode !== "movie" && (
              <>
            {/* REQUIREMENT BANNER — capability-driven. The always-on plain-words hint
                line (requirementSummary) was removed (operator 2026-07-13: static
                guidance clutter). What remains is real, actionable state: for
                edit-only models this surfaces an EXPLICIT required start-image
                control that gates Run (and the vi-gen-req-missing modifier still
                flags the missing case). */}
            {startImageRequired && (
              <div className={`vi-gen-req${hasStartImage ? "" : " vi-gen-req-missing"}`}>
                <button
                  type="button"
                  className={`vi-btn vi-btn-sm${hasStartImage ? "" : " vi-btn-accent"}`}
                  onClick={() => {
                    setAttachTarget(null); // append mode, not attach
                    setShowImagePicker(true);
                    setShowVideoPicker(false);
                  }}
                  title="Add the start image this edit model requires"
                >
                  {hasStartImage ? "✓ Start image added" : "+ Add start image (required)"}
                </button>
              </div>
            )}

            {/* DUPLICATE ADD-BAR — SUPPRESSED for image + scene modes (operator
                2026-07-13: this top add-bar duplicates the add/attach affordances
                already reachable from the composer below the viewer + the
                requirement banner's start-image button). Gated (not deleted —
                archive-in-place, reversible) so any FUTURE mode that renders the
                addFile slot and genuinely needs a top-level append bar can still
                reach it; image + scene are the only modes that reach here today, so
                this renders nowhere now. The underlying handlers (addText /
                addImage / addVideoFromLibrary) are untouched — the composer's
                per-part attach controls and the pickers still call them. (Scene is
                the only mode that reaches this branch since the Image tab folded
                into it, so the gate needs just the one test now.) */}
            {mode !== "scene" && (
            <div className="vi-gen-add-bar">
              <button type="button" className="vi-btn" onClick={addText}>
                + Add text (required)
              </button>
              {/* A pure text-to-image model can't consume a start image, so its
                  "Add image" affordance is omitted (req.image === "none"). */}
              {req.image !== "none" && (
                <button
                  type="button"
                  className="vi-btn"
                  onClick={() => {
                    setAttachTarget(null); // append mode, not attach
                    setShowImagePicker((v) => !v);
                    setShowVideoPicker(false);
                  }}
                >
                  {startImageRequired
                    ? "+ Add start image from library"
                    : "+ Add image from library"}
                </button>
              )}
              <button
                type="button"
                className="vi-btn"
                onClick={() => {
                  setAttachTarget(null); // append mode, not attach
                  setShowVideoPicker((v) => !v);
                  setShowImagePicker(false);
                }}
              >
                + Add video part
              </button>
            </div>
            )}
              </>
            )}

            {showImagePicker && (
              <div className="vi-gen-picker vi-studio-strip">
                {attachTarget != null && (
                  <p className="vi-crop-hint">
                    Attaching to part #
                    {parts.findIndex((p) => p.key === attachTarget) + 1} — the
                    picked image pairs with that part&apos;s text.
                  </p>
                )}
                {goalAttachTarget != null && (
                  <p className="vi-crop-hint">
                    Attaching a reference image to goal #
                    {goals.findIndex((g) => g.id === goalAttachTarget) + 1} of the
                    timeline.
                  </p>
                )}
                {renderImagePickerBody(addImageMedia)}
              </div>
            )}

            {showVideoPicker && (
              <div className="vi-gen-picker vi-studio-strip">
                {attachTarget != null && (
                  <p className="vi-crop-hint">
                    Attaching to part #
                    {parts.findIndex((p) => p.key === attachTarget) + 1} — the
                    picked video pairs with that part&apos;s text.
                  </p>
                )}
                <p className="vi-crop-hint">
                  A video part is resolved into frames automatically (uniform
                  sampling); for precise control, extract + pick frames in the
                  Frames station and add them as images.
                </p>
                {renderVideoPickerBody(addVideoMedia)}
              </div>
            )}
          </>
        }
        components={
          /* COMPONENTS — result/preview, the ordered prompt composer, and the
             in-panel Run footer (no floating buttons). */
          <div className="vi-station-components">
          {/* CENTER PREVIEW / RESULT — ALWAYS rendered (blank preview area in
              the idle state, before a job produces anything). */}
          <div className="vi-studio-preview">
              {mode === "movie" ? (
                movieJob.status === "done" && movieJob.outputs.length > 0 ? (
                  <figure className="vi-gen-result vi-scene-result vi-movie-result">
                    {movieClip && (
                      <div className="vi-scene-clip">
                        <div className="vi-video-clip">
                          <video
                            controls
                            className="vi-video-player"
                            src={mediaBytesUrl(movieClip.uri)}
                          />
                        </div>
                        <a
                          className="vi-btn vi-btn-sm"
                          href={mediaBytesUrl(movieClip.uri)}
                          download={`movie.${extFromMime(movieClip.mime)}`}
                        >
                          ↓ Download movie
                        </a>
                      </div>
                    )}
                    <div className="vi-frames-grid vi-scene-frames vi-studio-strip">
                      {movieFrames.map((f, i) => (
                        <span className="vi-scene-frame-cell" key={f.uri}>
                          <img
                            className="vi-scene-frame-img"
                            src={mediaBytesUrl(f.uri)}
                            alt={`Movie frame ${i + 1}`}
                            loading="lazy"
                          />
                          <span className="vi-frame-tag">#{i + 1}</span>
                          <a
                            className="vi-frame-dl"
                            href={mediaBytesUrl(f.uri)}
                            download={`movie-frame-${i + 1}.${extFromMime(f.mime)}`}
                            title={`Download frame ${i + 1}`}
                            aria-label={`Download movie frame ${i + 1}`}
                          >
                            ↓
                          </a>
                        </span>
                      ))}
                    </div>
                    <figcaption className="vi-gen-note">
                      {movieFrames.length} frame
                      {movieFrames.length === 1 ? "" : "s"}
                      {movieClip ? " + movie" : ""} added to the library — reuse them
                      below.
                      {project?.dir && (
                        <span className="vi-gen-saved">
                          {" · Saved to "}
                          {project.name ? `${project.name} · ` : ""}
                          <code>{project.dir}</code>
                        </span>
                      )}
                    </figcaption>
                  </figure>
                ) : movieJob.status === "failed" ? (
                  <p className="vi-error" role="alert">
                    {jobError ?? "Movie generation failed."}
                  </p>
                ) : (
                  <div className="vi-gen-preview-pending">
                    {/* OVERALL — segment d/t + stage (the nested movie readout). */}
                    <span className={`vi-status vi-status-${movieJob.status}`}>
                      {movieProgress
                        ? movieReadout(movieProgress)
                        : `${movieJob.status ?? "queued"}…`}
                    </span>
                    {movieProgress && (
                      <>
                        {/* CURRENT SEGMENT — the flat JobProgress-shaped `current`
                            fed VERBATIM to the shipped progressReadout + frame grid. */}
                        {movieProgress.current && (
                          <div className="vi-movie-current">
                            <span className="vi-status vi-status-running">
                              {progressReadout(movieProgress.current)}
                            </span>
                            {movieProgress.current.frames &&
                              movieProgress.current.frames.length > 0 && (
                                <div className="vi-frames-grid vi-scene-frames vi-studio-strip">
                                  {movieProgress.current.frames.map((f, i) => (
                                    <span className="vi-scene-frame-cell" key={f.uri}>
                                      <img
                                        className="vi-scene-frame-img"
                                        src={mediaBytesUrl(f.uri)}
                                        alt={`Current frame ${i + 1}`}
                                        loading="lazy"
                                      />
                                      <span className="vi-frame-tag">#{i + 1}</span>
                                    </span>
                                  ))}
                                </div>
                              )}
                          </div>
                        )}
                        {/* PER-SEGMENT STRIP — goal label + take + score + status. */}
                        <div className="vi-movie-seg-strip vi-studio-strip">
                          {movieProgress.segments.map((seg) => {
                            const st = seg.status ?? "pending";
                            return (
                              <span
                                key={seg.index}
                                className={`vi-movie-seg vi-movie-seg-${st}`}
                              >
                                <span className="vi-movie-seg-idx">
                                  #{seg.index + 1}
                                </span>
                                <span className="vi-movie-seg-label">
                                  {segmentLabel(seg.prompt, seg.index)}
                                </span>
                                <span className="vi-movie-seg-meta">
                                  <span className="vi-movie-seg-status">{st}</span>
                                  {seg.attempt != null && seg.attempt > 0 && (
                                    <span className="vi-movie-seg-attempt">
                                      take {seg.attempt}
                                    </span>
                                  )}
                                  <span
                                    className={`vi-movie-score${
                                      seg.score == null ? " vi-movie-score-none" : ""
                                    }`}
                                    title="Judge score (0–100)"
                                  >
                                    {seg.score != null ? Math.round(seg.score) : "—"}
                                  </span>
                                </span>
                              </span>
                            );
                          })}
                        </div>
                      </>
                    )}
                  </div>
                )
              ) : (
                /* SCENE — fans out per prompt now (sceneRowJobs); each prompt's
                   result renders under its own card. These single-sceneJob
                   done/failed branches are unreachable (kept only so a stale
                   durable record can never shadow the live fan-out summary). */
                false && sceneJob.status === "done" && sceneFrames.length > 0 ? (
                  <figure className="vi-gen-result vi-scene-result">
                    {sceneClip && (
                      <div className="vi-scene-clip">
                        <div className="vi-video-clip">
                          <video
                            controls
                            className="vi-video-player"
                            src={mediaBytesUrl(sceneClip.uri)}
                          />
                        </div>
                        {/* Download the assembled scene video — same
                            <a download> anchor idiom the frame tiles use. */}
                        <a
                          className="vi-btn vi-btn-sm"
                          href={mediaBytesUrl(sceneClip.uri)}
                          download={`scene.${extFromMime(sceneClip.mime)}`}
                        >
                          ↓ Download MP4
                        </a>
                      </div>
                    )}
                    <div className="vi-frames-grid vi-scene-frames vi-studio-strip">
                      {sceneFrames.map((f, i) => (
                        <span className="vi-scene-frame-cell" key={f.uri}>
                          <img
                            className="vi-scene-frame-img"
                            src={mediaBytesUrl(f.uri)}
                            alt={`Scene frame ${i + 1}`}
                            loading="lazy"
                          />
                          <span className="vi-frame-tag">#{i + 1}</span>
                          {/* Per-frame download, revealed on hover/focus. */}
                          <a
                            className="vi-frame-dl"
                            href={mediaBytesUrl(f.uri)}
                            download={`scene-frame-${i + 1}.${extFromMime(f.mime)}`}
                            title={`Download frame ${i + 1}`}
                            aria-label={`Download scene frame ${i + 1}`}
                          >
                            ↓
                          </a>
                        </span>
                      ))}
                    </div>
                    <figcaption className="vi-gen-note">
                      {sceneFrames.length} frame{sceneFrames.length === 1 ? "" : "s"}
                      {sceneClip ? " + clip" : ""} added to the library — reuse them
                      below.
                      {project?.dir && (
                        <span className="vi-gen-saved">
                          {" · Saved to "}
                          {project.name ? `${project.name} · ` : ""}
                          <code>{project.dir}</code>
                        </span>
                      )}
                    </figcaption>
                  </figure>
                ) : false && sceneJob.status === "failed" ? (
                  <p className="vi-error" role="alert">
                    {jobError ?? "Scene generation failed."}
                  </p>
                ) : (
                  // Scene fans out one job per prompt; the VIEWER (this central
                  // area) houses every prompt's output, labeled by its prompt. Each
                  // prompt card below keeps only its status/cancel.
                  renderSceneViewer()
                )
              )}
            </div>

          {/* COMPOSER — ordered parts list (Image/Scene), OR the goal timeline
              (Movie). Every part pairs text w/ optional media; every goal is a
              frame interval + prompt + optional reference. */}
          <div className="vi-gen-composer">
            {mode === "movie" ? (
              <div className="vi-movie-timeline">
                {/* k88 — the SHARED prompt-list toolbar (same structure Cinema
                    renders): goal count, select-all, the one-call spread, the
                    negative writer and the standard-negative bulk tick. */}
                <PromptListToolbar
                  count={{
                    id: "vi-gen-goal-count",
                    label: "Goals",
                    value: goals.length,
                    min: 1,
                    onChange: setGoalCount,
                    title:
                      "How many goals the timeline has. Lowering it parks rows from the END of the timeline; raising it brings them back before adding blanks.",
                  }}
                  noun="goal"
                  selectedCount={movieSelectedCount}
                  itemCount={goals.length}
                  allSelected={movieAllSelected}
                  onToggleSelectAll={toggleMovieSelectAll}
                  generateLabel={
                    assist.assistBusy === "spread"
                      ? "✨ Writing the spread…"
                      : `✨ Generate prompts (${movieSelectedCount} selected)`
                  }
                  generateTitle="ONE call writes every ticked goal against one shared world; the unticked ones ride along as locked context so the shots belong to the same movie."
                  generateDisabledTitle="Tick the goals you want written."
                  onGenerate={() => void onMovieSpreadSelected()}
                  negativesLabel={
                    assist.assistBusy === "negative"
                      ? "✨ Writing the negative…"
                      : `✨ Generate negative (${movieSelectedCount} selected)`
                  }
                  negativesTitle="Write an artifact/quality exclusion list for the movie's ONE negative, from the ticked goals' prompts."
                  onNegatives={() => void onMovieNegativesSelected()}
                  assistBusy={assist.assistBusy != null}
                  onStandardNegatives={onMovieStandardNegatives}
                  standardNegativesTitle="Tick [standard negative] on the movie's wired negative (goal #1 — the wire carries one negative per movie)."
                />
                {/* The spread's non-destructive result notice — reports, never edits. */}
                {movieSpreadNotice && movieSpreadNotice.length > 0 && (
                  <div
                    className="vi-input-callout"
                    role="status"
                    style={{ marginBottom: "0.6rem" }}
                  >
                    <ul style={{ margin: 0, paddingLeft: "1.1rem" }}>
                      {movieSpreadNotice.map((line, i) => (
                        <li key={i} style={{ wordBreak: "break-word" }}>
                          {line}
                        </li>
                      ))}
                    </ul>
                    <button
                      type="button"
                      className="vi-btn vi-btn-sm vi-btn-ghost"
                      style={{ marginTop: "0.35rem" }}
                      onClick={() => setMovieSpreadNotice(null)}
                    >
                      ✕ Dismiss
                    </button>
                  </div>
                )}
                <ol className="vi-gen-parts">
                  {goals.map((g, i) => {
                    const r = timeline.rows[i];
                    const rowMsgs: string[] = [];
                    if (r && !r.startOk) rowMsgs.push("start_frame must be an integer ≥ 0");
                    if (r && !r.endOk) rowMsgs.push("end_frame must be an integer > start");
                    if (r && !r.promptOk) rowMsgs.push("prompt required");
                    const contigMsg = timeline.flags[i];
                    if (contigMsg) rowMsgs.push(contigMsg);
                    return (
                      <PromptCard
                        key={g.id}
                        badge="goal"
                        index={i}
                        noun="goal"
                        select={{
                          checked: movieSelectedIds.has(g.id),
                          onChange: () => toggleMovieSelected(g.id),
                          title:
                            "Tick to let the group actions rewrite this goal. Unticked rows are held as locked context.",
                          label: `Select goal ${i + 1} for group prompt assist`,
                        }}
                        canMoveUp={i > 0}
                        canMoveDown={i < goals.length - 1}
                        onMoveUp={() => moveGoal(g.id, -1)}
                        onMoveDown={() => moveGoal(g.id, 1)}
                        moveUpTitle="Move up (re-tiles the timeline)"
                        moveDownTitle="Move down (re-tiles the timeline)"
                        onSplit={() => splitGoal(g.id)}
                        splitTitle="Split this goal at its midpoint"
                        canRemove={goals.length > 1}
                        onRemove={() => removeGoal(g.id)}
                        removeTitle="Remove"
                      >
                        {/* k92 — [Prompt][Negative] tabs over ONE GoalComposer pane.
                            When "share negative" is ON (default) the wire carries ONE
                            movie negative, edited on goal #1 (later rows disabled with
                            the reason). Untick it (in the goal's settings) and EACH goal
                            edits its OWN negative — the wire now grew a per-goal field. */}
                        <PromptFieldTabs
                          idPrefix={`vi-gen-goal-${g.id}`}
                          promptLabel="prompt"
                          prompt={g.prompt}
                          onPromptChange={(v) => editGoal(g.id, "prompt", v)}
                          promptPlaceholder="Describe what this interval should depict…"
                          rows={2}
                          negative={movieShare.negative ? negative : g.negative ?? ""}
                          onNegativeChange={
                            movieShare.negative
                              ? i === 0
                                ? setNegative
                                : () => undefined
                              : (v) => patchGoal(g.id, { negative: v })
                          }
                          negativePlaceholder="what to avoid — blank = none"
                          negativeHint={
                            movieShare.negative && i !== 0
                              ? undefined
                              : "Blank = no negative prompt."
                          }
                          stdNegative={
                            movieShare.negative
                              ? goals[0]?.stdNegative ?? false
                              : g.stdNegative ?? false
                          }
                          onStdNegativeChange={
                            movieShare.negative
                              ? (v) => {
                                  if (i !== 0) return;
                                  setGoals((prev) =>
                                    prev.map((row, j) =>
                                      j === 0 ? { ...row, stdNegative: v } : row,
                                    ),
                                  );
                                }
                              : (v) => patchGoal(g.id, { stdNegative: v })
                          }
                          negativeDisabledNote={
                            movieShare.negative && i !== 0
                              ? "Shared negative — edit it on goal #1, or untick “share negative” in this goal's settings for a per-goal one."
                              : undefined
                          }
                        />

                        {/* PROMPT ASSIST for THIS goal — Enhance enriches this goal's
                            prompt, Generate writes a fresh one (seeded by any current
                            text). The SAME shared cluster + hook the Scene composer and
                            the Cinema per-segment timeline use; Movie is a per-goal
                            timeline, so — exactly like Cinema — the cluster attaches per
                            goal, wired to that goal's own prompt. One in-flight assist at
                            a time across all goals; errors surface inline and never
                            clobber the goal's existing prompt. Enqueue is untouched. */}
                        <PromptAssistButtons
                          busy={assist.assistBusy}
                          error={assist.assistError}
                          onDismissError={assist.dismissError}
                          onEnhance={() =>
                            void assist.runAssist("detail", g.prompt, (p) =>
                              editGoal(g.id, "prompt", p),
                            )
                          }
                          onGenerate={() =>
                            void (async () => {
                              let fresh = "";
                              await assist.runAssist("generate", g.prompt, (p) => {
                                fresh = p;
                                editGoal(g.id, "prompt", p);
                              });
                              if (!fresh.trim()) return;
                              // NEGATIVE RIDES ALONG (operator 2026-08-13): the
                              // generated exclusion list APPENDS to the goal's (or
                              // shared) typed negative, never clobbers it.
                              await assist.runNegative({ subject: fresh }, (neg) => {
                                const cur = (movieShare.negative ? negative : g.negative ?? "")
                                  .trim().replace(/,\s*$/, "");
                                const merged = cur ? `${cur}, ${neg}` : neg;
                                if (movieShare.negative) setNegative(merged);
                                else patchGoal(g.id, { negative: merged });
                              });
                              const t = projectName.trim();
                              if (t === "" || t === autoProjectRef.current) {
                                const derived = deriveNameFrom(fresh);
                                if (derived) {
                                  autoProjectRef.current = derived;
                                  setProjectName(derived);
                                }
                              }
                            })()
                          }
                          canEnhance={g.prompt.trim() !== ""}
                          models={assist.assistModels}
                          model={assist.assistModel}
                          onModelChange={assist.setAssistModel}
                        />

                        {/* (The first-goal-only negative textarea moved into the
                            [Prompt][Negative] tabs above — k88.) */}

                        {g.ref ? (
                          <div className="vi-gen-part-media">
                            <img
                              className="vi-gen-part-thumb"
                              src={mediaBytesUrl(g.ref.ref.uri)}
                              alt="Goal reference image"
                              loading="lazy"
                            />
                            <span className="vi-gen-part-meta">
                              <code>{g.ref.ref.mime}</code>
                              {g.ref.origin ? ` · ${g.ref.origin}` : ""}
                              {g.ref.label ? ` · ${g.ref.label}` : ""}
                            </span>
                            <button
                              type="button"
                              className="vi-btn vi-btn-sm vi-btn-ghost"
                              onClick={() => detachGoalRef(g.id)}
                              aria-label="Remove reference image"
                              title="Remove reference image"
                            >
                              ✕ reference
                            </button>
                          </div>
                        ) : (
                          <div className="vi-gen-part-attach">
                            <button
                              type="button"
                              className="vi-btn vi-btn-sm vi-btn-ghost"
                              onClick={() => openGoalRefPicker(g.id)}
                            >
                              + add reference image
                            </button>
                          </div>
                        )}

                        {/* k92 — the settings expander now holds the goal's FULL
                            per-component knobs (model + size/steps/guidance/seed +
                            strength/motion/chain + the negative-share toggle), exactly
                            like a Scene part, ABOVE its frame-window (start/end + span).
                            fps/assemble/project stay movie-level (one clip). Validation
                            flags stay OUTSIDE it below, so a bad window is visible even
                            while the expander is closed. */}
                        <PromptCardSettings label={`settings — frames [${g.start}, ${g.end})`}>
                          {renderGoalKnobs(g)}
                          <div className="vi-knob-window vi-movie-window">
                            <label className="vi-knob-sub">
                              start_frame
                              <input
                                type="number"
                                min="0"
                                step="1"
                                inputMode="numeric"
                                className="vi-knob-input vi-knob-input-sm"
                                value={g.start}
                                onChange={(e) => editGoal(g.id, "start", e.target.value)}
                              />
                            </label>
                            <label className="vi-knob-sub">
                              end_frame
                              <input
                                type="number"
                                min="0"
                                step="1"
                                inputMode="numeric"
                                className="vi-knob-input vi-knob-input-sm"
                                value={g.end}
                                onChange={(e) => editGoal(g.id, "end", e.target.value)}
                              />
                            </label>
                            <span className="vi-movie-window-span">
                              [{g.start}, {g.end}) ·{" "}
                              {(() => {
                                const s = toNumberOrNull(g.start);
                                const e = toNumberOrNull(g.end);
                                return s != null && e != null && e > s ? e - s : "—";
                              })()}{" "}
                              frames
                            </span>
                          </div>
                        </PromptCardSettings>

                        {rowMsgs.length > 0 && (
                          <span className="vi-knob-flag">{rowMsgs.join(" · ")}</span>
                        )}
                      </PromptCard>
                    );
                  })}
                </ol>

                {/* k92 — GROUP settings for the ticked goals (mirror of the Scene
                    composer's group strip): edit once, apply to every selected goal. */}
                {renderMovieGroupSettings()}

                <div className="vi-movie-timeline-foot">
                  <button type="button" className="vi-btn" onClick={addGoal}>
                    + Add goal
                  </button>
                  {/* k92 TEST OPTIONS — the SAME reusable architecture Scene uses:
                      "test all models" / "per preset" APPEND one goal (prompt
                      component) per model / preset, each with its own model, so the
                      movie's segments become a model/preset comparison reel. */}
                  <button
                    type="button"
                    className="vi-btn vi-btn-ghost"
                    onClick={addGoalPerModel}
                    disabled={goalBasePrompt().trim() === "" || modelsLoading}
                    title="Append one goal per available model (each pinned to that model) — a cross-model comparison reel. Seeded from the last non-empty goal's prompt."
                  >
                    ✚ per model — test all
                  </button>
                  <button
                    type="button"
                    className="vi-btn vi-btn-ghost"
                    onClick={addGoalPerPreset}
                    disabled={goalBasePrompt().trim() === "" || presets.length === 0}
                    title="Append one goal per preset (each carrying the preset's model + knobs). Seeded from the last non-empty goal's prompt."
                  >
                    ✚ per preset
                  </button>
                  <span className="vi-movie-summary">
                    {timeline.total} frames total
                    {(() => {
                      const clip =
                        fpsNum && fpsNum > 0 ? timeline.total / fpsNum : null;
                      return clip != null
                        ? ` · ~${clip.toFixed(1)}s @ ${fps} fps`
                        : "";
                    })()}
                  </span>
                </div>

                {!timelineValid && (
                  <span className="vi-knob-flag">
                    The timeline must tile [0, total) contiguously from frame 0 —
                    every goal needs end &gt; start, a non-empty prompt, and no
                    gaps/overlaps.
                  </span>
                )}
              </div>
            ) : (
              <>
                {/* The station-wide "whole assembled prompt" Enhance/Generate cluster
                    was removed (operator): per-part Enhance/Generate on each card and
                    the toolbar's group "Generate prompts (N selected)" below cover it. */}
                {/* k89/k93 — the SHARED prompt-list toolbar in its GROUP layout:
                      prompt assist model  [select]
                      [start] - [end]  [ ] select all   negatives: [on/off]  [ ] std.
                      (✨ generate (N selected)) (✨ enhance (M selected)) [+ video]
                    The selection SET (sceneSelectedIds) stays the source of truth;
                    range / select-all / per-card ticks all write into it. */}
                <PromptListToolbar
                  count={{
                    id: "vi-gen-part-count",
                    label: "Parts",
                    value: parts.length,
                    min: 1,
                    max: MAX_SCENE_PARTS,
                    onChange: setScenePartCount,
                    title: `How many parts the prompt has (1–${MAX_SCENE_PARTS}). Lowering it removes TEXT parts from the END — media parts are never auto-removed.`,
                  }}
                  noun="part"
                  selectedCount={sceneSelectedCount}
                  itemCount={sceneTextParts.length}
                  allSelected={sceneAllSelected}
                  onToggleSelectAll={toggleSceneSelectAll}
                  modelRow={
                    /* PROMPT-ASSIST MODEL — ONE shared selector for which text
                       generator the group actions (and every card's default) use. */
                    assist.assistModels.length > 0 ? (
                      <div className="vi-prompt-assist vi-gen-assist-model">
                        <span className="vi-prompt-assist-lead" aria-hidden="true">
                          prompt assist model
                        </span>
                        <select
                          className="vi-select vi-select-sm"
                          value={assist.assistModel ?? ""}
                          disabled={assist.assistBusy != null}
                          onChange={(e) => assist.setAssistModel(e.target.value || null)}
                          aria-label="Text generator used to enhance or generate prompts"
                          title="Which text generator writes prompts for the group actions and every prompt's Enhance / Generate (a card can pick its own)."
                        >
                          <option value="">fleet default</option>
                          {assist.assistModels.map((m) => (
                            <option key={m.model} value={m.model}>
                              {m.model}
                              {m.state === "serving" || m.state === "loaded" ? ""
                : m.state === "hot" ? " · hot (loads)" : " · cold (downloads)"}
                            </option>
                          ))}
                        </select>
                      </div>
                    ) : null
                  }
                  range={{
                    start: sceneRangeStart,
                    end: sceneRangeEnd,
                    onChange: selectSceneRange,
                  }}
                  negativesOn={sceneGroupNegatives}
                  onNegativesOnChange={setSceneGroupNegatives}
                  negativesOnTitle="On: generate / enhance also write each selected part's Negative field (visible; sent as-is). Off: prompts only."
                  stdChecked={sceneStdAllOn}
                  onStdChange={onSceneStdChange}
                  stdTitle="Tick: insert the standard exclusion set into every selected part's Negative field. Untick: remove exactly those terms. What you see in the field is what is sent."
                  generateLabel={
                    assist.assistBusy === "spread"
                      ? "✨ writing the spread…"
                      : `✨ generate (${sceneSelectedCount} selected)`
                  }
                  generateTitle="ONE call writes every selected text part against one shared world; unselected parts and media captions ride along as locked context. Part #1's Negative field is sent as the shared negative."
                  generateDisabledTitle="Select the text parts you want written (range, select all, or the per-card ticks)."
                  onGenerate={() => void onSceneSpreadSelected()}
                  enhanceLabel={
                    assist.assistBusy === "detail"
                      ? "✨ enhancing…"
                      : `✨ enhance (${sceneEnhanceCount} selected)`
                  }
                  enhanceTitle={
                    sceneEnhanceCount === 0
                      ? "Enhance needs selected parts with a non-blank prompt."
                      : "Enrich each selected non-blank prompt with more descriptive detail (one part at a time)."
                  }
                  canEnhance={sceneEnhanceCount > 0}
                  onEnhance={() => void onSceneEnhanceSelected()}
                  // Legacy single-line props (unused in the group layout, kept typed).
                  negativesLabel=""
                  onNegatives={() => undefined}
                  onStandardNegatives={() => undefined}
                  assistBusy={assist.assistBusy != null}
                  actionsExtra={
                    <button
                      type="button"
                      className={`vi-btn vi-btn-sm vi-btn-ghost${scenePretext ? " vi-btn-on" : ""}`}
                      onClick={() => setScenePretextOpen((v) => !v)}
                      aria-expanded={scenePretextOpen}
                      title="Attach ONE video as pretext: it is analyzed server-side and its description is fed to generate / enhance as context (not a prompt part)."
                    >
                      {scenePretext ? "▶ video pretext ✓" : "+ video"}
                    </button>
                  }
                  below={
                    (scenePretextOpen || scenePretext) && (
                      <div className="vi-gen-inline-attach vi-gen-pretext">
                        {scenePretext ? (
                          <div className="vi-gen-part-media">
                            <span className="vi-gen-part-videoicon" aria-hidden>
                              ▶
                            </span>
                            <span className="vi-gen-part-meta">
                              <code>{scenePretext.ref.mime}</code>
                              {scenePretext.origin ? ` · ${scenePretext.origin}` : ""}
                              {scenePretext.label ? ` · ${scenePretext.label}` : ""}
                              {" · described server-side as pretext for generate / enhance"}
                            </span>
                            <button
                              type="button"
                              className="vi-btn vi-btn-sm vi-btn-ghost"
                              onClick={() => {
                                setScenePretext(null);
                                setScenePretextOpen(false);
                              }}
                              aria-label="Remove the pretext video"
                              title="Remove the pretext video"
                            >
                              ✕
                            </button>
                          </div>
                        ) : (
                          renderVideoPickerBody((it) => {
                            setScenePretext(it);
                            setScenePretextOpen(false);
                          })
                        )}
                      </div>
                    )
                  }
                />
                {/* GROUP SETTINGS — the settings counterpart to the group prompt
                    assist: whichever prompt components are selected get these
                    settings (applied as their own per-part overrides). */}
                {renderGroupSettings()}
                {/* BULK-ADD — "test all models" and "per preset" as COMPONENTS (the
                    single architecture: each added component is an independent prompt
                    that fans out to its own output). Replaces the old tester rows. */}
                <div className="vi-gen-bulk-actions">
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm"
                    onClick={addPartPerModel}
                    disabled={
                      sceneAnyRunning ||
                      basePromptText().trim() === "" ||
                      models.filter((m) => !m.disabled).length === 0
                    }
                    title="Add one prompt component per available model — same prompt, model varied. Run fans them out to one output each in the viewer. (This is 'test all models'.)"
                  >
                    ✚ per model — test all
                  </button>
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm"
                    onClick={addPartPerPreset}
                    disabled={
                      sceneAnyRunning ||
                      basePromptText().trim() === "" ||
                      presets.length === 0
                    }
                    title="Add one prompt component per curated preset (its model + knob defaults)."
                  >
                    ✚ per preset
                  </button>
                  {/* The server-side cross-model BATTERY (studio_tester) kept as a
                      consistent SELECTION alongside the component scaffolds. Unlike
                      the per-model scaffold (N local jobs, one component + output
                      each), this runs one central sweep that records a battery to the
                      generation log — a caveat, offered as an explicit option. */}
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm vi-btn-ghost"
                    onClick={() => void runStudioTesterSweep()}
                    disabled={studioTesterBusy || basePromptText().trim() === ""}
                    title="Server-side battery: sweep this prompt across every model of this type in ONE central run, recording results to the generation log (does not scaffold components)."
                  >
                    {studioTesterBusy ? "🧪 starting…" : "🧪 battery (server sweep)"}
                  </button>
                </div>
                {studioTesterMsg && (
                  <p className="vi-input-callout" role="status">
                    {studioTesterMsg}
                  </p>
                )}
                {/* The spread's non-destructive result notice — reports, never edits. */}
                {sceneSpreadNotice && sceneSpreadNotice.length > 0 && (
                  <div
                    className="vi-input-callout"
                    role="status"
                    style={{ marginBottom: "0.6rem" }}
                  >
                    <ul style={{ margin: 0, paddingLeft: "1.1rem" }}>
                      {sceneSpreadNotice.map((line, i) => (
                        <li key={i} style={{ wordBreak: "break-word" }}>
                          {line}
                        </li>
                      ))}
                    </ul>
                    <button
                      type="button"
                      className="vi-btn vi-btn-sm vi-btn-ghost"
                      style={{ marginTop: "0.35rem" }}
                      onClick={() => setSceneSpreadNotice(null)}
                    >
                      ✕ Dismiss
                    </button>
                  </div>
                )}
                {/* (The old empty-composer prose moved into the sidebar "About
                    Scene" expander — an empty list just shows the add bar below.) */}
                {parts.length > 0 && (
              <ol className="vi-gen-parts">
                {parts.map((p, i) => {
                  const kind = partKind(p);
                  return (
                    <PromptCard
                      key={p.key}
                      badge={kind}
                      index={i}
                      noun="part"
                      select={{
                        checked: sceneSelectedIds.has(p.key),
                        onChange: () => toggleSceneSelected(p.key),
                        disabled: !!p.media,
                        title: p.media
                          ? "Media parts ride the group actions as locked context — prompt-writing targets text parts."
                          : "Tick to let the group actions rewrite this part. Unticked parts are held as locked context.",
                        label: `Select part ${i + 1} for group prompt assist`,
                      }}
                      canMoveUp={i > 0}
                      canMoveDown={i < parts.length - 1}
                      onMoveUp={() => movePart(p.key, -1)}
                      onMoveDown={() => movePart(p.key, 1)}
                      moveUpTitle="Move up"
                      moveDownTitle="Move down"
                      canRemove={
                        !sceneRowJobs.byRow[p.key]?.running &&
                        !sceneRowJobs.byRow[p.key]?.capped
                      }
                      onRemove={() => removePart(p.key)}
                      removeTitle={
                        sceneRowJobs.byRow[p.key]?.running ||
                        sceneRowJobs.byRow[p.key]?.capped
                          ? "This prompt's job is still active — cancel or let it finish before removing"
                          : "Remove"
                      }
                    >
                      {/* [Prompt][Negative] tabs over ONE GoalComposer pane (labels
                          visually hidden — the tabs say it). k93: EVERY part sends
                          exactly what its own Negative field holds; blank is blank.
                          Part #1's field is also the group assist's shared negative
                          (its tab tooltip says so). The std. tick lives in the
                          assist row below and reads its state from the field. */}
                      <PromptFieldTabs
                        idPrefix={`vi-gen-part-${p.key}`}
                        hideLabels
                        hideStdTick
                        promptLabel={
                          p.media
                            ? `caption (optional — sent as text just before the ${kind})`
                            : "prompt"
                        }
                        prompt={p.text}
                        onPromptChange={(v) => editText(p.key, v)}
                        promptPlaceholder={
                          p.media
                            ? `Caption for this ${kind}…`
                            : "Describe what to generate…"
                        }
                        rows={2}
                        negativeLabel="negative prompt"
                        negative={p.negative ?? ""}
                        onNegativeChange={(v) => patchPart(p.key, { negative: v })}
                        negativePlaceholder="what to avoid (sent exactly as written; blank = none)"
                        negativeTabTitle={
                          !p.media && sceneTextParts[0]?.key === p.key
                            ? "This part's negative. Part #1's Negative field is ALSO sent as the shared negative of the group generate / enhance — explicitly, never merged."
                            : "This part's negative — sent exactly as written with this part's request."
                        }
                        stdNegative={hasStandardNegative(p.negative ?? "")}
                        onStdNegativeChange={(v) => onPartStdChange(p, v)}
                        negativeDisabledNote={undefined}
                        negativeExtras={undefined}
                      />

                      {/* k93 §B — the per-card assist row:
                            prompt assist model [select]
                            negatives: [on/off]  [ ] std.   (✨ generate) (✨ enhance)
                          Text parts only — media parts carry captions and ride the
                          group actions as locked context. One in-flight assist at a
                          time across all parts. */}
                      {!p.media && (
                        <div className="vi-gen-part-assist-row">
                          <label
                            className="vi-comfy-hint vi-prompt-toolbar-neg"
                            title="On: this card's generate / enhance also write its Negative field. Off: the prompt only."
                          >
                            negatives:
                            <button
                              type="button"
                              className={`vi-btn vi-btn-sm vi-toggle${(p.assistNegatives ?? true) ? " vi-toggle-on" : ""}`}
                              role="switch"
                              aria-checked={p.assistNegatives ?? true}
                              onClick={() =>
                                patchPart(p.key, { assistNegatives: !(p.assistNegatives ?? true) })
                              }
                            >
                              {(p.assistNegatives ?? true) ? "on" : "off"}
                            </button>
                          </label>
                          <label
                            className="vi-comfy-hint vi-prompt-toolbar-std"
                            title="Tick: insert the standard exclusion set into this part's Negative field. Untick: remove exactly those terms. What you see is what is sent."
                          >
                            <input
                              type="checkbox"
                              checked={hasStandardNegative(p.negative ?? "")}
                              onChange={(e) => onPartStdChange(p, e.target.checked)}
                              aria-label={`Insert the standard negative into part ${i + 1}`}
                            />
                            std.
                          </label>
                          <PromptAssistButtons
                            busy={assist.assistBusy}
                            error={assist.assistError}
                            onDismissError={assist.dismissError}
                            onEnhance={() => void runPartAssist(p, "detail")}
                            onGenerate={() => void runPartAssist(p, "generate")}
                            canEnhance={p.text.trim() !== ""}
                            models={assist.assistModels}
                            model={p.assistModel ?? assist.assistModel}
                            onModelChange={(m) =>
                              patchPart(p.key, { assistModel: m })
                            }
                          />
                        </div>
                      )}

                      {p.media ? (
                        <div className="vi-gen-part-media-block">
                          <div className="vi-gen-part-media">
                            {kind === "image" ? (
                              <img
                                className="vi-gen-part-thumb"
                                src={mediaBytesUrl(p.media.ref.uri)}
                                alt="Prompt image part"
                                loading="lazy"
                              />
                            ) : (
                              <span className="vi-gen-part-videoicon" aria-hidden>
                                ▶
                              </span>
                            )}
                            <span className="vi-gen-part-meta">
                              <code>{p.media.ref.mime}</code>
                              {p.media.origin ? ` · ${p.media.origin}` : ""}
                              {p.media.label ? ` · ${p.media.label}` : ""}
                              {kind === "video"
                                ? " · resolved to frames server-side"
                                : ""}
                            </span>
                            <button
                              type="button"
                              className="vi-btn vi-btn-sm vi-btn-ghost"
                              onClick={() => detachMedia(p.key)}
                              aria-label="Detach media"
                              title="Detach media (keeps the text)"
                            >
                              ✕ media
                            </button>
                          </div>
                          {/* k93 §B.6 — pull ONE frame out of the attached video
                              (frame-extract job → the frame becomes this part's
                              image / start frame AND lands in the library). */}
                          {kind === "video" && (
                            <div className="vi-gen-part-pullframe">
                              <label className="vi-comfy-hint" htmlFor={`vi-gen-pull-${p.key}`}>
                                pull frame @
                              </label>
                              <input
                                id={`vi-gen-pull-${p.key}`}
                                type="number"
                                className="vi-knob-input vi-gen-pull-input"
                                min={0}
                                step={0.1}
                                max={p.media.ref.duration_s ?? undefined}
                                value={pullFrameAt[p.key] ?? "0"}
                                onChange={(e) =>
                                  setPullFrameAt((prev) => ({ ...prev, [p.key]: e.target.value }))
                                }
                                aria-label="Time in seconds of the frame to pull"
                              />
                              <span className="vi-comfy-hint">
                                s{p.media.ref.duration_s != null ? ` / ${p.media.ref.duration_s.toFixed(1)}` : ""}
                              </span>
                              <button
                                type="button"
                                className="vi-btn vi-btn-sm"
                                disabled={pullFrameJobs[p.key] != null}
                                onClick={() => pullFrame(p)}
                                title="Extract the frame at this time (frame-extract job). The still becomes this part's start image and is added to the library."
                              >
                                {pullFrameJobs[p.key] != null ? "pulling…" : "pull frame"}
                              </button>
                              {pullFrameError[p.key] && (
                                <span className="vi-error">{pullFrameError[p.key]}</span>
                              )}
                            </div>
                          )}
                          {p.frame && (
                            <div className="vi-gen-part-media">
                              <img
                                className="vi-gen-part-thumb"
                                src={mediaBytesUrl(p.frame.ref.uri)}
                                alt="Frame pulled from the attached video"
                                loading="lazy"
                              />
                              <span className="vi-gen-part-meta">
                                <code>{p.frame.ref.mime}</code>
                                {p.frame.label ? ` · ${p.frame.label}` : ""}
                                {" · sent as this part's start image (before the video)"}
                              </span>
                              <button
                                type="button"
                                className="vi-btn vi-btn-sm vi-btn-ghost"
                                onClick={() => detachFrame(p.key)}
                                aria-label="Drop the pulled frame"
                                title="Drop the pulled frame (keeps the video)"
                              >
                                ✕ frame
                              </button>
                            </div>
                          )}
                        </div>
                      ) : (
                        <>
                          {/* k93 §B.5 — the attach row; each button EXPANDS its
                              picker inline right under this row. */}
                          <div className="vi-gen-part-attach">
                            <button
                              type="button"
                              className={`vi-btn vi-btn-sm vi-btn-ghost${partAttachOpen?.key === p.key && partAttachOpen.kind === "image" ? " vi-btn-on" : ""}`}
                              onClick={() => openAttachPicker(p.key, "image")}
                              aria-expanded={partAttachOpen?.key === p.key && partAttachOpen.kind === "image"}
                              title="Pick an image that pairs with this part's text"
                            >
                              + image
                            </button>
                            <button
                              type="button"
                              className={`vi-btn vi-btn-sm vi-btn-ghost${partAttachOpen?.key === p.key && partAttachOpen.kind === "video" ? " vi-btn-on" : ""}`}
                              onClick={() => openAttachPicker(p.key, "video")}
                              aria-expanded={partAttachOpen?.key === p.key && partAttachOpen.kind === "video"}
                              title="Pick a video that pairs with this part's text (a frame can then be pulled from it)"
                            >
                              + video
                            </button>
                            <button
                              type="button"
                              className={`vi-btn vi-btn-sm vi-btn-ghost${partAttachOpen?.key === p.key && partAttachOpen.kind === "library" ? " vi-btn-on" : ""}`}
                              onClick={() => openAttachPicker(p.key, "library")}
                              aria-expanded={partAttachOpen?.key === p.key && partAttachOpen.kind === "library"}
                              title="Append a media part from the library (a new part, not paired with this one)"
                            >
                              + library
                            </button>
                          </div>
                          {partAttachOpen?.key === p.key && (
                            <div className="vi-gen-inline-attach">
                              {partAttachOpen.kind === "image" && (
                                <>
                                  <p className="vi-crop-hint">
                                    The picked image pairs with this part&apos;s text.
                                  </p>
                                  {renderImagePickerBody(placeMedia)}
                                </>
                              )}
                              {partAttachOpen.kind === "video" && (
                                <>
                                  <p className="vi-crop-hint">
                                    The picked video pairs with this part&apos;s text; it is
                                    resolved into frames server-side, or pull one frame
                                    from it afterwards.
                                  </p>
                                  {renderVideoPickerBody(placeMedia)}
                                </>
                              )}
                              {partAttachOpen.kind === "library" && (
                                <>
                                  <p className="vi-crop-hint">
                                    Appends a NEW media part after the last one.
                                  </p>
                                  {renderLibraryPickerBody((m) => {
                                    setAttachTarget(null);
                                    placeMedia(m);
                                  })}
                                </>
                              )}
                            </div>
                          )}
                        </>
                      )}

                      {/* IDENTITY — the generation-wide character, selectable from
                          ANY part because ONE identity is shared by the whole
                          generation. k93 §B.7: rendered as a single `+ identity`
                          button that expands inline (same row idiom); shows bound
                          when one is set. */}
                      <div className="vi-gen-part-attach">
                        <button
                          type="button"
                          className={`vi-btn vi-btn-sm vi-btn-ghost${identityOpen.has(p.key) || boundIdentity ? " vi-btn-on" : ""}`}
                          aria-expanded={identityOpen.has(p.key)}
                          onClick={() =>
                            setIdentityOpen((prev) => {
                              const next = new Set(prev);
                              if (next.has(p.key)) next.delete(p.key);
                              else next.add(p.key);
                              return next;
                            })
                          }
                          title="Bind an identity profile (one per generation — shared by every part)"
                        >
                          {boundIdentity ? `+ identity ✓ ${boundIdentity.name}` : "+ identity"}
                        </button>
                      </div>
                      {identityOpen.has(p.key) && (
                        <div className="vi-gen-inline-attach">
                          <GenIdentityBar
                            profiles={identityProfiles}
                            loading={identityLoading}
                            error={identityError}
                            bound={boundIdentity}
                            onBind={bindIdentity}
                            onClear={clearIdentity}
                            disabled={running}
                          />
                        </div>
                      )}

                      {renderPartSettings(p)}
                      {renderPartResult(p)}
                    </PromptCard>
                  );
                })}
              </ol>
                )}

            {/* SETTINGS STRIP — the GENERATION SETTINGS, absorbed into the prompt
                component (operator ask 2026-08-04, k63). These used to stack down the
                left Settings rail, one full-width block each, at arm's length from the
                thing they describe; the operator's read is that they ARE part of the
                prompt component, so they render here, attached to the parts they
                govern, in ONE condensed line. Same relocation the negative prompt made
                on 07-13, same rule: state stays in the station (these inputs drive the
                very same setters the rail did), only the rendering moved.

                Image AND Scene both get it — the strip is the shared generation
                contract, and scene's own knobs (frames/motion/fps/assemble/strength/
                chain) stay in the rail because they describe the SCENE, not the frame.

                The ticks decide what a component seeded from this one receives; in
                scene mode generation itself just reads the values as it always has.

                k89: the strip stays STATION-WIDE (it configures the single
                generation, not one part) but folds into a collapsed "Settings"
                expander below the list — the shared PromptCardSettings idiom.
                Collapsed by default; no expanded-on-first-visit special case, since
                that would need a persistence hack (<details> keeps its open flag in
                the DOM only for the mount). */}
            {/* Superseded by the COMPLETE per-prompt settings on each card
                (renderPartSettings). Kept only as a shared-defaults editor when
                there are no prompt components yet. */}
            {parts.length === 0 && (
            <PromptCardSettings label="Settings">
            <CondensedKnobStrip
              idPrefix="vi-gen"
              values={{ width, height, steps, guidance, seed }}
              onPatch={(p) => {
                if (p.width !== undefined) setWidth(p.width);
                if (p.height !== undefined) setHeight(p.height);
                if (p.steps !== undefined) setSteps(p.steps);
                if (p.guidance !== undefined) setGuidance(p.guidance);
                if (p.seed !== undefined) setSeed(p.seed);
              }}
              carry={{
                model: sceneShare.model,
                size: sceneShare.size,
                steps: sceneShare.steps,
                guidance: sceneShare.guidance,
                seed: sceneShare.seed,
                negative: false,
              }}
              onCarry={(k, v) => setSharedKey(k as SceneShareKey, v)}
              lead={
                <span className="vi-knob-mini-group vi-knob-mini-model">
                  <CarryTick
                    id="vi-gen-carry-model"
                    setting="the model"
                    checked={sceneShare.model}
                    onChange={(v) => setSharedKey("model", v)}
                  />
                  <label className="vi-knob-mini-label" htmlFor="vi-gen-model">
                    model
                  </label>
                  <select
                    id="vi-gen-model"
                    className="vi-knob-select vi-knob-select-mini"
                    value={modelId}
                    onChange={(e) => setModelId(e.target.value)}
                    disabled={modelsLoading}
                  >
                    {modelsLoading && <option value="">Loading models…</option>}
                    {!modelsLoading && models.every((m) => m.disabled) && (
                      <option value="">No image models available</option>
                    )}
                    {/* Greyed rows are adapters / pipeline components / unclassified
                        models: present on disk, but not generation targets. Disabled,
                        with the backend's own reason as the tooltip (k61). */}
                    {models.map((m) => (
                      <option key={m.id} value={m.id} disabled={m.disabled} title={m.reason}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm vi-btn-ghost"
                    onClick={refreshModels}
                    disabled={modelsLoading}
                    title="Re-scan the registry for image models adopted this session"
                  >
                    ↻
                  </button>
                </span>
              }
              footer={
                <>
                  {/* CAPABILITY LINE — what THIS model needs, from its tasks[].
                      KEPT here (k89): it is model-dependent point-of-use state, not
                      static prose. The static "first image part is the start frame"
                      hint moved into the sidebar "About Scene" expander. */}
                  <span className="vi-knob-hint vi-knob-range">
                    {capabilityHint(req.image)}
                  </span>
                  {modelsError && <span className="vi-error">{modelsError}</span>}
                </>
              }
            />
            </PromptCardSettings>
            )}

            {/* ADD-PART BAR — the composer's own append affordance, for BOTH image
                and scene. Docked under the parts list exactly like Movie's
                "+ Add goal" sits under its timeline (vi-movie-timeline-foot), so the
                three modes now share one idiom: the list you are editing owns the
                verb that extends it.

                WHY THIS EXISTS (operator 2026-07-16: "scene in generate is lacking a
                button to add a prompt"). The 2026-07-13 de-duplication suppressed the
                TOP add-bar for image + scene on the rationale that the composer below
                already carried these affordances. That rationale held for MEDIA (each
                part card has "from library" / "+ attach image" / "+ attach video",
                and the requirement banner has its start-image button) — but NOT for
                TEXT: `addText` was left with its only call site inside the gated bar,
                so after 07-13 NO mode could add a text part. Scene, which has no other
                text affordance at all, was capped at the single part it mounts with;
                and either mode could reach the `vi-gen-empty` state by removing its
                last part, with no way back short of a page reload.

                So the 07-13 gate is CORRECT and stays (it removed a genuinely
                duplicate TOP bar, and remains reachable for any future mode that
                renders the addFile slot) — the real defect was that the composer never
                grew the replacement verb it was credited with. This is that verb, in
                the composer, where 07-13 said it should live. Image is not regressed:
                it gains the text-append it silently lost and keeps every existing
                media affordance; nothing is duplicated, because the top bar renders
                nowhere for these modes.

                Per the operator's "the component for prompting should be the single
                image generation": scene reuses this SAME composer + part cards +
                prompt-assist that single-image generation uses — it is literally the
                same JSX for both modes (this branch is not mode-gated), so they cannot
                drift apart. */}
            <div className="vi-gen-add-bar vi-gen-parts-foot">
              <button
                type="button"
                className="vi-btn"
                onClick={addText}
                title="Append a text part to the prompt (parts are sent in order)"
              >
                + Add text{parts.length === 0 ? " (required)" : ""}
              </button>
              {/* A pure text-to-image model can't consume a start image, so its
                  image-append affordance is omitted (req.image === "none") — the same
                  capability rule the suppressed top bar used. */}
              {req.image !== "none" && (
                <button
                  type="button"
                  className="vi-btn"
                  onClick={() => {
                    setAttachTarget(null); // append mode, not attach
                    setShowImagePicker((v) => !v);
                    setShowVideoPicker(false);
                  }}
                  title="Append an image part from the library"
                >
                  {startImageRequired && !hasStartImage
                    ? "+ Add start image from library"
                    : "+ Add image from library"}
                </button>
              )}
            </div>
              </>
            )}

          </div>

          {/* The separate image-tester / studio-sweep surface was removed:
              the prompt COMPONENT is the single reusable unit (each part has its
              own model/settings/negative/output). "Per model" / "Per preset" /
              "Test all models" now add prompt components (see addPartPerModel /
              addPartPerPreset) that fan out to one output each in the viewer. */}

          {/* FOOTER — Run action + feedback, docked INSIDE the components panel
              (no floating buttons outside the panels). */}
          <div className="vi-station-footer">
            <div className="vi-station-footer-main">
              {mode !== "movie" && !hasText && (
                <span className="vi-knob-flag">
                  Add at least one non-empty text part to enable Run.
                </span>
              )}
              {mode !== "movie" && startImageRequired && !hasStartImage && (
                <span className="vi-knob-flag">
                  This edit model needs a start image — add one to enable Run.
                </span>
              )}
              {mode === "movie" && !timelineValid && (
                <span className="vi-knob-flag">
                  Complete the goal timeline (contiguous from 0, every goal a
                  non-empty prompt) to enable Run.
                </span>
              )}
              {jobError && (
                <p className="vi-error" role="alert">
                  {jobError}
                </p>
              )}
              {/* (studioTesterMsg moved into the Tester collapsibles above — k89.) */}
              {capped && (
                <div className="vi-frames-run">
                  <span className="vi-knob-flag">{STILL_RUNNING_MESSAGE}</span>
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm"
                    onClick={resume}
                  >
                    Check again
                  </button>
                </div>
              )}
            </div>
            {nameWarn && (
              <UnnamedWarnPanel
                noun={mode}
                onProceed={(name) => {
                  setNameWarn(false);
                  if (name) setProjectName(name);
                  setPendingRun(true);
                }}
                onDismiss={() => setNameWarn(false)}
              />
            )}
            <div className="vi-station-footer-run">
              <button
                type="button"
                className="vi-btn vi-btn-accent"
                disabled={!canRun}
                onClick={() => onRun()}
              >
                {running
                  ? mode === "movie"
                    ? "Generating movie…"
                    : mode === "scene"
                      ? "Generating scene…"
                      : "Generating…"
                  : "Run"}
              </button>
              {/* (The cross-model "🧪 Test all models" ghost that sat here moved
                  into each tab's bottom Tester collapsible — k89.) */}
              {/* Cooperative cancel: queued dies outright; a running scene stops
                  between frames (a long frame finishes first). */}
              {running && activeJobId && (
                <button
                  type="button"
                  className="vi-btn"
                  disabled={status === "cancelling"}
                  onClick={() => cancelJob(activeJobId)}
                  title="Stop this generation — a frame already in flight finishes first"
                >
                  {status === "cancelling" ? "Cancelling…" : "✕ Cancel"}
                </button>
              )}
              {status && (
                <span className={`vi-status vi-status-${status}`}>
                  {running && runningReadout ? runningReadout : status}
                </span>
              )}
            </div>
          </div>
          </div>
        }
      />
      )}
    </section>
  );
}
