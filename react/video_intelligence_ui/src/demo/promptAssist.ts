// Canned PROMPT-ASSIST generator for the showroom (?demo=1). The demoFetch shim
// answers POST /video/prompt/assist entirely CLIENT-SIDE from these fragment banks
// — ZERO backend contact — so the Enhance / Generate buttons are genuinely live in
// the brochure without an LLM.
//
// Two behaviours, mirroring the real route's { mode } contract:
//   • "generate" — composes a WHOLE cinematic prompt from the fragment banks below.
//     Combinations are numerous (subject × setting × light × camera × style × grade),
//     so successive clicks (almost) never repeat; nextGeneratedPrompt() additionally
//     GUARANTEES the next prompt differs from the last one it handed out.
//   • "detail" (Enhance) — takes the submitted draft and APPENDS a few randomized
//     cinematic descriptors, so the box visibly grows with richer detail while keeping
//     whatever the visitor already wrote.
//
// Flavor is deliberately kept in the same cinematic world as the seeded amber-dune
// sample (src/demo/seed.ts DEMO_GENERATE_PROMPT): warm golden-hour / dusk light,
// volumetric atmosphere, film-grain grades — so a Generate result reads as a sibling
// of the prefilled prompt, not a jarring genre jump. Math.random is fine here (demo).

// ── fragment banks ──────────────────────────────────────────────────────────
// Each bank is a coherent slot in "SUBJECT, SETTING, LIGHT, CAMERA, STYLE, GRADE".
// Every combination reads as a plausible cinematic still/shot.
const SUBJECTS = [
  "a lone figure in a flowing cloak",
  "a weathered lighthouse keeper",
  "a wanderer with a lantern",
  "a solitary rider on horseback",
  "an astronaut adrift from the hatch",
  "a child chasing paper kites",
  "a violinist mid-performance",
  "a fox threading through tall grass",
  "twin travelers on a ridge line",
  "a diver breaking the surface",
  "a monk crossing a stone bridge",
  "a dancer caught mid-turn",
];

const SETTINGS = [
  "on a windswept dune",
  "at the edge of a mirror-still lake",
  "among towering redwoods wreathed in mist",
  "on a rain-slick city street",
  "beneath a cathedral of ice",
  "across a field of swaying wildflowers",
  "on a cliff above a churning sea",
  "inside an abandoned glasshouse",
  "along a deserted railway platform",
  "under a canopy of drifting cherry blossom",
  "in a canyon of red sandstone",
  "on a fog-bound harbor pier",
];

const LIGHTS = [
  "golden hour, amber sun low on the horizon",
  "soft dusk light deepening from orange to indigo",
  "cold blue pre-dawn glow",
  "warm lantern light against deep shadow",
  "shafts of volumetric sunlight through haze",
  "overcast silver light, gentle and diffuse",
  "neon reflections shimmering on wet ground",
  "moonlight silvering every edge",
  "backlit rim light haloing the subject",
  "candle-warm firelight flickering nearby",
];

const CAMERAS = [
  "cinematic wide shot",
  "intimate close-up, shallow depth of field",
  "slow dolly push-in",
  "sweeping aerial vantage",
  "low-angle hero framing",
  "over-the-shoulder tracking shot",
  "static locked-off medium shot",
  "drifting handheld follow",
  "dramatic Dutch-tilt angle",
];

const STYLES = [
  "35mm film grain",
  "anamorphic lens flare",
  "hyper-detailed photoreal render",
  "painterly matte-painting finish",
  "muted teal-and-orange color grade",
  "high dynamic range, crisp micro-contrast",
  "dreamy soft-focus bloom",
  "rich chiaroscuro lighting",
];

const GRADES = [
  "long violet shadows",
  "warm highlights, cool shadows",
  "deep cinematic contrast",
  "atmospheric depth haze",
  "delicate lens bokeh in the background",
  "subtle vignette drawing the eye inward",
];

// Extra descriptor fragments Enhance appends to an existing draft (kept distinct from
// the whole-prompt banks so the enrichment reads as ADDED cinematic polish).
const ENHANCERS = [
  "volumetric light",
  "ultra-detailed textures",
  "dramatic rim lighting",
  "shallow depth of field",
  "atmospheric haze",
  "rich color grading",
  "cinematic composition",
  "soft golden-hour glow",
  "fine film grain",
  "gentle lens bloom",
  "sweeping sense of scale",
  "crisp foreground detail",
];

// ── random helpers ────────────────────────────────────────────────────────────
function pick<T>(bank: readonly T[]): T {
  return bank[Math.floor(Math.random() * bank.length)];
}

// Pick `n` DISTINCT items from a bank (n clamped to the bank size), order preserved
// by draw. Used by Enhance to append a small handful of non-repeating descriptors.
function pickSome<T>(bank: readonly T[], n: number): T[] {
  const pool = bank.slice();
  const out: T[] = [];
  const count = Math.min(n, pool.length);
  for (let i = 0; i < count; i++) {
    const idx = Math.floor(Math.random() * pool.length);
    out.push(pool.splice(idx, 1)[0]);
  }
  return out;
}

// ── generate: a whole cinematic prompt from the banks ──────────────────────────
function composePrompt(): string {
  const subject = pick(SUBJECTS);
  const setting = pick(SETTINGS);
  const light = pick(LIGHTS);
  const camera = pick(CAMERAS);
  const style = pick(STYLES);
  const grade = pick(GRADES);
  // Sentence-cased subject, comma-joined clauses — reads like the seeded sample.
  const head = subject.charAt(0).toUpperCase() + subject.slice(1);
  return `${head} ${setting}, ${light}, ${camera}, ${grade}, ${style}`;
}

// The last prompt handed out by nextGeneratedPrompt — so we never return the SAME
// prompt twice in a row (the operator's "genuinely new each click" requirement).
let _lastGenerated: string | null = null;

/**
 * A freshly composed cinematic prompt, GUARANTEED to differ from the previous one.
 * With thousands of combinations a repeat is already unlikely; this makes "never the
 * same one twice in a row" a hard guarantee (bounded retry, then a forced tail
 * descriptor so it always terminates even in the degenerate single-bank case).
 */
import { assistPrompt, sectionFor } from "./sampleSets";
import { SAMPLES2 } from "./sampleData";

const s2cursor: Record<string, number> = {};

/**
 * 2026-08-13 (operator ruling): ✨ generate hands out the REAL prompts that
 * produced the bundled sample renders for the requesting section, and ARMS
 * that sample so the next generate renders exactly that media (sampleSets.ts).
 * The fragment-bank composer below stays as the implementation detail of
 * `enhancePrompt` only.
 */
export function samplePromptFor(kind: string): string {
  // Prefer the RECORDED samples (sampleData.ts): the prompt handed out is one
  // that provably produced a bundled render, and enqueue-side matching
  // (fixtures.pickReplay) then replays exactly that envelope.
  const section = sectionFor(kind === "image" ? "scene" : kind);
  const pool = SAMPLES2.filter((x) => x.kind === section && x.prompts.length);
  if (pool.length) {
    const i = (s2cursor[section] = ((s2cursor[section] ?? -1) + 1) % pool.length);
    return pool[i].prompts.join("\n\n");
  }
  return assistPrompt(sectionFor(kind));
}

export function nextGeneratedPrompt(): string {
  let candidate = composePrompt();
  for (let tries = 0; tries < 8 && candidate === _lastGenerated; tries++) {
    candidate = composePrompt();
  }
  if (candidate === _lastGenerated) {
    // Extremely unlikely; force a difference so we never violate the guarantee.
    candidate = `${candidate}, ${pick(ENHANCERS)}`;
  }
  _lastGenerated = candidate;
  return candidate;
}

/**
 * Enhance ("detail"): take the submitted draft and return it enriched with a few
 * appended cinematic descriptors (randomized, distinct from each other). A blank
 * draft falls back to a freshly generated prompt (mirrors the real route, which
 * treats Enhance-with-no-draft leniently rather than erroring). The visible effect
 * is the box GROWING with detail while keeping whatever the visitor already wrote.
 */
export function enhancePrompt(draft: string): string {
  const base = draft.trim();
  if (base === "") return nextGeneratedPrompt();
  const added = pickSome(ENHANCERS, 2 + Math.floor(Math.random() * 2)); // 2–3 descriptors
  return `${base}, ${added.join(", ")}`;
}
