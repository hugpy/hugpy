// Showroom SEED values for the canned demo (?demo=1) — the "sample gens +
// prefilled prompts" flavor the brochure opens with. Everything here is
// demo-only: it is imported ONLY behind an isCanned() gate (installVideoDemo for
// the workbench seeding, and the station composers for the prefilled prompt), so
// none of it can leak into a live session's normal path.
//
// Coherence: the prompts describe scenes that plausibly match the bundled
// gen-1/gen-2 stills (see fixtures.demoGenerated) — gen-1 is a warm amber/indigo
// sunset gradient, gen-2 a green/teal one — so the prefilled prompt, the completed
// sample generation, and the still the visitor sees all tell ONE story.
import {
  demoGenerated,
  demoFrames,
  enqueueJob,
  type JobKind,
} from "./fixtures";
import { addToLibrary } from "../video/mediaLibrary";
import { trackJob } from "../video/jobTracker";

/**
 * The prompt the Generate (Image/Scene) composer opens PRE-FILLED with in canned
 * mode — a cinematic scene that matches gen-1 (the primary sample still). The
 * station reads this only when isCanned(); its live path stays an empty prompt.
 */
export const DEMO_GENERATE_PROMPT =
  "A lone figure on a windswept dune at golden hour, amber sun low on the horizon, " +
  "long violet shadows, cinematic wide shot, volumetric light, 35mm film grain";

/**
 * The prompt the Studio composer opens PRE-FILLED with in canned mode — a motion-
 * flavored variant of the same world (a clip prompt reads better with camera/motion
 * phrasing). Matches the sunset palette so the whole demo stays coherent.
 */
export const DEMO_STUDIO_PROMPT =
  "Slow dolly across golden dunes at sunset, drifting sand catching amber light, " +
  "the sky deepening from orange to indigo, cinematic, smooth camera move";

/** The model the sample generation is attributed to (matches the fixtures registry). */
const DEMO_MODEL = "sd-turbo";

// A stable groupId for the seeded completed generation so its Session Library work
// section is a real gen run (prompt + model header, Continue/Replicate affordances)
// rather than a loose "demo asset" tile. Namespaced so it can never collide with a
// live job id.
const SEED_IMAGE_GROUP = "demo-seed-generate-image";

/**
 * Seed the workbench so the showroom opens with SAMPLE WORK already present:
 *
 *   • one COMPLETED "generate image" run — gen-1 pushed into the Session Library
 *     with full generation provenance (prompt + model + group), so it renders as a
 *     proper gen work section, not a bare demo tile; and
 *   • one COMPLETED frame-extract of the sample clip — a couple of stills tied to
 *     the bundled clip, so the library reads as a session that has done real work;
 *   • one slowly-progressing in-flight "generate image" job, enqueued through the
 *     SAME canned staged-poll machinery (enqueueJob → the shim's staged jobStatus →
 *     the real jobTracker polls it to done), so the visitor sees live-feeling
 *     progress in Active Processes that is entirely fake and never touches a backend.
 *
 * Best-effort + idempotent: addToLibrary de-dups by uri and trackJob de-dups by
 * jobId, so a double-call (in-tab reload) can't double-seed. No-op on any throw.
 */
export function seedDemoWorkbench(): void {
  try {
    // ── the completed sample generation (Session Library) ────────────────────
    addToLibrary(demoGenerated[0], "generate", "generated", {
      prompt: DEMO_GENERATE_PROMPT,
      model: DEMO_MODEL,
      groupId: SEED_IMAGE_GROUP,
      genKind: "generate_image",
    });

    // ── a completed frame-extract of the sample clip (a couple of stills) ─────
    addToLibrary(demoFrames[0], "frames", "frame 1");
    addToLibrary(demoFrames[1], "frames", "frame 2");

    // ── one live-feeling in-flight generation (Active Processes) ─────────────
    // Enqueue a canned generate_image job and register it with the real tracker.
    // The tracker's module poll loop then drives it through the shim's staged
    // jobStatus (unclaimed → running-with-progress → done), and on `done` its
    // output lands in the Session Library via the tracker's normal pushOutputs —
    // exactly the path a real generation takes, with zero backend contact.
    const kind: JobKind = "generate_image";
    const jobId = enqueueJob(kind, null);
    trackJob({
      jobId,
      kind: "generate_image",
      station: "generate",
      label: DEMO_GENERATE_PROMPT,
      enqueuedAt: Date.now(),
      prompt: DEMO_GENERATE_PROMPT,
      model: DEMO_MODEL,
    });
  } catch {
    /* seeding is best-effort — the shim + prefilled prompts still demo without it */
  }
}
