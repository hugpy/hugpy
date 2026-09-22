// PRE-RENDERED SAMPLE SETS for the canned /video/generate demo (2026-08-13).
//
// The operator's ruling: the demo must NEVER touch a GPU or a model — every
// "render" is a pre-rendered sample, and the ✨ Generate prompt button hands
// out the REAL prompts that produced those samples, per section. Source of
// truth: /mnt/llm_storage/video_intel/generate-samples (each render dir now
// carries prompt.txt + manifest.json, reconstructed from media_jobs.db where
// the renders originally shipped without them). A curated ~8 MB subset is
// bundled here; the full set stays on the share.
//
// THE COUPLING CONTRACT (what makes the demo read as real):
//   ✨ assist (mode "generate") → assistPrompt(section) ARMS a sample set and
//   returns its prompt; the next enqueue for that section CONSUMES the armed
//   set, so the media that "renders" is the media that prompt actually made.
//   Generating without assist just round-robins the sets. Selection happens
//   at ENQUEUE (stored on the job) so repeated polls stay stable.
//
// Honest exclusions: renders whose real prompt was operator keyboard-mash
// ("asdasd") keep their on-disk manifest but are not showcased in the assist
// bank — the demo only advertises prompts worth reading.

import type { MediaRef } from "../video/contract";

// scene: desert roads (job 187ac900…, sd-turbo, 6 frames @512)
import { demoMedia } from "./mediaBase";
const sdF0 = demoMedia("generate/scene-desert/frame_00000.png");
const sdF1 = demoMedia("generate/scene-desert/frame_00001.png");
const sdF2 = demoMedia("generate/scene-desert/frame_00002.png");
const sdF3 = demoMedia("generate/scene-desert/frame_00003.png");
const sdF4 = demoMedia("generate/scene-desert/frame_00004.png");
const sdF5 = demoMedia("generate/scene-desert/frame_00005.png");
const sdClip = demoMedia("generate/scene-desert/scene.mp4");
// scene: vintage frames (job 69a7bbcf…) — render showcased, prompt not
const svF0 = demoMedia("generate/scene-vintage/frame_00000.png");
const svF1 = demoMedia("generate/scene-vintage/frame_00001.png");
const svF2 = demoMedia("generate/scene-vintage/frame_00002.png");
const svClip = demoMedia("generate/scene-vintage/scene.mp4");
// movie: sunny meadow (job 5ca10b84…, 2 goals) + swirling galaxy (1eed0b95…, 4 goals)
const mmSeg0 = demoMedia("generate/movie-medow/seg_00.mp4");
const mmSeg1 = demoMedia("generate/movie-medow/seg_01.mp4");
const mmMovie = demoMedia("generate/movie-medow/movie.mp4");
const mgSeg0 = demoMedia("generate/movie-galaxy/seg_00.mp4");
const mgSeg1 = demoMedia("generate/movie-galaxy/seg_01.mp4");
const mgSeg2 = demoMedia("generate/movie-galaxy/seg_02.mp4");
const mgSeg3 = demoMedia("generate/movie-galaxy/seg_03.mp4");
const mgMovie = demoMedia("generate/movie-galaxy/movie.mp4");
// clips (studio i2v manifests carry the prompts)
const clCat = demoMedia("generate/clips/cat-lounge.mp4");
const clDrone = demoMedia("generate/clips/drone-city.mp4");
const clSails = demoMedia("generate/clips/paper-sails.mp4");
// cinema: dank cavern (movie.json segment prompts; seg_01 was a branch-only
// render, so the set is segment_00 + the assembled movie LAST)
const ccSeg0 = demoMedia("generate/cinema-cavern/seg_00.mp4");
const ccMovie = demoMedia("generate/cinema-cavern/movie.mp4");

function img(id: string, uri: string): MediaRef {
  return { asset_id: id, kind: "image", uri, mime: "image/png",
           width: 512, height: 512, duration_s: null, fps_native: null,
           sample_rate: null, channels: null };
}
function vid(id: string, uri: string, dur = 4.0): MediaRef {
  return { asset_id: id, kind: "video", uri, mime: "video/mp4",
           width: 512, height: 512, duration_s: dur, fps_native: 6,
           sample_rate: null, channels: null };
}

export type SectionKind = "image" | "scene" | "movie" | "clip" | "cinema";

export interface SampleSet {
  title: string;
  /** The REAL prompt(s) that produced this render (joined for the assist). */
  prompt: string;
  /** Contract-ordered outputs (frames first, assembled clip/movie LAST). */
  outputs: MediaRef[];
  /** Showcased by ✨ generate? (false = render-only; prompt was junk/empty) */
  showcase: boolean;
}

const DESERT_PROMPT =
  "this is the first scene in view. its california and the air is crisp\n\n" +
  "the last frame needs to be the gentlment in the left side having walked allwthe way across";

export const SAMPLE_SETS: Record<SectionKind, SampleSet[]> = {
  image: [
    { title: "california crisp", prompt: "this is the first scene in view. its california and the air is crisp",
      outputs: [img("gs-img-desert", sdF0)], showcase: true },
    { title: "meadow still", prompt: "a sunny green meadow with wildflowers",
      outputs: [img("gs-img-vintage", svF0)], showcase: true },
  ],
  scene: [
    { title: "desert roads", prompt: DESERT_PROMPT,
      outputs: [img("gs-sd-0", sdF0), img("gs-sd-1", sdF1), img("gs-sd-2", sdF2),
                img("gs-sd-3", sdF3), img("gs-sd-4", sdF4), img("gs-sd-5", sdF5),
                vid("gs-sd-clip", sdClip)], showcase: true },
    { title: "vintage frames", prompt: "asdasd",
      outputs: [img("gs-sv-0", svF0), img("gs-sv-1", svF1), img("gs-sv-2", svF2),
                vid("gs-sv-clip", svClip)], showcase: false },
  ],
  movie: [
    { title: "sunny meadow", prompt:
        "a sunny green meadow with wildflowers\n\nthe same meadow at golden sunset, warm light",
      outputs: [vid("gs-mm-0", mmSeg0), vid("gs-mm-1", mmSeg1),
                vid("gs-mm-movie", mmMovie, 8.0)], showcase: true },
    { title: "swirling galaxy", prompt:
        "a vast spiral galaxy in deep space, distant wide view, scattered stars\n\n" +
        "zooming toward a glowing blue and purple nebula, cosmic dust\n\n" +
        "a closer view of swirling cosmic gas clouds and brilliant clustered stars\n\n" +
        "a single radiant star filling the frame, blinding light, lens flare",
      outputs: [vid("gs-mg-0", mgSeg0), vid("gs-mg-1", mgSeg1), vid("gs-mg-2", mgSeg2),
                vid("gs-mg-3", mgSeg3), vid("gs-mg-movie", mgMovie, 16.0)], showcase: true },
  ],
  clip: [
    { title: "cat lounge", prompt: "a cat",
      outputs: [vid("gs-cl-cat", clCat)], showcase: true },
    { title: "drone over city lights", prompt: "the same glowing night city, camera slowly orbiting",
      outputs: [vid("gs-cl-drone", clDrone)], showcase: true },
    { title: "paper sails", prompt: "",
      outputs: [vid("gs-cl-sails", clSails)], showcase: false },
  ],
  cinema: [
    { title: "dank cavern", prompt:
        "In a subterranean cavern bathed in the harsh midday sun, a towering structure " +
        "dwarfs its surroundings, casting short, sharp black shadows that dance across " +
        "the cavern's walls. The shot begins with a sweeping establishing view; as the " +
        "camera slowly pushes in, the structure's intricate details come into focus.",
      outputs: [vid("gs-cc-0", ccSeg0), vid("gs-cc-movie", ccMovie, 10.0)], showcase: true },
  ],
};

// ── the assist⇄enqueue coupling ─────────────────────────────────────────────
const cursors: Record<SectionKind, number> = {
  image: 0, scene: 0, movie: 0, clip: 0, cinema: 0,
};
const armed: Partial<Record<SectionKind, SampleSet>> = {};

function showcased(kind: SectionKind): SampleSet[] {
  const all = SAMPLE_SETS[kind];
  const s = all.filter((x) => x.showcase);
  return s.length ? s : all;
}

/** ✨ generate: arm the next showcased set for `kind` and return its prompt. */
export function assistPrompt(kind: SectionKind): string {
  const sets = showcased(kind);
  const set = sets[cursors[kind] % sets.length];
  cursors[kind] = (cursors[kind] + 1) % sets.length;
  armed[kind] = set;
  return set.prompt;
}

let rr: Record<SectionKind, number> = { image: 0, scene: 0, movie: 0, clip: 0, cinema: 0 };

/** Enqueue-time selection: the armed set if ✨ ran, else round-robin. */
export function takeSet(kind: SectionKind): SampleSet {
  const a = armed[kind];
  if (a) { delete armed[kind]; return a; }
  const sets = showcased(kind);
  const set = sets[rr[kind] % sets.length];
  rr[kind] = (rr[kind] + 1) % sets.length;
  return set;
}

/** Map an assist/job context kind onto a sample section. */
export function sectionFor(kind: string): SectionKind {
  if (kind === "image" || kind === "scene" || kind === "movie" || kind === "clip" || kind === "cinema")
    return kind;
  if (kind === "generate_image") return "image";
  if (kind === "generate_scene") return "scene";
  if (kind === "generate_movie") return "movie";
  if (kind === "studio_i2v") return "clip";
  if (kind === "generate_studio_movie") return "cinema";
  return "scene";
}
