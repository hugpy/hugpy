// Public surface of the video-intelligence demo layer (canned-only — no live
// mode; see mode.ts). entry.tsx calls installVideoDemo() ONCE, before anything
// that could touch the network.
import { isCanned } from "./mode";
import { installDemoTransport } from "./demoFetch";
import { demoClip, demoAudio } from "./fixtures";
import { seedDemoWorkbench } from "./seed";
import { addToLibrary, getLibrary } from "../video/mediaLibrary";

/**
 * Boot the canned demo: install the fetch shim, then seed the workbench so every
 * station opens with SHOWROOM FLAVOR instead of empty — the worked example (clip +
 * audio) PLUS sample generations (a completed image gen, a couple of extracted
 * frames, and one live-feeling in-flight generation; see seed.ts). No-op outside
 * ?demo=1. Idempotent across in-tab reloads (the library + tracker are
 * sessionStorage-backed under their demo-namespaced keys — addToLibrary/trackJob
 * de-dup, so don't double-seed).
 */
export function installVideoDemo(): void {
  if (!isCanned()) return;
  installDemoTransport();
  try {
    if (getLibrary().length === 0) {
      addToLibrary(demoClip, "demo", "sample clip");
      addToLibrary(demoAudio, "demo", "sample audio");
    }
  } catch {
    /* library seeding is best-effort — the shim alone still demos */
  }
  // Sample gens + one live-feeling in-flight job (its own idempotent guards).
  seedDemoWorkbench();
}

export {
  getDemoConfig,
  isDemo,
  isCanned,
  isEmbedded,
  type DemoMode,
  type DemoConfig,
} from "./mode";
export { installDemoTransport, uninstallDemoTransport } from "./demoFetch";
export { default as DemoBanner } from "./DemoBanner";
