// Bundle the standalone intelligence ARMS into the console build so a
// production wheel serves them at /media and /video.
//
// Why this exists: in dev, webpack's devServer statically mounts each arm's
// dist (see webpack.config.js). In prod there is no devServer — the wheel
// ships ui/dist as console_dist/ and Flask serves it. So after the console
// build we (1) build each arm and (2) copy its dist into ui/dist/<arm>, which
// flows into console_dist/<arm>/* at packaging time (pyproject package-data)
// and is served by Flask's _mount_ui (which falls deep links back to the
// arm's own index.html).
//
// Runs as `postbuild` (after webpack's `clean:true` has already emitted
// dist/), so the copies are never wiped. Failures are non-fatal PER ARM: a
// console build must still succeed even if an arm can't be built (the arms
// are loosely coupled by design).
import { execSync } from "node:child_process";
import { existsSync, cpSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const uiDir = resolve(here, "..");

// Each arm is a sibling of ui/ with its own Vite toolchain whose vite.config
// sets base:"/<arm>/" so every asset URL resolves under its mount.
const ARMS = [
  { name: "media", dir: "../media_intelligence_ui" },
  { name: "video", dir: "../video_intelligence_ui" },
  // Dir is agents_ui but the mount/arm name is fleet — its vite.config sets
  // base:"/fleet/" to match (see agents_ui/vite.config.ts).
  { name: "fleet", dir: "../agents_ui" },
];

function log(msg) {
  console.log(`[bundle-arms] ${msg}`);
}

for (const arm of ARMS) {
  const armDir = resolve(uiDir, arm.dir);
  const armDist = resolve(armDir, "dist");
  const target = resolve(uiDir, `dist/${arm.name}`);

  if (!existsSync(armDir)) {
    log(`${arm.name} arm not found at ${armDir} — skipping (/${arm.name} absent).`);
    continue;
  }

  try {
    log(`building ${arm.name} arm…`);
    execSync("npm run build", { cwd: armDir, stdio: "inherit" });
  } catch (err) {
    log(`${arm.name} arm build failed (${err && err.message}); leaving /${arm.name} unbundled.`);
    continue;
  }

  if (!existsSync(armDist)) {
    log(`${arm.name} arm build produced no dist at ${armDist} — skipping.`);
    continue;
  }

  try {
    rmSync(target, { recursive: true, force: true });
    cpSync(armDist, target, { recursive: true });
    log(`copied ${armDist} -> ${target}`);
  } catch (err) {
    log(`copy failed (${err && err.message}); /${arm.name} not bundled.`);
  }
}
