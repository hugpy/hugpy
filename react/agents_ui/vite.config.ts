import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Standalone build of the agent-fleet "arm" of the hugpy ecosystem — sibling of
// media_intelligence_ui / video_intelligence_ui with the same posture:
// - Served at /fleet by the hugpy dev UI's origin, so `base` makes every asset
//   URL resolve under that path. (Dir name is agents_ui; the PUBLIC ROUTE is
//   /fleet — operator-confirmed, see dev/AGENTS-ARM-PLAN.md.)
// - Native toolchain (Vite + Tailwind v4), deliberately NOT integrated into the
//   dev UI's webpack build ("loosely coupled, NOT bundled into the SPA").
// - Dials the SAME-ORIGIN current-hugpy API (/api) if/when it needs to. Override
//   the base with VITE_HUGPY_API_BASE at build time if needed.
export default defineConfig({
  base: "/fleet/",
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  // Standalone `vite dev` / `vite preview` reach the same-origin API by proxying
  // /api to the local hugpy Flask backend, stripping the /api prefix exactly like
  // the webpack devServer (dev/ui) and prod nginx do. Override the target with
  // HUGPY_API_URL. The integrated path (webpack serving the built dist at /fleet)
  // has its own proxy and is unaffected.
  server: {
    proxy: {
      "/api": {
        target: process.env.HUGPY_API_URL || "http://127.0.0.1:7002",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
