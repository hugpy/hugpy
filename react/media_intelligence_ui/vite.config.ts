import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Standalone build of the media-intelligence "arm" of the hugpy ecosystem.
// - Served at /media by the hugpy dev UI's origin, so `base` makes every asset
//   URL resolve under that path.
// - Keeps the component on its NATIVE toolchain (Vite + Tailwind v4) instead of
//   forcing Tailwind into the dev UI's webpack build (deliberately NOT strictly
//   integrated).
// - Dials the SAME-ORIGIN current-hugpy API (/api) — see src/config.ts. Override
//   the base with VITE_HUGPY_API_BASE at build time if needed.
export default defineConfig({
  base: "/media/",
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  // Standalone `vite dev` / `vite preview` reach the same-origin API by proxying
  // /api to the local hugpy Flask backend, stripping the /api prefix exactly like
  // the webpack devServer (dev/ui) and prod nginx do. Override the target with
  // HUGPY_API_URL. The integrated path (webpack serving the built dist at /media)
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
