// Library build for the @hugpy/ui npm package.
//
// Separate from the app build (the console ships via webpack -> dist/). This
// emits a consumable package into dist-lib/: ESM + CJS bundles, a single CSS
// file (design tokens + every component's styles), and .d.ts types.
//
// React, react-router, and MUI/emotion are marked external (peerDependencies)
// so the consumer's copy is used — no duplicate React, no bundled design system.

import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import dts from 'vite-plugin-dts'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [
    react(),
    dts({
      include: ['src'],
      exclude: ['src/main.jsx', 'dist', 'dist-lib', 'node_modules', '**/*.test.*'],
      // .d.ts for JS files too (the panels are .jsx); types stay loose there.
      compilerOptions: { allowJs: true, declaration: true, emitDeclarationOnly: true },
      // A missing/loose type shouldn't fail the whole build.
      logLevel: 'warn',
    }),
  ],
  build: {
    outDir: 'dist-lib',
    emptyOutDir: true,
    sourcemap: true,
    cssCodeSplit: false, // collapse all imported CSS into one style file
    lib: {
      entry: path.resolve(__dirname, 'src/index.ts'),
      name: 'HugpyUI',
      formats: ['es', 'cjs'],
      fileName: (format) => `hugpy-ui.${format === 'es' ? 'mjs' : 'cjs'}`,
    },
    rollupOptions: {
      // Anything the consumer is expected to provide (peerDependencies) or that
      // ships as its own dependency stays out of the bundle.
      external: [
        'react',
        'react-dom',
        'react/jsx-runtime',
        'react-router-dom',
        /^@mui\//,
        /^@emotion\//,
        // the embeddable station console + xterm — lazy-loaded by the Console tab,
        // optional peers (only needed if a consumer opens that tab)
        '@hugpy/console',
        /^@xterm\//,
      ],
      output: {
        assetFileNames: (asset) =>
          asset.name && asset.name.endsWith('.css') ? 'style.css' : 'assets/[name][extname]',
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM',
          'react/jsx-runtime': 'jsxRuntime',
        },
      },
    },
  },
})
