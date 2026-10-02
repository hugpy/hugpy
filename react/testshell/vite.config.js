import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /api → the read-only DB mirror in api.py (HUGPY_TESTSHELL_PORT, default 7013).
const API = process.env.HUGPY_TESTSHELL_API || 'http://127.0.0.1:7013'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': { target: API, changeOrigin: true } } },
  preview: { proxy: { '/api': { target: API, changeOrigin: true } } },
})
