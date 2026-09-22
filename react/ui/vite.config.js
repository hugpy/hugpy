import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 6091,
    strictPort: true,
    // Vite blocks requests whose Host header it doesn't recognize.
    // nginx forwards Host: dev.hugpy.ai, so allow it.
    allowedHosts: ['dev.hugpy.ai'],
    // HMR runs over wss through nginx on 443, not directly on 6091.
    hmr: {
      protocol: 'wss',
      host: 'dev.hugpy.ai',
      clientPort: 443,
    },
    proxy: {
      '/api': {
        target: 'https://hugpy.ai',
        changeOrigin: true,
        secure: true,
      },
    },
  },
})