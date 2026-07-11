import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy target must match run-webapp.sh's backend host/port.
export default defineConfig({
  plugins: [react()],
  // plotly.js/lib (CJS source, needed for carpet traces) references node's `global`
  define: { global: 'globalThis' },
  server: { proxy: { '/api': 'http://127.0.0.1:8010' } },
})
