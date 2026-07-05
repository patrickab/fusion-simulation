import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy target must match run-webapp.sh's backend host/port.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8010' } },
})
