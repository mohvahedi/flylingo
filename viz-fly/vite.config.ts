import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Standalone harness. base './' so dist/ can be served from any prefix, including a
// static file server on the local network, with no rewrites.
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    port: 5191,
    strictPort: false,
  },
  preview: {
    port: 5192,
  },
  build: {
    outDir: 'dist',
    target: 'es2020',
    sourcemap: false,
    chunkSizeWarningLimit: 1600,
  },
});
