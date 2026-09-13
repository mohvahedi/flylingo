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
    rollupOptions: {
      // phone.html is built too. It is the scene the landing page embeds, so it has to exist
      // in dist; left out, the deploy 404s on it and the page shows an empty frame. Paths are
      // relative to the project root, which Vite resolves, so this needs no node builtins
      // (importing node:path would drag in @types/node for one call).
      input: {
        index: 'index.html',
        phone: 'phone.html',
      },
    },
  },
});
