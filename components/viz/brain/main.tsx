/**
 * Entry point. index.html mounts #root and loads /src/main.tsx.
 *
 * StrictMode is deliberately not used: it double mounts effects in dev, which
 * would open two websocket connections and build two WebGL contexts for the
 * same canvas. The harness is a measurement surface, so the numbers it shows
 * should come from a single mount.
 */

import { createRoot } from 'react-dom/client';
import App from './App';

const host = document.getElementById('root');
if (host === null) {
  throw new Error('index.html is missing <div id="root">');
}

createRoot(host).render(<App />);
