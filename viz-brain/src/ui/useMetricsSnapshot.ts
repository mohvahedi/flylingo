/**
 * Poll the shared metrics snapshot that src/metrics/fps.ts owns.
 *
 * The snapshot is mutated in place every rendered frame by BrainCloud
 * (recordFrame / updateFps) and published to window.__bc by publishMetrics,
 * so a poll is enough and costs no React renders inside the canvas.
 */

import { useEffect, useState } from 'react';
import { metrics, type MetricsSnapshot } from '../metrics/fps';

/** Deep copy so React sees a new object identity on every sample. */
function snapshot(): MetricsSnapshot {
  return { ...metrics };
}

export function useMetricsSnapshot(intervalMs = 250): MetricsSnapshot {
  const [value, setValue] = useState<MetricsSnapshot>(snapshot);

  useEffect(() => {
    const timer = window.setInterval(() => setValue(snapshot()), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs]);

  return value;
}
