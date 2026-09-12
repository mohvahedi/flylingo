// Shared metrics object. The React UI reads it to draw the FPS readout, and a
// headless harness reads the same numbers through window.__bc, so the reported
// FPS figure and the on-screen figure can never disagree.

export interface MetricsSnapshot {
  /** Rolling frames per second over the last FPS_WINDOW frames. */
  fps: number;
  /** Total frames rendered since mount. */
  frames: number;
  /** Duration of the most recent frame, milliseconds. */
  lastFrameMs: number;
  /** Mean frame duration over the rolling window, milliseconds. */
  avgFrameMs: number;
  /** Worst frame duration seen during the run. */
  worstFrameMs: number;
  /** Points rendered in the static cloud. */
  points: number;
  /** Live slots carried by the overlay. */
  liveSlots: number;
  /** Live slots whose sampled id resolved to a real neuron in the cloud. */
  resolvedIds: number;
  /** Either "full" (all cloud points re-uploaded per frame) or "subset". */
  driveMode: string;
  /** Where the live slots got their positions. */
  mappingBasis: string;
  /** WEBGL_debug_renderer_info string, so the number can be interpreted. */
  renderer: string;
  /** Bytes uploaded to the GPU per frame by the attribute updates. */
  bytesPerFrame: number;
  /** True while the synthetic idle animation is the source of the frame. */
  synthetic: boolean;
  /** Live mode label from the experiment, when the caller supplies one. */
  mode: string;
  /** The auto-range reference actually used this frame (95th pct of |state|). */
  refValue: number;
  /** Largest |state| seen this frame. Zero means a genuinely dead frame. */
  peakAbs: number;
  /** Spikes in the current frame. Measured typical value is about 5 of 512. */
  spikeCount: number;
  /** Per-point field bytes queued this frame. */
  fieldUploadBytes: number;
  /** Drawn hairline edges. A sampled illustration, not measured adjacency. */
  edgeCount: number;
  /** Hubs the edge set was built from, real high in-degree neurons. */
  edgeHubs: number;
  /** Mean drawn edge length in world units. */
  edgeMeanLength: number;
  /** One line stating what the edges were actually derived from. */
  edgeSource: string;
  /** Live dots in the decaying trail buffer this frame. */
  trailPoints: number;
  /** Capacity of the trail buffer. */
  trailCapacity: number;
  /** Bytes the trail re-uploads per frame while it is not empty. */
  trailUploadBytes: number;
  /** The caption line rendered under the cloud. */
  caption: string;
}

const FPS_WINDOW = 120;

const durations: number[] = [];
let worst = 0;

export function resetMetrics(): void {
  durations.length = 0;
  worst = 0;
}

export function recordFrame(dtMs: number): void {
  durations.push(dtMs);
  if (durations.length > FPS_WINDOW) durations.shift();
  if (dtMs > worst) worst = dtMs;
}

function mean(list: number[]): number {
  if (list.length === 0) return 0;
  let s = 0;
  for (const v of list) s += v;
  return s / list.length;
}

/**
 * Per-frame auto-range reference.
 *
 * Returns the 95th percentile of |state|, or 0 for a dead frame. Zero is the
 * important case: the no_edges control produces all zeros, and the renderer
 * must go genuinely dark rather than stretch numerical dust to full brightness.
 */
export function referenceOf(state: number[], scratch: Float32Array): number {
  const n = Math.min(state.length, scratch.length);
  if (n === 0) return 0;
  let peak = 0;
  for (let i = 0; i < n; i += 1) {
    const a = Math.abs(state[i]);
    scratch[i] = a;
    if (a > peak) peak = a;
  }
  // Absolute floor: below this the frame is numerically dead, not merely quiet.
  if (peak < 1e-4) return 0;
  const view = scratch.subarray(0, n);
  view.sort();
  const idx = Math.min(n - 1, Math.floor(0.95 * n));
  return view[idx];
}

/** The one live snapshot object. Mutated in place, never reallocated. */
export const metrics: MetricsSnapshot = {
  fps: 0,
  frames: 0,
  lastFrameMs: 0,
  avgFrameMs: 0,
  worstFrameMs: 0,
  points: 0,
  liveSlots: 0,
  resolvedIds: 0,
  driveMode: 'unknown',
  mappingBasis: 'unknown',
  renderer: 'unknown',
  bytesPerFrame: 0,
  synthetic: true,
  mode: 'unknown',
  refValue: 0,
  peakAbs: 0,
  spikeCount: 0,
  fieldUploadBytes: 0,
  edgeCount: 0,
  edgeHubs: 0,
  edgeMeanLength: 0,
  edgeSource: '',
  trailPoints: 0,
  trailCapacity: 0,
  trailUploadBytes: 0,
  caption: '',
};

export function updateFps(): void {
  const avg = mean(durations);
  metrics.avgFrameMs = avg;
  metrics.fps = avg > 0 ? 1000 / avg : 0;
  metrics.worstFrameMs = worst;
}

declare global {
  interface Window {
    __bc?: MetricsSnapshot;
  }
}

export function publishMetrics(): MetricsSnapshot {
  if (typeof window !== 'undefined') window.__bc = metrics;
  return metrics;
}
