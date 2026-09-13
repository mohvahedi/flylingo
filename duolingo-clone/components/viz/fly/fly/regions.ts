/**
 * Activity binning for the fly harness.
 *
 * The app hands us 512 sampled state values in -1..1 (BrainFrame.state, see
 * D:\Projects\flylingo\INTERFACES.md). We bin those 512 values onto the anatomy so the
 * fly reads as alive: five regions, 18 channels, each channel driving one visible part
 * (head shell, thorax shell, four abdomen segments, six legs, two wings).
 *
 * IMPORTANT, measured on the real MaleCNS graph (166,700 neurons, 25,582,938 edges) with
 * real challenge embeddings:
 *   p50 = 0.051, p75 = 0.106, p90 = 0.174, p95 = 0.274, p98 = 0.386, max = 0.717
 *   state_rms ~ 0.105, activeFraction ~ 0.0044
 *
 * Those numbers drive the normalization below. A linear 0..1 emissive map against raw
 * |state| would leave the fly essentially unlit (p50 is 5 percent). So every frame is
 * normalized against its own 95th percentile, with a floor on that reference so a nearly
 * dead frame cannot be amplified into fake life, and a hard inert gate at the bottom:
 * under the `no_edges` control every value is exactly 0.0 and the fly must be dark.
 *
 * The binning is an arbitrary but stable map from the sampled subset onto body parts, and
 * is documented as such in PREVIEW.md. It is a display choice, not an anatomical claim.
 */

export const STATE_LEN = 512;

export type ChannelKind = 'head' | 'thorax' | 'abdomen' | 'legs' | 'wings';

export type Channel = {
  id: string;
  kind: ChannelKind;
  /** index within its kind, e.g. abdomen segment 0..3, leg 0..5 */
  slot: number;
  /** inclusive slice into the 512-length state array */
  from: number;
  to: number;
};

function span(start: number, len: number, count: number, i: number): { from: number; to: number } {
  const from = start + Math.floor((i * len) / count);
  const to = start + Math.floor(((i + 1) * len) / count) - 1;
  return { from, to: Math.max(from, to) };
}

/**
 * Region boundaries over the 512 sampled values. These are a display binning, not an
 * anatomical claim about which neurons are being sampled.
 */
export const REGIONS = {
  head: { from: 0, len: 64, channels: 2 },
  thorax: { from: 64, len: 128, channels: 4 },
  abdomen: { from: 192, len: 128, channels: 4 },
  legs: { from: 320, len: 128, channels: 6 },
  wings: { from: 448, len: 64, channels: 2 },
} as const;

export const CHANNELS: Channel[] = (() => {
  const out: Channel[] = [];
  (Object.keys(REGIONS) as ChannelKind[]).forEach((kind) => {
    const r = REGIONS[kind];
    for (let i = 0; i < r.channels; i += 1) {
      const { from, to } = span(r.from, r.len, r.channels, i);
      out.push({ id: `${kind}-${i}`, kind, slot: i, from, to });
    }
  });
  return out;
})();

export const CHANNEL_COUNT = CHANNELS.length;

/** Per-kind emissive tints. Cool instrument palette, one hue per anatomical region. */
export const REGION_COLOR: Record<ChannelKind, string> = {
  head: '#7dd3fc',
  thorax: '#22d3ee',
  abdomen: '#a78bfa',
  legs: '#34d399',
  wings: '#93c5fd',
};

// ---------------------------------------------------------------------------
// Normalization constants, all pinned to the measured distribution above.
// ---------------------------------------------------------------------------

/** Measured p95 of |activity| on real frames. The default reference. */
export const REF_P95 = 0.274;

/**
 * Floor on the per-frame reference. A frame whose p95 is below this is quiet rather than
 * dead, and we refuse to stretch it: without this floor a frame with a handful of tiny
 * values would normalize to a fully lit fly.
 */
export const REF_FLOOR = 0.14;

/** Ceiling on the reference, so a single enormous outlier cannot dim a real frame. */
export const REF_CEIL = 0.5;

/**
 * Below this p95 the frame is treated as inert (drive exactly 0, no glow at all). This is
 * the `no_edges` case: W zeroed means every state value is exactly 0.0, and the fly has to
 * look genuinely inert rather than animated by noise.
 */
export const INERT_P95 = 0.01;

/** Measured activeFraction on real frames, used to normalize that vital. */
export const ACTIVE_FRACTION_REF = 0.02;

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export type ChannelValues = {
  /** mean |state| per channel, raw */
  mean: Float32Array;
  /** peak |state| per channel, raw */
  peak: Float32Array;
  /** count of values with |state| >= 0.5 per channel */
  spikes: Float32Array;
  /** overall mean |state| across the frame, raw */
  overall: number;
  /** the per-frame normalization reference actually used (p95 clamped) */
  reference: number;
  /** the raw p95 of |activity| before clamping, for the readout */
  rawP95: number;
  /** true when the frame carries no usable signal at all (all zeros, i.e. no_edges) */
  inert: boolean;
  /** 0..1 normalized drive per channel, what the visuals actually read */
  drive: Float32Array;
};

export function createChannelValues(): ChannelValues {
  return {
    mean: new Float32Array(CHANNEL_COUNT),
    peak: new Float32Array(CHANNEL_COUNT),
    spikes: new Float32Array(CHANNEL_COUNT),
    overall: 0,
    reference: REF_P95,
    rawP95: 0,
    inert: true,
    drive: new Float32Array(CHANNEL_COUNT),
  };
}

/**
 * Per-channel normalization against the frame reference.
 *
 * mean/reference alone gives 0.051/0.274 = 0.19 for a typical channel, which is a visible
 * but dim glow; a lively channel at p90 (0.174) lands at 0.63. Gamma < 1 lifts the low end
 * so a real quiet frame still reads as a sleeping animal rather than a dead prop, and the
 * spike term gives a channel that actually fires a hard local highlight.
 */
export function normalizeDrive(meanAbs: number, reference: number, spikes: number, span: number): number {
  if (reference <= 0) return 0;
  const ratio = clamp01(meanAbs / reference);
  const lit = ratio ** 0.78;
  const spikeShare = span > 0 ? clamp01(spikes / span) : 0;
  return clamp01(lit * 0.82 + spikeShare * 1.6);
}

/**
 * Vitals: how awake the whole network is, 0..1.
 *
 * stateRms (~0.105) and activeFraction (~0.0044) are different scales, so each is
 * normalized against its own measured reference before being combined. This is used only
 * as a global floor and for the startle trigger; region glow comes from the channels.
 */
export function vitality(v: ChannelValues, stateRms: number, activeFraction: number): number {
  if (v.inert) return 0;
  const ref = v.reference > 0 ? v.reference : REF_P95;
  const byMean = v.overall / ref;
  const byRms = stateRms / ref;
  const byActive = activeFraction / ACTIVE_FRACTION_REF;
  return clamp01(Math.max(byMean, byRms, byActive * 0.6));
}

/**
 * Bin one frame of state into channel values. Defensive about length: a short array is
 * read where it exists and treated as zero elsewhere rather than throwing. NaNs and
 * non-finite values count as zero.
 */
const pctScratch = new Float32Array(STATE_LEN);
const absScratch = new Float32Array(STATE_LEN);

export function readChannels(
  activity: ArrayLike<number> | null | undefined,
  out: ChannelValues,
): ChannelValues {
  const n = activity ? Math.min(activity.length, STATE_LEN) : 0;

  // pass 1: absolute values plus the frame total, so we can find the p95 reference
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    const raw = activity![i];
    const v = typeof raw === 'number' && Number.isFinite(raw) ? raw : 0;
    const a = v < 0 ? -v : v;
    absScratch[i] = a;
    sum += a;
  }

  out.overall = n > 0 ? sum / n : 0;

  // p95 of |activity| over this frame. TypedArray.sort() is numeric ascending, and 512
  // elements is a few microseconds, so this is cheap enough to run every frame.
  let p95 = 0;
  if (n > 0) {
    pctScratch.set(absScratch.subarray(0, n));
    const sub = pctScratch.subarray(0, n);
    sub.sort();
    p95 = sub[Math.min(n - 1, Math.floor(0.95 * n))];
  }
  out.rawP95 = p95;

  // inert gate: all-zero frames (no_edges) and near-zero frames must not be amplified
  out.inert = !(p95 >= INERT_P95) || out.overall <= 0;
  if (out.inert) {
    out.reference = 0;
    out.mean.fill(0);
    out.peak.fill(0);
    out.spikes.fill(0);
    out.drive.fill(0);
    out.overall = 0;
    return out;
  }

  const reference = Math.min(REF_CEIL, Math.max(REF_FLOOR, p95));
  out.reference = reference;

  for (let c = 0; c < CHANNELS.length; c += 1) {
    const ch = CHANNELS[c];
    let s = 0;
    let peak = 0;
    let spikes = 0;
    let k = 0;
    for (let i = ch.from; i <= ch.to; i += 1) {
      const a = i < n ? absScratch[i] : 0;
      s += a;
      if (a > peak) peak = a;
      if (a >= 0.5) spikes += 1;
      k += 1;
    }
    const mean = k > 0 ? s / k : 0;
    out.mean[c] = mean;
    out.peak[c] = peak;
    out.spikes[c] = spikes;
    out.drive[c] = normalizeDrive(mean, reference, spikes, k);
  }
  return out;
}

/** Exponential smoothing toward the incoming frame, so 20 Hz frames do not pop. */
export function smoothChannels(cur: Float32Array, target: Float32Array, k: number): void {
  for (let i = 0; i < cur.length; i += 1) {
    cur[i] += (target[i] - cur[i]) * k;
  }
}

export type SyntheticFrame = {
  state: number[];
  stateRms: number;
  activeFraction: number;
};

/**
 * Synthetic idle activity, used when the harness is mounted with no props at all (before
 * the socket connects) and by the standalone dev page in "no backend" mode.
 *
 * The magnitudes are chosen to land in the same regime as the real data (p95 near 0.27,
 * stateRms near 0.1) rather than saturating, so the standalone demo shows the same glow
 * range the real app will produce.
 *
 * Shape: two low-frequency standing waves plus one traveling excitation packet sweeping
 * across the 512 sampled slots. Values are clamped to the documented -1..1 range.
 */
export function syntheticFrame(t: number, state?: number[]): SyntheticFrame {
  const buf = state && state.length === STATE_LEN ? state : new Array<number>(STATE_LEN).fill(0);
  const center = ((t * 120) % (STATE_LEN + 160)) - 80;
  const width = 46;
  let sq = 0;
  let active = 0;
  for (let i = 0; i < STATE_LEN; i += 1) {
    const d = i - center;
    const packet = 0.62 * Math.exp(-(d * d) / (2 * width * width));
    const wave = 0.16 * Math.sin(0.083 * i - t * 2.1) + 0.1 * Math.sin(0.031 * i + t * 0.9);
    const band = 0.12 * Math.sin(0.006 * i + t * 0.35);
    let v = packet * Math.sin(t * 6.0 - i * 0.05) + wave * (0.5 + 3.2 * Math.abs(band));
    if (v > 1) v = 1;
    else if (v < -1) v = -1;
    buf[i] = v;
    sq += v * v;
    if (v >= 0.5 || v <= -0.5) active += 1;
  }
  return {
    state: buf,
    stateRms: Math.sqrt(sq / STATE_LEN),
    activeFraction: active / STATE_LEN,
  };
}

/** Honest labels for the mode badge. Nothing here claims more than the wiring gives. */
export const MODE_LABEL: Record<string, { title: string; note: string; color: string }> = {
  intact: {
    title: 'intact MaleCNS v1.0',
    note: 'measured connectome, 166,700 neurons frozen; only the readout learns',
    color: '#22d3ee',
  },
  shuffled: {
    title: 'control: shuffled',
    note: 'fixed node relabeling, topology preserved, same size and sparsity',
    color: '#f59e0b',
  },
  no_edges: {
    title: 'control: no edges',
    note: 'W zeroed, state is exactly zero, no spikes by construction',
    color: '#ef4444',
  },
  random_graph: {
    title: 'control: random graph',
    note: 'degree-matched random sparse matrix, same nnz, same parameters',
    color: '#a78bfa',
  },
};

export function modeLabel(mode: string | undefined) {
  const key = (mode || 'intact').toLowerCase();
  return (
    MODE_LABEL[key] ?? {
      title: key || 'unknown',
      note: 'unrecognized mode string, rendering as supplied',
      color: '#94a3b8',
    }
  );
}
