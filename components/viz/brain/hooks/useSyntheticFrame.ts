// Synthetic idle animation, used when BrainCloud is mounted with no props, which
// is the state the real app is in before the socket opens.
//
// THIS IS NOT FLY ACTIVITY. It is a deterministic oscillator bank over the 512
// sampled slots. The UI labels it as synthetic whenever it is active, so nobody
// can mistake it for connectome output.
//
// SCALE. The generator targets the measured telemetry distribution rather than
// an invented 0..1 spread:
//     median |state| 0.051, p95 0.274, state_rms 0.105, ~5 spikes of 512.
// A heavy tailed per-slot amplitude reproduces that shape, and the array is
// rescaled so its own rms lands on target. The returned stateRms is measured
// from the array that is actually returned, never hard coded.

export interface SyntheticFrame {
  state: number[];
  spikes: number[];
  activeFraction: number;
  stateRms: number;
}

const SLOTS = 512;
/** Threshold the experiment uses for a spike. Measured to fire on ~5 of 512. */
const SPIKE_THRESHOLD = 0.5;
/** Measured state_rms across all three live modes: 0.096 to 0.108. */
const TARGET_RMS = 0.105;
/** Slots that carry a travelling pulse, so the raster has something to draw. */
const HOT_SLOTS = [7, 63, 128, 199, 256, 311, 384, 455];

interface Oscillator {
  freq: number;
  phase: number;
  weight: number;
  sign: number;
}

/** Deterministic oscillator bank built from a small integer hash. */
function buildOscillators(): Oscillator[] {
  const out: Oscillator[] = [];
  let h = 0x2545f491;
  const next = () => {
    h ^= h << 13;
    h ^= h >>> 17;
    h ^= h << 5;
    h >>>= 0;
    return h / 4294967296;
  };
  for (let i = 0; i < SLOTS; i += 1) {
    // Cubed uniform gives a heavy tail: most slots sit near zero and a few
    // carry most of the energy, which is the measured shape.
    const u = next();
    out.push({
      freq: 0.15 + next() * 0.7,
      phase: next() * Math.PI * 2,
      weight: u * u * u,
      sign: next() < 0.5 ? -1 : 1,
    });
  }
  return out;
}

const OSC = buildOscillators();

export function syntheticFrame(tSeconds: number): SyntheticFrame {
  const state = new Array<number>(SLOTS);
  // Slow global envelope plus a travelling wave, so the raster strip shows
  // drift instead of static noise.
  const envelope = 0.6 + 0.4 * Math.sin(tSeconds * 0.31);
  const wave = tSeconds * 0.85;

  let sumSq = 0;
  for (let i = 0; i < SLOTS; i += 1) {
    const o = OSC[i];
    const travelling = 0.12 * Math.sin(wave - (i / SLOTS) * 6.4);
    const v =
      o.sign *
        o.weight *
        envelope *
        Math.sin(tSeconds * o.freq * Math.PI * 2 + o.phase) +
      travelling * o.weight;
    state[i] = v;
    sumSq += v * v;
  }

  // Rescale the quiet background so the frame's own rms matches the measured
  // 0.105 exactly. A frame that is already near zero is left alone rather than
  // being stretched, which keeps the dead-brain case honest.
  const rawRms = Math.sqrt(sumSq / SLOTS);
  if (rawRms > 1e-6) {
    const k = TARGET_RMS / rawRms;
    for (let i = 0; i < SLOTS; i += 1) state[i] *= k;
  }

  // Travelling pulses on a handful of slots. These are the spikes.
  let spikeSumSq = 0;
  for (let s = 0; s < HOT_SLOTS.length; s += 1) {
    const slot = HOT_SLOTS[s];
    const period = 1.6 + s * 0.23;
    const phase = tSeconds / period + s * 0.37;
    const frac = phase - Math.floor(phase);
    // Narrow, sharp pulse: elevated for about 8 percent of the cycle.
    if (frac < 0.08) {
      const shape = Math.sin((frac / 0.08) * Math.PI);
      const amp = 0.72 + 0.14 * shape;
      state[slot] = OSC[slot].sign * amp;
    }
    spikeSumSq += state[slot] * state[slot];
  }

  let active = 0;
  const spikes: number[] = [];
  for (let i = 0; i < SLOTS; i += 1) {
    if (Math.abs(state[i]) >= SPIKE_THRESHOLD) {
      spikes.push(i);
      active += 1;
    }
  }

  // Recompute the rms from the returned array so the readout is the truth.
  let finalSq = 0;
  for (let i = 0; i < SLOTS; i += 1) finalSq += state[i] * state[i];

  return {
    state,
    spikes,
    activeFraction: active / SLOTS,
    stateRms: Math.sqrt(finalSq / SLOTS),
  };
}
