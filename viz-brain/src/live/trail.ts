// Travelling activity pulses and their decaying trail.
//
// The reference look is a long-exposure photograph: bright cores on the active
// neurons, each one shedding a dotted tail of progressively smaller, dimmer
// dots that streams away from the cell. This module owns that tail.
//
// HOW IT IS DRIVEN
// ----------------
// Every frame the live slots carry the caller's 512 state values. A slot whose
// |state| clears ACTIVITY_FLOOR of the frame reference (or that just appeared in
// the real spikes list) advances a per-slot clock along its own ray and spawns
// one dot per SPAWN_INTERVAL of clock time. Successive dots from one slot line
// up along that ray, which is what makes the tail read as motion rather than
// noise. Each dot then ages; size and alpha fall with age and it is dropped at
// TRAIL_LIFE, so the trail is a real decay.
//
// Nothing here invents activity. With a dead frame (no_edges: all zeros, empty
// spikes) the reference is 0, no slot clears the floor, and the buffer empties
// by decay alone. Colours come from the caller's state and spike list only.
//
// The ray direction is fixed per slot from the real layout: the soma's offset
// from its (cell class, degree decile) bin centre, blended toward a global
// streaming axis so the tails leave the cloud on one side the way the reference
// does, plus a small deterministic wobble so two slots in one bin do not draw
// the same line. All of it is a display choice, not measured motion.

import type { BrainLayout } from '../layout';

export const TRAIL_CAPACITY = 3072;
export const TRAIL_LIFE = 0.6;
export const TRAIL_SLOTS = 512;

const TRAIL_SPEED = 0.6;
const TRAIL_MAX_OFFSET = 0.5;
/**
 * World-space spacing between successive dots of one trail. The old code
 * spawned one dot per frame, so the trail's dot spacing was a function of the
 * frame rate: at 30 fps that is a 16 px gap, and the tail read as a few loose
 * specks rather than a stream. Spawning on distance travelled instead makes the
 * dotted line the same length and density at 30 fps and at 300 fps.
 */
const TRAIL_SPACING = 0.02;
const ACTIVITY_FLOOR = 0.42;
const AMBER_SECONDS = 0.22;
const SPAWN_BUDGET = 192;

/** World-space bias so trails stream off to one side, as in the reference. */
const STREAM_AXIS: [number, number, number] = [0.82, 0.22, -0.53];

/** Palette: cyan to pale blue, amber reserved for the newest spikes. */
const CYAN: [number, number, number] = [0x7f / 255, 0xe0 / 255, 0xea / 255];
const PALE: [number, number, number] = [0xbf / 255, 0xef / 255, 0xf7 / 255];
const AMBER: [number, number, number] = [0xf0 / 255, 0xa0 / 255, 0x30 / 255];

export interface TrailBuffer {
  /** Written straight into the geometry attributes. */
  positions: Float32Array;
  colors: Float32Array;
  ages: Float32Array;
  powers: Float32Array;
  /** Live dots, densely packed at the front of every array. */
  count: number;
  /** Cumulative spawned dots, for the metrics panel. */
  spawned: number;
  slotClock: Float32Array;
  slotWait: Float32Array;
  slotAmber: Float32Array;
  slotDir: Float32Array;
  dirsFor: Int32Array | null;
  layout: BrainLayout;
}

export function createTrail(layout: BrainLayout): TrailBuffer {
  return {
    positions: new Float32Array(TRAIL_CAPACITY * 3),
    colors: new Float32Array(TRAIL_CAPACITY * 3),
    ages: new Float32Array(TRAIL_CAPACITY),
    powers: new Float32Array(TRAIL_CAPACITY),
    count: 0,
    spawned: 0,
    slotClock: new Float32Array(TRAIL_SLOTS),
    slotWait: new Float32Array(TRAIL_SLOTS),
    slotAmber: new Float32Array(TRAIL_SLOTS),
    slotDir: new Float32Array(TRAIL_SLOTS * 3),
    dirsFor: null,
    layout,
  };
}

function hash01(n: number): number {
  let x = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  x ^= x >>> 16;
  return x / 4294967296;
}

function rebuildDirections(t: TrailBuffer, liveIndices: Int32Array): void {
  const layout = t.layout;
  const pos = layout.positions;
  const binOf = layout.binOf;
  const bins = layout.bins;
  for (let s = 0; s < TRAIL_SLOTS; s += 1) {
    const idx = s < liveIndices.length ? liveIndices[s] : -1;
    const o = s * 3;
    if (idx < 0 || idx >= layout.count) {
      t.slotDir[o] = 0;
      t.slotDir[o + 1] = 0;
      t.slotDir[o + 2] = 0;
      continue;
    }
    const bin = bins[binOf[idx]];
    const c = bin ? bin.center : ([0, 0, 0] as [number, number, number]);
    let dx = pos[idx * 3] - c[0];
    let dy = pos[idx * 3 + 1] - c[1];
    let dz = pos[idx * 3 + 2] - c[2];
    let len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 1e-4) {
      dx = hash01(s * 3 + 1) - 0.5;
      dy = hash01(s * 3 + 2) - 0.5;
      dz = hash01(s * 3 + 3) - 0.5;
      len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    }
    dx /= len;
    dy /= len;
    dz /= len;
    // Blend toward the global streaming axis so the tails leave on one side.
    let bx = dx * 0.55 + STREAM_AXIS[0] * 0.45;
    let by = dy * 0.55 + STREAM_AXIS[1] * 0.45;
    let bz = dz * 0.55 + STREAM_AXIS[2] * 0.45;
    let bl = Math.sqrt(bx * bx + by * by + bz * bz) || 1;
    bx /= bl;
    by /= bl;
    bz /= bl;
    // Deterministic wobble so neighbouring slots do not draw the same ray.
    const w = 0.22;
    bx += (hash01(s * 7 + 11) - 0.5) * w;
    by += (hash01(s * 7 + 12) - 0.5) * w;
    bz += (hash01(s * 7 + 13) - 0.5) * w;
    bl = Math.sqrt(bx * bx + by * by + bz * bz) || 1;
    t.slotDir[o] = bx / bl;
    t.slotDir[o + 1] = by / bl;
    t.slotDir[o + 2] = bz / bl;
  }
  t.dirsFor = liveIndices;
}

/**
 * Append one dot to the trail buffer at `off` world units along slot `s`'s ray.
 * Returns false once the capacity is reached.
 */
function pushDot(
  t: TrailBuffer,
  pos: Float32Array,
  idx: number,
  s: number,
  off: number,
  power: number,
): boolean {
  if (t.count >= TRAIL_CAPACITY) return false;
  const i = t.count;
  t.count = i + 1;
  t.spawned += 1;
  const d = s * 3;
  t.positions[i * 3] = pos[idx * 3] + t.slotDir[d] * off;
  t.positions[i * 3 + 1] = pos[idx * 3 + 1] + t.slotDir[d + 1] * off;
  t.positions[i * 3 + 2] = pos[idx * 3 + 2] + t.slotDir[d + 2] * off;
  t.ages[i] = 0;
  t.powers[i] = power;
  const c = t.colors;
  if (t.slotAmber[s] > 0) {
    c[i * 3] = AMBER[0];
    c[i * 3 + 1] = AMBER[1];
    c[i * 3 + 2] = AMBER[2];
  } else {
    c[i * 3] = CYAN[0] + (PALE[0] - CYAN[0]) * power;
    c[i * 3 + 1] = CYAN[1] + (PALE[1] - CYAN[1]) * power;
    c[i * 3 + 2] = CYAN[2] + (PALE[2] - CYAN[2]) * power;
  }
  return true;
}

/**
 * Age, drop and spawn trail dots. Returns the live dot count.
 *
 * dt is in seconds. ref is the same per-frame reference the shaders use: a
 * value of zero means a dead frame and stops all spawning.
 */
export function updateTrail(
  t: TrailBuffer,
  liveIndices: Int32Array,
  state: number[],
  spikes: number[],
  ref: number,
  nslots: number,
  hasIds: boolean,
  dt: number,
): number {
  if (t.dirsFor !== liveIndices) rebuildDirections(t, liveIndices);

  const pos = t.layout.positions;
  const positions = t.positions;
  const colors = t.colors;
  const ages = t.ages;
  const powers = t.powers;

  // -------------------------------------------------------------- age out
  let n = t.count;
  for (let i = 0; i < n; ) {
    ages[i] += dt;
    if (ages[i] >= TRAIL_LIFE) {
      n -= 1;
      if (i !== n) {
        positions[i * 3] = positions[n * 3];
        positions[i * 3 + 1] = positions[n * 3 + 1];
        positions[i * 3 + 2] = positions[n * 3 + 2];
        colors[i * 3] = colors[n * 3];
        colors[i * 3 + 1] = colors[n * 3 + 1];
        colors[i * 3 + 2] = colors[n * 3 + 2];
        ages[i] = ages[n];
        powers[i] = powers[n];
      }
    } else {
      i += 1;
    }
  }
  t.count = n;

  for (let s = 0; s < TRAIL_SLOTS; s += 1) {
    if (t.slotAmber[s] > 0) {
      const a = t.slotAmber[s] - dt;
      t.slotAmber[s] = a > 0 ? a : 0;
    }
  }

  if (!(ref > 0.0001)) return t.count;

  // ------------------------------------------------------------ spawn dots
  const limit = Math.min(nslots, TRAIL_SLOTS);
  let budget = SPAWN_BUDGET;
  for (let s = 0; s < limit && budget > 0; s += 1) {
    const idx = s < liveIndices.length ? liveIndices[s] : -1;
    if (idx < 0 && hasIds) {
      t.slotClock[s] = 0;
      t.slotWait[s] = 0;
      continue;
    }
    if (idx < 0 || idx >= t.layout.count) continue;

    const value = state[s] ?? 0;
    const mag = Math.abs(value) / ref;
    let fresh = false;
    for (let k = 0; k < spikes.length; k += 1) {
      if (spikes[k] === s) {
        fresh = true;
        break;
      }
    }
    if (fresh) t.slotAmber[s] = AMBER_SECONDS;

    t.slotWait[s] += dt;
    if (!fresh && mag < ACTIVITY_FLOOR) {
      // Slot went quiet: the pulse stops and the ray resets toward the soma.
      const keep = 1 - dt * 3;
      t.slotClock[s] *= keep > 0 ? keep : 0;
      t.slotWait[s] = 0;
      continue;
    }
    const power = fresh ? 1 : Math.min(1, mag);
    const step = dt * TRAIL_SPEED * (0.5 + power);
    const prev = t.slotClock[s];
    let clock = prev + step;
    let wrapped = false;
    // The head cycles along its ray and relaunches at the soma, so a pulse
    // travels out, leaves its tail behind and is replaced by the next one.
    // Clamping the head at the far end (the old behaviour) made every later dot
    // spawn at one offset, which turned the tail into a single static blob.
    if (clock >= TRAIL_MAX_OFFSET) {
      clock -= TRAIL_MAX_OFFSET;
      wrapped = true;
    }
    t.slotClock[s] = clock;
    t.slotWait[s] = 0;

    // One dot per TRAIL_SPACING of travel, not one per frame, so spacing and
    // tail length are the same at any frame rate. The dot at the head is the
    // newest and brightest; the dots already laid down age behind it.
    let from = wrapped ? 0 : Math.ceil(prev / TRAIL_SPACING);
    const to = Math.floor(clock / TRAIL_SPACING);
    for (let k = from; k <= to && budget > 0; k += 1) {
      budget -= 1;
      if (!pushDot(t, pos, idx, s, k * TRAIL_SPACING, power)) break;
    }
    if (budget > 0) {
      budget -= 1;
      pushDot(t, pos, idx, s, clock, power);
    }
  }
  return t.count;
}
