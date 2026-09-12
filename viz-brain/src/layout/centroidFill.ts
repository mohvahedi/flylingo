// Which points sit on a group centroid rather than on a measured soma?
//
// public/data/soma_positions.f32 holds one position per neuron, but 27,038 of
// the 166,700 MaleCNS neurons have no measured soma, so the extract parks them
// at their (class, in-degree decile) group centroid. A centroid is a single
// coordinate, so hundreds or thousands of neurons land on the same triple.
// Measured on the shipped file (1e-4 quantum, exact float equality):
//
//   139,668 distinct coordinates across 166,700 points
//   5 coordinates are shared, holding 27,037 points
//   the four largest hold 14,418, 6,931, 5,095 and 590 points
//
// Additive blending turns a pile that size into a blown out white disc, which
// made the placeholder geometry the brightest thing in the panel while the
// measured somata stayed a dim speckle. That is backwards, so this module
// recovers the flag from the data itself instead of trusting a sidecar list.
//
// A coordinate shared by CENTROID_STACK_THRESHOLD or more points is a
// centroid placement. Nothing here asserts a count it did not measure: the
// caller gets the real totals and quotes those.

/** Coordinates shared by this many points or more are treated as a centroid. */
export const CENTROID_STACK_THRESHOLD = 3;

/** Quantum used to group coordinates. Coordinates are floats in [-1, 1]. */
const QUANTUM = 1e4;

export interface CentroidFill {
  /** 1 where the point has no measured soma and sits on a shared coordinate. */
  fill: Float32Array;
  /** 1 / (points sharing this coordinate), so a stack sums to about one point. */
  scale: Float32Array;
  /** Number of distinct coordinates in the file. */
  distinctCoordinates: number;
  /** Number of coordinates shared by the threshold or more. */
  sharedCoordinates: number;
  /** Points sitting on a shared coordinate. */
  piledPoints: number;
  /** Largest number of points sharing one coordinate. */
  maxStack: number;
  /** Points with a unique coordinate (a measured soma). */
  uniquePoints: number;
  threshold: number;
}

function key(rx: number, ry: number, rz: number): number {
  // Integers in [-20000, 20000]. The mixed key stays below 2^53 so every value
  // is exact and grouping is by true float equality at the chosen quantum.
  return ((rx + 40000) * 80001 + (ry + 40000)) * 80001 + (rz + 40000);
}

export function detectCentroidFill(positions: Float32Array, count: number): CentroidFill {
  const counts = new Map<number, number>();
  const keys = new Float64Array(count);
  for (let i = 0; i < count; i += 1) {
    const rx = Math.round(positions[i * 3] * QUANTUM);
    const ry = Math.round(positions[i * 3 + 1] * QUANTUM);
    const rz = Math.round(positions[i * 3 + 2] * QUANTUM);
    const k = key(rx, ry, rz);
    keys[i] = k;
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }

  const fill = new Float32Array(count);
  const scale = new Float32Array(count);
  let sharedCoordinates = 0;
  let piledPoints = 0;
  let uniquePoints = 0;
  let maxStack = 0;

  for (const n of counts.values()) {
    if (n >= CENTROID_STACK_THRESHOLD) {
      sharedCoordinates += 1;
      piledPoints += n;
      if (n > maxStack) maxStack = n;
    } else {
      uniquePoints += n;
    }
  }

  for (let i = 0; i < count; i += 1) {
    const n = counts.get(keys[i]) ?? 1;
    if (n >= CENTROID_STACK_THRESHOLD) {
      fill[i] = 1;
      // Cap accumulation: the pile contributes roughly what one point would.
      scale[i] = 1 / n;
    } else {
      scale[i] = 1;
    }
  }

  return {
    fill,
    scale,
    distinctCoordinates: counts.size,
    sharedCoordinates,
    piledPoints,
    maxStack,
    uniquePoints,
    threshold: CENTROID_STACK_THRESHOLD,
  };
}
