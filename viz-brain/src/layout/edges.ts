// Sampled hairline edge set for the connectome cloud.
//
// HONESTY CONTRACT, READ THIS FIRST
// ---------------------------------
// THESE ARE NOT MEASURED CONNECTOME ADJACENCY. The MaleCNS v1.0 graph is
// 25,582,938 directed connections and that edge list is not in this workspace.
// Even if it were, 25 million line segments cannot be drawn in real time. What
// is drawn here is a legible illustration:
//
//   * the hubs are REAL: the highest in-degree neurons, selected through the
//     layout's (cell class, in-degree decile) bins, which come from the real
//     in_degree sidecar;
//   * each hub is joined to its 1 or 2 NEAREST same-class somas in the real
//     measured coordinate set, so links stay inside a cell class and read as
//     structure rather than spaghetti across the brain;
//   * one hub in ten also gets a single longer same-class link, taken from the
//     same hub pool by a seeded hash, to hint at long-range structure.
//
// The pairing is a deterministic nearest-neighbour rule over measured soma
// positions. It is NOT a synapse, it is NOT a traced tract, and it must never
// be reported as either. Same layout in, same edges out: no Math.random here.

import type { BrainLayout } from './index';
import { QUANTILES } from './index';

/** Hubs drawn. 4,200 of 166,700 points, about the top 2.5% by in-degree. */
export const EDGE_HUBS = 4200;
/** Hard cap on drawn segments, 2 per hub plus long links. */
export const EDGE_CAP = 12000;
/** Neighbour search radius in world units. The cloud spans about [-1, 1]. */
export const EDGE_RADIUS = 0.075;
/** Below this the two somas share a voxel (group-centroid duplicates). */
export const EDGE_MIN_LENGTH = 0.004;
export const EDGE_PARTNERS = 2;
export const EDGE_LONG_EVERY = 10;
export const EDGE_LONG_MIN = 0.18;
export const EDGE_LONG_MAX = 0.75;

export interface EdgeSet {
  /** 6 floats per edge: x1, y1, z1, x2, y2, z2. */
  positions: Float32Array;
  /** Per edge brightness weight, 0..1, from the hub's degree percentile. */
  weights: Float32Array;
  count: number;
  hubs: number;
  longRange: number;
  meanLength: number;
  radius: number;
  minLength: number;
  /** One line, safe to print in a report, stating what was actually derived. */
  source: string;
}

function hash01(n: number): number {
  let x = (Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  x ^= x >>> 16;
  return x / 4294967296;
}

export function buildSampledEdges(layout: BrainLayout): EdgeSet {
  const count = layout.count;
  const pos = layout.positions;
  const cls = layout.cellClass;
  const deg = layout.inDegree;

  // -------------------------------------------------------------------------
  // Uniform grid over the measured positions. Built once per layout, O(count).
  // -------------------------------------------------------------------------
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < count; i += 1) {
    const x = pos[i * 3];
    const y = pos[i * 3 + 1];
    const z = pos[i * 3 + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const cell = EDGE_RADIUS;
  const nx = Math.max(1, Math.floor((maxX - minX) / cell) + 1);
  const ny = Math.max(1, Math.floor((maxY - minY) / cell) + 1);
  const nz = Math.max(1, Math.floor((maxZ - minZ) / cell) + 1);
  const cells = nx * ny * nz;

  const cellOf = new Int32Array(count);
  for (let i = 0; i < count; i += 1) {
    let ix = Math.floor((pos[i * 3] - minX) / cell);
    let iy = Math.floor((pos[i * 3 + 1] - minY) / cell);
    let iz = Math.floor((pos[i * 3 + 2] - minZ) / cell);
    if (ix < 0) ix = 0;
    else if (ix >= nx) ix = nx - 1;
    if (iy < 0) iy = 0;
    else if (iy >= ny) iy = ny - 1;
    if (iz < 0) iz = 0;
    else if (iz >= nz) iz = nz - 1;
    cellOf[i] = ix + iy * nx + iz * nx * ny;
  }
  const starts = new Int32Array(cells + 1);
  for (let i = 0; i < count; i += 1) starts[cellOf[i] + 1] += 1;
  for (let c = 0; c < cells; c += 1) starts[c + 1] += starts[c];
  const members = new Int32Array(count);
  const fill = starts.slice(0, cells);
  for (let i = 0; i < count; i += 1) {
    const c = cellOf[i];
    members[fill[c]] = i;
    fill[c] += 1;
  }

  // -------------------------------------------------------------------------
  // Hubs: real high in-degree neurons from the top two degree deciles.
  // -------------------------------------------------------------------------
  const cand: number[] = [];
  for (let b = 0; b < layout.bins.length; b += 1) {
    const bin = layout.bins[b];
    if (bin.quantile < QUANTILES - 2) continue;
    for (let k = bin.start; k < bin.end; k += 1) cand.push(layout.order[k]);
  }
  cand.sort((a, b) => deg[b] - deg[a] || a - b);
  const hubCount = Math.min(EDGE_HUBS, cand.length);

  const positions = new Float32Array(EDGE_CAP * 6);
  const weights = new Float32Array(EDGE_CAP);
  const seen = new Set<number>();
  const base = layout.baseIntensity;
  const min2 = EDGE_MIN_LENGTH * EDGE_MIN_LENGTH;
  const rad2 = EDGE_RADIUS * EDGE_RADIUS;
  const longMin2 = EDGE_LONG_MIN * EDGE_LONG_MIN;
  const longMax2 = EDGE_LONG_MAX * EDGE_LONG_MAX;
  const best = new Float64Array(EDGE_PARTNERS);
  const bestIdx = new Int32Array(EDGE_PARTNERS);

  let n = 0;
  let longRange = 0;
  let lenSum = 0;
  const addEdge = (a: number, b: number): boolean => {
    if (n >= EDGE_CAP) return false;
    const lo = a < b ? a : b;
    const hi = a < b ? b : a;
    if (hi === lo) return false;
    const key = lo * count + hi;
    if (seen.has(key)) return false;
    seen.add(key);
    const o = n * 6;
    positions[o] = pos[a * 3];
    positions[o + 1] = pos[a * 3 + 1];
    positions[o + 2] = pos[a * 3 + 2];
    positions[o + 3] = pos[b * 3];
    positions[o + 4] = pos[b * 3 + 1];
    positions[o + 5] = pos[b * 3 + 2];
    weights[n] = 0.35 + 0.65 * Math.min(1, Math.max(0, base[a]));
    const dx = pos[b * 3] - pos[a * 3];
    const dy = pos[b * 3 + 1] - pos[a * 3 + 1];
    const dz = pos[b * 3 + 2] - pos[a * 3 + 2];
    lenSum += Math.sqrt(dx * dx + dy * dy + dz * dz);
    n += 1;
    return true;
  };

  for (let h = 0; h < hubCount && n < EDGE_CAP; h += 1) {
    const a = cand[h];
    const ax = pos[a * 3];
    const ay = pos[a * 3 + 1];
    const az = pos[a * 3 + 2];
    const acls = cls[a];
    let ix = Math.floor((ax - minX) / cell);
    let iy = Math.floor((ay - minY) / cell);
    let iz = Math.floor((az - minZ) / cell);
    if (ix < 0) ix = 0;
    else if (ix >= nx) ix = nx - 1;
    if (iy < 0) iy = 0;
    else if (iy >= ny) iy = ny - 1;
    if (iz < 0) iz = 0;
    else if (iz >= nz) iz = nz - 1;

    for (let k = 0; k < EDGE_PARTNERS; k += 1) {
      best[k] = Infinity;
      bestIdx[k] = -1;
    }
    for (let dx = -1; dx <= 1; dx += 1) {
      const jx = ix + dx;
      if (jx < 0 || jx >= nx) continue;
      for (let dy = -1; dy <= 1; dy += 1) {
        const jy = iy + dy;
        if (jy < 0 || jy >= ny) continue;
        for (let dz = -1; dz <= 1; dz += 1) {
          const jz = iz + dz;
          if (jz < 0 || jz >= nz) continue;
          const c = jx + jy * nx + jz * nx * ny;
          const s0 = starts[c];
          const s1 = starts[c + 1];
          for (let m = s0; m < s1; m += 1) {
            const j = members[m];
            if (j === a || cls[j] !== acls) continue;
            const ex = pos[j * 3] - ax;
            const ey = pos[j * 3 + 1] - ay;
            const ez = pos[j * 3 + 2] - az;
            const d2 = ex * ex + ey * ey + ez * ez;
            if (d2 < min2 || d2 > rad2) continue;
            if (d2 < best[0]) {
              best[1] = best[0];
              bestIdx[1] = bestIdx[0];
              best[0] = d2;
              bestIdx[0] = j;
            } else if (d2 < best[1]) {
              best[1] = d2;
              bestIdx[1] = j;
            }
          }
        }
      }
    }
    for (let k = 0; k < EDGE_PARTNERS; k += 1) {
      if (bestIdx[k] >= 0) addEdge(a, bestIdx[k]);
    }

    if (h % EDGE_LONG_EVERY === 0 && hubCount > 0) {
      for (let t = 0; t < 6; t += 1) {
        const pick = (hash01(h * 131 + t * 7 + 1) * hubCount) | 0;
        const j = cand[pick];
        if (j === a || cls[j] !== acls) continue;
        const ex = pos[j * 3] - ax;
        const ey = pos[j * 3 + 1] - ay;
        const ez = pos[j * 3 + 2] - az;
        const d2 = ex * ex + ey * ey + ez * ez;
        if (d2 < longMin2 || d2 > longMax2) continue;
        if (addEdge(a, j)) longRange += 1;
        break;
      }
    }
  }

  const outCount = n;
  return {
    positions: positions.slice(0, outCount * 6),
    weights: weights.slice(0, outCount),
    count: outCount,
    hubs: hubCount,
    longRange,
    meanLength: outCount > 0 ? lenSum / outCount : 0,
    radius: EDGE_RADIUS,
    minLength: EDGE_MIN_LENGTH,
    source:
      'sampled illustration: nearest same-class soma pairs for the highest in-degree hubs, not measured adjacency',
  };
}
