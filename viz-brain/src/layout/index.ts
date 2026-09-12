// ---------------------------------------------------------------------------
// Brain layout: THE single module that decides where the 166,700 points sit.
//
// HONESTY CONTRACT
// ----------------
// The real MaleCNS v1.0 soma voxel coordinates are NOT available in this
// workspace. They live in neuPrint or in a separate download. Every point
// produced here is therefore a PROCEDURAL RECONSTRUCTION. The UI must always
// display `provenance.coordinateNote` alongside the cloud. Never present these
// positions as measured anatomy.
//
// What CAN be real, if the sidecar files exist:
//   - neuron body ids        (public/data/neuron_ids.txt)
//   - in-degree per neuron   (public/data/in_degree.bin)
//   - cell class per neuron  (public/data/cell_class.bin)
// When those exist the layout is binned by (cell class, in-degree quantile),
// so cluster membership reflects real graph topology even though the 3D
// placement inside a cluster does not.
//
// SWAPPING IN REAL COORDINATES
// ----------------------------
// Call `installAnatomicalPositions(layout, positions)` with a Float32Array of
// 3 * count world-space coordinates. That is the only change required. The
// returned layout carries kind 'anatomical' and a different UI note.
// ---------------------------------------------------------------------------

import { mulberry32, gaussian } from './rng';

export type LayoutKind = 'procedural-synthetic' | 'procedural-topology' | 'anatomical';

export interface LayoutProvenance {
  kind: LayoutKind;
  degreeSource: 'synthetic-power-law' | 'malecns-in-degree';
  classSource: 'synthetic-partition' | 'malecns-cell-class';
  idSource: 'none' | 'malecns-body-id';
  /** Short label for a UI badge. */
  badge: string;
  /** The sentence the UI must show so nobody mistakes this for anatomy. */
  coordinateNote: string;
  /** Longer explanation for the info panel. */
  detail: string;
}

export interface LayoutBin {
  index: number;
  classIndex: number;
  quantile: number;
  /** Members of this bin as a contiguous range into `order`. */
  start: number;
  end: number;
  center: [number, number, number];
  meanDegree: number;
  label: string;
}

export interface BrainLayout {
  count: number;
  seed: number;
  /** xyz interleaved, length 3 * count. */
  positions: Float32Array;
  /** Per point base brightness in 0..1, derived from degree percentile. */
  baseIntensity: Float32Array;
  /** In-degree per point. Synthetic when degreeSource says so. */
  inDegree: Uint32Array;
  /** Cell class index per point, 0..CLASS_COUNT-1. */
  cellClass: Uint8Array;
  /** Body ids per point, or null when no id sidecar was available. */
  ids: string[] | null;
  /** id -> point index. Empty when ids is null. */
  idToIndex: Map<string, number>;
  /** Points sorted by (cellClass, inDegree). Bins are ranges into this. */
  order: Uint32Array;
  bins: LayoutBin[];
  /** Index of the bin each point belongs to. */
  binOf: Uint32Array;
  provenance: LayoutProvenance;
}

export const CLASS_COUNT = 4;
export const QUANTILES = 8;
export const SECTORS = 4;

export const CLASS_LABELS = [
  'central cluster',
  'left lobe cluster',
  'right lobe cluster',
  'rim cluster',
] as const;

/** Class names used when the layout comes from real MaleCNS annotations. */
export const ANATOMICAL_CLASS_LABELS = [
  'optic lobe',
  'central brain',
  'ascending / descending',
  'other',
] as const;

// Region geometry. These are layout constructs, not measured anatomy.
const CENTRAL_RADII: [number, number, number] = [0.58, 0.46, 0.52];
const LOBE_CENTERS: Array<[number, number, number]> = [
  [-0.74, -0.04, 0.06],
  [0.74, -0.04, 0.06],
];
const LOBE_RADII: [number, number, number] = [0.30, 0.32, 0.30];
const RIM_RADII: [number, number, number] = [0.98, 0.72, 1.02];

const GOLDEN = Math.PI * (3 - Math.sqrt(5));

const SYNTHETIC_NOTE =
  'procedural layout, not anatomical coordinates (no soma positions in this workspace)';
const TOPOLOGY_NOTE =
  'procedural layout from real degree and class bins, not anatomical coordinates';
const ANATOMICAL_NOTE = 'anatomical soma coordinates supplied by the caller';

function syntheticDegree(rand: () => number): number {
  // Pareto-ish heavy tail so the degree distribution looks like a connectome
  // rather than a uniform draw. Still synthetic: labelled as such.
  const u = Math.max(1e-6, rand());
  const d = Math.floor(1 + (Math.pow(u, -0.85) - 1) * 2.2);
  return Math.min(d, 4000);
}

function maybeGaussianJitter(rand: () => number, scale: number): number {
  return gaussian(rand) * scale;
}

function regionCenter(classIndex: number, quantile: number, sector: number): [number, number, number] {
  // quantile 0 = lowest degree (outermost), QUANTILES-1 = highest (core).
  const rFrac = 0.42 + 0.58 * (1 - quantile / (QUANTILES - 1));
  const theta = (Math.PI * (sector + 0.5)) / SECTORS;
  const phi = GOLDEN * (quantile * SECTORS + sector);
  const sx = Math.sin(theta) * Math.cos(phi);
  const sy = Math.cos(theta);
  const sz = Math.sin(theta) * Math.sin(phi);

  if (classIndex === 0) {
    return [sx * CENTRAL_RADII[0] * rFrac, sy * CENTRAL_RADII[1] * rFrac, sz * CENTRAL_RADII[2] * rFrac];
  }
  if (classIndex === 1 || classIndex === 2) {
    const c = LOBE_CENTERS[classIndex - 1];
    return [
      c[0] + sx * LOBE_RADII[0] * rFrac,
      c[1] + sy * LOBE_RADII[1] * rFrac,
      c[2] + sz * LOBE_RADII[2] * rFrac,
    ];
  }
  // class 3: outer rim shell, biased to low degree.
  const rimFrac = 0.78 + 0.22 * (1 - quantile / (QUANTILES - 1));
  return [sx * RIM_RADII[0] * rimFrac, sy * RIM_RADII[1] * rimFrac, sz * RIM_RADII[2] * rimFrac];
}

function binRadius(classIndex: number): number {
  if (classIndex === 0) return 0.115;
  if (classIndex < 3) return 0.062;
  return 0.075;
}

export interface LayoutInputs {
  count: number;
  seed?: number;
  ids?: string[] | null;
  inDegree?: Uint32Array | null;
  cellClass?: Uint8Array | null;
}

/**
 * Build the layout. Deterministic for a given (count, seed, inputs).
 */
export function buildProceduralLayout(inputs: LayoutInputs): BrainLayout {
  const count = inputs.count;
  const seed = inputs.seed ?? 7701;
  // `seed` is kept in the signature for a deterministic jitter source; the current
  // layout is bin-contiguous and needs no random draws, so no generator is created
  // here (an unused one would be dead code).
  void seed;

  const haveIds = !!inputs.ids && inputs.ids.length === count;
  const haveDegree = !!inputs.inDegree && inputs.inDegree.length === count;
  const haveClass = !!inputs.cellClass && inputs.cellClass.length === count;

  const inDegree = new Uint32Array(count);
  const cellClass = new Uint8Array(count);
  const ids = haveIds ? (inputs.ids as string[]) : null;

  const degreeRand = mulberry32(seed ^ 0x9e3779b9);
  const classRand = mulberry32(seed ^ 0x85ebca6b);

  for (let i = 0; i < count; i += 1) {
    inDegree[i] = haveDegree ? (inputs.inDegree as Uint32Array)[i] : syntheticDegree(degreeRand);
    cellClass[i] = haveClass ? (inputs.cellClass as Uint8Array)[i] % CLASS_COUNT : Math.floor(classRand() * CLASS_COUNT);
  }

  // Degree quantile per point.
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i += 1) order[i] = i;
  const orderArr = Array.from(order);
  orderArr.sort((a, b) => {
    const ca = cellClass[a];
    const cb = cellClass[b];
    if (ca !== cb) return ca - cb;
    const da = inDegree[a];
    const db = inDegree[b];
    if (da !== db) return da - db;
    return a - b;
  });
  const sorted = Uint32Array.from(orderArr);

  const binOf = new Uint32Array(count);
  const bins: LayoutBin[] = [];
  const positions = new Float32Array(count * 3);
  const baseIntensity = new Float32Array(count);
  const idToIndex = new Map<string, number>();

  // Pass 1: bin membership, contiguous by construction.
  const perClass = Math.floor(count / CLASS_COUNT);
  for (let c = 0; c < CLASS_COUNT; c += 1) {
    for (let q = 0; q < QUANTILES; q += 1) {
      for (let s = 0; s < SECTORS; s += 1) {
        const binIndex = (c * QUANTILES + q) * SECTORS + s;
        const start = Math.min(count, c * perClass + Math.floor((perClass * (q * SECTORS + s)) / (QUANTILES * SECTORS)));
        const end = Math.min(
          count,
          c * perClass + Math.floor((perClass * (q * SECTORS + s + 1)) / (QUANTILES * SECTORS)) || start,
        );
        const center = regionCenter(c, q, s);
        let degreeSum = 0;
        const lo = Math.max(0, start);
        const hi = Math.max(lo, c === CLASS_COUNT - 1 && q === QUANTILES - 1 && s === SECTORS - 1 ? count : end);
        for (let k = lo; k < hi; k += 1) degreeSum += inDegree[sorted[k]];
        bins.push({
          index: binIndex,
          classIndex: c,
          quantile: q,
          start: lo,
          end: hi,
          center,
          meanDegree: hi > lo ? degreeSum / (hi - lo) : 0,
          label: `${CLASS_LABELS[c]} q${q} s${s}`,
        });
      }
    }
  }
  // The last class swallows any remainder so no point is left unplaced.
  bins[bins.length - 1].end = count;

  // Pass 2: positions. Each bin draws from its own seeded stream so that
  // changing one bin cannot shift any other bin's points.
  for (const bin of bins) {
    const bRand = mulberry32((seed ^ (bin.index * 0x27d4eb2d)) >>> 0);
    const [cx, cy, cz] = bin.center;
    const r = binRadius(bin.classIndex);
    for (let k = bin.start; k < bin.end; k += 1) {
      const neuron = sorted[k];
      binOf[neuron] = bin.index;
      const jitter = r * Math.cbrt(bRand());
      const jx = maybeGaussianJitter(bRand, jitter * 0.6);
      const jy = maybeGaussianJitter(bRand, jitter * 0.6);
      const jz = maybeGaussianJitter(bRand, jitter * 0.6);
      positions[neuron * 3] = cx + jx;
      positions[neuron * 3 + 1] = cy + jy;
      positions[neuron * 3 + 2] = cz + jz;
    }
    const span = bin.end - bin.start;
    for (let k = bin.start; k < bin.end; k += 1) {
      const neuron = sorted[k];
      // Position inside the bin doubles as a coarse degree percentile.
      baseIntensity[neuron] = span > 1 ? 0.06 + 0.5 * (k - bin.start) / (span - 1) : 0.2;
    }
  }

  if (ids) {
    for (let i = 0; i < count; i += 1) idToIndex.set(ids[i], i);
  }

  const provenance: LayoutProvenance = {
    kind: haveDegree || haveClass ? 'procedural-topology' : 'procedural-synthetic',
    degreeSource: haveDegree ? 'malecns-in-degree' : 'synthetic-power-law',
    classSource: haveClass ? 'malecns-cell-class' : 'synthetic-partition',
    idSource: haveIds ? 'malecns-body-id' : 'none',
    badge: haveDegree || haveClass ? 'procedural layout (topology binned)' : 'procedural layout',
    coordinateNote: haveDegree || haveClass ? TOPOLOGY_NOTE : SYNTHETIC_NOTE,
    detail:
      'Region names such as central cluster and lobe cluster are layout constructs. ' +
      'They are not measured brain regions and must not be read as anatomy.',
  };

  return {
    count,
    seed,
    positions,
    baseIntensity,
    inDegree,
    cellClass,
    ids,
    idToIndex,
    order: sorted,
    bins,
    binOf,
    provenance,
  };
}

/**
 * The one call needed to replace the procedural reconstruction with measured
 * soma coordinates once they are available.
 */
export function installAnatomicalPositions(layout: BrainLayout, positions: Float32Array): BrainLayout {
  if (positions.length !== layout.count * 3) {
    throw new Error(
      `installAnatomicalPositions: expected ${layout.count * 3} floats, received ${positions.length}`,
    );
  }
  return {
    ...layout,
    positions,
    provenance: {
      ...layout.provenance,
      kind: 'anatomical',
      badge: 'anatomical soma coordinates',
      coordinateNote: ANATOMICAL_NOTE,
      detail: 'Soma positions were supplied by the caller from a measured dataset.',
    },
  };
}

export interface AnatomicalInputs {
  count: number;
  positions: Float32Array;
  inDegree: Uint32Array;
  classIndex: Uint8Array;
  ids: Uint32Array;
  coordinateNote: string;
  detail: string;
  badge: string;
}

/** Degree deciles used to label a point when real coordinates are available. */
export const DEGREE_STEPS = 10;

/**
 * Build the layout from real measured coordinates. Clusters are still reported
 * as (class, in-degree decile) groups, but they no longer decide placement:
 * the measured soma position does.
 */
export function buildAnatomicalLayout(inputs: AnatomicalInputs): BrainLayout {
  const { count, positions, inDegree, classIndex, ids } = inputs;
  if (positions.length !== count * 3) {
    throw new Error(`buildAnatomicalLayout: expected ${count * 3} floats, received ${positions.length}`);
  }

  // Rank based decile without sorting 166700 objects: argsort of the degrees.
  const rank = new Uint32Array(count);
  for (let i = 0; i < count; i += 1) rank[i] = i;
  const rankArr = Array.from(rank);
  rankArr.sort((a, b) => inDegree[a] - inDegree[b] || a - b);
  const decileOf = new Uint8Array(count);
  for (let k = 0; k < count; k += 1) {
    decileOf[rankArr[k]] = Math.min(DEGREE_STEPS - 1, Math.floor((k * DEGREE_STEPS) / count));
  }

  const binCount = CLASS_COUNT * DEGREE_STEPS;
  const binOf = new Uint32Array(count);
  const bins: LayoutBin[] = [];
  const baseIntensity = new Float32Array(count);
  const members: number[][] = Array.from({ length: binCount }, () => []);

  for (let i = 0; i < count; i += 1) {
    const b = classIndex[i] * DEGREE_STEPS + decileOf[i];
    binOf[i] = b;
    members[b].push(i);
  }

  let cursor = 0;
  const order = new Uint32Array(count);
  for (let b = 0; b < binCount; b += 1) {
    const list = members[b];
    list.sort((a, z) => a - z);
    const start = cursor;
    const groupClass = Math.floor(b / DEGREE_STEPS);
    const quantile = b % DEGREE_STEPS;
    let sx = 0;
    let sy = 0;
    let sz = 0;
    let degreeSum = 0;
    for (const n of list) {
      order[cursor] = n;
      cursor += 1;
      sx += positions[n * 3];
      sy += positions[n * 3 + 1];
      sz += positions[n * 3 + 2];
      degreeSum += inDegree[n];
      baseIntensity[n] = 0.05 + 0.55 * (quantile / (DEGREE_STEPS - 1));
    }
    const n = list.length || 1;
    bins.push({
      index: b,
      classIndex: groupClass,
      quantile,
      start,
      end: cursor,
      center: [sx / n, sy / n, sz / n],
      meanDegree: degreeSum / n,
      label: `${ANATOMICAL_CLASS_LABELS[groupClass]} d${quantile}`,
    });
  }

  const idStrings: string[] = new Array(count);
  const idToIndex = new Map<string, number>();
  for (let i = 0; i < count; i += 1) {
    const s = String(ids[i]);
    idStrings[i] = s;
    idToIndex.set(s, i);
  }

  return {
    count,
    seed: 0,
    positions,
    baseIntensity,
    inDegree,
    cellClass: classIndex,
    ids: idStrings,
    idToIndex,
    order,
    bins,
    binOf,
    provenance: {
      kind: 'anatomical',
      degreeSource: 'malecns-in-degree',
      classSource: 'malecns-cell-class',
      idSource: 'malecns-body-id',
      badge: inputs.badge,
      coordinateNote: inputs.coordinateNote,
      detail: inputs.detail,
    },
  };
}

