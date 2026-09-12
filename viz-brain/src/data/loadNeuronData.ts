// Loader for the real MaleCNS metadata emitted by scripts/extract_real_layout.py.
// Read-only: this module never writes anything back to the dataset.

export interface NeuronDataMeta {
  dataset: string;
  count: number;
  edges: number;
  coordinateSource: string;
  coordinateNote: string;
  coordinateField: string;
  accuracy: {
    neuronsWithSomaAnnotation: number;
    neuronsFilledFromGroupCentroid: number;
    matchingRule: string;
  };
  degreeSource: string;
  classSource: string;
  classTable: string[];
  rawCentre: [number, number, number];
  uniformScale: number;
  sources: Record<string, string>;
  files: Record<string, { dtype: string; shape: number[] }>;
}

export interface NeuronData {
  meta: NeuronDataMeta;
  count: number;
  /** xyz interleaved, world units, length 3 * count. */
  positions: Float32Array;
  inDegree: Uint32Array;
  classIndex: Uint8Array;
  /** Body ids in canonical MaleCNS order. */
  ids: Uint32Array;
  /** id string -> point index. Built once, used to resolve frame.sampled_ids. */
  idToIndex: Map<string, number>;
}

/**
 * The binary sidecars live in public/data, so they are served from
 * `<base>data/`. Resolving them against import.meta.url would look for
 * src/data/data in dev and assets/data in a build, which never exists, so the
 * base is resolved against the page instead. tsconfig does not pull in
 * vite/client, so import.meta.env is not typed here and window.location is
 * used deliberately.
 */
// The real MaleCNS metadata lives in a `data/` directory served from the site root.
//
// A page-relative URL resolves against window.location.href, which breaks as soon as
// the component is mounted on a nested route (a lesson at /lesson/fly would look for
// /lesson/fly/data/...). Both Vite and Next serve `public/` at the root, so an
// absolute path behaves identically in the standalone harness and inside the app.
// The override exists for embedding the component somewhere else entirely.
const DATA_BASE = new URL(
  (globalThis as { __FLYLINGO_DATA_BASE__?: string }).__FLYLINGO_DATA_BASE__ ?? '/data/',
  window.location.origin,
);

async function fetchBinary(file: string): Promise<ArrayBuffer> {
  const res = await fetch(new URL(file, DATA_BASE).toString());
  if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`);
  return res.arrayBuffer();
}

export async function loadNeuronData(): Promise<NeuronData> {
  const metaRes = await fetch(new URL('layout_meta.json', DATA_BASE).toString());
  if (!metaRes.ok) throw new Error(`layout_meta.json: HTTP ${metaRes.status}`);
  const meta = (await metaRes.json()) as NeuronDataMeta;

  const [posBuf, degBuf, clsBuf, idBuf] = await Promise.all([
    fetchBinary('soma_positions.f32'),
    fetchBinary('in_degree.u32'),
    fetchBinary('class_index.u8'),
    fetchBinary('neuron_ids.u32'),
  ]);

  const count = meta.count;
  const positions = new Float32Array(posBuf);
  const inDegree = new Uint32Array(degBuf);
  const classIndex = new Uint8Array(clsBuf);
  const ids = new Uint32Array(idBuf);

  if (positions.length !== count * 3) {
    throw new Error(`soma_positions.f32 has ${positions.length} floats, expected ${count * 3}`);
  }
  if (inDegree.length !== count || classIndex.length !== count || ids.length !== count) {
    throw new Error('metadata array length does not match meta.count');
  }

  const idToIndex = new Map<string, number>();
  for (let i = 0; i < count; i += 1) idToIndex.set(String(ids[i]), i);

  return { meta, count, positions, inDegree, classIndex, ids, idToIndex };
}
