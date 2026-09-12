import type { BrainLayout } from '../layout';

export interface LiveFrame {
  state: number[];
  spikes: number[];
  sampledIds: string[];
  activeFraction: number;
  stateRms: number;
}

export interface LiveMapping {
  /** Cloud point index per live slot, or -1 when the slot could not be resolved. */
  indices: Int32Array;
  slots: number;
  resolved: number;
  unresolved: number;
  basis: 'sampled-ids' | 'synthetic-slots';
}

/**
 * Decide which cloud points the live slots sit on.
 *
 * When the frame carries sampled_ids, each id is looked up in the real MaleCNS
 * body id table, so a live slot renders at that neuron's measured soma.
 * Identifiers that are not present in the cloud are reported as unresolved
 * rather than being silently snapped onto a random point.
 *
 * With no sampled ids (the pre-socket synthetic case) slots are placed on a
 * deterministic spread of cloud points and the basis is reported as
 * synthetic-slots so the UI can say the placement is arbitrary.
 */
export function buildLiveMapping(layout: BrainLayout, sampledIds: string[]): LiveMapping {
  const slots = sampledIds.length > 0 ? sampledIds.length : 512;
  const indices = new Int32Array(slots);
  let resolved = 0;

  if (sampledIds.length > 0) {
    for (let i = 0; i < slots; i += 1) {
      const idx = layout.idToIndex.get(sampledIds[i]);
      indices[i] = idx === undefined ? -1 : idx;
      if (idx !== undefined) resolved += 1;
    }
    return {
      indices,
      slots,
      resolved,
      unresolved: slots - resolved,
      basis: 'sampled-ids',
    };
  }

  const step = Math.max(1, Math.floor(layout.count / slots));
  for (let i = 0; i < slots; i += 1) indices[i] = layout.order[(i * step) % layout.count];
  return { indices, slots, resolved: 0, unresolved: slots, basis: 'synthetic-slots' };
}

/** Cheap signature so a new array with identical contents does not rebuild the mapping. */
export function mappingSignature(frame: LiveFrame): string {
  const ids = frame.sampledIds;
  if (ids.length === 0) return 'none';
  let h = 0;
  const probe = Math.min(ids.length, 8);
  for (let i = 0; i < probe; i += 1) {
    const s = ids[i];
    for (let k = 0; k < s.length; k += 1) h = (Math.imul(h, 31) + s.charCodeAt(k)) | 0;
  }
  return `${ids.length}:${h}:${ids[ids.length - 1]}`;
}
