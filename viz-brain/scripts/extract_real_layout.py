"""
Extract real connectome metadata for the BrainCloud harness.

Reads, read-only:
  D:/Projects/flylingo/cache/malecns_v1/ids.npy      canonical neuron ids, order is authoritative
  D:/Projects/flylingo/cache/malecns_v1/indptr.npy   CSR row pointers, row = postsynaptic
  D:/Projects/flylingo/cache/malecns_v1/manifest.json
  D:/Projects/flylingo/cache/source/annotations.feather   bodyId, somaLocation, superclass

Writes into the harness public/data directory:
  layout_meta.json      provenance, counts, source hashes, class table
  soma_positions.f32    count * 3 float32, raw dataset units
  in_degree.u32         count uint32
  class_index.u8        count uint8
  neuron_ids.u32        count uint32 (MaleCNS body ids; all fit in uint32)

Never writes anywhere under cache/.
"""

import hashlib
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

ROOT = "D:/Projects/flylingo"
MALECNS = os.path.join(ROOT, "cache", "malecns_v1")
ANNOT = os.path.join(ROOT, "cache", "source", "annotations.feather")
OUT = os.path.join(ROOT, "viz-brain", "public", "data")

# superclass values are the measured annotation vocabulary. Collapse them into a
# small number of layout classes so the shader can colour by group. The mapping
# is recorded in layout_meta.json so nothing is hidden.
CLASS_TABLE = [
    ("optic_lobe", ("ol_intrinsic", "ol_sensory")),
    ("central_brain", ("cb_intrinsic", "cb_sensory", "cb_sensory_tbc", "cb_motor",
                       "cb_efferent", "cb_endocrine", "visual_projection",
                       "visual_projection_tbc", "visual_centrifugal")),
    ("ascending_descending", ("ascending_neuron", "descending_neuron",
                              "descending_neuron_tbc", "sensory_ascending",
                              "sensory_ascending_tbc", "sensory_descending",
                              "efferent_ascending", "efferent_descending")),
    ("other", ()),
]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    ids = np.load(os.path.join(MALECNS, "ids.npy"))
    indptr = np.load(os.path.join(MALECNS, "indptr.npy"))
    assert ids.dtype == np.int64 and indptr.dtype in (np.int64, np.int32)
    count = int(ids.shape[0])
    assert indptr.shape[0] == count + 1, "indptr does not match ids"

    # row = postsynaptic, so the row width is the in-degree of that neuron.
    in_degree = np.diff(indptr).astype(np.uint32)
    print(f"neurons={count} total_edges={int(in_degree.sum())}")
    print(f"in_degree min={int(in_degree.min())} median={int(np.median(in_degree))} "
          f"max={int(in_degree.max())}")

    with pa.memory_map(ANNOT, "r") as mm:
        ann = feather.read_table(mm, columns=["bodyId", "somaLocation", "superclass"])
    ann_ids = ann["bodyId"].to_numpy()
    loc = ann["somaLocation"].to_pylist()
    sup = [x.as_py() for x in ann["superclass"]]

    lookup = {}
    for i, bid in enumerate(ann_ids):
        lookup[int(bid)] = i

    positions = np.full((count, 3), np.nan, dtype=np.float32)
    class_index = np.full(count, 3, dtype=np.uint8)
    have_pos = np.zeros(count, dtype=bool)

    sup_to_class = {}
    for ci, (_, names) in enumerate(CLASS_TABLE):
        for n in names:
            sup_to_class[n] = ci

    matched = 0
    for i in range(count):
        j = lookup.get(int(ids[i]))
        if j is None:
            continue
        matched += 1
        s = sup[j]
        if s in sup_to_class:
            class_index[i] = sup_to_class[s]
        a = loc[j]
        if a is not None and len(a) == 3:
            positions[i] = (float(a[0]), float(a[1]), float(a[2]))
            have_pos[i] = True

    n_pos = int(have_pos.sum())
    print(f"annotation matched={matched}/{count} with_soma_position={n_pos}")

    if n_pos == 0:
        raise SystemExit("no soma positions found, refusing to write a fake coordinate file")

    # Fill the neurons that have no soma annotation with the centroid of their
    # (class, degree decile) group. Those points are flagged in the metadata so
    # the UI can say how many are interpolated rather than leave it implied.
    filled = ~have_pos
    if filled.any():
        deg = in_degree.astype(np.float64)
        decile = np.clip((np.argsort(np.argsort(deg)) * 10 // count), 0, 9).astype(np.uint8)
        group = class_index.astype(np.int64) * 10 + decile.astype(np.int64)
        for g in np.unique(group[filled]):
            sel = group == g
            known = sel & have_pos
            if known.any():
                positions[sel] = positions[known].mean(axis=0)
        # Any group with no known members at all falls back to the global centroid.
        still = np.isnan(positions[:, 0])
        if still.any():
            positions[still] = positions[have_pos].mean(axis=0)
    n_filled = int(filled.sum())

    # Centre and scale into world units. Uniform scale only, no per-axis stretch.
    centre = positions.mean(axis=0)
    positions = positions - centre
    span = float(np.abs(positions).max())
    scale = 1.0 / span
    world = (positions * scale).astype(np.float32)
    print(f"raw_span={span:.1f} world_span={float(np.abs(world).max()):.4f}")

    world.astype("<f4").tofile(os.path.join(OUT, "soma_positions.f32"))
    in_degree.astype("<u4").tofile(os.path.join(OUT, "in_degree.u32"))
    class_index.astype("u1").tofile(os.path.join(OUT, "class_index.u8"))
    ids.astype("<u4").tofile(os.path.join(OUT, "neuron_ids.u32"))

    manifest_path = os.path.join(MALECNS, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        upstream_manifest = json.load(fh)

    meta = {
        "dataset": "MaleCNS v1.0",
        "count": count,
        "edges": int(in_degree.sum()),
        "coordinateSource": "real soma voxel coordinates",
        "coordinateNote": "measured soma voxel coordinates from MaleCNS annotations.feather (somaLocation), centred and uniformly scaled to world units",
        "coordinateField": "somaLocation",
        "accuracy": {
            "neuronsWithSomaAnnotation": n_pos,
            "neuronsFilledFromGroupCentroid": n_filled,
            "matchingRule": "centroid of (class, in-degree decile) group",
        },
        "degreeSource": "malecns in-degree from CSR indptr diff (row = postsynaptic)",
        "classSource": "malecns superclass, collapsed into 4 layout classes",
        "classTable": [name for name, _ in CLASS_TABLE],
        "classMapping": {k: v for k, v in sorted(sup_to_class.items())},
        "rawCentre": [float(x) for x in centre],
        "uniformScale": scale,
        "sources": {
            "ids_npy": sha256_of(os.path.join(MALECNS, "ids.npy")),
            "indptr_npy": sha256_of(os.path.join(MALECNS, "indptr.npy")),
            "annotations_feather": sha256_of(ANNOT),
        },
        "upstreamManifest": upstream_manifest,
        "files": {
            "soma_positions.f32": {"dtype": "float32", "shape": [count, 3]},
            "in_degree.u32": {"dtype": "uint32", "shape": [count]},
            "class_index.u8": {"dtype": "uint8", "shape": [count]},
            "neuron_ids.u32": {"dtype": "uint32", "shape": [count]},
        },
    }
    with open(os.path.join(OUT, "layout_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("wrote", OUT)
    for name in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, name)
        print(f"  {name:24s} {os.path.getsize(p):>10d} bytes")


if __name__ == "__main__":
    main()
