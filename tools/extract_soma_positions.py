"""Extract real MaleCNS soma coordinates for the brain cloud, correctly.

The previous extraction collapsed 166,700 neurons onto only 40 distinct coordinates,
so the cloud rendered as a handful of dots. This reads the source of truth and asserts
the result is actually distinct before writing anything.

Reads, read-only:
  cache/malecns_v1/ids.npy          canonical neuron ids (order is authoritative)
  cache/malecns_v1/indptr.npy       CSR row pointers (row = postsynaptic) for in-degree
  cache/malecns_v1/manifest.json
  cache/source/annotations.feather  bodyId, somaLocation, superclass

Writes into duolingo-clone/public/data:
  soma_positions.f32   count * 3 float32, centred and uniformly scaled
  in_degree.u32        count uint32
  class_index.u8       count uint8
  neuron_ids.u32       count uint32
  layout_meta.json     provenance, counts, source hashes, class table
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

ROOT = Path(r"D:\Projects\flylingo")
MALECNS = ROOT / "cache" / "malecns_v1"
ANNOT = ROOT / "cache" / "source" / "annotations.feather"
OUT = ROOT / "duolingo-clone" / "public" / "data"

# Layout classes, collapsed from the 19 MaleCNS superclasses.
CLASS_TABLE = ["optic_lobe", "central_brain", "ascending_descending", "other"]
OL = {"ol_intrinsic", "ol_sensory"}
CB = {"cb_efferent", "cb_endocrine", "cb_intrinsic", "cb_motor", "cb_sensory",
      "cb_sensory_tbc", "visual_centrifugal", "visual_projection", "visual_projection_tbc"}
AD = {"ascending_neuron", "descending_neuron", "descending_neuron_tbc",
      "efferent_ascending", "efferent_descending", "sensory_ascending",
      "sensory_ascending_tbc", "sensory_descending"}


def class_of(superclass: str) -> int:
    if superclass in OL:
        return 0
    if superclass in CB:
        return 1
    if superclass in AD:
        return 2
    return 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ids = np.load(MALECNS / "ids.npy")
    indptr = np.load(MALECNS / "indptr.npy")
    manifest = json.loads((MALECNS / "manifest.json").read_text())
    n = len(ids)
    print(f"canonical neurons: {n}")
    print(f"edges:             {manifest['directed_edges']}")

    in_degree = np.diff(indptr).astype(np.uint32)
    print(f"in-degree: min {in_degree.min()} max {in_degree.max()} mean {in_degree.mean():.2f}")

    table = feather.read_table(ANNOT, columns=["bodyId", "somaLocation", "superclass"])
    body = np.asarray(table.column("bodyId").to_pylist(), np.int64)
    soma = table.column("somaLocation").to_pylist()
    sup = table.column("superclass").to_pylist()
    print(f"annotation rows:   {len(body)}")

    by_id_loc = {}
    by_id_sup = {}
    for b, loc, s in zip(body, soma, sup):
        if loc is not None:
            by_id_loc[int(b)] = (float(loc[0]), float(loc[1]), float(loc[2]))
        if s:
            by_id_sup[int(b)] = s

    coords = np.full((n, 3), np.nan, np.float64)
    have = 0
    for i, nid in enumerate(ids):
        loc = by_id_loc.get(int(nid))
        if loc is not None:
            coords[i] = loc
            have += 1
    print(f"neurons with a measured soma: {have} of {n}")

    classes = np.zeros(n, np.uint8)
    for i, nid in enumerate(ids):
        classes[i] = class_of(by_id_sup.get(int(nid), ""))
    print("class counts:", np.bincount(classes, minlength=4).tolist())

    # Fill the rest from the centroid of their (class, in-degree decile) group.
    need = np.isnan(coords[:, 0])
    filled = int(need.sum())
    keys = classes.astype(np.int64) * 10 + np.minimum(
        (in_degree / max(1, in_degree.max()) * 10).astype(np.int64), 9)
    centroids = {}
    for key in np.unique(keys[~need]):
        sel = (~need) & (keys == key)
        centroids[int(key)] = coords[sel].mean(axis=0)
    for key in np.unique(keys[need]):
        sel = need & (keys == key)
        c = centroids.get(int(key))
        if c is None:
            c = coords[~need].mean(axis=0)
        coords[sel] = c
    print(f"filled from (class, degree decile) centroid: {filled}")

    assert not np.isnan(coords).any(), "coordinates still contain NaN"

    distinct = len(np.unique(coords, axis=0))
    print(f"distinct coordinates: {distinct} of {n}")
    if distinct < n * 0.5:
        print(f"REFUSING to write: only {distinct} distinct positions for {n} neurons. "
              f"The cloud would render as a scatter of dots, not a brain.", file=sys.stderr)
        return 1

    # Centre on the midpoint of the measured extent and scale uniformly into ~[-1, 1].
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    centre = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    scale = 2.0 / extent if extent > 0 else 1.0
    scaled = ((coords - centre) * scale).astype(np.float32)
    print(f"world extent: min {np.round(scaled.min(0),3)} max {np.round(scaled.max(0),3)}")

    OUT.mkdir(parents=True, exist_ok=True)
    scaled.tofile(OUT / "soma_positions.f32")
    in_degree.astype(np.uint32).tofile(OUT / "in_degree.u32")
    classes.astype(np.uint8).tofile(OUT / "class_index.u8")
    ids.astype(np.uint32).tofile(OUT / "neuron_ids.u32")

    meta = {
        "dataset": "MaleCNS v1.0",
        "count": n,
        "edges": int(manifest["directed_edges"]),
        "coordinateSource": "real soma voxel coordinates",
        "coordinateNote": (
            "measured soma voxel coordinates from MaleCNS annotations.feather "
            "(somaLocation), centred and uniformly scaled to world units"),
        "coordinateField": "somaLocation",
        "accuracy": {
            "neuronsWithSomaAnnotation": have,
            "neuronsFilledFromGroupCentroid": filled,
            "matchingRule": "centroid of (class, in-degree decile) group",
            "distinctCoordinates": distinct,
        },
        "degreeSource": "malecns in-degree from CSR indptr diff (row = postsynaptic)",
        "classSource": "malecns superclass, collapsed into 4 layout classes",
        "classTable": CLASS_TABLE,
        "rawCentre": [float(x) for x in centre],
        "uniformScale": float(scale),
        "sources": {
            "ids_npy": sha256(MALECNS / "ids.npy"),
            "indptr_npy": sha256(MALECNS / "indptr.npy"),
            "annotations_feather": sha256(ANNOT),
        },
        "files": {
            "soma_positions.f32": {"dtype": "float32", "shape": [n, 3]},
            "in_degree.u32": {"dtype": "uint32", "shape": [n]},
            "class_index.u8": {"dtype": "uint8", "shape": [n]},
            "neuron_ids.u32": {"dtype": "uint32", "shape": [n]},
        },
    }
    (OUT / "layout_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("\nwrote:", ", ".join(sorted(p.name for p in OUT.iterdir())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
