"""Build the real MaleCNS v1.0 graph from the raw feather sources.

Clean reimplementation of ``flm/scripts/prepare_graph.py``. The output directory
is treated as write-once: anyone may read it, only a verified source build may
create it, and an existing manifest is never silently overwritten.

Integrity first. Every source file is checked against its exact byte count and
sha256 before a single row is read, so a truncated or still-downloading file
fails loudly instead of producing a plausible-looking wrong graph.

Retention rule (frozen, matches the published counts):
  annotations: keep rows where ``superclass`` is non-empty AND ``status`` != 'Glia'
  edges: keep rows where BOTH endpoints are retained neurons
Expected outcome: 166700 neurons, 25582938 directed edges, 124177617 contacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE = ROOT / "cache" / "source"
GRAPH = ROOT / "cache" / "malecns_v1"

DATASET = "MaleCNS v1.0"

# Published source files: local name -> (byte count, sha256)
SOURCES = {
    "annotations.feather": (
        14483314,
        "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    ),
    "edges.feather": (
        1051241946,
        "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    ),
}

EXPECTED_NEURONS = 166700
EXPECTED_EDGES = 25582938
EXPECTED_CONTACTS = 124177617

EDGE_COLUMNS = ["body_pre", "body_post", "weight"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources(source: Path, check_hash: bool = True) -> dict:
    """Refuse to proceed on any missing, short, oversized or altered source."""
    report = {}
    for name, (size, digest) in SOURCES.items():
        path = source / name
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        actual_size = path.stat().st_size
        if actual_size != size:
            raise ValueError(
                f"{name} is {actual_size} bytes, expected exactly {size}. "
                "The download is incomplete or truncated; refusing to build."
            )
        if check_hash:
            actual = sha256(path)
            if actual != digest:
                raise ValueError(
                    f"{name} sha256 is {actual}, expected {digest}. "
                    "Source version mismatch; refusing to build."
                )
        else:
            actual = None
        report[name] = {"bytes": size, "sha256": digest, "verified_sha256": actual}
        print(f"  ok {name}: {size} bytes"
              + (f", sha256 verified" if check_hash else ", sha256 unchecked"), flush=True)
    return report


def retained_ids(source: Path) -> np.ndarray:
    """Annotated, non-glial body IDs, sorted ascending int64."""
    table = feather.read_table(
        source / "annotations.feather", columns=["bodyId", "superclass", "status"]
    )
    body = np.asarray(table["bodyId"], np.int64)
    superclass = table["superclass"].to_pylist()
    status = table["status"].to_pylist()
    keep = np.array([bool(s) for s in superclass])
    keep &= np.array([s != "Glia" for s in status])
    ids = np.sort(body[keep].astype(np.int64))
    if len(ids) != EXPECTED_NEURONS:
        raise ValueError(
            f"Retention produced {len(ids)} neurons, expected {EXPECTED_NEURONS}."
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Retained neuron IDs are not unique.")
    return ids


def edge_arrays(source: Path, ids: np.ndarray):
    """Stream edges in Arrow record batches, keeping edges between retained cells.

    Returns (pre, post, contacts, raw_rows). Rows are read batch by batch so the
    1.05 GB source is never decompressed whole.
    """
    n = len(ids)
    pre_chunks, post_chunks, count_chunks = [], [], []
    rows = 0
    batches = 0
    with pa.memory_map(str(source / "edges.feather"), "r") as mapped:
        reader = pa.ipc.open_file(mapped)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            a, z, w = [
                np.asarray(batch.column(batch.schema.get_field_index(k)))
                for k in EDGE_COLUMNS
            ]
            rows += len(a)
            ai = np.searchsorted(ids, a)
            zi = np.searchsorted(ids, z)
            # searchsorted can return len(ids); clamp before indexing.
            valid = (ai < n) & (zi < n)
            ai_c = np.minimum(ai, n - 1)
            zi_c = np.minimum(zi, n - 1)
            valid &= (ids[ai_c] == a) & (ids[zi_c] == z)
            if valid.any():
                pre_chunks.append(ai_c[valid].astype(np.int32))
                post_chunks.append(zi_c[valid].astype(np.int32))
                count_chunks.append(w[valid].astype(np.int64))
            batches += 1
    pre = np.concatenate(pre_chunks)
    post = np.concatenate(post_chunks)
    contacts = np.concatenate(count_chunks)
    return pre, post, contacts, rows, batches


def build(source: Path = SOURCE, out: Path = GRAPH, check_hash: bool = True,
          force: bool = False) -> dict:
    source = Path(source)
    out = Path(out)
    started = time.time()

    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"{manifest_path} already exists. This directory is write-once; "
            "pass --force only if you deliberately intend to rebuild it."
        )

    print(f"Verifying sources in {source}", flush=True)
    source_report = verify_sources(source, check_hash=check_hash)

    print("Selecting retained neurons", flush=True)
    ids = retained_ids(source)
    print(f"  retained neurons: {len(ids)}", flush=True)

    print("Streaming edges", flush=True)
    t0 = time.time()
    pre, post, contacts, raw_rows, batches = edge_arrays(source, ids)
    keep_seconds = time.time() - t0
    print(f"  raw edge rows: {raw_rows} in {batches} batches ({keep_seconds:.1f}s)", flush=True)
    print(f"  retained directed edges: {len(pre)}", flush=True)

    nnz = int(len(pre))
    if nnz != EXPECTED_EDGES:
        raise ValueError(
            f"Retained edge count is {nnz}, expected exactly {EXPECTED_EDGES}. "
            "Refusing to write a graph that does not match the published release."
        )
    total_contacts = int(contacts.sum())
    if total_contacts != EXPECTED_CONTACTS:
        raise ValueError(
            f"Total synaptic contacts are {total_contacts}, expected exactly "
            f"{EXPECTED_CONTACTS}. Refusing to write."
        )

    # row = postsynaptic, column = presynaptic
    W = sparse.coo_matrix(
        (contacts.astype(np.float32), (post, pre)), shape=(len(ids), len(ids))
    ).tocsr()
    del pre, post, contacts
    if W.nnz != nnz:
        raise ValueError(
            f"CSR has {W.nnz} stored edges but {nnz} rows were retained, so duplicate "
            "directed edges exist. Inspect before proceeding."
        )

    # Rows sum to 1: each weight is divided by that neuron's total incoming contacts.
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    W.data /= np.repeat(np.maximum(row_sums, 1.0), np.diff(W.indptr)).astype(np.float32)
    W.sort_indices()

    out.mkdir(parents=True, exist_ok=True)
    arrays = {"ids": ids, "data": W.data, "indices": W.indices, "indptr": W.indptr}
    digests = {}
    for name, arr in arrays.items():
        path = out / f"{name}.npy"
        np.save(path, arr)
        digests[path.name] = sha256(path)
        print(f"  wrote {path.name} {arr.dtype} {arr.shape} "
              f"{path.stat().st_size} bytes", flush=True)

    manifest = {
        "release": DATASET,
        "neurons": int(len(ids)),
        "directed_edges": int(W.nnz),
        "synaptic_contacts": total_contacts,
        "raw_edge_rows": int(raw_rows),
        "retention": (
            "Annotated nonempty superclass, excluding status Glia; both endpoints retained."
        ),
        "matrix_orientation": "row=postsynaptic; column=presynaptic",
        "weights": "Unsigned contact counts, divided by total incoming contacts per neuron.",
        "biological_dynamics": False,
        "source_files": {
            name: {
                "bytes": info["bytes"],
                "sha256": info["sha256"],
                "sha256_verified_at_build": bool(info["verified_sha256"]),
            }
            for name, info in source_report.items()
        },
        "build_seconds": round(time.time() - started, 1),
        "arrays": digests,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(
        {k: manifest[k] for k in
         ["release", "neurons", "directed_edges", "synaptic_contacts", "raw_edge_rows"]}
    ), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=GRAPH)
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if a manifest already exists")
    parser.add_argument("--no-hash", action="store_true",
                        help="check byte counts but skip source sha256 (development only)")
    args = parser.parse_args()
    build(args.source, args.out, check_hash=not args.no_hash, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
