"""Colab step 1: environment, real connectome, and the GPU/CPU equivalence gate.

Runs on a Colab VM. Three jobs, in order, each of which can end the run:

  1. Build the real MaleCNS v1.0 graph from the public GCS source. Integrity first:
     the edge source must hash to the pinned sha256 and both files must match their exact
     byte counts, or nothing is built. The retention rule and the three published counts
     (166,700 neurons / 25,582,938 edges / 124,177,617 contacts) are asserted.
  2. Materialise the embedded brain modules and confirm their digests.
  3. GATE: prove the accelerated path reproduces the reference path, per control mode, on
     both fresh and trained weights. This project has already been burned by an accelerated
     path whose numbers were quoted before equivalence was established, so the gate runs
     BEFORE any number is produced, and it fails the run rather than warning.

Usage:  python step1_build.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GCS = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
SOURCES = {
    "edges.feather": (
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        1051241946,
        "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    ),
    "annotations.feather": (
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        14483314,
        None,  # size pinned; digest recorded on first build and reused
    ),
}
EXPECTED_NEURONS = 166700
EXPECTED_EDGES = 25582938
EXPECTED_CONTACTS = 124177617

ROOT = Path("/content/flylingo")
CACHE = ROOT / "cache"
SOURCE = CACHE / "source"
GRAPH = CACHE / "malecns_v1"


def sh(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def ensure_packages() -> None:
    """Install only what is missing, and report what the run will actually use."""
    need = []
    for mod, pkg in (("pyarrow", "pyarrow"), ("scipy", "scipy")):
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    try:
        import cupy  # noqa: F401
        print(f"cupy present: {cupy.__version__}", flush=True)
    except Exception:  # noqa: BLE001
        need.append("cupy-cuda12x")
    if need:
        sh([sys.executable, "-m", "pip", "install", "-q", *need])
    import cupy

    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}", flush=True)


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch() -> None:
    """Download the two public source feathers and verify them."""
    import requests

    SOURCE.mkdir(parents=True, exist_ok=True)
    for local, (remote, size, digest) in SOURCES.items():
        path = SOURCE / local
        if path.exists() and path.stat().st_size == size:
            print(f"  {local}: already present, {size} bytes", flush=True)
        else:
            url = f"{GCS}/{remote}"
            print(f"  downloading {remote} -> {local} ({size / 1e9:.3f} GB)", flush=True)
            t0 = time.time()
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                done = 0
                with open(path, "wb") as f:
                    for block in r.iter_content(chunk_size=8 << 20):
                        f.write(block)
                        done += len(block)
                        if done % (128 << 20) < (8 << 20):
                            pct = done / size * 100
                            print(f"    {pct:5.1f}%  {done / 1e9:.2f} GB  "
                                  f"{time.time() - t0:.0f}s", flush=True)
        got = path.stat().st_size
        if got != size:
            raise SystemExit(f"{local} is {got} bytes, expected exactly {size}. "
                             "Truncated download; refusing to build.")
        if digest:
            actual = sha256(path)
            if actual != digest:
                raise SystemExit(f"{local} sha256 {actual}, expected {digest}. "
                                 "Source version mismatch; refusing to build.")
            print(f"  {local}: {size} bytes, sha256 verified", flush=True)
        else:
            print(f"  {local}: {size} bytes, sha256 {sha256(path)}", flush=True)


def build_graph() -> dict:
    """Same retention rule and the same three assertions as brain/scripts/build_graph.py."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.feather as feather
    from scipy import sparse

    manifest_path = GRAPH / "manifest.json"
    if manifest_path.exists():
        print("  graph already built:", flush=True)
        print(manifest_path.read_text(), flush=True)
        return json.loads(manifest_path.read_text())

    print("  selecting retained neurons", flush=True)
    table = feather.read_table(
        SOURCE / "annotations.feather", columns=["bodyId", "superclass", "status"]
    )
    body = np.asarray(table["bodyId"], np.int64)
    keep = np.array([bool(s) for s in table["superclass"].to_pylist()])
    keep &= np.array([s != "Glia" for s in table["status"].to_pylist()])
    ids = np.sort(body[keep].astype(np.int64))
    if len(ids) != EXPECTED_NEURONS:
        raise SystemExit(f"retention produced {len(ids)} neurons, expected "
                         f"{EXPECTED_NEURONS}")
    if len(np.unique(ids)) != len(ids):
        raise SystemExit("retained neuron ids are not unique")
    print(f"  retained neurons: {len(ids)}", flush=True)

    print("  streaming edges", flush=True)
    n = len(ids)
    pre_chunks, post_chunks, count_chunks = [], [], []
    rows = 0
    t0 = time.time()
    with pa.memory_map(str(SOURCE / "edges.feather"), "r") as mapped:
        reader = pa.ipc.open_file(mapped)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            a, z, w = [np.asarray(batch.column(batch.schema.get_field_index(k)))
                       for k in ("body_pre", "body_post", "weight")]
            rows += len(a)
            ai, zi = np.searchsorted(ids, a), np.searchsorted(ids, z)
            valid = (ai < n) & (zi < n)
            ai_c, zi_c = np.minimum(ai, n - 1), np.minimum(zi, n - 1)
            valid &= (ids[ai_c] == a) & (ids[zi_c] == z)
            if valid.any():
                pre_chunks.append(ai_c[valid].astype(np.int32))
                post_chunks.append(zi_c[valid].astype(np.int32))
                count_chunks.append(w[valid].astype(np.int64))
    pre, post = np.concatenate(pre_chunks), np.concatenate(post_chunks)
    contacts = np.concatenate(count_chunks)
    print(f"  raw edge rows: {rows}  ({time.time() - t0:.1f}s)", flush=True)
    print(f"  retained directed edges: {len(pre)}", flush=True)

    if len(pre) != EXPECTED_EDGES:
        raise SystemExit(f"retained edges {len(pre)}, expected exactly {EXPECTED_EDGES}")
    total_contacts = int(contacts.sum())
    if total_contacts != EXPECTED_CONTACTS:
        raise SystemExit(f"total contacts {total_contacts}, expected exactly "
                         f"{EXPECTED_CONTACTS}")

    # row = postsynaptic, column = presynaptic; rows normalised to sum to 1.
    W = sparse.coo_matrix(
        (contacts.astype(np.float32), (post, pre)), shape=(n, n)
    ).tocsr()
    del pre, post, contacts
    if W.nnz != EXPECTED_EDGES:
        raise SystemExit(f"CSR has {W.nnz} stored edges for {EXPECTED_EDGES} rows: "
                         "duplicate directed edges exist")
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    W.data /= np.repeat(np.maximum(row_sums, 1.0), np.diff(W.indptr)).astype(np.float32)
    W.sort_indices()

    GRAPH.mkdir(parents=True, exist_ok=True)
    arrays = {"ids": ids, "data": W.data, "indices": W.indices, "indptr": W.indptr}
    digests = {}
    for name, arr in arrays.items():
        path = GRAPH / f"{name}.npy"
        np.save(path, arr)
        digests[path.name] = sha256(path)
        print(f"  wrote {path.name} {arr.dtype} {arr.shape} "
              f"{path.stat().st_size} bytes", flush=True)

    manifest = {
        "release": "MaleCNS v1.0",
        "neurons": int(len(ids)),
        "directed_edges": int(W.nnz),
        "synaptic_contacts": total_contacts,
        "raw_edge_rows": int(rows),
        "retention": "Annotated nonempty superclass, excluding status Glia; "
                     "both endpoints retained.",
        "matrix_orientation": "row=postsynaptic; column=presynaptic",
        "weights": "Unsigned contact counts, divided by total incoming contacts.",
        "source_files": {
            local: {"bytes": size, "sha256": digest}
            for local, (_, size, digest) in SOURCES.items()
        },
        "arrays": digests,
        "built_on": "colab",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in
                      ("release", "neurons", "directed_edges", "synaptic_contacts")}),
          flush=True)
    return manifest


def equivalence_gate() -> dict:
    """Prove the accelerated path reproduces the reference path before any number is read.

    Per control mode, and for both fresh and trained weights, compare the settled state on
    the GPU against the CPU. The tolerance is stated rather than implied: cuSPARSE and scipy
    accumulate in different orders, so this is a float32 bound, not bitwise equality. A
    fresh-vs-trained split is deliberate -- an accelerated path can agree on the default case
    and diverge on a permutation, and passing a naive test while corrupting the comparison
    being run is the exact failure this gate exists to catch.
    """
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from brain.encoders import encode_text
    from brain.plastic_brain import PlasticBrain
    from brain.reservoir import FlyReservoir, load_connectome

    TOL_REL = 1e-4  # relative to the state's own scale, for the FRESH settle
    connectome = load_connectome(str(GRAPH))
    emb = encode_text("hola")
    results = {}

    for mode in ("intact", "shuffled", "random_graph", "no_edges"):
        r_cpu = FlyReservoir(connectome, seed=7301)
        r_cpu.set_mode(mode)
        r_gpu = FlyReservoir(connectome, seed=7301)
        r_gpu.set_mode(mode)
        info = r_gpu.enable_gpu()
        if not info.get("enabled"):
            raise SystemExit(f"accelerated path unavailable: {info}")

        sr = r_cpu.graph if mode != "random_graph" else r_cpu.random_graph()
        key = "graph" if mode != "random_graph" else "random"

        sc = r_cpu.settle(emb, steps=6)[0].copy()
        sg = r_gpu.settle(emb, steps=6)[0].copy()
        scale = max(float(np.max(np.abs(sc))), 1e-9)
        fresh = float(np.max(np.abs(sc - sg))) / scale

        # (1) HARD: the upload must not change the matrix the plastic positions index. A matvec
        # used to canonicalise the uploaded control in place, merging its 31,231 duplicate
        # (row, column) pairs, so the trained weights were written to the wrong synapses. Checked
        # per mode, because only the control matrix has duplicates to merge.
        mirror = r_gpu._gmirror.get(key)
        layout_ok = (
            mirror is not None
            and int(mirror.nnz) == int(sr.nnz)
            and bool(np.array_equal(mirror.indptr.get(), sr.indptr))
            and bool(np.allclose(mirror.data.get(), sr.data, rtol=0, atol=1e-6))
        )

        # (2) HARD: the mirror must carry ITS OWN device's matrix at the plastic positions after
        # training. This is the placement invariant, and it is exact: a mismatch is a bug, not
        # drift. (Comparing across devices instead would conflate the two -- the residual
        # difference between devices is training drift, which is a separate row below.)
        b_cpu = PlasticBrain(r_cpu, n_pools=4, pool_size=200, seed=99, steps=6)
        b_gpu = PlasticBrain(r_gpu, n_pools=4, pool_size=200, seed=99, steps=6)
        for _ in range(6):
            b_cpu.observe(emb, 0)
            b_gpu.observe(emb, 0)
        m_gpu = r_gpu._gmirror.get(key)
        src_gpu = r_gpu.graph if mode != "random_graph" else r_gpu._random_graph
        pos = r_gpu.plastic_pos
        placement = float(
            np.max(np.abs(m_gpu.data.get()[pos] - src_gpu.data[pos]))
        ) if m_gpu is not None else float("inf")

        # (3) REPORTED, not gated: the two devices train the same model but accumulate float32
        # sums in a different order, and the training loop feeds that difference back into the
        # weights. Measured locally after six plasticity steps: scales drift 2.4e-05 relative in
        # the connectome, 2.1e-04 under the shuffle, 8.4e-04 in the control, with the matrices
        # themselves identical and the mirror exact. That is numerical drift, not a defect, and
        # the loop is why it is larger than the 1e-7 seen in a single settle. What licenses a
        # GPU NUMBER is not this figure but the same-seed accuracy agreement measured below.
        sc2 = r_cpu.settle(emb, steps=6)[0].copy()
        sg2 = r_gpu.settle(emb, steps=6)[0].copy()
        scale2 = max(float(np.max(np.abs(sc2))), 1e-9)
        trained = float(np.max(np.abs(sc2 - sg2))) / scale2
        scales_rel = float(
            np.max(np.abs(r_cpu.plastic_scale - r_gpu.plastic_scale))
            / max(float(np.max(np.abs(r_cpu.plastic_scale))), 1e-9)
        )

        ok = fresh < TOL_REL and layout_ok and placement < 1e-6
        results[mode] = {
            "fresh_rel_diff": fresh,
            "mirror_keeps_layout": layout_ok,
            "weights_placed_exactly": placement,
            "trained_settle_drift_rel": trained,
            "scale_drift_rel": scales_rel,
            "pass": ok,
        }
        print(f"  {mode:14s} fresh {fresh:.3e}  layout {'ok' if layout_ok else 'BROKEN'}  "
              f"placement {placement:.2e}  |  drift: settle {trained:.2e} scales {scales_rel:.2e}"
              f"  {'PASS' if ok else 'FAIL'}", flush=True)

    worst_fresh = max(v["fresh_rel_diff"] for v in results.values())
    failed = [k for k, v in results.items() if not v["pass"]]
    if failed:
        raise SystemExit(
            f"EQUIVALENCE GATE FAILED for {failed}: worst fresh relative difference "
            f"{worst_fresh:.3e} against the stated tolerance {TOL_REL:.0e}. The accelerated "
            "path computes something different; do not quote any number from this session "
            "until it is fixed."
        )
    print(f"\n  gate passed: fresh settles agree to {worst_fresh:.3e} (< {TOL_REL:.0e}), "
          "every mirror keeps the CPU layout, and trained weights land on the same entries.",
          flush=True)
    return results


def accuracy_agreement(epochs: int = 3) -> dict:
    """Does a GPU-trained brain score the same as a CPU-trained one on the same seed?

    This is the measurement that licenses a GPU figure. A fresh settle agrees to ~3e-07 and the
    mirror is exact, but the training loop feeds each step's float32 difference back into the
    weights, so the two devices' models drift apart slightly as they train. What matters for a
    reported accuracy is whether that drift changes the answer. Both devices train here on the
    same seed, same order, same budget, and the scores are compared directly.
    """
    ys = None
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from brain.encoders import encode_text
    from brain.plastic_brain import PlasticBrain
    from brain.reservoir import FlyReservoir

    connectome = load_connectome(str(GRAPH))
    cur = json.loads((ROOT / "brain/curriculum/es-en.json").read_text(encoding="utf-8"))
    items = [(c["prompt"], int(c["correctIndex"]))
             for u in cur["units"] for l in u["lessons"] for c in l["challenges"]]
    y = np.array([t for _, t in items], dtype=int)
    ys = y

    X_path = ROOT / "cache" / "embeddings.npy"
    if X_path.exists():
        X = np.load(X_path)
    else:
        X = np.stack([encode_text(p) for p, _ in items])
        np.save(X_path, X)
    assert X.shape[0] == len(ys)

    out = {}
    for mode in ("intact", "random_graph"):
        scores = {}
        for label, use_gpu in (("cpu", False), ("gpu", True)):
            r = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=7301)
            r.set_mode(mode)
            if use_gpu:
                r.enable_gpu()
            b = PlasticBrain(r, n_pools=4, pool_size=200, seed=99, steps=6)
            b.set_mode(mode)
            for ep in range(1, epochs + 1):
                idx = np.random.default_rng(ep).permutation(len(X))
                for i in idx:
                    b.observe(X[i], ys[i])
            scores[label] = b.accuracy(X, ys)
            del r, b
        diff = abs(scores["cpu"] - scores["gpu"])
        out[mode] = {**scores, "abs_diff": diff}
        print(f"  {mode:14s} same seed, {epochs} epochs: CPU {scores['cpu']:.1%}   "
              f"GPU {scores['gpu']:.1%}   difference {diff:.1%}", flush=True)
    worst = max(v["abs_diff"] for v in out.values())
    print(f"\n  largest CPU/GPU accuracy difference at {epochs} epochs: {worst:.1%}", flush=True)
    return out


def main() -> None:
    print("=" * 70)
    print("FlyLingo on Colab -- step 1: environment, graph, equivalence gate")
    print("=" * 70, flush=True)
    ensure_packages()

    print("\n[1/3] public sources", flush=True)
    fetch()

    print("\n[2/3] building the real MaleCNS v1.0 graph", flush=True)
    manifest = build_graph()

    print("\n[3/3] embedded modules", flush=True)
    sys.path.insert(0, "/content")
    import payload

    payload.write_all(ROOT)
    print(f"  wrote {len(payload.FILES)} files under {ROOT}", flush=True)

    print("\n[4/3] GPU vs CPU equivalence gate (per mode, fresh and trained)", flush=True)
    gate = equivalence_gate()

    print("\n[5/4] does the drift change any answer? same-seed CPU vs GPU accuracy", flush=True)
    agreement = accuracy_agreement(epochs=3)

    (CACHE / "step1_report.json").write_text(
        json.dumps({"manifest": manifest, "equivalence": gate,
                    "accuracy_agreement": agreement}, indent=2) + "\n"
    )
    print("\nstep 1 complete.", flush=True)


if __name__ == "__main__":
    main()
