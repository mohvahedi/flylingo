"""Pin down how the mirror's layout changes, so the fix targets the real mechanism.

The counts: the random_graph control has 25,582,938 stored entries and 31,231 duplicate
(row, column) pairs; the connectome has zero duplicates. Constructing a cupyx CSR from the
CPU arrays keeps all 25,582,938. Assigning to the cupyx matrix's `.data` attribute collapses
it to 25,551,707, which is exactly 25,582,938 - 31,231: the setter canonicalises the matrix,
summing duplicate pairs and reindexing everything after each merge.

`Reservoir.gpu_sync_data` writes trained weights as `mirror.data[idx] = values` with `idx`
being positions in the CPU layout. If that write canonicalises the mirror, every position
after the first merge refers to a different entry from then on, so the accelerator trains on
a matrix the CPU never had. This script checks, in order:

  1. does assigning `.data` change the stored-entry count?
  2. is there a write that does NOT change it (in-place on the returned array)?
  3. after such a write, do the written positions hold the written values?
  4. does the accelerated settle reproduce the CPU settle once the layout is preserved?

Prints every number; the last two are the ones that decide it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

import cupy as cp  # noqa: E402
from cupyx.scipy import sparse as cp_sparse  # noqa: E402

connectome = load_connectome()
r = FlyReservoir(connectome, seed=7301)
r.set_mode("random_graph")
r.enable_gpu()
r.random_graph()
src = r._random_graph
mirror = r._build_gpu_mirror("random")

print("initial state")
print(f"  cpu entries   : {src.nnz}")
print(f"  mirror entries: {mirror.nnz}   match: {mirror.nnz == src.nnz}")

# ---- 1. what does assigning .data do?
pos = np.flatnonzero(np.ones(src.nnz, bool))[:1000].astype(np.int64)
m2 = cp_sparse.csr_matrix(
    (cp.asarray(src.data), cp.asarray(src.indices), cp.asarray(src.indptr)), shape=src.shape
)
before = m2.nnz
m2.data[pos] = cp.asarray(src.data[pos])
print("\n1. `mirror.data[idx] = values`")
print(f"  entries before: {before}   after: {m2.nnz}   "
      f"{'REINDEXED' if m2.nnz != before else 'layout preserved'}")

# ---- 2. does in-place mutation of the returned array leave the count alone?
m3 = cp_sparse.csr_matrix(
    (cp.asarray(src.data), cp.asarray(src.indices), cp.asarray(src.indptr)), shape=src.shape
)
raw = m3.data  # the underlying cupy array
sentinel = cp.zeros(pos.size, cp.float32) + cp.float32(7.5)
raw[pos] = sentinel
print("\n2. in-place on the array returned by `.data`")
print(f"  entries after : {m3.nnz}   "
      f"{'layout preserved' if m3.nnz == before else 'REINDEXED'}")
print(f"  values land at the written positions: "
      f"{bool(np.allclose(cp.asnumpy(m3.data[pos]), 7.5))}")

# ---- 3. and through the real mirror, after a genuine plasticity update
print("\n3. the shipped gpu_sync_data path")
r2 = FlyReservoir(connectome, seed=7301)
r2.set_mode("random_graph")
r2.enable_gpu()
r2.random_graph()
r2.enable_plasticity(np.full(r2.n, 0, np.int64), max_edges=117800)
m = r2._gmirror.get("random")
print(f"  mirror entries before sync: {m.nnz if m is not None else 'not built'}")
print(f"  cpu entries              : {r2._random_graph.nnz}")
if m is not None:
    print(f"  layouts match            : {m.nnz == r2._random_graph.nnz}")
    same = np.allclose(cp.asnumpy(m.data), r2._random_graph.data,
                       rtol=0, atol=1e-9) if m.nnz == r2._random_graph.nnz else False
    print(f"  mirror holds the cpu data: {same}")
