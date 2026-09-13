"""Is it the matvec that reindexes the mirror, and does declaring the format stop it?

The instrumentation sequence: upload the control mirror -> stored entries 25,582,938
(correct) -> one sparse matvec -> the next inspection reads 25,551,707, i.e. exactly
25,582,938 - 31,231 duplicate (row, column) pairs merged, with the index pointers rewritten.
The control matrix is built with duplicates deliberately (so its nnz matches the connectome's
25,582,938 and it shares the connectome's CSR layout, which is what lets one set of plastic
edge positions address the same synapse rank in either wiring). cupyx's matvec canonicalises
such a matrix in place, so the layout moves under the caller and every later position-based
weight write lands on a different entry.

Three things are checked here:
  1. does one matvec change the mirror's stored-entry count?
  2. is the matvec's RESULT unchanged by canonicalisation (it should be: duplicate entries in
     the same row and column simply add, so merging cannot change a matrix-vector product)?
  3. does declaring the mirror canonical on the cupyx object stop the in-place rewrite while
     leaving the result identical?
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
src = r.random_graph()

x = np.random.default_rng(0).normal(size=src.shape[1]).astype(np.float32)
gx = cp.asarray(x)
expect = np.asarray(src @ x, np.float32)


def mirror():
    return cp_sparse.csr_matrix(
        (cp.asarray(src.data), cp.asarray(src.indices), cp.asarray(src.indptr)),
        shape=src.shape,
    )


print(f"cpu control matrix: nnz {src.nnz}  canonical {src.has_canonical_format}")

print("\n1. one matvec on a freshly built mirror")
m = mirror()
before = m.nnz
out = cp.asnumpy(m @ gx)
after = m.nnz
print(f"   nnz {before} -> {after}   "
      f"{'REINDEXED (in place)' if after != before else 'unchanged'}   "
      f"short by {before - after}")
print(f"   matvec result max|diff| vs cpu: {np.max(np.abs(out - expect)):.3e}")

print("\n2. is the result stable across the rewrite?")
out2 = cp.asnumpy(m @ gx)
print(f"   second matvec after the rewrite: max|diff| vs cpu "
      f"{np.max(np.abs(out2 - expect)):.3e}   "
      f"(so the product is unchanged: "
      f"{bool(np.allclose(out, out2, rtol=0, atol=1e-6))})")

print("\n3. declaring the mirror canonical before any matvec")
m2 = mirror()
try:
    m2.has_canonical_format = True
    declared = True
except Exception as e:  # noqa: BLE001
    declared = False
    print(f"   could not set the flag: {type(e).__name__}: {e}")
if declared:
    b2 = m2.nnz
    out3 = cp.asnumpy(m2 @ gx)
    a2 = m2.nnz
    print(f"   nnz {b2} -> {a2}   "
          f"{'REINDEXED' if a2 != b2 else 'LAYOUT PRESERVED'}")
    print(f"   matvec result max|diff| vs cpu: {np.max(np.abs(out3 - expect)):.3e}")
    print(f"   indptr matches the cpu matrix : "
          f"{bool(np.array_equal(m2.indptr.get(), src.indptr))}")
