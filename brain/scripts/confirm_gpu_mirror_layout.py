"""Confirm the mechanism behind the GPU/CPU divergence: does the upload reshape the matrix?

Established by scripts/diagnose_gpu_divergence.py: on the trained random_graph control the
GPU mirror holds 25,551,707 stored entries against the CPU matrix's 25,582,938, so the two
arrays are not the same length and every position after the first merge refers to a different
entry. `gpu_sync_data` writes scaled weights by position, so it writes them into the wrong
entries and the two devices train different matrices.

This checks the cause directly rather than inferring it: does the CPU control matrix contain
duplicate (row, column) pairs, and does constructing a cupyx matrix from those arrays reduce
the stored-entry count? It also checks whether the graph (intact/shuffled) matrix has the same
problem, which predicts where the divergence should and should not appear.
No verdict is pre-written; it prints the counts.
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


def probe(name: str, src) -> None:
    """Report whether a CSR matrix has duplicates, and what cupyx does to it."""
    indptr, indices = src.indptr, src.indices
    rows = np.repeat(np.arange(src.shape[0], dtype=np.int64), np.diff(indptr))
    # a duplicate is a repeated (row, col) pair
    order = np.lexsort((indices, rows))
    same = (rows[order][1:] == rows[order][:-1]) & (indices[order][1:] == indices[order][:-1])
    dup_pairs = int(same.sum())

    m = cp_sparse.csr_matrix(
        (cp.asarray(src.data), cp.asarray(src.indices), cp.asarray(src.indptr)),
        shape=src.shape,
    )
    print(f"{name}:")
    print(f"  cpu stored entries        : {src.nnz}")
    print(f"  duplicate (row,col) pairs : {dup_pairs}")
    print(f"  gpu mirror stored entries : {m.nnz}")
    print(f"  counts match              : {m.nnz == src.nnz}   "
          f"({'OK' if m.nnz == src.nnz else 'DIFFERENT LAYOUT'} by {src.nnz - m.nnz})")

    # does the divergence survive when positions are irrelevant? compare one matvec
    x = np.random.default_rng(0).normal(size=src.shape[1]).astype(np.float32)
    a = np.asarray(src @ x, np.float32)
    b = cp.asnumpy(m @ cp.asarray(x))
    scale = max(float(np.max(np.abs(a))), 1e-9)
    print(f"  one matvec max|diff|      : {np.max(np.abs(a - b)):.3e} "
          f"(rel {np.max(np.abs(a - b)) / scale:.3e})")

    # would setting the arrays directly (no canonicalisation) preserve the layout?
    m2 = cp_sparse.csr_matrix(src.shape, dtype=src.dtype)
    m2.data = cp.asarray(src.data)
    m2.indices = cp.asarray(src.indices)
    m2.indptr = cp.asarray(src.indptr)
    b2 = cp.asnumpy(m2 @ cp.asarray(x))
    print(f"  direct-attribute mirror   : entries {m2.nnz}  "
          f"({'SAME LAYOUT' if m2.nnz == src.nnz else 'still different'})  "
          f"matvec max|diff| {np.max(np.abs(a - b2)):.3e}")


r = FlyReservoir(connectome, seed=7301)
r.set_mode("random_graph")
probe("random_graph control matrix", r.random_graph())

r2 = FlyReservoir(connectome, seed=7301)
r2.set_mode("intact")
probe("intact connectome matrix", r2.graph.tocsr())
