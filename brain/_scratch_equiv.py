"""Scratch: prove reservoir.py is numerically equivalent to the flm reference.

Verification only. reservoir.py itself never imports flm at runtime.
"""
import sys

import numpy as np

sys.path.insert(0, "D:/Projects/flylingo/flm")
sys.path.insert(0, "D:/Projects/flylingo/brain")

from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402


class Shim:
    """flm's Reservoir only needs .matrix and .shape on its graph object."""

    def __init__(self, matrix):
        self.matrix = matrix


def main():
    from flm.graph import Reservoir as RefReservoir

    conn = load_connectome()
    mine = FlyReservoir(conn, embedding_dim=256, dims=128, seed=7301)
    ref = RefReservoir(Shim(conn.matrix), embedding_dim=256, dimensions=128, seed=7301)

    rng = np.random.default_rng(11)
    max_ds = max_df = 0.0
    for t in range(12):
        emb = rng.standard_normal(256).astype(np.float32)
        f_ref = ref.step(emb, "intact")
        f_mine = mine.step(emb, "intact")
        ds = float(np.max(np.abs(ref.state - mine.state)))
        df = float(np.max(np.abs(np.asarray(f_ref) - np.asarray(f_mine))))
        max_ds = max(max_ds, ds)
        max_df = max(max_df, df)
    print(f"intact    : max |dstate| = {max_ds:.3e}   max |dfeatures| = {max_df:.3e}")

    # shuffled
    mine.reset()
    ref.reset()
    max_ds = max_df = 0.0
    for t in range(8):
        emb = rng.standard_normal(256).astype(np.float32)
        f_ref = ref.step(emb, "shuffled")
        f_mine = mine.step(emb, "shuffled")
        max_ds = max(max_ds, float(np.max(np.abs(ref.state - mine.state))))
        max_df = max(max_df, float(np.max(np.abs(np.asarray(f_ref) - np.asarray(f_mine)))))
    print(f"shuffled  : max |dstate| = {max_ds:.3e}   max |dfeatures| = {max_df:.3e}")

    mine.reset()
    ref.reset()
    f_ref = ref.step(rng.standard_normal(256).astype(np.float32), "no_edges")
    f_mine = mine.step(rng.standard_normal(256).astype(np.float32), "no_edges")
    print(f"no_edges  : ref all zero = {bool(np.all(f_ref == 0))}  "
          f"mine all zero = {bool(np.all(f_mine == 0))}")

    # unit-RMS feature contract
    mine.reset()
    for _ in range(5):
        f = mine.step(rng.standard_normal(256).astype(np.float32), "intact")
    print(f"feature RMS = {float(np.sqrt(np.mean(f * f))):.6f} (contract: unit RMS)")


if __name__ == "__main__":
    main()
