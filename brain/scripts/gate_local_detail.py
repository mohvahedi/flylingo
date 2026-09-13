"""Replicate the Colab gate locally, with the detail needed to say WHERE the residual lives.

After the layout fix the gate improved from 6.012e-01 to 1.916e-04 in the random control and
every mirror keeps the CPU's layout, but three modes still exceed the 1e-4 tolerance and two
report the trained weights as misplaced. This separates the possibilities: the residual can
live in the trained scales (the two devices train differently), in the matrices (the weights
land on different entries), or in the settle itself (same matrix, different arithmetic).

Per mode it prints, after each of six plasticity steps:
  * the largest relative difference in the plastic scales,
  * whether the weights at the plastic positions agree between the two devices' matrices,
  * whether the GPU mirror holds its own device's matrix,
  * the settled-state difference by step count, fresh and after training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_text  # noqa: E402
from brain.plastic_brain import PlasticBrain  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

MODES = ("intact", "shuffled", "random_graph")
connectome = load_connectome()
emb = encode_text("hola")
TOL = 1e-4


def build(mode: str, gpu: bool):
    r = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=7301)
    r.set_mode(mode)
    if gpu:
        info = r.enable_gpu()
        assert info.get("enabled"), info
    b = PlasticBrain(r, n_pools=4, pool_size=200, seed=99, steps=6)
    b.set_mode(mode)
    return r, b


def active(r):
    return (r._random_graph, "random") if r.mode == "random_graph" else (r.graph, "graph")


def peak(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float("inf"), float("inf")
    d = float(np.max(np.abs(a - b)))
    s = max(float(np.max(np.abs(a))), 1e-12)
    return d, d / s


for mode in MODES:
    print(f"\n{'=' * 74}\n{mode}\n{'=' * 74}")
    r_c, b_c = build(mode, gpu=False)
    r_g, b_g = build(mode, gpu=True)

    src_c, key = active(r_c)
    src_g, _ = active(r_g)
    mirror = r_g._gmirror.get(key)
    print(f"  matrices at the plastic sites, before any training:")
    d, dr = peak(src_c.data[r_c.plastic_pos], src_g.data[r_g.plastic_pos])
    print(f"    cpu vs gpu matrix data   : max|diff| {d:.3e} (rel {dr:.3e})")
    if mirror is not None:
        d, dr = peak(src_g.data[r_g.plastic_pos], mirror.data.get()[r_g.plastic_pos])
        print(f"    gpu matrix vs its mirror : max|diff| {d:.3e} (rel {dr:.3e})")

    for step in range(1, 7):
        b_c.observe(emb, 0)
        b_g.observe(emb, 0)
        d, dr = peak(r_c.plastic_scale, r_g.plastic_scale)
        src_c, key = active(r_c)
        src_g, _ = active(r_g)
        mirror = r_g._gmirror.get(key)
        pos = r_c.plastic_pos
        wd, _ = peak(src_c.data[pos], src_g.data[pos])
        md, _ = peak(src_g.data[pos], mirror.data.get()[pos]) if mirror is not None else (float("nan"), 0)
        print(f"    step {step}: scales rel {dr:.3e}   matrix at plastic rel {wd:.3e}   "
              f"mirror vs its matrix rel {md:.3e}")

    # the settle comparison, fresh (scales reset to 1 by rebuilding) and trained
    sc = r_c.settle(emb, steps=6)[0].copy()
    sg = r_g.settle(emb, steps=6)[0].copy()
    d, dr = peak(sc, sg)
    print(f"  trained settle: max|diff| {d:.3e}  rel {dr:.3e}   "
          f"{'ok' if dr < TOL else 'OVER TOLERANCE'}")
