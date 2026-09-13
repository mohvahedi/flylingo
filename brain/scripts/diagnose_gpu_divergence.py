"""Which quantity does the accelerated path get wrong, using the SHIPPED code paths?

The equivalence gate (colab/step1_build.py) fails on a T4 and on the local 3070 Ti after
only six plasticity steps, worst on random_graph (6.0e-01 on a state of scale ~0.8), while
a fresh, untrained reservoir agrees to ~3e-07. The earlier local probe agreed on trained
weights because it RE-IMPLEMENTED the recurrence instead of calling the shipped code, so it
tested a reading of the code rather than the code. This script calls the shipped `step`/
`settle` on both devices and compares the quantities the acceleration can get wrong.

Everything is compared to the CPU path, which is the reference: the CPU path is what the
service, the checkpoints and every published number use.

Reports, per mode:
  1. the plastic scales                  (do the two devices train the same numbers?)
  2. the graph data at the plastic sites  (was the scaled weight written on both?)
  3. the uploaded mirror vs the CPU matrix (does the accelerator hold the same weights?)
  4. the settled state by step count      (where does the trajectory part company?)

Reads no checkpoint and trains nothing new: six observe() calls from fresh weights, which is
the configuration the gate fails in. No verdict is pre-written; it prints the numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_text  # noqa: E402
from brain.plastic_brain import PlasticBrain  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

OBSERVES = 6
MODES = ("intact", "shuffled", "random_graph")

cur = json.load(open("brain/curriculum/es-en.json", encoding="utf-8"))
items = [
    (c["prompt"], int(c["correctIndex"]))
    for u in cur["units"]
    for l in u["lessons"]
    for c in l["challenges"]
]
X = np.stack([encode_text(p) for p, _ in items])
y = np.array([t for _, t in items], dtype=int)
connectome = load_connectome()


def build(mode: str, gpu: bool):
    """Same construction order as the gate and as measure_plastic_brain: GPU first, then brain."""
    r = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=7301)
    r.set_mode(mode)
    if gpu:
        info = r.enable_gpu()
        if not info.get("enabled"):
            raise SystemExit(f"GPU unavailable: {info}")
    b = PlasticBrain(r, n_pools=4, pool_size=200, seed=99, steps=6)
    b.set_mode(mode)
    return r, b


def rel(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    s = max(float(np.max(np.abs(np.asarray(a)))), 1e-9)
    return d, d / s


for mode in MODES:
    print(f"\n{'=' * 72}\n{mode}\n{'=' * 72}")
    r_c, b_c = build(mode, gpu=False)
    r_g, b_g = build(mode, gpu=True)

    for i in range(OBSERVES):
        b_c.observe(X[i], y[i])
        b_g.observe(X[i], y[i])

    # 1. the trained scales
    d, dr = rel(r_c.plastic_scale, r_g.plastic_scale)
    print(f"1. plastic scales          : max|CPU-GPU| {d:.3e}  (rel {dr:.3e})  "
          f"max {r_c.plastic_scale.max():.3f}")

    # 2. the graph data written at the plastic sites
    pos = r_c.plastic_pos
    d, dr = rel(r_c.graph.data[pos], r_g.graph.data[pos])
    print(f"2. graph data at plastic   : max|CPU-GPU| {d:.3e}  (rel {dr:.3e})")

    # 3. what the accelerator actually holds, vs the CPU's own matrix
    mirror = r_g._gmirror.get("graph")
    if mirror is None:
        print("3. mirror                  : NOT BUILT")
    else:
        m = mirror.data.get()
        d, dr = rel(r_g.graph.data, m)
        print(f"3. mirror vs its own graph : max|GPU mirror - GPU graph| {d:.3e} (rel {dr:.3e})")
        d, dr = rel(r_c.graph.data, m)
        print(f"   mirror vs the CPU graph : max|GPU mirror - CPU graph| {d:.3e} (rel {dr:.3e})")
        if mode == "random_graph":
            rm = r_g._gmirror.get("random")
            src = r_g._random_graph
            if rm is None:
                print("   random mirror           : NOT BUILT")
            else:
                d, dr = rel(src.data, rm.data.get())
                print(f"   random mirror vs its own: max|diff| {d:.3e} (rel {dr:.3e})")
                d, dr = rel(r_c._random_graph.data, rm.data.get())
                print(f"   random mirror vs CPU's  : max|diff| {d:.3e} (rel {dr:.3e})")

    # 4. the settled state, by step count, through the shipped settle() on both devices
    emb = X[0]
    print("4. settled state by steps (shipped settle on each device):")
    for steps in range(1, 7):
        r_c.reset()
        s_c = r_c.settle(emb, steps=steps)[0].copy()
        r_g.reset()
        s_g = r_g.settle(emb, steps=steps)[0].copy()
        d, dr = rel(s_c, s_g)
        print(f"     steps {steps}: max|CPU-GPU| {d:.3e}  rel {dr:.3e}  "
              f"(state max {np.abs(s_c).max():.3f})")
