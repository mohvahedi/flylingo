"""Is the trained system a bug, or a numerically sensitive one?

The GPU and CPU paths agree on a fresh brain (2.98e-07) and disagree badly once the weights are
trained (0.465 on a matrix of scale 0.845). That has two very different explanations and they need
different remedies:

  (a) A BUG: the GPU path computes something different from the CPU path, so any GPU number is
      untrustworthy and the accelerator must be fixed before it is used again.
  (b) SENSITIVITY: the trained recurrence amplifies a float32 difference of order 1e-7 into a much
      larger one over six recurrent steps, so the two devices are each internally consistent but
      cannot be compared on trained weights. Nothing is wrong with either path.

The distinguishing measurement is a perturbation test, and it needs no GPU at all. If multiplying
every trained weight by (1 + 1e-6) moves the accuracy by many points, the trained system is
ill-conditioned and (b) is the explanation. If a 1e-6 perturbation leaves the accuracy untouched
while the GPU path still disagrees, it is (a).

Same-architecture reloads (CPU -> CPU) are checked first: if those do not reproduce exactly, the
saving code is broken and nothing else in this script means anything.

Runs on the CPU throughout, so it puts no load on the GPU.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_text  # noqa: E402
from brain.plastic_brain import PlasticBrain  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

import json  # noqa: E402

EPOCHS = 12
MODE = "intact"

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


def build():
    r = FlyReservoir(connectome, seed=7301)
    r.set_mode(MODE)
    b = PlasticBrain(r, n_pools=4, pool_size=200, seed=99, steps=6)
    b.set_mode(MODE)
    return r, b


print(f"training {EPOCHS} epochs on the CPU (no GPU load)...")
r, brain = build()
for ep in range(1, EPOCHS + 1):
    idx = np.random.default_rng(ep).permutation(len(X))
    for i in idx:
        brain.observe(X[i], y[i])
acc = brain.accuracy(X, y)
print(f"trained accuracy          : {acc:.1%}")
print(f"scale range               : {r.plastic_scale.min():.4f} .. {r.plastic_scale.max():.4f}")
print(f"score_w range             : {brain.score_w.min():.5f} .. {brain.score_w.max():.5f}")

ckpt = Path("runs/plastic_brain/_diag_sensitivity.npz")
brain.save(ckpt)

# ---------------------------------------------------------------- 1. same-architecture reload
r2, brain2 = build()
brain2.load(ckpt)
acc_reload = brain2.accuracy(X, y)
print(f"\n1. CPU -> CPU reload      : {acc_reload:.1%}")
print(f"   exact match            : {abs(acc_reload - acc) < 1e-12}")

# ---------------------------------------------------------------- 2. the perturbation test
print("\n2. perturb every trained weight by a relative epsilon, and re-score:")
print(f"   {'epsilon':>10} {'scale perturb':>14} {'score_w perturb':>16} {'both':>10}")
for eps in (0.0, 1e-7, 1e-6, 1e-5, 1e-4):
    _, b_s = build()
    b_s.load(ckpt)
    if eps:
        b_s.r.plastic_scale *= np.float32(1.0 + eps)
        b_s.r.apply_plastic()
    a_scale = b_s.accuracy(X, y)

    _, b_w = build()
    b_w.load(ckpt)
    if eps:
        b_w.score_w *= (1.0 + eps)
    a_w = b_w.accuracy(X, y)

    _, b_b = build()
    b_b.load(ckpt)
    if eps:
        b_b.r.plastic_scale *= np.float32(1.0 + eps)
        b_b.r.apply_plastic()
        b_b.score_w *= (1.0 + eps)
    a_b = b_b.accuracy(X, y)

    print(f"   {eps:>10.0e} {a_scale:>13.1%} {a_w:>15.1%} {a_b:>9.1%}")

# ---------------------------------------------------------------- 3. how knife-edge is a decision?
print("\n3. how close are the pool scores to tying, on a trained brain?")
_, b3 = build()
b3.load(ckpt)
margins = []
for i in range(len(X)):
    _, z, _ = b3.forward(X[i])
    order = np.sort(z)[::-1]
    margins.append(float(order[0] - order[1]))
margins = np.array(margins)
print(f"   top-2 pool margin: mean {margins.mean():.5f}  median {np.median(margins):.5f}")
print(f"   pool score scale : {np.abs(z).mean():.4f}")
print(f"   relative margin  : {(margins / np.abs(z).mean()).mean():.4f}")
print(f"   prompts within 1% of a tie: {(margins / np.abs(z).mean() < 0.01).mean():.1%}")
