"""Why does the intact brain plateau at 64%? Is it the optimiser or the read-out?

Two candidate ceilings, and they call for completely different fixes:

  (a) the optimiser. The gradient treats the presynaptic activity as fixed, dropping the
      recurrent dependence, so it is only first-order accurate. If that is the limit, the pool
      representation holds the answer and a better update would find it.

  (b) the read-out. The decision is the argmax over four pool MEANS, which is a 4-dimensional
      projection of a 166,700-dimensional state. If the four numbers simply cannot separate the
      97 prompts, no amount of training on these synapses will help.

The test separates them without training anything: freeze the brain at its initial weights, take
the representation it produces, and fit a linear classifier to it directly.

  * linear probe on the 4 pool means -> if this is high, (b) is false, the optimisation is the limit
  * linear probe on the 128 settled features -> the ceiling for ANY linear read-out on this state
  * linear probe on the raw encoding -> the no-brain baseline, for reference

Usage: python scripts/diagnose_ceiling.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_text  # noqa: E402
from brain.plastic_brain import PlasticBrain  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402


def fit_linear_readout(X, y, n_classes=4, epochs=3000, lr=0.5, l2=1e-4, seed=0):
    """Plain full-batch softmax regression. Deliberately simple: the question is whether the
    representation is linearly separable, not how well a fancy optimiser can do."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    n, d = X.shape
    W = np.zeros((n_classes, d))
    b = np.zeros(n_classes)
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1.0

    best = 0.0
    for _ in range(epochs):
        z = X @ W.T + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - Y) / n
        W -= lr * (g.T @ X + l2 * W)
        b -= lr * g.sum(axis=0)
        acc = float((p.argmax(axis=1) == y).mean())
        best = max(best, acc)
    return best


cur = json.load(open("brain/curriculum/es-en.json", encoding="utf-8"))
items = [
    (c["prompt"], int(c["correctIndex"]))
    for u in cur["units"]
    for l in u["lessons"]
    for c in l["challenges"]
]
X_raw = np.stack([encode_text(p) for p, _ in items])
y = np.array([t for _, t in items], dtype=int)

r = FlyReservoir(load_connectome(), embedding_dim=256, dims=128, seed=7301)
pb = PlasticBrain(r, n_pools=4, pool_size=200, seed=99, steps=6)
pb.set_mode("intact")

# representations from the FROZEN brain (scales are all 1.0 here)
Z_pools, X_feat = [], []
for i in range(len(items)):
    emb = X_raw[i]
    probs, z, _ = pb.forward(emb)
    Z_pools.append(z)
    # the 128 settled features, the same ones the old prompt_index read-out consumed
    r.reset()
    _state, feat = r.settle(emb, steps=6)
    X_feat.append(feat)
Z_pools = np.stack(Z_pools)
X_feat = np.stack(X_feat)

print(f"task: {len(items)} challenges, 4 classes, majority baseline "
      f"{np.bincount(y).max()/len(y):.1%}\n")
print(f"pool means (4 dims)  spread across prompts: {Z_pools.max()-Z_pools.min():.5f}")
print(f"settled features (128 dims) std across prompts: {X_feat.std(axis=0).mean():.5f}\n")

print(f"{'representation':<34} {'dims':>5}  {'linear probe':>13}")
print("-" * 58)
for name, rep in (
    ("raw prompt encoding (no brain)", X_raw),
    ("settled brain state, 128 features", X_feat),
    ("4 answer-pool means (the decision)", Z_pools),
):
    acc = fit_linear_readout(rep, y, seed=0)
    print(f"{name:<34} {rep.shape[1]:>5}  {acc:>12.1%}")

# how correlated are the pools? near-identical means four pools cannot disagree usefully
C = np.corrcoef(Z_pools.T)
print(f"\npool correlation matrix (frozen brain):")
for i in range(C.shape[0]):
    print("   " + "  ".join(f"{C[i, j]:+.3f}" for j in range(C.shape[1])))
off = C[np.triu_indices(4, 1)]
print(f"   mean off-diagonal correlation {off.mean():+.3f}")
print(f"   pool win counts (argmax): {np.bincount(Z_pools.argmax(axis=1), minlength=4)}")
print(f"   answers per option      : {np.bincount(y, minlength=4)}")

# and how separable is the FINAL trained state?
trained = Path("D:/Projects/flylingo/brain/runs/plastic_brain/convergence.json")
if trained.exists():
    print(f"\n(trained run reached {json.load(open(trained))['best_accuracy']:.1%})")
