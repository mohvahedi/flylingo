"""Throwaway diagnostic (NO tuning): why is the real-curriculum run at chance?

Two questions that a null result has to answer honestly:
  1. What is the real chance baseline? (if the correct index is not uniform,
     0.25 understates a trivial policy.)
  2. Do the real reservoir features carry the answer at all? A supervised
     linear softmax readout on the SAME intact features is an upper bound on
     what a linear policy can get; it is not a trained arm and not a result.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/Projects/flylingo")
sys.path.insert(0, str(ROOT / "brain"))

from brain.learning import train as T          # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome   # noqa: E402

tasks = T.load_task()
print("source", tasks.source, "n_useful", len(tasks.challenges),
      "train", len(tasks.train), "eval", len(tasks.eval))

print()
print("=== 1. what does a trivial policy get on the eval split? ===")
tr_ci = np.array([int(c['correctIndex']) for c in tasks.train])
ev_ci = np.array([int(c['correctIndex']) for c in tasks.eval])
print("correctIndex counts  train", np.bincount(tr_ci, minlength=4).tolist(),
      " eval", np.bincount(ev_ci, minlength=4).tolist())
print("train majority class", int(np.argmax(np.bincount(tr_ci, minlength=4))),
      "-> eval accuracy of that constant policy",
      round(float(np.mean(ev_ci == int(np.argmax(np.bincount(tr_ci, minlength=4))))), 3))
print("strict chance (uniform)", round(1.0 / 4, 3))

print()
print("=== 2. do the intact features carry the correct index? ===")
conn = load_connectome()
res = FlyReservoir(conn, embedding_dim=T.EMBED_DIM, dims=T.DIMS, seed=T.RESERVOIR_SEED)
Xtr = T.features_for(res, 'intact', tasks.train, T.EMBED_DIM, reset=True)
Xev = T.features_for(res, 'intact', tasks.eval, T.EMBED_DIM, reset=True)
print("train features", Xtr.shape, "rms", round(float(np.sqrt((Xtr ** 2).mean())), 4),
      "| eval features", Xev.shape, "rms", round(float(np.sqrt((Xev ** 2).mean())), 4))
print("mean per-step rms", round(float(np.sqrt((Xtr ** 2).mean(axis=1)).mean()), 4))
# are the rows actually different from each other?
c = Xtr - Xtr.mean(axis=0, keepdims=True)
print("pairwise cosine spread: min %0.3f max %0.3f" % (
    float((c @ c.T / (np.linalg.norm(c, axis=1, keepdims=True) @
                      np.linalg.norm(c, axis=1, keepdims=True).T + 1e-12))[~np.eye(len(c), dtype=bool)].min()),
    float((c @ c.T / (np.linalg.norm(c, axis=1, keepdims=True) @
                      np.linalg.norm(c, axis=1, keepdims=True).T + 1e-12))[~np.eye(len(c), dtype=bool)].max())))

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
A, B = (Xtr - mu) / sd, (Xev - mu) / sd
W = np.zeros((A.shape[1], 4))
b = np.zeros(4)
y = tr_ci
for it in range(4000):
    z = A @ W + b
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    g = (p - np.eye(4)[y]) / len(y)
    W -= 0.5 * (A.T @ g)
    b -= 0.5 * g.sum(0)
ze = B @ W + b
ev_pred = ze.argmax(1)
print("supervised linear readout on the SAME features: train acc",
      round(float(np.mean((A @ W + b).argmax(1) == y)), 3),
      " eval acc", round(float(np.mean(ev_pred == ev_ci)), 3))
