"""Tune the online learning rate against the REAL curriculum, offline.

The live service learns, but slowly: 0.20 accuracy after 60 answers. For a demo whose whole
point is watching the fly learn, that is too flat to see. This exercises the actual readout
class, the actual encoder and the actual connectome features over the actual curriculum, and
reports how fast each candidate learning rate reaches competence.

Two variables are tested, because the second is probably the real problem:
  - the learning rate itself
  - whether the feature normaliser is fitted ONLINE. A fresh readout has mu=0, sd=1, so raw
    reservoir features go in unscaled; if their magnitude is small the logits are tiny, the
    gradients are tiny, and no learning rate will look fast. Fitting the normaliser as features
    arrive is what makes the inputs the right size.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"D:\Projects\flylingo\brain")))

from brain.curriculum import load_curriculum          # noqa: E402
from brain.encoders import encode_text                 # noqa: E402
from brain.learning.prompt_index import PromptIndexReadout  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome   # noqa: E402

DIMS = 128
N_ACTIONS = 4

print("loading the connectome and the curriculum ...")
conn = load_connectome()
cur = load_curriculum()
challenges = cur.challenges()
print(f"  {len(challenges)} challenges across {len(cur.lessons())} lessons")

# Build the feature for every challenge ONCE, through the same path the service uses.
# The service builds this with embedding_dim=256 (the encoder width) and dims=128 (the
# reservoir's output). Those are different numbers, and using 128 for both was the first thing
# to go wrong here: the reservoir rejected the encoder's 256-wide vector.
dr = FlyReservoir(conn, embedding_dim=256, dims=DIMS, seed=7301)
feats = {}
for ch in challenges:
    dr.reset()
    feats[ch.id] = dr.step(encode_text(ch["prompt"])).astype(np.float64)

X = np.stack([feats[c.id] for c in challenges])
y = np.array([int(c["correctIndex"]) for c in challenges])
print(f"  feature matrix {X.shape}")
row_norms = np.linalg.norm(X, axis=1)
print(f"  feature row norms: min {row_norms.min():.4f} median {np.median(row_norms):.4f} max {row_norms.max():.4f}")
print(f"  per-dim std across the set: median {np.median(X.std(axis=0)):.5f}")
print()

# How separable is this task in this feature space, as an upper bound?
# (nearest-centroid style check: can a linear map do it at all?)
print("=" * 72)


def run(lr_scale, online_norm, steps=120, seed=0, order=None):
    rng = np.random.default_rng(seed)
    r = PromptIndexReadout(in_dim=DIMS, n_actions=N_ACTIONS, seed=0)
    r.lr = lr_scale
    r.standardise = True

    idx = np.arange(len(challenges)) if order is None else order
    hits = []
    seen = []
    for i in range(steps):
        j = int(idx[i % len(idx)])
        x = X[j]
        # online normaliser: keep mu/sd current with what has actually been seen
        if online_norm:
            seen.append(x)
            S = np.stack(seen)
            r.mu = S.mean(axis=0)
            r.sd = S.std(axis=0) + 1e-6
        a, _lp = r.sample(x, rng)
        ok = int(a) == int(y[j])
        hits.append(int(ok))
        r.observe(1.0 if ok else -0.25, 0.0)
    return hits


print(f"{'lr':>8} {'online-norm':>12} {'first20':>8} {'last20':>8} {'best20':>8}")
print("-" * 72)
results = []
for lr in (0.5, 2.0, 5.0, 20.0, 50.0):
    for onorm in (False, True):
        # average over 3 seeds so one lucky run cannot pick the winner
        accs = []
        for seed in (0, 1, 2):
            h = run(lr, onorm, steps=120, seed=seed)
            accs.append((np.mean(h[:20]), np.mean(h[-20:]),
                         max(np.mean(h[k:k + 20]) for k in range(0, 101, 10))))
        f, l, b = np.mean(accs, axis=0)
        results.append((lr, onorm, f, l, b))
        print(f"{lr:>8} {str(onorm):>12} {f:>8.3f} {l:>8.3f} {b:>8.3f}")

print()
best = max(results, key=lambda t: (t[3], t[4]))
print(f"best final accuracy: lr={best[0]}  online_norm={best[1]}  -> {best[3]:.3f}")
print()
print("Per-answer outcome for the best setting (1 = right):")
h = run(best[0], best[1], steps=120, seed=0)
print("  " + "".join("1" if v else "." for v in h[:60]))
print("  " + "".join("1" if v else "." for v in h[60:]))
