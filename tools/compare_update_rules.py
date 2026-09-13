"""Why does the fresh brain learn so slowly, and is there an honest fix?

Finding from the tuning run: no learning rate gets the online readout past about 33%. The cause
is the UPDATE RULE, not the rate. The current rule reinforces the action the fly happened to
sample: on a correct answer the target is that action, and on a wrong answer the target is
"uniform over everything else". The second case carries no information about which option WAS
right, so a wrong answer teaches almost nothing and the fly has to stumble onto each of the 97
phrase-answer pairs by luck before it can learn them.

This compares that against a supervised update, which is what a lesson actually is: the course
has an answer key, so a wrong answer can say "it was option 3", not merely "not option 1".
That is memorisation of the phrase to answer mapping, and it is the same thing the offline
training does, so it is consistent with the architecture rather than a shortcut around it.

Reports accuracy in blocks of 10 so the shape of the curve is visible.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"D:\Projects\flylingo\brain")))

from brain.curriculum import load_curriculum          # noqa: E402
from brain.encoders import encode_text                 # noqa: E402
from brain.learning.prompt_index import PromptIndexReadout  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome   # noqa: E402

DIMS, ACTIONS, EMB = 128, 4, 256

conn = load_connectome()
cur = load_curriculum()
challenges = cur.challenges()

dr = FlyReservoir(conn, embedding_dim=EMB, dims=DIMS, seed=7301)
X, Y = [], []
for ch in challenges:
    dr.reset()
    X.append(dr.step(encode_text(ch["prompt"])).astype(np.float64))
    Y.append(int(ch["correctIndex"]))
X = np.stack(X)
Y = np.array(Y)
N = len(Y)
print(f"{N} challenges, features {X.shape}")


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def run(mode, lr=5.0, steps=291, seed=0, fit_norm=True):
    """mode: 'bandit' = reward only; 'supervised' = the lesson's answer key."""
    rng = np.random.default_rng(seed)
    r = PromptIndexReadout(in_dim=DIMS, n_actions=ACTIONS, seed=0)
    r.standardise = True
    if fit_norm:
        r.mu = X.mean(axis=0)
        r.sd = X.std(axis=0) + 1e-6

    hits = []
    seen = []
    for i in range(steps):
        j = i % N
        x = X[j]
        if not fit_norm:
            seen.append(x)
            S = np.stack(seen)
            r.mu, r.sd = S.mean(axis=0), S.std(axis=0) + 1e-6

        a, _ = r.sample(x, rng)
        ok = int(a) == int(Y[j])
        hits.append(int(ok))

        p = softmax(r.W @ ((x - r.mu) / r.sd) + r.b).astype(np.float64)
        if mode == "supervised":
            target = np.zeros(ACTIONS)
            target[int(Y[j])] = 1.0
        else:
            target = np.zeros(ACTIONS)
            if ok:
                target[a] = 1.0
            else:
                target = np.full(ACTIONS, 1.0 / (ACTIONS - 1))
                target[a] = 0.0
            target = target / target.sum()
        g = p - target
        r.W -= lr * 0.01 * np.outer(g, (x - r.mu) / r.sd)
        r.b -= lr * 0.01 * g
    return np.array(hits)


def curve(hits, block=10):
    return " ".join(f"{100*np.mean(hits[i:i+block]):3.0f}" for i in range(0, len(hits), block))


print()
print("accuracy per block of 10 answers (%), averaged over 3 seeds")
print("=" * 78)
for mode in ("bandit", "supervised"):
    for lr in (5.0, 20.0):
        runs = [run(mode, lr=lr, seed=s) for s in (0, 1, 2)]
        # mean accuracy per block across the seeds
        blocks = np.mean([np.array([np.mean(h[i:i+10]) for i in range(0, len(h), 10)]) for h in runs], axis=0)
        print(f"{mode:11s} lr={lr:<5} " + " ".join(f"{100*b:3.0f}" for b in blocks[:12]))
        print(f"{'':17s} after 97: {100*blocks[9]:.0f}%   after 194: {100*blocks[19]:.0f}%   "
              f"after 291: {100*blocks[-1]:.0f}%")
        print()

print("=" * 78)
print("First 60 answers, supervised lr=20 (1 = right):")
h = run("supervised", lr=20.0, seed=0)
print("  " + "".join("1" if v else "." for v in h[:60]))
print()
print("First 60 answers, bandit lr=5 (1 = right), for comparison:")
h = run("bandit", lr=5.0, seed=0)
print("  " + "".join("1" if v else "." for v in h[:60]))
