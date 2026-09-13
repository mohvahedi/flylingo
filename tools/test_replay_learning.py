"""Can the fly learn a lesson WITHIN a session? Testing experience replay.

What the previous runs established:
  - The measured offline checkpoint reaches 1.000 memorisation, but that is MANY epochs over the
    same 97 phrase-answer pairs.
  - Online, one answer gives one update. One pass over 97 pairs is nowhere near enough, which is
    why every learning rate and both update rules plateaued around 27-40% with wild variance.

So the honest question is not "which learning rate" but "how many updates per interaction". A
real learner rehearses. Experience replay is the standard technique for exactly this: keep what
you have seen, and take extra gradient steps from it between real interactions.

This measures accuracy over a session of 100 answers for several replay depths. If a shallow
replay converges in roughly the number of answers a viewer would actually watch, then live
training is deliverable, and the number says how much rehearsal it needs. If none of them do,
that is the answer and it should be reported rather than tuned around.
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


def run(replay, lr, steps=140, seed=0):
    """One pass through the lesson, with `replay` extra supervised updates per answer."""
    rng = np.random.default_rng(seed)
    r = PromptIndexReadout(in_dim=DIMS, n_actions=ACTIONS, seed=0)
    r.standardise = True
    r.mu = X.mean(axis=0)
    r.sd = X.std(axis=0) + 1e-6

    buf_x: list[np.ndarray] = []
    buf_y: list[int] = []
    hits = []

    def step_update(x, y_true):
        xn = (x - r.mu) / r.sd
        p = softmax(r.W @ xn + r.b).astype(np.float64)
        t = np.zeros(ACTIONS)
        t[int(y_true)] = 1.0
        g = p - t
        r.W -= lr * np.outer(g, xn)
        r.b -= lr * g

    for i in range(steps):
        j = i % N
        x, y_true = X[j], Y[j]
        a, _ = r.sample(x, rng)
        hits.append(int(a) == int(y_true))

        buf_x.append(x)
        buf_y.append(y_true)
        # the real interaction, then rehearsal on what has been seen
        step_update(x, y_true)
        for _ in range(replay):
            if not buf_x:
                break
            k = int(rng.integers(0, len(buf_x)))
            step_update(buf_x[k], buf_y[k])
    return np.array(hits)


def blocks(h, b=20):
    return " ".join(f"{100*np.mean(h[i:i+b]):3.0f}%" for i in range(0, len(h), b))


print()
print("accuracy in blocks of 20 answers, mean of 3 seeds")
print("=" * 78)
print(f"{'replay':>7} {'lr':>6}  " + " ".join(f"{'a'+str(i+1):>6}" for i in range(7)))
print("-" * 78)
for replay in (0, 5, 20, 60, 150):
    for lr in (0.5, 0.2):
        rows = []
        for s in (0, 1, 2):
            h = run(replay, lr, seed=s)
            rows.append([100 * np.mean(h[i:i + 20]) for i in range(0, 140, 20)])
        m = np.mean(rows, axis=0)
        tag = "0 (plain)" if replay == 0 else str(replay)
        print(f"{tag:>7} {lr:>6}  " + " ".join(f"{v:5.0f}%" for v in m))
    print()

print("=" * 78)
print("Per-answer outcome, replay=150 lr=0.5 (1 = right), first 100 answers:")
h = run(150, 0.5, steps=140, seed=0)
for i in range(0, 100, 50):
    print("  " + "".join("1" if v else "." for v in h[i:i + 50]))
print()
print("Per-answer outcome, replay=0 (plain online), for comparison:")
h = run(0, 0.5, steps=140, seed=0)
for i in range(0, 100, 50):
    print("  " + "".join("1" if v else "." for v in h[i:i + 50]))
