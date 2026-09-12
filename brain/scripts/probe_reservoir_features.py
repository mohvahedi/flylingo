"""Does the measured connectome make the Spanish task linearly learnable?

Context, and why this is the experiment that matters.

The raw encoding cannot carry the task: a global linear head over it tops out at 0.495
(train, all 97 repeated) because "which option is correct" is a RELATIONAL property, not a
property of an option's own vector, and character n-grams carry no cross-lingual meaning.

But a reservoir is precisely a machine for turning an input into a high-dimensional
nonlinear feature space, and higher-dimensional random features raise linear separability
(that is the whole basis of echo state networks and random kitchen sinks). So the real
question about the fly is not "does a char n-gram encoder know Spanish" (it does not), it is
"does the MEASURED WIRING supply features that make the task learnable where the raw
encoding cannot".

That is a fair, falsifiable claim, and it is the one the project should be testing. It also
has a natural control: the same reservoir built on a shuffled graph, and on a
degree-matched random graph. If intact beats shuffled, the specific measured wiring is doing
work. If they tie, it does not.

This uses a plain supervised linear readout on purpose. A supervised probe is the cleanest
way to ask "is the information linearly available in these features", with the learning
rule removed from the picture.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_option_pairs  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

CURRICULUM = Path(r"D:\Projects\flylingo\brain\brain\curriculum\es-en.json")
MODES = ["intact", "shuffled", "random_graph"]
EPOCHS = 3000


def load():
    d = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    return [c for u in d["units"] for l in u["lessons"] for c in l["challenges"]]


def features_for(reservoir, chas, mode):
    """One reservoir feature vector per option, per challenge. Cached once per arm."""
    reservoir.set_mode(mode)
    out = []
    for ch in chas:
        reservoir.reset()
        E = encode_option_pairs(ch)
        rows = []
        for j in range(len(E)):
            rows.append(reservoir.step(E[j]))
        out.append(np.stack(rows))
    return out


def fit_binary(X, y, epochs, lr=0.5, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((2, X.shape[1])) * 0.01
    b = np.zeros(2)
    n = len(X)
    for _ in range(epochs):
        Z = X @ W.T + b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        onehot = np.zeros_like(P)
        onehot[np.arange(n), y] = 1
        G = (P - onehot) / n
        W -= lr * (G.T @ X + l2 * W)
        b -= lr * G.sum(axis=0)
    return W, b


def evaluate(name, feats, chas):
    X, y, group = [], [], []
    for i, E in enumerate(feats):
        for j in range(len(E)):
            X.append(E[j])
            y.append(1 if j == chas[i]["correctIndex"] else 0)
            group.append(i)
    X = np.stack(X).astype(np.float64)
    y = np.array(y)
    group = np.array(group)

    # Standardise: reservoirs emit unit-RMS vectors but the scale still matters here.
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd

    W, b = fit_binary(Xs, y, EPOCHS)
    row_acc = float((np.argmax(Xs @ W.T + b, axis=1) == y).mean())
    hits = 0
    for i, ch in enumerate(chas):
        sel = group == i
        s = (Xs[sel] @ W.T + b)[:, 1]
        hits += int(int(np.argmax(s)) == ch["correctIndex"])
    chal_acc = hits / len(chas)

    # A fair chance baseline: raw features have 4 correlated rows per challenge, and
    # always picking a fixed index gets about 0.25 given the near-uniform answer
    # distribution. Report both.
    majority_row = max(y.mean(), 1 - y.mean())
    return {
        "name": name,
        "dim": X.shape[1],
        "rows": X.shape[0],
        "row_acc": row_acc,
        "row_majority": majority_row,
        "challenge_acc": chal_acc,
    }


def main() -> int:
    chas = load()
    print(f"challenges: {len(chas)}, options per challenge: "
          f"{sorted(set(len(c['options']) for c in chas))}")
    print(f"reads: {sum(len(c['options']) for c in chas)} rows total\n")

    conn = load_connectome()
    print(f"graph: {conn.neurons} neurons, {conn.edges} edges")

    results = []

    # Baseline: the raw encoding, no reservoir at all.
    raw = [encode_option_pairs(c) for c in chas]
    results.append(evaluate("raw encoding (no reservoir)", raw, chas))
    print(f"  raw encoding evaluated: dim {results[-1]['dim']}")

    reservoir = FlyReservoir(conn, embedding_dim=256, dims=128, seed=7301)
    for mode in MODES:
        t0 = time.perf_counter()
        feats = features_for(reservoir, chas, mode)
        t1 = time.perf_counter()
        r = evaluate(mode, feats, chas)
        r["feature_seconds"] = round(t1 - t0, 1)
        results.append(r)
        print(f"  {mode}: features in {t1 - t0:.1f}s, dim {r['dim']}")

    print(f"\n=== supervised linear probe, {EPOCHS} full-batch epochs, "
          f"all 97 challenges repeated ===")
    print(f"  {'features':-<28} {'dim':>5} {'row acc':>8} {'majority':>9} {'challenge acc':>14}")
    for r in results:
        print(f"  {r['name']:<28} {r['dim']:>5} {r['row_acc']:>8.3f} "
              f"{r['row_majority']:>9.3f} {r['challenge_acc']:>14.3f}")

    print("\n=== verdict ===")
    raw_acc = results[0]["challenge_acc"]
    by_name = {r["name"]: r for r in results}
    intact = by_name["intact"]["challenge_acc"]
    shuffled = by_name["shuffled"]["challenge_acc"]
    randg = by_name["random_graph"]["challenge_acc"]

    print(f"  raw encoding ceiling          : {raw_acc:.3f} (chance 0.250)")
    print(f"  intact connectome features    : {intact:.3f}")
    print(f"  shuffled control              : {shuffled:.3f}")
    print(f"  degree-matched random control : {randg:.3f}")

    if intact > max(shuffled, randg) + 0.05 and intact > raw_acc + 0.05:
        print("\n  POSITIVE: the measured wiring makes the task more linearly learnable than")
        print("  both control graphs AND than the raw encoding. That is a real claim about the")
        print("  connectome: its specific structure supplies useful nonlinear features.")
    elif intact > raw_acc + 0.05 and intact <= max(shuffled, randg) + 0.05:
        print("\n  PARTIAL: any reservoir helps beyond the raw encoding, but the MEASURED")
        print("  wiring is not distinguishable from a shuffled or random graph. So the benefit")
        print("  is from having a reservoir at all, not from the biology.")
    else:
        print("\n  NULL: the connectome's features are not more linearly learnable than the")
        print("  raw encoding. Reported as such.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
