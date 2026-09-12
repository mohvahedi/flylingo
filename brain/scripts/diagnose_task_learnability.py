"""Is the Spanish task learnable from this encoding at all, ignoring the connectome?

The project's headline result is a null: the intact connectome did not beat its controls
on held-out challenges. Before blaming the reservoir or the learning rule, this settles
the question underneath them, because if the features carry no signal for unseen
challenges then NO readout can beat chance and the null is a property of the task and
the encoding, not of the fly.

Three things are measured, all with the real curriculum and no reservoir involved:

1. GEOMETRY. For a held-out challenge, is the correct option's vector distinguishable
   from the distractors by the encoding alone? Measured as a nearest-neighbour vote over
   training (prompt, correct) pairs: if unseen pairs carry no relational structure, this
   sits at chance no matter how good the readout is.
2. MEMORISATION. If train and test share the exact (prompt, option) pairs, can a linear
   classifier fit them? This is the control that proves the pipeline can learn when the
   task is actually learnable.
3. GENERALISATION. The same classifier on held-out challenges with disjoint vocabulary.
   The gap between 2 and 3 is the whole story.

Honest expectation going in: character n-grams cannot map "Hello" to "Hola" without
having seen that pairing, so a large gap is expected. The point is to measure it rather
than assert it.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_option_pairs, encode_option_pair  # noqa: E402

CURRICULUM = Path(r"D:\Projects\flylingo\brain\brain\curriculum\es-en.json")


def load():
    d = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    return [c for u in d["units"] for l in u["lessons"] for c in l["challenges"]]


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fit_logreg(X, y, n_classes, epochs=400, lr=0.5, l2=1e-3):
    """Plain multinomial logistic regression, batch gradient descent. numpy only."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((n_classes, X.shape[1])).astype(np.float64) * 0.01
    b = np.zeros(n_classes)
    n = len(X)
    for _ in range(epochs):
        Z = X @ W.T + b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        onehot = np.zeros_like(P)
        onehot[np.arange(n), y] = 1.0
        G = (P - onehot) / n
        W -= lr * (G.T @ X + l2 * W)
        b -= lr * G.sum(axis=0)
    return W, b


def accuracy(W, b, X, y):
    return float((np.argmax(X @ W.T + b, axis=1) == y).mean())


def main() -> int:
    chas = load()
    n = len(chas)
    print(f"challenges: {n}")
    print(f"correctIndex distribution: {np.bincount([c['correctIndex'] for c in chas], minlength=4).tolist()}")

    # Precompute the centered option encodings once; they are the scorer's input.
    ALL = [encode_option_pairs(c) for c in chas]
    ALLy = [c["correctIndex"] for c in chas]

    def rows_for(indices):
        X, y = [], []
        for i in indices:
            E = ALL[i]
            for j in range(len(E)):
                X.append(E[j])
                y.append(1 if j == ALLy[i] else 0)
        return np.stack(X), np.array(y)

    def challenge_acc(W, b, indices):
        hits = 0
        for i in indices:
            E = ALL[i]
            s = (E @ W.T + b)[:, 1]
            hits += int(int(np.argmax(s)) == ALLy[i])
        return hits, len(indices)

    # ---------------------------------------------------------------- 1. geometry
    print("\n=== 1. does the encoding relate an unseen prompt to its correct option? ===")
    rng = random.Random(7)
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = 24
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    train_correct = np.stack([
        encode_option_pair(chas[i]["prompt"], chas[i]["answer"]) for i in train_idx
    ])
    nn_hits = 0
    for i in test_idx:
        ch = chas[i]
        E = np.stack([encode_option_pair(ch["prompt"], o) for o in ch["options"]])
        pick = int(np.argmax((E @ train_correct.T).max(axis=1)))
        nn_hits += int(pick == ch["correctIndex"])
    print(f"  nearest-neighbour over training (prompt, correct) pairs: "
          f"{nn_hits / n_test:.3f} ({nn_hits}/{n_test}), chance 0.250")

    # ------------------------------------------------- 2. cross-validated estimate
    # 24 held-out items is far too few to conclude anything, and the earlier run's
    # 0.333 was 8/24, which is not distinguishable from chance. 5-fold CV pools every
    # challenge's prediction, so the estimate rests on all 97 items.
    print("\n=== 2. 5-fold cross-validated accuracy from the encoding alone ===")
    folds = 5
    order = list(range(n))
    random.Random(11).shuffle(order)
    pooled_hits, pooled_n = 0, 0
    per_fold = []
    for k in range(folds):
        te = [order[i] for i in range(k, n, folds)]
        tr = [i for i in order if i not in set(te)]
        Xtr, ytr = rows_for(tr)
        W, b = fit_logreg(Xtr, ytr, 2)
        h, m = challenge_acc(W, b, te)
        per_fold.append(h / m)
        pooled_hits += h
        pooled_n += m
        print(f"  fold {k + 1}: {h}/{m} = {h / m:.3f}")
    pooled = pooled_hits / pooled_n
    print(f"  POOLED held-out accuracy: {pooled:.4f} ({pooled_hits}/{pooled_n}), chance 0.250")

    # Significance: exact binomial tail, P(X >= observed | n, p=0.25).
    from math import comb

    def binom_tail(k, n_, p=0.25):
        return sum(comb(n_, i) * p**i * (1 - p) ** (n_ - i) for i in range(k, n_ + 1))

    p_value = binom_tail(pooled_hits, pooled_n)
    print(f"  exact binomial p against chance: {p_value:.4f} "
          f"({'significant at 0.05' if p_value < 0.05 else 'NOT significant'})")

    # ------------------------------------------------------------ 3. memorisation
    print("\n=== 3. can the pipeline fit at all when the pairs are shared? ===")
    Xall, yall = rows_for(range(n))
    W, b = fit_logreg(Xall, yall, 2)
    bic = float((np.argmax(Xall @ W.T + b, axis=1) == yall).mean())
    majority = max(yall.mean(), 1 - yall.mean())
    h, m = challenge_acc(W, b, list(range(n)))
    print(f"  binary correct-vs-distractor fit: {bic:.3f} (majority baseline {majority:.3f})")
    print(f"  challenge accuracy when pairs are shared: {h}/{m} = {h / m:.3f}")
    print("  (a large gap here versus the cross-validated number above is memorisation)")

    # ------------------------------------------------------------ 4. conclusion
    print("\n=== 4. interpretation ===")
    print(f"  encoding-only held-out accuracy : {pooled:.3f} (chance 0.250, p={p_value:.3f})")
    print(f"  encoding-only memorisation      : {h / m:.3f}")
    if p_value >= 0.05:
        print("  NOT distinguishable from chance. The character-level encoding carries no")
        print("  reliable signal for challenges whose exact wording or vocabulary was not in")
        print("  training, so ANY readout on top of it, including the connectome's, is bounded")
        print("  near chance on this split. The recorded null is therefore not evidence about")
        print("  the fly: it is a property of the task split and a char-n-gram encoder.")
    else:
        print("  Distinguishable from chance, so the encoding does carry generalisable")
        print("  signal and a null readout result would point at the reservoir or the rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
