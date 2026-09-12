"""Verify the learn-to-criterion protocol is actually learnable, before comparing on it.

The previous null was a category error: the evaluation asked for vocabulary-disjoint
generalisation, which a character n-gram encoder cannot do, so the architecture comparison
was meaningless. The rule that came out of that (see NOTES_task_learnability.md) is to
establish a task is learnable before comparing architectures on it.

So this applies that rule to the NEW protocol. It measures the learn-to-criterion ceiling
with the reservoir removed: can a plain linear readout over the encoding learn all 97
phrases when they are presented repeatedly?

If yes, the protocol is solvable and an intact-versus-control comparison on it is
meaningful. If no, the reframe failed and must be fixed before any further runs.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_option_pairs  # noqa: E402

CURRICULUM = Path(r"D:\Projects\flylingo\brain\brain\curriculum\es-en.json")


def load():
    d = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    return [c for u in d["units"] for l in u["lessons"] for c in l["challenges"]]


def fit_binary(X, y, epochs, lr=0.5, l2=1e-3, seed=0):
    """One linear correct-vs-distractor model, full-batch gradient descent."""
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


def main() -> int:
    chas = load()
    n = len(chas)
    print(f"challenges: {n}")

    # Feature rows: every option of every challenge, labelled correct or not.
    X, y, group = [], [], []
    for i, ch in enumerate(chas):
        E = encode_option_pairs(ch)
        for j in range(len(E)):
            X.append(E[j])
            y.append(1 if j == ch["correctIndex"] else 0)
            group.append(i)
    X = np.stack(X)
    y = np.array(y)
    group = np.array(group)
    print(f"rows: {X.shape}, positives {int(y.sum())} ({y.mean():.3f})")

    def challenge_acc(W, b):
        """Whole-challenge accuracy: pick the option with the highest correct score."""
        hits = 0
        for i, ch in enumerate(chas):
            sel = group == i
            s = (X[sel] @ W.T + b)[:, 1]
            hits += int(int(np.argmax(s)) == ch["correctIndex"])
        return hits / n

    print("\n=== learn-to-criterion: repeated full-batch fits over ALL 97 challenges ===")
    print("  epochs | train challenge acc | binary row acc | majority-row baseline")
    majority = max(y.mean(), 1 - y.mean())
    curve = []
    for epochs in (5, 25, 100, 300, 800, 2000):
        W, b = fit_binary(X, y, epochs)
        ca = challenge_acc(W, b)
        ra = float((np.argmax(X @ W.T + b, axis=1) == y).mean())
        curve.append((epochs, ca))
        print(f"  {epochs:6d} | {ca:19.3f} | {ra:14.3f} | {majority:.3f}")

    best = curve[-1][1]
    print("\n=== verdict ===")
    print(f"  encoding-only ceiling under learn-to-criterion: {best:.3f} (chance 0.250)")
    if best >= 0.6:
        print("  LEARNABLE. A readout over this encoding can fit all 97 phrases when they are")
        print("  repeated, so an intact-versus-control comparison on this protocol is")
        print("  meaningful: the task is solvable and the only question left is whether the")
        print("  measured connectome helps more than a shuffled graph or a plain linear head.")
    else:
        print("  NOT sufficiently learnable. The protocol must be fixed before any arm is")
        print("  compared, or the next result will repeat the same category error.")

    # For contrast, restate the vocabulary-disjoint ceiling that caused the original null.
    print("\n  for contrast, vocabulary-disjoint held-out was 0.2784 (p=0.294): the number")
    print("  that made the first null uninformative about the connectome.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
