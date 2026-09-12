"""Can the connectome memorize the 97 phrases? The learnable version of the goal.

The three measurements so far say the option-scoring framing is unlearnable:

  1. vocabulary-disjoint held-out, encoding only : 0.2784 (p=0.294)  no generalisation
  2. learn-to-criterion, all 97 repeated, raw   : 0.536             cannot even fit
  3. reservoir features vs raw vs controls      : 0.567 vs 0.536, intact == shuffled == random

The reason is structural. "Which option is correct" is a RELATIONAL property, and character
n-grams carry no cross-lingual meaning, so no global linear rule over (prompt, option)
vectors can do it. Random projection does not create semantics.

But there is a version of this that is genuinely learnable, and it is what "the fly learns
Spanish" actually means at 97 phrases: MEMORISATION. Give the readout the prompt, ask it for
the answer INDEX, and repeat until it knows them. That is drilling vocabulary, and a
128-dimensional feature space can hold 97 arbitrary associations if the features are
distinct enough.

This is also the architecture the reference FLM project uses: a frozen graph plus a trained
adapter that reads its state.

The comparison that matters stays the same: intact wiring versus a shuffled graph versus a
degree-matched random graph, at matched dimensions. If the measured connectome memorises
better than its controls, the biology is doing something. If all three tie, the connectome
is functioning as a generic random feature map, which is a legitimate and much weaker claim.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_text  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

CURRICULUM = Path(r"D:\Projects\flylingo\brain\brain\curriculum\es-en.json")
MODES = ["intact", "shuffled", "random_graph"]
EPOCHS = 4000
SEEDS = [0, 1, 2]


def load():
    d = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    return [c for u in d["units"] for l in u["lessons"] for c in l["challenges"]]


def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)


def fit_softmax(X, y, n_classes, epochs, lr=0.5, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_classes, X.shape[1])) * 0.01
    b = np.zeros(n_classes)
    n = len(X)
    for _ in range(epochs):
        P = softmax_rows(X @ W.T + b)
        onehot = np.zeros_like(P)
        onehot[np.arange(n), y] = 1
        G = (P - onehot) / n
        W -= lr * (G.T @ X + l2 * W)
        b -= lr * G.sum(axis=0)
    return W, b


def build_features(reservoir, chas, mode, feature_source):
    """One feature vector per challenge, from the prompt alone."""
    reservoir.set_mode(mode)
    out = []
    for ch in chas:
        reservoir.reset()
        text = ch["prompt"] if feature_source == "prompt" else (
            ch["prompt"] + " | " + " | ".join(ch["options"]))
        emb = encode_text(text)
        out.append(reservoir.step(emb))
    return np.stack(out)


def evaluate(name, X, y, n_classes, seeds=SEEDS):
    mu, sd = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    accs = []
    for s in seeds:
        W, b = fit_softmax(Xs, y, n_classes, EPOCHS, seed=s)
        accs.append(float((np.argmax(Xs @ W.T + b, axis=1) == y).mean()))
    return {"name": name, "dim": X.shape[1], "acc_mean": float(np.mean(accs)),
            "acc_min": float(np.min(accs)), "acc_max": float(np.max(accs))}


def main() -> int:
    chas = load()
    n = len(chas)
    y = np.array([c["correctIndex"] for c in chas])
    n_classes = max(len(c["options"]) for c in chas)
    print(f"challenges: {n} | classes: {n_classes} | chance: {1 / n_classes:.3f}")
    print(f"answer distribution: {np.bincount(y, minlength=n_classes).tolist()}")
    print("task: memorise prompt -> answer index, all challenges every epoch")
    print(f"readout: linear {n_classes}-way softmax, {EPOCHS} full-batch epochs, "
          f"{len(SEEDS)} seeds\n")

    conn = load_connectome()
    print(f"graph: {conn.neurons} neurons, {conn.edges} edges\n")

    results = []

    # Baselines with no reservoir: raw prompt encoding, and the option-list encoding used
    # by the original experiment.
    raw_prompt = np.stack([encode_text(c["prompt"]) for c in chas])
    results.append(evaluate("raw prompt encoding", raw_prompt, y, n_classes))

    reservoir = FlyReservoir(conn, embedding_dim=256, dims=128, seed=7301)
    for mode in MODES:
        t0 = time.perf_counter()
        X = build_features(reservoir, chas, mode, "prompt")
        dt = time.perf_counter() - t0
        r = evaluate(mode, X, y, n_classes)
        r["seconds"] = round(dt, 1)
        results.append(r)
        print(f"  {mode}: {X.shape} features in {dt:.1f}s")

    print(f"\n=== memorisation accuracy ===")
    print(f"  {'features':-<26} {'dim':>5} {'mean':>7} {'min':>7} {'max':>7}")
    for r in results:
        print(f"  {r['name']:<26} {r['dim']:>5} {r['acc_mean']:>7.3f} "
              f"{r['acc_min']:>7.3f} {r['acc_max']:>7.3f}")

    by = {r["name"]: r for r in results}
    raw = by["raw prompt encoding"]["acc_mean"]
    intact = by["intact"]["acc_mean"]
    shuf = by["shuffled"]["acc_mean"]
    rand = by["random_graph"]["acc_mean"]

    print("\n=== verdict ===")
    print(f"  chance                        : {1 / n_classes:.3f}")
    print(f"  raw prompt encoding           : {raw:.3f}")
    print(f"  intact connectome             : {intact:.3f}")
    print(f"  shuffled graph                : {shuf:.3f}")
    print(f"  degree-matched random graph   : {rand:.3f}")

    spread = max(intact, shuf, rand) - min(intact, shuf, rand)
    print(f"\n  spread across the three graphs: {spread:.3f}")

    if intact > 0.9 and intact > max(shuf, rand) + 0.05:
        print("  POSITIVE: the measured wiring memorises the phrases better than both")
        print("  control graphs. That is a claim about the specific biology.")
    elif intact > 0.9 and spread <= 0.05:
        print("  MIXED: the connectome as a reservoir CAN memorise the vocabulary (so the")
        print("  fly can be said to learn the phrases), but the measured wiring is")
        print("  indistinguishable from a shuffled or random graph of the same size. The")
        print("  result is therefore about reservoir computing in general, NOT about this")
        print("  connectome. Reported as such.")
    elif max(intact, shuf, rand) > 0.5:
        print("  PARTIAL: partial memorisation, still far from knowing the vocabulary.")
    else:
        print("  NULL: the features do not support memorising even 97 arbitrary")
        print("  associations, so no readout on this substrate learns the phrases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
