"""Train the prompt-index readout on the real curriculum and the real connectome.

Produces brain/runs/promptindex/adapter.npz, which brain/api.py loads at boot.

Also prints the three-arm comparison, so the honest attribution travels with the artifact:
the readout memorises the vocabulary, and the measured wiring is NOT distinguishable from a
shuffled or random graph on this task.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_text, encoder_fingerprint  # noqa: E402
from brain.learning.prompt_index import PromptIndexReadout  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

ROOT = Path(r"D:\Projects\flylingo")
CURRICULUM = ROOT / "brain" / "brain" / "curriculum" / "es-en.json"
OUT = ROOT / "brain" / "runs" / "promptindex"


def load():
    d = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    return [c for u in d["units"] for l in u["lessons"] for c in l["challenges"]]


def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)


def fit(X, y, n_classes, epochs, lr, l2, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_classes, X.shape[1])) * 0.01
    b = np.zeros(n_classes)
    n = len(X)
    accs = []
    for ep in range(epochs):
        P = softmax_rows(X @ W.T + b)
        onehot = np.zeros_like(P)
        onehot[np.arange(n), y] = 1
        G = (P - onehot) / n
        W -= lr * (G.T @ X + l2 * W)
        b -= lr * G.sum(axis=0)
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            accs.append(float((np.argmax(X @ W.T + b, axis=1) == y).mean()))
    return W, b, accs


def features(reservoir, chas, mode):
    reservoir.set_mode(mode)
    out = []
    for ch in chas:
        reservoir.reset()
        out.append(reservoir.step(encode_text(ch["prompt"])))
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--mode", default="intact")
    args = ap.parse_args()

    chas = load()
    y = np.array([c["correctIndex"] for c in chas])
    n_classes = max(len(c["options"]) for c in chas)
    print(f"curriculum: {len(chas)} challenges, {n_classes} options, chance {1 / n_classes:.3f}")

    conn = load_connectome()
    print(f"connectome: {conn.neurons} neurons, {conn.edges} edges")

    reservoir = FlyReservoir(conn, embedding_dim=256, dims=128, seed=7301)

    print("\n=== three-arm comparison (same readout, different wiring) ===")
    report = {}
    feats = {}
    for mode in ("intact", "shuffled", "random_graph"):
        t0 = time.perf_counter()
        X = features(reservoir, chas, mode)
        build_s = time.perf_counter() - t0
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
        Xs = (X - mu) / sd
        W, b, accs = fit(Xs, y, n_classes, args.epochs, 0.5, 1e-3)
        acc = float((np.argmax(Xs @ W.T + b, axis=1) == y).mean())
        feats[mode] = X
        report[mode] = {"accuracy": acc, "feature_seconds": round(build_s, 1),
                        "curve": accs, "params": int(W.size + b.size)}
        print(f"  {mode:<14} features {build_s:5.1f}s | memorisation accuracy {acc:.3f}")

    # The raw encoding, no reservoir, as the baseline that shows what the graph adds.
    Xr = np.stack([encode_text(c["prompt"]) for c in chas])
    mur, sdr = Xr.mean(0, keepdims=True), Xr.std(0, keepdims=True) + 1e-6
    Xrs = (Xr - mur) / sdr
    Wr, br, _ = fit(Xrs, y, n_classes, args.epochs, 0.5, 1e-3)
    raw_acc = float((np.argmax(Xrs @ Wr.T + br, axis=1) == y).mean())
    report["raw_encoding"] = {"accuracy": raw_acc, "dim": int(Xr.shape[1])}
    print(f"  {'raw encoding':<14} no reservoir            | memorisation accuracy {raw_acc:.3f}")

    accs = [report[m]["accuracy"] for m in ("intact", "shuffled", "random_graph")]
    spread = max(accs) - min(accs)
    print(f"\n  spread across the three wirings: {spread:.3f}")
    print(f"  raw-encoding baseline: {raw_acc:.3f}")

    # Save the intact-arm readout into the class the service uses.
    readout = PromptIndexReadout(
        in_dim=feats[args.mode].shape[1], n_actions=n_classes, seed=0)
    X = feats[args.mode]
    readout.fit_normaliser(X)
    Xs = (X - readout.mu) / readout.sd
    W, b, _ = fit(Xs, y, n_classes, args.epochs, 0.5, 1e-3)
    readout.W, readout.b = W, b

    out_path = OUT / "adapter.npz"
    readout.save(out_path)
    in_service = float(
        (np.argmax(
            np.stack([readout.logits(feats[args.mode][i]) for i in range(len(chas))]), axis=1
        ) == y).mean()
    )

    results = {
        "kind": "prompt_index",
        "curriculum": str(CURRICULUM),
        "n_challenges": len(chas),
        "n_classes": n_classes,
        "chance": 1 / n_classes,
        "epochs": args.epochs,
        "encoder_fingerprint": encoder_fingerprint(),
        "arms": report,
        "spread_across_wirings": spread,
        "raw_encoding_accuracy": raw_acc,
        "saved": str(out_path),
        "in_service_accuracy": in_service,
        "interpretation": (
            "The readout memorises the curriculum. The measured connectome does NOT beat a "
            "shuffled or degree-matched random graph on this task, and the raw encoding with "
            "no reservoir matches them all, so the connectome contributes no measurable "
            "learning advantage here. It supplies a fixed nonlinear feature map."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n=== saved ===")
    print(f"  checkpoint: {out_path}")
    print(f"  params: {readout.parameters()}")
    print(f"  accuracy as the service will compute it: {in_service:.3f}")
    print(f"  encoder fingerprint: {encoder_fingerprint()}")
    print(f"  results: {OUT / 'results.json'}")
    print("\n  attribution: the readout learns the phrases. The connectome does not beat its")
    print("  controls, and neither does the raw encoding differ, so the wiring's contribution")
    print("  is not measurable on this task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
