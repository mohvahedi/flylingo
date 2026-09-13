"""How much training does the plastic brain need before it picks the right option reliably?

The 15-epoch measurement left the intact arm at 53.6% with its loss still falling (1.85 -> 1.10),
so "is it undertrained?" is an open question, not a guess. This runs the intact connectome for as
long as it keeps improving and reports where it actually plateaus.

Two things are reported separately, because they answer different questions:
  * train accuracy, which is what the demo shows on screen
  * whether the loss is still falling, which says whether more epochs would keep helping

Usage: python scripts/train_plastic_to_convergence.py [max_epochs]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_text  # noqa: E402
from brain.plastic_brain import PlasticBrain  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

MAX_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
PATIENCE = 8  # stop after this many epochs with no new best

cur = json.load(open("brain/curriculum/es-en.json", encoding="utf-8"))
items = [
    (c["prompt"], int(c["correctIndex"]))
    for u in cur["units"]
    for l in u["lessons"]
    for c in l["challenges"]
]
X = np.stack([encode_text(p) for p, _ in items])
y = np.array([t for _, t in items], dtype=int)
chance = float(np.bincount(y).max() / len(y))

r = FlyReservoir(load_connectome(), embedding_dim=256, dims=128, seed=7301)
pb = PlasticBrain(r, n_pools=4, pool_size=200, seed=99, steps=6, lr=0.01)
pb.set_mode("intact")

print(f"task   : {len(items)} challenges, 4 options, majority baseline {chance:.1%}")
print(f"params : {r.plastic_scale.size:,} plastic synapses on real edges")
print(f"budget : up to {MAX_EPOCHS} epochs, stopping after {PATIENCE} with no improvement\n")
print(f"{'epoch':>5} {'train acc':>10} {'loss':>8} {'s_max':>7} {'s_std':>7}  note")
print("-" * 62)

best = pb.accuracy(X, y)
best_ep = 0
history = [{"epoch": 0, "accuracy": best, "loss": None}]
t0 = time.perf_counter()
stale = 0

for ep in range(1, MAX_EPOCHS + 1):
    idx = np.random.default_rng(ep).permutation(len(X))
    losses = []
    for i in idx:
        losses.append(pb.observe(X[i], y[i])["loss"])
    loss = float(np.mean(losses))
    acc = pb.accuracy(X, y)
    history.append({"epoch": ep, "accuracy": acc, "loss": loss})

    note = ""
    if acc > best:
        best, best_ep, stale = acc, ep, 0
        note = "new best"
    else:
        stale += 1

    print(
        f"{ep:>5} {acc:>10.1%} {loss:>8.4f} {r.plastic_scale.max():>7.2f} "
        f"{r.plastic_scale.std():>7.3f}  {note}"
    )

    if stale >= PATIENCE:
        print(f"\nstopping: no improvement for {PATIENCE} epochs")
        break

elapsed = time.perf_counter() - t0
print(f"\n{'='*62}")
print(f"best train accuracy : {best:.1%}  (epoch {best_ep})   majority baseline {chance:.1%}")
print(f"final loss          : {history[-1]['loss']:.4f}")
print(f"epochs run          : {len(history)-1}   in {elapsed/60:.1f} min")
print(f"scale max           : {r.plastic_scale.max():.2f}  (clip is 20)")

# is the remaining error made of near-misses or confident mistakes?
probs = np.stack([pb.forward(X[i])[0] for i in range(len(X))])
pred = probs.argmax(axis=1)
ok = pred == y
print(f"\nmisses: {int((~ok).sum())}/{len(y)}")
if (~ok).sum():
    conf = probs[~ok].max(axis=1)
    print(f"  mean confidence on a miss : {conf.mean():.3f}")
    print(f"  confident misses (>0.5)   : {int((conf > 0.5).sum())}")
    print(f"  near-miss misses (<0.4)   : {int((conf < 0.4).sum())}")

out = Path("D:/Projects/flylingo/brain/runs/plastic_brain")
out.mkdir(parents=True, exist_ok=True)
(out / "convergence.json").write_text(
    json.dumps(
        {
            "history": history,
            "best_accuracy": best,
            "best_epoch": best_ep,
            "majority_baseline": chance,
            "epochs_run": len(history) - 1,
            "minutes": elapsed / 60,
            "plastic_edges": int(r.plastic_scale.size),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\nwrote {out / 'convergence.json'}")
