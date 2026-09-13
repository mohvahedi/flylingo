"""The experiment that decides it: does the fly's real wiring beat its controls, now that the
brain is doing the choosing and the learning?

This is the question the earlier design could not answer. With a frozen connectome and a separate
readout, every arm scored exactly 1.000 (spread 0.000), because a fixed random projection is as
good as any other fixed random projection. The measurement was incapable of detecting a wiring
advantage even if one existed.

Here the answer comes from the brain's own populations and the learning happens on the brain's
own synapses, so the wiring is inside the loop and the comparison can finally mean something:

    intact        the measured MaleCNS connectome
    shuffled      the same graph under a fixed node relabeling (topology preserved, interfaces
                  misaligned)
    random_graph  a degree-matched random sparse matrix with the same nnz
    no_edges      disconnected; state is exactly zero, so pools are all equal and the argmax is
                  degenerate. Reported, never dressed up.

Every arm starts from the same pristine weights, uses the same pools, the same settling, the same
plastic edge set and the same example order. Only the wiring differs.

Writes artifacts to brain/runs/plastic_brain/measure.json.
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

def _maybe_gpu(reservoir, label=""):
    """Route the matvec to the GPU when there is one, and say so either way.

    The whole run is dominated by the sparse matrix-vector product, and on this machine that is
    26x faster on the GPU (5.2 ms per six-step settle against 136 ms). Left on the CPU a
    converged run takes about 45 minutes; on the GPU it takes under two. Enabling it is optional
    and reported rather than assumed, so a log always says which device produced the numbers.
    """
    info = reservoir.enable_gpu()
    if info.get("enabled"):
        print(f"device: GPU {info['device']}  ({label or 'matvec offloaded'})")
    else:
        print(f"device: CPU  ({info.get('reason', 'gpu not enabled')})")
    return info.get("enabled", False)


EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
MODES = ("intact", "shuffled", "random_graph", "no_edges")
OUT = Path("D:/Projects/flylingo/brain/runs/plastic_brain")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- the task
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
print(f"task: {len(items)} challenges, {len(np.unique(y))} classes")
print(f"majority-class baseline: {chance:.1%}   (uniform chance 25.0%)")

connectome = load_connectome()
results = {}

for mode in MODES:
    print(f"\n{'='*64}\n{mode}\n{'='*64}")
    r = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=7301)
    r.set_mode(mode)
    _maybe_gpu(r, mode)
    brain = PlasticBrain(r, n_pools=4, pool_size=200, seed=99, steps=6)
    brain.set_mode(mode)

    curve = [{"epoch": 0, "accuracy": brain.accuracy(X, y), "loss": None}]
    print(f"  epoch 0 (untrained): {curve[0]['accuracy']:.1%}")
    t0 = time.perf_counter()

    for ep in range(1, EPOCHS + 1):
        idx = np.random.default_rng(ep).permutation(len(X))
        losses = []
        for i in idx:
            losses.append(brain.observe(X[i], y[i])["loss"])
        acc = brain.accuracy(X, y)
        curve.append({"epoch": ep, "accuracy": acc, "loss": float(np.mean(losses))})
        print(f"  epoch {ep:>3}: acc {acc:6.1%}   loss {np.mean(losses):.4f}")

    s = r.plastic_scale
    results[mode] = {
        "curve": curve,
        "final_accuracy": curve[-1]["accuracy"],
        "start_accuracy": curve[0]["accuracy"],
        "gain": curve[-1]["accuracy"] - curve[0]["accuracy"],
        "final_loss": curve[-1]["loss"],
        "plastic_edges": int(s.size),
        "scale_max": float(s.max()),
        "scale_mean": float(s.mean()),
        "scale_std": float(s.std()),
        "seconds": time.perf_counter() - t0,
    }
    print(
        f"  --> {results[mode]['start_accuracy']:.1%} to {results[mode]['final_accuracy']:.1%} "
        f"(+{results[mode]['gain']:.1%})   scales max {s.max():.2f}   "
        f"{results[mode]['seconds']:.0f}s"
    )

# ---------------------------------------------------------------- the verdict
print(f"\n{'='*64}\nVERDICT\n{'='*64}")
print(f"  majority-class baseline        : {chance:.1%}")
for m in MODES:
    print(
        f"  {m:<14} start {results[m]['start_accuracy']:6.1%}  ->  "
        f"final {results[m]['final_accuracy']:6.1%}   gain {results[m]['gain']:+6.1%}"
    )

best = max(results[m]["final_accuracy"] for m in ("intact", "shuffled", "random_graph"))
spread = max(results[m]["final_accuracy"] for m in ("intact", "shuffled", "random_graph")) - min(
    results[m]["final_accuracy"] for m in ("intact", "shuffled", "random_graph")
)
intact = results["intact"]["final_accuracy"]
ties = [m for m in ("shuffled", "random_graph") if abs(intact - results[m]["final_accuracy"]) < 0.05]

print(f"\n  spread across intact / shuffled / random : {spread:.1%}")
if not ties:
    print("  The intact connectome finished clear of both controls. The wiring matters here, which")
    print("  is the opposite of the frozen-readout result and would need replication before it is")
    print("  claimed as a property of the fly.")
else:
    print(f"  Intact ties {', '.join(ties)} within 5 points. Letting the brain itself learn has not")
    print("  produced a wiring advantage on this task, which is now a result about a plastic")
    print("  connectome rather than about a frozen feature map.")

results["_meta"] = {
    "epochs": EPOCHS,
    "challenges": len(items),
    "chance_majority": chance,
    "spread_intact_shuffled_random": spread,
    "ties_within_5pts": ties,
}
(OUT / "measure.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'measure.json'}")
