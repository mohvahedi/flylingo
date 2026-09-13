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

WHY MULTIPLE SEEDS. This script used to run one seed per arm and then pick between two
pre-written narratives based on whether the single final-epoch figure happened to fall within 5
points. That produced a false finding: a 15-epoch run reported intact 83.5% against shuffled
74.2% and printed "the wiring matters here", but per-epoch the gap oscillates between -5 and +7
points around zero, so the reading was one lucky epoch out of a noisy series. A single epoch of a
single seed cannot separate a 5-point effect from seed noise, and the script must not pretend
otherwise. It now runs several seeds and reports the gap against the seed-to-seed spread, which
is the comparison that actually licenses a conclusion.

The score for an arm is the mean accuracy over the last SCORE_TAIL epochs, because a single final
epoch swings by several points run to run.

    python scripts/measure_plastic_brain.py [epochs] [seeds]

Writes artifacts to brain/runs/plastic_brain/measure.json and measure_seeds.json.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_text  # noqa: E402
from brain.plastic_brain import PlasticBrain  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402


def _maybe_gpu(reservoir, label=""):
    """Route the recurrence to the GPU when there is one, and say so either way.

    The run is dominated by the sparse matrix-vector product, and on this machine that is 26x
    faster on the GPU (5.2 ms per six-step settle against 136 ms). Enabling it is optional and
    reported rather than assumed, so a log always states which device produced the numbers.
    """
    info = reservoir.enable_gpu()
    if info.get("enabled"):
        print(f"device: GPU {info['device']}  ({label or 'recurrence on device'})")
    else:
        print(f"device: CPU  ({info.get('reason', 'gpu not enabled')})")
    return info.get("enabled", False)


EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SCORE_TAIL = 10  # epochs averaged into an arm's score
MODES = ("intact", "shuffled", "random_graph", "no_edges")
WIRED = ("intact", "shuffled", "random_graph")
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
print(f"epochs: {EPOCHS}   seeds: {SEEDS}   score = mean of last {SCORE_TAIL} epochs")

connectome = load_connectome()
per_seed = {}  # seed -> mode -> {"score", "final", "curve"}

for seed_i in range(SEEDS):
    r_seed = 7301 + seed_i
    b_seed = 99 + seed_i
    print(f"\n{'#'*64}\n# seed {seed_i + 1}/{SEEDS}  (reservoir {r_seed}, brain {b_seed})\n{'#'*64}")
    per_seed[seed_i] = {}

    for mode in MODES:
        print(f"\n{'='*64}\n{mode}   [seed {seed_i + 1}]\n{'='*64}")
        r = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=r_seed)
        r.set_mode(mode)
        _maybe_gpu(r, mode)
        brain = PlasticBrain(r, n_pools=4, pool_size=200, seed=b_seed, steps=6)
        brain.set_mode(mode)

        curve = [{"epoch": 0, "accuracy": brain.accuracy(X, y), "loss": None}]
        t0 = time.perf_counter()
        for ep in range(1, EPOCHS + 1):
            idx = np.random.default_rng(ep).permutation(len(X))  # fixed order across arms
            losses = [brain.observe(X[i], y[i])["loss"] for i in idx]
            acc = brain.accuracy(X, y)
            curve.append({"epoch": ep, "accuracy": acc, "loss": float(np.mean(losses))})
            # Print every epoch: this is flushed line by line into the run log, so a stall is
            # distinguishable from progress. A run that only reports at arm end looks identical
            # to a hung one, which is the trap that wastes an hour before anyone notices.
            print(f"    epoch {ep:>3}: acc {acc:6.1%}   loss {np.mean(losses):.4f}", flush=True)

        tail = [c["accuracy"] for c in curve[-SCORE_TAIL:]]
        s = r.plastic_scale

        # SAVE THE TRAINED WEIGHTS, and prove the reported number can be reproduced from them.
        #
        # This used to write only the accuracy curve, so every trained brain was discarded when
        # the process exited: 117,800 trained synaptic scales thrown away and unrecoverable, which
        # meant a reported figure could never be re-examined, re-scored or reused. It also hid a
        # real bug for hours -- a saved checkpoint would have exposed immediately that arm 2's
        # baseline sat at 8x the anatomical weights.
        #
        # Numbers are a description of an artifact; the artifact is the thing worth keeping.
        ckpt = OUT / f"arm_s{seed_i}_{mode}.npz"
        ckpt_meta = brain.save(ckpt)

        # Reload it through the same refusal guards the service uses and recompute the score.
        # If a stored brain cannot reproduce the number it is about to be reported for, the
        # number is not trustworthy and this arm must fail loudly rather than be written out.
        verify_r = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=r_seed)
        verify_r.set_mode(mode)
        verify = PlasticBrain(
            verify_r, n_pools=4, pool_size=200, seed=b_seed, steps=6
        )
        verify.set_mode(mode)
        verify.load(ckpt)
        reloaded_final = verify.accuracy(X, y)
        assert abs(reloaded_final - curve[-1]["accuracy"]) < 1e-6, (
            f"{mode} seed {seed_i}: checkpoint reproduces {reloaded_final:.4f} but the curve "
            f"reports {curve[-1]['accuracy']:.4f}; refusing to record an unreproducible number"
        )

        per_seed[seed_i][mode] = {
            "curve": curve,
            "score": float(statistics.mean(tail)),
            "score_std_within": float(statistics.pstdev(tail)),
            "final_accuracy": curve[-1]["accuracy"],
            "start_accuracy": curve[0]["accuracy"],
            "plastic_edges": int(s.size),
            "scale_max": float(s.max()),
            "seconds": time.perf_counter() - t0,
            "checkpoint": str(ckpt),
            "checkpoint_reproduces_final": float(reloaded_final),
            "checkpoint_updates": int(ckpt_meta["updates"]),
        }
        print(
            f"  --> start {curve[0]['accuracy']:.1%}  final-epoch {curve[-1]['accuracy']:.1%}  "
            f"last-{SCORE_TAIL} mean {per_seed[seed_i][mode]['score']:.1%}   "
            f"scales max {s.max():.2f}   {per_seed[seed_i][mode]['seconds']:.0f}s   "
            f"saved {ckpt.name} (reproduced {reloaded_final:.1%})"
        )

# ------------------------------------------------------- aggregate across seeds
def col(mode, key):
    return [per_seed[s][mode][key] for s in range(SEEDS)]


def mean_std(vals):
    return (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)


print(f"\n{'='*72}\nACROSS {SEEDS} SEEDS   (score = mean of the last {SCORE_TAIL} epochs)\n{'='*72}")
print(f"{'arm':<14} {'mean':>8} {'sd':>7} {'min':>7} {'max':>7}   {'final-epoch mean':>17}")
print("-" * 72)
summary = {}
for mode in MODES:
    m, sd = mean_std(col(mode, "score"))
    fe, _ = mean_std(col(mode, "final_accuracy"))
    summary[mode] = {"score_mean": m, "score_sd": sd, "final_epoch_mean": fe}
    print(
        f"{mode:<14} {m:>7.1%} {sd:>6.1%} {min(col(mode, 'score')):>6.1%} "
        f"{max(col(mode, 'score')):>6.1%}   {fe:>16.1%}"
    )

# The controlled comparison: pair each seed's intact score with the same seed's controls, so the
# seed's difficulty cancels and only the wiring differs.
gaps_vs_best = []
gaps_vs_shuffled = []
gaps_vs_random = []
print("\npaired per seed (same seed, same example order, only the wiring differs):")
print(f"{'seed':>5} {'intact':>8} {'shuffled':>9} {'random':>8} {'best ctl':>9} {'gap':>7}")
for s in range(SEEDS):
    i = per_seed[s]["intact"]["score"]
    sh = per_seed[s]["shuffled"]["score"]
    rg = per_seed[s]["random_graph"]["score"]
    best = max(sh, rg)
    gaps_vs_best.append(i - best)
    gaps_vs_shuffled.append(i - sh)
    gaps_vs_random.append(i - rg)
    print(f"{s + 1:>5} {i:>7.1%} {sh:>8.1%} {rg:>7.1%} {best:>8.1%} {i - best:>+6.1%}")

gm, gsd = mean_std(gaps_vs_best)
sm, ssd = mean_std(gaps_vs_shuffled)
rm, rsd = mean_std(gaps_vs_random)
wins = sum(1 for g in gaps_vs_best if g > 0)
print(f"\n  intact - best control : {gm:+.1%}  (sd {gsd:.1%}, seeds where intact leads {wins}/{SEEDS})")
print(f"  intact - shuffled     : {sm:+.1%}  (sd {ssd:.1%})")
print(f"  intact - random_graph : {rm:+.1%}  (sd {rsd:.1%})")

print(f"\n{'='*72}\nVERDICT\n{'='*72}")
print(f"  majority-class baseline : {chance:.1%}")
print(f"  edge-free control       : {summary['no_edges']['score_mean']:.1%} "
      f"(state is exactly zero, so this must stay at chance)")
# A claim requires the effect to clear the noise, not merely to have the right sign once.
if SEEDS < 3:
    print("  Too few seeds to separate an effect from noise. Re-run with more seeds.")
elif abs(gm) < gsd or wins in (0, SEEDS):
    print(f"  The intact connectome's {gm:+.1%} over the best control is within the seed-to-seed")
    print(f"  spread ({gsd:.1%}) and does not clear it. No wiring advantage is detectable on this")
    print("  task with this design. A recurrent graph is still required: the edge-free control is")
    print("  at chance while every connected arm learns.")
else:
    print(f"  The intact connectome leads the best control by {gm:+.1%} with a seed spread of")
    print(f"  {gsd:.1%}, on {wins}/{SEEDS} seeds. The effect clears the noise, which is the")
    print("  opposite of the frozen-readout result. Replicate on a second machine before calling")
    print("  it a property of the fly.")

(OUT / "measure_seeds.json").write_text(
    json.dumps(
        {
            "config": {"epochs": EPOCHS, "seeds": SEEDS, "score_tail": SCORE_TAIL},
            "per_seed": {str(k): v for k, v in per_seed.items()},
            "summary": summary,
            "gap": {
                "vs_best_control_mean": gm,
                "vs_best_control_sd": gsd,
                "vs_shuffled_mean": sm,
                "vs_random_mean": rm,
                "seeds_intact_leads": wins,
            },
            "chance_majority": chance,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\nwrote {OUT / 'measure_seeds.json'}")
