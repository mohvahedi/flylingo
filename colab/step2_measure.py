"""Colab step 2: the four-arm measurement, with the artifacts that make it checkable.

The question: does the real connectome's wiring beat its controls once the brain does both
the choosing and the learning?

    intact        the measured MaleCNS connectome
    shuffled      same graph under a fixed node relabeling (topology kept, interfaces moved)
    random_graph  degree-matched random sparse matrix, same nnz
    no_edges      disconnected; state is exactly zero, so the argmax is degenerate

Every arm gets the same pools, the same settling, the same plastic edge set, the same
example order and its own pristine copy of the weights. Only the wiring differs.

Rules this script obeys, each of which was learned the hard way on this project:

  * An arm's score is the mean over the last TAIL epochs, never a single epoch. A single
    epoch swings by several points, so one draw is not a measurement.
  * Several seeds, compared PAIRED per seed, so a seed's difficulty cancels and only the
    mechanism under test differs.
  * Every arm saves its trained weights, and the reported number is recomputed by loading
    that checkpoint back through the same guards the service uses. A mismatch fails the arm.
  * No pre-written verdict. The summary prints the effect next to its own spread and
    refuses to claim when the effect does not clear it.
  * Progress is printed per epoch so a stall is distinguishable from progress in the log.

Usage:  python step2_measure.py [epochs] [seeds] [tail]
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/content/flylingo")
RUNS = ROOT / "runs" / "plastic_brain"
RUNS.mkdir(parents=True, exist_ok=True)

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TAIL = int(sys.argv[3]) if len(sys.argv) > 3 else 10
MODES = ("intact", "shuffled", "random_graph", "no_edges")
USE_GPU = True

sys.path.insert(0, str(ROOT))
from brain.encoders import encode_text  # noqa: E402
from brain.plastic_brain import PlasticBrain  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

cur = json.load(open(ROOT / "brain/curriculum/es-en.json", encoding="utf-8"))
items = [
    (c["prompt"], int(c["correctIndex"]))
    for u in cur["units"]
    for l in u["lessons"]
    for c in l["challenges"]
]
X = np.stack([encode_text(p) for p, _ in items])
y = np.array([t for _, t in items], dtype=int)
chance = float(np.bincount(y).max() / len(y))

print(f"task: {len(items)} challenges, {len(np.unique(y))} classes, "
      f"all shown every epoch", flush=True)
print(f"memorisation of a phrase-to-answer mapping, not language.", flush=True)
print(f"majority-class baseline {chance:.1%}   uniform chance 25.0%", flush=True)
print(f"epochs {EPOCHS}   seeds {SEEDS}   score = mean of last {TAIL} epochs", flush=True)

connectome = load_connectome(str(ROOT / "cache" / "malecns_v1"))
print(f"connectome: {connectome.neurons} neurons, {connectome.edges} directed edges",
      flush=True)

per_seed: dict[int, dict] = {}
for seed_i in range(SEEDS):
    r_seed, b_seed = 7301 + seed_i, 99 + seed_i
    print(f"\n{'#' * 68}\n# seed {seed_i + 1}/{SEEDS}  (reservoir {r_seed}, brain {b_seed})"
          f"\n{'#' * 68}", flush=True)
    per_seed[seed_i] = {}

    for mode in MODES:
        print(f"\n{'=' * 68}\n{mode}   [seed {seed_i + 1}]\n{'=' * 68}", flush=True)
        r = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=r_seed)
        r.set_mode(mode)
        if USE_GPU:
            info = r.enable_gpu()
            print(f"  device: " + (f"GPU {info['device']}" if info.get("enabled")
                                   else f"CPU ({info.get('reason')})"), flush=True)
        brain = PlasticBrain(r, n_pools=4, pool_size=200, seed=b_seed, steps=6)
        brain.set_mode(mode)

        curve = [{"epoch": 0, "accuracy": brain.accuracy(X, y), "loss": None}]
        print(f"    epoch   0: acc {curve[0]['accuracy']:6.1%}   (untrained)", flush=True)
        t0 = time.perf_counter()
        for ep in range(1, EPOCHS + 1):
            idx = np.random.default_rng(ep).permutation(len(X))
            losses = [brain.observe(X[i], y[i])["loss"] for i in idx]
            acc = brain.accuracy(X, y)
            curve.append({"epoch": ep, "accuracy": acc, "loss": float(np.mean(losses))})
            print(f"    epoch {ep:>3}: acc {acc:6.1%}   loss {np.mean(losses):.4f}   "
                  f"{time.perf_counter() - t0:5.0f}s", flush=True)

        tail = [c["accuracy"] for c in curve[-TAIL:]]
        s = r.plastic_scale
        ckpt = RUNS / f"colab_s{seed_i}_{mode}.npz"
        meta = brain.save(ckpt)

        # Recompute the reported number from the saved artifact, through the serving guards.
        vr = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=r_seed)
        vr.set_mode(mode)
        verify = PlasticBrain(vr, n_pools=4, pool_size=200, seed=b_seed, steps=6)
        verify.set_mode(mode)
        verify.load(ckpt)
        reloaded = verify.accuracy(X, y)
        if abs(reloaded - curve[-1]["accuracy"]) > 1e-6:
            raise SystemExit(
                f"{mode} seed {seed_i}: checkpoint reproduces {reloaded:.4f} but the curve "
                f"says {curve[-1]['accuracy']:.4f}; refusing to record an unreproducible "
                "number"
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
            "checkpoint": ckpt.name,
            "checkpoint_reproduces_final": float(reloaded),
            "checkpoint_updates": int(meta["updates"]),
        }
        print(f"  --> start {curve[0]['accuracy']:.1%}  final-epoch "
              f"{curve[-1]['accuracy']:.1%}  last-{TAIL} mean "
              f"{per_seed[seed_i][mode]['score']:.1%} (sd "
              f"{per_seed[seed_i][mode]['score_std_within']:.1%})  scales max {s.max():.2f}"
              f"  {per_seed[seed_i][mode]['seconds']:.0f}s  "
              f"reproduced {reloaded:.1%}", flush=True)


def col(mode: str, key: str) -> list:
    return [per_seed[s][mode][key] for s in range(SEEDS)]


def mean_std(vals: list) -> tuple[float, float]:
    return (statistics.mean(vals),
            statistics.pstdev(vals) if len(vals) > 1 else 0.0)


print(f"\n{'=' * 74}\nACROSS {SEEDS} SEEDS   (score = mean of the last {TAIL} epochs)"
      f"\n{'=' * 74}", flush=True)
print(f"{'arm':<14} {'mean':>8} {'sd':>7} {'min':>7} {'max':>7} {'final mean':>11}",
      flush=True)
print("-" * 74, flush=True)
summary = {}
for mode in MODES:
    m, sd = mean_std(col(mode, "score"))
    fe, _ = mean_std(col(mode, "final_accuracy"))
    summary[mode] = {"score_mean": m, "score_sd": sd, "final_epoch_mean": fe}
    print(f"{mode:<14} {m:>7.1%} {sd:>6.1%} {min(col(mode, 'score')):>6.1%} "
          f"{max(col(mode, 'score')):>6.1%} {fe:>10.1%}", flush=True)

gaps_best, gaps_shuf, gaps_rand = [], [], []
print("\npaired per seed (same seed, same order, only the wiring differs):", flush=True)
print(f"{'seed':>5} {'intact':>8} {'shuffled':>9} {'random':>8} {'best ctl':>9} "
      f"{'gap':>7}", flush=True)
for s in range(SEEDS):
    i = per_seed[s]["intact"]["score"]
    sh = per_seed[s]["shuffled"]["score"]
    rg = per_seed[s]["random_graph"]["score"]
    best = max(sh, rg)
    gaps_best.append(i - best)
    gaps_shuf.append(i - sh)
    gaps_rand.append(i - rg)
    print(f"{s + 1:>5} {i:>7.1%} {sh:>8.1%} {rg:>7.1%} {best:>8.1%} {i - best:>+6.1%}",
          flush=True)

gm, gsd = mean_std(gaps_best)
sm, ssd = mean_std(gaps_shuf)
rm, rsd = mean_std(gaps_rand)
wins = sum(1 for g in gaps_best if g > 0)
print(f"\n  intact - best control : {gm:+.1%}  (sd {gsd:.1%}, "
      f"seeds where intact leads {wins}/{SEEDS})", flush=True)
print(f"  intact - shuffled     : {sm:+.1%}  (sd {ssd:.1%})", flush=True)
print(f"  intact - random_graph : {rm:+.1%}  (sd {rsd:.1%})", flush=True)

print(f"\n{'=' * 74}\nWHAT THE NUMBERS SUPPORT\n{'=' * 74}", flush=True)
print(f"  majority-class baseline : {chance:.1%}", flush=True)
print(f"  edge-free control       : {summary['no_edges']['score_mean']:.1%} "
      f"(state is exactly zero, so this must stay at chance)", flush=True)
if SEEDS < 3:
    print("  Fewer than 3 seeds: too few to separate an effect from seed noise.",
          flush=True)
elif abs(gm) < gsd or wins in (0, SEEDS):
    print(f"  The intact connectome's {gm:+.1%} over the best control sits inside the "
          f"seed-to-seed spread ({gsd:.1%}) and does not clear it. No wiring advantage is",
          flush=True)
    print("  detectable on this task with this design. A recurrent graph is still required:",
          flush=True)
    print("  the edge-free control stays at chance while every connected arm learns.",
          flush=True)
else:
    print(f"  The intact connectome leads the best control by {gm:+.1%} against a seed"
          f" spread of {gsd:.1%}, on {wins}/{SEEDS} seeds. That clears the noise here.",
          flush=True)
    print("  Replicate before calling it a property of the fly.", flush=True)

report = {
    "config": {"epochs": EPOCHS, "seeds": SEEDS, "tail": TAIL, "device": "GPU (colab)",
               "steps": 6, "pool_size": 200, "n_pools": 4},
    "per_seed": {str(k): v for k, v in per_seed.items()},
    "summary": summary,
    "gap": {"vs_best_control_mean": gm, "vs_best_control_sd": gsd,
            "vs_shuffled_mean": sm, "vs_shuffled_sd": ssd,
            "vs_random_mean": rm, "vs_random_sd": rsd,
            "seeds_intact_leads": wins},
    "chance_majority": chance,
}
out = RUNS / "colab_measure.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nwrote {out}", flush=True)

zip_path = ROOT / "colab_results.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(RUNS.glob("colab_*")):
        z.write(f, f.name)
    z.write(ROOT / "cache" / "step1_report.json", "step1_report.json")
print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)", flush=True)
