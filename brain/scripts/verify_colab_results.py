"""Recompute every number the Colab run reported, from the checkpoints it saved.

A reported figure is a claim about an artifact. This loads each of the 20 downloaded arm
checkpoints and recomputes its accuracy independently, then compares against the number in
colab_measure.json. Nothing here is read from the run's own logs.

Two things are checked per arm:
  1. the checkpoint's accuracy matches the run's reported final-epoch accuracy, and
  2. it matches the run's own `checkpoint_reproduces_final` field, which was computed on the
     Colab VM's CPU while training ran on its GPU -- so agreement is also a cross-device check.

Works in parallel over arms; each is independent. Reads only local files.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FLY = Path("D:/Projects/flylingo")
EX = FLY / "artifacts" / "colab_results" / "extracted"
REPORT = FLY / "artifacts" / "colab_results" / "colab_measure.json"
GRAPH = FLY / "cache" / "malecns_v1"
sys.path.insert(0, str(FLY / "brain"))


def check_one(args):
    import os

    os.environ.setdefault("FLY_VERIFY_ARM", "1")
    ckpt_name, mode, r_seed, b_seed, reported_final, reported_repro = args
    from brain.encoders import encode_text
    from brain.plastic_brain import PlasticBrain
    from brain.reservoir import FlyReservoir, load_connectome

    cur = json.loads((FLY / "brain" / "brain" / "curriculum" / "es-en.json").read_text("utf-8"))
    items = [(c["prompt"], int(c["correctIndex"]))
             for u in cur["units"] for l in u["lessons"] for c in l["challenges"]]
    X = np.stack([encode_text(p) for p, _ in items])
    y = np.array([t for _, t in items], dtype=int)

    connectome = load_connectome(str(GRAPH))
    r = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=r_seed)
    r.set_mode(mode)
    b = PlasticBrain(r, n_pools=4, pool_size=200, seed=b_seed, steps=6)
    b.set_mode(mode)
    meta = b.load(EX / ckpt_name, require_encoder_match=True)
    acc = b.accuracy(X, y)
    return {
        "arm": ckpt_name.replace("colab_", "").replace(".npz", ""),
        "mode": mode,
        "recomputed": acc,
        "reported_final": reported_final,
        "reported_repro": reported_repro,
        "matches_final": abs(acc - reported_final) < 1e-9,
        "matches_own_repro": abs(acc - reported_repro) < 1e-9,
        "updates": int(meta["updates"]),
    }


def main() -> int:
    d = json.loads(REPORT.read_text("utf-8"))
    jobs = []
    for seed_i, modes in d["per_seed"].items():
        r_seed, b_seed = 7301 + int(seed_i), 99 + int(seed_i)
        for mode, row in modes.items():
            jobs.append((f"colab_s{seed_i}_{mode}.npz", mode, r_seed, b_seed,
                         row["final_accuracy"], row["checkpoint_reproduces_final"]))

    print(f"verifying {len(jobs)} checkpoints against the reported numbers\n")
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check_one, jobs))

    ok = True
    for r in sorted(results, key=lambda x: x["arm"]):
        flag = "OK " if (r["matches_final"] and r["matches_own_repro"]) else "MISMATCH"
        if flag != "OK ":
            ok = False
        print(f"  {flag} {r['arm']:<22} recomputed {r['recomputed']:.4f}   "
              f"reported {r['reported_final']:.4f}   repro {r['reported_repro']:.4f}   "
              f"updates {r['updates']}")

    print(f"\n{sum(1 for r in results if r['matches_final'] and r['matches_own_repro'])}"
          f"/{len(results)} checkpoints reproduce their reported number")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
