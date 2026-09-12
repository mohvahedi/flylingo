"""Option-scoring readout on the real curriculum and the real MaleCNS connectome.

What this measures
------------------
1. Learn-to-criterion (reframe): all 97 challenges presented every epoch, so the
   task is solvable. Primary measure = per-epoch train accuracy per arm.
2. Vocabulary-disjoint generalisation: train on the 73-item split, report the 24
   held-out items. Expected at chance, see NOTES_task_learnability.md (ceiling
   0.2784 for ANY readout on this encoding).
3. The comparison that matters: intact connectome vs parameter-matched control
   (same architecture, same initialisation, gated rule off) vs shuffled wiring vs
   degree-matched random graph vs a disconnected graph.
4. Supervised upper bounds on the SAME intact features and the SAME split, so a
   null can be attributed: a linear probe and a supervised fit of the identical
   architecture. A probe that fits while the reward-driven arms do not puts the
   bottleneck in credit assignment; a probe that also fails puts it in the
   features.

Feature caching: the features depend on the graph and the encoding, not on the
epoch, so every (mode, steps_per_option) cache is built ONCE. A sweep over all
arms then costs no reservoir time at all.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_option_pairs, encoder_fingerprint  # noqa: E402
from brain.learning.option_scorer import (  # noqa: E402
    LinearOptionScorer,
    OptionScorer,
    supervised_fit,
)
from brain.learning.train import load_task  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

ROOT = Path(r"D:\Projects\flylingo")
OUT = ROOT / "brain" / "runs" / "optionscorer"
NOTES = ROOT / "brain" / "NOTES_optionscorer.md"

REWARD_CORRECT = 1.0
REWARD_WRONG = -0.25
CHANCE = 0.25
MODES = ("intact", "shuffled", "random_graph", "no_edges")


# --------------------------------------------------------------------- helpers


def sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def binom_tail(k: int, n: int, p: float = CHANCE) -> float:
    """Exact one-sided binomial tail P(X >= k) at n trials, probability p."""
    if n <= 0:
        return float("nan")
    from scipy.stats import binom

    return float(binom.sf(k - 1, n, p))


def accuracy(model, features, labels, indices=None) -> float:
    idx = range(len(features)) if indices is None else indices
    hits = 0
    n = 0
    for i in idx:
        if int(np.argmax(model.scores(features[i]))) == int(labels[i]):
            hits += 1
        n += 1
    return float(hits / n) if n else float("nan")


# ------------------------------------------------------------------- features


def option_features(challenges, reservoir, mode, steps, embedding_dim=256,
                    raw=False) -> np.ndarray:
    """(n_challenges, n_options, dims) features, one row per candidate.

    Each candidate starts from a zeroed state, so a candidate's feature depends
    only on that candidate: the feature map itself is permutation equivariant
    over the options, which is what the readout relies on. ``steps`` repeated
    steps of the same embedding let the recurrence engage the graph instead of
    applying it once.
    """
    reservoir.set_mode(mode)
    out = np.zeros((len(challenges), 4, reservoir.dims if not raw else embedding_dim),
                   np.float32)
    for i, ch in enumerate(challenges):
        rows = encode_option_pairs(ch, dim=embedding_dim)
        for j in range(rows.shape[0]):
            if raw:
                out[i, j] = rows[j]
                continue
            reservoir.reset()
            f = None
            for _ in range(steps):
                f = reservoir.step(rows[j])
            out[i, j] = f
    return out


# -------------------------------------------------------------------- training


def train_arm(name, model, features_by_challenge, labels, subset, epochs, seed,
              condition, eval_idx, all_idx, rng_seed=1000, order_seed=4242,
              verbose=True) -> dict:
    """Reward-driven training on a subset, with per-epoch curves.

    ``subset`` is the list of challenge indices trained on. Every arm sees the
    identical challenge order per epoch, so the comparison is by construction.
    """
    rng = np.random.default_rng(rng_seed + seed)
    curve = []
    t0 = time.perf_counter()
    for epoch in range(int(epochs)):
        order = np.random.default_rng(order_seed + epoch).permutation(len(subset))
        hits = []
        rewards = []
        losses = []
        entropies = []
        gated = []
        row = {"epoch": epoch}
        for j in order:
            i = int(subset[j])
            F = features_by_challenge[i]
            idx, logprob = model.sample(F, rng)
            reward = REWARD_CORRECT if idx == int(labels[i]) else REWARD_WRONG
            info = model.observe(reward, logprob, F)
            hits.append(1.0 if reward > 0 else 0.0)
            rewards.append(reward)
            losses.append(float(info.get("loss", 0.0)))
            entropies.append(float(info.get("entropy", 0.0)))
            gated.append(float(info.get("gated_weight_delta", 0.0)))
        row.update({
            "train_accuracy": float(np.mean(hits)) if hits else float("nan"),
            "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "entropy": float(np.mean(entropies)) if entropies else float("nan"),
            "gated_weight_delta": float(np.mean(gated)) if gated else float("nan"),
            "accuracy_all97": accuracy(model, features_by_challenge, labels, all_idx),
            "accuracy_heldout24": accuracy(model, features_by_challenge, labels, eval_idx),
            "gated_params": int(model.gated_parameters()),
            "updated_params": int(model.parameters()),
        })
        # The fit measure and the generalisation measure are the same computation
        # when the held-out ids were also trained on, and are labelled separately.
        row["fit_on_trained_subset"] = row["train_accuracy"]
        curve.append(row)
        if verbose:
            print(f"    {name:<20} ep {epoch:>3} {condition:<8} "
                  f"train {row['train_accuracy']:.3f} all97 {row['accuracy_all97']:.3f} "
                  f"held24 {row['accuracy_heldout24']:.3f} "
                  f"reward {row['mean_reward']:+.3f} H {row['entropy']:.3f}")
    final = curve[-1]
    return {
        "arm": name,
        "condition": condition,
        "epochs": int(epochs),
        "curve": curve,
        "final": {
            "train_accuracy": final["train_accuracy"],
            "accuracy_all97": final["accuracy_all97"],
            "accuracy_heldout24": final["accuracy_heldout24"],
            "mean_reward": final["mean_reward"],
            "entropy": final["entropy"],
            "gated_weight_delta": final["gated_weight_delta"],
        },
        "params": int(model.parameters()),
        "gated_params": int(model.gated_parameters()),
        "wall_s": round(time.perf_counter() - t0, 2),
        "epochs_to_criterion": next(
            (r["epoch"] for r in curve if r["train_accuracy"] >= 0.90), None),
    }


def train_supervised_arm(name, model, features_by_challenge, labels, subset, epochs,
                         condition, eval_idx, all_idx, lr, l2, verbose=True) -> dict:
    """Supervised upper bound: same features, same form, labels instead of reward."""
    t0 = time.perf_counter()
    Fs = [features_by_challenge[i] for i in subset]
    ys = [int(labels[i]) for i in subset]
    _, curve = supervised_fit(model, Fs, ys, epochs=int(epochs), lr=lr, l2=l2,
                              verbose=False)
    hits_subset = sum(int(np.argmax(model.scores(f)) == y) for f, y in zip(Fs, ys))
    final = {
        "train_accuracy": float(hits_subset / len(Fs)),
        "accuracy_all97": accuracy(model, features_by_challenge, labels, all_idx),
        "accuracy_heldout24": accuracy(model, features_by_challenge, labels, eval_idx),
        "mean_reward": None,
        "entropy": None,
        "gated_weight_delta": 0.0,
    }
    if verbose:
        print(f"    {name:<20} {condition:<8} train {final['train_accuracy']:.3f} "
              f"all97 {final['accuracy_all97']:.3f} "
              f"held24 {final['accuracy_heldout24']:.3f}")
    return {
        "arm": name,
        "condition": condition,
        "epochs": int(epochs),
        "curve": [{"epoch": i, "train_accuracy": float(a)} for i, a in enumerate(curve)],
        "final": final,
        "params": int(model.parameters()),
        "gated_params": 0,
        "wall_s": round(time.perf_counter() - t0, 2),
        "epochs_to_criterion": next(
            (i for i, a in enumerate(curve) if a >= 0.90), None),
    }


# ------------------------------------------------------------------------ main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30,
                    help="presentations of the whole curriculum (learn-to-criterion)")
    ap.add_argument("--steps", type=int, default=3,
                    help="reservoir steps per candidate from a zeroed state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-fraction", type=float, default=0.25)
    ap.add_argument("--sup-epochs", type=int, default=400)
    ap.add_argument("--sup-lr", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gate-lr", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gate-fraction", type=float, default=0.05)
    ap.add_argument("--modes", type=str,
                    default="intact,shuffled,random_graph,no_edges")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_task(eval_fraction=args.eval_fraction, seed=args.seed, n_actions=4)
    challenges = tasks.challenges
    labels = np.array([int(c["correctIndex"]) for c in challenges], np.int64)
    all_idx = list(range(len(challenges)))
    eval_idx = [int(np.where(np.array([c["id"] for c in challenges]) == c0["id"])[0][0])
                for c0 in tasks.eval]
    train_idx = [i for i in all_idx if i not in set(eval_idx)]
    print(f"curriculum {tasks.source}: {len(challenges)} challenges, "
          f"train {len(train_idx)}, held out {len(eval_idx)}, chance {CHANCE}")
    print(f"options per challenge: {[len(c['options']) for c in challenges[:5]]} ...")

    pos_counts = np.bincount(labels, minlength=4)
    train_pos = np.bincount(labels[train_idx], minlength=4)
    print(f"correctIndex distribution (all 97): {pos_counts.tolist()}")
    print(f"correctIndex distribution (train {len(train_idx)}): {train_pos.tolist()}")

    conn = load_connectome()
    print(f"connectome: {conn.release} {conn.neurons} neurons {conn.edges} edges")
    reservoir = FlyReservoir(conn, embedding_dim=256, dims=128, seed=7301)

    modes = [m for m in (args.modes or "").split(",") if m]
    feats: dict = {}
    meta_feats: dict = {}
    print("\n=== feature caches (built once per mode, reused for every epoch) ===")
    for mode in modes:
        t0 = time.perf_counter()
        X = option_features(challenges, reservoir, mode, args.steps, 256, raw=False)
        dt = time.perf_counter() - t0
        feats[mode] = X
        meta_feats[mode] = {
            "steps_per_option": int(args.steps),
            "shape": list(X.shape),
            "seconds": round(dt, 2),
            "reservoir_steps": int(X.shape[0] * X.shape[1] * args.steps),
            "sha256": sha(X),
        }
        print(f"  {mode:<14} {X.shape} {dt:5.1f}s steps={X.shape[0]*X.shape[1]*args.steps}")

    # Depth check: the same arm at one step per candidate, i.e. exactly the 388
    # steps per curriculum sweep that the brief budgeted.
    t0 = time.perf_counter()
    feats["intact_1step"] = option_features(challenges, reservoir, "intact", 1, 256,
                                            raw=False)
    meta_feats["intact_1step"] = {
        "steps_per_option": 1, "shape": list(feats["intact_1step"].shape),
        "seconds": round(time.perf_counter() - t0, 2),
        "reservoir_steps": int(feats["intact_1step"].shape[0] * 4),
        "sha256": sha(feats["intact_1step"]),
    }
    print(f"  {'intact_1step':<14} {feats['intact_1step'].shape} "
          f"{meta_feats['intact_1step']['seconds']:5.1f}s "
          f"steps={meta_feats['intact_1step']['reservoir_steps']}")

    # Raw encoding, no reservoir at all: what the graph adds, if anything.
    Xraw = np.stack([encode_option_pairs(ch, dim=256) for ch in challenges]).astype(np.float32)
    feats["raw_encoding"] = Xraw
    meta_feats["raw_encoding"] = {"steps_per_option": 0, "shape": list(Xraw.shape),
                                  "seconds": 0.0, "reservoir_steps": 0,
                                  "sha256": sha(Xraw)}
    print(f"  {'raw_encoding':<14} {Xraw.shape} (no reservoir)")

    # ---------------------------------------------------------------- arms
    arms = {}
    condition_curves = {}
    y = labels
    n_opt = 4
    print("\n=== learn-to-criterion: all 97 every epoch (primary) ===")
    conditions = {
        "all97": all_idx,
        f"split{len(train_idx)}": train_idx,
    }
    arm_specs = []
    for mode in modes:
        arm_specs.append((mode, mode, True))
    arm_specs.append(("param_matched", "intact", False))
    arm_specs.append(("intact_1step", "intact_1step", True))
    arm_specs.append(("raw_encoding_mlp", "raw_encoding", True))

    trained: dict = {}
    for cond, subset in conditions.items():
        print(f"\n-- condition {cond} (train on {len(subset)} challenges) --")
        for name, mode, dopamine in arm_specs:
            model = OptionScorer(in_dim=int(feats[mode].shape[2]), hidden=args.hidden,
                                 seed=args.seed, lr=args.lr, gate_lr=args.gate_lr,
                                 dopamine=dopamine, gate_fraction=args.gate_fraction)
            res = train_arm(name, model, feats[mode], y, subset, args.epochs, args.seed,
                            cond, eval_idx, all_idx)
            res["mode"] = mode
            res["dopamine"] = bool(dopamine)
            res["features_sha256"] = meta_feats[mode]["sha256"]
            arms[f"{name}::{cond}"] = res
            trained[f"{name}::{cond}"] = model
            condition_curves.setdefault(cond, {})[name] = res

    # ------------------------------------------------------- upper bounds
    print("\n=== supervised upper bounds (same features, same split) ===")
    bounds = {}
    for cond, subset in conditions.items():
        for label, mode, kind in (("linear_probe", "intact", "linear"),
                                  ("supervised_mlp", "intact", "mlp"),
                                  ("linear_probe_1step", "intact_1step", "linear"),
                                  ("linear_probe_raw", "raw_encoding", "linear"),
                                  ("supervised_mlp_raw", "raw_encoding", "mlp")):
            if kind == "linear":
                model = LinearOptionScorer(in_dim=int(feats[mode].shape[2]), seed=args.seed)
            else:
                model = OptionScorer(in_dim=int(feats[mode].shape[2]), hidden=args.hidden,
                                     seed=args.seed, dopamine=False)
            res = train_supervised_arm(label, model, feats[mode], y, subset,
                                       args.sup_epochs, cond, eval_idx, all_idx,
                                       args.sup_lr, 1e-3)
            res["mode"] = mode
            res["kind"] = kind
            bounds[f"{label}::{cond}"] = res

    # ---------------------------------------------------------------- report
    comparison = {}
    for cond in conditions:
        names = [f"{n}::{cond}" for n, _m, _d in arm_specs]
        finals = {n.split("::")[0]: arms[n]["final"]["train_accuracy"] for n in names}
        crit = {n.split("::")[0]: arms[n]["epochs_to_criterion"] for n in names}
        held = {n.split("::")[0]: arms[n]["final"]["accuracy_heldout24"] for n in names}
        comparison[cond] = {
            "final_train_accuracy": finals,
            "epochs_to_0.90": crit,
            "heldout24_accuracy": held,
            "intact_minus_param_matched": finals.get("intact", float("nan"))
            - finals.get("param_matched", float("nan")),
            "intact_minus_shuffled": finals.get("intact", float("nan"))
            - finals.get("shuffled", float("nan")),
            "intact_minus_random_graph": finals.get("intact", float("nan"))
            - finals.get("random_graph", float("nan")),
            "intact_minus_no_edges": finals.get("intact", float("nan"))
            - finals.get("no_edges", float("nan")),
        }
    # Proof rather than intent: the two arms that must differ only in the gated
    # rule have to have consumed byte-identical reservoir features.
    comparison["intact_and_param_matched_share_features"] = bool(
        arms["intact::all97"]["features_sha256"]
        == arms["param_matched::all97"]["features_sha256"])

    # The honest attribution: on the learn-to-criterion condition, does the intact
    # graph fit faster or higher than every control?
    all97 = comparison["all97"]
    others = ["param_matched", "shuffled", "random_graph", "no_edges"]
    diffs = {o: all97["intact_minus_" + o] if ("intact_minus_" + o) in all97 else None
             for o in others}
    higher = all(d is not None and d >= 0 for d in diffs.values())
    faster = all(
        all97["epochs_to_0.90"].get("intact") is not None
        and (all97["epochs_to_0.90"].get(o) is None
             or all97["epochs_to_0.90"]["intact"] <= all97["epochs_to_0.90"][o])
        for o in others)
    if higher and faster:
        verdict = ("intact connectome arm fitted the curriculum to at least the accuracy "
                   "and at least as fast as every control on the learn-to-criterion "
                   "condition")
    else:
        verdict = ("intact connectome arm did NOT both match or beat every control and "
                   "reach criterion at least as fast on the learn-to-criterion "
                   "condition; report this as a null on the connectome comparison")
    comparison["verdict"] = verdict

    split = comparison[f"split{len(train_idx)}"]
    tails = {}
    n_held = len(eval_idx)
    for name, res in list(condition_curves[f"split{len(train_idx)}"].items()):
        k = int(round(res["final"]["accuracy_heldout24"] * n_held))
        tails[name] = {"heldout_correct": k, "heldout_n": n_held,
                       "binomial_p_one_sided": binom_tail(k, n_held)}
    for label, res in bounds.items():
        if res["condition"] != f"split{len(train_idx)}":
            continue
        k = int(round(res["final"]["accuracy_heldout24"] * n_held))
        tails[label] = {"heldout_correct": k, "heldout_n": n_held,
                        "binomial_p_one_sided": binom_tail(k, n_held)}
    comparison["heldout_binomial_tails"] = tails

    majority_index = int(np.argmax(train_pos))
    baselines = {
        "chance": CHANCE,
        "majority_correct_index_in_train": majority_index,
        "majority_index_accuracy_all97": float((labels == majority_index).mean()),
        "majority_index_accuracy_heldout24": float(
            (labels[eval_idx] == majority_index).mean()),
        "correct_index_counts_heldout24": np.bincount(labels[eval_idx],
                                                      minlength=4).tolist(),
    }

    printed = []
    for cond in conditions:
        printed.append(f"  {cond}: " + ", ".join(
            f"{k} {v:.3f}" for k, v in comparison[cond]["final_train_accuracy"].items()))

    n_params = int(OptionScorer(in_dim=128, hidden=args.hidden).parameters())
    interpretation = (
        "Learn-to-criterion: the readout is asked to fit the 97-phrase mapping, which the "
        "encoding supports because the exact pairings are seen in training. The "
        "vocabulary-disjoint number stays at chance and that is a property of the encoding, "
        "not of the connectome (ceiling 0.2784, exact binomial p 0.294, "
        "NOTES_task_learnability.md). The comparison that carries information is the "
        "per-epoch curve of the intact graph against its controls."
    )

    results = {
        "kind": "optionscorer",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "curriculum": tasks.path,
        "curriculum_source": tasks.source,
        "n_challenges": len(challenges),
        "n_options": n_opt,
        "chance": CHANCE,
        "encoder_fingerprint": encoder_fingerprint(),
        "architecture": {
            "class": "OptionScorer",
            "form": "score_i = w . tanh(W1 f_i + b1) + b2, softmax over i",
            "permutation_equivariant": True,
            "hidden": int(args.hidden),
            "params": n_params,
            "gated_fraction": float(args.gate_fraction),
            "gated_params": int(OptionScorer(in_dim=128, hidden=args.hidden,
                                             gate_fraction=args.gate_fraction)
                                .gated_parameters()),
            "learning_rule": "REINFORCE with running baseline, entropy bonus, plus a "
                             "masked three-factor Hebbian gate on an enumerated subset",
        },
        "reservoir": {
            "source": "brain.reservoir.FlyReservoir",
            "class": type(reservoir).__name__,
            "dataset": conn.release,
            "neurons": int(reservoir.neurons),
            "edges": int(reservoir.edges),
            "full_graph_edges": int(reservoir.connectome_edges),
            "is_standin": False,
            "embedding_dim": 256,
            "dims": 128,
            "seed": 7301,
            "history": "each candidate starts from a zeroed state; steps_per_option "
                       "repeated steps of that candidate's own embedding",
        },
        "feature_caches": meta_feats,
        "split": {
            "eval_fraction": float(args.eval_fraction),
            "seed": int(args.seed),
            "n_train": len(train_idx),
            "n_heldout": len(eval_idx),
            "heldout_ids": [c["id"] for c in tasks.eval],
            "note": "load_task(seed=0) reproduces the split of the earlier "
                    "runs/curriculum run, so the 24 held-out items are the same items",
        },
        "config": {
            "epochs": int(args.epochs),
            "steps_per_option": int(args.steps),
            "lr": float(args.lr),
            "gate_lr": float(args.gate_lr),
            "entropy_coef": 0.01,
            "reward_correct": REWARD_CORRECT,
            "reward_wrong": REWARD_WRONG,
            "supervised_epochs": int(args.sup_epochs),
            "supervised_lr": float(args.sup_lr),
        },
        "arms": arms,
        "upper_bounds": bounds,
        "comparison": comparison,
        "baselines": baselines,
        "interpretation": interpretation,
        "curve_summary": printed,
    }

    # ------------------------------------------------------------ checkpoints
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, mode, dopamine in arm_specs:
        model = trained[f"{name}::all97"]
        path = out_dir / ("adapter.npz" if name == "intact"
                          else f"adapter_{name}.npz")
        model.save(path)
        saved[name] = str(path)
        arms[f"{name}::all97"]["saved"] = str(path)

    # Read the primary checkpoint back and assert the gate the service enforces.
    primary = out_dir / "adapter.npz"
    with np.load(primary, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
    verified = meta.get("encoder_fingerprint") == encoder_fingerprint()
    print("\n=== checkpoint verification ===")
    print(f"  path: {primary}")
    print(f"  meta.encoder_fingerprint = {meta.get('encoder_fingerprint')}")
    print(f"  encoder_fingerprint()    = {encoder_fingerprint()}")
    print(f"  match: {verified}")
    assert verified, "checkpoint meta does not carry the current encoder fingerprint"
    loaded = OptionScorer(in_dim=128, hidden=args.hidden, seed=123)
    loaded.load(primary)
    F0 = feats["intact"][0]
    round_trip = bool(np.array_equal(loaded.scores(F0),
                                     trained["intact::all97"].scores(F0)))
    print(f"  load-back scores identical to the trained model: {round_trip}")
    assert round_trip, "checkpoint on disk does not reproduce the trained model"

    results["checkpoint_verification"] = {
        "primary": str(primary),
        "encoder_fingerprint_in_meta": meta.get("encoder_fingerprint"),
        "encoder_fingerprint_current": encoder_fingerprint(),
        "match": bool(verified),
        "meta_kind": meta.get("kind"),
        "meta_gated_params": meta.get("gated_params"),
        "saved": saved,
    }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"\n  results: {out_dir / 'results.json'}")
    print(f"  saved: {list(saved)}")

    with NOTES.open("a", encoding="utf-8") as fh:
        fh.write("\n## Run: option-scoring readout (real connectome, real curriculum)\n\n")
        fh.write(f"- encoder_fingerprint {encoder_fingerprint()}\n")
        fh.write(f"- {len(challenges)} challenges, split {len(train_idx)}/"
                 f"{len(eval_idx)}, chance {CHANCE}\n")
        fh.write(f"- reservoir {conn.neurons} neurons, {conn.edges} edges, "
                 f"{args.steps} steps per candidate\n")
        for cond in conditions:
            fh.write(f"- {cond} final train accuracy: "
                     + json.dumps(comparison[cond]["final_train_accuracy"]) + "\n")
        fh.write(f"- verdict: {verdict}\n")
        fh.write(f"- checkpoint: {primary} (fingerprint match {verified})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
