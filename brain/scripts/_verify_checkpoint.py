"""Throwaway: verify the written checkpoint loads the way api.py loads it, and
report the measured per-arm numbers (no tuning, no re-running training)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/Projects/flylingo")
sys.path.insert(0, str(ROOT / "brain"))

from brain.learning.adapter import PolicyAdapter          # noqa: E402
from brain.learning import train as T                      # noqa: E402

CKPT = ROOT / "brain" / "runs" / "curriculum" / "adapter.npz"
res = json.loads((ROOT / "brain" / "runs" / "curriculum" / "results.json").read_text())

print("=== paths / constants ===")
print("DEFAULT_CHECKPOINT  ", T.DEFAULT_CHECKPOINT)
print("DEFAULT_RESULTS     ", T.DEFAULT_RESULTS)
print("CURRICULUM_JSON     ", T.CURRICULUM_JSON, T.CURRICULUM_JSON.exists())
print("ADAPTER_SEED        ", T.ADAPTER_SEED, " RESERVOIR_SEED", T.RESERVOIR_SEED)
print("DIMS/HIDDEN/N_ACTIONS", T.DIMS, T.HIDDEN, T.N_ACTIONS)

print()
print("=== checkpoint file ===")
st = CKPT.stat()
print("exists", CKPT.exists(), "bytes", st.st_size)
print("path equals api.py's:", str(CKPT) == str(ROOT / "brain" / "runs" / "curriculum" / "adapter.npz"))

# Load exactly the way brain/api.py does.
a = PolicyAdapter(in_dim=128, n_actions=4, hidden=256, seed=0)
before = a.steps
a.load(CKPT)
print("loaded into a fresh api-style adapter: steps", before, "->", a.steps,
      "params", a.parameters(), "gated", a.gated_parameters())
finite = all(np.all(np.isfinite(getattr(a, n))) for n in ("W1", "b1", "W2", "b2"))
print("all weights finite:", finite)
probe = a.probs(np.zeros(128, np.float64))
print("probs on a zero feature sum to", float(probe.sum()))

print()
print("=== measured per-arm result (real reservoir, real curriculum) ===")
print("reservoir:", res["reservoir"]["source"], "standin:", res["reservoir"]["is_standin"])
print("task:", res["task"]["source"], "|", {k: v for k, v in res["task"].items()
      if isinstance(v, (int, float, str))}, "| train", res["task"]["n_train"],
      "eval", res["task"]["n_eval"])
print("epochs:", res.get("epochs"), "| hparams:",
      {k: v for k, v in res.get("hparams", {}).items() if not isinstance(v, (dict, list))})
for name, arm in res["arms"].items():
    if not arm["ran"]:
        print(f"  {name:<14} DID NOT RUN: {arm['skipped_reason']}")
        continue
    curve = arm["curve"]
    tr = [r["train_accuracy"] for r in curve]
    ev = [r["eval_accuracy"] for r in curve]
    print(f"  {name:<14} final {arm['final']['eval_accuracy']:.3f} | train acc first/last "
          f"{tr[0]:.3f}/{tr[-1]:.3f} | eval last5 {[round(x,3) for x in ev[-5:]]}"
          f" | mode {arm['reservoir_mode_used']} | same_init {arm.get('same_init_as_intact')}"
          f" | gated {arm['gated_params']}")
    print("      final:", {k: v for k, v in arm["final"].items() if not isinstance(v, (dict, list))})
print("comparison:", json.dumps(res["comparison"], indent=2))


def binom_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), exact."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


print()
print("=== is any arm's final accuracy distinguishable from chance? ===")
n = res["task"]["n_eval"]
for name, arm in res["arms"].items():
    if arm["ran"]:
        acc = arm["final"]["eval_accuracy"]
        k = round(acc * n)
        print(f"  {name:<14} {k}/{n} = {acc:.3f}  P(X>={k}|p=0.25) = {binom_tail(k, n, 0.25):.3f}")
