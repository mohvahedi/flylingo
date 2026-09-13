"""Instrument the failing configuration stage by stage, in random_graph mode.

Three explanations for the divergent mirror have now been tested and refuted: the cupyx
constructor preserves the stored-entry count, `.tocsr()` returns the same object, and
writing `mirror.data[idx] = values` does not canonicalise. Yet in the real sequence -- build
with the GPU enabled, construct the plastic brain, then take six plasticity steps -- the
random control's mirror ends up with 31,231 FEWER stored entries than the CPU matrix it came
from. So the loss happens somewhere in that sequence and this script finds where, by printing
the count at each stage for both mirrors, plus whether the index pointers still agree.

No hypothesis is asserted here. The stage where the count changes is the answer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_text  # noqa: E402
from brain.plastic_brain import PlasticBrain  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

MODE = "random_graph"

cur = json.load(open("brain/curriculum/es-en.json", encoding="utf-8"))
items = [(c["prompt"], int(c["correctIndex"]))
         for u in cur["units"] for l in u["lessons"] for c in l["challenges"]]
X = np.stack([encode_text(p) for p, _ in items])
y = np.array([t for _, t in items], dtype=int)
connectome = load_connectome()


def report(r, label: str) -> None:
    gm = r._gmirror.get("graph")
    rm = r._gmirror.get("random")
    rg = r._random_graph
    g = r.graph
    print(f"  [{label}]")
    print(f"      cpu graph nnz {g.nnz:>9}   mirror graph "
          f"{gm.nnz if gm is not None else '-':>9}"
          f"   indptr {'same' if gm is not None and np.array_equal(gm.indptr.get(), g.indptr) else 'DIFFERS' if gm is not None else '-'}")
    if rg is not None:
        print(f"      cpu rand  nnz {rg.nnz:>9}   mirror rand  "
              f"{rm.nnz if rm is not None else '-':>9}"
              f"   indptr {'same' if rm is not None and np.array_equal(rm.indptr.get(), rg.indptr) else 'DIFFERS' if rm is not None else '-'}")
        if rm is not None and rm.nnz != rg.nnz:
            print(f"      --> mirror is short by {rg.nnz - rm.nnz}")


print("=" * 72)
print("random_graph mode, the sequence the gate fails in")
print("=" * 72)

r = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=7301)
r.set_mode(MODE)
r.enable_gpu()
report(r, "after enable_gpu")

b = PlasticBrain(r, n_pools=4, pool_size=200, seed=99, steps=6)
b.set_mode(MODE)
report(r, "after enable_plasticity (PlasticBrain ctor)")

for i in range(3):
    b.observe(X[i], y[i])
    report(r, f"after observe {i + 1}")

# the CPU side, for comparison: same construction, no GPU
print("\nCPU-only reference (same sequence):")
r2 = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=7301)
r2.set_mode(MODE)
b2 = PlasticBrain(r2, n_pools=4, pool_size=200, seed=99, steps=6)
b2.set_mode(MODE)
for i in range(3):
    b2.observe(X[i], y[i])
print(f"  cpu random matrix nnz: {r2._random_graph.nnz}")
print(f"  cpu graph  matrix nnz: {r2.graph.nnz}")

# do the two devices agree about the random matrix's contents where layouts match?
if r._gmirror.get("random") is not None and r._gmirror["random"].nnz == r._random_graph.nnz:
    import cupy as cp
    a = r._random_graph.data
    bb = cp.asnumpy(r._gmirror["random"].data)
    print(f"  mirror data vs cpu data: max|diff| {np.max(np.abs(a - bb)):.3e}")
else:
    print("  layouts already differ, contents not comparable")
