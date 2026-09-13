"""Which single call reindexes the control mirror: the upload, or the weight sync?

Instrumentation of the failing sequence localised the change to the first plasticity step
in random_graph mode: between "after enable_plasticity" (no mirror) and "after observe 1"
the control mirror exists with 31,231 fewer stored entries and different index pointers. Two
calls run in that window -- `_build_gpu_mirror("random")` (the upload) and `gpu_sync_data`
(the weight sync, called from `apply_plastic`). Each is wrapped here to print the stored-entry
count immediately before and after, so the culprit is named rather than inferred.
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

cur = json.load(open("brain/curriculum/es-en.json", encoding="utf-8"))
items = [(c["prompt"], int(c["correctIndex"]))
         for u in cur["units"] for l in u["lessons"] for c in l["challenges"]]
X = np.stack([encode_text(p) for p, _ in items])
y = np.array([t for _, t in items], dtype=int)
connectome = load_connectome()

r = FlyReservoir(connectome, embedding_dim=256, dims=128, seed=7301)
r.set_mode("random_graph")
r.enable_gpu()

orig_build = r._build_gpu_mirror
orig_sync = r.gpu_sync_data
orig_apply = r.apply_plastic


def counts(r):
    out = {}
    for key in ("graph", "random"):
        m = r._gmirror.get(key)
        out[key] = None if m is None else int(m.nnz)
    return out


def build_spy(key):
    before = counts(r)
    result = orig_build(key)
    after = counts(r)
    print(f"  _build_gpu_mirror({key!r}): nnz {before[key]} -> {after[key]}  "
          f"({'CHANGED' if before[key] != after[key] else 'same'})")
    return result


def sync_spy():
    before = counts(r)
    orig_sync()
    after = counts(r)
    for key in ("graph", "random"):
        tag = "CHANGED" if before[key] != after[key] else "same"
        print(f"  gpu_sync_data: {key} nnz {before[key]} -> {after[key]}  ({tag})")


def apply_spy():
    before = counts(r)
    orig_apply()
    after = counts(r)
    for key in ("graph", "random"):
        tag = "CHANGED" if before[key] != after[key] else "same"
        print(f"  apply_plastic: {key} nnz {before[key]} -> {after[key]}  ({tag})")


r._build_gpu_mirror = build_spy
r.gpu_sync_data = sync_spy
r.apply_plastic = apply_spy

b = PlasticBrain(r, n_pools=4, pool_size=200, seed=99, steps=6)
b.set_mode("random_graph")
print("observe 1:")
b.observe(X[0], y[0])

print("\nfinal:")
for key in ("graph", "random"):
    m = r._gmirror.get(key)
    src = r.graph if key == "graph" else r._random_graph
    print(f"  {key}: cpu {src.nnz}  mirror {m.nnz if m else None}"
          f"  indptr {'same' if m is not None and np.array_equal(m.indptr.get(), src.indptr) else 'DIFFERS'}")
