"""Why does the random_graph arm fail to reproduce its own accuracy from its checkpoint?

Only the random_graph arm failed: saved weights reproduced 45.4% where the run reported 87.6%.
intact and shuffled both reproduced exactly. The verification path I added to the experiment script
runs the reload on the CPU while the training run used the GPU, so the first suspect is a
device difference that the random wiring amplifies.

This isolates the cause into four measurements:
  1. does the reloaded CPU brain match the trained GPU brain?      (the reported failure)
  2. does the reloaded GPU brain match the trained GPU brain?      (device, or weights?)
  3. are the two random matrices actually identical?               (weights, or wiring?)
  4. what does one settle differ by on each device, fresh weights?  (is the mode inherently split?)

Every comparison prints the number, so the conclusion does not rest on an assertion.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoders import encode_text  # noqa: E402
from brain.plastic_brain import PlasticBrain  # noqa: E402
from brain.reservoir import FlyReservoir, load_connectome  # noqa: E402

import json  # noqa: E402

MODE = "random_graph"
EPOCHS = 8

cur = json.load(open("brain/curriculum/es-en.json", encoding="utf-8"))
items = [
    (c["prompt"], int(c["correctIndex"]))
    for u in cur["units"]
    for l in u["lessons"]
    for c in l["challenges"]
]
X = np.stack([encode_text(p) for p, _ in items])
y = np.array([t for _, t in items], dtype=int)

connectome = load_connectome()

# ---------------------------------------------------------------- train on the GPU
r_a = FlyReservoir(connectome, seed=7301)
r_a.set_mode(MODE)
r_a.enable_gpu()
brain_a = PlasticBrain(r_a, n_pools=4, pool_size=200, seed=99, steps=6)
brain_a.set_mode(MODE)

for ep in range(1, EPOCHS + 1):
    idx = np.random.default_rng(ep).permutation(len(X))
    for i in idx:
        brain_a.observe(X[i], y[i])
acc_trained = brain_a.accuracy(X, y)
print(f"trained on GPU            : {acc_trained:.1%}")

ckpt = Path("runs/plastic_brain/_diag_random.npz")
meta = brain_a.save(ckpt)
print(f"saved                     : {ckpt.name}  ({meta['plastic_edges']} edges)")

# ---------------------------------------------------------------- reload, CPU and GPU
def reloaded(use_gpu: bool):
    r = FlyReservoir(connectome, seed=7301)
    r.set_mode(MODE)
    if use_gpu:
        r.enable_gpu()
    b = PlasticBrain(r, n_pools=4, pool_size=200, seed=99, steps=6)
    b.set_mode(MODE)
    b.load(ckpt)
    return r, b


r_cpu, brain_cpu = reloaded(use_gpu=False)
acc_cpu = brain_cpu.accuracy(X, y)
print(f"reloaded, CPU path        : {acc_cpu:.1%}   {'MATCH' if abs(acc_cpu-acc_trained)<1e-9 else 'MISMATCH'}")

r_gpu, brain_gpu = reloaded(use_gpu=True)
acc_gpu = brain_gpu.accuracy(X, y)
print(f"reloaded, GPU path        : {acc_gpu:.1%}   {'MATCH' if abs(acc_gpu-acc_trained)<1e-9 else 'MISMATCH'}")

# ---------------------------------------------------------------- are the weights identical?
same_score = np.allclose(brain_a.score_w, brain_cpu.score_w)
same_scale = np.allclose(r_a.plastic_scale, r_cpu.plastic_scale)
print(f"\nweights identical         : pool {same_score}   scales {same_scale}")

# ---------------------------------------------------------------- are the random matrices identical?
ra = r_a._random_graph
rc = r_cpu._random_graph
if ra is None or rc is None:
    print("random matrix            : NOT BUILT on one side")
else:
    print(f"random matrix indices same: {np.array_equal(ra.indices, rc.indices)}")
    print(f"random matrix data same   : {np.allclose(ra.data, rc.data)}")
    print(f"  max |diff|              : {np.max(np.abs(ra.data - rc.data)):.3e}")
    print(f"  nnz                     : {ra.nnz} vs {rc.nnz}")

# ---------------------------------------------------------------- one settle, both devices
r_fresh_g = FlyReservoir(connectome, seed=7301)
r_fresh_g.set_mode(MODE)
r_fresh_g.enable_gpu()
r_fresh_c = FlyReservoir(connectome, seed=7301)
r_fresh_c.set_mode(MODE)

emb = encode_text("hola")
sg = r_fresh_g.settle(emb, steps=6)[0].copy()
sc = r_fresh_c.settle(emb, steps=6)[0].copy()
print(f"\none settle, fresh weights : max |GPU - CPU| = {np.max(np.abs(sg - sc)):.3e}")
print(f"  (matrix scale ~ {np.abs(sc).max():.3f}, so this is "
      f"{np.max(np.abs(sg - sc)) / max(np.abs(sc).max(), 1e-9):.1%} relative)")

# and with the trained weights, where the run actually diverged
r_gpu.plastic_scale[:] = r_a.plastic_scale
r_gpu.apply_plastic()
r_cpu.plastic_scale[:] = r_a.plastic_scale
r_cpu.apply_plastic()
sg2 = r_gpu.settle(emb, steps=6)[0].copy()
sc2 = r_cpu.settle(emb, steps=6)[0].copy()
print(f"one settle, trained wts   : max |GPU - CPU| = {np.max(np.abs(sg2 - sc2)):.3e}")
print(f"  (matrix scale ~ {np.abs(sc2).max():.3f})")
