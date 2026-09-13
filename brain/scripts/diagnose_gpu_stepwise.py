"""Localise the GPU/CPU divergence on TRAINED weights, step by step.

Established by scripts/diagnose_trained_sensitivity.py (CPU only):
  * CPU -> CPU reload is exact.
  * Perturbing every trained weight by 1e-4 relative moves accuracy by 0.0 points.
  * Decisions are far from ties (relative top-2 margin 1.19; 1% of prompts near a tie).
So the trained system is NOT ill-conditioned, and float32 sensitivity cannot explain a
0.465 disagreement on a state of scale 0.845. That points at the accelerated path
computing something different. This script finds WHICH quantity differs and WHERE.

Method: run the SAME trained brain on both devices and compare, per recurrence step:
  * the drive into the matvec,
  * the raw matvec product (before tanh),
  * the state after tanh,
  * the pooled features.
The first step at which a quantity differs by much more than float32 noise (say 1e-5)
is the defective one. Each mode (intact, shuffled, random_graph) is reported separately,
because a defect in one mode's mirror is invisible in the others.

No verdict is pre-written here: it prints the numbers per step and per mode, and states
which quantity first exceeded the noise floor. If nothing exceeds it, it says so.

Reads the trained checkpoints already on disk; trains nothing; runs a handful of settles.
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

NOISE = 1e-5  # float32 accumulated over a 25.6M-entry matvec; above this is not rounding

RUNS = Path("runs/plastic_brain")
CHECKPOINTS = {
    "intact": RUNS / "arm_s0_intact.npz",
    "shuffled": RUNS / "arm_s0_shuffled.npz",
    "random_graph": RUNS / "arm_s0_random_graph.npz",
}
STEPS = 6
POOL_SIZE, N_POOLS, SEED = 200, 4, 99


def build(mode: str, gpu: bool):
    r = FlyReservoir(connectome, seed=7301)
    r.set_mode(mode)
    if gpu:
        info = r.enable_gpu()
        if not info.get("enabled"):
            raise SystemExit(f"GPU unavailable: {info}")
    b = PlasticBrain(r, n_pools=N_POOLS, pool_size=POOL_SIZE, seed=SEED, steps=STEPS)
    b.set_mode(mode)
    return r, b


def cpu_trace(r, emb):
    """CPU recurrence, recording the drive, the raw product and the state per step."""
    r.reset()
    code = r.B_projection.T @ np.asarray(emb, np.float32)
    code /= np.sqrt(np.mean(code * code) + 1e-6)
    drive = np.take(code, r.input_bins) * r.input_sign
    drive = drive * 0.4 + 0.6 * r.state
    trace = []
    for _ in range(STEPS):
        if r.mode == "shuffled":
            prod = (r.graph @ drive[r.permutation])[r.inverse]
        elif r.mode == "random_graph":
            prod = r.random_graph() @ drive
        else:
            prod = r.graph @ drive
        state = np.tanh(prod).astype(np.float32)
        trace.append((drive.copy(), prod.copy(), state.copy()))
        drive = 0.4 * (np.take(code, r.input_bins) * r.input_sign) + 0.6 * state
    return trace, state


def gpu_trace(r, emb):
    """The shipped GPU recurrence, recording the same quantities per step."""
    cp = r._cp
    r._prepare_gpu_settle()
    r._build_gpu_mirror("graph")
    code = r.B_projection.T @ np.asarray(emb, np.float32)
    code /= np.sqrt(np.mean(code * code) + 1e-6)
    drive_in = cp.asarray(code[r.input_bins] * r.input_sign)
    trace = []
    gstate = cp.zeros(r.n, cp.float32)
    for _ in range(STEPS):
        gdrive = gstate * 0.6 + 0.4 * drive_in
        if r.mode == "shuffled":
            gprod = r._gmirror["graph"] @ gdrive[r._g_perm]
            gstate = cp.tanh(gprod[r._g_inv])
        elif r.mode == "random_graph":
            r.random_graph()
            r._build_gpu_mirror("random")
            gprod = r._gmirror["random"] @ gdrive
            gstate = cp.tanh(gprod)
        else:
            gprod = r._gmirror["graph"] @ gdrive
            gstate = cp.tanh(gprod)
        trace.append((gdrive.get().copy(), gprod.get().copy(), gstate.get().copy()))
    return trace, gstate.get().copy()


connectome = load_connectome()
cur = json.load(open("brain/curriculum/es-en.json", encoding="utf-8"))
prompt = cur["units"][0]["lessons"][0]["challenges"][0]["prompt"]
emb = encode_text(prompt)
print(f"prompt: {prompt!r}   noise floor: {NOISE:.0e}   steps: {STEPS}")

summary = {}
for mode, ckpt in CHECKPOINTS.items():
    if not ckpt.exists():
        print(f"\n=== {mode}: no checkpoint at {ckpt}, skipped")
        continue
    print(f"\n=== mode = {mode}   ({ckpt.name})")
    r_c, b_c = build(mode, gpu=False)
    r_g, b_g = build(mode, gpu=True)
    b_c.load(ckpt)
    b_g.load(ckpt)

    # are the matrices themselves the same on both sides?
    if mode == "random_graph":
        mc, mg = r_c.random_graph(), r_g.random_graph()
        d = np.max(np.abs(mc.data - mg.data)) if mc.nnz == mg.nnz else np.inf
        print(f"  random matrix data max|diff| : {d:.3e}   nnz {mc.nnz} vs {mg.nnz}")
    else:
        d = np.max(np.abs(r_c.graph.data - r_g.graph.data))
        print(f"  graph data max|diff|         : {d:.3e}")
    print(f"  plastic scales max|diff|     : "
          f"{np.max(np.abs(r_c.plastic_scale - r_g.plastic_scale)):.3e}")

    tc, _ = cpu_trace(r_c, emb)
    tg, _ = gpu_trace(r_g, emb)

    first = None
    for k, ((dc, pc, sc), (dg, pg, sg)) in enumerate(zip(tc, tg), start=1):
        dd = float(np.max(np.abs(dc - dg)))
        dp = float(np.max(np.abs(pc - pg)))
        ds = float(np.max(np.abs(sc - sg)))
        scale = float(np.max(np.abs(sc))) or 1.0
        print(f"  step {k}: drive {dd:.3e}  product {dp:.3e}  "
              f"state {ds:.3e} (state max {scale:.3f}, rel {ds/scale:.2e})")
        if first is None:
            for name, val in (("drive", dd), ("product", dp), ("state", ds)):
                if val > NOISE:
                    first = (k, name, val)
                    break

    # do the two devices disagree about the answer?
    a_c, a_g = b_c.choose(emb), b_g.choose(emb)
    print(f"  answer: CPU {a_c}  GPU {a_g}  {'SAME' if a_c == a_g else 'DIFFERENT'}")
    summary[mode] = {"first_divergence": first}

print("\n--- summary (first quantity to exceed the noise floor) ---")
for mode, s in summary.items():
    f = s["first_divergence"]
    print(f"  {mode:14s}: " + (f"step {f[0]}, {f[1]}, max|diff| {f[2]:.3e}" if f
                               else "no divergence above the noise floor"))
