"""Does the GPU matvec path compute the same thing as the CPU path, and how much faster?

Two claims to pin, because a speedup that changes the answer is worthless:

  1. EQUIVALENCE. A settled state on the GPU must match the CPU's to float32 tolerance. cuSPARSE
     accumulates in a different order than scipy, so the two cannot be bitwise identical, and the
     tolerance has to be stated rather than waved at. This also checks the controls, since the
     shuffled path permutes and unpermutes and the random path uses a second matrix.

  2. SPEED. End-to-end settle timing on both, and the implied epoch cost.

Skips cleanly when no GPU is available, so the suite stays green on a machine without one.
"""
import time

import numpy as np
import pytest

from brain.reservoir import FlyReservoir, load_connectome

cupy = pytest.importorskip("cupy", reason="no GPU on this machine")

MODES = ("intact", "shuffled", "random_graph", "no_edges")
TOL = 2e-3  # float32 accumulation differs by summation order; this is the stated bound


@pytest.fixture(scope="module")
def connectome():
    return load_connectome()


def _settled(reservoir, emb, steps=6):
    reservoir.reset()
    state, feat = reservoir.settle(emb, steps=steps)
    return state.copy(), feat.copy()


def test_gpu_matvec_matches_cpu(connectome):
    """Every control must agree between the two paths, not just the default one."""
    emb = np.random.default_rng(11).standard_normal(256).astype(np.float32)

    cpu = FlyReservoir(connectome, seed=7301)
    gpu = FlyReservoir(connectome, seed=7301)
    info = gpu.enable_gpu()
    assert info["enabled"], info

    for mode in MODES:
        cpu.set_mode(mode)
        gpu.set_mode(mode)
        cs, cf = _settled(cpu, emb)
        gs, gf = _settled(gpu, emb)
        if mode == "no_edges":
            # exactly zero on both, by construction
            assert np.all(cs == 0) and np.all(gs == 0)
            continue
        assert np.allclose(cs, gs, atol=TOL), f"{mode}: settled state diverged"
        assert np.allclose(cf, gf, atol=TOL), f"{mode}: features diverged"


def test_gpu_plasticity_matches_cpu(connectome):
    """The plastic weights must reach the GPU mirror, or training would use stale weights."""
    emb = np.random.default_rng(12).standard_normal(256).astype(np.float32)

    cpu = FlyReservoir(connectome, seed=7301)
    gpu = FlyReservoir(connectome, seed=7301)
    gpu.enable_gpu()

    rng = np.random.default_rng(99)
    pool_of_row = np.full(cpu.n, -1, dtype=np.int64)
    sel = rng.permutation(cpu.n)[:800]
    for k in range(4):
        pool_of_row[sel[k * 200 : (k + 1) * 200]] = k

    cpu.enable_plasticity(pool_of_row, max_edges=20_000)
    gpu.enable_plasticity(pool_of_row, max_edges=20_000)

    # change the scales on both, identically
    scales = 1.0 + 0.5 * np.random.default_rng(7).random(cpu.plastic_scale.size).astype(np.float32)
    cpu.plastic_scale[:] = scales
    gpu.plastic_scale[:] = scales
    cpu.apply_plastic()
    gpu.apply_plastic()

    cs, _ = _settled(cpu, emb)
    gs, _ = _settled(gpu, emb)
    assert np.allclose(cs, gs, atol=TOL), "plasticity did not reach the GPU mirror"


def test_gpu_is_faster(connectome):
    """Measure both, and require a real speedup rather than reporting a hoped-for one."""
    emb = np.random.default_rng(13).standard_normal(256).astype(np.float32)

    cpu = FlyReservoir(connectome, seed=7301)
    gpu = FlyReservoir(connectome, seed=7301)
    gpu.enable_gpu()

    for _ in range(3):  # warm up both kernels and the cuSPARSE plan
        cpu.reset(); cpu.settle(emb, steps=6)
        gpu.reset(); gpu.settle(emb, steps=6)
    cupy.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(15):
        cpu.reset(); cpu.settle(emb, steps=6)
    cpu_ms = (time.perf_counter() - t0) / 15 * 1000

    cupy.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(15):
        gpu.reset(); gpu.settle(emb, steps=6)
    cupy.cuda.Stream.null.synchronize()
    gpu_ms = (time.perf_counter() - t0) / 15 * 1000

    print(f"\n  CPU 6-step settle: {cpu_ms:7.2f} ms")
    print(f"  GPU 6-step settle: {gpu_ms:7.2f} ms   speedup {cpu_ms/gpu_ms:.1f}x")
    print(f"  implied epoch (194 settles): CPU {cpu_ms*194/1000:5.1f}s  GPU {gpu_ms*194/1000:4.1f}s")
    assert gpu_ms < cpu_ms / 3.0, f"GPU was only {cpu_ms/gpu_ms:.1f}x faster"
