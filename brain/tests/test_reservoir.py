"""Tests for the FlyLingo simulation core.

Two layers:

  - synthetic: an exact, hand-computable reservoir where the recurrence is
    checked against a direct numpy transcription of the frozen formula. These
    always run and pin the semantics of every control mode.
  - real: assertions against the built MaleCNS v1.0 graph in
    ``cache/malecns_v1``. Skipped, not faked, if the build is absent.

No test fabricates connectome activity. Where the graph is missing the test is
skipped and says so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.reservoir import (  # noqa: E402
    GRAPH,
    MODES,
    SAMPLES,
    SPIKE_THRESHOLD,
    Connectome,
    FlyReservoir,
    load_connectome,
    sha256,
)

EMBED = 256
DIMS = 128


# ------------------------------------------------------------------ fixtures


def tiny_connectome(n=64, seed=5):
    """A small duck-typed connectome: only .matrix and .ids are used."""
    rng = np.random.default_rng(seed)
    density = 0.15
    mask = rng.random((n, n)) < density
    counts = rng.integers(1, 6, (n, n)).astype(np.float32) * mask
    W = sparse.csr_matrix(counts)
    rows = np.asarray(W.sum(axis=1)).ravel()
    W.data /= np.repeat(np.maximum(rows, 1.0), np.diff(W.indptr)).astype(np.float32)
    ids = np.sort(rng.choice(np.arange(10**9, 10**9 + 100000, dtype=np.int64),
                             n, replace=False))
    return SimpleNamespace(matrix=W, ids=ids, neurons=n, edges=int(W.nnz),
                           directed_edges=int(W.nnz))


@pytest.fixture
def tiny():
    return FlyReservoir(tiny_connectome(), embedding_dim=EMBED, dims=DIMS, seed=7301)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def manual_reference(res, embeddings, W, mode="intact"):
    """Direct transcription of the frozen recurrence, independent of fast paths."""
    x = np.zeros(res.n, np.float32)
    dims = res.dims
    out = []
    for emb in embeddings:
        code = res.B_projection.T @ np.asarray(emb, np.float32)
        code = code / np.sqrt(np.mean(code * code) + 1e-6)
        drive = 0.6 * x + 0.4 * code[res.input_bins] * res.input_sign
        if mode == "no_edges":
            x = np.zeros(res.n, np.float32)
        elif mode == "shuffled":
            x = np.tanh((W @ drive[res.permutation])[res.inverse]).astype(np.float32)
        else:
            x = np.tanh(W @ drive).astype(np.float32)
        f = np.bincount(res.output_bins,
                        weights=x * res.output_sign,
                        minlength=dims).astype(np.float32) / res.output_scale
        f = f / np.sqrt(np.mean(f * f) + 1e-6)
        out.append(f)
    return np.stack(out)


# ------------------------------------------------------------- recurrence


def test_recurrence_matches_direct_formula(tiny, rng):
    embs = rng.standard_normal((6, EMBED)).astype(np.float32)
    got = tiny.sequence(embs)
    want = manual_reference(tiny, embs, tiny.graph)
    assert got.shape == (6, DIMS)
    assert np.allclose(got, want, atol=1e-5)


def test_state_update_is_tanh_of_matvec(tiny, rng):
    """x_t must equal tanh(W @ (0.6 x_(t-1) + 0.4 B e_t)) exactly."""
    emb = rng.standard_normal(EMBED).astype(np.float32)
    tiny.reset()
    tiny.step(emb)
    code = tiny.B_projection.T @ emb
    code = code / np.sqrt(np.mean(code * code) + 1e-6)
    expected = np.tanh(
        tiny.graph @ (0.4 * code[tiny.input_bins] * tiny.input_sign)
    ).astype(np.float32)
    assert np.allclose(tiny.state, expected, atol=1e-5)


def test_second_step_uses_previous_state(tiny, rng):
    a = rng.standard_normal(EMBED).astype(np.float32)
    b = rng.standard_normal(EMBED).astype(np.float32)
    tiny.reset()
    tiny.step(a)
    x1 = tiny.state.copy()
    tiny.step(b)
    code = tiny.B_projection.T @ b
    code = code / np.sqrt(np.mean(code * code) + 1e-6)
    drive = 0.6 * x1 + 0.4 * code[tiny.input_bins] * tiny.input_sign
    assert np.allclose(tiny.state, np.tanh(tiny.graph @ drive), atol=1e-5)


def test_features_are_unit_rms(tiny, rng):
    f = tiny.step(rng.standard_normal(EMBED).astype(np.float32))
    assert f.shape == (DIMS,)
    assert f.dtype == np.float32
    assert np.isclose(np.sqrt(np.mean(f * f)), 1.0, atol=1e-3)


def test_sequence_resets_state(tiny, rng):
    embs = rng.standard_normal((4, EMBED)).astype(np.float32)
    first = tiny.sequence(embs)
    second = tiny.sequence(embs)
    assert np.allclose(first, second), "sequence must be reproducible from a reset"
    assert tiny.updates == 4


def test_shapes_are_checked(tiny):
    with pytest.raises(ValueError):
        tiny.step(np.zeros(EMBED + 1, np.float32))
    with pytest.raises(ValueError):
        tiny.sequence(np.zeros((3, EMBED + 1), np.float32))


def test_determinism_for_a_fixed_seed(rng):
    embs = rng.standard_normal((3, EMBED)).astype(np.float32)
    a = FlyReservoir(tiny_connectome(), embedding_dim=EMBED, dims=DIMS, seed=7301)
    b = FlyReservoir(tiny_connectome(), embedding_dim=EMBED, dims=DIMS, seed=7301)
    assert np.allclose(a.sequence(embs), b.sequence(embs))


def test_different_seed_changes_interface_only(tiny):
    other = FlyReservoir(tiny_connectome(), embedding_dim=EMBED, dims=DIMS, seed=99)
    assert not np.array_equal(tiny.input_bins, other.input_bins)
    assert np.array_equal(np.sort(tiny.permutation), np.arange(tiny.n))


# ------------------------------------------------------------------- modes


def test_default_mode_is_intact_and_set_mode_roundtrip(tiny):
    assert tiny.mode == "intact"
    for mode in MODES:
        tiny.set_mode(mode)
        assert tiny.mode == mode


def test_unknown_mode_raises_value_error(tiny):
    with pytest.raises(ValueError):
        tiny.set_mode("superior")
    with pytest.raises(ValueError):
        tiny.set_mode("")
    assert tiny.mode == "intact"


def test_no_edges_gives_exactly_zero_features(tiny, rng):
    f = tiny.step(rng.standard_normal(EMBED).astype(np.float32), mode="no_edges")
    assert np.array_equal(f, np.zeros(DIMS, np.float32))
    assert not np.any(tiny.state)
    f2 = tiny.step(rng.standard_normal(EMBED).astype(np.float32))
    assert np.array_equal(f2, np.zeros(DIMS, np.float32)), "mode must persist"


def test_no_edges_telemetry_is_honest(tiny, rng):
    tiny.set_mode("no_edges")
    for _ in range(3):
        tiny.step(rng.standard_normal(EMBED).astype(np.float32))
    tel = tiny.telemetry()
    assert tel["spikes"] == []
    assert tel["active_fraction"] == 0.0
    assert tel["state_rms"] == 0.0
    assert all(v == 0.0 for v in tel["sampled_state"])


def test_no_edges_sequence_is_all_zero(tiny, rng):
    embs = rng.standard_normal((5, EMBED)).astype(np.float32)
    feats = tiny.sequence(embs, mode="no_edges")
    assert np.array_equal(feats, np.zeros((5, DIMS), np.float32))


def test_shuffled_preserves_topology_but_moves_interface(tiny, rng):
    emb = rng.standard_normal(EMBED).astype(np.float32)
    tiny.reset()
    intact = tiny.step(emb).copy()
    tiny.reset()
    shuffled = tiny.step(emb, mode="shuffled").copy()
    # Same number of stored edges: the wiring is relabeled, not changed.
    assert tiny.edges == tiny.graph.nnz
    assert not np.allclose(intact, shuffled), "relabeling must change the readout"
    # A permutation is a bijection.
    assert np.array_equal(np.sort(tiny.permutation), np.arange(tiny.n))
    assert np.array_equal(tiny.inverse[tiny.permutation], np.arange(tiny.n))


def test_random_graph_same_nnz(tiny):
    W = tiny.random_graph()
    assert W.shape == tiny.graph.shape
    # The contract: same nnz, degree-matched, rows renormalized to 1.
    assert W.nnz == tiny.graph.nnz
    sums = np.asarray(W.sum(axis=1)).ravel()
    nz = sums[sums > 0]
    assert np.allclose(nz, 1.0, atol=1e-4)
    assert np.array_equal(np.diff(W.indptr), np.diff(tiny.graph.indptr))
    tiny.set_mode("random_graph")
    f = tiny.step(np.ones(EMBED, np.float32) * 0.1)
    assert f.shape == (DIMS,)
    assert np.all(np.isfinite(f))


def test_modes_produce_distinct_activity(tiny, rng):
    embs = rng.standard_normal((3, EMBED)).astype(np.float32)
    outputs = {}
    for mode in MODES:
        outputs[mode] = tiny.sequence(embs, mode=mode)
    assert np.array_equal(outputs["no_edges"], np.zeros((3, DIMS), np.float32))
    assert not np.allclose(outputs["intact"], outputs["shuffled"])
    assert not np.allclose(outputs["intact"], outputs["random_graph"])


# --------------------------------------------------------------- telemetry


def test_telemetry_contract(tiny, rng):
    for _ in range(4):
        tiny.step(rng.standard_normal(EMBED).astype(np.float32))
    tel = tiny.telemetry()
    assert set(tel) == {
        "updates", "state_rms", "active_fraction",
        "sampled_ids", "sampled_state", "spikes",
    }
    assert isinstance(tel["updates"], int) and tel["updates"] == 4
    assert isinstance(tel["state_rms"], float) and tel["state_rms"] >= 0.0
    assert isinstance(tel["active_fraction"], float) and 0.0 <= tel["active_fraction"] <= 1.0
    assert len(tel["sampled_state"]) == SAMPLES == 512
    assert len(tel["sampled_ids"]) == 512
    assert all(isinstance(s, str) for s in tel["sampled_ids"])
    assert all(int(s) > 0 for s in tel["sampled_ids"])
    assert all(isinstance(v, float) for v in tel["sampled_state"])
    assert all(-1.0 <= v <= 1.0 for v in tel["sampled_state"])
    # spikes are indices INTO sampled_state and agree with the threshold
    for i in tel["spikes"]:
        assert isinstance(i, int) and 0 <= i < 512
        assert abs(tel["sampled_state"][i]) >= SPIKE_THRESHOLD
    below = [i for i, v in enumerate(tel["sampled_state"])
             if abs(v) < SPIKE_THRESHOLD]
    assert set(tel["spikes"]).isdisjoint(below)
    assert len(tel["spikes"]) == 512 - len(below)


def test_sampled_ids_come_from_the_connectome(tiny):
    tel = tiny.telemetry()
    real = {str(int(i)) for i in tiny.connectome.ids}
    assert set(tel["sampled_ids"]) <= real


def test_telemetry_updates_counter_resets(tiny, rng):
    tiny.step(rng.standard_normal(EMBED).astype(np.float32))
    assert tiny.telemetry()["updates"] == 1
    tiny.reset()
    assert tiny.telemetry()["updates"] == 0


# ------------------------------------------------------- subgraph fallback


def test_subgraph_is_a_seeded_subset_with_fewer_neurons():
    full = FlyReservoir(tiny_connectome(), embedding_dim=EMBED, dims=DIMS, seed=7301)
    sub = FlyReservoir(tiny_connectome(), embedding_dim=EMBED, dims=DIMS,
                       seed=7301, subgraph_size=32)
    assert sub.n == 32 < full.n
    assert sub.subgraph_size == 32
    assert sub.graph.shape == (32, 32)
    assert sub.connectome_edges == full.connectome_edges
    assert set(sub.sample_ids) <= {str(int(i)) for i in
                                  tiny_connectome().ids[sub.sub_index]}
    f = sub.step(np.ones(EMBED, np.float32))
    assert f.shape == (DIMS,) and np.isfinite(f).all()
    assert len(sub.telemetry()["sampled_state"]) == 512


def test_subgraph_is_reproducible():
    a = FlyReservoir(tiny_connectome(), embedding_dim=EMBED, dims=DIMS,
                     seed=7301, subgraph_size=32)
    b = FlyReservoir(tiny_connectome(), embedding_dim=EMBED, dims=DIMS,
                     seed=7301, subgraph_size=32)
    emb = np.random.default_rng(3).standard_normal(EMBED).astype(np.float32)
    assert np.allclose(a.step(emb), b.step(emb))


def test_subgraph_no_edges_still_exactly_zero():
    sub = FlyReservoir(tiny_connectome(), embedding_dim=EMBED, dims=DIMS,
                       seed=7301, subgraph_size=32)
    f = sub.step(np.ones(EMBED, np.float32), mode="no_edges")
    assert np.array_equal(f, np.zeros(DIMS, np.float32))
    tel = sub.telemetry()
    assert tel["spikes"] == [] and tel["active_fraction"] == 0.0


# ------------------------------------------------------------- real graph

REAL = Path(GRAPH) / "manifest.json"
needs_graph = pytest.mark.skipif(
    not REAL.exists(), reason=f"built graph not present at {GRAPH}"
)


@needs_graph
def test_real_connectome_counts_and_integrity():
    conn = load_connectome()
    assert conn.neurons == 166700
    assert conn.edges == 25582938
    assert conn.matrix.nnz == 25582938
    assert conn.matrix.shape == (166700, 166700)
    assert conn.ids.dtype == np.int64
    assert np.all(np.diff(conn.ids) > 0)


@needs_graph
def test_real_manifest_hash_verification_catches_tampering(tmp_path):
    """Corruption must raise, not degrade into a silent wrong graph."""
    import shutil
    for name in ["ids.npy", "data.npy", "indices.npy", "indptr.npy", "manifest.json"]:
        shutil.copyfile(Path(GRAPH) / name, tmp_path / name)
    arr = np.load(tmp_path / "data.npy").copy()
    arr[:10] = arr[:10] + 1.0
    np.save(tmp_path / "data.npy", arr)
    with pytest.raises(ValueError):
        Connectome(tmp_path, verify=True)


@needs_graph
def test_real_rows_sum_to_one():
    conn = load_connectome()
    sums = np.asarray(conn.matrix.sum(axis=1)).ravel()
    nz = sums[sums > 0]
    assert np.allclose(nz, 1.0, atol=1e-4)
    assert nz.size > 160000


@needs_graph
def test_real_full_graph_step_and_telemetry():
    conn = load_connectome()
    res = FlyReservoir(conn, embedding_dim=EMBED, dims=DIMS, seed=7301)
    assert res.n == 166700
    assert res.sample_index.shape == (512,)
    emb = np.random.default_rng(1).standard_normal(EMBED).astype(np.float32)
    f = res.step(emb)
    assert f.shape == (DIMS,) and np.isfinite(f).all()
    tel = res.telemetry()
    assert len(tel["sampled_state"]) == 512 and len(tel["spikes"]) >= 0
    assert sum(tel["sampled_state"]) != 0.0


@needs_graph
def test_real_no_edges_all_zero():
    conn = load_connectome()
    res = FlyReservoir(conn, embedding_dim=EMBED, dims=DIMS, seed=7301)
    res.set_mode("no_edges")
    f = res.step(np.ones(EMBED, np.float32))
    assert np.array_equal(f, np.zeros(DIMS, np.float32))
    tel = res.telemetry()
    assert tel["spikes"] == []
    assert tel["active_fraction"] == 0.0


@needs_graph
def test_real_step_timing_is_measured():
    """Reported, not asserted into existence. 20 Hz needs <= 50 ms."""
    conn = load_connectome()
    res = FlyReservoir(conn, embedding_dim=EMBED, dims=DIMS, seed=7301)
    bench = res.benchmark(steps=10)
    print("\nfull-graph step: median %.2f ms (%.1f Hz single-thread)"
          % (bench["median_ms"], bench["hz_single_thread"]))
    assert bench["median_ms"] > 0.0
    assert bench["twenty_hz_ok"], "full graph fell below the 20 Hz budget"


@needs_graph
def test_real_subgraph_fallback_timing_is_measured():
    """The fallback exists and is faster; both numbers are real measurements."""
    conn = load_connectome()
    full = FlyReservoir(conn, embedding_dim=EMBED, dims=DIMS, seed=7301)
    sub = FlyReservoir(conn, embedding_dim=EMBED, dims=DIMS, seed=7301,
                       subgraph_size=30000)
    assert sub.n == 30000
    assert sub.connectome_edges == 25582938
    bf = full.benchmark(steps=8)
    bs = sub.benchmark(steps=8)
    print("\nfull graph:  %.2f ms/step (%.1f Hz), %d edges"
          % (bf["median_ms"], bf["hz_single_thread"], full.edges))
    print("subgraph %d: %.2f ms/step (%.1f Hz), %d edges"
          % (sub.n, bs["median_ms"], bs["hz_single_thread"], sub.edges))
    sub.set_mode("no_edges")
    assert np.array_equal(
        sub.step(np.ones(EMBED, np.float32)), np.zeros(DIMS, np.float32)
    )
    assert sub.telemetry()["spikes"] == []
