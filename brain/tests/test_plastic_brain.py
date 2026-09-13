"""Tests for the plastic brain: settling, real-synapse plasticity, and the learning rule.

These are deliberately small and use a reduced-dimension reservoir where possible, because the
point is to pin the mechanics, not to reproduce the full 166,700-neuron measurement (that lives
in scripts/measure_plastic_brain.py).
"""
import numpy as np
import pytest

from brain.plastic_brain import PlasticBrain, _softmax
from brain.encoders import encode_text
from brain.reservoir import FlyReservoir, load_connectome


@pytest.fixture(scope="module")
def connectome():
    return load_connectome()


@pytest.fixture(scope="module")
def reservoir(connectome):
    return FlyReservoir(connectome, embedding_dim=256, dims=128, seed=7301)


@pytest.fixture
def brain(reservoir):
    """A fresh brain per test: plasticity mutates the shared connectome's weights in place."""
    reservoir.set_mode("intact")
    reservoir.plastic_scale = None
    reservoir.plastic_pos = None
    reservoir._pristine = {}
    b = PlasticBrain(reservoir, n_pools=4, pool_size=200, seed=99, steps=6,
                     max_plastic_edges=20_000)
    yield b
    reservoir.plastic_scale = np.ones_like(reservoir.plastic_scale)
    reservoir.apply_plastic()


# ------------------------------------------------------------------ settling

def test_settling_actually_runs_the_recurrence(reservoir):
    """One pass after a reset is a single matrix multiply; settling must be more than that.

    This is the defect that made the wiring irrelevant: with a reset state of zero the recurrence
    never executed, so structure could not contribute. If this test ever fails, that is back.
    """
    emb = np.random.default_rng(0).standard_normal(256).astype(np.float32)
    reservoir.set_mode("intact")
    reservoir.reset()
    one, _ = reservoir.settle(emb, steps=1)
    reservoir.reset()
    eight, _ = reservoir.settle(emb, steps=8)
    assert np.abs(eight - one).max() > 1e-3, "settling did not change the state"


def test_settling_converges(reservoir):
    """It should settle to a fixed point, not diverge or oscillate."""
    emb = np.random.default_rng(1).standard_normal(256).astype(np.float32)
    reservoir.set_mode("intact")
    reservoir.reset()
    a, _ = reservoir.settle(emb, steps=8)
    reservoir.reset()
    b, _ = reservoir.settle(emb, steps=20)
    assert np.abs(a - b).max() < 0.05, "state had not settled"
    assert np.all(np.isfinite(a))


def test_settle_rejects_zero_steps(reservoir):
    emb = np.zeros(256, np.float32)
    with pytest.raises(ValueError):
        reservoir.settle(emb, steps=0)


# --------------------------------------------------------------- plasticity

def test_plastic_edges_are_real_edges(brain):
    """Every plastic position must index a genuine edge, and its target must be pooled."""
    r = brain.r
    assert r.plastic_pos.size > 0
    assert r.plastic_pos.max() < r.graph.data.size
    rows = np.searchsorted(r.graph.indptr, r.plastic_pos, side="right") - 1
    assert np.all(brain.pool_of_row[rows] == r.plastic_pool), "plastic edge targets a non-pool neuron"


def test_plasticity_changes_the_dynamics(brain):
    """Changing a synaptic scale must change what the brain computes."""
    emb = np.random.default_rng(2).standard_normal(256).astype(np.float32)
    brain.r.reset()
    before, _ = brain.r.settle(emb, steps=6)
    brain.r.plastic_scale[:] = 3.0
    brain.r.apply_plastic()
    brain.r.reset()
    after, _ = brain.r.settle(emb, steps=6)
    assert np.abs(before - after).max() > 1e-3


def test_restoring_the_scale_restores_the_state_exactly(brain):
    """apply_plastic must be idempotent and lossless, or the controls contaminate each other."""
    emb = np.random.default_rng(3).standard_normal(256).astype(np.float32)
    brain.r.reset()
    original, _ = brain.r.settle(emb, steps=6)
    brain.r.plastic_scale[:] = 2.5
    brain.r.apply_plastic()
    brain.r.plastic_scale[:] = 1.0
    brain.r.apply_plastic()
    brain.r.reset()
    restored, _ = brain.r.settle(emb, steps=6)
    assert np.allclose(original, restored, atol=1e-6)


def test_controls_do_not_share_weights(brain):
    """intact and random_graph must differ, and each must keep its own pristine set."""
    emb = np.random.default_rng(4).standard_normal(256).astype(np.float32)
    brain.r.plastic_scale[:] = 1.7
    brain.r.apply_plastic()
    brain.r.set_mode("intact")
    brain.r.reset()
    a, _ = brain.r.settle(emb, steps=6)
    brain.r.set_mode("random_graph")
    brain.r.reset()
    b, _ = brain.r.settle(emb, steps=6)
    assert np.abs(a - b).max() > 1e-4
    assert set(brain.r._pristine) == {"graph", "random"}


def test_no_edges_gives_exactly_zero_and_no_signal(brain):
    """The disconnected control must be genuinely dead, not nearly dead."""
    brain.set_mode("no_edges")
    emb = np.random.default_rng(5).standard_normal(256).astype(np.float32)
    probs, z, state = brain.forward(emb)
    assert np.all(state == 0.0)
    assert np.allclose(z, 0.0)
    # all pools equal -> softmax is uniform
    assert np.allclose(probs, 0.25, atol=1e-9)


# ------------------------------------------------------------------ choosing

def test_the_choice_comes_from_the_pools(brain):
    """argmax must be over pool activity, with no classifier in between."""
    emb = np.random.default_rng(6).standard_normal(256).astype(np.float32)
    probs, z, _ = brain.forward(emb)
    assert np.argmax(probs) == int(np.argmax(z))
    assert len(z) == brain.n_pools
    assert abs(probs.sum() - 1.0) < 1e-9


def test_the_readout_starts_as_the_unweighted_mean(brain):
    """The learned weights must begin at 1/pool_size, so the starting point is the plain mean and
    the effect of learning is measured against it rather than against an arbitrary start."""
    import numpy as np

    assert np.allclose(brain.score_w, 1.0 / brain.pool_size)
    emb = encode_text("How do you say 'Hello' in Spanish?")
    _, z, state = brain.forward(emb)
    means = np.array([float(state[ix].mean()) for ix in brain.pool_index])
    assert np.allclose(z, means), "initial readout is not the mean"


def test_the_readout_weights_are_trained(brain):
    """The pool weights are part of what learns, not fixed structure.

    Measured reason they exist: an equal-weight mean of a pool is 37.1% linearly separable while
    the same neurons read with learned weights are 100%. Averaging discards which of the pool's
    neurons fired, and that is where the answer is.
    """
    import numpy as np

    before = brain.score_w.copy()
    for _ in range(5):
        brain.observe(encode_text("How do you say 'Thank you' in Spanish?"), 0)
    assert not np.allclose(before, brain.score_w), "the readout weights never moved"


def test_pools_are_disjoint_real_neurons(brain):
    seen = np.concatenate(brain.pool_index)
    assert seen.size == brain.n_pools * brain.pool_size
    assert np.unique(seen).size == seen.size, "pools overlap"
    assert seen.max() < brain.r.n


# ------------------------------------------------------------------ learning

def test_trainable_parameters_count_both_parts(brain):
    """Reporting only the synapses would understate what learns, and only the readout would hide
    the synaptic plasticity entirely."""
    st = brain.stats()
    assert st["trainable_params"] == st["synapse_params"] + st["readout_params"]
    assert st["readout_params"] == brain.n_pools * brain.pool_size


def test_a_gradient_step_moves_the_scales(brain):
    X = np.random.default_rng(7).standard_normal((4, 256)).astype(np.float32)
    before = brain.r.plastic_scale.copy()
    out = brain.observe(X[0], 0)
    assert not np.allclose(before, brain.r.plastic_scale)
    assert np.isfinite(out["loss"]) and out["loss"] >= 0.0
    assert np.isfinite(out["step_rms"]) and out["step_rms"] > 0.0


def test_scales_stay_positive_and_bounded(brain):
    """A negative scale would flip an excitatory synapse into an inhibitory one, which this
    connectome carries no transmitter sign to justify."""
    X = np.random.default_rng(8).standard_normal((6, 256)).astype(np.float32)
    for i in range(6):
        brain.observe(X[i], i % brain.n_pools)
    s = brain.r.plastic_scale
    assert np.all(s >= 0.0)
    assert np.all(s <= 20.0)


def test_repeated_identical_examples_reduce_the_loss(brain):
    """The clearest possible statement of learning: feed one example, watch the loss fall."""
    emb = encode_text("How do you say 'Good morning' in Spanish?")
    target = 1
    first = brain.observe(emb, target)["loss"]
    for _ in range(40):
        last = brain.observe(emb, target)["loss"]
    assert last < first, f"loss did not fall ({first:.4f} -> {last:.4f})"


def test_the_learned_pool_wins_after_training(brain):
    """The decision itself should change, not just the loss.

    Uses a real encoded prompt: a random 256-vector is not an input this brain would ever see,
    and asking it to fit one tests the encoder's off-distribution behaviour rather than learning.

    The confidence threshold is set for this fixture, which caps at 20,000 plastic edges to stay
    quick. At the full 117,800 the same prompt reaches p(target) = 0.998 by the 100th update; the
    decision should be correct either way, which is what the assertion that matters checks.
    """
    emb = encode_text("How do you say 'Hello' in Spanish?")
    target = 1
    for _ in range(120):
        brain.observe(emb, target)
    probs, _, _ = brain.forward(emb)
    assert brain.choose(emb) == target
    assert probs[target] > 0.5, "the target pool should dominate, not merely edge ahead"


def test_the_update_is_a_gradient_descent_direction(brain):
    """The step must actually reduce the loss on the example it was taken on.

    Checked by measuring the loss before and after one update with a small learning rate, so the
    move is small enough that the first-order argument holds.
    """
    emb = encode_text("How do you say 'Thank you' in Spanish?")
    target = 3
    brain.lr = 0.002
    before = brain.forward(emb)[0][target]
    brain.observe(emb, target)
    after = brain.forward(emb)[0][target]
    assert after >= before, f"one small step lowered p(target): {before:.6f} -> {after:.6f}"


# ------------------------------------------------------------------ helpers

def test_softmax_is_stable_on_extremes():
    z = np.array([1000.0, -1000.0, 0.0])
    p = _softmax(z)
    assert np.all(np.isfinite(p))
    assert abs(p.sum() - 1.0) < 1e-12
    assert p[0] > 0.999
