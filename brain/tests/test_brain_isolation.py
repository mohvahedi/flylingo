"""Training one brain must not change the weights another brain starts from.

This is the guard for a defect that silently invalidated a four-arm comparison. `self.graph` is a
reference to the shared connectome matrix and `apply_plastic` writes scaled values into its CSR
data in place, so a brain built after a trained one captured the trained weights as its own
"pristine anatomical" baseline. Measured before the fix: the second brain's baseline sat at
exactly 8.00x the anatomical weights after the first had been trained to a 8x scale, which means
the control arms did not start from the same weights and the comparison's central premise was
false.

The check is on the weights a second brain actually uses, not on whether an array was copied,
because a copy that still shares its buffers would pass a shape check and fail here.
"""
import numpy as np
import pytest

from brain.plastic_brain import PlasticBrain
from brain.reservoir import FlyReservoir, load_connectome


@pytest.fixture(scope="module")
def connectome():
    return load_connectome()


def _brain(connectome, seed=99):
    r = FlyReservoir(connectome, seed=7301)
    pb = PlasticBrain(r, n_pools=4, pool_size=200, seed=seed, steps=6)
    return r, pb


def test_training_one_brain_leaves_the_next_brains_baseline_alone(connectome):
    """The exact failure: train, then build a second brain and compare its baseline."""
    r_a, _ = _brain(connectome)
    pos = r_a.plastic_pos[:8]
    anatomical = r_a._pristine["graph"][:8].copy()

    # Train hard, as a converged run does (scales reach ~10).
    r_a.plastic_scale[:] = 8.0
    r_a.apply_plastic()
    # The scaled weights must live in THIS brain's own matrix...
    assert np.allclose(r_a.graph.data[pos], anatomical * 8.0, rtol=1e-4), (
        "the trained weights should be visible in the first brain's own matrix"
    )
    # ...and NOT in the shared connectome, which every later brain reads from.
    assert np.allclose(connectome.matrix.data[pos], anatomical, rtol=1e-4), (
        "training mutated the shared connectome, so every later brain inherits the scales"
    )

    r_b, _ = _brain(connectome)
    baseline = r_b._pristine["graph"][:8]
    assert np.allclose(baseline, anatomical, rtol=1e-4), (
        "a second brain inherited the first brain's trained weights as its pristine baseline; "
        f"got ratios {np.round(baseline / anatomical, 3)}"
    )


def test_training_does_not_mutate_the_shared_connectome(connectome):
    """The connectome itself must stay anatomical, whatever any brain does."""
    pos = None
    before = None
    r, _ = _brain(connectome)
    pos = r.plastic_pos[:8]
    before = r._pristine["graph"][:8].copy()

    r.plastic_scale[:] = 5.0
    r.apply_plastic()

    # A brand new reservoir reading the same connectome must see the anatomical values.
    fresh = FlyReservoir(connectome, seed=7301)
    assert np.allclose(fresh.graph.data[pos], before, rtol=1e-4), (
        "the shared connectome was mutated by plasticity; every later reader inherits it"
    )


def test_a_second_brain_starts_at_the_same_accuracy_as_the_first(connectome):
    """The end-to-end consequence: two fresh brains must behave identically."""
    from brain.encoders import encode_text

    r_a, pb_a = _brain(connectome)
    r_a.plastic_scale[:] = 6.0
    r_a.apply_plastic()

    emb = encode_text("hola")
    r_b, pb_b = _brain(connectome)

    probs_a = _brain(connectome)[1].forward(emb)[0]  # a clean third brain, built after training
    probs_b = pb_b.forward(emb)[0]
    assert np.allclose(probs_a, probs_b, atol=1e-9), (
        "two freshly built brains disagree, so a trained brain is leaking into later ones"
    )
