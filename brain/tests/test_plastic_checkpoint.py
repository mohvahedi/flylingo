"""A saved brain must answer identically, and a mismatched one must refuse to load.

Two failure modes this guards, both of which produce a brain that looks trained and answers
arbitrarily:

  - the pool assignment is a seeded shuffle of neuron ids and is NOT recomputable from the
    weights, so a checkpoint that stored only the scales would load onto the wrong neurons;
  - a checkpoint fit under one encoding scheme, applied under another, is the same class of
    failure the frozen-readout guard was added for.

So the round-trip is checked on real decisions, not on array equality, and each refusal path is
exercised rather than assumed.
"""
import json

import numpy as np
import pytest

from brain.encoders import encode_text
from brain.plastic_brain import PlasticBrain
from brain.reservoir import FlyReservoir, load_connectome


@pytest.fixture(scope="module")
def connectome():
    return load_connectome()


def _fresh(connectome, mode="intact"):
    r = FlyReservoir(connectome, seed=7301)
    r.set_mode(mode)
    pb = PlasticBrain(r, n_pools=4, pool_size=200, seed=99, steps=6)
    pb.set_mode(mode)
    return pb


def _train(pb, n=12):
    rng = np.random.default_rng(3)
    for i in range(n):
        emb = encode_text(f"frase numero {i}")
        pb.observe(emb, int(rng.integers(0, pb.n_pools)))


def test_save_load_reproduces_decisions(connectome, tmp_path):
    """The whole point: reload and the same prompt gets the same answer and the same probs."""
    pb = _fresh(connectome)
    _train(pb)
    probes = [encode_text(p) for p in ("hola", "gracias", "el gato", "la manzana")]
    before = [pb.forward(e)[0].copy() for e in probes]

    path = tmp_path / "brain.npz"
    meta = pb.save(path)
    assert meta["kind"] == "plastic_brain"
    assert meta["plastic_edges"] == pb.r.plastic_scale.size

    restored = _fresh(connectome)
    restored.load(path)
    after = [restored.forward(e)[0].copy() for e in probes]

    for i, (b, a) in enumerate(zip(before, after)):
        assert np.allclose(b, a, atol=1e-6), f"probe {i} changed after reload"
        assert int(np.argmax(b)) == int(np.argmax(a))


def test_load_restores_the_learned_weights(connectome, tmp_path):
    """Not just the outputs: the scales and pool weights must come back changed."""
    pb = _fresh(connectome)
    _train(pb)
    assert not np.allclose(pb.score_w, 1.0 / pb.pool_size), "training did not move the weights"

    path = tmp_path / "brain.npz"
    pb.save(path)

    restored = _fresh(connectome)
    assert np.allclose(restored.score_w, 1.0 / restored.pool_size), "fresh brain should start at 1/pool_size"
    restored.load(path)

    assert np.allclose(restored.score_w, pb.score_w), "pool weights did not restore"
    assert np.allclose(restored.r.plastic_scale, pb.r.plastic_scale), "synaptic scales did not restore"
    assert restored.updates == pb.updates


def test_a_fresh_brain_does_not_reproduce_a_trained_one(connectome, tmp_path):
    """The negative control: if reload changed nothing, the first test would prove nothing."""
    pb = _fresh(connectome)
    _train(pb)
    path = tmp_path / "brain.npz"
    pb.save(path)

    fresh = _fresh(connectome)
    trained = _fresh(connectome)
    trained.load(path)

    emb = encode_text("hola")
    # Compare the actual decision inputs, not the softmax: after 12 updates the two can still
    # agree on the argmax by chance, and this test must not depend on that.
    fresh_z = fresh.forward(emb)[1]
    trained_z = trained.forward(emb)[1]
    assert not np.allclose(fresh_z, trained_z), "a fresh brain matched a trained one; nothing trained"


def test_a_wrong_encoder_is_refused(connectome, tmp_path, monkeypatch):
    """A checkpoint from another encoding scheme must not load, and must say why."""
    pb = _fresh(connectome)
    _train(pb, n=4)
    path = tmp_path / "brain.npz"
    pb.save(path)

    import brain.plastic_brain as module

    real = module.encoder_fingerprint if hasattr(module, "encoder_fingerprint") else None
    monkeypatch.setattr("brain.encoders.encoder_fingerprint", lambda: "deadbeefdeadbeef")

    restored = _fresh(connectome)
    with pytest.raises(ValueError, match="encoder"):
        restored.load(path)


def test_a_different_pool_seed_is_refused(connectome, tmp_path):
    """A brain that assigns different neurons to each answer must not load this checkpoint.

    Both guards are legitimate here and which one fires depends on the pool seed: a different
    pool assignment also selects a different plastic edge set (measured: 119,535 edges against
    117,800), so the edge-count check can catch it first. Either refusal is correct; what matters
    is that the load does not proceed.
    """
    pb = _fresh(connectome)
    _train(pb, n=4)
    path = tmp_path / "brain.npz"
    pb.save(path)

    other_r = FlyReservoir(connectome, seed=7301)
    other = PlasticBrain(other_r, n_pools=4, pool_size=200, seed=1234, steps=6)
    with pytest.raises(ValueError, match="pool layout|plastic edges"):
        other.load(path)


def test_a_tampered_pool_layout_is_refused(connectome, tmp_path):
    """The pool-layout guard on its own, with the edge count matching so it is the one that fires.

    Storing only the weights would load them onto the wrong neurons, the assignment is a seeded
    shuffle of neuron ids and is not recomputable from the scales, so this path must be caught
    even when everything else lines up.
    """
    pb = _fresh(connectome)
    _train(pb, n=4)
    path = tmp_path / "brain.npz"
    pb.save(path)

    # Rewrite the stored pool layout, keeping the shape and edge count intact.
    with np.load(path, allow_pickle=False) as d:
        score_w, scale, pool_index, meta = (
            d["score_w"], d["plastic_scale"], d["pool_index"], d["meta"],
        )
    shuffled = np.stack([np.roll(p, 1) for p in pool_index]).astype(np.int64)
    assert shuffled.shape == pool_index.shape and not np.array_equal(shuffled, pool_index)
    np.savez_compressed(
        path, score_w=score_w, plastic_scale=scale, pool_index=shuffled, meta=meta
    )

    restored = _fresh(connectome)
    with pytest.raises(ValueError, match="pool layout"):
        restored.load(path)


def test_saved_meta_is_json_readable(connectome, tmp_path):
    """The sidecar has to be readable by a human debugging a weird answer."""
    pb = _fresh(connectome)
    _train(pb, n=4)
    path = tmp_path / "brain.npz"
    pb.save(path)
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
    for key in ("kind", "n_pools", "pool_size", "encoder_fingerprint", "updates", "plastic_edges"):
        assert key in meta, f"{key} missing from the checkpoint metadata"
