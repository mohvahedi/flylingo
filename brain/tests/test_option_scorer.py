"""Tests for the option-scoring readout: permutation equivariance and the contract.

The equivariance tests are the point. If the scores are not exactly equivariant
under a permutation of the candidate rows, the architecture has not changed and
this is index lookup wearing a new name.
"""
import json

import numpy as np
import pytest

from brain.learning.option_scorer import (
    LinearOptionScorer,
    OptionScorer,
    supervised_fit,
)

IN_DIM = 32
HIDDEN = 48
N_OPT = 4


def _features(seed=0, n=N_OPT, dim=IN_DIM):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _perm(rng, n=N_OPT):
    return rng.permutation(n)


@pytest.mark.parametrize("dopamine", [True, False])
def test_scores_are_permutation_equivariant(dopamine):
    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN, seed=3, dopamine=dopamine)
    F = _features(1)
    base = scorer.scores(F)
    for trial in range(4):
        order = _perm(np.random.default_rng(trial))
        got = scorer.scores(F[order])
        assert np.array_equal(got, base[order]), (
            f"shuffling candidates did not permute the scores identically (order={order})")


def test_probs_permute_with_candidates():
    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN, seed=5)
    F = _features(2)
    base = scorer.probs(F)
    order = np.array([2, 0, 3, 1])
    got = scorer.probs(F[order])
    assert np.allclose(got, base[order], atol=1e-7)
    # As a set, the probabilities are unchanged, which is what makes the readout
    # order-independent rather than merely order-reindexed.
    assert np.allclose(np.sort(got), np.sort(base), atol=1e-7)


def test_a_trained_scorer_stays_permutation_equivariant():
    """Equivariance must survive training, including the gated rule."""
    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN, seed=7)
    rng = np.random.default_rng(11)
    F = _features(3)
    for _ in range(25):
        idx, lp = scorer.sample(F, rng)
        scorer.observe(1.0 if idx == 1 else -0.25, lp, F)
    base = scorer.scores(F)
    order = np.array([3, 1, 0, 2])
    assert np.array_equal(scorer.scores(F[order]), base[order])


def test_score_of_one_row_matches_the_batch():
    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN, seed=9)
    F = _features(4)
    scores = scorer.scores(F)
    for i in range(N_OPT):
        assert scorer.score(F[i]) == pytest.approx(float(scores[i]), rel=1e-5)


def test_linear_option_scorer_is_permutation_equivariant():
    lin = LinearOptionScorer(in_dim=IN_DIM, seed=1)
    F = _features(5)
    base = lin.scores(F)
    order = np.array([1, 3, 0, 2])
    assert np.array_equal(lin.scores(F[order]), base[order])


def test_a_one_dimensional_input_is_rejected_with_a_usable_message():
    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN)
    with pytest.raises(ValueError) as err:
        scorer.scores(np.zeros(IN_DIM, np.float32))
    assert "one row per candidate" in str(err.value)


def test_observe_returns_the_keys_the_service_reads():
    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN, seed=2)
    rng = np.random.default_rng(0)
    F = _features(6)
    _idx, lp = scorer.sample(F, rng)
    info = scorer.observe(1.0, lp, F)
    for key in ("loss", "grad_norm", "entropy", "updated_params", "gated_params",
                "gated_weight_delta"):
        assert key in info, key
    assert info["updated_params"] == scorer.parameters()
    assert info["gated_params"] == scorer.gated_parameters()


def test_observe_without_a_sample_never_invents_an_update():
    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN)
    before = [getattr(scorer, n).copy() for n in ("W1", "b1", "w", "b2")]
    info = scorer.observe(1.0, -0.5, _features(7))
    assert info["updated_params"] == 0 and info["gated_params"] == 0
    after = [getattr(scorer, n) for n in ("W1", "b1", "w", "b2")]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_the_gated_count_is_the_enumerated_mask_and_nothing_else():
    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN, seed=4, gate_fraction=0.25,
                          dopamine=True)
    k = int(round(0.25 * HIDDEN))
    assert scorer.gated_parameters() == k * (IN_DIM + 2)
    assert int(scorer.gate["W1"].sum()) == k * IN_DIM
    assert int(scorer.gate["b1"].sum()) == k
    assert int(scorer.gate["w"].sum()) == k
    assert not bool(scorer.gate["b2"].any())
    off = scorer.clone(dopamine=False)
    assert off.gated_parameters() == 0
    assert off.parameters() == scorer.parameters()


def test_the_gate_actually_moves_parameters_and_only_the_gated_ones():
    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN, seed=8, gate_fraction=0.1,
                          lr=0.0, gate_lr=0.5, dopamine=True)
    rng = np.random.default_rng(0)
    F = _features(8)
    before = {n: getattr(scorer, n).copy() for n in ("W1", "b1", "w", "b2")}
    for _ in range(5):
        idx, lp = scorer.sample(F, rng)
        info = scorer.observe(1.0 if idx == 0 else -0.25, lp, F)
    assert info["gated_weight_delta"] > 0.0
    assert scorer.gated_steps == 5
    # The disabled rule must be exactly inert: no parameter outside the mask moved.
    for name in ("W1", "b1", "w"):
        moved = np.abs(getattr(scorer, name) - before[name]) > 0.0
        assert np.array_equal(moved, scorer.gate[name]), name
    assert float(scorer.b2) == pytest.approx(float(before["b2"]))


def test_checkpoint_round_trip_carries_the_encoder_fingerprint(tmp_path):
    from brain.encoders import encoder_fingerprint

    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN, seed=6, dopamine=True)
    rng = np.random.default_rng(1)
    F = _features(9)
    for _ in range(10):
        idx, lp = scorer.sample(F, rng)
        scorer.observe(1.0 if idx == 2 else -0.25, lp, F)
    path = tmp_path / "adapter.npz"
    scorer.save(path)

    with np.load(path, allow_pickle=False) as d:
        assert "meta" in d.files
        meta = json.loads(str(d["meta"]))
    assert meta["encoder_fingerprint"] == encoder_fingerprint()
    assert meta["kind"] == "option_scorer"
    assert meta["gated_params"] == scorer.gated_parameters()

    fresh = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN, seed=99)
    fresh.load(path)
    assert np.array_equal(fresh.scores(F), scorer.scores(F))
    assert fresh.gated_parameters() == scorer.gated_parameters()
    assert fresh.steps == scorer.steps


def test_a_checkpoint_from_another_architecture_is_rejected(tmp_path):
    from brain.learning.adapter import PolicyAdapter

    path = tmp_path / "adapter.npz"
    PolicyAdapter(in_dim=IN_DIM, n_actions=N_OPT, hidden=HIDDEN, seed=0).save(path)
    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN)
    with pytest.raises(ValueError):
        scorer.load(path)


def test_it_learns_a_matching_task_the_features_can_solve():
    """Sanity: when the correct candidate is linearly separated, training finds it.

    The point of this test is that the reward-driven rule works at all on a
    solvable matching task, so a null on the real curriculum is a statement about
    that curriculum and not about a broken update.
    """
    scorer = OptionScorer(in_dim=IN_DIM, hidden=HIDDEN, seed=12, lr=0.2, dopamine=True)
    rng = np.random.default_rng(21)
    key = np.zeros(IN_DIM, np.float32)
    key[0] = 1.0
    n_ch = 6
    Fs = [rng.standard_normal((N_OPT, IN_DIM)).astype(np.float32) for _ in range(n_ch)]
    ys = []
    for F in Fs:
        t = int(rng.integers(N_OPT))
        F[t, 0] = 3.0
        ys.append(t)
    before = np.mean([int(np.argmax(scorer.scores(F))) == y for F, y in zip(Fs, ys)])
    for _ in range(60):
        for F, y in zip(Fs, ys):
            idx, lp = scorer.sample(F, rng)
            scorer.observe(1.0 if idx == y else -0.25, lp, F)
    after = np.mean([int(np.argmax(scorer.scores(F))) == y for F, y in zip(Fs, ys)])
    assert after >= 0.8, f"matching task not learned: before {before:.2f} after {after:.2f}"


def test_supervised_fit_fits_a_separable_matching_task():
    rng = np.random.default_rng(31)
    Fs = []
    ys = []
    for _ in range(8):
        F = rng.standard_normal((N_OPT, IN_DIM)).astype(np.float32)
        t = int(rng.integers(N_OPT))
        F[t, 1] = 4.0
        Fs.append(F)
        ys.append(t)
    model, curve = supervised_fit(LinearOptionScorer(in_dim=IN_DIM, seed=0), Fs, ys,
                                  epochs=60, lr=0.5)
    assert curve[-1] == 1.0
