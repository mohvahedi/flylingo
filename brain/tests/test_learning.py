"""Math verification for the FlyLingo learning layer.

These tests check the arithmetic of the training layer rather than its plumbing:
the entropy term against finite differences, the dopamine rule at ``gate_lr = 0``,
the sign of a policy update under reward, and which weights the gated rule is
allowed to touch.

Run them with the project venv:

    D:/Projects/flylingo/brain/.venv/Scripts/python.exe -m pytest tests/test_learning.py -v

NOTE ON THE ENTROPY TERM. ``observe`` used to add ``+entropy_coef * p * (H - log p)``
to ``dz``, which is not the gradient of entropy: the ``H`` term had the wrong sign,
so the "entropy bonus" measurably sharpened the policy (cosine -0.997 with the true
gradient) and the suite carried a strict xfail for it. The adapter now uses
``-entropy_coef * p * (H + log p)``, and the xfail is gone; the finite-difference
check below is a plain assertion again. Measurements: NOTES_learning.md, finding 1.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]  # D:/Projects/flylingo
BRAIN = ROOT / 'brain'
if str(BRAIN) not in sys.path:
    sys.path.insert(0, str(BRAIN))

from brain.learning.adapter import (  # noqa: E402
    PARAMETER_NAMES,
    PolicyAdapter,
)

from brain.learning import train as train_mod  # noqa: E402


# --------------------------------------------------------------------- helpers


def softmax64(z):
    z = np.asarray(z, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy64(z):
    p = softmax64(z)
    return float(-np.sum(p * np.log(p)))


def fd_gradient(fn, z, eps=1e-6):
    """Central finite differences of a scalar function of a vector."""
    z = np.asarray(z, np.float64)
    g = np.zeros_like(z)
    for i in range(z.size):
        up = z.copy()
        dn = z.copy()
        up[i] += eps
        dn[i] -= eps
        g[i] = (fn(up) - fn(dn)) / (2.0 * eps)
    return g


def _entropy_probe(in_dim=6, hidden=5, seed=11):
    """An adapter whose only active learning rule is the entropy term.

    ``lr = 1.0`` with ``grad_normalize = False`` makes the parameter update equal
    the raw gradient, ``dopamine = False`` removes the gated rule, and the caller
    passes ``reward = 0.0`` on the first step so the advantage is exactly zero and
    the REINFORCE term vanishes. What is left in ``b2`` is the entropy term alone.
    """
    return PolicyAdapter(in_dim=in_dim, n_actions=4, hidden=hidden, seed=seed, lr=1.0,
                         dopamine=False, gate_lr=0.0, entropy_coef=1.0,
                         grad_normalize=False)


def _entropy_probe_step(adapter, feat, seed=3):
    """Return (delta_b2, delta_W2, h, W2_before, b2_before) for one zero-reward step."""
    rng = np.random.default_rng(seed)
    action, logprob = adapter.sample(feat, rng)
    w2_before = adapter.W2.copy()
    b2_before = adapter.b2.copy()
    _f, h, _z = adapter._forward(feat)
    out = adapter.observe(0.0, logprob, feat)
    assert out['advantage'] == 0.0, 'probe is only valid when the advantage is exactly 0'
    return adapter.b2 - b2_before, adapter.W2 - w2_before, h, w2_before, b2_before


# ------------------------------------------------- the finite-difference method


def test_true_entropy_gradient_analytic_form_matches_finite_difference():
    """The reference: dH/dz = -p * (H + log p), confirmed by finite differences.

    This is the control for the test below. If this passes, the finite-difference
    harness is trustworthy, so any disagreement found in the adapter is the
    adapter's fault and not the numerics'.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for n in (4, 8):
        for _ in range(6):
            z = rng.normal(0.0, 1.5, n)
            p = softmax64(z)
            h = float(-np.sum(p * np.log(p)))
            analytic = -p * (h + np.log(p))
            fd = fd_gradient(entropy64, z)
            worst = max(worst, float(np.max(np.abs(analytic - fd))))
    assert worst < 1e-6, f'analytic entropy gradient disagrees with FD by {worst}'


def test_entropy_is_highest_at_the_uniform_point_after_ascent():
    """The uniform point is the top of H, and ascent runs into it without passing it.

    Starting the ascent AT the uniform point tests nothing: it sits at z = 0,
    where dH/dz = -p*(H + log p) is exactly zero, so an ascender cannot move and
    the old assertion could never hold. The claims with content are that no
    logits beat the uniform point's entropy, and that ascending the true
    gradient from a non-uniform start raises H towards log n without exceeding
    it. Both are asserted below.
    """
    h_max = entropy64(np.zeros(4, np.float64))  # log 4, the max over all z
    rng = np.random.default_rng(0)
    for _ in range(64):
        z = rng.normal(0.0, 1.5, 4)
        assert entropy64(z) < h_max + 1e-12, 'uniform point is not the entropy maximum'

    z = np.array([2.4, -1.7, 0.6, -1.2], np.float64)
    h0 = entropy64(z)
    assert h0 < h_max - 1e-3, 'the ascent needs a genuinely non-uniform start'
    for _ in range(200):
        p = softmax64(z)
        h = float(-np.sum(p * np.log(p)))
        g = -p * (h + np.log(p))
        z = z + 0.5 * g
    h_end = entropy64(z)
    assert h_end > h0, 'ascending the true gradient did not raise entropy'
    assert h_end <= h_max + 1e-9, 'entropy exceeded log n, which is impossible'


# ------------------------------------------------------ the adapter under test


def test_adapter_entropy_term_is_what_the_source_line_computes():
    """Documents the implemented term: dz = -entropy_coef * p * (H + log p)."""
    a = _entropy_probe()
    feat = np.array([0.4, -1.1, 0.7, 0.2, -0.6, 1.3], np.float32)
    delta_b2, delta_w2, h, w2_before, b2_before = _entropy_probe_step(a, feat)

    # Recompute the probs from the weights the step was taken FROM. Calling
    # a.probs(feat) here (the old version) reads W2/b2 AFTER observe mutated
    # them, so the comparison was against a different weight setting: the two
    # sides only agreed by coincidence.
    p = softmax64(w2_before @ h + b2_before)
    h_ent = float(-np.sum(p * np.log(p)))
    implemented = (-(p * (h_ent + np.log(p)))).astype(np.float32)

    assert np.allclose(delta_b2, implemented, atol=1e-5), (
        'b2 update no longer equals the documented entropy expression; '
        f'max diff {float(np.max(np.abs(delta_b2 - implemented)))}')
    # The whole gradient is consistent with that dz: dW2 = outer(dz, h).
    assert np.allclose(delta_w2, np.outer(implemented, h).astype(np.float32), atol=1e-5)


def test_adapter_entropy_term_matches_entropy_finite_difference():
    """The claim in the adapter docstring, checked for real.

    Was ``xfail(strict=True)`` while ``observe`` used ``p * (H - log p)``; the
    adapter now uses ``-p * (H + log p)``, so this is a plain assertion again.
    """
    a = _entropy_probe()
    feat = np.array([0.4, -1.1, 0.7, 0.2, -0.6, 1.3], np.float32)
    delta_b2, _dw2, h, w2_before, b2_before = _entropy_probe_step(a, feat)

    def entropy_of_b2(b2):
        return entropy64(w2_before @ h + b2)

    fd = fd_gradient(entropy_of_b2, b2_before)
    assert np.allclose(delta_b2, fd, atol=1e-4), (
        'entropy term is not the gradient of entropy\n'
        f'  implemented b2 delta: {delta_b2}\n'
        f'  finite difference   : {fd}\n'
        f'  correct form -p(H+logp): '
        f'{(-softmax64(w2_before @ h + b2_before) * (entropy64(w2_before @ h + b2_before) + np.log(softmax64(w2_before @ h + b2_before)))).astype(np.float32)}')


# ------------------------------------------------------------------ the rules


def test_dopamine_off_at_zero_gate_lr_gives_bit_identical_trajectory():
    """gate_lr = 0 makes factor 2 contribute exactly nothing to the weights.

    The trajectory (weights, probabilities, sampled actions) of ``dopamine=True``
    must then be indistinguishable from ``dopamine=False``, i.e. from the
    parameter-matched control. The ``observe`` return value still differs, because
    telemetry such as ``gated_params`` reports the enabled rule rather than the
    arithmetic effect, so that is asserted separately.
    """
    a = PolicyAdapter(in_dim=8, n_actions=4, hidden=6, seed=3, lr=0.05,
                      entropy_coef=0.01, dopamine=True, gate_lr=0.0)
    b = a.clone(dopamine=False)
    for name in PARAMETER_NAMES:
        assert np.array_equal(getattr(a, name), getattr(b, name)), \
            f'{name} differs before any step'

    last_a = last_b = None
    rng_data = np.random.default_rng(1)
    for i in range(25):
        feat = rng_data.normal(size=8).astype(np.float32)
        act_a, lp_a = a.sample(feat, np.random.default_rng(100 + i))
        act_b, lp_b = b.sample(feat, np.random.default_rng(100 + i))
        assert act_a == act_b
        assert lp_a == lp_b
        reward = 1.0 if i % 3 == 0 else -0.25
        last_a = a.observe(reward, lp_a, feat)
        last_b = b.observe(reward, lp_b, feat)
        for name in PARAMETER_NAMES:
            assert np.array_equal(getattr(a, name), getattr(b, name)), \
                f'{name} diverged at step {i}'

    assert a.steps == b.steps == 25
    assert a.probs(np.zeros(8, np.float32)).tolist() == \
        b.probs(np.zeros(8, np.float32)).tolist()
    # Telemetry is the only place they differ.
    assert last_a['gated_params'] > 0
    assert last_b['gated_params'] == 0
    assert last_a['gated_weight_delta'] == 0.0
    assert last_b['gated_weight_delta'] == 0.0


def test_positive_reward_raises_probability_of_the_sampled_action():
    a = PolicyAdapter(in_dim=8, n_actions=4, hidden=6, seed=5, lr=0.1,
                      dopamine=False, entropy_coef=0.0, grad_normalize=False)
    feat = np.array([1.0, -0.5, 0.3, 0.8, -1.2, 0.4, 0.9, -0.3], np.float32)
    action, logprob = a.sample(feat, np.random.default_rng(11))
    before = float(a.probs(feat)[action])
    out = a.observe(1.0, logprob, feat)
    after = float(a.probs(feat)[action])
    assert out['advantage'] > 0.0
    assert after > before, f'p(action) fell from {before} to {after} under +1 reward'


def test_negative_reward_lowers_probability_of_the_sampled_action():
    a = PolicyAdapter(in_dim=8, n_actions=4, hidden=6, seed=5, lr=0.1,
                      dopamine=False, entropy_coef=0.0, grad_normalize=False)
    feat = np.array([1.0, -0.5, 0.3, 0.8, -1.2, 0.4, 0.9, -0.3], np.float32)
    action, logprob = a.sample(feat, np.random.default_rng(11))
    before = float(a.probs(feat)[action])
    out = a.observe(-1.0, logprob, feat)
    after = float(a.probs(feat)[action])
    assert out['advantage'] < 0.0
    assert after < before


def test_positive_reward_still_raises_probability_at_default_entropy_coef():
    """The entropy bug is small at the shipped coefficient, so this must hold."""
    a = PolicyAdapter(in_dim=8, n_actions=4, hidden=6, seed=5, lr=0.1,
                      dopamine=False, entropy_coef=0.01, grad_normalize=True)
    feat = np.array([1.0, -0.5, 0.3, 0.8, -1.2, 0.4, 0.9, -0.3], np.float32)
    action, logprob = a.sample(feat, np.random.default_rng(11))
    before = float(a.probs(feat)[action])
    a.observe(1.0, logprob, feat)
    assert float(a.probs(feat)[action]) > before


def test_factor_two_moves_gated_weights_and_never_ungated_ones():
    """With the policy rule switched off, only the enumerated gated weights move."""
    a = PolicyAdapter(in_dim=64, n_actions=4, hidden=32, seed=7, lr=0.0,
                      dopamine=True, gate_lr=0.05, gate_fraction=0.05,
                      entropy_coef=0.0, grad_normalize=False)
    assert a.gated_param_count > 0
    before = {n: getattr(a, n).copy() for n in PARAMETER_NAMES}
    rng = np.random.default_rng(4)
    feat = rng.normal(size=64).astype(np.float32)
    _act, lp = a.sample(feat, rng)
    out = a.observe(1.0, lp, feat)

    moved_total = 0
    for n in PARAMETER_NAMES:
        delta = getattr(a, n) - before[n]
        gated = a.gate[n]
        assert not np.any(delta[~gated]), f'{n}: an ungated weight moved'
        moved_total += int(np.count_nonzero(delta[gated]))
    assert moved_total > 0, 'factor 2 moved nothing'
    assert out['gated_weight_delta'] > 0.0
    assert out['gated_params'] == a.gated_param_count


# ------------------------------------------------------------------- plumbing


def test_parameter_counts_are_the_documented_ones():
    a = PolicyAdapter(in_dim=128, n_actions=4, hidden=256, seed=0,
                      dopamine=True, gate_lr=0.05)
    assert a.parameters() == 34052
    assert a.gated_parameters() == 1703
    assert a.gated_parameters() == int(sum(int(m.sum()) for m in a.gate.values()))
    b = a.clone(dopamine=False)
    assert b.parameters() == 34052, 'clone must not change the parameter count'
    assert b.gated_parameters() == 0
    assert np.array_equal(a.W1, b.W1) and np.array_equal(a.W2, b.W2)


def test_checkpoint_path_is_the_one_the_api_loads():
    """Regression guard: train.py once wrote one directory too deep."""
    assert train_mod.ROOT == ROOT
    assert train_mod.DEFAULT_CHECKPOINT == BRAIN / 'runs' / 'curriculum' / 'adapter.npz'
    api_src = (BRAIN / 'brain' / 'api.py').read_text(encoding='utf-8')
    assert 'adapter.npz' in api_src and 'curriculum' in api_src
    assert train_mod.DEFAULT_CHECKPOINT.name == 'adapter.npz'
    assert train_mod.DEFAULT_CHECKPOINT.parent.name == 'curriculum'
    assert train_mod.DEFAULT_CHECKPOINT.parent.parent.name == 'runs'
    assert train_mod.DEFAULT_CHECKPOINT.parent.parent.parent.name == 'brain'


def test_required_arms_are_three_and_param_matched_shares_the_intact_reservoir():
    """Regression guard: param_matched once named a reservoir mode that does not exist."""
    names = [s[0] for s in train_mod.ARM_SPECS]
    assert names == ['intact', 'param_matched', 'shuffled']
    mode = {s[0]: s[1] for s in train_mod.ARM_SPECS}
    assert mode['param_matched'] == 'intact', \
        'the parameter-matched control must see the same features as intact'
    assert mode['intact'] == 'intact'
    assert mode['shuffled'] == 'shuffled'
    which = {s[0]: s[2] for s in train_mod.ARM_SPECS}
    assert which['param_matched'] == 'param_matched'
    assert len(set(which.values())) == 3


class _FakeReservoir:
    """Minimal object matching the documented reservoir interface."""

    def __init__(self, dims=8):
        self.dims = dims

    def reset(self):
        return None

    def sequence(self, embeddings, mode='intact'):
        return np.zeros((len(embeddings), self.dims), np.float32)

    def step(self, emb, mode='intact'):
        return np.zeros(self.dims, np.float32)


def test_features_come_from_the_reservoir_object_and_not_from_a_rebuild():
    """Features are whatever the reservoir returns, not a re-encode of the input.

    The stub challenges must satisfy the curriculum schema that ``_validate``
    enforces: ``encode_challenge`` reads ``ch['prompt']`` and the old fixture
    omitted it, so the test raised KeyError before it could check anything about
    the reservoir. Note the empty case still reports DIMS, because with no
    challenges there is nothing to ask the reservoir about.
    """
    res = _FakeReservoir(dims=8)
    tasks = [{'id': 'a', 'type': 'translate', 'prompt': 'hola', 'answer': 'hello',
              'options': ['hello', 'goodbye', 'please', 'thanks'], 'correctIndex': 0,
              'difficulty': 1},
             {'id': 'b', 'type': 'translate', 'prompt': 'adios', 'answer': 'goodbye',
              'options': ['hello', 'goodbye', 'please', 'thanks'], 'correctIndex': 1,
              'difficulty': 1}]
    feats = train_mod.features_for(res, 'intact', tasks, 16, reset=True)
    assert feats.shape == (2, 8)
    assert train_mod.features_for(res, 'intact', [], 16).shape == (0, train_mod.DIMS)


def test_training_runs_the_three_arms_on_a_stand_in_reservoir(tmp_path):
    """Integration: three arms, per-epoch curves, honest labelling."""
    res = train_mod.train(epochs=2, force_standin=True, save_checkpoints=False,
                          results_path=tmp_path / 'r.json', eval_fraction=0.25,
                          reservoir_seed=1, adapter_seed=1, verbose=False)
    arms = res['arms']
    assert set(arms) == {'intact', 'param_matched', 'shuffled'}
    for name, a in arms.items():
        assert a['ran'] is True, f'{name} did not run'
        assert len(a['curve']) == 2, f'{name} curve is not per-epoch'
        for row in a['curve']:
            assert 'epoch' in row and 'train_accuracy' in row and 'eval_accuracy' in row
        assert a['params'] == 34052
    assert res['arms']['intact']['gated_params'] == 1703
    assert res['arms']['param_matched']['gated_params'] == 0
    assert res['arms']['param_matched']['same_init_as_intact'] is True
    assert res['reservoir']['is_standin'] is True
    assert res['task']['source'] in ('synthetic', 'curriculum')
    assert len(res['notes']) > 0
    assert any('STAND-IN' in n or 'stand-in' in n for n in res['notes']), \
        'a stand-in run must say so in its notes'
    assert res['comparison']['intact_and_param_matched_share_features'] is True
    assert (tmp_path / 'r.json').exists()


def test_results_file_is_written_with_every_arms_curve(tmp_path):
    import json
    out = tmp_path / 'res.json'
    train_mod.train(epochs=1, force_standin=True, save_checkpoints=False,
                    results_path=out, reservoir_seed=2, adapter_seed=2, verbose=False)
    data = json.loads(out.read_text(encoding='utf-8'))
    for name in ('intact', 'param_matched', 'shuffled'):
        assert data['arms'][name]['ran'] is True
        assert len(data['arms'][name]['curve']) == 1
    assert 'comparison' in data and 'verdict' in data['comparison']
