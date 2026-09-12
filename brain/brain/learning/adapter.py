"""Reward-modulated policy adapter for the frozen FlyLingo connectome readout.

Two learning signals act on the same small numpy policy head:

1. REINFORCE with a running baseline. Every trainable weight is updated by the
   policy gradient of the sampled action, scaled by the reward prediction error
   (reward minus the running baseline plus a small entropy bonus). This is the
   broad, slow, whole-network signal.

2. A dopamine-gated memory rule. A small, explicitly enumerated subset of the
   adapter weights (``gate_fraction`` of each parameter array, selected by a fixed
   seeded mask, named ``dopamine_gate`` and counted by ``gated_parameters()``) is
   ADDITIONALLY modulated by a phasic dopamine signal. Those weights keep a Hebbian
   eligibility trace, an exponentially weighted average of recent pre/post
   co-activity, and are nudged by ``gate_lr * phasic_dopamine * eligibility``. This
   is a three-factor rule: it never uses the action advantage or the action
   likelihood, and it touches only the masked weights. Disabling it
   (``dopamine=False``) yields the parameter matched control adapter: identical
   architecture, identical parameter count, identical initialisation, no gated
   update.

Honest framing: the gated rule is a coarse Hebbian consolidation that is gated by
a reward prediction error, not a claim about fly dopamine. Whether it helps is an
empirical question answered by the parameter matched control in
``brain.learning.train``. It is deliberately NOT a second copy of REINFORCE: if it
were, the matched control would measure nothing.

Interface: frozen in INTERFACES.md. ``PolicyAdapter(in_dim, n_actions, hidden,
seed)`` plus ``logits``, ``probs``, ``sample``, ``observe``, ``save``, ``load``,
``parameters``. Everything after ``seed`` is optional and keyword only.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

#: Name of the gated weight set, reported in telemetry and checkpoints.
GATE_NAME = 'dopamine_gate'

#: Parameter arrays, in update order.
PARAMETER_NAMES = ('W1', 'b1', 'W2', 'b2')

#: Fraction of each parameter array that belongs to the gated set.
DEFAULT_GATE_FRACTION = 0.05


def _softmax(z):
    """Numerically stable softmax, returned as normalised float32."""
    z = np.asarray(z, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


def unit_rms(x, eps=1e-6):
    """Scale a vector to unit root mean square, matching reservoir features."""
    x = np.asarray(x, np.float32)
    return x / (np.sqrt(np.mean(x * x)) + eps)


class PolicyAdapter:
    """Two-layer policy head over reservoir features, reward modulated.

    ``observe`` returns ``loss``, ``grad_norm``, ``entropy``, ``updated_params``
    (the number of weights moved by the policy gradient) and ``gated_params``
    (the number of weights moved by the dopamine-gated rule). Extra diagnostic
    keys are included and are safe for callers to ignore.
    """

    def __init__(self, in_dim=128, n_actions=4, hidden=256, seed=0, *,
                 lr=0.05, gate_fraction=DEFAULT_GATE_FRACTION, dopamine=True,
                 gate_lr=0.05, eligibility_decay=0.9, tonic_decay=0.99,
                 baseline_decay=0.95, entropy_coef=0.01, grad_normalize=True):
        rng = np.random.default_rng(int(seed))
        self.in_dim = int(in_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.seed = int(seed)

        # Same initialisation as the verified baseline stub: the policy must keep
        # learning the synthetic 4-class task it was checked against.
        self.W1 = (rng.standard_normal((hidden, in_dim)) / np.sqrt(in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = (rng.standard_normal((n_actions, hidden)) / np.sqrt(hidden)).astype(np.float32)
        self.b2 = np.zeros(n_actions, np.float32)

        self.lr = float(lr)
        self.grad_normalize = bool(grad_normalize)
        self.entropy_coef = float(entropy_coef)
        self.baseline = 0.0
        self.baseline_decay = float(baseline_decay)

        # Dopamine-gated memory rule.
        self.dopamine = bool(dopamine)
        self.gate_fraction = float(gate_fraction)
        self.gate_lr = float(gate_lr)
        self.eligibility_decay = float(eligibility_decay)
        self.tonic_decay = float(tonic_decay)
        self.tonic = 0.0
        self.elig = {name: np.zeros_like(getattr(self, name)) for name in PARAMETER_NAMES}
        self.gate = self._build_gate()
        self.gated_param_count = int(sum(int(m.sum()) for m in self.gate.values()))

        self.steps = 0
        self.gated_steps = 0
        self._cache = {}

    # ------------------------------------------------------------------ setup

    def _build_gate(self):
        """Enumerate the gated weights exactly, from the seed, once.

        The mask seed is derived from the adapter seed but decoupled from the
        weight initialisation stream, so changing the gate fraction never changes
        the initial policy. Exactly ``round(fraction * size)`` entries are taken
        from each array (at least one while the fraction is positive), never a
        random count, so the reported gated count is deterministic.
        """
        g = np.random.default_rng([int(self.seed) & 0xFFFFFFFF, 0x0D0FA6A7])
        masks = {}
        for name in PARAMETER_NAMES:
            arr = getattr(self, name)
            size = int(arr.size)
            if self.gate_fraction <= 0.0:
                count = 0
            else:
                count = max(1, int(round(self.gate_fraction * size)))
                count = min(count, size)
            flat = np.zeros(size, bool)
            if count:
                flat[g.choice(size, count, replace=False)] = True
            masks[name] = flat.reshape(arr.shape)
        return masks

    def config(self):
        """Hyperparameters that must travel with a checkpoint."""
        return {
            'in_dim': self.in_dim,
            'n_actions': self.n_actions,
            'hidden': self.hidden,
            'seed': self.seed,
            'lr': self.lr,
            'grad_normalize': self.grad_normalize,
            'entropy_coef': self.entropy_coef,
            'baseline_decay': self.baseline_decay,
            'gate_name': GATE_NAME,
            'dopamine': self.dopamine,
            'gate_fraction': self.gate_fraction,
            'gate_lr': self.gate_lr,
            'eligibility_decay': self.eligibility_decay,
            'tonic_decay': self.tonic_decay,
            'gated_params': int(self.gated_param_count),
            'total_params': int(self.parameters()),
        }

    # ---------------------------------------------------------------- forward

    def _forward(self, features):
        f = np.asarray(features, np.float32).ravel()
        if f.size != self.in_dim:
            raise ValueError(f'expected {self.in_dim} features, got {f.size}')
        h = np.tanh(self.W1 @ f + self.b1)
        return f, h, self.W2 @ h + self.b2

    def logits(self, features):
        return self._forward(features)[2]

    def probs(self, features):
        # Pure: never touches the sample cache, so the service can call it after
        # sample() and before observe().
        return _softmax(self._forward(features)[2])

    def sample(self, features, rng):
        f, h, z = self._forward(features)
        p = _softmax(z)
        p64 = p.astype(np.float64)
        p64 /= p64.sum()
        action = int(rng.choice(self.n_actions, p=p64))
        logprob = float(np.log(max(float(p[action]), 1e-9)))
        onehot = np.zeros(self.n_actions, np.float32)
        onehot[action] = 1.0
        self._cache = {'f': f, 'h': h, 'z': z, 'p': p, 'action': action, 'onehot': onehot}
        return action, logprob

    # --------------------------------------------------------------- learning

    def observe(self, reward, logprob, features=None):
        """One reward-modulated update. See the module docstring for the two rules."""
        reward = float(reward)
        if 'p' not in self._cache:
            # Defensive: the service always samples before observing. Never invent
            # an update for an action that was not taken.
            return {'loss': 0.0, 'grad_norm': 0.0, 'entropy': 0.0, 'updated_params': 0,
                    'gated_params': 0, 'gated_weight_delta': 0.0, 'dopamine': 0.0,
                    'advantage': 0.0, 'baseline': float(self.baseline), 'steps': self.steps}
        c = self._cache
        f, h, p, onehot = c['f'], c['h'], c['p'], c['onehot']

        # Reward prediction error against the running baseline.
        advantage = reward - float(self.baseline)
        self.baseline += (1.0 - self.baseline_decay) * (reward - self.baseline)

        entropy = float(-(p * np.log(p + 1e-9)).sum())

        # ---- factor 1: REINFORCE with baseline, plus an entropy bonus -------
        # dH/dz_j = -p_j * (H + log p_j) for H = -sum_k p_k log p_k. Verified
        # against a central finite difference in tests/test_learning.py and in
        # scripts/_entropy_probe_check.py (max abs difference ~1e-7 on the probe
        # point, and the identity itself matches FD to ~2e-10).
        # The sign here matters: with +p*(H - log p) the "bonus" pointed down the
        # entropy gradient (cosine -0.997 with it) and measurably sharpened the
        # policy, contradicting the -entropy_coef * entropy term in `loss` below.
        # See NOTES_learning.md, finding 1.
        logp = np.log(p.astype(np.float64) + 1e-9)
        dz = (onehot.astype(np.float64) - p) * advantage
        dz = dz - self.entropy_coef * (p * (entropy + logp))
        dz = dz.astype(np.float32)
        grads = {
            'W2': np.outer(dz, h).astype(np.float32),
            'b2': dz.copy(),
            'W1': np.outer(((self.W2.T @ dz) * (1.0 - h * h)).astype(np.float32), f).astype(np.float32),
            'b1': ((self.W2.T @ dz) * (1.0 - h * h)).astype(np.float32),
        }
        grad_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())) + 1e-12)
        # Normalised steepest ascent, the update the verified stub used. Keeps the
        # step length independent of feature scale, which matters because the
        # reservoir emits unit-RMS features.
        scale = self.lr / max(grad_norm, 1e-3) if self.grad_normalize else self.lr
        for name in PARAMETER_NAMES:
            getattr(self, name)[...] += (scale * grads[name]).astype(np.float32)

        # ---- factor 2: dopamine-gated memory rule on the enumerated subset ---
        phasic = 0.0
        gated_delta = 0.0
        if self.dopamine and self.gated_param_count:
            phasic = reward - float(self.tonic)
            self.tonic += (1.0 - self.tonic_decay) * phasic
            tau = self.eligibility_decay
            # Reward-independent Hebbian co-activity, then gate it by dopamine.
            instant = {
                'W1': np.outer(h, f).astype(np.float32),
                'b1': h.copy(),
                'W2': np.outer(onehot, h).astype(np.float32),
                'b2': onehot.copy(),
            }
            for name in PARAMETER_NAMES:
                trace = self.elig[name]
                trace *= tau
                trace += (1.0 - tau) * instant[name]
                delta = ((self.gate_lr * phasic) * trace) * self.gate[name]
                getattr(self, name)[...] += delta
                gated_delta += float(np.abs(delta).sum())
            self.gated_steps += 1

        self.steps += 1
        loss = -(float(logprob) * advantage) - self.entropy_coef * entropy
        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'entropy': entropy,
            'updated_params': int(self.parameters()),
            'gated_params': int(self.gated_param_count if (self.dopamine and self.gated_param_count) else 0),
            'gated_weight_delta': gated_delta,
            'dopamine': float(phasic),
            'advantage': float(advantage),
            'baseline': float(self.baseline),
            'steps': int(self.steps),
        }

    # ----------------------------------------------------------------- stats

    def parameters(self):
        return int(sum(int(getattr(self, n).size) for n in PARAMETER_NAMES))

    def gated_parameters(self):
        """How many adapter weights the dopamine-gated rule can move."""
        return int(self.gated_param_count if self.dopamine else 0)

    # ------------------------------------------------------------- checkpoint

    def save(self, path):
        """Write a checkpoint atomically so a live service never reads a partial file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {n: getattr(self, n) for n in PARAMETER_NAMES}
        arrays.update({f'E_{n}': self.elig[n] for n in PARAMETER_NAMES})
        arrays.update({f'gate_{n}': self.gate[n] for n in PARAMETER_NAMES})
        arrays['meta'] = np.array(json.dumps(self.config()))
        arrays['baseline'] = np.array(self.baseline)
        arrays['tonic'] = np.array(self.tonic)
        arrays['steps'] = np.array(self.steps)
        arrays['gated_steps'] = np.array(self.gated_steps)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + '.', suffix='.npz')
        os.close(fd)
        try:
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def load(self, path):
        """Load a checkpoint. Older checkpoints saved without a gate are still read.

        A legacy file (the pre-learning baseline stub) carries only W1, b1, W2, b2,
        baseline and steps. Its missing gate is rebuilt from the constructor seed and
        its missing traces start at zero, so an old checkpoint never silently disables
        a rule the caller believes is running.
        """
        with np.load(path, allow_pickle=False) as d:
            files = set(d.files)
            shapes = {n: getattr(self, n).shape for n in PARAMETER_NAMES}
            for name in PARAMETER_NAMES:
                if name not in files:
                    raise ValueError(f'checkpoint is missing parameter {name}')
                arr = np.ascontiguousarray(d[name], np.float32)
                if arr.shape != shapes[name]:
                    raise ValueError(
                        f'checkpoint {name} has shape {arr.shape}, adapter expects {shapes[name]}')
                setattr(self, name, arr)
            self.baseline = float(d['baseline']) if 'baseline' in files else 0.0
            self.tonic = float(d['tonic']) if 'tonic' in files else 0.0
            self.steps = int(d['steps']) if 'steps' in files else 0
            self.gated_steps = int(d['gated_steps']) if 'gated_steps' in files else 0
            meta = json.loads(str(d['meta'])) if 'meta' in files else {}
            for name in PARAMETER_NAMES:
                trace = f'E_{name}'
                if trace in files and d[trace].shape == shapes[name]:
                    self.elig[name] = np.ascontiguousarray(d[trace], np.float32)
                else:
                    self.elig[name] = np.zeros(shapes[name], np.float32)
                mask = f'gate_{name}'
                if mask in files and d[mask].shape == shapes[name]:
                    self.gate[name] = np.asarray(d[mask], bool)
                else:
                    self.gate[name] = self._build_gate()[name]
        self.gated_param_count = int(sum(int(m.sum()) for m in self.gate.values()))
        for key in ('lr', 'entropy_coef', 'baseline_decay', 'gate_lr', 'eligibility_decay',
                    'tonic_decay', 'gate_fraction'):
            if key in meta:
                setattr(self, key, float(meta[key]))
        for key in ('grad_normalize', 'dopamine'):
            if key in meta:
                setattr(self, key, bool(meta[key]))
        self._cache = {}
        return self

    # --------------------------------------------------------------- helpers

    def clone(self, *, dopamine=None, seed=None, **overrides):
        """A fresh adapter with this architecture, optionally a different rule.

        Used to build the parameter matched control: identical architecture,
        identical parameter count, same initialisation when the seed is unchanged.
        """
        kwargs = {k: v for k, v in self.config().items()
                  if k in ('in_dim', 'n_actions', 'hidden', 'seed', 'lr', 'grad_normalize',
                           'entropy_coef', 'baseline_decay', 'gate_fraction', 'gate_lr',
                           'eligibility_decay', 'tonic_decay', 'dopamine')}
        kwargs.update(overrides)
        if dopamine is not None:
            kwargs['dopamine'] = dopamine
        if seed is not None:
            kwargs['seed'] = seed
        return PolicyAdapter(**kwargs)
