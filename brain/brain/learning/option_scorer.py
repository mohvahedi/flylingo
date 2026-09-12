"""Option-scoring readout: a permutation-equivariant matching head.

Why this module exists
----------------------
``PolicyAdapter`` maps ONE challenge vector to ``n_actions`` logits. For a fixed
random feature map that asks the readout to memorise arbitrary index-to-answer
associations, because the only thing that separates option 0 from option 1 is a
row of ``W2``. Nothing about the candidate itself enters the decision.

``OptionScorer`` changes the interface to matching. It consumes the per-candidate
features from ``brain.encoders.encode_option_pairs`` (shape ``(n_options,
in_dim)``, one row per candidate) and produces ONE scalar score per candidate:

    a_i     = tanh(W1 f_i + b1)          (hidden,)
    score_i = w . a_i + b2               scalar
    p       = softmax(score)

The same weights are applied to every row, so the map is permutation equivariant
by construction: permuting the rows permutes the scores identically and never
changes the set of probabilities. That property is the entire point of the
architecture change, and it is asserted in tests/test_option_scorer.py rather
than asserted in prose.

The optional gate, stated honestly
----------------------------------
With ``dopamine=True`` a fixed, explicitly enumerated subset of hidden units is
also updated by a three-factor rule: a reward-independent Hebbian eligibility
trace (pre-synaptic feature times post-synaptic unit activation), scaled by the
phasic dopamine signal ``reward - tonic``. The gated subset is exactly the rows
of ``W1``, the entries of ``b1`` and the entries of ``w`` belonging to
``k = max(1, round(gate_fraction * hidden))`` seeded hidden units, so the gated
count is ``k * (in_dim + 2)`` and is reported by ``gated_parameters()`` rather
than estimated. This is the same rule family as ``PolicyAdapter``'s gated rule,
re-indexed for a scorer whose "action" is the chosen candidate. It is not
claimed to be biologically derived: it is a masked low-rank Hebbian update,
nothing more.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from ..encoders import encoder_fingerprint

__all__ = ["OptionScorer", "LinearOptionScorer", "supervised_fit", "softmax"]

#: Parameters the gradient rule moves for every candidate-scoring readout.
PARAMETER_NAMES = ("W1", "b1", "w", "b2")

GATE_NAME = "dopamine_gated_hidden_subset"

KIND = "option_scorer"


def softmax(z) -> np.ndarray:
    """Numerically stable softmax over a 1-D array of scores."""
    z = np.asarray(z, np.float64).ravel()
    z = z - z.max() if z.size else z
    e = np.exp(z)
    total = float(e.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(z.shape, 1.0 / max(1, z.size), np.float32)
    return (e / total).astype(np.float32)


class OptionScorer:
    """Permutation-equivariant scorer over a set of candidate options.

    Parameters
    ----------
    in_dim:
        Width of one candidate feature row. 128 for the reservoir features.
    hidden:
        Width of the tanh hidden layer applied to every candidate.
    dopamine:
        Enable the enumerated masked Hebbian gate described in the module docstring.
    """

    def __init__(self, in_dim: int = 128, hidden: int = 256, seed: int = 0, lr: float = 0.05,
                 grad_normalize: bool = True, entropy_coef: float = 0.01,
                 baseline_decay: float = 0.99, dopamine: bool = True, gate_lr: float = 0.05,
                 gate_fraction: float = 0.05, eligibility_decay: float = 0.9,
                 tonic_decay: float = 0.99):
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.lr = float(lr)
        self.grad_normalize = bool(grad_normalize)
        self.entropy_coef = float(entropy_coef)
        self.baseline_decay = float(baseline_decay)
        self.dopamine = bool(dopamine)
        self.gate_lr = float(gate_lr)
        self.gate_fraction = float(gate_fraction)
        self.eligibility_decay = float(eligibility_decay)
        self.tonic_decay = float(tonic_decay)

        rng = np.random.default_rng(self.seed)
        self.W1 = (rng.standard_normal((self.hidden, self.in_dim))
                   / np.sqrt(self.in_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden, np.float32)
        self.w = (rng.standard_normal(self.hidden) / np.sqrt(self.hidden)).astype(np.float32)
        self.b2 = np.zeros((), np.float32)

        self.baseline = 0.0
        self.tonic = 0.0
        self.steps = 0
        self.gated_steps = 0

        self.elig = {n: np.zeros(getattr(self, n).shape, np.float32) for n in PARAMETER_NAMES}
        self.gate = self._build_gate()
        self.gated_param_count = int(sum(int(m.sum()) for m in self.gate.values()))
        self._cache: dict = {}

    # ------------------------------------------------------------------- setup

    def _build_gate(self) -> dict:
        """The enumerated gated subset: whole hidden units, chosen by seed.

        A hidden unit is either gated on all of the parameters that touch it (its
        row of ``W1``, its ``b1`` entry, its ``w`` entry) or not gated at all. The
        count is therefore exact and reproducible, and ``b2`` is never gated
        because it is shared by every candidate.
        """
        k = int(round(self.gate_fraction * self.hidden))
        k = max(1, min(self.hidden, k))
        rng = np.random.default_rng(self.seed + 991)
        idx = np.sort(rng.choice(self.hidden, k, replace=False))
        gate = {
            "W1": np.zeros((self.hidden, self.in_dim), bool),
            "b1": np.zeros(self.hidden, bool),
            "w": np.zeros(self.hidden, bool),
            "b2": np.zeros((), bool),
        }
        gate["W1"][idx, :] = True
        gate["b1"][idx] = True
        gate["w"][idx] = True
        return gate

    def config(self) -> dict:
        """Everything a checkpoint needs to be rebuilt, plus the encoder identity."""
        return {
            "kind": KIND,
            "class": type(self).__name__,
            "in_dim": int(self.in_dim),
            "hidden": int(self.hidden),
            "seed": int(self.seed),
            "lr": self.lr,
            "grad_normalize": self.grad_normalize,
            "entropy_coef": self.entropy_coef,
            "baseline_decay": self.baseline_decay,
            "dopamine": self.dopamine,
            "gate_name": GATE_NAME,
            "gate_lr": self.gate_lr,
            "gate_fraction": self.gate_fraction,
            "eligibility_decay": self.eligibility_decay,
            "tonic_decay": self.tonic_decay,
            "gated_params": int(self.gated_param_count),
            "total_params": int(self.parameters()),
            "encoder_fingerprint": encoder_fingerprint(),
            "interface": "score(features: (n_options, in_dim)) -> (n_options,)",
        }

    # ----------------------------------------------------------------- forward

    def _options(self, feature_rows) -> np.ndarray:
        F = np.asarray(feature_rows, np.float32)
        if F.ndim == 1:
            raise ValueError(
                "OptionScorer needs one row per candidate, shape (n_options, "
                f"in_dim={self.in_dim}); got a 1-D vector of size {F.size}. "
                "Use brain.encoders.encode_option_pairs for the option features.")
        if F.ndim != 2 or F.shape[1] != self.in_dim:
            raise ValueError(f"expected (n_options, {self.in_dim}) features, got {F.shape}")
        return F

    def _forward(self, feature_rows):
        F = self._options(feature_rows)
        A = np.tanh(F @ self.W1.T + self.b1).astype(np.float32)   # (n, hidden)
        s = (A @ self.w + self.b2).astype(np.float32)             # (n,)
        return F, A, s

    def score(self, features) -> float:
        """Scalar score for a single candidate."""
        return float(self._forward(np.asarray(features, np.float32)[None, :])[2][0])

    def scores(self, feature_rows) -> np.ndarray:
        """One score per candidate, in the order the candidates were given."""
        return self._forward(feature_rows)[2]

    def logits(self, feature_rows) -> np.ndarray:
        """Alias of ``scores`` so the service can treat readouts alike."""
        return self.scores(feature_rows)

    def probs(self, feature_rows) -> np.ndarray:
        """Softmax over the candidate scores. Pure: never touches the sample cache."""
        return softmax(self._forward(feature_rows)[2])

    # ---------------------------------------------------------------- sampling

    def sample(self, feature_rows, rng):
        """Sample one candidate, return ``(index, logprob)``.

        The pair pattern matches ``PolicyAdapter.sample`` so the service's
        sample-then-observe flow is unchanged.
        """
        F, A, s = self._forward(feature_rows)
        p = softmax(s)
        p64 = p.astype(np.float64)
        p64 = p64 / p64.sum()
        index = int(rng.choice(p.size, p=p64))
        logprob = float(np.log(max(float(p[index]), 1e-9)))
        onehot = np.zeros(p.size, np.float32)
        onehot[index] = 1.0
        self._cache = {"F": F, "A": A, "s": s, "p": p, "index": index, "onehot": onehot}
        return index, logprob

    # ---------------------------------------------------------------- learning

    def observe(self, reward, logprob, feature_rows=None) -> dict:
        """One reward-modulated update.

        Returns the same keys ``PolicyAdapter.observe`` returns, so a caller can
        consume either readout without branching.
        """
        reward = float(reward)
        empty = {"loss": 0.0, "grad_norm": 0.0, "entropy": 0.0, "updated_params": 0,
                 "gated_params": 0, "gated_weight_delta": 0.0, "dopamine": 0.0,
                 "advantage": 0.0, "baseline": float(self.baseline), "steps": self.steps}
        if "p" not in self._cache:
            # Never invent an update for a candidate that was not sampled.
            return empty
        c = self._cache
        F, A, p, onehot = c["F"], c["A"], c["p"], c["onehot"]

        advantage = reward - float(self.baseline)
        self.baseline += (1.0 - self.baseline_decay) * (reward - self.baseline)
        entropy = float(-(p.astype(np.float64) * np.log(p.astype(np.float64) + 1e-9)).sum())

        # Factor 1: REINFORCE with baseline plus the entropy bonus. Same identity
        # as PolicyAdapter: dH/dz_j = -p_j (H + log p_j), so the bonus term enters
        # with the sign that actually increases entropy.
        logp = np.log(p.astype(np.float64) + 1e-9)
        ds = (onehot.astype(np.float64) - p.astype(np.float64)) * advantage
        ds = ds - self.entropy_coef * (p.astype(np.float64) * (entropy + logp))

        # Backprop through the shared hidden layer, summed over candidates.
        G = (ds[:, None] * (self.w.astype(np.float64)[None, :]
                            * (1.0 - A.astype(np.float64) ** 2)))          # (n, hidden)
        grads = {
            "W1": (G.T @ F.astype(np.float64)).astype(np.float32),
            "b1": G.sum(axis=0).astype(np.float32),
            "w": (ds @ A.astype(np.float64)).astype(np.float32),
            "b2": np.asarray(ds.sum(), np.float32),
        }
        grad_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())) + 1e-12)
        scale = self.lr / max(grad_norm, 1e-3) if self.grad_normalize else self.lr
        for name in PARAMETER_NAMES:
            getattr(self, name)[...] += (scale * grads[name]).astype(np.float32)

        # Factor 2: masked three-factor Hebbian rule on the enumerated subset.
        # pre = the chosen candidate's feature row, post = that candidate's hidden
        # activations, third factor = phasic dopamine.
        phasic = 0.0
        gated_delta = 0.0
        if self.dopamine and self.gated_param_count:
            index = int(c["index"])
            phasic = reward - float(self.tonic)
            self.tonic += (1.0 - self.tonic_decay) * phasic
            tau = self.eligibility_decay
            instant = {
                "W1": np.outer(A[index], F[index]).astype(np.float32),
                "b1": A[index].copy(),
                "w": A[index].copy(),
                "b2": np.zeros((), np.float32),
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
            "loss": loss,
            "grad_norm": grad_norm,
            "entropy": entropy,
            "updated_params": int(self.parameters()),
            "gated_params": int(self.gated_param_count
                                if (self.dopamine and self.gated_param_count) else 0),
            "gated_weight_delta": gated_delta,
            "dopamine": float(phasic),
            "advantage": float(advantage),
            "baseline": float(self.baseline),
            "steps": int(self.steps),
        }

    # ------------------------------------------------------------------- stats

    def parameters(self) -> int:
        return int(sum(int(getattr(self, n).size) for n in PARAMETER_NAMES))

    def gated_parameters(self) -> int:
        return int(self.gated_param_count if self.dopamine else 0)

    # -------------------------------------------------------------- checkpoint

    def save(self, path):
        """Write a checkpoint atomically, carrying the encoder fingerprint in meta."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {n: getattr(self, n) for n in PARAMETER_NAMES}
        arrays.update({f"E_{n}": self.elig[n] for n in PARAMETER_NAMES})
        arrays.update({f"gate_{n}": self.gate[n] for n in PARAMETER_NAMES})
        arrays["meta"] = np.array(json.dumps(self.config()))
        arrays["baseline"] = np.array(self.baseline)
        arrays["tonic"] = np.array(self.tonic)
        arrays["steps"] = np.array(self.steps)
        arrays["gated_steps"] = np.array(self.gated_steps)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".npz")
        os.close(fd)
        try:
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def load(self, path):
        """Load a checkpoint. Missing gate or traces are rebuilt, never silently zeroed."""
        with np.load(path, allow_pickle=False) as d:
            files = set(d.files)
            shapes = {n: getattr(self, n).shape for n in PARAMETER_NAMES}
            for name in PARAMETER_NAMES:
                if name not in files:
                    raise ValueError(f"checkpoint is missing parameter {name}")
                arr = np.ascontiguousarray(d[name], np.float32)
                if arr.shape != shapes[name]:
                    # np.savez widens a 0-d array to shape (1,). Accept that one
                    # case explicitly, and only that one, so a real shape
                    # mismatch is still refused.
                    if shapes[name] == () and arr.size == 1:
                        arr = arr.reshape(())
                    else:
                        raise ValueError(
                            f"checkpoint {name} has shape {arr.shape}, "
                            f"scorer expects {shapes[name]}")
                setattr(self, name, arr)
            self.baseline = float(d["baseline"]) if "baseline" in files else 0.0
            self.tonic = float(d["tonic"]) if "tonic" in files else 0.0
            self.steps = int(d["steps"]) if "steps" in files else 0
            self.gated_steps = int(d["gated_steps"]) if "gated_steps" in files else 0
            meta = json.loads(str(d["meta"])) if "meta" in files else {}
            for name in PARAMETER_NAMES:
                trace = f"E_{name}"
                if trace in files and d[trace].shape == shapes[name]:
                    self.elig[name] = np.ascontiguousarray(d[trace], np.float32)
                else:
                    self.elig[name] = np.zeros(shapes[name], np.float32)
                mask = f"gate_{name}"
                if mask in files and d[mask].shape == shapes[name]:
                    self.gate[name] = np.asarray(d[mask], bool)
                else:
                    self.gate[name] = self._build_gate()[name]
        self.gated_param_count = int(sum(int(m.sum()) for m in self.gate.values()))
        for key in ("lr", "entropy_coef", "baseline_decay", "gate_lr", "eligibility_decay",
                    "tonic_decay", "gate_fraction"):
            if key in meta:
                setattr(self, key, float(meta[key]))
        for key in ("grad_normalize", "dopamine"):
            if key in meta:
                setattr(self, key, bool(meta[key]))
        self._cache = {}
        return self

    def clone(self, *, dopamine=None, seed=None, **overrides) -> "OptionScorer":
        """A fresh scorer with this architecture, optionally a different rule."""
        kwargs = {k: v for k, v in self.config().items()
                  if k in ("in_dim", "hidden", "seed", "lr", "grad_normalize", "entropy_coef",
                           "baseline_decay", "gate_fraction", "gate_lr", "eligibility_decay",
                           "tonic_decay", "dopamine")}
        kwargs.update(overrides)
        if dopamine is not None:
            kwargs["dopamine"] = dopamine
        if seed is not None:
            kwargs["seed"] = seed
        return OptionScorer(**kwargs)


class LinearOptionScorer:
    """A single linear layer over candidates: ``score_i = u . f_i + c``.

    The supervised upper bound in the option-scoring architecture with the hidden
    layer removed, so a failure here is a statement about the features rather than
    about depth.
    """

    def __init__(self, in_dim: int = 128, seed: int = 0):
        self.in_dim = int(in_dim)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.u = (rng.standard_normal(self.in_dim) * 0.01).astype(np.float32)
        self.c = np.zeros((), np.float32)

    def scores(self, feature_rows) -> np.ndarray:
        F = np.asarray(feature_rows, np.float32)
        if F.ndim == 1:
            F = F[None, :]
        return (F @ self.u + self.c).astype(np.float32)

    logits = scores

    def probs(self, feature_rows) -> np.ndarray:
        return softmax(self.scores(feature_rows))

    def parameters(self) -> int:
        return int(self.u.size + 1)

    def gated_parameters(self) -> int:
        return 0

    def config(self) -> dict:
        return {"kind": "linear_option_scorer", "in_dim": int(self.in_dim),
                "seed": int(self.seed), "parameters": int(self.parameters()),
                "encoder_fingerprint": encoder_fingerprint()}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, u=self.u, c=np.asarray(self.c),
                 meta=np.array(json.dumps(self.config())))

    def load(self, path):
        with np.load(path, allow_pickle=False) as d:
            self.u = np.ascontiguousarray(d["u"], np.float32)
            self.c = np.asarray(d["c"], np.float32)
        return self


def supervised_fit(model, feature_rows_by_challenge, labels, epochs: int = 400,
                   lr: float = 0.5, l2: float = 1e-3, seed: int = 0, verbose: bool = False):
    """Fit any option-scoring model by cross-entropy on cached features.

    This is the supervised upper bound: it uses the same features, the same
    architecture and the same candidate-scoring form as the reward-driven arms,
    and differs only in receiving the correct index as a label instead of a
    scalar reward. Full-batch gradient descent with a fixed schedule; no early
    stopping and no evaluation-driven tuning, so the reported accuracy is what
    the fixed schedule reaches rather than the best number seen.
    """
    Fs = [np.asarray(F, np.float32) for F in feature_rows_by_challenge]
    y = np.asarray(labels, np.int64)
    n, n_opt = len(Fs), Fs[0].shape[0]
    curve = []

    def logits_of(model, F):
        if isinstance(model, LinearOptionScorer):
            return F @ model.u + model.c
        A = np.tanh(F @ model.W1.T + model.b1)
        return A @ model.w + model.b2

    for epoch in range(int(epochs)):
        gW1 = np.zeros_like(model.W1) if not isinstance(model, LinearOptionScorer) else None
        gb1 = np.zeros_like(model.b1) if not isinstance(model, LinearOptionScorer) else None
        gw = np.zeros_like(model.w) if not isinstance(model, LinearOptionScorer) else None
        gb2 = 0.0
        gu = np.zeros_like(model.u) if isinstance(model, LinearOptionScorer) else None
        gc = 0.0
        loss_sum = 0.0
        for i in range(n):
            F = Fs[i]
            z = logits_of(model, F).astype(np.float64)
            p = softmax(z)
            t = int(y[i])
            loss_sum += -float(np.log(max(float(p[t]), 1e-12)))
            ds = p.copy()
            ds[t] -= 1.0
            if isinstance(model, LinearOptionScorer):
                gu += ds @ F
                gc += float(ds.sum())
            else:
                A = np.tanh(F @ model.W1.T + model.b1).astype(np.float64)
                G = ds[:, None] * model.w.astype(np.float64)[None, :] * (1.0 - A ** 2)
                gW1 += (G.T @ F).astype(np.float32)
                gb1 += G.sum(axis=0).astype(np.float32)
                gw += (ds @ A).astype(np.float32)
                gb2 += float(ds.sum())
        if isinstance(model, LinearOptionScorer):
            model.u -= (lr * (gu / n + l2 * model.u)).astype(np.float32)
            model.c -= np.asarray(lr * gc / n, np.float32)
        else:
            model.W1 -= (lr * (gW1 / n + l2 * model.W1)).astype(np.float32)
            model.b1 -= (lr * (gb1 / n)).astype(np.float32)
            model.w -= (lr * (gw / n + l2 * model.w)).astype(np.float32)
            model.b2 -= np.asarray(lr * gb2 / n, np.float32)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            hits = sum(int(np.argmax(logits_of(model, Fs[i])) == int(y[i])) for i in range(n))
            curve.append(hits / n)
            if verbose:
                print(f"    supervised epoch {epoch:>4} train {hits / n:.3f} loss {loss_sum / n:.4f}")
    return model, curve
