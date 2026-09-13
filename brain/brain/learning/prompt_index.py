"""Prompt-index readout: the architecture that actually learns the curriculum.

Why this module exists
----------------------
Four measurements established what works and what does not (see
brain/NOTES_task_learnability.md and scripts/probe_*.py):

  1. Option-scoring over (prompt, option) vectors: unlearnable. Vocabulary-disjoint held-out
     0.2784 (p=0.294); learn-to-criterion 0.536. Correctness is a relational property and
     character n-grams carry no cross-lingual meaning, so no global linear rule recovers it.
  2. Reservoir features versus controls on that framing: 0.567 vs 0.536, intact == shuffled
     == random. The connectome contributed nothing.
  3. Prompt-index memorisation, reservoir features: 1.000, and the SAME 1.000 for the
     shuffled graph, the random graph, and the raw encoding with no reservoir at all.

So the architecture that learns the vocabulary is: reservoir features of the PROMPT, mapped
to the answer index by a trained linear readout. That is drilling 97 phrases, which is what
"learns Spanish" means at this scale, and it is also the shape the reference FLM project
uses (frozen graph, trained adapter reading its state).

The honest caveat, which this module records rather than hides: measurement 3 shows the
measured wiring is indistinguishable from a shuffled or random graph on this task. The
166,700-neuron connectome is therefore not needed for the result. It supplies a fixed
nonlinear feature map, and any fixed nonlinear feature map of the same width does the same
job. Reporting the learning without that caveat would be the exact overclaim this project
was built to avoid.

API compatibility: exposes the same method names as PolicyAdapter so the service can use it
unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


class PromptIndexReadout:
    """Linear map from reservoir features of a prompt to answer-index logits."""

    def __init__(self, in_dim: int = 128, n_actions: int = 4, seed: int = 0,
                 lr: float = 0.5, l2: float = 1e-3, standardise: bool = True):
        rng = np.random.default_rng(int(seed))
        self.in_dim = int(in_dim)
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        self.W = (rng.standard_normal((self.n_actions, self.in_dim)) * 0.01).astype(np.float64)
        self.b = np.zeros(self.n_actions, dtype=np.float64)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.standardise = bool(standardise)
        # Feature normalisation, captured from the training set.
        self.mu = np.zeros(self.in_dim, dtype=np.float64)
        self.sd = np.ones(self.in_dim, dtype=np.float64)
        self.steps = 0
        self._cache: dict = {}

    # ------------------------------------------------------------- normalise

    def _norm(self, features: np.ndarray) -> np.ndarray:
        f = np.asarray(features, np.float64).ravel()
        if self.standardise:
            f = (f - self.mu) / self.sd
        return f

    def fit_normaliser(self, X: np.ndarray) -> None:
        X = np.asarray(X, np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6

    # --------------------------------------------------------------- forward

    def logits(self, features: np.ndarray) -> np.ndarray:
        return (self.W @ self._norm(features) + self.b).astype(np.float32)

    def probs(self, features: np.ndarray) -> np.ndarray:
        return _softmax(self.W @ self._norm(features) + self.b)

    def sample(self, features: np.ndarray, rng) -> tuple[int, float]:
        z = self.W @ self._norm(features) + self.b
        p = _softmax(z)
        p64 = p.astype(np.float64)
        p64 /= p64.sum()
        a = int(rng.choice(self.n_actions, p=p64))
        onehot = np.zeros(self.n_actions, dtype=np.float64)
        onehot[a] = 1.0
        self._cache = {"x": self._norm(features), "p": p, "action": a, "onehot": onehot}
        return a, float(np.log(max(float(p[a]), 1e-9)))

    # -------------------------------------------------------------- learning

    def observe(self, reward: float, logprob: float, features=None) -> dict:
        """One supervised step is not possible from a scalar reward alone.

        The service calls this after an answer to report progress. A reward of +1 means
        the chosen index was correct, which is exactly the label a supervised update needs,
        so the gradient is taken against the action that was actually taken. This keeps the
        online path consistent with the offline training in train_prompt_index.py.
        """
        if "p" not in self._cache:
            return {"loss": 0.0, "grad_norm": 0.0, "entropy": 0.0,
                    "updated_params": 0, "gated_params": 0, "steps": self.steps}
        x, p, onehot = self._cache["x"], self._cache["p"], self._cache["onehot"]
        correct = float(reward) > 0.0
        # Reinforce the chosen index when correct, suppress it when wrong.
        target = onehot if correct else (1.0 - onehot) / max(1, self.n_actions - 1)
        target = target / target.sum()
        g = (p.astype(np.float64) - target)
        gW = np.outer(g, x)
        gb = g
        gn = float(np.sqrt((gW * gW).sum() + (gb * gb).sum()) + 1e-12)
        self.W -= self.lr * 0.01 * gW
        self.b -= self.lr * 0.01 * gb
        entropy = float(-(p * np.log(p + 1e-9)).sum())
        self.steps += 1
        return {"loss": float(-np.log(max(float(p[int(np.argmax(onehot))]), 1e-9))),
                "grad_norm": gn, "entropy": entropy,
                "updated_params": int(self.parameters()), "gated_params": 0,
                "steps": self.steps}

    def parameters(self) -> int:
        return int(self.W.size + self.b.size)

    def observe_supervised(self, target_index: int, lr: float | None = None) -> dict:
        """One supervised step toward a KNOWN correct option.

        Why this exists alongside observe(). The reward-only rule can only reinforce the action
        the model happened to sample: on a correct answer it learns that action, and on a wrong
        answer it learns only "not that one", which says nothing about which option was right.
        Measured, that plateaus around 45% and never converges, however the learning rate is
        set, because the fly has to stumble onto each phrase-answer pair by luck before it can
        learn it.

        A lesson is not like that: the course HAS an answer key, so a wrong answer can say "it
        was option 3". Using it is what the offline training does as well (train_prompt_index.py
        fits against the known index), so this is consistent with the architecture rather than a
        shortcut around it. It is memorisation of the phrase-answer mapping, which is exactly
        what this project measured the readout to be good at.

        Takes the features of the previous sample() call from the cache, so it costs one forward
        and one rank-1 update: 4x128 numbers.
        """
        if "x" not in self._cache:
            return {"loss": 0.0, "grad_norm": 0.0, "entropy": 0.0, "steps": self.steps}
        x = self._cache["x"]
        p = _softmax(self.W @ x + self.b).astype(np.float64)
        target = np.zeros(self.n_actions, dtype=np.float64)
        target[int(target_index) % self.n_actions] = 1.0
        g = p - target
        gW = np.outer(g, x)
        gn = float(np.sqrt((gW * gW).sum() + (g * g).sum()) + 1e-12)
        step = self.lr if lr is None else float(lr)
        self.W -= step * gW
        self.b -= step * g
        # L2, matching the offline trainer, so the online path cannot drift somewhere the
        # offline one could not reach.
        self.W -= step * self.l2 * self.W
        entropy = float(-(p * np.log(p + 1e-9)).sum())
        self.steps += 1
        return {
            "loss": float(-np.log(max(float(p[int(target_index) % self.n_actions]), 1e-9))),
            "grad_norm": gn,
            "entropy": entropy,
            "steps": self.steps,
        }

    def rehearse(self, x: np.ndarray, target_index: int, lr: float | None = None) -> dict:
        """A supervised step from a REMEMBERED example rather than the current one.

        This is experience replay, and it is the difference between learning in a session and
        not. Measured over the real 97-challenge curriculum, from a fresh readout:
            no rehearsal      reaches about 45% and stays there
            60 rehearsal steps per answer  reaches 92-100% within about 100 answers
        One answer is one update, and one pass over 97 pairs is nowhere near enough to fit 516
        parameters; a learner rehearses. The step itself is identical to observe_supervised,
        only the features come from the buffer.
        """
        x = np.asarray(x, np.float64).ravel()
        saved = self._cache.get("x")
        self._cache["x"] = self._norm(x)
        try:
            return self.observe_supervised(target_index, lr=lr)
        finally:
            if saved is None:
                self._cache.pop("x", None)
            else:
                self._cache["x"] = saved

    def gated_parameters(self) -> int:
        return 0

    # ------------------------------------------------------------ checkpoint

    def save(self, path) -> None:
        from brain.encoders import encoder_fingerprint

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": "prompt_index_readout",
            "in_dim": self.in_dim,
            "n_actions": self.n_actions,
            "seed": self.seed,
            "lr": self.lr,
            "l2": self.l2,
            "standardise": self.standardise,
            "encoder_fingerprint": encoder_fingerprint(),
            # Recorded so no reader can mistake what the connectome did here.
            "connectome_contribution": (
                "none measurable: intact, shuffled and random graphs all reach 1.000 on this "
                "task, as does the raw encoding with no reservoir"
            ),
        }
        np.savez(
            path,
            W=self.W, b=self.b, mu=self.mu, sd=self.sd,
            meta=np.array(json.dumps(meta)),
            steps=np.array(self.steps),
        )

    def load(self, path) -> "PromptIndexReadout":
        with np.load(path, allow_pickle=False) as d:
            self.W = np.asarray(d["W"], np.float64)
            self.b = np.asarray(d["b"], np.float64)
            if "mu" in d.files:
                self.mu = np.asarray(d["mu"], np.float64)
                self.sd = np.asarray(d["sd"], np.float64)
            self.steps = int(d["steps"]) if "steps" in d.files else 0
        self.in_dim = int(self.W.shape[1])
        self.n_actions = int(self.W.shape[0])
        return self
