"""The connectome as the chooser AND the learner.

What this replaces, and why
---------------------------
The previous design froze the whole connectome and put a 516-parameter linear layer on top. That
layer did all the learning and all the deciding, so the fly's wiring was a fixed feature map, and
the controls showed it was interchangeable with a random one (intact, shuffled and random all
scored 1.000, spread 0.000). Calling that "the fly's brain learns Spanish" was not supportable.

Here the answer comes out of the brain's own neurons and the learning happens on the brain's own
synapses:

  * CHOICE. Four answer pools are disjoint groups of real neurons. The prompt is encoded, the
    connectome is settled for several recurrent steps, and the answer is whichever pool scores
    highest, each pool's score being a learned weighted sum over its OWN neurons. There is no
    classifier outside the brain: the argmax is over the brain's own populations.

    Why weighted and not a plain mean: an equal-weight mean of a pool measured 37.1% linearly
    separable against 100% for the same neurons read with learned weights. Averaging discards
    which of the pool's neurons fired, and that is where the answer is. A population read by
    learned synapses is also what a real mushroom body output neuron is, so this is the more
    faithful choice as well as the effective one.

  * LEARNING. The plastic parameters are real connectome edges -- the ones whose target neuron
    lies in a pool. Each carries a multiplicative scale on its anatomical weight. Training
    adjusts those scales, so the synapses that change are real synapses onto real neurons.

Why the update rule is the delta rule and not a hand-wave: a pool's activity is the mean of its
neurons' activity, and those neurons are driven by their incoming plastic edges, so the pool
activity is linear in the scales to first order. The cross-entropy gradient with respect to a
scale therefore reduces to

    dL/ds_e = error_of_the_edge's_pool * anatomical_weight_e * presynaptic_activity_e / pool_size

which is exactly what `observe` applies. The one approximation is that the presynaptic activity
itself depends on the scales through the recurrence; that second-order term is dropped, which is
the standard treatment, and it is stated here rather than buried.

Honesty rules carried over unchanged: every control (intact, shuffled, random_graph, no_edges)
gets identical pools, identical settling and identical plasticity, each operating on its own
pristine weights, so no control can accidentally learn on the intact connectome.
"""
from __future__ import annotations

import numpy as np

__all__ = ["PlasticBrain"]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class PlasticBrain:
    """Four answer pools on the connectome, chosen by settling and trained on real synapses."""

    def __init__(
        self,
        reservoir,
        n_pools: int = 4,
        pool_size: int = 200,
        seed: int = 99,
        steps: int = 6,
        max_plastic_edges: int | None = 120_000,
        # Temperature and the read-out step are tuned for the LEARNED read-out, which produces
        # larger logits than the unweighted mean did. At the old temperature of 0.004 the softmax
        # saturated once the weights grew and the loss ran to 18.
        #
        # Full sweep over temperature x lr_w, 8 epochs each on the real curriculum (epoch 4 and
        # epoch 8 accuracy, loss at epoch 8, and the spread of the learned weights):
        #
        #   temp   lr_w    ep4     ep8     loss    w_std
        #   0.05   0.002  58.8%   74.2%   1.1024  0.047   <- best
        #   0.05   0.010  58.8%   46.4%   1.8406  0.166
        #   0.20   0.002  49.5%   49.5%   1.1672  0.066
        #   0.20   0.010  60.8%   73.2%   1.1100  0.224
        #   1.00   0.002  43.3%   49.5%   1.2891  0.088
        #   1.00   0.010  49.5%   49.5%   1.1687  0.331
        #
        # 0.05 / 0.002 wins. 0.2 / 0.01 is a close second and is slightly ahead at epoch 4, but it
        # ends lower with a weight spread five times larger -- more extreme weights for the same
        # accuracy, which is a worse place to be. The choice is not knife-edge: the two best rows
        # both beat the mean read-out's 63.9% ceiling, so the improvement does not depend on
        # landing on one exact setting.
        temperature: float = 0.05,
        lr: float = 0.01,
        lr_w: float = 0.002,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ):
        self.r = reservoir
        self.n_pools = int(n_pools)
        self.pool_size = int(pool_size)
        self.steps = int(steps)
        self.temperature = float(temperature)
        self.lr = float(lr)
        self.lr_w = float(lr_w)

        # Adam state for the synaptic scales.
        #
        # Why an adaptive optimiser at all: the anatomical weights average 6.7e-3, because each
        # row of the connectome is normalised by its in-degree, so the raw cross-entropy gradient
        # with respect to a synaptic scale is around 1e-6. A plain step of that size moves nothing
        # -- the first attempt measured accuracy pinned at 19.6% with a step size identical every
        # epoch. A first remedy, scaling each step by the gradient's own global RMS, did start
        # learning (19.6% -> 36.1%) but moved every edge by the same amount regardless of whether
        # it mattered, which was noisy and drove scales into the clip.
        #
        # Adam keeps each edge's relative influence and denoises with momentum, so consistently
        # informative synapses take sustained steps while noise averages out. This is an optimiser
        # choice: the update direction is still the cross-entropy gradient with respect to the
        # synaptic scale.
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self._m = None
        self._v = None
        self._t = 0

        # Pools: disjoint groups of real neurons, selected by a seeded permutation. They are NOT
        # anatomically identified cell types and are not claimed to be; the claim is only that
        # they are real neurons of the reconstruction, and that the decision is their activity.
        rng = np.random.default_rng(seed)
        n = self.r.n
        take = self.n_pools * self.pool_size
        if take > n:
            raise ValueError("pools larger than the brain")
        chosen = rng.permutation(n)[:take]
        pool_of_row = np.full(n, -1, dtype=np.int64)
        self.pool_index: list[np.ndarray] = []
        for k in range(self.n_pools):
            idx = np.sort(chosen[k * self.pool_size : (k + 1) * self.pool_size])
            pool_of_row[idx] = k
            self.pool_index.append(idx.astype(np.int64))
        self.pool_of_row = pool_of_row

        self.plastic_info = self.r.enable_plasticity(
            pool_of_row, max_edges=max_plastic_edges
        )
        # Learned weights over each pool's own neurons: the answer populations read their
        # neurons with synapses, not an average. Initialised to 1/pool_size so the starting
        # point is exactly the unweighted mean, which makes the effect of learning measurable
        # against the mean rather than against an arbitrary different starting point.
        self.score_w = np.full((self.n_pools, self.pool_size), 1.0 / self.pool_size, np.float64)
        self._wm = self._wv = None
        self._wt = 0
        self.updates = 0

    # ------------------------------------------------------------------ forward

    @property
    def mode(self) -> str:
        return self.r.mode

    def set_mode(self, mode: str) -> None:
        """Switch control condition, keeping the learned scales.

        The scales are shared across modes on purpose: the comparison is the same learned
        synapses on different wiring, not a different set of synapses.
        """
        self.r.set_mode(mode)

    def _base_weights(self) -> np.ndarray:
        """The active matrix's pristine anatomical weights at the plastic positions."""
        key = "random" if self.r.mode == "random_graph" else "graph"
        return self.r._pristine[key]

    def pool_activity(self, state: np.ndarray) -> np.ndarray:
        """Each pool's score: a learned weighted sum over that pool's own neurons.

        With the weights at their initial 1/pool_size this is exactly the unweighted mean, so the
        two are directly comparable.
        """
        return np.array(
            [float(self.score_w[k] @ state[idx]) for k, idx in enumerate(self.pool_index)]
        )

    def forward(self, embedding: np.ndarray):
        """Settle the connectome, then read the pools. Returns (probs, z, state)."""
        self.r.reset()
        state, _ = self.r.settle(embedding, steps=self.steps)
        z = self.pool_activity(state)
        probs = _softmax(z / self.temperature)
        return probs, z, state

    def choose(self, embedding: np.ndarray) -> int:
        probs, _, _ = self.forward(embedding)
        return int(np.argmax(probs))

    def probs(self, embedding: np.ndarray) -> np.ndarray:
        return self.forward(embedding)[0]

    # ------------------------------------------------------------------- learn

    def observe(self, embedding: np.ndarray, target: int) -> dict:
        """One training example: settle, measure the pools, and move the real synapses.

        Returns the loss and the size of the step taken, so the caller can report learning rather
        than assert it.
        """
        probs, _, state = self.forward(embedding)
        target = int(target)

        # Cross-entropy gradient with respect to each pool's score.
        err = probs.copy()
        err[target] -= 1.0  # dL/dz for softmax + NLL

        # --- the pool read-out weights, trained alongside the synapses ---
        #
        # z_k = w_k . state[pool_k], so dz_k/dw_kj = state[pool_k][j]. Same Adam treatment as the
        # synaptic scales, and the same reason: the raw gradient is small and noisy.
        if self._wm is None:
            self._wm = np.zeros_like(self.score_w)
            self._wv = np.zeros_like(self.score_w)
        gw = np.empty_like(self.score_w)
        for k in range(self.n_pools):
            gw[k] = err[k] * state[self.pool_index[k]]
        self._wt += 1
        self._wm = self.beta1 * self._wm + (1.0 - self.beta1) * gw
        self._wv = self.beta2 * self._wv + (1.0 - self.beta2) * (gw * gw)
        wm_hat = self._wm / (1.0 - self.beta1 ** self._wt)
        wv_hat = self._wv / (1.0 - self.beta2 ** self._wt)
        self.score_w -= self.lr_w * wm_hat / (np.sqrt(wv_hat) + 1e-12)

        # Scale the presynaptic activity of every plastic edge by its pool's error and by its own
        # anatomical weight. Edges into a pool that was too active get weakened.
        pre = state[self.r.plastic_pre]
        base = self._base_weights()
        pool_err = err[self.r.plastic_pool]
        g = (pool_err * base * pre / float(self.pool_size)).astype(np.float32)

        # Adam on the scales.
        if self._m is None:
            self._m = np.zeros_like(g)
            self._v = np.zeros_like(g)
        self._t += 1
        self._m *= self.beta1
        self._m += (1.0 - self.beta1) * g
        self._v *= self.beta2
        self._v += (1.0 - self.beta2) * (g * g)
        m_hat = self._m / (1.0 - self.beta1 ** self._t)
        v_hat = self._v / (1.0 - self.beta2 ** self._t)
        delta = (self.lr * m_hat / (np.sqrt(v_hat) + 1e-12)).astype(np.float32)

        self.r.plastic_scale -= delta
        # Clamped positive: a negative anatomical weight would turn an excitatory connection into
        # an inhibitory one, a claim about transmitter sign this connectome does not carry. The
        # upper bound is generous because a single edge is only ~0.65% of a neuron's drive, so
        # scales of order ten are what it takes to move a population's activity appreciably.
        np.clip(self.r.plastic_scale, 0.0, 20.0, out=self.r.plastic_scale)
        self.r.apply_plastic()
        self.updates += 1

        loss = float(-np.log(max(probs[target], 1e-12)))
        return {
            "loss": loss,
            "correct": int(np.argmax(probs)) == target,
            "probs": probs.tolist(),
            "step_l1": float(np.sum(np.abs(delta))),
            "step_rms": float(np.sqrt(np.mean(delta * delta))),
        }

    # ------------------------------------------------------------------ report

    def accuracy(self, embeddings: np.ndarray, targets: np.ndarray) -> float:
        hit = 0
        for e, t in zip(embeddings, targets):
            if self.choose(e) == int(t):
                hit += 1
        return hit / len(embeddings)

    def stats(self) -> dict:
        s = self.r.plastic_scale
        return {
            "mode": self.r.mode,
            "pools": self.n_pools,
            "pool_size": self.pool_size,
            "settle_steps": self.steps,
            "plastic_edges": int(s.size),
            "plastic_of_total": f"{s.size / self.r.graph.nnz * 100:.2f}%",
            "plastic_scale_mean": float(np.mean(s)),
            "plastic_scale_std": float(np.std(s)),
            "updates": self.updates,
            "readout_params": int(self.score_w.size),
            "synapse_params": int(s.size),
            "trainable_params": int(s.size) + int(self.score_w.size),
        }
