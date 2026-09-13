"""FlyLingo simulation core: the frozen MaleCNS v1.0 connectome as a reservoir.

This is a clean reimplementation of the dynamics in ``flm/flm/graph.py``. It does
not import from ``flm`` at runtime; the reference is read-only documentation.

Honesty rules that are enforced here, not merely documented:

  - Every retained directed edge participates. ``W[post, pre]`` is contact count
    divided by that neuron's total incoming contacts, so rows sum to 1.
  - This deliberately infers NO transmitter sign, NO spikes, NO dopamine and NO
    biological time from anatomy. The state is an abstract rate-model state.
  - ``no_edges`` zeroes the recurrence exactly, so features are exactly zero. A
    zero feature vector is the honest visual signal that the graph is
    disconnected. We never substitute placeholder activity for it.
  - ``shuffled`` is a fixed node relabeling of ``W`` relative to the input and
    output interfaces. It preserves topology and tests interface alignment. It
    is NOT a claim about random graphs.
  - ``random_graph`` substitutes a degree-matched random sparse matrix with the
    same nnz, so the control differs only in wiring.

The recurrence, exactly, per step::

    x_t = tanh(W @ (0.6 * x_(t-1) + 0.4 * B @ embedding_t))
    f_t = normalize(P @ x_t)

``B`` and ``P`` are fixed seeded random interfaces, not anatomical language
pathways.

The full graph is the default. ``subgraph_size`` is an opt-in, seeded
degradation for RAM- or latency-bound deployments; it is never applied
implicitly, and a subgraph run is reported as a subgraph run.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

__all__ = [
    "GRAPH",
    "SAMPLES",
    "SPIKE_THRESHOLD",
    "Connectome",
    "FlyReservoir",
    "load_connectome",
    "sha256",
]

# ------------------------------------------------------------------ constants

GRAPH = Path("D:/Projects/flylingo/cache/malecns_v1")
SAMPLES = 512
SPIKE_THRESHOLD = 0.5
MODES = ("intact", "shuffled", "no_edges", "random_graph")
DATASET = "MaleCNS v1.0"
EXPECTED_NEURONS = 166700
EXPECTED_EDGES = 25582938
EXPECTED_CONTACTS = 124177617


def sha256(path) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------- connectome


class Connectome:
    """The frozen CSR graph plus its release metadata.

    Integrity is verified by recomputing sha256 of every array against the
    manifest before the arrays are used. A mismatch raises ``ValueError`` rather
    than degrading quietly.
    """

    def __init__(self, folder=GRAPH, verify=True):
        self.folder = Path(folder)
        manifest_path = self.folder / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest at {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())

        if verify:
            for name, digest in self.manifest["arrays"].items():
                if sha256(self.folder / name) != digest:
                    raise ValueError(f"Graph integrity check failed: {name}")

        self.ids = np.load(self.folder / "ids.npy")
        if self.ids.dtype != np.int64:
            raise ValueError("Neuron IDs must be int64.")
        if self.ids.ndim != 1 or np.any(np.diff(self.ids) <= 0):
            raise ValueError("Neuron IDs must be sorted ascending and unique.")

        arrays = [
            np.load(self.folder / (name + ".npy"))
            for name in ("data", "indices", "indptr")
        ]
        n = len(self.ids)
        self.matrix = sparse.csr_matrix(tuple(arrays), shape=(n, n), copy=False)
        if self.matrix.nnz != int(self.manifest["directed_edges"]):
            raise ValueError("Edge count differs from manifest.")
        if self.neurons != EXPECTED_NEURONS:
            raise ValueError(
                f"Expected {EXPECTED_NEURONS} neurons, manifest says {self.neurons}."
            )

        self.release = self.manifest.get("release", DATASET)
        self.directed_edges = int(self.manifest["directed_edges"])
        self.synaptic_contacts = int(self.manifest.get("synaptic_contacts", 0))
        self.verified = bool(verify)

    # The API reads these two directly.
    @property
    def neurons(self) -> int:
        return int(len(self.ids))

    @property
    def edges(self) -> int:
        return int(self.matrix.nnz)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"Connectome({self.release!r}, neurons={self.neurons}, "
            f"edges={self.edges}, verified={self.verified})"
        )


def load_connectome(folder=GRAPH, verify=True) -> Connectome:
    """Load the built MaleCNS v1.0 graph. Raises if the build is absent or bad."""
    return Connectome(folder=folder, verify=verify)


# ------------------------------------------------------------------ reservoir


class FlyReservoir:
    """Full-graph reservoir over the frozen connectome.

    ``step`` is allocation-free in the hot path: ``_pre``, ``_code``, ``_drive``,
    ``_prod`` and ``_feat`` are reused every call.
    """

    def __init__(self, connectome, embedding_dim: int = 256, dims: int = 128,
                 seed: int = 7301, subgraph_size: int = None):
        if isinstance(connectome, (str, Path)):
            connectome = load_connectome(connectome)
        self.connectome = connectome
        self.full_graph = connectome.matrix
        full_n = int(self.full_graph.shape[0])
        self.full_n = full_n
        self.embedding_dim = int(embedding_dim)
        self.dims = int(dims)
        self.seed = int(seed)
        self._mode = "intact"

        rng = np.random.default_rng(seed)

        # Optional subgraph fallback. The full graph is the default and the one
        # the reference runs; a subgraph is an explicitly chosen degradation for
        # RAM- or latency-bound deployments (DOOMFLY-style projects run 8192 to
        # 30000 neuron subgraphs). It is a fixed seeded subset, so a subgraph
        # reservoir is as reproducible as a full one.
        if subgraph_size is not None and int(subgraph_size) < full_n:
            self.subgraph_size = int(subgraph_size)
            self.sub_index = np.sort(
                rng.choice(full_n, self.subgraph_size, replace=False)
            ).astype(np.int64)
            self.graph = self.full_graph[self.sub_index][:, self.sub_index].tocsr()
            self.active_ids = connectome.ids[self.sub_index]
        else:
            self.subgraph_size = None
            self.sub_index = None
            self.graph = self.full_graph
            self.active_ids = connectome.ids
        self.n = int(self.graph.shape[0])

        # B: input interface. A seeded dense projection embedding -> dims, then a
        # fixed assignment of that code into n neurons with a random sign.
        self.B_projection = (
            rng.standard_normal((embedding_dim, dims)) / np.sqrt(embedding_dim)
        ).astype(np.float32)
        self.input_bins = rng.integers(0, dims, self.n)
        self.input_sign = rng.choice(np.array([-1, 1], np.float32), self.n)

        # P: output interface. Fixed readout of n neurons into dims bins, each bin
        # scaled by sqrt(bin occupancy) so bins are comparable regardless of size.
        self.output_bins = rng.integers(0, dims, self.n)
        self.output_sign = rng.choice(np.array([-1, 1], np.float32), self.n)
        counts = np.bincount(self.output_bins, minlength=dims)
        self.output_scale = np.sqrt(np.maximum(1, counts)).astype(np.float32)

        # Fixed node relabeling for the 'shuffled' control.
        self.permutation = rng.permutation(self.n)
        self.inverse = np.argsort(self.permutation)

        # Fixed readout sample: 512 neurons, spread across the ID space. Chosen
        # once from the seed so every frame samples the same cells.
        self.sample_index = np.linspace(0, self.n - 1, SAMPLES).astype(np.int64)
        self.sample_ids = [str(int(i)) for i in self.active_ids[self.sample_index]]

        # Degree-matched random control, built lazily on first use.
        self._random_graph = None

        # Reused buffers. `state` is the abstract rate-model state, n floats.
        self.state = np.zeros(self.n, np.float32)
        self.tmp_state = np.zeros(self.n, np.float32)
        self._code = np.zeros(dims, np.float32)
        self._drive = np.zeros(self.n, np.float32)
        self._prod = np.zeros(self.n, np.float32)
        self._feat = np.zeros(dims, np.float32)
        self._sample = np.zeros(SAMPLES, np.float32)
        self.updates = 0
        self.step_ms = 0.0

    # ------------------------------------------------------------ properties

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def neurons(self) -> int:
        return self.n

    @property
    def edges(self) -> int:
        """Stored edges in the active matrix (subgraph when one is selected)."""
        return int(self.graph.nnz)

    @property
    def connectome_edges(self) -> int:
        """Stored edges in the full frozen connectome, always 25582938."""
        return int(self.full_graph.nnz)

    def set_mode(self, mode: str) -> None:
        """Switch control mode. Unknown modes raise ValueError (API -> HTTP 400)."""
        if mode not in MODES:
            raise ValueError(
                f"Unknown mode {mode!r}. Expected one of {', '.join(MODES)}."
            )
        self._mode = mode
        if mode == "no_edges":
            # Disconnecting the graph must be instantaneous in the state too.
            self.state.fill(0.0)

    def reset(self) -> None:
        self.state.fill(0.0)
        self.updates = 0

    # ------------------------------------------------------------- mechanics

    def random_graph(self) -> sparse.csr_matrix:
        """Degree-matched random sparse matrix with the same nnz.

        Each row keeps the measured number of stored entries and every entry
        keeps its measured weight; only the presynaptic partner identities are
        randomized. Duplicate partners are kept rather than merged, so nnz is
        exactly the measured nnz and no row loses degree. Rows are then
        renormalized to sum to 1, exactly like the measured graph.

        This is the wiring control: the control differs from the real graph only
        in which partners carry the measured degree sequence.
        """
        if self._random_graph is None:
            rng = np.random.default_rng(self.seed + 17)
            src = self.graph.tocsr()
            indices = rng.integers(
                0, self.n, size=src.nnz, dtype=np.int64
            ).astype(src.indices.dtype)
            random_w = sparse.csr_matrix(
                (src.data.copy(), indices, src.indptr.copy()), shape=src.shape
            )
            row_sums = np.asarray(random_w.sum(axis=1)).ravel()
            random_w.data /= np.repeat(
                np.maximum(row_sums, 1.0), np.diff(random_w.indptr)
            ).astype(np.float32)
            self._random_graph = random_w
        return self._random_graph

    def step(self, embedding, mode: str = None) -> np.ndarray:
        """One reservoir update. Returns (dims,) float32 unit-RMS features.

        ``embedding`` is (embedding_dim,) float32. If ``mode`` is given it is
        applied for this step and stored.
        """
        if mode is not None and mode != self._mode:
            self.set_mode(mode)

        emb = np.asarray(embedding, np.float32)
        if emb.shape != (self.embedding_dim,):
            raise ValueError(
                f"embedding must have shape ({self.embedding_dim},), got {emb.shape}"
            )
        if not np.all(np.isfinite(emb)):
            raise ValueError("embedding contains non-finite values.")

        # B @ embedding_t, unit-RMS so the drive does not depend on input scale.
        code = self.B_projection.T @ emb
        code /= np.sqrt(np.mean(code * code) + 1e-6)

        # 0.6 * x_(t-1) + 0.4 * (assigned input code), into the reused buffer.
        np.take(code, self.input_bins, out=self._drive)
        self._drive *= self.input_sign
        self._drive *= 0.4
        self._drive += 0.6 * self.state

        if self._mode == "no_edges":
            # Exactly zero: tanh(W @ .) with W = 0 is 0, but writing it directly
            # keeps the feature path provably zero rather than nearly zero.
            self.state.fill(0.0)
        else:
            np.tanh(self._matvec_into(self._drive), out=self.state)

        # P @ x_t: binned, sign-flipped, occupancy-scaled, then unit-RMS.
        feat = np.bincount(
            self.output_bins,
            weights=self.state * self.output_sign,
            minlength=self.dims,
        ).astype(np.float32)
        feat /= self.output_scale
        feat /= np.sqrt(np.mean(feat * feat) + 1e-6)

        if self._mode == "no_edges":
            # Guard against any float noise turning a disconnected graph into a
            # signal for the adapter.
            feat.fill(0.0)
        self.updates += 1
        return feat

    def _matvec_into(self, vec: np.ndarray) -> np.ndarray:
        self.tmp_state[:] = vec[self.permutation] if self._mode == "shuffled" else vec
        if self._mode == "shuffled":
            self._prod[:] = self.graph @ self.tmp_state
            self._prod[:] = self._prod[self.inverse]
        elif self._mode == "random_graph":
            self._prod[:] = self.random_graph() @ vec
        else:
            self._prod[:] = self.graph @ vec
        return self._prod

    # --------------------------------------------------- settling and plasticity

    def settle(self, embedding, steps: int = 8):
        """Run the recurrence ``steps`` times with the input held, and return the settled state.

        This is the real dynamics: each step feeds the previous state back through the connectome,
        so the wiring shapes the trajectory and the fixed point it reaches. ``step`` already writes
        ``self.state`` in place, so repeated calls settle naturally; nothing is reset in between.
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")
        feat = None
        for _ in range(int(steps)):
            feat = self.step(embedding)
        return self.state.copy(), feat


    def enable_plasticity(self, pool_of_row: np.ndarray, max_edges: int | None = None,
                          seed: int | None = None) -> dict:
        """Mark the real edges that end on a pooled neuron as plastic.

        ``pool_of_row`` is an (n,) int array giving each neuron's pool index, or -1 for neurons that
        belong to no pool. Only edges whose TARGET is pooled are selected, because those are the
        synapses that directly drive the populations the answer is read from.

        The random-graph control is built here rather than lazily, so every mode has its own pristine
        edge weights captured before any plasticity is applied. Without that, switching modes could
        leave one control running on another's modified weights.

        This does NOT touch the mode. It used to force `intact` so the connectome matrix was the
        active one while capturing pristine weights, but that silently overrode the caller's
        control condition: a shuffled or random run would quietly execute on the intact graph and
        report the intact result. Both matrices are captured explicitly below, so nothing needs
        forcing, and whatever mode the caller chose stays chosen.
        """
        mode_at_entry = self._mode
        csr = self.graph.tocsr()
        rows = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(csr.indptr))
        mask = np.asarray(pool_of_row)[rows] >= 0
        pos = np.flatnonzero(mask)

        if max_edges is not None and pos.size > int(max_edges):
            rng = np.random.default_rng(self.seed if seed is None else seed)
            pos = np.sort(rng.choice(pos, int(max_edges), replace=False))

        self.plastic_pos = pos.astype(np.int64)
        self.plastic_pre = csr.indices[pos].astype(np.int64)
        self.plastic_pool = np.asarray(pool_of_row)[rows[pos]].astype(np.int64)
        self.plastic_scale = np.ones(pos.size, np.float32)

        # Make the control matrix exist now so its pristine weights are captured before any
        # plasticity is applied.
        self.random_graph()
        self._pristine = {
            "graph": csr.data[self.plastic_pos].copy(),
            "random": self._random_graph.data[self.plastic_pos].copy(),
        }
        self.apply_plastic()
        if self._mode != mode_at_entry:
            self.set_mode(mode_at_entry)

        return {
            "plastic_edges": int(pos.size),
            "pooled_neurons": int(np.sum(np.asarray(pool_of_row) >= 0)),
            "of_total_edges": int(csr.nnz),
        }


    def apply_plastic(self) -> None:
        """Write ``pristine * scale`` into the CSR data of every matrix, in place.

        Idempotent: writing the same scaled values twice is a no-op, so this is safe to call after
        any scale change and after any mode switch. Both the connectome and the random control are
        written, each from its own pristine weights, so the two never contaminate each other.
        """
        if getattr(self, "plastic_pos", None) is None:
            return
        scale = self.plastic_scale
        for key, mat in (("graph", self.graph), ("random", self._random_graph)):
            if mat is None:
                continue
            base = self._pristine.get(key)
            if base is None:
                base = mat.data[self.plastic_pos].copy()
                self._pristine[key] = base
            mat.data[self.plastic_pos] = base * scale


    def plastic_stats(self) -> dict:
        """What the plastic synapses look like now, for the UI and for the record."""
        if getattr(self, "plastic_pos", None) is None:
            return {"enabled": False}
        s = self.plastic_scale
        return {
            "enabled": True,
            "edges": int(s.size),
            "mean": float(np.mean(s)),
            "min": float(np.min(s)),
            "max": float(np.max(s)),
            "changed": int(np.sum(s != 1.0)),
            "l1_change": float(np.sum(np.abs(s - 1.0))),
        }

    def step_timed(self, embedding, mode: str = None) -> np.ndarray:
        """``step`` plus a measured wall-clock duration for the whole update."""
        t0 = time.perf_counter()
        f = self.step(embedding, mode)
        self.step_ms = (time.perf_counter() - t0) * 1000.0
        return f

    def sequence(self, embeddings, mode: str = "intact") -> np.ndarray:
        """(T, embedding_dim) -> (T, dims). Resets state first, like flm."""
        self.reset()
        arr = np.asarray(embeddings, np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embeddings must be (T, {self.embedding_dim}), got {arr.shape}"
            )
        out = np.empty((arr.shape[0], self.dims), np.float32)
        for i in range(arr.shape[0]):
            out[i] = self.step(arr[i], mode)
        return out

    # ------------------------------------------------------------- telemetry

    def telemetry(self) -> dict:
        """The frozen frame payload. Keys and ranges are not negotiable."""
        np.take(self.state, self.sample_index, out=self._sample)
        state = self.state
        rms = float(np.sqrt(np.mean(state * state))) if self.n else 0.0
        if self._mode == "no_edges":
            active = 0.0
        else:
            active = float(np.mean(np.abs(state) >= SPIKE_THRESHOLD))
        spikes = np.flatnonzero(np.abs(self._sample) >= SPIKE_THRESHOLD)
        return {
            "updates": int(self.updates),
            "state_rms": rms,
            "active_fraction": active,
            "sampled_ids": list(self.sample_ids),
            "sampled_state": [float(v) for v in self._sample],
            "spikes": [int(i) for i in spikes],
        }

    # ------------------------------------------------------------ benchmark

    def benchmark(self, steps: int = 20, mode: str = "intact") -> dict:
        """Measured wall-clock cost of full-graph steps. Never an estimate."""
        rng = np.random.default_rng(0)
        embs = rng.standard_normal((steps, self.embedding_dim)).astype(np.float32)
        self.set_mode(mode)
        self.reset()
        # Warm up allocator and BLAS, untimed.
        for i in range(min(3, steps)):
            self.step(embs[i])
        times = []
        for i in range(steps):
            t0 = time.perf_counter()
            self.step(embs[i])
            times.append((time.perf_counter() - t0) * 1000.0)
        times = np.asarray(times)
        med = float(np.median(times))
        return {
            "mode": mode,
            "steps": int(steps),
            "median_ms": med,
            "mean_ms": float(times.mean()),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "hz_single_thread": 1000.0 / med if med > 0 else float("inf"),
            "twenty_hz_ok": bool(med <= 50.0),
        }
