"""Training harness for the FlyLingo learning layer.

This module trains ``PolicyAdapter`` (see ``brain/learning/adapter.py``) on a
curriculum of Spanish multiple-choice challenges whose features come from a
reservoir over the frozen connectome. It exists to answer one question, and it
answers it by construction rather than by hand:

    does the measured connectome beat a parameter matched or shuffled graph
    of identical size, at matched parameters, on the same task?

Three arms always run, in the same process, over the same challenge order, with
the same adapter seed and the same reservoir seed:

  ``intact``        connectome in mode ``intact``, adapter with the dopamine
                    gated rule ON (``dopamine=True``).
  ``param_matched`` same connectome mode, adapter from ``clone(dopamine=False)``:
                    identical architecture, identical parameter count, identical
                    initialisation, dopamine gated rule OFF.
  ``shuffled``      same adapter as ``intact`` (fresh, same seed), but the
                    reservoir runs in ``shuffled`` (a fixed node relabeling of W
                    relative to the input/output interfaces). If the reservoir
                    refuses ``shuffled``, the arm falls back to ``random_graph``
                    and records which mode it actually used.

Extra diagnostic arms (``no_edges``, ``random_graph``) can be requested but are
never silently substituted for one of the three.

Honesty rules this file follows:
  * Every arm entry in the results JSON carries ``ran`` and, when false,
    ``skipped_reason``. Nothing is reported as run that did not run.
  * The reservoir source is recorded per run: ``is_standin`` plus a warning
    string. A result produced with the stand-in reservoir is never presented as
    a result about the connectome.
  * Challenges whose correct option falls outside the ``n_actions`` head are
    dropped and counted, not silently scored as wrong.

Reservoir interface used here (frozen in INTERFACES.md), and nothing else:

    .reset() -> None
    .step(embedding: (embedding_dim,) -> (dims,))
    .sequence(embeddings: (T, embedding_dim) -> (T, dims))
    .set_mode(mode: str) -> None
    .mode -> str
    .telemetry() -> dict

The harness never rebuilds the reservoir internals; if ``brain/reservoir.py``
exists it is used, otherwise a clearly labelled stand-in is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .adapter import PARAMETER_NAMES, PolicyAdapter

# --------------------------------------------------------------------- paths

# Repo root (D:/Projects/flylingo). api.py computes this as parents[2] from
# brain/brain/api.py; this file sits one level deeper (brain/brain/learning/
# train.py) so it is parents[3]. Getting this wrong silently writes the
# checkpoint where the service never looks, so test_learning.py asserts that
# DEFAULT_CHECKPOINT equals the path brain/api.py resolves.
ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / 'brain' / 'runs' / 'curriculum'
DEFAULT_CHECKPOINT = RUNS_DIR / 'adapter.npz'
DEFAULT_RESULTS = RUNS_DIR / 'results.json'
CURRICULUM_JSON = ROOT / 'brain' / 'brain' / 'curriculum' / 'es-en.json'
# The curriculum has already moved once, so accept either location.
CURRICULUM_CANDIDATES = (CURRICULUM_JSON, ROOT / 'brain' / 'curriculum' / 'es-en.json')

EMBED_DIM = 256
DIMS = 128
HIDDEN = 256
N_ACTIONS = 4
RESERVOIR_SEED = 7301
ADAPTER_SEED = 0

#: Text that must accompany any number produced with the stand-in reservoir.
STANDIN_WARNING = (
    'features came from the stand-in reservoir (StandInReservoir), NOT from the '
    'MaleCNS connectome. Numbers from this run say nothing about the real '
    'connectome; re-run with brain/reservoir.py present to measure that.'
)

MODE_FALLBACK = {'shuffled': 'random_graph'}


# ------------------------------------------------------------------- encoder


def encode_text(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic hashed char-ngram encoder, unit L2 norm.

    Same construction as ``brain/api.py:encode_text`` (orders 1..3, signed
    buckets, plus a fixed sinusoid so no input maps to the zero vector) but with
    a *stable* hash. ``api.py`` uses Python's builtin ``hash()`` on str, which is
    salted per process (PYTHONHASHSEED), so its embeddings are not reproducible
    across processes and a checkpoint trained against one salting would not match
    a service started with another. Recorded in the results JSON as
    ``encoder.hash`` so this is visible rather than assumed away.
    """
    v = np.zeros(dim, np.float32)
    t = ' ' + text.strip().lower() + ' '
    for n in (1, 2, 3):
        for i in range(max(0, len(t) - n + 1)):
            gram = t[i:i + n]
            h = int.from_bytes(
                hashlib.blake2b(gram.encode('utf-8'), digest_size=8).digest(), 'big')
            h &= 0xFFFFFFFF
            v[h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    v += np.sin(np.arange(dim, dtype=np.float32) * 0.017)
    nrm = float(np.linalg.norm(v)) + 1e-6
    return (v / nrm).astype(np.float32)


def encode_challenge(ch: dict, dim: int = EMBED_DIM) -> np.ndarray:
    parts = [ch['prompt'], ch.get('type', ''), 'en']
    parts += [str(o) for o in ch.get('options', [])]
    return encode_text(' | '.join(parts), dim)


# ---------------------------------------------------------------- stand-in


class StandInReservoir:
    """A labelled stand-in with the frozen reservoir interface, and no more.

    It is a plain seeded echo-state network, x_t = tanh(0.6 * W @ (x_{t-1} +
    ...) style, matching the recurrence shape documented in INTERFACES.md:

        x_t = tanh(W @ (0.6 * x_{t-1} + 0.4 * B @ e_t))
        f_t = unit_rms(P @ x_t)

    ``W`` here is a random sparse matrix, NOT Janelia MaleCNS. The class name,
    ``telemetry()['standin'] = True`` and the ``is_standin`` flag in the results
    JSON all say so. Modes:

      intact       W as built.
      shuffled     W relabelled by a fixed permutation of its nodes while B and P
                   are left alone, which is exactly what breaks the alignment
                   between graph and interfaces. Dynamics stay a relabeling of
                   themselves.
      random_graph Erdos-Renyi sparse matrix with the same nnz.
      no_edges     W identically zero, so features are exactly zero and
                   ``active_fraction`` is 0.0 with an empty spike list.
    """

    MODES = ('intact', 'shuffled', 'no_edges', 'random_graph')

    def __init__(self, embedding_dim: int = EMBED_DIM, dims: int = DIMS,
                 seed: int = RESERVOIR_SEED, n_nodes: int = 512, density: float = 0.05):
        self.embedding_dim = int(embedding_dim)
        self.dims = int(dims)
        self.seed = int(seed)
        self.n_nodes = int(n_nodes)
        self.density = float(density)
        rng = np.random.default_rng(self.seed)
        n = self.n_nodes
        nnz = max(n, int(round(self.density * n * n)))
        rows = rng.integers(0, n, nnz)
        cols = rng.integers(0, n, nnz)
        vals = rng.standard_normal(nnz).astype(np.float32)
        self.W = np.zeros((n, n), np.float32)
        np.add.at(self.W, (rows, cols), vals)
        # Row normalisation, the same orientation convention as the connectome.
        rowsum = np.abs(self.W).sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        self.W /= rowsum
        self.W_intact = self.W.copy()
        perm = rng.permutation(n)
        self.W_shuffled = self.W[perm][:, perm]
        rnd = np.zeros((n, n), np.float32)
        rr = rng.integers(0, n, nnz)
        cc = rng.integers(0, n, nnz)
        np.add.at(rnd, (rr, cc), rng.standard_normal(nnz).astype(np.float32))
        rs = np.abs(rnd).sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        self.W_random = rnd / rs
        self.B = (rng.standard_normal((n, self.embedding_dim)) / np.sqrt(self.embedding_dim)
                  ).astype(np.float32)
        self.B[rng.random((n, self.embedding_dim)) > 0.1] = 0.0
        self.P = (rng.standard_normal((self.dims, n)) / np.sqrt(n)).astype(np.float32)
        self._mode = 'intact'
        self.x = np.zeros(n, np.float32)
        self.updates = 0

    # -- mode ---------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f'unknown mode {mode!r}, expected one of {self.MODES}')
        self._mode = mode

    def _W(self):
        if self._mode == 'intact':
            return self.W_intact
        if self._mode == 'shuffled':
            return self.W_shuffled
        if self._mode == 'random_graph':
            return self.W_random
        if self._mode == 'no_edges':
            return None
        raise ValueError(f'unknown mode {self._mode!r}')

    # -- interface ----------------------------------------------------------

    def reset(self) -> None:
        self.x = np.zeros(self.n_nodes, np.float32)
        self.updates = 0

    def step(self, embedding: np.ndarray, mode: str | None = None) -> np.ndarray:
        if mode is not None and mode != self._mode:
            self.set_mode(mode)
        e = np.asarray(embedding, np.float32).ravel()
        if e.size != self.embedding_dim:
            raise ValueError(f'expected {self.embedding_dim} embedding dims, got {e.size}')
        W = self._W()
        if W is None:
            self.x = np.zeros(self.n_nodes, np.float32)
        else:
            drive = 0.6 * self.x + 0.4 * (self.B @ e)
            self.x = np.tanh(W @ drive)
        self.updates += 1
        f = self.P @ self.x
        return (f / (np.sqrt(np.mean(f * f)) + 1e-6)).astype(np.float32)

    def sequence(self, embeddings: np.ndarray, mode: str | None = None) -> np.ndarray:
        E = np.asarray(embeddings, np.float32)
        if E.ndim == 1:
            E = E[None, :]
        return np.stack([self.step(e, mode) for e in E], 0)

    def telemetry(self) -> dict:
        s = self.x[:512] if self.n_nodes >= 512 else np.pad(self.x, (0, 512 - self.n_nodes))
        s = np.clip(np.tanh(s), -1.0, 1.0).astype(np.float64)
        return {
            'updates': int(self.updates),
            'state_rms': float(np.sqrt(np.mean(self.x * self.x))) if self.n_nodes else 0.0,
            'active_fraction': float(np.mean(np.abs(s) >= 0.5)),
            'sampled_ids': [f'standin-{i}' for i in range(512)],
            'sampled_state': [float(v) for v in s],
            'spikes': [int(i) for i in np.flatnonzero(np.abs(s) >= 0.5)],
            'standin': True,
        }


# ------------------------------------------------------- reservoir loading


def _smoke_reservoir(res, embedding_dim: int, dims: int) -> None:
    """Fail loudly if an object does not honour the frozen feature interface."""
    if hasattr(res, 'reset'):
        res.reset()
    e = np.zeros(embedding_dim, np.float32)
    f = np.asarray(res.step(e), np.float32)
    if f.shape != (dims,):
        raise ValueError(f'step() returned shape {f.shape}, expected {(dims,)}')
    if not np.all(np.isfinite(f)):
        raise ValueError('step() returned non-finite features')
    seq = np.asarray(res.sequence(np.stack([e, e, e])), np.float32)
    if seq.shape != (3, dims):
        raise ValueError(f'sequence() returned shape {seq.shape}, expected {(3, dims)}')


def load_reservoir(embedding_dim: int = EMBED_DIM, dims: int = DIMS,
                   seed: int = RESERVOIR_SEED, force_standin: bool = False):
    """Return ``(reservoir, meta)``. Prefers ``brain.reservoir`` when it exists.

    ``meta`` records which source was used and, when the real module was present
    but unusable, why it was not used. A failure here never fabricates features:
    it falls back to the stand-in and says so.
    """
    meta = {
        'source': 'standin',
        'class': 'StandInReservoir',
        'module': None,
        'is_standin': True,
        'warning': STANDIN_WARNING,
        'embedding_dim': int(embedding_dim),
        'dims': int(dims),
        'seed': int(seed),
        'error': None,
    }
    if force_standin:
        meta['error'] = 'forced by --standin'
        return StandInReservoir(embedding_dim, dims, seed), meta

    try:
        from ..reservoir import FlyReservoir, load_connectome  # type: ignore
    except Exception as exc:  # module absent, or its deps missing
        meta['error'] = f'brain.reservoir unavailable: {exc!r}'
        return StandInReservoir(embedding_dim, dims, seed), meta

    try:
        try:
            conn = load_connectome()
        except TypeError:
            conn = load_connectome(ROOT / 'cache' / 'malecns_v1')
        res = FlyReservoir(conn, embedding_dim=embedding_dim, dims=dims, seed=seed)
        _smoke_reservoir(res, embedding_dim, dims)
    except Exception as exc:
        meta['error'] = f'brain.reservoir present but unusable: {exc!r}'
        return StandInReservoir(embedding_dim, dims, seed), meta

    meta.update(source='brain.reservoir.FlyReservoir', class_=None, module='brain.reservoir',
                is_standin=False, warning=None, error=None)
    meta['class'] = type(res).__name__
    meta.pop('class_', None)
    return res, meta


# ------------------------------------------------------------------- tasks


@dataclass
class TaskSet:
    """A curriculum split into a training stream and a held-out challenge set."""

    source: str                      # 'curriculum' | 'synthetic'
    path: str | None
    challenges: list
    train: list = field(default_factory=list)
    eval: list = field(default_factory=list)
    n_actions: int = N_ACTIONS
    dropped_no_room: int = 0         # fewer options than the head has actions
    dropped_unreachable: int = 0     # correct option index >= n_actions
    note: str = ''

    def summary(self) -> dict:
        return {
            'source': self.source,
            'path': self.path,
            'note': self.note,
            'n_challenges': len(self.challenges),
            'n_train': len(self.train),
            'n_eval': len(self.eval),
            'n_actions': int(self.n_actions),
            'dropped_no_room': int(self.dropped_no_room),
            'dropped_unreachable': int(self.dropped_unreachable),
            'difficulties': sorted({int(c['difficulty']) for c in self.challenges}),
        }


#: A small synthetic Spanish-like set, used only when the real curriculum file
#: is not on disk yet. Options and answers are real Spanish words; the mapping
#: from prompt to answer is what the adapter must learn.
_SYNTHETIC_PAIRS = [
    ('Hello', 'Hola'), ('Goodbye', 'Adios'), ('Thank you', 'Gracias'),
    ('Please', 'Por favor'), ('Yes', 'Si'), ('No', 'No'),
    ('Water', 'Agua'), ('Bread', 'Pan'), ('Milk', 'Leche'),
    ('Apple', 'Manzana'), ('Cat', 'Gato'), ('Dog', 'Perro'),
    ('House', 'Casa'), ('Book', 'Libro'), ('Friend', 'Amigo'),
    ('School', 'Escuela'), ('Teacher', 'Maestro'), ('Student', 'Estudiante'),
    ('Morning', 'Manana'), ('Night', 'Noche'), ('Today', 'Hoy'),
    ('Tomorrow', 'Manana'), ('Coffee', 'Cafe'), ('Tea', 'Te'),
    ('Rice', 'Arroz'), ('Chicken', 'Pollo'), ('Fish', 'Pescado'),
    ('Red', 'Rojo'), ('Blue', 'Azul'), ('Green', 'Verde'),
    ('Big', 'Grande'), ('Small', 'Pequeno'), ('Hot', 'Caliente'),
    ('Cold', 'Frio'), ('Mother', 'Madre'), ('Father', 'Padre'),
    ('Brother', 'Hermano'), ('Sister', 'Hermana'), ('City', 'Ciudad'),
    ('Street', 'Calle'), ('Money', 'Dinero'), ('Time', 'Tiempo'),
]

# Instances generated per (prompt, answer) pair. The correct slot cycles with the
# pair index, so the label distribution stays balanced and answer position is
# uninformative. One instance per pair left the eval split able to miss a class
# entirely, which made accuracy coarse and noisy.
_SYNTHETIC_REPEATS = 4


def synthetic_challenges() -> list:
    """Build a Spanish-like 4-option multiple-choice set, deterministically.

    Distractor sets are draws from the same answer pool, so no distractor is a
    semantic cue: the only way to score is to learn the prompt to answer mapping.
    """
    rng = np.random.default_rng(4242)
    answers = [a for _, a in _SYNTHETIC_PAIRS]
    out = []
    for i, (prompt, answer) in enumerate(_SYNTHETIC_PAIRS):
        # Several instances per pair, with the correct slot cycling 0..3, so the
        # answer position carries no information and the label distribution is
        # balanced. Without this the eval split can miss a class entirely and
        # accuracy is coarse and noisy.
        for k in range(_SYNTHETIC_REPEATS):
            pool = [a for a in answers if a != answer]
            rng.shuffle(pool)
            ci = (i + k) % 4
            opts = [None] * 4
            opts[ci] = answer
            fill = iter(pool[:3])
            for j in range(4):
                if j != ci:
                    opts[j] = next(fill)
            out.append({
                'id': f'syn{i + 1:03d}r{k + 1}',
                'type': 'translate',
                'prompt': prompt,
                'promptLang': 'en',
                'answer': answer,
                'options': opts,
                'correctIndex': int(ci),
                'audio': answer,
                'difficulty': int(1 + (i % 5)),
            })
    return out


def _validate(ch: dict) -> str | None:
    """Return a reason string if the challenge is unusable, else None."""
    for key in ('id', 'prompt', 'answer', 'options', 'correctIndex'):
        if key not in ch:
            return f'missing key {key}'
    opts = ch['options']
    if len(opts) < 3:
        return 'fewer than 3 options'
    ci = int(ch['correctIndex'])
    if not 0 <= ci < len(opts):
        return f'correctIndex {ci} out of range'
    if opts[ci] != ch['answer']:
        return 'correctIndex does not point at answer'
    return None


def load_task(path: Path | str | None = None, eval_fraction: float = 0.25,
              seed: int = 0, n_actions: int = N_ACTIONS) -> TaskSet:
    """Load the real curriculum if present, else the labelled synthetic set."""
    if path is None:
        path = next((p for p in CURRICULUM_CANDIDATES if p.exists()), CURRICULUM_JSON)
    path = Path(path)
    source, note, raw = 'synthetic', '', synthetic_challenges()
    if path.exists():
        raw_json = json.loads(path.read_text(encoding='utf-8'))
        chs = []
        for unit in raw_json.get('units', []):
            for lesson in unit.get('lessons', []):
                for ch in lesson.get('challenges', []):
                    chs.append(ch)
        if chs:
            source, raw = 'curriculum', chs
            note = 'challenges in file order per lesson'
        else:
            note = f'{path.name} had no challenges, used the synthetic set'
    else:
        note = (f'{path} does not exist yet, used the built-in synthetic Spanish-like set. '
                'These numbers are about the harness, not about the real curriculum.')

    bad = [(ch.get('id'), _validate(ch)) for ch in raw if _validate(ch)]
    good = [ch for ch in raw if not _validate(ch)]
    dropped_no_room = sum(1 for ch in good if len(ch['options']) < n_actions)
    usable = [ch for ch in good if len(ch['options']) >= n_actions]
    dropped_unreachable = sum(1 for ch in usable if int(ch['correctIndex']) >= n_actions)
    usable = [ch for ch in usable if int(ch['correctIndex']) < n_actions]
    if bad:
        note += f'; dropped {len(bad)} invalid challenge(s): {bad[:3]}'

    # Curriculum order: easy first, stable by id. Same order for every arm.
    usable = sorted(usable, key=lambda c: (int(c.get('difficulty', 1)), str(c['id'])))
    n_eval = max(1, int(round(eval_fraction * len(usable)))) if len(usable) > 1 else 0
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(usable))
    eval_idx = set(int(i) for i in idx[:n_eval])
    train = [usable[i] for i in range(len(usable)) if i not in eval_idx]
    evalset = [usable[i] for i in range(len(usable)) if i in eval_idx]
    evalset.sort(key=lambda c: str(c['id']))
    return TaskSet(source=source, path=str(path), challenges=usable, train=train,
                   eval=evalset, n_actions=n_actions, dropped_no_room=dropped_no_room,
                   dropped_unreachable=dropped_unreachable, note=note)


# -------------------------------------------------------------- feature run


def features_for(reservoir, mode: str, challenges, embedding_dim: int = EMBED_DIM,
                 reset: bool = True) -> np.ndarray:
    """Run a challenge list through the reservoir.

    Features are stateful across the list on purpose: this is the same call
    pattern ``brain/api.py`` uses (one ``.sequence([...])`` per challenge with the
    reservoir carrying state between answers), so the adapter trains on the same
    kind of feature stream it will see in the service.
    """
    if reset and hasattr(reservoir, 'reset'):
        reservoir.reset()
    if mode is not None and getattr(reservoir, 'mode', mode) != mode:
        reservoir.set_mode(mode)
    E = np.stack([encode_challenge(ch, embedding_dim) for ch in challenges]) if challenges \
        else np.zeros((0, embedding_dim), np.float32)
    if len(challenges) == 0:
        return np.zeros((0, DIMS), np.float32)
    return np.asarray(reservoir.sequence(E, mode), np.float32)


def _argmax_accuracy(probs: np.ndarray, correct: np.ndarray) -> float:
    if probs.size == 0:
        return float('nan')
    return float(np.mean(np.argmax(probs, axis=1) == correct))


# --------------------------------------------------------------------- arms


def make_adapters(n_actions: int = N_ACTIONS, in_dim: int = DIMS, hidden: int = HIDDEN,
                  seed: int = ADAPTER_SEED, lr: float = 0.05, gate_lr: float = 0.05,
                  entropy_coef: float = 0.01, gate_fraction: float = 0.05):
    """The three adapters, built so the comparison exists by construction.

    ``intact`` and ``param_matched`` share a seed, so the dopamine rule is the
    only difference between them. ``shuffled`` gets the same rule as ``intact``
    and a fresh adapter from the same seed.
    """
    common = dict(in_dim=in_dim, n_actions=n_actions, hidden=hidden, seed=seed,
                  lr=lr, entropy_coef=entropy_coef, gate_fraction=gate_fraction)
    intact = PolicyAdapter(**common, dopamine=True, gate_lr=gate_lr)
    matched = intact.clone(dopamine=False)
    shuffled = PolicyAdapter(**common, dopamine=True, gate_lr=gate_lr)
    return {'intact': intact, 'param_matched': matched, 'shuffled': shuffled}


ARM_SPECS = (
    # (arm name, reservoir mode, adapter variant)
    # param_matched uses the SAME reservoir and features as intact; the only
    # difference is that the dopamine-gated rule is switched off via clone().
    ('intact', 'intact', 'intact'),
    ('param_matched', 'intact', 'param_matched'),
    ('shuffled', 'shuffled', 'shuffled'),
)
EXTRA_ARM_SPECS = (
    # Diagnostics, off by default. Each gets its own fresh adapter so it cannot
    # inherit weights already trained by another arm.
    ('no_edges', 'no_edges', 'no_edges'),
    ('random_graph', 'random_graph', 'random_graph'),
)


def run_arm(*, name: str, adapter: PolicyAdapter, reservoir, mode: str,
            tasks: TaskSet, epochs: int, seed: int, eval_every: int = 1,
            embedding_dim: int = EMBED_DIM, shuffle_epochs: bool = False,
            feature_cache=None, on_epoch=None) -> dict:
    """Train one arm and return its curve plus honest bookkeeping."""
    out = {
        'arm': name,
        'ran': False,
        'skipped_reason': None,
        'reservoir_mode_requested': mode,
        'reservoir_mode_used': None,
        'params': None,
        'gated_params': None,
        'curve': [],
        'final': {},
        'features_sha256': None,
        'wall_s': 0.0,
    }
    t0 = time.time()
    mode_used = mode
    try:
        if getattr(reservoir, 'mode', mode) != mode:
            reservoir.set_mode(mode)
    except Exception as exc:
        fb = MODE_FALLBACK.get(mode)
        if fb is None:
            out['skipped_reason'] = f'reservoir refused mode {mode!r}: {exc!r}'
            out['wall_s'] = round(time.time() - t0, 3)
            return out
        try:
            reservoir.set_mode(fb)
            mode_used = fb
        except Exception as exc2:
            out['skipped_reason'] = (
                f'reservoir refused mode {mode!r} ({exc!r}) and fallback {fb!r} ({exc2!r})')
            out['wall_s'] = round(time.time() - t0, 3)
            return out
    out['reservoir_mode_used'] = mode_used
    if mode_used != mode:
        out['skipped_reason'] = None
        out['mode_fallback'] = f'{mode} -> {mode_used}'

    out['params'] = int(adapter.parameters())
    out['gated_params'] = int(adapter.gated_parameters())
    out['adapter'] = dict(adapter.config())

    eval_correct = np.array([int(c['correctIndex']) for c in tasks.eval], np.int64)
    rng = np.random.default_rng(int(seed))

    for epoch in range(1, int(epochs) + 1):
        order = list(tasks.train)
        if shuffle_epochs:
            # seeded per (seed, epoch) and identical for every arm
            rng_o = np.random.default_rng([int(seed), epoch, 7])
            perm = rng_o.permutation(len(order))
            order = [order[i] for i in perm]
        # The reservoir resets at the start of every .sequence() call and the
        # epoch order is deterministic given (seed, epoch), so features are a
        # pure function of (mode, seed, epoch, shuffle). Cache them. This is what
        # makes intact vs param_matched a genuine parameter match: they train on
        # byte-identical features rather than on two similar draws.
        cache_key = ('epoch', mode_used, int(seed), int(epoch), bool(shuffle_epochs))
        feats = None if feature_cache is None else feature_cache.get(cache_key)
        if feats is None:
            feats = features_for(reservoir, mode_used, order, embedding_dim, reset=True)
            if feature_cache is not None:
                feature_cache[cache_key] = feats
        if epoch == 1:
            out['features_sha256'] = hashlib.sha256(
                np.ascontiguousarray(feats, np.float32).tobytes()).hexdigest()[:16]
        n = len(order)
        correct_flags = np.zeros(n, np.float64)
        rewards = np.zeros(n, np.float64)
        losses = np.zeros(n, np.float64)
        entropies = np.zeros(n, np.float64)
        grad_norms = np.zeros(n, np.float64)
        gate_deltas = np.zeros(n, np.float64)
        for i, ch in enumerate(order):
            f = feats[i]
            action, logprob = adapter.sample(f, rng)
            reward = 1.0 if action == int(ch['correctIndex']) else -0.25
            info = adapter.observe(reward, logprob, f)
            correct_flags[i] = 1.0 if reward > 0 else 0.0
            rewards[i] = reward
            losses[i] = float(info.get('loss', 0.0))
            entropies[i] = float(info.get('entropy', 0.0))
            grad_norms[i] = float(info.get('grad_norm', 0.0))
            gate_deltas[i] = float(info.get('gated_weight_delta', 0.0))
        row = {
            'epoch': epoch,
            'train_accuracy': float(correct_flags.mean()) if n else float('nan'),
            'mean_reward': float(rewards.mean()) if n else float('nan'),
            'loss': float(losses.mean()) if n else float('nan'),
            'entropy': float(entropies.mean()) if n else float('nan'),
            'grad_norm': float(grad_norms.mean()) if n else float('nan'),
            'gated_weight_delta': float(gate_deltas.mean()) if n else float('nan'),
        }
        if epoch % max(1, int(eval_every)) == 0 or epoch == int(epochs):
            if len(tasks.eval):
                eval_key = ('eval', mode_used, int(seed))
                ef = None if feature_cache is None else feature_cache.get(eval_key)
                if ef is None:
                    ef = features_for(reservoir, mode_used, tasks.eval, embedding_dim,
                                      reset=True)
                    if feature_cache is not None:
                        feature_cache[eval_key] = ef
                probs = np.stack([adapter.probs(f) for f in ef])
                row['eval_accuracy'] = _argmax_accuracy(probs, eval_correct)
            else:
                row['eval_accuracy'] = float('nan')
        out['curve'].append(row)
        if on_epoch is not None:
            on_epoch(name, row)
    last = out['curve'][-1] if out['curve'] else {}
    best = max((r.get('eval_accuracy', float('nan')) for r in out['curve']
                if r.get('eval_accuracy') is not None), default=float('nan'))
    out['final'] = {
        'epoch': last.get('epoch'),
        'eval_accuracy': last.get('eval_accuracy'),
        'best_eval_accuracy': float(best),
        'train_accuracy': last.get('train_accuracy'),
        'mean_reward': last.get('mean_reward'),
        'entropy': last.get('entropy'),
        'steps': int(getattr(adapter, 'steps', 0)),
        'gated_steps': int(getattr(adapter, 'gated_steps', 0)),
    }
    out['ran'] = True
    out['wall_s'] = round(time.time() - t0, 3)
    return out


# -------------------------------------------------------------------- train


def train(*, epochs: int = 30, arms: tuple = ('intact', 'param_matched', 'shuffled'),
          extra_arms: tuple = (), curriculum: Path | str | None = None,
          results_path: Path | str | None = DEFAULT_RESULTS,
          checkpoint_path: Path | str | None = DEFAULT_CHECKPOINT,
          eval_fraction: float = 0.25, adapter_seed: int = ADAPTER_SEED,
          reservoir_seed: int = RESERVOIR_SEED, dims: int = DIMS,
          embedding_dim: int = EMBED_DIM, hidden: int = HIDDEN, n_actions: int = N_ACTIONS,
          lr: float = 0.05, gate_lr: float = 0.05, entropy_coef: float = 0.01,
          gate_fraction: float = 0.05,
          shuffle_epochs: bool = False, force_standin: bool = False,
          save_checkpoints: bool = True, verbose: bool = True) -> dict:
    """Train every requested arm and write the results JSON. Returns the results."""
    tasks = load_task(curriculum, eval_fraction=eval_fraction, seed=adapter_seed,
                      n_actions=n_actions)
    reservoir, rmeta = load_reservoir(embedding_dim=embedding_dim, dims=dims,
                                      seed=reservoir_seed, force_standin=force_standin)
    if verbose:
        print(f'task: {tasks.source} n_train={len(tasks.train)} n_eval={len(tasks.eval)} '
              f'note={tasks.note}')
        print(f'reservoir: {rmeta["source"]} standin={rmeta["is_standin"]} '
              f'err={rmeta["error"]}')

    adapters = make_adapters(n_actions=n_actions, in_dim=dims, hidden=hidden,
                             seed=adapter_seed, lr=lr, gate_lr=gate_lr,
                             entropy_coef=entropy_coef, gate_fraction=gate_fraction)
    specs = [s for s in ARM_SPECS if s[0] in arms]
    missing = [a for a in arms if a not in {s[0] for s in ARM_SPECS}]
    specs += [s for s in EXTRA_ARM_SPECS if s[0] in extra_arms]
    # A diagnostic arm needs its own fresh adapter. Reusing another arm's adapter
    # would hand the diagnostic a readout already trained on the task (and would
    # break the parameter count match), which is why "_which" is now a distinct
    # variant name per diagnostic instead of pointing at "intact".
    for _n, _mode, _which in specs:
        if _which not in adapters:
            adapters[_which] = PolicyAdapter(
                in_dim=dims, n_actions=n_actions, hidden=hidden, seed=adapter_seed,
                lr=lr, entropy_coef=entropy_coef, gate_fraction=gate_fraction,
                dopamine=True, gate_lr=gate_lr)
    feature_cache = {}
    if missing:
        raise ValueError(f'unknown arms {missing}; known: '
                         f'{[s[0] for s in ARM_SPECS]} extras: {[s[0] for s in EXTRA_ARM_SPECS]}')

    results = {
        'kind': 'flylingo_learning_curve',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'host_note': 'Windows, bash/MSYS, venv at brain/.venv',
        'reservoir': rmeta,
        'task': tasks.summary(),
        'encoder': {
            'kind': 'blake2b hashed char ngram 1..3, signed buckets, unit L2 norm',
            'embedding_dim': int(embedding_dim),
            'hash': 'blake2b-64 stable; brain/api.py encode_text uses builtin hash() '
                    '(per process salted), recorded here because it matters for reuse',
        },
        'config': {
            'epochs': int(epochs),
            'eval_fraction': float(eval_fraction),
            'adapter_seed': int(adapter_seed),
            'reservoir_seed': int(reservoir_seed),
            'dims': int(dims),
            'hidden': int(hidden),
            'n_actions': int(n_actions),
            'lr': float(lr),
            'gate_lr': float(gate_lr),
            'entropy_coef': float(entropy_coef),
            'shuffle_epochs': bool(shuffle_epochs),
            'reward_correct': 1.0,
            'reward_wrong': -0.25,
            'order': 'train stream sorted by difficulty then id, identical for all arms',
        },
        'arms': {},
        'comparison': {},
        'notes': [],
    }
    if rmeta['is_standin']:
        results['notes'].append(STANDIN_WARNING)
    if tasks.source == 'synthetic':
        results['notes'].append(tasks.note)
    if not all(s[0] in arms for s in ARM_SPECS):
        results['notes'].append(
            'the required three arms were reduced by the caller; a comparison against a '
            'missing arm is not meaningful')

    def on_epoch(name, row):
        if verbose:
            ev = row.get('eval_accuracy')
            ev_s = 'n/a' if ev is None else f'{ev:.3f}'
            print(f'  {name:<14} epoch {row["epoch"]:>3} train {row["train_accuracy"]:.3f} '
                  f'eval {ev_s} reward {row["mean_reward"]:+.3f} H {row["entropy"]:.3f}')

    # Same-init proof, captured HERE and not after the loop: the field means "did
    # this arm's adapter start from the intact arm's weights?", which is what makes
    # the control parameter-matched. The old version compared the two adapters
    # after both had trained, so it measured the training history instead and
    # reported False for every non-zero learning rate.
    def _same_init_as_intact(adapter) -> bool:
        return bool(all(np.array_equal(getattr(adapters['intact'], n), getattr(adapter, n))
                        for n in PARAMETER_NAMES))

    same_init = {name: _same_init_as_intact(adapters[which]) for name, _m, which in specs}

    for name, mode, which in specs:
        results['arms'][name] = run_arm(
            name=name, adapter=adapters[which], reservoir=reservoir, mode=mode,
            tasks=tasks, epochs=epochs, seed=adapter_seed + 1000, eval_every=1,
            embedding_dim=embedding_dim, shuffle_epochs=shuffle_epochs,
            feature_cache=feature_cache, on_epoch=on_epoch)
        results['arms'][name]['adapter_variant'] = which
        results['arms'][name]['same_init_as_intact'] = same_init[name]

    # ---------------------------------------------------------- comparison
    def acc(arm, key='eval_accuracy', best=False):
        a = results['arms'].get(arm)
        if not a or not a['ran'] or not a['curve']:
            return None
        if best:
            return a['final'].get('best_eval_accuracy')
        return a['curve'][-1].get(key)

    cmp_out = {}
    fh = {n: a.get('features_sha256') for n, a in results['arms'].items()}
    cmp_out['features_sha256'] = fh
    # Proof rather than intent: the two arms that are supposed to differ only in
    # the dopamine-gated rule must have consumed byte-identical reservoir
    # features. If this is False the parameter match is not clean.
    cmp_out['intact_and_param_matched_share_features'] = bool(
        fh.get('intact') is not None and fh.get('intact') == fh.get('param_matched'))
    base = acc('intact')
    for other in ('param_matched', 'shuffled'):
        o = acc(other)
        if base is None or o is None:
            cmp_out[f'intact_minus_{other}'] = None
            cmp_out[f'{other}_ran'] = bool(results['arms'].get(other, {}).get('ran', False))
        else:
            cmp_out[f'intact_minus_{other}'] = float(base - o)
    cmp_out['intact_final_eval_accuracy'] = base
    cmp_out['param_matched_final_eval_accuracy'] = acc('param_matched')
    cmp_out['shuffled_final_eval_accuracy'] = acc('shuffled')
    cmp_out['chance_eval_accuracy'] = float(1.0 / n_actions)
    if base is not None and cmp_out.get('intact_minus_param_matched') is not None:
        d_pm = cmp_out['intact_minus_param_matched']
        d_sh = cmp_out.get('intact_minus_shuffled')
        if d_pm > 0 and (d_sh is None or d_sh > 0):
            verdict = ('intact connectome arm beat every control that ran on final held-out '
                       'accuracy')
        elif d_pm == 0 and (d_sh is None or d_sh == 0):
            verdict = 'intact arm tied every control that ran; no measurable advantage'
        else:
            verdict = ('intact arm did NOT beat every control that ran on final held-out '
                       'accuracy; report this as a null result')
        cmp_out['verdict'] = verdict
    else:
        cmp_out['verdict'] = 'comparison not available: an arm did not run'
    results['comparison'] = cmp_out

    if verbose:
        print('comparison:', json.dumps(cmp_out, indent=2))

    # ---------------------------------------------------------- checkpoint
    if save_checkpoints:
        runs = Path(checkpoint_path).parent if checkpoint_path else RUNS_DIR
        runs.mkdir(parents=True, exist_ok=True)
        saved = {}
        if checkpoint_path and results['arms'].get('intact', {}).get('ran'):
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            adapters['intact'].save(checkpoint_path)
            saved['intact'] = str(checkpoint_path)
        for name, which in (('param_matched', 'param_matched'), ('shuffled', 'shuffled')):
            if results['arms'].get(name, {}).get('ran'):
                p = runs / f'adapter_{name}.npz'
                adapters[which].save(p)
                saved[name] = str(p)
        results['checkpoints'] = saved

    if results_path:
        rp = Path(results_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(results, indent=2), encoding='utf-8')
        if verbose:
            print(f'wrote {rp}')
    results['results_path'] = str(results_path) if results_path else None
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='Train the FlyLingo readout on the Spanish curriculum and compare '
                    'the intact connectome arm against parameter matched and shuffled arms.')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--curriculum', type=str, default=None,
                   help='path to curriculum JSON (default brain/curriculum/es-en.json)')
    p.add_argument('--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT))
    p.add_argument('--results', type=str, default=str(DEFAULT_RESULTS))
    p.add_argument('--extra-arms', type=str, default='',
                   help='comma list from no_edges,random_graph')
    p.add_argument('--eval-fraction', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=ADAPTER_SEED)
    p.add_argument('--standin', action='store_true',
                   help='force the stand-in reservoir even if brain/reservoir.py exists')
    p.add_argument('--no-save', action='store_true')
    args = p.parse_args(argv)
    extra = tuple(a for a in (args.extra_arms or '').split(',') if a)
    train(epochs=args.epochs, curriculum=args.curriculum, results_path=args.results,
          checkpoint_path=args.checkpoint, extra_arms=extra,
          eval_fraction=args.eval_fraction, adapter_seed=args.seed,
          force_standin=args.standin, save_checkpoints=not args.no_save)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
