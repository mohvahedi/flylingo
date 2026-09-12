"""Deterministic text encoders for the connectome reservoir.

Why this module exists
----------------------
The first version of the encoder used Python's builtin ``hash()``. That function is
salted per process for strings (PYTHONHASHSEED), so the same text encoded differently
in every new interpreter. Measured: the same input gave checksums 3.857, 5.022 and
3.440 across three consecutive processes.

The consequence was severe. Training encoded the curriculum in one process and the
serving API encoded it in another, so a checkpoint fit on one set of features was
applied to a different set at run time. The model was being asked to answer questions
in an encoding it had never seen. Every encoder here is therefore built on a stable
hash and the stability is asserted in tests.

The encoders are deliberately training-free and carry no pretrained knowledge: they
cannot smuggle in task information from a language model, so any learning measured on
top of them is the readout learning rather than a language model answering for it.

Design note on the pair encoder
-------------------------------
A single vector describing "this prompt with these four options" asks the readout to
memorise arbitrary index-to-answer associations, which is a poor fit for fixed random
wiring. ``encode_option_pair`` instead scores one option at a time against the prompt,
which turns the task into matching rather than index lookup.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

EMBED_DIM = 256

#: Parameterless sinusoidal version of the hashing trick. Deterministic by construction.
_PHASE = np.arange(EMBED_DIM, dtype=np.float32) * 0.017


def stable_hash64(token: str) -> int:
    """A 64-bit hash that does not vary between processes.

    blake2b is used rather than crc32 because collisions between neighbouring n-grams
    should be as rare as reasonably possible in a 256-wide space, and rather than
    Python's ``hash`` because that is salted per process.
    """
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def _sign_and_index(token: str, dim: int, salt: int = 0) -> tuple[float, int]:
    h = stable_hash64(token if salt == 0 else f"{salt}\x00{token}")
    idx = h % dim
    sign = 1.0 if (h >> 63) & 1 else -1.0
    return sign, idx


def encode_text(text: str, dim: int = EMBED_DIM, seed: int = 0) -> np.ndarray:
    """Hashed character n-gram bag, signed, unit norm. Deterministic everywhere."""
    v = np.zeros(dim, np.float32)
    t = " " + text.strip().lower() + " "
    for n in (1, 2, 3):
        for i in range(max(0, len(t) - n + 1)):
            sign, idx = _sign_and_index(t[i : i + n], dim, salt=seed)
            v[idx] += sign
    v += np.sin(_PHASE + float(seed))
    nrm = float(np.linalg.norm(v)) + 1e-6
    return (v / nrm).astype(np.float32)


def encode_option_pair(prompt: str, option: str, dim: int = EMBED_DIM, seed: int = 0) -> np.ndarray:
    """Encode one (prompt, option) candidate as a single vector.

    The prompt and the option are hashed into separate halves of the space and the
    cross term is included, so a readout can learn "this option fits this prompt"
    without having to memorise a position-dependent label.
    """
    half = dim // 2
    v = np.zeros(dim, np.float32)
    p = " " + prompt.strip().lower() + " "
    o = " " + option.strip().lower() + " "

    # Prompt side, option side, and the union, each in its own index range so a
    # coincidence in one does not cancel the others.
    for n in (1, 2, 3):
        for i in range(max(0, len(p) - n + 1)):
            sign, idx = _sign_and_index(p[i : i + n], half, salt=seed + 1)
            v[idx] += sign
        for i in range(max(0, len(o) - n + 1)):
            sign, idx = _sign_and_index(o[i : i + n], half, salt=seed + 2)
            v[half + idx] += sign
    for n in (1, 2):
        text = f"{p.strip()}|{o.strip()}"
        for i in range(max(0, len(text) - n + 1)):
            sign, idx = _sign_and_index(text[i : i + n], half, salt=seed + 3)
            v[half + idx] += 0.5 * sign

    v += np.sin(_PHASE + float(seed))
    nrm = float(np.linalg.norm(v)) + 1e-6
    return (v / nrm).astype(np.float32)


def encode_challenge(ch: dict, dim: int = EMBED_DIM, seed: int = 0) -> np.ndarray:
    """One vector for a whole challenge, options included.

    Kept because the streaming path and the existing harnesses use it. For learning,
    prefer ``encode_option_pairs``: see the module docstring.
    """
    parts = [ch["prompt"], ch.get("type", ""), "en"]
    parts += [str(o) for o in ch.get("options", [])]
    return encode_text(" | ".join(parts), dim=dim, seed=seed)


def encode_option_pairs(
    ch: dict, dim: int = EMBED_DIM, seed: int = 0, center: bool = True
) -> np.ndarray:
    """Encode every option of a challenge against its prompt.

    Returns (n_options, dim), one row per candidate, in the order the options are
    presented. This is what the option-scoring readout consumes.

    ``center`` subtracts the per-challenge mean across options. Measured before centering,
    the four candidates sat at cosine 0.92 to 0.97 of each other because the shared prompt
    dominated the vector. That shared component is identical for every option, so it
    contributes a constant to each score and cancels in the softmax, but it also drives all
    four candidates into the same operating region of the reservoir's tanh, where the
    differences that actually decide the answer get squashed. Removing it leaves only the
    part that distinguishes the options, which is the part the readout has to learn from.
    """
    prompt = ch["prompt"]
    opts = [str(o) for o in ch.get("options", [])]
    if not opts:
        return np.zeros((1, dim), np.float32)
    raw = np.stack([encode_option_pair(prompt, o, dim=dim, seed=seed) for o in opts])
    if center and len(raw) > 1:
        raw = raw - raw.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        raw = raw / np.maximum(norms, 1e-6)
    return raw.astype(np.float32)


def encoder_fingerprint() -> str:
    """A short stable identity for the encoding scheme.

    Checkpoints record this so a model trained under one encoder can never be silently
    loaded under another. That is exactly the failure that happened here: the first
    encoder used Python's builtin hash(), which is salted per process, so a checkpoint fit
    during training was applied at run time to features the server never reproduced. The
    model looked trained and its probabilities were meaningless.

    Any change to how text becomes a vector must change this string, which is why the
    scheme name and the salient constants are hashed together rather than a version
    number being maintained by hand.
    """
    scheme = "|".join(
        [
            "hashed-char-ngram-signed-v2",
            "blake2b8",
            f"dim={EMBED_DIM}",
            "ngrams=1,2,3",
            "pair=split-half+cross",
            "pair_center=mean-subtract+unit-norm",
            f"idle_mix={IDLE_MIX}",
        ]
    )
    return hashlib.blake2b(scheme.encode("utf-8"), digest_size=8).hexdigest()


def encode_idle(tick: int, dim: int = EMBED_DIM) -> np.ndarray:
    """Slowly varying drive, so the connectome keeps moving between answers.

    Feeding the reservoir one constant embedding converges it to a fixed point within
    about a dozen steps, which freezes the live view. This is an input signal chosen so
    the visualization stays alive. It is not evidence of the fly attending to anything
    and it carries no task information.
    """
    i = np.arange(dim, dtype=np.float32)
    v = (
        np.sin(0.013 * float(tick) + i * 0.11)
        + 0.5 * np.sin(0.0071 * float(tick) + i * 0.037)
        + 0.25 * np.sin(0.0031 * float(tick) + i * 0.211)
    ).astype(np.float32)
    nrm = float(np.linalg.norm(v)) + 1e-6
    return (v / nrm).astype(np.float32)


#: Strength of the continuously varying drive component, relative to the challenge text.
#: See encode_idle for why the drifting component exists.
IDLE_MIX = 0.25


def tick_embedding(
    challenge: dict, tick: int, mix: float = IDLE_MIX, dim: int = EMBED_DIM
) -> np.ndarray:
    """Drive for one background step: the challenge, plus a drifting component."""
    base = encode_challenge(challenge, dim=dim)
    mixed = (1.0 - mix) * base + mix * encode_idle(tick, dim=dim)
    nrm = float(np.linalg.norm(mixed)) + 1e-6
    return (mixed / nrm).astype(np.float32)
