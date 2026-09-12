"""Regression tests for the text encoders.

The bug these exist to prevent: the first encoder used Python's builtin ``hash()``,
which is salted per process (PYTHONHASHSEED). The same text produced different vectors
in different interpreters, so a checkpoint fit during training was applied at serving
time to features it had never seen. Stability across processes is therefore a
correctness property, not a nicety, and it is asserted here by actually spawning
separate interpreters.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BRAIN = Path(__file__).resolve().parents[1]

from brain.encoders import (  # noqa: E402
    EMBED_DIM,
    encode_challenge,
    encode_idle,
    encode_option_pair,
    encode_option_pairs,
    encode_text,
    stable_hash64,
    tick_embedding,
)


def _run_in_fresh_process(expr: str) -> str:
    """Evaluate an expression in a brand new interpreter and return its stdout."""
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import numpy as np\n"
        "from brain.encoders import *\n"
        "print(%s)\n" % (BRAIN, expr)
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    return out.stdout.strip()


def test_stable_hash_is_process_independent():
    """The hash must not depend on PYTHONHASHSEED."""
    text = "How do you say 'Hello' in Spanish?"
    first = _run_in_fresh_process(f"stable_hash64({text!r})")
    for _ in range(2):
        assert _run_in_fresh_process(f"stable_hash64({text!r})") == first
    # And it must agree with the value computed in this process.
    assert str(stable_hash64(text)) == first


def test_encode_text_is_identical_across_processes():
    """The exact failure that made the old checkpoint unusable."""
    expr = "round(float(encode_text(\"How do you say 'Hello' in Spanish?\").sum()), 10)"
    sums = {_run_in_fresh_process(expr) for _ in range(3)}
    assert len(sums) == 1, f"encode_text differed across processes: {sums}"


def test_encode_option_pairs_identical_across_processes():
    ch = {
        "prompt": "How do you say 'Hello' in Spanish?",
        "options": ["Buenas tardes", "Hola", "Buenas noches", "Buenos días"],
    }
    expr = (
        "round(float(np.abs(encode_option_pairs(%r)[0]).sum()), 10)" % (ch,)
    )
    sums = {_run_in_fresh_process(expr) for _ in range(3)}
    assert len(sums) == 1, f"encode_option_pairs differed across processes: {sums}"


def test_encode_text_shape_and_norm():
    v = encode_text("Hola")
    assert v.shape == (EMBED_DIM,)
    assert v.dtype == np.float32
    assert np.isfinite(v).all()
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4


def test_encoding_is_case_and_whitespace_insensitive():
    a = encode_text("How do you say Hello in Spanish?")
    b = encode_text("  how do you say hello in spanish?  ")
    assert np.allclose(a, b), "encoder should normalise case and surrounding space"


def test_different_texts_encode_differently():
    a = encode_text("Hola")
    b = encode_text("Adiós")
    assert not np.allclose(a, b)
    assert float(a @ b) < 0.95


class TestOptionPairs:
    ch = {
        "prompt": "How do you say 'Hello' in Spanish?",
        "options": ["Buenas tardes", "Hola", "Buenas noches", "Buenos días"],
    }

    def test_shape_matches_option_count(self):
        E = encode_option_pairs(self.ch)
        assert E.shape == (4, EMBED_DIM)
        assert E.dtype == np.float32
        assert np.isfinite(E).all()

    def test_centering_separates_the_candidates(self):
        """Before centering the shared prompt dominated and candidates were near-identical."""
        def mean_pairwise_cos(E):
            cs = []
            for i in range(len(E)):
                for j in range(i + 1, len(E)):
                    a, b = E[i], E[j]
                    cs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
            return float(np.mean(cs))

        raw = mean_pairwise_cos(encode_option_pairs(self.ch, center=False))
        centered = mean_pairwise_cos(encode_option_pairs(self.ch, center=True))
        assert raw > 0.7, f"expected the uncentered candidates to be similar, got {raw}"
        assert centered < raw - 0.3, (
            f"centering should separate the candidates: raw {raw}, centered {centered}"
        )

    def test_centering_shrinks_the_shared_component(self):
        """The invariant centering is for: the part common to all options goes away.

        Note the row mean cannot be exactly zero, because each row is renormalised to unit
        length after centering and the per-row norms differ. So the property tested is that
        the mean across options collapses from "nearly a unit vector" (all options almost
        identical) to a small residual.
        """
        def mean_norm(E):
            return float(np.linalg.norm(E.mean(axis=0)))

        raw = mean_norm(encode_option_pairs(self.ch, center=False))
        centered = mean_norm(encode_option_pairs(self.ch, center=True))
        assert raw > 0.8, f"uncentered options should share a large component, got {raw}"
        assert centered < 0.3, f"centering should shrink the shared component, got {centered}"
        assert centered < raw / 3.0

    def test_rows_are_unit_norm(self):
        E = encode_option_pairs(self.ch, center=True)
        norms = np.linalg.norm(E, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), f"row norms {norms}"

    def test_single_option_does_not_divide_by_zero(self):
        out = encode_option_pairs({"prompt": "x", "options": ["only"]})
        assert out.shape == (1, EMBED_DIM)
        assert np.isfinite(out).all()

    def test_no_options_is_handled(self):
        out = encode_option_pairs({"prompt": "x", "options": []})
        assert out.shape == (1, EMBED_DIM)
        assert np.isfinite(out).all()

    def test_option_order_is_preserved(self):
        """Row i must correspond to options[i]; the readout relies on this."""
        E = encode_option_pairs(self.ch)
        for i, opt in enumerate(self.ch["options"]):
            single = encode_option_pair(self.ch["prompt"], opt)
            # The centered row is the raw pair minus the mean, so correlate rather than
            # compare directly.
            r = float(np.corrcoef(E[i], single)[0, 1])
            assert r > 0.0, f"row {i} should reflect option {opt!r}, correlation {r}"


def test_idle_drive_keeps_moving():
    """A constant drive converges the reservoir to a fixed point; this must not."""
    a = encode_idle(0)
    b = encode_idle(1)
    c = encode_idle(500)
    assert not np.allclose(a, b)
    assert not np.allclose(a, c)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-4


def test_tick_embedding_carries_the_challenge():
    ch = {"prompt": "How do you say 'water' in Spanish?", "options": ["el agua", "el pan"]}
    e = tick_embedding(ch, 0)
    assert e.shape == (EMBED_DIM,)
    assert abs(float(np.linalg.norm(e)) - 1.0) < 1e-4
    # It must differ from pure idle, i.e. the challenge is actually in there.
    assert not np.allclose(e, encode_idle(0))
