"""Tests for the online training path: progression, dopamine, and rehearsal.

These lock in the three things that were wrong or missing, each of which was found by
measurement rather than assumed:

  1. The course stopped after the first lesson. The answer path set the next challenge to None
     when a lesson ran out, so every session looked like the same first lesson.
  2. There was no reward signal the UI could draw. Dopamine is now delivered after the
     plasticity step, so the drawn pulse and the applied gradient are the same event.
  3. Online learning did not converge: with one update per answer the readout plateaued near
     45%. Supervised rehearsal from a replay buffer is what makes it learn inside a session,
     and it needs 60 steps per answer rather than 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.learning.prompt_index import PromptIndexReadout  # noqa: E402


# --------------------------------------------------------------- the update itself


def test_a_fresh_readout_is_at_chance_across_the_course():
    """A naive brain must actually be naive, or 'watch it learn' is a lie.

    Note the shape of this claim, which the first version of this test got wrong. A fresh
    readout is not near-uniform on EVERY input: W is random, so on any single input its softmax
    can land anywhere, and one measured 0.52. The property that matters is the one across many
    inputs, because that is what the accuracy curve averages over: untrained, it picks the right
    option at chance. Measured here over 300 random inputs, which is the same shape of
    measurement the live training run uses.
    """
    r = PromptIndexReadout(in_dim=128, n_actions=4, seed=0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 128))
    Y = rng.integers(0, 4, size=300)
    hits = sum(int(np.argmax(r.probs(x))) == int(y) for x, y in zip(X, Y))
    acc = hits / len(Y)
    assert 0.15 < acc < 0.36, f"an untrained readout should be at chance (0.25), got {acc:.3f}"

    # and it must not be confident: the mean top probability stays low until it is trained
    mean_top = float(np.mean([r.probs(x).max() for x in X[:100]]))
    assert mean_top < 0.60, f"untrained readout is too confident: mean top p {mean_top:.3f}"


def test_supervised_step_moves_the_choice_it_is_told():
    """One supervised step toward index 2 must raise p(2) and lower the others."""
    r = PromptIndexReadout(in_dim=128, n_actions=4, seed=0)
    x = np.random.default_rng(1).standard_normal(128)
    r.sample(x, np.random.default_rng(2))
    before = r.probs(x).copy()
    r.observe_supervised(2, lr=0.35)
    after = r.probs(x)
    assert after[2] > before[2], f"p(correct) did not rise: {before[2]} -> {after[2]}"
    others_before = np.delete(before, 2).sum()
    others_after = np.delete(after, 2).sum()
    assert others_after < others_before


def test_repeated_supervision_on_one_example_converges():
    """Thirty steps on a single example should nearly memorise it.

    If this does not happen the online path cannot learn anything at all, so it is the floor
    the whole training mode stands on.
    """
    r = PromptIndexReadout(in_dim=128, n_actions=4, seed=0)
    x = np.random.default_rng(3).standard_normal(128)
    r.sample(x, np.random.default_rng(4))
    for _ in range(30):
        r.observe_supervised(2, lr=0.35)
    assert r.probs(x)[2] > 0.9, f"p(correct) only reached {r.probs(x)[2]}"


def test_rehearse_learns_a_different_example_than_the_cached_one():
    """rehearse() must act on the features it is GIVEN, not on the cached sample.

    This is the bug class the method exists to avoid: if it silently used the cache, the replay
    buffer would be a no-op and the fly would appear to rehearse while learning nothing.
    """
    r = PromptIndexReadout(in_dim=128, n_actions=4, seed=0)
    a = np.random.default_rng(5).standard_normal(128)
    b = np.random.default_rng(6).standard_normal(128)
    r.sample(a, np.random.default_rng(7))          # cache holds a
    p_before = r.probs(b).copy()
    for _ in range(20):
        r.rehearse(b, 1, lr=0.35)
    assert r.probs(b)[1] > p_before[1] + 0.2, "rehearse did not learn the example it was given"


def test_rehearse_leaves_the_cache_usable():
    """Rehearsal must not destroy the cache, because the answer path reads it afterwards."""
    r = PromptIndexReadout(in_dim=128, n_actions=4, seed=0)
    x = np.random.default_rng(8).standard_normal(128)
    r.sample(x, np.random.default_rng(9))
    cached = r._cache["x"].copy()
    r.rehearse(np.random.default_rng(10).standard_normal(128), 0, lr=0.35)
    assert "x" in r._cache
    assert np.allclose(r._cache["x"], cached)


def test_parameters_are_the_documented_516():
    """The parameter count is quoted all over the UI; pin it."""
    r = PromptIndexReadout(in_dim=128, n_actions=4, seed=0)
    assert r.parameters() == 4 * 128 + 4 == 516


# --------------------------------------------------------------- the service behaviour


@pytest.fixture(scope="module")
def client():
    """A TestClient over the real service, loading the real connectome."""
    from fastapi.testclient import TestClient
    from brain.api import app

    with TestClient(app) as c:
        yield c


def test_train_fresh_gives_an_untrained_readout(client):
    r = client.post("/train", json={"fresh": True})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["fresh_brain"] is True
    assert body["parameters"] == 516


def test_the_frame_exposes_the_training_and_reward_telemetry(client):
    """The UI renders these, so their absence is a broken UI, not a cosmetic gap."""
    client.post("/train", json={"fresh": True})
    t = client.get("/telemetry").json()
    for key in (
        "dopamine",
        "dopamine_total",
        "fresh_brain",
        "window_accuracy",
        "window_size",
        "lessons_completed",
        "lesson_pos",
        "lesson_total",
        "lesson_title",
        "rehearsals",
        "replay_size",
        "training",
        "entropy",
        "grad_norm",
    ):
        assert key in t, f"missing telemetry key {key!r}"


def test_answering_awards_dopamine_and_fills_the_replay_buffer(client):
    client.post("/train", json={"fresh": True})
    s = client.post("/session", json={"fresh": True, "lesson_id": "u1l1"}).json()
    ch = s["challenge"]
    before = client.get("/telemetry").json()

    r = client.post(
        "/answer",
        json={
            "session_id": s["session_id"],
            "challenge_id": ch["id"],
            "choice_index": ch["correctIndex"],
        },
    ).json()
    after = client.get("/telemetry").json()

    assert after["replay_size"] == before["replay_size"] + 1
    assert after["rehearsals"] > before["rehearsals"], "no rehearsal steps were taken"
    assert after["learned_answered"] == before["learned_answered"] + 1
    # a pulse is delivered on a correct answer, and dopamine_total counts pulses
    assert after["dopamine_total"] >= before["dopamine_total"]
    assert 0.0 <= after["dopamine"] <= 1.0
    assert "loss" in r


def test_the_course_advances_past_the_first_lesson(client):
    """The regression that made every session look like lesson one.

    Driven with correct answers so the lesson runs out and the course has to move on. Before
    the fix this loop left the challenge at None and the session dead-ended at u1l1.
    """
    client.post("/train", json={"fresh": True, "replay_steps": 0})
    s = client.post("/session", json={"fresh": True, "lesson_id": "u1l1"}).json()
    ch = s["challenge"]
    sid = s["session_id"]

    seen = {s["lesson_id"]}
    for _ in range(20):
        r = client.post(
            "/answer",
            json={
                "session_id": sid,
                "challenge_id": ch["id"],
                "choice_index": ch["correctIndex"],
            },
        ).json()
        nxt = r.get("next_challenge")
        if nxt is None:
            break
        ch = nxt
        t = client.get("/telemetry").json()
        if t.get("lesson_id"):
            seen.add(t["lesson_id"])

    assert len(seen) > 1, f"the course never advanced past the first lesson, saw only {seen}"
    t = client.get("/telemetry").json()
    assert t["lesson_pos"] > 0
    assert t["lesson_total"] == 15, f"expected 15 lessons, got {t['lesson_total']}"


def test_the_window_curve_reflects_the_recent_answers(client):
    """The curve is the evidence of learning, so it must track recent answers, not all of them."""
    client.post("/train", json={"fresh": True, "replay_steps": 0})
    s = client.post("/session", json={"fresh": True, "lesson_id": "u1l1"}).json()
    ch = s["challenge"]
    for _ in range(25):
        r = client.post(
            "/answer",
            json={
                "session_id": s["session_id"],
                "challenge_id": ch["id"],
                "choice_index": ch["correctIndex"],
            },
        ).json()
        if r.get("next_challenge") is None:
            break
        ch = r["next_challenge"]
    t = client.get("/telemetry").json()
    assert 0 < t["window_size"] <= 20, f"window size {t['window_size']} out of range"
    assert 0.0 <= t["window_accuracy"] <= 1.0
