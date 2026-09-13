"""FlyLingo service: the frozen HTTP + WebSocket interface from INTERFACES.md.

Wires the frozen connectome (brain.reservoir), the trainable readout
(brain.learning.PolicyAdapter) and the Spanish curriculum (brain.curriculum) into the
single event stream the two visualization harnesses render.

Honesty rules enforced here, not just documented:
  - the mode actually in force is reported in every frame, so a demo can never be
    mistaken for the intact connectome when it is not;
  - control accuracies are only populated once those controls have actually been
    evaluated; they are never invented;
  - a feature vector of exact zeros is treated as a disconnected graph and reported as
    such rather than silently scored.
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .curriculum import load_curriculum
from .learning.adapter import PolicyAdapter
from .reservoir import FlyReservoir, load_connectome

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "brain" / "runs" / "curriculum" / "adapter.npz"
#: The option-scoring checkpoint, produced once the matching architecture is trained.
CHECKPOINT_OPTIONS = ROOT / "brain" / "runs" / "optionscorer" / "adapter.npz"
#: The prompt-index checkpoint. This is the architecture that actually learns the
#: curriculum: reservoir features of the prompt, mapped to the answer index. See
#: brain/learning/prompt_index.py for why, and for the honest attribution caveat.
CHECKPOINT_PROMPT_INDEX = ROOT / "brain" / "runs" / "promptindex" / "adapter.npz"

EMBED_DIM = 256
DIMS = 128
N_ACTIONS = 4
STREAM_HZ = 20

# The connectome keeps running between answers.
#
# A measured full-graph step costs about 24.5 ms, so ticking at 10 Hz uses roughly a
# quarter of one core. Without this the reservoir only advanced when a challenge was
# answered, which made the "live" brain view a frozen picture between questions: the
# activity is the whole point of the visualization, so it has to actually move.
TICK_HZ = 10.0
TICK_INTERVAL = 1.0 / TICK_HZ

app = FastAPI(title="FlyLingo brain", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: dict = {}


# --------------------------------------------------------------------------- state


class Session(BaseModel):
    id: str
    lesson_id: str
    index: int = 0
    streak: int = 0
    hearts: int = 5
    xp: int = 0
    answered: int = 0
    correct: int = 0
    #: which lesson within the course this session is on, and how many lessons exist. The
    #: demo used to stop dead at the end of the first lesson because nothing advanced past it.
    lesson_pos: int = 0
    lessons_completed: int = 0


def _load_readout(candidate: Path):
    """Load whichever readout class a checkpoint declares, or None if it is unusable.

    Each class shares the same method surface (logits, probs, sample, observe,
    parameters, gated_parameters, save, load), so the service can treat them alike.
    """
    from .learning.prompt_index import PromptIndexReadout

    try:
        with np.load(candidate, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    except Exception as exc:
        return None, f"skipped: {type(exc).__name__}: {exc}"

    recorded = meta.get("encoder_fingerprint")
    if recorded != ENCODER_FINGERPRINT:
        return None, (
            f"skipped: trained under encoder {recorded!r}, current is {ENCODER_FINGERPRINT!r}"
        )

    kind = meta.get("kind", "policy_adapter")
    try:
        # A prompt-index readout. Both spellings are accepted because the saver writes the
        # descriptive "prompt_index_readout" while the service key is the shorter
        # "prompt_index"; a mismatch here silently fell through to PolicyAdapter and failed
        # on a missing W1, which looked like a corrupt checkpoint rather than a name bug.
        if kind in ("prompt_index", "prompt_index_readout"):
            in_dim = int(meta.get("in_dim", DIMS))
            n_actions = int(meta.get("n_actions", N_ACTIONS))
            obj = PromptIndexReadout(in_dim=in_dim, n_actions=n_actions)
            obj.load(candidate)
            return (obj, "prompt_index"), "loaded"
        obj = PolicyAdapter(in_dim=DIMS, n_actions=N_ACTIONS, hidden=256, seed=0)
        obj.load(candidate)
    except Exception as exc:
        return None, f"skipped: {type(exc).__name__}: {exc}"
    return (obj, kind), "loaded"


def _boot() -> None:
    conn = load_connectome()
    reservoir = FlyReservoir(conn, embedding_dim=EMBED_DIM, dims=DIMS, seed=7301)
    adapter = PolicyAdapter(in_dim=DIMS, n_actions=N_ACTIONS, hidden=256, seed=0)

    # A checkpoint is only loaded if it was trained under the CURRENT encoder.
    #
    # This guard exists because it would have caught a real failure: the first encoder was
    # built on Python's salted builtin hash(), so a checkpoint fit during training was
    # applied at run time to features the server never reproduced. The model appeared
    # trained and its probabilities were arbitrary. Loading a stale checkpoint silently is
    # worse than loading none, so a mismatch is reported rather than ignored.
    #
    # Order matters: the prompt-index readout is the architecture measured to actually
    # learn the curriculum, so it wins if present.
    checkpoint_status = "none"
    loaded_from = None
    readout_kind = "policy_adapter"
    for candidate in (CHECKPOINT_PROMPT_INDEX, CHECKPOINT_OPTIONS, CHECKPOINT):
        if not candidate.exists():
            continue
        loaded, checkpoint_status = _load_readout(candidate)
        if loaded is not None:
            adapter, readout_kind = loaded
            loaded_from = str(candidate)
        break

    # The decision path steps the reservoir on the PROMPT when the readout is a
    # prompt-index model, and on the prompt-plus-options otherwise. Decisions run on their
    # own reservoir instance so the live-streamed state, which the tick thread advances, is
    # not perturbed by answering a question.
    decision_reservoir = FlyReservoir(conn, embedding_dim=EMBED_DIM, dims=DIMS, seed=7301)

    STATE.update(
        connectome=conn,
        reservoir=reservoir,
        adapter=adapter,
        readout_kind=readout_kind,
        decision_reservoir=decision_reservoir,
        checkpoint_status=checkpoint_status,
        checkpoint_loaded_from=loaded_from,
        encoder_fingerprint=ENCODER_FINGERPRINT,
        rng=np.random.default_rng(1234),
        curriculum=load_curriculum(),
        # ---- training and reward state, all exposed in the live frame ----
        dopamine=0.0,
        dopamine_last_t=time.time(),
        dopamine_total=0,
        #: recent fly outcomes, for the visible learning curve
        window=[],
        #: experience replay buffer: (features, correct index) pairs the fly has seen
        replay_x=[],
        replay_y=[],
        #: how many rehearsal steps have been taken, shown in the UI
        rehearsals=0,
        #: whether the loaded readout started from no training at all
        fresh_brain=False,
        lessons_completed=0,
        sessions={},
        session=None,
        started=time.time(),
        step=0,
        challenge=None,
        last=None,
        curve=deque(maxlen=500),
        # Populated only by an actual control evaluation. Left out until then.
        controls={},
        frames=0,
        # Serializes reservoir and adapter access between the answer path and the
        # background tick thread. Both mutate the same state.
        lock=threading.RLock(),
        ticks=0,
    )


def _lesson_order() -> list[str]:
    """Every lesson id, in teaching order. Recomputed cheaply from the loaded curriculum."""
    return [lesson.id for lesson in STATE["curriculum"].lessons()]


def _challenges_for(lesson_id: str) -> list[dict]:
    return STATE["curriculum"].all_challenges().get(lesson_id, [])


def _go_to_lesson(lesson_id: str) -> dict:
    """Make `lesson_id` the lesson in play and return its first challenge.

    This is what was missing: the old code advanced within a lesson and then set the next
    challenge to None when the lesson ran out, so the course ended after seven questions and
    every session looked like the same first lesson. Wrapping back to the start keeps a public
    demo running indefinitely, which is the point of it.
    """
    challenges = _challenges_for(lesson_id)
    if not challenges:
        raise ValueError(f"lesson {lesson_id!r} has no challenges")
    sess: Session = STATE["session"]
    sess.lesson_id = lesson_id
    sess.index = 0
    STATE["lesson_challenges"] = challenges
    STATE["challenge"] = challenges[0]
    return challenges[0]


def _advance_lesson() -> dict | None:
    """Move to the next lesson, wrapping at the end of the course. Returns its first challenge."""
    order = _lesson_order()
    if not order:
        return None
    sess: Session = STATE["session"]
    try:
        pos = order.index(sess.lesson_id)
    except ValueError:
        pos = -1
    nxt_pos = (pos + 1) % len(order)
    sess.lesson_pos = nxt_pos
    if nxt_pos == 0:
        # wrapped: the whole course has been seen through
        STATE["lessons_completed"] = int(STATE.get("lessons_completed", 0)) + 1
    return _go_to_lesson(order[nxt_pos])


def _fresh_readout():
    """A readout with no training at all, so learning can be watched from scratch.

    Measured, not assumed: PromptIndexReadout initialises W from N(0, 0.01) with b = 0, so at
    step zero the softmax over four options is near uniform. That is a genuinely naive brain,
    and the accuracy curve climbing from about chance is the whole point of the training mode.
    """
    from .learning.prompt_index import PromptIndexReadout

    obj = PromptIndexReadout(in_dim=DIMS, n_actions=N_ACTIONS, seed=0)
    return (obj, "prompt_index")


def _lesson_title(lesson_id: str | None) -> str | None:
    if not lesson_id:
        return None
    try:
        lesson = STATE["curriculum"].get_lesson(lesson_id)
    except Exception:
        return None
    return getattr(lesson, "title", None)


def _dopamine_now() -> float:
    """The live dopamine trace, decayed to now.

    Kept as a deadline-free exponential evaluated on read, so the number cannot depend on how
    often anything polls it.
    """
    last = float(STATE.get("dopamine_last_t", 0.0))
    level = float(STATE.get("dopamine", 0.0))
    if level <= 0.0:
        return 0.0
    dt = max(0.0, time.time() - last)
    return float(level * math.exp(-dt / DOPAMINE_TAU))


def _pulse_dopamine(correct: bool) -> float:
    """Deliver a reward pulse and return the new level."""
    now = time.time()
    level = _dopamine_now()
    if correct:
        level = min(1.0, level + DOPAMINE_UP)
        STATE["dopamine_total"] = int(STATE.get("dopamine_total", 0)) + 1
    else:
        level = max(0.0, level - DOPAMINE_DOWN)
    STATE["dopamine"] = level
    STATE["dopamine_last_t"] = now
    return level


def _reset_readout_learning() -> str:
    """Start the active readout's learning over, WITHOUT changing which readout it is.

    This distinction matters. Starting a session used to call `_fresh_readout()` unconditionally,
    which replaced whatever readout was selected with the default one -- so the HUD, which starts
    a session on load, would silently revert the plastic brain back to the old prompt-index
    readout and the demo would quietly stop running on the connectome.

    "Fresh" should mean the learning starts over, not that the architecture changes.
    """
    kind = STATE.get("readout_kind")
    if kind == "plastic_brain":
        pb = _plastic_brain()
        pb.r.plastic_scale[:] = 1.0
        pb.r.apply_plastic()
        # 1/pool_size is exactly the unweighted mean, so this is the anatomical brain with no
        # learning of either kind.
        pb.score_w[:] = 1.0 / pb.pool_size
        pb._m = pb._v = pb._wm = pb._wv = None
        pb._t = pb._wt = 0
        pb.updates = 0
        STATE["checkpoint_status"] = "plastic connectome, scales at 1.0"
        return kind
    STATE["adapter"], STATE["readout_kind"] = _fresh_readout()
    STATE["checkpoint_status"] = "fresh (untrained)"
    return STATE["readout_kind"]


def _active_parameters() -> int:
    """How many trainable parameters the ACTIVE readout has.

    The plastic brain's parameters are synaptic scales on real connectome edges, and it has no
    `adapter`, so reporting the adapter's 516 here would have been a flat lie in the UI: it said
    516 while the thing learning had 117,800 plastic synapses.
    """
    if STATE.get("readout_kind") == "plastic_brain":
        pb = STATE.get("plastic_brain")
        if pb is not None:
            # Both parts train: the synaptic scales on real edges AND the weights that read each
            # answer population. Reporting only the synapses understated what learns.
            return int(pb.r.plastic_scale.size) + int(pb.score_w.size)
        return 0
    return int(STATE["adapter"].parameters())


def _plastic_telemetry() -> dict:
    """What the brain's own synapses are doing, for the live panel.

    Without this the UI could not show that the thing being learned is the wiring: every number
    it had described the old bolted-on readout.
    """
    pb = STATE.get("plastic_brain")
    if pb is None or STATE.get("readout_kind") != "plastic_brain":
        return {}
    s = pb.r.plastic_scale
    return {
        "plastic_readout_params": int(pb.score_w.size),
        "plastic_readout_std": float(np.std(pb.score_w)),
        "plastic_edges": int(s.size),
        "plastic_of_total": float(s.size / max(1, pb.r.graph.nnz)),
        "plastic_scale_mean": float(np.mean(s)),
        "plastic_scale_std": float(np.std(s)),
        "plastic_scale_max": float(np.max(s)),
        "plastic_updates": int(pb.updates),
        "plastic_settle_steps": int(pb.steps),
        "plastic_pools": int(pb.n_pools),
        "plastic_pool_size": int(pb.pool_size),
    }


def _plastic_brain():
    """The plastic brain, built once and cached.

    It gets its OWN reservoir over the SAME connectome object, for two reasons. Settling resets
    the state, and the tick loop steps the state continuously, so sharing one instance would have
    them overwriting each other's dynamics. And because both reservoirs point at the same
    connectome matrix, the plastic weights this one writes are the same weights the live view
    renders -- so the brain on screen is the brain that is learning, not a copy of it.
    """
    pb = STATE.get("plastic_brain")
    if pb is None:
        from .plastic_brain import PlasticBrain

        own = FlyReservoir(STATE["connectome"], embedding_dim=EMBED_DIM, dims=DIMS, seed=7301)
        pb = PlasticBrain(
            own,
            n_pools=4,
            pool_size=200,
            seed=99,
            steps=PLASTIC_SETTLE_STEPS,
            max_plastic_edges=PLASTIC_MAX_EDGES,
            lr=PLASTIC_LR,
        )
        STATE["plastic_brain"] = pb
    return pb


def _decision_features(ch: dict) -> np.ndarray:
    """The feature vector the readout decides on. ONE definition, used everywhere.

    The answer path and the streamed probabilities must agree, or the UI shows
    probabilities that do not correspond to the choice the fly actually makes. They
    diverged once: the tick loop refreshed `probs` from its own drifting drive while
    answering used the challenge embedding, so a trained readout displayed near-uniform
    probabilities next to confident answers.
    """
    if STATE.get("readout_kind") == "plastic_brain":
        # Same path as the decision, so the displayed probabilities always belong to the choice
        # the fly actually makes.
        emb = encode_text(ch["prompt"])
        STATE["probs"] = _plastic_brain().probs(emb)
        return
    if STATE.get("readout_kind") == "prompt_index":
        # Trained on features of the PROMPT alone.
        dr = STATE["decision_reservoir"]
        dr.reset()
        return dr.step(encode_text(ch["prompt"]))
    return STATE["reservoir"].step(encode_challenge(ch))


def _refresh_probs(ch: dict | None = None) -> None:
    """Recompute the displayed probabilities for the current challenge."""
    ch = ch if ch is not None else STATE.get("challenge")
    if ch is None:
        return
    f = _decision_features(ch)
    STATE["last_features"] = f
    STATE["probs"] = STATE["adapter"].probs(f)


def _tick_loop() -> None:
    """Step the connectome continuously so the live view is actually live.

    The step is CPU-bound (about 24.5 ms for 166,700 neurons), so it runs on its own
    daemon thread rather than in the event loop, and every access to the reservoir and
    the adapter is taken under STATE['lock'] so an answer can never interleave with a
    tick and read a half-updated state.

    This drives the VISUALIZATION only. Probabilities are refreshed at challenge
    boundaries instead, so what is displayed always matches what the readout decides.
    """
    while True:
        time.sleep(TICK_INTERVAL)
        ch = STATE.get("challenge")
        if ch is None:
            continue
        try:
            with STATE["lock"]:
                r = STATE["reservoir"]
                tick = STATE.get("ticks", 0)
                r.step(tick_embedding(ch, tick))
                STATE["ticks"] = tick + 1
        except Exception:
            # A failed tick must never kill the process; the next one will retry.
            continue


@app.on_event("startup")
def _startup() -> None:
    _boot()
    thread = threading.Thread(target=_tick_loop, name="connectome-tick", daemon=True)
    thread.start()
    STATE["tick_thread"] = thread


# ------------------------------------------------------------------- encoder
# The encoders live in brain/encoders.py so that training and serving use ONE
# implementation. They were previously duplicated here and built on Python's builtin
# hash(), which is salted per process: the same text produced different vectors in
# different interpreters, so a checkpoint fit during training was applied to features the
# server never reproduced. Anything that encodes text must import from brain.encoders.
from .encoders import (  # noqa: E402
    IDLE_MIX,
    encode_challenge,
    encode_idle,
    encode_option_pairs,
    encode_text,
    encoder_fingerprint,
    tick_embedding,
)

#: Fingerprint of the encoding scheme in force. Recorded in checkpoints and verified on load.
ENCODER_FINGERPRINT = encoder_fingerprint()

#: How long a dopamine pulse takes to fall to 1/e, seconds. Dopamine here is not decoration:
#: it is the reward signal that drives the readout's plasticity, so the trace that the UI draws
#: is the same number that gated the last weight update.
#:
#: 2.5 s rather than a snappier value on purpose: the pulse fires on an answer, and the viewer
#: is then reading the result and the next question, so a 1.4 s trace had already decayed to
#: nothing by the time anyone looked. The count of pulses is what persists; this is the pulse.
DOPAMINE_TAU = 2.5
#: A correct answer delivers this much dopamine in one pulse, and a wrong one takes this much
#: away. Negative is a real part of the signal (a prediction error below expectation), not a
#: punishment added for effect.
DOPAMINE_UP = 0.62
DOPAMINE_DOWN = 0.22
#: Rolling window for the visible learning curve, in answers.
LEARNING_WINDOW = 20
#: Supervised rehearsal steps taken from the replay buffer after each answer.
#:
#: Measured against the real 97-challenge curriculum, from a fresh readout: with 0 the fly
#: plateaus near 45% and never converges, and with 60 it reaches 92-100% within about 100
#: answers. One answer is one update, and one pass over 97 pairs cannot fit 516 parameters, so
#: the rehearsal is what makes learning visible inside a session rather than over many.
REPLAY_STEPS = 60
#: Cap on the replay buffer. The course is 97 challenges, so this holds it all.
REPLAY_CAPACITY = 256
#: Learning rate for the online supervised steps. 0.2-0.5 measured stable; the loss curve and
#: entropy in the live frame are the check that it stays sane.
TRAIN_LR = 0.35

# --- the plastic brain's own settings
#
# Settling steps: the recurrence has to run for the wiring to shape the trajectory. Measured
# convergence is around 10 steps (8 and 14 steps differ by 0.0016), so 6 keeps most of the
# dynamics at about 105 ms a decision.
PLASTIC_SETTLE_STEPS = 6
# Plastic edge cap. Pooling 200 neurons per answer gives ~117,800 real edges onto those
# populations from the full connectome; this is the cap, not a target.
PLASTIC_MAX_EDGES = 120_000
# Adam step for the synaptic scales. The raw gradient is ~1e-6 because anatomical weights are
# row-normalised, so a plain step of that size does nothing at all -- see plastic_brain.py.
PLASTIC_LR = 0.01
# Rehearsal steps per answer for the plastic brain. Each is a settle plus an update (~105 ms), so
# this sets how fast the demo learns on screen. Measured: without it the live curve needs about
# ninety minutes to leave chance, with it the change is visible inside a minute.
PLASTIC_REPLAY_STEPS = 20


# --------------------------------------------------------------------- helpers


def _challenge_payload(ch: dict) -> dict:
    return {
        "id": ch["id"],
        "type": ch["type"],
        "prompt": ch["prompt"],
        "promptLang": ch.get("promptLang", "en"),
        "answer": ch["answer"],
        "options": list(ch["options"]),
        "correctIndex": int(ch["correctIndex"]),
        "audio": ch.get("audio", ch["answer"]),
        "difficulty": int(ch.get("difficulty", 1)),
    }


def _current_frame(state: dict, chosen=None, reward=None, correct=None) -> dict:
    r = state["reservoir"]
    with state["lock"]:
        tel = r.telemetry()
    sess = state["session"]
    probs = state.get("probs")
    if probs is None:
        probs = [0.25] * N_ACTIONS
    acc = (sess.correct / sess.answered) if sess and sess.answered else 0.0
    lesson_progress = 0.0
    if sess:
        total = len(state["lesson_challenges"])
        lesson_progress = (sess.index / total) if total else 0.0
    return {
        "t": tel["updates"],
        "step": state["step"],
        "challenge_id": state["challenge"]["id"] if state["challenge"] else None,
        "spikes": tel["spikes"],
        "state": tel["sampled_state"],
        "sampled_ids": tel["sampled_ids"],
        "active_fraction": tel["active_fraction"],
        "state_rms": tel["state_rms"],
        "probs": [float(p) for p in probs],
        "chosen": chosen,
        "reward": reward,
        "correct": correct,
        "mode": r.mode,
        "ticks": int(state.get("ticks", 0)),
        "lesson_id": sess.lesson_id if sess else None,
        "lesson_progress": float(lesson_progress),
        "accuracy": float(acc),
        "fly_accuracy": float(
            (STATE["fly_correct"] / STATE["fly_answered"]) if STATE.get("fly_answered") else 0.0
        ),
        "streak": sess.streak if sess else 0,
        # ---- reward and training telemetry ----
        "dopamine": _dopamine_now(),
        "dopamine_total": int(STATE.get("dopamine_total", 0)),
        "fresh_brain": bool(STATE.get("fresh_brain", False)),
        "learned_correct": int(STATE.get("fly_correct", 0)),
        "learned_answered": int(STATE.get("fly_answered", 0)),
        "window_accuracy": (
            sum(STATE.get("window", [])) / len(STATE["window"])
            if STATE.get("window")
            else 0.0
        ),
        "window_size": len(STATE.get("window", [])),
        "lessons_completed": int(STATE.get("lessons_completed", 0)),
        "lesson_pos": sess.lesson_pos if sess else 0,
        "lesson_total": len(STATE.get("lesson_order_cache") or _lesson_order()),
        "lesson_title": _lesson_title(sess.lesson_id) if sess else None,
        "readout_kind": STATE.get("readout_kind"),
        "checkpoint_status": STATE.get("checkpoint_status"),
        # ---- the brain's own synapses, when it is the thing that learns ----
        **_plastic_telemetry(),
        "entropy": float(STATE.get("last", {}).get("entropy", 0.0)),
        "grad_norm": float(STATE.get("last", {}).get("grad_norm", 0.0)),
        "rehearsals": int(STATE.get("rehearsals", 0)),
        "replay_size": len(STATE.get("replay_x", [])),
        "training": bool(STATE.get("training", True)),
        "hearts": sess.hearts if sess else 5,
        "xp": sess.xp if sess else 0,
        "controls": dict(state["controls"]),
    }


# ------------------------------------------------------------------- endpoints


@app.get("/health")
def health() -> dict:
    conn = STATE["connectome"]
    return {
        "status": "ok",
        "neurons": int(conn.neurons),
        "edges": int(conn.edges),
        "dataset": "MaleCNS v1.0",
        "checkpoint": STATE.get("checkpoint_loaded_from"),
        "checkpoint_status": STATE.get("checkpoint_status", "none"),
        "readout_kind": STATE.get("readout_kind"),
        "encoder_fingerprint": STATE.get("encoder_fingerprint"),
        "uptime_s": round(time.time() - STATE["started"], 1),
    }


@app.get("/curriculum")
def curriculum() -> dict:
    return STATE["curriculum"].model_dump()


@app.get("/stats")
def stats() -> dict:
    return {
        "lessons_completed": int(STATE.get("lessons_completed", 0)),
        "accuracy": float(
            (STATE["session"].correct / STATE["session"].answered)
            if STATE.get("session") and STATE["session"].answered
            else 0.0
        ),
        "accuracy_by_difficulty": STATE.get("accuracy_by_difficulty", {}),
        "learning_curve": list(STATE["curve"]),
        "controls": dict(STATE["controls"]),
    }


class StartReq(BaseModel):
    lesson_id: str | None = None
    #: start from an untrained readout, so the learning is visible rather than already done
    fresh: bool = False
    #: which lesson in the course to start at
    lesson_pos: int | None = None


@app.post("/session")
def start_session(req: StartReq) -> dict:
    cur = STATE["curriculum"]
    lesson = None
    if req.lesson_id:
        lesson = cur.get_lesson(req.lesson_id)
    if lesson is None:
        lesson = cur.first_lesson()
    challenges = cur.all_challenges().get(lesson.id, [])
    if not challenges:
        raise HTTPException(status_code=500, detail="lesson has no challenges")

    if req.fresh:
        # A genuinely naive readout. The shipped checkpoint has already memorised the whole
        # curriculum, so with it loaded there is nothing to watch learn; this is the state the
        # demo wants when the point is to SEE the training happen.
        #
        # Resets the ACTIVE readout's learning rather than swapping architectures: the HUD starts
        # a session on load, and replacing the readout here silently reverted the plastic brain
        # to the old prompt-index readout.
        _reset_readout_learning()
        STATE["fresh_brain"] = True
        STATE["dopamine"] = 0.0
        STATE["dopamine_total"] = 0
        STATE["window"] = []
        STATE["fly_answered"] = 0
        STATE["fly_correct"] = 0
        STATE["curve"] = []
        STATE["replay_x"] = []
        STATE["replay_y"] = []
        STATE["rehearsals"] = 0
    # NOTE: there is deliberately no `else` branch here resetting fresh_brain to False.
    # fresh_brain describes the ADAPTER in memory, not the session, and the UI starts a session
    # on load without passing `fresh`. Resetting it here therefore made the UI report
    # "pretrained" while the readout was in fact naive, contradicting /health, which derives the
    # same claim from checkpoint_status and got it right. The two must not be able to disagree.

    order = _lesson_order()
    pos = order.index(lesson.id) if lesson.id in order else 0
    if req.lesson_pos is not None:
        pos = int(req.lesson_pos) % max(1, len(order))
        lesson = STATE["curriculum"].get_lesson(order[pos])
        challenges = _challenges_for(lesson.id)

    sid = f"s{int(time.time()*1000)}"
    STATE["lesson_challenges"] = challenges
    STATE["session"] = Session(id=sid, lesson_id=lesson.id, lesson_pos=pos)
    STATE["challenge"] = challenges[0]
    STATE["step"] = 0

    r = STATE["reservoir"]
    # Carry neural state across the lesson so the activity you watch is a continuous
    # trajectory, not a fresh look at each question. Reset at lesson start only.
    r.reset()
    _refresh_probs(challenges[0])
    STATE["last_chosen"] = None
    return {
        "session_id": sid,
        "lesson_id": lesson.id,
        "challenge": _challenge_payload(challenges[0]),
    }


class AnswerReq(BaseModel):
    session_id: str
    challenge_id: str
    choice_index: int = -1
    # Whose answer drives the lesson.
    #
    # False (default): the human clicked an option, and that answer moves hearts, XP and
    # progress while the fly is scored on its own separate action.
    #
    # True: nobody is playing. The fly's own sampled action is what counts, so the lesson
    # advances exactly as if the fly had clicked. This is what makes the demo run hands-off and
    # recordable: without it the frame sat on the first challenge forever because nothing ever
    # submitted an answer, so the fly never chose, never moved and never earned reward.
    as_fly: bool = False


@app.post("/answer")
def answer(req: AnswerReq) -> dict:
    sess: Session = STATE["session"]
    if sess is None or sess.id != req.session_id:
        raise HTTPException(status_code=409, detail="no such session")
    ch = STATE["challenge"]
    if ch is None or ch["id"] != req.challenge_id:
        raise HTTPException(status_code=409, detail="challenge is not current")

    adapter = STATE["adapter"]
    r = STATE["reservoir"]
    lock = STATE["lock"]

    # The fly's decision. Held under the tick thread's lock so a tick cannot interleave
    # between the step and the reward assignment, which would associate the reward with the
    # wrong state.
    with lock:
        if STATE.get("readout_kind") == "plastic_brain":
            # The brain chooses: settle the connectome, then read the answer pools. There is no
            # classifier between the neurons and the decision.
            pb = _plastic_brain()
            emb = encode_text(ch["prompt"])
            probs = pb.probs(emb)
            action = int(np.argmax(probs))
            logprob = float(np.log(max(float(probs[action]), 1e-12)))

            fly_correct = action == int(ch["correctIndex"])
            reward = 1.0 if fly_correct else -0.25
            # The brain learns: a gradient step on its own synapses, onto the answer population
            # that should have won. This is the whole point of the architecture -- the wiring
            # being changed is real wiring, not a readout beside it.
            if STATE.get("training", True):
                info = pb.observe(emb, int(ch["correctIndex"]))

                # Rehearsal, for the plastic brain specifically.
                #
                # Without it the live demo learns too slowly to watch: one update per answer
                # means ~5 seconds of wall clock per gradient step, and the measured curve needs
                # roughly a thousand steps to leave chance, which is about ninety minutes of
                # screen time. Rehearsing from what the fly has already seen is what makes the
                # learning visible in a recording without changing what is being learned -- the
                # same synapses, the same rule, the same examples, just revisited.
                pb_buf = STATE.setdefault("pb_replay", [])
                pb_buf.append((emb.copy(), int(ch["correctIndex"])))
                if len(pb_buf) > REPLAY_CAPACITY:
                    del pb_buf[0]
                if len(pb_buf) > 1:
                    for _ in range(PLASTIC_REPLAY_STEPS):
                        k = int(STATE["rng"].integers(0, len(pb_buf)))
                        pb.observe(pb_buf[k][0], pb_buf[k][1])
                    STATE["rehearsals"] = int(STATE.get("rehearsals", 0)) + PLASTIC_REPLAY_STEPS
            else:
                info = {"loss": float(-np.log(max(float(probs[action]), 1e-12)))}
            f = None
        else:
            f = _decision_features(ch)
            action, logprob = adapter.sample(f, STATE["rng"])
            probs = adapter.probs(f)

            # The fly is rewarded for its OWN sampled action being right, never for the user's.
            # Rewarding it for a choice it did not make would train on noise.
            fly_correct = int(action) == int(ch["correctIndex"])
            reward = 1.0 if fly_correct else -0.25
            info = adapter.observe(reward, logprob, f)

        # Rehearsal. The reward-only step above reinforces the sampled action and nothing more,
        # which measured as a plateau near 45%; these steps learn from the lesson's answer key
        # against remembered examples, which measured as reaching 92-100% within a session.
        if STATE.get("training", True) and STATE.get("readout_kind") == "prompt_index":
            # Not applicable to the plastic brain: its observe() already trains the connectome's
            # own synapses, so rehearsal here would be a second, unrelated mechanism.
            truth = int(ch["correctIndex"])
            if hasattr(adapter, "observe_supervised"):
                info = adapter.observe_supervised(truth, lr=TRAIN_LR)

            buf_x = STATE.setdefault("replay_x", [])
            buf_y = STATE.setdefault("replay_y", [])
            buf_x.append(np.asarray(f, np.float64).copy())
            buf_y.append(truth)
            if len(buf_x) > REPLAY_CAPACITY:
                del buf_x[0]
                del buf_y[0]

            if hasattr(adapter, "rehearse") and buf_x:
                n = len(buf_x)
                for _ in range(REPLAY_STEPS):
                    k = int(STATE["rng"].integers(0, n))
                    adapter.rehearse(buf_x[k], buf_y[k], lr=TRAIN_LR)
                STATE["rehearsals"] = int(STATE.get("rehearsals", 0)) + REPLAY_STEPS
        # Dopamine is delivered AFTER the plasticity step, because it is the same signal: the
        # reward gates the weight update, and the trace is what the UI draws. Computing it here
        # keeps the drawn pulse and the applied gradient from ever being different numbers.
        dopamine = _pulse_dopamine(bool(fly_correct))

    # Whose answer drives the lesson itself. In the hands-off demo the fly answers for itself,
    # so its own sampled action is the one that moves hearts, XP and progress; otherwise the
    # human's click is. Either way the fly is scored on its own action.
    driving_choice = int(action) if req.as_fly else int(req.choice_index)
    correct = driving_choice == int(ch["correctIndex"])

    STATE["step"] += 1
    sess.answered += 1
    sess.correct += int(correct)
    sess.streak = sess.streak + 1 if correct else 0
    if correct:
        sess.xp += 10 + 5 * int(ch.get("difficulty", 1))
    else:
        sess.hearts = max(0, sess.hearts - 1)

    # The fly's own scoreboard, tracked separately so its curve is never confused
    # with the user's.
    STATE["fly_answered"] = STATE.get("fly_answered", 0) + 1
    STATE["fly_correct"] = STATE.get("fly_correct", 0) + int(fly_correct)

    # The rolling window is what makes learning visible: a cumulative average barely moves after
    # a hundred answers, so a curve built on it looks flat even while the model is improving.
    win = STATE.setdefault("window", [])
    win.append(int(fly_correct))
    if len(win) > LEARNING_WINDOW:
        del win[0]

    STATE["curve"].append(
        {
            "step": STATE["step"],
            "correct": bool(correct),
            "fly_correct": bool(fly_correct),
            "accuracy": sess.correct / sess.answered,
            "fly_accuracy": STATE["fly_correct"] / STATE["fly_answered"],
            "user_choice": int(driving_choice),
            "fly_choice": int(action),
            "answer_index": int(ch["correctIndex"]),
            "reward": reward,
            "loss": info.get("loss", 0.0),
            "mode": r.mode,
        }
    )
    STATE["last"] = info
    STATE["probs"] = probs
    STATE["last_features"] = f
    STATE["last_chosen"] = action

    # Advance only on a correct answer. A wrong answer keeps the same challenge current,
    # matching the lesson flow (wrong, then retry the same question): advancing past it
    # would make the retry target a challenge the server no longer considers current.
    if correct:
        if sess.index + 1 < len(STATE["lesson_challenges"]):
            sess.index += 1
            nxt = STATE["lesson_challenges"][sess.index]
        else:
            # The lesson is finished. Move to the next one instead of stopping: the old code
            # left the next challenge as None here, so the demo dead-ended on the first lesson.
            sess.lessons_completed += 1
            try:
                nxt = _advance_lesson()
            except Exception:
                nxt = None
    else:
        nxt = ch
    STATE["challenge"] = nxt
    if nxt is not None:
        # The displayed probabilities must belong to the challenge now on screen.
        with lock:
            _refresh_probs(nxt)
            STATE["last_chosen"] = None

    return {
        "correct": bool(correct),
        "fly_correct": bool(fly_correct),
        "fly_accuracy": (
            STATE["fly_correct"] / STATE["fly_answered"] if STATE["fly_answered"] else 0.0
        ),
        "fly_choice": int(action),
        "user_choice": int(req.choice_index),
        "reward": reward,
        "chosen": int(action),
        "answer_index": int(ch["correctIndex"]),
        "probs": [float(p) for p in probs],
        "loss": float(info.get("loss", 0.0)),
        "grad_norm": float(info.get("grad_norm", 0.0)),
        "entropy": float(info.get("entropy", 0.0)),
        "updated_params": int(info.get("updated_params", 0)),
        "gated_params": int(info.get("gated_params", 0)),
        "step": STATE["step"],
        "streak": sess.streak,
        "xp": sess.xp,
        "hearts": sess.hearts,
        "next_challenge": _challenge_payload(nxt) if nxt else None,
    }


class TrainReq(BaseModel):
    fresh: bool = True
    training: bool | None = None
    lr: float | None = None
    replay_steps: int | None = None
    #: Which readout answers the questions. "plastic_brain" makes the connectome itself the
    #: chooser and the learner; "prompt_index" is the older frozen-wiring design kept for
    #: comparison. None leaves the current one alone.
    kind: str | None = None


@app.post("/train")
def train(req: TrainReq) -> dict:
    """Reset the readout to untrained, or reload the shipped checkpoint.

    Exposed so the UI can offer "watch it learn" and "already trained" as a real choice rather
    than pretending the pretrained model is learning.
    """
    if req.kind is not None:
        if req.kind not in ("plastic_brain", "prompt_index", "policy_adapter"):
            return {"ok": False, "detail": f"unknown readout kind {req.kind!r}"}
        STATE["readout_kind"] = req.kind
        # One reset path, used by /train and /session alike, so the two cannot drift apart.
        _reset_readout_learning()
        STATE["fresh_brain"] = True
    elif req.fresh:
        _reset_readout_learning()
        STATE["fresh_brain"] = True
    else:
        loaded, status = _load_readout(CHECKPOINT_PROMPT_INDEX)
        if loaded is None:
            return {"ok": False, "checkpoint_status": status}
        STATE["adapter"], STATE["readout_kind"] = loaded
        STATE["checkpoint_status"] = status
        STATE["fresh_brain"] = False

    global TRAIN_LR, REPLAY_STEPS
    if req.training is not None:
        STATE["training"] = bool(req.training)
    if req.lr is not None:
        TRAIN_LR = float(req.lr)
    if req.replay_steps is not None:
        REPLAY_STEPS = int(max(0, req.replay_steps))

    STATE["dopamine"] = 0.0
    STATE["dopamine_total"] = 0
    STATE["window"] = []
    STATE["replay_x"] = []
    STATE["replay_y"] = []
    STATE["pb_replay"] = []
    STATE["rehearsals"] = 0
    STATE["fly_answered"] = 0
    STATE["fly_correct"] = 0
    STATE["curve"] = []
    STATE["step"] = 0
    STATE["last"] = {}
    with STATE["lock"]:
        _refresh_probs()
    return {
        "ok": True,
        "fresh_brain": bool(STATE["fresh_brain"]),
        "checkpoint_status": STATE.get("checkpoint_status"),
        "readout_kind": STATE.get("readout_kind"),
        "parameters": _active_parameters(),
        "training": bool(STATE.get("training", True)),
        "lr": TRAIN_LR,
        "replay_steps": REPLAY_STEPS,
    }


class ControlReq(BaseModel):
    mode: str


@app.post("/control")
def control(req: ControlReq) -> dict:
    r = STATE["reservoir"]
    try:
        r.set_mode(req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Re-step under the new mode so the streamed activity reflects it immediately.
    # Without this the telemetry keeps showing the previous mode's state until the next
    # answer, so switching to a control would change the label but not the brain, and a
    # viewer could not tell whether the control was doing anything.
    reapplied = False
    ch = STATE.get("challenge")
    if ch is not None:
        with STATE["lock"]:
            r.reset()
            _refresh_probs(ch)
            STATE["last_chosen"] = None
            reapplied = True

    return {
        "mode": r.mode,
        "note": "control mode now applies to the streamed activity and the next answers",
        "applied_to_current_state": reapplied,
    }


@app.post("/reset")
def reset() -> dict:
    STATE["reservoir"].reset()
    state = STATE["adapter"]
    fresh = PolicyAdapter(in_dim=DIMS, n_actions=N_ACTIONS, hidden=256, seed=0)
    state.__dict__.update(fresh.__dict__)
    STATE["step"] = 0
    STATE["curve"].clear()
    return {"ok": True}


@app.get("/telemetry")
def telemetry() -> dict:
    return _current_frame(STATE, STATE.get("last_chosen"))


# ------------------------------------------------------------------- websocket


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    interval = 1.0 / STREAM_HZ
    try:
        while True:
            frame = _current_frame(STATE, STATE.get("last_chosen"))
            await ws.send_text(json.dumps(frame))
            STATE["frames"] += 1
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return
    except Exception:
        return
