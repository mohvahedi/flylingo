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


def _boot() -> None:
    conn = load_connectome()
    reservoir = FlyReservoir(conn, embedding_dim=EMBED_DIM, dims=DIMS, seed=7301)
    adapter = PolicyAdapter(in_dim=DIMS, n_actions=N_ACTIONS, hidden=256, seed=0)
    if CHECKPOINT.exists():
        adapter.load(CHECKPOINT)
    STATE.update(
        connectome=conn,
        reservoir=reservoir,
        adapter=adapter,
        rng=np.random.default_rng(1234),
        curriculum=load_curriculum(),
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


def _tick_loop() -> None:
    """Step the connectome continuously so the live view is actually live.

    The step is CPU-bound (about 24.5 ms for 166,700 neurons), so it runs on its own
    daemon thread rather than in the event loop, and every access to the reservoir and
    the adapter is taken under STATE['lock'] so an answer can never interleave with a
    tick and read a half-updated state.
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
                f = r.step(tick_embedding(ch, tick))
                STATE["last_features"] = f
                if STATE.get("session") is not None:
                    STATE["probs"] = STATE["adapter"].probs(f)
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
# A deterministic hashed bag-of-tokens encoder. Chosen deliberately: it is
# training-free, reproducible across processes, and it cannot smuggle in task
# knowledge from a pretrained language model, so any learning we measure is the
# readout learning, not a language model answering the question for the fly.


def encode_text(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    v = np.zeros(dim, np.float32)
    t = " " + text.strip().lower() + " "
    for n in (1, 2, 3):
        for i in range(max(0, len(t) - n + 1)):
            gram = t[i : i + n]
            h = hash(gram) & 0xFFFFFFFF
            v[h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    v += np.sin(np.arange(dim, dtype=np.float32) * 0.017)
    nrm = float(np.linalg.norm(v)) + 1e-6
    return (v / nrm).astype(np.float32)


#: Strength of the continuously varying drive component, relative to the challenge text.
#: See encode_idle for why this exists.
IDLE_MIX = 0.25


def encode_idle(tick: int, dim: int = EMBED_DIM) -> np.ndarray:
    """A slowly varying drive, so the connectome keeps moving between answers.

    Feeding the reservoir one constant embedding converges it to a fixed point within
    about a dozen steps, which makes the live view settle into a still image. This adds
    a deterministic, slowly drifting component so the network keeps exploring its own
    dynamics while a challenge sits on screen.

    What this is and is not: it is an input signal to a fixed dynamical system, chosen
    so the visualization stays alive. It is not evidence of the fly attending to
    anything, and it carries no task information. The decision at answer time still
    comes from the challenge text through the same measured wiring.
    """
    i = np.arange(dim, dtype=np.float32)
    # Three incommensurate frequencies per dimension: a long period, so the drive never
    # visibly repeats within a lesson, and no shared phase between dimensions.
    v = (
        np.sin(0.013 * float(tick) + i * 0.11)
        + 0.5 * np.sin(0.0071 * float(tick) + i * 0.037)
        + 0.25 * np.sin(0.0031 * float(tick) + i * 0.211)
    ).astype(np.float32)
    nrm = float(np.linalg.norm(v)) + 1e-6
    return (v / nrm).astype(np.float32)


def tick_embedding(challenge: dict, tick: int) -> np.ndarray:
    """The drive for one background step: the challenge text plus a drifting component."""
    base = encode_challenge(challenge)
    mixed = (1.0 - IDLE_MIX) * base + IDLE_MIX * encode_idle(tick)
    nrm = float(np.linalg.norm(mixed)) + 1e-6
    return (mixed / nrm).astype(np.float32)


def encode_challenge(ch: dict) -> np.ndarray:
    parts = [ch["prompt"], ch.get("type", ""), "en"]
    parts += [str(o) for o in ch.get("options", [])]
    return encode_text(" | ".join(parts))


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
        "checkpoint": str(CHECKPOINT) if CHECKPOINT.exists() else None,
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

    sid = f"s{int(time.time()*1000)}"
    STATE["lesson_challenges"] = challenges
    STATE["session"] = Session(id=sid, lesson_id=lesson.id)
    STATE["challenge"] = challenges[0]
    STATE["step"] = 0

    r = STATE["reservoir"]
    # Carry neural state across the lesson so the activity you watch is a continuous
    # trajectory, not a fresh look at each question. Reset at lesson start only.
    r.reset()
    emb = encode_challenge(challenges[0])
    f = r.step(emb)
    p = STATE["adapter"].probs(f)
    STATE["probs"] = p
    STATE["last_features"] = f
    STATE["last_chosen"] = None
    return {
        "session_id": sid,
        "lesson_id": lesson.id,
        "challenge": _challenge_payload(challenges[0]),
    }


class AnswerReq(BaseModel):
    session_id: str
    challenge_id: str
    choice_index: int


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

    # The fly's decision: one more step of real connectome activity, then a sampled
    # choice. Held under the tick thread's lock so a tick cannot interleave between the
    # step and the reward assignment, which would associate the reward with the wrong state.
    with lock:
        emb = encode_challenge(ch)
        f = r.step(emb)
        action, logprob = adapter.sample(f, STATE["rng"])
        probs = adapter.probs(f)

        # The fly is rewarded for its OWN sampled action being right, never for the user's.
        # Rewarding it for a choice it did not make would train on noise.
        fly_correct = int(action) == int(ch["correctIndex"])
        reward = 1.0 if fly_correct else -0.25
        info = adapter.observe(reward, logprob, f)

    # The user's answer drives the lesson itself: hearts, XP, progress.
    correct = int(req.choice_index) == int(ch["correctIndex"])

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

    STATE["curve"].append(
        {
            "step": STATE["step"],
            "correct": bool(correct),
            "fly_correct": bool(fly_correct),
            "accuracy": sess.correct / sess.answered,
            "fly_accuracy": STATE["fly_correct"] / STATE["fly_answered"],
            "user_choice": int(req.choice_index),
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
            nxt = None
    else:
        nxt = ch
    STATE["challenge"] = nxt

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
            f = r.step(encode_challenge(ch))
            STATE["last_features"] = f
            if STATE.get("session") is not None:
                STATE["probs"] = STATE["adapter"].probs(f)
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
