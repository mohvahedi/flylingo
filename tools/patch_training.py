"""Apply the training/progression patch to brain/api.py.

Every replacement asserts its anchor exists, so a silent no-op is impossible: if the file
changed under us the script fails loudly instead of half-applying.
"""
from pathlib import Path

P = Path(r"D:\Projects\flylingo\brain\brain\api.py")
s = P.read_text(encoding="utf-8")
orig = s


def sub(old: str, new: str, count: int = 1) -> None:
    global s
    assert old in s, f"ANCHOR MISSING:\n{old[:200]}"
    s = s.replace(old, new, count)


# ---------------------------------------------------------------- 1. constants
sub(
    """#: Fingerprint of the encoding scheme in force. Recorded in checkpoints and verified on load.
ENCODER_FINGERPRINT = encoder_fingerprint()""",
    """#: Fingerprint of the encoding scheme in force. Recorded in checkpoints and verified on load.
ENCODER_FINGERPRINT = encoder_fingerprint()

#: How long a dopamine pulse takes to fall to 1/e, seconds. Dopamine here is not decoration:
#: it is the reward signal that drives the readout's plasticity, so the trace that the UI draws
#: is the same number that gated the last weight update.
DOPAMINE_TAU = 1.4
#: A correct answer delivers this much dopamine in one pulse, and a wrong one takes this much
#: away. Negative is a real part of the signal (a prediction error below expectation), not a
#: punishment added for effect.
DOPAMINE_UP = 0.62
DOPAMINE_DOWN = 0.22
#: Rolling window for the visible learning curve, in answers.
LEARNING_WINDOW = 20""",
)

# ---------------------------------------------------------------- 2. session fields
sub(
    """    answered: int = 0
    correct: int = 0""",
    """    answered: int = 0
    correct: int = 0
    #: which lesson within the course this session is on, and how many lessons exist. The
    #: demo used to stop dead at the end of the first lesson because nothing advanced past it.
    lesson_pos: int = 0
    lessons_completed: int = 0""",
)

# ---------------------------------------------------------------- 3. progression helpers
sub(
    """def _decision_features(ch: dict) -> np.ndarray:""",
    '''def _lesson_order() -> list[str]:
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


def _decision_features(ch: dict) -> np.ndarray:''',
)

# ---------------------------------------------------------------- 4. boot: new state
sub(
    """        curriculum=load_curriculum(),""",
    """        curriculum=load_curriculum(),
        # ---- training and reward state, all exposed in the live frame ----
        dopamine=0.0,
        dopamine_last_t=time.time(),
        dopamine_total=0,
        #: recent fly outcomes, for the visible learning curve
        window=[],
        #: whether the loaded readout started from no training at all
        fresh_brain=False,
        lessons_completed=0,""",
)

# ---------------------------------------------------------------- 5. session start
sub(
    """class StartReq(BaseModel):
    lesson_id: str | None = None""",
    """class StartReq(BaseModel):
    lesson_id: str | None = None
    #: start from an untrained readout, so the learning is visible rather than already done
    fresh: bool = False
    #: which lesson in the course to start at
    lesson_pos: int | None = None""",
)

sub(
    """    sid = f"s{int(time.time()*1000)}"
    STATE["lesson_challenges"] = challenges
    STATE["session"] = Session(id=sid, lesson_id=lesson.id)
    STATE["challenge"] = challenges[0]
    STATE["step"] = 0""",
    """    if req.fresh:
        # A genuinely naive readout. The shipped checkpoint has already memorised the whole
        # curriculum, so with it loaded there is nothing to watch learn; this is the state the
        # demo wants when the point is to SEE the training happen.
        STATE["adapter"], STATE["readout_kind"] = _fresh_readout()
        STATE["checkpoint_status"] = "fresh (untrained)"
        STATE["fresh_brain"] = True
        STATE["dopamine"] = 0.0
        STATE["dopamine_total"] = 0
        STATE["window"] = []
        STATE["fly_answered"] = 0
        STATE["fly_correct"] = 0
        STATE["curve"] = []
    else:
        STATE["fresh_brain"] = False

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
    STATE["step"] = 0""",
)

# ---------------------------------------------------------------- 6. answer path
sub(
    """        fly_correct = int(action) == int(ch["correctIndex"])
        reward = 1.0 if fly_correct else -0.25
        info = adapter.observe(reward, logprob, f)""",
    """        fly_correct = int(action) == int(ch["correctIndex"])
        reward = 1.0 if fly_correct else -0.25
        info = adapter.observe(reward, logprob, f)
        # Dopamine is delivered AFTER the plasticity step, because it is the same signal: the
        # reward gates the weight update, and the trace is what the UI draws. Computing it here
        # keeps the drawn pulse and the applied gradient from ever being different numbers.
        dopamine = _pulse_dopamine(bool(fly_correct))""",
)

sub(
    """    STATE["fly_answered"] = STATE.get("fly_answered", 0) + 1
    STATE["fly_correct"] = STATE.get("fly_correct", 0) + int(fly_correct)""",
    """    STATE["fly_answered"] = STATE.get("fly_answered", 0) + 1
    STATE["fly_correct"] = STATE.get("fly_correct", 0) + int(fly_correct)

    # The rolling window is what makes learning visible: a cumulative average barely moves after
    # a hundred answers, so a curve built on it looks flat even while the model is improving.
    win = STATE.setdefault("window", [])
    win.append(int(fly_correct))
    if len(win) > LEARNING_WINDOW:
        del win[0]""",
)

# ---------------------------------------------------------------- 7. lesson advance
sub(
    """    if correct:
        if sess.index + 1 < len(STATE["lesson_challenges"]):
            sess.index += 1
            nxt = STATE["lesson_challenges"][sess.index]
        else:
            nxt = None
    else:
        nxt = ch
    STATE["challenge"] = nxt""",
    """    if correct:
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
    STATE["challenge"] = nxt""",
)

# ---------------------------------------------------------------- 8. the frame
sub(
    """        "streak": sess.streak if sess else 0,""",
    """        "streak": sess.streak if sess else 0,
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
        "entropy": float(STATE.get("last", {}).get("entropy", 0.0)),
        "grad_norm": float(STATE.get("last", {}).get("grad_norm", 0.0)),""",
)

# a tiny helper for the lesson title, used by the frame
sub(
    """def _dopamine_now() -> float:""",
    '''def _lesson_title(lesson_id: str | None) -> str | None:
    if not lesson_id:
        return None
    try:
        lesson = STATE["curriculum"].get_lesson(lesson_id)
    except Exception:
        return None
    return getattr(lesson, "title", None)


def _dopamine_now() -> float:''',
)

# ---------------------------------------------------------------- 9. training endpoint
sub(
    """class ControlReq(BaseModel):
    mode: str""",
    '''class TrainReq(BaseModel):
    fresh: bool = True


@app.post("/train")
def train(req: TrainReq) -> dict:
    """Reset the readout to untrained, or reload the shipped checkpoint.

    Exposed so the UI can offer "watch it learn" and "already trained" as a real choice rather
    than pretending the pretrained model is learning.
    """
    if req.fresh:
        STATE["adapter"], STATE["readout_kind"] = _fresh_readout()
        STATE["checkpoint_status"] = "fresh (untrained)"
        STATE["fresh_brain"] = True
    else:
        loaded, status = _load_readout(CHECKPOINT_PROMPT_INDEX)
        if loaded is None:
            return {"ok": False, "checkpoint_status": status}
        STATE["adapter"], STATE["readout_kind"] = loaded
        STATE["checkpoint_status"] = status
        STATE["fresh_brain"] = False

    STATE["dopamine"] = 0.0
    STATE["dopamine_total"] = 0
    STATE["window"] = []
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
        "parameters": int(STATE["adapter"].parameters()),
    }


class ControlReq(BaseModel):
    mode: str''',
)

sub("import math\n", "") if "import math\n" in s else None
if "\nimport math" not in s:
    sub("import json", "import json\nimport math", 1)

P.write_text(s, encoding="utf-8")
print(f"patched: {len(orig)} -> {len(s)} chars")
