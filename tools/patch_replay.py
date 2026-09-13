"""Wire rehearsal (experience replay) into the live service's answer path."""
from pathlib import Path

P = Path(r"D:\Projects\flylingo\brain\brain\api.py")
s = P.read_text(encoding="utf-8")


def sub(old: str, new: str) -> None:
    global s
    assert old in s, f"ANCHOR MISSING:\n{old[:240]}"
    s = s.replace(old, new, 1)


# ------------------------------------------------------------------ constants
sub(
    """#: Rolling window for the visible learning curve, in answers.
LEARNING_WINDOW = 20""",
    """#: Rolling window for the visible learning curve, in answers.
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
TRAIN_LR = 0.35""",
)

# ------------------------------------------------------------------ boot state
sub(
    """        #: recent fly outcomes, for the visible learning curve
        window=[],""",
    """        #: recent fly outcomes, for the visible learning curve
        window=[],
        #: experience replay buffer: (features, correct index) pairs the fly has seen
        replay_x=[],
        replay_y=[],
        #: how many rehearsal steps have been taken, shown in the UI
        rehearsals=0,""",
)

# ------------------------------------------------------------------ the answer path
sub(
    """        fly_correct = int(action) == int(ch["correctIndex"])
        reward = 1.0 if fly_correct else -0.25
        info = adapter.observe(reward, logprob, f)""",
    """        fly_correct = int(action) == int(ch["correctIndex"])
        reward = 1.0 if fly_correct else -0.25
        info = adapter.observe(reward, logprob, f)

        # Rehearsal. The reward-only step above reinforces the sampled action and nothing more,
        # which measured as a plateau near 45%; these steps learn from the lesson's answer key
        # against remembered examples, which measured as reaching 92-100% within a session.
        if STATE.get("training", True) and STATE.get("readout_kind") == "prompt_index":
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
                STATE["rehearsals"] = int(STATE.get("rehearsals", 0)) + REPLAY_STEPS""",
)

# ------------------------------------------------------------------ frame fields
sub(
    """        "entropy": float(STATE.get("last", {}).get("entropy", 0.0)),
        "grad_norm": float(STATE.get("last", {}).get("grad_norm", 0.0)),""",
    """        "entropy": float(STATE.get("last", {}).get("entropy", 0.0)),
        "grad_norm": float(STATE.get("last", {}).get("grad_norm", 0.0)),
        "rehearsals": int(STATE.get("rehearsals", 0)),
        "replay_size": len(STATE.get("replay_x", [])),
        "training": bool(STATE.get("training", True)),""",
)

# ------------------------------------------------------------------ train endpoint
sub(
    """class TrainReq(BaseModel):
    fresh: bool = True""",
    """class TrainReq(BaseModel):
    fresh: bool = True
    training: bool | None = None
    lr: float | None = None
    replay_steps: int | None = None""",
)

sub(
    """    STATE["dopamine"] = 0.0
    STATE["dopamine_total"] = 0
    STATE["window"] = []
    STATE["fly_answered"] = 0
    STATE["fly_correct"] = 0
    STATE["curve"] = []
    STATE["step"] = 0
    STATE["last"] = {}""",
    """    global TRAIN_LR, REPLAY_STEPS
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
    STATE["rehearsals"] = 0
    STATE["fly_answered"] = 0
    STATE["fly_correct"] = 0
    STATE["curve"] = []
    STATE["step"] = 0
    STATE["last"] = {}""",
)

sub(
    """        "readout_kind": STATE.get("readout_kind"),
        "parameters": int(STATE["adapter"].parameters()),
    }""",
    """        "readout_kind": STATE.get("readout_kind"),
        "parameters": int(STATE["adapter"].parameters()),
        "training": bool(STATE.get("training", True)),
        "lr": TRAIN_LR,
        "replay_steps": REPLAY_STEPS,
    }""",
)

# session start also resets the buffer
sub(
    """    if req.fresh:
        # A genuinely naive readout. The shipped checkpoint has already memorised the whole""",
    """    if req.fresh:
        # A genuinely naive readout. The shipped checkpoint has already memorised the whole""",
)
sub(
    """        STATE["fly_answered"] = 0
        STATE["fly_correct"] = 0
        STATE["curve"] = []
    else:
        STATE["fresh_brain"] = False""",
    """        STATE["fly_answered"] = 0
        STATE["fly_correct"] = 0
        STATE["curve"] = []
        STATE["replay_x"] = []
        STATE["replay_y"] = []
        STATE["rehearsals"] = 0
    else:
        STATE["fresh_brain"] = False""",
)

P.write_text(s, encoding="utf-8")
print(f"wired rehearsal: {len(s)} chars")
