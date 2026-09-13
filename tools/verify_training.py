"""Does a fresh brain actually LEARN, and does the course actually advance?

This is the verification that matters for the user's request ("it should be real training").
If the readout does not measurably improve from a naive start, or the lesson does not move off
the first one, the honest answer is "it does not work" and that is what this prints.
"""
import json
import sys
import urllib.request

API = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8770"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60


def post(path, payload):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(API + path, timeout=60) as r:
        return json.load(r)


# ---- reset to an untrained brain -------------------------------------------------
print("=== reset to a fresh, untrained readout ===")
t = post("/train", {"fresh": True})
print(json.dumps(t, indent=1))
assert t["ok"] and t["fresh_brain"], "reset did not produce a fresh brain"

s = post("/session", {"fresh": True})
print("session:", s["lesson_id"], "| first challenge:", s["challenge"]["prompt"])

# ---- answer repeatedly, recording the curve --------------------------------------
start_lesson = s["lesson_id"]
lessons_seen = [start_lesson]
curve = []
first20 = []
last20 = []

for i in range(N):
    ch = s["challenge"]
    # Answer with the SAME choice the fly made, so the user's side also progresses and the
    # lesson is not blocked retrying a wrong answer.
    pick = ch["correctIndex"]
    r = post(
        "/answer",
        {"session_id": s["session_id"], "challenge_id": ch["id"], "choice_index": pick},
    )
    curve.append((i + 1, r["fly_correct"], r["fly_accuracy"]))
    nxt = r.get("next_challenge")
    if nxt is None:
        print(f"  !! ran out of challenges at answer {i+1}")
        break
    s["challenge"] = nxt
    if nxt.get("lesson_id") and nxt["lesson_id"] not in lessons_seen:
        lessons_seen.append(nxt["lesson_id"])

    if i < 20:
        first20.append(int(r["fly_correct"]))
    last20.append(int(r["fly_correct"]))

# ---- the verdict -----------------------------------------------------------------
print()
print(f"=== after {len(curve)} answers ===")
final = get("/health")
print("fly answered      :", len(curve))
acc_first = sum(first20) / max(1, len(first20))
acc_last = sum(last20[-20:]) / max(1, len(last20[-20:]))
print(f"fly accuracy, first 20 : {acc_first:.3f}")
print(f"fly accuracy, last  20 : {acc_last:.3f}")
print(f"change                 : {acc_last - acc_first:+.3f}")
print()
print("lessons visited   :", len(lessons_seen), lessons_seen[:6], "..." if len(lessons_seen) > 6 else "")
print("started at        :", start_lesson)
print("ended at          :", s["challenge"].get("lesson_id", "?"))
print()
print("REAL TRAINING:", "YES, accuracy improved" if acc_last > acc_first + 0.10 else
      ("unchanged" if abs(acc_last - acc_first) <= 0.10 else "NO, it got worse"))
print("PROGRESSION  :", "YES, the course advanced" if len(lessons_seen) > 1 else
      "NO sign of advancing past the first lesson")

print()
print("per-answer fly outcome (1 = right):")
print("  " + "".join("1" if c else "." for _, c, _ in curve))
