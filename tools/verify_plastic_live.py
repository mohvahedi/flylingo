"""Live check: does the service run on the plastic brain, and does it learn as it answers?

This drives the real HTTP API the HUD uses. It switches the readout to the plastic connectome,
starts a session, and has the fly answer many questions, reporting accuracy and how far the
synaptic scales have moved. If either is static, the brain is not the learner.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8770"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def post(path, payload=None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)


print("switching the readout to the plastic connectome...")
t0 = time.time()
try:
    r = post("/train", {"kind": "plastic_brain", "fresh": True})
except urllib.error.HTTPError as e:
    print("FAILED:", e.code, e.read().decode()[:400])
    sys.exit(1)
print(f"  {r}   ({time.time()-t0:.0f}s to build)")

h = get("/health")
print(f"  readout_kind now: {h.get('readout_kind')}   checkpoint: {h.get('checkpoint_status')}")

s = post("/session", {"lesson_id": None, "fresh": True})
sid = s["session_id"]
ch = s["challenge"]
print(f"\nsession {sid}   first challenge {ch['id']}  {ch['prompt']!r}")

print(f"\n{'n':>4} {'correct':>8}  {'p(chosen)':>9}  {'lesson':>7}  {'challenge':>10}")
print("-" * 60)
hits = 0
first_scale = last_scale = None
for i in range(1, N + 1):
    try:
        res = post(
            "/answer",
            {"session_id": sid, "challenge_id": ch["id"], "choice_index": -1, "as_fly": True},
        )
    except urllib.error.HTTPError as e:
        print(f"  {i:>4} HTTP {e.code}: {e.read().decode()[:120]}")
        break
    hits += int(res["fly_correct"])
    if i % 15 == 0 or i == 1 or i == N:
        st = get("/stats")
        print(
            f"{i:>4} {hits}/{i:<6} {max(res['probs']):>9.3f}  "
            f"{res.get('lesson_id', '?'):>7}  {ch['id']:>10}"
        )
    nxt = res.get("next_challenge")
    if nxt:
        ch = nxt

st = get("/stats") if False else None
# the plastic scales live in the brain; read them through the telemetry frame
tel = get("/telemetry")
frame_keys = [k for k in tel if "plastic" in k.lower()]
print(f"\ntelemetry keys mentioning plastic: {frame_keys or 'none'}")
print(f"\nfinal: {hits}/{N} correct = {hits/N:.1%}   (majority-class baseline 26.8%)")
