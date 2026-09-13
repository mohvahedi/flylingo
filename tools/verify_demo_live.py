"""Drive the demo the way the HUD does, and confirm the fly's own choices advance the lesson."""
import json
import urllib.request

B = "http://127.0.0.1:8770"


def post(path, body=None):
    req = urllib.request.Request(
        B + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=120))


def get(path):
    return json.load(urllib.request.urlopen(B + path, timeout=120))


s = post("/session", {"fresh": False})
print("health before:", get("/health")["readout_kind"], "|", get("/health")["checkpoint_status"])
sid = s.get("session_id") or s.get("id")

before = get("/telemetry")
print("challenge before:", before.get("challenge_id"), "| step:", before.get("step"),
      "| mode:", before.get("mode"), "| chosen:", before.get("chosen"))

for i in range(12):
    t = get("/telemetry")
    cid = t.get("challenge_id")
    if cid is None:
        print("  no challenge:", list(t.keys())[:14])
        break
    r = post("/answer", {"session_id": sid, "challenge_id": cid, "choice_index": 0, "as_fly": True})
    picked = r.get("choice_index", r.get("driving_choice"))
    if i < 4 or i == 11:
        print(f"  answer {i+1:2d}: fly chose {picked}, correct={r.get('correct')}, "
              f"next challenge={r.get('next_challenge_id')}")

after = get("/telemetry")
print("\nchallenge after:", after.get("challenge_id"), "| step:", after.get("step"),
      "| ticks:", after.get("ticks"), "| active_fraction:", after.get("active_fraction"))
h = get("/health")
print("health after:", h["readout_kind"], "|", h["checkpoint_status"])
