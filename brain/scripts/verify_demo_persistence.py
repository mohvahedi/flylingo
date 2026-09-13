"""Prove the demo persistence end to end, over HTTP, across a REAL process restart.

The unit tests drive `_boot()` directly. This exercises the thing a user actually does: start the
server, train the brain, stop the server, start it again, and check the fly still knows what it
learned. Two separate processes, so nothing can be carried in memory.

Prints the health status at each stage rather than asserting a hidden intermediate, because the
status string is the claim being checked.

Usage:  python scripts/verify_demo_persistence.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("D:/Projects/flylingo")
BRAIN = ROOT / "brain"
PY = BRAIN / ".venv" / "Scripts" / "python.exe"
PORT = 8771
BASE = f"http://127.0.0.1:{PORT}"


def get(path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def start_server():
    log = open(ROOT / "artifacts" / "persistence_server.log", "a")
    p = subprocess.Popen(
        [str(PY), "-m", "uvicorn", "brain.api:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(BRAIN), stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(120):
        try:
            get("/health")
            return p
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1)
    p.kill()
    raise SystemExit("server did not come up")


def stop_server(p):
    """Terminate and WAIT, so the shutdown hook's save is guaranteed to have run."""
    p.terminate()
    try:
        p.wait(timeout=180)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=30)


ckpt = BRAIN / "runs" / "plastic_brain" / "demo.npz"
print(f"checkpoint under test: {ckpt}")
print(f"exists before we start: {ckpt.exists()}")
if ckpt.exists():
    ckpt.unlink()
    print("removed it, so this measures a genuine first run")

print("\n--- process 1: fresh brain, train it, then stop ---")
srv = start_server()
try:
    health = get("/health")
    print("  /health readout_kind :", health["readout_kind"])
    print("  /health checkpoint   :", health["checkpoint_status"])

    sel = get("/train", method="POST", body={"kind": "plastic_brain", "fresh": True})
    print("  selected plastic brain, status:", sel["checkpoint_status"])
    s = get("/session", method="POST", body={"fresh": True})
    sid, ch = s["session_id"], s["challenge"]
    # Each answer advances the lesson, so the current challenge id changes. The response carries
    # the new one; reusing the old id is a 409 ("challenge is not current"), which is the guard
    # doing its job. Enough answers to cross PLASTIC_CHECKPOINT_EVERY updates, since the point is
    # that the checkpoint is written WITHOUT a graceful shutdown.
    for step in range(12):
        resp = get("/answer", method="POST",
                   body={"session_id": sid, "challenge_id": ch["id"],
                         "choice_index": -1, "as_fly": True})
        nxt = resp.get("next_challenge")
        if nxt is None:
            break
        ch = nxt
    t = get("/telemetry")
    updates_before = t["plastic_updates"]
    print(f"  trained: plastic_updates={updates_before}  scale_mean={t['plastic_scale_mean']:.6f}")
    assert updates_before > 0
finally:
    stop_server(srv)

print(f"\n  checkpoint written during training: {ckpt.exists()}  "
      f"({ckpt.stat().st_size if ckpt.exists() else 0} bytes)")
# NOT "written on shutdown": on Windows terminate() hard-kills the process, so uvicorn's shutdown
# never runs. This is a HARD KILL on purpose, because that is how a demo actually stops.
assert ckpt.exists(), (
    "no checkpoint after an abrupt stop. Either the automatic checkpoint during training did not "
    "run, or persistence depends on a graceful shutdown that Windows never delivers."
)
print("  (process was hard-killed, so this checkpoint came from the periodic save)")

print("\n--- process 2: a genuine restart ---")
srv = start_server()
try:
    health = get("/health")
    print("  /health checkpoint before selecting:", health["checkpoint_status"])

    # A restart must be able to RESUME. fresh=False is the resume switch; the plastic brain has to
    # be selected first because the server starts on the readout its own checkpoints name.
    sel = get("/train", method="POST", body={"kind": "plastic_brain", "fresh": False})
    print("  selected plastic brain, status :", sel["checkpoint_status"])

    t = get("/telemetry")
    print(f"  plastic_updates after restart: {t['plastic_updates']}")
    print(f"  scale_mean after restart     : {t['plastic_scale_mean']:.6f}")

    assert "resumed" in sel["checkpoint_status"], sel["checkpoint_status"]
    assert t["plastic_updates"] > 0, "the restart resumed nothing"

    # The resumed state is the LAST PERIODIC CHECKPOINT, not the final update, so a little progress
    # is expected to be missing. What must hold is that the loss is bounded by the checkpoint
    # interval plus the updates one answer performs (1 observe + 20 rehearsal steps = 21), rather
    # than being the whole run. Asserting exact equality here would be wrong: it would demand a
    # save on every single update.
    lost = updates_before - t["plastic_updates"]
    print(f"  updates lost to the checkpoint interval: {lost} of {updates_before} "
          f"({lost / updates_before:.1%})")
    assert 0 <= lost <= 100 + 21, f"lost {lost} updates, more than the checkpoint interval allows"
finally:
    stop_server(srv)

print("\nPASS: the fly resumed the run it had trained, across a real process restart,")
print("      with at most one checkpoint interval of progress at risk.")
