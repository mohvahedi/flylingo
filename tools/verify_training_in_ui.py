"""End-to-end: does the TRAINING PANEL in the browser actually show learning?

The point of this project is that you can watch the fly learn. That claim is only met if the
numbers in the HUD move when answers happen, and if a fresh brain visibly climbs. This drives
the real service through the real API while the real page is open, and reads the panel out of
the DOM before and after, so the check is against what a viewer sees rather than against a log.
"""
import json
import re
import urllib.request

from playwright.sync_api import sync_playwright

API = "http://127.0.0.1:8770"
APP = "http://127.0.0.1:3300/fly"


def post(path, payload):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def read_panel(pg):
    """Pull the learning panel's numbers out of the rendered DOM."""
    txt = pg.evaluate("document.body.innerText")
    def grab(label, pattern):
        m = re.search(pattern, txt, re.I)
        return m.group(1).strip() if m else None
    return {
        "accuracy": grab("acc", r"fly accuracy[^\n]*\n[^\n]*?(\d{1,3})%"),
        "dopamine_pulses": grab("dop", r"(\d+)\s*pulses"),
        "rehearsals": grab("reh", r"rehearsals\s*\n?\s*([\d,]+)"),
        "learnt": grab("lea", r"learnt\s*\n?\s*([\d/]+)"),
        "lesson": grab("les", r"lesson\s+(\d+\s*/\s*\d+)"),
        "course": grab("crs", r"course\s*\n?\s*([^\n]{0,40})"),
        "state": "from scratch" if "from scratch" in txt.lower() else "pretrained",
    }


def answer_n(n, pg):
    s = post("/session", {"lesson_id": "u1l1"})
    ch = s["challenge"]
    for i in range(n):
        r = post(
            "/answer",
            {
                "session_id": s["session_id"],
                "challenge_id": ch["id"],
                "choice_index": ch["correctIndex"],
            },
        )
        nxt = r.get("next_challenge")
        if nxt is None:
            break
        ch = nxt
    return n


with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:150]))
    pg.goto(APP, wait_until="load", timeout=60000)
    pg.wait_for_timeout(18000)

    # ---- reset to a naive brain and reload so the panel reflects it ----
    t = post("/train", {"fresh": True, "replay_steps": 60})
    print(f"brain reset: fresh={t['fresh_brain']} params={t['parameters']} "
          f"lr={t['lr']} replay/answer={t['replay_steps']}")
    pg.reload(wait_until="load")
    pg.wait_for_timeout(16000)

    before = read_panel(pg)
    print()
    print("BEFORE any answers, as the panel shows it:")
    for k, v in before.items():
        print(f"   {k:16s}: {v}")

    # ---- answer, in three waves, reading the panel between them ----
    marks = []
    for wave in range(3):
        answer_n(40, pg)
        pg.wait_for_timeout(2500)
        snap = read_panel(pg)
        marks.append(snap)
        print()
        print(f"after {(wave + 1) * 40} answers:")
        for k, v in snap.items():
            print(f"   {k:16s}: {v}")

    pg.screenshot(path=r"D:\Projects\flylingo\artifacts\hud-training-live.png")

    print()
    print("=" * 68)
    print("VERDICT")
    first, last = marks[0]["accuracy"], marks[-1]["accuracy"]
    print(f"  accuracy shown in the panel: {first} -> {last}")
    print(f"  rehearsals:                 {before['rehearsals']} -> {last and marks[-1]['rehearsals']}")
    print(f"  dopamine pulses:            {before['dopamine_pulses']} -> {marks[-1]['dopamine_pulses']}")
    print(f"  learnt:                     {before['learnt']} -> {marks[-1]['learnt']}")
    print(f"  course advanced:            {before['lesson']} -> {marks[-1]['lesson']}")
    print(f"  console errors:             {errs[:3] if errs else 'none'}")
    br.close()
