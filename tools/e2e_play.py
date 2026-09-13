"""End-to-end lesson play: answers correctly by looking up the real correct index.

Confirms the full loop against the live service: session start, a correct answer,
advancement to the next challenge, the wrong-then-retry path (which previously
returned HTTP 409), and that the fly panel and canvases are live.

Requires the API and the Next app. Ports resolve via tools/flylingo_env.py, because
Windows re-randomises its Hyper-V excluded port ranges on every boot and a hardcoded
port can become unbindable without warning.
"""
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flylingo_env import app_url

API = "http://127.0.0.1:8770"
APP = app_url("/lesson/fly")
OUT = Path(r"D:\Projects\flylingo\artifacts")
OUT.mkdir(parents=True, exist_ok=True)


def get(path):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.load(r)


def post(path, body):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


# Map every challenge id to its correct option index, from the real curriculum.
cur = get("/curriculum")
answer_of = {}
for unit in cur["units"]:
    for lesson in unit["lessons"]:
        for ch in lesson["challenges"]:
            answer_of[ch["id"]] = ch["correctIndex"]

print(f"curriculum: {cur['course']}, {len(cur['units'])} units, "
      f"{sum(len(u['lessons']) for u in cur['units'])} lessons, "
      f"{len(answer_of)} challenges")

errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 950})
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(f"{m.type}: {m.text}")
            if m.type == "error" else None)

    page.goto(APP, wait_until="domcontentloaded")
    page.wait_for_timeout(9000)

    def current_challenge_id():
        return get("/telemetry")["challenge_id"]

    def click_button(label, timeout=8000):
        btn = page.get_by_role("button", name=label, exact=True)
        if btn.count() and btn.first.is_visible():
            btn.first.click(timeout=timeout)
            return True
        return False

    def click_correct():
        """Advance if needed, then answer the current challenge correctly."""
        # After a correct answer the clone's footer shows "Next"; click it to advance
        # the UI to the challenge the server has already moved on to.
        for _ in range(3):
            if click_button("Next"):
                page.wait_for_timeout(900)
                break
            break

        cid = current_challenge_id()
        idx = answer_of.get(cid)
        if idx is None:
            return None, None
        cards = page.locator("div.cursor-pointer")
        if cards.count() <= idx:
            return cid, None
        cards.nth(idx).click()
        page.wait_for_timeout(300)
        if not click_button("Check"):
            return cid, None
        page.wait_for_timeout(2200)
        return cid, idx

    print("\n=== answering correctly through the lesson ===")
    seen = []
    for i in range(6):
        cid, idx = click_correct()
        if cid is None:
            print(f"  {i+1}: no current challenge (lesson finished?)")
            break
        seen.append(cid)
        body = page.inner_text("body")
        picked = ""
        for line in body.splitlines():
            if "fly picked" in line:
                picked = line.strip()
                break
        print(f"  {i+1}: {cid} clicked option {idx} | {picked}")

    st = get("/stats")
    curve = st["learning_curve"]
    print(f"\nanswered: {len(seen)} | distinct challenges advanced: {len(set(seen))}")
    print(f"server recorded {len(curve)} answers")
    if curve:
        last = curve[-1]
        print(f"final: user accuracy {st['accuracy']*100:.0f}%, "
              f"fly accuracy {last.get('fly_accuracy', 0)*100:.0f}%, "
              f"mode {last['mode']}")
        print("all responses were correct:",
              all(c["correct"] for c in curve))

    # The retry path that used to 409: answer wrong, then retry the same challenge.
    print("\n=== wrong answer then retry (the path that returned 409) ===")
    cid = current_challenge_id()
    idx = answer_of.get(cid)
    wrong = (idx + 1) % 4
    cards = page.locator("div.cursor-pointer")
    cards.nth(wrong).click()
    page.wait_for_timeout(300)
    click_button("Check")
    page.wait_for_timeout(2200)
    print("  after wrong answer, server challenge:", current_challenge_id(),
          "| unchanged:", current_challenge_id() == cid)
    click_button("Retry")
    page.wait_for_timeout(600)
    cid2, idx2 = click_correct()
    print("  retried and answered correctly:", cid2 == cid)
    status_codes = [e for e in errors if "409" in e]
    print("  409 errors seen:", status_codes if status_codes else "none")

    # Panel liveness and canvas painting.
    body = page.inner_text("body")
    print("\n=== panel ===")
    print("  stream live:", "LIVE" in body.upper())
    print("  mode badge shown:", "INTACT CONNECTOME" in body.upper())
    painted = page.evaluate(
        """() => [...document.querySelectorAll('canvas')].map(c => {
             const ctx = c.getContext('2d');
             if (!ctx) return 'no-2d';
             const d = ctx.getImageData(0,0,c.width,c.height).data;
             let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 8) n++;
             return {size: c.width + 'x' + c.height, lit: n};
           })"""
    )
    print("  canvases:", painted)
    page.screenshot(path=str(OUT / "final-lesson.png"), full_page=True)

    print("\n=== console errors ===")
    for e in errors[:15]:
        print("  ", e[:200])
    if not errors:
        print("   none")

    browser.close()

print("\nartifacts:", OUT)
