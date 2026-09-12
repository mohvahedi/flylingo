"""Drive every control mode from the browser and check the UI tells the truth.

This is the check that matters for the project's central claim: the app must never
let a viewer mistake a control condition for the intact connectome, and the no_edges
control must visibly produce an inert brain rather than a plausible-looking one.

Requires the API on 127.0.0.1:8770 and the Next app on 127.0.0.1:3100.
"""
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

API = "http://127.0.0.1:8770"
APP = "http://127.0.0.1:3100/lesson/fly"
OUT = Path(r"D:\Projects\flylingo\artifacts")
OUT.mkdir(parents=True, exist_ok=True)

MODES = ["intact", "shuffled", "random_graph", "no_edges"]
LABEL = {
    "intact": "INTACT CONNECTOME",
    "shuffled": "SHUFFLED CONTROL",
    "random_graph": "RANDOM GRAPH CONTROL",
    "no_edges": "NO EDGES CONTROL",
}


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


def lit_pixels(page):
    """Lit-pixel counts for the 2D chart canvases only.

    The panel now also contains the fly's WebGL canvas, which has no 2D context, so
    canvases are filtered to the ones a 2D context can be obtained from rather than
    indexed positionally.
    """
    return page.evaluate(
        """() => [...document.querySelectorAll('canvas')].map(c => {
             const ctx = c.getContext('2d');
             if (!ctx) return null;
             const d = ctx.getImageData(0,0,c.width,c.height).data;
             let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 8) n++;
             return {size: c.width + 'x' + c.height, lit: n};
           }).filter(Boolean)"""
    )


errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 950})
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(f"{m.type}: {m.text}")
            if m.type == "error" else None)

    page.goto(APP, wait_until="domcontentloaded")
    page.wait_for_timeout(9000)

    results = {}
    for mode in MODES:
        post("/control", {"mode": mode})
        page.wait_for_timeout(2500)

        body = page.inner_text("body").upper()
        badge_ok = LABEL[mode] in body
        # The telemetry the app is rendering from.
        t = get("/telemetry")
        lit = lit_pixels(page)
        results[mode] = {
            "badge_shown": badge_ok,
            "server_mode": t["mode"],
            "spikes": len(t["spikes"]),
            "active_fraction": round(t["active_fraction"], 6),
            "state_rms": round(t["state_rms"], 6),
            "canvas_lit_pixels": lit,
        }
        print(f"\n[{mode}]")
        print(f"  badge '{LABEL[mode]}' visible : {badge_ok}")
        print(f"  server mode                   : {t['mode']}")
        print(f"  spikes                        : {len(t['spikes'])}")
        print(f"  active_fraction               : {t['active_fraction']:.6f}")
        print(f"  state_rms                     : {t['state_rms']:.6f}")
        print(f"  canvas lit pixels             : {lit}")
        page.screenshot(path=str(OUT / f"mode-{mode}.png"), full_page=True)

    print("\n=== assertions ===")
    ok = True

    # Every mode must be labelled, and the label must match the server's mode.
    for mode in MODES:
        r = results[mode]
        if not r["badge_shown"]:
            print(f"  FAIL: {mode} badge not visible")
            ok = False
        if r["server_mode"] != mode:
            print(f"  FAIL: server mode is {r['server_mode']}, expected {mode}")
            ok = False

    # no_edges must be a genuinely dead brain: exactly zero everywhere, empty spikes.
    ne = results["no_edges"]
    if ne["state_rms"] != 0.0 or ne["active_fraction"] != 0.0 or ne["spikes"] != 0:
        print(f"  FAIL: no_edges is not inert: {ne}")
        ok = False
    else:
        print("  PASS: no_edges yields state_rms 0, active_fraction 0, 0 spikes")

    # The dead brain must render darker than the live ones, not equally bright.
    # Compare the sampled-neuron canvas (288x288), which is the one that renders state.
    def state_canvas_lit(shields):
        for c in shields:
            if c["size"].startswith("288x288"):
                return c["lit"]
        return 0

    live_lit = max(
        state_canvas_lit(results[m]["canvas_lit_pixels"])
        for m in ("intact", "shuffled", "random_graph")
    )
    dead_lit = state_canvas_lit(ne["canvas_lit_pixels"])
    print(f"  live canvas lit pixels (max): {live_lit} | no_edges: {dead_lit}")
    if dead_lit >= live_lit:
        print("  FAIL: the no_edges brain is not visibly darker than a live one")
        ok = False
    else:
        print("  PASS: no_edges renders visibly darker than the live modes")

    # Restore and leave it running.
    post("/control", {"mode": "intact"})
    print("\nrestored to intact")

    print("\n=== console errors ===")
    for e in errors[:15]:
        print("  ", e[:200])
    if not errors:
        print("   none")

    browser.close()

print("\nRESULT:", "all control-mode checks passed" if ok else "FAILURES PRESENT")
raise SystemExit(0 if ok else 1)
