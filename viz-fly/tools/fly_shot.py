"""Screenshot the hero scene in a chosen state, and print the probe markers with it.

Usage: python tools/fly_shot.py OUT.png [PICK] [SETTLE_MS] [?qs]
   PICK   FLY PICKS button to press (1-4), or 0 to leave the fly wherever it lands
   SETTLE ms to wait after the click before the shot
   QS    extra query string for the page, e.g. 'noshadow'
"""
import sys
import time

from playwright.sync_api import sync_playwright

OUT = sys.argv[1] if len(sys.argv) > 1 else "shot.png"
PICK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SETTLE = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
QS = sys.argv[4] if len(sys.argv) > 4 else ""
URL = "http://localhost:5191/phone.html" + (f"?{QS}" if QS else "")

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]

with sync_playwright() as p:
    br = p.chromium.launch(headless=True, args=ARGS)
    pg = br.new_page(viewport={"width": 1400, "height": 625})
    pg.goto(URL, wait_until="load", timeout=60000)
    pg.wait_for_function("() => !!window.__flyProbe && !!window.__flyPos", timeout=60000)
    pg.wait_for_timeout(10000)
    if PICK:
        pg.get_by_role("button", name=str(PICK), exact=True).first.click(timeout=5000)
        print("clicked FLY PICKS", PICK)
    t0 = time.time()
    while (time.time() - t0) * 1000 < SETTLE:
        pg.wait_for_timeout(500)
    pg.screenshot(path=OUT)
    st = pg.evaluate(
        """() => ({fly: window.__flyPos, node: window.__flyProbe, stage: window.__phoneStage})"""
    )
    print("saved", OUT)
    print("behavior", st["fly"]["behavior"], "t", st["fly"]["t"], "dist", st["fly"]["dist"])
    print("span", st["node"]["span"] if st["node"] else None)
    br.close()
