"""Screenshot the phone scene and read back the screen's real world orientation.

A phone that came up sideways, mirrored or upside down would still pass every structural
check, so this measures the screen plane's world normal and up vector, and reports where the
fly was sent relative to the card it picked.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ART = Path(r"D:\Projects\flylingo\artifacts")
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5191/phone.html"
WAIT_MS = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
TAG = sys.argv[3] if len(sys.argv) > 3 else "a"

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1400, "height": 625}, device_scale_factor=1)
    errs = []
    pg.on("pageerror", lambda e: errs.append("PAGEERROR " + str(e)[:220]))
    pg.on("console", lambda m: errs.append(f"{m.type}: {m.text[:180]}") if m.type == "error" else None)

    pg.goto(BASE, wait_until="load", timeout=60000)
    pg.wait_for_timeout(WAIT_MS)

    probe = pg.evaluate("() => window.__phoneStage || null")
    behavior = pg.evaluate("() => window.__flyBehavior || null")
    print("screen normal   :", probe and probe.get("screenNormal"))
    print("screen up       :", probe and probe.get("screenUp"))
    print("rects           :", (probe or {}).get("rects"))
    print("fly behaviour   :", behavior)
    print("phone world h   :", probe and probe.get("phoneWorldH"))
    print()
    print("errors:", errs[:6])

    pg.screenshot(path=str(ART / f"phone-scene-{TAG}.png"))
    print("saved", ART / f"phone-scene-{TAG}.png")
    br.close()
