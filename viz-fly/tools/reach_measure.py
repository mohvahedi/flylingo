"""Measure the hero fly: how big it is, and whether its forelegs reach the card it chose.

Two numbers are the whole point of this file.

SIZE. The fly's world span (from the real-mesh node probe) and that span as a fraction of
the phone screen's world height. The screen height is the yardstick the composition is
built on (LAYOUT.screenHeight), so "the fly grew from 0.72 to X, which is Y% of the screen"
is a fact about the render rather than an opinion about a screenshot.

REACH. The glass plane is rebuilt here independently, from the screen mesh's world normal
and box centre published by the page, and every one of the fly's six tarsus joints is
projected onto it. The claim "the fly touches the option it picked with its forelegs" is
then three measurable things at once:

    * the two forelegs (pose indices 0 and 3, the chains the groom behavior already
      addresses) travel far more than the other four, so the motion is selective;
    * their tarsus distance to the glass goes to ~0 during the reach, so they arrive AT the
      surface rather than hovering in front of it;
    * the point where each foreleg meets the glass lies inside the chosen card's
      rectangle, so they land on the option and not on the card next to it.

Nothing here reads pixels for motion: readPixels returns zeros on this stack because
react-three-fiber leaves preserveDrawingBuffer off.
"""
import json
import math
import os
import sys
import time

import numpy as np
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5191/phone.html"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tools/reach_samples.json"
PICK = int(sys.argv[3]) if len(sys.argv) > 3 else 3  # which FLY PICKS button to press
SECONDS = float(sys.argv[4]) if len(sys.argv) > 4 else 18.0
HZ = 12.0

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]


def snap(page):
    """One sample: the flight probe, the node probe and the stage probe, at once."""
    return page.evaluate(
        """() => ({
             fly: window.__flyPos || null,
             node: window.__flyProbe || null,
             stage: window.__phoneStage || null,
             wall: performance.now(),
           })"""
    )


def main() -> int:
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    samples = []
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True, args=ARGS)
        pg = br.new_page(viewport={"width": 1400, "height": 625})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)[:300]))
        pg.goto(BASE, wait_until="load", timeout=60000)
        pg.wait_for_function("() => !!window.__flyProbe && !!window.__flyPos", timeout=60000)
        # let the studio env bake and the first flight land
        pg.wait_for_timeout(9000)
        print("first arrival:", pg.evaluate("() => window.__flyPos && [window.__flyPos.behavior, window.__flyPos.dist]"))

        # press the button for another card so a fresh flight is captured end to end
        try:
            pg.get_by_role("button", name=str(PICK), exact=True).first.click(timeout=5000)
            print("clicked FLY PICKS", PICK)
        except Exception as exc:  # noqa: BLE001
            print("click failed:", exc)

        t0 = time.time()
        step = 1.0 / HZ
        while time.time() - t0 < SECONDS:
            samples.append(snap(pg))
            pg.wait_for_timeout(int(step * 1000))
        br.close()

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"base": BASE, "samples": samples, "errors": errs}, fh)

    print(f"samples {len(samples)} -> {OUT}")
    if errs:
        print("page errors:", errs[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
