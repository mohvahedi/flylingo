"""Which card is the fly being sent to, moment by moment -- and is it the choice or the geometry
that keeps moving?

The trace showed the fly's target changing every 1-3 seconds, so it hops between cards and every
reach gets cut short instead of holding for the 2.8s it was tuned for. Two very different causes
produce that, and they need opposite fixes:

  * the CHOSEN CARD changes -- i.e. the choice the stage is given is unstable (a live readout
    flipping between two near-tied options flips the target with it); or
  * the chosen card is stable but its MEASURED POSITION moves -- the card rectangles are
    re-measured from the rendered screen and jitter or shift.

The probe publishes `cardWorld`: the world position the stage would fly to for each of the four
options. So the chosen index is recoverable as the card whose position the target matches. This
prints that index, the distance to it, and all four card positions, so the moving part is named.

Usage:  python tools/trace_choice.py [seconds]
"""
from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 90
URL = "http://127.0.0.1:3300/fly"

READ = """() => {
  const p = window.__flyPos;
  if (!p) return null;
  return { t: p.t, behavior: p.behavior, target: p.target, cardWorld: p.cardWorld || null };
}"""


def dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


with sync_playwright() as pw:
    br = pw.chromium.launch(args=["--use-gl=angle", "--enable-unsafe-swiftshader"])
    pg = br.new_page(viewport={"width": 1600, "height": 900})
    pg.goto(URL, wait_until="load", timeout=90000)
    for _ in range(150):
        if pg.evaluate("() => Boolean(window.__flyPos)"):
            break
        time.sleep(1)
    print("t      behavior   choice  d_to_choice   card0y    card1y    card2y    card3y",
          flush=True)
    print("-" * 82, flush=True)
    prev_idx = None
    prev_cards = None
    changes = 0
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        s = pg.evaluate(READ)
        if s and s["cardWorld"] and s["target"]:
            cards = s["cardWorld"]
            ds = [dist(s["target"], c) for c in cards]
            idx = min(range(len(ds)), key=lambda i: ds[i])
            mark = ""
            if prev_idx is not None and idx != prev_idx:
                changes += 1
                mark += f"   choice {prev_idx} -> {idx}"
            if prev_cards is not None:
                moved = max(abs(a[1] - b[1]) for a, b in zip(cards, prev_cards))
                if moved > 0.05:
                    mark += f"   cards moved {moved:.2f} in y"
            prev_idx, prev_cards = idx, cards
            print(f"{s['t']:6.2f} {s['behavior']:<10} {idx:<7} {ds[idx]:<13.4f} "
                  f"{cards[0][1]:<9.3f} {cards[1][1]:<9.3f} {cards[2][1]:<9.3f} "
                  f"{cards[3][1]:<9.3f}{mark}", flush=True)
        time.sleep(0.5)
    br.close()
    print(f"\nchoice changed {changes} times in {SECONDS}s")
