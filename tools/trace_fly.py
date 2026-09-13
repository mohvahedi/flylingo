"""Dump the fly's per-frame behavior as a readable timeline, one line per sample.

The aggregate checks disagreed with the report: every arrival showed a reach (44/44), every shot
had the fly at its target with a tarsus on the card. So the question is no longer "does the reach
ever play" but "what does the sequence of an answer actually look like" -- whether the fly travels
to a new card, sits where it already was, or reaches somewhere other than the card it chose.

Prints one row per sample: animator time, behavior, whether it is travelling, the remaining
distance to its target, the reach extension, how many tarsi are on the chosen card, and the target.
A blank line marks each time the target changes, which is the moment a new card is chosen.

Usage:  python tools/trace_fly.py [seconds]
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
  const legs = p.legs || [];
  return {
    t: p.t, behavior: p.behavior, moving: p.moving, dist: p.dist,
    on: p.reach ? p.reach.on : 0, planted: p.reach ? p.reach.planted : 0,
    target: p.target, pos: p.pos, hasRect: p.hasRect,
    onCard: legs.filter((l) => l.onCard).length,
  };
}"""

with sync_playwright() as pw:
    br = pw.chromium.launch(args=["--use-gl=angle", "--enable-unsafe-swiftshader"])
    pg = br.new_page(viewport={"width": 1600, "height": 900})
    pg.goto(URL, wait_until="load", timeout=90000)
    for _ in range(150):
        if pg.evaluate("() => Boolean(window.__flyPos)"):
            break
        time.sleep(1)
    print("t      behavior   moving  dist    on    plant  onCard  target", flush=True)
    print("-" * 78, flush=True)
    prev = None
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        s = pg.evaluate(READ)
        if s:
            tgt = s["target"]
            shown = f"({tgt[0]:.2f},{tgt[1]:.2f},{tgt[2]:.2f})" if tgt else "-"
            if prev is not None and s["target"] != prev:
                print(f"        ---- target -> {shown} ----", flush=True)
            prev = s["target"]
            flag = ""
            if s["planted"] > 0.5 and s["onCard"] == 0:
                flag = "  <-- reaching but NO tarsus on the card"
            if s["planted"] > 0.5 and s["moving"]:
                flag = "  <-- planted while still travelling"
            print(f"{s['t']:6.2f} {s['behavior']:<10} {str(s['moving']):<7} "
                  f"{s['dist']:<7.3f} {s['on']:<5.2f} {s['planted']:<6.2f} "
                  f"{s['onCard']:<7} {shown}{flag}", flush=True)
        time.sleep(0.5)
    br.close()
