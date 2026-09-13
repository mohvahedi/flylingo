"""Screenshot the fly at the moment it reaches, so the complaint can be checked against pixels.

The probe-based check ("did the reach behavior play?") came back 44/44, so whatever the viewer is
seeing is not an absent reach: it is something about where the fly is or what is on screen when the
reach plays. The only way to tell is to look.

This loads the demo, waits for the reach to be at full extension (probe reach.planted >= 0.9), and
saves a PNG at each one, along with the numbers behind it (chosen card target, fly position, how
many tarsi the probe says are on the card). Reaching only counts once per arrival so the shots are
of distinct reaches rather than the same one twice.

    python tools/shoot_reaches.py [count]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = Path("artifacts/reach_shots")
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:3300/fly"

READ = """() => {
  const p = window.__flyPos;
  if (!p) return null;
  const legs = p.legs || [];
  const t = p.target, q = p.pos;
  const d = Math.hypot(t[0]-q[0], t[1]-q[1], t[2]-q[2]);
  return {
    planted: p.reach ? p.reach.planted : 0,
    on: p.reach ? p.reach.on : 0,
    behavior: p.behavior,
    moving: p.moving,
    dist: +d.toFixed(4),
    target: t, pos: q,
    onCard: legs.filter((l) => l.onCard).length,
    tarsi: legs.length,
    hasRect: p.hasRect,
  };
}"""

shots = []
with sync_playwright() as pw:
    br = pw.chromium.launch(args=["--use-gl=angle", "--enable-unsafe-swiftshader"])
    pg = br.new_page(viewport={"width": 1600, "height": 900})
    pg.goto(URL, wait_until="load", timeout=90000)
    for _ in range(150):
        if pg.evaluate("() => Boolean(window.__flyPos)"):
            break
        time.sleep(1)
    print("stage live; waiting for reaches", flush=True)

    last_target = None
    t0 = time.time()
    while len(shots) < COUNT and time.time() - t0 < 300:
        s = pg.evaluate(READ)
        if not s:
            time.sleep(0.2)
            continue
        # one shot per distinct arrival, taken at full extension
        key = tuple(s["target"])
        if s["planted"] >= 0.9 and key != last_target:
            last_target = key
            name = OUT / f"reach_{len(shots):02d}.png"
            pg.screenshot(path=str(name))
            s["file"] = name.name
            s["t"] = round(time.time() - t0, 2)
            shots.append(s)
            print(f"  shot {len(shots)}/{COUNT}: {name.name}  onCard={s['onCard']}/{s['tarsi']} "
                  f"dist_to_target={s['dist']}  behavior={s['behavior']}", flush=True)
        time.sleep(0.15)
    br.close()

(OUT / "shots.json").write_text(json.dumps(shots, indent=2), encoding="utf-8")
print(f"\n{len(shots)} screenshots in {OUT}")
print(json.dumps([{k: v for k, v in s.items() if k in
                   ("file", "onCard", "tarsi", "dist", "behavior", "hasRect")} for s in shots],
                 indent=1))
