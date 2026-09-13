"""Watch every answer the demo makes, and check the fly actually reaches for the card it picked.

The bug being chased: for some answers the fly does not travel to its chosen card and put its
forelegs on it. The suspect is the flight trigger in PhoneStage, which restarts only when the
card's world position CHANGES:

    if (!lastTarget.current.equals(target.current)) { ...start a new flight... }

Two consecutive answers on the SAME option produce an identical target, so no new flight starts,
`moving` stays false, the dwell counter keeps running from the previous arrival, and the behavior
stays on 'groom' -- the fly never reaches. With four options that is roughly a quarter of answers,
and more while the model is still near chance and repeating itself.

The page publishes its own state every frame (PhoneStage sets window.__flyPos), including
`reach.on` / `reach.planted`, the flight target and distance, and where each tarsus lands against
the card. This polls that, segments the run into ARRIVALS (contiguous spells where the fly is not
travelling), and reports per arrival whether the reach actually played.

No verdict is pre-written: it prints every arrival and the tally.

Usage:  python tools/check_fly_reaches.py [seconds]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 180
URL = "http://127.0.0.1:3300/fly"

READ = """() => {
  const p = window.__flyPos;
  if (!p) return null;
  const legs = p.legs || [];
  return {
    t: p.t,
    behavior: p.behavior,
    moving: p.moving,
    dist: p.dist,
    target: p.target,
    hasRect: p.hasRect,
    reachOn: p.reach ? p.reach.on : 0,
    planted: p.reach ? p.reach.planted : 0,
    onCard: legs.filter((l) => l.onCard).length,
    tarsi: legs.length,
  };
}"""

print(f"watching {URL} for {SECONDS}s\n", flush=True)
with sync_playwright() as pw:
    br = pw.chromium.launch(args=["--use-gl=angle", "--enable-unsafe-swiftshader"])
    pg = br.new_page(viewport={"width": 1600, "height": 900})
    pg.goto(URL, wait_until="load", timeout=90000)

    for _ in range(150):
        if pg.evaluate("() => Boolean(window.__flyPos)"):
            break
        time.sleep(1)
    else:
        print("no __flyPos after 150s: the stage never started", flush=True)
        sys.exit(2)
    print("stage is live, sampling...\n", flush=True)

    samples = []
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        s = pg.evaluate(READ)
        if s:
            s["wall"] = round(time.time() - t0, 2)
            samples.append(s)
        time.sleep(0.2)
    br.close()

if not samples:
    print("no samples collected")
    sys.exit(2)


def key(target):
    return tuple(target) if target else None


# --- segment into ARRIVALS: a spell of not-travelling, ended by the fly starting to travel ----
arrivals = []
cur = None
for s in samples:
    if s["moving"]:
        if cur is not None:
            arrivals.append(cur)
            cur = None
        continue
    if cur is None:
        cur = {"target": key(s["target"]), "samples": 0, "max_reach": 0.0,
               "max_planted": 0.0, "max_on_card": 0, "behaviors": set(), "tarsi": s["tarsi"],
               "start": s["wall"]}
    cur["samples"] += 1
    cur["max_reach"] = max(cur["max_reach"], s["reachOn"])
    cur["max_planted"] = max(cur["max_planted"], s["planted"])
    cur["max_on_card"] = max(cur["max_on_card"], s["onCard"])
    cur["behaviors"].add(s["behavior"])
    # a long pause between samples means a stalled tab, not a continuous arrival
    if s["wall"] - cur["start"] > 60:
        arrivals.append(cur)
        cur = None
if cur is not None:
    arrivals.append(cur)

print(f"{len(samples)} samples over {SECONDS}s -> {len(arrivals)} arrivals\n", flush=True)
print(f"{'#':>3} {'target':>26} {'secs':>5} {'reachOn':>8} {'planted':>8} {'onCard':>7}  "
      f"behaviors", flush=True)
print("-" * 100, flush=True)

missing = []
for i, a in enumerate(arrivals):
    tgt = (f"({a['target'][0]:.2f},{a['target'][1]:.2f},{a['target'][2]:.2f})"
           if a["target"] else "-")
    secs = a["samples"] * 0.2
    planted = a["max_planted"]
    tag = ""
    if planted < 0.5:
        missing.append(i)
        tag = "   <-- NO REACH"
    print(f"{i:>3} {tgt:>26} {secs:>5.1f} {a['max_reach']:>8.3f} {planted:>8.3f} "
          f"{a['max_on_card']:>7}  {','.join(sorted(a['behaviors']))}{tag}", flush=True)

reached = sum(1 for a in arrivals if a["max_planted"] >= 0.5)
print(f"\narrivals where the fly reached for its card: {reached}/{len(arrivals)}", flush=True)
if missing:
    print(f"arrivals with NO reach: {missing}", flush=True)
else:
    print("every arrival showed the fly reaching for its card", flush=True)

Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/fly_reach_check.json").write_text(
    json.dumps({"seconds": SECONDS, "samples": len(samples), "arrivals": len(arrivals),
                "reached": reached, "no_reach_arrivals": missing,
                "detail": [{k: (sorted(v) if isinstance(v, set) else v) for k, v in a.items()}
                           for a in arrivals]},
               indent=2),
    encoding="utf-8",
)
print("wrote artifacts/fly_reach_check.json", flush=True)
