"""Sweep the reach's numbers in a live page and report what each one does to the tarsi.

The reach has several dials and the fly is a scanned mesh whose leg axes are not documented,
so guessing the sign of a dial from the pose vocabulary is a coin flip. This drives the real
page instead: it plants the fly on its card with the reach held open (window.__REACH.hold is
set huge, which freezes the pose at full extension and defers the handover), then moves one
dial at a time and reads back, for every tarsus, the perpendicular distance to the glass and
where the tarsus sits in the image. Negative d = behind the glass, 0 = on it. The image
deltas say whether the motion is visible from the camera at all.

Two passes: the leg dials are swept with no body lean, so the standing feet are off the glass
and what they travel is their own doing; the body dials are swept on top of the leg base.

Usage: python tools/reach_sweep.py [URL] [CARD] [OUT.json]
"""
import json
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5191/phone.html"
CARD = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = sys.argv[3] if len(sys.argv) > 3 else "tools/reach_sweep.json"

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]

# what the reach does without any body movement: the legs have to earn their own way
LEG_BASE = {
    "extend": 0.36,
    "hold": 1e6,  # frozen open for the whole sweep
    "settle": 0.55,
    "femur": -0.55,
    "knee": -0.35,
    "tarsus": 0.3,
    "hipYaw": -0.18,
    "hipRoll": 0.55,
    "bodyPitch": 0,
    "bodyRoll": 0,
    "lean": 0,
    "drop": 0,
    "tap": 0,
    "tapHz": 9,
    "headPitch": 0,
}

LEG_PROBES = [
    ("femur", -1.1),
    ("femur", -1.7),
    ("femur", 0.4),
    ("knee", 1.0),
    ("knee", -1.2),
    ("tarsus", -0.4),
    ("tarsus", 0.9),
    ("hipYaw", -0.6),
    ("hipYaw", 0.4),
    ("hipRoll", -0.55),
    ("hipRoll", 1.1),
    ("hipRoll", 1.6),
]

BODY_PROBES = [
    ("lean", 0.2),
    ("lean", 0.35),
    ("lean", 0.5),
    ("drop", 0.25),
    ("bodyRoll", 0.35),
    ("bodyPitch", 0.4),
]


def sample(pg):
    return pg.evaluate(
        """() => {
             const f = window.__flyPos, n = window.__flyProbe, px = window.__flyPixel;
             const legs = f.legs || [];
             return {
               t: f.t, behavior: f.behavior, reach: f.reach, rotY: f.rotY, span: n.span,
               d: legs.map(l => +l.d.toFixed(4)),
               onCard: legs.map(l => l.onCard),
               pp: legs.map(l => [+l.px.toFixed(0), +l.py.toFixed(0)]),
               img: (n.foot || []).map(p => px(p[0], p[1], p[2])),
               foot: (n.foot || []).map(p => p.map(v => +v.toFixed(3))),
             };
           }"""
    )


def main() -> int:
    log = {}
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True, args=ARGS)
        pg = br.new_page(viewport={"width": 1400, "height": 625})
        pg.goto(URL, wait_until="load", timeout=60000)
        pg.wait_for_function("() => !!window.__flyProbe && !!window.__flyPos", timeout=60000)
        pg.wait_for_timeout(9000)
        # the stance before any reach: what the legs do with no behavior driving them
        log["neutral"] = sample(pg)
        pg.get_by_role("button", name=str(CARD), exact=True).first.click(timeout=5000)
        print("picked card", CARD)
        pg.evaluate("() => { window.__BODY_BASE = null; }")
        pg.evaluate("(b) => Object.assign(window.__REACH, b)", LEG_BASE)
        pg.wait_for_function(
            "() => window.__flyPos.behavior === 'reach' && window.__flyPos.reach.on > 0.99",
            timeout=240000,
        )
        print("planted at t =", pg.evaluate("() => window.__flyPos.t"))
        steps = []

        def step(label, patch):
            pg.evaluate("(b) => Object.assign(window.__REACH, b)", patch)
            # let the pose settle for a few frames: animator time advances <= 0.05 s per frame
            t0 = pg.evaluate("() => window.__flyPos.t")
            deadline = time.time() + 60
            while time.time() < deadline:
                pg.wait_for_timeout(300)
                if pg.evaluate("() => window.__flyPos.t") - t0 >= 0.12:
                    break
            s = sample(pg)
            # average two more reads a moment apart, so the idle foot jitter averages out
            pg.wait_for_timeout(200)
            s2 = sample(pg)
            s["d"] = [(a + b) / 2 for a, b in zip(s["d"], s2["d"])]
            s["img"] = [[(a + b) / 2 for a, b in zip(p1, p2)] for p1, p2 in zip(s["img"], s2["img"])]
            steps.append({"label": label, "patch": patch, "s": s})
            return s

        leg_base = step("leg-base", LEG_BASE)
        for name, value in LEG_PROBES:
            step(f"{name}={value}", {**LEG_BASE, name: value})
        # body dials on top of the leg base
        for name, value in BODY_PROBES:
            step(f"{name}={value}", {**LEG_BASE, name: value})
        log["steps"] = steps
        br.close()

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)

    b = leg_base

    def line(lbl, s, ref):
        parts = [f"{lbl:>14}"]
        for i in (0, 3):
            parts.append(
                f"L{i} d {s['d'][i]:+.3f} ({s['d'][i] - ref['d'][i]:+.3f}) "
                f"img {s['img'][i][0] - ref['img'][i][0]:+5.1f},{s['img'][i][1] - ref['img'][i][1]:+5.1f}"
            )
        parts.append("others " + " ".join(f"{s['d'][i] - ref['d'][i]:+.2f}" for i in (1, 2, 4, 5)))
        parts.append("onCard " + "".join("1" if c else "." for c in s["onCard"]))
        return "  ".join(parts)

    print("\nneutral stance (no behavior):", [round(v, 3) for v in log["neutral"]["d"]])
    print("deltas below are against leg-base =", [round(v, 3) for v in b["d"]])
    for st in steps[1:]:
        print(line(st["label"], st["s"], b))
    print("\nwritten", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
