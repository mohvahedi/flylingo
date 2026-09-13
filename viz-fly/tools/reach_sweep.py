"""Sweep the reach's numbers in a live page and report what each one does to the tarsi.

The reach has nine dials and the fly is a scanned mesh whose leg axes are not documented, so
guessing signs from the pose vocabulary is a coin flip. This drives the real page: it plants
the fly on its card with the reach held open (window.__REACH.hold is set huge, which both
freezes the pose at full extension and defers the handover), then moves one dial at a time and
reads back, for every tarsus, the perpendicular distance to the glass and where the tarsus sits
in the image. Positive d = in front of the glass. dy in image pixels tells whether the motion
is visible at all.

Usage: python tools/reach_sweep.py [URL] [CARD]
"""
import json
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5191/phone.html"
CARD = int(sys.argv[2]) if len(sys.argv) > 2 else 3

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]

BASE = {
    "extend": 0.36,
    "hold": 1.15,
    "settle": 0.55,
    "femur": -0.55,
    "knee": -0.35,
    "tarsus": 0.3,
    "hipYaw": -0.18,
    "hipRoll": 0.55,
    "bodyPitch": 0.16,
    "bodyRoll": 0.1,
    "lean": 0.34,
    "drop": 0.07,
    "tap": 0.07,
    "tapHz": 9,
    "headPitch": 0.3,
}

# one dial at a time, away from the base
PROBES = [
    ("femur", -1.1),
    ("femur", -1.7),
    ("knee", 0.4),
    ("knee", 1.0),
    ("tarsus", -0.4),
    ("hipYaw", 0.4),
    ("hipYaw", -0.6),
    ("hipRoll", -0.55),
    ("hipRoll", 1.1),
    ("bodyRoll", 0.5),
    ("bodyPitch", 0.5),
    ("drop", 0.3),
    ("lean", 0.8),
]


def sample(pg):
    return pg.evaluate(
        """() => {
             const f = window.__flyPos, n = window.__flyProbe;
             const px = window.__flyPixel;
             const legs = f.legs || [];
             return {
               t: f.t, behavior: f.behavior, reach: f.reach,
               d: legs.map(l => l.d),
               onCard: legs.map(l => l.onCard),
               px: legs.map(l => l.px.toFixed(0)),
               py: legs.map(l => l.py.toFixed(0)),
               img: (n.foot || []).map(p => px(p[0], p[1], p[2])),
               foot: n.foot,
             };
           }"""
    )


def main() -> int:
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True, args=ARGS)
        pg = br.new_page(viewport={"width": 1400, "height": 625})
        pg.goto(URL, wait_until="load", timeout=60000)
        pg.wait_for_function("() => !!window.__flyProbe && !!window.__flyPos", timeout=60000)
        pg.wait_for_timeout(9000)
        pg.get_by_role("button", name=str(CARD), exact=True).first.click(timeout=5000)
        print("picked card", CARD)
        # hold the reach open forever: planted at full extension, no handover
        pg.evaluate("() => { window.__REACH.hold = 1e6; }")
        pg.wait_for_function(
            "() => window.__flyPos.behavior === 'reach' && window.__flyPos.reach.on > 0.99",
            timeout=180000,
        )
        print("planted; t =", pg.evaluate("() => window.__flyPos.t"))
        rows = []

        def step(label):
            # let the pose settle a few frames (animator time advances <= 0.05 s per frame)
            t0 = pg.evaluate("() => window.__flyPos.t")
            deadline = time.time() + 40
            while time.time() < deadline:
                pg.wait_for_timeout(300)
                if pg.evaluate("() => window.__flyPos.t") - t0 >= 0.12:
                    break
            s = sample(pg)
            rows.append((label, s))
            return s

        pg.evaluate("(b) => Object.assign(window.__REACH, b)", BASE)
        base = step("base")
        print("base:", json.dumps({k: base[k] for k in ("d", "img")}))
        for name, value in PROBES:
            pg.evaluate("(b) => Object.assign(window.__REACH, b)", {**BASE, name: value})
            step(f"{name}={value}")
        pg.evaluate("(b) => Object.assign(window.__REACH, b)", BASE)
        br.close()

    def fmt(label, s, b):
        out = [f"{label:>16}"]
        for i in (0, 3):
            dd = s["d"][i] - b["d"][i]
            dxi = s["img"][i][0] - b["img"][i][0]
            dyi = s["img"][i][1] - b["img"][i][1]
            out.append(f"leg{i}: d {s['d'][i]:+.3f} ({dd:+.3f}) img dx {dxi:+6.1f} dy {dyi:+6.1f}")
        for i in (1, 2, 4, 5):
            out.append(f"L{i} {s['d'][i] - b['d'][i]:+.3f}")
        return "  ".join(out)

    print("\nper-leg effect of each dial, relative to base (d = perpendicular distance to glass,")
    print("negative delta = toward the surface; img dx/dy = how far the tarsus moved on screen)")
    print(fmt("base", base, base))
    for label, s in rows[1:]:
        print(fmt(label, s, base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
