"""Which way does the fly have to be turned so its FORELEGS land on the card?

Reads the scene's own per-tarsus telemetry at each candidate yaw and reports, for the two
forelegs, the perpendicular distance to the glass and their screen pixel position versus the
chosen option's rectangle. The winner is the one whose fore tarsi are actually on the card.
"""
import math

from playwright.sync_api import sync_playwright

CANDIDATES = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]


def run(pg):
    d = pg.evaluate("() => window.__flyPos || null")
    if not d:
        return None
    legs = d.get("legs") or []
    if len(legs) < 6:
        return None
    fore = [legs[0], legs[3]]
    dist_glass = [abs(f["d"]) for f in fore]
    return {
        "rotY": d.get("rotY"),
        "nearest": min(dist_glass),
        "oncard": [bool(f.get("onCard")) for f in fore],
        "px": [(round(f["px"]), round(f["py"])) for f in fore],
        "all_d": [round(l["d"], 2) for l in legs],
    }


with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    best = None
    for yaw in CANDIDATES:
        pg = br.new_page(viewport={"width": 1400, "height": 625})
        pg.goto(
            f"http://localhost:5191/phone.html?yaw={yaw:.4f}", wait_until="load", timeout=60000
        )
        pg.wait_for_timeout(8000)
        # pick card 2 and give it time to fly there and reach
        pg.evaluate(
            """() => {
              const b = [...document.querySelectorAll('button')]
                .filter(x => /^[1-4]$/.test(x.textContent.trim()));
              if (b[1]) b[1].click();
            }"""
        )
        pg.wait_for_timeout(6000)
        r = run(pg)
        label = f"yaw={yaw:.2f} ({math.degrees(yaw):.0f} deg)"
        if r:
            print(f"{label:22s} rotY={r['rotY']}  fore tarsi off glass: {r['nearest']:.3f}")
            print(f"{'':22s}   on card: {r['oncard']}   at px {r['px']}")
            print(f"{'':22s}   all six d: {r['all_d']}")
            score = r["nearest"] - (1.0 if any(r["oncard"]) else 0.0)
            if best is None or score < best[0]:
                best = (score, yaw, r)
        else:
            print(f"{label:22s} no telemetry")
        print()
        pg.close()

    if best:
        print("=" * 62)
        print(f"WINNER: yaw = {best[1]:.4f} rad ({math.degrees(best[1]):.0f} deg)")
        print(f"   fore tarsi {best[2]['nearest']:.3f} units off the glass, on card {best[2]['oncard']}")
        print(f"   rotY settled at {best[2]['rotY']}")
    else:
        print("no candidate produced usable telemetry")
    br.close()
