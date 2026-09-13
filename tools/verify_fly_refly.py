"""Prove the fly RE-FLIES to a different card when the choice changes.

The app test could not settle this: if the fly happens to pick the same option on two
consecutive questions it would legitimately not move. The standalone viewer has buttons to set
the choice directly, so this drives 1 -> 4 -> 2 and measures the fly's position after each,
together with its distance to the card it should be on.
"""
import math

from playwright.sync_api import sync_playwright


def close(a, b):
    return math.dist(a, b)


with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1400, "height": 625})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:180]))
    pg.goto("http://localhost:5191/phone.html", wait_until="load", timeout=60000)
    pg.wait_for_timeout(9000)

    def state():
        return pg.evaluate(
            "() => ({fly: window.__flyPos||null, probe: window.__phoneStage||null})"
        )

    def choose(n):
        # the viewer's pick buttons are labelled 1..4 inside the FLY PICKS row
        pg.evaluate(
            """(n) => {
              const btns = [...document.querySelectorAll('button')]
                .filter(b => /^[1-4]$/.test(b.textContent.trim()));
              if (btns[n-1]) btns[n-1].click();
            }""",
            n,
        )

    print("card perch points (world space):")
    s0 = state()
    for i, c in enumerate(s0["fly"]["cardWorld"]):
        print(f"   card {i+1}: {c}")
    print()

    prev = None
    for n in (1, 4, 2):
        choose(n)
        pg.wait_for_timeout(4000)  # the flight takes 1.6s; allow generous settle time
        s = state()
        pos = s["fly"]["pos"]
        tgt = s["fly"]["target"]
        cards = s["fly"]["cardWorld"]
        gaps = [round(close(c, pos), 3) for c in cards]
        nearest = gaps.index(min(gaps)) + 1
        moved = close(prev, pos) if prev else 0.0
        print(f"chose card {n}:  fly at {pos}")
        print(f"   target {tgt}  distance to target {s['fly']['dist']}")
        print(f"   distances to each card: {gaps}  -> nearest card {nearest}")
        print(f"   moved {moved:.3f} units since the previous choice")
        print(f"   {'OK - on the card it chose' if nearest == n else 'MISMATCH'}")
        print()
        prev = pos

    pg.screenshot(path=r"D:\Projects\flylingo\artifacts\phone-choice-4.png")
    print("errors:", errs[:4])
    br.close()
