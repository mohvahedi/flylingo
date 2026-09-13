"""Measure whether the fly's FORELEGS actually reach the chosen card.

The user's requirement is that the fly "use its arms to choose the option", so the claim that
needs measuring is not that the fly is near the phone but that its fore tarsi land on the card
it picked. The scene already reports every tarsus's perpendicular distance to the glass and
where that lands in screen pixels; this reads that back and checks it against the option's own
rectangle.
"""
import math

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1400, "height": 625})
    pg.goto("http://localhost:5191/phone.html", wait_until="load", timeout=60000)
    pg.wait_for_timeout(9000)

    # drive it onto each card in turn and watch the fore tarsi
    for pick in (1, 3, 2):
        pg.evaluate(
            """(n) => {
              const b = [...document.querySelectorAll('button')]
                .filter(x => /^[1-4]$/.test(x.textContent.trim()));
              if (b[n-1]) b[n-1].click();
            }""",
            pick,
        )
        pg.wait_for_timeout(5000)
        d = pg.evaluate("() => window.__flyPos || null")
        if not d:
            print("no telemetry")
            continue
        legs = d.get("legs")
        print(f"chose card {pick}:  behaviour={d.get('behavior')}")
        print(f"   fly pos    {[round(v,2) for v in d['pos']]}")
        print(f"   target     {[round(v,2) for v in d['target']]}   dist {d['dist']}")
        if legs:
            names = ["fore-L", "mid-L", "hind-L", "fore-R", "mid-R", "hind-R"]
            for i, lg in enumerate(legs):
                name = names[i] if i < len(names) else f"leg{i}"
                mark = "  <-- ON THE CARD" if lg.get("onCard") else ""
                print(f"   {name:7s} d={lg['d']:+.3f}  px=({lg['px']:.0f},{lg['py']:.0f}){mark}")
            fore = [legs[i] for i in (0, 3) if i < len(legs)]
            near = min(abs(f["d"]) for f in fore)
            on = any(f.get("onCard") for f in fore)
            print(f"   => fore tarsi: nearest {near:.3f} units off the glass, on the chosen card: {on}")
        else:
            print("   (no per-leg telemetry published)")
        print()
    br.close()
