"""Verify the fly actually moves to a NEW card when the fly's choice changes, inside the real HUD.

The whole claim of the hero panel is that the fly flies to the answer it picked. That is only
demonstrated if the fly's position CHANGES when the choice changes, and lands on the new
card's perch point. This drives the real lesson in the app and samples the fly's live position
across an answer, so the movement is measured rather than assumed from a still.
"""
import math
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:3300/fly"

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:180]))
    pg.goto(BASE, wait_until="load", timeout=60000)
    pg.wait_for_timeout(18000)

    def fly():
        return pg.evaluate("() => window.__flyPos || null")

    def hud_choice():
        return pg.evaluate(
            """() => {
              const t = document.body.innerText;
              const m = t.match(/FLY IS CHOOSING\\s*\\n?\\s*([^\\n]+)/i)
                     || t.match(/FLY ANSWERED\\s*\\n?\\s*([^\\n]+)/i);
              return m ? m[1].trim() : null;
            }"""
        )

    start = fly()
    print("=== at rest ===")
    print("  fly pos :", start["pos"] if start else None)
    print("  target  :", start["target"] if start else None)
    print("  picking :", hud_choice())
    print("  cards   :", (start.get("cardWorld") if start else None))
    print()

    # Answer the question so the fly must commit and the next challenge arrives.
    pg.locator("button").filter(has_text="Hola").first.click()
    pg.wait_for_timeout(400)
    pg.locator("button").filter(has_text="CHECK").first.click()
    pg.wait_for_timeout(2500)
    print("=== after answering ===")
    mid = fly()
    print("  fly pos :", mid["pos"] if mid else None)
    print("  picking :", hud_choice())

    # advance to the next question
    for name in ("NEXT", "CHECK"):
        b = pg.locator("button").filter(has_text=name)
        if b.count():
            try:
                b.first.click()
                break
            except Exception:
                pass
    pg.wait_for_timeout(3500)
    end = fly()
    print()
    print("=== on the next question ===")
    print("  fly pos :", end["pos"] if end else None)
    print("  target  :", end["target"] if end else None)
    print("  picking :", hud_choice())
    print("  cards   :", (end.get("cardWorld") if end else None))

    if start and end:
        moved = math.dist(start["pos"], end["pos"])
        print()
        print(f"FLY MOVED {moved:.3f} world units between questions")
        print("  (0 would mean it never re-flew, which would mean the hero panel is a still)")
        tgt = end["target"]
        cards = end.get("cardWorld") or []
        if cards:
            gaps = [round(math.dist(c, tgt), 3) for c in cards]
            best = gaps.index(min(gaps))
            print(f"  distances from the fly's target to each card: {gaps}")
            print(f"  nearest card: {best + 1}   (the HUD says the fly picked {hud_choice()})")

    pg.screenshot(path=r"D:\Projects\flylingo\artifacts\hud-choice-moved.png")
    print()
    print("errors:", errs[:4])
    br.close()
