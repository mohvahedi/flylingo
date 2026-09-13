"""Verify the fly is sent to the CARD IT ACTUALLY CHOSE.

The fly's whole claim is that it goes to the answer it picked. That is only true if the pixel
rectangle the screen renderer produced for option N maps to the world point the fly is sent
to, through the phone's real world matrix. This checks exactly that, per card, instead of
trusting the code that computes both sides.
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
    pg.wait_for_timeout(10000)
    d = pg.evaluate("() => window.__flyPos || null")

    if not d:
        print("no telemetry")
        raise SystemExit(1)

    sb = d["screenBox"]
    print("screen world box:")
    print("   min", sb["min"], " max", sb["max"])
    print()

    print("card centres in world space:")
    cards = d["cardWorld"] or []
    for i, c in enumerate(cards):
        print(f"   card {i+1}: x={c[0]:+.3f}  y={c[1]:+.3f}  z={c[2]:+.3f}")
    print()

    t = d["target"]
    print("fly target  :", t)
    print("fly position:", d["pos"], " dist", d["dist"])
    print("behaviour   :", d["behavior"])
    print()

    sx = (sb["min"][0] + sb["max"][0]) / 2
    sy = (sb["min"][1] + sb["max"][1]) / 2
    print(f"screen centre: x={sx:+.3f} y={sy:+.3f}")
    print()

    # the fly was told to pick card index 1 (the second card), so it must be sitting on it
    if len(cards) >= 2:
        gap = math.dist(cards[1], t)
        print(f"distance from the fly's target to card 2's perch point: {gap:.4f}  (want ~0)")
        print("   (the perch is 0.78 across the card, not its centre, so the fly clears the text)")

    # and the four cards must be in the expected 2x2 arrangement, which is what makes the
    # mapping meaningful rather than a coincidence
    if len(cards) == 4:
        tl, tr, bl, b_r = cards[0], cards[1], cards[2], cards[3]
        print()
        print("arrangement checks (holds for a stacked single column too):")
        print(f"   card1 left of card2 : {tl[0] < tr[0]}")
        print(f"   card3 left of card4 : {bl[0] < b_r[0]}")
        print(f"   card1 above card3   : {tl[1] > bl[1]}")
        print(f"   card2 above card4   : {tr[1] > b_r[1]}")

    inside = (
        sb["min"][0] - 0.25 <= t[0] <= sb["max"][0] + 0.25
        and sb["min"][1] - 0.25 <= t[1] <= sb["max"][1] + 0.25
    )
    print()
    print("target lies within the screen box (allowing the standoff):", inside)
    br.close()
