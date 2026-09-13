"""Does the fly keep its fore tarsi ON the chosen card through the whole reach?

A single sample is misleading: the reach is an animation (extend, hold, settle, and a tap), so
the tarsus moves across the card and the answer to a one-shot reading flips on timing alone.
This samples through the cycle and reports the fraction of the time each fore tarsus is on the
card, which is the honest form of the claim.
"""
import sys

from playwright.sync_api import sync_playwright

PICKS = [2, 1, 3]

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1400, "height": 625})
    pg.goto("http://localhost:5191/phone.html", wait_until="load", timeout=60000)
    pg.wait_for_timeout(7000)

    ok = True
    for pick in PICKS:
        pg.evaluate(
            """(n) => {
              const b = [...document.querySelectorAll('button')]
                .filter(x => /^[1-4]$/.test(x.textContent.trim()));
              if (b[n-1]) b[n-1].click();
            }""",
            pick,
        )
        pg.wait_for_timeout(4500)  # arrive + let the reach cycle start
        samples, on_card, near = 0, [0, 0], []
        for _ in range(26):
            d = pg.evaluate("() => window.__flyPos || null")
            legs = (d or {}).get("legs")
            if legs and len(legs) >= 6:
                samples += 1
                for k, idx in enumerate((0, 3)):
                    if legs[idx].get("onCard"):
                        on_card[k] += 1
                near.append(min(abs(legs[0]["d"]), abs(legs[3]["d"])))
            pg.wait_for_timeout(180)
        frac = [c / samples for c in on_card] if samples else [0, 0]
        best = min(near) if near else 99
        verdict = "PASS" if max(frac) > 0.5 else "FAIL"
        if max(frac) <= 0.5:
            ok = False
        print(
            f"chose card {pick}: fore tarsus on the chosen card for "
            f"{frac[0]*100:.0f}% / {frac[1]*100:.0f}% of the reach cycle "
            f"({samples} samples), closest approach {best:.2f} units -> {verdict}"
        )

    print()
    print("RESULT:", "the forelegs hold the answer through the reach" if ok else "NOT holding")
    br.close()
    sys.exit(0 if ok else 1)
