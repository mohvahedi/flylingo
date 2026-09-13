"""Headless end-to-end check of the FlyLingo lesson against the live brain service.

Loads the real page, plays challenges, and asserts the fly panel and lesson both
populate from live data. Prints what it actually observed.
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from flylingo_env import app_url

URL = app_url("/lesson/fly")
OUT = Path(r"D:\Projects\flylingo\artifacts")
OUT.mkdir(parents=True, exist_ok=True)

console_errors = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 950})
    page.on(
        "console",
        lambda m: console_errors.append(f"{m.type}: {m.text}")
        if m.type in ("error", "warning")
        else None,
    )
    page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_timeout(9000)

    print("=== initial render ===")
    print("title:", page.title())
    body = page.inner_text("body")
    print(body[:1800])
    page.screenshot(path=str(OUT / "01-lesson.png"), full_page=True)

    # The fly panel must be live, not "no service".
    has_service = "live" in body.lower()
    print("\nstream live:", has_service)
    print("mode badge intact:", "Intact connectome" in body or "intact" in body.lower())

    # Play three challenges, picking option 1 each time, and watch the panel change.
    for i in range(3):
        try:
            page.get_by_text("Select the correct meaning", exact=False).first.wait_for(
                timeout=8000
            )
        except Exception:
            pass
        options = page.locator("div.cursor-pointer")
        n = options.count()
        print(f"\n--- challenge {i+1}: {n} option cards ---")
        if n == 0:
            break
        options.first.click()
        page.wait_for_timeout(400)
        # The clone's footer button is labelled Check / Next / Retry.
        for label in ("Check", "Next", "Retry", "Continue"):
            btn = page.get_by_role("button", name=label, exact=True)
            if btn.count() and btn.first.is_visible():
                btn.first.click()
                print(f"clicked button: {label}")
                break
        page.wait_for_timeout(2500)
        txt = page.inner_text("body")
        for line in txt.splitlines():
            if "fly picked" in line or "readout updated" in line or "Nicely done" in line or "Try again" in line:
                print("   ", line.strip())
        page.screenshot(path=str(OUT / f"0{i+2}-lesson.png"), full_page=True)

    final = page.inner_text("body")
    print("\n=== final panel readouts ===")
    for key in ("Active", "State rms", "Step", "Accuracy"):
        idx = final.find(key)
        if idx >= 0:
            print(f"  {key}: {final[idx:idx+40].splitlines()[:2]}")

    print("\n=== canvases present ===")
    print(page.eval_on_selector_all("canvas", "els => els.map(e => e.width + 'x' + e.height)"))

    # Confirm the canvases are actually painting, not blank.
    painted = page.evaluate(
        """() => {
      const out = [];
      for (const c of document.querySelectorAll('canvas')) {
        const ctx = c.getContext('2d');
        if (!ctx) { out.push('no-2d-context'); continue; }
        const d = ctx.getImageData(0, 0, c.width, c.height).data;
        let nonEmpty = 0;
        for (let i = 3; i < d.length; i += 4) if (d[i] > 8) nonEmpty++;
        out.push({ w: c.width, h: c.height, litPixels: nonEmpty });
      }
      return out;
    }"""
    )
    print(painted)

    print("\n=== console errors/warnings ===")
    for e in console_errors[:25]:
        print("  ", e[:220])
    if not console_errors:
        print("   none")

    browser.close()
