"""Where did the fly go? Read the live probes back."""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1400, "height": 625})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:200]))
    pg.on("console", lambda m: errs.append(f"{m.type}: {m.text[:160]}") if m.type in ("error", "warning") else None)
    pg.goto("http://localhost:5191/phone.html", wait_until="load", timeout=60000)
    pg.wait_for_timeout(12000)

    d = pg.evaluate("() => window.__flyPos || null")
    probe = pg.evaluate("() => window.__flyProbe || null")
    print("__flyPos:")
    if d:
        for k in ("pos", "target", "dist", "moving", "behavior", "hasScreen", "hasRect", "screenBox"):
            print(f"   {k:12s}: {d.get(k)}")
        print("   cardWorld  :", d.get("cardWorld"))
    else:
        print("   MISSING")
    print()
    print("__flyProbe present:", bool(probe))
    if probe and probe.get("foot"):
        print("   foot positions (first 2):", probe["foot"][:2])
    print()
    print("errors:", errs[:8] if errs else "none")
    br.close()
