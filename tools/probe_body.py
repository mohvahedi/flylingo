"""Read back the handset's material colours to confirm the black-body fix actually applied."""
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    br = p.chromium.launch(headless=True, args=["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"])
    pg = br.new_page(viewport={"width":1400,"height":625})
    pg.goto("http://localhost:5191/phone.html", wait_until="load", timeout=60000)
    pg.wait_for_timeout(8000)
    d = pg.evaluate("() => window.__phoneStage || null")
    print("body materials:", d.get("bodyMaterials") if d else None)
    print("screen normal  :", d.get("screenNormal") if d else None)
    br.close()
