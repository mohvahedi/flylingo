"""Poll the phone scene while the fly is in flight, so the flight can be verified rather than
assumed. Reports when the probe appears, and how the fly's distance to its target closes."""
import sys, time
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5191/phone.html"

with sync_playwright() as p:
    br = p.chromium.launch(headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = br.new_page(viewport={"width": 1400, "height": 625})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:200]))
    pg.goto(BASE, wait_until="load", timeout=60000)
    for i in range(26):
        pg.wait_for_timeout(1000)
        st = pg.evaluate("() => ({probe: window.__phoneStage||null, fly: window.__flyPos||null})")
        probe, fly = st["probe"], st["fly"]
        line = f"t+{i+1:02d}s "
        if probe:
            line += f"normal={probe.get('screenNormal')} up={probe.get('screenUp')} "
        else:
            line += "probe=… "
        if fly:
            line += (f"pos={fly['pos']} target={fly['target']} dist={fly['dist']} "
                     f"moving={fly['moving']} beh={fly['behavior']} "
                     f"screen={fly['hasScreen']} rect={fly['hasRect']}")
        print(line, flush=True)
        if fly and not fly["moving"] and i > 3:
            print("ARRIVED at t+%ds" % (i + 1))
            break
    print("errors:", errs[:4])
    br.close()
