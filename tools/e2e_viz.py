"""Verify both 3D views render: the fly in the lesson panel, and the full brain route.

Uses element screenshots rather than WebGL readPixels: react-three-fiber does not set
preserveDrawingBuffer, so an out-of-frame readPixels returns zeros and proves nothing.

Requires the API on 127.0.0.1:8770 and the Next app on 127.0.0.1:3100.
"""
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

OUT = Path(r"D:\Projects\flylingo\artifacts")
OUT.mkdir(parents=True, exist_ok=True)
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]


def analyse(png):
    im = Image.open(io.BytesIO(png)).convert("RGB")
    px = list(im.getdata())
    n = len(px)
    lit = sum(1 for r, g, b in px if r + g + b > 30)
    mean = sum(sum(p) for p in px) / (3 * n)
    return {"size": f"{im.width}x{im.height}", "lit_fraction": round(lit / n, 4),
            "mean": round(mean, 2)}


def changed(a, b):
    ia, ib = Image.open(io.BytesIO(a)).convert("RGB"), Image.open(io.BytesIO(b)).convert("RGB")
    if ia.size != ib.size:
        return 1.0
    pa, pb = list(ia.getdata()), list(ib.getdata())
    return sum(1 for x, y in zip(pa, pb) if x != y) / len(pa)


errors = []
ok = True
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=ARGS)
    page = browser.new_page(viewport={"width": 1500, "height": 950})
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(f"{m.type}: {m.text}")
            if m.type == "error" else None)

    # ---------- 1. the fly, in the lesson panel ----------
    print("=== LESSON: 3D fly ===")
    page.goto("http://127.0.0.1:3100/lesson/fly", wait_until="domcontentloaded")
    page.wait_for_timeout(13000)
    stage = page.locator("canvas").first
    a1 = stage.screenshot()
    page.wait_for_timeout(1100)
    a2 = stage.screenshot()
    print(f"  render      : {analyse(a1)}")
    print(f"  after 1.1s  : {analyse(a2)}")
    mv = changed(a1, a2)
    print(f"  pixels changed: {mv:.4f}")
    if analyse(a1)["lit_fraction"] < 0.05:
        print("  FAIL: fly is essentially black")
        ok = False
    elif mv < 0.005:
        print("  WARN: fly did not animate")
    else:
        print("  PASS: fly renders and animates")
    print("  brain link present:", "brain cloud" in page.inner_text("body").lower())
    page.screenshot(path=str(OUT / "lesson-fly.png"), full_page=True)

    # ---------- 2. the brain cloud, on its own route ----------
    print("\n=== BRAIN ROUTE: /lesson/fly/brain ===")
    page.goto("http://127.0.0.1:3100/lesson/fly/brain", wait_until="domcontentloaded")
    page.wait_for_timeout(16000)
    body = page.inner_text("body")
    print("  title:", page.title())
    for probe in ("MaleCNS v1.0, live", "stream open", "166,700", "25,582,938",
                  "measured soma voxels"):
        print(f"  shows {probe!r}:", probe in body)

    cloud = page.locator("canvas").first
    b1 = cloud.screenshot()
    page.wait_for_timeout(1400)
    b2 = cloud.screenshot()
    print(f"  render      : {analyse(b1)}")
    print(f"  after 1.4s  : {analyse(b2)}")
    mv2 = changed(b1, b2)
    print(f"  pixels changed: {mv2:.4f}")
    if analyse(b1)["lit_fraction"] < 0.01:
        print("  FAIL: brain cloud is essentially black")
        ok = False
    else:
        print("  PASS: brain cloud renders")
        if mv2 < 0.001:
            print("  WARN: cloud did not change between samples")
    page.screenshot(path=str(OUT / "brain-route.png"), full_page=True)

    # ---------- 3. the control modes must differ here too ----------
    print("\n=== control modes on the brain route ===")
    import json
    import urllib.request

    def post(path, payload):
        req = urllib.request.Request(
            "http://127.0.0.1:8770" + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    # no_edges is the interesting case. Its ACTIVITY must be exactly zero, and the view
    # must be visibly dimmer, but the anatomy deliberately stays faintly legible: the
    # shader keeps a small ambient term so a quiet brain still shows its structure. So
    # the check is "much dimmer and provably inert", not "literally black".
    live_mean = max(0.0, float(analyse(cloud.screenshot())["mean"]))
    for mode in ("intact", "shuffled", "random_graph", "no_edges"):
        post("/control", {"mode": mode})
        page.wait_for_timeout(2600)
        shot = cloud.screenshot()
        st = analyse(shot)
        tel = json.load(urllib.request.urlopen("http://127.0.0.1:8770/telemetry", timeout=30))
        inert = tel["state_rms"] == 0.0 and tel["active_fraction"] == 0.0 and tel["spikes"] == []
        print(f"  {mode:13s} lit={st['lit_fraction']:.4f} mean={st['mean']:6.2f} "
              f"| state_rms={tel['state_rms']} spikes={len(tel['spikes'])} provably_inert={inert}")
        if mode == "no_edges":
            if not inert:
                print("     FAIL: no_edges is not inert in the stream")
                ok = False
            elif st["mean"] >= live_mean * 0.5:
                print(f"     FAIL: no_edges is not visibly dimmer (mean {st['mean']} vs live {live_mean})")
                ok = False
            else:
                print(f"     PASS: no_edges provably inert and {live_mean / max(st['mean'], 1e-6):.1f}x dimmer")
        page.screenshot(path=str(OUT / f"brain-{mode}.png"), full_page=True)
    post("/control", {"mode": "intact"})

    print("\n=== console ===")
    for e in errors[:20]:
        print("  ", e[:230])
    if not errors:
        print("   none")
    browser.close()

print("\nRESULT:", "both 3D views render" if ok else "PROBLEMS FOUND")
raise SystemExit(0 if ok else 1)
