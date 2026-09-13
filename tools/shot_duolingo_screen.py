"""Render the Duolingo phone screen on its own and save it, so the design can be judged
before it is wrapped around a 3D phone.

The TS module is loaded straight from the Vite dev server, which transforms it on the fly,
so there is no build step between editing the screen and seeing it.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5191/"
ART = Path(__file__).resolve().parent.parent / ".." / "artifacts"
ART = Path(r"D:\Projects\flylingo\artifacts")
ART.mkdir(parents=True, exist_ok=True)

# Each state is rendered in turn so the live pick, the correct and the wrong views are all
# checked, not just the happy path.
STATES = [
    ("open", {"flyChoice": 1, "userChoice": -1, "status": "none", "answerIndex": -1, "progress": 0.28}),
    ("correct", {"flyChoice": 1, "userChoice": 1, "status": "correct", "answerIndex": 1, "progress": 0.42}),
    ("wrong", {"flyChoice": 1, "userChoice": 0, "status": "wrong", "answerIndex": 1, "progress": 0.42}),
]

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 900, "height": 1000}, device_scale_factor=3)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:200]))
    pg.goto(BASE, wait_until="load", timeout=40000)

    for name, over in STATES:
        res = pg.evaluate(
            """async (over) => {
              const m = await import('/src/fly/duolingoScreen.ts');
              await m.ensureDuolingoFonts();
              const canvas = m.createScreenCanvas(2);
              canvas.id = 'shot';
              const ctx = canvas.getContext('2d');
              const out = m.drawDuolingoLesson(ctx, Object.assign({
                prompt: "How do you say 'Hello' in Spanish?",
                options: [{text:'Buenas tardes'},{text:'Hola'},
                          {text:'Buenas noches'},{text:'Buenos días'}],
                hearts: 5, course: 'es',
              }, over));
              canvas.style.width = m.SCREEN_W + 'px';
              canvas.style.height = m.SCREEN_H + 'px';
              document.body.style.background = '#222';
              document.body.innerHTML = '';
              document.body.appendChild(canvas);
              return { rects: out.options, w: m.SCREEN_W, h: m.SCREEN_H };
            }""",
            over,
        )
        pg.wait_for_timeout(500)
        pg.wait_for_timeout(400)
        box = pg.evaluate("() => { const c=document.getElementById('shot'); const r=c.getBoundingClientRect(); return {w:r.width, h:r.height}; }")
        pg.screenshot(path=str(ART / f"duolingo-screen-{name}.png"),
                      clip={"x": 0, "y": 0, "width": 390, "height": 865})
        print(f"{name:8s} rects={res['rects']}")

    print("canvas:", res["w"], "x", res["h"])
    print("errors:", errs)
    br.close()
print("saved to", ART)
