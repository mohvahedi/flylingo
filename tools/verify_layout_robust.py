"""Do the panels still fit across MANY questions, not just the one that was on screen?

Question lengths vary a lot ("How do you say 'Hello' in Spanish?" against "Fill the blank:
'Buenos ___' (Good morning)"), and only the long ones stress the layout. A single screenshot
only proves the frame that happened to be up. This samples repeatedly and reports the worst
overflow and any collision seen across the sample.
"""
import sys

from playwright.sync_api import sync_playwright

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 24

OVERFLOW_JS = r"""() => {
  const out = [];
  for (const el of document.querySelectorAll('section, div')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none') continue;
    const b = el.getBoundingClientRect();
    if (b.width < 200 || b.height < 60) continue;
    const over = el.scrollHeight - el.clientHeight;
    if (over > 1) out.push({ over, txt: (el.textContent || '').trim().slice(0, 44) });
  }
  return out;
}"""

COLLIDE_JS = r"""() => {
  const out = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || Number(cs.opacity) === 0) continue;
    const b = el.getBoundingClientRect();
    if (b.width < 2 || b.height < 2) continue;
    let own = '';
    for (const n of el.childNodes) if (n.nodeType === 3) own += n.textContent;
    own = own.trim();
    if (!own) continue;
    out.push({ t: own.slice(0, 40), x: b.x, y: b.y, w: b.width, h: b.height });
  }
  return out;
}"""

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:120]))
    pg.goto("http://127.0.0.1:3300/fly", wait_until="load", timeout=60000)
    pg.wait_for_timeout(18000)

    worst_over = 0
    worst_where = ""
    collision_counts = []
    questions = set()

    for i in range(SAMPLES):
        q = pg.evaluate(
            "() => { const h = document.querySelector('h2'); return h ? h.textContent.trim() : null; }"
        )
        if q:
            questions.add(q)
        for o in pg.evaluate(OVERFLOW_JS):
            if o["over"] > worst_over:
                worst_over = o["over"]
                worst_where = o["txt"]
        items = pg.evaluate(COLLIDE_JS)
        n = 0
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                p1, p2 = items[a], items[b]
                ca = p1["x"] <= p2["x"] and p1["y"] <= p2["y"] and p1["x"]+p1["w"] >= p2["x"]+p2["w"] and p1["y"]+p1["h"] >= p2["y"]+p2["h"]
                cb = p2["x"] <= p1["x"] and p2["y"] <= p1["y"] and p2["x"]+p2["w"] >= p1["x"]+p1["w"] and p2["y"]+p2["h"] >= p1["y"]+p1["h"]
                if ca or cb:
                    continue
                ox = min(p1["x"]+p1["w"], p2["x"]+p2["w"]) - max(p1["x"], p2["x"])
                oy = min(p1["y"]+p1["h"], p2["y"]+p2["h"]) - max(p1["y"], p2["y"])
                if ox > 0 and oy > 0:
                    f = (ox*oy) / min(p1["w"]*p1["h"], p2["w"]*p2["h"])
                    if f > 0.12:
                        n += 1
        collision_counts.append(n)
        pg.wait_for_timeout(2200)

    br.close()

print(f"sampled {SAMPLES} frames, {len(questions)} distinct questions")
print(f"  worst container overflow : {worst_over}px   {worst_where!r}")
print(f"  collisions per frame     : max {max(collision_counts)}, total {sum(collision_counts)}")
print(f"  console errors           : {len(errs)}")
print()
print("questions seen:")
for q in sorted(questions):
    print(f"  {len(q):>3} chars  {q}")
print()
ok = worst_over <= 12 and sum(collision_counts) == 0
print("RESULT:", "layout holds across the sample" if ok else "STILL OVERFLOWING/COLLIDING")
sys.exit(0 if ok else 1)
