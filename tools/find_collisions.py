"""Find elements that actually collide, measured from the live DOM.

Reading a downscaled screenshot cannot distinguish "cramped" from "overlapping", and the vision
model mis-maps positions on a 1920-wide frame. This walks the real DOM, takes every element that
directly contains visible text, and reports pairs whose boxes intersect. That is ground truth for
"things are bugged into each other".

Annotations are ignored (MEDIA/aria), and so are ancestor/descendant pairs, because a child box
inside its parent is not a collision.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:3300/fly"
WAIT = int(sys.argv[2]) if len(sys.argv) > 2 else 30000

JS = r"""() => {
  const out = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || Number(cs.opacity) === 0) continue;
    const b = el.getBoundingClientRect();
    if (b.width < 2 || b.height < 2) continue;
    // only elements that own text directly
    let own = '';
    for (const n of el.childNodes) if (n.nodeType === 3) own += n.textContent;
    own = own.trim();
    if (!own) continue;
    out.push({
      t: own.slice(0, 46),
      x: b.x, y: b.y, w: b.width, h: b.height,
      fs: parseFloat(cs.fontSize),
      tag: el.tagName.toLowerCase(),
      path: (el.className || '').toString().slice(0, 24),
    });
  }
  return out;
}"""

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(URL, wait_until="load", timeout=60000)
    pg.wait_for_timeout(WAIT)
    items = pg.evaluate(JS)
    br.close()

print(f"{len(items)} text-bearing elements\n")


def overlap(a, b):
    ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    oy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    if ox <= 0 or oy <= 0:
        return 0.0
    inter = ox * oy
    return inter / min(a["w"] * a["h"], b["w"] * b["h"])


hits = []
n = len(items)
for i in range(n):
    for j in range(i + 1, n):
        a, b = items[i], items[j]
        # skip nesting: one box fully containing the other is layout, not collision
        ca = a["x"] <= b["x"] and a["y"] <= b["y"] and a["x"]+a["w"] >= b["x"]+b["w"] and a["y"]+a["h"] >= b["y"]+b["h"]
        cb = b["x"] <= a["x"] and b["y"] <= a["y"] and b["x"]+b["w"] >= a["x"]+a["w"] and b["y"]+b["h"] >= a["y"]+a["h"]
        if ca or cb:
            continue
        f = overlap(a, b)
        if f > 0.12:
            hits.append((f, a, b))

hits.sort(key=lambda h: -h[0])
print(f"COLLISIONS (>12% of the smaller box overlapped): {len(hits)}\n")
for f, a, b in hits[:20]:
    print(f"  {f*100:4.0f}%  {a['tag']} {a['t']!r}")
    print(f"        x{a['x']:.0f}y{a['y']:.0f} {a['w']:.0f}x{a['h']:.0f} fs{a['fs']:.0f}")
    print(f"        {b['tag']} {b['t']!r}")
    print(f"        x{b['x']:.0f}y{b['y']:.0f} {b['w']:.0f}x{b['h']:.0f} fs{b['fs']:.0f}")
    print()

# also: anything whose box runs outside the 1920x1080 stage, or is clipped by its parent
print("OFF-STAGE / CLIPPED:")
for it in items:
    if it["x"] < 0 or it["y"] < 0 or it["x"] + it["w"] > 1920.5 or it["y"] + it["h"] > 1080.5:
        print(f"  {it['tag']} {it['t']!r} at x{it['x']:.0f} y{it['y']:.0f} w{it['w']:.0f} h{it['h']:.0f}")

# smallest type in the frame
print()
print("SMALLEST TYPE:")
for it in sorted(items, key=lambda i: i["fs"])[:8]:
    print(f"  {it['fs']:4.1f}px  {it['t']!r}")
