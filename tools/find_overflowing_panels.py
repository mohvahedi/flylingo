"""Which panels are taller than the box they were given?

Every collision found so far looks like content escaping its container, so this measures each
panel's scrollHeight against its clientHeight. A panel whose content is taller than its box
paints over whatever sits below it, which is what "bugged into each other" describes.
"""
from playwright.sync_api import sync_playwright

JS = r"""() => {
  const out = [];
  for (const el of document.querySelectorAll('section, div')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none') continue;
    const b = el.getBoundingClientRect();
    if (b.width < 200 || b.height < 60) continue;
    const over = el.scrollHeight - el.clientHeight;
    const overX = el.scrollWidth - el.clientWidth;
    if (over > 1 || overX > 1) {
      out.push({
        h: Math.round(b.height), w: Math.round(b.width),
        x: Math.round(b.x), y: Math.round(b.y),
        sh: el.scrollHeight, ch: el.clientHeight,
        over, overX,
        ov: cs.overflow, ovY: cs.overflowY,
        txt: (el.textContent || '').trim().slice(0, 50),
      });
    }
  }
  return out;
}"""

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto("http://127.0.0.1:3300/fly", wait_until="load", timeout=60000)
    pg.wait_for_timeout(34000)
    rows = pg.evaluate(JS)
    br.close()

print(f"{len(rows)} overflowing containers\n")
for r in sorted(rows, key=lambda r: -r["over"]):
    print(f"  over Y by {r['over']:>4}px (over X {r['overX']:>3})  box {r['w']}x{r['h']} at ({r['x']},{r['y']})")
    print(f"      scrollHeight {r['sh']} vs clientHeight {r['ch']}   overflow={r['ov']}/{r['ovY']}")
    print(f"      {r['txt']!r}")
    print()
