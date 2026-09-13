"""Verify the HUD layout: canvas buffers match their CSS boxes, and nothing overflows.

Both properties were real defects found in this frame, so they are checked rather than assumed:
  - a canvas whose buffer and CSS box disagree is being STRETCHED by the browser. That happened
    to the specimen render (728x358 CSS over a 700x295 buffer, about 21% vertical stretch)
    because it had no positioned ancestor, so it also drew over its own panel header.
  - an element whose scrollWidth exceeds its clientWidth is clipped or overflowing.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:3300/fly"
OUT = Path(r"D:\Projects\flylingo\artifacts") / (sys.argv[2] if len(sys.argv) > 2 else "layout-check.png")

JS = """() => {
  const canvases = [];
  document.querySelectorAll('canvas').forEach((c, i) => {
    const r = c.getBoundingClientRect();
    canvases.push({
      i,
      css: Math.round(r.width) + 'x' + Math.round(r.height),
      buf: c.width + 'x' + c.height,
      matches: Math.abs(r.width - c.width) < 2 && Math.abs(r.height - c.height) < 2,
    });
  });
  const overflowing = [];
  document.querySelectorAll('*').forEach((e) => {
    if (e.clientWidth > 0 && e.scrollWidth > e.clientWidth + 2) {
      overflowing.push({
        tag: e.tagName,
        text: (e.textContent || '').trim().slice(0, 50),
        scrollW: e.scrollWidth,
        clientW: e.clientWidth,
      });
    }
  });
  const panels = [];
  document.querySelectorAll('section').forEach((s) => {
    const r = s.getBoundingClientRect();
    panels.push(Math.round(r.width) + 'x' + Math.round(r.height));
  });
  return { canvases, overflowing, panels };
}"""

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:140]))
    pg.on(
        "console",
        lambda m: errs.append("console error: " + m.text[:120]) if m.type == "error" else None,
    )
    pg.goto(URL, wait_until="load", timeout=60000)
    pg.wait_for_timeout(20000)
    pg.screenshot(path=str(OUT))

    r = pg.evaluate(JS)
    print("panels (w x h):")
    for i, q in enumerate(r["panels"]):
        print(f"   panel {i}: {q}")
    print()
    print("canvases:")
    stretched = 0
    for c in r["canvases"]:
        flag = "MATCH" if c["matches"] else "*** STRETCHED ***"
        if not c["matches"]:
            stretched += 1
        print(f"   {c['i']}: css {c['css']:>10}   buffer {c['buf']:>10}   {flag}")
    print()
    print(f"overflowing elements: {len(r['overflowing'])}")
    for o in r["overflowing"][:6]:
        print(f"   {o['tag']:6} scrollW {o['scrollW']} > clientW {o['clientW']}  {o['text']!r}")
    print()
    print("console errors:", errs[:4] if errs else "none")
    print()
    print("VERDICT:", "clean" if stretched == 0 and not r["overflowing"] and not errs else "problems above")
    br.close()
