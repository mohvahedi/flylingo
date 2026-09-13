"""Isolate the cause: does the longer caption overflow the row, or was it already overflowing?

Measured in the live page by rewriting the caption's text in place and re-reading the geometry,
which changes nothing on disk. This distinguishes "my restore broke the row" from "the row was
already over and the count is measuring a wrapping container".
"""
from playwright.sync_api import sync_playwright

LONG = "male cns v1.0 · brain and ventral nerve cord · measured connectome"
SHORT = "male cns v1.0 · measured connectome"

PROBE = """(want) => {
  const el = [...document.querySelectorAll('*')].find(e =>
    ['male cns v1.0 · brain and ventral nerve cord · measured connectome',
     'male cns v1.0 · measured connectome'].includes((e.textContent || '').trim())
    && !e.children.length);
  if (!el) return null;
  const box = el.parentElement;
  const kids = [...box.children].map(c => ({
    w: Math.round(c.getBoundingClientRect().width),
    t: (c.textContent || '').trim().slice(0, 34),
  }));
  return {
    spanW: Math.round(el.getBoundingClientRect().width),
    boxClient: box.clientWidth,
    boxScroll: box.scrollWidth,
    boxOver: box.scrollWidth - box.clientWidth,
    wrap: getComputedStyle(box).flexWrap,
    kids,
  };
}"""

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto("http://127.0.0.1:3300/fly", wait_until="load", timeout=60000)
    pg.wait_for_timeout(16000)

    for label, want in (("LONG (as shipped now)", LONG), ("SHORT (agent's version)", SHORT)):
        # rewrite in place
        pg.evaluate(
            """(args) => {
              const el = [...document.querySelectorAll('*')].find(e =>
                ['male cns v1.0 · brain and ventral nerve cord · measured connectome',
                 'male cns v1.0 · measured connectome'].includes((e.textContent || '').trim())
                && !e.children.length);
              if (el) el.textContent = args;
            }""",
            want,
        )
        pg.wait_for_timeout(700)
        r = pg.evaluate(PROBE, want)
        if not r:
            print(f"{label}: caption not found")
            continue
        print(f"{label}")
        print(f"   span width      : {r['spanW']}px")
        print(f"   row             : client {r['boxClient']}, scroll {r['boxScroll']}  (over by {r['boxOver']})")
        print(f"   row flex-wrap   : {r['wrap']}")
        for k in r["kids"]:
            print(f"     child {k['w']:>4}px  {k['t']!r}")
        print()
    br.close()
