"""Which elements overflow, and is it the longer caption that caused it?

The layout agent measured 0 overflowing elements. After restoring the fuller connectome caption
the count reads 6, so either the caption pushed a row over or the count includes things that
were always there and simply were not counted. This names them so the answer is a measurement.
"""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto("http://127.0.0.1:3300/fly", wait_until="load", timeout=60000)
    pg.wait_for_timeout(16000)

    rows = pg.evaluate(
        """() => {
          const out = [];
          for (const e of document.querySelectorAll('*')) {
            if (e.scrollWidth > e.clientWidth + 1 && e.clientWidth > 0) {
              const cs = getComputedStyle(e);
              out.push({
                tag: e.tagName.toLowerCase(),
                text: (e.textContent || '').trim().slice(0, 58),
                client: e.clientWidth,
                scroll: e.scrollWidth,
                over: e.scrollWidth - e.clientWidth,
                ws: cs.whiteSpace,
                ov: cs.overflow,
                txtOv: cs.textOverflow,
                cls: (e.className || '').toString().slice(0, 30),
              });
            }
          }
          return out;
        }"""
    )
    print(f"{len(rows)} overflowing elements:\n")
    for r in rows:
        print(f"  <{r['tag']}> over by {r['over']}px   {r['client']} -> {r['scroll']}")
        print(f"      text: {r['text']!r}")
        print(f"      white-space={r['ws']}  overflow={r['ov']}  text-overflow={r['txtOv']}")
        print(f"      class: {r['cls']}")

    # specifically: is the caption's own row one of them?
    cap = pg.evaluate(
        """() => {
          const want = 'male cns v1.0 · brain and ventral nerve cord · measured connectome';
          const el = [...document.querySelectorAll('*')].find(e =>
            (e.textContent || '').trim() === want && !e.children.length);
          if (!el) return null;
          let p = el, chain = [];
          while (p && chain.length < 5) {
            chain.push({ tag: p.tagName.toLowerCase(), client: p.clientWidth,
                         scroll: p.scrollWidth, over: p.scrollWidth - p.clientWidth,
                         disp: getComputedStyle(p).display });
            p = p.parentElement;
          }
          return chain;
        }"""
    )
    print("\nthe caption's ancestor chain (does any box clip it?):")
    if cap:
        for i, c in enumerate(cap):
            flag = "  <-- CLIPPING" if c["over"] > 1 else ""
            print(f"  {'self' if i == 0 else 'up ' + str(i)} <{c['tag']:8s}> {c['client']} -> {c['scroll']}{flag}")
    br.close()
