"""Find the most informative connectome caption that actually fits the row.

My restore of the full wording overflowed the row by 207px (caption 526px + colour legend 365px
in a 700px nowrap row), so the layout agent's trim was correct and my canvas-based measurement
was wrong: it under-counted letter-spacing by about 15%. This measures the real rendered width of
each candidate in the live DOM instead, and reports the budget the legend leaves behind.

The point of the wording is that this panel is captioned as a brain while MaleCNS is the whole
central nervous system, brain plus ventral nerve cord, so the ideal wording says both without
spending a line the row does not have.
"""
from playwright.sync_api import sync_playwright

CANDIDATES = [
    "male cns v1.0 · measured connectome",
    "male cns v1.0 · brain + vnc",
    "male cns v1.0 · brain + nerve cord",
    "male cns v1.0 · brain and vnc · measured",
    "male cns · brain + ventral nerve cord",
    "male cns v1.0 · brain + vnc, measured",
]

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto("http://127.0.0.1:3300/fly", wait_until="load", timeout=60000)
    pg.wait_for_timeout(15000)

    # the caption span itself, so the real font and letter-spacing are used
    res = pg.evaluate(
        """(cands) => {
          const el = [...document.querySelectorAll('*')].find(e =>
            (e.textContent || '').trim().startsWith('male cns') && !e.children.length);
          if (!el) return null;
          const box = el.parentElement;
          const legend = [...box.children].find(c => c !== el);
          const legendW = legend ? Math.round(legend.getBoundingClientRect().width) : 0;
          const boxW = box.clientWidth;
          const gap = parseFloat(getComputedStyle(box).columnGap || getComputedStyle(box).gap || '0') || 0;
          const c = document.createElement('canvas').getContext('2d');
          const cs = getComputedStyle(el);
          c.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
          const ls = cs.letterSpacing && cs.letterSpacing !== 'normal' ? parseFloat(cs.letterSpacing) : 0;
          const meas = s => Math.round(c.measureText(s).width + ls * s.length);
          // real rendered width of the current text, as the calibration check
          const actual = Math.round(el.getBoundingClientRect().width);
          const predicted = meas(el.textContent.trim());
          return { boxW, legendW, gap, actual, predicted,
                   budget: Math.round(boxW - legendW - gap),
                   cands: cands.map(t => ({ t, w: meas(t) })) };
        }""",
        CANDIDATES,
    )

    if not res:
        print("caption not found")
    else:
        print(f"row width          : {res['boxW']}px")
        print(f"colour legend      : {res['legendW']}px")
        print(f"gap                : {res['gap']}px")
        print(f"caption budget     : {res['budget']}px")
        print()
        print(f"calibration: canvas predicted {res['predicted']}px vs real rendered {res['actual']}px")
        err = (res["predicted"] - res["actual"]) / res["actual"] * 100 if res["actual"] else 0
        print(f"             the canvas measure is off by {err:+.1f}%")
        print()
        for c in res["cands"]:
            ok = "FITS" if c["w"] <= res["budget"] else "OVER"
            print(f"  [{ok}] {c['w']:>4}px  {c['t']}")
    br.close()
