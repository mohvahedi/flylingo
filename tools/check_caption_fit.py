"""Does the provenance row have room for the fuller connectome description?

The layout agent trimmed this row to "male cns v1.0 · measured connectome" to make it fit,
flagging that it drops the words "brain and ventral nerve cord". That is a real loss of
information: this panel is captioned as a brain, and MaleCNS is the whole central nervous
system, brain plus ventral nerve cord. So the question is whether the fuller wording fits
the box now, and this measures it instead of guessing.
"""
from playwright.sync_api import sync_playwright

FULL = "male cns v1.0 · brain and ventral nerve cord · measured connectome"
SHORT = "male cns v1.0 · measured connectome"

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto("http://127.0.0.1:3300/fly", wait_until="load", timeout=60000)
    pg.wait_for_timeout(14000)

    info = pg.evaluate(
        """() => {
          const out = [];
          for (const el of document.querySelectorAll('*')) {
            const t = (el.textContent || '').trim();
            if (t === 'male cns v1.0 · measured connectome' && el.children.length === 0) {
              const cs = getComputedStyle(el);
              const parent = el.parentElement;
              out.push({
                avail: Math.round(parent.getBoundingClientRect().width),
                own: Math.round(el.getBoundingClientRect().width),
                scroll: el.scrollWidth,
                font: cs.fontSize + ' ' + cs.fontFamily.split(',')[0],
              });
            }
          }
          return out;
        }"""
    )
    for i in info:
        print(f"  row available width : {i['avail']}px")
        print(f"  current text width  : {i['own']}px  ({i['font']})")
        print(f"  scrollWidth (clip)  : {i['scroll']}px")

    # measure both candidate strings in the same font
    m = pg.evaluate(
        """([full, short]) => {
          const el = [...document.querySelectorAll('*')].find(e =>
            (e.textContent || '').trim() === 'male cns v1.0 · measured connectome' && !e.children.length);
          if (!el) return null;
          const cs = getComputedStyle(el);
          const c = document.createElement('canvas').getContext('2d');
          c.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
          const sp = cs.letterSpacing && cs.letterSpacing !== 'normal' ? parseFloat(cs.letterSpacing) : 0;
          const w = (s) => c.measureText(s).width + sp * s.length;
          return {
            avail: Math.round(el.parentElement.getBoundingClientRect().width),
            full: Math.round(w(full)),
            short: Math.round(w(short)),
          };
        }""",
        [FULL, SHORT],
    )
    if m:
        print()
        print(f"  available            : {m['avail']}px")
        print(f"  SHORT wording        : {m['short']}px  {'FITS' if m['short'] <= m['avail'] else 'CLIPS'}")
        print(f"  FULL wording         : {m['full']}px  {'FITS' if m['full'] <= m['avail'] else 'CLIPS'}")
        room = m["full"] - m["short"]
        print(f"  extra needed for full: {room}px")
    br.close()
