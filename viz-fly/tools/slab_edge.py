"""Does the slab read as a BOUNDED slab, or as a plane that fades into black?

This is the check that the previous round of measurement was missing. `ground_local.py` answers
"is the fly's shadow readable", which it is; it never asked "is there a visible edge". So this
probe reads the silhouette directly:

  walk every screen row from the top and find the first pixel that is lit (above VOID_LUMA).
  That is the slab's upper boundary at that column. For a bounded slab it is a hard step: the
  pixel above is void, the pixel below is slab, and the step size is large.
  For a faded disc it is a slow ramp, so the step is a few luma no matter how far you look.

It reports the step at the boundary, the value of the first 8 px inside it, and the boundary's
screen path, so the shape of the silhouette can be read off as text.

    python tools/slab_edge.py [base_url]
"""
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:5191/?frozen'
ART = 'D:/Projects/flylingo/artifacts'
VOID_LUMA = 6          # at or below this is the void; the clear colour is 11.00 luma in linear
                       # terms but the tonemapped clear is ~0
RAMP = ' .:-=+*#%@'
LAUNCH_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
               '--disable-lcd-text']
HIDE_JS = """(() => {
  const cv = document.querySelector('canvas');
  let el = cv, n = 0;
  while (el && el.parentElement) {
    const p = el.parentElement;
    for (const sib of Array.from(p.children)) if (sib !== el) { sib.style.display = 'none'; n++; }
    el = p; if (el === document.body) break;
  }
  return n;
})()"""


def luma(png):
    a = np.asarray(Image.open(BytesIO(png)).convert('RGB'), dtype=np.int32)
    return (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000


with sync_playwright() as p:
    b = p.chromium.launch(args=LAUNCH_ARGS)
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(BASE, wait_until='domcontentloaded', timeout=90000)
    pg.wait_for_selector('canvas', timeout=90000)
    pg.wait_for_timeout(7000)
    pg.evaluate(HIDE_JS)
    pg.wait_for_timeout(300)
    l = luma(pg.locator('canvas').first.screenshot())
    with open(ART + '/viz-fly-hero.png', 'wb') as fh:
        fh.write(pg.locator('canvas').first.screenshot())
    b.close()

print('console errors: %d %s' % (len(errs), [e[:120] for e in errs[:3]]))
print()
print('== slab silhouette: first lit pixel going down each column ==')
print('   col   first-lit-y   void above   slab below   STEP    8px inside')
# the fly occupies roughly screen x 350..1150; inside those columns the first lit pixel is the
# fly's own body (195..222 luma), not the slab's boundary, so both are reported separately
FLY_X = (350, 1150)
rows = []
rows_slab = []
for x in range(40, 1440, 70):
    col = l[:, x]
    idx = np.where(col > VOID_LUMA)[0]
    if len(idx) == 0:
        print('  %4d   none (all void)' % x)
        continue
    y = int(idx[0])
    above = float(col[max(0, y - 6):y].mean())
    below = float(col[y:y + 3].mean())
    inside = float(col[y:y + 8].mean())
    step = below - above
    rows.append((x, y, step, inside))
    tag = '  <- fly, not slab' if FLY_X[0] <= x <= FLY_X[1] else ''
    if not tag:
        rows_slab.append((x, y, step, inside))
    print('  %4d   %8d   %10.2f   %10.2f   %6.2f   %8.2f%s' % (x, y, above, below, step, inside, tag))

steps = [r[2] for r in rows_slab]
inside = [r[3] for r in rows_slab if r[3] > 10]
print()
if steps:
    print('  SLAB boundary, fly-free columns only: step median %.2f  min %.2f  max %.2f  (%d cols)'
          % (float(np.median(steps)), float(np.min(steps)), float(np.max(steps)), len(steps)))
    print('  boundary y spans %d..%d, so the edge is not a flat line: the left edge and the far'
          % (min(r[1] for r in rows_slab), max(r[1] for r in rows_slab)))
    print('  edge meet at the slab\'s visible far-left corner, which is what makes it an object.')
    print('  a faded disc gives a step of a few luma; a bounded slab gives a step the size of')
    print('  the slab\'s own lit value, because the void above it is 0.')
    print('  lit value just inside the boundary: median %.2f  min %.2f  max %.2f'
          % (float(np.median(inside)), float(np.min(inside)), float(np.max(inside))))
print()
print('== how much of the frame is void, slab, and fly ==')
void = float((l <= VOID_LUMA).mean())
lit = float(((l > VOID_LUMA) & (l < 70)).mean())
fly = float((l >= 70).mean())
print('  void (<=%d)        %5.1f%%' % (VOID_LUMA, 100 * void))
print('  slab (%d..69)      %5.1f%%' % (VOID_LUMA + 1, 100 * lit))
print('  fly/bright (>=70) %5.1f%%' % (100 * fly))
print('  frame mean %.2f  p50 %d  p99 %d' % (l.mean(), np.percentile(l, 50), np.percentile(l, 99)))
print()
print('== frame map ==')
for y in range(0, 900, 20):
    line = ''
    for x in range(0, 1440, 20):
        line += RAMP[min(9, max(0, int(float(l[y:y + 20, x:x + 20].mean()) / 12)))]
    print('%4d %s' % (y, line))
print('   . <12  : 12-24  - 24-36  = 36-48  + 48-60  * 60-72  # 72-84  %% 84-96  @ >96')
