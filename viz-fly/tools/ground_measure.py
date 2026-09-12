"""Grounding measurement probe for the viz-fly hero stage.

Screenshots the same frame twice, once with the whole shadow rig on and once with it
switched off (?noshadow), and reports the numbers that decide whether the fly reads as
standing on a lit slab inside a dark frame:

  under      luma of the ground directly under the body, in a band under the fly
  slab       luma of the ground immediately around that pool, same frame
  void       luma of the far background, same frame
  contrast   (slab - under) / slab            <- what a viewer actually sees
  shadow_on/off ratio of under-body luma      <- does the shadow do anything at all

    python tools/ground_measure.py [base_url]
"""
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:5191/?frozen'
ART = 'D:/Projects/flylingo/artifacts'
BG = (8, 12, 17)

LAUNCH_ARGS = [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--disable-lcd-text',
]

HIDE_JS = """(() => {
  const c = document.querySelector('canvas');
  if (!c || !c.parentElement) return 0;
  for (const el of Array.from(c.parentElement.children)) {
    if (el.tagName !== 'CANVAS') el.style.display = 'none';
  }
  return 1;
})()"""

# The fly occupies roughly the middle 40% of the frame. These boxes are in canvas pixels
# (1440x900) and were placed by looking at the luma map of the shot, not guessed:
# the body sits near x 560..880, its feet land around y 560..600.
BOX = {
    'under': (620, 900, 505, 575),    # pool beneath thorax/legs
    'slab':  (620, 900, 620, 690),    # slab just in front of the pool, same column range
    'void':  (1050, 1400, 120, 320),  # far background, upper right
    'void2': (1050, 1400, 760, 880),  # far background, lower right
}


def shot(page, url, path):
    page.goto(url, wait_until='load', timeout=45000)
    page.wait_for_selector('canvas', timeout=45000)
    page.wait_for_timeout(5000)
    page.evaluate(HIDE_JS)
    page.wait_for_timeout(150)
    png = page.locator('canvas').first.screenshot()
    with open(path, 'wb') as fh:
        fh.write(png)
    a = np.asarray(Image.open(BytesIO(png)).convert('RGB'), dtype=np.int32)
    # Rec.601 luma in integer math; the weights wrap if left in int16
    l = (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000
    return a, l


def box(l, name):
    y0, y1, x0, x1 = BOX[name]
    return float(l[y0:y1, x0:x1].mean())


def rowprofile(l, y0, y1):
    out = []
    for y in range(y0, y1, 10):
        out.append((y, round(float(l[y:y + 10, 560:940].mean()), 1)))
    return out


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=LAUNCH_ARGS)
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        on_a, on_l = shot(page, BASE, ART + '/viz-fly-ground-on.png')
        # the same frame is the hero shot: shadows on, camera frozen
        with open(ART + '/viz-fly-hero.png', 'wb') as fh:
            fh.write(open(ART + '/viz-fly-ground-on.png', 'rb').read())
        cam = page.evaluate('() => { const c = window.__flyCamera; '
                            'return c ? [c.position.toArray().map(n=>+n.toFixed(3)), c.fov] : null; }')
        print('camera', cam)
        off_a, off_l = shot(page, BASE + ('&' if '?' in BASE else '?') + 'noshadow',
                            ART + '/viz-fly-ground-off.png')
        browser.close()

    print('== per-box luma (shadows ON / OFF) ==')
    for k in BOX:
        print('  %-6s %7.2f  / %7.2f' % (k, box(on_l, k), box(off_l, k)))

    under, slab, void = box(on_l, 'under'), box(on_l, 'slab'), box(on_l, 'void')
    contr = (slab - under) / slab * 100.0 if slab else 0.0
    print()
    print('LOCAL CONTRAST  pool %.2f  vs  slab-around %.2f   -> shadow %.1f%% darker'
          % (under, slab, contr))
    print('VOID            near %.2f  far %.2f' % (void, box(on_l, 'void2')))
    print('SHADOW EFFECT   under-body on %.2f off %.2f  -> %.1f%% darker'
          % (under, box(off_l, 'under'),
             (box(off_l, 'under') - under) / box(off_l, 'under') * 100.0 if box(off_l, 'under') else 0))

    print()
    print('== vertical luma profile through the fly column (x 560..940), shadows ON ==')
    for y, v in rowprofile(on_l, 380, 900):
        print('  y=%4d %6.1f %s' % (y, v, '#' * int(v / 3)))

    print()
    print('== frame stats ==')
    for name, l in (('on', on_l), ('off', off_l)):
        print('  %-4s mean %.2f  p99 %d  max %d  frac>40 %.4f  frac<26 %.4f'
              % (name, l.mean(), np.percentile(l, 99), l.max(),
                 (l > 40).mean(), (l < 26).mean()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
