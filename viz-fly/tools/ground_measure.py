"""Grounding measurement probe for the viz-fly hero stage.

Screenshots the same frozen frame twice, once with the whole shadow rig on and once with it
switched off (?noshadow), and reports the numbers that decide whether the fly reads as
standing on a lit slab inside a dark frame:

  under      luma of the slab under the body, in the pooled band beneath the fly
  slab       luma of the lit slab immediately beside that pool, same rows, same frame
  void       luma of the far background above the slab's far edge, same frame
  contrast   (slab - under) / slab            <- what a viewer actually sees
  effect     ratio of under-body luma on/off  <- does the shadow rig do anything at all

Read the box table before trusting a number from it. Three defects were fixed here and each
one had been producing a plausible-looking figure:

  1. BOX rows were documented (x0, x1, y0, y1) and unpacked as (y0, y1, x0, x1). The "under"
     and "slab" boxes were therefore both sampling the near floor to the LEFT of the fly -
     columns 505..575 and 620..690, rows 620..900 - where the fly's own legs are, and the
     "17% local contrast" in HANDOFF_grounding.md is that measurement, not a shadow.
  2. HIDE_JS hid the canvas's siblings, which is the canvas itself: the canvas is the only
     child of its R3F container. The badge and the dev strip are siblings of that container,
     so both stayed in every shot and the top left of the frame was text, not void.
  3. The void boxes fell outside the frame (x up to 1400 while the sampled rows ran 1050..1400
     the wrong way), which is why that row printed nan.

For the self-locating version - which finds the shadow footprint from the A/B difference
instead of trusting a box pair - use tools/ground_local.py.

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
# pixels brighter than this are fly body, legs or bloom, never slab, and are dropped from
# every box mean so a box that clips a leg is not read as a bright floor
FLY_LUMA = 70

LAUNCH_ARGS = [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--disable-lcd-text',
]

HIDE_JS = """(() => {
  const cv = document.querySelector('canvas');
  let el = cv, n = 0;
  while (el && el.parentElement) {
    const p = el.parentElement;
    for (const sib of Array.from(p.children)) {
      if (sib !== el) { sib.style.display = 'none'; n++; }
    }
    el = p;
    if (el === document.body) break;
  }
  return n;
})()"""

# Canvas pixels (1440x900), (x0, x1, y0, y1), placed from the projection of the frozen camera
# (position [1.85, 1.15, 2.35], fov 32, target [0, 0.5, 0]): the fly's ground contact point
# projects to (720, 692), the slab's far edge runs from (550, 357) to the right of the frame,
# so everything above y=340 is void and above y=280 is void across the whole width.
BOX = {
    'under': (620, 1000, 640, 760),    # pooled slab under and behind the fly's feet
    'slab':  (60, 300, 640, 760),      # lit slab at the same rows, outside the pool
    'slabL': (60, 300, 440, 560),      # lit slab beside the fly, further back
    'void':  (1050, 1400, 120, 300),   # background, upper right
    'void2': (60, 420, 60, 280),       # background, upper left
}


def shot(page, url, path):
    page.goto(url, wait_until='domcontentloaded', timeout=90000)
    page.wait_for_selector('canvas', timeout=90000)
    page.wait_for_timeout(6000)
    page.evaluate(HIDE_JS)
    page.wait_for_timeout(200)
    png = page.locator('canvas').first.screenshot()
    with open(path, 'wb') as fh:
        fh.write(png)
    a = np.asarray(Image.open(BytesIO(png)).convert('RGB'), dtype=np.int32)
    # Rec.601 luma in integer math; the weights wrap if left in int16
    l = (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000
    return a, l


def box(l, name):
    x0, x1, y0, y1 = BOX[name]
    b = l[y0:y1, x0:x1]
    m = b < FLY_LUMA
    return float(b[m].mean()) if m.sum() > 20 else float('nan')


def rowprofile(l, y0, y1):
    out = []
    for y in range(y0, y1, 10):
        out.append((y, round(float(l[y:y + 10, 560:940].mean()), 1)))
    return out


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=LAUNCH_ARGS)
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        errors = []
        page.on('console', lambda m: errors.append('console.%s: %s' % (m.type, m.text))
                if m.type == 'error' else None)
        page.on('pageerror', lambda e: errors.append('pageerror: %s' % e))
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

    print('== per-box luma, (x0,x1,y0,y1) = %s, fly pixels dropped ==' % (BOX['under'],))
    for k in BOX:
        print('  %-6s %s  %7.2f  / %7.2f' % (k, str(BOX[k]), box(on_l, k), box(off_l, k)))

    under, slab, void = box(on_l, 'under'), box(on_l, 'slab'), box(on_l, 'void')
    contr = (slab - under) / slab * 100.0 if slab else 0.0
    print()
    print('LOCAL CONTRAST  pool-under %.2f  vs  slab-beside %.2f   -> %.1f%% darker'
          % (under, slab, contr))
    print('SLAB LIT        beside fly %.2f   further back %.2f   (the surface the shadow needs)'
          % (slab, box(on_l, 'slabL')))
    print('VOID            upper right %.2f   upper left %.2f  (clear colour luma %.2f)'
          % (void, box(on_l, 'void2'), float((np.array(BG) * np.array([299, 587, 114])).sum() // 1000)))
    print('SHADOW RIG      under on %.2f off %.2f -> %.1f%% darker'
          % (under, box(off_l, 'under'),
             (box(off_l, 'under') - under) / box(off_l, 'under') * 100.0
             if box(off_l, 'under') else 0.0))
    print('   note: the rig figure is what the key light\'s cast shadow removes, and the key is')
    print('   21 degrees above the horizon, so on its own it is only a few luma. The pooled')
    print('   contact read is a material term, ablated separately by tools/ground_local.py.')

    print()
    print('== vertical luma profile through the fly column (x 560..940), shadows ON ==')
    for y, v in rowprofile(on_l, 300, 900):
        print('  y=%4d %6.1f %s' % (y, v, '#' * int(v / 3)))

    print()
    print('== frame stats ==')
    for name, l in (('on', on_l), ('off', off_l)):
        print('  %-4s mean %.2f  p50 %d  p99 %d  max %d  frac>%d %.4f  frac<26 %.4f'
              % (name, l.mean(), np.percentile(l, 50), np.percentile(l, 99), l.max(),
                 FLY_LUMA, (l > FLY_LUMA).mean(), (l < 26).mean()))

    print()
    print('== console errors: %d ==' % len(errors))
    for e in errors[:10]:
        print('  ' + e[:300])
    return 0


if __name__ == '__main__':
    sys.exit(main())
