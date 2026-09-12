"""Isolate parts of the viz-fly scene and print a luma map for each variant, so the ground
can be measured on its own instead of inferred from a frame the fly shares.

    python tools/isolate.py [url] [--w 96] [--h 30]

Variants: as-is, fly hidden, ground hidden, fly+rings hidden. Prints, per variant, an ASCII
luma map and the share of the frame above a few luma thresholds.
"""
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

URL = 'http://localhost:5191/?frozen'
MW, MH = 96, 30
for i, a in enumerate(sys.argv[1:]):
    if a == '--w':
        MW = int(sys.argv[i + 2])
    if a == '--h':
        MH = int(sys.argv[i + 2])
    if not a.startswith('--') and i == 0:
        URL = a

LAUNCH_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
HIDE_JS = """(() => {
  const c = document.querySelector('canvas');
  if (c && c.parentElement) for (const el of Array.from(c.parentElement.children)) {
    if (el.tagName !== 'CANVAS') el.style.display = 'none';
  }
  return 1;
})()"""

RAMP = ' .:-=+*#%@'


def luma(png):
    a = np.asarray(Image.open(BytesIO(png)).convert('RGB'), dtype=np.int32)
    return (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000, a


def report(tag, png, probe):
    l, a = luma(png)
    h, w = l.shape
    print('=== %s ===  %dx%d' % (tag, w, h))
    print('   mean luma  %.1f   >20 %.3f  >40 %.3f  >80 %.3f' % (
        l.mean(), (l > 20).mean(), (l > 40).mean(), (l > 80).mean()))
    step_y, step_x = max(1, h // MH), max(1, w // MW)
    # average into cells first, then map: the fly is thin, a raw sample would miss it
    for y0 in range(0, h - step_y + 1, step_y):
        row = ''
        for x0 in range(0, w - step_x + 1, step_x):
            v = l[y0:y0 + step_y, x0:x0 + step_x].mean()
            row += RAMP[min(9, int(v) // 26)]
        print('%4d %s' % (y0, row))
    if probe:
        for name, (x, y) in probe.items():
            print('   %-14s at (%4d,%4d) rgb %s luma %.1f' % (
                name, x, y, tuple(a[y, x]), l[y, x]))


VARIANTS = [
    ('as-is', ''),
    ('no fly', "s.getObjectByName('fly-fit').visible=false"),
    ('no ground', "s.getObjectByName('ground').visible=false"),
    ('fly+ground+rings off', """s.getObjectByName('fly-fit').visible=false;
      s.getObjectByName('ground').visible=false;
      s.getObjectByName('ring-inner').visible=false;
      s.getObjectByName('ring-outer').visible=false"""),
]

with sync_playwright() as p:
    b = p.chromium.launch(args=LAUNCH_ARGS)
    for tag, js in VARIANTS:
        page = b.new_page(viewport={'width': 1440, 'height': 900})
        page.goto(URL, wait_until='load', timeout=45000)
        page.wait_for_selector('canvas', timeout=45000)
        page.wait_for_timeout(4500)
        page.evaluate(HIDE_JS)
        if js:
            ok = page.evaluate('(js) => { const s = window.__flyScene; if (!s) return false;'
                               ' for (const part of js.split(";")) { const m = part.match('
                               '/getObjectByName\\(\'([^\']+)\'\\)\\.visible=false/);'
                               ' if (m) { const o = s.getObjectByName(m[1]); if (o) o.visible=false; } }'
                               ' return true; }', js)
            if not ok:
                print('!! scene not published, variant "%s" is not isolated' % tag)
            page.wait_for_timeout(400)
        report(tag, page.locator('canvas').first.screenshot(), None)
        page.close()
    b.close()
