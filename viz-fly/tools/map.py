"""Print an ASCII luma map of the rendered canvas, plus the row/column profiles, so the
measurement boxes in ground_measure.py can be placed from data instead of guessed.

    python tools/map.py [url] [--w 120] [--h 60]
"""
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

args = [a for a in sys.argv[1:] if not a.startswith('--')]
URL = args[0] if args else 'http://localhost:5191/'
GW = 120
GH = 54
for i, a in enumerate(sys.argv):
    if a == '--w':
        GW = int(sys.argv[i + 1])
    if a == '--h':
        GH = int(sys.argv[i + 1])

LAUNCH_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
HIDE_JS = """(() => {
  const c = document.querySelector('canvas');
  if (!c || !c.parentElement) return 0;
  for (const el of Array.from(c.parentElement.children)) {
    if (el.tagName !== 'CANVAS') el.style.display = 'none';
  }
  return 1;
})()"""


def ch(v):
    for t, c in ((6, ' '), (12, '.'), (18, ':'), (24, '-'), (32, '='), (48, '+'),
                 (70, '*'), (110, '#'), (170, '%')):
        if v < t:
            return c
    return '@'


with sync_playwright() as p:
    browser = p.chromium.launch(args=LAUNCH_ARGS)
    page = browser.new_page(viewport={'width': 1440, 'height': 900})
    page.goto(URL, wait_until='load', timeout=45000)
    page.wait_for_selector('canvas', timeout=45000)
    page.wait_for_timeout(5000)
    page.evaluate(HIDE_JS)
    page.wait_for_timeout(150)
    png = page.locator('canvas').first.screenshot()
    browser.close()

a = np.asarray(Image.open(BytesIO(png)).convert('RGB'), dtype=np.int32)
l = (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000
h, w = l.shape
print('canvas %dx%d' % (w, h))
for j in range(GH):
    y = j * h // GH
    print('%4d ' % y + ''.join(ch(float(l[y:y + h // GH, i * w // GW:(i + 1) * w // GW].mean()))
                               for i in range(GW)))
print()
print('col profile (x, mean luma over full height, every 60 px)')
print(' '.join('%d:%.0f' % (x, l[:, x:x + 60].mean()) for x in range(0, w, 60)))
print()
print('row profile (y, mean luma, every 20 px)')
for y in range(0, h, 20):
    print('  y=%4d %6.1f %s' % (y, l[y:y + 20].mean(), '#' * int(l[y:y + 20].mean() / 3)))
