"""Coarse RGB grid of the rendered canvas, so scene regions can be identified by hue
(amber = fly, slate = ground, dark teal = void, cyan = the scale rings).

    python tools/grid.py [url] [--cols 16] [--rows 12]
"""
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

args = [a for a in sys.argv[1:] if not a.startswith('--')]
URL = args[0] if args else 'http://localhost:5191/'
COLS, ROWS = 16, 12
for i, a in enumerate(sys.argv):
    if a == '--cols':
        COLS = int(sys.argv[i + 1])
    if a == '--rows':
        ROWS = int(sys.argv[i + 1])

LAUNCH_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
HIDE_JS = """(() => {
  const c = document.querySelector('canvas');
  if (c && c.parentElement) for (const el of Array.from(c.parentElement.children)) {
    if (el.tagName !== 'CANVAS') el.style.display = 'none';
  }
  return 1;
})()"""

with sync_playwright() as p:
    b = p.chromium.launch(args=LAUNCH_ARGS)
    page = b.new_page(viewport={'width': 1440, 'height': 900})
    page.goto(URL, wait_until='load', timeout=45000)
    page.wait_for_selector('canvas', timeout=45000)
    page.wait_for_timeout(5000)
    page.evaluate(HIDE_JS)
    page.wait_for_timeout(150)
    png = page.locator('canvas').first.screenshot()
    b.close()

a = np.asarray(Image.open(BytesIO(png)).convert('RGB'), dtype=np.int32)
l = (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000
h, w = l.shape
print('canvas %dx%d' % (w, h))
print('grid mean RGB + luma, %d cols x %d rows' % (COLS, ROWS))
hdr = '  y\\\\x '
for i in range(COLS):
    hdr += '%13d' % (i * w // COLS)
print(hdr)
for j in range(ROWS):
    y0, y1 = j * h // ROWS, (j + 1) * h // ROWS
    row = '%5d ' % y0
    for i in range(COLS):
        x0, x1 = i * w // COLS, (i + 1) * w // COLS
        c = a[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        row += ' %4d,%4d,%4d' % (c[0], c[1], c[2])
    print(row)

print()
print('warm/cyan/asymmetry probes')
print('  left third  luma %.1f   right third luma %.1f' % (
    l[:, :w // 3].mean(), l[:, 2 * w // 3:].mean()))
amber = (a[..., 0] > a[..., 2] + 12) & (l > 30)
print('  amber pixel frac %.4f' % amber.mean())
