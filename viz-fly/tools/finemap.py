"""Fine luma map of the lower half of the frame (where the ground and the contact pool are),
sampled at 20 px cells, so the measurement boxes in ground_measure.py can be placed exactly.

    python tools/finemap.py [url]
"""
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:5191/?frozen'
LAUNCH_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
X0, X1, Y0, Y1, CELL = 240, 1200, 480, 900, 20
RAMP = ' .:-=+*#%@'


def grab(page, url):
    page.goto(url, wait_until='load', timeout=45000)
    page.wait_for_selector('canvas', timeout=45000)
    page.wait_for_timeout(5000)
    page.evaluate("""() => { const c = document.querySelector('canvas');
      if (c && c.parentElement) for (const el of Array.from(c.parentElement.children))
        if (el.tagName !== 'CANVAS') el.style.display = 'none'; return 1; }""")
    page.wait_for_timeout(150)
    a = np.asarray(Image.open(BytesIO(page.locator('canvas').first.screenshot())).convert('RGB'),
                   dtype=np.int32)
    return (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000


def draw(tag, l):
    print('=== %s ===   (x %d..%d, y %d..%d, cell %d px)' % (tag, X0, X1, Y0, Y1, CELL))
    hdr = '     ' + ''.join((str(x // 100 % 10) if (x - X0) % 100 == 0 else ' ') for x in range(X0, X1, CELL))
    print(hdr)
    for y in range(Y0, Y1, CELL):
        row = ''
        for x in range(X0, X1, CELL):
            v = l[y:y + CELL, x:x + CELL].mean()
            row += RAMP[min(9, int(v) // 26)]
        print('%4d %s' % (y, row))


with sync_playwright() as p:
    b = p.chromium.launch(args=LAUNCH_ARGS)
    page = b.new_page(viewport={'width': 1440, 'height': 900})
    on = grab(page, URL)
    off = grab(page, URL + ('&' if '?' in URL else '?') + 'noshadow')
    draw('shadows ON', on)
    draw('shadows OFF', off)
    d = on.astype(np.int32) - off.astype(np.int32)
    print('=== ON minus OFF (positive = the shadow rig darkened that cell) ===')
    print('     ' + ''.join((str(x // 100 % 10) if (x - X0) % 100 == 0 else ' ') for x in range(X0, X1, CELL)))
    for y in range(Y0, Y1, CELL):
        row = ''
        for x in range(X0, X1, CELL):
            v = -d[y:y + CELL, x:x + CELL].mean()  # darkening, so positive reads bright here
            row += RAMP[min(9, int(max(0, v)) // 6)]
        print('%4d %s' % (y, row))
    b.close()
