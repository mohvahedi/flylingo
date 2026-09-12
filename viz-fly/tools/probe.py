"""Fast diagnosis probe for the viz-fly dev server: what does the console say, and is
anything lit? Used while iterating on the shaders, where the full witness harness is slow.

    python tools/probe.py [url] [--shader-dump]

Prints every console error (full text for THREE.WebGLProgram failures), every page error,
the canvas/GL info, and the lit-fraction of a clean element screenshot. Exits 1 if the
console carries a shader error.
"""
import sys
import re
from io import BytesIO

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

args = [a for a in sys.argv[1:] if not a.startswith('--')]
URL = args[0] if args else 'http://localhost:5191/'
DUMP = '--shader-dump' in sys.argv
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


def main() -> int:
    logs = []
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=LAUNCH_ARGS)
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        page.on('console', lambda m: logs.append(('%s' % m.type, m.text)))
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(URL, wait_until='load', timeout=45000)
        page.wait_for_selector('canvas', timeout=45000)
        page.wait_for_timeout(4500)
        info = page.evaluate(
            """(() => {
          const c = document.querySelector('canvas');
          const gl = c && (c.getContext('webgl2') || c.getContext('webgl'));
          const dbg = gl && gl.getExtension('WEBGL_debug_renderer_info');
          return {
            w: c && c.width, h: c && c.height,
            webgl2: !!gl && typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext,
            renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : null,
            stats: window.__flyStats || null,
          };
        })()"""
        )
        page.evaluate(HIDE_JS)
        page.wait_for_timeout(120)
        png = page.locator('canvas').first.screenshot()
        with open(ART + '/viz-fly-probe.png', 'wb') as fh:
            fh.write(png)
        browser.close()

    # int32: the luma weights multiply by up to 587 and wrap if left in int16, which
    # would report a properly lit frame as a black one.
    a = np.asarray(Image.open(BytesIO(png)).convert('RGB'), dtype=np.int32)
    l = (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000
    dist = np.abs(a - np.array(BG, dtype=np.int16)).sum(axis=2)

    print('canvas:', info['w'], 'x', info['h'], 'webgl2:', info['webgl2'])
    print('renderer:', info['renderer'])
    print('stats:', info['stats'])
    print('mean_luma %.2f  p99 %d  max %d' % (l.mean(), np.percentile(l, 99), l.max()))
    print('non_background_frac %.4f  lit_frac %.4f' % ((dist > 12).mean(), (l > 40).mean()))

    shader_err = [t for _, t in logs if 'WebGLProgram' in t or 'Shader Error' in t]
    other_err = [(k, t) for k, t in logs if k == 'error' and t not in shader_err]
    warn = [(k, t) for k, t in logs if k == 'warning']
    print('\nconsole errors: %d (shader: %d)  warnings: %d  pageerrors: %d'
          % (len(shader_err) + len(other_err), len(shader_err), len(warn), len(errors)))
    for k, t in other_err:
        print('  [%s] %s' % (k, t[:600]))
    for e in errors:
        print('  [pageerror] %s' % e[:600])
    for t in shader_err:
        body = t if DUMP else '\n'.join(t.splitlines()[:16])
        head = re.search(r'ERROR: \d+:\d+:.*', t)
        print('  [shader] %s' % (head.group(0) if head else body[:400]))
        if DUMP:
            print(body[:6000])
    return 1 if (shader_err or errors) else 0


if __name__ == '__main__':
    sys.exit(main())
