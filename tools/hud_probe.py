"""Runtime probe for the connectome HUD.

Answers the question the screenshot statistics cannot: is the amber spike
path actually live at runtime, and are the live pins / edges / trail dots
present in the scene graph with sane uniforms. Reads the react-three-fiber
root off the canvas element (canvas.__r3f) rather than using readPixels,
because r3f does not set preserveDrawingBuffer and an out-of-frame read
returns zeros.

Usage: python tools/hud_probe.py [base_url]
"""

import json
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:4180'
ART = Path('D:/Projects/flylingo/artifacts')
CANVAS = '[data-testid="cloud-section"] canvas'
CAPTION = '[data-testid="cloud-caption"]'

PROBE = """
() => {
  // react-three-fiber 9 does not expose __r3f on the canvas here, so walk the
  // React fiber tree instead: every THREE object created from JSX ends up as a
  // fiber stateNode, which gives read-only access to the live scene without
  // changing the component.
  const host = document.getElementById('root') || document.body;
  const rootKey = Object.keys(host).find((k) => k.startsWith('__reactContainer$'));
  const root = rootKey ? host[rootKey] : null;
  const found = [];
  const stack = root ? [root] : [];
  let guard = 0;
  while (stack.length && guard < 400000) {
    const f = stack.pop();
    guard += 1;
    const sn = f.stateNode;
    if (sn && sn.isObject3D) found.push(sn);
    if (f.child) stack.push(f.child);
    if (f.sibling) stack.push(f.sibling);
    if (f.alternate && f.alternate.child && guard < 2000) stack.push(f.alternate.child);
  }
  const out = { r3f_keys: rootKey ? ['fiber-walk'] : [], fibers_seen: guard, points: [], gl: {} };
  const want = ['uSize','uGate','uQuiet','uAmbient','uShockColor','uToneGain','uWidthPx',
                'uOpacity','uPixelRatio','uViewHeight','uRef','uColorDim','uClassMix',
                'uFogNear','uFogFar','uAlpha','uCore','uTrailLife'];
  for (const o of found) {
    if (!o.isPoints && !o.isMesh && !o.isLineSegments) continue;
    const g = o.geometry;
    const m = o.material;
    const rec = {
      type: o.type,
      name: o.name || '',
      visible: o.visible,
      frustumCulled: o.frustumCulled,
      position_count: g && g.attributes && g.attributes.position ? g.attributes.position.count : 0,
      attrs: g ? Object.keys(g.attributes) : [],
      drawRange: g && g.drawRange ? [g.drawRange.start, g.drawRange.count] : null,
    };
    if (m) {
      rec.material = m.type;
      rec.transparent = m.transparent;
      rec.blending = m.blending;
      rec.depthWrite = m.depthWrite;
      rec.uniforms = {};
      if (m.uniforms) {
        for (const k of want) {
          if (m.uniforms[k]) {
            const v = m.uniforms[k].value;
            rec.uniforms[k] = (v && v.isColor) ? [v.r, v.g, v.b] : v;
          }
        }
      }
    }
    // aShock / aActivity live on the pin geometry.
    if (g && g.attributes.aShock) {
      const a = g.attributes.aShock.array;
      let nz = 0;
      const hot = [];
      for (let i = 0; i < a.length; i += 1) {
        if (a[i] > 0.004) { nz += 1; if (hot.length < 8) hot.push([i, +a[i].toFixed(3)]); }
      }
      rec.aShock = { count: a.length, nonzero: nz, sample: hot, at_353: a[353], at_362: a[362], at_377: a[377] };
    }
    if (g && g.attributes.aActivity) {
      const a = g.attributes.aActivity.array;
      let nz = 0;
      let mx = 0;
      for (let i = 0; i < a.length; i += 1) { if (a[i] !== 0) nz += 1; if (Math.abs(a[i]) > mx) mx = Math.abs(a[i]); }
      rec.aActivity = { count: a.length, nonzero: nz, max_abs: +mx.toFixed(4) };
    }
    out.points.push(rec);
  }
  return out;
}
"""


def stats(png_bytes):
    import io

    im = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    w, h = im.size
    px = list(im.getdata())
    lit = warm = hot = 0
    sr = sg = sb = 0
    for r, g, b in px:
        m = max(r, g, b)
        sr += r
        sg += g
        sb += b
        if m > 28:
            lit += 1
        if r > g + 6 and r - b > 20 and r > 45:
            warm += 1
            if r > 110:
                hot += 1
    n = len(px)
    return {
        'size': [w, h],
        'lit_pct': round(100.0 * lit / n, 3),
        'mean_rgb': [round(sr / n, 2), round(sg / n, 2), round(sb / n, 2)],
        'warm_px': warm,
        'hot_px': hot,
    }


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1600, 'height': 900}, device_scale_factor=1)
    errs = []
    page.on('pageerror', lambda e: errs.append(str(e)))
    page.goto(BASE + '/?drive=full', wait_until='load')
    page.wait_for_function('() => window.__bc && window.__bc.points === 166700', timeout=60000)
    page.wait_for_timeout(3500)
    print('--- scene ---')
    print(json.dumps(page.evaluate(PROBE), indent=1)[:6000])
    print('--- metrics ---')
    m = page.evaluate('() => JSON.parse(JSON.stringify(window.__bc))')
    print(json.dumps(m, indent=1)[:2500])
    print('--- caption ---')
    print(page.inner_text(CAPTION))
    box = page.query_selector(CANVAS).bounding_box()
    png = page.screenshot(clip=box)
    (ART / 'hud-probe-canvas.png').write_bytes(png)
    print('--- canvas stats (hud-probe-canvas.png) ---')
    print(json.dumps(stats(png)))
    print('--- page errors ---')
    print(errs[:10])
    browser.close()
