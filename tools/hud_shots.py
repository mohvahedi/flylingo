"""Headless HUD verification for the connectome cloud.

Drives the built viz-brain preview in headless Chromium and measures the
rendered PNG bytes. It never uses WebGL readPixels for content: the renderer
is created without preserveDrawingBuffer, so an out-of-frame readPixels
returns zeros and would report a perfectly good scene as black. readPixels is
used in exactly one place, as a GPU fence for the frame rate measurement.

Usage:
    python tools/hud_shots.py [base_url]

Writes screenshots to D:/Projects/flylingo/artifacts/ and prints a JSON
summary to stdout.
"""

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

ART = Path('D:/Projects/flylingo/artifacts')
BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:4180'
CONTROL = 'http://127.0.0.1:8770/control'
CAPTION = '[data-testid="connectome-caption"]'
CANVAS = '[data-testid="cloud-section"] canvas'

# A pixel is "lit" when it is clearly above the near-black ground (#080C11).
LIT = 28
# Warm, i.e. amber-leaning: red dominant over blue and not just a grey tick.
WARM_DELTA = 20


def control(mode):
    body = json.dumps({'mode': mode}).encode()
    req = urllib.request.Request(CONTROL, data=body, headers={'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def stats(png):
    im = Image.open(io.BytesIO(png)).convert('RGB')
    px = im.tobytes()
    n = len(px) // 3
    lit = 0
    warm = 0
    hot = 0
    sr = sg = sb = 0
    max_rb = -999
    for i in range(0, len(px), 3):
        r = px[i]
        g = px[i + 1]
        b = px[i + 2]
        sr += r
        sg += g
        sb += b
        m = r if r > g else g
        if b > m:
            m = b
        if m > LIT:
            lit += 1
        if r > g + 6 and r - b > WARM_DELTA and r > 45:
            warm += 1
        if r - b > max_rb:
            max_rb = r - b
        if r > 150 and g > 110 and b < 120:
            hot += 1
    return {
        'size': list(im.size),
        'lit_px': lit,
        'lit_pct': round(100.0 * lit / n, 3),
        'warm_px': warm,
        'hot_px': hot,
        'max_r_minus_b': max_rb,
        'mean_rgb': [round(sr / n, 2), round(sg / n, 2), round(sb / n, 2)],
    }


def diff(a, b):
    ia = Image.open(io.BytesIO(a)).convert('RGB')
    ib = Image.open(io.BytesIO(b)).convert('RGB')
    d = ImageChops.difference(ia, ib).convert('L')
    px = d.tobytes()
    w = ia.size[0]
    moved = 0
    left = 0
    right = 0
    for i, v in enumerate(px):
        if v > 8:
            moved += 1
            if (i % w) < w // 2:
                left += 1
            else:
                right += 1
    return {
        'moved_px': moved,
        'moved_pct': round(100.0 * moved / len(px), 3),
        'left_px': left,
        'right_px': right,
        'symmetric': left > 0 and right > 0 and 0.6 < (left / max(1, right)) < 1.67,
    }


FPS_JS = """
async (ms) => {
  const c = document.querySelector('[data-testid="cloud-section"] canvas');
  const gl = c.getContext('webgl2') || c.getContext('webgl');
  const px = new Uint8Array(4);
  let frames = 0;
  const t0 = performance.now();
  await new Promise((res) => {
    const step = () => {
      // GPU fence only, not a content read: forces the pipeline to finish so
      // the sample measures completed frames rather than queued ones.
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      frames += 1;
      if (performance.now() - t0 >= ms) { res(); return; }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
  const elapsed = performance.now() - t0;
  return { frames, ms: Math.round(elapsed), fps: +(frames / (elapsed / 1000)).toFixed(1) };
}
"""

out = {'base': BASE, 'shots': {}, 'fps': {}, 'metrics': {}, 'scenarios': {}}


def shoot(page, name, box, shots=None):
    png = page.screenshot(clip=box)
    (ART / name).write_bytes(png)
    out['shots'][name] = stats(png)
    if shots is not None:
        shots.append((name, png))
    return png


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1600, 'height': 900}, device_scale_factor=1)
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('console', lambda m: errors.append('console.' + m.type + ': ' + m.text) if m.type == 'error' else None)

    # ---------------------------------------------------------------- live
    control('intact')
    page.goto(BASE + '/?drive=full', wait_until='load')
    page.wait_for_function('() => window.__bc && window.__bc.points === 166700', timeout=60000)
    page.wait_for_timeout(3000)
    out['metrics'] = page.evaluate('() => JSON.parse(JSON.stringify(window.__bc))')
    out['caption_text'] = page.inner_text(CAPTION)
    canvas_box = page.query_selector(CANVAS).bounding_box()

    burst = []
    for i in range(5):
        shoot(page, 'hud-cloud-live-%d.png' % i, canvas_box, burst)
        page.wait_for_timeout(220)

    out['motion'] = []
    for i in range(len(burst) - 1):
        d = diff(burst[i][1], burst[i + 1][1])
        d['pair'] = [burst[i][0], burst[i + 1][0]]
        out['motion'].append(d)

    best = max(burst, key=lambda s: out['shots'][s[0]]['warm_px'])
    page.wait_for_timeout(250)
    best_now = shoot(page, 'hud-cloud-live.png', canvas_box)
    out['shots']['hud-cloud-live.png'] = stats(best_now)
    out['best_warm_shot'] = best[0]
    page.screenshot(path=str(ART / 'hud-frame-live.png'))
    out['shots']['hud-frame-live.png'] = stats((ART / 'hud-frame-live.png').read_bytes())

    out['fps']['live_fence_run1'] = page.evaluate(FPS_JS, 2000)
    out['fps']['live_fence_run2'] = page.evaluate(FPS_JS, 2000)
    out['metrics_after'] = page.evaluate('() => JSON.parse(JSON.stringify(window.__bc))')

    # ------------------------------------------------------------ no_edges
    out['control_no_edges'] = control('no_edges')
    page.wait_for_timeout(2500)
    ne = shoot(page, 'hud-cloud-no-edges.png', canvas_box)
    out['metrics_no_edges'] = page.evaluate('() => JSON.parse(JSON.stringify(window.__bc))')
    for i in range(2):
        page.wait_for_timeout(260)
        ne2 = shoot(page, 'hud-cloud-no-edges-%d.png' % (i + 1), canvas_box)
    out['motion_no_edges'] = diff(ne, ne2)

    # ------------------------------------------------------------ restored
    out['control_intact'] = control('intact')
    page.wait_for_timeout(3000)
    re_png = shoot(page, 'hud-cloud-restored.png', canvas_box)
    out['metrics_restored'] = page.evaluate('() => JSON.parse(JSON.stringify(window.__bc))')
    page.wait_for_timeout(300)
    re2 = shoot(page, 'hud-cloud-restored-b.png', canvas_box)
    out['motion_restored'] = diff(re_png, re2)

    # ----------------------------------------------------------- zero props
    page.goto(BASE + '/?props=none&stream=off', wait_until='load')
    page.wait_for_function('() => window.__bc && window.__bc.points === 166700', timeout=60000)
    page.wait_for_timeout(3000)
    zb = page.query_selector(CANVAS).bounding_box()
    zp = shoot(page, 'hud-cloud-zero-props.png', zb)
    out['metrics_zero_props'] = page.evaluate('() => JSON.parse(JSON.stringify(window.__bc))')
    out['caption_zero_props'] = page.inner_text(CAPTION)

    out['page_errors'] = errors[:10]
    browser.close()

# ------------------------------------------------------------- comparisons
live = out['shots']['hud-cloud-live.png']
ne = out['shots']['hud-cloud-no-edges.png']
out['scenarios'] = {
    'live_vs_no_edges_lit_ratio': round(live['lit_pct'] / max(0.001, ne['lit_pct']), 2),
    'live_vs_no_edges_mean_luma_ratio': round(
        (live['mean_rgb'][0] + live['mean_rgb'][1] + live['mean_rgb'][2])
        / max(0.001, ne['mean_rgb'][0] + ne['mean_rgb'][1] + ne['mean_rgb'][2]),
        2,
    ),
    'nodes_visible': live['lit_pct'] > 5.0,
    'amber_visible_live': live['warm_px'] > 0 or out['shots'][out['best_warm_shot']]['warm_px'] > 0,
    'no_edges_dark': ne['lit_pct'] < 0.35 * live['lit_pct'],
    'trail_moving': max(m['moved_pct'] for m in out['motion']) > 0.05,
    'zero_props_renders': out['shots']['hud-cloud-zero-props.png']['lit_pct'] > 3.0,
}
out['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
# Persist as well as print. This script previously only printed its result, so
# artifacts/hud_shots.json kept whatever an older run had left there, and a
# reader measuring that file was measuring a stale build: it still held warm_px
# 0 and no_edges 12.87 percent long after both had changed. File and stdout now
# come from the same `out` dict in the same run.
(ART / 'hud_shots.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(out, indent=1))
