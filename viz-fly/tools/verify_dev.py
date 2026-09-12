"""Headless witness for the viz-fly hero panel.

Drives the dev server in a real Chromium (headless, swiftshader WebGL) and measures what a
screenshot alone cannot tell you:

  * how much of the panel is actually lit fly and not background,
  * that each of the five behaviours draws something different from idle,
  * that no_edges (all 512 values exactly zero) goes visibly inert,
  * that the frozen prop contract still works with no props at all,
  * what the scene costs in triangles and draw calls.

    python tools/verify_dev.py [url]

Defaults to http://localhost:5191/ (bun run dev). Prints a JSON summary, writes PNGs to
D:/Projects/flylingo/artifacts/, and exits non-zero if a hard check fails.

MEASUREMENT NOTE: never use WebGL readPixels to decide whether this panel rendered.
react-three-fiber does not set preserveDrawingBuffer, so an out-of-frame readPixels returns
all zeros and reports a working scene as black. Everything here reads the composited canvas
back through an element screenshot instead.
"""
import json
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:5191/'
ART = 'D:/Projects/flylingo/artifacts'
VIEW = {'width': 1440, 'height': 900}

# the panel's cleared colour, set in FlyStage on the renderer and on the wrapper
BG = (8, 12, 17)
# luma above which a pixel is unambiguously lit geometry, not floor and not scrim
LIT = 40

STATES = ('idle', 'walk', 'groom', 'proboscis', 'startle')

LAUNCH_ARGS = [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--disable-lcd-text',
]

# hides every HUD overlay that sits on top of the canvas, so a pixel measurement sees the
# render and nothing else. The dev strip is a DOM sibling of the canvas, not part of it.
HIDE_JS = """(() => {
  const c = document.querySelector('canvas');
  if (!c || !c.parentElement) return 0;
  let n = 0;
  for (const el of Array.from(c.parentElement.children)) {
    if (el.tagName !== 'CANVAS') { el.style.display = 'none'; n += 1; }
  }
  return n;
})()"""

SHOW_JS = """(() => {
  const c = document.querySelector('canvas');
  if (!c || !c.parentElement) return 0;
  for (const el of Array.from(c.parentElement.children)) {
    if (el.tagName !== 'CANVAS') el.style.display = '';
  }
  return 1;
})()"""

CANVAS_JS = """(() => {
  const c = document.querySelector('canvas');
  if (!c) return null;
  const gl = c.getContext('webgl2') || c.getContext('webgl');
  const dbg = gl && gl.getExtension('WEBGL_debug_renderer_info');
  return {
    w: c.width,
    h: c.height,
    cw: c.clientWidth,
    ch: c.clientHeight,
    webgl2: !!(c.getContext('webgl2')),
    renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : (gl ? gl.getParameter(gl.RENDERER) : null),
  };
})()"""

FPS_JS = """() => new Promise((res) => {
  const dts = [];
  let last = performance.now();
  const step = () => {
    const now = performance.now();
    dts.push(now - last);
    last = now;
    if (dts.length < 70) requestAnimationFrame(step);
    else res(dts.slice(10));
  };
  requestAnimationFrame(step);
})"""


def rgb(png: bytes) -> np.ndarray:
    # int32, not int16: the luma weights below multiply by 299/587 and silently wrap in
    # int16, which reports a fully lit frame as near black.
    return np.asarray(Image.open(BytesIO(png)).convert('RGB'), dtype=np.int32)


def luma(a: np.ndarray) -> np.ndarray:
    return (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000


def brightness(png: bytes) -> float:
    return round(float(luma(rgb(png)).mean()), 3)


def moved(a: bytes, b: bytes, thresh: int = 8) -> float:
    """Fraction of pixels that differ by more than `thresh` between two frames."""
    la, lb = luma(rgb(a)).ravel(), luma(rgb(b)).ravel()
    n = min(la.size, lb.size)
    return round(float((np.abs(la[:n].astype(np.int32) - lb[:n].astype(np.int32)) > thresh).mean()), 4)


def visibility(png: bytes, bg=BG, lit=LIT) -> dict:
    """Is the fly actually there, and how much of the panel does it own."""
    a = rgb(png)
    h, w, _ = a.shape
    dist = np.abs(a - np.array(bg, dtype=np.int32)).sum(axis=2)
    l = luma(a)
    mask = l > lit
    ys, xs = np.nonzero(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None
    return {
        'panel': [w, h],
        'non_background_frac': round(float((dist > 12).mean()), 4),
        'lit_frac': round(float(mask.mean()), 4),
        'lit_pixels': int(mask.sum()),
        'lit_bbox': box,
        'mean_luma': round(float(l.mean()), 2),
        'p50_luma': int(np.percentile(l, 50)),
        'p99_luma': int(np.percentile(l, 99)),
        'max_luma': int(l.max()),
    }


def mean_img(pngs) -> np.ndarray:
    return np.stack([luma(rgb(p)).astype(np.float32) for p in pngs]).mean(axis=0)


def frac_diff(ma: np.ndarray, mb: np.ndarray, thresh: float = 6.0) -> float:
    return round(float((np.abs(ma - mb) > thresh).mean()), 4)


def save(path: str, png: bytes) -> None:
    with open(path, 'wb') as fh:
        fh.write(png)


def shot(page, clean: bool = False) -> bytes:
    """Element screenshot of the WebGL canvas, optionally with the HUD hidden."""
    if clean:
        page.evaluate(HIDE_JS)
        page.wait_for_timeout(50)
    png = page.locator('canvas').first.screenshot()
    if clean:
        page.evaluate(SHOW_JS)
    return png


def fps(page) -> dict:
    dts = np.asarray(page.evaluate(FPS_JS), dtype=np.float64)
    return {
        'median_frame_ms': round(float(np.median(dts)), 2),
        'p90_frame_ms': round(float(np.percentile(dts, 90)), 2),
        'fps': round(1000.0 / float(np.median(dts)), 1),
        'samples': int(dts.size),
    }


def main() -> int:
    result = {'url': URL, 'view': VIEW, 'console_errors': [], 'page_errors': []}
    failures = []

    with sync_playwright() as p:
        browser = p.chromium.launch(args=LAUNCH_ARGS)
        page = browser.new_page(viewport=VIEW)
        page.on(
            'console',
            lambda m: result['console_errors'].append(m.text) if m.type == 'error' else None,
        )
        page.on('pageerror', lambda e: result['page_errors'].append(str(e)))

        page.goto(URL, wait_until='load', timeout=45000)
        page.wait_for_selector('canvas', timeout=45000)
        page.wait_for_timeout(4000)

        result['canvas'] = page.evaluate(CANVAS_JS)
        result['stats'] = page.evaluate('() => window.__flyStats || null')
        result['buttons'] = page.locator('button').all_inner_texts()
        result['perf_bloom_on'] = fps(page)

        # --- 1. is the fly actually visible against the near black panel --------------
        canvas_png = shot(page, clean=True)
        result['visibility'] = visibility(canvas_png)
        save(ART + '/viz-fly-canvas.png', canvas_png)
        page.screenshot(path=ART + '/viz-fly-hero.png')
        vis = result['visibility']
        if result['canvas'] is None or result['canvas']['webgl2'] is not True:
            failures.append('no WebGL2 context on the canvas')
        if vis['lit_frac'] < 0.004:
            failures.append('fly is not visibly lit: lit_frac %.4f below 0.4%%' % vis['lit_frac'])
        if vis['non_background_frac'] < 0.02:
            failures.append('panel is almost all background: %.4f' % vis['non_background_frac'])
        if vis['p99_luma'] < 90:
            failures.append('no bright pixels at all: p99 luma %d' % vis['p99_luma'])

        # --- 2. does it animate at all, on the default (intact) mode -------------------
        a = shot(page)
        page.wait_for_timeout(320)
        b = shot(page)
        page.wait_for_timeout(320)
        c = shot(page)
        result['intact'] = {
            'brightness': brightness(a),
            'a_to_b': moved(a, b),
            'b_to_c': moved(b, c),
            'a_to_c': moved(a, c),
        }
        if not (result['intact']['a_to_b'] > 0.002 or result['intact']['a_to_c'] > 0.002):
            failures.append('canvas did not change between frames (fly not animating)')

        # --- 3. the five behaviour buttons each change what is drawn -------------------
        result['behaviors'] = {}
        for name in STATES:
            btn = page.get_by_role('button', name=name, exact=True)
            if btn.count() == 0:
                failures.append('no %r button in the dev strip' % name)
                continue
            btn.first.click()
            page.wait_for_timeout(700)
            x = shot(page)
            page.wait_for_timeout(260)
            y = shot(page)
            result['behaviors'][name] = {
                'brightness': brightness(x),
                'moved': moved(x, y),
                'vs_idle': None,
            }
            if name != 'idle':
                prev = result['behaviors'].get('idle', {}).get('png')
                if prev:
                    result['behaviors'][name]['vs_idle'] = moved(prev, x)
            result['behaviors'][name]['png'] = x

        for name, rec in result['behaviors'].items():
            if rec['moved'] < 0.002:
                failures.append('behavior %r looks frozen' % name)

        idle_png = result['behaviors'].get('idle', {}).get('png')
        for name in ('walk', 'groom', 'proboscis', 'startle'):
            rec = result['behaviors'].get(name)
            if rec and idle_png and moved(idle_png, rec['png']) <= 0.002:
                failures.append('behavior %r draws the same thing as idle' % name)

        # --- 4. no_edges must go inert -------------------------------------------------
        page.get_by_role('button', name='no_edges', exact=True).first.click()
        page.wait_for_timeout(1600)
        ne = shot(page)
        zero = page.get_by_role('button', name='idle', exact=True).first
        result['no_edges'] = {
            'brightness': brightness(ne),
            'vs_intact_brightness': round(brightness(ne) - result['intact']['brightness'], 3),
            'lit_frac': visibility(ne)['lit_frac'],
            'badge_visible': 'no_edges' in page.inner_text('body'),
        }
        save(ART + '/viz-fly-state-no_edges.png', shot(page, clean=True))
        if not result['no_edges']['badge_visible']:
            failures.append('mode badge for no_edges is not visible')
        if result['no_edges']['brightness'] >= result['intact']['brightness']:
            failures.append('no_edges frame is not dimmer than the intact frame')
        if zero.count() == 0:
            failures.append('dev strip vanished after switching to no_edges')

        # --- 5. no props at all: the frozen contract's synthetic idle path ---------------
        page.get_by_role('button', name='no props', exact=True).first.click()
        page.wait_for_timeout(1500)
        n1 = shot(page)
        page.wait_for_timeout(340)
        n2 = shot(page)
        result['no_props'] = {
            'brightness': brightness(n1),
            'moved': moved(n1, n2),
            'lit_frac': visibility(n1)['lit_frac'],
        }
        if result['no_props']['moved'] <= 0.002:
            failures.append('FlyStage with no props is not animating')

        # --- 6. correct=true / false drive the reaction overlays -----------------------
        page.get_by_role('button', name='intact', exact=True).first.click()
        page.wait_for_timeout(900)
        page.get_by_role('button', name='correct=false', exact=True).first.click()
        page.wait_for_timeout(500)
        r1 = shot(page)
        page.wait_for_timeout(300)
        r2 = shot(page)
        result['recoil'] = {'brightness': brightness(r1), 'moved': moved(r1, r2)}
        page.get_by_role('button', name='correct=true', exact=True).first.click()
        page.wait_for_timeout(500)
        c1 = shot(page)
        result['celebrate'] = {
            'brightness': brightness(c1),
            'vs_recoil_brightness': round(brightness(c1) - result['recoil']['brightness'], 3),
        }

        # --- 7. frozen camera: is each behaviour different from idle, or is the orbit
        #        doing the work? With the camera held still, every pixel that moves is
        #        the fly. Idle is sampled twice to establish the breathing noise floor.
        page.goto(URL + '?frozen', wait_until='load', timeout=45000)
        page.wait_for_selector('canvas', timeout=45000)
        page.wait_for_timeout(3500)

        def state_mean(name: str, n: int = 4):
            btn = page.get_by_role('button', name=name, exact=True)
            if btn.count() == 0:
                return None
            btn.first.click()
            page.wait_for_timeout(900)
            frames = [shot(page, clean=True) for _ in range(n)]
            page.wait_for_timeout(140)
            save(ART + '/viz-fly-state-%s.png' % name, frames[-1])
            return mean_img(frames)

        idle_a = state_mean('idle')
        idle_b = state_mean('idle')
        floor = frac_diff(idle_a, idle_b) if idle_a is not None and idle_b is not None else None
        idle_mean = (idle_a + idle_b) / 2 if floor is not None else None

        result['frozen_camera'] = {'idle_vs_idle_floor': floor, 'states': {}}
        if idle_mean is None:
            failures.append('idle state could not be sampled with the camera frozen')
        else:
            for name in ('walk', 'groom', 'proboscis', 'startle'):
                m = state_mean(name)
                if m is None:
                    failures.append('no %r button in the dev strip (frozen pass)' % name)
                    continue
                d = frac_diff(idle_mean, m)
                result['frozen_camera']['states'][name] = {'vs_idle_pixels_differing': d}
                if d <= max(2.5 * floor, 0.002):
                    failures.append(
                        'state %r is not distinguishable from idle with the camera frozen '
                        '(%.4f vs floor %.4f)' % (name, d, floor)
                    )

        # --- 8. bloom off: the same panel, one prop, for the cost comparison ------------
        page.goto(URL + '?frozen&nobloom', wait_until='load', timeout=45000)
        page.wait_for_selector('canvas', timeout=45000)
        page.wait_for_timeout(3500)
        nb = shot(page, clean=True)
        result['perf_bloom_off'] = fps(page)
        result['visibility_nobloom'] = visibility(nb)
        save(ART + '/viz-fly-nobloom.png', nb)

        browser.close()

    for rec in result.get('behaviors', {}).values():
        rec.pop('png', None)

    if result['console_errors']:
        failures.append('%d console error(s): %s' % (len(result['console_errors']), result['console_errors'][:2]))
    if result['page_errors']:
        failures.append('%d page error(s): %s' % (len(result['page_errors']), result['page_errors'][:2]))

    result['failures'] = failures
    result['ok'] = not failures
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
