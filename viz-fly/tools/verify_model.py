"""Headless verification of the real-mesh hero path.

Measures the panel the way the task asks for: element screenshots of the canvas, analysed as
PNGs, plus a read of the live node-position probe the scene publishes.

Why not WebGL readPixels: react-three-fiber does not set preserveDrawingBuffer, so an
out-of-frame readPixels returns all zeros and reports a perfectly working scene as black.
That trap has already cost this project one round, so this script never touches readPixels.

What it produces, all in D:/Projects/flylingo/artifacts:
  viz-fly-hero.png            the hero shot, bloom on, HUD overlays hidden
  viz-fly-hero-hud.png        the same frame with the real HUD overlay in place
  viz-fly-state-<name>.png    one per animation state
  viz-fly-model-verify.json   every number printed, for the record

Usage: python tools/verify_model.py [base_url]
"""
import json
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:5191/'
ART = 'D:/Projects/flylingo/artifacts'
VIEW = {'width': 1440, 'height': 900}
# the panel's cleared colour, set in FlyStage on the renderer and on the wrapper
BG = (8, 12, 17)
# luma above which a pixel is unambiguously lit geometry, not floor and not scrim
LIT = 40

STATES = ('idle', 'walk', 'groom', 'proboscis', 'startle')
# enough frames to average out the phase of a periodic walk cycle
FRAMES_PER_STATE = 5
# a smaller panel for the state sweep: same numbers, far less software rasterisation
SWEEP_VIEW = {'width': 1024, 'height': 640}

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
    w: c.width, h: c.height, cw: c.clientWidth, ch: c.clientHeight,
    webgl2: !!(c.getContext('webgl2')),
    renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : null,
  };
})()"""

FPS_JS = """() => new Promise((res) => {
  const dts = [];
  let last = performance.now();
  const step = () => {
    const now = performance.now();
    dts.push(now - last);
    last = now;
    if (dts.length < 40) requestAnimationFrame(step);
    else res(dts.slice(8));
  };
  requestAnimationFrame(step);
})"""


def rgb(png: bytes) -> np.ndarray:
    # int32, not int16: the luma weights below multiply by 299/587 and silently wrap in
    # int16, which reports a fully lit frame as near black.
    return np.asarray(Image.open(BytesIO(png)).convert('RGB'), dtype=np.int32)


def luma(a: np.ndarray) -> np.ndarray:
    return (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000


def visibility(png: bytes, lit=LIT) -> dict:
    """Is the fly actually there, and how much of the panel does it own."""
    a = rgb(png)
    h, w, _ = a.shape
    dist = np.abs(a - np.array(BG, dtype=np.int32)).sum(axis=2)
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


def shot(page) -> bytes:
    """Element screenshot of the WebGL canvas. Deliberately not a page screenshot.

    The selector is canvas[data-engine], not canvas: the sparkline overlay draws into its
    own 2D canvas, so a bare locator('canvas') is ambiguous.
    """
    return page.locator('canvas[data-engine]').screenshot()


def clean_shot(page) -> bytes:
    page.evaluate(HIDE_JS)
    page.wait_for_timeout(120)
    png = shot(page)
    page.evaluate(SHOW_JS)
    return png


def settle(page, ms: int = 900) -> None:
    page.wait_for_timeout(ms)


def probe(page):
    return page.evaluate('window.__flyProbe || null')


def probe_range(page, ms: int, hz: int = 20) -> dict:
    """Sample the node-position probe for `ms` and report how far each node travelled.

    This is the honest replacement for reading the framebuffer: it says where the animated
    nodes ended up, so a walk cycle can be proven to move feet rather than merely change
    pixels.
    """
    samples = []
    steps = max(2, int(ms / 1000 * hz))
    for _ in range(steps):
        p = probe(page)
        if p:
            samples.append(p)
        page.wait_for_timeout(int(1000 / hz))
    if not samples:
        return {'samples': 0}
    out = {'samples': len(samples)}
    # wing tips
    for i in range(2):
        pts = np.array([s['wing'][i] for s in samples], dtype=float)
        out[f'wing{i}_excursion'] = round(float(np.linalg.norm(pts.max(0) - pts.min(0))), 5)
    # the deepest joint of each leg chain
    exc = []
    for i in range(6):
        pts = np.array([s['foot'][i] for s in samples], dtype=float)
        exc.append(round(float(np.linalg.norm(pts.max(0) - pts.min(0))), 5))
    out['foot_excursion'] = exc
    out['foot_excursion_max'] = round(float(max(exc)), 5)
    head = np.array([s['head'] for s in samples], dtype=float)
    out['head_excursion'] = round(float(np.linalg.norm(head.max(0) - head.min(0))), 5)
    out['behavior'] = samples[-1].get('behavior')
    out['glow_last'] = [round(float(x), 4) for x in samples[-1].get('glow', [])]
    out['glow_max'] = round(float(max(max(s.get('glow', [0])) for s in samples)), 4)
    out['span_last'] = [round(float(x), 4) for x in samples[-1].get('span', [])]
    return out


def main() -> int:
    report = {'base': BASE, 'console': [], 'results': {}}
    errors = []
    warns = []
    rig_logs = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=LAUNCH_ARGS)
        page = browser.new_page(viewport=VIEW)

        def on_console(msg):
            text = msg.text
            if msg.type == 'error':
                errors.append(text)
            elif msg.type == 'warning':
                warns.append(text)
            if '[fly] rig' in text:
                rig_logs.append(text)

        page.on('console', on_console)
        page.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))

        # ---------------------------------------------------------------- hero path
        page.goto(f'{BASE}?frozen', wait_until='load')
        page.wait_for_function('() => !!window.__flyProbe', timeout=60000)
        settle(page, 6000)  # let the studio env bake, textures decode and the pose settle

        p = probe(page)
        report['results']['model_path'] = {
            'probe_present': bool(p),
            'behavior': p.get('behavior') if p else None,
            'source': p.get('source') if p else None,
            'span': [round(float(x), 4) for x in (p.get('span') or [])],
            'glow': [round(float(x), 4) for x in (p.get('glow') or [])],
        }
        report['results']['canvas'] = page.evaluate(CANVAS_JS)

        hero = shot(page)
        save(f'{ART}/viz-fly-hero-hud.png', hero)
        report['results']['hero_with_hud'] = visibility(hero)
        hero_clean = clean_shot(page)
        save(f'{ART}/viz-fly-hero.png', hero_clean)
        report['results']['hero_clean'] = visibility(hero_clean)

        stats = page.evaluate('window.__flyStats || null')
        report['results']['stats'] = stats

        # ------------------------------------------------------------------ state sweep
        page.set_viewport_size(SWEEP_VIEW)
        page.goto(f'{BASE}?frozen&nobloom', wait_until='load')
        page.wait_for_function('() => !!window.__flyProbe', timeout=60000)
        settle(page, 5000)

        states = {}
        for name in STATES:
            page.get_by_role('button', name=name, exact=True).click()
            settle(page, 2200)  # blend in over BLEND_TIME, then hold
            frames = []
            for _ in range(FRAMES_PER_STATE):
                frames.append(clean_shot(page))
                page.wait_for_timeout(220)
            save(f'{ART}/viz-fly-state-{name}.png', frames[-1])
            states[name] = {
                'mean': mean_img(frames),
                'visibility': visibility(frames[-1]),
                'probe': probe_range(page, 1400),
            }

        base = states['idle']['mean']
        for name in STATES:
            if name == 'idle':
                continue
            states[name]['frac_diff_vs_idle'] = frac_diff(base, states[name]['mean'])
        report['results']['states'] = {
            k: {
                'frac_diff_vs_idle': v.get('frac_diff_vs_idle'),
                'behavior': v['probe'].get('behavior'),
                'lit_frac': v['visibility']['lit_frac'],
                'foot_excursion_max': v['probe'].get('foot_excursion_max'),
                'wing0_excursion': v['probe'].get('wing0_excursion'),
                'head_excursion': v['probe'].get('head_excursion'),
                'glow_max': v['probe'].get('glow_max'),
            }
            for k, v in states.items()
        }
        report['results']['foot_excursion_all'] = {
            k: v['probe'].get('foot_excursion') for k, v in states.items()
        }

        # ------------------------------------------------------------------- reactions
        page.get_by_role('button', name='idle', exact=True).click()
        settle(page, 2500)
        calm = mean_img([clean_shot(page) for _ in range(3)])
        page.get_by_role('button', name='correct=true', exact=True).click()
        page.wait_for_timeout(350)
        celebrate = mean_img([clean_shot(page) for _ in range(3)])
        settle(page, 2500)
        page.get_by_role('button', name='correct=false', exact=True).click()
        page.wait_for_timeout(250)
        recoil = mean_img([clean_shot(page) for _ in range(3)])
        report['results']['reactions'] = {
            'celebrate_frac_diff_vs_idle': frac_diff(calm, celebrate),
            'recoil_frac_diff_vs_idle': frac_diff(calm, recoil),
        }

        # ------------------------------------------------------- glow range and inert
        page.get_by_role('button', name='idle', exact=True).click()
        page.get_by_role('button', name='intact', exact=True).click()
        settle(page, 2500)
        live_probe = probe_range(page, 1400)
        page.get_by_role('button', name='no_edges', exact=True).click()
        settle(page, 2500)
        inert_probe = probe_range(page, 1400)
        report['results']['glow'] = {
            'intact_glow_max': live_probe.get('glow_max'),
            'intact_glow_last': live_probe.get('glow_last'),
            'no_edges_glow_max': inert_probe.get('glow_max'),
            'no_edges_glow_last': inert_probe.get('glow_last'),
        }

        # ----------------------------------------------------------------- fallback path
        page.set_viewport_size(VIEW)
        page.goto(f'{BASE}?frozen&procedural', wait_until='load')
        settle(page, 5000)
        proc = visibility(clean_shot(page))
        report['results']['fallback_forced'] = proc

        # and the real failure mode: the asset 404s or is unreachable
        page.route('**/models/fly.glb', lambda route: route.abort())
        page.on('console', on_console)
        page.goto(f'{BASE}?frozen', wait_until='load')
        settle(page, 7000)
        try:
            broken = visibility(clean_shot(page))
        except Exception as exc:  # noqa: BLE001
            broken = {'error': str(exc)}
        broken['probe_present'] = bool(probe(page))
        report['results']['fallback_on_asset_failure'] = broken

        # --------------------------------------------------------------------- fps
        page.unroute('**/models/fly.glb')
        page.goto(f'{BASE}?frozen', wait_until='load')
        page.wait_for_function('() => !!window.__flyProbe', timeout=60000)
        settle(page, 4000)
        try:
            dts = page.evaluate(FPS_JS)
            med = float(np.median(dts))
            report['results']['fps'] = {
                'median_ms': round(med, 2),
                'fps': round(1000 / med, 1) if med > 0 else None,
            }
        except Exception as exc:  # noqa: BLE001
            report['results']['fps'] = {'error': str(exc)}

        report['console'] = {
            'error_count': len(errors),
            'errors': errors[:20],
            'webglprogram_errors': [e for e in errors if 'WebGLProgram' in e or 'THREE.WebGLProgram' in e],
            'warn_count': len(warns),
            'warns': warns[:10],
            'rig_logs': rig_logs,
        }
        browser.close()

    with open(f'{ART}/viz-fly-model-verify.json', 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2)[:3000])
    print(json.dumps(report['results'], indent=2, default=str)[:9000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
