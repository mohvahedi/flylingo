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
    """Is the fly actually there, and how much of the panel does it own.

    `near_white_frac` is the clipping metric: the fraction of the frame inside 20 luma of
    pure white. A frame that pins its brightest areas at 255 has no separation left at the
    top of the range, which is what the previous backlight did (max_luma 252, wings washed
    out). It is reported per state so a dead frame cannot hide behind an average.
    """
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
        'p999_luma': int(np.percentile(l, 99.9)),
        'max_luma': int(l.max()),
        'near_white_frac': round(float((l > 235).mean()), 4),
        'near_white_pixels': int((l > 235).sum()),
        'over_200_frac': round(float((l > 200).mean()), 4),
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
        # three frames averaged for the shadow A/B below: the idle pose keeps breathing, and
        # a single frame against a three frame mean would read as a false difference
        hero_frames = [clean_shot(page) for _ in range(3)]
        hero_clean = hero_frames[-1]
        save(f'{ART}/viz-fly-hero.png', hero_clean)
        report['results']['hero_clean'] = visibility(hero_clean)

        stats = page.evaluate('window.__flyStats || null')
        report['results']['stats'] = stats
        # the renderer's own settings and the setae count, read back from the live page rather
        # than copied out of the source
        report['results']['settings'] = page.evaluate(
            '({ gl: window.__flyGl || null, bloom: window.__flyBloom || null,'
            ' setae: (typeof window.__flySetae === "number" ? window.__flySetae : null),'
            ' probe_source: (window.__flyProbe && window.__flyProbe.source) || null })'
        )

        # ---------------------------------------------------------------- shadow A/B
        # Same camera (?frozen), same frame size, same lights, same materials: the only
        # difference is the shadow switch. So anything that got darker in the shadowed frame
        # is shadow, and the floor region below and left of the fly is where the key light's
        # wedged cast shadow lands.
        #
        # Before switching, read the contact patch instances back out of the live scene: seven
        # quads, six of them on the tarsus x/z the probe reports, all of them on the floor.
        contacts = page.evaluate(
            '''(() => {
              const m = window.__flyContacts;
              const p = window.__flyProbe;
              if (!m || !p) return null;
              const a = m.instanceMatrix.array;
              const t = [];
              for (let i = 0; i < m.count; i++) t.push([a[i * 16 + 12], a[i * 16 + 13], a[i * 16 + 14]]);
              return { count: m.count, translations: t, visible: m.visible,
                       feet_xz: p.foot.map((f) => [f[0], f[2]]) };
            })()'''
        )
        if contacts:
            feet = contacts['feet_xz']
            trans = contacts['translations']
            # Bound by BOTH lists. Guarding only `feet` indexes past the end of
            # `translations` whenever the instance list is shorter, which crashed this
            # diagnostic and told us nothing. Reporting the raw lengths is the point of a
            # verification script: an unexpected shape should be visible, not fatal.
            n = min(6, len(feet), len(trans))
            # feet_xz rows are [x, z] (the probe maps foot to [f[0], f[2]]), so the z
            # component is index 1, not 2. Indexing 2 read past the end of a 2-element row.
            dist = [
                round(float(((trans[i][0] - feet[i][0]) ** 2
                             + (trans[i][2] - feet[i][1]) ** 2) ** 0.5), 4)
                for i in range(n)
            ]
            report['results']['contact_shadows'] = {
                'instances': contacts['count'],
                'translations_len': len(trans),
                'feet_len': len(feet),
                'compared': n,
                'visible': contacts['visible'],
                'layer_y': [round(float(t[1]), 4) for t in contacts['translations']],
                'foot_to_patch_xz': dist,
                'max_foot_to_patch_xz': max(dist) if dist else None,
                'note': 'six tarsus patches (indices 0-5) plus one thorax pool (index 6); '
                        'y is the layer height above the floor, xz is checked against the '
                        'foot positions the same probe publishes',
            }
        page.goto(f'{BASE}?frozen&noshadow', wait_until='load')
        page.wait_for_function('() => !!window.__flyProbe', timeout=60000)
        settle(page, 6000)
        noshadow_frames = [clean_shot(page) for _ in range(3)]
        save(f'{ART}/viz-fly-hero-noshadow.png', noshadow_frames[-1])
        report['results']['hero_clean_noshadow'] = visibility(noshadow_frames[-1])

        with_shadow = mean_img(hero_frames).astype(np.float32)
        without = mean_img(noshadow_frames).astype(np.float32)
        delta = with_shadow - without
        darker = delta < -6
        shadow_report = {
            'darker_pixels': int(darker.sum()),
            'darker_frac': round(float(darker.mean()), 4),
            'mean_darkening_where_darker': round(float(delta[darker].mean()), 2) if darker.any() else 0.0,
            'max_darkening': round(float(delta.min()), 2),
            'mean_luma_with_shadow': round(float(with_shadow.mean()), 2),
            'mean_luma_without_shadow': round(float(without.mean()), 2),
        }
        box = report['results']['hero_clean'].get('lit_bbox')
        if box:
            x0, y0, x1, y1 = box
            wb = max(24, x1 - x0)
            hb = max(24, y1 - y0)
            rx0 = max(0, int(x0 - 0.7 * wb))
            rx1 = min(with_shadow.shape[1], int(x0 + 0.6 * wb))
            ry0 = max(0, int(y1 - 0.1 * hb))
            ry1 = min(with_shadow.shape[0], int(y1 + 1.0 * hb))
            shadow_report['floor_roi'] = {
                'roi_xyxy': [rx0, ry0, rx1, ry1],
                'note': 'below and left of the fly, where the key light throws the shadow',
                'mean_with_shadow': round(float(with_shadow[ry0:ry1, rx0:rx1].mean()), 2),
                'mean_without_shadow': round(float(without[ry0:ry1, rx0:rx1].mean()), 2),
            }
        report['results']['shadow'] = shadow_report

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
                'non_background_frac': v['visibility']['non_background_frac'],
                'mean_luma': v['visibility']['mean_luma'],
                'p99_luma': v['visibility']['p99_luma'],
                'max_luma': v['visibility']['max_luma'],
                'near_white_frac': v['visibility']['near_white_frac'],
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
        live_png = clean_shot(page)
        save(f'{ART}/viz-fly-mode-intact.png', live_png)
        page.get_by_role('button', name='no_edges', exact=True).click()
        settle(page, 2500)
        inert_probe = probe_range(page, 1400)
        dead_png = clean_shot(page)
        save(f'{ART}/viz-fly-mode-no_edges.png', dead_png)
        # And again after a long settle. The animator's release constants are per FRAME, not
        # per second, so under software rasterisation (single digit fps) the drive and the
        # vitality decay slowly in wall clock terms and leave a visible tail on an inert
        # frame. At 60 fps the same number of frames passes 20x faster. Sampling both is how
        # the tail gets attributed rather than guessed at.
        settle(page, 10000)
        inert_settled = probe_range(page, 1400)
        report['results']['glow'] = {
            'intact_glow_max': live_probe.get('glow_max'),
            'intact_glow_last': live_probe.get('glow_last'),
            'no_edges_glow_max': inert_probe.get('glow_max'),
            'no_edges_glow_last': inert_probe.get('glow_last'),
            'no_edges_glow_max_settled': inert_settled.get('glow_max'),
            'no_edges_glow_last_settled': inert_settled.get('glow_last'),
            # the same two modes as pixels, not just as probe values: the dead frame has to be
            # the dimmer one, or the "inert" claim is only true of a number
            'pixels': {
                'intact': visibility(live_png),
                'no_edges': visibility(dead_png),
                'panel_note': 'measured at 1024x640, the state sweep viewport',
            },
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
        # Every number here is headless Chromium on SwiftShader, a software rasteriser, on a
        # machine whose real GPU is an RTX 3070 Ti. So it is a FLOOR, not the panel's frame
        # rate, and quoting it bare would be misleading. What it is good for is the ratio
        # between paths, which is why all three are measured the same way in one session.
        page.unroute('**/models/fly.glb')
        fps = {}
        for label, url, has_probe in (
            ('model_bloom', f'{BASE}?frozen', True),
            ('model_nobloom', f'{BASE}?frozen&nobloom', True),
            ('procedural_bloom', f'{BASE}?frozen&procedural', False),
        ):
            try:
                page.goto(url, wait_until='load')
                if has_probe:
                    page.wait_for_function('() => !!window.__flyProbe', timeout=60000)
                settle(page, 4000)
                dts = page.evaluate(FPS_JS)
                med = float(np.median(dts))
                fps[label] = {
                    'median_ms': round(med, 2),
                    'fps': round(1000 / med, 1) if med > 0 else None,
                }
            except Exception as exc:  # noqa: BLE001
                fps[label] = {'error': str(exc)}
        try:
            fps['model_over_procedural'] = round(
                fps['model_bloom']['median_ms'] / fps['procedural_bloom']['median_ms'], 2
            )
            fps['bloom_over_nobloom'] = round(
                fps['model_bloom']['median_ms'] / fps['model_nobloom']['median_ms'], 2
            )
        except (KeyError, TypeError, ZeroDivisionError):
            pass
        fps['renderer'] = (report['results'].get('canvas') or {}).get('renderer')
        fps['note'] = (
            'SwiftShader software rasterisation: a floor, not the panel frame rate. The '
            'ratios between paths are the meaningful part of these numbers.'
        )
        report['results']['fps'] = fps

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
