"""Headless smoke check for the viz-fly standalone harness.

Drives the dev server in a real Chromium (headless, swiftshader WebGL) and answers the
questions a screenshot cannot: does the stage mount, does the fly actually move, does each
of the five behaviors change what is drawn, and does no_edges go visibly inert.

    python tools/verify_dev.py [url]

Run it against `bun run dev` (default http://localhost:5191/) or `bun run preview` with the
dist build. Prints a JSON summary and exits non-zero if a hard check fails.
"""
import json
import sys
from io import BytesIO

from PIL import Image
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:5191/'

LAUNCH_ARGS = [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--disable-lcd-text',
]


def brightness(png: bytes) -> float:
    im = Image.open(BytesIO(png)).convert('L')
    px = list(im.getdata())
    return round(sum(px) / len(px), 3)


def moved(a: bytes, b: bytes, thresh: int = 8) -> float:
    """Fraction of pixels that differ by more than `thresh` between two frames."""
    ia = Image.open(BytesIO(a)).convert('L')
    ib = Image.open(BytesIO(b)).convert('L')
    da, db = list(ia.getdata()), list(ib.getdata())
    n = min(len(da), len(db))
    changed = sum(1 for i in range(n) if abs(da[i] - db[i]) > thresh)
    return round(changed / n, 4)


def canvas_shot(page):
    return page.locator('canvas').first.screenshot()


def main() -> int:
    result = {'url': URL, 'console_errors': [], 'page_errors': []}
    failures = []

    with sync_playwright() as p:
        browser = p.chromium.launch(args=LAUNCH_ARGS)
        page = browser.new_page(viewport={'width': 1280, 'height': 800})
        page.on('console', lambda m: result['console_errors'].append(m.text) if m.type == 'error' else None)
        page.on('pageerror', lambda e: result['page_errors'].append(str(e)))

        page.goto(URL, wait_until='load', timeout=30000)
        page.wait_for_selector('canvas', timeout=30000)
        page.wait_for_timeout(2500)

        result['canvas'] = page.evaluate(
            "(() => { const c = document.querySelector('canvas');"
            " return { w: c.width, h: c.height, cw: c.clientWidth, ch: c.clientHeight,"
            " gl: !!(c.getContext('webgl2') || c.getContext('webgl')) }; })()"
        )
        result['buttons'] = page.locator('button').all_inner_texts()
        result['body_text'] = ' | '.join(page.inner_text('body').split('\n'))

        # --- 1. does it animate at all, on the default (intact) mode -------------
        a = canvas_shot(page)
        page.wait_for_timeout(320)
        b = canvas_shot(page)
        page.wait_for_timeout(320)
        c = canvas_shot(page)
        result['intact'] = {
            'brightness': brightness(a),
            'a_to_b': moved(a, b),
            'b_to_c': moved(b, c),
            'a_to_c': moved(a, c),
        }
        if result['canvas']['gl'] is not True:
            failures.append('no WebGL context on the canvas')
        if not (result['intact']['a_to_b'] > 0.002 or result['intact']['a_to_c'] > 0.002):
            failures.append('canvas did not change between frames (fly not animating)')

        # --- 2. the five behavior buttons each change what is drawn -------------
        result['behaviors'] = {}
        for name in ('idle', 'walk', 'groom', 'proboscis', 'startle'):
            btn = page.get_by_role('button', name=name, exact=True)
            if btn.count() == 0:
                failures.append('no %r button in the dev strip' % name)
                continue
            btn.first.click()
            page.wait_for_timeout(700)
            x = canvas_shot(page)
            page.wait_for_timeout(260)
            y = canvas_shot(page)
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

        # --- 3. no_edges must go inert -----------------------------------------
        page.get_by_role('button', name='no_edges', exact=True).first.click()
        page.wait_for_timeout(1600)
        ne = canvas_shot(page)
        zero = page.get_by_role('button', name='idle', exact=True).first
        result['no_edges'] = {
            'brightness': brightness(ne),
            'vs_intact_brightness': round(brightness(ne) - result['intact']['brightness'], 3),
            'badge_visible': 'no_edges' in page.inner_text('body'),
        }
        if not result['no_edges']['badge_visible']:
            failures.append('mode badge for no_edges is not visible')
        if result['no_edges']['brightness'] >= result['intact']['brightness']:
            failures.append('no_edges frame is not dimmer than the intact frame')
        if zero.count() == 0:
            failures.append('dev strip vanished after switching to no_edges')

        # --- 4. no props at all: the frozen contract's synthetic idle path ------
        page.get_by_role('button', name='no props', exact=True).first.click()
        page.wait_for_timeout(1500)
        n1 = canvas_shot(page)
        page.wait_for_timeout(340)
        n2 = canvas_shot(page)
        result['no_props'] = {'brightness': brightness(n1), 'moved': moved(n1, n2)}
        if result['no_props']['moved'] <= 0.002:
            failures.append('FlyStage with no props is not animating')

        # --- 5. correct=true / false drive the reaction overlays ----------------
        page.get_by_role('button', name='intact', exact=True).first.click()
        page.wait_for_timeout(900)
        page.get_by_role('button', name='correct=false', exact=True).first.click()
        page.wait_for_timeout(500)
        r1 = canvas_shot(page)
        page.wait_for_timeout(300)
        r2 = canvas_shot(page)
        result['recoil'] = {'brightness': brightness(r1), 'moved': moved(r1, r2)}
        page.get_by_role('button', name='correct=true', exact=True).first.click()
        page.wait_for_timeout(500)
        c1 = canvas_shot(page)
        result['celebrate'] = {
            'brightness': brightness(c1),
            'vs_recoil_brightness': round(brightness(c1) - result['recoil']['brightness'], 3),
        }

        browser.close()

    for rec in result.get('behaviors', {}).values():
        rec.pop('png', None)

    result['failures'] = failures
    result['ok'] = not failures
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
