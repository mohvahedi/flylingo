"""Fast smoke check: does the real mesh mount, and are there any console errors.

Separate from verify_model.py because the full sweep under SwiftShader takes minutes and
the useful question first is simply "did the model load at all".
Usage: python tools/smoke.py [base_url]
"""
import json
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:5191/'
ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']

with sync_playwright() as pw:
    browser = pw.chromium.launch(args=ARGS)
    page = browser.new_page(viewport={'width': 900, 'height': 600})
    errors, logs = [], []
    page.on('console', lambda m: (errors if m.type == 'error' else logs).append(m.text))
    page.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))

    page.goto(f'{BASE}?frozen', wait_until='load')
    try:
        page.wait_for_function('() => !!window.__flyProbe', timeout=45000)
        ok = True
    except Exception:  # noqa: BLE001
        ok = False
    page.wait_for_timeout(3000)

    p = page.evaluate('window.__flyProbe || null')
    out = {
        'probe_present': ok,
        'probe': p,
        'stats': page.evaluate('window.__flyStats || null'),
        'canvas': page.evaluate(
            "(() => { const c=document.querySelector('canvas'); return c ? {w:c.width,h:c.height} : null; })()"
        ),
        'errors': errors[:10],
        'error_count': len(errors),
        'rig_logs': [t for t in logs if '[fly]' in t],
    }
    png = page.locator('canvas[data-engine]').screenshot()
    with open('D:/Projects/flylingo/artifacts/viz-fly-smoke.png', 'wb') as fh:
        fh.write(png)
    out['png_bytes'] = len(png)
    browser.close()

print(json.dumps(out, indent=2, default=str))
