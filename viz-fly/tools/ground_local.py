"""Local grounding measurement for the viz-fly hero stage.

The question this answers: does the fly read as standing ON a lit surface, or floating in a
void? The eye judges that by LOCAL contrast - how much darker the shadow is than the lit
ground immediately beside it - so hand-placed boxes are the wrong instrument: the 17%
figure in HANDOFF_grounding.md came from a box pair that was placed in the near floor
beside the fly's legs rather than in the shadow at all (see the note in that file).

This probe locates the shadow footprint from the data instead:

  1. shoot the same frozen frame twice, shadow rig ON and OFF (?noshadow), N frames each
  2. the temporal spread of each set is the noise floor of any A/B difference
  3. the footprint is the set of GROUND pixels that the shadow darkens by more than that
     noise floor; ground pixels are the ones neither frame paints as bright fly body
  4. `under`  = mean luma of the footprint with shadows ON
     `beside` = mean luma of the lit ring around the footprint, same frame
     `contrast` = (beside - under) / beside

  void      the band above the slab's far lip, which must stay near black
  profile   vertical luma strip through the fly's own column

    python tools/ground_local.py [base_url] [--shots N]
"""
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

BASE = 'http://localhost:5191/?frozen'
SHOTS = 3
args = [a for a in sys.argv[1:]]
if args and not args[0].startswith('--'):
    BASE = args.pop(0)
if '--shots' in args:
    SHOTS = int(args[args.index('--shots') + 1])

ART = 'D:/Projects/flylingo/artifacts'
BG = (8, 12, 17)          # gl.setClearColor('#080c11')
FLY_LUMA = 70             # above this is fly body / legs / bloom, not ground
SHADOW_DELTA = 6          # luma the shadow must remove before a pixel counts as shadowed
RAMP = ' .:-=+*#%@'

LAUNCH_ARGS = [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--disable-lcd-text',
]

# Hide every DOM overlay in the shot: walk from the canvas up to <body> and hide each
# ancestor's other children. The previous probe hid the canvas's *siblings*, which is the
# same element as the canvas, so the badge and the dev strip stayed in the frame.
HIDE_JS = """(() => {
  const cv = document.querySelector('canvas');
  let el = cv, n = 0;
  while (el && el.parentElement) {
    const p = el.parentElement;
    for (const sib of Array.from(p.children)) {
      if (sib !== el) { sib.style.display = 'none'; n++; }
    }
    el = p;
    if (el === document.body) break;
  }
  return n;
})()"""


def luma(png):
    a = np.asarray(Image.open(BytesIO(png)).convert('RGB'), dtype=np.int32)
    return (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000


def annotate(png, text):
    im = Image.open(BytesIO(png)).convert('RGB')
    from PIL import ImageDraw
    ImageDraw.Draw(im).text((12, 8), text, fill=(255, 80, 80))
    out = BytesIO()
    im.save(out, format='PNG')
    return out.getvalue()


def shoot_all(page, tag):
    """N frames of one state, plus the hero PNG of the first."""
    frames = []
    for i in range(SHOTS):
        if i == 0:
            png = page.locator('canvas').first.screenshot()
            with open('%s/viz-fly-%s.png' % (ART, tag), 'wb') as fh:
                fh.write(png)
        else:
            png = page.locator('canvas').first.screenshot()
        frames.append(luma(png))
        page.wait_for_timeout(700)
    stack = np.stack(frames)
    return stack.mean(axis=0), stack


def dilate(mask, k):
    """Cheap square dilation by k pixels, via shifted ORs (k small)."""
    out = mask.copy()
    for d in range(1, k + 1):
        out |= np.roll(mask, d, 0) | np.roll(mask, -d, 0)
        out |= np.roll(mask, d, 1) | np.roll(mask, -d, 1)
    return out


def show(l, title, step=18, cw=20, mark=None, mk='S'):
    print(title)
    for y in range(0, l.shape[0], step):
        line = ''
        for x in range(0, l.shape[1], cw):
            cell = l[y:y + step, x:x + cw]
            ch = RAMP[min(9, max(0, int(float(cell.mean()) / 12)))]
            if mark is not None and mark[y:y + step, x:x + cw].mean() > 0.30:
                ch = mk
            line += ch
        print('%4d %s' % (y, line))
    print('   cells: %s  [%s = shadow footprint]' % (
        '. <12  : 12-24  - 24-36  = 36-48  + 48-60  * 60-72  # 72-84  %% 84-96  @ >96', mk))
    print()


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=LAUNCH_ARGS)
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        on_url = BASE
        off_url = BASE + ('&' if '?' in BASE else '?') + 'noshadow'

        page.goto(on_url, wait_until='load', timeout=45000)
        page.wait_for_selector('canvas', timeout=45000)
        page.wait_for_timeout(5000)
        hidden = page.evaluate(HIDE_JS)
        page.wait_for_timeout(200)
        on, on_stack = shoot_all(page, 'ground-on')
        cam = page.evaluate('() => { const c = window.__flyCamera; '
                            'return c ? [c.position.toArray().map(n=>+n.toFixed(3)), c.fov] : null; }')
        gl = page.evaluate('() => window.__flyGl || null')
        print('camera %s  overlays hidden %s' % (cam, hidden))
        print('renderer %s' % (gl,))

        page.goto(off_url, wait_until='load', timeout=45000)
        page.wait_for_selector('canvas', timeout=45000)
        page.wait_for_timeout(5000)
        page.evaluate(HIDE_JS)
        page.wait_for_timeout(200)
        off, off_stack = shoot_all(page, 'ground-off')
        browser.close()

    # hero shot of the shipped state, annotated with the camera state
    hero = annotate(open(ART + '/viz-fly-ground-on.png', 'rb').read(),
                    'shadows ON  camera frozen  %s' % (cam,))
    with open(ART + '/viz-fly-hero.png', 'wb') as fh:
        fh.write(hero)

    # noise floor: two frames of the SAME state
    noise = float(np.abs(on_stack[0] - on_stack[-1]).mean())
    still = float(np.abs(on_stack[0] - on_stack[-1])[100:400].mean())
    print('temporal noise (same state, %.1fs apart): whole frame %.2f luma, '
          'upper third %.2f luma' % ((SHOTS - 1) * 0.7, noise, still))

    # ground pixels: neither frame paints them bright, and neither is the background
    upper = np.maximum(on, off)
    ground = (upper < FLY_LUMA) & (upper > 4)
    bright = dilate(on > FLY_LUMA, 10) | dilate(off > FLY_LUMA, 10)

    delta = off - on
    shadow = (delta > SHADOW_DELTA) & ground & ~bright
    # a footprint has to be a blob, not speckle: keep only richly shadowed pixels and
    # require local support
    core = (delta > 10) & ground & ~bright
    shadow = dilate(core, 3) & ground & ~bright

    ring = (dilate(shadow, 26) & ~dilate(shadow, 4) & ground & ~bright)

    def m(mask):
        return float(on[mask].mean()) if mask.any() else float('nan')

    under, beside = m(shadow), m(ring)
    contrast = (beside - under) / beside * 100.0 if beside else 0.0

    print()
    print('== frame stats (mean of %d frames per state) ==' % SHOTS)
    for name, l in (('on', on), ('off', off)):
        print('  %-4s mean %.2f  p50 %d  p99 %d  max %d  frac>%d %.4f'
              % (name, l.mean(), np.percentile(l, 50), np.percentile(l, 99), l.max(),
                 FLY_LUMA, (l > FLY_LUMA).mean()))

    print()
    print('== shadow footprint, located from the A/B difference ==')
    if shadow.sum() < 500:
        print('  no footprint found (area %d px): the shadow rig is not darkening ground pixels'
              % shadow.sum())
    else:
        ys, xs = np.where(shadow)
        print('  area %6d px   x %4d..%4d   y %4d..%4d   centroid (%4d,%4d)'
              % (shadow.sum(), xs.min(), xs.max(), ys.min(), ys.max(), xs.mean(), ys.mean()))
        off_under = float(off[shadow].mean())
        print('  darkening where it lands: %.2f luma (on) vs %.2f luma (off) -> %.1f%% darker'
              % (under, off_under, (off_under - under) / off_under * 100.0))
        print()
        print('  shadow (under)  %6.2f' % under)
        print('  lit ring beside %6.2f' % beside)
        print('  LOCAL CONTRAST  %6.1f%% darker than the surface immediately beside it'
              % contrast)
        vs_noise = (float(off[shadow].mean()) - under) / max(noise, 1e-6)
        print('  signal/noise     %6.1fx the temporal noise floor' % vs_noise)

    print()
    print('== void ==')
    # self-locating: the top of the lit content, then the band well above it
    rowmax = on.max(axis=1)
    top = int(np.argmax(rowmax > 40)) if (rowmax > 40).any() else 0
    vo = on[60:max(120, top - 30)]
    print('  top of lit content y=%d; void band y 60..%d: mean %.2f  p99 %d  max %d'
          % (top, max(120, top - 30), vo.mean() if vo.size else float('nan'),
             np.percentile(vo, 99) if vo.size else 0, vo.max() if vo.size else 0))
    corner = on[40:200, 0:360]
    print('  upper-left corner y40..200 x0..360: mean %.2f  max %d'
          % (corner.mean(), corner.max()))
    print('  background clear colour luma %.2f' % float(
        (np.array(BG) * np.array([299, 587, 114])).sum() // 1000))

    print()
    print('== vertical profile through the fly column (x 560..940), shadows ON ==')
    for y in range(240, 900, 30):
        v = float(on[y:y + 30, 560:940].mean())
        s = float(shadow[y:y + 30, 560:940].mean())
        print('  y=%4d %6.1f %-30s shadow %.0f%%' % (y, v, '#' * int(v / 4), 100 * s))

    print()
    show(on, '== shadow map (S) over the lit frame, shadows ON ==', mark=shadow)
    show(np.clip(off - on, 0, 60), '== |off - on| difference (the shadow signal) ==')
    return 0


if __name__ == '__main__':
    sys.exit(main())
