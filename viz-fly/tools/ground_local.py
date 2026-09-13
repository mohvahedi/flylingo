"""Local grounding measurement for the viz-fly hero stage.

The question this answers: does the fly read as standing ON a lit surface, or floating in a
void? The eye judges that by LOCAL contrast - how much darker the shadow is than the lit
ground immediately beside it - so hand-placed boxes are the wrong instrument. The 17% figure
in HANDOFF_grounding.md came from a box pair that had been placed in the near floor beside the
fly's legs rather than in the shadow at all (the BOX table was documented (x0,x1,y0,y1) and
unpacked the other way round, and the void boxes fell outside the frame, which is why that row
read nan).

This probe locates the darkening from the data instead. It shoots the same frozen frame in
three states and differences them:

  shipped          the frame as it ships
  pool off         uPoolStrength = 0 on the live ground material: the contact pool alone
  ?noshadow        the whole shadow rig off: the key light does not cast, no contact patches

Each difference is the set of pixels that state removes, restricted to ground pixels (neither
frame paints them as bright fly body). For each one it reports the mean luma inside the
footprint and on the lit ring just outside it:

  contrast = (beside - under) / beside

  void      the band above the slab's far lip, which must stay near black
  profile   vertical luma strip through the fly's own column

    python tools/ground_local.py [base_url] [--shots N]
"""
import sys
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

BASE = 'http://localhost:5191/?frozen'
SHOTS = 3
args = list(sys.argv[1:])
if args and not args[0].startswith('--'):
    BASE = args.pop(0)
if '--shots' in args:
    SHOTS = int(args[args.index('--shots') + 1])

ART = 'D:/Projects/flylingo/artifacts'
FLY_LUMA = 70             # above this is fly body / legs / bloom, not slab
DELTA = 6                 # luma a state must remove before a pixel counts as darkened
RAMP = ' .:-=+*#%@'

LAUNCH_ARGS = [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--disable-lcd-text',
]

# Hide every DOM overlay in the shot: walk from the canvas up to <body> and hide each
# ancestor's other children. Hiding the canvas's *siblings* hides nothing, because the canvas
# is the only child of its own R3F container; the badge and the dev strip are siblings of that.
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

POOL_OFF_JS = """(v) => {
  const g = window.__flyScene.getObjectByName('ground');
  const u = g && g.material && g.material.userData && g.material.userData.uniforms;
  if (!u || !u.uPoolStrength) return 'no uPoolStrength';
  u.uPoolStrength.value = v;
  return 'uPoolStrength=' + v;
}"""


def luma(png):
    a = np.asarray(Image.open(BytesIO(png)).convert('RGB'), dtype=np.int32)
    return (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000


def annotate(png, text):
    im = Image.open(BytesIO(png)).convert('RGB')
    ImageDraw.Draw(im).text((12, 8), text, fill=(255, 80, 80))
    out = BytesIO()
    im.save(out, format='PNG')
    return out.getvalue()


def dilate(mask, k):
    out = mask.copy()
    for d in range(1, k + 1):
        out |= np.roll(mask, d, 0) | np.roll(mask, -d, 0)
        out |= np.roll(mask, d, 1) | np.roll(mask, -d, 1)
    return out


def show(l, title, step=20, cw=20, mark=None, mk='X'):
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
    print('   . <12  : 12-24  - 24-36  = 36-48  + 48-60  * 60-72  # 72-84  %% 84-96  @ >96'
          '   [%s = removed by the named state]' % mk)
    print()


class Shot:
    def __init__(self, page, tag):
        self.frames = []
        for i in range(SHOTS):
            png = page.locator('canvas').first.screenshot()
            if i == 0:
                with open('%s/viz-fly-%s.png' % (ART, tag), 'wb') as fh:
                    fh.write(png)
            self.frames.append(luma(png))
            page.wait_for_timeout(700)
        st = np.stack(self.frames)
        self.mean = st.mean(axis=0)
        self.noise = float(np.abs(st[0] - st[-1]).mean())


def contrast(name, shipped, other, ground, bright, l):
    """Report the footprint `other` removes relative to the shipped frame.

    `under` is the DEEP part of the footprint (pixels this state darkens by more than 10
    luma), not the dilated outline: averaging over a dilated mask mixes in pixels the state
    does not touch and understates the darkening by an order of magnitude. `beside` is the
    2..8 px band around that footprint, which is what "immediately beside it" means to the
    eye. Both are read off the shipped frame, so this is a spatial contrast, not an A/B.
    """
    s, o = shipped.mean, other.mean
    # the footprint is where the ABLATED state is brighter, i.e. the pixels the state under test
    # is darkening. The opposite sign selects pixels the shipped frame happens to be brighter on
    # for unrelated reasons, which is how a box pair can report a confident wrong number.
    d = o - s
    core = (d > 10) & ground & ~bright
    foot = dilate(core, 2) & ground & ~bright
    ring = dilate(foot, 8) & ~dilate(foot, 2) & ground & ~bright
    print('== %s ==' % name)
    if core.sum() < 200:
        print('   no footprint found (%d px): nothing measurable is being removed' % core.sum())
        print()
        return
    ys, xs = np.where(core)
    under = float(s[core].mean())
    beside = float(s[ring].mean())
    print('   footprint %6d px   x %4d..%4d  y %4d..%4d  centroid (%4d,%4d)'
          % (core.sum(), xs.min(), xs.max(), ys.min(), ys.max(), xs.mean(), ys.mean()))
    print('   removed where it lands: %.2f luma -> %.2f luma (%.1f%% darker)'
          % (float(o[core].mean()), under,
             (float(o[core].mean()) - under) / max(float(o[core].mean()), 1e-6) * 100.0))
    print('   under the darkening   %6.2f  (deep pixels only)' % under)
    print('   surface beside it     %6.2f  (2..8 px band around it)' % beside)
    print('   LOCAL CONTRAST        %6.1f%% darker than the surface immediately beside it'
          % ((beside - under) / beside * 100.0))
    print('   signal / noise        %6.1fx the temporal noise floor (%.2f luma)'
          % ((beside - under) / max(shipped.noise, 1e-6), shipped.noise))
    print('   (only the darkening matters here; the fly walks between frames, so the')
    print('    noise floor is measured on slab pixels that exclude the fly)')
    print()
    return foot


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=LAUNCH_ARGS)
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        # a shader that fails to compile renders as an empty stage and only says so on the
        # console, so never measure without watching it
        errors = []
        page.on('console', lambda m: errors.append('console.%s: %s' % (m.type, m.text))
                if m.type == 'error' else None)
        page.on('pageerror', lambda e: errors.append('pageerror: %s' % e))

        page.goto(BASE, wait_until='domcontentloaded', timeout=90000)
        page.wait_for_selector('canvas', timeout=90000)
        page.wait_for_timeout(6000)
        hidden = page.evaluate(HIDE_JS)
        cam = page.evaluate('() => { const c = window.__flyCamera; '
                            'return c ? [c.position.toArray().map(n => +n.toFixed(3)), c.fov] : null; }')
        gl = page.evaluate('() => window.__flyGl || null')
        print('camera %s   renderer %s   overlays hidden %s' % (cam, gl, hidden))
        shipped = Shot(page, 'ground-on')
        print('pool ablation: %s' % page.evaluate(POOL_OFF_JS, 0))
        pool_off = Shot(page, 'ground-pool-off')
        page.evaluate(POOL_OFF_JS, 1)
        # how much of the slab does the key light actually supply? A shadow can only ever
        # remove the key's share, so this is the ceiling on the cast shadow's contrast.
        print('key ablation: %s' % page.evaluate(
            "() => { let n = 0; window.__flyScene.traverse(o => {"
            " if (o.isDirectionalLight && o.castShadow) { o.userData._i = o.intensity;"
            " o.intensity = 0; n++; } }); return 'key intensity -> 0 (' + n + ' light)'; }"))
        key_off = Shot(page, 'ground-key-off')

        off_url = BASE + ('&' if '?' in BASE else '?') + 'noshadow'
        page.goto(off_url, wait_until='domcontentloaded', timeout=90000)
        page.wait_for_selector('canvas', timeout=90000)
        page.wait_for_timeout(6000)
        page.evaluate(HIDE_JS)
        rig_off = Shot(page, 'ground-off')
        # The fly's own pixels are not floor: ?noshadow changes the fly's self shadowing and its
        # contact patches too, so any A/B footprint that is allowed to include the body will
        # report a confident contrast between the fly and the slab. Derive the fly's real
        # silhouette here, with the shadow rig already off, so the only variable is the fly.
        page.evaluate("() => { window.__flyScene.getObjectByName('fly-fit').visible = false; }")
        page.wait_for_timeout(700)
        nofly = Shot(page, 'ground-nofly')
        browser.close()

    print()
    print('== console errors during the run: %d ==' % len(errors))
    for e in errors[:12]:
        print('   ' + e[:300])
    print()

    # the hero is the shipped frame, untouched: no annotation on the deliverable. The labelled
    # copy is a diagnostic, for anyone cross-checking what the numbers were read from.
    with open(ART + '/viz-fly-hero.png', 'wb') as fh:
        fh.write(open(ART + '/viz-fly-ground-on.png', 'rb').read())
    with open(ART + '/viz-fly-ground-on-labelled.png', 'wb') as fh:
        fh.write(annotate(open(ART + '/viz-fly-ground-on.png', 'rb').read(),
                          'shadows ON  camera frozen  %s' % (cam,)))

    on = shipped.mean
    upper = np.maximum(on, rig_off.mean)
    # the fly's silhouette, from the shadow-rig-off pair where the only difference is the fly
    fly = dilate(np.abs(rig_off.mean - nofly.mean) > 6, 8)
    # slab pixels: not the fly, not the void, and not painted bright by either frame
    ground = ~fly & (upper < FLY_LUMA) & (upper > 3)
    bright = dilate(on > FLY_LUMA, 10) | dilate(rig_off.mean > FLY_LUMA, 10)
    print('   fly silhouette %d px (%.1f%% of the frame) excluded from every footprint'
          % (fly.sum(), 100.0 * fly.mean()))

    print('== frame stats (mean of %d frames per state) ==' % SHOTS)
    for name, s in (('shipped ', shipped), ('pool off', pool_off), ('rig off ', rig_off),
                    ('key off ', key_off)):
        l = s.mean
        print('   %s mean %6.2f  p50 %3d  p99 %3d  max %3d  frac>%d %.4f  noise %.2f'
              % (name, l.mean(), np.percentile(l, 50), np.percentile(l, 99), l.max(),
                 FLY_LUMA, (l > FLY_LUMA).mean(), s.noise))
    print()

    pool_foot = contrast('contact pool (material term): shipped vs uPoolStrength=0',
                         shipped, pool_off, ground, bright, on)
    rig_foot = contrast('shadow rig: shipped vs ?noshadow', shipped, rig_off, ground, bright, on)

    print('== what the key light actually supplies the slab ==')
    for (y0, y1, x0, x1, nm) in [(560, 620, 200, 520, 'slab beside fly '),
                                 (640, 760, 60, 300, 'slab left of it '),
                                 (640, 760, 620, 1000, 'slab under fly  '),
                                 (500, 560, 60, 300, 'slab mid left   ')]:
        a = on[y0:y1, x0:x1]
        b = key_off.mean[y0:y1, x0:x1]
        m = a < FLY_LUMA
        if m.sum() > 20:
            v, w = float(a[m].mean()), float(b[m].mean())
            print('   %s y%4d..%4d x%4d..%4d: shipped %6.2f  key at 0 %6.2f  -> key supplies %.1f%%'
                  % (nm, y0, y1, x0, x1, v, w, (v - w) / max(v, 1e-6) * 100.0))
    print()

    print('== void ==')
    rowmax = on.max(axis=1)
    top = int(np.argmax(rowmax > 40)) if (rowmax > 40).any() else 0
    band = on[60:max(140, top - 30)]
    print('   top of lit content y=%d; void band y 60..%d: mean %.2f  p99 %d  max %d'
          % (top, max(140, top - 30), band.mean() if band.size else float('nan'),
             np.percentile(band, 99) if band.size else 0, band.max() if band.size else 0))
    for (y0, y1, x0, x1, nm) in [(40, 200, 0, 360, 'upper left '), (40, 200, 1080, 1440, 'upper right'),
                                 (200, 300, 400, 1040, 'behind fly ')]:
        b = on[y0:y1, x0:x1]
        print('   %s y%4d..%4d x%4d..%4d: mean %.2f  max %d' % (nm, y0, y1, x0, x1, b.mean(), b.max()))
    print()

    print('== vertical profile through the fly column (x 560..940) ==')
    for y in range(240, 900, 30):
        v = float(on[y:y + 30, 560:940].mean())
        pf = float(pool_foot[y:y + 30, 560:940].mean()) if pool_foot is not None else 0
        rf = float(rig_foot[y:y + 30, 560:940].mean()) if rig_foot is not None else 0
        print('   y=%4d %6.1f %-28s pool %3.0f%%  rig %3.0f%%'
              % (y, v, '#' * int(v / 4), 100 * pf, 100 * rf))
    print()

    show(on, '== shipped frame ==', mark=pool_foot, mk='X')
    show(np.clip(shipped.mean - rig_off.mean, 0, 60), '== removed by ?noshadow (the rig) ==')
    return 0


if __name__ == '__main__':
    sys.exit(main())
