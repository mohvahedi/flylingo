"""Find the framing that composes the hero best, measured on the real app canvas.

The harness page carries a light debug overlay, which a luminance mask reads as subject, so the
sweep runs against the app's own canvas where the only bright things are the phone and the fly.
For each candidate camera the subject's bounding box is measured and the four margins reported;
the best fills the panel while keeping the margins balanced, because an off-centre subject with
dead space down one side is what read as uncomposed.

The camera is read from the page's own query string (see camVec/camParam in PhoneStage), so no
rebuild is needed between candidates.
"""
import sys

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

APP = "http://127.0.0.1:3300/fly"

CANDIDATES = [
    # distance 14, sweeping the look-at right so the 23-29% dead band on the left closes up.
    ("d14_t0",   0.4, 8.0, 14.0,  0.4, 3.9, 0.6, 38),
    ("d14_t2",   0.4, 8.0, 14.0,  2.0, 3.9, 0.6, 38),
    ("d14_t4",   0.4, 8.0, 14.0,  3.6, 3.9, 0.6, 38),
    ("d13_t4",   0.4, 7.8, 13.0,  3.6, 3.9, 0.6, 38),
    ("d13_t6",   0.4, 7.8, 13.0,  5.0, 3.9, 0.6, 38),
    ("d12_t6",   0.4, 7.6, 12.0,  5.0, 3.8, 0.6, 38),
]


def measure(pg, shot):
    box = pg.evaluate(
        """() => {
          const c = document.querySelector('canvas');
          if (!c) return null;
          const b = c.getBoundingClientRect();
          return { x: Math.round(b.x), y: Math.round(b.y),
                   w: Math.round(b.width), h: Math.round(b.height) };
        }"""
    )
    if not box:
        return None
    pg.screenshot(path=shot)
    im = Image.open(shot).convert("RGB")
    crop = im.crop((box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))
    a = np.asarray(crop).astype(np.int16)
    h, w = a.shape[:2]
    lum = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)

    phone = lum > 150
    fly = (sat > 45) & (lum > 35) & ~phone
    both = fly | phone
    ys, xs = np.where(both)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return {
        "box": f"{(x1-x0)/w*100:.0f}%x{(y1-y0)/h*100:.0f}%",
        "left": x0 / w, "right": (w - x1) / w,
        "top": y0 / h, "bot": (h - y1) / h,
        "cover": both.mean(),
        "canvas": f"{w}x{h}",
    }


with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    out = []
    for name, cx, cy, cz, tx, ty, tz, fov in CANDIDATES:
        pg = br.new_page(viewport={"width": 1920, "height": 1080})
        url = f"{APP}?cam={cx},{cy},{cz}&tgt={tx},{ty},{tz}&fov={fov}"
        try:
            pg.goto(url, wait_until="load", timeout=60000)
            pg.wait_for_timeout(23000)
            m = measure(pg, rf"D:\Projects\flylingo\artifacts\f-{name}.png")
        except Exception as e:
            print(f"{name:15s} FAILED {type(e).__name__}: {str(e)[:60]}")
            pg.close()
            continue
        pg.close()
        if not m:
            print(f"{name:15s} no subject detected")
            continue
        imb = abs(m["left"] - m["right"]) + abs(m["top"] - m["bot"])
        out.append((imb, name, (cx, cy, cz, tx, ty, tz, fov), m))
        print(
            f"{name:15s} canvas {m['canvas']:>8s}  box {m['box']:>7s}  "
            f"L{m['left']*100:3.0f} R{m['right']*100:3.0f} T{m['top']*100:3.0f} B{m['bot']*100:3.0f}  "
            f"imbalance {imb*100:3.0f}  fill {m['cover']*100:4.1f}%"
        )

    print()
    if out:
        out.sort(key=lambda r: r[0])
        for imb, name, params, m in out[:3]:
            print(f"  best: {name:15s} imbalance {imb*100:.0f}  fill {m['cover']*100:.1f}%  params {params}")
    br.close()
    sys.exit(0)
