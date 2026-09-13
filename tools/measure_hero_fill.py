"""How much of the hero panel do the fly and the phone actually occupy?

An "it looks empty" judgement is not actionable, so this measures the subject's bounding box as a
fraction of the panel. The scene is a dark blue gradient, the phone is near-white and the fly is
strongly coloured, so the two are separable by luminance and saturation respectively.
"""
import sys

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:3300/fly"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"D:\Projects\flylingo\artifacts\hero-fill.png"

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(URL, wait_until="load", timeout=60000)
    pg.wait_for_timeout(26000)
    box = pg.evaluate(
        """() => {
          const c = document.querySelector('canvas');
          if (!c) return null;
          const b = c.getBoundingClientRect();
          return { x: Math.round(b.x), y: Math.round(b.y),
                   w: Math.round(b.width), h: Math.round(b.height) };
        }"""
    )
    pg.screenshot(path=OUT)
    br.close()

if not box:
    print("no canvas")
    sys.exit(1)

im = Image.open(OUT).convert("RGB")
crop = im.crop((box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))
crop.save(OUT.replace(".png", "-crop.png"))
a = np.asarray(crop).astype(np.int16)
H, W = a.shape[:2]

lum = a.mean(axis=2)
sat = a.max(axis=2) - a.min(axis=2)

phone = lum > 150                      # the near-white handset and its screen
fly = (sat > 45) & (lum > 35)          # the fly's coloured body, eyes and wings
# exclude the phone's own pixels from the fly mask, so the two boxes stay distinct
fly = fly & ~phone

def bbox(mask, label):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        print(f"{label:8s}: not found")
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    print(
        f"{label:8s}: {mask.mean()*100:5.1f}% of pixels   "
        f"box {x1-x0:>4}x{y1-y0:<4} = {(x1-x0)/W*100:4.0f}% x {(y1-y0)/H*100:4.0f}% of panel   "
        f"x {x0/W*100:3.0f}-{x1/W*100:3.0f}%  y {y0/H*100:3.0f}-{y1/H*100:3.0f}%"
    )
    return x0, x1, y0, y1

print(f"panel: {W}x{H}\n")
bp = bbox(phone, "phone")
bf = bbox(fly, "fly")
if bp and bf:
    x0 = min(bp[0], bf[0]); x1 = max(bp[1], bf[1])
    y0 = min(bp[2], bf[2]); y1 = max(bp[3], bf[3])
    print(
        f"{'BOTH':8s}: {'':18s}"
        f"box {x1-x0:>4}x{y1-y0:<4} = {(x1-x0)/W*100:4.0f}% x {(y1-y0)/H*100:4.0f}% of panel   "
        f"x {x0/W*100:3.0f}-{x1/W*100:3.0f}%  y {y0/H*100:3.0f}-{y1/H*100:3.0f}%"
    )
    print()
    print(f"margin left/right: {x0/W*100:.0f}% / {(W-x1)/W*100:.0f}%")
    print(f"margin top/bottom: {y0/H*100:.0f}% / {(H-y1)/H*100:.0f}%")
