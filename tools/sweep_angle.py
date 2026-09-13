"""Find a camera angle where the fly does NOT sit on top of the phone.

The problem this solves, measured: the fly stands in front of the handset and a level camera
therefore projects the fly straight over the phone's screen, hiding the question, and the
handset's lower edge falls outside the frame. Raising the camera makes the two separate in screen
space, because the fly is several units nearer the lens, so a downward view drops the fly in the
frame while the phone stays higher up.

The objective is the phone's VISIBLE area (near-white screen pixels, bright enough not to be the
fly's own body), with the fly still large. So the sweep reports both and the winner is the one
with the most visible phone.
"""
import sys

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

APP = "http://127.0.0.1:3300/fly"

CANDIDATES = [
    # The fly's WINGSPAN is what covers the phone: it spans the whole frame because the fly is
    # close to the lens and seen head-on. Viewing from the side foreshortens the wings, so the
    # sweep moves the camera round rather than up.
    ("front",    0.4,  8.0, 14.0, 0.4, 3.9, 0.6, 38),
    ("side_x5",  5.0,  8.0, 13.0, 1.0, 3.9, 0.6, 38),
    ("side_x9",  9.0,  8.5, 11.0, 1.0, 3.9, 0.6, 38),
    ("side_x12", 12.0, 9.0,  9.0, 1.0, 3.9, 0.6, 38),
    ("side_x9_y11", 9.0, 11.0, 11.0, 1.0, 3.6, 0.6, 38),
    ("side_x9_t2",  9.0,  8.5, 11.0, 2.2, 3.9, 0.6, 38),
]


def measure(pg, shot):
    box = pg.evaluate(
        """() => { const c=document.querySelector('canvas'); if(!c) return null;
          const b=c.getBoundingClientRect();
          return {x:Math.round(b.x),y:Math.round(b.y),w:Math.round(b.width),h:Math.round(b.height)}; }"""
    )
    if not box:
        return None
    pg.screenshot(path=shot)
    im = Image.open(shot)
    crop = im.crop((box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))
    crop.save(shot.replace(".png", "-crop.png"))
    a = np.asarray(crop.convert("RGB")).astype(np.int16)
    lum = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)

    phone = lum > 165
    fly = (sat > 45) & (lum > 35) & ~phone
    h, w = a.shape[:2]

    def boxf(m):
        ys, xs = np.where(m)
        if len(xs) == 0:
            return None
        return (xs.min() / w, xs.max() / w, ys.min() / h, ys.max() / h)

    pf, ff = boxf(phone), boxf(fly)
    return {
        "phone_px": phone.mean() * 100,
        "fly_px": fly.mean() * 100,
        "phone_box": None if not pf else f"x{pf[0]*100:.0f}-{pf[1]*100:.0f} y{pf[2]*100:.0f}-{pf[3]*100:.0f}",
        "fly_box": None if not ff else f"x{ff[0]*100:.0f}-{ff[1]*100:.0f} y{ff[2]*100:.0f}-{ff[3]*100:.0f}",
    }


with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    rows = []
    for name, cx, cy, cz, tx, ty, tz, fov in CANDIDATES:
        pg = br.new_page(viewport={"width": 1920, "height": 1080})
        try:
            pg.goto(f"{APP}?cam={cx},{cy},{cz}&tgt={tx},{ty},{tz}&fov={fov}",
                    wait_until="load", timeout=60000)
            pg.wait_for_timeout(22000)
            m = measure(pg, rf"D:\Projects\flylingo\artifacts\a-{name}.png")
        except Exception as e:
            print(f"{name:12s} FAILED {type(e).__name__}: {str(e)[:50]}")
            pg.close()
            continue
        pg.close()
        if not m:
            print(f"{name:12s} no canvas")
            continue
        rows.append((m["phone_px"], name, (cx, cy, cz, tx, ty, tz, fov), m))
        print(f"{name:12s} phone {m['phone_px']:5.1f}% px  fly {m['fly_px']:5.1f}% px")
        print(f"{'':12s}   phone {m['phone_box']}")
        print(f"{'':12s}   fly   {m['fly_box']}")

    print()
    if rows:
        rows.sort(key=lambda r: -r[0])
        print("ranked by VISIBLE PHONE (the fly must not cover it):")
        for px, name, params, m in rows:
            print(f"  {name:12s} {px:5.1f}% phone visible   fly {m['fly_px']:4.1f}%   {params}")
    br.close()
    sys.exit(0)
