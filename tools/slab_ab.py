"""Verify the slab grounding myself: does the shadow now darken a surface the eye can see?

Two questions, both answered from pixels with shadows ON vs OFF (?noshadow), sampling ONLY
floor regions so the fly's own bright pixels cannot contaminate the comparison:
  1. is the shadowed floor darker than the same floor with the rig off?
  2. does the shadowed floor read darker than the lit floor BESIDE it in the same frame?
     That local contrast is what makes a shadow visible at all.
"""
import io
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

BASE = "http://localhost:5191/?frozen"


def shoot(pg, url, path):
    pg.goto(url, wait_until="load", timeout=60000)
    pg.wait_for_timeout(9000)
    b = pg.screenshot()
    open(path, "wb").write(b)
    return Image.open(io.BytesIO(b)).convert("RGB")


def luma(im):
    d = list(im.getdata())
    return sum(sum(q) for q in d) / (3 * len(d))


with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1440, "height": 900})
    on = shoot(pg, BASE, r"D:\Projects\flylingo\artifacts\slab-on.png")
    off = shoot(pg, BASE + "&noshadow", r"D:\Projects\flylingo\artifacts\slab-off.png")
    br.close()

W, H = on.size
print(f"frame {W}x{H}")

# Floor strips only: below the fly, and to either side, all well under the body line.
bands = {
    "floor left  ": (0.06, 0.80, 0.26, 0.97),
    "floor mid   ": (0.38, 0.80, 0.62, 0.97),
    "floor right ": (0.74, 0.80, 0.94, 0.97),
    "shadow zone ": (0.40, 0.62, 0.62, 0.78),
}
res = {}
for name, (x0, y0, x1, y1) in bands.items():
    box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
    a, b = luma(on.crop(box)), luma(off.crop(box))
    res[name] = (a, b)
    delta = b - a
    pct = 100 * delta / max(b, 1e-6)
    verdict = "shadowed" if delta > 1.5 else ("flat" if abs(delta) <= 1.5 else "brighter?!")
    print(f"  {name} on {a:6.2f}  off {b:6.2f}  delta {delta:+6.2f} ({pct:+5.1f}%)  {verdict}")

print()
sh = res["shadow zone "][0]
bes = res["floor mid   "][0]
print(f"LOCAL CONTRAST: shadow zone {sh:.2f} vs floor beside it {bes:.2f} -> "
      f"{100 * (bes - sh) / max(bes, 1e-6):.1f}% darker")
print("(a shadow needs to be clearly darker than the surface NEXT to it, not just darker")
print(" than the same pixels with the rig switched off)")

diff = ImageChops.difference(on, off).getbbox()
print("A/B difference bounding box:", diff)
