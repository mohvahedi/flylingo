"""Print a row-by-row ink profile of the rendered Duolingo screen.

Guessing crop coordinates wasted two attempts, so measure where the content actually is:
for every scanline, count pixels that are not near-white, and print the bands. That gives the
true y position of the prompt, each card row and the button.
"""
import sys

from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else r"D:\Projects\flylingo\artifacts\duolingo-screen-open.png"
im = Image.open(path).convert("RGB")
W, H = im.size
print(f"image {W}x{H}")

px = im.load()
# Scale factor between the screenshot and the 390x865 CSS layout.
scale = W / 390.0
print(f"scale vs 390 CSS px: {scale}")

bands = []
run = None
for y in range(H):
    ink = 0
    for x in range(0, W, 2):  # every other column is plenty for a profile
        r, g, b = px[x, y]
        if r < 235 or g < 235 or b < 235:
            ink += 1
    if ink > 1:
        if run is None:
            run = [y, y, 0]
        run[1] = y
        run[2] = max(run[2], ink)
    else:
        if run is not None and run[1] - run[0] >= 2:
            bands.append(tuple(run))
        run = None
if run is not None:
    bands.append(tuple(run))

print(f"\n{len(bands)} content bands (screenshot px / CSS px):")
for y0, y1, mx in bands:
    print(f"  y {y0:5d}-{y1:5d}   CSS {y0/scale:6.1f}-{y1/scale:6.1f}   peak ink {mx}")
