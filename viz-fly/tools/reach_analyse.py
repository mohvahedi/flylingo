"""Analyse a reach_measure.py sample file: fly size, foreleg selectivity, and contact.

Everything is computed from published world-space numbers: the screen mesh's world box and
normal (the page's own reading of the loaded asset), and the world position of the deepest
joint of each of the six leg chains. The glass plane is rebuilt from those; no pixel is read.

The pixel->world mapping is validated against the page's own cardWorld readout before any
conclusion is drawn from it, so the "the tarsus landed inside the chosen card" claim cannot
be an artefact of a wrong frame.
"""
import json
import sys

import numpy as np

FORELegS = (0, 3)  # pose indices of the two forelegs: the chains the groom behavior addresses
SCREEN_W, SCREEN_H = 780, 1730  # duolingoScreen canvas size, the pixel space of the rects


def basis(stage):
    n = np.array(stage["screenNormal"], dtype=float)
    n /= np.linalg.norm(n)
    u = np.array(stage["screenUp"], dtype=float)
    u = u - n * float(u @ n)
    u /= np.linalg.norm(u)
    r = np.cross(u, n)  # the plane's local +X: right, for a right-handed frame
    return n, u, r


def main() -> int:
    path = sys.argv[1]
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    samples = blob["samples"]
    st = next((s["stage"] for s in samples if s.get("stage")), None)
    if not st or "screenNormal" not in st:
        print("no stage probe:", st)
        return 1
    node = [s["node"] for s in samples if s.get("node") and "foot" in s["node"]]
    fly = [s["fly"] for s in samples if s.get("fly") and s["fly"].get("target")]
    if not node or not fly:
        print("probe missing from samples")
        return 1

    n, u, r = basis(st)
    H = float(st["screenHeight"])
    W = H * SCREEN_W / SCREEN_H
    c = (np.array(fly[0]["screenBox"]["min"]) + np.array(fly[0]["screenBox"]["max"])) / 2.0

    def plane_to_world(px, py):
        return (
            c
            + r * ((px / SCREEN_W - 0.5) * W)
            + u * ((0.5 - py / SCREEN_H) * H)
        )

    def world_to_plane(p):
        d = np.asarray(p, dtype=float) - c
        return (0.5 + float(d @ r) / W) * SCREEN_W, (0.5 - float(d @ u) / H) * SCREEN_H

    # ---- validate the frame: the page's own perch points must come out of this mapping
    cards = fly[-1]["cardWorld"] or []
    all_rects = st["rects"]
    print("rects published by the page:")
    for i, rct in enumerate(all_rects):
        print(f"   card {i}: x {rct['x']}..{rct['x'] + rct['w']}  y {rct['y']}..{rct['y'] + rct['h']}"
              f"  centre ({rct['cx']},{rct['cy']})")
    print("perch points published by the page:", [list(np.round(np.array(cc), 3)) for cc in cards])
    errs = []
    for i, card in enumerate(cards):
        if i >= len(all_rects):
            break
        rct = all_rects[i]
        pred = plane_to_world(rct["x"] + rct["w"] * 0.84, rct["cy"])
        errs.append(float(np.linalg.norm(pred - np.array(card))))
        print(f"   card {i}: predicted {np.round(pred, 3)} published {np.round(np.array(card), 3)}"
              f"  err {errs[-1]:.3f}")
    print("frame validation: max |predicted perch - published cardWorld| =", round(max(errs), 5))
    print("  (nonzero would mean the plane frame below is wrong; the page is the reference)")
    print()
    print("phone scale          :", st.get("phoneScale"), " screenHeight", H, " screen W", round(W, 3))
    print("screen normal        :", np.round(n, 4), " up", np.round(u, 4), " right", np.round(r, 4))
    print("glass centre (world) :", np.round(c, 4))
    print("fly target (last)    :", fly[-1]["target"], " pos", fly[-1]["pos"], " dist", fly[-1]["dist"])

    # ---- size
    span = np.array(node[-1]["span"], dtype=float)
    print()
    print("FLY SIZE (world units)")
    print("  bbox span            :", np.round(span, 4))
    print("  longest axis         :", round(float(span.max()), 4))
    print("  as fraction of the screen height:", round(float(span.max()) / H, 4),
          f"= {100 * float(span.max()) / H:.1f}%")
    print("  nose-to-tail (Z) fraction      :", round(float(span[2]) / H, 4))

    feet = np.array([s["foot"] for s in node], dtype=float)
    wings = np.array([s["wing"] for s in node], dtype=float)
    print("  wingtip-to-wingtip span        :", round(float(np.linalg.norm(wings[:, 0] - wings[:, 1], axis=1).max()), 4))
    print("  distance of the lowest tarsus to the glass plane:",
          round(float(np.einsum("nij,j->ni", feet - c, n).min()), 4))

    # ---- selectivity of the leg motion
    exc = [float(np.linalg.norm(pts.max(0) - pts.min(0))) for pts in feet.transpose(1, 0, 2)]
    print()
    print("LEG EXCURSION over the window (world units)")
    for i, e in enumerate(exc):
        print(f"  leg {i}: {e:.4f}{'   <- foreleg' if i in FORELegS else ''}")
    others = [e for i, e in enumerate(exc) if i not in FORELegS]
    print(f"  mean of the other four: {np.mean(others):.4f}   ratio: {np.mean([exc[i] for i in FORELegS]) / max(1e-9, np.mean(others)):.2f}x")

    # ---- contact with the glass, per leg
    d = np.einsum("nij,j->ni", feet - c, n)
    print()
    print("TARSUS DISTANCE TO THE GLASS, perpendicular (world units, 0 = touching)")
    idle = float(np.median(d[:, [1, 2, 4, 5]]))
    for i in range(6):
        print(f"  leg {i}: min {d[:, i].min():+.4f}  median {np.median(d[:, i]):+.4f}"
              f"{'   <- foreleg' if i in FORELegS else ''}")
    print(f"  the four non-forelegs sit at a median of {idle:+.4f} off the glass")

    # ---- where the closest foreleg contact lands, in the chosen card's rect
    ch = st.get("flyChoice")
    rct = all_rects[ch] if ch is not None and ch < len(all_rects) else None
    print()
    print("CONTACT POSITION of each foreleg at its closest approach to the glass")
    for i in FORELegS:
        j = int(np.argmin(d[:, i]))
        px, py = world_to_plane(feet[j, i])
        inside = None
        if rct:
            inside = (rct["x"] <= px <= rct["x"] + rct["w"]) and (rct["y"] <= py <= rct["y"] + rct["h"])
        print(f"  leg {i}: sample {j} t={fly[min(j, len(fly) - 1)]['t']}s behavior={node[j]['behavior']}"
              f"  dist_to_glass {d[j, i]:+.4f}  card px ({px:.0f},{py:.0f})")
        if rct:
            print(f"           chosen card rect x {rct['x']}..{rct['x'] + rct['w']}  y {rct['y']}..{rct['y'] + rct['h']}"
                  f"  -> inside: {inside}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
