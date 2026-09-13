"""Work out how the phone's screen mesh maps UVs to positions.

The texture orientation has to be known rather than guessed: the screen plane spans two axes
and the UVs could be rotated or mirrored, which would put Duolingo on the screen sideways or
upside down. This prints the UV->position correspondence for the four corners so the correct
texture rotation and flip can be derived exactly.
"""
import json
import struct

import numpy as np

PATH = r"D:\Downloads\smartphone_with_green_screen.glb"

with open(PATH, "rb") as f:
    struct.unpack("<4sII", f.read(12))
    clen, _ = struct.unpack("<I4s", f.read(8))
    gltf = json.loads(f.read(clen).decode("utf-8"))
    blen, _ = struct.unpack("<I4s", f.read(8))
    bin_ = f.read(blen)

CT = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def acc(i):
    a = gltf["accessors"][i]
    bv = gltf["bufferViews"][a["bufferView"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    n = a["count"] * NC[a["type"]]
    arr = np.frombuffer(
        bin_, dtype=np.dtype(CT[a["componentType"]]).newbyteorder("<"), count=n, offset=off
    )
    return arr.reshape(a["count"], NC[a["type"]])


# mesh 2 is the chroma / screen
prim = gltf["meshes"][2]["primitives"][0]
print("attributes:", list(prim["attributes"].keys()))
pos = acc(prim["attributes"]["POSITION"])
uv = acc(prim["attributes"]["TEXCOORD_0"])
idx = acc(prim["indices"]).reshape(-1) if "indices" in prim else None

print(f"verts {len(pos)}  uvs {len(uv)}  tris {0 if idx is None else len(idx) // 3}")
print()
print("position bounds: min", np.round(pos.min(axis=0), 5), " max", np.round(pos.max(axis=0), 5))
print("uv bounds:        min", np.round(uv.min(axis=0), 5), " max", np.round(uv.max(axis=0), 5))
print()

# For each corner of the UV square, print the 3D position it corresponds to.
print("UV corner -> 3D position  (this is the mapping the texture must respect)")
for name, target in [
    ("u=0,v=0", (0.0, 0.0)),
    ("u=1,v=0", (1.0, 0.0)),
    ("u=0,v=1", (0.0, 1.0)),
    ("u=1,v=1", (1.0, 1.0)),
]:
    d = np.hypot(uv[:, 0] - target[0], uv[:, 1] - target[1])
    j = int(np.argmin(d))
    print(f"  {name}: vert {j:3d}  pos {np.round(pos[j], 5)}   uv {np.round(uv[j], 4)}")

print()
# Which axes do the UV directions run along? Take two orthogonal UV steps.
u_lo = uv[:, 0].min()
u_hi = uv[:, 0].max()
du = (uv[:, 0] - u_lo) / max(1e-9, (u_hi - u_lo))
v_lo = uv[:, 1].min()
v_hi = uv[:, 1].max()
dv = (uv[:, 1] - v_lo) / max(1e-9, (v_hi - v_lo))

for label, w in (("u (horizontal)", du), ("v (vertical)", dv)):
    # linear fit of position against the uv coordinate
    A = np.vstack([w, np.ones_like(w)]).T
    slopes = []
    for ax in range(3):
        m, _ = np.linalg.lstsq(A, pos[:, ax], rcond=None)[0]
        slopes.append(m)
    s = np.array(slopes)
    dom = int(np.argmax(np.abs(s)))
    print(f"  {label} runs along axis {'XYZ'[dom]}  slope {s[dom]:+.5f}  (full vector {np.round(s, 5)})")

print()
print("So: 'XYZ'[argmax] above tells you which model axis the texture's u and v follow.")
print("A positive slope means the texture direction agrees with that axis' + direction.")
