"""Describe a .glb without loading it: read the JSON chunk straight out of the binary.

Why not just trust the asset brief: node names, mesh counts and material assignments are
the contract the rig drives, and a typo in a node name is a silent no-op at runtime (the
part simply never moves). This prints the ground truth so the animator can be checked
against it, and it fails loudly if the group names the rig expects are missing.

    python tools/inspect_glb.py [path] [--group PREFIX ...]
"""
import json
import numpy as np
import struct
import sys
from collections import Counter, defaultdict

DEFAULT = 'D:/Projects/flylingo/viz-fly/public/models/fly.glb'


def chunks(data: bytes):
    magic, version, length = struct.unpack_from('<III', data, 0)
    if magic != 0x46546C67:
        raise SystemExit('not a glb (bad magic)')
    off = 12
    while off < length:
        clen, ctype = struct.unpack_from('<II', data, off)
        body = data[off + 8:off + 8 + clen]
        yield ctype, body
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    print('glb version', version, 'bytes', length)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else DEFAULT
    with open(path, 'rb') as fh:
        data = fh.read()

    doc = None
    images = []
    for ctype, body in chunks(data):
        if ctype == 0x4E4F534A:  # JSON
            doc = json.loads(body.decode('utf-8'))
        elif ctype == 0x004E4942:  # BIN
            images.append(len(body))
    if doc is None:
        raise SystemExit('no JSON chunk')

    nodes = doc.get('nodes', [])
    meshes = doc.get('meshes', [])
    mats = doc.get('materials', [])
    print('nodes %d  meshes %d  materials %d  images %d  textures %d  animations %d  skins %d'
          % (len(nodes), len(meshes), len(mats), len(doc.get('images', [])),
             len(doc.get('textures', [])), len(doc.get('animations', [])), len(doc.get('skins', []))))
    print('extensionsUsed:', doc.get('extensionsUsed'))
    print('bin chunk bytes:', images)

    verts = 0
    prims = 0
    for m in meshes:
        for p in m.get('primitives', []):
            prims += 1
            acc = doc['accessors'][p['attributes']['POSITION']]
            verts += acc['count']
    print('primitives %d  POSITION vertices (sum, incl. duplicated per primitive) %d' % (prims, verts))

    # material -> mesh/primitives/verts
    by_mat = defaultdict(lambda: [0, 0])
    for m in meshes:
        for p in m.get('primitives', []):
            name = mats[p['material']]['name'] if 'material' in p else '<none>'
            by_mat[name][0] += 1
            by_mat[name][1] += doc['accessors'][p['attributes']['POSITION']]['count']
    print('\nmaterials -> primitives, verts:')
    for name, (pc, vc) in sorted(by_mat.items()):
        print('  %-12s %3d  %6d' % (name, pc, vc))

    # --- world-space bbox: exactly what three's Box3.setFromObject will report -----------
    # The raw accessor min/max are in each mesh's own local space, so they do not describe
    # the assembled fly. Composing the node matrices first is what makes the numbers
    # meaningful for choosing up/forward and for the fit-to-camera scale.
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get('children', []):
            parent[c] = i
    world = {}

    def wm(i):
        if i in world:
            return world[i]
        local = np.eye(4)
        if 'matrix' in nodes[i]:
            local = np.array(nodes[i]['matrix'], dtype=np.float64).reshape(4, 4).T
        else:
            t = nodes[i].get('translation', [0, 0, 0])
            x, y, z, w = nodes[i].get('rotation', [0, 0, 0, 1])
            s = nodes[i].get('scale', [1, 1, 1])
            rot = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
                [0, 0, 0, 1],
            ], dtype=np.float64)
            for k in range(3):
                rot[:3, k] = rot[:3, k] * s[k]
            rot[:3, 3] = t
            local = rot
        m = wm(parent[i]) @ local if i in parent else local
        world[i] = m
        return m

    lo = np.array([1e18] * 3)
    hi = np.array([-1e18] * 3)
    for i, n in enumerate(nodes):
        if 'mesh' not in n:
            continue
        m = wm(i)
        for p in meshes[n['mesh']].get('primitives', []):
            acc = doc['accessors'][p['attributes']['POSITION']]
            amin = acc.get('min')
            amax = acc.get('max')
            if not amin or not amax:
                continue
            for cx in (amin[0], amax[0]):
                for cy in (amin[1], amax[1]):
                    for cz in (amin[2], amax[2]):
                        v = (m @ np.array([cx, cy, cz, 1.0]))[:3]
                        lo = np.minimum(lo, v)
                        hi = np.maximum(hi, v)
    size = hi - lo
    ctr = (lo + hi) / 2
    print('\nworld-space bbox (what three Box3.setFromObject reports):')
    print('  min [%10.1f %10.1f %10.1f]' % tuple(lo))
    print('  max [%10.1f %10.1f %10.1f]' % tuple(hi))
    print('  size [%10.1f %10.1f %10.1f]  centre [%9.1f %9.1f %9.1f]' % (*size, *ctr))
    order = np.argsort(size)[::-1]
    print('  axes longest to shortest: %s   span %.1f  max_dim %.1f'
          % (['XYZ'[k] for k in order], float(np.linalg.norm(size)), float(size.max())))

    # material of the biggest contributors, to sanity-check which part owns which axis
    print('\nper-material world extents:')
    by_mat = defaultdict(lambda: [np.array([1e18] * 3), np.array([-1e18] * 3)])
    for i, n in enumerate(nodes):
        if 'mesh' not in n:
            continue
        m = wm(i)
        for p in meshes[n['mesh']].get('primitives', []):
            acc = doc['accessors'][p['attributes']['POSITION']]
            if not acc.get('min'):
                continue
            mn, mx = by_mat[p['material']]
            for cx in (acc['min'][0], acc['max'][0]):
                for cy in (acc['min'][1], acc['max'][1]):
                    for cz in (acc['min'][2], acc['max'][2]):
                        v = (m @ np.array([cx, cy, cz, 1.0]))[:3]
                        mn = np.minimum(mn, v)
                        mx = np.maximum(mx, v)
            by_mat[p['material']] = [mn, mx]
    for mi, (mn, mx) in sorted(by_mat.items()):
        print('  %-12s y[%8.1f %8.1f]  x[%8.1f %8.1f]  z[%8.1f %8.1f]'
              % (mats[mi].get('name', '?'), mn[1], mx[1], mn[0], mx[0], mn[2], mx[2]))

    # node name census by prefix
    prefixes = Counter()
    for n in nodes:
        nm = n.get('name', '<unnamed>')
        # longest alphabetic prefix, e.g. FLYPAT33 -> FLYPAT
        p = ''
        for ch in nm:
            if ch.isalpha():
                p += ch
            else:
                break
        prefixes[p or nm] += 1
    print('\nnode name prefixes:')
    for p, c in sorted(prefixes.items(), key=lambda t: (-t[1], t[0])):
        print('  %-14s %3d' % (p, c))

    for group in [a for a in sys.argv[1:] if a.startswith('--group=')]:
        want = group.split('=', 1)[1]
        hits = [n.get('name', '') for n in nodes if n.get('name', '').startswith(want)]
        print('\n%s* (%d): %s' % (want, len(hits), sorted(hits)))

    # full census of the interesting groups
    for want in ('FLYALA', 'FLYOJO', 'FLYCUL', 'FLYMAIN', 'FLYALI'):
        hits = sorted(n.get('name', '') for n in nodes if n.get('name', '').startswith(want))
        print('\n%-8s %2d  %s' % (want, len(hits), hits))

    legs = sorted(n.get('name', '') for n in nodes if n.get('name', '').startswith('FLYPAT'))
    print('\nFLYPAT %2d  %s' % (len(legs), legs))

    # which nodes carry a mesh, i.e. what actually draws
    with_mesh = [n for n in nodes if 'mesh' in n]
    print('\nnodes with a mesh: %d ; nodes that are pure transform: %d' % (len(with_mesh), len(nodes) - len(with_mesh)))
    no_mesh_names = sorted(n.get('name', '') for n in nodes if 'mesh' not in n)
    print('pure-transform nodes:', no_mesh_names[:40])

    # hierarchy depth + parent map sanity
    roots = [i for i in range(len(nodes))]
    children = set()
    for n in nodes:
        for c in n.get('children', []):
            children.add(c)
    roots = [i for i in roots if i not in children]
    depth = {i: 0 for i in roots}
    changed = True
    guard = 0
    while changed and guard < 50:
        changed = False
        guard += 1
        for i, n in enumerate(nodes):
            for c in n.get('children', []):
                d = depth.get(i, 0) + 1
                if depth.get(c, -1) < d:
                    depth[c] = d
                    changed = True
    print('roots: %d  max depth: %d' % (len(roots), max(depth.values()) if depth else 0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
