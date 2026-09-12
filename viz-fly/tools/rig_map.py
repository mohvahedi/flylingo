"""Map the fly's node rig into something a walk cycle can drive.

Reads the glb JSON, composes world matrices from the default TRS, and answers the three
questions the animator needs:
  1. which FLY* transform nodes are the leg chains, and in what order do segments nest
  2. where each leg root sits in space, so front/mid/back and left/right can be labelled
  3. what the body/wing/head/eye nodes look like, and how the 3DSMeshMatrix layer sits
     between a FLY* node and the mesh that actually draws

Prints the leg table and a species-neutral summary; use --tree for the full hierarchy.

    python tools/rig_map.py [path] [--tree] [--legs]
"""
import json
import struct
import sys
from collections import defaultdict

import numpy as np

DEFAULT = 'D:/Projects/flylingo/viz-fly/public/models/fly.glb'


def load(path):
    with open(path, 'rb') as fh:
        data = fh.read()
    off = 12
    doc = None
    while off < len(data):
        clen, ctype = struct.unpack_from('<II', data, off)
        if ctype == 0x4E4F534A:
            doc = json.loads(data[off + 8:off + 8 + clen].decode('utf-8'))
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    return doc


def trs(node):
    m = np.eye(4)
    if 'matrix' in node:
        return np.array(node['matrix'], dtype=np.float64).reshape(4, 4).T
    t = node.get('translation', [0, 0, 0])
    r = node.get('rotation', [0, 0, 0, 1])
    s = node.get('scale', [1, 1, 1])
    x, y, z, w = r
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    rot[:3, 0] *= s[0]
    rot[:3, 1] *= s[1]
    rot[:3, 2] *= s[2]
    rot[0, 3], rot[1, 3], rot[2, 3] = t
    return rot


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else DEFAULT
    doc = load(path)
    nodes = doc['nodes']
    name = lambda i: nodes[i].get('name', '<%d>' % i)

    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get('children', []):
            parent[c] = i

    world = {}

    def wm(i):
        if i in world:
            return world[i]
        local = trs(nodes[i])
        m = wm(parent[i]) @ local if i in parent else local
        world[i] = m
        return m

    for i in range(len(nodes)):
        wm(i)

    def pos(i):
        return wm(i)[:3, 3]

    def label(v):
        u = v / (np.linalg.norm(v) or 1)
        return '%+.2f,%+.2f,%+.2f' % (u[0], u[1], u[2])

    print('root node:', name([i for i in range(len(nodes)) if i not in parent][0]))
    # leaf-with-mesh chain for the first leg node, to confirm the FLY* -> 3DSMeshMatrix -> Object nesting
    for i, n in enumerate(nodes):
        if n.get('name', '').startswith('FLYPAT01'):
            chain = []
            j = i
            while True:
                chain.append('%s(mesh=%s)' % (name(j), nodes[j].get('mesh')))
                kids = nodes[j].get('children', [])
                if not kids:
                    break
                j = kids[0]
            print('chain from FLYPAT01:', ' -> '.join(chain))
            break

    print('\n=== FLY transform nodes: world position (units) and local translation ===')
    groups = defaultdict(list)
    for i, n in enumerate(nodes):
        nm = n.get('name', '')
        if nm.startswith('FLY') and not any(k in nm for k in ('3DS',)):
            groups[nm.rstrip('0123456789') if nm[:6] not in ('FLYMAI',) else nm].append(i)

    legs = [i for i, n in enumerate(nodes) if n.get('name', '').startswith('FLYPAT')]
    legs.sort(key=lambda i: name(i))
    legroots = [i for i in legs if name(parent.get(i, -1)).startswith('FLYCAB') or
                not name(parent.get(i, -1)).startswith('FLYPAT')]
    print('FLYPAT nodes: %d ; legs (chains whose parent is not another FLYPAT): %d'
          % (len(legs), len(legroots)))

    rows = []
    for r in legroots:
        chain = [r]
        j = r
        while True:
            kids = [c for c in nodes[j].get('children', []) if name(c).startswith('FLYPAT')]
            if not kids:
                break
            j = kids[0]
            chain.append(j)
        p = pos(r)
        rows.append((name(r), chain, p))

    print('\n%-12s %-5s  world position                len  chain'
          % ('leg root', 'depth'))
    for nm, chain, p in sorted(rows, key=lambda t: t[2][0]):
        print('%-12s %-5s  [%9.1f %9.1f %9.1f]  %2d   %s'
              % (nm, len(chain), p[0], p[1], p[2], len(chain), ','.join(name(c) for c in chain)))

    if rows:
        # body-frame classification: convert to the FLYMAIN frame so "front" means the head end
        body = [i for i, n in enumerate(nodes) if n.get('name') == 'FLYMAIN'][0]
        inv = np.linalg.inv(wm(body))
        print('\nleg roots in body frame (x=right, y=up, z=front/back as authored):')
        for nm, chain, _ in rows:
            local = (inv @ np.append(pos(chain[0]), 1))[:3]
            print('  %-12s local [%8.1f %8.1f %8.1f]  segments %d' % (nm, local[0], local[1], local[2], len(chain)))

        # do leg chains branch (some segments having 2 FLYPAT children)?
        branch = [(name(i), len([c for c in nodes[i].get('children', []) if name(c).startswith('FLYPAT')]))
                  for i in legs if len([c for c in nodes[i].get('children', []) if name(c).startswith('FLYPAT')]) > 1]
        print('\nFLYPAT segments with >1 FLYPAT child:', branch if branch else 'none (each leg is a single chain)')

    print('\n=== other named nodes ===')
    for want in ('FLYMAIN', 'FLYALA', 'FLYALA0', 'FLYALI', 'FLYALI0', 'FLYOJO', 'FLYOJO0', 'FLYCUL', 'FLYCAB'):
        for i, n in enumerate(nodes):
            if n.get('name') == want:
                p = pos(i)
                print('  %-9s parent=%-14s world [%9.1f %9.1f %9.1f] children=%d mesh=%s'
                      % (want, name(parent.get(i, -1)), p[0], p[1], p[2],
                         len(n.get('children', [])), n.get('mesh')))
                break

    # which mesh material each of the named groups ultimately draws
    matname = lambda mi: doc['materials'][mi].get('name', '<%d>' % mi)
    print('\n=== material of each named group (walking to the first mesh child) ===')
    seen = set()
    for i, n in enumerate(nodes):
        nm = n.get('name', '')
        if nm.startswith(('FLYALA', 'FLYALI', 'FLYOJO', 'FLYCUL', 'FLYCAB', 'FLYMAIN', 'FLYPAT01')):
            key = nm.rstrip('0123456789') or nm
            if key in seen:
                continue
            seen.add(key)
            mats = []
            stack = [i]
            while stack:
                j = stack.pop()
                if 'mesh' in nodes[j]:
                    for pr in doc['meshes'][nodes[j]['mesh']]['primitives']:
                        if 'material' in pr:
                            mats.append(matname(pr['material']))
                stack.extend(nodes[j].get('children', []))
            print('  %-10s -> %s' % (nm, sorted(set(mats))))

    if '--tree' in sys.argv:
        print('\n=== tree ===')

        def dump(i, d):
            n = nodes[i]
            print('%s%s%s' % ('  ' * d, name(i), ' [mesh]' if 'mesh' in n else ''))
            for c in n.get('children', []):
                dump(c, d + 1)
        for r in [i for i in range(len(nodes)) if i not in parent]:
            dump(r, 0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
