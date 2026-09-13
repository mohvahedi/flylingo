/**
 * realFly.ts: the non-React half of the hero path.
 *
 * The supplied asset (public/models/fly.glb) is a STATIC mesh: no skin, no baked clips.
 * What it does have is 172 separate nodes, so motion is produced by driving named node
 * transforms. That is a different kind of rig from the procedural fly in Fly.tsx, which
 * poses a skeleton of groups it created itself. This module is the bridge: it maps the
 * SAME Pose struct (the five-behaviour state machine in pose.ts) onto the asset's real
 * node chains, so both paths animate from one source of truth.
 *
 * Measured facts about the asset that this file relies on. tools/inspect_glb.py and
 * tools/rig_map.py print all of it, so none of it is guessed:
 *
 *  - 172 nodes, 57 meshes, 7 materials, 9 embedded textures, 6511 vertices, 0 skins,
 *    0 animations. World-space bounds 4017 x 2097 x 5056 units centred near the origin.
 *  - +Y is up and +Z is the head end, so no orientation correction is needed and
 *    ORIENTATION stays identity. Up: wings reach y=+1252, feet reach y=-845. Head end:
 *    FLYOJOS ("ojos", eyes) hangs under FLYCAB ("cabeza", head) spanning z 1263..2120,
 *    while FLYTCUL ("culo", abdomen) spans z -1910..-252. Note this means the head node
 *    is FLYCAB, not FLYCUL: the eyes being children of FLYCAB is what settles it.
 *  - the 6 legs are parent-child chains of 8 segments, and the chain order is NOT the
 *    node numbering order (FLYPAT01 -> FLYPAT07 -> FLYPAT08 -> ... -> FLYPAT02), so
 *    chains are walked at runtime rather than hard-coded.
 *  - wings are FLYALA / FLYALA0 (material FLYALAS). FLYALI / FLYALI0 carry FLYTCORT and
 *    are the halteres, which is what a fly actually has behind its wings.
 *  - which node of each pair is the left one is derived from its world x, not assumed.
 *
 * Axis handling. The nodes sit under 11 levels of 3DSMeshMatrix transforms, so a node's
 * own local axes are NOT the model's axes. Each driven node therefore precomputes, once,
 * how the three model axes are expressed in its own post-rest local frame, and per-frame
 * rotations are built about those. That is what makes "flap the wing about the body's
 * longitudinal axis" correct without hand-tuning a sign per node.
 *
 * The whole per-frame path writes into quaternions and vectors allocated at load time and
 * allocates nothing.
 */
import * as THREE from 'three';
import { BODY_BASE_Y, legNeutral, type LegPose, type Pose } from './pose';

export const MODEL_URL = `${import.meta.env.BASE_URL}models/fly.glb`;

/** Longest axis of the asset is scaled to this, in scene units. The camera sees ~1.75 units. */
export const TARGET_SPAN = 1.75;

/**
 * Orientation correction, in radians of XYZ euler, applied between the fit group and the
 * mesh. Identity because the asset was measured upright and head-forward. It exists so a
 * replacement asset that loads on its side is one constant away from being fixed.
 */
export const ORIENTATION: [number, number, number] = [0, 0, 0];

/** The pose struct's neutral wing angles, so idle maps to the asset's authored rest pose. */
const WING_REST_FLAP = 0.32;
const WING_REST_SWEEP = -0.16;

/** Angle gains. 1.0 means "use the procedural rig's own amplitudes unchanged". */
const LEG_GAIN = 1.0;
const WING_GAIN = 1.0;
const HALTERE_GAIN = 0.9;
/** The head dip that stands in for a proboscis this mesh does not have. Radians at full extension. */
const PROBOSCIS_DIP = 0.55;

export type Region = 'head' | 'thorax' | 'abdomen' | 'legs' | 'wings';

type Driven = {
  node: THREE.Object3D;
  name: string;
  rest: THREE.Quaternion;
  restPos: THREE.Vector3;
  restScale: THREE.Vector3;
  /** model +X / +Y / +Z expressed in this node's own post-rest local frame */
  ax: THREE.Vector3;
  ay: THREE.Vector3;
  az: THREE.Vector3;
  /** scratch, reused every frame */
  q: THREE.Quaternion;
  /** the local point furthest from this node's origin, used to read a tip position */
  tip: THREE.Vector3 | null;
};

export type LegChain = {
  /** segment 0 is the hip, the rest run down the leg */
  seg: Driven[];
  /** +1 = left (+x), -1 = right (-x) */
  side: number;
  /** pose row convention: 0 fore, 1 mid, 2 hind */
  row: number;
  /** index into pose.legs, 0..5 */
  index: number;
};

export type Rig = {
  root: Driven;
  head: Driven;
  abdomen: Driven;
  wings: Driven[];
  halteres: Driven[];
  legs: LegChain[];
  /** every FLY* transform that gets written to, in one list */
  all: Driven[];
};

export type ProbeSample = {
  t: number;
  behavior: string;
  source: string;
  /** wingtip positions in the fitted model's own space */
  wing: [number, number, number][];
  /** deepest joint of each leg chain */
  foot: [number, number, number][];
  head: [number, number, number];
  /** mesh-space bounding info of the whole rig, refreshed on demand */
  span: [number, number, number];
  /** the five additive overlay opacities, so an inert frame can be proven to be inert */
  glow: number[];
};

/* ------------------------------------------------------------------ rig reading */

function byName(root: THREE.Object3D, name: string): THREE.Object3D | null {
  let found: THREE.Object3D | null = null;
  root.traverse((o) => {
    if (!found && o.name === name) found = o;
  });
  return found;
}

/**
 * Builds a driven record for a node: its rest transform and the three model axes in its
 * own local frame. Requires world matrices to be up to date.
 */
function drivenFor(node: THREE.Object3D): Driven {
  const rest = node.quaternion.clone();
  const invRest = rest.clone().invert();
  const invParent = new THREE.Quaternion();
  if (node.parent) node.parent.getWorldQuaternion(invParent);
  invParent.invert();
  const toLocal = (x: number, y: number, z: number) =>
    new THREE.Vector3(x, y, z).applyQuaternion(invParent).applyQuaternion(invRest).normalize();
  return {
    node,
    name: node.name,
    rest,
    restPos: node.position.clone(),
    restScale: node.scale.clone(),
    ax: toLocal(1, 0, 0),
    ay: toLocal(0, 1, 0),
    az: toLocal(0, 0, 1),
    q: new THREE.Quaternion(),
    tip: null,
  };
}

/**
 * The local point of a node's subtree furthest from the node's own origin. Used to read a
 * wingtip rather than the hinge, so the debug probe reports motion, not stillness.
 */
function furthestLocalPoint(node: THREE.Object3D): THREE.Vector3 {
  const box = new THREE.Box3().setFromObject(node);
  const inv = new THREE.Matrix4().copy(node.matrixWorld).invert();
  const out = new THREE.Vector3();
  const best = new THREE.Vector3();
  let bestLen = -1;
  const corners: THREE.Vector3[] = [];
  for (let i = 0; i < 8; i += 1) {
    corners.push(
      new THREE.Vector3(
        i & 1 ? box.max.x : box.min.x,
        i & 2 ? box.max.y : box.min.y,
        i & 4 ? box.max.z : box.min.z,
      ),
    );
  }
  for (const c of corners) {
    const local = c.applyMatrix4(inv);
    const len = local.length();
    if (len > bestLen) {
      bestLen = len;
      best.copy(local);
    }
  }
  out.copy(best);
  return out;
}

/**
 * Walks the FLYPAT chains. A chain root is a FLYPAT node with no FLYPAT ancestor; its
 * chain is the FLYPAT descendants down its first FLYPAT child at each level.
 */
function readLegs(root: THREE.Object3D): LegChain[] {
  const roots: THREE.Object3D[] = [];
  root.traverse((o) => {
    if (!/^FLYPAT/.test(o.name)) return;
    let p = o.parent;
    while (p) {
      if (/^FLYPAT/.test(p.name)) return;
      p = p.parent;
    }
    roots.push(o);
  });

  const chains: LegChain[] = [];
  const world = new THREE.Vector3();
  for (const r of roots) {
    const seg: Driven[] = [];
    let cur: THREE.Object3D | null = r;
    while (cur) {
      seg.push(drivenFor(cur));
      let next: THREE.Object3D | null = null;
      for (const child of cur.children) {
        if (/^FLYPAT/.test(child.name)) {
          next = child;
          break;
        }
      }
      cur = next;
    }
    r.getWorldPosition(world);
    chains.push({
      seg,
      side: world.x >= 0 ? 1 : -1,
      row: -1,
      index: -1,
    });
  }

  // Row is the fore/aft rank. Head is +z, so the largest z is the fore leg.
  for (const side of [1, -1]) {
    const group = chains.filter((c) => c.side === side);
    const zs = group.map((c) => {
      const w = new THREE.Vector3();
      c.seg[0].node.getWorldPosition(w);
      return { c, z: w.z };
    });
    zs.sort((a, b) => a.z - b.z);
    zs.forEach((entry, rank) => {
      // rank 0 is the smallest z, which is the hindmost leg.
      const row = 2 - rank;
      entry.c.row = row;
      entry.c.index = (side > 0 ? 0 : 3) + row;
    });
  }

  chains.sort((a, b) => a.index - b.index);
  return chains;
}

/**
 * Region of a mesh node for the glow shell: the nearest FLY* ancestor decides.
 * FLYMAIN itself is the thorax, which is the remainder.
 */
export function regionOf(node: THREE.Object3D): Region {
  let o: THREE.Object3D | null = node;
  while (o) {
    const n = o.name;
    if (/^FLYPAT/.test(n)) return 'legs';
    if (/^FLYALA/.test(n)) return 'wings';
    if (/^FLYALI/.test(n)) return 'wings';
    if (/^FLYOJO/.test(n)) return 'head';
    if (n === 'FLYCAB') return 'head';
    if (n === 'FLYCUL') return 'abdomen';
    o = o.parent;
  }
  return 'thorax';
}

/**
 * Reads the asset's rig. The node must be at its authored rest transform, and its world
 * matrices must be current.
 */
export function buildRig(root: THREE.Object3D, options: { quiet?: boolean } = {}): Rig {
  root.updateMatrixWorld(true);

  const main = byName(root, 'FLYMAIN');
  const head = byName(root, 'FLYCAB');
  const abdomen = byName(root, 'FLYCUL');
  if (!main || !head || !abdomen) {
    throw new Error(
      `realFly: asset is missing the expected nodes (FLYMAIN ${!!main}, FLYCAB ${!!head}, FLYCUL ${!!abdomen})`,
    );
  }

  const wings: Driven[] = [];
  const halteres: Driven[] = [];
  for (const n of ['FLYALA', 'FLYALA0']) {
    const node = byName(root, n);
    if (node) {
      const d = drivenFor(node);
      d.tip = furthestLocalPoint(node);
      wings.push(d);
    }
  }
  for (const n of ['FLYALI', 'FLYALI0']) {
    const node = byName(root, n);
    if (node) {
      const d = drivenFor(node);
      d.tip = furthestLocalPoint(node);
      halteres.push(d);
    }
  }
  const legs = readLegs(root);

  const rig: Rig = {
    root: drivenFor(main),
    head: drivenFor(head),
    abdomen: drivenFor(abdomen),
    wings,
    halteres,
    legs,
    all: [],
  };
  rig.all = [rig.root, rig.head, rig.abdomen, ...wings, ...halteres];
  for (const c of legs) for (const s of c.seg) rig.all.push(s);

  if (import.meta.env.DEV && !options.quiet) {
    const names = rig.all.map((d) => d.name).join(',');
    console.info(
      `[fly] rig: legs ${legs.length} chains (${legs.map((l) => `${l.index}:${l.seg.length}segz=${l.side > 0 ? 'L' : 'R'}`).join(' ')}), wings ${wings.length}, halteres ${halteres.length}, driven ${rig.all.length}`,
    );
    if (names.length === 0) console.warn('[fly] rig has no driven nodes');
  }
  return rig;
}

/* ------------------------------------------------------------------ posing */

/**
 * Reused by all six legs on every frame. legNeutral writes all five fields, so a single
 * scratch object is enough and the per-frame path stays allocation free.
 */
const neutralScratch: LegPose = { hipYaw: 0, hipRoll: 0, femur: 0, knee: 0, tarsus: 0 };

function mulAxis(d: Driven, axis: THREE.Vector3, angle: number): void {
  if (angle === 0) return;
  d.q.setFromAxisAngle(axis, angle);
  d.node.quaternion.multiply(d.q);
}

/** Hinge angle for a leg segment: positive lifts the foot in the body's XY plane. */
function hingeZ(d: Driven, angle: number): void {
  mulAxis(d, d.az, angle);
}

/**
 * Applies the shared Pose to the asset. Allocation free: writes into the node transforms
 * and the preallocated scratch quaternion only.
 */
export function applyPose(rig: Rig, pose: Pose): void {
  // ---- root: bob, walk translation, yaw, pitch, roll -------------------------------
  const r = rig.root;
  r.node.position.copy(r.restPos);
  r.node.position.y += pose.rootY - BODY_BASE_Y + pose.bodyY;
  r.node.position.z += pose.rootZ;
  r.node.quaternion.copy(r.rest);
  // positive bodyPitch / headPitch reads as "nose down" in the shared pose vocabulary
  mulAxis(r, r.ax, -pose.bodyPitch);
  mulAxis(r, r.ay, pose.rootYaw + pose.bodyYaw);
  mulAxis(r, r.az, pose.bodyRoll);

  // ---- head: this asset's head node carries the eyes --------------------------------
  const h = rig.head;
  h.node.quaternion.copy(h.rest);
  const headDip = pose.headPitch + PROBOSCIS_DIP * pose.proboscis;
  mulAxis(h, h.ax, -headDip);
  mulAxis(h, h.ay, pose.headYaw + 0.12 * pose.proboscisWiggle);

  // ---- abdomen: one rigid node, so the four-segment ripple becomes one bend ---------
  const ab = rig.abdomen;
  ab.node.quaternion.copy(ab.rest);
  const abdMean = (pose.abdPitch[0] + pose.abdPitch[1] + pose.abdPitch[2] + pose.abdPitch[3]) / 4;
  const abdBreathe = (pose.abdScale[0] + pose.abdScale[1] + pose.abdScale[2] + pose.abdScale[3]) / 4;
  mulAxis(ab, ab.ax, abdMean);
  // the segments do not scale, but a little swell keeps the idle breathing readable
  ab.node.scale.copy(ab.restScale).multiplyScalar(1 + 0.6 * (abdBreathe - 1));

  // ---- legs: hip yaw about the vertical, everything else about the body axis ---------
  for (const chain of rig.legs) {
    const leg = pose.legs[chain.index];
    legNeutral(chain.index, neutralScratch);
    const neutral = neutralScratch;
    const seg = chain.seg;
    // hip: fore/aft about model Y, then lift about model Z
    seg[0].node.quaternion.copy(seg[0].rest);
    mulAxis(seg[0], seg[0].ay, leg.hipYaw - neutral.hipYaw);
    mulAxis(seg[0], seg[0].az, leg.hipRoll - neutral.hipRoll);
    const bends = [
      leg.femur - neutral.femur,
      leg.knee - neutral.knee,
      leg.tarsus - neutral.tarsus,
    ];
    for (let i = 0; i < seg.length; i += 1) {
      const d = seg[i];
      if (i > 0) d.node.quaternion.copy(d.rest);
      if (i >= 1 && i <= 3) hingeZ(d, bends[i - 1] * LEG_GAIN);
    }
  }

  // ---- wings and halteres ------------------------------------------------------------
  for (const wing of rig.wings) {
    const side = wingSide(wing);
    const flap = (side > 0 ? pose.wingL.flap : pose.wingR.flap) - WING_REST_FLAP;
    const sweep = (side > 0 ? pose.wingL.sweep : pose.wingR.sweep) - WING_REST_SWEEP;
    wing.node.quaternion.copy(wing.rest);
    // flap about the body's longitudinal axis; mirrored by side so both wings rise together
    mulAxis(wing, wing.az, side * flap * WING_GAIN);
    mulAxis(wing, wing.ay, -side * sweep * WING_GAIN);
  }
  for (const hal of rig.halteres) {
    const side = wingSide(hal);
    hal.node.quaternion.copy(hal.rest);
    mulAxis(hal, hal.az, side * (pose.haltere - 0.5) * HALTERE_GAIN);
    mulAxis(hal, hal.ay, -side * 0.35 * (pose.haltere - 0.5) * HALTERE_GAIN);
  }
}

/** Cached side per wing/haltere node, read from the node's rest world position once. */
const sideCache = new WeakMap<THREE.Object3D, number>();
function wingSide(d: Driven): number {
  let s = sideCache.get(d.node);
  if (s === undefined) {
    const w = new THREE.Vector3();
    d.node.getWorldPosition(w);
    s = w.x >= 0 ? 1 : -1;
    sideCache.set(d.node, s);
  }
  return s;
}

/* ------------------------------------------------------------------ fit and probe */

/**
 * Centres the asset on x/z, puts its feet on y=0 and scales its longest axis to
 * `span`. Computed from the loaded bounding box rather than hard-coded, so replacing the
 * asset with a differently sized one keeps the framing.
 */
export function fitModel(root: THREE.Object3D, span = TARGET_SPAN) {
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const longest = Math.max(size.x, size.y, size.z) || 1;
  const scale = span / longest;
  const position = new THREE.Vector3(
    -centre.x * scale,
    -box.min.y * scale,
    -centre.z * scale,
  );
  return { scale, position, size, box };
}

export function createProbe(): ProbeSample {
  return {
    t: 0,
    behavior: 'idle',
    source: 'auto',
    wing: [
      [0, 0, 0],
      [0, 0, 0],
    ],
    foot: [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ],
    head: [0, 0, 0],
    span: [0, 0, 0],
    glow: [0, 0, 0, 0, 0],
  };
}

const probeW = new THREE.Vector3();
const probeV = new THREE.Vector3();

/**
 * Fills the probe in place with real world positions, so a headless run can prove the
 * nodes actually move. Cheaper and far more honest than a WebGL readPixels, which returns
 * zeros on this stack because react-three-fiber leaves preserveDrawingBuffer off.
 */
export function sampleProbe(rig: Rig, probe: ProbeSample, t: number, behavior: string, source: string): void {
  probe.t = t;
  probe.behavior = behavior;
  probe.source = source;
  rig.wings.forEach((wing, i) => {
    if (i > 1) return;
    probeW.copy(wing.tip ?? probeV.set(0, 0, 0));
    wing.node.localToWorld(probeW);
    probe.wing[i][0] = probeW.x;
    probe.wing[i][1] = probeW.y;
    probe.wing[i][2] = probeW.z;
  });
  rig.legs.forEach((chain, i) => {
    if (i > 5) return;
    const deepest = chain.seg[chain.seg.length - 1];
    deepest.node.getWorldPosition(probeW);
    probe.foot[i][0] = probeW.x;
    probe.foot[i][1] = probeW.y;
    probe.foot[i][2] = probeW.z;
  });
  rig.head.node.getWorldPosition(probeW);
  probe.head[0] = probeW.x;
  probe.head[1] = probeW.y;
  probe.head[2] = probeW.z;
}
