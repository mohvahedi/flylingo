/**
 * The fly itself: every mesh is generated in code from three.js primitives. No .glb, no
 * texture files, no CDN. If the host is offline the fly still renders, which matters
 * because this harness is embedded in an app that may be demoed without network.
 *
 * Hierarchy (each group is rotated by the pose values computed in pose.ts):
 *
 *   root (rootY, rootZ, rootYaw)
 *     body (breathing Y, pitch, roll, yaw)
 *       thorax -> setae
 *       head -> eyes, ocelli, antennae (each with setae), proboscis
 *       abdomen -> 4 segments, chained, each with its own material and its own setae
 *       leg x6 -> hip(yaw,roll) -> femur(+setae) -> knee -> tibia -> tarsus
 *       wing x2 -> flap, sweep
 *       haltere x2
 *
 * Glow: 18 channels from regions.ts map onto 18 visible parts (head, thorax, 4 abdomen
 * segments, 6 legs, 2 wings). Every part owns its own material, so a locally active patch
 * of the sampled subset lights exactly that part and no other. That per-part mapping is
 * what makes the connectome state legible on the body instead of being an average glow.
 *
 * The materials are also the only place the fly knows about magnitude: `live.vitality` is
 * already normalized in regions.ts against the measured p95 of |activity|, and is exactly
 * 0 for an all-zero frame, so nothing here can amplify a dead network into fake life.
 *
 * Look: the materials in materials.ts are physical (clearcoat, thin film iridescence,
 * sheen) with a shader patch for cuticle microstructure. The emissive channel stays the
 * instrument layer on top of a warm amber insect, so a hot frame glows and a dead frame is
 * a lit but lifeless specimen.
 */
import { useEffect, useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { REGION_COLOR, type ChannelKind } from './regions';
import { LEG_LEN, LEG_ROWS, legRow, legSide, type Pose } from './pose';
import type { LiveReadout } from './animator';
import {
  chitinMaterial,
  eyeMaterial,
  setaeMaterial,
  wingGeometry,
  wingMaterial,
} from './materials';

/** Materials, one per visible part that carries a channel. */
type FlyMats = {
  head: THREE.MeshPhysicalMaterial;
  thorax: THREE.MeshPhysicalMaterial;
  abdomen: THREE.MeshPhysicalMaterial[];
  legs: THREE.MeshPhysicalMaterial[];
  wings: THREE.MeshPhysicalMaterial;
  eyes: THREE.MeshPhysicalMaterial;
  gloss: THREE.MeshPhysicalMaterial;
  setae: THREE.MeshPhysicalMaterial;
};

/** Signature used to find the channel index for a given region slot. */
export type SlotOf = (kind: ChannelKind, slot: number) => number;

export type FlyProps = {
  /** 0..1 smoothed per-channel drive, length CHANNELS.length (18) */
  drive: Float32Array;
  /** the current pose, mutated in place each frame by the animator */
  pose: Pose;
  /** live animator readout, mutated in place: vitality, behavior, inert */
  live: LiveReadout;
  /** the per-channel slot a mesh belongs to, so meshes can look up their own drive */
  slotOf: SlotOf;
};

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Base emissive intensity with no network at all. Deliberately dim but not black. */
const BASE_GLOW = 0.1;

/** Wing span and chord, shared by the geometry and the vein pattern. */
const WING_LEN = 0.66;
const WING_CHORD = 0.2;

type FlyAssets = {
  mats: FlyMats;
  wingGeo: THREE.BufferGeometry;
  setaeGeo: THREE.BufferGeometry;
};

function makeAssets(): FlyAssets {
  const mats: FlyMats = {
    head: chitinMaterial('head', 'head'),
    thorax: chitinMaterial('thorax', 'thorax'),
    abdomen: [0, 1, 2, 3].map(() => chitinMaterial('abdomen', 'abdomen')),
    legs: [0, 1, 2, 3, 4, 5].map(() => chitinMaterial('legs', 'legs')),
    wings: wingMaterial(WING_LEN, WING_CHORD),
    // one facet frequency for the big compound eyes, a finer one for the small glossy
    // parts (ocelli, antenna tips, proboscis labellum) so the facets stay the same size
    // in world units instead of turning into moire on a 3 mm sphere
    eyes: eyeMaterial(8.5),
    gloss: eyeMaterial(26),
    setae: setaeMaterial(),
  };
  const wingGeo = wingGeometry(WING_LEN, WING_CHORD);
  // one unit height cone, reused by every setae instance and scaled per bristle
  const setaeGeo = new THREE.ConeGeometry(0.0068, 1, 4, 1);
  setaeGeo.translate(0, 0.5, 0);
  return { mats, wingGeo, setaeGeo };
}

function disposeAssets(a: FlyAssets) {
  a.wingGeo.dispose();
  a.setaeGeo.dispose();
  Object.values(a.mats)
    .flat()
    .forEach((m) => m.dispose());
}

/* ------------------------------------------------------------------- setae */

/** Deterministic LCG, so the same fly is generated on every mount and every machine. */
function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export type SetaeProps = {
  /** how many bristles */
  count: number;
  seed: number;
  /** ellipsoid the bristles are sampled from, in the parent's frame */
  cx: number;
  cy: number;
  cz: number;
  rx: number;
  ry: number;
  rz: number;
  /**
   * cos(theta) band to sample over, measured from the local +Y pole. [0.2, 1] is the top
   * cap of a body part, [-1, 1] is all round a limb.
   */
  capLo: number;
  capHi: number;
  /** how far each bristle leans toward the tail (-z) */
  sweep: number;
  /** bristle length in world units */
  length: number;
  /** per bristle length jitter, 0..1 */
  variance?: number;
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
};

/**
 * Instanced bristles. Built once, never touched again: the matrices are baked in the
 * parent's frame, so the setae on a femur ride the walk cycle for free and the ones on the
 * thorax ride the breathing.
 */
export function Setae({
  count,
  seed,
  cx,
  cy,
  cz,
  rx,
  ry,
  rz,
  capLo,
  capHi,
  sweep,
  length,
  variance = 0.6,
  geometry,
  material,
}: SetaeProps) {
  const mesh = useMemo(() => {
    const m = new THREE.InstancedMesh(geometry, material, count);
    m.castShadow = true;
    // the instances are baked here, so the whole part's bounds are the part's bounds
    m.frustumCulled = false;

    const rnd = lcg(seed);
    const pos = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const quat = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const mat4 = new THREE.Matrix4();

    for (let i = 0; i < count; i += 1) {
      const c = capLo + (capHi - capLo) * rnd();
      const st = Math.sqrt(Math.max(0, 1 - c * c));
      const ph = rnd() * Math.PI * 2;
      const nx = st * Math.cos(ph);
      const ny = c;
      const nz = st * Math.sin(ph);

      pos.set(cx + nx * rx, cy + ny * ry, cz + nz * rz);
      // surface normal of the ellipsoid, then leaned towards the tail like real setae
      dir.set(nx / rx, ny / ry, nz / rz).normalize();
      dir.z -= sweep;
      dir.normalize();

      quat.setFromUnitVectors(up, dir);
      scale.set(1, length * (1 - variance + rnd() * variance), 1);
      mat4.compose(pos, quat, scale);
      m.setMatrixAt(i, mat4);
    }
    m.instanceMatrix.needsUpdate = true;
    m.computeBoundingSphere();
    return m;
  }, [count, seed, cx, cy, cz, rx, ry, rz, capLo, capHi, sweep, length, variance, geometry, material]);

  return <primitive object={mesh} />;
}

/* --------------------------------------------------------------------- leg */

/** One leg: hip (yaw/roll) -> femur -> knee -> tibia -> tarsus, all driven by the pose. */
function Leg({
  pose,
  i,
  mat,
  assets,
}: {
  pose: Pose;
  i: number;
  mat: THREE.MeshPhysicalMaterial;
  assets: FlyAssets;
}) {
  const row = LEG_ROWS[legRow(i)];
  const side = legSide(i);
  const hip = useRef<THREE.Group>(null);
  const femur = useRef<THREE.Group>(null);
  const knee = useRef<THREE.Group>(null);

  useFrame(() => {
    const l = pose.legs[i];
    if (hip.current) hip.current.rotation.set(l.hipRoll, l.hipYaw * side, 0);
    if (femur.current) femur.current.rotation.x = l.femur;
    if (knee.current) knee.current.rotation.x = l.knee;
  });

  return (
    <group position={[row.x * side, row.y, row.z]}>
      <group ref={hip}>
        <mesh castShadow material={mat} position={[0, -0.02, 0]}>
          <sphereGeometry args={[0.045, 12, 10]} />
        </mesh>
        <group ref={femur}>
          <mesh castShadow material={mat} position={[0, -LEG_LEN.femur / 2, 0]}>
            <capsuleGeometry args={[0.028, LEG_LEN.femur - 0.056, 4, 10]} />
          </mesh>
          {/* setae ride the femur, so the walk cycle carries them */}
          <Setae
            count={11}
            seed={9001 + i * 131}
            cx={0}
            cy={-LEG_LEN.femur * 0.5}
            cz={0}
            rx={0.028}
            ry={LEG_LEN.femur * 0.45}
            rz={0.028}
            capLo={-1}
            capHi={1}
            sweep={0.55}
            length={0.026}
            geometry={assets.setaeGeo}
            material={assets.mats.setae}
          />
          <group ref={knee} position={[0, -LEG_LEN.femur, 0]}>
            <mesh castShadow material={mat} position={[0, -LEG_LEN.tibia / 2, 0]}>
              <capsuleGeometry args={[0.022, LEG_LEN.tibia - 0.044, 4, 10]} />
            </mesh>
            <group position={[0, -LEG_LEN.tibia, 0]}>
              <mesh castShadow material={mat} position={[0, -LEG_LEN.tarsus / 2, 0]}>
                <capsuleGeometry args={[0.016, LEG_LEN.tarsus - 0.032, 3, 8]} />
              </mesh>
              <mesh castShadow material={mat} position={[0, -LEG_LEN.tarsus, 0]}>
                <sphereGeometry args={[0.02, 10, 8]} />
              </mesh>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}

/* --------------------------------------------------------------------- fly */

export function Fly({ drive, pose, live, slotOf }: FlyProps) {
  const assets = useMemo(makeAssets, []);
  const { mats } = assets;
  useEffect(() => () => disposeAssets(assets), [assets]);

  const root = useRef<THREE.Group>(null);
  const body = useRef<THREE.Group>(null);
  const head = useRef<THREE.Group>(null);
  const abd = useRef<(THREE.Group | null)[]>([]);
  /** inner shell group per abdomen segment: the only children the jitter loop touches */
  const abdShell = useRef<(THREE.Group | null)[]>([]);
  const antL = useRef<THREE.Group>(null);
  const antR = useRef<THREE.Group>(null);
  const prob = useRef<THREE.Group>(null);
  const wingL = useRef<THREE.Group>(null);
  const wingR = useRef<THREE.Group>(null);
  const halL = useRef<THREE.Group>(null);
  const halR = useRef<THREE.Group>(null);

  // preallocated scratch so the frame loop never allocates
  const tintBase = useMemo(() => new THREE.Color(), []);
  const tintWarm = useMemo(() => new THREE.Color('#ffd479'), []);
  const tintCool = useMemo(() => new THREE.Color('#38bdf8'), []);

  useFrame(() => {
    const r = root.current;
    if (r) {
      r.position.set(0, pose.rootY, pose.rootZ);
      r.rotation.set(0, pose.rootYaw, 0);
    }
    const b = body.current;
    if (b) {
      b.position.y = pose.bodyY;
      b.rotation.set(pose.bodyPitch, pose.bodyYaw, pose.bodyRoll);
    }
    const h = head.current;
    if (h) h.rotation.set(pose.headPitch, pose.headYaw, 0);

    for (let s = 0; s < 4; s += 1) {
      const g = abd.current[s];
      if (!g) continue;
      g.rotation.x = pose.abdPitch[s];
      const k = pose.abdScale[s];
      g.scale.set(1 + (k - 1) * 0.35, k, 1 + (k - 1) * 0.35);
    }

    if (antL.current) antL.current.rotation.set(pose.antL.pitch, pose.antL.yaw, 0);
    if (antR.current) antR.current.rotation.set(pose.antR.pitch, -pose.antR.yaw, 0);

    if (prob.current) {
      const e = pose.proboscis;
      const on = e > 0.004;
      const sc = on ? 0.42 + 0.58 * e : 0.001;
      prob.current.scale.set(sc, on ? 0.5 + 0.5 * e : 0.001, sc);
      prob.current.position.set(0, -0.1 - 0.34 * e, 0.1 + 0.26 * e);
      prob.current.rotation.set(0.5 + 0.55 * (1 - e), pose.proboscisWiggle, 0);
      prob.current.visible = on;
    }

    if (wingL.current) wingL.current.rotation.set(pose.wingL.flap, 0, pose.wingL.sweep);
    if (wingR.current) wingR.current.rotation.set(pose.wingR.flap, 0, -pose.wingR.sweep);
    if (halL.current) halL.current.rotation.z = pose.haltere;
    if (halR.current) halR.current.rotation.z = -pose.haltere;

    // --- connectome driven glow -------------------------------------------
    // `live.vitality` is normalized against the measured p95 of |activity| (see regions.ts)
    // and is exactly 0 on an all-zero frame, so an unlit network stays unlit.
    const rg = pose.reactionGlow;
    const hit = 1.6 * rg;
    const floor = BASE_GLOW + 0.34 * clamp01(live.vitality);
    const tint = rg > 0.001 ? (pose.reactionTint >= 0 ? tintWarm : tintCool) : null;

    const paint = (
      kind: ChannelKind,
      m: THREE.MeshPhysicalMaterial,
      v: number,
      gain: number,
    ) => {
      m.emissiveIntensity = floor + gain * clamp01(v) + hit;
      tintBase.set(REGION_COLOR[kind]);
      if (tint) tintBase.lerp(tint, 0.6 * rg);
      m.emissive.copy(tintBase);
    };

    paint('head', mats.head, drive[slotOf('head', 0)] * 0.6 + drive[slotOf('head', 1)] * 0.4, 1.7);
    paint('thorax', mats.thorax, drive[slotOf('thorax', 1)] * 0.7 + drive[slotOf('thorax', 2)] * 0.3, 1.5);

    // each abdomen segment reads its OWN channel, and jitters its own scale with it
    for (let s = 0; s < 4; s += 1) {
      const dd = drive[slotOf('abdomen', s)] || 0;
      paint('abdomen', mats.abdomen[s], dd, 1.6);
      const g = abd.current[s];
      const shell = abdShell.current[s];
      const j = 0.012 * dd * Math.sin(pose.haltere * 0.11 + s * 1.7);
      // only the inner shell group is scaled, so the setae and the tergite ridge parented
      // beside it (and the group's own pose scale below) are not stamped to 1 every frame
      if (shell) {
        shell.position.y = j;
        shell.scale.setScalar(1 + 0.02 * dd * Math.cos(s * 2.1 + pose.haltere * 0.07));
      }
      if (!g) continue;
      const k = pose.abdScale[s];
      g.scale.set(1 + (k - 1) * 0.35 + j, k + j * 0.5, 1 + (k - 1) * 0.35 + j);
    }

    // each leg reads its own channel: a leg whose channel is active is the bright one
    for (let i = 0; i < 6; i += 1) {
      paint('legs', mats.legs[i], drive[slotOf('legs', i)] || 0, 1.9);
    }

    // wings take the mean of their two channels plus their own flap speed
    const wingGlow = ((drive[slotOf('wings', 0)] || 0) + (drive[slotOf('wings', 1)] || 0)) * 0.5;
    paint('wings', mats.wings, wingGlow, 1.3);

    // wings hold a little more open when the fly is doing anything but resting
    const openBias = live.behavior === 'idle' ? 0 : 0.07;
    mats.wings.opacity = clamp01(
      pose.wingOpacity + openBias + 0.24 * wingGlow + 0.14 * Math.min(1, Math.abs(pose.wingL.flap)),
    );

    // eyes take only a little of the head channel: they are specular, not a light source
    const headGlow = drive[slotOf('head', 0)] || 0;
    const eyeGlow = 0.1 + 0.45 * headGlow + 0.7 * hit * (pose.reactionTint >= 0 ? 1 : 0.35);
    mats.eyes.emissiveIntensity = eyeGlow;
    mats.gloss.emissiveIntensity = eyeGlow;
  });

  const abdSegments = [
    { pos: [0, 0.02, -0.3] as const, r: 0.2 },
    { pos: [0, 0.0, -0.44] as const, r: 0.185 },
    { pos: [0, -0.02, -0.56] as const, r: 0.16 },
    { pos: [0, -0.045, -0.66] as const, r: 0.12 },
  ] as const;

  return (
    <group ref={root}>
      <group ref={body}>
        {/* thorax: one ellipsoid, the mechanical centre of mass */}
        <mesh
          castShadow
          receiveShadow
          material={mats.thorax}
          position={[0, 0.02, 0.06]}
          scale={[0.3, 0.26, 0.38]}
        >
          <sphereGeometry args={[1, 32, 24]} />
        </mesh>
        {/* scutellum bump */}
        <mesh castShadow material={mats.thorax} position={[0, 0.1, -0.2]} scale={[0.19, 0.13, 0.16]}>
          <sphereGeometry args={[1, 24, 16]} />
        </mesh>
        {/* dorsal setae: densest on the thorax, and they cross the silhouette */}
        <Setae
          count={88}
          seed={2311}
          cx={0}
          cy={0.02}
          cz={0.06}
          rx={0.3}
          ry={0.26}
          rz={0.38}
          capLo={0.18}
          capHi={1}
          sweep={0.7}
          length={0.046}
          geometry={assets.setaeGeo}
          material={mats.setae}
        />

        {/* head */}
        <group ref={head} position={[0, 0.06, 0.38]}>
          <mesh castShadow material={mats.head} scale={[0.21, 0.19, 0.19]}>
            <sphereGeometry args={[1, 30, 22]} />
          </mesh>
          <Setae
            count={44}
            seed={7717}
            cx={0}
            cy={0}
            cz={0}
            rx={0.21}
            ry={0.19}
            rz={0.19}
            capLo={0.05}
            capHi={1}
            sweep={0.55}
            length={0.034}
            geometry={assets.setaeGeo}
            material={mats.setae}
          />
          {/* compound eyes: the biggest single visual cue that this is a fly */}
          <mesh
            material={mats.eyes}
            position={[0.12, 0.02, 0.05]}
            rotation={[0, 0.34, -0.18]}
            scale={[0.1, 0.13, 0.12]}
          >
            <sphereGeometry args={[1, 32, 24]} />
          </mesh>
          <mesh
            material={mats.eyes}
            position={[-0.12, 0.02, 0.05]}
            rotation={[0, -0.34, 0.18]}
            scale={[0.1, 0.13, 0.12]}
          >
            <sphereGeometry args={[1, 32, 24]} />
          </mesh>
          {/* ocelli */}
          <mesh material={mats.gloss} position={[0, 0.11, 0.11]} scale={[0.035, 0.035, 0.035]}>
            <sphereGeometry args={[1, 14, 12]} />
          </mesh>

          {/* antennae: 3 segments, tip brighter, twitching constantly */}
          <group ref={antL} position={[0.09, 0.09, 0.1]}>
            <mesh material={mats.head} position={[0.03, 0.09, 0.03]} rotation={[0, 0, -0.5]}>
              <capsuleGeometry args={[0.014, 0.14, 4, 8]} />
            </mesh>
            <mesh material={mats.head} position={[0.11, 0.17, 0.05]} rotation={[0, 0, -1.2]}>
              <capsuleGeometry args={[0.011, 0.12, 4, 8]} />
            </mesh>
            <mesh material={mats.gloss} position={[0.19, 0.19, 0.06]}>
              <sphereGeometry args={[0.034, 14, 12]} />
            </mesh>
          </group>
          <group ref={antR} position={[-0.09, 0.09, 0.1]}>
            <mesh material={mats.head} position={[-0.03, 0.09, 0.03]} rotation={[0, 0, 0.5]}>
              <capsuleGeometry args={[0.014, 0.14, 4, 8]} />
            </mesh>
            <mesh material={mats.head} position={[-0.11, 0.17, 0.05]} rotation={[0, 0, 1.2]}>
              <capsuleGeometry args={[0.011, 0.12, 4, 8]} />
            </mesh>
            <mesh material={mats.gloss} position={[-0.19, 0.19, 0.06]}>
              <sphereGeometry args={[0.034, 14, 12]} />
            </mesh>
          </group>

          {/* proboscis: drumstick of coils, hidden entirely when retracted */}
          <group ref={prob} position={[0, -0.1, 0.1]}>
            <mesh castShadow material={mats.head} position={[0, -0.08, 0.02]}>
              <capsuleGeometry args={[0.032, 0.16, 4, 12]} />
            </mesh>
            <mesh material={mats.gloss} position={[0, -0.2, 0.05]} scale={[0.06, 0.05, 0.06]}>
              <sphereGeometry args={[1, 16, 12]} />
            </mesh>
          </group>
        </group>

        {/* abdomen: 4 chained segments, each with its own channel and its own emissive */}
        {abdSegments.map((s, i) => (
          <group
            key={i}
            ref={(el) => {
              abd.current[i] = el;
            }}
            position={s.pos as unknown as [number, number, number]}
          >
            <group
              ref={(el) => {
                abdShell.current[i] = el;
              }}
            >
              <mesh
                castShadow
                receiveShadow
                material={mats.abdomen[i]}
                scale={[s.r, s.r * 0.92, s.r * 1.15]}
              >
                <sphereGeometry args={[1, 28, 20]} />
              </mesh>
              <mesh castShadow material={mats.abdomen[i]} scale={[s.r * 0.82, s.r * 0.7, s.r * 0.7]}>
                <sphereGeometry args={[1, 20, 16]} />
              </mesh>
            </group>
            {/* the tergite grooves themselves are drawn by the shader band term; these
                are the setae that make each plate edge read as an edge */}
            <Setae
              count={15}
              seed={4400 + i * 977}
              cx={0}
              cy={0}
              cz={0}
              rx={s.r}
              ry={s.r * 0.92}
              rz={s.r * 1.15}
              capLo={0.05}
              capHi={1}
              sweep={0.7}
              length={0.03}
              geometry={assets.setaeGeo}
              material={mats.setae}
            />
          </group>
        ))}

        {/* all six legs, each its own hip/femur/knee chain and its own material */}
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <Leg key={i} i={i} pose={pose} mat={mats.legs[i]} assets={assets} />
        ))}

        {/* wings: real outlines with a veined translucent membrane, pivot at the shoulder */}
        <group ref={wingL} position={[0.14, 0.2, 0.02]}>
          <mesh
            material={mats.wings}
            geometry={assets.wingGeo}
            position={[0.02, 0, -0.02]}
            rotation={[Math.PI / 2, 0, -0.16]}
          />
        </group>
        <group ref={wingR} position={[-0.14, 0.2, 0.02]}>
          {/* mirrored through negative x scale; the membrane is DoubleSide so the flipped
              winding is harmless and both faces light correctly */}
          <mesh
            material={mats.wings}
            geometry={assets.wingGeo}
            position={[-0.02, 0, -0.02]}
            scale={[-1, 1, 1]}
            rotation={[Math.PI / 2, 0, 0.16]}
          />
        </group>

        {/* halteres: tiny gyroscopes behind the wings, and they genuinely beat */}
        <group ref={halL} position={[0.07, 0.06, -0.14]}>
          <mesh material={mats.wings} position={[0.05, 0, 0]}>
            <capsuleGeometry args={[0.008, 0.07, 3, 6]} />
          </mesh>
        </group>
        <group ref={halR} position={[-0.07, 0.06, -0.14]}>
          <mesh material={mats.wings} position={[-0.05, 0, 0]}>
            <capsuleGeometry args={[0.008, 0.07, 3, 6]} />
          </mesh>
        </group>
      </group>
    </group>
  );
}
