/**
 * RealFly: the hero path. Loads the supplied static mesh and animates it by driving its
 * named nodes through the shared Pose, with a per-region additive glow overlay that
 * tracks the connectome.
 *
 * Why an overlay and not a recolour: the asset's own PBR textures and specular maps are
 * what make it read as a photograph, and tinting them would trade that away for the sake
 * of an effect. The glow is therefore a second, additive pass over the same geometry: a
 * clone of the node graph whose meshes all carry a fresnel rim shader, one instance per
 * connectome region, blended in on top. When the connectome is inert every one of those
 * opacities is exactly zero, so the fly renders as a plain photograph with no glow at all,
 * and nothing amplifies a dead frame.
 *
 * The overlay is posed from its own copy of the rig. It is a clone, so driving the model's
 * nodes moves the model and not the shell; without a second rig the glow would hang in the
 * rest pose while the animal walked. applyPose is absolute (every node is reset from its
 * rest transform before the pose is applied), so running it twice a frame is exact and
 * cannot accumulate.
 *
 * Cost: the clone shares geometry and is never uploaded again, so it is 57 extra draw
 * calls against the same 6511 vertices, not a second mesh. Well inside the budget.
 *
 * Two things are added on top of the loaded mesh and neither of them touches its materials:
 * procedural setae, parented to the model's own mesh nodes so they ride the animation
 * (fly/setae.ts), and the per tarsus contact patches that ground it (fly/contacts.ts).
 */
import { useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { REGION_COLOR } from './regions';
import type { Behavior, Pose } from './pose';
import type { LiveReadout } from './animator';
import { createContacts } from './contacts';
import { buildSetae } from './setae';
import {
  applyPose,
  buildRig,
  createProbe,
  fitModel,
  MODEL_URL,
  ORIENTATION,
  regionOf,
  sampleProbe,
  type ProbeSample,
  type Region,
} from './flyRig';

const REGIONS: Region[] = ['head', 'thorax', 'abdomen', 'legs', 'wings'];

/** Glow tuning. The floor rides on live.vitality, which is pinned to 0 on an inert frame. */
const GLOW_FLOOR = 0.28;
// deliberately under 1: typical drive sits near 0.25, so the response stays in the middle
// of the range instead of pinning every region against the ceiling. Measured on the first
// pass, the leg region sat at the 0.92 ceiling in both the live and the brisker frames,
// which meant the busiest region stopped modulating at all.
const GLOW_GAIN = 0.95;
const GLOW_MAX = 0.95;
const GLOW_REACTION = 0.3;

const GOLD = new THREE.Color('#ffd9a2');

/**
 * The overlay shader: an additive fresnel rim.
 *
 * A flat additive wash over the whole surface was the first attempt and it was wrong. The
 * overlay colour is a cool instrument tint (cyan, violet, mint, ice blue) and at the
 * opacity needed for the connectome to read, a flat wash repaints the whole animal and
 * discards the asset's own shading, which is the entire reason it looks photographic. On
 * the first pass the overlay measured 0.46 on every region at once, i.e. saturated across
 * a fly that occupies about 60 pixels of thorax.
 *
 * This shader makes the contribution proportional to how edge on the surface is to the
 * camera instead. Face on, the middle of the thorax or an eye pointing at us, it is nearly
 * zero and the underlying PBR shading is untouched. At the silhouette it is bright, so an
 * active connectome reads as the animal glowing along its outline. It also puts the glow
 * exactly where it flatters the geometry: wings, legs, antennae and the ridged abdomen are
 * grazing angle almost everywhere, so thin parts carry the most light with no per part
 * special casing.
 *
 * Raw ShaderMaterial because no built in material has a fresnel term to modulate. It is
 * deliberately not tone mapped: under the bloom chain it lands in the linear HDR buffer
 * and OutputPass maps it once, with everything else.
 */
const GLOW_VERT = `
varying vec3 vN;
varying vec3 vV;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}
`;

const GLOW_FRAG = `
uniform vec3 uColor;
uniform float uOpacity;
varying vec3 vN;
varying vec3 vV;
void main() {
  float f = 1.0 - abs(dot(normalize(vN), normalize(vV)));
  float rim = pow(clamp(f, 0.0, 1.0), 2.4);
  gl_FragColor = vec4(uColor * (0.14 + 1.5 * rim), uOpacity);
}
`;

type ShellEntry = {
  kind: Region;
  mat: THREE.ShaderMaterial;
  base: THREE.Color;
};

type Shell = {
  group: THREE.Group;
  root: THREE.Object3D;
  entries: ShellEntry[];
};

/** Scratch for the probe, so the sampling path allocates nothing either. */
const probeBox = new THREE.Box3();
const probeSize = new THREE.Vector3();

/**
 * Builds the additive overlay: a clone of the node graph where every mesh carries the rim
 * shader, tinted by its anatomy region.
 *
 * The clone is coincident with the model, so the depth test against the surface it sits on
 * is a tie. Ties are resolved deterministically rather than by luck: the overlay draws
 * with a negative polygon offset, which pulls it one depth unit in front of its own surface
 * without moving a single vertex. So the overlay always wins against the surface it sits on
 * and is still correctly occluded by anything genuinely in front of it.
 */
function buildShell(root: THREE.Object3D): Shell {
  const entries: ShellEntry[] = [];
  const byKind = new Map<Region, THREE.ShaderMaterial>();
  for (const kind of REGIONS) {
    const base = new THREE.Color(REGION_COLOR[kind]);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: base.clone() },
        uOpacity: { value: 0 },
      },
      vertexShader: GLOW_VERT,
      fragmentShader: GLOW_FRAG,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    mat.name = `glow-${kind}`;
    byKind.set(kind, mat);
    entries.push({ kind, mat, base });
  }

  const cloned = root.clone(true);
  cloned.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!(mesh as unknown as { isMesh?: boolean }).isMesh) return;
    mesh.material = byKind.get(regionOf(o)) ?? byKind.get('thorax')!;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.renderOrder = 2;
  });

  const group = new THREE.Group();
  group.name = 'fly-glow-shell';
  group.add(cloned);
  return { group, root: cloned, entries };
}

function applyGlow(
  shell: Shell,
  pose: Pose,
  drive: Float32Array,
  slotOf: (kind: string, slot: number) => number,
  vitality: number,
): void {
  const tint = Math.max(0, Math.min(1, pose.reactionTint + 0.25 * pose.reactionGlow));
  const react = Math.max(0, Math.min(1, pose.reactionGlow));
  for (let i = 0; i < shell.entries.length; i += 1) {
    const entry = shell.entries[i];
    const slot = slotOf(entry.kind, 0);
    const d = slot >= 0 && slot < drive.length ? drive[slot] : 0;
    let o = GLOW_FLOOR * vitality + GLOW_GAIN * d + GLOW_REACTION * react;
    if (o > GLOW_MAX) o = GLOW_MAX;
    else if (o < 0) o = 0;
    entry.mat.uniforms.uOpacity.value = o;
    (entry.mat.uniforms.uColor.value as THREE.Color).copy(entry.base).lerp(GOLD, tint);
  }
}

export type RealFlyProps = {
  drive: Float32Array;
  pose: Pose;
  live: LiveReadout;
  slotOf: (kind: string, slot: number) => number;
  /** draw the per tarsus contact patches and the thorax occlusion pool. Default true. */
  contacts?: boolean;
};

export function RealFly({ drive, pose, live, slotOf, contacts = true }: RealFlyProps) {
  const gltf = useGLTF(MODEL_URL);

  // the drei cache hands back one shared scene, and this component mutates node
  // transforms every frame, so it works on a clone. The clone shares geometry, textures
  // and materials with the cache, so nothing is duplicated on the GPU.
  const root = useMemo(() => {
    const clone = gltf.scene.clone(true);
    clone.name = 'fly-model';
    return clone;
  }, [gltf]);

  const fit = useMemo(() => fitModel(root), [root]);
  const rig = useMemo(() => buildRig(root), [root]);
  const shell = useMemo(() => buildShell(root), [root]);
  // the overlay's own copy of the rig, so the glow follows the animation instead of
  // hanging in the rest pose. Silent: the rig shape is already reported by the model's.
  const shellRig = useMemo(() => buildRig(shell.root, { quiet: true }), [shell]);
  const probe = useMemo<ProbeSample>(() => createProbe(), []);

  // Bristles and ground contact patches, built once from the loaded asset. The setae parent
  // themselves to the model's own mesh nodes, so they ride the animation for free; the
  // contact patches live at scene level, because the ground does not move with the fly.
  const setae = useMemo(() => buildSetae(root, fit.scale), [root, fit]);
  const contact = useMemo(() => createContacts(), []);

  useEffect(() => {
    (window as unknown as { __flySetae?: number }).__flySetae = setae.count;
    if (import.meta.env.DEV) {
      console.info(
        `[fly] setae: ${setae.count} bristles across ${setae.meshes} meshes on ${setae.parts.join(', ')}`,
      );
    }
  }, [setae]);

  // The mesh is published so the headless check can read the seven instances back and
  // confirm the patches sit where the feet are, rather than take the code's word for it.
  useEffect(() => {
    (window as unknown as { __flyContacts?: unknown }).__flyContacts = contact.mesh;
    return () => contact.dispose();
  }, [contact]);

  // the real mesh is the shadow caster: this is the single biggest realism win over the
  // procedural path, because the cast shadow is now the actual silhouette of the animal
  useEffect(() => {
    root.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!(mesh as unknown as { isMesh?: boolean }).isMesh) return;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });
  }, [root]);

  useEffect(
    () => () => {
      // the shell materials are ours to own; geometry and textures belong to the cache
      for (const entry of shell.entries) entry.mat.dispose();
    },
    [shell],
  );

  const lastProbe = useRef(0);

  useFrame((state) => {
    applyPose(rig, pose);
    applyPose(shellRig, pose);
    applyGlow(shell, pose, drive, slotOf, live.vitality);

    // The contact patches read the rig after the pose has been applied, so the patches and
    // the feet can never disagree, and a foot that lifts during the walk cycle visibly
    // lightens and shrinks its patch.
    contact.mesh.visible = contacts;
    if (contacts) contact.update(rig);

    // kept current every frame so whatever the probe publishes is this frame's truth
    for (let i = 0; i < shell.entries.length; i += 1) {
      probe.glow[i] = shell.entries[i].mat.uniforms.uOpacity.value as number;
    }

    // Node-position witness. The scene must not be read back through WebGL, because
    // react-three-fiber leaves preserveDrawingBuffer off and an out-of-frame readPixels
    // returns zeros; this reports where the animated nodes actually ended up instead.
    const t = state.clock.elapsedTime;
    if (t - lastProbe.current < 0.1) return;
    lastProbe.current = t;
    const node = rig.root.node;
    if (node.parent) node.updateMatrixWorld(true);
    sampleProbe(rig, probe, t, live.behavior as Behavior, live.source);
    probeBox.setFromObject(root);
    probeBox.getSize(probeSize);
    probe.span[0] = probeSize.x;
    probe.span[1] = probeSize.y;
    probe.span[2] = probeSize.z;
    (window as unknown as { __flyProbe?: ProbeSample }).__flyProbe = probe;
  });

  return (
    <>
      <group position={fit.position} scale={fit.scale} name="fly-fit">
        <group rotation={ORIENTATION} name="fly-orient">
          <primitive object={root} />
          {/* inside the same orient group as the model: if the correction rotation ever has
              to change for a replacement asset, the overlay follows it automatically */}
          <primitive object={shell.group} />
        </group>
      </group>
      {/* a sibling of the fit group, not a child: these live on the ground, in scene space,
          at the world positions of the feet the rig just posed */}
      <primitive object={contact.mesh} />
    </>
  );
}

useGLTF.preload(MODEL_URL);
