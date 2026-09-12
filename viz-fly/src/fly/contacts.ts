/**
 * Ground contacts: what puts the fly ON the floor.
 *
 * The real cast shadow (the key light's shadow map, onto the ground plane) gives the long
 * dramatic wedge, and drei's ContactShadows adds a depth based pool. Neither of them
 * produces a separate patch under each tarsus, and a fly whose feet touch nothing reads as a
 * sticker pasted on a gradient. This adds the two missing shapes:
 *
 *   - one small soft patch under each of the six tarsi, tracked every frame from the same
 *     rig nodes the animator drives, so the patches walk with the feet. A lifted foot gets a
 *     smaller, fainter patch, which is the cue that reads as contact rather than as a decal.
 *   - one broader elliptical ambient occlusion pool under the thorax, following the body.
 *
 * Implementation: a single InstancedMesh of seven quads with a radial falloff shader, so the
 * whole thing is one extra transparent draw call and no textures. Per instance opacity
 * travels in instanceColor.r, which three declares for us under USE_INSTANCING_COLOR, so
 * there is no custom attribute to maintain.
 *
 * Allocation: created once, and the update path writes into preallocated vectors and
 * matrices. Seven instance matrices a frame is nothing.
 */
import * as THREE from 'three';
import type { Rig } from './flyRig';

const FOOT_COUNT = 6;
/** six tarsi plus one thorax pool */
const INSTANCES = FOOT_COUNT + 1;
/** scene units. The fly spans 1.75, a tarsus footprint is a few hundredths of that. */
const CONTACT_RADIUS = 0.075;
const POOL_RADIUS = 0.42;
/** heights above the ground of each layer, so nothing z fights */
const CONTACT_Y = 0.01;
const POOL_Y = 0.008;

const VERT = /* glsl */ `
varying vec2 vUv;
varying float vA;
void main() {
	vUv = uv;
	#ifdef USE_INSTANCING_COLOR
	vA = instanceColor.r;
	#else
	vA = 1.0;
	#endif
	vec4 local = vec4( position, 1.0 );
	#ifdef USE_INSTANCING
	local = instanceMatrix * local;
	#endif
	gl_Position = projectionMatrix * modelViewMatrix * local;
}
`;

const FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uStrength;
varying vec2 vUv;
varying float vA;
void main() {
	float d = length( vUv - 0.5 ) * 2.0;
	float a = 1.0 - smoothstep( 0.0, 1.0, d );
	a *= a;
	gl_FragColor = vec4( uColor, a * vA * uStrength );
}
`;

export type Contacts = {
  mesh: THREE.InstancedMesh;
  /** call once a frame; reads the rig's own world positions, allocates nothing */
  update: (rig: Rig) => void;
  dispose: () => void;
};

export function createContacts(): Contacts {
  const geo = new THREE.PlaneGeometry(1, 1);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color('#01060c') },
      uStrength: { value: 1 },
    },
    vertexShader: VERT,
    fragmentShader: FRAG,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    side: THREE.FrontSide,
  });
  material.name = 'ground-contacts';

  const mesh = new THREE.InstancedMesh(geo, material, INSTANCES);
  mesh.name = 'ground-contacts';
  mesh.frustumCulled = false;
  mesh.renderOrder = 3;
  mesh.castShadow = false;
  mesh.receiveShadow = false;

  // allocates instanceColor, which is where the per instance opacity lives
  const white = new THREE.Color('#ffffff');
  for (let i = 0; i < INSTANCES; i += 1) mesh.setColorAt(i, white);
  const colors = mesh.instanceColor as THREE.InstancedBufferAttribute;

  const flat = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));
  const pos = new THREE.Vector3();
  const scl = new THREE.Vector3();
  const matrix = new THREE.Matrix4();
  /** each leg's own idle height, so a lifted foot can be told from the ground contact */
  const restY = new Float32Array(FOOT_COUNT).fill(-1);

  const update = (rig: Rig) => {
    for (let i = 0; i < FOOT_COUNT; i += 1) {
      const chain = rig.legs[i];
      if (!chain || chain.seg.length === 0) continue;
      const tip = chain.seg[chain.seg.length - 1].node;
      tip.updateWorldMatrix(true, false);
      tip.getWorldPosition(pos);
      const y = pos.y;
      if (restY[i] < 0) restY[i] = y;
      // lift above the leg's own idle height, never below it: a crouch does not lighten a
      // contact patch, a raised foot does
      const lift = Math.max(0, y - restY[i]);
      const k = 1 / (1 + lift * 9);
      const r = CONTACT_RADIUS * (0.6 + 0.4 * k);
      scl.set(r, r, 1);
      pos.y = CONTACT_Y;
      matrix.compose(pos, flat, scl);
      mesh.setMatrixAt(i, matrix);
      colors.setX(i, 0.80 * k);
    }

    // the occlusion pool under the thorax, longer along the body than across it
    rig.root.node.updateWorldMatrix(true, false);
    rig.root.node.getWorldPosition(pos);
    scl.set(POOL_RADIUS * 0.8, POOL_RADIUS * 1.35, 1);
    pos.y = POOL_Y;
    matrix.compose(pos, flat, scl);
    mesh.setMatrixAt(FOOT_COUNT, matrix);
    colors.setX(FOOT_COUNT, 0.55);

    mesh.instanceMatrix.needsUpdate = true;
    colors.needsUpdate = true;
  };

  const dispose = () => {
    geo.dispose();
    material.dispose();
    mesh.dispose();
  };

  return { mesh, update, dispose };
}
