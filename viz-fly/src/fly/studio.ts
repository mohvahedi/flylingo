/**
 * The studio the fly is shot in.
 *
 * Two separate things live here because they are the two halves of "expensive looking":
 *
 *  1. The punctual rig (RIG below): one hard warm key from the upper right, one cool rim
 *     off axis behind, a soft front-left fill, a low cool bounce and a very shallow
 *     ambient. The key is the only shadow caster and it sits low, roughly 21 degrees above
 *     the horizon, which is what stretches the cast shadow into the long dramatic wedge the
 *     reference has.
 *
 *     The rim is deliberately NOT dead centre. A rim directly behind the subject puts the
 *     brightest pixel of the frame on the halo behind the head, which destroys separation
 *     instead of creating it, and clips the wings to flat white. Measured on the dead centre
 *     version, the top of the range sat pinned at luma 252. It now sits roughly 46 degrees
 *     off the camera axis to the back left, so the background behind the head stays black
 *     and the bright edge lands on the fly's flank and wing tips.
 *
 *  2. A procedural HDR environment (StudioEnvironment below). Punctual lights alone give
 *     a dielectric nothing to reflect, and bare specular from a directional light reads
 *     as plastic. So the fly is lit by, and reflects, a baked room: a near black shell with
 *     three emissive panels placed exactly where the key, the fill and the rim are. It is
 *     built in code from planes and PMREM filtered once at mount, so there is still no
 *     .hdr file to download, and it is applied at low intensity so it adds highlight and
 *     reflection without lifting the background out of black.
 */
import { useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';

type Vec3 = [number, number, number];

/** The punctual light rig. Position is in world units, the fly root sits at y ~ 0.5. */
export const RIG: {
  key: { position: Vec3; intensity: number; color: string };
  rim: { position: Vec3; intensity: number; color: string };
  fill: { position: Vec3; intensity: number; color: string };
  bounce: { position: Vec3; intensity: number; color: string };
  ambient: { intensity: number; color: string };
} = {
  // hard, warm, upper right, about 21 degrees above the horizon. The only shadow caster in
  // the scene, and low enough that the wedge it throws runs long and to the left.
  key: { position: [5.6, 2.35, 2.15], intensity: 3.0, color: '#ffd7a0' },
  // cool, off axis to the back left: about 46 degrees from the camera axis, never dead
  // centre, so the halo behind the head stays black and the edge lands on the flank
  rim: { position: [-4.2, 2.05, -1.7], intensity: 1.05, color: '#7fb2ff' },
  // soft key from the front left. Without it the face and the proboscis get no directional
  // light at all and read as flat blue grey mush.
  fill: { position: [-3.1, 1.55, 2.7], intensity: 0.6, color: '#f2dcc2' },
  // low and cool from the lower left: catches the belly and the underside of the wings
  bounce: { position: [-2.6, -0.5, 1.4], intensity: 0.4, color: '#3f6f9f' },
  // shallow on purpose. Anything brighter flattens the shadow side into grey.
  ambient: { intensity: 0.1, color: '#4f6f8f' },
};

/** Half extent of the key light's shadow frustum. The fly spans about 0.9 units. */
export const SHADOW_EXTENT = 2.4;

/* --------------------------------------------------------------- environment */

/** An emissive panel. Colour components above 1 are deliberate: this is HDR. */
function panel(r: number, g: number, b: number, w: number, h: number, at: Vec3): THREE.Mesh {
  const material = new THREE.MeshBasicMaterial();
  material.color.setRGB(r, g, b);
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), material);
  mesh.position.set(at[0], at[1], at[2]);
  mesh.lookAt(0, 0, 0);
  return mesh;
}

function buildStudio(): THREE.Scene {
  const scene = new THREE.Scene();

  // a near black room, so the only light in the reflections is the light that is really there
  const shellMat = new THREE.MeshBasicMaterial({ side: THREE.BackSide });
  shellMat.color.setRGB(0.05, 0.062, 0.078);
  const shell = new THREE.Mesh(new THREE.BoxGeometry(26, 26, 26), shellMat);
  scene.add(shell);

  // key panel, upper right, aligned with RIG.key
  scene.add(panel(5.5, 3.4, 1.6, 7, 4.5, [7.5, 8, 4]));
  // fill panel, lower left, broad and dim
  scene.add(panel(0.5, 1.0, 2.0, 9, 6, [-8, 1.2, 3]));
  // rim strip: off axis to the back left, matching RIG.rim. Large and soft on purpose, so
  // the rim is a broad grazing highlight along the flank rather than a punctual hotspot
  // sitting in the middle of the frame behind the head.
  scene.add(panel(0.9, 1.5, 3.2, 14, 3.4, [-9, 4.6, -6]));

  return scene;
}

function disposeScene(scene: THREE.Scene) {
  scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const m = mesh.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(m)) m.forEach((x) => x.dispose());
    else if (m) m.dispose();
  });
}

/**
 * Bakes the studio room into a PMREM environment and hangs it on the scene. Applied at low
 * intensity: high enough that the clearcoat has something to reflect, low enough that the
 * background stays black and the shadow side stays dark.
 *
 * Fails soft. An environment is a luxury; if a driver cannot build one the fly still
 * renders from the punctual rig, it just loses the wet reflections.
 */
export function StudioEnvironment({ intensity = 0.32 }: { intensity?: number }) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);

  useEffect(() => {
    let pmrem: THREE.PMREMGenerator | null = null;
    let target: THREE.WebGLRenderTarget | null = null;
    let studio: THREE.Scene | null = null;
    try {
      pmrem = new THREE.PMREMGenerator(gl);
      studio = buildStudio();
      target = pmrem.fromScene(studio, 0);
      scene.environment = target.texture;
      scene.environmentIntensity = intensity;
    } catch (err) {
      console.warn('fly: studio environment unavailable, falling back to lights only', err);
    }
    return () => {
      scene.environment = null;
      scene.environmentIntensity = 1;
      if (studio) disposeScene(studio);
      if (target) target.dispose();
      if (pmrem) pmrem.dispose();
    };
  }, [gl, scene, intensity]);

  return null;
}
