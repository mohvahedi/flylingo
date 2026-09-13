/**
 * Loads the supplied phone asset and prepares its screen for a live texture.
 *
 * Two things about this asset were established by reading its binary rather than guessing,
 * and both matter:
 *
 * 1. The node chain already normalises it. `Sketchfab_model` applies the Z-up to Y-up
 *    rotation, then `.fbx` scales by 0.01, then `Cube` scales by 100 and rotates again. The
 *    net effect is that the handset is already standing upright and portrait in world space.
 *    An earlier version of this file applied its own "stand it up" rotation on top, which
 *    turned the phone a second time and threw the screen out of alignment with the body.
 *    Nothing here rotates the handset; only a small presentation yaw is applied, by the
 *    caller.
 *
 * 2. The screen mesh (`Cube_chroma_0`) cannot be textured directly. It is 47 triangles whose
 *    UVs span only 0.62 x 0.74 of the unit square, so a texture would land skewed and
 *    cropped. A clean plane is used instead, sized to the screen's measured rectangle, and
 *    it is parented to the same node the screen mesh sits in (`Cube`) so it inherits the
 *    identical transform chain. Parenting it to the scene root instead was the first bug:
 *    the plane then ignored the asset's scale and rotation and floated off the handset.
 *
 * Measured, in the mesh's own space:
 *   screen plane   x = 0.00554 (constant), Y span 0.06982, Z span 0.15482
 *   screen normal  +X, uniform across every vertex
 *   body           X 0.0114, Y 0.0756, Z 0.1603
 */

import { useGLTF } from '@react-three/drei';
import { useMemo } from 'react';
import * as THREE from 'three';

export const PHONE_URL = `${import.meta.env.BASE_URL}models/phone.glb`;

/** Screen rectangle in the phone's mesh space. */
export const SCREEN = {
  /** offset along the phone's local X, the screen normal; just proud of the body */
  x: 0.00554,
  /** along local Y */
  width: 0.06982,
  /** along local Z */
  height: 0.15482,
};

/**
 * Rotation of the screen plane inside the handset.
 *
 * A PlaneGeometry lies in its own XY facing +Z, with u along +X and v along +Y. The phone's
 * screen spans its local Y (width) and Z (height) and faces +X, so the plane's axes are
 * mapped onto the phone's:
 *
 *   plane +X (u, left to right) -> phone +Y   (screen width)
 *   plane +Y (v, bottom to top) -> phone +Z   (screen height, the portrait axis)
 *   plane +Z (the face)         -> phone +X   (out of the glass)
 *
 * With the handset standing, that puts the texture's top at world +Y and its left at world
 * -X, so the screen reads upright and is not mirrored.
 */
export function screenPlaneQuaternion(): THREE.Quaternion {
  const m = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(1, 0, 0),
  );
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

export type PhoneParts = {
  /** the handset, cloned so the drei cache is never mutated */
  root: THREE.Object3D;
  /** the node the screen mesh lives in: the plane is parented here to share its transform */
  anchor: THREE.Object3D | null;
  /** the original green screen mesh, hidden */
  chroma: THREE.Object3D | null;
  /** every mesh name found, for the on screen report */
  found: string[];
};

export function usePhone(): PhoneParts {
  const gltf = useGLTF(PHONE_URL);
  return useMemo(() => {
    const root = gltf.scene.clone(true);
    root.name = 'phone-model';
    const found: string[] = [];
    // collected into an array rather than assigned to a `let`: TypeScript's flow analysis
    // narrows a `let` that is only written inside a callback back to its initial null.
    const screenMeshes: THREE.Object3D[] = [];

    root.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!(mesh as unknown as { isMesh?: boolean }).isMesh) return;
      found.push(mesh.name);
      const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      const matName = (mat as THREE.Material | undefined)?.name ?? '';
      if (matName === 'chroma' || /chroma/i.test(mesh.name)) {
        mesh.visible = false;
        screenMeshes.push(mesh);
      }
    });
    const chroma: THREE.Object3D | null = screenMeshes[0] ?? null;

    // Give the handset a body that can actually be seen. The asset's own materials are pure
    // black (baseColor 0,0,0), which is invisible against a near-black stage: the first
    // renders read as a white card floating in a void because only the lit screen showed.
    // A real phone photographed on a dark set shows its edges, so the body is lifted to a
    // dark grey and given a little metalness for the rim light to catch. Materials are
    // cloned first, because the clone shares them with the drei cache.
    root.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!(mesh as unknown as { isMesh?: boolean }).isMesh) return;
      const current = mesh.material;
      if (!current || Array.isArray(current)) return;
      const mat = current.clone() as THREE.MeshStandardMaterial;
      const name = (current as THREE.Material).name;
      if (name === 'phone_body' || name === 'phone_back') {
        mat.color = new THREE.Color('#2b323c');
        mat.roughness = 0.36;
        mat.metalness = 0.22;
      } else if (name === 'Camera') {
        mat.color = new THREE.Color('#0f1319');
        mat.roughness = 0.25;
        mat.metalness = 0.5;
      }
      mat.needsUpdate = true;
      mesh.material = mat;

      // Cast the phone's shadow. Without this the handset has no shadow at all and reads as a
      // graphic pasted over a gradient: the floor had `receiveShadow` and the key light had
      // `castShadow` from the start, but the GLB's meshes never had `castShadow`, so there was
      // nothing for the floor to receive. The body casts; the glass does not, because a screen
      // is an emitter and a shadow from it would be wrong.
      mesh.castShadow = true;
      mesh.receiveShadow = false;
    });

    // Parent the replacement plane to the screen mesh's own parent, so it inherits every
    // transform the body does. Falling back to the mirror of the chroma node keeps this
    // working if a future export names things differently.
    const anchor = chroma?.parent ?? null;

    return { root, anchor, chroma, found };
  }, [gltf]);
}

/**
 * Map a pixel position on the 390x865 screen to a point in the SCREEN PLANE's own space.
 *
 * This deliberately works in the plane's local frame and then lets the caller apply the
 * plane's world matrix. The first version mapped pixels onto the phone's mesh axes instead
 * (u to local Y, v to local Z), which silently rotated the result: measured, the two rows of
 * cards came out separated along world Z and the columns along world Y, so the fly was sent
 * to the right coordinates for a screen that was turned on its side.
 *
 * A PlaneGeometry lies in its own XY plane facing +Z with u along +X and v along +Y, so:
 *   x = (u - 0.5) * screen width    -> horizontal, to the viewer's right
 *   y = (0.5 - v) * screen height   -> vertical; canvas y counts downward, hence the flip
 *   z = standoff                    -> out along the face, off the glass
 */
export function screenPointToPlaneLocal(
  px: number,
  py: number,
  screenW: number,
  screenH: number,
  standoff = 0,
): THREE.Vector3 {
  return new THREE.Vector3(
    (px / screenW - 0.5) * SCREEN.width,
    (0.5 - py / screenH) * SCREEN.height,
    standoff,
  );
}
