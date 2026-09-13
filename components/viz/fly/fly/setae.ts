/**
 * Setae: the fuzz that sells the scale.
 *
 * The supplied mesh has no bristle geometry, and at this scale a bare cuticle silhouette
 * reads as a toy. So this scatters short instanced cones over the parts of the asset that
 * really carry them: the dorsal thorax and its scutellum, the thorax/neck plates, the head
 * and the legs.
 *
 * Placement is read off the asset's own geometry rather than guessed. Every bristle stands
 * on an actual surface vertex, along that vertex's own normal, and is parented to the mesh
 * it grew out of. That parentage is the whole trick: the fuzz on a femur rides the walk
 * cycle and the fuzz on the thorax rides the breathing, with no per frame work at all.
 *
 * Cost: one InstancedMesh per source mesh (the legs alone are 48 primitives), about 2000
 * instances in total at 8 triangles each. They share one unit cone geometry and one
 * material, nothing here runs per frame, and they do not cast shadows. Bristle length is
 * requested in scene units and divided by each mesh's own accumulated scale, so the fuzz
 * stays the same physical size whatever units the asset happens to be authored in.
 */
import * as THREE from 'three';
import { setaeMaterial } from './materials';

/** Bristles to aim for per material, split across that material's meshes by vertex count. */
const BUDGET: Record<string, number> = {
  FLYPATA: 1300, // the six legs, 48 primitives, 3042 vertices
  FLYTCORT: 420, // dorsal thorax, which carries the scutellum at its hind margin
  FLYTCUE: 280, // thorax and neck plates
  FLYTCAB: 140, // head capsule
};

/** Bristle length in scene units. The fly spans 1.75, so this is a little under 2 percent. */
const BRISTLE_WORLD = 0.04;
/** Radius as a fraction of length: thin enough to read as a bristle, thick enough to survive. */
const BRISTLE_ASPECT = 0.11;
/** How far a bristle leans off the surface normal. */
const LEAN = 0.42;

export type Setae = {
  /** total instances created, i.e. the number the HUD reports */
  count: number;
  /** how many InstancedMeshes were added, i.e. the extra draw calls */
  meshes: number;
  /** the materials the bristles were distributed over, for the record */
  parts: string[];
};

/** Deterministic 0..1 hash, so a rebuild of the same asset produces the same fuzz. */
function hash01(n: number): number {
  const x = Math.sin(n * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

/**
 * Builds the bristles in place: the returned count is what was added, and the InstancedMesh
 * children hang off the model's own mesh nodes, so there is nothing to render here.
 */
export function buildSetae(root: THREE.Object3D, sceneScale: number): Setae {
  root.updateMatrixWorld(true);

  const buckets = new Map<string, THREE.Mesh[]>();
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const m of mats) {
      if (!m || !(m.name in BUDGET)) continue;
      const list = buckets.get(m.name) ?? [];
      list.push(mesh);
      buckets.set(m.name, list);
    }
  });

  // a unit cone with its base at the origin and its tip at +y, so one compose can place,
  // orient and size a bristle: y scale is the length, x/z scale is the thickness
  const cone = new THREE.ConeGeometry(1, 1, 4, 1, true);
  cone.translate(0, 0.5, 0);
  const material = setaeMaterial();

  const up = new THREE.Vector3(0, 1, 0);
  const p = new THREE.Vector3();
  const n = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const scl = new THREE.Vector3();
  const world = new THREE.Vector3();
  const matrix = new THREE.Matrix4();

  let count = 0;
  let meshes = 0;
  const parts: string[] = [];

  for (const [part, list] of buckets) {
    const budget = BUDGET[part];
    let total = 0;
    const verts = list.map((mesh) => {
      const c = mesh.geometry.getAttribute('position')?.count ?? 0;
      total += c;
      return c;
    });
    if (total === 0) continue;
    parts.push(part);

    list.forEach((mesh, mi) => {
      const geo = mesh.geometry as THREE.BufferGeometry;
      const attr = geo.getAttribute('position');
      if (!attr) return;
      const nrm = geo.getAttribute('normal');
      if (!geo.boundingSphere) geo.computeBoundingSphere();
      const centre = geo.boundingSphere?.center ?? null;

      const take = Math.min(attr.count, Math.max(4, Math.round((budget * verts[mi]) / total)));
      if (take < 2) return;
      // an even stride, offset per mesh, so two meshes of the same part do not sample the
      // same relative vertices
      const stride = Math.max(1, Math.floor(attr.count / take));
      const offset = Math.floor(hash01(mi + 1.7) * stride);

      const inst = new THREE.InstancedMesh(cone, material, take);
      inst.name = `setae-${part}-${mi}`;
      inst.castShadow = false;
      inst.receiveShadow = false;

      // this mesh's accumulated scale, so BRISTLE_WORLD can be asked for in scene units
      mesh.getWorldScale(world);
      const scale = sceneScale * world.x || sceneScale || 1;

      for (let i = 0; i < take; i += 1) {
        const k = (i * stride + offset) % attr.count;
        p.fromBufferAttribute(attr, k);
        if (nrm) n.fromBufferAttribute(nrm, k);
        else if (centre) n.copy(p).sub(centre);
        else n.copy(up);
        if (n.lengthSq() < 1e-12) n.copy(up);
        n.normalize();

        // spread the fuzz: a perfect shagreen reads as a texture, not as hair
        const j1 = hash01(k * 1.31 + mi * 3.17);
        const j2 = hash01(k * 2.73 + mi * 5.91);
        const j3 = hash01(k * 4.19 + mi * 7.43);
        n.x += (j1 - 0.5) * LEAN;
        n.y += (j2 - 0.5) * LEAN * 0.5;
        n.z += (j3 - 0.5) * LEAN;
        n.normalize();

        const len = (BRISTLE_WORLD / scale) * (0.6 + 0.9 * j2);
        q.setFromUnitVectors(up, n);
        scl.set(len * BRISTLE_ASPECT, len, len * BRISTLE_ASPECT);
        // lifted a little off the surface, so half the cone is not buried in the cuticle
        p.addScaledVector(n, len * 0.22);
        matrix.compose(p, q, scl);
        inst.setMatrixAt(i, matrix);
      }
      inst.instanceMatrix.needsUpdate = true;
      // the instance matrices are in the source mesh's local space, which for this asset is
      // thousands of units from the origin, so the bounding volume must be the instances'
      // own, not the unit cone's, or the whole patch gets culled
      inst.computeBoundingSphere();
      mesh.add(inst);
      count += take;
      meshes += 1;
    });
  }

  return { count, meshes, parts };
}
