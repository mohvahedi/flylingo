/**
 * The stage: a bounded slab the fly stands on.
 *
 * The floor used to be a 7 unit radius disc whose albedo faded to zero at the rim, so it had
 * no edge, no silhouette and no value where the fly's shadow landed. Measured with the
 * ?noshadow A/B, the shadow was the correct darkness but the surface it fell on was not a
 * surface: the eye had nothing to compare it against and the fly read as floating.
 *
 * This is the replacement: a rounded rectangle with real thickness, its top face on y=0 and
 * a 0.16 skirt below it. It is deliberately small enough that all four edges stay inside the
 * hero framing, so the slab reads as an object sitting in a near black void instead of as an
 * infinite plane that happens to be dark.
 *
 * The dimensions are shared: `FlyStage` builds the geometry from them and `groundMaterial`
 * takes the half extents as uniforms, so the gradient in the shader and the silhouette in
 * the scene can never disagree.
 */
import * as THREE from 'three';

export const SLAB = {
  /** half extent along world x. The camera sits at x=1.85, so the near edge is off frame. */
  hx: 3.35,
  /** half extent along world z. The far edge lands around y=350 in the hero framing. */
  hz: 2.95,
  /** slab thickness. Read as a lit skirt under the far lip, which is what makes it a solid. */
  thickness: 0.16,
  /** corner radius of the rounded rectangle. */
  radius: 0.62,
  /**
   * The contact pool, in the slab's own object space (x, y) = (world x, -world z).
   * An ellipse under the body: the fly's feet span about +-0.6, so this clears them and
   * eases out over a further 0.4, which is the soft edge the reference shows.
   */
  poolX: 0,
  poolY: 0,
  poolRx: 1.12,
  poolRz: 1.5,
} as const;

/** The rounded rectangle outline, centred on the origin. */
function roundedRect(w: number, h: number, r: number): THREE.Shape {
  const s = new THREE.Shape();
  const x = -w / 2;
  const y = -h / 2;
  s.moveTo(x + r, y);
  s.lineTo(x + w - r, y);
  s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + h - r);
  s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  s.lineTo(x + r, y + h);
  s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r);
  s.quadraticCurveTo(x, y, x + r, y);
  return s;
}

/**
 * Extruded rounded rectangle, positioned so that the TOP face lies exactly on y=0 after the
 * mesh's own -90 degree x rotation: object +z becomes world +y, so the extrusion is shifted
 * down by one thickness. The fly's feet, the rings and the contact patches all assume y=0 is
 * the walkable surface, so this must not move.
 */
export function createSlabGeometry(): THREE.ExtrudeGeometry {
  const geo = new THREE.ExtrudeGeometry(
    roundedRect(SLAB.hx * 2, SLAB.hz * 2, SLAB.radius),
    { depth: SLAB.thickness, bevelEnabled: false, curveSegments: 26 },
  );
  geo.translate(0, 0, -SLAB.thickness);
  return geo;
}
