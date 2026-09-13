/**
 * The stage: a bounded slab the fly stands on.
 *
 * The floor used to be a 7 unit radius disc whose albedo faded to zero at the rim, so it had no
 * edge, no silhouette and no value where the fly's shadow landed. Measured with the ?noshadow
 * A/B, the shadow was the correct darkness but the surface it fell on was not a surface: the eye
 * had nothing to compare it against and the fly read as floating.
 *
 * This is the replacement: a rounded rectangle with real thickness, its top face on y=0 and a
 * skirt below it, with a lit rim band (fly/materials.ts, GROUND_BODY) so its boundary is an edge
 * rather than a gradient. The dimensions are shared: `FlyStage` builds the geometry from them and
 * `groundMaterial` takes the half extents as uniforms, so the shading and the silhouette cannot
 * disagree.
 *
 * WHY THESE NUMBERS, and it is worth writing down because two successive failures here were both
 * framing errors rather than material errors. The hero camera is fixed at [1.85, 1.15, 2.35]
 * looking at [0, 0.5, 0] with fov 32, and it is LOW and CLOSE. Ray casting the frame against the
 * slab's own box gives, for this camera:
 *
 *   - The frame's bottom edge crosses the floor at world z = 1.087 - 0.786x, i.e. z ~ 1.09 at the
 *     frame's centre column and only ~0.30 at its right edge. So the floor is only visible from
 *     about 2.1 world units in front of the camera outward: any surface nearer than that is below
 *     the frame. A slab's NEAR edge can therefore only be shown by pulling hz below ~1.1, which
 *     would put the near edge inside the fly's own footprint (its feet reach z=+0.9 and the pool
 *     reaches z=+1.5). The near lip is not available at this camera and no thickness shows it.
 *   - The slab's side walls are never visible either. The far wall is hidden under the top face,
 *     the left wall faces away from a camera that sits at x=+1.85, and the right and near walls
 *     are below the frame. Thickness was ray cast at 0.16, 0.20 and 0.34 and changes nothing.
 *
 * So the ONLY readable boundary from this camera is the top face's own silhouette against the
 * void, and that is what has to carry the edge: hence the lit rim band in the material, and hence
 * half extents small enough that the far-left corner and the left edge fall well inside the
 * frame instead of just off its left border. At hx=3.35 the left edge sat at screen x=-76: one
 * pixel of black margin, so the slab appeared to bleed off the frame with no visible limit. At
 * hx=2.30 the left edge crosses the frame at x~98 and the far-left corner lands at (591, 408),
 * which puts the slab's corner and two of its edges inside the frame with black beyond them.
 *
 * The rings in `FlyStage` are scaled to match: at their old 2.6 radius they would have extended
 * past a 2.30 x 2.10 slab and drawn a circle out over the void.
 */
import * as THREE from 'three';

export const SLAB = {
  /** half extent along world x. Small enough that the left edge lands inside the frame. */
  hx: 2.3,
  /** half extent along world z. The far-left corner still projects into frame at (591, 408). */
  hz: 2.1,
  /**
   * Slab thickness. No wall of this box is visible from the hero camera (see above), so this is
   * here for the silhouette and for any other framing, not because it shows in the hero shot.
   */
  thickness: 0.2,
  /** corner radius of the rounded rectangle. */
  radius: 0.5,
  /**
   * The contact pool, in the slab's own object space (x, y) = (world x, -world z).
   * An ellipse under the body: the fly's feet span about +-0.6 in x and +-0.9 in z, so this clears
   * them and eases out over a further ~0.4, which is the soft edge the reference shows. Its reach
   * (1.5 in z) stays clear of the rim band, which begins at 0.80 of the half extents.
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
