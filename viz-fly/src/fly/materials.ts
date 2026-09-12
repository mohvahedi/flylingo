/**
 * Procedural materials for the fly. No textures, no downloads, no HDR files: every
 * pattern below is analytic GLSL injected into three's own physical shader with
 * onBeforeCompile.
 *
 * Why patch rather than texture: the fly ships with no assets at all, because the app it
 * lives in may be demoed offline. A chitin normal map plus an eye AO map plus a wing vein
 * map would be three downloads, three licences and a build step. The patterns here are a
 * few ALU ops per fragment, resolution independent, and cost nothing to load.
 *
 * The base look is three's own MeshPhysicalMaterial, which is what actually makes this
 * read as expensive: clearcoat gives the hard specular of lacquered cuticle, thin film
 * iridescence gives the gold to green to blue shift with viewing angle, sheen gives the
 * fuzz on the setae, and real dielectric specular replaces the plastic look of
 * MeshStandardMaterial. The injected code only adds the microstructure three cannot
 * supply without a map: tergite grooves, cuticle grain, ommatidial facets, wing venation.
 *
 * Cost: one compiled program per `key` below. All three chitin materials for a region
 * share `fly-chitin-1`, so the whole fly is 3 programs plus the ground.
 */
import * as THREE from 'three';
import type { ChannelKind } from './regions';

type UniformValue = { value: number } | { value: THREE.Vector3 };

type Patch = {
  /** program cache key. Materials that share a key share one compiled program. */
  key: string;
  /** statements appended to the end of the fragment main(), right after the emissive term */
  body: string;
  /** extra uniforms. Values are per material instance, so a shared program is still safe. */
  uniforms: Record<string, UniformValue>;
};

/**
 * Injected once per program, immediately after <common>. Declares the object space
 * position varying and the small hash toolkit the pattern bodies use.
 */
const HELPERS = /* glsl */ `
varying vec3 vObjPos;
uniform float uPatA;
uniform float uPatB;
uniform float uGroove;
uniform float uGrain;
uniform vec3  uSheenTint;
uniform float uFresnel;
uniform float uFresnelPow;
uniform float uGlint;

float flHash13( vec3 p ) {
	p = fract( p * vec3( 0.1031, 0.1030, 0.0973 ) );
	p += dot( p, p.yxz + 33.33 );
	return fract( ( p.x + p.y ) * p.z );
}
float flHash21( vec2 p ) {
	p = fract( p * vec2( 0.1031, 0.1030 ) );
	p += dot( p, p.yx + 33.33 );
	return fract( ( p.x + p.y ) * p.z );
}
float flLine( float v, float c, float w ) {
	return 1.0 - smoothstep( 0.0, w, abs( v - c ) );
}
`;

/**
 * Insert `add` after `needle`, and fail loudly if the anchor is gone. A silent miss would
 * leave a stock shader and look "almost right", which is worse than an obvious error.
 */
function after(src: string, needle: string, add: string, what: string): string {
  if (src.indexOf(needle) === -1) {
    throw new Error(`fly material patch: anchor "${needle}" not found in ${what}`);
  }
  return src.replace(needle, `${needle}\n${add}`);
}

/**
 * Install a fragment patch and an object space position varying.
 *
 * `vObjPos = position` is the raw geometry attribute, deliberately not the transformed
 * vertex: it means the pattern is defined on the primitive the part is built from, so a
 * unit sphere that has been squashed into a thorax still carries a stable, seamless
 * coordinate system that the non-uniform mesh scale cannot smear.
 */
function patchMaterial(material: THREE.MeshPhysicalMaterial, p: Patch): THREE.MeshPhysicalMaterial {
  material.onBeforeCompile = (shader) => {
    shader.vertexShader =
      'varying vec3 vObjPos;\n' +
      after(shader.vertexShader, 'void main() {', '\tvObjPos = position;', 'the vertex main');
    shader.fragmentShader = after(
      after(shader.fragmentShader, '#include <common>', HELPERS, 'the fragment prelude'),
      '#include <emissivemap_fragment>',
      p.body,
      'the physical fragment main',
    );
    Object.assign(shader.uniforms, p.uniforms);
    // keep a handle so the render loop can nudge a uniform without a recompile
    material.userData.uniforms = shader.uniforms;
  };
  material.customProgramCacheKey = () => p.key;
  return material;
}

/* ------------------------------------------------------------------ chitin */

/**
 * Tergite grooves, cuticle grain, and a fresnel sheen. Runs on every hard part of the
 * body: thorax, head, four abdomen segments, six legs.
 */
const CHITIN_BODY = /* glsl */ `
{
	vec3 op = vObjPos;

	// tergite bands: grooves across the body's long axis. uPatA is bands per object unit,
	// so the abdomen gets a couple per segment and the head gets a nearly smooth plate.
	float band = fract( op.z * uPatA + op.y * 0.4 );
	float groove = smoothstep( 0.38, 0.50, abs( band - 0.5 ) );

	// cuticle grain, two hash octaves. Sells the scale: without it the shell is a mirror.
	float g1 = flHash13( op * uPatB );
	float g2 = flHash13( floor( op * uPatB ) * 0.37 + 4.1 );
	float grain = mix( g1, g2, 0.5 );
	float pit = smoothstep( 0.66, 0.98, grain );

	diffuseColor.rgb *= 1.0 - uGroove * groove;
	diffuseColor.rgb *= mix( 1.0, 0.82, pit * 0.8 );

	roughnessFactor = clamp(
		roughnessFactor * ( 1.0 + 1.1 * groove ) + ( 0.10 * pit + 0.05 * grain ) * uGrain,
		0.03, 1.0
	);

	// fresnel: warm gold head on, cool green at grazing angles. The gold to blue part of
	// the shift is the real thin film iridescence on the material; this adds the green.
	float fres = pow( 1.0 - saturate( dot( normal, normalize( vViewPosition ) ) ), uFresnelPow );
	diffuseColor.rgb = mix( diffuseColor.rgb, diffuseColor.rgb * 0.35 + uSheenTint, fres * uFresnel );

	// sparkle on the raised cuticle, so the silhouette is not a smooth silhouette
	totalEmissiveRadiance += uSheenTint * ( uGlint * fres * pit );
}
`;

export type ChitinTone = 'thorax' | 'head' | 'abdomen' | 'legs';

/**
 * Per region look. `band` and `grain` are in object space, and the body parts are all
 * unit spheres squashed by mesh.scale while the legs are capsules already in world
 * metres, hence the two very different grain frequencies.
 */
const CHITIN_TONES: Record<
  ChitinTone,
  {
    albedo: string;
    rough: number;
    band: number;
    groove: number;
    grainFreq: number;
    grain: number;
    irid: number;
    coat: number;
    sheen: number;
  }
> = {
  thorax: {
    albedo: '#7a4d1c', rough: 0.26, band: 1.5, groove: 0.22,
    grainFreq: 46, grain: 1.0, irid: 0.62, coat: 1.0, sheen: 0.35,
  },
  head: {
    albedo: '#6b4218', rough: 0.30, band: 1.0, groove: 0.14,
    grainFreq: 40, grain: 1.0, irid: 0.55, coat: 1.0, sheen: 0.4,
  },
  abdomen: {
    albedo: '#87571f', rough: 0.30, band: 1.5, groove: 0.34,
    grainFreq: 44, grain: 0.95, irid: 0.7, coat: 1.0, sheen: 0.45,
  },
  legs: {
    albedo: '#412b14', rough: 0.38, band: 42, groove: 0.30,
    grainFreq: 520, grain: 0.8, irid: 0.42, coat: 0.9, sheen: 0.55,
  },
};

/**
 * Warm amber gold cuticle. The emissive channel is the connectome instrument layer and is
 * driven per frame by Fly.tsx; it is set to BASE_GLOW here and never touched again.
 */
export function chitinMaterial(tone: ChitinTone, region: ChannelKind): THREE.MeshPhysicalMaterial {
  const t = CHITIN_TONES[tone];
  const m = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(t.albedo),
    emissive: new THREE.Color(region === 'legs' ? '#0a1f22' : '#0d1b22'),
    emissiveIntensity: 0.1,
    roughness: t.rough,
    metalness: 0.0,
    clearcoat: t.coat,
    clearcoatRoughness: 0.08,
    iridescence: t.irid,
    iridescenceIOR: 1.34,
    iridescenceThicknessRange: [180, 560],
    sheen: t.sheen,
    sheenColor: new THREE.Color('#f2cf94'),
    sheenRoughness: 0.5,
    specularIntensity: 1.0,
    envMapIntensity: 1.0,
  });
  return patchMaterial(m, {
    key: 'fly-chitin-1',
    body: CHITIN_BODY,
    uniforms: {
      uPatA: { value: t.band },
      uPatB: { value: t.grainFreq },
      uGroove: { value: t.groove },
      uGrain: { value: t.grain },
      uSheenTint: { value: new THREE.Vector3(0.34, 0.86, 0.6) },
      uFresnel: { value: 0.75 },
      uFresnelPow: { value: 3.0 },
      uGlint: { value: 0.05 },
    },
  });
}

/* -------------------------------------------------------------------- eyes */

/**
 * Ommatidial facets. A staggered grid in spherical coordinates: the theta axis is offset
 * by half a cell for every other phi row, which is the cheapest honest approximation of
 * hexagonal packing, and the cell borders are darker and rougher than the lenses they
 * separate.
 */
const EYE_BODY = /* glsl */ `
{
	vec3 d = normalize( vObjPos );
	float theta = atan( d.z, d.x );
	float phi = acos( clamp( d.y, -1.0, 1.0 ) );

	vec2 g = vec2( theta * uPatA, phi * uPatB );
	g.x += 0.5 * floor( g.y );

	vec2 cell = floor( g );
	vec2 gv = abs( fract( g ) - 0.5 );
	float edge = smoothstep( 0.34, 0.50, max( gv.x, gv.y ) );
	float ch = flHash21( cell );

	diffuseColor.rgb *= 1.0 - uGroove * edge;
	diffuseColor.rgb *= mix( 0.72, 1.06, ch );
	roughnessFactor = clamp( roughnessFactor + 0.34 * edge + 0.06 * ch, 0.02, 1.0 );

	float fres = pow( 1.0 - saturate( dot( normal, normalize( vViewPosition ) ) ), uFresnelPow );
	// one glint per ommatidium, brighter on the ridges, plus a hard dome highlight
	totalEmissiveRadiance += uSheenTint * uGlint * ( 0.45 + 0.55 * ch ) * ( 0.35 + 0.65 * edge );
	totalEmissiveRadiance += uSheenTint * uGlint * 0.6 * fres;
}
`;

/**
 * Deep red crimson, high specular, faceted. Also used for the ocelli, the antenna tips and
 * the proboscis labellum, which is why the facet frequency is a parameter: the ocelli are
 * a tenth the size of the eyes and would alias at the eye's frequency.
 */
export function eyeMaterial(facets = 9.0): THREE.MeshPhysicalMaterial {
  const m = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color('#48101a'),
    emissive: new THREE.Color('#7f1d1d'),
    emissiveIntensity: 0.16,
    roughness: 0.14,
    metalness: 0.0,
    clearcoat: 1.0,
    clearcoatRoughness: 0.05,
    iridescence: 0.4,
    iridescenceIOR: 1.4,
    iridescenceThicknessRange: [120, 380],
    sheen: 0.3,
    sheenColor: new THREE.Color('#ff6a55'),
    sheenRoughness: 0.4,
    envMapIntensity: 1.2,
  });
  return patchMaterial(m, {
    key: 'fly-eye-1',
    body: EYE_BODY,
    uniforms: {
      uPatA: { value: facets },
      uPatB: { value: facets * 1.05 },
      uGroove: { value: 0.42 },
      uGrain: { value: 0.0 },
      uSheenTint: { value: new THREE.Vector3(1.0, 0.3, 0.22) },
      uFresnel: { value: 0.5 },
      uFresnelPow: { value: 2.4 },
      uGlint: { value: 0.09 },
    },
  });
}

/* ------------------------------------------------------------------- wings */

/**
 * Veined translucent membrane. The wing shape is generated in the XY plane with the root
 * at the origin and x running root to tip, so x and y are already the natural vein
 * coordinates: five longitudinal veins fanned down the length, three cross veins across
 * it, a smooth fade at the root and the tip where a real wing has no veins.
 *
 * The membrane keeps its low base opacity and the veins add to it, so a lit membrane is
 * see through and a vein is not.
 */
const WING_BODY = /* glsl */ `
{
	vec3 op = vObjPos;
	float t = clamp( op.x / uPatA, 0.0, 1.0 );

	// chord widens from the root to about two thirds span, then tapers to the tip
	float chord = uPatB * ( 0.18 + 0.82 * sin( 3.14159 * pow( max( t, 0.0001 ), 0.62 ) ) );
	float yn = op.y / max( chord, 0.0001 );

	float lon =
		flLine( yn, -0.74, 0.085 ) +
		flLine( yn, -0.34, 0.075 ) +
		flLine( yn,  0.02, 0.070 ) +
		flLine( yn,  0.38, 0.075 ) +
		flLine( yn,  0.74, 0.085 );
	float cross =
		flLine( t, 0.26, 0.030 ) +
		flLine( t, 0.55, 0.026 ) +
		flLine( t, 0.82, 0.030 );

	float inside = smoothstep( 1.0, 0.76, abs( yn ) ) * smoothstep( 1.0, 0.94, t );
	float veins = clamp( ( lon + cross ) * inside, 0.0, 1.0 );

	// membrane: thin and slightly cool, thickening towards the tip
	diffuseColor.rgb = mix( diffuseColor.rgb, uSheenTint, 0.35 * t );
	diffuseColor.a = clamp( diffuseColor.a + uGroove * veins + 0.08 * t, 0.0, 1.0 );

	// veins run thicker, darker and much less transparent than the membrane
	diffuseColor.rgb = mix( diffuseColor.rgb, diffuseColor.rgb * 0.22, veins );
	roughnessFactor = clamp( mix( roughnessFactor, 0.28, veins ) + ( 1.0 - veins ) * 0.04, 0.02, 1.0 );

	// rim light on the membrane: the whole point of a translucent wing
	float fres = pow( 1.0 - saturate( dot( normal, normalize( vViewPosition ) ) ), uFresnelPow );
	diffuseColor.rgb += uSheenTint * uFresnel * fres * ( 0.35 + 0.65 * t );
	totalEmissiveRadiance += uSheenTint * uGlint * fres;
}
`;

/**
 * Wing geometry: a real fly wing outline (elongated, blunt tip, narrow waist at the root)
 * built from four cubic beziers and tessellated flat. ~96 triangles, and it gives the
 * silhouette something to bite on instead of a three sided blade.
 */
export function wingGeometry(len = 0.66, chord = 0.2): THREE.BufferGeometry {
  const s = new THREE.Shape();
  s.moveTo(0, 0.02);
  s.bezierCurveTo(len * 0.16, chord * 0.92, len * 0.6, chord * 1.02, len * 0.95, chord * 0.3);
  s.bezierCurveTo(len * 1.04, chord * 0.06, len * 0.9, -chord * 0.5, len * 0.54, -chord * 0.7);
  s.bezierCurveTo(len * 0.28, -chord * 0.8, len * 0.07, -chord * 0.44, 0.0, 0.02);
  const g = new THREE.ShapeGeometry(s, 24);
  g.computeVertexNormals();
  return g;
}

export function wingMaterial(len = 0.66, chord = 0.2): THREE.MeshPhysicalMaterial {
  const m = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color('#cfe3f2'),
    emissive: new THREE.Color('#0b1a22'),
    emissiveIntensity: 0.1,
    roughness: 0.08,
    metalness: 0.0,
    transparent: true,
    opacity: 0.3,
    depthWrite: false,
    side: THREE.DoubleSide,
    clearcoat: 0.8,
    clearcoatRoughness: 0.08,
    iridescence: 0.75,
    iridescenceIOR: 1.28,
    iridescenceThicknessRange: [140, 460],
    sheen: 0.6,
    sheenColor: new THREE.Color('#9fd8ff'),
    sheenRoughness: 0.35,
    envMapIntensity: 1.4,
  });
  return patchMaterial(m, {
    key: 'fly-wing-1',
    body: WING_BODY,
    uniforms: {
      uPatA: { value: len },
      uPatB: { value: chord },
      uGroove: { value: 0.62 },
      uGrain: { value: 0.0 },
      uSheenTint: { value: new THREE.Vector3(0.6, 0.82, 1.0) },
      uFresnel: { value: 0.55 },
      uFresnelPow: { value: 2.6 },
      uGlint: { value: 0.05 },
    },
  });
}

/* ------------------------------------------------------------------ setae */

/**
 * Short bristles. One cone geometry, instanced, never touched after setup: the setae ride
 * whatever group they are parented to, so the ones on a femur follow the walk cycle for
 * free. Sheen is doing the work here, it is what makes a 7 mm cone read as fuzz and not as
 * a spike.
 */
export function setaeMaterial(): THREE.MeshPhysicalMaterial {
  return new THREE.MeshPhysicalMaterial({
    color: new THREE.Color('#2a1d0f'),
    roughness: 0.42,
    metalness: 0.0,
    clearcoat: 0.5,
    clearcoatRoughness: 0.2,
    sheen: 1.0,
    sheenColor: new THREE.Color('#d9b073'),
    sheenRoughness: 0.45,
    envMapIntensity: 1.1,
  });
}

/* ----------------------------------------------------------------- ground */

/**
 * Ground. A radial falloff to true black rather than a plane that ends, so there is no
 * visible edge and no horizon; the fly and its shadow sit in a pool of light that dies
 * out. Clearcoat on a low frequency grain gives the key light something to streak across,
 * which is what makes a floor look like a floor under a hard source.
 */
const GROUND_BODY = /* glsl */ `
{
	float r = length( vObjPos.xy );
	float fade = smoothstep( 6.2, 1.5, r );
	diffuseColor.rgb *= fade * fade;
	roughnessFactor = clamp(
		roughnessFactor + 0.14 * flHash13( vObjPos * 26.0 ),
		0.02, 1.0
	);
}
`;

export function groundMaterial(radius = 7): THREE.MeshPhysicalMaterial {
  const m = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color('#0d1319'),
    roughness: 0.46,
    metalness: 0.0,
    clearcoat: 0.45,
    clearcoatRoughness: 0.42,
    envMapIntensity: 0.8,
  });
  return patchMaterial(m, {
    key: 'fly-ground-1',
    body: GROUND_BODY,
    uniforms: {
      uPatA: { value: radius },
      uPatB: { value: 1 },
      uGroove: { value: 0 },
      uGrain: { value: 0 },
      uSheenTint: { value: new THREE.Vector3(0, 0, 0) },
      uFresnel: { value: 0 },
      uFresnelPow: { value: 2 },
      uGlint: { value: 0 },
    },
  });
}
