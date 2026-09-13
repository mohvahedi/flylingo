// GLSL for the connectome cloud. Four programs:
//   1. the full 166,700 point cloud,
//   2. the small live overlay that carries the 512 sampled state values,
//   3. a sampled hairline edge set, drawn as screen-space ribbons,
//   4. the decaying trail of travelling activity pulses.
//
// AUTO-RANGING. Measured telemetry state is mostly small: median |state| is
// 0.051, the 95th percentile is 0.274, max observed 0.717, state_rms 0.105.
// Mapping brightness linearly from 0..1 therefore renders an almost black
// screen. Every program here normalises by uRef, a per-frame reference derived
// from the 95th percentile of |state|. uRef is driven to exactly zero on a dead
// frame (the no_edges control gives all zeros), and a zero reference collapses
// the activity term to nothing rather than amplifying noise into a light show.
//
// The full cloud keeps a low ambient term from aBase (degree percentile) so the
// anatomy stays legible when the brain is quiet, but that ambient term is
// deliberately dim and carries no activity claim. On a dead frame the ambient
// term is multiplied by uAmbientQuiet, so the all-zero control stays dark
// instead of showing an unearned structure.
//
// HONESTY OF BRIGHTNESS. 27,038 neurons have no measured soma and sit on a
// group centroid, so dozens or thousands of points share one coordinate. Those
// points arrive with aFill = 1 and aScale = 1 / (points sharing that position),
// are drawn smaller and tinted a cooler, darker cyan, and their alpha is
// weighted by aScale. Without the weight, additive blending summed a 14,418
// point pile into a blown out white disc and the placeholder geometry became
// the loudest thing in the panel. The measured somata are the foreground.
//
// PALETTE. Deep near-black background, nodes on a cyan to pale blue ramp
// (#5BC8D6, #7FE0EA, #BFEFF7), amber (#F0A030) reserved for the newest spikes.
// Hue and core brightness carry real per neuron values: cell class moves a node
// along the cyan to pale blue ramp, in-degree percentile drives the core,
// activity drives the hot end. No third hue family is introduced.
//
// TONE. Nothing here is allowed to reach pure white. Every layer runs its
// accumulated fragment colour through a soft saturating curve,
// 1 - exp(-gain * x), which is the identity-ish for dim nodes and rolls off
// towards 1 for bright ones, so a core keeps its hue instead of clipping. Amber
// on the spike ring is composited after the curve so the one colour that must
// stay readable stays saturated.

export const STATIC_VERT = /* glsl */ `
attribute float aBase;
attribute float aField;
attribute float aFill;
attribute float aScale;
attribute float aClass;
uniform float uSize;
uniform float uPixelRatio;
uniform float uViewHeight;
uniform float uGlobal;
uniform float uRef;
uniform float uFillSize;
uniform float uAttenNear;
uniform float uAttenFar;
uniform float uFogNear;
uniform float uFogFar;
varying float vAct;
varying float vHot;
varying float vBase;
varying float vFill;
varying float vScale;
varying float vClass;
varying float vFog;
float normField(float v) {
  // uRef <= 0 means a dead frame: contribute nothing at all.
  if (uRef <= 0.0001) return 0.0;
  return clamp(abs(v) / uRef, 0.0, 1.0);
}
void main() {
  float act = normField(aField);
  vAct = act;
  vHot = aField >= 0.0 ? act : -act;
  vBase = clamp(aBase, 0.0, 1.0);
  vFill = aFill;
  vScale = clamp(aScale, 0.0, 1.0);
  vClass = clamp(aClass, 0.0, 1.0);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float depth = max(-mv.z, 0.35);
  // Depth cue: far nodes fade, which is what gives the volume its depth
  // instead of reading as a flat scatter.
  vFog = smoothstep(uFogNear, uFogFar, depth);
  float sizeScale = (1.0 + 0.9 * act) * (0.85 + 0.3 * aBase);
  // Centroid-placed neurons are drawn visibly smaller than measured somata.
  sizeScale *= mix(1.0, uFillSize, aFill);
  // Distance falloff, clamped at both ends. The earlier form was
  // 1.0 / max(-mv.z, 0.35) on its own: the cloud spans about [-1, 1] with the
  // camera roughly two units back, so that factor was close to 0.33 and a 2.4px
  // point became sub-pixel. 166,700 sub-pixel points rasterise to almost
  // nothing, which is why the brain once rendered as a sparse scatter. The
  // clamp keeps the falloff a real depth cue without ever collapsing a point
  // below the viewport-scaled size, and the final floor of 1.15px guarantees a
  // point always covers a pixel.
  float atten = clamp(uAttenNear / depth, 0.72, 1.4);
  float px = uSize * uPixelRatio * (uViewHeight / 600.0) * sizeScale * atten;
  gl_PointSize = clamp(px, 1.15, 26.0);
}
`;

export const STATIC_FRAG = /* glsl */ `
precision mediump float;
uniform vec3 uColorDim;
uniform vec3 uColorPale;
uniform vec3 uColorHot;
uniform vec3 uColorNeg;
uniform vec3 uColorInterp;
uniform float uAmbient;
uniform float uQuiet;
uniform float uClassMix;
uniform float uFillTint;
uniform float uToneGain;
uniform float uFogDim;
varying float vAct;
varying float vHot;
varying float vBase;
varying float vFill;
varying float vScale;
varying float vClass;
varying float vFog;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float mask = smoothstep(0.25, 0.015, r2);
  // A tight core sitting inside the soft disc. Round sprites with a bright
  // centre are what give the cloud the long-exposure look of the reference
  // rather than a flat scatter of equally lit dots.
  float core = smoothstep(0.055, 0.0, r2);
  // Cell class moves the node along the cyan to pale blue ramp. This is the
  // real per neuron value the panel carries, inside one hue family.
  vec3 col = mix(uColorDim, uColorPale, vClass * uClassMix);
  col = mix(col, uColorHot, pow(vAct, 0.6));
  col = mix(col, uColorNeg, clamp(-vHot, 0.0, 1.0) * 0.45);
  col += core * (0.18 + 0.5 * vAct) * (0.45 + 0.55 * vBase);
  // Centroid-placed points take a cooler, darker tint so they read as
  // interpolated fill rather than as measured anatomy.
  col = mix(col, uColorInterp, vFill * uFillTint);
  // uAmbient carries the anatomy. It is scaled by degree percentile so dense
  // cells outline the shape, gated by uAmbientQuiet so an all-zero state renders
  // dark instead of glowing, and cut back again for centroid points. Every term
  // is weighted by vScale so a pile of hundreds on one coordinate sums to about
  // what a single point would, instead of burning a hole in the frame.
  float quiet = uQuiet;
  float ambient = uAmbient * quiet * (0.35 + 0.65 * vBase) * (1.0 - 0.8 * vFill) * vScale;
  float alpha = mask * (ambient + 0.95 * vAct * vAct * vScale);
  // Soft saturation: luminous, never clipped to flat white.
  vec3 tone = 1.0 - exp(-max(col, 0.0) * uToneGain);
  // Depth fog for an additive layer is a dimming, not a blend towards a fog
  // colour: adding a colour would brighten the far side.
  tone *= mix(1.0, uFogDim, vFog);
  alpha *= mix(1.0, 0.55, vFog);
  gl_FragColor = vec4(tone, clamp(alpha, 0.0, 0.92));
}
`;

export const LIVE_VERT = /* glsl */ `
attribute float aActivity;
attribute float aShock;
uniform float uSize;
uniform float uPixelRatio;
uniform float uViewHeight;
uniform float uRef;
varying float vAct;
varying float vSign;
varying float vShock;
varying float vFog;
void main() {
  // Same auto-range rule as the big cloud.
  float mag = uRef > 0.0001 ? clamp(abs(aActivity) / uRef, 0.0, 1.0) : 0.0;
  vAct = mag;
  vSign = aActivity >= 0.0 ? 1.0 : -1.0;
  vShock = aShock;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float depth = max(-mv.z, 0.35);
  vFog = smoothstep(1.5, 4.2, depth);
  // Spikes are rare, about 5 of 512 slots per frame, so individual spike points
  // have to be legible on their own. A spike pin gets a large fixed boost.
  float sizeScale = 1.0 + 2.0 * mag + 6.0 * aShock;
  // Same viewport-scaled pixel sizing as the static cloud, so the live overlay sits
  // on the same visual scale and never collapses below a visible pixel.
  float px = uSize * uPixelRatio * (uViewHeight / 600.0) * sizeScale * clamp(2.4 / depth, 0.75, 1.35);
  // A spike pin is rasterised at a size floor. The accent ring lives at radius
  // 0.35-0.47 of the sprite, so a shrunken sprite put the ring inside a couple
  // of pixels and it was lost in the fragment coverage test.
  float minPx = aShock > 0.02 ? 16.0 : 1.5;
  gl_PointSize = clamp(px, minPx, 40.0);
}
`;

export const LIVE_FRAG = /* glsl */ `
precision mediump float;
uniform vec3 uPos;
uniform vec3 uNeg;
uniform vec3 uShockColor;
uniform float uGate;
uniform float uToneGain;
varying float vAct;
varying float vSign;
varying float vShock;
varying float vFog;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float mask = smoothstep(0.25, 0.01, r2);
  float core = smoothstep(0.055, 0.0, r2);
  vec3 base = vSign >= 0.0 ? uPos : uNeg;
  float shock = clamp(vShock, 0.0, 1.0) * uGate;
  vec3 col = 1.0 - exp(-max(base, 0.0) * uToneGain);
  col += core * (0.22 + 0.45 * vAct);
  col *= mix(1.0, 0.7, vFog);
  // A live slot with no activity and no spike is a dim marker only, so a quiet
  // brain shows its anatomy and little else. uGate is zero on a dead frame:
  // every one of the 512 pins then disappears instead of leaving 512 dim dots
  // glowing over an all-zero state.
  float energy = (0.15 + 1.1 * vAct + vShock) * uGate;
  float alpha = mask * clamp(energy, 0.0, 1.0);
  // A fresh spike gets a ring at the edge of its (large) sprite. A spike pin
  // lands on the brightest node in the frame, and additive amber on top of a
  // bright cyan node blends straight to white, which is why the amber read as
  // absent in an earlier attempt. The ring sits about 8 px out, where the cyan
  // underneath is dim, so the amber survives; the centre is pulled back so the
  // ring is what the eye catches. The amber is composited after the tone curve
  // so the one colour that must stay legible is not greyed out by it.
  //
  // The ring is computed BEFORE the coverage test and carries its own alpha
  // floor. It used to be drawn after the alpha < 0.02 discard, and its peak
  // band sat at r2 = 0.225 where the mask has already fallen to 0.03, so the
  // strongest part of the ring was culled and only a washed-out inner sliver
  // survived. Measured in the panel: spikes reported 3-5 per frame with warm_px
  // 0 and r - b up to 64 but no pixel passing r > g + 6.
  float ring = 0.0;
  if (shock > 0.02) {
    ring = smoothstep(0.085, 0.155, r2) * (1.0 - smoothstep(0.215, 0.248, r2));
    alpha = max(alpha, ring * shock);
  }
  if (alpha < 0.02) discard;
  if (ring > 0.0) {
    // The accent must not be able to lose the red/green race. Two changes: the
    // pin's own cyan is pushed down under the ring, then a low-green amber is
    // mixed in at full strength. Cyan (r low, g high) added to amber pulls
    // green level with red once the local cyan accumulation exceeds roughly the
    // amber alpha, and past that point no pixel passes r > g + 6, so the accent
    // is invisible in the capture even though it is drawing.
    col *= 1.0 - 0.85 * ring;
    col = mix(col, uShockColor, ring);
  }
  gl_FragColor = vec4(col, clamp(alpha, 0.0, 1.0));
}
`;

// Hairline edges, drawn as screen-space ribbons instead of GL lines. A GL line
// is exactly one device pixel wide, and at this camera distance the sampled
// edges are a few pixels long and mostly diagonal, so a single-pixel line
// disappeared under the cloud and the panel read as a point cloud rather than a
// connectome. Each edge is a four vertex quad: the vertex shader projects both
// endpoints, takes the screen-space direction, and offsets the quad sideways by
// uWidthPx, so an edge is never thinner than that many pixels at any distance.
export const EDGE_VERT = /* glsl */ `
attribute vec3 aStart;
attribute vec3 aEnd;
attribute float aSide;
attribute float aT;
attribute float aWeight;
uniform float uWidthPx;
uniform float uViewHeight;
varying float vWeight;
varying float vFog;
void main() {
  vWeight = aWeight;
  mat4 mvp = projectionMatrix * modelViewMatrix;
  vec4 clipA = mvp * vec4(aStart, 1.0);
  vec4 clipB = mvp * vec4(aEnd, 1.0);
  vec4 self = aT < 0.5 ? clipA : clipB;
  vec4 other = aT < 0.5 ? clipB : clipA;
  vec2 ndcA = clipA.xy / max(abs(clipA.w), 1e-5);
  vec2 ndcB = clipB.xy / max(abs(clipB.w), 1e-5);
  vec2 dir = ndcB - ndcA;
  float len = length(dir);
  dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
  vec2 perp = vec2(-dir.y, dir.x);
  float pxToNdc = 2.0 / max(uViewHeight, 1.0);
  float halfWidth = max(uWidthPx, 1.0) * 0.5 * pxToNdc;
  vFog = smoothstep(1.5, 4.2, max(self.w, 0.35));
  gl_Position = vec4(self.xy + perp * aSide * halfWidth * self.w, self.zw);
}
`;

export const EDGE_FRAG = /* glsl */ `
precision mediump float;
uniform vec3 uColor;
uniform float uOpacity;
uniform float uToneGain;
varying float vWeight;
varying float vFog;
void main() {
  // Low opacity cyan, dimmed with depth so the mesh sits under the cloud
  // instead of drawing a grid over it.
  float a = uOpacity * (0.35 + 0.65 * vWeight) * (1.0 - 0.65 * vFog);
  if (a < 0.002) discard;
  vec3 col = 1.0 - exp(-max(uColor, 0.0) * uToneGain);
  gl_FragColor = vec4(col, a);
}
`;

// Travelling pulses. Each dot is one spawn event: position along the pulse ray,
// age in seconds, and the power the slot had when it spawned. Age drives size
// and alpha down, so the tail is a real decay rather than a static streak.
export const TRAIL_VERT = /* glsl */ `
attribute float aAge;
attribute float aPower;
attribute vec3 aColor;
uniform float uSize;
uniform float uPixelRatio;
uniform float uViewHeight;
uniform float uLife;
varying float vFade;
varying vec3 vColor;
varying float vFog;
void main() {
  // 1.0 at spawn, 0.0 at uLife. Squaring makes the tail fall away quickly.
  float f = clamp(1.0 - aAge / uLife, 0.0, 1.0);
  vFade = f * f;
  vColor = aColor;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float depth = max(-mv.z, 0.35);
  vFog = smoothstep(1.5, 4.2, depth);
  // Same viewport-relative pixel sizing as the cloud, so a trail dot is never
  // sub-pixel, and it shrinks along the tail instead of vanishing at once.
  float sizeScale = (0.35 + 1.05 * aPower) * (0.25 + 0.75 * f);
  float px = uSize * uPixelRatio * (uViewHeight / 600.0) * sizeScale * clamp(2.4 / depth, 0.72, 1.4);
  gl_PointSize = clamp(px, 1.0, 20.0);
}
`;

export const TRAIL_FRAG = /* glsl */ `
precision mediump float;
uniform float uAlpha;
uniform float uToneGain;
varying float vFade;
varying vec3 vColor;
varying float vFog;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float mask = smoothstep(0.25, 0.02, r2);
  float core = smoothstep(0.055, 0.0, r2);
  float a = uAlpha * vFade * mask * (1.0 - 0.5 * vFog);
  if (a < 0.004) discard;
  vec3 col = 1.0 - exp(-max(vColor + core * vFade * 0.45, 0.0) * uToneGain);
  gl_FragColor = vec4(col, a);
}
`;
