// GLSL for the connectome cloud. Four programs:
//   1. the full 166,700 point cloud,
//   2. the small live overlay that carries the 512 sampled state values,
//   3. a sampled hairline edge set,
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
// deliberately dim and carries no activity claim.
//
// PALETTE. Deep near-black background, nodes on a cyan to pale blue ramp
// (#5BC8D6, #7FE0EA, #BFEFF7), amber (#F0A030) reserved for the newest spikes.

export const STATIC_VERT = /* glsl */ `
attribute float aBase;
attribute float aField;
uniform float uSize;
uniform float uPixelRatio;
uniform float uViewHeight;
uniform float uGlobal;
uniform float uRef;
varying float vAct;
varying float vHot;
varying float vBase;
float normField(float v) {
  // uRef <= 0 means a dead frame: contribute nothing at all.
  if (uRef <= 0.0001) return 0.0;
  return clamp(abs(v) / uRef, 0.0, 1.0);
}
void main() {
  float act = normField(aField);
  vAct = act;
  vHot = aField >= 0.0 ? act : -act;
  vBase = aBase;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float sizeScale = (1.0 + 0.9 * act) * (0.85 + 0.3 * aBase);
  // Point size in PIXELS, scaled by viewport height so the cloud looks the same at
  // any resolution. The previous form multiplied by 1/max(-mv.z, 0.35), and since the
  // cloud spans only about [-1, 1] with the camera roughly 3 units back, that factor
  // is about 0.33: a 2.4px point became about 0.8px, i.e. sub-pixel. 166,700 sub-pixel
  // points rasterise to almost nothing, which is why the brain rendered as a sparse
  // scatter of a few dozen dots instead of a cloud. The floor of 1.25px guarantees a
  // point always covers a pixel, and the ceiling keeps a close-up from filling the
  // screen with blobs.
  float px = uSize * uPixelRatio * (uViewHeight / 600.0) * sizeScale;
  gl_PointSize = clamp(px, 1.25, 26.0);
}
`;

export const STATIC_FRAG = /* glsl */ `
precision mediump float;
uniform vec3 uColorDim;
uniform vec3 uColorHot;
uniform vec3 uColorNeg;
uniform float uAmbient;
varying float vAct;
varying float vHot;
varying float vBase;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float mask = smoothstep(0.25, 0.015, r2);
  // A tight core sitting inside the soft disc. Round sprites with a bright
  // centre are what give the cloud the long-exposure look of the reference
  // rather than a flat scatter of equally lit dots.
  float core = smoothstep(0.055, 0.0, r2);
  vec3 col = mix(uColorDim, uColorHot, pow(vAct, 0.6));
  col = mix(col, uColorNeg, clamp(-vHot, 0.0, 1.0) * 0.5);
  col += core * (0.2 + 0.55 * vAct) * (0.45 + 0.55 * vBase);
  // uAmbient carries the anatomy. It is scaled by degree percentile so dense
  // cells outline the shape, and it is driven to a small value on a dead frame
  // so an all-zero state renders dark instead of glowing.
  float ambient = uAmbient * (0.35 + 0.65 * vBase);
  float alpha = mask * (ambient + 0.95 * vAct * vAct);
  gl_FragColor = vec4(col, alpha);
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
void main() {
  // Same auto-range rule as the big cloud.
  float mag = uRef > 0.0001 ? clamp(abs(aActivity) / uRef, 0.0, 1.0) : 0.0;
  vAct = mag;
  vSign = aActivity >= 0.0 ? 1.0 : -1.0;
  vShock = aShock;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  // Spikes are rare, about 5 of 512 slots per frame, so individual spike points
  // have to be legible on their own. A spike pin gets a large fixed boost.
  float sizeScale = 1.0 + 2.0 * mag + 6.0 * aShock;
  // Same viewport-scaled pixel sizing as the static cloud, so the live overlay sits
  // on the same visual scale and never collapses below a visible pixel.
  float px = uSize * uPixelRatio * (uViewHeight / 600.0) * sizeScale;
  gl_PointSize = clamp(px, 1.5, 40.0);
}
`;

export const LIVE_FRAG = /* glsl */ `
precision mediump float;
uniform vec3 uPos;
uniform vec3 uNeg;
uniform vec3 uShockColor;
uniform float uGate;
varying float vAct;
varying float vSign;
varying float vShock;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float mask = smoothstep(0.25, 0.01, r2);
  float core = smoothstep(0.055, 0.0, r2);
  vec3 base = vSign >= 0.0 ? uPos : uNeg;
  float shock = clamp(vShock, 0.0, 1.0);
  // uShockColor is the amber. Only a live spike drives vShock, so amber never
  // appears anywhere else in the picture.
  vec3 col = mix(base, uShockColor, shock);
  col += core * (0.22 + 0.45 * vAct + 0.75 * shock);
  // A live slot with no activity and no spike is a dim marker only, so a quiet
  // brain shows its anatomy and little else. uGate is zero on a dead frame:
  // every one of the 512 pins then disappears instead of leaving 512 dim dots
  // glowing over an all-zero state.
  float energy = (0.15 + 1.1 * vAct + vShock) * uGate;
  float alpha = mask * clamp(energy, 0.0, 1.0);
  if (alpha < 0.02) discard;
  // A fresh spike gets a ring at the edge of its (large) sprite. A spike pin
  // lands on the brightest node in the frame, and additive amber on top of a
  // bright cyan node blends straight to white, which is why the amber read as
  // absent in an earlier attempt. The ring sits about 8 px out, where the cyan
  // underneath is dim, so the amber survives; the centre is pulled back so the
  // ring is what the eye catches.
  if (shock > 0.02) {
    float ring = smoothstep(0.14, 0.225, r2) * (1.0 - smoothstep(0.225, 0.25, r2));
    col = mix(col, uShockColor * 1.18, ring);
    alpha = max(alpha * (1.0 - 0.65 * ring), ring * shock * uGate);
  }
  gl_FragColor = vec4(col, alpha);
}
`;

// Hairline edges. WebGL draws lines one pixel wide, which is exactly the
// reference look: a faint cyan mesh under the cloud, never a fat tube.
export const EDGE_VERT = /* glsl */ `
attribute float aWeight;
varying float vWeight;
void main() {
  vWeight = aWeight;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const EDGE_FRAG = /* glsl */ `
precision mediump float;
uniform vec3 uColor;
uniform float uOpacity;
varying float vWeight;
void main() {
  float a = uOpacity * vWeight;
  if (a < 0.002) discard;
  gl_FragColor = vec4(uColor, a);
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
void main() {
  // 1.0 at spawn, 0.0 at uLife. Squaring makes the tail fall away quickly.
  float f = clamp(1.0 - aAge / uLife, 0.0, 1.0);
  vFade = f * f;
  vColor = aColor;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  // Same viewport-relative pixel sizing as the cloud, so a trail dot is never
  // sub-pixel, and it shrinks along the tail instead of vanishing at once.
  float sizeScale = (0.35 + 1.05 * aPower) * (0.25 + 0.75 * f);
  float px = uSize * uPixelRatio * (uViewHeight / 600.0) * sizeScale;
  gl_PointSize = clamp(px, 1.0, 20.0);
}
`;

export const TRAIL_FRAG = /* glsl */ `
precision mediump float;
uniform float uAlpha;
varying float vFade;
varying vec3 vColor;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float mask = smoothstep(0.25, 0.02, r2);
  float core = smoothstep(0.055, 0.0, r2);
  float a = uAlpha * vFade * mask;
  if (a < 0.004) discard;
  gl_FragColor = vec4(vColor + core * vFade * 0.45, a);
}
`;
