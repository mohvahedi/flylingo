// GLSL for the connectome cloud. Two programs: the full 166,700 point cloud and
// the small live overlay that carries the 512 sampled state values.
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
  vec3 col = mix(uColorDim, uColorHot, pow(vAct, 0.6));
  col = mix(col, uColorNeg, clamp(-vHot, 0.0, 1.0) * 0.55);
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
varying float vAct;
varying float vSign;
varying float vShock;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float mask = smoothstep(0.25, 0.01, r2);
  vec3 base = vSign >= 0.0 ? uPos : uNeg;
  vec3 col = mix(base, uShockColor, clamp(vShock, 0.0, 1.0));
  // A live slot with no activity and no spike is invisible, so a quiet brain
  // shows only its anatomy. A spike is drawn at full strength regardless of
  // how small the underlying state value was.
  float alpha = mask * clamp(0.15 + 1.1 * vAct + vShock, 0.0, 1.0);
  if (alpha < 0.02) discard;
  gl_FragColor = vec4(col, alpha);
}
`;
