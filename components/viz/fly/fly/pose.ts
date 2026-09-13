/**
 * Motion for the fly rig.
 *
 * Everything here is pure trigonometry over primitives assembled in Fly.tsx. There is no
 * downloaded rig and no external asset: the fly is built from spheres, capsules and boxes
 * in code, and this module says what angle every joint is at a given time.
 *
 * Frame convention: the fly faces +Z, up is +Y, right is +X. Ground plane is y = 0.
 */

export type Behavior = 'idle' | 'walk' | 'groom' | 'proboscis' | 'startle';

export type ReactionKind = 'none' | 'celebrate' | 'recoil';

export type LegPose = {
  hipYaw: number;
  hipRoll: number;
  femur: number;
  knee: number;
  tarsus: number;
};

export type Pose = {
  /** root translation, metres-ish, fly is about 2.4 units nose to tail */
  rootY: number;
  rootZ: number;
  rootYaw: number;
  bodyPitch: number;
  bodyRoll: number;
  bodyYaw: number;
  bodyY: number;
  headPitch: number;
  headYaw: number;
  /** four abdomen segments, pitch relative to the one in front */
  abdPitch: number[];
  abdScale: number[];
  /** antennae: [yaw, pitch] */
  antL: { yaw: number; pitch: number };
  antR: { yaw: number; pitch: number };
  /** 0 = fully retracted into the head capsule, 1 = fully extended */
  proboscis: number;
  proboscisWiggle: number;
  wingL: { flap: number; sweep: number };
  wingR: { flap: number; sweep: number };
  wingOpacity: number;
  haltere: number;
  legs: LegPose[];
  /** 0..1 extra emissive added by the celebrate/recoil overlays */
  reactionGlow: number;
  /** -1 cool down, 0 neutral, +1 warm pulse; drives the reaction tint */
  reactionTint: number;
};

/** Leg segment lengths. Kept here so pose math and mesh layout cannot drift apart. */
export const LEG_LEN = { femur: 0.34, tibia: 0.38, tarsus: 0.18 };

/** Hip attach points, body-local. x is the left side (+X), mirrored for the right. */
export const LEG_ROWS = [
  { z: 0.3, x: 0.29, y: -0.1, name: 'fore' },
  { z: 0.02, x: 0.32, y: -0.1, name: 'mid' },
  { z: -0.28, x: 0.3, y: -0.1, name: 'hind' },
] as const;

/** Leg index 0..2 = left fore/mid/hind, 3..5 = right fore/mid/hind. */
export function legSide(i: number): number {
  return i < 3 ? 1 : -1;
}

export function legRow(i: number): number {
  return i % 3;
}

/**
 * Root height that puts the tarsi on y = 0 with the neutral leg pose below.
 *
 * The chain is a Z-shaped insect leg: femur points forward-down, tibia points back-down,
 * tarsus lies nearly flat on the ground. Direction of a segment with chain angle `a`
 * (measured from straight down, positive = backward) is (0, -cos a, -sin a), and a hip
 * splay roll r scales the vertical drop by cos(r).
 *
 * drop = cos(r) * (Lfemur*cos(f) + Ltibia*cos(f+k) + Ltarsus*cos(f+k+t))
 *      = 0.813878 * (0.34*cos(0.45) + 0.38*cos(0.85) + 0.18*cos(0.20))
 *      = 0.813878 * 0.733357
 *      = 0.596848
 * the hip sits at body-local y = -0.10, so the foot is 0.696848 below the body origin.
 */
export const BODY_BASE_Y = 0.6968;

const NEUTRAL = { femur: -0.45, knee: 1.3, tarsus: -1.05, roll: 0.62 };

/** Tripod gait: left fore, left hind and right mid step together, the other three follow. */
export function gaitPhase(i: number): number {
  const left = i < 3;
  const row = i % 3;
  const groupA = (left && row !== 1) || (!left && row === 1);
  return groupA ? 0 : Math.PI;
}

function baseLeg(i: number, out: LegPose): void {
  const side = legSide(i);
  const row = legRow(i);
  // forelegs splay forward, hind legs back, mid legs out to the side
  const rowYaw = row === 0 ? -0.34 : row === 1 ? 0.02 : 0.36;
  out.hipYaw = side * rowYaw;
  out.hipRoll = side * NEUTRAL.roll;
  out.femur = NEUTRAL.femur;
  out.knee = NEUTRAL.knee;
  out.tarsus = NEUTRAL.tarsus;
}

/**
 * The neutral pose of leg `i`, exported so the real-mesh rig can pose the asset relative
 * to it. The authored rest pose of a static mesh already encodes a standing leg, so the
 * real rig applies deviations from this rather than absolute angles: that is what keeps
 * the two paths animating with the same amplitudes from one source of truth.
 */
export function legNeutral(i: number, out: LegPose): void {
  baseLeg(i, out);
}

/**
 * Crossfade between two fully computed poses. Used so a behavior change (say walk to
 * groom) never snaps: both behaviors are evaluated at the same time `t` and interpolated.
 */
export function lerpPose(out: Pose, a: Pose, b: Pose, w: number): Pose {
  const lerp = (x: number, y: number) => x + (y - x) * w;
  out.rootY = lerp(a.rootY, b.rootY);
  out.rootZ = lerp(a.rootZ, b.rootZ);
  out.rootYaw = lerp(a.rootYaw, b.rootYaw);
  out.bodyPitch = lerp(a.bodyPitch, b.bodyPitch);
  out.bodyRoll = lerp(a.bodyRoll, b.bodyRoll);
  out.bodyYaw = lerp(a.bodyYaw, b.bodyYaw);
  out.bodyY = lerp(a.bodyY, b.bodyY);
  out.headPitch = lerp(a.headPitch, b.headPitch);
  out.headYaw = lerp(a.headYaw, b.headYaw);
  for (let s = 0; s < 4; s += 1) {
    out.abdPitch[s] = lerp(a.abdPitch[s], b.abdPitch[s]);
    out.abdScale[s] = lerp(a.abdScale[s], b.abdScale[s]);
  }
  out.antL.yaw = lerp(a.antL.yaw, b.antL.yaw);
  out.antL.pitch = lerp(a.antL.pitch, b.antL.pitch);
  out.antR.yaw = lerp(a.antR.yaw, b.antR.yaw);
  out.antR.pitch = lerp(a.antR.pitch, b.antR.pitch);
  out.proboscis = lerp(a.proboscis, b.proboscis);
  out.proboscisWiggle = lerp(a.proboscisWiggle, b.proboscisWiggle);
  out.wingL.flap = lerp(a.wingL.flap, b.wingL.flap);
  out.wingL.sweep = lerp(a.wingL.sweep, b.wingL.sweep);
  out.wingR.flap = lerp(a.wingR.flap, b.wingR.flap);
  out.wingR.sweep = lerp(a.wingR.sweep, b.wingR.sweep);
  out.wingOpacity = lerp(a.wingOpacity, b.wingOpacity);
  out.haltere = lerp(a.haltere, b.haltere);
  for (let i = 0; i < 6; i += 1) {
    const la = a.legs[i];
    const lb = b.legs[i];
    const lo = out.legs[i];
    lo.hipYaw = lerp(la.hipYaw, lb.hipYaw);
    lo.hipRoll = lerp(la.hipRoll, lb.hipRoll);
    lo.femur = lerp(la.femur, lb.femur);
    lo.knee = lerp(la.knee, lb.knee);
    lo.tarsus = lerp(la.tarsus, lb.tarsus);
  }
  // reaction overlays are additive in both inputs, so interpolating them keeps the hit
  out.reactionGlow = lerp(a.reactionGlow, b.reactionGlow);
  out.reactionTint = lerp(a.reactionTint, b.reactionTint);
  return out;
}

/** Hermite smoothstep on 0..1. */
export function smoothstep(x: number): number {
  const u = x < 0 ? 0 : x > 1 ? 1 : x;
  return u * u * (3 - 2 * u);
}

export function createPose(): Pose {
  const legs: LegPose[] = [];
  for (let i = 0; i < 6; i += 1) {
    const l: LegPose = { hipYaw: 0, hipRoll: 0, femur: 0, knee: 0, tarsus: 0 };
    baseLeg(i, l);
    legs.push(l);
  }
  return {
    rootY: BODY_BASE_Y,
    rootZ: 0,
    rootYaw: 0,
    bodyPitch: 0,
    bodyRoll: 0,
    bodyYaw: 0,
    bodyY: 0,
    headPitch: 0,
    headYaw: 0,
    abdPitch: [0, 0, 0, 0],
    abdScale: [1, 1, 1, 1],
    antL: { yaw: 0, pitch: 0 },
    antR: { yaw: 0, pitch: 0 },
    proboscis: 0,
    proboscisWiggle: 0,
    wingL: { flap: 0, sweep: 0 },
    wingR: { flap: 0, sweep: 0 },
    wingOpacity: 0.34,
    haltere: 0,
    legs,
    reactionGlow: 0,
    reactionTint: 0,
  };
}

/** Two-pulse antenna twitch every `period` seconds, plus a slow sway. */
function twitch(t: number, period: number, phase: number): number {
  const u = ((t / period + phase) % 1 + 1) % 1;
  // The square is applied to the scaled offset, then negated: exp(-(x^2)).
  // Writing it as -x ** 2 is a syntax error in JS/TS (unary minus binds looser
  // than **), which is why the negation sits outside the parentheses.
  const pulseA = Math.exp(-(((u - 0.08) * 26) ** 2));
  const pulseB = Math.exp(-(((u - 0.2) * 26) ** 2));
  return pulseA + 0.7 * pulseB;
}

export type ReactState = { kind: ReactionKind; age: number };

/**
 * Fill `pose` for time `t` (seconds) in behavior `behavior`, blended in over `blend`
 * (0 = pure idle, 1 = fully in the behavior), with an optional reaction overlay.
 */
export function computePose(
  pose: Pose,
  t: number,
  behavior: Behavior,
  blend: number,
  react: ReactState,
): Pose {
  const b = blend < 0 ? 0 : blend > 1 ? 1 : blend;
  const breath = Math.sin(t * 2.1);
  const sway = Math.sin(t * 0.7);

  // ---- neutral reset -------------------------------------------------------
  pose.rootY = BODY_BASE_Y;
  pose.rootZ = 0;
  pose.rootYaw = 0;
  pose.bodyPitch = 0;
  pose.bodyRoll = 0;
  pose.bodyYaw = 0;
  pose.bodyY = 0;
  pose.headPitch = 0;
  pose.headYaw = 0;
  pose.proboscis = 0;
  pose.proboscisWiggle = 0;
  pose.wingOpacity = 0.34;
  pose.haltere = 0;
  pose.reactionGlow = 0;
  pose.reactionTint = 0;
  for (let i = 0; i < 6; i += 1) baseLeg(i, pose.legs[i]);
  for (let s = 0; s < 4; s += 1) {
    pose.abdPitch[s] = 0;
    pose.abdScale[s] = 1;
  }

  // ---- idle layer, always present and always breathing ---------------------
  pose.bodyY += 0.011 * breath;
  pose.bodyPitch += 0.022 * breath;
  pose.headPitch += 0.035 * Math.sin(t * 2.1 + 0.6);
  pose.headYaw += 0.05 * Math.sin(t * 0.43);
  pose.abdScale[0] = 1 + 0.016 * Math.sin(t * 2.1 + 0.3);
  pose.abdScale[1] = 1 + 0.018 * Math.sin(t * 2.1 + 0.5);
  pose.abdScale[2] = 1 + 0.02 * Math.sin(t * 2.1 + 0.7);
  pose.abdScale[3] = 1 + 0.022 * Math.sin(t * 2.1 + 0.9);
  pose.abdPitch[0] = 0.05 * Math.sin(t * 1.05);
  pose.abdPitch[1] = 0.05 * Math.sin(t * 1.05 + 0.25);
  pose.abdPitch[2] = 0.05 * Math.sin(t * 1.05 + 0.5);
  pose.abdPitch[3] = 0.05 * Math.sin(t * 1.05 + 0.75);
  // resting wings: folded back over the abdomen, breathing with the body
  pose.wingL.flap = 0.32 + 0.03 * Math.sin(t * 2.1 + 0.2);
  pose.wingR.flap = pose.wingL.flap;
  pose.wingL.sweep = -0.16 + 0.06 * sway;
  pose.wingR.sweep = pose.wingL.sweep;
  // halteres beat constantly, they are the gyroscopes
  pose.haltere = 0.5 + 0.42 * Math.sin(t * 34);

  const antTw = twitch(t, 2.6, 0);
  pose.antL.yaw = 0.16 + 0.12 * Math.sin(t * 0.9) + 0.34 * antTw;
  pose.antL.pitch = -0.3 + 0.06 * Math.sin(t * 1.7) - 0.5 * antTw;
  pose.antR.yaw = 0.16 + 0.12 * Math.sin(t * 0.9 + 0.4) + 0.34 * twitch(t, 2.6, 0.5);
  pose.antR.pitch = -0.3 + 0.06 * Math.sin(t * 1.7 + 0.4) - 0.5 * twitch(t, 2.6, 0.5);

  // gentle idle foot pressure, so the legs are never frozen
  for (let i = 0; i < 6; i += 1) {
    const l = pose.legs[i];
    const w = Math.sin(t * 1.3 + i * 1.1);
    l.knee += 0.02 * w;
    l.hipYaw += 0.02 * Math.sin(t * 0.9 + i * 0.7);
  }

  // ---- behavior layer ------------------------------------------------------
  if (behavior === 'walk' && b > 0) {
    const stepHz = 3.1;
    const p0 = t * stepHz * Math.PI * 2;
    pose.bodyY += 0.016 * Math.sin(p0 * 2) * 0.5 * b;
    pose.bodyPitch += -0.05 * b;
    pose.rootZ += 0.055 * Math.sin(p0) * b;
    pose.rootYaw += 0.07 * Math.sin(p0) * b;
    pose.bodyRoll += 0.05 * Math.sin(p0) * b;
    for (let i = 0; i < 6; i += 1) {
      const l = pose.legs[i];
      const side = legSide(i);
      const p = p0 + gaitPhase(i);
      const swing = Math.sin(p);
      const lift = Math.max(0, Math.sin(p + Math.PI * 0.5));
      const liftCurve = lift * lift * (3 - 2 * lift); // smoothstep, keeps stance flat
      // step forward and back: hip yaw negative swings the left leg forward (+Z)
      l.hipYaw += -side * 0.5 * swing * b;
      // fold the knee to pick the foot up during swing
      l.knee += (0.62 * liftCurve) * b;
      l.femur += -0.16 * liftCurve * b;
      l.tarsus += 0.5 * liftCurve * b;
    }
    pose.headYaw += 0.09 * Math.sin(p0 * 0.5) * b;
    pose.wingL.flap += 0.08 * Math.sin(p0 * 2) * b;
    pose.wingR.flap = pose.wingL.flap;
    pose.antL.pitch += -0.16 * b;
    pose.antR.pitch += -0.16 * b;
  } else if (behavior === 'groom' && b > 0) {
    // forelegs come up over the head and wipe, head tips down to meet them
    pose.bodyPitch += 0.12 * b;
    pose.bodyY += -0.03 * b;
    pose.headPitch += 0.42 * b;
    const rub = Math.sin(t * 13);
    const rub2 = Math.sin(t * 13 + 1.2);
    const fore = [0, 3];
    for (let k = 0; k < fore.length; k += 1) {
      const i = fore[k];
      const l = pose.legs[i];
      const side = legSide(i);
      l.femur += -1.72 * b + 0.12 * (k === 0 ? rub : rub2) * b;
      l.knee += 1.0 * b + 0.22 * (k === 0 ? rub2 : rub) * b;
      l.tarsus += 0.5 * b;
      l.hipYaw += -side * 0.34 * b;
      l.hipRoll += -side * 0.1 * b;
    }
    pose.antL.pitch += -0.55 * b;
    pose.antR.pitch += -0.55 * b;
    pose.antL.yaw += 0.3 * b;
    pose.antR.yaw += 0.3 * b;
    pose.wingL.flap += 0.06 * b;
    pose.wingR.flap += 0.06 * b;
  } else if (behavior === 'proboscis' && b > 0) {
    // probe: head down, proboscis out and swinging side to side
    const ext = Math.min(1, b) * (0.85 + 0.15 * Math.sin(t * 1.6));
    pose.proboscis = ext;
    pose.proboscisWiggle = 0.5 * Math.sin(t * 5.2) + 0.3 * Math.sin(t * 11.7);
    pose.headPitch += 0.34 * b;
    pose.bodyPitch += 0.06 * b;
    pose.bodyY += -0.02 * b;
    pose.antL.pitch += -0.2 * b;
    pose.antR.pitch += -0.2 * b;
    // forelegs brace slightly outward while probing
    for (const i of [0, 3]) {
      pose.legs[i].hipYaw += -legSide(i) * 0.16 * b;
    }
  } else if (behavior === 'startle' && b > 0) {
    // hop arc, repeating every 1.9 s while this behavior is selected
    const u = (t % 1.9) / 1.9;
    if (u < 0.62) {
      const arc = Math.sin(Math.PI * (u / 0.62));
      pose.rootY += 0.3 * arc * b;
      pose.rootZ += 0.1 * arc * b;
    }
    pose.legs.forEach((l) => {
      l.knee += 0.75 * b;
      l.femur += -0.2 * b;
    });
    pose.wingL.flap += 1.15 * b;
    pose.wingR.flap += 1.15 * b;
    pose.headPitch += -0.25 * b;
    pose.antL.pitch += -0.7 * b;
    pose.antR.pitch += -0.7 * b;
    pose.abdPitch[3] += 0.2 * b;
  }

  // ---- reaction overlay ----------------------------------------------------
  if (react.kind === 'celebrate' && react.age < 1.3) {
    const u = react.age / 1.3;
    const env = Math.sin(Math.PI * u); // smooth in and out
    const flick = Math.sin(u * Math.PI * 8);
    // clean wing raise plus a small confident hop
    pose.wingL.flap += (0.85 + 0.35 * flick) * env;
    pose.wingR.flap = pose.wingL.flap;
    pose.wingL.sweep += 0.3 * env;
    pose.wingR.sweep += 0.3 * env;
    pose.wingOpacity += 0.3 * env;
    pose.rootY += 0.14 * Math.sin(Math.PI * Math.min(1, u * 1.15)) * env;
    pose.bodyPitch += -0.16 * env;
    pose.abdPitch[0] += -0.1 * env;
    pose.headPitch += -0.12 * env;
    pose.antL.pitch += -0.6 * env;
    pose.antR.pitch += -0.6 * env;
    pose.reactionGlow = env;
    pose.reactionTint = 1;
  } else if (react.kind === 'recoil' && react.age < 0.85) {
    const u = react.age / 0.85;
    const env = Math.exp(-3.2 * u) * (1 - Math.exp(-26 * u)); // sharp hit, quick settle
    // brace and lean back, wings half open as if arrested mid-start
    pose.bodyPitch += -0.42 * env;
    pose.bodyY += -0.05 * env;
    pose.abdPitch[0] += 0.22 * env;
    pose.abdPitch[1] += 0.16 * env;
    pose.abdPitch[2] += 0.1 * env;
    pose.headPitch += 0.3 * env;
    pose.wingL.flap += 0.42 * env;
    pose.wingR.flap += 0.42 * env;
    pose.wingL.sweep += -0.32 * env;
    pose.wingR.sweep += -0.32 * env;
    pose.legs.forEach((l, i) => {
      l.knee += 0.3 * env * (1 + 0.3 * (i % 2));
      l.hipRoll *= 1 + 0.14 * env;
    });
    pose.antL.pitch += -0.85 * env;
    pose.antR.pitch += -0.85 * env;
    pose.reactionGlow = env;
    pose.reactionTint = -1;
  }

  return pose;
}

/**
 * Autonomous behavior schedule for when nobody is driving: idle, then a few seconds of
 * walking, a quick groom, a probe, and back. Cycle length 26 s. Returns the behavior and
 * how long it has been running, so the caller can blend.
 */
export const AUTO_SCHEDULE: Array<{ behavior: Behavior; seconds: number }> = [
  { behavior: 'idle', seconds: 7 },
  { behavior: 'walk', seconds: 6 },
  { behavior: 'idle', seconds: 3 },
  { behavior: 'groom', seconds: 4 },
  { behavior: 'proboscis', seconds: 4 },
  { behavior: 'idle', seconds: 2 },
];

export const AUTO_PERIOD = AUTO_SCHEDULE.reduce((a, s) => a + s.seconds, 0);

export function autoBehavior(t: number): { behavior: Behavior; elapsed: number; remaining: number } {
  let u = t % AUTO_PERIOD;
  for (const step of AUTO_SCHEDULE) {
    if (u < step.seconds) return { behavior: step.behavior, elapsed: u, remaining: step.seconds - u };
    u -= step.seconds;
  }
  return { behavior: 'idle', elapsed: 0, remaining: 1 };
}
