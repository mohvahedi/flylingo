/**
 * The animator: one hook that owns all mutable animation state and advances it every
 * frame inside the react-three-fiber render loop.
 *
 * Why a hook returning refs instead of React state: pose updates run at 60 fps and touch
 * ~40 numbers. Routing that through setState would re-render the tree every frame. Instead
 * the animator mutates plain objects and the meshes read them in their own useFrame
 * callbacks. React state is used only for things a human triggers (behavior override).
 */
import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import {
  CHANNEL_COUNT,
  createChannelValues,
  readChannels,
  syntheticFrame,
  vitality,
  type ChannelValues,
} from './regions';
import {
  autoBehavior,
  computePose,
  createPose,
  lerpPose,
  smoothstep,
  type Behavior,
  type Pose,
  type ReactionKind,
  type ReactState,
} from './pose';

/** Crossfade duration between behaviors, seconds. */
export const BLEND_TIME = 0.45;

/** How long each transient reaction runs, seconds. */
const REACT_SECONDS: Record<ReactionKind, number> = {
  none: 0,
  celebrate: 1.3,
  recoil: 0.85,
};

/**
 * How much the vitality reading must jump in one frame to read as a startle. A real
 * network at 20 Hz does not move this far in one step, so this fires on genuine bursts
 * (and never under no_edges, where vitality is pinned to 0).
 */
const STARTLE_JUMP = 0.2;

export type AnimatorInput = {
  /** raw state from the app, or null/undefined when no frame has arrived */
  activity: number[] | null | undefined;
  /** 0..1 measures of how awake the network is */
  stateRms: number;
  activeFraction: number;
  /** manual behavior override from the dev strip, or null to follow the auto schedule */
  override: Behavior | null;
  /** a transient reaction triggered by a `correct` transition */
  reaction: { kind: ReactionKind; seq: number };
  /** pause everything (used by the dev strip freeze button) */
  paused: boolean;
  /** extra time scaling, 1 = real time */
  timeScale: number;
};

/** Mutable live readout, safe to read from any useFrame and from the DOM overlays. */
export type LiveReadout = {
  /** which behavior is actually playing right now */
  behavior: Behavior;
  /** whether that behavior came from the auto schedule or a human */
  source: 'auto' | 'manual';
  /** elapsed seconds of animator time */
  t: number;
  /** how long the behavior now playing has been running, seconds */
  age: number;
  /** vitality 0..1, the same number that drives the global glow floor */
  vitality: number;
  /** per-frame normalization reference actually used (p95, clamped) */
  reference: number;
  /** true when the frame carries no usable signal (no_edges) */
  inert: boolean;
  /** true when the harness is running on synthetic activity, not app frames */
  synthetic: boolean;
  /** smoothed stateRms, raw (synthetic value when no frame is supplied) */
  rms: number;
};

export type AnimatorOutput = {
  pose: Pose;
  /** smoothed per-channel drive, length CHANNEL_COUNT */
  drive: Float32Array;
  /** live readout object, mutated in place every frame */
  live: LiveReadout;
};

export function useAnimator(input: AnimatorInput) {
  // live mirror of the props so useFrame never closes over a stale value
  const inRef = useRef(input);
  inRef.current = input;

  const pose = useMemo(createPose, []);
  const poseA = useMemo(createPose, []);
  const poseB = useMemo(createPose, []);
  const drive = useMemo(() => new Float32Array(CHANNEL_COUNT), []);
  const channels = useMemo<ChannelValues>(createChannelValues, []);
  const synthBuf = useMemo(() => new Array<number>(512).fill(0), []);
  const live = useMemo<LiveReadout>(
    () => ({
      behavior: 'idle',
      source: 'auto',
      t: 0,
      age: 0,
      vitality: 0,
      reference: 0,
      inert: true,
      synthetic: true,
      rms: 0,
    }),
    [],
  );

  const react = useRef<ReactState>({ kind: 'none', age: 0 });
  const lastSeq = useRef(-1);
  const curBehavior = useRef<Behavior>('idle');
  const blend = useRef(1);
  const prevBehavior = useRef<Behavior>('idle');
  /** how long the current (and, across a crossfade, the previous) behavior has been running */
  const behaviorAge = useRef(0);
  const prevAge = useRef(0);
  const ownTime = useRef(0);
  const vital = useRef(0);
  const prevVital = useRef(0);
  const startleUntil = useRef(-1);

  useFrame((_, rawDelta) => {
    const cfg = inRef.current;
    const delta = Math.min(0.05, Math.max(0, rawDelta)) * (cfg.timeScale || 0);
    if (!cfg.paused) ownTime.current += delta;
    const t = ownTime.current;

    // ---- 1. bin the connectome state into anatomical channels ---------------
    const hasFrame = Array.isArray(cfg.activity) && cfg.activity.length > 0;
    let rms: number;
    if (hasFrame) {
      readChannels(cfg.activity, channels);
      live.synthetic = false;
      rms = cfg.stateRms || 0;
    } else {
      // No props at all: synthesize a live-looking idle drive so the harness is never a
      // frozen mesh before the socket connects. Same magnitude regime as the real data.
      const s = syntheticFrame(t, synthBuf);
      readChannels(s.state, channels);
      live.synthetic = true;
      rms = s.stateRms;
    }
    // vitality: falls to exactly 0 for an inert frame (no_edges), so nothing amplifies it
    const v = vitality(
      channels,
      hasFrame ? cfg.stateRms || 0 : rms,
      hasFrame ? cfg.activeFraction || 0 : 0,
    );
    live.reference = channels.reference;
    live.inert = channels.inert;
    if (!cfg.paused) {
      // fast attack, slower release: spikes are visible, decay is smooth
      const vTarget = hasFrame && channels.inert ? 0 : v;
      vital.current += (vTarget - vital.current) * (vTarget > vital.current ? 0.3 : 0.1);
      for (let c = 0; c < CHANNEL_COUNT; c += 1) {
        const k = channels.drive[c] > drive[c] ? 0.34 : 0.12;
        drive[c] += (channels.drive[c] - drive[c]) * k;
      }
    }

    // A sharp jump in how much of the network is firing reads as a startle: the fly hops.
    // This is the one behavior the connectome itself can trigger, which is what makes the
    // rig feel coupled to the simulation rather than to a timer. Under no_edges vitality is
    // pinned to 0 and the jump is always 0, so an inert fly never startles.
    const jump = vital.current - prevVital.current;
    prevVital.current = vital.current;
    if (!cfg.paused && jump > STARTLE_JUMP && t > startleUntil.current) {
      startleUntil.current = t + 1.4;
    }

    // ---- 2. reaction bookkeeping -------------------------------------------
    if (cfg.reaction.seq !== lastSeq.current) {
      lastSeq.current = cfg.reaction.seq;
      if (cfg.reaction.kind !== 'none') {
        react.current.kind = cfg.reaction.kind;
        react.current.age = 0;
      }
    }
    if (react.current.kind !== 'none') {
      if (!cfg.paused) react.current.age += delta;
      if (react.current.age > REACT_SECONDS[react.current.kind]) react.current.kind = 'none';
    }

    // ---- 3. pick the behavior and crossfade --------------------------------
    let want: Behavior;
    let source: 'auto' | 'manual';
    if (cfg.override) {
      want = cfg.override;
      source = 'manual';
    } else if (t < startleUntil.current) {
      want = 'startle';
      source = 'auto';
    } else {
      want = autoBehavior(t).behavior;
      source = 'auto';
    }
    if (want !== curBehavior.current) {
      prevBehavior.current = curBehavior.current;
      curBehavior.current = want;
      blend.current = 0;
      prevAge.current = behaviorAge.current;
      behaviorAge.current = 0;
    }
    if (!cfg.paused) {
      behaviorAge.current += delta;
      prevAge.current += delta;
    }
    if (blend.current < 1) {
      blend.current = Math.min(1, blend.current + delta / BLEND_TIME);
    }

    // both behaviors are evaluated at the same time t and interpolated, so a behavior
    // change never snaps the fly's limbs. Each is evaluated at its own age: a one-shot
    // behavior leaving the stage has to stay at the end of its envelope while it fades out,
    // not restart at the beginning of it.
    if (blend.current >= 1) {
      computePose(pose, t, curBehavior.current, 1, react.current, behaviorAge.current);
    } else {
      computePose(poseA, t, prevBehavior.current, 1, react.current, prevAge.current);
      computePose(poseB, t, curBehavior.current, 1, react.current, behaviorAge.current);
      lerpPose(pose, poseA, poseB, smoothstep(blend.current));
    }

    // ---- 4. publish the live readout ---------------------------------------
    live.behavior = curBehavior.current;
    live.source = source;
    live.t = t;
    live.age = behaviorAge.current;
    live.vitality = vital.current;
    // the sparkline tracks raw stateRms; smoothed so a 20 Hz feed does not jitter
    live.rms += (rms - live.rms) * 0.2;
  });

  return { pose, drive, live } satisfies AnimatorOutput;
}
