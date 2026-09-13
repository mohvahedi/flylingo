/**
 * The reach: the moment the fly arrives on the answer it picked and puts its forelegs down on
 * it.
 *
 * Why this is its own module rather than another branch in pose.ts: the reach is the only
 * behavior that moves the fly's BODY as well as its legs. The legs are posed (pose.ts), but
 * the body has to lean in toward the glass and settle onto the card, and the fly's body
 * transform belongs to whoever is flying it (PhoneStage). Both halves have to use the same
 * envelope and the same numbers, so they live here and both import them.
 *
 * Measured facts this is built around (see tools/reach_analyse.py):
 *   - the fly stands in front of the glass with its tarsi spread quite wide along the screen
 *     normal, so the two forelegs start at very different distances from the glass: bringing
 *     them both onto the surface needs the body to come toward the card as well as the legs to
 *     extend;
 *   - the pose vocabulary swings a leg in the plane of its own splay (femur / knee / tarsus /
 *     hipRoll) and swings it fore and aft with hipYaw, so "down and forward" is a femur/knee
 *     extension plus a hipYaw, and lateral travel toward the glass comes from hipRoll applied
 *     with opposite sign to the two sides, which is how groom moves both forelegs together.
 */

export type ReachTune = {
  /** seconds of the extension before the tarsi land */
  extend: number;
  /** seconds planted on the surface */
  hold: number;
  /** seconds easing back to the resting stance */
  settle: number;
  /** foreleg bone rotations at full extension, radians (added to the rest pose) */
  femur: number;
  knee: number;
  tarsus: number;
  hipYaw: number;
  hipRoll: number;
  /** body: nose down a touch while leaning in */
  bodyPitch: number;
  /** body roll toward the glass, which carries the legs closer to it */
  bodyRoll: number;
  /** how far the whole fly moves toward the glass at full lean, world units */
  lean: number;
  /** how far the body settles toward the card, world units, down the image */
  drop: number;
  /** the small press at the end of the hold: amplitude of the tap, radians */
  tap: number;
  /** tap rate, radians per second */
  tapHz: number;
  /** how much the head comes down with the reach */
  headPitch: number;
};

/**
 * Tuned against measurements, not by eye: each of these was swept in the browser while
 * tools/reach_measure.py watched the perpendicular distance from the fore tarsi to the glass
 * (see the commit message / tools/reach_analyse.py output).
 */
export const REACH: ReachTune = {
  extend: 0.36,
  hold: 1.15,
  settle: 0.55,
  femur: -0.55,
  knee: -0.35,
  tarsus: 0.3,
  hipYaw: -0.18,
  hipRoll: 0.55,
  bodyPitch: 0.16,
  // Level. Measured with this at 0.1: the bank put the two fore tarsi about 56 screen units
  // apart in height, so one sat on the card band and the other just below it, and only one of
  // them ever held the answer. Level, both land inside the band.
  bodyRoll: 0.0,
  // Scaled for an 8.6 unit specimen. These are world-space offsets, so at the old 1.35 they
  // were tuned for a body six times smaller and would barely move the tarsi now. The lean is
  // what carries the forelegs the last of the way onto the glass, so it has to be sized to the
  // gap it has to close.
  lean: 1.30,
  drop: 0.30,
  tap: 0.07,
  tapHz: 9,
  headPitch: 0.3,
};

/** Total seconds the reach runs for, so the caller knows when to hand over. */
export function reachSeconds(tune: ReachTune = REACH): number {
  return tune.extend + tune.hold + tune.settle;
}

export type ReachState = {
  /** 0..1 extension of the legs; 1 = tarsi planted on the card */
  on: number;
  /** 1 through the hold, easing to 0 as the fly settles */
  planted: number;
  /** the extra press at the end of the hold, -1..1, for a small tap */
  tap: number;
};

/** The reach doing nothing. Shared so a frame that is not reaching allocates nothing. */
export const ZERO_REACH: ReachState = { on: 0, planted: 0, tap: 0 };

const smooth = (u: number) => {
  const x = u < 0 ? 0 : u > 1 ? 1 : u;
  return x * x * (3 - 2 * x);
};

/**
 * The reach envelope as a function of how long the behavior has been running. Pure, so the
 * leg pose and the body lean cannot drift apart.
 */
export function reachCurve(age: number, tune: ReachTune = REACH): ReachState {
  if (!(age > 0)) return { on: 0, planted: 0, tap: 0 };
  const land = tune.extend;
  const off = tune.extend + tune.hold;
  if (age < land) return { on: smooth(age / land), planted: 0, tap: 0 };
  if (age < off) {
    const t = age - land;
    // a slow press that fades in: the fly is not drumming, it is leaning on the answer
    const tap = Math.sin(t * tune.tapHz) * smooth(t / 0.35) * (1 - smooth((t - tune.hold + 0.3) / 0.3));
    return { on: 1, planted: 1, tap };
  }
  const u = (age - off) / tune.settle;
  const k = 1 - smooth(u);
  return { on: k, planted: k, tap: 0 };
}
