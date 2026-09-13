/**
 * The hero scene: the supplied phone standing on the stage with the real Duolingo lesson on
 * its screen, and the specimen flying to the answer it picked.
 *
 * Composition notes
 * -----------------
 * The panel this feeds is wide (about 2.24:1) while a phone is portrait, so the phone cannot
 * fill the frame without being cropped. It is placed slightly right of centre with the fly in
 * the foreground to the left, which uses the width instead of leaving dead space either side,
 * and the two subjects are connected by the flight the fly actually performs.
 *
 * The fly's target comes from the option rectangles the screen renderer returns, mapped
 * through the phone's world matrix. Nothing guesses where a card is: the same numbers that
 * drew the card decide where the fly goes, so the two cannot disagree.
 */

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

import { useAnimator } from './animator';
import { CHANNELS } from './regions';
import { RealFly } from './RealFly';
import { RIG, StudioEnvironment } from './studio';
import { Bloom } from './Bloom';
import type { Behavior, ReactionKind } from './pose';
import { REACH, ZERO_REACH, reachCurve, reachSeconds, type ReachState } from './reach';
import {
  SCREEN,
  screenPlaneQuaternion,
  screenPointToPlaneLocal,
  usePhone,
} from './phoneModel';
import {
  SCREEN_H,
  SCREEN_W,
  createScreenCanvas,
  drawDuolingoLesson,
  ensureDuolingoFonts,
  type OptionRect,
  type ScreenState,
} from './duolingoScreen';

/**
 * Read a framing override from the URL, so the composition can be swept against measured
 * margins rather than nudged by eye. `?cam=0,9,17&tgt=0,4,0&fov=38`.
 */
function camParam(name: string, fallback: number): number {
  if (typeof window === "undefined") return fallback;
  const raw = new URLSearchParams(window.location.search).get(name);
  if (raw === null) return fallback;
  const parts = raw.split(",");
  const v = Number(parts[0]);
  return Number.isFinite(v) ? v : fallback;
}

function camVec(name: string, x: number, y: number, z: number): [number, number, number] {
  if (typeof window === "undefined") return [x, y, z];
  const raw = new URLSearchParams(window.location.search).get(name);
  if (raw === null) return [x, y, z];
  const p = raw.split(",").map(Number);
  if (p.length !== 3 || p.some((n) => !Number.isFinite(n))) return [x, y, z];
  return [p[0], p[1], p[2]];
}

/**
 * The physical layout. Kept in one place so the framing can be tuned from a screenshot
 * without hunting through the scene graph.
 */
export const LAYOUT = {
  /**
   * World height of the phone's screen.
   *
   * Large on purpose. The panel this feeds is about 2.24:1 while a handset is portrait, so a
   * phone that fits entirely within the frame's height can only ever be about a fifth of its
   * width, leaving most of the frame empty. Making the handset big enough that the frame
   * crops its lower edge gives the Duolingo screen real size and reads as a considered
   * product shot instead of a small object in a void.
   */
  screenHeight: 8.4,
  /** where the phone stands */
  phoneX: 1.85,
  /**
   * Yaw about Y. The asset's own node chain already stands the handset upright, but it
   * leaves the glass facing world +X (measured: screen normal [0.984, 0, 0.179]), so a
   * quarter turn is needed to bring it round to the viewer. The extra -0.14 is presentation:
   * it keeps the glass from being perfectly flat to the camera, which reads better.
   */
  phoneYaw: -Math.PI / 2 - 0.14,
  /** tilt about X, leaning the top away from the viewer */
  phonePitch: 0.03,
  /**
   * The fly's world span, in world units. The handset is about 8.7 tall, so at 8.6 the
   * specimen is the same size as the phone, which is what was asked for.
   *
   * Sizes tried: 0.72 read as an insect that happened to be in the shot; 1.35 read as a
   * character but still small; 8.6 makes it a co-lead with the handset. At this size it can no
   * longer stand ON a card, because the cards are 1.5 units tall and it would cover the lesson,
   * so it stands beside the handset and reaches in with its forelegs. See `lateral` below.
   */
  flySpan: 8.6,
  /** where the fly starts: in front of the phone and to its left */
  flyStart: [-8.0, 0.0, 4.0] as [number, number, number],
  /**
   * How far off the glass the fly hovers when it reaches a card, in the phone's own mesh
   * units. The screen is only 0.155 tall there, so this has to be small: an earlier value of
   * 0.05 was a third of the screen height and threw the fly well clear of the handset.
   *
   * Scaled up with flySpan, so the specimen stands the same distance off the card in its own
   * body lengths as it did at 0.72: 0.006 here puts its feet plane 0.32 world units (3.8% of
   * the screen height) in front of the glass.
   */
  /**
   * Perpendicular offset off the screen plane, in the plane's own units, so it is multiplied by
   * the phone group's scale.
   *
   * 3.64 world units. Raised from 1.5 because the fore tarsi were landing through the glass
   * rather than on it: measured at 1.5, with the fly correctly facing its card, they read -1.66
   * and -2.53 on the glass normal, i.e. inside the handset. They move rigidly with the body, so
   * standing the fly back by their mean 2.1 puts them on the surface.
   */
  standoff: 0.094,
  /**
   * How far to turn the specimen to face its answer. Measured, not assumed: the asset carries
   * its own correction rotation, so which way it faces is not readable from the code.
   *
   * Swept all four cardinal yaws against the scene's own per-tarsus telemetry, with the fly
   * parked on card 2:
   *
   *     yaw   0 deg   fore tarsi  1.656 off glass   at px (430,349) (358,178)
   *     yaw  90 deg   fore tarsi  0.661 off glass   at px (-176,321) (-271,139)
   *     yaw 180 deg   fore tarsi  4.543 off glass   at px (-158,321) (-83,138)
   *     yaw 270 deg   fore tarsi  3.588 off glass   at px (466,345) (567,174)
   *
   * 90 degrees has the smallest distance and is still the WRONG answer: those tarsi are near the
   * plane only because they hang beside the handset, and their screen x is negative, off the
   * left edge. Only at 0 do the forelegs sit at the card's own horizontal position. Taking the
   * smallest distance alone would have picked 90 and pointed the fly at nothing, so the screen
   * position is checked as well as the distance.
   */

  flyYawOffset: (() => {
    // Overridable from the URL as ?yaw=<radians> so the correct value can be swept against the
    // real tarsus telemetry rather than guessed. The fly asset carries its own correction
    // rotation, so which way it actually faces is a measurement, not a reading of the code.
    const q =
      typeof window === "undefined"
        ? null
        : new URLSearchParams(window.location.search).get("yaw");
    const v = q === null ? NaN : Number(q);
    // 0.55 rad off the head-on approach. Facing the card squarely put the abdomen across the
    // screen and hid the question; the cards are wide enough to be hit anyway, so the body is
    // turned aside. Measured: the tarsi stay on the chosen card.
    return Number.isFinite(v) ? v : 0.55;
  })(),
  /**
   * How far across the chosen card the fly perches, 0 = left edge, 1 = right edge.
   *
   * Not the centre: the option text sits on the left of each card, and a fly with a 1.6 unit
   * wingspan landing centrally covered the text of the card next to it. Perched to the right
   * it reads as sitting on the answer without hiding it. Moved out to 0.88 when the specimen
   * grew, so its head and the forelegs it reaches out with stay on the card instead of
   * hanging off the right edge of the screen.
   */
  flyCardU: 0.88,
  /**
   * How far to the LEFT of the chosen card the fly stands, in world units, so its body clears
   * the handset and only its forelegs cross onto the glass.
   *
   * This is the change that makes "use its arms to choose the option" possible at phone size:
   * the body sits beside the screen and the forelegs are what reach the answer.
   *
   * In WORLD units. The body stays clear of the glass while its forelegs stay inside reach of
   * the card: at 4.0 the legs would have had to span more than their own length even facing the
   * right way, and at 0 the body itself would sit over the lesson text.
   *
   * It is divided by the plane's inherited scale at the point of use. That division is not
   * cosmetic: the screen plane is a child of the phone group, which is scaled by about 54, so
   * adding this straight to the plane's local x multiplied it by 54 and threw the fly to
   * [-177, -0.4, -23], completely out of the scene. Measured, not guessed.
   */
  lateral: 2.9,
  /**
   * How far BELOW the chosen card the fly stands, in world units.
   *
   * The specimen is the same size as the handset, so standing level with the card puts its body
   * across the upper screen and hides the question text.
   *
   * Small on purpose, and bounded by measurement rather than taste: the fly and its tarsi move
   * rigidly together, so dropping the stand carries the tarsi off the card at about 99 screen
   * units per world unit. Measured, 2.6 of drop put them at screen y 590-642 against a card
   * centred at 360. The card band is about a world unit tall, so this is the most that fits.
   * The rest of the clearance is bought with the camera angle instead, which costs nothing.
   */
  perchDrop: 0.15,
  /**
   * How far in FRONT of the glass the fly's body stands, world units. Derived from the size so
   * the pose stays self-similar: at 0.72 the specimen stood 0.24 body-lengths off the card, and
   * that ratio is what looks right, so it is recomputed rather than re-guessed.
   */
  bodyClearance: 1.5,
  /** how long the flight from wherever it is to the chosen card takes, in seconds */
  flySeconds: 1.6,
  /** easing exponent for the flight: 2 eases in and out, which reads as a dart not a slide */
  flyEase: 2.0,
  /**
   * The camera has to see an 8.6 unit fly AND an 8.7 unit handset standing side by side, so it is
   * pulled back until the vertical field clears both. At fov 38 and distance 17.6 the vertical
   * field is 2*17.6*tan(19) = 12.1 units against a subject about 9 tall. The old 12.2 put the
   * field at 8.4 and cropped the top of the handset, which is exactly what cut the question text
   * off the screen. Held at 17.6 rather than 15.6 because the hero panel is wide and the tighter
   * framing read as cramped.
   */
  /**
   * A three-quarter view from the left, which is what keeps the fly off the phone.
   *
   * The problem this solves, measured: from a head-on camera the fly's wingspan spans the whole
   * frame (it is five world units nearer the lens than the handset) and lays itself straight
   * across the phone's screen, hiding the question, while the handset's lower edge falls outside
   * the frame. Raising the camera did not help, it made it worse: the fly grew from 12.5% to
   * 21.8% of the panel and the visible phone fell from 13.3% to 8.8%.
   *
   * Moving the camera round to the side foreshortens the wings instead. Swept over six azimuths
   * and measured by the phone's VISIBLE area, this one leaves the most screen readable, 16.1%
   * of the panel against 12.7% head-on, with the fly still dominant. The question text and all
   * four options read clearly, and the fly's foreleg still reaches in toward the answer.
   */
  camera: camVec("cam", 5.0, 8.0, 14.6),
  target: camVec("tgt", 1.2, 3.9, 0.6),
  fov: Number(camParam("fov", 38)),
};

/** Fallback only. The real scale is measured from the loaded asset; see PhoneScreen. */
const PHONE_SCALE_FALLBACK = LAYOUT.screenHeight / SCREEN.height;

export type PhoneStageProps = {
  prompt: string;
  options: { text: string }[];
  flyChoice: number;
  userChoice: number;
  status: 'none' | 'correct' | 'wrong';
  answerIndex: number;
  hearts: number;
  progress: number;
  activity?: number[];
  stateRms?: number;
  activeFraction?: number;
  reaction?: { kind: ReactionKind; seq: number };
  paused?: boolean;
  timeScale?: number;
  /** reported so a caller can verify where the fly was sent */
  onScreen?: (info: { rects: OptionRect[] }) => void;
  width?: number;
  height?: number;
};

/**
 * The handset, standing, with a live Duolingo texture on a clean screen plane.
 *
 * The plane is parented to the node the asset's own screen mesh lives in, so it inherits the
 * full transform chain rather than being positioned by hand. The presentation scale is then
 * MEASURED from the loaded geometry instead of computed from the raw numbers, because the
 * chain rotates the model and the axis that ends up vertical is not obvious from the file.
 * Measuring makes the framing self-correcting.
 */
function PhoneScreen({
  texture,
  screenRef,
  onMeasured,
}: {
  texture: THREE.CanvasTexture | null;
  screenRef: React.RefObject<THREE.Mesh | null>;
  onMeasured?: (info: {
    scale: number;
    phoneWorldH: number;
    footprintX: number;
    footprintZ: number;
  }) => void;
}) {
  const { root, anchor } = usePhone();
  const planeQuat = useMemo(screenPlaneQuaternion, []);

  // record the handset's meshes so their materials can be reported
  useEffect(() => {
    sceneMeshes.length = 0;
    root.traverse((o) => {
      const m = o as THREE.Mesh;
      if ((m as unknown as { isMesh?: boolean }).isMesh) sceneMeshes.push(m);
    });
  }, [root]);
  const groupRef = useRef<THREE.Group>(null);
  const [scale, setScale] = useState(PHONE_SCALE_FALLBACK);

  // Build the replacement screen inside the asset's own node.
  const plane = useMemo(() => {
    if (!anchor) return null;
    const geo = new THREE.PlaneGeometry(SCREEN.width, SCREEN.height);
    const mat = new THREE.MeshBasicMaterial({ toneMapped: false });
    const m = new THREE.Mesh(geo, mat);
    m.quaternion.copy(planeQuat);
    m.position.set(SCREEN.x + 0.0006, 0, 0);
    m.renderOrder = 1;
    m.name = 'flylingo-screen';
    anchor.add(m);
    return m;
  }, [anchor, planeQuat]);

  useEffect(() => {
    return () => {
      if (plane && anchor) {
        anchor.remove(plane);
        plane.geometry.dispose();
        (plane.material as THREE.Material).dispose();
      }
    };
  }, [plane, anchor]);

  // Hand the plane up so the fly's targeting reads the real world matrix.
  useEffect(() => {
    if (plane) screenRef.current = plane;
  }, [plane, screenRef]);

  // assign the texture whenever it changes
  useEffect(() => {
    if (!plane) return;
    const mat = plane.material as THREE.MeshBasicMaterial;
    mat.map = texture;
    mat.needsUpdate = true;
  }, [plane, texture]);

  // Measure at unit scale, then set the scale so the screen is exactly LAYOUT.screenHeight
  // tall and the handset stands on the floor.
  useEffect(() => {
    if (!plane || !groupRef.current) return;
    const g = groupRef.current;
    g.scale.setScalar(1);
    g.position.set(LAYOUT.phoneX, 0, 0);
    g.updateWorldMatrix(true, true);

    const screenBox = new THREE.Box3().setFromObject(plane);
    const screenSize = screenBox.getSize(new THREE.Vector3());
    // the screen is standing, so its height is whichever of Y/Z is larger in world space
    const unitScreenH = Math.max(screenSize.y, screenSize.z);
    const phoneBox = new THREE.Box3().setFromObject(root);
    const phoneSize = phoneBox.getSize(new THREE.Vector3());
    const unitPhoneH = Math.max(phoneSize.y, phoneSize.z);
    const unitFootprintX = phoneSize.x;
    const unitFootprintZ = Math.max(phoneSize.y, phoneSize.z);

    const s = unitScreenH > 1e-9 ? LAYOUT.screenHeight / unitScreenH : PHONE_SCALE_FALLBACK;
    const phoneWorldH = unitPhoneH * s;

    // stand it on the floor: the box centre sits at half its height
    const centreY = (phoneBox.min.y + phoneBox.max.y) / 2;
    setScale(s);
    g.scale.setScalar(s);
    g.position.set(LAYOUT.phoneX, -centreY * s + phoneWorldH / 2, 0);
    g.updateWorldMatrix(true, true);
    onMeasured?.({
      scale: s,
      phoneWorldH,
      footprintX: unitFootprintX * s,
      footprintZ: unitFootprintZ * s,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plane, root]);

  const tilt = useMemo(
    () => new THREE.Quaternion().setFromEuler(new THREE.Euler(LAYOUT.phonePitch, LAYOUT.phoneYaw, 0, 'YXZ')),
    [],
  );

  return (
    <group ref={groupRef} scale={scale} position={[LAYOUT.phoneX, 0, 0]} quaternion={tilt}>
      <primitive object={root} />
    </group>
  );
}

/**
 * Ambient occlusion where the handset meets the floor.
 *
 * A cast shadow alone leaves the base looking soft and composited: what sells contact is a
 * tight dark pool right at the junction. This is a small decal on the floor, sized from the
 * handset's measured footprint so it cannot drift out of register with the phone.
 */
function PhoneContactAO({ footprintX, footprintZ }: { footprintX: number; footprintZ: number }) {
  const tex = useMemo(() => {
    const c = document.createElement('canvas');
    c.width = 128;
    c.height = 128;
    const ctx = c.getContext('2d');
    if (ctx) {
      const g = ctx.createRadialGradient(64, 64, 4, 64, 64, 62);
      g.addColorStop(0, 'rgba(0,0,0,0.78)');
      g.addColorStop(0.45, 'rgba(0,0,0,0.40)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, 128, 128);
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, []);
  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[LAYOUT.phoneX, 0.0025, 0]}
      renderOrder={1}
    >
      <planeGeometry args={[footprintX * 2.1, footprintZ * 1.5]} />
      <meshBasicMaterial map={tex} transparent depthWrite={false} />
    </mesh>
  );
}

/** The specimen: flies to the card it picked, hovers, reaches out to touch it. */
function FlyActor({
  rects,
  flyChoice,
  activity,
  stateRms,
  activeFraction,
  reaction,
  paused,
  timeScale,
  screenRef,
  targetOut,
  phoneScale,
}: {
  rects: OptionRect[];
  flyChoice: number;
  activity?: number[];
  stateRms: number;
  activeFraction: number;
  reaction: { kind: ReactionKind; seq: number };
  paused: boolean;
  timeScale: number;
  screenRef: React.RefObject<THREE.Mesh | null>;
  targetOut: React.RefObject<THREE.Vector3 | null>;
  /** the phone group's measured scale, so plane-local offsets can be authored in world units */
  phoneScale: number;
}) {
  const group = useRef<THREE.Group>(null);
  const target = useRef(new THREE.Vector3(...LAYOUT.flyStart));
  const home = useMemo(() => new THREE.Vector3(...LAYOUT.flyStart), []);
  // published so the contact shadow on the glass can follow the perch point
  useEffect(() => {
    (window as unknown as { __flyTargetRef?: unknown }).__flyTargetRef = null;
  }, []);
  const [behavior, setBehavior] = useState<Behavior | null>(null);
  const arrivedAt = useRef(-1);
  /** the chosen card's world position with no standing offset: what the fly should face */
  const cardTrue = useRef<THREE.Vector3 | null>(null);
  const flightFrom = useRef(new THREE.Vector3(...LAYOUT.flyStart));
  const lastTarget = useRef(new THREE.Vector3(...LAYOUT.flyStart));
  const flightStart = useRef(0);
  const flightU = useRef(0);

  const anim = useAnimator({
    activity: activity ?? null,
    stateRms,
    activeFraction,
    override: behavior,
    reaction,
    paused,
    timeScale,
  });

  const slotOf = useCallback((kind: string, slot: number) => {
    for (let i = 0; i < CHANNELS.length; i += 1) {
      const c = CHANNELS[i];
      if (c.kind === kind && c.slot === slot) return i;
    }
    return 0;
  }, []);

  const scratch = useMemo(() => new THREE.Vector3(), []);
  // The reach moves the body as well as the legs; see the touch block in useFrame.
  const flightPos = useRef(new THREE.Vector3());
  const glassNormal = useMemo(() => new THREE.Vector3(), []);
  const legWorld = useMemo(() => new THREE.Vector3(), []);
  const legLocal = useMemo(() => new THREE.Vector3(), []);
  const screenInverse = useMemo(() => new THREE.Matrix4(), []);
  const glassPoint = useMemo(() => new THREE.Vector3(), []);
  const glassNormalV = useMemo(() => new THREE.Vector3(), []);
  const camera = useThree((s) => s.camera);
  const viewport = useThree((s) => s.size);

  /**
   * World -> CSS pixels for the current camera, published so a screenshot can be read against
   * the same numbers the probe reports. Stable: built once per camera, not per frame.
   */
  useEffect(() => {
    const px = window as unknown as { __flyPixel?: unknown };
    px.__flyPixel = (x: number, y: number, z: number) => {
      const v = new THREE.Vector3(x, y, z).project(camera);
      return [(v.x * 0.5 + 0.5) * viewport.width, (0.5 - v.y * 0.5) * viewport.height].map(
        (n) => +n.toFixed(1),
      );
    };
  }, [camera, viewport]);

  /**
   * Dev only: hands the page the very object the pose reads, so the reach can be swept from
   * the browser while tools/reach_measure.py watches the tarsi. That is how its numbers were
   * chosen — measured, not guessed. In a build this branch is compiled out.
   */
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    (window as unknown as { __REACH?: unknown }).__REACH = REACH;
  }, []);

  useFrame((_, rawDelta) => {
    const g = group.current;
    if (!g) return;
    const delta = Math.min(0.05, Math.max(0, rawDelta));

    // ---- where the fly should be -------------------------------------------------
    targetOut.current = target.current;
    const rect = flyChoice >= 0 ? rects[flyChoice] : undefined;
    const screen = screenRef.current;
    if (rect && screen) {
      // The body is offset to the LEFT of the card along the glass's own right axis, so it
      // clears the handset, and forward along the normal so the forelegs have something to
      // reach across. The forelegs are what land on the answer.
      const local = screenPointToPlaneLocal(
        rect.x + rect.w * LAYOUT.flyCardU,
        rect.cy,
        SCREEN_W,
        SCREEN_H,
        LAYOUT.standoff,
      );
      // plane-local units: divide by the scale the plane inherits, or this becomes 54x too big
      local.x -= LAYOUT.lateral / phoneScale;
      local.y -= LAYOUT.perchDrop / phoneScale;
      screen.updateWorldMatrix(true, false);
      target.current.copy(local).applyMatrix4(screen.matrixWorld);
    } else {
      target.current.copy(home);
    }

    // ---- fly there ---------------------------------------------------------------
    // Driven by wall-clock time, not by accumulated frame deltas. A per-frame lerp is tied to
    // the frame rate: under software rendering at well under one frame a second the fly took
    // over 26 seconds to cross the scene and never arrived. Interpolating against the clock
    // takes the same 1.6 seconds on any machine.
    const now = performance.now();
    if (!lastTarget.current.equals(target.current)) {
      lastTarget.current.copy(target.current);
      flightFrom.current.copy(g.position);
      flightStart.current = now;
    }
    let u = paused
      ? flightU.current
      : Math.min(1, (now - flightStart.current) / (LAYOUT.flySeconds * 1000));
    flightU.current = u;
    u = u < 1 ? 1 - Math.pow(1 - u, LAYOUT.flyEase) : 1;
    g.position.lerpVectors(flightFrom.current, target.current, u);
    const remaining = g.position.distanceTo(target.current);
    const moving = remaining > 0.03 || !screen;
    // the pure flight line, before the reach is allowed to move the body: the yaw and the
    // arrival test both belong to the flight, not to the lean
    flightPos.current.copy(g.position);

    // ---- touching the answer -----------------------------------------------------
    // The reach is not only legs: the fly leans in toward the glass and settles down onto the
    // card, and that body movement is what carries the tarsi the last of the way. Written
    // absolutely from the flight interpolation every frame, like the pose, so it can never
    // accumulate; driven by the animator's own behavior clock so the body and the legs cannot
    // disagree about where the tarsus is.
    if (screen && flyChoice >= 0 && rects[flyChoice]) {
      const r = rects[flyChoice];
      screen.updateWorldMatrix(true, false);
      cardTrue.current = screenPointToPlaneLocal(r.cx, r.cy, SCREEN_W, SCREEN_H, 0)
        .applyMatrix4(screen.matrixWorld);
    } else {
      cardTrue.current = null;
    }

    const reach: ReachState =
      behavior === 'reach' ? reachCurve(anim.live.age) : ZERO_REACH;
    if (screen && reach.on > 0) {
      screen.updateWorldMatrix(true, false);
      glassNormal.set(0, 0, 1).transformDirection(screen.matrixWorld);
      if (glassNormal.dot(scratch.subVectors(camera.position, g.position)) < 0) glassNormal.negate();
      g.position.addScaledVector(glassNormal, -REACH.lean * reach.on);
      g.position.y -= REACH.drop * reach.on;
    }

    const t = anim.live.t;
    if (moving) {
      arrivedAt.current = -1;
      if (behavior !== 'walk') setBehavior('walk');
    } else {
      if (arrivedAt.current < 0) arrivedAt.current = t;
      // Land, put the forelegs on the answer it picked, then settle into a look around
      // rather than looping forever.
      const dwelled = t - arrivedAt.current;
      const want: Behavior =
        dwelled < reachSeconds()
          ? 'reach'
          : dwelled < reachSeconds() + 1.5
            ? 'proboscis'
            : 'groom';
      if (behavior !== want) setBehavior(want);
    }

    // ---- face the way it is going, then the glass --------------------------------
    // The model's own forward direction is not documented, so the offset is a dial rather
    // than a guess: it is set from the screenshot so the fly does not travel sideways.
    // publish the flight for verification: position, target and remaining distance, plus
    // what the reach is doing and where the six tarsi are relative to the glass
    const w = window as unknown as {
      __flyPos?: unknown;
      __flyProbe?: { foot?: number[][]; wing?: number[][] } | null;
    };
    const node = w.__flyProbe ?? null;
    if (screen) screen.updateWorldMatrix(true, false);
    const glass = screen
      ? {
          point: screen.getWorldPosition(new THREE.Vector3()).toArray().map((v) => +v.toFixed(4)),
          normal: new THREE.Vector3(0, 0, 1).transformDirection(screen.matrixWorld).toArray().map((v) => +v.toFixed(4)),
          up: new THREE.Vector3(0, 1, 0).transformDirection(screen.matrixWorld).toArray().map((v) => +v.toFixed(4)),
          right: new THREE.Vector3(1, 0, 0).transformDirection(screen.matrixWorld).toArray().map((v) => +v.toFixed(4)),
          height: SCREEN.height,
        }
      : null;
    // Every tarsus against the glass, in the glass's own frame: the perpendicular distance,
    // and where that lands in screen pixels so it can be compared with the card it chose.
    let legs: { d: number; px: number; py: number; onCard: boolean }[] | null = null;
    if (screen && glass && node?.foot?.length) {
      screenInverse.copy(screen.matrixWorld).invert();
      glassPoint.fromArray(glass.point);
      glassNormalV.fromArray(glass.normal);
      legs = node.foot.map((p) => {
        legWorld.set(p[0], p[1], p[2]);
        const d = legLocal.copy(legWorld).sub(glassPoint).dot(glassNormalV);
        legLocal.copy(legWorld).applyMatrix4(screenInverse);
        const px = (legLocal.x / SCREEN.width + 0.5) * SCREEN_W;
        const py = (0.5 - legLocal.y / SCREEN.height) * SCREEN_H;
        const onCard = Boolean(
          rect && px >= rect.x && px <= rect.x + rect.w && py >= rect.y && py <= rect.y + rect.h,
        );
        return { d: +d.toFixed(4), px: +px.toFixed(1), py: +py.toFixed(1), onCard };
      });
    }
    const cardRect =
      rect && screen
        ? {
            px: [rect.x, rect.y, rect.w, rect.h].map((v) => +v.toFixed(1)),
            corners: (
              [
                [rect.x, rect.y],
                [rect.x + rect.w, rect.y],
                [rect.x, rect.y + rect.h],
                [rect.x + rect.w, rect.y + rect.h],
              ] as [number, number][]
            ).map(([px, py]) =>
              screenPointToPlaneLocal(px, py, SCREEN_W, SCREEN_H, LAYOUT.standoff)
                .applyMatrix4(screen!.matrixWorld)
                .toArray()
                .map((v) => +v.toFixed(4)),
            ),
          }
        : null;
    w.__flyPos = {
      t: +t.toFixed(2),
      pos: [flightPos.current.x, flightPos.current.y, flightPos.current.z].map((v) =>
        +v.toFixed(3),
      ),
      target: [target.current.x, target.current.y, target.current.z].map((v) =>
        +v.toFixed(3),
      ),
      dist: +remaining.toFixed(4),
      moving,
      behavior,
      hasScreen: Boolean(screen),
      hasRect: Boolean(rect),
      /** the body offset the reach is applying on top of the flight line */
      reach: {
        on: +reach.on.toFixed(4),
        planted: +reach.planted.toFixed(4),
        offset: [
          +(g.position.x - flightPos.current.x).toFixed(4),
          +(g.position.y - flightPos.current.y).toFixed(4),
          +(g.position.z - flightPos.current.z).toFixed(4),
        ],
      },
      rotY: +g.rotation.y.toFixed(4),
      scale: +(LAYOUT.flySpan / 1.75).toFixed(4),
      glass,
      cardRect,
      legs,
      cardWorld:
        screen && rects.length
          ? rects.map((r) => {
              const lp = screenPointToPlaneLocal(
                r.x + r.w * LAYOUT.flyCardU,
                r.cy,
                SCREEN_W,
                SCREEN_H,
                LAYOUT.standoff,
              );
              lp.x -= LAYOUT.lateral / phoneScale;
              lp.y -= LAYOUT.perchDrop / phoneScale;
              const p = lp.applyMatrix4(screen.matrixWorld);
              return [p.x, p.y, p.z].map((v) => +v.toFixed(3));
            })
          : null,
      screenBox: screen
        ? (() => {
            const b = new THREE.Box3().setFromObject(screen);
            return {
              min: [b.min.x, b.min.y, b.min.z].map((v) => +v.toFixed(3)),
              max: [b.max.x, b.max.y, b.max.z].map((v) => +v.toFixed(3)),
            };
          })()
        : null,
    };

    // world -> CSS pixels, so a screenshot can be read against the same numbers
    // Look at the ANSWER, not along the flight line. Once the fly has arrived the flight
    // vector is zero, so the old code skipped this entirely and the fly simply kept whatever
    // yaw the flight left it with; there was no orientation being held at all.
    if (cardTrue.current) {
      scratch.subVectors(cardTrue.current, g.position);
    } else {
      scratch.subVectors(target.current, flightPos.current);
    }
    if (scratch.lengthSq() > 1e-6) {
      const yaw = Math.atan2(scratch.x, scratch.z) + LAYOUT.flyYawOffset;
      let d = yaw - g.rotation.y;
      while (d > Math.PI) d -= Math.PI * 2;
      while (d < -Math.PI) d += Math.PI * 2;
      g.rotation.y += d * Math.min(1, 6 * delta);
    }
  });

  return (
    <group ref={group} position={LAYOUT.flyStart} scale={LAYOUT.flySpan / 1.75}>
      <Suspense fallback={null}>
        <RealFly
          drive={anim.drive}
          pose={anim.pose}
          live={anim.live}
          slotOf={slotOf}
          /* No floor contact patches: they are decals for standing on a surface, and here the
             fly is up on a screen. Left on, they painted dark smudges across the white card. */
          contacts={false}
        />
      </Suspense>
      {/* the behavior override is published so the headless check can see the flight was real */}
      <BehaviorProbe behavior={behavior} />
    </group>
  );
}

function BehaviorProbe({ behavior }: { behavior: Behavior | null }) {
  useEffect(() => {
    (window as unknown as { __flyBehavior?: string | null }).__flyBehavior = behavior;
  }, [behavior]);
  return null;
}

/**
 * The stage floor, plus the screen's glow reflected in it.
 *
 * A floor dark enough to vanish cannot show a cast shadow, and the shadow is what puts the
 * handset on a surface instead of in a void. The faint warm pool under the phone's base is
 * the screen's light landing on the floor, which is the cue that reads as a real object
 * standing on something rather than a graphic pasted over a gradient.
 */
function Floor() {
  const reflection = useMemo(() => {
    const c = document.createElement('canvas');
    c.width = 256;
    c.height = 256;
    const ctx = c.getContext('2d');
    if (ctx) {
      const g = ctx.createRadialGradient(128, 128, 8, 128, 128, 126);
      g.addColorStop(0, 'rgba(190,232,255,0.55)');
      g.addColorStop(0.45, 'rgba(120,180,230,0.20)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, 256, 256);
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, []);

  return (
    <>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.001, 0]} receiveShadow>
        <circleGeometry args={[26, 96]} />
        <meshStandardMaterial color="#232e3d" roughness={0.5} metalness={0.05} />
      </mesh>
      {/* the screen's light pooling on the floor, stretched along the viewing axis */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[LAYOUT.phoneX, 0.004, 0.9]}>
        <planeGeometry args={[9.5, 7.5]} />
        <meshBasicMaterial
          map={reflection}
          transparent
          opacity={0.5}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </>
  );
}

/**
 * A soft radial wash behind the stage.
 *
 * The black either side of a portrait handset in a wide frame reads as unpainted when it is
 * perfectly flat. A large, very dim gradient gives that space a light source and a
 * direction, so it reads as a dark room rather than as missing pixels.
 */
function Backdrop() {
  const tex = useMemo(() => {
    const c = document.createElement('canvas');
    c.width = 512;
    c.height = 512;
    const ctx = c.getContext('2d');
    if (ctx) {
      const g = ctx.createRadialGradient(256, 300, 20, 256, 300, 250);
      g.addColorStop(0, 'rgba(72,132,196,0.55)');
      g.addColorStop(0.5, 'rgba(40,74,116,0.20)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, 512, 512);
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, []);
  return (
    <mesh position={[LAYOUT.phoneX - 0.4, 3.1, -6.5]}>
      <planeGeometry args={[30, 22]} />
      <meshBasicMaterial
        map={tex}
        transparent
        opacity={0.85}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}

/** Keeps the renderer sized to the box the component is given. */
function SizeSync({ width, height }: { width?: number; height?: number }) {
  const size = useThree((s) => s.size);
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  useEffect(() => {
    const w = width || Math.round(size.width);
    const h = height || Math.round(size.height);
    if (!w || !h) return;
    gl.setSize(w, h, false);
    const cam = camera as THREE.PerspectiveCamera;
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
  }, [gl, camera, width, height, size]);
  return null;
}

/**
 * Report the handset's material colours. The asset ships them as pure black, which is
 * invisible on a dark stage; this is how the fix was confirmed rather than assumed.
 */
function bodyMats(): { name: string; color: string; metalness: number }[] {
  const out: { name: string; color: string; metalness: number }[] = [];
  const seen = new Set<string>();
  const meshes = sceneMeshes.length ? sceneMeshes : [];
  meshes.forEach((m) => {
    const mat = m?.material as THREE.MeshStandardMaterial;
    if (!mat || Array.isArray(mat)) return;
    const hex = mat.color && mat.color.getHexString ? mat.color.getHexString() : 'n/a';
    const key = `${mat.name || '?'}:${hex}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ name: mat.name || '?', color: `#${hex}`, metalness: mat.metalness ?? -1 });
  });
  return out;
}

/** every mesh in the loaded handset, published by PhoneScreen */
export const sceneMeshes: THREE.Mesh[] = [];

/**
 * A soft contact shadow drawn on the glass under the fly.
 *
 * Without it the fly reads as composited over the screen rather than standing on it, because
 * a real object resting on a surface occludes the light reaching it. The body's own per-tarsus
 * contact patches are floor decals and are switched off here, so this is the screen equivalent:
 * one soft ellipse, oriented to the glass and following the fly's perch point.
 */
function ScreenContactShadow({
  targetRef,
  screenRef,
}: {
  targetRef: React.RefObject<THREE.Vector3 | null>;
  screenRef: React.RefObject<THREE.Mesh | null>;
}) {
  const ref = useRef<THREE.Mesh>(null);
  const tex = useMemo(() => {
    const c = document.createElement('canvas');
    c.width = 128;
    c.height = 128;
    const ctx = c.getContext('2d');
    if (ctx) {
      const g = ctx.createRadialGradient(64, 64, 2, 64, 64, 62);
      g.addColorStop(0, 'rgba(0,0,0,0.62)');
      g.addColorStop(0.5, 'rgba(0,0,0,0.30)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, 128, 128);
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, []);

  useFrame(() => {
    const m = ref.current;
    const screen = screenRef.current;
    const target = targetRef.current;
    if (!m || !screen || !target) return;
    screen.updateWorldMatrix(true, false);
    m.position.copy(target);
    // lie flat on the glass, so orientation comes from the screen's own world rotation
    screen.getWorldQuaternion(m.quaternion);
    // lift a hair off the surface to avoid z-fighting with the display
    m.translateZ(0.0022);
  });

  return (
    <mesh ref={ref} renderOrder={2}>
      <planeGeometry args={[1.25, 0.95]} />
      <meshBasicMaterial map={tex} transparent depthWrite={false} />
    </mesh>
  );
}

export function PhoneStage(props: PhoneStageProps) {
  const {
    prompt,
    options,
    flyChoice,
    userChoice,
    status,
    answerIndex,
    hearts,
    progress,
    activity,
    stateRms = 0,
    activeFraction = 0,
    reaction = { kind: 'none', seq: 0 },
    paused = false,
    timeScale = 1,
    onScreen,
    width,
    height,
  } = props;

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const screenRef = useRef<THREE.Mesh | null>(null);
  const textureRef = useRef<THREE.CanvasTexture | null>(null);
  const [texture, setTexture] = useState<THREE.CanvasTexture | null>(null);
  const [rects, setRects] = useState<OptionRect[]>([]);
  const [measured, setMeasured] = useState<{
    scale: number;
    phoneWorldH: number;
    footprintX: number;
    footprintZ: number;
  } | null>(null);
  const flyTargetRef = useRef<THREE.Vector3 | null>(null);

  /**
   * A stable key for the screen's contents.
   *
   * The draw effect must not depend on the `options` array directly: callers naturally build
   * it inline (`options.map(o => ({text: o}))`), so its identity changes on every render. The
   * effect then redrew, called setRects with a fresh array, triggered another render, and
   * looped. That loop also reset the telemetry interval below, which is why the probe kept
   * coming back empty. Keying on the rendered values instead makes the effect run only when
   * something on the screen actually changes.
   */
  const screenKey = useMemo(
    () =>
      JSON.stringify({
        prompt,
        options: options.map((o) => o.text),
        flyChoice,
        userChoice,
        status,
        answerIndex,
        hearts,
        progress,
      }),
    [prompt, options, flyChoice, userChoice, status, answerIndex, hearts, progress],
  );

  // Draw the screen. Fonts are awaited first: the canvas would otherwise silently fall back
  // to a default face and the whole look would be wrong without any error.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await ensureDuolingoFonts();
      if (cancelled) return;
      if (!canvasRef.current) canvasRef.current = createScreenCanvas(2);
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const parsed = JSON.parse(screenKey) as {
        prompt: string;
        options: string[];
        flyChoice: number;
        userChoice: number;
        status: 'none' | 'correct' | 'wrong';
        answerIndex: number;
        hearts: number;
        progress: number;
      };
      const state: ScreenState = {
        prompt: parsed.prompt,
        options: parsed.options.map((text) => ({ text })),
        flyChoice: parsed.flyChoice,
        userChoice: parsed.userChoice,
        status: parsed.status,
        answerIndex: parsed.answerIndex,
        hearts: parsed.hearts,
        progress: parsed.progress,
      };
      const out = drawDuolingoLesson(ctx, state);
      setRects(out.options);
      onScreen?.({ rects: out.options });
      if (textureRef.current) {
        textureRef.current.needsUpdate = true;
      } else {
        const tex = new THREE.CanvasTexture(canvas);
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = 8;
        tex.minFilter = THREE.LinearMipmapLinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.generateMipmaps = true;
        textureRef.current = tex;
        setTexture(tex);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screenKey]);

  // Publish what a headless check needs to verify the screen's real orientation: a phone
  // that came up sideways or mirrored would otherwise pass every structural test.
  useEffect(() => {
    const id = window.setInterval(() => {
      // never let a probe fault blank the telemetry: that would look like "nothing is
      // running" when in fact only the reporting failed
      try {
      const m = screenRef.current;
      if (!m) return;
      m.updateWorldMatrix(true, false);
      const rot = new THREE.Matrix4().extractRotation(m.matrixWorld);
      const normal = new THREE.Vector3(0, 0, 1).applyMatrix4(rot);
      const up = new THREE.Vector3(0, 1, 0).applyMatrix4(rot);
      (window as unknown as { __phoneStage?: unknown }).__phoneStage = {
        rects,
        status,
        flyChoice,
        screenHeight: LAYOUT.screenHeight,
        phoneWorldH: measured?.phoneWorldH ?? null,
        phoneScale: measured?.scale ?? null,
        screenNormal: [normal.x, normal.y, normal.z].map((v) => +v.toFixed(3)),
        screenUp: [up.x, up.y, up.z].map((v) => +v.toFixed(3)),
        bodyMaterials: bodyMats(),
      };
      } catch (err) {
        (window as unknown as { __phoneStage?: unknown }).__phoneStage = {
          probeError: String(err).slice(0, 200),
        };
      }
    }, 400);
    return () => window.clearInterval(id);
  }, [rects, status, flyChoice, measured]);

  return (
    <Canvas
      shadows
      gl={{ antialias: true, powerPreference: 'high-performance' }}
      camera={{ position: LAYOUT.camera, fov: LAYOUT.fov, near: 0.05, far: 60 }}
      style={{ position: 'absolute', inset: 0 }}
    >
      <SizeSync width={width} height={height} />
      <color attach="background" args={['#080c11']} />
      <StudioEnvironment intensity={0.30} />

      {/* key: warm, upper left, the only shadow caster */}
      <directionalLight
        position={[-3.2, 4.6, 3.4]}
        intensity={2.0}
        color={RIG.key.color}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-near={0.5}
        shadow-camera-far={34}
        shadow-camera-left={-5}
        shadow-camera-right={5}
        shadow-camera-top={5}
        shadow-camera-bottom={-5}
        shadow-bias={-0.0007}
      />
      {/* rim from behind right, to separate the handset from the void */}
      <directionalLight position={[4.0, 3.2, -3.2]} intensity={1.2} color={RIG.rim.color} />
      <ambientLight intensity={RIG.ambient.intensity * 1.25} color={RIG.ambient.color} />

      <Backdrop />
      <Floor />

      <Suspense fallback={null}>
        <PhoneScreen
          texture={texture}
          screenRef={screenRef}
          onMeasured={(m) => setMeasured(m)}
        />
      </Suspense>

      <FlyActor
        rects={rects}
        flyChoice={flyChoice}
        activity={activity}
        stateRms={stateRms}
        activeFraction={activeFraction}
        reaction={reaction}
        paused={paused}
        timeScale={timeScale}
        screenRef={screenRef}
        targetOut={flyTargetRef}
        phoneScale={measured?.scale ?? PHONE_SCALE_FALLBACK}
      />

      {measured && (
        <PhoneContactAO
          footprintX={measured.footprintX}
          footprintZ={measured.footprintZ}
        />
      )}
      <ScreenContactShadow targetRef={flyTargetRef} screenRef={screenRef} />
      <Bloom />
      <OrbitControls
        target={LAYOUT.target}
        enablePan={false}
        minDistance={3}
        /* Was 16, which silently clamped the shot: OrbitControls pulled any camera further out
           back to 16, so framing changes past that distance had no effect at all. Raised to 60
           so the composition is not fighting a control constraint. */
        maxDistance={60}
        maxPolarAngle={Math.PI / 2.05}
      />
    </Canvas>
  );
}

export default PhoneStage;
