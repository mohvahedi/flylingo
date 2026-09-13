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
  screenHeight: 7.0,
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
  /** the fly's world span: large enough to read as the protagonist */
  flySpan: 1.6,
  /** where the fly starts: in front of the phone and to its left */
  flyStart: [-4.6, 0.0, 2.4] as [number, number, number],
  /**
   * How far off the glass the fly hovers when it reaches a card, in the phone's own mesh
   * units. The screen is only 0.155 tall there, so this has to be small: an earlier value of
   * 0.05 was a third of the screen height and threw the fly well clear of the handset.
   */
  standoff: 0.004,
  /** corrected so the model's own forward lines up with the direction of travel */
  flyYawOffset: 0,
  /**
   * How far across the chosen card the fly perches, 0 = left edge, 1 = right edge.
   *
   * Not the centre: the option text sits on the left of each card, and a fly with a 1.6 unit
   * wingspan landing centrally covered the text of the card next to it. Perched to the right
   * it reads as sitting on the answer without hiding it.
   */
  flyCardU: 0.78,
  /** how long the flight from wherever it is to the chosen card takes, in seconds */
  flySeconds: 1.6,
  /** easing exponent for the flight: 2 eases in and out, which reads as a dart not a slide */
  flyEase: 2.0,
  camera: [1.35, 3.55, 11.4] as [number, number, number],
  target: [1.35, 3.3, 0.4] as [number, number, number],
  fov: 38,
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
  onMeasured?: (info: { scale: number; phoneWorldH: number; screenWorldH: number }) => void;
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

    const s = unitScreenH > 1e-9 ? LAYOUT.screenHeight / unitScreenH : PHONE_SCALE_FALLBACK;
    const phoneWorldH = unitPhoneH * s;

    // stand it on the floor: the box centre sits at half its height
    const centreY = (phoneBox.min.y + phoneBox.max.y) / 2;
    setScale(s);
    g.scale.setScalar(s);
    g.position.set(LAYOUT.phoneX, -centreY * s + phoneWorldH / 2, 0);
    g.updateWorldMatrix(true, true);
    onMeasured?.({ scale: s, phoneWorldH, screenWorldH: LAYOUT.screenHeight });
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
}) {
  const group = useRef<THREE.Group>(null);
  const target = useRef(new THREE.Vector3(...LAYOUT.flyStart));
  const home = useMemo(() => new THREE.Vector3(...LAYOUT.flyStart), []);
  const [behavior, setBehavior] = useState<Behavior | null>(null);
  const arrivedAt = useRef(-1);
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

  useFrame((_, rawDelta) => {
    const g = group.current;
    if (!g) return;
    const delta = Math.min(0.05, Math.max(0, rawDelta));

    // ---- where the fly should be -------------------------------------------------
    const rect = flyChoice >= 0 ? rects[flyChoice] : undefined;
    const screen = screenRef.current;
    if (rect && screen) {
      const local = screenPointToPlaneLocal(
        rect.x + rect.w * LAYOUT.flyCardU,
        rect.cy,
        SCREEN_W,
        SCREEN_H,
        LAYOUT.standoff,
      );
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

    const t = anim.live.t;
    if (moving) {
      arrivedAt.current = -1;
      if (behavior !== 'walk') setBehavior('walk');
    } else {
      if (arrivedAt.current < 0) arrivedAt.current = t;
      // hold at the card, reach out to touch it, then settle rather than looping forever
      const dwelled = t - arrivedAt.current;
      const want: Behavior = dwelled < 1.5 ? 'proboscis' : 'groom';
      if (behavior !== want) setBehavior(want);
    }

    // ---- face the way it is going, then the glass --------------------------------
    // The model's own forward direction is not documented, so the offset is a dial rather
    // than a guess: it is set from the screenshot so the fly does not travel sideways.
    // publish the flight for verification: position, target and remaining distance
    const w = window as unknown as { __flyPos?: unknown };
    w.__flyPos = {
      t: +t.toFixed(2),
      pos: [g.position.x, g.position.y, g.position.z].map((v) => +v.toFixed(3)),
      target: [target.current.x, target.current.y, target.current.z].map((v) =>
        +v.toFixed(3),
      ),
      dist: +remaining.toFixed(4),
      moving,
      behavior,
      hasScreen: Boolean(screen),
      hasRect: Boolean(rect),
      cardWorld:
        screen && rects.length
          ? rects.map((r) => {
              const p = screenPointToPlaneLocal(
                r.x + r.w * LAYOUT.flyCardU,
                r.cy,
                SCREEN_W,
                SCREEN_H,
                LAYOUT.standoff,
              ).applyMatrix4(screen.matrixWorld);
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

    scratch.subVectors(target.current, g.position);
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
  const [measured, setMeasured] = useState<{ scale: number; phoneWorldH: number } | null>(null);

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
        shadow-camera-far={24}
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
          onMeasured={({ scale, phoneWorldH }) => setMeasured({ scale, phoneWorldH })}
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
      />

      <Bloom />
      <OrbitControls
        target={LAYOUT.target}
        enablePan={false}
        minDistance={3}
        maxDistance={16}
        maxPolarAngle={Math.PI / 2.05}
      />
    </Canvas>
  );
}

export default PhoneStage;
