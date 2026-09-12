/**
 * FlyStage: the single exported entry point for embedding the fly.
 *
 * Its props are exactly the frozen contract in D:\Projects\flylingo\INTERFACES.md
 * ("Frontend contract for the visualization components"). Every prop is optional so the
 * app can mount <FlyStage /> before the websocket has produced a frame; with nothing
 * supplied the harness runs a synthetic idle connectome and the autonomous behavior
 * schedule, and it looks alive rather than blank.
 *
 * Rendering choices worth knowing about:
 *  - no external assets. Every mesh is procedural and every material is built here, so
 *    the stage works with no network at all (no HDR file, no .glb, no fonts). The
 *    environment map is a room of emissive planes, baked at mount with PMREM.
 *  - cinematic rig: a hard warm key from the upper right that is the only shadow caster and
 *    sits low enough to throw a long cast shadow, a cool rim from behind, a low cool fill,
 *    and a shallow ambient. See fly/studio.ts.
 *  - the ground fades radially to true black, so there is no horizon and no plane edge.
 *  - optional bloom (on by default) so the speculars and the connectome glow read as light.
 *  - UI is limited to the mode badge, an optional corner sparkline, and a dev strip that
 *    only appears when a frame is supplied and `dev` is not disabled.
 */
import {
  Component,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { ContactShadows, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { Fly } from './fly/Fly';
import { RealFly } from './fly/RealFly';
import { useAnimator, type LiveReadout } from './fly/animator';
import { CHANNELS, modeLabel } from './fly/regions';
import { groundMaterial } from './fly/materials';
import { RIG, SHADOW_EXTENT, StudioEnvironment } from './fly/studio';
import { Bloom } from './fly/Bloom';
import type { Behavior, Pose, ReactionKind } from './fly/pose';
import { ControlStrip } from './components/ControlStrip';
import { Sparkline } from './components/Sparkline';

/** The frozen contract. Optional here so <FlyStage /> with no props is valid. */
export type FlyStageProps = {
  /** frame.state from the BrainFrame, -1..1, length 512 */
  activity?: number[];
  stateRms?: number;
  activeFraction?: number;
  /** drives the celebrate / recoil reaction on transition */
  correct?: boolean | null;
  reward?: number;
  /** 'intact' | 'shuffled' | 'no_edges' | 'random_graph', rendered as a badge */
  mode?: string;
  width?: number;
  height?: number;
  /** render the dev-only inspection strip. Default: true when activity is supplied. */
  dev?: boolean;
  /** render the corner stateRms sparkline. Default: true. */
  sparkline?: boolean;
  /** bloom pass over the whole stage. On by default; off costs one prop. */
  bloom?: boolean;
  /**
   * Which fly to render. 'auto' (the default) loads the supplied mesh in
   * public/models/fly.glb and falls back to the procedural fly if it cannot be loaded.
   * 'procedural' skips the asset entirely, which is also how the fallback path is tested.
   */
  asset?: 'auto' | 'procedural';
  /** slow orbit for the hero shot. Turn off when the caller drives the camera. */
  autoRotate?: boolean;
  /** geometry and renderer census, sampled twice a second. Optional, no-op by default. */
  onStats?: (stats: StageStats) => void;
  className?: string;
  style?: CSSProperties;
};

export type StageStats = {
  /** triangles in the scene graph, counted once at mount (instances included) */
  triangles: number;
  meshes: number;
  instances: number;
  /** peak draw calls in a frame, from the renderer's own counters */
  drawCalls: number;
  programs: number;
  geometries: number;
  textures: number;
};

const SPARK_POINTS = 180;

/**
 * Reports what the scene actually costs.
 *
 * The renderer's counters are turned off auto-reset and cleared at the top of the frame by
 * a negative priority subscriber, so they accumulate across every pass of the frame, bloom
 * included, and are read back before the next frame renders. Reading them the naive way
 * would only ever report the last pass of a composer chain.
 */
function StatsProbe({ onStats }: { onStats: (s: StageStats) => void }) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const census = useRef<Pick<StageStats, 'triangles' | 'meshes' | 'instances'> | null>(null);
  const peak = useRef(0);
  const last = useRef(0);

  useEffect(() => {
    let triangles = 0;
    let meshes = 0;
    let instances = 0;
    scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh || !mesh.geometry) return;
      const geo = mesh.geometry as THREE.BufferGeometry;
      const index = geo.getIndex();
      const position = geo.getAttribute('position');
      const tris = index ? index.count / 3 : position ? position.count / 3 : 0;
      const count = (mesh as THREE.InstancedMesh).isInstancedMesh
        ? (mesh as THREE.InstancedMesh).count
        : 1;
      triangles += tris * count;
      meshes += 1;
      if (count > 1) instances += count;
    });
    census.current = {
      triangles: Math.round(triangles),
      meshes,
      instances,
    };
  }, [scene]);

  useEffect(() => {
    const prev = gl.info.autoReset;
    gl.info.autoReset = false;
    return () => {
      gl.info.autoReset = prev;
    };
  }, [gl]);

  // runs before every other subscriber, so the counters only ever hold one frame of work
  useFrame(() => {
    gl.info.reset();
  }, -50);

  useFrame((state) => {
    const calls = gl.info.render.calls;
    if (calls > peak.current) peak.current = calls;
    const t = state.clock.elapsedTime;
    if (!census.current || t - last.current < 0.5) return;
    last.current = t;
    onStats({
      ...census.current,
      drawCalls: peak.current,
      programs: gl.info.programs ? gl.info.programs.length : 0,
      geometries: gl.info.memory.geometries,
      textures: gl.info.memory.textures,
    });
    peak.current = 0;
  });

  return null;
}

/**
 * Renders the real mesh, and falls back to the procedural fly if the asset cannot be
 * loaded at all (404, corrupt file, a driver that refuses the extensions). The page must
 * never break because of the asset, so this pair catches both failure shapes: Suspense for
 * a loader that throws while fetching, the boundary for one that throws while parsing.
 */
class ModelBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  componentDidCatch(error: unknown): void {
    console.warn('fly: model unavailable, rendering the procedural fallback', error);
  }

  render(): ReactNode {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

type FlyOrModelProps = {
  drive: Float32Array;
  pose: Pose;
  live: LiveReadout;
  slotOf: (kind: string, slot: number) => number;
  asset: 'auto' | 'procedural';
};

/**
 * The hero path and its safety net.
 *
 * The procedural fly serves as both the Suspense fallback and the error fallback, so a
 * viewer always sees an animated fly: the placeholder while the 729 KB mesh streams in,
 * then the real one. If the asset never arrives, the placeholder simply stays, and the
 * page is unaffected.
 */
function FlyOrModel({ drive, pose, live, slotOf, asset }: FlyOrModelProps) {
  const procedural = <Fly drive={drive} pose={pose} live={live} slotOf={slotOf} />;
  if (asset === 'procedural') return procedural;
  return (
    <ModelBoundary fallback={procedural}>
      <Suspense fallback={procedural}>
        <RealFly drive={drive} pose={pose} live={live} slotOf={slotOf} />
      </Suspense>
    </ModelBoundary>
  );
}

/** Lightweight live state shared between the render loop and the React overlays. */
type LiveRef = {
  behavior: Behavior;
  source: 'auto' | 'manual';
  rms: number;
};

function Stage({
  activity,
  stateRms,
  activeFraction,
  reward,
  override,
  timeScale,
  paused,
  reaction,
  history,
  live,
  autoRotate,
  bloom,
  asset,
  onStats,
}: {
  activity?: number[];
  stateRms: number;
  activeFraction: number;
  reward: number;
  override: Behavior | null;
  timeScale: number;
  paused: boolean;
  reaction: { kind: ReactionKind; seq: number };
  history: number[];
  live: React.RefObject<LiveRef>;
  autoRotate: boolean;
  bloom: boolean;
  asset: 'auto' | 'procedural';
  onStats?: (stats: StageStats) => void;
}) {
  const anim = useAnimator({
    activity: activity ?? null,
    stateRms,
    activeFraction,
    override,
    reaction,
    paused,
    timeScale,
  });

  // the floor is procedural too: one material, patched to fade radially to true black
  const ground = useMemo(() => groundMaterial(7), []);
  useEffect(() => () => ground.dispose(), [ground]);

  // publish the cheap bits for the overlays; the overlays poll a few times a second
  const lastPush = useRef(0);
  const tickPush = useCallback(
    (t: number, rms: number, behavior: Behavior, source: 'auto' | 'manual') => {
      if (live.current) {
        live.current.behavior = behavior;
        live.current.source = source;
        live.current.rms = rms;
      }
      if (t - lastPush.current > 0.06) {
        lastPush.current = t;
        history.push(Math.max(0, Math.min(1, rms)));
        if (history.length > SPARK_POINTS) history.shift();
      }
    },
    [history, live],
  );

  // The render loop is the only thing that ticks, so publish from there. This is a side
  // effect (it pushes a sparkline sample into `history`), so it must live in useFrame:
  // React is free to discard or re-run a useMemo, which would drop or duplicate samples.
  useFrame(() => {
    const l = anim.live;
    tickPush(l.t, l.rms, l.behavior, l.source);
  });

  const slotOf = useCallback((kind: string, slot: number) => {
    for (let i = 0; i < CHANNELS.length; i += 1) {
      const c = CHANNELS[i];
      if (c.kind === kind && c.slot === slot) return i;
    }
    return 0;
  }, []);

  return (
    <>
      <color attach="background" args={['#080c11']} />

      {/* the studio: a room of emissive panels, PMREM baked once, applied at low
          intensity. This is what gives the clearcoat and the eyes something to reflect
          without lifting the background out of black. */}
      <StudioEnvironment intensity={0.32} />

      {/* key: hard and warm from the upper right. The only shadow caster, and low enough
          (~25 degrees above the horizon) that the cast shadow runs long and to the left. */}
      <directionalLight
        position={RIG.key.position}
        intensity={RIG.key.intensity}
        color={RIG.key.color}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-near={0.5}
        shadow-camera-far={18}
        shadow-camera-left={-SHADOW_EXTENT}
        shadow-camera-right={SHADOW_EXTENT}
        shadow-camera-top={SHADOW_EXTENT}
        shadow-camera-bottom={-SHADOW_EXTENT}
        shadow-bias={-0.0006}
        shadow-normalBias={0.02}
      />
      {/* rim: cool, from behind and a little left, separates the fly from the black */}
      <directionalLight
        position={RIG.rim.position}
        intensity={RIG.rim.intensity}
        color={RIG.rim.color}
      />
      {/* fill: low and cool from the lower left, catches the belly and the wing undersides */}
      <directionalLight
        position={RIG.fill.position}
        intensity={RIG.fill.intensity}
        color={RIG.fill.color}
      />
      {/* ambient: shallow on purpose. Anything brighter flattens the shadow side to grey. */}
      <ambientLight intensity={RIG.ambient.intensity} color={RIG.ambient.color} />
      {/* reward accent: a warm under-glow that scales with the reward channel */}
      <pointLight
        position={[0, 0.5, 0.4]}
        intensity={1.5 * Math.max(0, Math.min(1, reward || 0))}
        color="#fbbf24"
        distance={3.2}
      />

      <FlyOrModel drive={anim.drive} pose={anim.pose} live={anim.live} slotOf={slotOf} asset={asset} />

      {/* ground: one dark disc that fades to true black at the rim, so there is no plane
          edge and no horizon, plus a scale ring, no texture, no clutter */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow material={ground}>
        <circleGeometry args={[7, 96]} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.003, 0]}>
        <ringGeometry args={[1.55, 1.575, 96]} />
        <meshBasicMaterial color="#22d3ee" transparent opacity={0.14} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.003, 0]}>
        <ringGeometry args={[2.6, 2.615, 96]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.05} />
      </mesh>

      {/* soft contact shadow under the body, in addition to the real cast shadow: it is
          what grounds the tarsi. Real shadow map above gives the long dramatic wedge. */}
      <ContactShadows
        position={[0, 0.004, 0]}
        opacity={0.55}
        scale={5}
        blur={2.6}
        far={2.4}
        resolution={512}
        color="#000814"
      />

      <OrbitControls
        makeDefault
        enablePan={false}
        minDistance={1.5}
        maxDistance={7}
        minPolarAngle={0.12}
        maxPolarAngle={Math.PI / 2.08}
        autoRotate={autoRotate}
        autoRotateSpeed={0.42}
        enableDamping
        dampingFactor={0.06}
        target={[0, 0.5, 0]}
      />

      {bloom && <Bloom />}
      {onStats && <StatsProbe onStats={onStats} />}
    </>
  );
}

export function FlyStage({
  activity,
  stateRms = 0,
  activeFraction = 0,
  correct = null,
  reward = 0,
  mode = 'intact',
  width,
  height,
  dev,
  sparkline = true,
  bloom = true,
  asset = 'auto',
  autoRotate = true,
  onStats,
  className,
  style,
}: FlyStageProps) {
  const [override, setOverride] = useState<Behavior | null>(null);
  const [paused, setPaused] = useState(false);
  const [timeScale, setTimeScale] = useState(1);
  const [reaction, setReaction] = useState<{ kind: ReactionKind; seq: number }>({
    kind: 'none',
    seq: 0,
  });

  const live = useRef<LiveRef>({ behavior: 'idle', source: 'auto', rms: 0 });
  const history = useMemo<number[]>(() => [], []);
  const [shown, setShown] = useState<{ behavior: Behavior; source: 'auto' | 'manual'; rms: number }>({
    behavior: 'idle',
    source: 'auto',
    rms: 0,
  });

  // poll the live ref at 4 Hz: the control strip label and sparkline do not need 60 fps
  useEffect(() => {
    const id = window.setInterval(() => {
      const l = live.current;
      if (!l) return;
      setShown({ behavior: l.behavior, source: l.source, rms: l.rms });
    }, 250);
    return () => window.clearInterval(id);
  }, []);

  // `correct` transition -> a clean celebratory motion, or a brief recoil
  const prevCorrect = useRef<boolean | null | undefined>(undefined);
  useEffect(() => {
    const p = prevCorrect.current;
    prevCorrect.current = correct;
    if (p === correct) return;
    if (correct === true) setReaction((r) => ({ kind: 'celebrate', seq: r.seq + 1 }));
    else if (correct === false) setReaction((r) => ({ kind: 'recoil', seq: r.seq + 1 }));
  }, [correct]);

  const trigger = useCallback((kind: 'celebrate' | 'recoil') => {
    setReaction((r) => ({ kind, seq: r.seq + 1 }));
  }, []);

  const hasFrame = Array.isArray(activity) && activity.length > 0;
  const showDev = dev ?? hasFrame;
  const badge = modeLabel(mode);

  const wrap: CSSProperties = {
    position: 'relative',
    width: width != null ? `${width}px` : '100%',
    height: height != null ? `${height}px` : '100%',
    background: '#080c11',
    overflow: 'hidden',
    ...style,
  };

  return (
    <div style={wrap} className={className}>
      <Canvas
        shadows
        dpr={[1, 2]}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        camera={{ position: [1.85, 1.15, 2.35], fov: 32, near: 0.05, far: 60 }}
        onCreated={({ gl }) => {
          gl.setClearColor('#080c11', 1);
        }}
      >
        <Stage
          activity={activity}
          stateRms={stateRms}
          activeFraction={activeFraction}
          reward={reward}
          override={override}
          timeScale={timeScale}
          paused={paused}
          reaction={reaction}
          history={history}
          live={live}
          autoRotate={autoRotate}
          bloom={bloom}
          asset={asset}
          onStats={onStats}
        />
      </Canvas>

      {/* mode badge: a demo can never be mistaken for the intact connectome */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          padding: '6px 10px',
          border: `1px solid ${badge.color}55`,
          borderLeft: `3px solid ${badge.color}`,
          borderRadius: 4,
          background: 'rgba(2,6,23,0.72)',
          backdropFilter: 'blur(6px)',
          maxWidth: 'min(420px, calc(100% - 24px))',
          pointerEvents: 'none',
        }}
      >
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: badge.color,
          }}
        >
          {badge.title}
        </div>
        <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2, lineHeight: 1.4 }}>
          {badge.note}
        </div>
        <div style={{ fontSize: 10, color: '#64748b', marginTop: 3 }}>
          {hasFrame ? 'live frames' : 'synthetic idle (no frames connected)'}
        </div>
      </div>

      {sparkline && (
        <div style={{ position: 'absolute', top: 12, right: 12, opacity: 0.9 }}>
          <Sparkline value={shown.rms} history={history} label="stateRms" />
        </div>
      )}

      {showDev && (
        <ControlStrip
          behavior={shown.behavior}
          source={shown.source}
          override={override}
          onOverride={setOverride}
          paused={paused}
          onTogglePause={() => setPaused((p) => !p)}
          timeScale={timeScale}
          onTimeScale={setTimeScale}
          onTrigger={trigger}
        />
      )}
    </div>
  );
}

export default FlyStage;
