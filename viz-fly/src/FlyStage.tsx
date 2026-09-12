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
 *    the stage works with no network at all (no HDR environment, no .glb, no fonts).
 *  - dark instrument look: one key light, one cool rim light, contact shadow only.
 *  - UI is limited to the mode badge, an optional corner sparkline, and a dev strip that
 *    only appears when a frame is supplied and `dev` is not disabled.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { ContactShadows, OrbitControls } from '@react-three/drei';
import { Fly } from './fly/Fly';
import { useAnimator } from './fly/animator';
import { CHANNELS, modeLabel } from './fly/regions';
import type { Behavior, ReactionKind } from './fly/pose';
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
  className?: string;
  style?: CSSProperties;
};

const SPARK_POINTS = 180;

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
      <color attach="background" args={['#05070b']} />
      <fog attach="fog" args={['#05070b', 6, 14]} />

      {/* key light: warm white, the only shadow caster */}
      <directionalLight
        position={[3.2, 5.0, 2.6]}
        intensity={2.4}
        color="#fff4e6"
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-near={0.5}
        shadow-camera-far={16}
        shadow-camera-left={-3}
        shadow-camera-right={3}
        shadow-camera-top={3}
        shadow-camera-bottom={-3}
        shadow-bias={-0.0012}
      />
      {/* rim light: cool, from behind, separates the fly from the background */}
      <directionalLight position={[-3.4, 2.2, -3.6]} intensity={2.1} color="#60a5fa" />
      {/* instrument fill so nothing is ever pure black */}
      <ambientLight intensity={0.22} color="#8fa6c4" />
      <pointLight position={[0, 1.1, 1.9]} intensity={0.5} color="#bae6fd" distance={7} />
      {/* reward accent: a warm under-glow that scales with the reward channel */}
      <pointLight
        position={[0, 0.5, 0.4]}
        intensity={1.5 * Math.max(0, Math.min(1, reward || 0))}
        color="#fbbf24"
        distance={3.2}
      />

      <Fly
        drive={anim.drive}
        pose={anim.pose}
        live={anim.live}
        slotOf={slotOf}
      />

      {/* ground: a dark disc plus a scale ring, no texture, no clutter */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <circleGeometry args={[7, 64]} />
        <meshStandardMaterial color="#080c12" roughness={0.94} metalness={0.04} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.003, 0]}>
        <ringGeometry args={[1.55, 1.575, 96]} />
        <meshBasicMaterial color="#22d3ee" transparent opacity={0.16} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.003, 0]}>
        <ringGeometry args={[2.6, 2.615, 96]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.07} />
      </mesh>

      {/* soft contact shadow instead of a hard shadow map alone */}
      <ContactShadows
        position={[0, 0.004, 0]}
        opacity={0.62}
        scale={5}
        blur={2.4}
        far={2.2}
        resolution={1024}
        color="#000814"
      />

      <OrbitControls
        makeDefault
        enablePan={false}
        minDistance={1.5}
        maxDistance={7}
        minPolarAngle={0.12}
        maxPolarAngle={Math.PI / 2.08}
        autoRotate
        autoRotateSpeed={0.42}
        enableDamping
        dampingFactor={0.06}
        target={[0, 0.5, 0]}
      />
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
    background: '#05070b',
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
          gl.setClearColor('#05070b', 1);
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
