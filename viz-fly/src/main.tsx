/**
 * Standalone dev entry for the viz-fly harness. index.html mounts this file.
 *
 * There is deliberately no backend here. The page drives the frozen FlyStage contract two
 * ways so both code paths can be seen offline:
 *
 *  - "none" mounts <FlyStage /> with no props at all. That is the contract's synthetic idle
 *    path (an app can mount the stage before its websocket has produced a frame).
 *  - a live mode feeds synthetic frames whose magnitudes match the measured real data
 *    (p95 of |activity| near 0.27, stateRms near 0.1), so the glow range, the mode badge
 *    and the startle reactions are exercised through the same props the app uses.
 *
 * "no_edges" is special-cased to all zeros, because that is what the real thing measures:
 * under no_edges every one of the 512 values is exactly 0.0 and the fly must look inert.
 *
 * This file is a viewer, not part of the public API. See PREVIEW.md.
 */
import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { FlyStage, type StageStats } from './FlyStage';
import { STATE_LEN, syntheticFrame } from './fly/regions';

/**
 * Viewer-only switches, read once from the query string. They exist so the headless check
 * can hold the camera still and compare animation states against each other, and so it can
 * measure the bloom pass separately from the scene.
 *   ?frozen   no auto orbit: any pixel difference between two frames is the fly, not the camera
 *   ?nobloom  skip the composer and render straight to the canvas
 */
const PARAMS = new URLSearchParams(window.location.search);
const FROZEN_CAM = PARAMS.has('frozen');
const BLOOM = !PARAMS.has('nobloom');

const LIVE_MODES = ['intact', 'shuffled', 'no_edges', 'random_graph'] as const;
type LiveMode = (typeof LIVE_MODES)[number];

const CHOICES = ['none', ...LIVE_MODES] as const;
type Choice = (typeof CHOICES)[number];

const LABELS: Record<Choice, string> = {
  none: 'no props',
  intact: 'intact',
  shuffled: 'shuffled',
  no_edges: 'no_edges',
  random_graph: 'random_graph',
};

/** The app sends frames at 20 Hz; feed the harness at the same rate. */
const FRAME_HZ = 20;

type Frame = {
  activity: number[];
  stateRms: number;
  activeFraction: number;
};

function Demo() {
  const [choice, setChoice] = useState<Choice>('intact');
  const [correct, setCorrect] = useState<boolean | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [stats, setStats] = useState<StageStats | null>(null);

  // null means "mount FlyStage with no props at all".
  const liveMode: LiveMode | null = choice === 'none' ? null : choice;

  /**
   * StageStats go straight onto window, so the headless check can read the triangle and
   * draw call census without the viewer having to re-render twice a second on its behalf.
   */
  const onStats = (s: StageStats) => {
    (window as unknown as { __flyStats?: StageStats }).__flyStats = s;
  };

  useEffect(() => {
    const id = window.setInterval(() => {
      const s = (window as unknown as { __flyStats?: StageStats }).__flyStats;
      if (s) setStats(s);
    }, 700);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (choice === 'none') {
      setFrame(null);
      return;
    }
    const start = performance.now();
    let last = -1;
    let raf = 0;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const t = (now - start) / 1000;
      const slot = Math.floor(t * FRAME_HZ);
      if (slot === last) return;
      last = slot;
      if (choice === 'no_edges') {
        setFrame({ activity: new Array<number>(STATE_LEN).fill(0), stateRms: 0, activeFraction: 0 });
        return;
      }
      const f = syntheticFrame(t);
      setFrame({ activity: f.state, stateRms: f.stateRms, activeFraction: f.activeFraction });
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [choice]);

  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      {liveMode ? (
        <FlyStage
          activity={frame?.activity}
          stateRms={frame?.stateRms ?? 0}
          activeFraction={frame?.activeFraction ?? 0}
          correct={correct}
          reward={correct ? 1 : 0}
          mode={liveMode}
          dev
          bloom={BLOOM}
          autoRotate={!FROZEN_CAM}
          onStats={onStats}
        />
      ) : (
        <FlyStage bloom={BLOOM} autoRotate={!FROZEN_CAM} onStats={onStats} />
      )}

      {/* standalone-only switcher, bottom right: the stage's own UI owns the top corners
          and the bottom left, so nothing overlaps */}
      <div
        style={{
          position: 'absolute',
          right: 12,
          bottom: 12,
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
          padding: '8px 10px',
          border: '1px solid rgba(120,150,190,0.28)',
          borderRadius: 6,
          background: 'rgba(6,10,16,0.72)',
          font: '11px/1.3 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
          color: '#93a4bb',
          zIndex: 10,
        }}
      >
        <div style={{ color: '#5f7186', letterSpacing: '0.08em' }}>STANDALONE (no backend)</div>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: 230 }}>
          {CHOICES.map((c) => (
            <button
              key={c}
              onClick={() => setChoice(c)}
              style={{
                font: 'inherit',
                padding: '3px 6px',
                borderRadius: 4,
                cursor: 'pointer',
                color: c === choice ? '#e6f4ff' : '#93a4bb',
                background: c === choice ? 'rgba(34,211,238,0.18)' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${c === choice ? 'rgba(34,211,238,0.5)' : 'rgba(120,150,190,0.24)'}`,
              }}
            >
              {LABELS[c]}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          {([null, true, false] as const).map((v) => (
            <button
              key={String(v)}
              disabled={!frame}
              onClick={() => setCorrect(v)}
              style={{
                font: 'inherit',
                padding: '3px 6px',
                borderRadius: 4,
                cursor: frame ? 'pointer' : 'default',
                opacity: frame ? 1 : 0.4,
                color: v === correct ? '#e6f4ff' : '#93a4bb',
                background: v === correct ? 'rgba(251,191,36,0.16)' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${v === correct ? 'rgba(251,191,36,0.5)' : 'rgba(120,150,190,0.24)'}`,
              }}
            >
              {v === null ? 'correct=null' : v ? 'correct=true' : 'correct=false'}
            </button>
          ))}
        </div>
        {stats && (
          <div
            data-testid="fly-stats"
            style={{ color: '#5f7186', fontSize: 10, lineHeight: 1.45, maxWidth: 230 }}
          >
            {`tris ${stats.triangles.toLocaleString()} | meshes ${stats.meshes}`}
            <br />
            {`setae ${stats.instances} | calls ${stats.drawCalls}`}
            <br />
            {`programs ${stats.programs} | geo ${stats.geometries} | tex ${stats.textures}`}
            <br />
            {`${BLOOM ? 'bloom on' : 'bloom off'} | ${FROZEN_CAM ? 'camera frozen' : 'auto orbit'}`}
          </div>
        )}
      </div>
    </div>
  );
}

const root = document.getElementById('root');
if (!root) throw new Error('#root is missing from index.html');
createRoot(root).render(<Demo />);
