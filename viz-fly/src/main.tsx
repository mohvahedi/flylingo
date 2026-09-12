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
import { FlyStage } from './FlyStage';
import { STATE_LEN, syntheticFrame } from './fly/regions';

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

  /** null means "mount FlyStage with no props at all". */
  const liveMode: LiveMode | null = choice === 'none' ? null : choice;

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
        />
      ) : (
        <FlyStage />
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
      </div>
    </div>
  );
}

const root = document.getElementById('root');
if (!root) throw new Error('#root is missing from index.html');
createRoot(root).render(<Demo />);
