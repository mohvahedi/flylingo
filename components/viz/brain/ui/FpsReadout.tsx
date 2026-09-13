/**
 * Live FPS readout.
 *
 * The number comes from src/metrics/fps.ts (updateFps, driven by the RAF loop
 * inside BrainCloud), not from a counter invented here. Reading the same
 * snapshot that scripts/measure-fps.mjs reads keeps the on screen number and
 * the reported number in agreement.
 */

import { useMetricsSnapshot } from './useMetricsSnapshot';

export interface FpsReadoutProps {
  /** Rolling window the fps value is measured over, for the label. */
  windowFrames?: number;
}

export function FpsReadout({ windowFrames }: FpsReadoutProps) {
  const m = useMetricsSnapshot(200);
  const hasFrames = m.frames > 0;

  return (
    <div style={{ textAlign: 'right', lineHeight: 1.15 }}>
      <div style={{ fontSize: '1.5rem', fontWeight: 700, color: hasFrames ? '#eaf6ff' : '#5f7385' }}>
        {hasFrames ? m.fps.toFixed(1) : '--'}
        <span style={{ fontSize: '0.7rem', fontWeight: 400, marginLeft: '0.25rem', color: '#7f98ad' }}>fps</span>
      </div>
      <div style={{ fontSize: '0.68rem', color: '#7f98ad' }}>
        {windowFrames ? `rolling ${windowFrames} frames | ` : ''}
        avg {m.avgFrameMs.toFixed(2)} ms | worst {m.worstFrameMs.toFixed(2)} ms
      </div>
      <div style={{ fontSize: '0.68rem', color: '#5f7385' }}>
        {m.points.toLocaleString()} points | {m.liveSlots} live slots | drive {m.driveMode}
      </div>
    </div>
  );
}

export default FpsReadout;
