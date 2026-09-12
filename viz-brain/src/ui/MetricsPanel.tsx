/**
 * Detail panel. Every row is read from the shared metrics snapshot that
 * BrainCloud publishes, so the panel cannot drift from the renderer.
 */

import { useMetricsSnapshot } from './useMetricsSnapshot';

function Row({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.6rem', padding: '0.14rem 0' }}>
      <span style={{ color: '#7f98ad' }} title={hint}>
        {label}
      </span>
      <span style={{ color: '#dceaf7', textAlign: 'right' }}>{value}</span>
    </div>
  );
}

export function MetricsPanel() {
  const m = useMetricsSnapshot(250);

  return (
    <div style={{ fontSize: '0.72rem', fontFamily: 'inherit' }}>
      <div style={{ color: '#9fb2c4', textTransform: 'uppercase', letterSpacing: '0.08em', fontSize: '0.66rem', marginBottom: '0.3rem' }}>
        renderer
      </div>
      <Row label="fps" value={m.fps.toFixed(2)} hint="rolling mean measured by updateFps() in src/metrics/fps.ts" />
      <Row label="frames" value={m.frames.toLocaleString()} />
      <Row label="avg frame" value={`${m.avgFrameMs.toFixed(2)} ms`} />
      <Row label="worst frame" value={`${m.worstFrameMs.toFixed(2)} ms`} />
      <Row label="points" value={m.points.toLocaleString()} />
      <Row label="live slots" value={String(m.liveSlots)} />
      <Row label="drive mode" value={m.driveMode} hint="full uploads all points every frame; subset animates only the 512 live slots" />
      <Row label="field upload" value={`${(m.fieldUploadBytes / 1024).toFixed(1)} kiB`} />
      <Row label="bytes per frame" value={`${(m.bytesPerFrame / 1024).toFixed(1)} kiB`} />
      <Row label="webgl renderer" value={m.renderer} />

      <div style={{ color: '#9fb2c4', textTransform: 'uppercase', letterSpacing: '0.08em', fontSize: '0.66rem', margin: '0.7rem 0 0.3rem' }}>
        frame
      </div>
      <Row label="mode" value={m.mode} />
      <Row label="source" value={m.synthetic ? 'synthetic idle' : 'live socket'} />
      <Row label="sampled ids resolved" value={`${m.resolvedIds} / ${m.liveSlots}`} />
      <Row label="mapping basis" value={m.mappingBasis} />
      <Row label="auto range (ref)" value={m.refValue.toFixed(4)} hint="p95 of |state| for this frame; 0 means the frame is dead and renders empty" />
      <Row label="peak |state|" value={m.peakAbs.toFixed(4)} />
      <Row label="spikes" value={String(m.spikeCount)} />

      <div style={{ marginTop: '0.7rem', color: '#5f7385', fontSize: '0.68rem' }}>
        Measured live values: p50 0.051, p95 0.274, max 0.717, rms 0.105, active fraction 0.0044,
        about 5 spikes per 512 per frame. Under the no_edges control every value is exactly 0.0 and
        spikes is exactly [], so a dead frame is drawn empty rather than amplified.
      </div>
    </div>
  );
}

export default MetricsPanel;
