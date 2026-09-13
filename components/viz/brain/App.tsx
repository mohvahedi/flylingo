/**
 * Demo app for the BrainCloud harness.
 *
 * Responsibilities:
 *   - own exactly one load of the real MaleCNS layout in public/data and tell
 *     the viewer, in the UI, whether the cloud is showing measured soma
 *     coordinates or the procedural fallback (with the accuracy numbers);
 *   - connect to the brain service at ws://127.0.0.1:8770/stream and feed the
 *     real frames into BrainCloud through the frozen prop contract;
 *   - fall back to the synthetic idle animation from useSyntheticFrame.ts when
 *     the socket is absent, so the page is never blank;
 *   - publish the shared metrics snapshot from src/metrics/fps.ts to
 *     window.__bc, which backs both the on screen FPS readout and
 *     scripts/measure-fps.mjs.
 *
 * Query parameters, used by scripts/measure-fps.mjs:
 *   drive=full|subset          initial BrainCloud drive mode (default subset)
 *   stream=off                 never open the socket, stay on the idle path
 *   stream=<ws url>            override the stream endpoint
 */

import { useCallback, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import BrainCloud from './components/BrainCloud';
import RasterStrip from './components/RasterStrip';
import { loadNeuronData, type NeuronDataMeta } from './data/loadNeuronData';
import { useBrainStream } from './hooks/useBrainStream';
import {
  buildAnatomicalLayout,
  buildProceduralLayout,
  type BrainLayout,
} from './layout';
import { buildLiveMapping } from './live/mapping';
import { EDGE_HUBS, EDGE_RADIUS } from './layout/edges';
import { DEFAULT_STREAM_URL } from './live/socket';
import { metrics, publishMetrics } from './metrics/fps';
import { ModeBadge } from './ui/ModeBadge';
import { FpsReadout } from './ui/FpsReadout';
import { MetricsPanel } from './ui/MetricsPanel';

/** Matches the seed used by the extractor notes, so the fallback is stable. */
const SEED = 7701;
/** layout_meta.json "count", the retained MaleCNS neuron count. */
const FALLBACK_COUNT = 166700;
/** The RasterStrip history length lives in the hook; this mirrors it for the caption. */
const HISTORY_LABEL = 'approx 12 s';
/** Measured state_rms per live mode. Magnitudes are nearly identical, so the badge carries the difference. */
const MODE_RMS_NOTE = 'intact 0.105 / shuffled 0.108 / random_graph 0.096';

type LayoutState =
  | { status: 'procedural'; layout: BrainLayout; reason: string; buildMs: number }
  | { status: 'anatomical'; layout: BrainLayout; buildMs: number };

function accuracyDetail(meta: NeuronDataMeta): string {
  return (
    `${meta.dataset}: ${meta.accuracy.neuronsWithSomaAnnotation} of ${meta.count} neurons carry ` +
    `a measured soma annotation (${meta.coordinateField}); ` +
    `${meta.accuracy.neuronsFilledFromGroupCentroid} are placed at their ` +
    `${meta.accuracy.matchingRule} because no soma was annotated. ` +
    `${meta.edges.toLocaleString()} directed edges, ${meta.degreeSource}.`
  );
}

function buildProceduralFallback(reason: string): LayoutState {
  const t0 = performance.now();
  const layout = buildProceduralLayout({ count: FALLBACK_COUNT, seed: SEED });
  return { status: 'procedural', layout, reason, buildMs: performance.now() - t0 };
}

function Pill({ children, tone = 'slate', title }: { children: ReactNode; tone?: string; title?: string }) {
  const tones: Record<string, { fg: string; bg: string; edge: string }> = {
    slate: { fg: '#9fb2c4', bg: 'rgba(120,140,160,0.14)', edge: 'rgba(159,178,196,0.4)' },
    green: { fg: '#7ff0c8', bg: 'rgba(46,160,126,0.16)', edge: 'rgba(127,240,200,0.45)' },
    amber: { fg: '#ffd479', bg: 'rgba(180,132,32,0.16)', edge: 'rgba(255,212,121,0.45)' },
    red: { fg: '#ff9d9d', bg: 'rgba(180,60,60,0.16)', edge: 'rgba(255,157,157,0.45)' },
  };
  const t = tones[tone] ?? tones.slate;
  const style: CSSProperties = {
    padding: '0.18rem 0.5rem',
    borderRadius: '999px',
    border: `1px solid ${t.edge}`,
    background: t.bg,
    color: t.fg,
    fontSize: '0.72rem',
    letterSpacing: '0.04em',
    whiteSpace: 'nowrap',
  };
  return (
    <span style={style} title={title}>
      {children}
    </span>
  );
}

export function App() {
  const [driveMode, setDriveMode] = useState<'full' | 'subset'>(() => {
    const raw = new URLSearchParams(window.location.search).get('drive');
    return raw === 'full' ? 'full' : 'subset';
  });
  const [streamEnabled] = useState(() => new URLSearchParams(window.location.search).get('stream') !== 'off');
  /**
   * Dev escape hatch: `?props=none` mounts BrainCloud with zero props, which
   * is the synthetic idle path. It exists so the "renders with no props"
   * requirement can be checked in a real browser instead of argued from code.
   */
  const [zeroProps] = useState(() => new URLSearchParams(window.location.search).get('props') === 'none');
  const [streamUrl] = useState(() => {
    const raw = new URLSearchParams(window.location.search).get('stream');
    if (raw === null || raw === 'off' || raw === 'on') return DEFAULT_STREAM_URL;
    return raw;
  });
  const [inspectId, setInspectId] = useState<string | null>(null);
  const [layoutState, setLayoutState] = useState<LayoutState>(() =>
    buildProceduralFallback('measured coordinates still loading'),
  );

  // One load of the real dataset per page. The anatomical layout wins as soon
  // as the binaries decode; on any failure the procedural layout stays up and
  // the reason string is shown to the viewer.
  useEffect(() => {
    let cancelled = false;
    const t0 = performance.now();
    loadNeuronData()
      .then((data) => {
        if (cancelled) return;
        const layout = buildAnatomicalLayout({
          count: data.count,
          positions: data.positions,
          inDegree: data.inDegree,
          classIndex: data.classIndex,
          ids: data.ids,
          badge: 'anatomical soma coordinates',
          coordinateNote: data.meta.coordinateNote,
          detail: accuracyDetail(data.meta),
        });
        setLayoutState({ status: 'anatomical', layout, buildMs: performance.now() - t0 });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const reason = error instanceof Error ? error.message : String(error);
        setLayoutState(buildProceduralFallback(`public/data unavailable (${reason})`));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const stream = useBrainStream(streamUrl, streamEnabled);
  const { frame, history, sampledIds } = stream;
  const live = stream.source === 'live';

  // Live mapping is rebuilt when the sampled id set changes. BrainCloud does
  // its own placement; this copy exists so the metrics panel reports how many
  // of the 512 slots landed on a real neuron.
  const mapping = useMemo(
    () => buildLiveMapping(layoutState.layout, sampledIds),
    [layoutState.layout, sampledIds],
  );

  // Publish what only the app knows. BrainCloud owns the renderer, points,
  // frame timing and reference level fields.
  useEffect(() => {
    metrics.mode = live ? stream.mode : 'idle';
    metrics.synthetic = !live;
    metrics.resolvedIds = mapping.resolved;
    metrics.mappingBasis = mapping.basis;
    publishMetrics();
  }, [live, stream.mode, mapping]);

  const onInspectSampled = useCallback(() => {
    setInspectId(sampledIds.length > 0 ? sampledIds[0] : null);
  }, [sampledIds]);

  const provenance = layoutState.layout.provenance;
  const anatomical = layoutState.status === 'anatomical';

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        minHeight: '100%',
        color: '#cfe3f5',
        background: '#080c11',
      }}
    >
      <header
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: '1rem',
          padding: '0.7rem 0.9rem',
          borderBottom: '1px solid rgba(120,160,200,0.18)',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.98rem', fontWeight: 700, letterSpacing: '0.02em' }}>FlyLingo BrainCloud harness</span>
            <ModeBadge mode={stream.mode} live={live} controls={stream.controls} />
            <Pill tone={live ? 'green' : 'amber'} title={`socket status: ${stream.status}`}>
              {live ? `live ${streamUrl}` : streamEnabled ? `socket ${stream.status}, idle animation` : 'socket disabled, idle animation'}
            </Pill>
            <Pill tone={anatomical ? 'green' : 'amber'} title={provenance.coordinateNote}>
              {provenance.badge}
            </Pill>
            <Pill tone="slate" title="BrainCloud drive mode: full uploads the activity attribute for every point each frame">
              drive {driveMode}
            </Pill>
          </div>
          <div style={{ fontSize: '0.72rem', color: '#7f98ad' }}>
            frozen BrainFrame contract: 512 floats in -1..1, spikes are indices into state, sampled ids are strings.
            Frame {stream.step} / challenge {stream.challengeId ?? 'none'} / t {stream.tMs.toFixed(0)} ms /{' '}
            {stream.framesReceived} frames received since open.
          </div>
        </div>
        <FpsReadout windowFrames={60} />
      </header>

      <main style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <section style={{ position: 'relative', flex: 1, minWidth: 0 }} data-testid="cloud-section">
          {zeroProps ? (
            <BrainCloud />
          ) : (
            <BrainCloud
              layout={layoutState.layout}
              state={frame.state}
              spikes={frame.spikes}
              sampledIds={frame.sampledIds}
              activeFraction={frame.activeFraction}
              stateRms={frame.stateRms}
              driveMode={driveMode}
              inspectId={inspectId}
            />
          )}

          <div
            style={{
              position: 'absolute',
              top: '0.7rem',
              left: '0.8rem',
              right: '0.8rem',
              pointerEvents: 'none',
              fontSize: '0.72rem',
              color: '#b9d2e6',
              textShadow: '0 1px 3px rgba(0,0,0,0.85)',
            }}
          >
            <div style={{ color: anatomical ? '#9fe8cd' : '#ffd479', fontWeight: 600 }}>
              {anatomical
                ? 'coordinates: measured soma voxels from the real connectome'
                : 'coordinates: procedural fallback, not anatomy'}
            </div>
            <div style={{ color: '#9fb2c4', maxWidth: '62ch' }}>{provenance.coordinateNote}</div>
            <div style={{ color: '#8fa6bb', maxWidth: '72ch', marginTop: '0.2rem' }}>
              {anatomical ? provenance.detail : `${layoutState.reason}; ${provenance.detail}`}
            </div>
          </div>

        </section>

        <aside
          style={{
            width: '310px',
            flex: '0 0 310px',
            borderLeft: '1px solid rgba(120,160,200,0.18)',
            padding: '0.7rem 0.8rem',
            overflowY: 'auto',
          }}
        >
          <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
            <button
              type="button"
              onClick={() => setDriveMode('subset')}
              style={buttonStyle(driveMode === 'subset')}
              title="static 166,700 point cloud, animated 512 live slots"
            >
              subset
            </button>
            <button
              type="button"
              onClick={() => setDriveMode('full')}
              style={buttonStyle(driveMode === 'full')}
              title="re-upload the activity attribute for all 166,700 points every frame"
            >
              full
            </button>
            <button
              type="button"
              onClick={onInspectSampled}
              disabled={sampledIds.length === 0}
              style={buttonStyle(false, sampledIds.length === 0)}
              title="highlight the layout point of the first sampled id in this frame"
            >
              inspect first sampled id
            </button>
            <button type="button" onClick={() => setInspectId(null)} disabled={inspectId === null} style={buttonStyle(false, inspectId === null)}>
              clear
            </button>
          </div>

          <div style={{ fontSize: '0.72rem', marginBottom: '0.6rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f98ad' }}>layout</span>
              <span style={{ color: '#dceaf7' }}>{anatomical ? 'anatomical' : 'procedural'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f98ad' }}>layout build</span>
              <span style={{ color: '#dceaf7' }}>{layoutState.buildMs.toFixed(0)} ms</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f98ad' }}>id index</span>
              <span style={{ color: '#dceaf7' }}>{layoutState.layout.idToIndex.size.toLocaleString()} ids</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f98ad' }}>inspect</span>
              <span style={{ color: '#dceaf7' }}>{inspectId ?? 'none'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f98ad' }}>frames received</span>
              <span style={{ color: '#dceaf7' }}>{stream.framesReceived.toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f98ad' }}>dropped spikes</span>
              <span style={{ color: '#dceaf7' }}>{stream.droppedSpikes}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f98ad' }}>frame rms</span>
              <span style={{ color: '#dceaf7' }}>{frame.stateRms.toFixed(4)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f98ad' }}>active fraction</span>
              <span style={{ color: '#dceaf7' }}>{frame.activeFraction.toFixed(4)}</span>
            </div>
          </div>

          <div style={{ fontSize: '0.68rem', color: '#7f98ad', lineHeight: 1.45 }}>
            <div>
              Auto range is the 95th percentile of |state| for the frame in view (measured p95 0.274).
              A frame whose |state| is all zeros renders empty on purpose: under the no_edges control
              every value is exactly 0.0 and spikes is exactly [], so an empty cloud is a real
              measurement, not a display failure.
            </div>
            <div style={{ marginTop: '0.25rem' }}>
              Mode is a label for the connectivity behind the frame, not a magnitude cue: state_rms is
              nearly identical across live modes ({MODE_RMS_NOTE}).
            </div>
            <div style={{ marginTop: '0.25rem' }}>
              Edges: the {EDGE_HUBS.toLocaleString()} highest in-degree hubs joined to their nearest
              same-class somas within {EDGE_RADIUS} world units, over measured soma positions. A sampled
              illustration, not measured adjacency. The real graph is 25,582,938 directed edges and is
              not drawn.
            </div>
            <div style={{ marginTop: '0.25rem' }}>
              Pulses: each active live slot sheds a decaying dotted trail along a fixed per-slot ray,
              driven only by the real spikes and the real state values in the frame.
            </div>
          </div>

          <MetricsPanel />
        </aside>
      </main>

      <footer style={{ borderTop: '1px solid rgba(120,160,200,0.18)', padding: '0.5rem 0.9rem 0.7rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#7f98ad', marginBottom: '0.3rem' }}>
          <span>
            raster strip: {history.length} frames ({HISTORY_LABEL}, oldest left), columns are frames,
            rows are the 512 state slots, bright marks are spikes
          </span>
          <span>{live ? 'source: live socket' : 'source: synthetic idle'}</span>
        </div>
        <RasterStrip history={history} height={200} />
      </footer>
    </div>
  );
}

function buttonStyle(active: boolean, disabled = false): CSSProperties {
  return {
    padding: '0.22rem 0.55rem',
    borderRadius: '6px',
    border: `1px solid ${active ? 'rgba(127,240,200,0.55)' : 'rgba(120,160,200,0.3)'}`,
    background: active ? 'rgba(46,160,126,0.18)' : 'rgba(20,32,44,0.7)',
    color: disabled ? '#5f7385' : active ? '#9fe8cd' : '#cfe3f5',
    fontSize: '0.72rem',
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.6 : 1,
  };
}

export default App;
