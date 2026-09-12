import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { buildAnatomicalLayout, type BrainLayout } from '../layout';
import { loadNeuronData } from '../data/loadNeuronData';
import { syntheticFrame } from '../hooks/useSyntheticFrame';
import { LIVE_FRAG, LIVE_VERT, STATIC_FRAG, STATIC_VERT, EDGE_FRAG, EDGE_VERT, TRAIL_FRAG, TRAIL_VERT } from '../shaders/points';
import {
  metrics,
  publishMetrics,
  recordFrame,
  referenceOf,
  resetMetrics,
  updateFps,
  type MetricsSnapshot,
} from '../metrics/fps';
import type { LiveFrame } from '../live/mapping';
import { buildSampledEdges } from '../layout/edges';
import { createTrail, TRAIL_CAPACITY, TRAIL_LIFE, updateTrail } from '../live/trail';

const LIVE_SLOTS = 512;

// ---------------------------------------------------------------------------
// Caption figures. Every number here is measured and is quoted from
// public/data/layout_meta.json: 166,700 retained MaleCNS v1.0 neurons,
// 25,582,938 directed edges in the graph, 139,668 distinct measured soma
// positions, 27,038 neurons with no soma annotation placed at a group centroid
// (139,662 + 27,038 = 166,700). The disclaimer below the figures is not
// decorative: the coordinates really are measured soma voxels, and the drawn
// edges really are a sampled illustration rather than the 25M edge graph.
// ---------------------------------------------------------------------------
const CONNECTOME_FACTS = {
  neurons: 166700,
  directedEdges: 25582938,
  measuredSomaPositions: 139668,
  centroidFilled: 27038,
};

const CAPTION_LABEL = 'MaleCNS v1.0 · measured connectome';
const CAPTION_VALUE =
  `${CONNECTOME_FACTS.neurons.toLocaleString('en-US')} neurons · ` +
  `${CONNECTOME_FACTS.directedEdges.toLocaleString('en-US')} directed edges · ` +
  `${CONNECTOME_FACTS.measuredSomaPositions.toLocaleString('en-US')} measured soma positions`;
const CAPTION_NOTE =
  `Coordinates are measured soma voxels; the ${CONNECTOME_FACTS.centroidFilled.toLocaleString('en-US')} ` +
  'neurons with no soma annotation sit at their class and in-degree decile group centroid. ' +
  'All 166,700 neurons are drawn; the hairline edges are a sampled subset of the ' +
  'highest in-degree hubs, not the measured edge list.';

export interface BrainCloudProps {
  /** frame.state, -1..1, length 512 */
  state?: number[];
  /** indices INTO state */
  spikes?: number[];
  sampledIds?: string[];
  activeFraction?: number;
  stateRms?: number;
  width?: number;
  height?: number;

  // ---------------------------------------------------------------------
  // Harness-only props. All optional, so the app can mount the component
  // with nothing but the frozen contract.
  // ---------------------------------------------------------------------
  /** Pre-built layout. The harness supplies the real MaleCNS one. */
  layout?: BrainLayout;
  /**
   * "full" re-uploads the activity attribute for all 166,700 points every
   * frame. "subset" leaves the big cloud static and animates only the 512
   * live slots. Both are measured; see PREVIEW.md.
   */
  driveMode?: 'full' | 'subset';
  /** Called on every rendered frame with the shared metrics snapshot. */
  onMetrics?: (m: MetricsSnapshot) => void;
  inspectId?: string | null;
}

/** Decide which cloud point each live slot sits on. */
function resolveLiveIndices(layout: BrainLayout, sampledIds: string[]): Int32Array {
  const out = new Int32Array(LIVE_SLOTS).fill(-1);
  if (sampledIds.length > 0) {
    for (let i = 0; i < LIVE_SLOTS && i < sampledIds.length; i += 1) {
      const idx = layout.idToIndex.get(sampledIds[i]);
      if (idx !== undefined) out[i] = idx;
    }
    return out;
  }
  // No ids in the frame: place the slots on a deterministic spread of real
  // neurons. The placement is arbitrary and the UI reports it as such.
  const step = Math.max(1, Math.floor(layout.count / LIVE_SLOTS));
  for (let i = 0; i < LIVE_SLOTS; i += 1) out[i] = layout.order[(i * step) % layout.count];
  return out;
}

function Cloud({
  layout,
  frame,
  liveIndices,
  driveMode,
  inspectIndex,
}: {
  layout: BrainLayout;
  frame: LiveFrame;
  liveIndices: Int32Array;
  driveMode: 'full' | 'subset';
  inspectIndex: number;
}) {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  // Drawing-buffer height, used to size points in resolution-independent pixels.
  const viewport = useThree((s) => s.size);

  const fieldAttr = useRef<THREE.BufferAttribute | null>(null);
  const activityAttr = useRef<THREE.BufferAttribute | null>(null);
  const shockAttr = useRef<THREE.BufferAttribute | null>(null);
  const posAttr = useRef<THREE.BufferAttribute | null>(null);
  const activityArr = useRef<Float32Array | null>(null);
  const shockArr = useRef<Float32Array | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const lastFrameTime = useRef(0);
  const shockState = useRef(new Float32Array(LIVE_SLOTS));
  const lastSpikeKey = useRef('');
  const refScratch = useRef(new Float32Array(LIVE_SLOTS));
  const refValue = useRef(0);
  const trailPosAttr = useRef<THREE.BufferAttribute | null>(null);
  const trailColAttr = useRef<THREE.BufferAttribute | null>(null);
  const trailAgeAttr = useRef<THREE.BufferAttribute | null>(null);
  const trailPowAttr = useRef<THREE.BufferAttribute | null>(null);

  // -------------------------------------------------------------------
  // Geometry and materials. Built once, never reallocated.
  // -------------------------------------------------------------------
  const staticGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(layout.positions, 3));
    g.setAttribute('aBase', new THREE.BufferAttribute(layout.baseIntensity, 1));
    // One mutable float per point. This is the only attribute that changes in
    // "full" mode; in "subset" mode it is written once and left alone.
    const field = new THREE.BufferAttribute(new Float32Array(layout.count), 1);
    field.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aField', field);
    fieldAttr.current = field;
    // The cloud spans roughly [-1, 1]. Setting the bounding sphere by hand
    // keeps three.js from walking 166,700 points to compute it.
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 3.2);
    return g;
  }, [layout]);

  const staticMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: STATIC_VERT,
        fragmentShader: STATIC_FRAG,
        uniforms: {
          uSize: { value: 2.4 },
          uPixelRatio: { value: 1 },
          uViewHeight: { value: 600 },
          uGlobal: { value: 0.85 },
          uAmbient: { value: 0.1 },
          uRef: { value: 0.274 },
          uColorDim: { value: new THREE.Color('#5bc8d6') },
          uColorHot: { value: new THREE.Color('#bfeff7') },
          uColorNeg: { value: new THREE.Color('#6fa6de') },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );

  const liveGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const positions = new Float32Array(LIVE_SLOTS * 3);
    const activity = new Float32Array(LIVE_SLOTS);
    const shock = new Float32Array(LIVE_SLOTS);
    const pa = new THREE.BufferAttribute(positions, 3);
    const aa = new THREE.BufferAttribute(activity, 1);
    const sa = new THREE.BufferAttribute(shock, 1);
    pa.setUsage(THREE.DynamicDrawUsage);
    aa.setUsage(THREE.DynamicDrawUsage);
    sa.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', pa);
    g.setAttribute('aActivity', aa);
    g.setAttribute('aShock', sa);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 3.2);
    posAttr.current = pa;
    activityAttr.current = aa;
    shockAttr.current = sa;
    activityArr.current = activity;
    shockArr.current = shock;
    return g;
  }, []);

  const liveMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: LIVE_VERT,
        fragmentShader: LIVE_FRAG,
        uniforms: {
          uSize: { value: 6.0 },
          uPixelRatio: { value: 1 },
          uViewHeight: { value: 600 },
          uRef: { value: 0.274 },
          uPos: { value: new THREE.Color('#7fe0ea') },
          uNeg: { value: new THREE.Color('#6fa6de') },
          uShockColor: { value: new THREE.Color('#f0a030') },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );

  // -------------------------------------------------------------------
  // Sampled hairline edges. Built once per layout from the real high
  // in-degree hubs over the measured positions. src/layout/edges.ts states
  // exactly what is real here: the hubs and the somas are, the pairing is a
  // nearest-same-class illustration and is not measured adjacency.
  // -------------------------------------------------------------------
  const edgeSet = useMemo(() => buildSampledEdges(layout), [layout]);
  const edgeGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(edgeSet.positions, 3));
    g.setAttribute('aWeight', new THREE.BufferAttribute(edgeSet.weights, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 3.4);
    return g;
  }, [edgeSet]);
  const edgeMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: EDGE_VERT,
        fragmentShader: EDGE_FRAG,
        uniforms: {
          uColor: { value: new THREE.Color('#5bc8d6') },
          uOpacity: { value: 0.13 },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );

  // -------------------------------------------------------------------
  // Travelling pulse trail. One buffer, one draw call, reused every frame:
  // the geometry is never rebuilt, only the attributes are rewritten.
  // -------------------------------------------------------------------
  const trail = useMemo(() => createTrail(layout), [layout]);
  const trailGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pa = new THREE.BufferAttribute(trail.positions, 3);
    const ca = new THREE.BufferAttribute(trail.colors, 3);
    const ga = new THREE.BufferAttribute(trail.ages, 1);
    const wa = new THREE.BufferAttribute(trail.powers, 1);
    pa.setUsage(THREE.DynamicDrawUsage);
    ca.setUsage(THREE.DynamicDrawUsage);
    ga.setUsage(THREE.DynamicDrawUsage);
    wa.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', pa);
    g.setAttribute('aColor', ca);
    g.setAttribute('aAge', ga);
    g.setAttribute('aPower', wa);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 3.4);
    g.setDrawRange(0, 0);
    trailPosAttr.current = pa;
    trailColAttr.current = ca;
    trailAgeAttr.current = ga;
    trailPowAttr.current = wa;
    return g;
  }, [trail]);
  const trailMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: TRAIL_VERT,
        fragmentShader: TRAIL_FRAG,
        uniforms: {
          uSize: { value: 3.6 },
          uPixelRatio: { value: 1 },
          uViewHeight: { value: 600 },
          uLife: { value: TRAIL_LIFE },
          uAlpha: { value: 0.85 },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );

  useEffect(
    () => () => {
      staticGeo.dispose();
      staticMat.dispose();
      liveGeo.dispose();
      liveMat.dispose();
      edgeGeo.dispose();
      edgeMat.dispose();
      trailGeo.dispose();
      trailMat.dispose();
    },
    [staticGeo, staticMat, liveGeo, liveMat, edgeGeo, edgeMat, trailGeo, trailMat],
  );

  // -------------------------------------------------------------------
  // Renderer settings, camera controls, metrics bootstrap.
  // -------------------------------------------------------------------
  useEffect(() => {
    const ctx = gl.getContext();
    const info = ctx.getExtension('WEBGL_debug_renderer_info');
    metrics.renderer = info
      ? String(ctx.getParameter(info.UNMASKED_RENDERER_WEBGL))
      : 'renderer string unavailable';
    metrics.points = layout.count;
    metrics.liveSlots = LIVE_SLOTS;
    metrics.edgeCount = edgeSet.count;
    metrics.edgeHubs = edgeSet.hubs;
    metrics.edgeMeanLength = edgeSet.meanLength;
    metrics.edgeSource = edgeSet.source;
    metrics.trailCapacity = TRAIL_CAPACITY;
    metrics.caption = CAPTION_VALUE;
    metrics.driveMode = driveMode;
    metrics.bytesPerFrame =
      driveMode === 'full' ? layout.count * 4 : LIVE_SLOTS * (3 * 4 + 4 + 4);

    const controls = new OrbitControls(camera, gl.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.6;
    controls.zoomSpeed = 0.9;
    camera.position.set(2.1, 1.25, 1.75);
    controls.target.set(0, 0, 0);
    controlsRef.current = controls;

    resetMetrics();
    publishMetrics();
    return () => {
      controls.dispose();
      controlsRef.current = null;
    };
  }, [gl, camera, layout, driveMode, edgeSet]);

  // -------------------------------------------------------------------
  // Live slot positions: written only when the mapping changes.
  // -------------------------------------------------------------------
  const lastMapping = useRef<Int32Array | null>(null);
  useEffect(() => {
    const pa = posAttr.current;
    if (!pa || lastMapping.current === liveIndices) return;
    lastMapping.current = liveIndices;
    const arr = pa.array as Float32Array;
    const src = layout.positions;
    for (let s = 0; s < LIVE_SLOTS; s += 1) {
      const idx = s < liveIndices.length ? liveIndices[s] : -1;
      if (idx >= 0) {
        arr[s * 3] = src[idx * 3];
        arr[s * 3 + 1] = src[idx * 3 + 1];
        arr[s * 3 + 2] = src[idx * 3 + 2];
      } else {
        // Unresolved slot: parked offscreen rather than drawn at an invented
        // position. Its activity is forced to zero every frame.
        arr[s * 3] = 1e4;
        arr[s * 3 + 1] = 1e4;
        arr[s * 3 + 2] = 1e4;
      }
    }
    pa.needsUpdate = true;
  }, [liveIndices, layout]);

  // -------------------------------------------------------------------
  // Per-frame attribute updates.
  // -------------------------------------------------------------------
  useFrame(() => {
    const nslots = frame.state.length || LIVE_SLOTS;
    const hasIds = frame.sampledIds.length > 0;

    // ------------------------------------------------------------------
    // Auto-range. Measured telemetry: median |state| 0.051, p95 0.274,
    // state_rms 0.105. A linear 0..1 map would be near black, so every
    // frame is normalised against its own 95th percentile. A dead frame
    // (no_edges: all zeros) yields a reference of exactly 0, which the
    // shaders treat as "contribute nothing" rather than amplifying dust.
    // ------------------------------------------------------------------
    const p95 = referenceOf(frame.state, refScratch.current);
    const prev = refValue.current;
    // Rise fast so a genuine burst is not dimmed, fall slowly so the scale
    // does not flicker frame to frame. Exact zero is never smoothed away.
    const next = p95 === 0 ? 0 : prev === 0 ? p95 : prev + (p95 - prev) * (p95 > prev ? 0.5 : 0.08);
    refValue.current = next;
    let peak = 0;
    for (let i = 0; i < nslots; i += 1) {
      const a = Math.abs(frame.state[i]);
      if (a > peak) peak = a;
    }
    metrics.refValue = next;
    metrics.peakAbs = peak;
    metrics.spikeCount = frame.spikes.length;

    if (driveMode === 'full' && fieldAttr.current) {
      const fa = fieldAttr.current;
      const arr = fa.array as Float32Array;
      if (hasIds) {
        // Real per-neuron values: decay everything, then write the ids the
        // frame actually names. Nothing is invented for unnamed neurons.
        for (let i = 0; i < arr.length; i += 1) arr[i] *= 0.93;
        for (let s = 0; s < nslots && s < liveIndices.length; s += 1) {
          const idx = liveIndices[s];
          if (idx >= 0) arr[idx] = frame.state[s];
        }
      } else {
        // No ids available: spread the 512 values over the cloud so the whole
        // shape breathes. This is a display spread, not biology, and the UI
        // labels the frame as synthetic.
        for (let i = 0; i < arr.length; i += 1) {
          const slot = ((i * nslots) / arr.length) | 0;
          const v = frame.state[slot < nslots ? slot : nslots - 1];
          arr[i] = arr[i] * 0.6 + v * 0.4;
        }
      }
      fa.needsUpdate = true;
      metrics.fieldUploadBytes = arr.length * 4;
    } else {
      metrics.fieldUploadBytes = 0;
    }

    // Live overlay: 1,024 floats per frame in both modes.
    const aa = activityAttr.current;
    const sa = shockAttr.current;
    if (aa && sa && activityArr.current && shockArr.current) {
      const av = activityArr.current;
      const sv = shockArr.current;
      const spiked = frame.spikes;
      const spikeKey = `${spiked.length}:${spiked.length ? spiked[0] : -1}:${spiked.length ? spiked[spiked.length - 1] : -1}`;
      const freshSpikes = spikeKey !== lastSpikeKey.current;
      lastSpikeKey.current = spikeKey;
      if (freshSpikes) {
        for (let k = 0; k < spiked.length; k += 1) {
          const s = spiked[k];
          if (s >= 0 && s < LIVE_SLOTS) shockState.current[s] = 1;
        }
      }
      for (let s = 0; s < LIVE_SLOTS; s += 1) {
        const idx = s < liveIndices.length ? liveIndices[s] : -1;
        const live = s < nslots && (idx >= 0 || !hasIds);
        av[s] = live ? frame.state[s] : 0;
        sv[s] = live ? shockState.current[s] : 0;
        if (shockState.current[s] > 0) shockState.current[s] *= 0.72;
        if (shockState.current[s] < 0.004) shockState.current[s] = 0;
      }
      aa.needsUpdate = true;
      sa.needsUpdate = true;
    }

    // Frame delta in seconds, shared by the trail and the fps record below.
    const frameNow = performance.now();
    const dtMs = lastFrameTime.current === 0 ? 16.7 : frameNow - lastFrameTime.current;
    lastFrameTime.current = frameNow;
    const dtSec = Math.min(0.05, dtMs / 1000);

    // ------------------------------------------------------------------
    // Travelling pulses. Driven by the same real spikes and real state
    // values as the live overlay: a dead frame spawns nothing and the
    // buffer empties by decay alone.
    // ------------------------------------------------------------------
    const liveDots = updateTrail(
      trail,
      liveIndices,
      frame.state,
      frame.spikes,
      refValue.current,
      nslots,
      hasIds,
      dtSec,
    );
    trailGeo.setDrawRange(0, liveDots);
    if (liveDots > 0) {
      if (trailPosAttr.current) trailPosAttr.current.needsUpdate = true;
      if (trailColAttr.current) trailColAttr.current.needsUpdate = true;
      if (trailAgeAttr.current) trailAgeAttr.current.needsUpdate = true;
      if (trailPowAttr.current) trailPowAttr.current.needsUpdate = true;
    }
    metrics.trailPoints = liveDots;
    metrics.trailUploadBytes = liveDots > 0 ? TRAIL_CAPACITY * 8 * 4 : 0;

    const pr = gl.getPixelRatio();
    const viewHeight = viewport.height || gl.domElement.height || 600;
    staticMat.uniforms.uPixelRatio.value = pr;
    staticMat.uniforms.uViewHeight.value = viewHeight;
    staticMat.uniforms.uRef.value = refValue.current;
    // Ambient structure light. This is what draws the anatomy: every non-active
    // neuron is rendered in uColorDim at this alpha, blended additively, so the
    // value has to be high enough to clear black on its own. Measured: at 0.55 with
    // a dark navy uColorDim the whole cloud landed near RGB(5,14,24) and the brain
    // was invisible. Driven near zero on a dead frame so no_edges looks genuinely dark.
    staticMat.uniforms.uAmbient.value = refValue.current > 0 ? 0.75 : 0.06;
    // The sampled edges are a display aid, not a claim about a dead frame, so
    // they follow the same activity gate as the ambient term. Under the
    // no_edges control the edge layer contributes nothing at all.
    edgeMat.uniforms.uOpacity.value = refValue.current > 0 ? 0.13 : 0;
    trailMat.uniforms.uPixelRatio.value = pr;
    trailMat.uniforms.uViewHeight.value = viewHeight;
    trailMat.uniforms.uAlpha.value = refValue.current > 0 ? 0.85 : 0;
    staticMat.uniforms.uGlobal.value = 0.8 + 0.5 * frame.stateRms;
    liveMat.uniforms.uPixelRatio.value = pr;
    liveMat.uniforms.uViewHeight.value = viewHeight;
    liveMat.uniforms.uRef.value = refValue.current;

    controlsRef.current?.update();

    const now = performance.now();
    const dt = lastFrameTime.current === 0 ? 16.7 : now - lastFrameTime.current;
    lastFrameTime.current = now;
    recordFrame(dt);
    metrics.frames += 1;
    metrics.lastFrameMs = dt;
    updateFps();
    publishMetrics();
  });

  // Brighten the inspected point by lifting its own aField entry.
  useEffect(() => {
    const fa = fieldAttr.current;
    if (!fa || inspectIndex < 0) return;
    const arr = fa.array as Float32Array;
    arr[inspectIndex] = 1;
    fa.needsUpdate = true;
  }, [inspectIndex]);

  return (
    <>
      <points geometry={staticGeo} material={staticMat} frustumCulled={false} />
      <lineSegments geometry={edgeGeo} material={edgeMat} frustumCulled={false} />
      <points geometry={trailGeo} material={trailMat} frustumCulled={false} />
      <points geometry={liveGeo} material={liveMat} frustumCulled={false} />
    </>
  );
}

function ResizeSync({ width, height }: { width?: number; height?: number }) {
  const set = useThree((s) => s.set);
  const size = useThree((s) => s.size);
  useEffect(() => {
    if (width && height && (size.width !== width || size.height !== height)) {
      set({ size: { width, height, top: 0, left: 0 } });
    }
  }, [width, height, set, size.width, size.height]);
  return null;
}

/**
 * Frozen prop contract component.
 *
 *   state: number[]        // frame.state, -1..1, length 512
 *   spikes: number[]       // indices INTO state
 *   sampledIds: string[]
 *   activeFraction: number
 *   stateRms: number
 *   width?: number; height?: number
 *
 * Renders correctly with no props at all: it falls back to the synthetic idle
 * animation and the harness badge reports the frame as synthetic. It never
 * throws on a partial or empty frame.
 */
export function BrainCloud({
  state,
  spikes,
  sampledIds,
  activeFraction = 0,
  stateRms = 0,
  width,
  height,
  layout,
  driveMode = 'subset',
  inspectId = null,
}: BrainCloudProps) {
  const [loaded, setLoaded] = useState<BrainLayout | null>(layout ?? null);
  const usingSynthetic = !state || state.length === 0;

  // Load the real MaleCNS metadata once, unless a layout was supplied.
  useEffect(() => {
    if (layout) {
      setLoaded(layout);
      return;
    }
    if (loaded) return;
    let cancelled = false;
    loadNeuronData()
      .then((data) => {
        if (cancelled) return;
        setLoaded(
          buildAnatomicalLayout({
            count: data.count,
            positions: data.positions,
            inDegree: data.inDegree,
            classIndex: data.classIndex,
            ids: data.ids,
            badge: 'anatomical soma coordinates',
            coordinateNote: data.meta.coordinateNote,
            detail:
              `${data.meta.dataset}: ${data.meta.accuracy.neuronsWithSomaAnnotation} of ` +
              `${data.count} neurons carry a measured soma annotation; ` +
              `${data.meta.accuracy.neuronsFilledFromGroupCentroid} are placed at their ` +
              `(class, degree decile) group centroid because no soma was annotated.`,
          }),
        );
      })
      .catch(() => {
        if (!cancelled) setLoaded(null);
      });
    return () => {
      cancelled = true;
    };
  }, [layout, loaded]);

  // Synthetic idle animation, driven only when the caller supplies no state.
  const [tMs, setTMs] = useState(0);
  useEffect(() => {
    if (!usingSynthetic) return;
    let raf = 0;
    const t0 = performance.now();
    const loop = () => {
      setTMs(performance.now() - t0);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [usingSynthetic]);

  const frame: LiveFrame = useMemo(() => {
    if (state && state.length > 0) {
      return {
        state,
        spikes: spikes ?? [],
        sampledIds: sampledIds ?? [],
        activeFraction,
        stateRms,
      };
    }
    return { ...syntheticFrame(tMs / 1000), sampledIds: [] };
  }, [state, spikes, sampledIds, activeFraction, stateRms, tMs]);

  const liveIndices = useMemo(
    () => (loaded ? resolveLiveIndices(loaded, frame.sampledIds) : new Int32Array(LIVE_SLOTS).fill(-1)),
    [loaded, frame.sampledIds],
  );

  const inspectIndex = useMemo(() => {
    if (!loaded || !inspectId) return -1;
    const idx = loaded.idToIndex.get(inspectId);
    return idx === undefined ? -1 : idx;
  }, [loaded, inspectId]);

  return (
    <div
      style={{
        position: 'relative',
        width: width ? `${width}px` : '100%',
        height: height ? `${height}px` : '100%',
        // Deep near-black HUD ground. The reference ground is #080C11.
        background: '#080c11',
        overflow: 'hidden',
      }}
    >
      <Canvas
        dpr={[1, 1.5]}
        gl={{ antialias: false, powerPreference: 'high-performance', alpha: false }}
        camera={{ fov: 40, near: 0.01, far: 40, position: [2.1, 1.25, 1.75] }}
        style={{ position: 'absolute', inset: 0 }}
      >
        <ResizeSync width={width} height={height} />
        <color attach="background" args={['#080c11']} />
        {loaded ? (
          <Cloud
            layout={loaded}
            frame={frame}
            liveIndices={liveIndices}
            driveMode={driveMode}
            inspectIndex={inspectIndex}
          />
        ) : null}
      </Canvas>
      {/*
        The caption. Real figures only, and the note underneath is the honesty
        clause: measured soma voxels, group centroid for the unannotated 27,038,
        sampled edges. Styled as a wide-tracked uppercase label plus a value
        line, which is how the reference broadcasts its numbers.
      */}
      <div
        data-testid="connectome-caption"
        style={{
          position: 'absolute',
          left: '1.1rem',
          bottom: '0.95rem',
          pointerEvents: 'none',
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          textShadow: '0 1px 4px rgba(0,0,0,0.9)',
        }}
      >
        <div
          data-testid="connectome-caption-label"
          style={{
            fontSize: '0.58rem',
            letterSpacing: '0.34em',
            textTransform: 'uppercase',
            color: '#6c8aa1',
          }}
        >
          {CAPTION_LABEL}
        </div>
        <div
          data-testid="connectome-caption-value"
          style={{
            marginTop: '0.3rem',
            fontSize: '0.98rem',
            letterSpacing: '0.04em',
            color: '#d8eef6',
          }}
        >
          {CAPTION_VALUE}
        </div>
        <div
          data-testid="connectome-caption-note"
          style={{
            marginTop: '0.3rem',
            maxWidth: '58ch',
            fontSize: '0.6rem',
            lineHeight: 1.45,
            color: '#5e7a90',
          }}
        >
          {CAPTION_NOTE}
        </div>
      </div>
      {!loaded ? (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'grid',
            placeItems: 'center',
            color: '#5b7ea6',
            fontFamily: 'ui-monospace, monospace',
            fontSize: 12,
          }}
        >
          loading neuron metadata...
        </div>
      ) : null}
    </div>
  );
}

export default BrainCloud;
