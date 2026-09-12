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
import { detectCentroidFill } from '../layout/centroidFill';
import { createTrail, TRAIL_CAPACITY, TRAIL_LIFE, updateTrail } from '../live/trail';

const LIVE_SLOTS = 512;

/**
 * Queue a partial attribute upload. The live prefix is written every frame and
 * the rest of the buffer is stale, so only that prefix is worth sending.
 */
function markUploaded(attr: THREE.BufferAttribute, count: number, itemSize: number): void {
  attr.clearUpdateRanges();
  attr.addUpdateRange(0, Math.max(0, count * itemSize));
  attr.needsUpdate = true;
}

/**
 * Wall-clock time constant of the amber spike flash. The old code decayed the
 * flash by a fixed 0.72 per frame, which lasted about 170 ms at 60 fps and
 * under a millisecond in an unthrottled headless run, so the amber was
 * invisible to a screenshot. This is the time to fall to 1/e.
 */
const SPIKE_FLASH_SECONDS = 0.16;

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
  // 139,662 neurons carry a measured soma annotation and 27,038 do not, and
  // 139,662 + 27,038 = 166,700 exactly. 139,668 is a different quantity: the
  // number of DISTINCT positions in the file, which counts the shared group
  // centroids as one coordinate each. The two numbers are never mixed in the
  // caption.
  measuredSomaNeurons: 139662,
  centroidFilled: 27038,
  distinctCoordinates: 139668,
  /** Measured with src/layout/centroidFill.ts on the shipped positions file. */
  measuredSharedCoordinates: 5,
  measuredPiledPoints: 27037,
  measuredMaxPile: 14418,
};

const CAPTION_LABEL = 'MaleCNS v1.0 · male central nervous system · measured connectome';
const CAPTION_VALUE =
  `${CONNECTOME_FACTS.neurons.toLocaleString('en-US')} neurons · ` +
  `${CONNECTOME_FACTS.directedEdges.toLocaleString('en-US')} directed edges · ` +
  `${CONNECTOME_FACTS.distinctCoordinates.toLocaleString('en-US')} distinct soma positions`;
const CAPTION_NOTE =
  'MaleCNS v1.0 is the whole male central nervous system, brain and ventral nerve cord, ' +
  'so the elongated form is the real measured geometry, not a rendering error. ' +
  'Coordinates are measured soma voxels: ' +
  `${CONNECTOME_FACTS.measuredSomaNeurons.toLocaleString('en-US')} of ` +
  `${CONNECTOME_FACTS.neurons.toLocaleString('en-US')} neurons carry a measured soma, and the ` +
  `other ${CONNECTOME_FACTS.centroidFilled.toLocaleString('en-US')} sit at their class and ` +
  `in-degree decile group centroid, ${CONNECTOME_FACTS.measuredPiledPoints.toLocaleString('en-US')} ` +
  `of them sharing just ${CONNECTOME_FACTS.measuredSharedCoordinates} coordinates (largest pile ` +
  `${CONNECTOME_FACTS.measuredMaxPile.toLocaleString('en-US')} points). Those are drawn smaller ` +
  'and cooler, as interpolated fill, so the measured somata stay the foreground. Edges: ' +
  'sampled illustration: nearest same-class soma pairs for the highest in-degree hubs, ' +
  'not measured adjacency.';
const CAPTION_LEGEND =
  'cyan measured soma · dim cooler cyan interpolated centroid fill · amber fresh spike · ' +
  'node hue cell class, core brightness in-degree percentile';

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
  void lastSpikeKey;
  const refScratch = useRef(new Float32Array(LIVE_SLOTS));
  const refValue = useRef(0);
  const trailPosAttr = useRef<THREE.BufferAttribute | null>(null);
  const trailColAttr = useRef<THREE.BufferAttribute | null>(null);
  const trailAgeAttr = useRef<THREE.BufferAttribute | null>(null);
  const trailPowAttr = useRef<THREE.BufferAttribute | null>(null);

  // -------------------------------------------------------------------
  // Geometry and materials. Built once, never reallocated.
  // -------------------------------------------------------------------
  // Which points sit on a group centroid rather than on a measured soma? The
  // flag is recovered from the positions themselves: a coordinate shared by
  // CENTROID_STACK_THRESHOLD or more points is a centroid placement. Measured
  // on the shipped file: 5 shared coordinates holding 27,037 points, the
  // largest pile 14,418. Those points are drawn smaller, tinted cooler, and
  // their alpha is weighted by 1 / stack size so a pile cannot accumulate into
  // a blown out disc.
  const centroidFill = useMemo(
    () => detectCentroidFill(layout.positions, layout.count),
    [layout],
  );

  // Cell class normalised to 0..1 so node hue carries a real per neuron value.
  const classNorm = useMemo(() => {
    const out = new Float32Array(layout.count);
    let maxClass = 1;
    for (let i = 0; i < layout.count; i += 1) {
      if (layout.cellClass[i] > maxClass) maxClass = layout.cellClass[i];
    }
    for (let i = 0; i < layout.count; i += 1) out[i] = layout.cellClass[i] / maxClass;
    return out;
  }, [layout]);

  const staticGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(layout.positions, 3));
    g.setAttribute('aBase', new THREE.BufferAttribute(layout.baseIntensity, 1));
    g.setAttribute('aFill', new THREE.BufferAttribute(centroidFill.fill, 1));
    g.setAttribute('aScale', new THREE.BufferAttribute(centroidFill.scale, 1));
    g.setAttribute('aClass', new THREE.BufferAttribute(classNorm, 1));
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
  }, [layout, centroidFill, classNorm]);

  const staticMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: STATIC_VERT,
        fragmentShader: STATIC_FRAG,
        uniforms: {
          uSize: { value: 2.6 },
          uPixelRatio: { value: 1 },
          uViewHeight: { value: 600 },
          uGlobal: { value: 0.85 },
          uAmbient: { value: 0.1 },
          uQuiet: { value: 0.06 },
          uRef: { value: 0.274 },
          uFillSize: { value: 0.55 },
          uClassMix: { value: 1 },
          uFillTint: { value: 0.8 },
          uToneGain: { value: 1.15 },
          uFogDim: { value: 0.3 },
          uFogNear: { value: 1.5 },
          uFogFar: { value: 4.2 },
          uAttenNear: { value: 2.4 },
          uColorDim: { value: new THREE.Color('#5bc8d6') },
          uColorPale: { value: new THREE.Color('#7fe0ea') },
          uColorHot: { value: new THREE.Color('#bfeff7') },
          uColorNeg: { value: new THREE.Color('#6fa6de') },
          uColorInterp: { value: new THREE.Color('#1d5a6b') },
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
          uGate: { value: 0 },
          uPos: { value: new THREE.Color('#7fe0ea') },
          uNeg: { value: new THREE.Color('#6fa6de') },
          uShockColor: { value: new THREE.Color('#f0a030') },
          uToneGain: { value: 1.45 },
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
  // Each edge becomes a four vertex ribbon (see EDGE_VERT). A GL line is one
  // device pixel wide and vanished under the cloud, so the width is set in CSS
  // pixels and applied in screen space: a hairline edge survives at any
  // distance and at any canvas size.
  const edgeGeo = useMemo(() => {
    const n = edgeSet.count;
    const src = edgeSet.positions;
    const start = new Float32Array(n * 4 * 3);
    const end = new Float32Array(n * 4 * 3);
    const side = new Float32Array(n * 4);
    const tFlag = new Float32Array(n * 4);
    const weight = new Float32Array(n * 4);
    const index = new Uint32Array(n * 6);
    const lengths = new Float32Array(n);
    let lengthSum = 0;
    for (let e = 0; e < n; e += 1) {
      const ax = src[e * 6];
      const ay = src[e * 6 + 1];
      const az = src[e * 6 + 2];
      const bx = src[e * 6 + 3];
      const by = src[e * 6 + 4];
      const bz = src[e * 6 + 5];
      lengths[e] = Math.hypot(bx - ax, by - ay, bz - az);
      lengthSum += lengths[e];
      const w = edgeSet.weights[e];
      for (let k = 0; k < 4; k += 1) {
        const o = (e * 4 + k) * 3;
        start[o] = ax;
        start[o + 1] = ay;
        start[o + 2] = az;
        end[o] = bx;
        end[o + 1] = by;
        end[o + 2] = bz;
        side[e * 4 + k] = k % 2 === 0 ? -1 : 1;
        tFlag[e * 4 + k] = k >= 2 ? 1 : 0;
        weight[e * 4 + k] = w;
      }
      const v = e * 4;
      const io = e * 6;
      index[io] = v;
      index[io + 1] = v + 1;
      index[io + 2] = v + 2;
      index[io + 3] = v + 1;
      index[io + 4] = v + 3;
      index[io + 5] = v + 2;
    }
    lengths.sort();
    const median = n > 0 ? lengths[n >> 1] : 0;
    Object.assign(metrics, {
      edgeLengthMean: n > 0 ? lengthSum / n : 0,
      edgeLengthMedian: median,
      edgeVertices: n * 4,
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('aStart', new THREE.BufferAttribute(start, 3));
    g.setAttribute('aEnd', new THREE.BufferAttribute(end, 3));
    g.setAttribute('aSide', new THREE.BufferAttribute(side, 1));
    g.setAttribute('aT', new THREE.BufferAttribute(tFlag, 1));
    g.setAttribute('aWeight', new THREE.BufferAttribute(weight, 1));
    g.setIndex(new THREE.BufferAttribute(index, 1));
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
          uOpacity: { value: 0.28 },
          uToneGain: { value: 1.15 },
          uWidthPx: { value: 1.7 },
          uViewHeight: { value: 600 },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
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
          uSize: { value: 4.2 },
          uPixelRatio: { value: 1 },
          uViewHeight: { value: 600 },
          uLife: { value: TRAIL_LIFE },
          uAlpha: { value: 0.85 },
          uToneGain: { value: 1.35 },
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
    // Honesty accounting, measured from the shipped positions file by
    // detectCentroidFill: how many points sit on a shared group centroid and how
    // large the biggest pile is. Read by the harness, printed by PREVIEW.md.
    Object.assign(metrics, {
      centroidPoints: centroidFill.piledPoints,
      measuredPoints: centroidFill.uniquePoints,
      sharedCoordinates: centroidFill.sharedCoordinates,
      distinctCoordinates: centroidFill.distinctCoordinates,
      maxCentroidStack: centroidFill.maxStack,
      centroidFillThreshold: centroidFill.threshold,
    });
    metrics.bytesPerFrame =
      driveMode === 'full' ? layout.count * 4 : LIVE_SLOTS * (3 * 4 + 4 + 4);

    const controls = new OrbitControls(camera, gl.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.6;
    controls.zoomSpeed = 0.9;
    camera.position.set(2.1, 1.25, 1.75);
    controls.target.set(0, 0, 0);
    controls.minDistance = 1.1;
    controls.maxDistance = 6;
    // Framing. The measured cloud is a rod: principal axis (0, 0.376, 0.926)
    // with extents 2.044 along it, 1.466 across x and 0.865 across the third
    // axis, and its bounding box is already centred on the origin. The previous
    // camera sat 46 degrees off that axis, so the 2.044 unit rod projected to
    // about 1.6 units inside a 2.2 unit tall frame and roughly 60 percent of the
    // panel was empty. This position is 13 degrees off broadside at distance
    // 1.94, so the rod fills the frame and lies diagonally across it.
    camera.position.set(-1.86, 0.46, 0.3);
    camera.lookAt(0, 0, 0);
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
    // Frame delta in seconds. Computed first because the amber spike flash
    // and the pulse trail are both driven by real time, not by frames: a
    // fixed per-frame decay makes the flash last about 170 ms at 60 fps
    // and well under a millisecond in an unthrottled headless run.
    // ------------------------------------------------------------------
    const frameNow = performance.now();
    const dtMs = lastFrameTime.current === 0 ? 16.7 : frameNow - lastFrameTime.current;
    lastFrameTime.current = frameNow;
    const dtSec = Math.min(0.05, Math.max(0.0005, dtMs / 1000));

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
      // Amber marks the spikes the CURRENT frame reports rather than a one-shot
      // change in the spike list. The earlier version flashed only when the
      // spike key changed, which is fine for a live stream but wrong for a held
      // frame: the reservoir telemetry keeps reporting the same three spike
      // indices while the network is idle, so the flash had already decayed
      // before any screenshot and the amber measured as present in none of them
      // (warm and hot pixel counts of 0 in every live shot). A slot the frame
      // still lists as spiking is, by definition, among the newest spikes, so it
      // is held at full while it stays in the list and decays only after it
      // leaves it.
      for (let k = 0; k < spiked.length; k += 1) {
        const s = spiked[k];
        if (s >= 0 && s < LIVE_SLOTS) shockState.current[s] = 1;
      }
      for (let s = 0; s < LIVE_SLOTS; s += 1) {
        const idx = s < liveIndices.length ? liveIndices[s] : -1;
        const live = s < nslots && (idx >= 0 || !hasIds);
        av[s] = live ? frame.state[s] : 0;
        sv[s] = live ? shockState.current[s] : 0;
        // Real-time decay, so the amber flash lasts the same wall-clock
        // time at 30 fps and at 300 fps.
        if (shockState.current[s] > 0) {
          shockState.current[s] *= Math.exp(-dtSec / SPIKE_FLASH_SECONDS);
        }
        if (shockState.current[s] < 0.004) shockState.current[s] = 0;
      }
      aa.needsUpdate = true;
      sa.needsUpdate = true;
    }

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
      // Upload only the live prefix. The dots are packed at the front of every
      // array, so re-uploading the whole 3,072 dot capacity every frame would be
      // roughly twice the traffic for a buffer that is usually half empty.
      if (trailPosAttr.current) markUploaded(trailPosAttr.current, liveDots, 3);
      if (trailColAttr.current) markUploaded(trailColAttr.current, liveDots, 3);
      if (trailAgeAttr.current) markUploaded(trailAgeAttr.current, liveDots, 1);
      if (trailPowAttr.current) markUploaded(trailPowAttr.current, liveDots, 1);
    }
    metrics.trailPoints = liveDots;
    metrics.trailUploadBytes = liveDots > 0 ? liveDots * 8 * 4 : 0;

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
    staticMat.uniforms.uAmbient.value = 0.85;
    // Dead frame economics. The ambient term is the only structure light in the
    // picture, so on an all-zero state it is multiplied by uQuiet rather than
    // brightened, and a pile of 14,418 points on one centroid can no longer sum
    // into a disc. The cut is sourced from the ambient term alone, never from a
    // global brightness reduction of the live frame.
    //
    // Measured why 0.055 was still too bright: the all-zero control came out at
    // 12.87 percent lit with a mean of RGB(10.8, 17.2, 22.6) against the live
    // frame's 28.2 percent and RGB(31.3, 57.3, 63.1), so the "dark" control was
    // reading as a dim but visible brain (the harness's lit test is a channel
    // over 28, and 139,663 measured points at this alpha accumulate past it).
    // 0.012 is a 4.6x cut on the ambient term only, which leaves the live frame
    // untouched and its own measured bright.
    const liveFrame = refValue.current > 0;
    staticMat.uniforms.uQuiet.value = liveFrame ? 1 : 0.012;
    staticMat.uniforms.uClassMix.value = 1;
    staticMat.uniforms.uToneGain.value = 1.15;
    staticMat.uniforms.uFogNear.value = 1.5;
    staticMat.uniforms.uFogFar.value = 4.2;
    staticMat.uniforms.uAttenNear.value = 2.4;
    // Low opacity cyan hairlines, still visible under the cloud because
    // EDGE_VERT draws them 1.7 px wide in screen space.
    edgeMat.uniforms.uOpacity.value = liveFrame ? 0.28 : 0;
    edgeMat.uniforms.uViewHeight.value = viewHeight;
    edgeMat.uniforms.uWidthPx.value = 1.7;
    trailMat.uniforms.uPixelRatio.value = pr;
    trailMat.uniforms.uViewHeight.value = viewHeight;
    trailMat.uniforms.uAlpha.value = liveFrame ? 0.85 : 0;
    staticMat.uniforms.uGlobal.value = 0.8 + 0.5 * frame.stateRms;
    liveMat.uniforms.uPixelRatio.value = pr;
    liveMat.uniforms.uViewHeight.value = viewHeight;
    liveMat.uniforms.uRef.value = refValue.current;
    // Dead frame: the 512 pins are switched off entirely, so an all-zero
    // state does not leave 512 dim dots glowing over the cloud.
    liveMat.uniforms.uGate.value = refValue.current > 0 ? 1 : 0;

    controlsRef.current?.update();

    recordFrame(dtMs);
    metrics.frames += 1;
    metrics.lastFrameMs = dtMs;
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
      <mesh geometry={edgeGeo} material={edgeMat} frustumCulled={false} />
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
        camera={{ fov: 38, near: 0.01, far: 40, position: [-1.86, 0.46, 0.3] }}
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
        The caption. Real figures only. Three different quantities are kept
        apart on purpose: 139,662 neurons carry a measured soma annotation,
        27,038 have none and sit at their group centroid, and 139,668 is the
        number of distinct positions in the file. The note under the figures is
        the honesty clause, and the legend says what the colours mean. Styled as
        a wide-tracked uppercase label plus a value line, which is how the
        reference broadcasts its numbers.
      */}
      <div
        data-testid="connectome-caption"
        data-neurons={CONNECTOME_FACTS.neurons}
        data-measured-soma-neurons={CONNECTOME_FACTS.measuredSomaNeurons}
        data-centroid-fill-neurons={CONNECTOME_FACTS.centroidFilled}
        data-distinct-positions={CONNECTOME_FACTS.distinctCoordinates}
        data-measured-shared-coordinates={CONNECTOME_FACTS.measuredSharedCoordinates}
        data-measured-piled-points={CONNECTOME_FACTS.measuredPiledPoints}
        data-measured-max-pile={CONNECTOME_FACTS.measuredMaxPile}
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
        <div
          data-testid="connectome-legend"
          style={{
            marginTop: '0.35rem',
            maxWidth: '62ch',
            fontSize: '0.56rem',
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: '#4f6a7e',
          }}
        >
          {CAPTION_LEGEND}
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
