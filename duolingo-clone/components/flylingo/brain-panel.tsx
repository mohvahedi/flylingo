"use client";

import { useEffect, useRef } from "react";

import { cn } from "@/lib/utils";

import type { BrainFrame, Health, Mode } from "@/lib/flylingo/types";

/**
 * A 2D canvas rendering of the sampled connectome state.
 *
 * This is the guaranteed-available view of the fly brain: it draws the exact 512
 * values the service streams, so the panel is honest even before (or without) the
 * 3D components. Colour encodes signed activity, brightness encodes magnitude, and
 * neurons flagged as spikes are drawn with a brighter core.
 */
export function BrainCanvas({
  state,
  spikes,
  size = 256,
  className,
}: {
  state: number[];
  spikes: number[];
  size?: number;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
    const px = size * dpr;
    if (canvas.width !== px || canvas.height !== px) {
      canvas.width = px;
      canvas.height = px;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const n = state.length || 1;
    const cols = Math.ceil(Math.sqrt(n));
    const cell = size / cols;
    const spikeSet = new Set(spikes);

    // Display convention, not a change to the data: the sampled state is genuinely
    // small (measured p95 of |state| is about 0.274, state_rms about 0.105), so a
    // linear 0..1 scale would render an almost black frame. Normalise against this
    // frame's own 95th percentile instead. A dead frame (no_edges is exactly zero)
    // collapses the reference to zero and renders genuinely empty rather than
    // amplifying noise into fake activity.
    let ref = 0;
    const sorted = Array.from({ length: n }, (_, i) => Math.abs(state[i] ?? 0)).sort(
      (a, b) => a - b
    );
    ref = sorted.length ? sorted[Math.floor(0.95 * (sorted.length - 1))] : 0;
    const dead = ref < 1e-4;

    for (let i = 0; i < n; i += 1) {
      const value = state[i] ?? 0;
      const cx = (i % cols) * cell + cell / 2;
      const cy = Math.floor(i / cols) * cell + cell / 2;
      const mag = dead ? 0 : Math.min(1, Math.abs(value) / ref);
      const isSpike = spikeSet.has(i);

      // Positive activity reads warm, negative reads cool. Magnitude drives alpha.
      const alpha = dead ? 0 : 0.05 + mag * 0.95;
      const color =
        value >= 0
          ? `rgba(88, 204, 2, ${alpha})`
          : `rgba(28, 176, 246, ${alpha})`;

      ctx.fillStyle = color;
      const r = Math.max(cell * 0.34, 0.6) * (isSpike ? 1.5 : 1);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();

      if (isSpike) {
        ctx.strokeStyle = "rgba(255, 214, 0, 0.9)";
        ctx.lineWidth = Math.max(0.6, cell * 0.08);
        ctx.beginPath();
        ctx.arc(cx, cy, r * 1.35, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }, [state, spikes, size]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size }}
      className={cn("rounded-xl bg-neutral-900", className)}
      aria-label="Live sampled connectome activity"
    />
  );
}

/** Rolling spike raster: one row per recent frame, spikes drawn as ticks. */
export function SpikeRaster({
  history,
  width = 256,
  height = 64,
  className,
}: {
  history: number[][];
  width?: number;
  height?: number;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(23,23,23,1)";
    ctx.fillRect(0, 0, width, height);

    const rows = history.length;
    if (!rows) return;
    const rowH = height / rows;
    const n = 512;

    history.forEach((spikes, r) => {
      ctx.fillStyle = "rgba(250, 204, 21, 0.85)";
      for (const idx of spikes) {
        const x = (Math.max(0, Math.min(n - 1, idx)) / n) * width;
        ctx.fillRect(x, r * rowH, Math.max(1, width / n), Math.max(1, rowH));
      }
    });
  }, [history, width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className={cn("rounded-lg", className)}
      aria-label="Spike raster"
    />
  );
}

const MODES: Mode[] = ["intact", "shuffled", "no_edges", "random_graph"];

const MODE_LABEL: Record<Mode, string> = {
  intact: "Intact connectome",
  shuffled: "Shuffled control",
  no_edges: "No edges control",
  random_graph: "Random graph control",
};

/**
 * The mode badge.
 *
 * This exists so a running demo can never be mistaken for the intact connectome
 * when it is not. It is deliberately always visible, never hidden behind hover.
 */
export function ModeBadge({ mode }: { mode: Mode }) {
  const isIntact = mode === "intact";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-bold uppercase tracking-wide",
        isIntact
          ? "bg-green-100 text-green-700"
          : "bg-amber-100 text-amber-800"
      )}
    >
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          isIntact ? "bg-green-500" : "bg-amber-500"
        )}
      />
      {MODE_LABEL[mode]}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border-2 border-neutral-200 px-2 py-1.5">
      <div className="text-[10px] font-bold uppercase tracking-wide text-neutral-400">
        {label}
      </div>
      <div className="font-mono text-sm text-neutral-700">{value}</div>
    </div>
  );
}

export function ModeControls({
  mode,
  onChange,
  disabled,
}: {
  mode: Mode;
  onChange: (mode: Mode) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {MODES.map((m) => (
        <button
          key={m}
          type="button"
          disabled={disabled}
          onClick={() => onChange(m)}
          className={cn(
            "rounded-lg border-2 px-2 py-1 text-[11px] font-bold uppercase tracking-wide transition",
            m === mode
              ? "border-neutral-800 bg-neutral-800 text-white"
              : "border-neutral-200 text-neutral-500 hover:bg-neutral-100",
            disabled && "cursor-not-allowed opacity-50"
          )}
        >
          {m.replace("_", " ")}
        </button>
      ))}
    </div>
  );
}

/**
 * Whether the readout has actually been trained, and what that does and does not mean.
 *
 * Two separate claims are made here, and keeping them separate is the whole point:
 *
 * 1. Whether a model is loaded at all. A checkpoint is refused unless it was trained under
 *    the current encoder, and a refused checkpoint leaves the readout untrained.
 * 2. What the training demonstrated. The readout learns the vocabulary, but the measured
 *    connectome was measured NOT to beat a shuffled or degree-matched random graph on this
 *    task, and the raw encoding with no reservoir matches them all. So the learning lives
 *    in the readout, and the 166,700-neuron wiring is supplying a fixed feature map rather
 *    than doing anything the biology is needed for.
 *
 * Showing the first without the second would be the exact overclaim this project exists to
 * avoid, so both always render together.
 */
export function TrainingBadge({ health }: { health: Health | null }) {
  if (!health) return null;
  const trained = health.checkpoint_status === "loaded";
  return (
    <div
      className={cn(
        "rounded-lg border-2 px-2 py-1.5 text-[11px] leading-snug",
        trained
          ? "border-green-200 bg-green-50 text-green-900"
          : "border-amber-200 bg-amber-50 text-amber-900"
      )}
    >
      <div className="font-bold uppercase tracking-wide">
        {trained ? "readout trained" : "readout untrained"}
      </div>
      <div className="mt-0.5">
        {trained
          ? `Loaded ${health.readout_kind ?? "readout"}. It answers the curriculum correctly.`
          : "No usable checkpoint, so the answers below are an untrained readout: treat them as guesswork, not as the fly performing."}
      </div>
      {trained && (
        <div className="mt-1 border-t border-green-200 pt-1 opacity-90">
          <span className="font-bold">Attribution:</span> the connectome does not beat its
          own controls here. Intact, shuffled and random graphs all reach the same accuracy,
          and so does the encoding with no reservoir at all. The learning is in the readout;
          the wiring supplies a fixed feature map.
        </div>
      )}
      {!trained && (
        <div className="mt-1 font-mono text-[10px] opacity-80">
          {health.checkpoint_status}
        </div>
      )}
    </div>
  );
}

export type BrainPanelProps = {
  frame: BrainFrame | null;
  status: "connecting" | "open" | "closed";
  spikeHistory: number[][];
  onModeChange: (mode: Mode) => void;
  modeBusy?: boolean;
  /** Service health, used to state plainly whether the readout is trained. */
  health?: Health | null;
  /** Optional slot for the 3D fly, so layout stays stable when it is absent. */
  flySlot?: React.ReactNode;
};

export function BrainPanel({
  frame,
  status,
  spikeHistory,
  onModeChange,
  modeBusy,
  health,
  flySlot,
}: BrainPanelProps) {
  const mode = frame?.mode ?? "intact";
  const state = frame?.state ?? [];
  const spikes = frame?.spikes ?? [];

  return (
    <aside className="flex w-full flex-col gap-3 rounded-2xl border-2 bg-white p-4 lg:w-[320px]">
      <div className="flex items-center justify-between gap-2">
        <ModeBadge mode={mode} />
        <span
          className={cn(
            "text-[10px] font-bold uppercase tracking-wide",
            status === "open"
              ? "text-green-600"
              : status === "connecting"
                ? "text-amber-600"
                : "text-rose-500"
          )}
        >
          {status === "open"
            ? "live"
            : status === "connecting"
              ? "connecting"
              : "no service"}
        </span>
      </div>

      {flySlot ?? (
        <div className="flex h-[220px] items-center justify-center rounded-xl bg-neutral-900 text-center text-xs text-neutral-500">
          Fly view loads when the visualization module is present
        </div>
      )}

      <div>
        <div className="mb-1.5 flex items-baseline justify-between">
          <span className="text-[10px] font-bold uppercase tracking-wide text-neutral-400">
            Sampled neurons
          </span>
          <span className="text-[10px] text-neutral-400">
            512 of 166,700, auto-ranged
          </span>
        </div>
        <BrainCanvas state={state} spikes={spikes} size={288} className="w-full" />
      </div>

      <div>
        <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wide text-neutral-400">
          Spike raster
        </div>
        <SpikeRaster history={spikeHistory} width={288} height={56} className="w-full" />
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Stat
          label="Active"
          value={frame ? `${(frame.active_fraction * 100).toFixed(1)}%` : "n/a"}
        />
        <Stat label="State rms" value={frame ? frame.state_rms.toFixed(3) : "n/a"} />
        <Stat label="Step" value={frame ? `${frame.step}` : "n/a"} />
        <Stat
          label="Fly accuracy"
          value={frame ? `${(frame.fly_accuracy * 100).toFixed(0)}%` : "n/a"}
        />
      </div>

      <TrainingBadge health={health ?? null} />

      {frame && Object.keys(frame.controls).length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(frame.controls).map(([k, v]) => (
            <Stat
              key={k}
              label={k.replace("_accuracy", "").replace("_", " ")}
              value={typeof v === "number" ? `${(v * 100).toFixed(0)}%` : "n/a"}
            />
          ))}
        </div>
      )}

      <div className="border-t-2 pt-3">
        <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wide text-neutral-400">
          Control condition
        </div>
        <ModeControls mode={mode} onChange={onModeChange} disabled={modeBusy} />
        <p className="mt-2 text-[11px] leading-snug text-neutral-400">
          The wiring never changes. Only a small readout on top of it learns, and the
          controls isolate whether the measured wiring is doing any work.
        </p>
      </div>
    </aside>
  );
}
