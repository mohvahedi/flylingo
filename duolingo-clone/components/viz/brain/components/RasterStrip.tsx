import { useEffect, useMemo, useRef } from 'react';
import type { LiveFrame } from '../live/mapping';

/**
 * Spike raster strip.
 *
 * One row per state slot, one column per recent frame, so the last few seconds
 * of activity read left to right. Drawn on a 2D canvas because 512 x 240 cells
 * would otherwise be 122,880 DOM nodes.
 *
 * SPARSITY. Measured telemetry spikes about 5 of 512 slots per frame, and the
 * no_edges control gives an empty list. The strip therefore draws two layers:
 * a faint auto-ranged heat layer from |state| so quiet frames still show
 * structure, and a bright unambiguous mark per spike. With an empty spike list
 * nothing is drawn on the bright layer, which is correct: an empty list is a
 * real measurement, not a rendering failure.
 */

export interface RasterStripProps {
  /** Ring of recent frames, oldest first. */
  history: LiveFrame[];
  width?: number;
  height?: number;
}

export function RasterStrip({ history, width, height = 220 }: RasterStripProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Auto-range across the whole visible history, using the 95th percentile of
  // |state|, so the heat layer is readable at the measured 0.05 median and
  // collapses to nothing when every frame is zeros.
  const ref = useMemo(() => {
    const vals: number[] = [];
    for (const f of history) {
      for (let i = 0; i < f.state.length; i += 1) {
        const a = Math.abs(f.state[i]);
        if (a > 0) vals.push(a);
      }
    }
    if (vals.length === 0) return 0;
    vals.sort((a, b) => a - b);
    const p = vals[Math.min(vals.length - 1, Math.floor(0.95 * vals.length))];
    return p < 1e-4 ? 0 : p;
  }, [history]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cssW = width ?? canvas.clientWidth ?? 480;
    const cssH = height;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(cssH * dpr)) {
      canvas.width = Math.floor(cssW * dpr);
      canvas.height = Math.floor(cssH * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#04070d';
    ctx.fillRect(0, 0, cssW, cssH);

    const cols = Math.max(1, Math.round(cssW / 2));
    const rows = 512;
    const cellW = cssW / cols;
    // 512 rows do not fit in 220 px, so rows collapse onto the nearest pixel
    // row. The strip is a density view, not a one row per neuron view.
    const rowH = Math.max(1, cssH / rows);
    const start = Math.max(0, history.length - cols);
    const shown = history.length - start;
    const pad = cols - shown;

    for (let c = 0; c < shown; c += 1) {
      const f = history[start + c];
      const x = Math.round((pad + c) * cellW);

      // Layer 1: auto-ranged heat from |state|.
      if (ref > 0) {
        const st = f.state;
        for (let s = 0; s < rows && s < st.length; s += 1) {
          const v = Math.abs(st[s]) / ref;
          if (v < 0.25) continue;
          const a = Math.min(0.7, (v - 0.25) * 0.9);
          ctx.fillStyle = `rgba(72,140,196,${a.toFixed(3)})`;
          ctx.fillRect(x, Math.round(s * rowH), Math.max(cellW, 1), Math.max(rowH, 1));
        }
      }

      // Layer 2: spikes. About 5 per frame at the measured threshold.
      const spikes = f.spikes;
      if (spikes.length > 0) {
        ctx.fillStyle = '#eaf6ff';
        for (let k = 0; k < spikes.length; k += 1) {
          const s = spikes[k];
          if (s < 0 || s >= rows) continue;
          ctx.fillRect(x, Math.round(s * rowH), Math.max(cellW, 1.5), Math.max(rowH, 2));
        }
      }
    }

    // Time gridlines, one per second at the measured ~5 Hz frame rate.
    ctx.strokeStyle = 'rgba(90,130,170,0.28)';
    ctx.lineWidth = 1;
    for (let t = 1; t < 4; t += 1) {
      const xx = Math.round((t / 4) * cssW) + 0.5;
      ctx.beginPath();
      ctx.moveTo(xx, 0);
      ctx.lineTo(xx, cssH);
      ctx.stroke();
    }
    ctx.fillStyle = 'rgba(120,160,200,0.8)';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText('oldest', 4, cssH - 4);
    ctx.fillText('now', cssW - 26, cssH - 4);
  }, [history, width, height, ref]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        display: 'block',
        width: width ? `${width}px` : '100%',
        height: `${height}px`,
        background: '#04070d',
        border: '1px solid #16324f',
        borderRadius: 4,
      }}
    />
  );
}

export default RasterStrip;
