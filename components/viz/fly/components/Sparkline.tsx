/**
 * Small corner readout: a rolling sparkline of stateRms. Canvas 2D on purpose, it costs
 * nothing and never competes with the WebGL context for frames.
 *
 * This is the only ambient UI in the harness besides the mode badge.
 */
import { useEffect, useRef } from 'react';

export type SparklineProps = {
  /** latest value, expected 0..1 */
  value: number;
  /** ring buffer the caller pushes into */
  history: number[];
  width?: number;
  height?: number;
  label?: string;
  color?: string;
};

export function Sparkline({
  value,
  history,
  width = 168,
  height = 40,
  label = 'stateRms',
  color = '#22d3ee',
}: SparklineProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const valueRef = useRef(value);
  valueRef.current = value;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = Math.round(width * dpr);
    cv.height = Math.round(height * dpr);
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const v = Math.max(0, Math.min(1, valueRef.current || 0));
      ctx.clearRect(0, 0, width, height);

      // frame
      ctx.strokeStyle = 'rgba(148,163,184,0.22)';
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

      // axis guides at 0.5
      ctx.strokeStyle = 'rgba(148,163,184,0.12)';
      ctx.beginPath();
      ctx.moveTo(0, height - height * 0.5);
      ctx.lineTo(width, height - height * 0.5);
      ctx.stroke();

      const n = history.length;
      if (n > 1) {
        ctx.beginPath();
        for (let i = 0; i < n; i += 1) {
          const x = (i / (n - 1)) * (width - 2) + 1;
          const y = height - 1 - Math.max(0, Math.min(1, history[i])) * (height - 2);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.25;
        ctx.stroke();
      }

      // head marker
      const y = height - 1 - v * (height - 2);
      ctx.fillStyle = color;
      ctx.fillRect(width - 3, y - 1.5, 2, 3);

      ctx.fillStyle = 'rgba(203,213,225,0.7)';
      ctx.font = '9px ui-monospace, Menlo, Consolas, monospace';
      ctx.fillText(label, 4, 11);
      ctx.fillText(v.toFixed(3), width - 40, 11);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [history, width, height, label, color]);

  return <canvas ref={ref} style={{ width, height, display: 'block' }} />;
}
