"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Design tokens for the broadcast HUD.
 *
 * The reference look is a tech-noir esports overlay: near-black with a blue cast, hairline
 * borders, wide-tracked uppercase grey labels, exactly one cool hue family (blue to cyan)
 * and one warm accent (amber to yellow) used only for interactive and attention elements.
 * Holding to that restraint is what makes it read as expensive, so nothing here introduces
 * a third hue.
 */
export const HUD = {
  bg: "#080C11",
  panel: "#0C1117",
  panelAlt: "#0A0E14",
  line: "rgba(120,160,200,0.20)",
  lineStrong: "rgba(120,160,200,0.34)",
  text: "#FFFFFF",
  dim: "#9AA8B6",
  faint: "#74828F",
  cyan: "#5BC8D6",
  cyanBright: "#7FE0EA",
  cyanPale: "#BFEFF7",
  blue: "#326CE5",
  amber: "#F0A030",
  amberBright: "#F5C542",
  green: "#4ADE80",
  rose: "#F0616B",
} as const;

/** The stage is composed at 16:9 and scaled to fit, exactly like a capture overlay. */
export const STAGE_W = 1920;
export const STAGE_H = 1080;

export function useStageScale() {
  const [scale, setScale] = useState(1);
  useEffect(() => {
    const fit = () =>
      setScale(Math.min(window.innerWidth / STAGE_W, window.innerHeight / STAGE_H));
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);
  return scale;
}

/**
 * A fixed 16:9 stage scaled to fit the window.
 *
 * Composing at fixed dimensions and scaling guarantees the reference's grid discipline,
 * baseline alignment and negative space at any window size. A responsive layout would
 * reflow and lose exactly the properties that make it look considered.
 */
export function HudStage({ children }: { children: React.ReactNode }) {
  const scale = useStageScale();
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: HUD.bg,
        display: "grid",
        placeItems: "center",
        overflow: "hidden",
      }}
    >
      {/* A very soft vignette, as in the reference. */}
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(120% 90% at 50% 40%, rgba(30,60,90,0.18) 0%, rgba(8,12,17,0) 60%)",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          width: STAGE_W,
          height: STAGE_H,
          transform: `scale(${scale})`,
          transformOrigin: "center center",
          position: "relative",
          color: HUD.text,
        }}
      >
        {children}
      </div>
    </div>
  );
}

/** Wide-tracked uppercase caption. The single most recognisable signature of the style. */
export function Label({
  children,
  tone = "dim",
  size = 11.5,
}: {
  children: React.ReactNode;
  tone?: "dim" | "faint" | "cyan" | "amber" | "text";
  size?: number;
}) {
  const color =
    tone === "cyan" ? HUD.cyan : tone === "amber" ? HUD.amber
      : tone === "faint" ? HUD.faint : tone === "text" ? HUD.text : HUD.dim;
  return (
    <span
      style={{
        fontSize: size,
        letterSpacing: "0.16em",
        textTransform: "uppercase",
        color,
        fontWeight: 600,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </span>
  );
}

/**
 * Panel with a hairline border and an optional wide-tracked label in its corner.
 *
 * `pad` insets the header and the body together; `bodyPad` overrides the body alone, so a
 * rendered scene can run full-bleed under a header that keeps the frame's inset.
 */
export function Panel({
  label,
  right,
  children,
  grow,
  pad = 14,
  bodyPad,
  glow = false,
  height,
}: {
  label?: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
  grow?: boolean;
  pad?: number;
  bodyPad?: number;
  glow?: boolean;
  height?: number;
}) {
  return (
    <section
      style={{
        position: "relative",
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        flex: grow ? "1 1 auto" : "0 0 auto",
        height,
        background: HUD.panel,
        border: `1px solid ${glow ? HUD.lineStrong : HUD.line}`,
        borderRadius: 3,
        boxShadow: glow ? "0 0 14px rgba(70,150,215,0.18)" : undefined,
      }}
    >
      {(label || right) && (
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            padding: `8px ${pad}px`,
            borderBottom: `1px solid ${HUD.line}`,
            flex: "0 0 auto",
          }}
        >
          <Label>{label}</Label>
          {right}
        </header>
      )}
      {/* `overflow: hidden` is a structural safety net, not a style choice. These panels are
          fixed-height boxes in a fixed 1920x1080 frame, and when content grew past its box it
          painted straight over the panel below -- which is what made text from the training
          panel collide with the lower-third metrics row. Content is now sized to fit as well,
          but a panel should never be able to escape its own border. */}
      <div
        style={{
          flex: "1 1 auto",
          minHeight: 0,
          overflow: "hidden",
          padding: bodyPad ?? pad,
        }}
      >
        {children}
      </div>
    </section>
  );
}

/** Label over a large bold value, the HUD's standard readout cell. */
export function Stat({
  label,
  value,
  tone = "text",
  size = 30,
}: {
  label: string;
  value: string;
  tone?: "text" | "cyan" | "amber" | "green" | "rose";
  size?: number;
}) {
  const color =
    tone === "cyan" ? HUD.cyanBright : tone === "amber" ? HUD.amberBright
      : tone === "green" ? HUD.green : tone === "rose" ? HUD.rose : HUD.text;
  return (
    <div style={{ minWidth: 0 }}>
      {/* 10.5 rather than 9.5: at recording scale every Stat label in the frame was the
          smallest type on screen, and the labels are what make the numbers mean anything. */}
      <Label size={10.5}>{label}</Label>
      <div
        style={{
          fontSize: size,
          fontWeight: 700,
          letterSpacing: "-0.01em",
          color,
          lineHeight: 1.15,
          marginTop: 2,
        }}
      >
        {value}
      </div>
    </div>
  );
}

/** A thin horizontal meter, used for progress and for activity levels. */
export function Meter({
  value,
  tone = HUD.cyan,
  height = 4,
  track = "rgba(120,160,200,0.14)",
}: {
  value: number;
  tone?: string;
  height?: number;
  track?: string;
}) {
  return (
    <div style={{ width: "100%", height, background: track, borderRadius: 2 }}>
      <div
        style={{
          width: `${Math.max(0, Math.min(1, value)) * 100}%`,
          height: "100%",
          background: tone,
          borderRadius: 2,
          transition: "width 200ms linear",
        }}
      />
    </div>
  );
}

/** Small pill used for mode and status badges. */
export function Chip({
  children,
  tone = "cyan",
  solid = false,
}: {
  children: React.ReactNode;
  tone?: "cyan" | "amber" | "green" | "rose" | "dim";
  solid?: boolean;
}) {
  const color =
    tone === "amber" ? HUD.amber : tone === "green" ? HUD.green
      : tone === "rose" ? HUD.rose : tone === "dim" ? HUD.dim : HUD.cyan;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "3px 8px",
        borderRadius: 2,
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: "0.12em",
        textTransform: "uppercase",
        color: solid ? HUD.bg : color,
        background: solid ? color : "transparent",
        border: `1px solid ${color}`,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </span>
  );
}

/**
 * Measures its own box and passes the exact pixel size to a WebGL child.
 *
 * Both visualizations take explicit width/height props and the HUD previously hardcoded
 * them. That silently drifted from the panel geometry: the specimen canvas was rendered at
 * 698x430 inside a panel whose body was only 315px tall, so `overflow: hidden` clipped the
 * bottom ~115px of every frame. That cut off the fly's legs and its contact shadow, which is
 * exactly the detail the grounding work depends on, and it is a large part of why the fly
 * read as floating in the app while looking better in the standalone harness.
 *
 * Measuring instead of guessing means the canvas always matches its box, and it cannot drift
 * again when the surrounding layout changes.
 *
 * The box is also a positioned ancestor. A WebGL child that fills itself with `position:
 * absolute; inset: 0` otherwise resolves against the panel's padding box, because nothing
 * between the two is positioned. The specimen scene did exactly that: it drew over its own
 * panel header, so the label and the attribution line vanished under the render, and, with the
 * drawing buffer no longer matching its CSS box, the browser stretched the frame (a 728x358 CSS
 * box for a 700x295 buffer, ~21% of vertical stretch) and squashed the handset.
 */
export function FitBox({
  children,
  minHeight = 80,
}: {
  children: (size: { width: number; height: number }) => React.ReactNode;
  minHeight?: number;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      setSize((prev) => {
        const w = Math.max(1, Math.round(r.width));
        const h = Math.max(minHeight, Math.round(r.height));
        return prev.width === w && prev.height === h ? prev : { width: w, height: h };
      });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [minHeight]);

  return (
    <div
      ref={ref}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        minHeight: 0,
        overflow: "hidden",
      }}
    >
      {size.width > 0 ? children(size) : null}
    </div>
  );
}

/**
 * A colour key entry for a canvas-based view.
 *
 * The connectome cloud draws its own legend inside the canvas, but the HUD scales the whole
 * stage to fit 16:9, which turned that legend into an unreadable grey smear over the point
 * cloud. The HUD therefore renders the key itself, in DOM type at a size that survives the
 * scale, next to the figures. `swatch` is the exact colour the shader uses.
 */
export function LegendDot({
  swatch,
  children,
}: {
  swatch: string;
  children: React.ReactNode;
}) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: 2,
          background: swatch,
          flex: "0 0 auto",
        }}
      />
      <Label size={10} tone="dim">
        {children}
      </Label>
    </span>
  );
}

/** Rolling sparkline for a metric, matching the HUD's restrained chart language. */
export function Spark({
  values,
  width = 200,
  height = 34,
  tone = HUD.cyan,
}: {
  values: number[];
  width?: number;
  height?: number;
  tone?: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = width * dpr;
    c.height = height * dpr;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (values.length < 2) return;
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const span = hi - lo || 1;
    ctx.strokeStyle = tone;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - 3 - ((v - lo) / span) * (height - 8);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [values, width, height, tone]);
  return <canvas ref={ref} style={{ width, height, display: "block" }} />;
}
