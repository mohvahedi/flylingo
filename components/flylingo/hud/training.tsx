"use client";

import { useEffect, useRef, useState } from "react";

import { HUD, Label, Meter, Spark, Stat } from "./primitives";

/**
 * The training panel: what the fly is learning, and the reward that drives it.
 *
 * Every number here is measured by the service and reported in the live frame. Nothing in this
 * panel is decorative or interpolated:
 *
 *  - the LEARNING CURVE is windowed accuracy over the last N answers. A cumulative average was
 *    deliberately not used: it barely moves after a hundred answers, so it looks flat while the
 *    model is visibly improving.
 *  - the DOPAMINE trace is the same value that gated the last weight update, not a separate
 *    number that merely correlates with it. The service delivers the pulse immediately after
 *    the plasticity step, so the two cannot drift apart.
 *  - the REHEARSAL count is how many supervised steps were taken from the replay buffer. It is
 *    shown because it is the honest explanation of why the curve moves at all: one answer gives
 *    one update, and one update cannot fit 516 parameters. Measured, without rehearsal the fly
 *    plateaus near 45%; with it, it reaches 92-100% within about a hundred answers.
 *
 * The honesty caveat stays on screen rather than being tucked away, because this panel is the
 * one that looks most like a claim.
 */
export function HudTraining({
  windowAccuracy,
  windowSize,
  dopamine,
  dopamineTotal,
  rehearsals,
  replaySize,
  entropy,
  freshBrain,
  training,
  lessonPos,
  lessonTotal,
  lessonTitle,
  lessonsCompleted,
  learnedCorrect,
  learnedAnswered,
  params,
  busy,
  onToggleTraining,
  onRestartFresh,
}: {
  windowAccuracy: number;
  windowSize: number;
  dopamine: number;
  dopamineTotal: number;
  rehearsals: number;
  replaySize: number;
  entropy: number;
  freshBrain: boolean;
  training: boolean;
  lessonPos: number;
  lessonTotal: number;
  lessonTitle: string | null;
  lessonsCompleted: number;
  learnedCorrect: number;
  learnedAnswered: number;
  params: number;
  busy: boolean;
  onToggleTraining: () => void;
  onRestartFresh: () => void;
}) {
  // The curve's own history. The service reports the current window; the SHAPE over time is
  // what shows learning, so it is accumulated here from the frames.
  const [curve, setCurve] = useState<number[]>([]);
  const last = useRef(-1);
  useEffect(() => {
    if (windowSize === last.current) return;
    last.current = windowSize;
    setCurve((h) => [...h.slice(-119), windowAccuracy]);
  }, [windowSize, windowAccuracy]);

  const early = curve.length >= 6 ? curve.slice(0, Math.max(3, Math.floor(curve.length / 3))) : [];
  const late = curve.length >= 6 ? curve.slice(-Math.max(3, Math.floor(curve.length / 4))) : [];
  const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
  const gain = early.length && late.length ? mean(late) - mean(early) : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%" }}>
      {/* ---- what it is doing, and the controls ---- */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, flex: "0 0 auto" }}>
        <Label tone={freshBrain ? "amber" : "cyan"} size={10}>
          {freshBrain ? "learning from scratch" : "pretrained"}
        </Label>
        <div style={{ flex: "1 1 auto" }} />
        <button
          type="button"
          disabled={busy}
          onClick={onToggleTraining}
          style={{
            padding: "3px 9px",
            background: training ? HUD.cyan : "transparent",
            color: training ? HUD.bg : HUD.faint,
            border: `1px solid ${training ? HUD.cyan : HUD.line}`,
            borderRadius: 2,
            fontSize: 9.5,
            fontWeight: 700,
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            cursor: busy ? "not-allowed" : "pointer",
          }}
          title="When on, each answer also trains the readout. Turn it off to freeze the weights and compare."
        >
          {training ? "training on" : "training off"}
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={onRestartFresh}
          style={{
            padding: "3px 9px",
            background: "transparent",
            color: HUD.amber,
            border: `1px solid ${HUD.amber}`,
            borderRadius: 2,
            fontSize: 9.5,
            fontWeight: 700,
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            cursor: busy ? "not-allowed" : "pointer",
          }}
          title="Wipe the weights and start again from a naive brain."
        >
          reset brain
        </button>
      </div>

      {/* ---- the learning curve ---- */}
      <div style={{ flex: "0 0 auto" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
          <Label tone="faint" size={10}>
            fly accuracy · last {windowSize || 0} answers
          </Label>
          <span
            style={{
              fontVariantNumeric: "tabular-nums",
              fontSize: 26,
              fontWeight: 700,
              color: windowAccuracy > 0.6 ? HUD.green : windowAccuracy > 0.35 ? HUD.cyan : HUD.amber,
              lineHeight: 1,
            }}
          >
            {(windowAccuracy * 100).toFixed(0)}%
          </span>
        </div>
        <div style={{ marginTop: 6 }}>
          <Spark values={curve.length > 1 ? curve : [0, 0]} width={330} height={46} tone={HUD.cyan} />
        </div>
        <div
          style={{
            marginTop: 4,
            display: "flex",
            justifyContent: "space-between",
            fontSize: 10,
            color: HUD.faint,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          <span>chance 25%</span>
          {curve.length >= 6 ? (
            <span style={{ color: gain > 0.05 ? HUD.green : HUD.faint }}>
              {gain > 0.05 ? `improving  ${gain > 0 ? "+" : ""}${(gain * 100).toFixed(0)} points` : "no clear trend yet"}
            </span>
          ) : (
            <span>collecting answers…</span>
          )}
          <span>100%</span>
        </div>
      </div>

      {/* ---- dopamine ---- */}
      <div style={{ flex: "0 0 auto" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
          <Label tone="amber" size={10}>
            dopamine
          </Label>
          <span style={{ fontSize: 10, color: HUD.faint, fontVariantNumeric: "tabular-nums" }}>
            {dopamineTotal} pulses
          </span>
        </div>
        <div style={{ marginTop: 5 }}>
          <Meter value={dopamine} tone={HUD.amber} height={7} track="rgba(240,160,48,0.14)" />
        </div>
        <div style={{ marginTop: 4, fontSize: 10, color: HUD.faint, lineHeight: 1.4 }}>
          reward on a correct pick, withheld on a miss. This is the signal that gates the weight
          update, so the bar and the learning above are the same event.
        </div>
      </div>

      {/* ---- the course, advancing ---- */}
      <div
        style={{
          flex: "0 0 auto",
          borderTop: `1px solid ${HUD.line}`,
          paddingTop: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
          <Label tone="faint" size={10}>
            course
          </Label>
          <span style={{ fontSize: 10, color: HUD.faint, fontVariantNumeric: "tabular-nums" }}>
            lesson {lessonPos + 1} / {lessonTotal}
          </span>
        </div>
        <div
          style={{
            marginTop: 4,
            fontSize: 14,
            fontWeight: 600,
            color: HUD.text,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {lessonTitle ?? "—"}
        </div>
        <div style={{ marginTop: 5 }}>
          <Meter
            value={lessonTotal ? (lessonPos + 1) / lessonTotal : 0}
            tone={HUD.cyan}
            height={3}
          />
        </div>
      </div>

      {/* ---- the numbers behind it ---- */}
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", flex: "0 0 auto" }}>
        <Stat label="learnt" value={`${learnedCorrect}/${learnedAnswered}`} size={18} tone="cyan" />
        <Stat label="rehearsals" value={rehearsals.toLocaleString("en-US")} size={18} />
        <Stat label="memory" value={`${replaySize} seen`} size={18} />
        <Stat label="entropy" value={entropy.toFixed(2)} size={18} />
        <Stat label="params" value={String(params)} size={18} />
        <Stat
          label="lessons done"
          value={String(lessonsCompleted)}
          size={18}
          tone={lessonsCompleted > 0 ? "green" : "text"}
        />
      </div>

      <div style={{ flex: "1 1 auto", minHeight: 6 }} />

      {/* ---- the honest caveat, kept on screen ---- */}
      <div style={{ flex: "0 0 auto", fontSize: 10, color: HUD.faint, lineHeight: 1.45 }}>
        {training ? (
          <>
            Each answer trains the 516-parameter readout supervised against the lesson&apos;s own
            answer key, with {rehearsals.toLocaleString("en-US")} rehearsal steps taken from the
            examples it has already seen. That is memorisation of the phrase-to-answer mapping,
            which is what this readout was measured to be good at. The connectome contributes no
            measurable learning advantage on this task: intact, shuffled, degree-matched random
            and the raw encoding with no reservoir all reach the same accuracy.
          </>
        ) : (
          <>
            Training is OFF, so the weights are frozen and the curve above should flatten. This is
            the control: the same lesson, the same connectome, the same readout, with the
            plasticity disabled.
          </>
        )}
      </div>
    </div>
  );
}
