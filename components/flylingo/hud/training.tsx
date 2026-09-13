"use client";

import { useEffect, useRef, useState } from "react";

import { HUD, Label, Meter, Spark } from "./primitives";

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

  const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
  // Peak of the recent window against the opening, not last-point against first-third: accuracy
  // dips when the course moves on to harder material, and a strict end-to-end comparison then
  // reported "no clear trend" across a climb from chance to the high nineties.
  const early = curve.length >= 8 ? curve.slice(0, 8) : [];
  const late = curve.length >= 8 ? curve.slice(-10) : [];
  const gain = early.length && late.length ? Math.max(...late) - mean(early) : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, height: "100%" }}>
      {/* ---- what it is doing, and the controls ---- */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, flex: "0 0 auto" }}>
        {/* The training state lives in the panel header ("from scratch" / "pretrained"), so it
            is stated once. Repeating it here as "already trained" put the same fact twice on one
            screen, beside a "training on" button that reads as its opposite. */}
        <Label tone="faint" size={10}>
          weights update every answer
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
            fontSize: 10.5,
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
            fontSize: 10.5,
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
          <Spark values={curve.length > 1 ? curve : [0, 0]} width={330} height={30} tone={HUD.cyan} />
        </div>
        <div
          style={{
            marginTop: 3,
            display: "flex",
            justifyContent: "space-between",
            fontSize: 10.5,
            color: HUD.faint,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          <span>chance 25%</span>
          {curve.length >= 6 ? (
            <span style={{ color: gain > 0.05 ? HUD.green : HUD.faint }}>
              {gain > 0.05
                ? `peak +${(gain * 100).toFixed(0)} points above its start`
                : "no gain yet"}
            </span>
          ) : (
            <span>collecting answers…</span>
          )}
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
        <div style={{ marginTop: 3, fontSize: 10.5, color: HUD.faint, lineHeight: 1.35 }}>
          reward on a correct pick, withheld on a miss — it gates the weight update.
        </div>
      </div>

      {/* ---- the course, advancing ---- */}
      <div
        style={{
          flex: "0 0 auto",
          borderTop: `1px solid ${HUD.line}`,
          paddingTop: 8,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: 8, minWidth: 0 }}>
          <Label tone="faint" size={10}>
            course
          </Label>
          <span
            style={{
              fontSize: 15,
              fontWeight: 600,
              color: HUD.text,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
              minWidth: 0,
            }}
          >
            {lessonTitle ?? "—"}
          </span>
          <span style={{ flex: "1 1 auto" }} />
          <span
            style={{
              fontSize: 10.5,
              color: HUD.faint,
              fontVariantNumeric: "tabular-nums",
              flex: "0 0 auto",
            }}
          >
            {lessonPos + 1} / {lessonTotal}
          </span>
        </div>
        <div style={{ marginTop: 4 }}>
          <Meter
            value={lessonTotal ? (lessonPos + 1) / lessonTotal : 0}
            tone={HUD.cyan}
            height={3}
          />
        </div>
      </div>

      {/* ---- the numbers behind it, as one compact line ----
          Six separate stat cells read as a research dashboard and crowded the panel past its
          box. What a viewer can use is how much it has learnt and how much rehearsal that took;
          entropy, parameters and full laps are kept in the title attribute and the README. */}
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 8,
          flex: "0 0 auto",
          fontSize: 11.5,
          color: HUD.faint,
          fontVariantNumeric: "tabular-nums",
        }}
        title={`entropy ${Math.abs(entropy) < 0.005 ? "0.00" : entropy.toFixed(2)} · ${params} parameters · ${lessonsCompleted} full laps through the course`}
      >
        <span>
          learnt{" "}
          <span style={{ color: HUD.cyan, fontWeight: 700, fontSize: 15 }}>
            {learnedCorrect}
          </span>
          /{learnedAnswered}
        </span>
        <span style={{ color: HUD.line }}>·</span>
        <span>
          {" "}
          <span style={{ color: HUD.text, fontWeight: 700, fontSize: 15 }}>
            {rehearsals.toLocaleString("en-US")}
          </span>{" "}
          rehearsals
        </span>
        <span style={{ color: HUD.line }}>·</span>
        <span>{replaySize} remembered</span>
      </div>

      <div style={{ flex: "1 1 auto", minHeight: 0 }} />

      {/* ---- the honest caveat, kept on screen ----
          Tightened from five lines of 10px type to three of 11.5px: at recording scale the old
          block was unreadable, and an honesty note nobody can read is not doing its job. The two
          claims that must survive are kept explicitly: what is being trained, and that the
          connectome buys nothing measurable. */}
      <div style={{ flex: "0 0 auto", fontSize: 11.5, color: HUD.faint, lineHeight: 1.42 }}>
        {training ? (
          <>
            Supervised against the lesson&apos;s own answer key, with{" "}
            {rehearsals.toLocaleString("en-US")} rehearsal steps from what it has already seen:
            memorisation of the phrase-to-answer mapping. The connectome confers{" "}
            <span style={{ color: HUD.dim }}>no measurable learning advantage</span> — intact,
            shuffled, random and no recurrence all reach the same accuracy.
          </>
        ) : (
          <>
            Training is OFF, so the weights are frozen and the curve above should flatten. This is
            the control: same lesson, same connectome, same readout, plasticity disabled.
          </>
        )}
      </div>
    </div>
  );
}
