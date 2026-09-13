"use client";

import { HUD, Label, Meter, Chip } from "./primitives";

import type { AnswerResult, ApiChallenge } from "@/lib/flylingo/types";

/**
 * The Duolingo lesson, restyled into the HUD's dark palette.
 *
 * The structure is deliberately the clone's, so it still reads as Duolingo: the progress
 * bar and hearts strip up top, a bold prompt, numbered option cards you can keyboard-select,
 * a single Check button, and green/red answer feedback. Only the surface changes, from the
 * clone's light card UI to hairline panels on near-black, so the lesson can sit inside a
 * broadcast frame without looking like an embedded website.
 */
export function HudLesson({
  challenge,
  selected,
  status,
  result,
  hearts,
  progress,
  pending,
  onSelect,
  onCheck,
  flyProbs,
}: {
  challenge: ApiChallenge | null;
  selected: number | undefined;
  status: "none" | "correct" | "wrong";
  result: AnswerResult | null;
  hearts: number;
  progress: number;
  pending: boolean;
  onSelect: (i: number) => void;
  onCheck: () => void;
  flyProbs: number[];
}) {
  const shortcutFor = (i: number) => `${i + 1}`;

  if (!challenge) {
    return (
      <div
        style={{
          display: "grid",
          placeItems: "center",
          height: "100%",
          color: HUD.faint,
          fontSize: 12,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
        }}
      >
        loading challenge
      </div>
    );
  }

  const accent =
    status === "correct" ? HUD.green : status === "wrong" ? HUD.rose : HUD.cyan;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      {/* progress and hearts */}
      <div style={{ display: "flex", alignItems: "center", gap: 14, flex: "0 0 auto" }}>
        <Label tone="faint">progress</Label>
        <div style={{ flex: "1 1 auto" }}>
          <Meter value={progress} tone={accent} height={5} />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ color: HUD.rose, fontSize: 14 }}>♥</span>
          <span style={{ fontWeight: 700, fontSize: 15, color: HUD.rose }}>{hearts}</span>
        </div>
      </div>

      {/* prompt */}
      <div style={{ marginTop: 18, flex: "0 0 auto" }}>
        <Label tone="faint">
          {challenge.type} · difficulty {challenge.difficulty}
        </Label>
        <h2
          style={{
            margin: "8px 0 0",
            fontSize: 30,
            lineHeight: 1.2,
            fontWeight: 700,
            letterSpacing: "-0.015em",
            color: HUD.text,
          }}
        >
          {challenge.prompt}
        </h2>
      </div>

      {/* options */}
      <div
        style={{
          marginTop: 20,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 12,
          flex: "0 0 auto",
        }}
      >
        {challenge.options.map((opt, i) => {
          const isSel = selected === i;
          const isAnswer = result ? i === result.answer_index : false;
          const showState = status !== "none" && (isSel || (status === "wrong" && isAnswer));
          const border = showState
            ? status === "correct"
              ? HUD.green
              : isAnswer
                ? HUD.green
                : HUD.rose
            : isSel
              ? HUD.cyan
              : HUD.line;
          return (
            <button
              key={`${i}-${opt}`}
              type="button"
              disabled={pending || status !== "none"}
              onClick={() => onSelect(i)}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 12,
                padding: "14px 16px",
                background: showState
                  ? status === "correct"
                    ? "rgba(74,222,128,0.10)"
                    : "rgba(240,97,107,0.10)"
                  : isSel
                    ? "rgba(91,200,214,0.10)"
                    : HUD.panelAlt,
                border: `1px solid ${border}`,
                borderRadius: 3,
                cursor: pending || status !== "none" ? "default" : "pointer",
                color: HUD.text,
                textAlign: "left",
                font: "inherit",
                transition: "border-color 120ms, background 120ms",
              }}
            >
              <span style={{ fontSize: 18, fontWeight: 600 }}>{opt}</span>
              <span
                style={{
                  flex: "0 0 auto",
                  width: 24,
                  height: 24,
                  display: "grid",
                  placeItems: "center",
                  border: `1px solid ${isSel ? HUD.cyan : HUD.line}`,
                  borderRadius: 3,
                  fontSize: 11,
                  fontWeight: 700,
                  color: isSel ? HUD.cyan : HUD.faint,
                }}
              >
                {shortcutFor(i)}
              </span>
            </button>
          );
        })}
      </div>

      {/* the fly's own readout over the same options */}
      {flyProbs.length > 0 && (
        <div style={{ marginTop: 16, flex: "0 0 auto" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: 6,
            }}
          >
            <Label tone="faint">fly readout over these options</Label>
            <Label tone="faint">softmax</Label>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {flyProbs.map((p, i) => (
              <div key={i} style={{ flex: 1, minWidth: 0 }}>
                <Meter value={p} tone={i === (result?.fly_choice ?? -1) ? HUD.amber : HUD.cyan} height={5} />
                <div
                  style={{
                    marginTop: 3,
                    fontSize: 10,
                    color: HUD.faint,
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {(p * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ flex: "1 1 auto", minHeight: 12 }} />

      {/* footer: feedback plus the single action button */}
      <div
        style={{
          flex: "0 0 auto",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 16,
          borderTop: `1px solid ${HUD.line}`,
          paddingTop: 14,
        }}
      >
        <div style={{ minWidth: 0 }}>
          {status === "correct" && <Chip tone="green">correct</Chip>}
          {status === "wrong" && <Chip tone="rose">not yet</Chip>}
          {status === "none" && <Label tone="faint">select an answer</Label>}
          {result && status !== "none" && (
            <div style={{ marginTop: 6, fontSize: 11, color: HUD.dim }}>
              fly picked{" "}
              <span style={{ color: HUD.cyan, fontWeight: 700 }}>
                {challenge.options[result.fly_choice] ?? result.fly_choice}
              </span>{" "}
              · {result.fly_correct ? "right" : "wrong"}
            </div>
          )}
        </div>
        <button
          type="button"
          disabled={pending || selected === undefined}
          onClick={onCheck}
          style={{
            flex: "0 0 auto",
            padding: "12px 26px",
            background:
              status === "correct" ? HUD.green : status === "wrong" ? HUD.rose : HUD.amber,
            color: HUD.bg,
            border: "none",
            borderRadius: 3,
            fontSize: 12,
            fontWeight: 800,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            cursor: pending || selected === undefined ? "not-allowed" : "pointer",
            opacity: pending || selected === undefined ? 0.45 : 1,
          }}
        >
          {status === "none" ? "check" : status === "correct" ? "next" : "retry"}
        </button>
      </div>
    </div>
  );
}
