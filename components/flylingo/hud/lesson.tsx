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
 *
 * Two things differ from the clone on purpose, because this is a demo of a brain answering
 * rather than a person answering:
 *
 *  - The fly's own softmax is attached to each OPTION CARD, not drawn as a separate row of
 *    anonymous bars. An earlier version showed four unlabelled bars under the options and a
 *    reviewer could not tell which bar belonged to which answer, which defeats the point of
 *    showing the readout at all. The probability now sits on the card it describes.
 *  - The fly's pick is shown LIVE, before the user answers, from the streamed frame. That is
 *    the honest version of the claim: the readout is computed continuously, so you can watch
 *    it commit before you commit. It is marked as provisional until the answer locks it in.
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
  answered,
  total,
  history,
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
  /** questions answered so far, and how many the lesson holds */
  answered: number;
  total: number;
  /** per-question outcome, oldest first, for the user and for the fly separately */
  history: { user: boolean; fly: boolean }[];
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

  // The fly's live pick, taken straight from the streamed softmax. Once the user answers,
  // the server's recorded choice is authoritative, so prefer it.
  const livePick =
    result?.fly_choice ??
    (flyProbs.length > 0
      ? flyProbs.reduce((best, p, i) => (p > flyProbs[best] ? i : best), 0)
      : -1);

  const pct = Math.round(progress * 100);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      {/* progress and hearts */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, flex: "0 0 auto" }}>
        <Label tone="faint">progress</Label>
        <div style={{ flex: "1 1 auto", display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ flex: "1 1 auto" }}>
            <Meter value={progress} tone={accent} height={6} />
          </div>
          {/* A 0% bar reads as a broken bar, so the number carries the meaning and the
              track is always visible behind it. */}
          <span
            style={{
              fontVariantNumeric: "tabular-nums",
              fontSize: 12,
              fontWeight: 700,
              color: progress > 0 ? HUD.text : HUD.faint,
              minWidth: 34,
              textAlign: "right",
            }}
          >
            {pct}%
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ color: HUD.rose, fontSize: 14 }}>♥</span>
          <span style={{ fontWeight: 700, fontSize: 15, color: HUD.rose }}>{hearts}</span>
        </div>
      </div>

      {/* The question block is centred in the space between the progress strip and the
          action bar. The panel is tall, so pinning the content to the top left a large
          dead gap under the options; centring uses the height instead of wasting it. */}
      <div
        style={{
          flex: "1 1 auto",
          minHeight: 0,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
        }}
      >
        {/* prompt */}
        <div style={{ flex: "0 0 auto" }}>
          <Label tone="faint">
            {challenge.type} · difficulty {challenge.difficulty}
          </Label>
          <h2
            style={{
              margin: "8px 0 0",
              fontSize: 30,
              lineHeight: 1.18,
              fontWeight: 700,
              letterSpacing: "-0.015em",
              color: HUD.text,
            }}
          >
            {challenge.prompt}
          </h2>
        </div>

        {/* options, each carrying the fly's own probability for that option */}
        <div
          style={{
            marginTop: 12,
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 10,
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
            const p = flyProbs[i] ?? 0;
            const isFlyPick = i === livePick;
            return (
              <button
                key={`${i}-${opt}`}
                type="button"
                disabled={pending || status !== "none"}
                onClick={() => onSelect(i)}
                style={{
                  position: "relative",
                  display: "flex",
                  flexDirection: "column",
                  gap: 6,
                  padding: "11px 16px 9px",
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
                <div style={{ display: "flex", alignItems: "center", gap: 10, width: "100%" }}>
                  <span style={{ fontSize: 20, fontWeight: 600, flex: "1 1 auto", minWidth: 0 }}>
                    {opt}
                  </span>
                  {/* the fly's own number for THIS option, so the mapping is unambiguous */}
                  <span
                    style={{
                      fontVariantNumeric: "tabular-nums",
                      fontSize: 15,
                      fontWeight: 700,
                      color: isFlyPick ? HUD.amberBright : HUD.faint,
                    }}
                  >
                    {(p * 100).toFixed(1)}%
                  </span>
                  <span
                    style={{
                      flex: "0 0 auto",
                      width: 28,
                      height: 28,
                      display: "grid",
                      placeItems: "center",
                      border: `1px solid ${isSel ? HUD.cyan : HUD.line}`,
                      borderRadius: 3,
                      fontSize: 12,
                      fontWeight: 700,
                      color: isSel ? HUD.cyan : HUD.faint,
                    }}
                  >
                    {shortcutFor(i)}
                  </span>
                </div>
                {/* the fly's distribution for this option, directly under its text */}
                <Meter
                  value={p}
                  tone={isFlyPick ? HUD.amber : HUD.cyan}
                  height={4}
                  track="rgba(120,160,200,0.16)"
                />
              </button>
            );
          })}
        </div>

      </div>

      {/* The lesson's own record: one dot per question plus who is ahead. This is the
          Duolingo progress idea and it gives the lower panel something real to say; the
          earlier layout left a dead zone here. A dot is empty until answered, then green or
          red, and the next question is ringed in amber. */}
      <div
        style={{
          flex: "0 0 auto",
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          gap: 24,
          marginTop: 10,
        }}
      >
        <div>
          <Label tone="faint" size={10}>
            question track
          </Label>
          <div style={{ display: "flex", gap: 7, marginTop: 8 }}>
            {Array.from({ length: Math.max(total, answered, 1) }).map((_, i) => {
              const h = history[i];
              const colour = h ? (h.user ? HUD.green : HUD.rose) : HUD.line;
              const isNow = i === answered && status === "none";
              return (
                <div
                  key={i}
                  title={
                    h
                      ? `question ${i + 1}: you ${h.user ? "correct" : "wrong"}, fly ${
                          h.fly ? "correct" : "wrong"
                        }`
                      : `question ${i + 1}`
                  }
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    background: h ? colour : "transparent",
                    border: `1px solid ${isNow ? HUD.amber : colour}`,
                    boxShadow: isNow ? `0 0 9px ${HUD.amber}66` : undefined,
                  }}
                />
              );
            })}
          </div>
        </div>
        <div style={{ display: "flex", gap: 22, alignItems: "baseline" }}>
          {/* Kept compact and inline: the lower third already reports "fly accuracy" and "you"
              as percentages, so these two are a per-lesson tally, not a second headline. */}
          <span style={{ fontSize: 11, color: HUD.faint }}>
            fly{" "}
            <span style={{ color: HUD.cyan, fontWeight: 700, fontSize: 15 }}>
              {history.filter((h) => h.fly).length}
            </span>
            <span style={{ color: HUD.faint }}>/{answered}</span>
          </span>
          <span style={{ fontSize: 11, color: HUD.faint }}>
            you{" "}
            <span style={{ color: HUD.green, fontWeight: 700, fontSize: 15 }}>
              {history.filter((h) => h.user).length}
            </span>
            <span style={{ color: HUD.faint }}>/{answered}</span>
          </span>
        </div>
      </div>

      {/* footer: feedback plus the single action button */}
      <div
        style={{
          flex: "0 0 auto",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 16,
          borderTop: `1px solid ${HUD.line}`,
          paddingTop: 8,
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
            padding: "10px 22px",
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
