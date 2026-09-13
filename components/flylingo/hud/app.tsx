"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import dynamic from "next/dynamic";
import { toast } from "sonner";

import { api, useBrainStream, useServiceHealth } from "@/lib/flylingo/api";
import type { AnswerResult, ApiChallenge, Mode, SessionStart } from "@/lib/flylingo/types";

import { HudLesson } from "./lesson";
import { HudTraining } from "./training";
import { Chip, FitBox, HUD, HudStage, Label, LegendDot, Panel, Spark, Stat } from "./primitives";

/** Both visualizations are WebGL and client-only; neither has a server renderer. */
/**
 * The hero: the specimen flying to the answer it picked, on the phone the lesson runs on.
 *
 * This replaces the bare specimen view. The Duolingo lesson on the phone is drawn by the same
 * numbers the fly is targeted with, so the fly cannot be sent to a card other than the one it
 * chose. The dark lesson panel on the right stays as the instrumentation view, where the
 * fly's softmax and your own answer live.
 */
const PhoneStage = dynamic(() => import("@/components/viz/fly/fly/PhoneStage"), {
  ssr: false,
  loading: () => <ViewLoading label="specimen" />,
});
const BrainCloud = dynamic(
  () => import("@/components/viz/brain/components/BrainCloud"),
  { ssr: false, loading: () => <ViewLoading label="connectome" /> },
);

function ViewLoading({ label }: { label: string }) {
  return (
    <div
      style={{
        display: "grid",
        placeItems: "center",
        height: "100%",
        color: HUD.faint,
        fontSize: 10.5,
        letterSpacing: "0.16em",
        textTransform: "uppercase",
      }}
    >
      loading {label}
    </div>
  );
}

const MODES: Mode[] = ["intact", "shuffled", "random_graph", "no_edges"];

export function HudApp() {
  const { frame, status: streamStatus } = useBrainStream();
  const { health } = useServiceHealth();

  const [session, setSession] = useState<SessionStart | null>(null);
  const [challenge, setChallenge] = useState<ApiChallenge | null>(null);
  const [selected, setSelected] = useState<number | undefined>();
  const [status, setStatus] = useState<"none" | "correct" | "wrong">("none");
  const [result, setResult] = useState<AnswerResult | null>(null);
  const [pending, setPending] = useState(false);
  const [hearts, setHearts] = useState(5);
  const [answered, setAnswered] = useState(0);
  const [total, setTotal] = useState(0);
  /** Per-question outcomes, so the panel can show a track and a scoreboard. Kept for the
      fly and the user separately, because the whole point of the demo is that they differ. */
  const [history, setHistory] = useState<{ user: boolean; fly: boolean }[]>([]);
  const [bootError, setBootError] = useState<string | null>(null);
  const [modeBusy, setModeBusy] = useState(false);
  const [trainBusy, setTrainBusy] = useState(false);
  const [rmsHistory, setRmsHistory] = useState<number[]>([]);
  const [activeHistory, setActiveHistory] = useState<number[]>([]);

  // Rolling metric history for the sparklines.
  const lastTick = useRef(-1);
  useEffect(() => {
    if (!frame || frame.t === lastTick.current) return;
    lastTick.current = frame.t;
    setRmsHistory((h) => [...h.slice(-59), frame.state_rms]);
    setActiveHistory((h) => [...h.slice(-59), frame.active_fraction * 100]);
  }, [frame]);

  const start = useCallback(async () => {
    setBootError(null);
    try {
      const s = await api.startSession();
      setSession(s);
      setChallenge(s.challenge);
      setSelected(undefined);
      setStatus("none");
      setResult(null);
      setAnswered(0);
      setHistory([]);
      try {
        const cur = await api.curriculum();
        const lesson = cur.units.flatMap((u) => u.lessons).find((l) => l.id === s.lesson_id);
        setTotal(lesson?.challenges.length ?? 0);
      } catch {
        setTotal(0);
      }
    } catch (e) {
      setBootError(e instanceof Error ? e.message : "brain service unreachable");
    }
  }, []);

  useEffect(() => {
    void start();
  }, [start]);

  const onCheck = useCallback(async () => {
    if (!session || !challenge) return;
    if (status === "wrong") {
      setStatus("none");
      setSelected(undefined);
      return;
    }
    if (status === "correct") {
      if (result?.next_challenge) {
        setChallenge(result.next_challenge);
        setSelected(undefined);
        setStatus("none");
      }
      return;
    }
    if (selected === undefined) return;
    setPending(true);
    try {
      const res = await api.answer(session.session_id, challenge.id, selected);
      setResult(res);
      setStatus(res.correct ? "correct" : "wrong");
      setHearts(res.hearts);
      setAnswered((n) => n + 1);
      setHistory((h) => [...h, { user: res.correct, fly: res.fly_correct }]);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "brain service did not respond");
    } finally {
      setPending(false);
    }
  }, [session, challenge, selected, status, result]);

  // Duolingo-style number-key selection.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!challenge || status !== "none" || pending) return;
      const n = Number(e.key);
      if (n >= 1 && n <= challenge.options.length) setSelected(n - 1);
      if (e.key === "Enter") void onCheck();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [challenge, status, pending, onCheck]);

  const onModeChange = useCallback(async (mode: Mode) => {
    setModeBusy(true);
    try {
      await api.control(mode);
    } catch {
      toast.error("could not switch control condition");
    } finally {
      setModeBusy(false);
    }
  }, []);

  /**
   * Freeze or resume plasticity, keeping the current weights.
   *
   * Turning training off is the control, not a convenience: the same lesson, the same
   * connectome and the same readout with the weights frozen, so the learning curve can be
   * seen to flatten. That comparison is the difference between showing training and claiming it.
   */
  const onToggleTraining = useCallback(async () => {
    setTrainBusy(true);
    try {
      await api.train(false, { training: !(frame?.training ?? true) });
    } catch {
      toast.error("could not change the training state");
    } finally {
      setTrainBusy(false);
    }
  }, [frame?.training]);

  /** Wipe the readout back to a naive brain and start the course again so it can be watched. */
  const onRestartFresh = useCallback(async () => {
    setTrainBusy(true);
    try {
      await api.train(true, { training: true });
      const s = await api.startSession(undefined, { fresh: true });
      setSession(s);
      setChallenge(s.challenge);
      setSelected(undefined);
      setStatus("none");
      setResult(null);
      setAnswered(0);
      setHistory([]);
      toast.success("brain reset · watching it learn from scratch");
    } catch {
      toast.error("could not reset the brain");
    } finally {
      setTrainBusy(false);
    }
  }, []);

  const progress = total > 0 ? Math.min(1, answered / total) : (frame?.lesson_progress ?? 0);
  /** The fly's pick: the server's recorded choice once answered, otherwise live off the stream. */
  const flyProbs = result?.probs ?? frame?.probs ?? [];
  const flyChoice =
    result?.fly_choice ??
    (flyProbs.length > 0
      ? flyProbs.reduce((best, p, i) => (p > flyProbs[best] ? i : best), 0)
      : -1);
  const flyAcc = frame?.fly_accuracy ?? 0;
  const userAcc = frame?.accuracy ?? 0;
  const mode = frame?.mode ?? "intact";
  const trained = health?.checkpoint_status === "loaded";

  const headline = useMemo(() => {
    if (status === "correct") return "Correct";
    if (status === "wrong") return "Not yet";
    return "Translate to Spanish";
  }, [status]);

  return (
    <HudStage>
      <div
        style={{
          position: "absolute",
          inset: 0,
          padding: 26,
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}
      >
        {/* ---------------------------------------------------------------- header */}
        <header
          style={{
            flex: "0 0 54px",
            display: "flex",
            alignItems: "center",
            gap: 16,
            borderBottom: `1px solid ${HUD.line}`,
            paddingBottom: 12,
          }}
        >
          <span style={{ fontSize: 26, fontWeight: 800, letterSpacing: "0.02em" }}>
            FLYLINGO
          </span>
          <Label>fruit fly learns spanish</Label>
          <div style={{ flex: 1 }} />
          <Chip tone={streamStatus === "open" ? "cyan" : "dim"}>
            {streamStatus === "open" ? "live stream" : streamStatus}
          </Chip>
          <Chip tone={mode === "intact" ? "green" : "amber"} solid={mode !== "intact"}>
            {mode.replace("_", " ")}
          </Chip>
          <Label tone="faint">
            real malecns v1.0 · {health ? health.neurons.toLocaleString() : "166,700"} neurons
          </Label>
        </header>

        {/* ----------------------------------------------------------------- body */}
        <div
          style={{
            flex: "1 1 auto",
            minHeight: 0,
            display: "grid",
            gridTemplateColumns: "730px 1fr",
            gap: 20,
          }}
        >
          {/* left column: specimen over connectome */}
          {/* One gap value for the whole frame: the column used to sit on 18px against 20px
              everywhere else, which is the kind of 2px mismatch that makes a grid look
              accidental rather than composed. */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20, minHeight: 0 }}>
            {/* The scene runs full-bleed: the header keeps the frame's 14px inset, the render
                goes to the panel edges, so the wide shot is not inset twice. The label is the
                specimen alone, which leaves the header room for the whole attribution line --
                with the longer label that line ran 133px past the panel edge and was clipped. */}
            <Panel
              grow
              bodyPad={0}
              label="specimen"
              right={
                <Label tone="faint">
                  fly.glb · victorberdugo1 (CC-BY-4.0) · handset by peroroo (CC-BY-SA-4.0)
                </Label>
              }
            >
              <FitBox>
                {({ width, height }) => (
                  <PhoneStage
                    prompt={challenge?.prompt ?? "Loading the lesson…"}
                    options={(challenge?.options ?? []).map((text) => ({ text }))}
                    flyChoice={flyChoice}
                    userChoice={selected ?? -1}
                    status={status}
                    answerIndex={result?.answer_index ?? -1}
                    hearts={hearts}
                    progress={progress}
                    activity={frame?.state ?? []}
                    stateRms={frame?.state_rms ?? 0}
                    activeFraction={frame?.active_fraction ?? 0}
                    width={width}
                    height={height}
                  />
                )}
              </FitBox>
            </Panel>

            <Panel
              height={330}
              glow
              label="connectome · activity pulses"
              right={
                <Label tone="faint">
                  {frame ? `${frame.spikes.length} spikes this frame` : "awaiting data"}
                </Label>
              }
            >
              {/* The figures and the colour key are rendered by the HUD, outside the canvas,
                  rather than by the canvas itself. The cloud draws its own caption for the
                  standalone harness, but here the stage is scaled to fit 16:9 and that
                  caption became a grey smear over the point cloud. Keeping it in DOM type
                  next to the canvas means it stays legible at any scale and stops competing
                  with the render. */}
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  height: "100%",
                  gap: 10,
                }}
              >
                <div style={{ flex: "1 1 auto", minHeight: 0 }}>
                  <FitBox>
                    {({ width, height }) => (
                      <BrainCloud
                        state={frame?.state ?? []}
                        spikes={frame?.spikes ?? []}
                        sampledIds={frame?.sampled_ids ?? []}
                        activeFraction={frame?.active_fraction ?? 0}
                        stateRms={frame?.state_rms ?? 0}
                        width={width}
                        height={height}
                        caption="none"
                      />
                    )}
                  </FitBox>
                </div>
                <div style={{ flex: "0 0 auto" }}>
                  {/* Key and provenance share a row: as three stacked lines this caption left
                      a 23px band of dead space under the cloud and cost the specimen panel the
                      same height, which it needs more than this panel does. */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 16,
                    }}
                  >
                    <Label tone="faint" size={10.5}>
                      male cns v1.0 · measured connectome
                    </Label>
                    <div style={{ display: "flex", gap: 18 }}>
                      <LegendDot swatch={HUD.cyan}>measured soma</LegendDot>
                      <LegendDot swatch="rgba(91,200,214,0.42)">centroid fill</LegendDot>
                      <LegendDot swatch={HUD.amber}>fresh spike</LegendDot>
                    </div>
                  </div>
                  <div
                    style={{
                      marginTop: 5,
                      fontSize: 15,
                      color: HUD.text,
                      fontVariantNumeric: "tabular-nums",
                      letterSpacing: "0.005em",
                    }}
                  >
                    166,700 neurons · 25,582,938 directed edges · 139,668 distinct soma positions
                  </div>
                </div>
              </div>
            </Panel>
          </div>

          {/* right column: the lesson the fly is answering, and what it is learning */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20, minHeight: 0 }}>
            <Panel
              grow
              glow
              pad={20}
              label="lesson · fly answers in real time"
              right={
                <Label tone="faint">
                  {answered}
                  {total ? ` / ${total}` : ""} answered
                </Label>
              }
            >
            {bootError ? (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                  height: "100%",
                  justifyContent: "center",
                }}
              >
                <Label tone="amber">brain service not running</Label>
                <p style={{ color: HUD.dim, fontSize: 13, margin: 0, lineHeight: 1.6 }}>
                  The frame, the connectome loader and the course data are all in place.
                  Start the service and this panel fills with live activity.
                </p>
                <pre
                  style={{
                    margin: 0,
                    padding: 14,
                    background: "#000",
                    border: `1px solid ${HUD.line}`,
                    borderRadius: 3,
                    color: HUD.cyan,
                    fontSize: 12,
                    overflowX: "auto",
                  }}
                >
{`cd D:/Projects/flylingo/brain
.venv/Scripts/python.exe -m uvicorn brain.api:app --port 8770`}
                </pre>
                <button
                  type="button"
                  onClick={() => void start()}
                  style={{
                    alignSelf: "flex-start",
                    padding: "10px 20px",
                    background: HUD.amber,
                    color: HUD.bg,
                    border: "none",
                    borderRadius: 3,
                    fontSize: 11,
                    fontWeight: 800,
                    letterSpacing: "0.14em",
                    textTransform: "uppercase",
                    cursor: "pointer",
                  }}
                >
                  retry
                </button>
              </div>
            ) : (
              <HudLesson
                challenge={challenge}
                selected={selected}
                status={status}
                result={result}
                hearts={hearts}
                progress={progress}
                pending={pending}
                onSelect={setSelected}
                onCheck={() => void onCheck()}
                flyProbs={result?.probs ?? frame?.probs ?? []}
                answered={answered}
                total={total}
                history={history}
              />
            )}
            </Panel>

            {/* What the fly is learning, and the reward driving it. Every number is measured by
                the service and reported in the live frame; nothing here is interpolated. */}
            <Panel
              height={344}
              pad={16}
              label="learning · reward · rehearsal"
              right={
                <Label tone={frame?.fresh_brain ? "amber" : "cyan"} size={10}>
                  {frame?.fresh_brain ? "from scratch" : "pretrained"}
                </Label>
              }
            >
              <HudTraining
                windowAccuracy={frame?.window_accuracy ?? 0}
                windowSize={frame?.window_size ?? 0}
                dopamine={frame?.dopamine ?? 0}
                dopamineTotal={frame?.dopamine_total ?? 0}
                rehearsals={frame?.rehearsals ?? 0}
                replaySize={frame?.replay_size ?? 0}
                entropy={frame?.entropy ?? 0}
                freshBrain={frame?.fresh_brain ?? false}
                training={frame?.training ?? true}
                lessonPos={frame?.lesson_pos ?? 0}
                lessonTotal={frame?.lesson_total ?? 15}
                lessonTitle={frame?.lesson_title ?? null}
                lessonsCompleted={frame?.lessons_completed ?? 0}
                learnedCorrect={frame?.learned_correct ?? 0}
                learnedAnswered={frame?.learned_answered ?? 0}
                params={516}
                busy={trainBusy}
                onToggleTraining={() => void onToggleTraining()}
                onRestartFresh={() => void onRestartFresh()}
              />
            </Panel>
          </div>
        </div>

        {/* ---------------------------------------------------------- lower third */}
        <div
          style={{
            flex: "0 0 150px",
            display: "grid",
            gridTemplateColumns: "1fr auto",
            gap: 24,
            borderTop: `1px solid ${HUD.line}`,
            paddingTop: 16,
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
            {/* The headline is the frame's status line, not its title. At 44px it outranked the
                lesson prompt (38px) that the viewer actually has to read, so the eye landed on
                a restatement instead of the question. 36px keeps the sting of Correct / Not
                yet and lets the prompt lead; 1.15 also clears the descender, which overflowed
                the old 1.05 line box by 4px. */}
            <h2
              style={{
                margin: 0,
                fontSize: 36,
                fontWeight: 700,
                letterSpacing: "-0.02em",
                lineHeight: 1.15,
                color:
                  status === "correct" ? HUD.green : status === "wrong" ? HUD.rose : HUD.text,
              }}
            >
              {headline}
            </h2>
            <div
              style={{
                marginTop: 10,
                fontFamily: "var(--font-mono), ui-monospace, monospace",
                fontSize: 15,
                color: HUD.dim,
                letterSpacing: "0.01em",
              }}
            >
              <span style={{ color: HUD.text }}>prompt</span>
              <span style={{ color: HUD.faint }}> → </span>
              <span style={{ color: HUD.amber }}>connectome.step()</span>
              <span style={{ color: HUD.faint }}> → </span>
              <span style={{ color: HUD.amber }}>readout</span>
              <span style={{ color: HUD.faint }}> → </span>
              <span style={{ color: HUD.cyan }}>argmax</span>
            </div>
            <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 10 }}>
              {MODES.map((m) => (
                <button
                  key={m}
                  type="button"
                  disabled={modeBusy}
                  onClick={() => void onModeChange(m)}
                  style={{
                    padding: "4px 10px",
                    background: m === mode ? HUD.cyan : "transparent",
                    color: m === mode ? HUD.bg : HUD.faint,
                    border: `1px solid ${m === mode ? HUD.cyan : HUD.line}`,
                    borderRadius: 2,
                    fontSize: 10,
                    fontWeight: 700,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    cursor: modeBusy ? "not-allowed" : "pointer",
                  }}
                >
                  {m.replace("_", " ")}
                </button>
              ))}
              <Label tone="faint">control condition</Label>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 34 }}>
            {/* One instrument, not two. The trace used to sit in its own group labelled "state
                rms" while a stat cell in the row beside it carried the same label and the same
                value, so the tier read as two competing readouts. The trace and its number now
                share a cell whose label matches the rest of the row (9.5px) and whose height
                matches theirs (line box + 2 + 35), so the labels stay on one line. The wrapper
                is a plain block on purpose: as a flex column it would blockify the label and
                drop its baseline 5px below the row's. */}
            <div>
              <Label tone="faint" size={9.5}>
                state rms
              </Label>
              <div
                style={{
                  marginTop: 2,
                  height: 35,
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                }}
              >
                <Spark values={rmsHistory} width={170} height={35} tone={HUD.cyan} />
                <div
                  style={{
                    fontSize: 26,
                    fontWeight: 700,
                    letterSpacing: "-0.01em",
                    lineHeight: 1.15,
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {frame ? frame.state_rms.toFixed(3) : "0.000"}
                </div>
              </div>
            </div>
            <div style={{ display: "flex", gap: 30 }}>
              <Stat
                label="fly accuracy"
                value={`${(flyAcc * 100).toFixed(0)}%`}
                tone={flyAcc > 0.5 ? "green" : "amber"}
              />
              <Stat label="you" value={`${(userAcc * 100).toFixed(0)}%`} />
              <Stat label="step" value={frame ? `${frame.step}` : "0"} />
              <Stat
                label="spikes"
                value={frame ? `${frame.spikes.length}` : "0"}
                tone="cyan"
              />
              <Stat
                label="readout"
                value={trained ? "trained" : "untrained"}
                tone={trained ? "green" : "amber"}
                size={20}
              />
            </div>
          </div>
        </div>

        {/* honesty footer: the attribution travels with the demo, always visible */}
        <div
          style={{
            flex: "0 0 auto",
            display: "flex",
            alignItems: "baseline",
            gap: 12,
            color: HUD.dim,
            fontSize: 11.5,
            letterSpacing: "0.01em",
            lineHeight: 1.5,
          }}
        >
          <Label tone="faint" size={9.5}>
            attribution
          </Label>
          <span style={{ flex: "1 1 auto", minWidth: 0 }}>
            <span style={{ color: HUD.cyanPale }}>fly.glb</span> by victorberdugo1
            (CC-BY-4.0). Handset{" "}
            <span style={{ color: HUD.cyanPale }}>smartphone_with_green_screen.glb</span> by{" "}
            <span style={{ color: HUD.cyanPale }}>peroroo</span> (CC-BY-SA-4.0, share-alike see
            README). Wiring frozen; a 516-parameter readout learns the 97 phrases.{" "}
            <span style={{ color: HUD.faint }}>
              Measured: intact, shuffled, degree-matched random and the raw encoding with no
              reservoir all reach the same accuracy, so the connectome confers no measurable
              learning advantage on this task.
            </span>
          </span>
        </div>
      </div>
    </HudStage>
  );
}
