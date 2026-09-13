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

/**
 * Cadence of the hands-off demo, in milliseconds.
 *
 * These are pacing, not padding: a viewer has to watch the fly settle onto its answer and the
 * reward trace fire before the card is replaced, or the sequence reads as numbers changing
 * rather than a creature deciding. THINK_MS covers the flight and the reach animation;
 * SETTLE_MS covers the dopamine decaying from 0.60 to 0.17 so the reward is actually seen.
 */
const THINK_MS = 3200;
const SETTLE_MS = 2000;

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
  /** Hands-off mode: the fly answers for itself. On by default, because the frame exists to be
      watched -- a demo where nothing moves until someone clicks is not a demo. */
  const [autoDemo, setAutoDemo] = useState(true);
  /** Counts completed turns of the hands-off loop; the loop re-arms on each increment. */
  const [cycle, setCycle] = useState(0);
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
        // Keep the whole course to hand: the lesson size has to be re-read every time the
        // course advances, not just at boot.
        const sizes = new Map<string, number>();
        for (const u of cur.units) for (const l of u.lessons) sizes.set(l.id, l.challenges.length);
        lessonSizes.current = sizes;
        setTotal(sizes.get(s.lesson_id) ?? 0);
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

  /**
   * The current challenge, readable from inside the hands-off loop.
   *
   * The loop must not depend on `challenge` directly: it sets the challenge itself at the end of
   * every cycle, so a `challenge` dependency would re-run the effect mid-cycle and submit the
   * answer twice. Reading it through a ref keeps the loop sequential while still seeing the
   * latest value.
   */
  const challengeRef = useRef<ApiChallenge | null>(null);
  challengeRef.current = challenge;
  /** Lesson id -> its challenge count, so the counter can follow the course as it advances. */
  const lessonSizes = useRef<Map<string, number>>(new Map());
  /** The lesson the counter currently describes, so a change resets it exactly once. */
  const countedLesson = useRef<string | null>(null);

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

  /**
   * The hands-off loop: think, choose, answer, settle, next question.
   *
   * Written as one strictly sequential cycle rather than as timers keyed on `status`. The
   * status-driven version overlapped: setting `pending` changed `flyTakeTurn`'s identity, which
   * re-ran the effect while the first request was still in flight, so two answers were submitted
   * for one challenge and the server answered the loser with HTTP 409 "challenge is not current".
   * One answer per cycle, and the next cycle starts only after the previous one has finished.
   *
   * The delays are the point rather than padding. A viewer has to see the fly settle on the
   * answer and the reward trace fire before the card is replaced, or the whole thing reads as
   * numbers changing rather than a creature making a decision. THINK_MS covers the reach
   * animation; SETTLE_MS covers the dopamine decaying.
   *
   * A wrong answer keeps the same challenge current, which is the lesson flow (miss, then
   * retry), so the next cycle simply re-attempts it.
   */
  useEffect(() => {
    if (!autoDemo || !session) return;
    let cancelled = false;
    const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

    const run = async () => {
      const ch = challengeRef.current;
      if (!ch) {
        await sleep(600);
        if (!cancelled) setCycle((c) => c + 1);
        return;
      }

      await sleep(THINK_MS);
      if (cancelled) return;

      let res: AnswerResult;
      try {
        res = await api.answer(session.session_id, ch.id, -1, true);
      } catch {
        if (cancelled) return;
        // Retry the same challenge on the next cycle rather than spinning here.
        await sleep(900);
        if (!cancelled) setCycle((c) => c + 1);
        return;
      }
      if (cancelled) return;

      setResult(res);
      setStatus(res.correct ? "correct" : "wrong");
      setHearts(res.hearts);
      setAnswered((n) => n + 1);
      setHistory((h) => [...h, { user: res.correct, fly: res.fly_correct }]);
      // Deliberately NOT setting `selected`: that paints the cyan "your pick" bar, and in the
      // hands-off demo nobody picked. The fly's choice is drawn in amber from `flyChoice`, so
      // the two marks keep meaning what they say.

      await sleep(SETTLE_MS);
      if (cancelled) return;

      if (res.next_challenge) setChallenge(res.next_challenge);
      setSelected(undefined);
      setStatus("none");
      setCycle((c) => c + 1);
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [autoDemo, session, cycle]);

  /**
   * Keep the counter describing the lesson that is actually on screen.
   *
   * The header used to read "19 / 7 answered": `total` was read once at boot from the first
   * lesson while `answered` kept climbing across lessons, so the denominator went stale and the
   * number was impossible. Following the course here fixes the fraction and gives the lesson
   * dot-track the right number of dots.
   */
  useEffect(() => {
    const id = frame?.lesson_id;
    if (!id || countedLesson.current === id) return;
    if (!lessonSizes.current.has(id)) return;
    countedLesson.current = id;
    setTotal(lessonSizes.current.get(id) ?? 0);
    setAnswered(0);
    setHistory([]);
  }, [frame?.lesson_id]);

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
  /** The option text the fly is on right now: shown in the lesson panel's header. */
  const pickedOption = flyChoice >= 0 ? (challenge?.options[flyChoice] ?? null) : null;
  /** True once the answer has been scored, which is when "choosing" becomes "answered". */
  const locked = Boolean(result) && status !== "none";
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
            flex: "0 0 46px",
            display: "flex",
            alignItems: "center",
            gap: 16,
            borderBottom: `1px solid ${HUD.line}`,
            paddingBottom: 10,
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
                      male cns v1.0 · brain + vnc · measured
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
                <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
                  {/* The fly's choice lives in the header. It used to be a bordered block inside
                      the panel, where it cost ~46px of the height the question needed -- and the
                      header had the width to spare. Same information, zero column height. */}
                  <Label tone="amber" size={11}>
                    {locked ? "fly answered" : "fly is choosing"}
                  </Label>
                  <span
                    style={{
                      fontSize: 17,
                      fontWeight: 700,
                      color: pickedOption ? HUD.text : HUD.faint,
                      maxWidth: 260,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {pickedOption ?? "—"}
                  </span>
                  <Label tone="faint">
                    {answered}
                    {total ? ` / ${total}` : ""} answered
                  </Label>
                </div>
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
            /* 150 -> 128: the lower third is a headline, a pipeline line and a control row, and
               it was carrying 150px of a fixed 1080 frame. The body needed those 22px far more
               than this tier needed the padding. */
            flex: "0 0 110px",
            display: "grid",
            gridTemplateColumns: "1fr auto",
            gap: 24,
            borderTop: `1px solid ${HUD.line}`,
            paddingTop: 14,
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
              <button
                type="button"
                onClick={() => setAutoDemo((v) => !v)}
                style={{
                  padding: "4px 10px",
                  background: autoDemo ? HUD.green : "transparent",
                  color: autoDemo ? HUD.bg : HUD.faint,
                  border: `1px solid ${autoDemo ? HUD.green : HUD.line}`,
                  borderRadius: 2,
                  fontSize: 10,
                  fontWeight: 700,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                }}
              >
                {autoDemo ? "fly is playing" : "paused"}
              </button>
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
                share a cell whose label matches the rest of the row (10.5px) and whose height
                matches theirs (line box + 2 + 35), so the labels stay on one line. The wrapper
                is a plain block on purpose: as a flex column it would blockify the label and
                drop its baseline 5px below the row's. */}
            <div>
              <Label tone="faint" size={10.5}>
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
                /* "untrained" was read as "not training" and looked like a contradiction beside
                   a panel reporting thousands of rehearsal steps. It means the readout STARTED
                   from nothing, so it now uses the panel's own wording. */
                value={trained ? "pretrained" : "from scratch"}
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
          <Label tone="faint" size={10.5}>
            attribution
          </Label>
          <span style={{ flex: "1 1 auto", minWidth: 0 }}>
            <span style={{ color: HUD.cyanPale }}>fly.glb</span> by victorberdugo1 (CC-BY-4.0).
            Handset <span style={{ color: HUD.cyanPale }}>smartphone_with_green_screen.glb</span>{" "}
            by <span style={{ color: HUD.cyanPale }}>peroroo</span> (CC-BY-SA-4.0, share-alike —
            see README). Wiring frozen. The honesty statement on what trains, and on the
            connectome conferring no measurable advantage, is in the panel above.
          </span>
        </div>
      </div>
    </HudStage>
  );
}
