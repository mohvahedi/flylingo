"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import dynamic from "next/dynamic";
import { toast } from "sonner";

import { api, useBrainStream, useServiceHealth } from "@/lib/flylingo/api";
import type { AnswerResult, ApiChallenge, Mode, SessionStart } from "@/lib/flylingo/types";

import { HudLesson } from "./lesson";
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
          <div style={{ display: "flex", flexDirection: "column", gap: 18, minHeight: 0 }}>
            <Panel
              grow
              label="specimen · lesson on screen"
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
              height={352}
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
                  <Label tone="faint" size={10.5}>
                    male cns v1.0 · measured connectome · brain and ventral nerve cord
                  </Label>
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
                  <div style={{ marginTop: 9, display: "flex", gap: 20, flexWrap: "wrap" }}>
                    <LegendDot swatch={HUD.cyan}>measured soma</LegendDot>
                    <LegendDot swatch="rgba(91,200,214,0.42)">centroid fill</LegendDot>
                    <LegendDot swatch={HUD.amber}>fresh spike</LegendDot>
                  </div>
                </div>
              </div>
            </Panel>
          </div>

          {/* right column: the lesson the fly is answering */}
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
            <h2
              style={{
                margin: 0,
                fontSize: 44,
                fontWeight: 700,
                letterSpacing: "-0.02em",
                lineHeight: 1.05,
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
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <Label tone="faint">state rms</Label>
              <Spark values={rmsHistory} width={150} height={40} tone={HUD.cyan} />
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
                label="state rms"
                value={frame ? frame.state_rms.toFixed(3) : "0.000"}
                size={26}
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
