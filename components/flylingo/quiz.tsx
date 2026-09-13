"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import Confetti from "react-confetti";
import { useWindowSize } from "react-use";
import { toast } from "sonner";

import { Challenge } from "@/components/flylingo/challenge";
import {
  BrainPanel,
  ModeControls,
  ModeBadge,
} from "@/components/flylingo/brain-panel";
import { LiveBrain } from "@/components/flylingo/live-brain";
import { Header } from "@/app/lesson/header";
import { Footer } from "@/app/lesson/footer";
import { QuestionBubble } from "@/app/lesson/question-bubble";
import { ResultCard } from "@/app/lesson/result-card";
import { MAX_HEARTS } from "@/constants";
import { api, useBrainStream, useServiceHealth } from "@/lib/flylingo/api";
import { cardTypeFor } from "@/lib/flylingo/types";
import type {
  AnswerResult,
  ApiChallenge,
  Mode,
  SessionStart,
} from "@/lib/flylingo/types";

const HISTORY = 40;

export function FlyLingoQuiz() {
  const { width, height } = useWindowSize();
  const { frame, status: streamStatus } = useBrainStream();
  const { health, reachable } = useServiceHealth();

  const [session, setSession] = useState<SessionStart | null>(null);
  const [challenge, setChallenge] = useState<ApiChallenge | null>(null);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<number | undefined>();
  const [status, setStatus] = useState<"none" | "wrong" | "correct">("none");
  const [result, setResult] = useState<AnswerResult | null>(null);
  const [pending, setPending] = useState(false);
  const [hearts, setHearts] = useState(MAX_HEARTS);
  const [xp, setXp] = useState(0);
  const [answered, setAnswered] = useState(0);
  const [finished, setFinished] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);
  const [spikeHistory, setSpikeHistory] = useState<number[][]>([]);
  const [modeBusy, setModeBusy] = useState(false);

  // Rolling spike raster, one row per received frame.
  const lastFrameRef = useRef<number>(-1);
  useEffect(() => {
    if (!frame || frame.t === lastFrameRef.current) return;
    lastFrameRef.current = frame.t;
    setSpikeHistory((h) => [...h.slice(-(HISTORY - 1)), frame.spikes]);
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
      setFinished(false);
      // Lesson length, only for the progress bar. Absence must not block the lesson.
      try {
        const cur = await api.curriculum();
        const lesson = cur.units
          .flatMap((u) => u.lessons)
          .find((l) => l.id === s.lesson_id);
        setTotal(lesson?.challenges.length ?? 0);
      } catch {
        setTotal(0);
      }
    } catch (e) {
      setBootError(
        e instanceof Error ? e.message : "Could not reach the brain service."
      );
    }
  }, []);

  useEffect(() => {
    void start();
  }, [start]);

  const onContinue = useCallback(async () => {
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
      } else {
        setFinished(true);
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
      setXp(res.xp);
      setAnswered((n) => n + 1);
    } catch (e) {
      toast.error(
        e instanceof Error ? e.message : "The brain service did not respond."
      );
    } finally {
      setPending(false);
    }
  }, [session, challenge, selected, status, result]);

  const onModeChange = useCallback(async (mode: Mode) => {
    setModeBusy(true);
    try {
      const res = await api.control(mode);
      toast.success(`Control condition: ${res.mode}`);
    } catch {
      toast.error("Could not switch control condition.");
    } finally {
      setModeBusy(false);
    }
  }, []);

  const percentage =
    total > 0 ? Math.min(100, (answered / total) * 100) : frame?.lesson_progress
      ? frame.lesson_progress * 100
      : 0;

  const panel = (
    <BrainPanel
      frame={frame}
      status={streamStatus}
      spikeHistory={spikeHistory}
      onModeChange={onModeChange}
      modeBusy={modeBusy}
      health={health}
      flySlot={
        <LiveBrain
          frame={frame}
          correct={status === "none" ? null : status === "correct"}
        />
      }
    />
  );

  if (bootError) {
    return (
      <div className="mx-auto flex h-full max-w-xl flex-col items-center justify-center gap-y-4 px-6 text-center">
        <h1 className="text-xl font-bold text-neutral-700 lg:text-3xl">
          The brain service is not running.
        </h1>
        <p className="text-sm text-neutral-500">
          The lesson shell, the connectome loader and the course data are all here. Start
          the service and this page will fill in with live activity.
        </p>
        <pre className="w-full overflow-x-auto rounded-xl bg-neutral-900 p-4 text-left text-xs text-neutral-300">
{`cd D:/Projects/flylingo/brain
.venv/Scripts/python.exe -m uvicorn brain.api:app --port 8770`}
        </pre>
        <p className="font-mono text-[11px] text-neutral-400">{bootError}</p>
        <button
          onClick={() => void start()}
          className="rounded-xl border-2 border-b-4 px-4 py-2 text-sm font-bold uppercase text-neutral-600 hover:bg-black/5"
        >
          Retry
        </button>
      </div>
    );
  }

  if (finished) {
    const flyAcc = result ? result.fly_accuracy : 0;
    return (
      <>
        <Confetti
          recycle={false}
          numberOfPieces={400}
          tweenDuration={8_000}
          width={width}
          height={height}
        />
        <div className="mx-auto flex h-full max-w-lg flex-col items-center justify-center gap-y-6 px-6 text-center">
          <h1 className="text-lg font-bold text-neutral-700 lg:text-3xl">
            Lesson complete.
          </h1>
          <div className="flex w-full items-center justify-center gap-x-4">
            <ResultCard variant="points" value={xp} />
            <ResultCard variant="hearts" value={hearts} />
          </div>
          <p className="max-w-md text-sm leading-relaxed text-neutral-500">
            You answered {answered} challenge{answered === 1 ? "" : "s"}. Over the same
            challenges the fly&apos;s readout was right {(flyAcc * 100).toFixed(0)}% of the
            time. Both numbers come from the same connectome activity.
          </p>
          <div className="flex flex-col items-center gap-3">
            <ModeBadge mode={frame?.mode ?? "intact"} />
            <ModeControls
              mode={frame?.mode ?? "intact"}
              onChange={onModeChange}
              disabled={modeBusy}
            />
          </div>
          <button
            onClick={() => void start()}
            className="rounded-xl border-2 border-b-4 border-green-300 bg-green-100 px-6 py-3 text-sm font-bold uppercase text-green-700 hover:bg-green-200"
          >
            New lesson
          </button>
        </div>
      </>
    );
  }

  if (!challenge) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="animate-pulse text-sm font-bold uppercase tracking-wide text-neutral-400">
          Loading the fly and the course
        </p>
      </div>
    );
  }

  const cardType = cardTypeFor(challenge.type);
  const title =
    challenge.type === "translate" || challenge.type === "select"
      ? "Select the correct meaning"
      : challenge.prompt;

  return (
    <>
      <Header
        hearts={hearts}
        percentage={percentage}
        hasActiveSubscription={false}
      />

      <div className="flex-1">
        <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-6 px-6 py-6 lg:flex-row lg:items-start">
          <div className="flex flex-1 justify-center">
            <div className="flex w-full flex-col gap-y-8 lg:min-h-[350px] lg:w-[600px]">
              <h1 className="text-center text-lg font-bold text-neutral-700 lg:text-start lg:text-3xl">
                {title}
              </h1>

              <div>
                {cardType === "ASSIST" && challenge.type === "translate" && (
                  <QuestionBubble question={challenge.prompt} />
                )}

                <Challenge
                  options={challenge.options}
                  onSelect={(i) => {
                    if (status !== "none") return;
                    setSelected(i);
                  }}
                  status={status}
                  selectedOption={selected}
                  disabled={pending}
                  type={cardType}
                />
              </div>

              {result && status !== "none" && (
                <div className="rounded-xl border-2 border-neutral-200 p-3 text-xs text-neutral-500">
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono">
                    <span>
                      fly picked{" "}
                      <strong className="text-neutral-700">
                        {challenge.options[result.fly_choice] ?? result.fly_choice}
                      </strong>{" "}
                      {result.fly_correct ? "(right)" : "(wrong)"}
                    </span>
                    <span>
                      readout updated {result.updated_params.toLocaleString()} weights
                      {result.gated_params
                        ? `, ${result.gated_params.toLocaleString()} dopamine-gated`
                        : ""}
                    </span>
                    <span>loss {result.loss.toFixed(3)}</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {panel}
        </div>
      </div>

      <Footer
        disabled={pending || selected === undefined}
        status={status}
        onCheck={() => void onContinue()}
      />
    </>
  );
}
