"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type {
  AnswerResult,
  BrainFrame,
  Curriculum,
  Health,
  Mode,
  SessionStart,
} from "./types";

/** The brain service. Override with NEXT_PUBLIC_FLYLINGO_API if it moves. */
export const API_BASE =
  process.env.NEXT_PUBLIC_FLYLINGO_API ?? "http://127.0.0.1:8770";

export const WS_URL = API_BASE.replace(/^http/, "ws") + "/stream";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${path} failed: ${res.status} ${detail.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return (await res.json()) as T;
}

export const api = {
  health: () => get<Health>("/health"),
  curriculum: () => get<Curriculum>("/curriculum"),
  startSession: (lessonId?: string, opts?: { fresh?: boolean }) =>
    post<SessionStart>("/session", {
      lesson_id: lessonId ?? null,
      fresh: opts?.fresh ?? false,
    }),
  /**
   * Submit an answer.
   *
   * `asFly` is for the hands-off demo: the fly answers with its own sampled action and the
   * lesson advances on that, so the frame runs by itself. Without it the human's click is what
   * counts and the fly is scored separately.
   */
  answer: (sessionId: string, challengeId: string, choiceIndex: number, asFly = false) =>
    post<AnswerResult>("/answer", {
      session_id: sessionId,
      challenge_id: challengeId,
      choice_index: choiceIndex,
      as_fly: asFly,
    }),
  control: (mode: Mode) => post<{ mode: Mode; note: string }>("/control", { mode }),
  reset: () => post<{ ok: boolean }>("/reset", {}),
  /**
   * Reset the readout to untrained, or reload the shipped checkpoint.
   *
   * The shipped checkpoint has already memorised the whole curriculum, so with it loaded there
   * is nothing to watch learn. This is how the UI offers "watch it learn from scratch" and
   * "already trained" as a real, honest choice instead of implying the pretrained model trains.
   */
  train: (fresh: boolean, extra?: { lr?: number; replaySteps?: number; training?: boolean }) =>
    post<{
      ok: boolean;
      fresh_brain: boolean;
      checkpoint_status: string | null;
      readout_kind: string | null;
      parameters: number;
      training: boolean;
      lr: number;
      replay_steps: number;
    }>("/train", {
      fresh,
      training: extra?.training,
      lr: extra?.lr,
      replay_steps: extra?.replaySteps,
    }),
};

export type StreamStatus = "connecting" | "open" | "closed";

/**
 * Subscribe to the live brain stream.
 *
 * Reconnects with backoff, and keeps the last frame so the panels can render
 * something real the moment the socket opens. When the service is absent the
 * caller still gets `status: "closed"` and `frame: null`, which the
 * visualization components must tolerate.
 */
export function useBrainStream(enabled = true) {
  const [frame, setFrame] = useState<BrainFrame | null>(null);
  const [status, setStatus] = useState<StreamStatus>("connecting");
  const [framesReceived, setFramesReceived] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);

  useEffect(() => {
    if (!enabled) {
      setStatus("closed");
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const connect = () => {
      if (cancelled) return;
      setStatus("connecting");
      let ws: WebSocket;
      try {
        ws = new WebSocket(WS_URL);
      } catch {
        setStatus("closed");
        return;
      }
      socketRef.current = ws;

      ws.onopen = () => {
        retryRef.current = 0;
        setStatus("open");
      };
      ws.onmessage = (event) => {
        try {
          setFrame(JSON.parse(event.data) as BrainFrame);
          setFramesReceived((n) => n + 1);
        } catch {
          // A malformed frame is not worth tearing the socket down for.
        }
      };
      ws.onclose = () => {
        setStatus("closed");
        if (cancelled) return;
        const delay = Math.min(500 * 2 ** retryRef.current, 5000);
        retryRef.current += 1;
        timer = setTimeout(connect, delay);
      };
      ws.onerror = () => {
        ws.close();
      };
    };

    connect();
    return () => {
      cancelled = true;
      clearTimeout(timer);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [enabled]);

  return { frame, status, framesReceived };
}

/** Poll the service for availability so the UI can explain an absent backend. */
export function useServiceHealth(enabled = true) {
  const [health, setHealth] = useState<Health | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  const check = useCallback(async () => {
    try {
      const h = await api.health();
      setHealth(h);
      setReachable(true);
    } catch {
      setReachable(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const run = async () => {
      if (cancelled) return;
      await check();
      if (!cancelled) timer = setTimeout(run, 5000);
    };
    let timer = setTimeout(run, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [check, enabled]);

  return { health, reachable, refresh: check };
}
