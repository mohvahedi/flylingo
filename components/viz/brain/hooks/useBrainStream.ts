/**
 * One hook that owns the frame source for the harness.
 *
 * It opens the websocket, keeps a ring of recent frames for the raster strip,
 * and drives the synthetic idle animation from useSyntheticFrame whenever no
 * live frame has arrived. The UI never has to decide which source it is on:
 * the hook reports `source` and hands back a frame either way.
 */

import { useEffect, useRef, useState } from 'react';
import { BrainStream, DEFAULT_STREAM_URL, type DecodedFrame, type StreamStatus } from '../live/socket';
import type { LiveFrame } from '../live/mapping';
import { syntheticFrame } from './useSyntheticFrame';

/** Recent frames kept for the raster strip. 240 frames at 20 Hz is 12 s. */
export const HISTORY_FRAMES = 240;
/** The service streams at 20 Hz; the idle animation matches it. */
const SYNTHETIC_INTERVAL_MS = 50;

export interface BrainStreamState {
  frame: LiveFrame;
  mode: string;
  source: 'live' | 'synthetic';
  status: StreamStatus;
  history: LiveFrame[];
  sampledIds: string[];
  activeFraction: number;
  stateRms: number;
  step: number;
  challengeId: string | null;
  tMs: number;
  controls: Record<string, number>;
  framesReceived: number;
  droppedSpikes: number;
}

function idleFrame(): LiveFrame {
  return { ...syntheticFrame(0), sampledIds: [] };
}

export function useBrainStream(url: string = DEFAULT_STREAM_URL, enabled = true): BrainStreamState {
  const [live, setLive] = useState<DecodedFrame | null>(null);
  const [status, setStatus] = useState<StreamStatus>('connecting');
  const [idle, setIdle] = useState<LiveFrame>(idleFrame);
  const [history, setHistory] = useState<LiveFrame[]>([]);
  const [framesReceived, setFramesReceived] = useState(0);
  const [droppedSpikes, setDroppedSpikes] = useState(0);
  const ring = useRef<LiveFrame[]>([]);

  const isLive = enabled && status === 'live' && live !== null;

  // Socket lifetime. A dead socket is not an error state here: the synthetic
  // branch below keeps drawing while the client retries in the background.
  useEffect(() => {
    if (!enabled) {
      setStatus('closed');
      return;
    }
    const stream = new BrainStream({
      url,
      onStatus: setStatus,
      onFrame: (decoded) => {
        setLive(decoded);
        setFramesReceived((n) => n + 1);
        if (decoded.droppedSpikes > 0) setDroppedSpikes((n) => n + decoded.droppedSpikes);
        pushFrame(decoded.frame);
      },
    });
    stream.start();
    return () => stream.stop();
    // pushFrame is stable (a ref based function declared below).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, enabled]);

  function pushFrame(frame: LiveFrame): void {
    const next = ring.current;
    next.push(frame);
    if (next.length > HISTORY_FRAMES) next.splice(0, next.length - HISTORY_FRAMES);
    setHistory(next.slice());
  }

  // Synthetic idle animation. Runs only while no live frame is arriving, and
  // the ring is cleared on every source switch so the strip never mixes a
  // live trace with an idle one.
  useEffect(() => {
    ring.current = [];
    setHistory([]);
    if (isLive) return;

    let cancelled = false;
    const t0 = performance.now();
    setIdle({ ...syntheticFrame(0), sampledIds: [] });
    const timer = window.setInterval(() => {
      if (cancelled) return;
      const frame: LiveFrame = { ...syntheticFrame((performance.now() - t0) / 1000), sampledIds: [] };
      setIdle(frame);
      pushFrame(frame);
    }, SYNTHETIC_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLive]);

  const frame = isLive && live !== null ? live.frame : idle;
  const sampledIds = isLive && live !== null ? live.frame.sampledIds : [];

  return {
    frame,
    mode: isLive && live !== null ? live.mode : 'idle',
    source: isLive ? 'live' : 'synthetic',
    status,
    history,
    sampledIds,
    activeFraction: frame.activeFraction,
    stateRms: frame.stateRms,
    step: isLive && live !== null ? live.step : 0,
    challengeId: isLive && live !== null ? live.challengeId : null,
    tMs: isLive && live !== null ? live.tMs : 0,
    controls: isLive && live !== null ? live.controls : {},
    framesReceived,
    droppedSpikes,
  };
}
