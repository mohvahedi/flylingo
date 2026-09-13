/**
 * WebSocket client for the FlyLingo brain service.
 *
 * The wire format is the frozen `BrainFrame` from INTERFACES.md: snake_case
 * keys, `state` values in -1..1 (length 512), `spikes` as indices INTO `state`
 * (never global neuron indices), `sampled_ids` as decimal strings, `t` in
 * milliseconds.
 *
 * The client never invents a frame. If a payload does not carry a non-empty
 * numeric `state` array it is dropped, and the caller stays on whichever
 * source it was already showing.
 */

import type { LiveFrame } from './mapping';

export type StreamStatus = 'connecting' | 'live' | 'closed';

export const DEFAULT_STREAM_URL = 'ws://127.0.0.1:8770/stream';

/** The subset of the frozen frame this harness reads. Everything is optional. */
export interface WireBrainFrame {
  t?: number;
  step?: number;
  challenge_id?: string | null;
  spikes?: number[];
  state?: number[];
  sampled_ids?: (string | number)[] | null;
  active_fraction?: number;
  state_rms?: number;
  probs?: number[];
  chosen?: number | null;
  reward?: number | null;
  correct?: boolean | null;
  mode?: string;
  lesson_id?: string | null;
  lesson_progress?: number;
  accuracy?: number;
  streak?: number;
  hearts?: number;
  xp?: number;
  controls?: Record<string, number | null> | null;
}

export interface DecodedFrame {
  /** Exactly the shape BrainCloud consumes. */
  frame: LiveFrame;
  mode: string;
  tMs: number;
  step: number;
  challengeId: string | null;
  /** Accuracy under each control mode, numbers only. */
  controls: Record<string, number>;
  /** Spikes dropped because they were not a valid index into `state`. */
  droppedSpikes: number;
}

function finite(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/**
 * Decode one wire payload. Returns null for anything that is not a usable
 * frame, so the socket layer can ignore it instead of rendering noise.
 */
export function decodeFrame(raw: unknown): DecodedFrame | null {
  if (typeof raw !== 'string' || raw.length === 0) return null;

  let parsed: WireBrainFrame;
  try {
    parsed = JSON.parse(raw) as WireBrainFrame;
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== 'object') return null;
  if (!Array.isArray(parsed.state) || parsed.state.length === 0) return null;

  const state = new Array<number>(parsed.state.length);
  for (let i = 0; i < state.length; i += 1) {
    const v = parsed.state[i];
    state[i] = typeof v === 'number' && Number.isFinite(v) ? v : 0;
  }

  const spikes: number[] = [];
  let droppedSpikes = 0;
  const wireSpikes = Array.isArray(parsed.spikes) ? parsed.spikes : [];
  for (let i = 0; i < wireSpikes.length; i += 1) {
    const s = wireSpikes[i];
    if (typeof s === 'number' && Number.isInteger(s) && s >= 0 && s < state.length) {
      spikes.push(s);
    } else {
      droppedSpikes += 1;
    }
  }

  const sampledIds: string[] = [];
  if (Array.isArray(parsed.sampled_ids)) {
    for (let i = 0; i < parsed.sampled_ids.length; i += 1) {
      const id = parsed.sampled_ids[i];
      if (id === null || id === undefined) continue;
      sampledIds.push(String(id));
    }
  }

  const controls: Record<string, number> = {};
  const wireControls = parsed.controls;
  if (wireControls !== null && wireControls !== undefined && typeof wireControls === 'object') {
    const keys = Object.keys(wireControls);
    for (let i = 0; i < keys.length; i += 1) {
      const v = wireControls[keys[i]];
      if (typeof v === 'number' && Number.isFinite(v)) controls[keys[i]] = v;
    }
  }

  return {
    frame: {
      state,
      spikes,
      sampledIds,
      activeFraction: finite(parsed.active_fraction),
      stateRms: finite(parsed.state_rms),
    },
    mode: typeof parsed.mode === 'string' && parsed.mode.length > 0 ? parsed.mode : 'unknown',
    tMs: finite(parsed.t),
    step: finite(parsed.step),
    challengeId: typeof parsed.challenge_id === 'string' ? parsed.challenge_id : null,
    controls,
    droppedSpikes,
  };
}

export interface BrainStreamOptions {
  url?: string;
  onFrame: (frame: DecodedFrame) => void;
  onStatus?: (status: StreamStatus) => void;
  /** Optional hook for decoded-but-unusable payloads. */
  onDrop?: (reason: string) => void;
}

/**
 * Reconnecting websocket client. The server is expected to be a local aiohttp
 * service that may not be running yet, so every close schedules a retry with
 * exponential backoff capped at 4 s. `stop()` cancels the retry loop.
 */
export class BrainStream {
  private readonly url: string;
  private readonly onFrame: (frame: DecodedFrame) => void;
  private readonly onStatus: ((status: StreamStatus) => void) | undefined;
  private readonly onDrop: ((reason: string) => void) | undefined;

  private ws: WebSocket | null = null;
  private timer: number | null = null;
  private attempt = 0;
  private stopped = true;
  private received = 0;

  constructor(options: BrainStreamOptions) {
    this.url = options.url ?? DEFAULT_STREAM_URL;
    this.onFrame = options.onFrame;
    this.onStatus = options.onStatus;
    this.onDrop = options.onDrop;
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.open();
  }

  stop(): void {
    this.stopped = true;
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      ws.close();
    }
    this.onStatus?.('closed');
  }

  /** Frames decoded since the last start(). Used by the UI as a traffic proof. */
  get framesReceived(): number {
    return this.received;
  }

  private open(): void {
    if (this.stopped) return;
    this.onStatus?.('connecting');

    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url);
    } catch {
      this.scheduleRetry();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.attempt = 0;
      this.received = 0;
      this.onStatus?.('live');
    };

    ws.onmessage = (event: MessageEvent) => {
      const decoded = decodeFrame(event.data);
      if (decoded === null) {
        this.onDrop?.(`unusable payload at t=${this.received}`);
        return;
      }
      this.received += 1;
      this.onFrame(decoded);
    };

    ws.onerror = () => {
      // The close handler runs next and owns the retry.
    };

    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      if (this.stopped) return;
      this.onStatus?.('closed');
      this.scheduleRetry();
    };
  }

  private scheduleRetry(): void {
    if (this.stopped || this.timer !== null) return;
    const delay = Math.min(4000, 500 * 2 ** this.attempt);
    this.attempt = Math.min(this.attempt + 1, 4);
    this.timer = window.setTimeout(() => {
      this.timer = null;
      this.open();
    }, delay);
  }
}
