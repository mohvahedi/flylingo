/**
 * Duolingo's own sounds, plus the Spanish word spoken aloud.
 *
 * The four UI cues are the assets the clone ships: correct.wav, incorrect.wav and finish.mp3.
 * They are the real Duolingo sounds, so the lesson sounds like Duolingo.
 *
 * The Spanish word is spoken by the browser's own speech engine from the challenge's `audio`
 * field, which the curriculum sets to exactly the right answer (all 97 challenges carry it).
 * The shipped es_*.mp3 clips are deliberately NOT used for this: they are six ~1s placeholder
 * recordings reused across the whole course, so playing one on a given answer would be claiming
 * it says a word it does not say. Speaking the actual phrase cannot make that mistake.
 *
 * Autoplay: browsers refuse to play audio until the page has been interacted with. This runs
 * hands-off, so `unlock()` is called from the first real gesture anywhere on the page and the
 * unlock state is published, so the HUD can say "click once for sound" rather than silently
 * doing nothing while someone records.
 */

export type Cue = "correct" | "incorrect" | "finish";

const SRC: Record<Cue, string> = {
  correct: "/correct.wav",
  incorrect: "/incorrect.wav",
  finish: "/finish.mp3",
};

/** The finish fanfare is much louder than the two clicks, so it is trimmed relative to them. */
const VOLUME: Record<Cue, number> = {
  correct: 0.5,
  incorrect: 0.45,
  finish: 0.65,
};

const elements = new Map<Cue, HTMLAudioElement>();
let enabled = false;
let unlocked = false;
const listeners = new Set<() => void>();

function notify() {
  for (const l of listeners) l();
}

/** Created lazily and guarded, because this module is imported by a client component that
 *  Next.js still evaluates on the server for the first render. */
function elementFor(cue: Cue): HTMLAudioElement | null {
  if (typeof window === "undefined") return null;
  let a = elements.get(cue);
  if (!a) {
    a = new Audio(SRC[cue]);
    a.preload = "auto";
    elements.set(cue, a);
  }
  return a;
}

/** Pick a Spanish voice, preferring the European/Spanish-American ones the browser exposes. */
function spanishVoice(): SpeechSynthesisVoice | null {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return null;
  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return null;
  const es = voices.filter((v) => v.lang.toLowerCase().startsWith("es"));
  if (!es.length) return null;
  // Prefer an exact es-ES / es-MX, otherwise any Spanish voice at all.
  return (
    es.find((v) => /^es[-_](ES|MX|US)$/i.test(v.lang)) ??
    es.find((v) => /^es[-_]419$/i.test(v.lang)) ??
    es[0]
  );
}

export const sound = {
  get enabled() {
    return enabled;
  },

  get unlocked() {
    return unlocked;
  },

  /** True when the browser has a usable speech engine and at least one Spanish voice. */
  get canSpeak() {
    return typeof window !== "undefined" && "speechSynthesis" in window && spanishVoice() !== null;
  },

  /** Subscribe to changes in `enabled` / `unlocked`, so the control can render the truth. */
  subscribe(fn: () => void) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },

  /**
   * Turn sound on: this is itself the gesture browsers require, so the click that enables it is
   * also what unlocks playback.
   */
  async enable() {
    enabled = true;
    await this.unlock();
    notify();
  },

  disable() {
    enabled = false;
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    notify();
  },

  /**
   * Satisfy the autoplay policy by playing each clip muted, then rewinding it.
   *
   * Muted playback is always permitted, and once an element has been started inside a real user
   * gesture the browser lets it play audibly afterwards.
   */
  async unlock() {
    if (typeof window === "undefined") return;
    let ok = true;
    for (const cue of Object.keys(SRC) as Cue[]) {
      const a = elementFor(cue);
      if (!a) continue;
      try {
        a.muted = true;
        await a.play();
        a.pause();
        a.currentTime = 0;
        a.muted = false;
      } catch {
        ok = false;
      }
    }
    // The speech engine needs the same gesture.
    try {
      if ("speechSynthesis" in window) window.speechSynthesis.resume();
    } catch {
      ok = false;
    }
    if (ok !== unlocked) {
      unlocked = ok;
      notify();
    }
  },

  /**
   * Play a cue. The element is cloned per call so a finish fanfare cannot cut off a correct
   * chime that is still ringing.
   */
  play(cue: Cue, delayMs = 0) {
    if (!enabled) return;
    const fire = () => {
      const base = elementFor(cue);
      if (!base) return;
      const node = base.cloneNode(true) as HTMLAudioElement;
      node.volume = VOLUME[cue];
      void node.play().catch(() => {
        // Blocked before a gesture, or unsupported. The control surfaces this state.
      });
    };
    if (delayMs > 0) window.setTimeout(fire, delayMs);
    else fire();
  },

  /** Speak a Spanish phrase. Cancels anything still being said so answers never overlap. */
  speak(text: string, rate = 0.92) {
    if (!enabled || !text || typeof window === "undefined") return;
    if (!("speechSynthesis" in window)) return;
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = spanishVoice()?.lang ?? "es-ES";
      const v = spanishVoice();
      if (v) u.voice = v;
      u.rate = rate;
      u.pitch = 1;
      window.speechSynthesis.speak(u);
    } catch {
      /* speaking is a nicety; never let it break the lesson */
    }
  },

  /**
   * The full moment: Duolingo's chime, then the answer spoken in Spanish a beat later, so the
   * ear hears "that was right" and then "this is the word" as two distinct events.
   */
  correctAnswer(spanish: string) {
    this.play("correct");
    this.speak(spanish, 0.92);
  },

  wrongAnswer(spanish: string) {
    this.play("incorrect");
    // The word is still said on a miss, which is when a learner most needs to hear it.
    this.speak(spanish, 0.86);
  },
};

/** Kept so callers can read the lesson id out of a challenge id ("u1l1c3" -> "u1l1"). */
export function lessonOf(challengeId: string | undefined | null): string | null {
  if (!challengeId) return null;
  const m = /^([a-z]\d+l\d+)/i.exec(challengeId);
  return m ? m[1] : null;
}
