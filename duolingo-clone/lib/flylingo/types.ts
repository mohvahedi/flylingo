/**
 * Types mirroring the frozen service contract in INTERFACES.md.
 *
 * These are hand-written rather than generated so the frontend has no build-time
 * dependency on the Python service.
 */

export type Mode = "intact" | "shuffled" | "no_edges" | "random_graph";

export type ChallengeType = "translate" | "select" | "match" | "fill";

/** One live frame from ws://127.0.0.1:8770/stream (20 Hz). */
export type BrainFrame = {
  t: number;
  step: number;
  challenge_id: string | null;
  /** Indices INTO `state`, not global neuron indices. */
  spikes: number[];
  /** Sampled neuron activity, -1..1, length 512. */
  state: number[];
  sampled_ids: string[];
  active_fraction: number;
  state_rms: number;
  probs: number[];
  chosen: number | null;
  reward: number | null;
  correct: boolean | null;
  mode: Mode;
  lesson_id: string | null;
  lesson_progress: number;
  accuracy: number;
  /** The fly readout's own accuracy, tracked separately from the user's. */
  fly_accuracy: number;
  streak: number;
  hearts: number;
  xp: number;
  controls: Partial<Record<Mode, number>>;
  // ---- reward and training telemetry, added with online training ----
  /** The live dopamine trace, 0..1. This is the same signal that gated the last weight update. */
  dopamine: number;
  /** How many reward pulses have fired this session. */
  dopamine_total: number;
  /** True when the readout started from no training at all, so learning is watchable. */
  fresh_brain: boolean;
  learned_correct: number;
  learned_answered: number;
  /** Accuracy over the last `window_size` answers. This is the curve that shows learning; the
   *  cumulative figure barely moves after a hundred answers and looks flat while it improves. */
  window_accuracy: number;
  window_size: number;
  lessons_completed: number;
  /** Which lesson of the course is in play, and how many there are. */
  lesson_pos: number;
  lesson_total: number;
  lesson_title: string | null;
  readout_kind: string | null;
  checkpoint_status: string | null;
  entropy: number;
  grad_norm: number;
  /** Supervised rehearsal steps taken from the replay buffer. */
  rehearsals: number;
  replay_size: number;
  training: boolean;
};

export type ApiChallenge = {
  id: string;
  type: ChallengeType;
  prompt: string;
  promptLang: string;
  answer: string;
  options: string[];
  correctIndex: number;
  audio: string;
  difficulty: number;
};

export type AnswerResult = {
  correct: boolean;
  /** Whether the fly's own sampled action was right. */
  fly_correct: boolean;
  fly_accuracy: number;
  fly_choice: number;
  user_choice: number;
  reward: number;
  /** The action the fly's readout sampled. */
  chosen: number;
  answer_index: number;
  probs: number[];
  loss: number;
  grad_norm: number;
  entropy: number;
  updated_params: number;
  gated_params: number;
  step: number;
  streak: number;
  xp: number;
  hearts: number;
  next_challenge: ApiChallenge | null;
};

export type SessionStart = {
  session_id: string;
  lesson_id: string;
  challenge: ApiChallenge;
};

export type Health = {
  status: string;
  neurons: number;
  edges: number;
  dataset: string;
  /** Path of the checkpoint that was actually loaded, or null if none was. */
  checkpoint: string | null;
  /**
   * Why the checkpoint was or was not loaded. A refused checkpoint means the readout
   * below is untrained, and the UI must say so rather than presenting its answers as
   * a trained model's.
   */
  checkpoint_status: string;
  /** Which readout architecture is loaded: 'prompt_index', 'policy_adapter', etc. */
  readout_kind: string | null;
  encoder_fingerprint: string | null;
  uptime_s: number;
};

export type CurriculumChallenge = ApiChallenge;
export type CurriculumLesson = {
  id: string;
  title: string;
  order: number;
  challenges: CurriculumChallenge[];
};
export type CurriculumUnit = {
  id: string;
  title: string;
  color: string;
  order: number;
  lessons: CurriculumLesson[];
};
export type Curriculum = {
  course: string;
  from: string;
  version: number;
  units: CurriculumUnit[];
};

/** The clone's visual variants, kept local so no DB schema is imported. */
export type CardType = "SELECT" | "ASSIST";

/** Map a curriculum challenge type onto the clone's two card layouts. */
export function cardTypeFor(type: ChallengeType): CardType {
  return type === "translate" || type === "select" ? "ASSIST" : "SELECT";
}
