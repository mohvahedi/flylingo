# FlyLingo — frozen interface contract

Everyone building in this repo codes against this file. Do not change a signature here
without saying so in your final report; other workstreams are running against it.

## What the project is

FlyLingo is a Duolingo-style Spanish lesson where the thing choosing the answers is a
real fruit-fly connectome. The wiring is Janelia/Google **MaleCNS v1.0**: 166,700 neurons,
25,582,938 directed edges, 124,177,617 synaptic contacts. The graph is **frozen**. What
learns is a small readout adapter bolted onto it, updated by a reward signal on each
correct or incorrect answer. The web app shows the fly in 3D and the whole brain's activity
streaming live while this happens.

The honest description, which every user-facing surface must respect: the fly does not
understand Spanish. Measured fly wiring supplies fixed dynamics; a small trained readout
reads them out. The point of the project is the **control experiment** — does the measured
connectome beat a shuffled, edge-free, or random graph of identical size and sparsity at
matched parameters on the same task? Almost nobody in the September 2026 wave ran that
comparison. We do.

## Layout

```
D:\Projects\flylingo\
  cache\source\            raw MaleCNS feather files (1.05 GB + 14 MB)
  cache\malecns_v1\        built CSR graph: ids.npy data.npy indices.npy indptr.npy manifest.json
  brain\                   Python service (this package). Owns the sim, learning, API.
    brain\                 importable package
    tests\
  viz-fly\                 standalone React harness: 3D fly + animations
  viz-brain\               standalone React harness: 166,700-neuron live view
  duolingo-clone\          upstream sanidhyy/duolingo-clone, being stripped of auth/db/payments
  flm\                     reference implementation (read-only). graph.py has the reservoir.
```

## The graph on disk

`cache\malecns_v1\` holds the exact arrays `flm\scripts\prepare_graph.py` writes:

- `ids.npy` — int64, sorted ascending, exactly 166,700 entries. Neuron body IDs.
- `data.npy` `indices.npy` `indptr.npy` — scipy CSR, shape (166700, 166700).
- `manifest.json` — counts plus sha256 of every `.npy`.

Orientation: **row = postsynaptic, column = presynaptic**. `W[post, pre]` is contact count
divided by that neuron's total incoming contacts, so rows sum to 1. Edges are unsigned;
this deliberately does not infer transmitter sign, spikes, or biological time.

Verify integrity by recomputing sha256 against the manifest before use. Never write into
this directory.

## Feature interface (frozen)

`brain.reservoir.FlyReservoir`

```python
FlyReservoir(connectome, embedding_dim=256, dims=128, seed=7301)

.reset() -> None
.step(embedding: np.ndarray, mode: str = 'intact') -> np.ndarray
    # embedding: (embedding_dim,) float32. returns (dims,) float32, unit-RMS features.
.sequence(embeddings: np.ndarray, mode: str = 'intact') -> np.ndarray
    # (T, embedding_dim) -> (T, dims)
`.telemetry() -> dict`
    # {'updates': int, 'state_rms': float, 'active_fraction': float,
    #  'sampled_ids': [str], 'sampled_state': [float],   # -1..1, len 512
    #  'spikes': [int]}                                  # indices INTO sampled_state
    # No em dashes above; keep it that way.
    #
    # `spikes` is required: the API and both viz harnesses read it every frame. A value
    # counts as a spike when abs(sampled_state[i]) >= 0.5. Under 'no_edges' every state is
    # exactly zero, so `spikes` must come back empty and `active_fraction` must be 0.0.
    # That empty list is the honest visual signal that the graph is disconnected; do not
    # substitute placeholder activity for it.
    #
    # MEASURED ON THE REAL GRAPH (166,700 neurons, 25,582,938 edges), so components can be
    # built against reality rather than an assumed 0..1 range:
    #   state_rms = 0.105, active_fraction = 0.0044, about 5 spikes per frame.
    #   abs(sampled_state) percentiles: p50 0.051, p75 0.106, p90 0.174, p95 0.274,
    #   p98 0.386, p99 0.496, max 0.717.
    # Two consequences. First, a linear 0..1 colour scale renders an almost black frame,
    # so visualizations must auto-range against a per-frame reference such as the 95th
    # percentile (about 0.274) while still collapsing to genuinely empty on a dead frame.
    # Second, all three non-degenerate modes have nearly the SAME activity magnitude
    # (intact 0.105, shuffled 0.108, random_graph 0.096); the difference is in the pattern
    # (cosine between intact and shuffled features is 0.17, intact vs random_graph 0.01).
    # So no visualization may imply "the intact brain looks busier". The mode badge carries
    # that distinction, which is exactly why it is required and always visible.
    #
    # `spikes` is a visualization convention on an abstract rate model, not a claim about
    # biological spiking, consistent with the reference implementation's own wording.
.set_mode(mode: str) -> None
.mode -> str
```

`mode` is one of `'intact' | 'shuffled' | 'no_edges' | 'random_graph'`. `shuffled` is a fixed
node relabeling of W relative to the input/output interfaces, preserving topology.
`no_edges` zeroes W, so features go exactly to zero. `random_graph` substitutes a
degree-matched random sparse matrix with the same nnz. The single recurrence per step is

```
x_t = tanh(W @ (0.6 * x_(t-1) + 0.4 * B @ embedding_t))
f_t = normalize(P @ x_t)
```

`B` and `P` are fixed seeded sparse random interfaces, not anatomical language pathways.
This is the same rule as `flm\flm\graph.py`; read that file before implementing.

## Learning interface (frozen)

`brain.learning.adapter.PolicyAdapter`

```python
PolicyAdapter(in_dim=128, n_actions=4, hidden=256, seed=0)

.logits(features: np.ndarray) -> np.ndarray          # (n_actions,)
.probs(features: np.ndarray) -> np.ndarray           # softmax
.sample(features, rng) -> tuple[int, float]          # (action, logprob)
.observe(reward: float, logprob: float, features: np.ndarray) -> dict
    # one reward-modulated update. returns
    # {'loss': float, 'grad_norm': float, 'entropy': float, 'updated_params': int}
.save(path) / .load(path)
.parameters() -> int                                  # total param count
```

Reward is `+1.0` for a correct answer, `-0.25` for wrong, plus a small shaping term the
caller may pass. `.observe` must keep a running baseline so the policy update is
REINFORCE-with-baseline, and it must also apply a **dopamine-gated memory rule**: a small,
explicitly enumerated subset of adapter weights (name it, count it, expose the count) is
additionally modulated by the reward signal, mirroring DOOMFLY's dopamine-gated rule. Report
both the count of gated weights and the count of gradient-updated weights.

## Curriculum interface (frozen)

`brain.curriculum` loads `brain/curriculum/es-en.json`, validated by a pydantic model.

Module-level helpers the service calls, in addition to the loaders above:
`load_curriculum(path=None) -> Curriculum` with `.get_lesson(lesson_id) -> Lesson | None`,
`.first_lesson() -> Lesson`, and `.all_challenges() -> dict[str, list[Challenge]]` keyed by
lesson id. Keep those exact names; `brain/api.py` already imports them.

```json
{
  "course": "Spanish", "from": "English", "version": 1,
  "units": [
    {
      "id": "u1", "title": "Basics 1", "color": "#58cc02", "order": 1,
      "lessons": [
        {
          "id": "u1l1", "title": "Greetings", "order": 1,
          "challenges": [
            {
              "id": "u1l1c1",
              "type": "translate",
              "prompt": "Hello",
              "promptLang": "en",
              "answer": "Hola",
              "options": ["Hola", "Adiós", "Gracias", "Perro"],
              "correctIndex": 0,
              "audio": "Hola",
              "difficulty": 1
            }
          ]
        }
      ]
    }
  ]
}
```

`type` is one of `translate | select | match | fill`. Every challenge needs at least 3
`options`, exactly one correct, `correctIndex` consistent with `answer`, and `difficulty`
1-5. Option order is fixed in the file and audio is the Spanish text to speak.

## HTTP + WebSocket interface (frozen)

Service runs on `http://127.0.0.1:8770`.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{status, neurons, edges, dataset, checkpoint, uptime_s}` |
| GET | `/curriculum` | — | the curriculum JSON |
| GET | `/stats` | — | `{lessons_completed, accuracy, accuracy_by_difficulty, learning_curve[], controls{}}` |
| POST | `/session` | `{}` | `{session_id, lesson_id, challenge}` |
| POST | `/answer` | `{session_id, challenge_id, choice_index}` | see AnswerResult |
| POST | `/control` | `{mode}` | `{mode, note}` |
| POST | `/reset` | `{}` | `{ok}` |
| GET | `/telemetry` | — | one `BrainFrame`, current state |
| WS | `/stream` | — | `BrainFrame` at 20 Hz |

`challenge` is the current challenge object in curriculum form, with `options` already
ordered for display.

`AnswerResult`

```json
{
  "correct": true, "reward": 1.0, "chosen": 0, "answer_index": 0,
  "probs": [0.71, 0.12, 0.09, 0.08],
  "loss": 0.42, "grad_norm": 0.031, "entropy": 0.88,
  "updated_params": 81920, "gated_params": 8192,
  "step": 137, "streak": 4, "xp": 40, "hearts": 5,
  "next_challenge": { ...same shape as challenge... }
}
```

`BrainFrame` (both the WS payload and `/telemetry`)

```json
{
  "t": 12840, "step": 137, "challenge_id": "u1l1c1",
  "spikes": [12, 88, 1043], "state": [0.31, -0.92],
  "sampled_ids": ["72057594000000001"], "active_fraction": 0.041,
  "state_rms": 0.88, "probs": [0.71, 0.12, 0.09, 0.08],
  "chosen": 0, "reward": 1.0, "correct": true,
  "mode": "intact", "lesson_id": "u1l1", "lesson_progress": 0.6,
  "accuracy": 0.72, "streak": 4, "hearts": 5, "xp": 40,
  "controls": {"shuffled_accuracy": 0.51, "no_edges_accuracy": 0.25, "random_graph_accuracy": 0.48}
}
```

`spikes` are indices **into** `state`/`sampled_ids` (the sampled subset, not global neuron
indices). `state` values are -1..1, length 512. This is the exact payload the two
visualization workstreams render, so field names and ranges are not negotiable.

## Frontend contract for the visualization components

Both harnesses export a component taking the same prop:

```ts
type BrainFrame = { /* exactly the JSON above, camelCase as written */ };

// viz-fly
export function FlyStage(props: {
  activity: number[];        // frame.state, -1..1, len 512
  stateRms: number;
  activeFraction: number;
  correct: boolean | null;   // drives the reaction animation
  reward: number;
  mode: string;              // 'intact' etc, drives a badge
  width?: number; height?: number;
}): JSX.Element;

// viz-brain
export function BrainCloud(props: {
  state: number[];           // frame.state, -1..1, len 512
  spikes: number[];          // indices into state
  sampledIds: string[];
  activeFraction: number;
  stateRms: number;
  width?: number; height?: number;
}): JSX.Element;
```

Both must render correctly with no props supplied at all (synthetic idle animation), so the
app can mount them before the socket connects.

## Rules

- Windows 11 host. `terminal` runs bash (MSYS). Native tools (node, python, git) need
  forward-slash native paths like `D:/Projects/flylingo/...`; MSYS `/d/...` paths are not
  translated for them. Use `$LOCALAPPDATA/Temp` for scratch files native tools must read.
- No em dashes anywhere in code comments, docs, or copy.
- Never fabricate a number. If an evaluation did not run, say it did not run.
- The connectome is the only source of dynamics. Do not add simulated extra neurons and
  call them fly neurons.
