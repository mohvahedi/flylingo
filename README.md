# FlyLingo: a fruit fly learns Spanish

A Duolingo-style Spanish lesson where the thing choosing the answers is a real fruit-fly
connectome, with the fly's body and its neural activity streamed live beside the question.

The wiring is Janelia/Google **MaleCNS v1.0**: 166,700 neurons, 25,582,938 directed edges,
124,177,617 synaptic contacts, from the complete male fly central nervous system published
September 2026. The graph is **frozen**. What learns is a small readout bolted onto it,
nudged by reward on each answer.

## What this actually is, and is not

The fly does not understand Spanish. Measured wiring supplies fixed dynamics; a small
trained readout reads those dynamics out into a choice. Anyone who tells you a connectome
"learned" something is describing a decoder they trained.

What makes this project worth building is the part the September 2026 wave skipped: the
**control experiment**. Does the measured connectome beat a shuffled, edge-free, or
degree-matched random graph of identical size at matched parameters on the same task?
Every answer runs in one of four modes, and the mode is displayed at all times so a demo
can never be mistaken for the intact brain:

| Mode | What it is |
|---|---|
| `intact` | the measured MaleCNS wiring |
| `shuffled` | fixed node relabeling of W relative to the input/output interfaces, topology preserved |
| `random_graph` | degree-matched random sparse matrix, same nnz |
| `no_edges` | W zeroed, so features go exactly to zero |

## Verified numbers

Every figure below came from a command that ran on this machine against the real data.

| Quantity | Value | How |
|---|---|---|
| Retained neurons | 166,700 | manifest, and independently recomputed |
| Directed edges | 25,582,938 | manifest, and recomputed from CSR `nnz` |
| Synaptic contacts | 124,177,617 | manifest |
| Raw edge rows | 151,856,684 | manifest |
| Source integrity | both feathers match their published sha256 | recomputed |
| Full-graph step | 24.5 ms median (min 20.3, max 35.7) | 20 timed steps |
| Implied rate | 40.9 Hz single-threaded | derived from the above |
| 20 Hz stream budget | fits, with room | 50 ms budget vs 24.5 ms step |
| Sampled state rms | 0.105 | real challenge embeddings |
| abs(state) p50 / p95 / max | 0.051 / 0.274 / 0.717 | 512 sampled neurons |
| Spikes per frame | about 5 of 512 | `abs(state) >= 0.5` |
| Readout parameters | 34,052 total, 1,703 dopamine-gated | adapter |
| Spanish curriculum | 4 units, 15 lessons, 97 challenges | served by `/curriculum` |
| Distinct soma coordinates | 139,668 of 166,700 | `tools/extract_soma_positions.py` |
| Neurons with a measured soma | 139,662 | `somaLocation` in the annotations feather |
| Placement filled from group centroid | 27,038 | (class, in-degree decile) centroid |
| Connectome tick rate | 10 Hz, continuously between answers | background thread, locked |
| Test suite | 82 passed | `pytest tests/` |

Two honest notes on those last rows. The spike threshold is a **visualization convention on
an abstract rate model**, not a claim about biological spiking, matching the reference
implementation's own wording. And all three live modes have nearly the same activity
magnitude (0.105 intact, 0.108 shuffled, 0.096 random), so a viewer cannot tell them apart
by looking; they differ in pattern, not level. That is why the mode badge exists.

### Result: the readout learns the vocabulary; the connectome does not beat its controls

Reported in full, because the interesting part is *what* the learning is attributable to.

**The learning works.** A trained prompt-index readout answers all 97 curriculum challenges
correctly: measured 1.000, chance 0.250, 516 parameters.

**But the connectome contributes nothing measurable.** Four independent measurements:

| # | Measurement | Result | What it rules out |
|---|---|---|---|
| 1 | Vocabulary-disjoint held-out, encoding only, 5-fold CV over all 97 | 0.2784 (p=0.294) | No generalisation is possible from character n-grams |
| 2 | Learn-to-criterion, option-scoring framing, all 97 repeated | 0.536 | Correctness is relational and cannot be recovered by a global linear rule |
| 3 | Same framing, reservoir features vs raw vs controls | 0.567 vs 0.536, intact == shuffled == random | The connectome's features add nothing on that framing |
| 4 | Prompt-index memorisation, reservoir features | **1.000** | It works, but see the spread |

Measurement 4 is the one that matters for attribution. All three wirings and the raw
encoding reach exactly the same score:

| Features | dim | Accuracy (3 seeds) |
|---|---|---|
| raw prompt encoding, no reservoir | 256 | 1.000 |
| intact connectome | 128 | 1.000 |
| shuffled graph | 128 | 1.000 |
| degree-matched random graph | 128 | 1.000 |

Spread across the three wirings: **0.000**. The encoding with *no reservoir at all* also
reaches 1.000. So the 166,700-neuron wiring is not needed for this result: it supplies a
fixed nonlinear feature map, and any fixed nonlinear feature map of the same width does the
same job. The learning is in the readout.

This is a negative result about the connectome and it is reported as one, on the live UI as
well as here. Earlier iterations of this README claimed the null was uninformative because
the task was unsolvable; that was correct about the *first* framing (measurements 1 and 2)
but it stopped being the whole story once a learnable framing existed (measurement 4). Being
able to learn the task and finding the biology irrelevant to it is a stronger and more
useful statement than "the task was too hard".

What would actually test the connectome's dynamics is a task where memory or temporal
integration is the bottleneck, rather than one-to-one vocabulary lookup that a linear map
solves outright.

## Layout

```
flylingo/
  cache/source/       raw MaleCNS feathers (1.05 GB edges + 14 MB annotations), hash-verified
  cache/malecns_v1/   built CSR graph + manifest (write-once, everyone reads it)
  brain/              Python service: connectome, reservoir, learning, curriculum, API
  viz-fly/            standalone harness: the 3D fly
  viz-brain/          standalone harness: the 166,700-neuron cloud
  duolingo-clone/     sanidhyy/duolingo-clone, stripped of auth for local use
  flm/                reference implementation, read-only
  INTERFACES.md       the frozen contract every component was built against
  tools/              end-to-end checks and the viz sync script
```

## Running it

Two processes. The brain service first:

```bash
cd D:/Projects/flylingo/brain
.venv/Scripts/python.exe -m uvicorn brain.api:app --host 127.0.0.1 --port 8770
```

Then the app:

```bash
cd D:/Projects/flylingo/duolingo-clone
bun run dev --port 3100
```

Open http://127.0.0.1:3100/lesson/fly

There is also a full-screen connectome view at http://127.0.0.1:3100/lesson/fly/brain,
which renders all 166,700 neurons and lets you switch control modes while watching.

Without the brain service the page still renders and tells you the service is not running,
with the exact command to start it. The lesson needs no account, database, or payment
provider, which is why the clone's auth was removed for this route.

**Restart the app after every rebuild.** `next start` caches its build manifest at boot, so a
rebuild underneath a running server leaves it serving the old HTML against new hashed chunks.
The symptom is a page stuck on its loading state plus a 404 on a chunk, which looks like a
code bug and is not one. This cost real debugging time here: `/lesson/fly` appeared broken
while the API was perfectly healthy. Build, then restart, then verify.

The standalone harnesses run on their own, with synthetic idle animation and no backend:

```bash
cd D:/Projects/flylingo/viz-fly   && bun run dev      # or viz-brain
```

## Rebuilding the connectome

```bash
cd D:/Projects/flylingo
brain/.venv/Scripts/python.exe build_graph_from_reference.py
```

This verifies both source hashes, streams the 1 GB edge table in Arrow batches so peak
memory stays low, and writes `cache/malecns_v1/` plus a manifest carrying the sha256 of
every array. It refuses to proceed on a hash mismatch.

## Verifying it end to end

```bash
python tools/e2e_play.py
```

It loads the real page in headless Chromium, answers challenges correctly by looking up
the true option index, and reports what actually happened: advancement, accuracy, the
wrong-then-retry path, whether the canvases are painting, and any console errors.

## The two visualization components

Both are standalone Vite + React + TypeScript projects exporting one frozen interface, and
both must render with no props at all so the app can mount them before the socket opens.

- `FlyStage` (viz-fly): the fly built from primitives in code, with idle, walk, groom,
  proboscis, and startle states, its glow driven by the connectome.
- `BrainCloud` (viz-brain): the neuron cloud on its own route. The soma coordinates are
  **real**: 139,668 distinct measured positions out of 166,700, from the `somaLocation`
  column of `annotations.feather`. The UI states which points are measured and which are
  filled from a group centroid rather than implying every point is measured.

`components/viz/` in the app is generated by `python tools/sync_viz.py` from those two
projects. Edit the harnesses, not the copy.

Both point-size handling and the cloud's ambient colour needed real fixes before the brain
was visible at all, and the recorded lessons are in the source comments:

- Point size was computed as `uSize * (1.0 / max(-mv.z, 0.35))`. The cloud spans only about
  [-1, 1] with the camera roughly 3 units back, so that factor is about 0.33 and a 2.4px
  point became roughly 0.8px. Sub-pixel points rasterise to almost nothing. Sizes are now
  in pixels scaled by viewport height, with a floor so a point always covers a pixel.
- Non-active neurons were drawn in a dark navy that, additively blended at its alpha, landed
  near RGB(5,14,24), i.e. invisible. The ambient colour is now bright enough to read the
  anatomy on its own.

## Verifying it end to end

```bash
python tools/e2e_play.py     # the lesson loop, answering correctly
python tools/e2e_modes.py    # all four control modes, in the browser
python tools/e2e_viz.py      # both 3D views render, and no_edges is inert
```

`e2e_play.py` loads the real page in headless Chromium, answers challenges correctly by
looking up the true option index, and reports advancement, accuracy, the wrong-then-retry
path, whether the canvases are painting, and any console errors.

One measurement note worth keeping: `e2e_viz.py` uses element screenshots rather than
`readPixels`. react-three-fiber does not set `preserveDrawingBuffer`, so reading pixels
outside a frame returns zeros and would report a working scene as black.
