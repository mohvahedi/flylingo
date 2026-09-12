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

Two honest notes on those last rows. The spike threshold is a **visualization convention on
an abstract rate model**, not a claim about biological spiking, matching the reference
implementation's own wording. And all three live modes have nearly the same activity
magnitude (0.105 intact, 0.108 shuffled, 0.096 random), so a viewer cannot tell them apart
by looking; they differ in pattern, not level. That is why the mode badge exists.

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

Without the brain service the page still renders and tells you the service is not running,
with the exact command to start it. The lesson needs no account, database, or payment
provider, which is why the clone's auth was removed for this route.

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
- `BrainCloud` (viz-brain): the neuron cloud. The soma coordinates are **real**: 139,662 of
  the 166,700 retained neurons carry a measured soma annotation from `somaLocation`, and
  27,038 are placed at their (class, in-degree decile) group centroid because no soma was
  annotated. The UI states which is which rather than implying every point is measured.

`components/viz/` in the app is generated by `python tools/sync_viz.py` from those two
projects. Edit the harnesses, not the copy.

## Honest status

Working and verified: the built connectome, the reservoir with all four control modes, the
Spanish curriculum served over HTTP, the live 20 Hz WebSocket stream, the lesson loop in a
real browser, the admin-free app build, and the real soma extraction.

Not yet proven: whether the intact connectome beats its controls on this task. That
comparison is the point of the project and its result gets reported whichever way it falls,
including a null. A seven-answer sample from one lesson is not evidence of anything.
