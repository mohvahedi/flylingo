# FlyLingo

A real fruit fly connectome answers Spanish questions, and a control experiment that asks whether the fly's wiring is what makes it work.

**Live: [flylingo.mohv80s.workers.dev](https://flylingo.mohv80s.workers.dev)**

The thing choosing the answers is the **MaleCNS v1.0** reconstruction from the FlyEM project at Janelia: 166,700 neurons, 25,582,938 directed edges, 124,177,617 synaptic contacts. A prompt is encoded, the connectome is settled for six recurrent steps, and the answer is read out of the activity of the fly's own neurons. Then the same experiment runs three more times on graphs that are not the fly's, to find out whether any of it matters.

It does not, or at least the specific wiring does not. That is the finding, and it took three attempts to get there honestly.

## The question

Feed a fixed recurrent network a prompt, watch it settle, and read the answer off its state. That much is standard reservoir computing, and it would work with almost any recurrent graph. So the interesting question is not whether it works. It is whether **this** graph, a real one measured synapse by synapse from an actual fly, beats a graph of the same size that is not the fly's.

Four versions of the brain are trained identically and compared:

| arm | what the wiring is |
|---|---|
| `intact` | the measured connectome |
| `shuffled` | the same graph with nodes relabelled, so the topology survives but the input and output interfaces move |
| `random_graph` | a degree matched random matrix with the same number of edges |
| `no_edges` | disconnected, so the state is exactly zero |

## The answer

| arm | score | spread | final epoch |
|---|---|---|---|
| `intact` | **87.7%** | 1.8% | 78.1% |
| `shuffled` | 88.2% | 2.3% | 80.2% |
| `random_graph` | 87.0% | 1.9% | 81.4% |
| `no_edges` | 21.6% | 0.0% | 21.6% |

40 epochs, 5 seeds, paired per seed, score is the mean of the last 10 epochs. The majority class baseline is 26.8%.

**A recurrent graph is required.** Take the edges away and the fly is at 21.6%, below the baseline, while every connected arm reaches the high eighties. The recurrence is doing real work.

**Which recurrent graph makes no measurable difference.** Against each control separately, the real connectome is 0.5 points behind the shuffle and 0.7 points ahead of the random graph, against spreads of 2.1 and 1.7. It led the best control on 0 of 5 seeds.

One trap when reading that table. "Best control" is a maximum over two noisy arms, so it is biased against `intact`, and "intact is 1.4 points behind the best control" is **not** evidence that the real wiring is worse. Read the per control differences, each against its own spread.

The final epoch column is there for a reason too. It sits 6 to 9 points below the run's own level every time, which is why the score is a tail mean and not the last number the loop happened to print.

## What is not claimed

- **This is not language.** It is memorisation of a 97 item phrase to answer mapping, and every phrase is shown every epoch. Nothing here generalises to unseen sentences.
- **This is not a fly thinking.** The connectome is used as a reservoir. It supplies fixed dynamics, a readout turns those into a choice, and the learning happens in the readout.
- **The answer pools are not cell types.** They are disjoint groups of real neurons chosen by a seeded shuffle. They are real neurons of the reconstruction, and nothing more is claimed.
- **The null result is not a failure to find something.** It is the third time this project landed there, and the first time with a measurement that could have come out the other way.

## How it works

The design worth describing is the second readout, `plastic_brain`, because it is the one where the wiring question can actually be asked.

**Choosing.** Four answer pools are disjoint groups of real neurons. The prompt is encoded, the connectome settles, and the answer is whichever pool scores highest, each pool's score being a learned weighted sum over its own neurons. There is no classifier outside the brain and the argmax is over the brain's own populations.

**Learning.** The trained parameters are **117,800 real connectome edges**, the ones whose target neuron lies in a pool, each carrying a multiplicative scale on its anatomical weight. The synapses that change are real synapses onto real neurons.

The readout is weighted rather than averaged for a measured reason. An equal weight mean over a pool was only 37.1% linearly separable against 100% for the same neurons read with learned weights. Averaging discards *which* neurons fired, and that is where the signal is.

There is also an older readout, `prompt_index`, which keeps the graph frozen and learns in a 516 parameter layer beside it. It reaches 100% on the 97 challenges. It is the honest control for the newer one, because in the frozen design the readout can absorb any weakness in the wiring, and a control comparison run through a readout like that measures the readout.

## Where this went wrong

Three real errors, all of them kept in the repository rather than tidied away.

**The first wiring claim came from a readout too weak to see the answer.** An equal weight mean readout produced intact +34.0%, shuffled +4.1%, random -5.2% and was reported as "the wiring matters". Probing the frozen representations showed the readout's own input was only 37.1% separable while the state it was averaging carried 100%. The spread belonged to the readout. Retracted.

**The second came from one epoch of one seed.** A 15 epoch run reported intact 83.5% against shuffled 74.2% and printed a verdict. The same run's per epoch gap averaged +0.1 points with a standard deviation of 3.7, ranging from -9.3 to +8.2, and intact led in 18 of 40 epochs, which is 45%, where a coin flip is 50%. Stopping at epoch 10, 20, 30 or 60 says the opposite of stopping at 15. Retracted.

**The third was a genuine bug in the accelerated path.** A `cupyx` sparse matvec canonicalises a non-canonical CSR matrix in place. The random control is built with 31,231 duplicate `(row, column)` pairs on purpose, so the first matvec merged them, rewrote the index pointers, and collapsed the uploaded mirror from 25,582,938 stored entries to 25,551,707. The trainer writes weights **by position**, so from that moment the GPU trained a matrix the CPU never had. It agreed to 3e-07 on an untrained brain and diverged by 6.0e-01 after six plasticity steps, which means it passed the obvious test and corrupted exactly the comparison being run. Every GPU figure from that period was wrong and is retracted.

That one is fixed, pinned by a test that fails on the old code, and the equivalence gate now runs **before** any number is read: fresh settles must agree, the uploaded matrix must keep the CPU's exact layout, and trained weights must land on the same entries. What remains between the two devices is float32 training drift of order 1e-3, which is reported rather than gated, because gating on it would fail a correct implementation.

## Verified numbers

| quantity | value |
|---|---|
| retained neurons | 166,700 |
| directed edges | 25,582,938 |
| synaptic contacts | 124,177,617 |
| distinct measured soma positions | 139,668 of 166,700 |
| full graph step | 24.5 ms median, 40.9 Hz single threaded |
| GPU speedup on the recurrence | 26x, 5.2 ms against 136 ms per six step settle |
| plastic synapses trained | 117,800 |
| Spanish curriculum | 4 units, 15 lessons, 97 challenges |
| tests | 167 passing |

Source integrity is checked, not assumed. Both Feather files are verified against their published byte counts and SHA-256 before a single row is read, and the build refuses to proceed on a mismatch. The edge source is 1,051,241,946 bytes with SHA-256 `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1`, which matches the value pinned in `brain/scripts/build_graph.py`.

## Running it

Two processes. The brain service first:

```bash
cd brain
.venv/Scripts/python.exe -m uvicorn brain.api:app --host 127.0.0.1 --port 8770
```

Then the app:

```bash
cd duolingo-clone
bun run start --port 3300
```

Open `http://127.0.0.1:3300/fly` for the HUD, or `/lesson/fly` for the phone-in-front-of-the-fly view, or `/lesson/fly/brain` for all 166,700 neurons.

**Build, then restart, then verify.** `next start` caches its build manifest at boot, so rebuilding under a running server leaves it serving old HTML against new hashed chunks. The page hangs on its loading state with a chunk 404 and it looks exactly like a code bug. It cost a full debugging pass here.

One clone gets all of it:

```bash
git clone https://github.com/mohvahedi/flylingo.git
```

`duolingo-clone/` used to be a git submodule and needed `--recursive`. It is now part of this tree, grafted in as a subtree with its history, so ordinary commits work across the whole project.

The visualization harnesses run standalone with synthetic drive and no backend:

```bash
cd viz-fly    # or viz-brain
bun run dev
```

## Rebuilding the connectome

```bash
brain/.venv/Scripts/python.exe build_graph_from_reference.py
```

It verifies both source hashes, streams the 1 GB edge table in Arrow batches so peak memory stays low, and writes `cache/malecns_v1/` with a manifest carrying the SHA-256 of every array.

## Layout

```
brain/            Python service: connectome, reservoir, learning, curriculum, API
viz-fly/          the 3D fly and the phone it plays on
viz-brain/        the 166,700 neuron cloud
duolingo-clone/   the lesson shell, a fork of sanidhyy/duolingo-clone (MIT)
colab/            the reproducible notebook and the single file Colab runner
tools/            end to end checks, sync scripts, and the analysis tools
INTERFACES.md     the frozen contract every component was built against
```

`viz-fly` and `viz-brain` are the source of truth and `duolingo-clone/components/viz/` is generated from them by `python tools/sync_viz.py`. Edit the harnesses, not the copy.

## The notebook

`colab/FlyLingo.ipynb` runs the whole thing on a free Colab GPU in about 25 minutes. It has no git clone step, because the parent repository had no remote when it was written, so it embeds the project's own modules as base64 with their SHA-256 verified as they land on the VM. It reimplements nothing, so the numbers it reports come from the code the test suite covers.

The data needs no upload. The source Feathers are in a public bucket and are downloaded, byte counted and hashed at run time.

## Deployment

The live site is a Cloudflare Worker serving static assets from `site/`:

```bash
npx wrangler deploy
```

`site/` holds the landing page, both built harnesses, the rendered notebook, and a 45 second clip of the demo answering questions. The three.js bundles account for almost all of the 18 MB.

The app itself cannot be hosted the same way. It needs Clerk, Postgres, Stripe, and the Python service that steps the connectome, so the live site shows the visualizations running on synthetic drive and says so rather than implying the full pipeline is up there.

## Licences and credits

Two third party 3D models are used and both require attribution. Attribution is rendered in the HUD, where the work is shown, not only in this file.

| asset | author | licence |
|---|---|---|
| `fly.glb` | [victorberdugo1](https://sketchfab.com/victorberdugo1) | CC-BY-4.0 |
| `smartphone_with_green_screen.glb` | peroroo | CC-BY-SA-4.0 |

**The handset is share alike.** Adaptations of it must be released under the same licence, and that obligation would travel with any distribution of this project.

The connectome is the MaleCNS v1.0 release from the FlyEM project at Janelia Research Campus. It is downloaded at run time and never redistributed here. Check the dataset's own terms before reusing it.

The lesson shell starts from [sanidhyy/duolingo-clone](https://github.com/sanidhyy/duolingo-clone) (MIT, Sanidhya Kumar Verma). The reference implementation this work was checked against is [nftechie/flm](https://github.com/nftechie/flm) (MIT, Alex Wormuth), which is deliberately **not** included in this repository.

Built with the help of an AI agent. Every number in this file came from a command that ran, and the scripts that produced them are in `tools/` and `brain/scripts/`.
