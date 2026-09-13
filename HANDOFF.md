# FlyLingo — session handoff

Written 2026-09-13, end of a very long session. Read this first; it is the state of the world, not
a plan of record. Everything below was verified at the time of writing.

---

## 1. What this project is

A fruit-fly brain — the real Janelia/Google **MaleCNS v1.0** connectome, 166,700 neurons and
25,582,938 directed edges — shown learning Spanish, framed as a Duolingo-style lesson. A 3D fly
sits beside a phone whose screen renders real Duolingo; the fly's own sampled action picks the
answer and its forelegs tap the chosen card. The connectome is stepped live and its spikes stream
to a dashboard.

**Location:** `D:\Projects\flylingo` (parent git repo; `duolingo-clone/` is a **nested** git repo).

**The honest framing, which is a project standard and has been fought over more than once:**

- The fly **processes** (real connectome stepped at 20 Hz), **picks** (softmax over four pools of
  real neurons → the fly flies to that card), and a **trainable component learns**.
- What is NOT true in the frozen configuration: the connectome's own weights never change, and a
  raw encoding with no brain in it scores the same. Do not let a demo imply otherwise.
- With plasticity on, 117,800 real connectome edges **do** train. Then the brain genuinely both
  chooses and learns.
- 97 phrases, all shown every epoch. This is memorisation of a phrase-to-answer mapping, **not
  language**. Say that wherever the number is shown.

---

## 2. Current status

### Services (both up at time of writing)

| surface | status |
|---|---|
| http://127.0.0.1:3300/fly | **200** — flagship HUD, plays itself |
| http://127.0.0.1:3300/lesson/fly | 200 — phone-in-front-of-fly lesson |
| http://127.0.0.1:3300/lesson/fly/brain | 200 — neuron view, spikes streaming |
| http://127.0.0.1:3300 | 307 → redirects to `/fly` |
| http://127.0.0.1:8770/health | **200** — brain service |

Brain health right now: `readout_kind: plastic_brain`, `checkpoint_status: "plastic connectome,
scales at 1.0"` — i.e. **fresh, untrained**. Its brain runs on the **CPU** (the GPU path is only
wired into the experiment scripts).

App is started with `bun run start --port 3300` in `duolingo-clone`; brain with
`.venv/Scripts/python.exe -m uvicorn brain.api:app --host 127.0.0.1 --port 8770` in `brain`.
**Build before restarting the app**, or the chunk manifest goes stale and every route 500s.

### Tests

**158 passing** (`cd brain && .venv/Scripts/python.exe -m pytest tests/`). Includes new guard
files: `test_brain_isolation.py` (4), `test_plastic_checkpoint.py` (6), `test_gpu_matvec.py` (3).

### Last commits

Parent repo (`D:\Projects\flylingo`), newest first:

```
a4175db Save a checkpoint per arm, and prove each reported number reproduces from it
56e94fd Stop training one brain from rewriting the weights the next brain starts from
4860e0b Retract the wiring claim for a second time: the gap is epoch noise, not an effect
63add15 Record the full temperature/lr sweep and add the live demo verifier
f6e6c8d Correct the wiring claim: with an adequate read-out, the intact connectome ties the controls
ab035ce Run the recurrence on the GPU: 26x, and the experiments take minutes not an hour
```

Nested repo (`duolingo-clone/`): `5b0debc`, `d2cf020`, `2eeba12`, `5f2622f`.

### Uncommitted / untracked

- `brain/runs/plastic_brain/` — **6 arm checkpoints** (`.npz`) plus `convergence.json`,
  `convergence-mean.json`, `measure.json`, `measure_seeds.json`. Deliberately not committed.
- `brain/scripts/diagnose_random_graph_reload.py` — untracked, **new, just written**
- `brain/scripts/diagnose_trained_sensitivity.py` — untracked, **new, was interrupted mid-run**
- `tools/record_demo.py`, `viz-fly/tools/reach_sweep.{py,json,log}` — untracked odds and ends
- `duolingo-clone` shows modified content (its own commits above are made separately)

---

## 3. THE OPEN PROBLEM — start here

**The measurement is broken and I do not trust any GPU number.**

The four-arm control comparison (does the real connectome beat shuffled / random / no-edges?) has
now failed **four times, each for a different reason**. Every failure was found late. Attempts:

| # | what broke it |
|---|---|
| 1 | Equal-weight pool-mean read-out was the bottleneck (37.1% probe vs 100% for the same neurons read with learned weights). Spread was the read-out's, not the wiring's. Retracted. |
| 2 | Read a finding off **one noisy epoch**. The same run's per-epoch gap averaged **+0.1 pts, sd 3.7, intact led 18/40 = 45%**. Stopping at epoch 15 said "wiring matters"; stopping at 10/20/30/60 said no. Retracted. |
| 3 | `self.graph` is a **reference** to the shared connectome and `apply_plastic` writes into it in place, so training arm 1 rewrote the weights arm 2 started from. Measured: arm 2's baseline sat at exactly **8.00× the anatomical weights**. Fixed (private copy). |
| 4 | **GPU and CPU disagree once weights are trained.** UNRESOLVED. |

### Attempt 4, the live defect, in detail

The checkpoint verification I added caught it: the `random_graph` arm's checkpoint reproduced
**45.4%** where the curve claimed **87.6%**, and the script correctly refused to record the number.

Then `scripts/diagnose_random_graph_reload.py` narrowed it (this is the last thing that ran):

| check | result |
|---|---|
| weights in the checkpoint | **identical** (pool and scales) |
| the two random matrices | **identical**, max diff 0.000e+00 |
| one settle, **fresh** weights, GPU vs CPU | diff **2.98e-07** (fine) |
| one settle, **trained** weights, GPU vs CPU | diff **4.651e-01** on a matrix of scale 0.845 |

So: the two devices agree perfectly on a fresh brain and diverge badly on a trained one. The
divergence grows with weight scale. **Every GPU figure in this session, including the 94.7%
ceiling and the 26× speedup, comes from that path and is therefore suspect.**

Two candidate explanations, and they need different remedies:

- **(a) A BUG** — the GPU path computes something different, so it must be fixed before use.
- **(b) SENSITIVITY** — the trained recurrence amplifies a float32 difference of order 1e-7 across
  six recurrent steps, so each device is internally consistent but the two cannot be compared on
  trained weights. Nothing is wrong with either path.

### The very next step

`brain/scripts/diagnose_trained_sensitivity.py` was written to **distinguish (a) from (b)** and its
run was interrupted before it printed anything. It is CPU-only, so it is silent and safe. Run it:

```bash
cd D:/Projects/flylingo/brain
.venv/Scripts/python.exe scripts/diagnose_trained_sensitivity.py
```

It answers three things, all on CPU:

1. **CPU → CPU reload exact?** If not, the save/load code is broken and nothing else matters.
2. **The perturbation test.** Multiply trained weights by `(1 + eps)` for eps in 0, 1e-7 … 1e-4 and
   re-score. If eps=1e-6 moves accuracy by many points, the trained system is ill-conditioned →
   **(b)**, a numerical-sensitivity finding. If a 1e-6 perturbation changes nothing while the GPU
   still disagrees → **(a)**, a real bug in the accelerator.
3. **How close are decisions to ties?** The top-2 pool margin relative to the pool-score scale.
   If margins are tiny, decisions are knife-edge and this is (b).

**Only after that** should any further experiment be run, and **only on CPU or on Colab** — see
the constraint in §5.

---

## 4. Things asked for but NOT done

- **"Fresh or pretrained?" for the demo.** Now answerable (checkpointing works, 6 `.npz` on disk),
  not yet decided or wired. Service still starts fresh. Suggestion: fresh + checkpoint on exit so a
  reload does not wipe progress.
- **A public reproducible Colab notebook.** This is the *right* use of Colab (see §5) and it also
  solves the "no public repo" problem below. Not started.
  - **Blocker:** the parent repo has **NO GIT REMOTE**. The code exists only on this machine, so a
    Colab notebook cannot clone it. It must **embed** the three modules it needs
    (`brain/brain/reservoir.py` 738 lines, `plastic_brain.py` 380, `encoders.py` 198) — or the repo
    must be published first.
  - **Good news:** the data needs no upload. The source feathers are in a **public GCS bucket**,
    `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`, reachable over HTTPS
    (`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/…`).
    Verified: `connectome-weights-male-cns-v1.0-minconf-0.5.feather` returns 200 with
    `Content-Length: 1051241946` — **exactly** the byte count pinned in `build_graph.py`.
    Annotations 14,483,314 bytes. A Colab session can fetch these at datacenter speed; Colab's disk
    is ephemeral, so it must re-download per session (~15 min, automatic).
  - Local cache for reference: `cache/source/{edges,annotations}.feather` (1,017 MB) and
    `cache/malecns_v1/{data,indices,indptr,ids}.npy` (198 MB, the built CSR).
- **The viral post / video.** A 45 s H.264 capture exists at
  `C:/Users/Lion/Desktop/FlyLingo-45s.mp4` (1920×1080, no audio). **Do not publish a wiring claim** —
  see §3. The defensible claim is: the fly learns to pick the Spanish answer, 19.6% → ~95%, only
  removing the edges breaks it. Note "~95%" was measured on the suspect GPU path.

---

## 5. HARD CONSTRAINTS

**The GPU.** Mid-session the user said: *"the training is putting loads on my card, which is
running at full speed fans"*. All training was stopped immediately (GPU dropped to 11%, 19 W,
41 °C). **Do not start GPU work without asking.** If GPU work is wanted, prefer Colab — that is
what Colab is for here, and the user is right that it fits. Options offered, not yet chosen:
(1) pause GPU work entirely, (2) `nvidia-smi -pl 150` to cap power, (3) CPU-only.

**Safety / standing rules (from memory, still in force):**
- Never run destructive commands (`rm -rf`, `del -Recurse`, `git reset --hard`, `diskpart`,
  `reg delete`) unless explicitly requested for that exact path. `approvals.mode=smart`.
- **Never `git add -A`** in a repo with live subagents — it commits their WIP under your message.
  Add explicit paths. (This has bitten before.)
- Never touch v2rayN. Never run Hermes desktop/installer binaries.
- Research artifacts and reports go to `D:\Research\`, **never Desktop**. (The one exception:
  `FlyLingo-45s.mp4` is on Desktop because the user asked for it there.)

**Shell:** `terminal` runs bash (git-bash/MSYS) on this Windows host. Use POSIX syntax. Native
tools (git, node, python) do **not** get MSYS path translation — pass `C:/...` forward-slash paths,
and use `$LOCALAPPDATA/Temp` not `/tmp` for scratch files a native tool must read. Reference python
is `D:/Projects/flylingo/brain/.venv/Scripts/python.exe` (uv-managed, **no pip**: use
`VIRTUAL_ENV=.venv uv pip install`).

---

## 6. Lessons earned this session (do not relearn these)

- **A single final epoch from a single seed is not a measurement.** Single-epoch accuracy swung
  with an **sd of 7–8.5 points**, range up to 36 points. The 60-epoch run's honest headline is
  **94.7%** (last-10 mean, sd ~3.9), not the 100% that was the best of 61 draws.
- **Never let a script hold pre-written verdicts.** `measure_plastic_brain.py` used to pick between
  "the wiring matters" and "it does not" using a 5-point threshold on one number. That emitted a
  false finding. It now averages over a tail of epochs, runs several seeds, pairs each seed's intact
  score against the same seed's controls, and **refuses to claim when the effect does not clear the
  spread**.
- **Save what you trained.** Three training runs were discarded because the scripts only wrote JSON
  curves. The weights *are* the deliverable. Now every arm saves a checkpoint and **loads it back
  through the same guards the service uses, recomputing the accuracy and failing if it disagrees**.
  That check is what caught the attempt-4 bug.
- **A saved checkpoint is a debugging instrument.** Attempt 3's contamination was invisible for
  hours and would have been obvious in seconds from a saved arm's baseline.
- **A control comparison run through an inadequate read-out measures the read-out.** Probe the
  frozen representations before interpreting any spread.
- **Offload the whole loop, not the hot operation.** Offloading only the matvec gave 5×; keeping
  state resident on the device gave 26×. The bottleneck was twelve host/device syncs per settle,
  not throughput. A reported "3450×" was a ms-vs-s unit slip; the real figure is 26×.
- **Stream long runs to a log and tail it.** Output piped through a buffering filter reports nothing
  until exit, so a stall looks identical to progress. (I then reintroduced this by dropping a
  per-epoch print, and had to fix it again.)
- These are all recorded in the `ml-experiment-integrity` skill, which was extended four times
  during this session. **Read it before running another experiment.**

---

## 7. What is solid, and what is not

**Solid (CPU-verified, reproducible):**

- The connectome loads and is hash-verified: 166,700 neurons, 25,582,938 edges; edges.feather
  sha256 `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1`.
- The 26× GPU speedup **per se** (5.2 ms vs 135.7 ms per six-step settle) — measured, but see the
  caveat that the GPU path's *numbers* are suspect.
- The demo plays itself honestly: driven via the API, the lesson advanced `u1l1c1 → u1l1c7` over 12
  answers with the fly's own `as_fly` action, 16,241 brain ticks, spikes streaming, and
  `readout_kind` correctly surviving a session start.
- Checkpointing round-trips: `intact` 434 KB, `shuffled` 402 KB, `random_graph` 402 KB, `no_edges`
  **3.8 KB** (its scales never moved — a nice built-in confirmation the control does nothing).
- 158 tests passing.

**NOT solid — do not report these as findings:**

- The wiring comparison, in any form. Four failed attempts; no trustworthy answer.
- The 94.7% ceiling and 92.8% final figures: measured on the suspect GPU path. Re-measure on CPU.
- The 15-epoch table (intact 83.5 / shuffled 74.2 / random 79.4 / no_edges 21.6) — epoch noise.
- Anything implying the connectome's specific wiring matters, or that it does not. **Both have been
  claimed and retracted.**
- The 45 s demo video was recorded before the checkpoint/session fixes, so it may show the fly
  re-learning from scratch mid-clip.
