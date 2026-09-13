# FlyLingo — session handoff

Written 2026-09-13, end of a very long session. Read this first; it is the state of the world, not
a plan of record. Everything below was verified at the time of writing.

---

## 0. THE ANSWER — the wiring does not matter (2026-09-13, attempt 5)

The four-arm comparison finally ran on a **verified** accelerated path. Two machines, two budgets,
same conclusion.

**Colab T4, 40 epochs, 5 seeds** (`artifacts/colab_results/`, score = mean of last 10 epochs):

| arm | mean | sd | intact − this arm | sd of that gap |
|---|---|---|---|---|
| **intact** (real connectome) | **87.7%** | 1.8% | — | — |
| shuffled | 88.2% | 2.3% | **−0.5%** | 2.1% |
| random_graph | 87.0% | 1.9% | **+0.7%** | 1.7% |
| no_edges | 21.6% | 0.0% | — | — |

```
paired per seed:
 seed   intact  shuffled   random     best ctl     gap
    1   87.2%    87.4%     85.9%      87.4%      -0.2%
    2   89.2%    85.8%     90.4%      90.4%      -1.2%
    3   84.8%    86.2%     86.2%      86.2%      -1.3%
    4   87.3%    89.7%     84.9%      89.7%      -2.4%
    5   90.0%    91.8%     87.4%      91.8%      -1.8%
intact leads the best control on 0/5 seeds.
```

**Local CPU, 12 epochs, 3 seeds** (`brain/runs/plastic_brain/colab_measure.json`): intact 63.7%,
shuffled 62.8%, random_graph 64.3%, no_edges 21.6%; intact leads on 1/3 seeds. Same tie.

**Read it correctly.** Against each control *separately*, the real connectome is inside the noise:
−0.5% against the shuffle and +0.7% against the random graph, each against a spread of 1.7–2.1
points. The "−1.4% versus the best control" in the headline is **not** a finding either, and it is
partly an artifact of the comparison itself: `best control` is a `max()` over two noisy arms, which
biases the denominator upward. Do not report it as "the real wiring is worse".

**What IS established:** a recurrent graph is **required**. The edge-free control sits at 21.6% — the
majority-class baseline is 26.8% and uniform chance is 25.0% — while every connected arm reaches
63–88%. Disconnecting the brain destroys the behaviour; which connected graph you use does not
measurably matter.

**Verified, not asserted.** All 20 Colab checkpoints were re-loaded on this machine (a different
host from the VM that trained them) and their accuracy recomputed from scratch:
**20/20 reproduce the reported number exactly** (`brain/scripts/verify_colab_results.py`).

**This is attempt 1's conclusion, now properly supported.** The read-out was the ceiling then; the
accelerator was corrupting the weights until §3; with both fixed, the answer is unchanged. The 15
epoch table in §7 and the retracted claims in §3 were all pointing at this.

**What is still not claimed:** anything about *why* the wiring does not matter, and anything about
language. This remains memorisation of a 97-item phrase-to-answer mapping — every phrase is shown
every epoch — and that caveat belongs wherever the number is shown.

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

## 3. THE OPEN PROBLEM — RESOLVED 2026-09-13 (attempt 5)

**It was (a) A BUG, in the accelerated path, and it is fixed and pinned by a test.**

One commit-summary line, then the history below is kept because the reasoning still applies:

> `Reservoir._build_gpu_mirror` now declares the uploaded mirror `has_canonical_format = True`
> and refuses an upload whose stored-entry count or index pointers differ from the CPU matrix's.
> A cupyx sparse matvec was canonicalising the non-canonical control matrix **in place**,
> merging its 31,231 duplicate (row, column) pairs and renumbering every entry after each merge.
> `gpu_sync_data` writes trained weights **by position**, so from the first matvec onward the
> accelerator trained a matrix the CPU never had.

**How it was found, in the order it actually happened** (`brain/scripts/` has each step):

| step | script | what it settled |
|---|---|---|
| 1 | `diagnose_trained_sensitivity.py` | CPU→CPU reload exact; perturbing every trained weight by 1e-4 moves accuracy by **0.0 points**; relative top-2 margin 1.19 with 1% of prompts near a tie. **Trained system is not ill-conditioned, so this is not sensitivity → (a).** |
| 2 | `diagnose_gpu_stepwise.py` | First probe said all modes agreed — but it **re-implemented** the recurrence, so it tested a reading of the code, not the code. Kept only as a record of that mistake. |
| 3 | `diagnose_gpu_divergence.py` | Using the **shipped** `settle`, the mirror's stored-entry count came back **25,551,707 against the CPU's 25,582,938** — a shape error that named the mechanism. |
| 4 | `confirm_gpu_mirror_layout.py`, `probe_mirror_layout.py`, `instrument_mirror_stages.py`, `instrument_mirror_calls.py` | Localised it: the control matrix has **31,231 duplicate pairs**, the connectome has **0**, the constructor preserves the count, and the change happens on the **first matvec**. |
| 5 | `prove_matvec_reindex.py` | One matvec: 25,582,938 → 25,551,707, short by exactly 31,231, **product unchanged** (5.4e-07). With `has_canonical_format = True`: **layout preserved, indptr matches, same product.** |

Three hypotheses were tested and **refuted** before the real one: the cupyx constructor
preserving the count, `.tocsr()` returning a new object, and `mirror.data[idx] = …`
canonicalising. Each was a plausible story that a measurement killed.

**Why the existing tests could not catch it** — both gaps are now closed:

- `test_gpu_matvec_matches_cpu` runs **no plasticity at all**.
- `test_gpu_plasticity_matches_cpu` ran **only the default mode**, whose matrix has zero
  duplicate pairs and therefore nothing to collapse. It now loops every mode and additionally
  checks the mirror carries its own device's weights at the plastic positions, exactly.
- New: `test_gpu_mirror_keeps_the_cpu_layout_per_mode` — asserts entry count, index pointers and
  data per mode, after a settle. **Proven red on the old code** (reverting the one line reproduces
  `25551707 == 25582938` naming `random_graph`) and green with the fix.

**The equivalence gate now passes on an independent machine.** `colab/step1_build.py` rebuilds
the connectome from the public bucket (sha256 verified, 166,700 / 25,582,938 / 124,177,617
asserted) and then gates the accelerator before any number is read:

```
intact        fresh 2.725e-07  layout ok  placement 0.00e+00  | drift: settle 5.40e-03 scales 2.83e-05  PASS
shuffled      fresh 2.794e-07  layout ok  placement 0.00e+00  | drift: settle 1.34e-02 scales 1.84e-04  PASS
random_graph  fresh 3.881e-07  layout ok  placement 0.00e+00  | drift: settle 1.92e-04 scales 3.84e-04  PASS
no_edges      fresh 0.000e+00  layout ok  placement 0.00e+00  | drift: settle 0.00e+00 scales 0.00e+00  PASS
```

Before the fix the same gate read `intact 5.398e-03 / shuffled 1.341e-02 / random_graph 6.012e-01`.

**Two kinds of difference, now kept apart.** The mirror is **exact** (placement 0.0). The
remaining cross-device figures are **float32 training drift**, not a defect: a training loop feeds
each step's summation-order difference back into the weights, which is why it is 1e-3-ish rather
than the 1e-7 of a single settle. The gate therefore **gates** the exact invariants (fresh
agreement, layout, placement) and **reports** the drift, rather than papering over both with one
tolerance. 159 tests pass.

**What this does to §7's "not solid" list, as of now:**

- The 94.7% ceiling and 92.8% final: **re-measurable on the GPU**, but not yet re-measured. Still
  do not report them.
- The 26× speedup: measured on this path; the layout bug did not affect a matvec's arithmetic, so
  the figure stands, but re-verify alongside the accuracy.
- The wiring comparison: **still not answered.** The bug was one reason it kept failing; the other
  reasons (attempts 1-3) were separate and also real. The four-arm run needs re-running on the
  fixed path before anything is claimed.
- Replicated: the divergence is device-independent — a local 3070 Ti and a Colab T4 both showed
  it, with the same per-mode ordering.

**What is still owed:** the four-arm measurement on the fixed path (`colab/step2_measure.py`,
multi-seed, paired, saves checkpoints and recomputes every reported number from them). Its Colab
blocker is recorded in §4.

### The history — attempts 1 to 4

**The measurement was broken and I did not trust any GPU number.**

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

### The very next step — DONE, see the resolution at the top of §3

`brain/scripts/diagnose_trained_sensitivity.py` ran and answered its three questions: reload is
exact, a 1e-4 perturbation moves accuracy by zero, and decisions are not near ties. That ruled out
sensitivity and sent the search to the accelerator, where `prove_matvec_reindex.py` found the
in-place canonicalisation. The full chain is in the table at the top of §3.

The next step now is **§4's first item**: the four-arm measurement on the fixed path.

---

## 3b. Colab: what works, and the one blocker

`colab/step1_build.py` and `colab/step2_measure.py` are written and step 1 has been run end to end
on a T4. `tools/gen_payload.py` embeds the three modules plus the curriculum (with digests) so the
VM needs no clone, and `tools/colab_mcp_bridge.py` drives googlecolab/colab-mcp over stdio.

- **The browser MCP route is a dead end here.** Colab's page never made a TCP connection to the
  local websocket server: with connection-level logging added, the server logged the proxy URL it
  opened and then no request at all, from either `colab.research.google.com` or `colab.google.com`.
  The CLI route works instead.
- **The CLI must run on Windows, not WSL.** Everything is pinned in `./run_colab.sh`'s equivalent:
  a venv at `~/.local/colab-cli-venv` with `google-colab-cli` **and `jupyter-kernel-client<1`**
  (1.0.2 renamed `KernelClient` to `JupyterKernelClient` and the CLI breaks on it), plus `termios.py`
  and `tty.py` stubs in its site-packages because `colab_cli.console` imports them at module scope.
  WSL also failed `uv` installs with `os error 12` despite 25 GB free.
- **`colab exec` on a long script times out client-side** ("Timeout waiting for output") while the
  VM keeps working. Launch detached with `nohup <sys.executable> -u script.py > log 2>&1 &` through
  a short `colab exec`, then poll the log. `sys.executable` in the kernel is `/usr/bin/python3`,
  which has cupy 14.0.1 and pyarrow already.
- **BLOCKER: the T4 slot is held by an orphan assignment.** `colab sessions` reports
  `[?] gpu-t4-s-kkb-ass1c2-2illcv5vb8gzj`, with no local record, so the CLI cannot unassign it and
  `colab new --gpu T4` fails `TooManyAssignmentsError`. A CPU session allocates fine, which is the
  giveaway that it is the GPU slot specifically. Direct calls to
  `/tun/m/unassign/<vm>` return 400 without the CLI's headers. Resolutions, in order of effort:
  1. Open Colab, use *Manage sessions*, and terminate the T4 runtime by hand, then re-run
     `colab new -s fly --gpu T4`.
  2. Let it time out (the VM is idle, and an unused assignment is reclaimed) and retry later.
  3. Run `colab/step2_measure.py` on any other GPU machine — it needs only a GPU and network, and
     it rebuilds the connectome itself.

Note the account is `vmoh80s@gmail.com` and the token lives at `~/.config/colab-cli/token.json`
(mirrored into Windows so the Windows CLI can read it).

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
