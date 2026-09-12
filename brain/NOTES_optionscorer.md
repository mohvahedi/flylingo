# Option-scoring readout: does the real connectome learn the Spanish task?

Working notes, written as the work happens so a crash cannot lose the reasoning.
Every number below is a measurement unless it is explicitly labelled as a plan or a
budget.

## 1. The two causes this work addresses

1. **Encoder drift.** The old encoder used Python's builtin `hash()`, salted per process.
   The same input gave different vectors in different interpreters, so a checkpoint fit
   during training was applied at serving time to features the server never reproduced.
   The encoder now lives in `brain/encoders.py`, is built on blake2b, and declares
   `encoder_fingerprint()`, currently `4ef0063b75dc44c7`.
2. **The architecture.** `PolicyAdapter` mapped ONE challenge vector to four logits, so
   the readout had to memorise index-to-answer associations that nothing about the
   candidate could support. `brain/encoders.py` now offers
   `encode_option_pairs(challenge) -> (n_options, 256)`, one row per candidate, centred
   across candidates (measured mean pairwise cosine -0.317 after centring, against 0.92
   to 0.97 before). The task becomes matching rather than index lookup.

## 2. What was built

- `brain/learning/option_scorer.py`
  - `OptionScorer`: `score_i = w . tanh(W1 f_i + b1) + b2`, then softmax over `i`. One
    weight set applied to every candidate, so the map is **permutation equivariant by
    construction**: permuting the candidate rows permutes the scores identically and
    leaves the probability set unchanged.
  - `LinearOptionScorer`: the same form with the hidden layer removed, used as the
    supervised linear probe inside the option-scoring architecture.
  - `supervised_fit`: full-batch cross-entropy training for either model, fixed schedule,
    no early stopping and no evaluation-driven tuning.
  - API: `score`, `scores`, `logits`, `probs`, `sample`, `observe`, `parameters`,
    `gated_parameters`, `save`, `load`, `clone`, `config`. `observe` returns the same keys
    `PolicyAdapter.observe` returns, so a caller can consume either readout unchanged.
  - `PolicyAdapter` and its tests are untouched. `brain/api.py` still imports it and still
    boots.
- `tests/test_option_scorer.py`, 15 tests, all passing:
  - scores are exactly permutation equivariant, with and without the gated rule, and
    remain so after training;
  - probabilities permute and the probability set is invariant;
  - the single-candidate `score` matches the batch;
  - a 1-D input is refused with a message that names the reason;
  - `observe` without a preceding `sample` returns zeroes and moves nothing;
  - the gated count is exactly the enumerated mask: `k = round(0.05 * 256) = 13` hidden
    units, so `13 * (128 + 2) = 1690` parameters, and a test proves that with the gradient
    rule switched off (lr = 0) **only the gated entries move**;
  - checkpoint round trip carries `meta["encoder_fingerprint"]`, and a `PolicyAdapter`
    checkpoint is refused by `OptionScorer.load`.
- `scripts/train_optionscorer.py`: the experiment. Features are cached once per
  `(mode, steps_per_option)` because they depend on the graph and the encoding, not on the
  epoch, so a 30-epoch sweep over every arm costs no reservoir time.

## 3. The honest gate on the gated rule

The three-factor rule is real and its scope is enumerated rather than decorative: a fixed
seeded subset of whole hidden units, an eligibility trace that is reward-independent
Hebbian co-activity (pre-synaptic feature row times post-synaptic unit activation of the
**sampled** candidate), scaled by phasic dopamine `reward - tonic`. It is the same rule
family as `PolicyAdapter`'s gated rule, re-indexed for a scorer whose action is the chosen
candidate. It is a masked low-rank Hebbian update and is not claimed to be more than that.
The gated count is reported, not estimated.

## 4. The design of the measurement

- Real connectome: MaleCNS v1.0, 166700 neurons, 25582938 edges, loaded from
  `D:/Projects/flylingo/cache/malecns_v1`, reservoir `dims=128`, `embedding_dim=256`,
  `seed=7301`.
- Real curriculum: `brain/curriculum/es-en.json`, 97 usable challenges, the same split as
  the earlier run (`load_task(seed=0, eval_fraction=0.25)`), 73 train and 24 held out.
- Each candidate starts from a zeroed reservoir state and is driven for `steps_per_option`
  steps with its own embedding, so a candidate's feature depends only on that candidate.
  That keeps the whole pipeline, encoder then graph then readout, permutation equivariant.
- Arms on the real graph and its controls: `intact`, `param_matched` (identical
  architecture, identical initialisation, gated rule off), `shuffled` (fixed node
  relabelling), `random_graph` (degree-matched random sparse graph), `no_edges`
  (disconnected, features are exactly zero: the harness-validity arm), `intact_1step`
  (depth check), `raw_encoding_mlp` (no reservoir at all).
- Two conditions: `all97` (learn-to-criterion, the reframe: every challenge every epoch)
  and `split73` (train on the 73, report the 24 held out, the vocabulary-disjoint number).
- Supervised upper bounds on the same features and the same split: `linear_probe`,
  `supervised_mlp`, plus the same two on the 1-step features and on the raw encoding. These
  separate "the features cannot carry it" from "the learning rule cannot find it".

## 5. Expected numbers to compare against, from the prior work

- Vocabulary-disjoint ceiling for ANY readout on this encoding: 0.2784 (27/97), exact
  binomial p 0.294, i.e. not significant. See `NOTES_task_learnability.md`.
- The earlier index-lookup null: intact 0.208, parameter-matched 0.167, shuffled 0.250 on
  24 held-out items, chance 0.25.
- `runs/promptindex/results.json`: a 516-parameter prompt-index readout reaches 1.0 on the
  learn-to-criterion task under every wiring, and the raw encoding without any reservoir
  also reaches 1.0, so the connectome is not distinguishable from its controls there either.

## 6. MEASURED RESULTS (real connectome, real 97-challenge curriculum)

Run: `scripts/train_optionscorer.py --epochs 30 --steps 3`. Reservoir features cached once
per mode; 1164 full-graph steps per mode, 26.6 s for the intact cache. The run reached the
end of the supervised-bounds block for the `all97` condition and all but the last row of
the `split73` block, then was stopped on instruction. **No checkpoint and no results.json
were written by this run**, so `runs/optionscorer/` is present but empty and the live app
is unaffected (it checks `runs/promptindex/adapter.npz` first).

Split: 97 challenges, 73 train, 24 held out, chance 0.25, `correctIndex` counts
`[21, 26, 25, 25]` overall and `[15, 16, 21, 21]` in the 73 train items.

### Condition `all97` (learn-to-criterion, all 97 every epoch). Final epoch, epoch 29

| arm | fit on all 97 | on the same 24 items | mean reward | entropy |
|---|---|---|---|---|
| intact | 0.433 | 0.458 (11/24, fit not generalisation) | +0.291 | 0.354 |
| param_matched | 0.433 | 0.417 (10/24) | +0.291 | 0.413 |
| shuffled | 0.495 | 0.542 (13/24) | +0.369 | 0.304 |
| random_graph | 0.402 | 0.458 (11/24) | +0.253 | 0.354 |
| no_edges | 0.268 | 0.250 (6/24, exactly chance) | +0.085 | 1.386 |
| intact_1step | 0.423 | 0.500 (12/24) | +0.278 | 0.358 |
| raw_encoding_mlp (no reservoir) | 0.423 | 0.417 (10/24) | +0.278 | 0.573 |

### Condition `split73` (train on 73, the 24 are genuinely held out). Final epoch

| arm | fit on the 73 | held out (24) | binomial p vs chance |
|---|---|---|---|
| intact | 0.479 | 0.208 | 0.753 |
| param_matched | 0.507 | 0.125 | 0.960 |
| shuffled | 0.425 | 0.167 | 0.885 |
| random_graph | 0.521 | 0.042 | 0.999 |
| no_edges | 0.247 | 0.250 | 0.578 |
| intact_1step | 0.466 | 0.125 | 0.960 |
| raw_encoding_mlp (no reservoir) | 0.521 | 0.125 | 0.960 |

None is above chance; the best held-out arm is the intact graph at 5/24 and its tail is
0.753. The majority-index baseline alone gives 10/24 here, which is what the reward-driven
arms' 0.4 to 0.5 "fit" numbers actually are: the encoder's per-candidate features are
separable by an arbitrary constant, so sitting on one index scores about 0.42 on 24 items
without any matching.

### Supervised upper bounds, same features, same split

| bound | condition | fit | held out |
|---|---|---|---|
| linear_probe (intact) | all97 | 0.423 | 0.375 (9/24, p 0.121) |
| supervised_mlp (intact) | all97 | 0.701 | 0.625 (15/24, p 0.0001) |
| linear_probe_1step | all97 | 0.392 | 0.375 |
| linear_probe_raw | all97 | 0.464 | 0.333 |
| supervised_mlp_raw | all97 | 0.505 | 0.375 |
| linear_probe (intact) | split73 | 0.479 | 0.125 (3/24) |
| supervised_mlp (intact) | split73 | 0.671 | 0.250 (6/24, p 0.578) |
| linear_probe_1step | split73 | 0.452 | 0.083 |
| linear_probe_raw | split73 | 0.534 | 0.125 |
| supervised_mlp_raw | split73 | not reached | not reached |

### What this says

1. The connectome does **not** beat its controls. On the learn-to-criterion condition the
   intact arm fits to 0.433 while the parameter-matched control fits to exactly the same
   0.433 and the shuffled graph to a higher 0.495. Intact minus shuffled is -0.062. The
   `no_edges` arm, whose features are exactly zero, sits at 0.268 fit and exactly chance
   held out, which is the harness-validity anchor: the real-feature arms are above it, so
   the readout is fitting something, but every wiring fits it equally.
2. The supervised bound separates the two causes. A supervised MLP on the **same** intact
   features reaches 0.701 fit and 0.625 on the 24, p 0.0001, while the reward-driven intact
   arm reaches 0.433. So the features are not the bottleneck at this scale; credit
   assignment is. The gradient-driven rule extracts roughly 0.43 where a supervised fit of
   the identical architecture extracts 0.70.
3. But even the bound is not usable: 0.701 fit on 97 items is far from the 1.0 that the
   516-parameter prompt-index readout reaches, so the encoding, not the learning rule, is
   what caps this framing below usable accuracy.
4. Depth and wiring are not the lever: 1 step per candidate and 3 steps per candidate give
   the same numbers (0.423 vs 0.433), and the raw encoding with no reservoir at all fits to
   0.423, the same as the full MaleCNS graph.

### Relation to the lead's numbers

The lead measured, on the same framing: encoding only 0.536, reservoir features 0.567,
shuffled 0.577, random 0.567, raw encoding no reservoir 0.536. My numbers are in the same
regime and support the same conclusion, but they are lower (intact 0.433, shuffled 0.495,
raw 0.423 as a reward-driven fit; 0.464 linear probe and 0.505 supervised MLP on the raw
encoding). The difference is method, not a contradiction: the lead's figures come from a
supervised probe over paired features, mine from the reward-driven arms plus separate
probes with a different schedule and no per-feature standardisation. Nothing here
contradicts the lead's conclusion, and nothing here is a positive result.

## 7. Superseded

This framing was superseded while the run was in flight. The learnable framing that was
actually found is prompt-index memorisation of the prompt to the answer index:
`brain/learning/prompt_index.py`, trained by `scripts/train_prompt_index.py`, saved to
`brain/runs/promptindex/adapter.npz`, intact 1.000, shuffled 1.000, random_graph 1.000, raw
encoding with no reservoir 1.000, spread across the three wirings 0.000, and live in the
app. The option-scoring architecture was a real attempt at the same question, and the
question it was built to answer has a clean null for an answer: on this encoding and this
graph, matching over candidate features is indistinguishable from matching over a shuffled
or random graph, and both fall short of usable accuracy. The prompt-index result does not
contradict that; it says the learnable part of this task is which prompt maps to which
answer, which is memorisation, and that the connectome is not what supplies it.

Nothing was deleted. `brain/learning/option_scorer.py` and
`tests/test_option_scorer.py` are left in place and green.
