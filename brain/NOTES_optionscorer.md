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
