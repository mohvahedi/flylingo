# FlyLingo learning layer: measured findings

Owner: learning-layer workstream (trainer + verification tests).
All numbers below were produced by running code in this repo, not estimated.

## Environment

- Interpreter: `D:/Projects/flylingo/brain/.venv/Scripts/python.exe` (Python 3.12).
- Reservoir: real `brain/reservoir.py`, modes `intact`, `shuffled`, `no_edges`,
  `random_graph`. `load_connectome()` loads the real male-CNS connectome
  (`standin=False`). `FlyReservoir.sequence(embeddings, mode)` resets state at the
  start of every call, so features are a pure function of (mode, embedding list).
- Curriculum: `brain/brain/curriculum/es-en.json` exists (97 challenges, 4 units).

## Finding 1 (BUG, high): the entropy term in adapter.py is not the gradient of entropy

`PolicyAdapter.observe` builds the entropy contribution to the logit gradient as

```
dz += entropy_coef * p * (entropy - logp)
```

The docstring claims this was checked against finite differences. It was not
checked correctly. The true gradient of `H(z) = -sum_k p_k log p_k` is

```
dH/dz_j = -p_j * (H + log p_j)
```

The implemented expression differs in the sign of the `H` term.

Measured, float64 softmax, central finite differences (eps=1e-6):

- Analytic `-p (H + log p)` vs FD: max abs difference 2.2e-10 over random
  `z` with n_actions in {4, 8} and |z| up to 4. The analytic form is correct.
- Implemented `p (H - log p)` vs the same FD gradient: max abs difference 1.33.
- At the entropy maximum `z = (0, 0)`, FD gradient is exactly `(0, 0)` (float64
  prints 0.0). The implemented term returns `(0.693147, 0.693147)`.
- Projected onto the zero-sum subspace (the only part a softmax can see), the
  implemented term has cosine -0.885 with the true gradient: it pushes in the
  opposite direction.
- `sum_j` of the implemented term is `2H > 0` rather than 0, so it also carries a
  spurious component along the all-ones direction, i.e. it rescales the logits.
- Consequence, 200 gradient-ascent steps at lr 0.5, n_actions 4, starting at the
  uniform point H = 1.386003:
  - true formula: H rises to 1.386292 (entropy increases, as an entropy bonus should)
  - implemented term: H falls to 0.964310, a 30.4 percent drop.

So the "entropy bonus" as written actively sharpens the policy instead of
encouraging exploration. Effect size scales with `entropy_coef` (default 0.01), so
at the default coefficient this is a small perturbation on top of REINFORCE, but it
is the wrong sign and it is not what the docstring claims.

Estimated correct line for the implementer:

```
dz = dz - entropy_coef * probs * (entropy + logprobs)
```

This was applied on 2026-09-13: `observe` now uses `dz -= entropy_coef * p * (H +
log p)`, and the strict xfail was replaced by a real finite-difference pass (see
"Finding 1 fixed" below). `tests/test_learning.py` now checks the corrected line
directly.

## Finding 2 (BUG, was blocking the experiment): parameter-matched arm used a non-existent reservoir mode

`train.py` had `ARM_SPECS` mapping the `param_matched` arm to reservoir mode
`param_matched`, which the reservoir does not provide. The parameter-matched
control must use the SAME reservoir and features as the intact arm; the only
intended difference is `dopamine=False` via `clone()`. Fixed to
`('param_matched', 'intact', 'param_matched')`.

## Finding 3 (BUG, would have silently orphaned the checkpoint): wrong repo root

`brain/api.py` resolves `CHECKPOINT = parents[2] / "brain" / "runs" / "curriculum"
/ "adapter.npz"`, which is `D:/Projects/flylingo/brain/runs/curriculum/adapter.npz`.
`train.py` sits one directory deeper than `api.py`, so its `parents[2]` was
`brain/`, not the repo root, and it was writing to
`brain/brain/runs/curriculum/adapter.npz`, a path nothing loads. Fixed to
`parents[3]`. `tests/test_learning.py` asserts the two paths agree so this cannot
regress silently.

## Finding 4: dopamine rule at gate_lr 0 is a no-op on the trajectory

With `gate_lr = 0`, `dopamine=True` and `dopamine=False` produce bit-identical
weights, probabilities and sampled actions, because factor 2 contributes exactly
`0.0` to the weights. The reported telemetry still differs (`gated_params`,
`dopamine`, `gated_weight_delta`), so the identity is about the trajectory, not the
`observe` return value. Measured in `tests/test_learning.py`.

## Files in this workstream

- `brain/brain/learning/train.py` - three-arm trainer (intact, param_matched,
  shuffled) plus opt-in diagnostics, writes JSON with per-epoch accuracy per arm.
- `brain/scripts/train_curriculum.py` - CLI, saves the checkpoint the API loads.
- `brain/tests/test_learning.py` - math verification.
- `brain/runs/curriculum/results.json` - the measured learning curves.

## Finding 1 fixed (2026-09-13): entropy gradient corrected, verified by finite difference

`adapter.py` line in the "factor 1: REINFORCE plus an entropy bonus" block is now

```
dz = dz - entropy_coef * p * (H + log p)     # dH/dz_j = -p_j * (H + log p_j)
```

Independently re-derived: `H = -sum_k p_k log p_k`, `dp_k/dz_j = p_k(delta_kj -
p_j)`, `dH/dp_k = -(log p_k + 1)`, so

```
dH/dz_j = -p_j log p_j + p_j * sum_k p_k log p_k
```

and since `sum_k p_k log p_k = -H` that is `-p_j log p_j - p_j H = -p_j(H + log
p_j)`. Note where the earlier hand check slipped: its intermediate
`-p_j log p_j + p_j sum_k p_k log p_k` is already the correct gradient, but the
substitution `sum_k p_k log p_k -> +H` in the last step was a sign error and
produced `p_j(H - log p_j)`. The shipped line uses the correct form, so the old
`p*(H - log p)` was the wrong side and the two entropy tests were right about the
defect existing.

Measured with central finite differences (float64 softmax, eps=1e-6, random `z`
with `|z| <= 4`, `n_actions` in {4, 8}): analytic `-p(H + log p)` agrees with the
FD gradient to 2.259e-10; the old `p(H - log p)` is off by 1.369. End to end
through the shipped `observe` (REINFORCE factor pinned to exactly zero by
`reward == baseline == 0`, dopamine off, so only the entropy line can act): the
logit change the adapter actually produces has cosine +0.707 to +0.943 with the
true FD gradient and -0.703 to -0.985 with the old expression, and policy entropy
rose in 6 of 6 trials. (Cosine below 1 is expected: the step is taken in
parameter space, so the logit move is `J J^T dz`, a smoothed `dz`.) At the uniform
point the FD gradient is exactly 0 and the corrected adapter line moves the logits
by exactly 0; the old line moved them by 0.693147 per coordinate, which is the
wrong sign and the reason entropy fell. Ascent from 200 random starts for 400
steps each tops out at 1.386294361120 = `log 4` to 12 decimals and never exceeds
it, so uniform really is the highest-entropy point and a correct entropy bonus
has to push toward it.

`tests/test_learning.py`: the strict xfail on
`test_adapter_entropy_term_matches_entropy_finite_difference` was removed because
it now passes for the right reason (it pins the corrected formula against a
central finite difference of entropy). `test_adapter_entropy_term_is_what_the_
source_line_computes` and `test_entropy_is_highest_at_the_uniform_point_after_
ascent` were updated to the corrected sign; the latter now checks that uniform is
the top of H and that ascent from just off it does not overshoot past it.

## Finding 5 (BUG, reporting only): `same_init_as_intact` was computed after training

`train.py` computed the control's same-init flag at the end of the arm loop, i.e.
after the control had been trained, so two adapters that started identical
disagreed by then and the field reported `False`. Fixed: the flag is now captured
before any arm trains, by comparing the intact arm's `W1/b1/W2/b2` with the arm's
own (name -> array from the frozen `parameters()` tuple set). It is a genuine
property of the run now: all three arms report `same_init_as_intact == True`, and
the intact and parameter-matched arms use byte-identical feature matrices
(`features_sha256` `45370c3f4eb2dbff` for both; `shuffled` differs), so the
comparison isolates the learning rule.

## Finding 6 (RESULT, null): real curriculum + real connectome, 30 epochs, three arms

Command: `.venv/Scripts/python.exe scripts/train_curriculum.py` (defaults).
Real `brain/reservoir.py` (`standin: false`, 166700 neurons), real
`brain/brain/curriculum/es-en.json` (97 challenges, 73 train / 24 eval, 0 dropped
as no-room or unreachable). 30 epochs x 73 train challenges = 2190 steps per arm.
Chance on the eval split is 0.25, and the train-majority constant policy would
score 0.167, so 0.25 is a fair baseline.

| arm | final eval | best epoch eval | train acc first -> last | mean reward | policy entropy |
|---|---|---|---|---|---|
| intact | 0.208 (5/24) | 0.500 | 0.288 -> 0.397 | 0.247 | 0.595 |
| param_matched | 0.167 (4/24) | 0.375 | 0.301 -> 0.466 | 0.332 | 0.650 |
| shuffled | 0.250 (6/24) | 0.333 | 0.219 -> 0.411 | 0.264 | 0.641 |

**The intact connectome arm did NOT beat the controls.** It finished 0.042 above
the parameter-matched control and 0.042 below the shuffled-graph arm, both one
eval item (1/24) apart. Exact binomial tail probabilities against 0.25: intact
0.753, param_matched 0.885, shuffled 0.578, so no arm is distinguishable from
chance on 24 held-out challenges. The intact arm's best epoch (0.500, raw
`P(X>=12|24, p=0.25) = 0.007`) is the maximum of 30 correlated evaluations, so it
is selection-inflated and is not evidence of learning; its last five epochs sit at
0.167-0.208. Net: a null result on this curriculum, at these settings, for one
run.

Diagnostics that bound the interpretation (not tuning, no hyperparameters moved):

- All three arms do rise above chance on the TRAIN split (0.40-0.47 by epoch 30),
  including the shuffled-graph arm, which means the arms are memorising 73
  training challenges rather than generalising. Train accuracy for the
  parameter-matched (ungated) arm is the highest of the three (0.466) while its
  held-out accuracy is the lowest, so on this run the dopamine gate shows no
  measured advantage.
- `scripts/_null_diagnostic.py`: `correctIndex` is near-uniform (train
  15/16/21/21, eval 6/10/4/4). A SUPERVISED linear softmax readout trained on the
  same intact features (4000 full-batch steps, standardised, 4 classes) reaches
  train accuracy 1.000 but only 0.333 (8/24, p=0.234) on the same held-out split.
  So the features are not degenerate (per-step rms 1.0, pairwise cosine spread
  -0.669 to 0.957) but a linear readout of them also fails to generalise to the
  held-out challenges. The null is therefore at least partly a feature/encoding
  generalisation limit, not only a credit-assignment failure.

Checkpoint actually written and reloaded the way the API loads it:
`brain/runs/curriculum/adapter.npz`, 311912 bytes, md5
`8249ffde691ed3635db33937807d2030`, written by the intact arm of
this run, loads into a fresh `PolicyAdapter(in_dim=128, n_actions=4, hidden=256,
seed=0)` (the `api.py` construction) as steps 2190, 34052 params, 1703 gated,
all weights finite, probabilities summing to 1.0. Sibling per-arm checkpoints
`adapter_param_matched.npz` and `adapter_shuffled.npz` sit next to it; full curves
and the comparison verdict are in `brain/runs/curriculum/results.json`.

Test suite after all of the above: `.venv/Scripts/python.exe -m pytest tests/ -q`
-> 82 passed, 0 failed, 0 xfail (15 of them `tests/test_learning.py`).

Throwaway evidence scripts (safe to delete): `scripts/_entropy_probe_check.py`
(FD check), `scripts/_verify_checkpoint.py` (reload + per-arm report),
`scripts/_null_diagnostic.py` (baseline + supervised ceiling),
`scripts/_real_run_probe*.py`, `scripts/_real_run.log` (full run output).
