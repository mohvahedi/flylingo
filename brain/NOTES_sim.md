# Simulation core: progress notes

Workstream: `brain/brain/reservoir.py`, `brain/scripts/build_graph.py`,
`brain/tests/test_reservoir.py`. Written incrementally so a host freeze cannot
lose the work.

## Timeline

- Module skeleton written to disk first, before any data exploration.
- `scripts/build_graph.py` written second, with byte-count AND sha256
  verification of both feather sources before reading a row.
- `tests/test_reservoir.py` written third: synthetic tests that pin the
  recurrence against a direct numpy transcription of the frozen formula, plus
  real-graph tests that skip (never fake) if the build is absent.
- Refinements applied in place by patch: vectorised `random_graph`, removed the
  dead `_matvec` duplicate, added the opt-in `subgraph_size` fallback, fixed a
  typo in `load_connectome`'s docstring.

## Data state as observed

- `annotations.feather` 14483314 bytes, sha256 2177e246...a9a3b2. Verified.
- `edges.feather` reached exactly 1051241946 bytes and its recomputed sha256 is
  e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1, matching
  the published release hash. The download is COMPLETE.
- Correction to an earlier note: the "125.4M readable rows" figure measured
  against the partial file was NOT the full row count. The complete file holds
  151856684 raw edge rows (from the canonical manifest), which is why the
  retention filter matters: it brings 151.9M raw rows down to 25.58M directed
  edges.
- `cache/malecns_v1/` already existed when this workstream reached the build
  step (created by another workstream at 23:51) with the exact expected counts.
  `build_graph.py` refused to overwrite it, which is the intended write-once
  behaviour. The canonical build was therefore verified rather than replaced,
  and `build_graph.py` was proven end to end against the real sources by
  building into a separate scratch directory and comparing array hashes.

## Findings

- Retention rule reproduced exactly: nonempty `superclass` AND `status != Glia`,
  then both endpoints must be retained. 166700 neurons.
- 25582938 directed edges, 124177617 synaptic contacts, 151856684 raw rows,
  matching the canonical release exactly.
- Rows of W sum to 1 after dividing by total incoming contacts.

## Verification of the builder

`build_graph.py --out cache/_verify_build` was run against the real, complete
sources and its four arrays were hashed against the canonical artifact:

    MATCH  data.npy     7acb4f7cedf72c00
    MATCH  indices.npy  917e8d715e4075ea
    MATCH  indptr.npy   2efdd6bcfd538d4a
    MATCH  ids.npy      ab90597b7b0ce07c
    ALL ARRAYS BYTE-IDENTICAL: True

Guards were exercised deliberately and all three refuse rather than proceed:

1. truncated `edges.feather` (1048576 bytes) ->
   `ValueError: edges.feather is 1048576 bytes, expected exactly 1051241946.
   The download is incomplete or truncated; refusing to build.`
2. right byte count (14483314), wrong sha256 ->
   `ValueError: annotations.feather sha256 is 5876b896..., expected 2177e246...
   Source version mismatch; refusing to build.`
3. existing manifest without `--force` ->
   `FileExistsError: ...manifest.json already exists. This directory is
   write-once; pass --force only if you deliberately intend to rebuild it.`

Scratch directories `cache/_verify_build` (198 MB) and `cache/_guard_test`
(14 MB) were deleted after verification. `cache/malecns_v1/` was never written
to by this workstream.

## API integration

Exercised through the real FastAPI app with `TestClient(app)` as a context
manager (the startup hook must run, otherwise STATE is empty and `/health`
raises KeyError):

- `/health` -> 200, `neurons 166700, edges 25582938, dataset MaleCNS v1.0`
- `POST /control` with an unknown mode -> 400
- `POST /control` with each of intact / shuffled / no_edges / random_graph -> 200
- `POST /answer` -> 200
- `/telemetry` frame carries `t` (= `updates`), `mode`, `active_fraction`,
  `state_rms`, 512 `state` (= `sampled_state`), 512 `sampled_ids`, `spikes`.
  All values inside -1..1, every sampled id parses as int64.
- Under `no_edges`: `active_fraction == 0.0` exactly, `spikes == []`, and the
  sum of absolute state is exactly 0.0. Nothing synthetic is substituted.

## Performance (measured, not estimated)

Full graph, 166700 neurons, 25582938 nnz, CPU, single thread, numpy 2.5.3 /
scipy 1.18.1 in the project venv:

    full-graph step: 17.5 ms median  -> 57.0 Hz   (quiet machine)
    full-graph step: 21.2 ms median  -> 47.1 Hz
    full-graph step: 25.8 ms median  -> 38.8 Hz   (under pytest, loaded machine)

So the wall-clock per full-graph step is roughly 17.5 to 25.8 ms, i.e. 39 to
57 Hz single-threaded. The 20 Hz budget is 50 ms per step, so 20 Hz is met with
roughly 2x to 3x headroom on this box. The spread is machine load, not
variance in the kernel; the same code and the same data produce it.

Opt-in subgraph fallback (NOT required, but implemented and measured):

    full graph:      24.74 ms/step (40.4 Hz),  25582938 edges
    subgraph 30000:   0.99 ms/step (1014.7 Hz),  822805 edges

The fallback is a seeded contiguous-free subset of neurons with the recurrence
restricted to that sub-block of W; it is never applied implicitly.

## Rules observed

- No em dashes anywhere in code or docs.
- Nothing is fabricated: every number here comes from a command that ran.
- `no_edges` yields exactly zero features, empty spike list, active_fraction 0.0.
- Steps are allocation-free: the pre-activation, tanh output and readout buffers
  are allocated once in `__init__` and reused every frame.
