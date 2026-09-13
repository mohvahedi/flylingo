# Does the connectome's wiring matter?

## Short answer

**No, and this time the answer is supported by the noise, not just a point estimate.** A
recurrent graph is required; which graph it is cannot be detected on this task.

Two claims, kept separate because they have different evidence:

1. **A recurrent graph is necessary.** The edge-free control sits at chance (21.6%) because its
   state is exactly zero. This is a hard, unambiguous result.
2. **Which graph is irrelevant.** intact, shuffled and random_graph all reach the same place. The
   best evidence for this is not a spread figure but the per-epoch series below.

## Two false findings, self-caught

This comparison has now produced two wrong answers, and both were measurement errors rather than
facts about the fly. They are recorded because the failure mode is instructive.

### False finding 1: "the wiring matters" (equal-weight mean read-out)

intact +34.0% / shuffled +4.1% / random -5.2%, reported as *"the intact connectome finished clear
of both controls; the wiring matters here."*

The read-out was the equal-weight mean of each pool's 200 neurons, which is 37.1% linearly
separable where the same neurons read with learned weights are 100%. It was measuring its own
weakness, and the intact graph happened to survive that weakness slightly better. **A control
comparison run through an inadequate read-out measures the read-out.**

### False finding 2: "the wiring matters" again (single epoch)

After fixing the read-out, a 15-epoch run produced intact 83.5% against shuffled 74.2%, a 9.3
point spread, and the script printed *"The intact connectome finished clear of both controls.
The wiring matters here."*

That reading is **one epoch out of a noisy series, and it was the lucky end of it.** From the same
run's own per-epoch record:

| statistic, epochs 21-60 | value |
|---|---|
| mean gap (intact − best control) | **+0.1 pts** |
| sd of the gap | 3.7 pts |
| range of the gap | −9.3 to +8.2 pts |
| epochs where intact led | **18/40 = 45%** (a coin flip is 50%) |
| epochs that would have printed "wiring matters" | 4/60 = **7%** |

Stopping at epoch 15 would have claimed a wiring advantage. Stopping at 10, 20, 30 or 60 would
not. The verdict was decided by where the loop stopped, not by the wiring.

Single-epoch accuracy swings by **7 to 8.5 points** (sd) within one arm, with a spread of up to 36
points between the best and worst epoch. Any conclusion drawn from one final epoch is a draw from
that distribution.

## What the numbers actually are

60 epochs, one seed, last-10-epoch means (a single final epoch is too noisy to quote):

| arm | last-10 mean | final epoch | start |
|---|---|---|---|
| intact (MaleCNS v1.0) | 94.8% | 92.8% | 19.6% |
| shuffled | 92.0% | 89.7% | 19.6% |
| random_graph | **95.1%** | 92.8% | 23.7% |
| no_edges | 21.6% | 21.6% | 21.6% |

Majority-class baseline 26.8%. The random graph is nominally *highest*.

**The ceiling run's honest headline is 94.7%, not 100%.** A peak of 100% occurred twice in 61
epochs; the last-10 mean is 94.7% with an sd of ~3.9 points, and only 9 of 61 epochs reached 95%.

## What is actually established

- The brain chooses: the answer is the argmax over its own four neuron populations.
- The brain learns: 117,800 plastic synapses on real connectome edges plus 800 read-out weights.
- It reaches ~95% train accuracy on the 97 challenges, from 19.6%.
- A recurrent graph is necessary and the specific wiring is not detectable as an advantage.

## Limits, stated plainly

- **Train accuracy on 97 examples, not a held-out set.** This measures fit, not generalisation.
- **The noisy single-seed figures above are superseded** by the multi-seed run
  (`artifacts/seeds60.log`, 5 seeds, paired per seed). A single seed cannot separate a 5-point
  effect from seed noise, which is the whole lesson of this file.
- **Learnable-by-construction.** All 97 challenges are presented every epoch, so a readout of
  this size can fit them. That is what "learns Spanish" means at 97 phrases: memorisation of a
  phrase-to-answer mapping, not language.
- **The pools are seeded groups of real neurons, not identified cell types.**
- **No task with time in it has been run.** Every input is a static prompt, which is exactly the
  setting where a reservoir's memory cannot help and therefore cannot be shown to help. This
  remains the honest reason a wiring advantage might exist but be invisible here.

## The rule this produced

> A single final epoch from a single seed is not a measurement of a noisy training curve. Report
> an average over epochs, vary the seed, and compare the effect against the seed-to-seed spread
> before making a claim.

A script must not hold two pre-written verdicts (one for "the wiring matters", one for "it does
not") and choose between them on a 5-point threshold applied to one number. That is a claim
generator, and it produced two false findings. `measure_plastic_brain.py` now averages over a
tail of epochs, runs several seeds, pairs each seed's intact score against the same seed's
controls so the seed's difficulty cancels, and refuses to make a claim when the effect does not
clear the noise.
