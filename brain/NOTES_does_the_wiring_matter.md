# Does the connectome's wiring matter? Corrected.

## Short answer

**No.** Any connected graph works; only a disconnected one fails. But reaching that answer
required fixing two of my own defects first, and one earlier claim of mine was wrong.

| arm | start | final (60 epochs) | gain |
|---|---|---|---|
| intact (MaleCNS v1.0) | 19.6% | **92.8%** | +73.2% |
| shuffled | 19.6% | **89.7%** | +70.1% |
| random_graph (degree-matched) | 23.7% | **92.8%** | +69.1% |
| no_edges | 21.6% | **21.6%** | +0.0% |

Spread across intact / shuffled / random: **3.1%**. Majority-class baseline 26.8%.

So two separate statements, both true:

1. **A recurrent graph is necessary.** `no_edges` is exactly chance, because its state is exactly
   zero and every pool reads the same value.
2. **Which graph is irrelevant.** The measured connectome is statistically indistinguishable from
   a shuffled relabelling of itself and from a degree-matched random matrix.

That is the classic reservoir-computing result: a sufficiently rich fixed recurrent structure
provides a useful feature space, and the specific wiring is not what carries the information.

## A claim I got wrong

Earlier, running with the equal-weight pool mean, the arms came out at intact +34.0%, shuffled
+4.1%, random -5.2%, and I reported that **"the intact connectome finished clear of both
controls; the wiring matters here."**

That was an artifact of a defective read-out, not a property of the fly. The mean read-out was
itself the bottleneck (37.1% linearly separable against 100% for the same neurons read with
learned weights), so it was measuring its own weakness, and the intact graph happened to survive
that weakness slightly better than the controls. Once the read-out is adequate, the gap closes to
3.1%.

The lesson: a control comparison run through an inadequate read-out measures the read-out.

## Why this now agrees with the original finding

The very first measurement in this project, with a frozen connectome and a separate 516-parameter
readout, had every arm at exactly 1.000 (spread 0.000) and was dismissed as uninformative because
a task everything can memorise cannot discriminate.

The conclusion is now the same from a design where the brain itself both chooses and learns. Two
independent architectures, one answer: **the connectome confers no measurable advantage on this
task.** The earlier dismissal was right about the measurement and wrong to imply the question was
still open.

## What the two defects were

Both were mine, and both had to be fixed before the comparison could mean anything:

**1. The recurrence never ran.** `_decision_features` reset the state and took a single step, so
with a previous state of zero the update collapsed to one matrix multiply. The dynamics never
executed and the wiring had nothing to contribute. `settle()` now runs the recurrence with the
input held.

**2. The decision discarded the answer.** The read-out was the equal-weight mean of each pool's
200 neurons. Linear probe on frozen representations: 4 pool means **37.1%**, the brain's 128-dim
settled state **100%**. Pool size was not the cause (1 neuron 38.1%, 200 neurons 37.1%) and
neither was neuron selection (most-selective 34.0%); averaging discards *which* neurons fired.
Reading each pool with learned weights fixed it, and that is also what a real mushroom body
output neuron is.

## What is actually established

- The brain chooses: the answer is the argmax over its own four neuron populations.
- The brain learns: 117,800 plastic synapses on real connectome edges plus 800 read-out weights.
- It reaches **100% peak / 92.8% final** train accuracy on the 97 challenges, from 19.6%.
- The specific wiring does not matter; a graph does.

## Limits, stated plainly

- **Train accuracy on 97 examples, not a held-out set.** This measures fit, not generalisation.
- **One seed per arm.** Not replicated.
- **Learnable-by-construction.** All 97 challenges are presented every epoch, so a readout of
  this size can fit them. That is what "learns Spanish" means at 97 phrases, and it is
  memorisation of the phrase-to-answer mapping rather than language.
- **The pools are seeded groups of real neurons, not identified cell types.**
- **`shuffled` preserves topology and misaligns the input/output interfaces**, so its failure
  would be about interface alignment specifically. It did not fail, so the point is moot here.
- **No task with time in it has been run.** Every input is a static prompt, which is exactly the
  setting where a reservoir's memory cannot help and therefore cannot be shown to help.
