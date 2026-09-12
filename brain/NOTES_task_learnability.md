# Task learnability: why the first null was not about the fly

`scripts/diagnose_task_learnability.py`, run against the real 97-challenge curriculum with
no reservoir involved, so it measures the ceiling for ANY readout on top of the encoding.

## Measured

| Measurement | Result | Chance |
|---|---|---|
| 5-fold cross-validated, encoding only, pooled | 0.2784 (27/97) | 0.250 |
| Exact binomial p against chance | 0.294, not significant | |
| Memorisation, shared (prompt, option) pairs | 0.454 | |
| Nearest neighbour over training correct pairs | 0.208 (5/24) | 0.250 |
| Per-fold CV | 0.200, 0.350, 0.263, 0.316, 0.263 | |

## Conclusion

The encoding carries no reliable signal for challenges whose exact wording or vocabulary
was absent from training. A character n-gram bag cannot relate "Hello" to "Hola" without
having seen that exact pairing, because the Spanish and English strings share no
informative structure.

So no readout on these features can beat chance on a vocabulary-disjoint split, whatever
the readout is. The recorded null (intact 0.208, parameter-matched 0.167, shuffled 0.250)
is therefore **not evidence about the connectome**. The fly was given an unsolvable task.

That cut both ways during the work: the earlier supervised probe reaching 0.333 on 24 items
looked like a promising signal, but 8/24 is not significant, and it was measured in a
different encoding pipeline that was also broken at the time (see the process-salted hash
bug). The 97-item cross-validated figure of 0.2784 is the number to quote.

## What was changed as a result

The evaluation was reframed from vocabulary-disjoint generalisation to
**learn-to-criterion with repetitions**, which is what "learns Spanish" means at this
scale: drilling the same 97 phrases until the mapping is known.

- All 97 challenges are presented every epoch, so the task is solvable by a readout.
- The primary measure becomes the train-accuracy learning curve per arm.
- The vocabulary-disjoint number is still reported, with the 0.2784 ceiling cited, so it
  cannot be mistaken for a failure of the connectome.
- The comparison that matters is now intact vs parameter-matched vs shuffled **on the
  learn-to-criterion curve**. A task that can be learned is the precondition for that
  comparison meaning anything.

## The rule this leaves behind

Before comparing an architecture's learning against a control, establish that the task is
learnable at all by measuring the ceiling with the readout removed. Otherwise a null
measures the task, not the architecture, and reporting it as a finding about the
architecture is a category error.
