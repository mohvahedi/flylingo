"""Generate colab/FlyLingo.ipynb: a notebook a stranger can run end to end.

The handoff's §4 asks for a public reproducible notebook, and notes the blocker: the parent repo has
no git remote, so a Colab session cannot clone it. This embeds the already-tested files inside the
notebook itself (base64, digest-checked as they are written), so the notebook is self-contained. It
also solves the other half of the problem: the data needs no upload, because the source feathers are
in a public GCS bucket.

The notebook does NOT reimplement anything. It writes out payload.py, step1_build.py and
step2_measure.py and runs them, so the numbers it reports come from the same code the project is
tested against -- one implementation, not a notebook copy that can drift.

Run:  python tools/gen_colab_notebook.py
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLY = HERE.parent
FILES = {
    "payload.py": FLY / "tools" / "payload.py",
    "step1_build.py": FLY / "colab" / "step1_build.py",
    "step2_measure.py": FLY / "colab" / "step2_measure.py",
}
OUT = FLY / "colab" / "FlyLingo.ipynb"


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.strip("\n").splitlines(keepends=True)}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def main() -> None:
    blobs, digests = {}, {}
    for name, path in FILES.items():
        raw = path.read_bytes()
        blobs[name] = base64.b64encode(raw).decode("ascii")
        digests[name] = hashlib.sha256(raw).hexdigest()
        print(f"  {name}: {len(raw)} bytes  {digests[name][:16]}...")

    cells = [
        md("""
# FlyLingo: does a real fly brain's wiring matter, on a task it can learn?

This notebook runs one controlled experiment. A fruit-fly connectome — the
[Janelia/Google MaleCNS v1.0](https://www.janelia.org/project-team/flyem), 166,700 neurons and
25,582,938 directed edges — is used as a reservoir that answers multiple-choice Spanish questions,
and the answer is read out of the activity of its own neurons. Four versions of the brain are
trained identically and compared:

| arm | what the wiring is |
|---|---|
| `intact` | the measured connectome |
| `shuffled` | the same graph, nodes relabelled (topology kept, interfaces moved) |
| `random_graph` | a degree-matched random matrix with the same number of edges |
| `no_edges` | disconnected — the control that must fail |

**What this establishes:** whether the specific measured wiring gives any advantage over a
same-sized graph that is not the fly's.

**What it does NOT establish, stated here rather than in a footnote:**

- This is **memorisation of a 97-item phrase-to-answer mapping**, not language. Every phrase is
  shown every epoch. Nothing here generalises to unseen sentences.
- The connectome is used as a **reservoir**, not as a simulation of a fly. No claim is made that
  the fly's brain "speaks Spanish", or that these dynamics resemble anything biological.
- The answer pools are **disjoint groups of real neurons chosen by a seeded shuffle**. They are not
  identified cell types and are not claimed to be.

**Runtime:** on a free Colab GPU, roughly 25 minutes at the default budget below.
"""),

        md("## 1. Environment\n\nColab already ships `cupy`, `pyarrow` and `scipy`. This checks the GPU and installs only what is missing."),

        code("""
import subprocess, sys, json, hashlib, base64, os, time
from pathlib import Path

def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

need = []
for mod, pkg in (("pyarrow", "pyarrow"), ("scipy", "scipy")):
    try:
        __import__(mod)
    except ImportError:
        need.append(pkg)
try:
    import cupy
    print("cupy", cupy.__version__)
except Exception:
    need.append("cupy-cuda12x")
if need:
    sh([sys.executable, "-m", "pip", "install", "-q", *need])

import cupy
props = cupy.cuda.runtime.getDeviceProperties(0)
print("GPU:", props["name"].decode())
print("All statements below are produced by the code in this notebook.")
"""),

        md("""
## 2. The code, embedded

This repository has no public remote, so the notebook cannot clone it. The three tested modules are
embedded as base64 and written to disk with their SHA-256 verified as they land, so the notebook is
self-contained and the files are provably the ones the project's own test suite covers.

They are **not** reimplemented here. `step1_build.py` builds the connectome and gates the
accelerator; `step2_measure.py` runs the four-arm comparison.
"""),

        code(f"""
FILES = {{
{chr(10).join(f"    {n!r}: {b!r}," for n, b in blobs.items())}
}}

DIGESTS = {digests!r}
CONTENT = Path("/content")
ROOT = CONTENT / "flylingo"

for name, blob in FILES.items():
    raw = base64.b64decode(blob)
    got = hashlib.sha256(raw).hexdigest()
    assert got == DIGESTS[name], f"{{name}} decoded to {{got}}, expected {{DIGESTS[name]}}"
    (CONTENT / name).write_bytes(raw)
    print(f"wrote {{name}}: {{len(raw)}} bytes, sha256 verified")
"""),

        md("""
## 3. The data

The connectome is public. `step1_build.py` downloads two Feather files over HTTPS and refuses to
build anything unless they match their exact byte counts and SHA-256 — a truncated download would
otherwise produce a plausible-looking wrong graph.

The published counts are then asserted, and the build fails loudly if any disagrees:

- **166,700** neurons (annotated, non-glial)
- **25,582,938** directed edges between them
- **124,177,617** synaptic contacts summed over those edges

Nothing is uploaded, and Colab's disk is ephemeral, so this re-runs in about five minutes per
session. That is the cost of not needing an account or a dataset upload.
"""),

        code("""
t0 = time.time()
sys.argv = ["step1_build.py"]
exec(compile((CONTENT / "step1_build.py").read_text(), "step1_build.py", "exec"),
     {"__name__": "__main__"})
print(f"\\nstep 1 took {(time.time() - t0) / 60:.1f} min")
"""),

        md("""
### Why step 1 gates the accelerator before measuring anything

An accelerated path is a second implementation of the same quantity, so it has to be shown to agree
with the reference before its numbers mean anything. This project learned that the hard way: a
`cupyx` sparse matvec was canonicalising the uploaded control matrix **in place**, merging its
31,231 duplicate `(row, column)` pairs and renumbering every entry after each merge, while the
trainer wrote weights **by position**. The accelerator therefore trained a matrix the CPU never had,
which disagreed by 6.0e-01 on a state of scale 0.8 after only six plasticity steps — and agreed
perfectly on an untrained brain, so a naive test passed.

The gate checks three things per arm:

1. an untrained settle agrees with the CPU to a **stated** float32 tolerance,
2. the uploaded matrix keeps the CPU's exact layout (stored-entry count and index pointers), and
3. after training, the weights have landed on the **same entries** on both devices.

The remaining cross-device difference is reported, not gated: a training loop feeds each step's
summation-order difference back into the weights, so drift of order 1e-3 is expected and is not a
defect. Gating on it would fail a correct implementation.
"""),

        md("""
## 4. The measurement

Rules this follows, each of which the project got wrong at least once before:

- **An arm's score is the mean over the last N epochs, never one epoch.** Single-epoch accuracy on a
  converged run was measured swinging with a standard deviation of 7–8.5 points, with a within-run
  range up to 36 points.
- **Several seeds, compared paired per seed**, so a seed's difficulty cancels and only the mechanism
  under test differs.
- **Every arm saves its trained weights**, and the number it reports is recomputed by loading that
  checkpoint back through the same guards the service uses.
- **No pre-written verdict.** The summary prints the effect next to its own spread and refuses to
  claim when the effect does not clear it.
- **The read-out is part of the experiment.** An earlier design read the populations with an equal-weight
  mean and concluded the wiring mattered; that spread belonged to the read-out, and the claim was
  retracted. Here each pool is read with learned weights, which is also what a real output neuron does.

Default budget: **25 epochs, 3 seeds**. Raise the first two numbers for a tighter estimate; on a T4,
40 epochs x 5 seeds takes about 45 minutes.
"""),

        code("""
EPOCHS, SEEDS, TAIL = 25, 3, 10   # modest by default so this notebook is runnable end to end
os.environ["FLYLINGO_ROOT"] = str(ROOT)
sys.argv = ["step2_measure.py", str(EPOCHS), str(SEEDS), str(TAIL)]
exec(compile((CONTENT / "step2_measure.py").read_text(), "step2_measure.py", "exec"),
     {"__name__": "__main__"})
"""),

        md("""
## 5. How to read the result

The number that answers the question is **`intact − best control`**, read against its own spread and
the count of seeds where intact leads.

Two traps, both of which this project fell into:

- **Do not read a single epoch.** If the summary says the effect is within the spread, there is no
  detectable difference — that is the result, not a failure to find one.
- **`best control` is a `max()` over two noisy arms.** It is a useful summary, but it biases the
  comparison *against* `intact`, so "intact is 1.4 points behind the best control" is not evidence
  that the real wiring is worse. Read the two per-control differences (`− shuffled`, `− random_graph`)
  separately, each against its own spread.

Also worth checking yourself: the `no_edges` arm must sit at chance. If it does not, something is
wrong with the harness and no other number should be believed. Its state is exactly zero, so its
pools are all equal and its argmax is degenerate.
"""),

        md("""
## 6. What this notebook found when it was run

Colab T4, 40 epochs, 5 seeds, score = mean of the last 10 epochs:

| arm | mean | sd |
|---|---|---|
| **intact** | **87.7%** | 1.8% |
| shuffled | 88.2% | 2.3% |
| random_graph | 87.0% | 1.9% |
| no_edges | 21.6% | 0.0% |

Per-control differences: **−0.5%** (sd 2.1) against the shuffle, **+0.7%** (sd 1.7) against the
random graph. Intact led the best control on **0 of 5** seeds.

**The specific wiring does not matter on this task.** What does: a recurrent graph is required —
the edge-free control sits at 21.6%, below the 26.8% majority-class baseline, while every connected
arm reaches 87–88%.

A local CPU run at 12 epochs and 3 seeds (63.7% / 62.8% / 64.3% / 21.6%) ties the same way, and all
20 T4 checkpoints were re-loaded on a different machine and their accuracies recomputed from
scratch: 20/20 reproduced the reported number exactly.

This is a **null result, and it is the third time this project reached one** — twice before it was
retracted, once because the read-out was the ceiling and once because the accelerator was corrupting
the weights. Those reasons are fixed and tested now, which is why the null is reported rather than
explained away.
"""),

        md("""
## 7. License and citation

The connectome is the **MaleCNS v1.0** release from the FlyEM project at Janelia Research Campus
(Google/Janealia `flyem-male-cns` public bucket), and is used here unmodified apart from the
retention rule and row normalisation described above. Check its own license before redistributing
the data; this notebook only downloads it at runtime.
"""),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size} bytes, {len(cells)} cells)")


if __name__ == "__main__":
    main()
