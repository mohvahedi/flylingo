"""Run colab/step1_build.py's accuracy_agreement LOCALLY, so the Colab run does not discover
its next import error after a GPU has been allocated.

The function has never executed: the first attempt died on a missing numpy import, the second on
names imported only inside the other function, and the Colab launch that would have run it died on
a shell-quoting error before the code started. Each would have been a wasted T4 slot.

This loads the Colab module with its paths rewritten to the local layout (the module expects the
Colab root, where brain/ sits directly under ROOT; locally it is nested one level deeper and the
graph cache lives beside it) and calls the function with one epoch. One epoch is deliberate: the
question here is whether the code RUNS and produces a comparable pair of scores, not what the
scores are.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

FLY = Path("D:/Projects/flylingo")
spec = importlib.util.spec_from_file_location("step1_build", FLY / "colab" / "step1_build.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# rewrite the module's paths for the local tree
mod.ROOT = FLY / "brain"          # contains the `brain` package and brain/curriculum
mod.GRAPH = FLY / "cache" / "malecns_v1"
mod.CACHE = FLY / "cache"

print(f"ROOT  -> {mod.ROOT}")
print(f"GRAPH -> {mod.GRAPH}   exists: {mod.GRAPH.exists()}")
print(f"curriculum exists: {(mod.ROOT / 'brain/curriculum/es-en.json').exists()}")
print()

res = mod.accuracy_agreement(epochs=3)
print()
print("returned:", res)
assert set(res) == {"intact", "random_graph"}, res
for mode, row in res.items():
    assert {"cpu", "gpu", "abs_diff"} <= set(row), (mode, row)
    assert 0.0 <= row["cpu"] <= 1.0 and 0.0 <= row["gpu"] <= 1.0, (mode, row)
print("OK: accuracy_agreement runs and returns comparable scores per mode")
