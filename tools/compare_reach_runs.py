"""Compare two trace_choice.py logs: how many of the fly's reaches were cut short.

The report to fix: for some answers the fly does not put its forelegs on the card it chose. The
measurement that found it counts reach RUNS in the live demo and how long each lasted. The reach is
tuned for 3.5s (0.36 extend + 2.8 hold + 0.55 settle), so a run well under that is a reach that was
interrupted before the forelegs went down -- which is the gesture.

A reach shorter than 2.5s counts as truncated: that is a conservative line, well below the 3.5s
the envelope is built for and above the half-second sampling interval.

    python tools/compare_reach_runs.py before.log after.log
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FULL = 2.5  # seconds; a reach at least this long is treated as having completed


def analyse(path: str, label: str) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        m = re.match(r"\s*([\d.]+)\s+(\w+)\s+(\d)\s+([\d.]+)", line)
        if m:
            rows.append({"t": float(m.group(1)), "b": m.group(2),
                         "change": ("choice" in line and "->" in line),
                         "moved": "cards moved" in line})

    runs, cur = [], None
    for r in rows:
        if r["b"] == "reach":
            cur = cur or [r["t"], r["t"]]
            cur[1] = r["t"]
        elif cur:
            runs.append(tuple(cur))
            cur = None
    if cur:
        runs.append(tuple(cur))

    dur = [round(b - a, 1) for a, b in runs]
    full = [d for d in dur if d >= FULL]
    trunc = [d for d in dur if d < FULL]
    print(f"--- {label}")
    print(f"    samples {len(rows)}   choice changes {sum(1 for r in rows if r['change'])}"
          f"   card-layout shifts {sum(1 for r in rows if r['moved'])}")
    print(f"    reaches {len(dur)}   completing(>={FULL}s) {len(full)}   "
          f"TRUNCATED(<{FULL}s) {len(trunc)}")
    print(f"    durations {dur}")
    if dur:
        print(f"    truncated {len(trunc)}/{len(dur)} = {len(trunc) / len(dur):.0%}")
    return {"reaches": len(dur), "full": len(full), "truncated": len(trunc), "durations": dur}


if __name__ == "__main__":
    before, after = sys.argv[1], sys.argv[2]
    b = analyse(before, "BEFORE (old build)")
    print()
    a = analyse(after, "AFTER (fixed build)")
    print()
    if b["reaches"] and a["reaches"]:
        print(f"truncated reaches: {b['truncated']}/{b['reaches']} -> "
              f"{a['truncated']}/{a['reaches']}")
