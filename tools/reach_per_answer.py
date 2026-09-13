"""Per answer: did the fly complete a reach on the card it chose?

Reach runs are the wrong unit. A run can end early because the next answer legitimately arrived, and
a single-sample blip is usually the tail of a long reach or the fly starting up. What matters to a
viewer is simpler: for each answer, did the fly land on its card and hold its forelegs there?

So this segments the trace by CHOICE CHANGE (one segment per answer), and asks in each segment
whether a reach of at least --min seconds happened. The reach envelope is 3.5s (0.36 extend + 2.8
hold + 0.55 settle), so 2.5s means the forelegs went down and stayed for most of the hold.

Answers are also split by whether the choice was a repeat of the previous one, because that was a
separate suspected failure mode.

    python tools/reach_per_answer.py before.log [after.log ...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MIN_COMPLETE = 2.5


def parse(path: str):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*([\d.]+)\s+(\w+)\s+(-?\d)\s+([\d.]+)", line)
        if m:
            rows.append({"t": float(m.group(1)), "b": m.group(2), "c": int(m.group(3)),
                         "change": ("choice" in line and "->" in line)})
    return rows


def per_answer(path: str, label: str):
    rows = parse(path)
    # segment boundaries: every sample that reports a choice change, plus the start
    bounds = [r["t"] for r in rows if r["change"]]
    if not bounds:
        print(f"--- {label}: no choice changes found")
        return None
    segs = []
    prev_choice = rows[0]["c"]
    for start, end in zip(bounds, bounds[1:] + [rows[-1]["t"] + 0.5]):
        seg = [r for r in rows if start <= r["t"] < end]
        choice = seg[0]["c"] if seg else None
        # longest reach run inside this answer
        best, cur = 0.0, None
        for r in seg:
            if r["b"] == "reach":
                cur = cur or [r["t"], r["t"]]
                cur[1] = r["t"]
            elif cur:
                best = max(best, cur[1] - cur[0])
                cur = None
        if cur:
            best = max(best, cur[1] - cur[0])
        segs.append({"t": start, "c": choice, "reach": best,
                     "repeat": choice == prev_choice})
        prev_choice = choice

    complete = [s for s in segs if s["reach"] >= MIN_COMPLETE]
    short = [s for s in segs if s["reach"] < MIN_COMPLETE]
    repeats = [s for s in segs if s["repeat"]]
    rep_ok = [s for s in repeats if s["reach"] >= MIN_COMPLETE]
    print(f"--- {label}")
    print(f"    answers {len(segs)}   completed a reach (>={MIN_COMPLETE}s) {len(complete)}   "
          f"short {len(short)}  ({len(complete) / len(segs):.0%} complete)")
    if repeats:
        print(f"    of which repeats of the previous card: {len(repeats)}, "
              f"completed {len(rep_ok)}")
    if short:
        print(f"    answers with a short reach: "
              f"{[(round(s['t'], 1), round(s['reach'], 1)) for s in short]}")
    return {"answers": len(segs), "complete": len(complete), "short": len(short)}


if __name__ == "__main__":
    out = []
    for p in sys.argv[1:]:
        out.append(per_answer(p, Path(p).name))
        print()
    if len(out) == 2 and all(out):
        print(f"answers completing a reach: {out[0]['complete']}/{out[0]['answers']} -> "
              f"{out[1]['complete']}/{out[1]['answers']}")
