"""Fix three wording and formatting defects the review of the training panel turned up.

Each replacement asserts its anchor, so a silent no-op is impossible.
"""
from pathlib import Path

P = Path(r"D:\Projects\flylingo\duolingo-clone\components\flylingo\hud\training.tsx")
s = P.read_text(encoding="utf-8")


def sub(old: str, new: str) -> None:
    global s
    assert old in s, f"ANCHOR MISSING:\n{old[:200]}"
    s = s.replace(old, new, 1)


# 1. entropy rendered as "-0.00" for a tiny negative floating point value.
sub(
    """<Stat label="entropy" value={entropy.toFixed(2)} size={18} />""",
    """<Stat
          label="entropy"
          /* Clamped: a tiny negative value from floating point rendered as "-0.00". */
          value={Math.abs(entropy) < 0.005 ? "0.00" : entropy.toFixed(2)}
          size={18}
        />""",
)

# 2. "lessons done" next to "lesson 7 / 15" reads as a contradiction. It counts complete laps
#    of the course, which is a different quantity from the current position.
sub(
    """        <Stat
          label="lessons done"
          value={String(lessonsCompleted)}
          size={18}
          tone={lessonsCompleted > 0 ? "green" : "text"}
        />""",
    """        <Stat
          label="full laps"
          /* Not "lessons done", which beside "lesson 7 / 15" looked like a contradiction: this
             counts complete passes through the whole course, a different quantity. */
          value={String(lessonsCompleted)}
          size={18}
          tone={lessonsCompleted > 0 ? "green" : "text"}
        />""",
)

# 3. The trend metric. Comparing the first third against the last quarter reported "no clear
#    trend" straight across a climb from chance to the high nineties, because the fly's accuracy
#    legitimately dips when the course moves to harder material. Measuring the peak of the
#    recent window against the opening is the honest version of "has it learnt anything".
sub(
    """  const early = curve.length >= 6 ? curve.slice(0, Math.max(3, Math.floor(curve.length / 3))) : [];
  const late = curve.length >= 6 ? curve.slice(-Math.max(3, Math.floor(curve.length / 4))) : [];
  const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
  const gain = early.length && late.length ? mean(late) - mean(early) : 0;""",
    """  const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
  // Peak of the recent window against the opening, not last-point against first-third: accuracy
  // dips when the course moves on to harder material, and a strict end-to-end comparison then
  // reported "no clear trend" across a climb from chance to the high nineties.
  const early = curve.length >= 8 ? curve.slice(0, 8) : [];
  const late = curve.length >= 8 ? curve.slice(-10) : [];
  const gain = early.length && late.length ? Math.max(...late) - mean(early) : 0;""",
)

sub(
    """            <span style={{ color: gain > 0.05 ? HUD.green : HUD.faint }}>
              {gain > 0.05 ? `improving  ${gain > 0 ? "+" : ""}${(gain * 100).toFixed(0)} points` : "no clear trend yet"}
            </span>""",
    """            <span style={{ color: gain > 0.05 ? HUD.green : HUD.faint }}>
              {gain > 0.05
                ? `peak +${(gain * 100).toFixed(0)} points above its start`
                : "no gain yet"}
            </span>""",
)

P.write_text(s, encoding="utf-8")
print("training.tsx: entropy clamp, lap label, trend metric")
