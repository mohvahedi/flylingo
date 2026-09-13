"""Let the yaw offset be set from the URL, so it can be SWEPT instead of guessed.

The fly's built-in correction rotation (ORIENTATION in RealFly) means the asset's own forward
axis is not something to reason about from the code: it has to be measured. Rather than edit and
reload four times, the offset is read once from `?yaw=` so a single script can test all four
cardinal options against the real per-tarsus telemetry and keep the winner.
"""
from pathlib import Path

P = Path(r"D:\Projects\flylingo\viz-fly\src\fly\PhoneStage.tsx")
s = P.read_text(encoding="utf-8")

old = """  flyYawOffset: Math.PI,"""
assert old in s, "anchor missing"
new = """  flyYawOffset: (() => {
    // Overridable from the URL as ?yaw=<radians> so the correct value can be swept against the
    // real tarsus telemetry rather than guessed. The fly asset carries its own correction
    // rotation, so which way it actually faces is a measurement, not a reading of the code.
    const q =
      typeof window === "undefined"
        ? null
        : new URLSearchParams(window.location.search).get("yaw");
    const v = q === null ? NaN : Number(q);
    return Number.isFinite(v) ? v : Math.PI;
  })(),"""
s = s.replace(old, new, 1)
P.write_text(s, encoding="utf-8")
print("yaw offset is now sweepable via ?yaw=")
