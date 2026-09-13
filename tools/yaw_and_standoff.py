"""Set the yaw from measurement, and stand the fly back so its fore tarsi land ON the glass.

The sweep over four cardinal yaws, reading the scene's own per-tarsus telemetry with the fly
parked on card 2:

    yaw   0 deg  fore tarsi off glass 1.656   at px (430,349) (358,178)   <- forelegs point at the phone
    yaw  90 deg  fore tarsi off glass 0.661   at px (-176,321) (-271,139) <- forelegs point away
    yaw 180 deg  fore tarsi off glass 4.543   at px (-158,321) (-83,138)
    yaw 270 deg  fore tarsi off glass 3.588   at px (466,345) (567,174)

The smallest distance is not the right answer: at 90 degrees the tarsi are near the glass only
because they hang beside the handset and their screen x is NEGATIVE, off the left edge. Only at
0 degrees do the fore tarsi sit at the card's own horizontal position, so that is the offset that
has the fly facing its answer.

The residual is then depth, not angle. At 0 degrees the two fore tarsi measure -1.66 and -2.53
on the glass normal: past the plane, inside the handset. Tarsus positions move rigidly with the
body, so standing the fly back by their mean 2.1 world units puts them on the surface.
"""
from pathlib import Path

P = Path(r"D:\Projects\flylingo\viz-fly\src\fly\PhoneStage.tsx")
s = P.read_text(encoding="utf-8")


def sub(old: str, new: str) -> None:
    global s
    assert old in s, f"ANCHOR MISSING:\n{old[:200]}"
    s = s.replace(old, new, 1)


# the yaw default: the measured winner, not the half turn it was guessed to be
sub(
    """    const v = q === null ? NaN : Number(q);
    return Number.isFinite(v) ? v : Math.PI;""",
    """    const v = q === null ? NaN : Number(q);
    return Number.isFinite(v) ? v : 0;""",
)

# 0.067 mesh units x ~54.25 = 3.64 world, i.e. the measured 1.5 plus the 2.1 the tarsi overshot
sub(
    """  standoff: 0.028,""",
    """  /**
   * Perpendicular offset off the screen plane, in the plane's own units, so it is multiplied by
   * the phone group's scale.
   *
   * 3.64 world units. Raised from 1.5 because the fore tarsi were landing through the glass
   * rather than on it: measured at 1.5, with the fly correctly facing its card, they read -1.66
   * and -2.53 on the glass normal, i.e. inside the handset. They move rigidly with the body, so
   * standing the fly back by their mean 2.1 puts them on the surface.
   */
  standoff: 0.067,""",
)

P.write_text(s, encoding="utf-8")
print("yaw default set to the measured 0; standoff raised to 3.64 world")
