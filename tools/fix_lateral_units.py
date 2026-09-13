"""Fix the lateral offset: it must be in the screen plane's OWN units, not world units.

The bug this corrects: the screen plane is a child of the phone group, which is scaled by
PHONE_SCALE (about 54x). So an offset of 3.4 added to the plane's local x became 3.4 * 54 = 184
world units, and the fly was flung to [-177, -0.4, -23], far outside the scene and invisible.
Measured from the live probe before the fix.

`standoff` was already in plane-local units and correct at 0.028 (= 1.5 world). The lateral
offset was authored in world units by mistake. It is now divided by the same scale the plane
inherits, so the two offsets are expressed in one consistent unit and cannot drift apart again.
"""
from pathlib import Path

P = Path(r"D:\Projects\flylingo\viz-fly\src\fly\PhoneStage.tsx")
s = P.read_text(encoding="utf-8")


def sub(old: str, new: str) -> None:
    global s
    assert old in s, f"ANCHOR MISSING:\n{old[:220]}"
    s = s.replace(old, new, 1)


# the constant, restated in the unit the code actually needs
sub(
    """  lateral: 3.4,""",
    """  lateral: 4.0,""",
)
sub(
    """   * This is the change that makes "use its arms to choose the option" possible at phone size:
   * the body sits beside the screen and the forelegs are what reach the answer. Tuned against
   * the measured tarsus-to-glass distance, not by eye.
   */""",
    """   * This is the change that makes "use its arms to choose the option" possible at phone size:
   * the body sits beside the screen and the forelegs are what reach the answer.
   *
   * In WORLD units. 4.0 clears the handset, whose screen is 3.8 wide and whose body is 8.7
   * tall, so the fly's body sits fully off the glass.
   *
   * It is divided by the plane's inherited scale at the point of use. That division is not
   * cosmetic: the screen plane is a child of the phone group, which is scaled by about 54, so
   * adding this straight to the plane's local x multiplied it by 54 and threw the fly to
   * [-177, -0.4, -23], completely out of the scene. Measured, not guessed.
   */""",
)

# both places that use it, in the plane's local units
sub(
    """      local.x -= LAYOUT.lateral;""",
    """      // plane-local units: divide by the scale the plane inherits, or this becomes 54x too big
      local.x -= LAYOUT.lateral / PHONE_SCALE;""",
)
sub(
    """              lp.x -= LAYOUT.lateral;""",
    """              lp.x -= LAYOUT.lateral / PHONE_SCALE;""",
)

P.write_text(s, encoding="utf-8")
print("lateral offset converted to plane-local units")
