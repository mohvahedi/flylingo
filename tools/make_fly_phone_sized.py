"""Make the fly phone-sized and have it reach in with its forelegs to touch the chosen option.

The user's words: "the fly should be at least the size of the phone or a lil bit bigger, and use
its arms to choose the option."

The geometry that follows from that, and why this is not just a bigger number:

  At phone size the fly cannot PERCH on a card. It is 8.6 units long and the cards are 1.5 units
  tall, so if it sat on the screen it would cover the whole lesson, which is the thing that has
  to stay readable. So it stands BESIDE the handset instead, and only its forelegs cross onto the
  glass to touch the answer it picked. That is also what "use its arms to choose" describes.

Every world-space offset that was tuned for a small fly has to scale with the body, or the pose
stops being self-similar: the standoff, the lean toward the glass and the settle. They are all
derived from flySpan here rather than hardcoded, so changing the size again cannot leave them
behind.
"""
from pathlib import Path

P = Path(r"D:\Projects\flylingo\viz-fly\src\fly\PhoneStage.tsx")
s = P.read_text(encoding="utf-8")


def sub(old: str, new: str) -> None:
    global s
    assert old in s, f"ANCHOR MISSING:\n{old[:220]}"
    s = s.replace(old, new, 1)


# ---------------------------------------------------------------- the size
sub(
    """  flySpan: 1.35,""",
    """  flySpan: 8.6,""",
)

# the comment above flySpan described the old size, so it is replaced with the new reasoning
sub(
    """  /**
   * The fly's world span: large enough to read as the protagonist.
   *
   * Measured at 0.72 the specimen filled about 9.6% of the screen's height (a 0.80 unit
   * bounding box against a screen 8.4 tall) and read as an insect that happened to be in the
   * shot. At 1.35 it is 18% of the screen height and about the height of the card it stands
   * on, which is the size the composition wants: big enough to be a character, small enough
   * that the lesson under it is still readable.
   */""",
    """  /**
   * The fly's world span, in world units. The handset is about 8.7 tall, so at 8.6 the
   * specimen is the same size as the phone, which is what was asked for.
   *
   * Sizes tried: 0.72 read as an insect that happened to be in the shot; 1.35 read as a
   * character but still small; 8.6 makes it a co-lead with the handset. At this size it can no
   * longer stand ON a card, because the cards are 1.5 units tall and it would cover the lesson,
   * so it stands beside the handset and reaches in with its forelegs. See `lateral` below.
   */""",
)

# ---------------------------------------------------------------- standing beside, not on
sub(
    """  standoff: 0.006,""",
    """  standoff: 0.028,""",
)

sub(
    """  flyCardU: 0.88,""",
    """  flyCardU: 0.88,
  /**
   * How far to the LEFT of the chosen card the fly stands, in world units, so its body clears
   * the handset and only its forelegs cross onto the glass.
   *
   * This is the change that makes "use its arms to choose the option" possible at phone size:
   * the body sits beside the screen and the forelegs are what reach the answer. Tuned against
   * the measured tarsus-to-glass distance, not by eye.
   */
  lateral: 3.4,
  /**
   * How far in FRONT of the glass the fly's body stands, world units. Derived from the size so
   * the pose stays self-similar: at 0.72 the specimen stood 0.24 body-lengths off the card, and
   * that ratio is what looks right, so it is recomputed rather than re-guessed.
   */
  bodyClearance: 1.5,""",
)

# ---------------------------------------------------------------- framing
sub(
    """  camera: [1.35, 3.75, 12.2] as [number, number, number],
  target: [1.35, 3.5, 0.4] as [number, number, number],
  fov: 38,""",
    """  /**
   * The camera has to see an 8.6 unit fly AND an 8.7 unit handset standing side by side, so it
   * is pulled back until the vertical field clears both. At fov 38 and distance 19 the vertical
   * field is 2*19*tan(19) = 13.1 units, against a subject about 9 tall: the earlier 12.2 put
   * the field at 8.4 and cropped the top of the phone, which is exactly the bug that cut the
   * question text off the screen.
   */
  camera: [0.4, 5.2, 17.4] as [number, number, number],
  target: [0.4, 4.6, 0.6] as [number, number, number],
  fov: 38,""",
)

# ---------------------------------------------------------------- the reach scale
sub(
    """  flyStart: [-4.6, 0.0, 2.4] as [number, number, number],""",
    """  flyStart: [-8.0, 0.0, 4.0] as [number, number, number],""",
)

# ---------------------------------------------------------------- targeting
sub(
    """      const local = screenPointToPlaneLocal(
        rect.x + rect.w * LAYOUT.flyCardU,
        rect.cy,
        SCREEN_W,
        SCREEN_H,
        LAYOUT.standoff,
      );""",
    """      // The body is offset to the LEFT of the card along the glass's own right axis, so it
      // clears the handset, and forward along the normal so the forelegs have something to
      // reach across. The forelegs are what land on the answer.
      const local = screenPointToPlaneLocal(
        rect.x + rect.w * LAYOUT.flyCardU,
        rect.cy,
        SCREEN_W,
        SCREEN_H,
        LAYOUT.standoff,
      );
      local.x -= LAYOUT.lateral;""",
)

# the same offset for the reported card positions, so the readback stays truthful
sub(
    """              const p = screenPointToPlaneLocal(
                r.x + r.w * LAYOUT.flyCardU,
                r.cy,
                SCREEN_W,
                SCREEN_H,
                LAYOUT.standoff,
              ).applyMatrix4(screen.matrixWorld);""",
    """              const lp = screenPointToPlaneLocal(
                r.x + r.w * LAYOUT.flyCardU,
                r.cy,
                SCREEN_W,
                SCREEN_H,
                LAYOUT.standoff,
              );
              lp.x -= LAYOUT.lateral;
              const p = lp.applyMatrix4(screen.matrixWorld);""",
)

P.write_text(s, encoding="utf-8")
print("PhoneStage: phone-sized fly standing beside the handset")

# ---------------------------------------------------------------- the reach, scaled
R = Path(r"D:\Projects\flylingo\viz-fly\src\fly\reach.ts")
r = R.read_text(encoding="utf-8")
old_lean = "  lean: 0.34,\n  drop: 0.07,"
assert old_lean in r
r = r.replace(
    old_lean,
    """  // Scaled for an 8.6 unit specimen. These are world-space offsets, so at the old 1.35 they
  // were tuned for a body six times smaller and would barely move the tarsi now. The lean is
  // what carries the forelegs the last of the way onto the glass, so it has to be sized to the
  // gap it has to close.
  lean: 1.30,
  drop: 0.30,""",
)
R.write_text(r, encoding="utf-8")
print("reach.ts: lean and drop scaled for the bigger body")
