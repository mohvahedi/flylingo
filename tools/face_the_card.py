"""Turn the fly to face the chosen card, and tighten its stand.

Measured before this change, from the per-tarsus telemetry:

    fore-L d=+5.118   mid-L d=+0.979   hind-L d=-1.054
    fore-R d=+4.924   mid-R d=+0.861   hind-R d=-1.110

The two legs the reach drives (indices 0 and 3, the forelegs) sit five units off the glass while
the middle and hind legs are effectively touching it. The fly has its BACK to the phone: the
model's own forward is opposite the direction of travel, so it flew in tail-first and then
reached into empty space behind it. `flyYawOffset` is the dial for exactly this and was left at 0.

Also: the orientation was never actually being held. It was computed from
(target - flightPos), and once the fly arrives that vector is zero, so the old code skipped the
whole block and the fly kept whatever yaw the flight left behind. It now looks at the card it
chose, which is well defined whether or not the fly has arrived.
"""
from pathlib import Path

P = Path(r"D:\Projects\flylingo\viz-fly\src\fly\PhoneStage.tsx")
s = P.read_text(encoding="utf-8")


def sub(old: str, new: str) -> None:
    global s
    assert old in s, f"ANCHOR MISSING:\n{old[:200]}"
    s = s.replace(old, new, 1)


sub(
    """  flyYawOffset: 0,""",
    """  /**
   * A half turn. Measured with this at 0: the forelegs sat 4.9-5.1 world units off the glass
   * while the middle and hind legs were one unit away, i.e. the specimen flew in tail-first and
   * reached into empty space behind it. Set from that measurement rather than from an assumption
   * about which axis the asset calls forward.
   */
  flyYawOffset: Math.PI,""",
)

sub(
    """  lateral: 4.0,""",
    """  lateral: 1.8,""",
)
sub(
    """   * In WORLD units. 4.0 clears the handset, whose screen is 3.8 wide and whose body is 8.7
   * tall, so the fly's body sits fully off the glass.""",
    """   * In WORLD units. The body stays clear of the glass while its forelegs stay inside reach of
   * the card: at 4.0 the legs would have had to span more than their own length even facing the
   * right way, and at 0 the body itself would sit over the lesson text.""",
)

# --- remember the card's true position, with no standing offset, for the fly to face
sub(
    """  const flightFrom = useRef(new THREE.Vector3(...LAYOUT.flyStart));""",
    """  /** the chosen card's world position with no standing offset: what the fly should face */
  const cardTrue = useRef<THREE.Vector3 | null>(null);
  const flightFrom = useRef(new THREE.Vector3(...LAYOUT.flyStart));""",
)

# fill it in the same frame pass that resolves the standing target
sub(
    """    const reach: ReachState =
      behavior === 'reach' ? reachCurve(anim.live.age) : ZERO_REACH;""",
    """    if (screen && flyChoice >= 0 && rects[flyChoice]) {
      const r = rects[flyChoice];
      screen.updateWorldMatrix(true, false);
      cardTrue.current = screenPointToPlaneLocal(r.cx, r.cy, SCREEN_W, SCREEN_H, 0)
        .applyMatrix4(screen.matrixWorld);
    } else {
      cardTrue.current = null;
    }

    const reach: ReachState =
      behavior === 'reach' ? reachCurve(anim.live.age) : ZERO_REACH;""",
)

# --- hold the orientation on the card
sub(
    """    scratch.subVectors(target.current, flightPos.current);
    if (scratch.lengthSq() > 1e-6) {
      const yaw = Math.atan2(scratch.x, scratch.z) + LAYOUT.flyYawOffset;""",
    """    // Look at the ANSWER, not along the flight line. Once the fly has arrived the flight
    // vector is zero, so the old code skipped this entirely and the fly simply kept whatever
    // yaw the flight left it with; there was no orientation being held at all.
    if (cardTrue.current) {
      scratch.subVectors(cardTrue.current, g.position);
    } else {
      scratch.subVectors(target.current, flightPos.current);
    }
    if (scratch.lengthSq() > 1e-6) {
      const yaw = Math.atan2(scratch.x, scratch.z) + LAYOUT.flyYawOffset;""",
)

P.write_text(s, encoding="utf-8")
print("fly faces the card; lateral tightened to 1.8")
