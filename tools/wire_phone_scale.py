"""Pass the phone's measured scale into FlyActor so the lateral offset uses the real scale.

PHONE_SCALE no longer exists as a module constant: the scale is measured from the loaded asset
at runtime (the node chain rotates the model, so which axis ends up vertical cannot be assumed).
The lateral offset needs that same number, so it is threaded through as a prop rather than
re-derived, which also means it can never disagree with the transform the plane actually has.
"""
from pathlib import Path

P = Path(r"D:\Projects\flylingo\viz-fly\src\fly\PhoneStage.tsx")
s = P.read_text(encoding="utf-8")


def sub(old: str, new: str) -> None:
    global s
    assert old in s, f"ANCHOR MISSING:\n{old[:220]}"
    s = s.replace(old, new, 1)


# 1. accept it as a prop
sub(
    """  screenRef,
  targetOut,
}: {""",
    """  screenRef,
  targetOut,
  phoneScale,
}: {""",
)
sub(
    """  timeScale: number;
  screenRef: React.RefObject<THREE.Mesh | null>;
  targetOut: React.RefObject<THREE.Vector3 | null>;
}) {""",
    """  timeScale: number;
  screenRef: React.RefObject<THREE.Mesh | null>;
  targetOut: React.RefObject<THREE.Vector3 | null>;
  /** the phone group's measured scale, so plane-local offsets can be authored in world units */
  phoneScale: number;
}) {""",
)

# 2. use it
sub("      local.x -= LAYOUT.lateral / PHONE_SCALE;",
    "      local.x -= LAYOUT.lateral / phoneScale;")
sub("              lp.x -= LAYOUT.lateral / PHONE_SCALE;",
    "              lp.x -= LAYOUT.lateral / phoneScale;")

# 3. pass it at the call site, with the measured value and a sane fallback for the first frames
sub(
    """        screenRef={screenRef}
        targetOut={flyTargetRef}
      />""",
    """        screenRef={screenRef}
        targetOut={flyTargetRef}
        phoneScale={measured?.scale ?? PHONE_SCALE_FALLBACK}
      />""",
)

P.write_text(s, encoding="utf-8")
print("phoneScale threaded into FlyActor")
