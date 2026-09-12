"""Read-only probe: encoder surface plus curriculum size. Prints a compact summary."""
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "brain"))

print("ROOT", ROOT)

import brain.encoders as E  # noqa: E402

print("encoders names:", [n for n in dir(E) if not n.startswith("_")])
for name in ("encoder_fingerprint", "encode_option_pairs", "encode_challenge"):
    fn = getattr(E, name, None)
    if fn is None:
        print(name, "MISSING")
        continue
    try:
        src = inspect.getsource(fn)
    except Exception as exc:  # pragma: no cover
        src = f"<no source: {exc}>"
    print("=" * 20, name, inspect.signature(fn))
    print(src)
    if name == "encoder_fingerprint":
        print("VALUE", fn())

print("EMBED_DIM", getattr(E, "EMBED_DIM", None))
print("constants", {k: getattr(E, k) for k in dir(E) if k.isupper()})

raw = json.loads((ROOT / "brain" / "curriculum" / "es-en.json").read_text(encoding="utf-8"))
chs = [c for u in raw.get("units", []) for l in u.get("lessons", []) for c in l.get("challenges", [])]
print("challenges", len(chs))
print("first", {k: chs[0][k] for k in chs[0]} if chs else None)
print("keys union", sorted({k for c in chs for k in c}))
