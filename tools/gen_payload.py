"""Generate the self-contained payload for the Colab run.

The parent repo has no git remote, so a Colab VM cannot clone it. This embeds the exact
bytes of the modules the run needs (plus the curriculum) as base64 inside one Python
file, which the notebook writes back out to disk before importing them. Same code, same
encoder, therefore the same checkpoint fingerprints as anything trained locally.

Run:  python gen_payload.py
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

BRAIN = Path(__file__).resolve().parents[1] / "brain"
OUT = Path(__file__).resolve().parent / "payload.py"

FILES = {
    "brain/__init__.py": BRAIN / "brain" / "__init__.py",
    "brain/reservoir.py": BRAIN / "brain" / "reservoir.py",
    "brain/plastic_brain.py": BRAIN / "brain" / "plastic_brain.py",
    "brain/encoders.py": BRAIN / "brain" / "encoders.py",
    "brain/curriculum/es-en.json": BRAIN / "brain" / "curriculum" / "es-en.json",
}


def main() -> None:
    blobs, digests = {}, {}
    for rel, path in FILES.items():
        raw = path.read_bytes()
        blobs[rel] = base64.b64encode(raw).decode("ascii")
        digests[rel] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        print(f"  {rel}: {len(raw)} bytes  {digests[rel]['sha256'][:16]}...")

    body = "\n".join(
        f'    {rel!r}: {blob!r},' for rel, blob in blobs.items()
    )
    text = f'''"""Embedded FlyLingo modules for the Colab run. GENERATED -- do not edit by hand.

Regenerate with tools/gen_payload.py in the FlyLingo repo. Content digests:

{json.dumps(digests, indent=2)}
"""
from __future__ import annotations

import base64
from pathlib import Path

FILES = {{
{body}
}}

DIGESTS = {digests!r}


def write_all(target: str | Path = "/content/flylingo") -> Path:
    """Materialise every embedded file under ``target`` and return the root."""
    import hashlib

    root = Path(target)
    for rel, blob in FILES.items():
        raw = base64.b64decode(blob)
        got = hashlib.sha256(raw).hexdigest()
        want = DIGESTS[rel]["sha256"]
        if got != want:
            raise RuntimeError(f"{{rel}} decoded to sha256 {{got}}, expected {{want}}")
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return root
'''
    OUT.write_text(text, encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
