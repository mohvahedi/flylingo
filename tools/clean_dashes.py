"""Replace em dashes and en dashes in this project's own text, leaving third-party files alone.

The house style for anything published is plain punctuation: em dashes read as machine written to
this project's owner, so they are converted to commas, periods or parentheses rather than kept.

Scope, deliberately narrow:

  * parent repo tracked files only
  * NOT flm/, which is someone else's project (MIT, Alex Wormuth) kept locally for reference
  * NOT duolingo-clone/, except the paths this project actually authored (checked separately by
    tools/clean_dashes_app.py), because the rest of that tree is an upstream fork

Conversions, in order:

  1. a spaced em dash (U+2014), the appositive case   -> ", "
  2. an em dash opening a list continuation           -> removed
  3. en dash (U+2013) ranges like "6 to 9"            -> " to "
  4. any remaining em or en dash                      -> ", " / "-"

Reports what it changed and refuses to leave a file inconsistent: it re-counts afterwards and
prints the residue, so a silent miss is visible.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

EM = "\u2014"
EN = "\u2013"

#: binary or generated files where a rewrite would be wrong
SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm", ".glb", ".ttf", ".woff",
               ".woff2", ".zip", ".npz", ".npy", ".feather", ".ico", ".pdf", ".u8", ".u32"}

DRY = "--apply" not in sys.argv


def tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    files = []
    for line in out.splitlines():
        if line.startswith("flm/") or line.startswith("duolingo-clone/"):
            continue
        p = Path(line)
        if p.suffix.lower() in SKIP_SUFFIX or not p.is_file():
            continue
        files.append(p)
    return files


def convert(text: str) -> tuple[str, int]:
    """Return the rewritten text and how many dashes were replaced."""
    before = text.count(EM) + text.count(EN)

    # 1 and 2: the em dash is the main offender
    text = text.replace(f" {EM} ", ", ")
    text = re.sub(rf"^(\s*[-*]?\s*){EM}\s*", r"\1", text, flags=re.MULTILINE)
    # 3: en dash between digits or short words is a range
    text = re.sub(rf"(\d)\s*{EN}\s*(\d)", r"\1 to \2", text)
    # 4: anything left
    text = text.replace(f" {EN} ", ", ").replace(EM, ", ").replace(EN, "-")

    return text, before


def main() -> None:
    files = tracked()
    total = 0
    changed = []
    for p in files:
        try:
            original = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if EM not in original and EN not in original:
            continue
        new, n = convert(original)
        total += n
        changed.append((p, n))
        if not DRY:
            p.write_text(new, encoding="utf-8")

    mode = "WOULD CHANGE" if DRY else "changed"
    for p, n in sorted(changed, key=lambda x: -x[1]):
        print(f"  {n:>3} dashes  {mode}  {p}")
    print(f"\n{total} dashes across {len(changed)} files"
          + ("  (dry run; pass --apply to write)" if DRY else ""))

    if not DRY:
        residue = 0
        for p in files:
            try:
                t = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            residue += t.count(EM) + t.count(EN)
        print(f"residue in project files after the pass: {residue}")


if __name__ == "__main__":
    main()
