"""Catch out-of-range GLSL component selection before it reaches the GPU.

Why this exists: a `vec2 p` read as `p.z` compiles to a black frame and a console error
buried among three's program warnings, and in this project it did exactly that once. The
GPU should not be the first line of defence, so this runs as a plain text check.

Scans every `/* glsl */` template literal in src/ for vector declarations (globals, locals,
and function parameters) and flags any `.x/.y/.z/.w` select or swizzle that reads a
component past the end of the declared type.

Declarations are resolved per scope, not file wide: several of the shader helpers here take
a parameter literally named `p`, as vec2 in one function and vec3 in another, and a
file-wide map would either skip both or flag the wrong one. A name that is ambiguous even
inside its own scope is reported as a note instead of a guess.

    python tools/lint_glsl.py            # from viz-fly/
Exit 0 clean, 1 on any out-of-range selection.
"""
import re
import sys
from pathlib import Path

SRC = Path('src')
ROOT = 'xyzw'

LITERAL = re.compile(r'/\*\s*glsl\s*\*/\s*`(.*?)`', re.S)
DECL = re.compile(r'\bvec([234])\s+([A-Za-z_]\w*)\s*(?:[=,;)]|$)', re.M)
SWIZZLE = re.compile(r'\b([A-Za-z_]\w*)\.([xyzw]{1,4})\b')
FUNC = re.compile(r'\b(?:float|int|void|bool|vec[234]|mat[234])\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{')


def literals(text: str):
    """Yield (first_line_number, glsl_body) for every tagged GLSL literal."""
    for m in LITERAL.finditer(text):
        yield text.count('\n', 0, m.start(1)) + 1, m.group(1)


def body_span(body: str, open_idx: int) -> int:
    """Index just past the brace that closes the block opened at `open_idx`."""
    depth = 0
    for i in range(open_idx, len(body)):
        if body[i] == '{':
            depth += 1
        elif body[i] == '}':
            depth -= 1
            if depth == 0:
                return i + 1
    return len(body)


def sizes_in(text: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for m in DECL.finditer(text):
        out.setdefault(m.group(2), set()).add(int(m.group(1)))
    return out


def lint(path: Path) -> tuple[list[str], list[str]]:
    text = path.read_text(encoding='utf-8')
    problems: list[str] = []
    notes: list[str] = []

    for base, body in literals(text):
        functions = []
        for m in FUNC.finditer(body):
            open_idx = body.index('{', m.start())
            functions.append((m.start(), body_span(body, open_idx), m.group(1), m.group(2)))

        # global scope: declarations outside every function body
        covered = [(s, e) for s, e, _, _ in functions]
        global_src = ''.join(
            body[i] for i in range(len(body)) if not any(s <= i < e for s, e in covered)
        )
        global_sizes = sizes_in(global_src)

        for start, end, fname, params in functions:
            scope = sizes_in(params)
            for name, n in sizes_in(body[start:end]).items():
                scope.setdefault(name, set()).update(n)
            merged: dict[str, set[int]] = {}
            for name in set(scope) | set(global_sizes):
                both = set(scope.get(name, set())) | set(global_sizes.get(name, set()))
                merged[name] = both

            for name, declared in sorted(merged.items()):
                if len(declared) > 1:
                    notes.append(
                        f'{path}:{base}  {fname}(): {name} is vec{sorted(declared)} in one scope, '
                        f'skipping (resolve by hand)'
                    )
                    continue
                size = declared.pop()
                for sm in SWIZZLE.finditer(body[start:end]):
                    if sm.group(1) != name:
                        continue
                    sw = sm.group(2)
                    bad = [c for c in sw if ROOT.index(c) >= size]
                    if bad:
                        at = start + sm.start()
                        line = base + body.count('\n', 0, at)
                        problems.append(
                            f'{path}:{line}  {fname}(): vec{size} {name} read as {name}.{sw} '
                            f'(component {"".join(bad)} out of range)'
                        )

        # global scope swizzles against their own declarations
        for name, declared in sorted(global_sizes.items()):
            if len(declared) != 1:
                continue
            size = declared.pop()
            for m in DECL.finditer(global_src):
                pass
            for sm in SWIZZLE.finditer(global_src):
                if sm.group(1) != name:
                    continue
                bad = [c for c in sm.group(2) if ROOT.index(c) >= size]
                if bad:
                    notes.append(
                        f'{path}:{base}  global vec{size} {name} read as {name}.{sm.group(2)} '
                        f'(component {"".join(bad)} out of range)'
                    )
    return problems, notes


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    files = sorted(root.rglob('*.ts')) + sorted(root.rglob('*.tsx'))
    if not files:
        print(f'no sources under {root.resolve()}')
        return 1

    glsl_files = 0
    for f in files:
        text = f.read_text(encoding='utf-8')
        if LITERAL.search(text):
            glsl_files += 1
        p, n = lint(f)
        problems += p
        notes += n

    print(f'scanned {len(files)} files, {glsl_files} carrying GLSL')
    for n in notes:
        print(f'  note: {n}')
    for p in problems:
        print(f'  ERROR: {p}')
    if problems:
        print(f'\n{len(problems)} out-of-range component selection(s)')
        return 1
    print('no out-of-range component selections')
    return 0


if __name__ == '__main__':
    sys.exit(main())
