"""Semantic audit of the shipped Spanish course.

Covers what the schema cannot: that no option carries the meaning the prompt is
asking for (the one defensible answer rule), that every option is Spanish, that
correctIndex really points at the answer, and which option sets mix parts of
speech. Reads es-en.json and imports the generator's SKELETON table, so it
audits the generated file the way the course author sees it.

Run from D:/Projects/flylingo/brain:

    .venv/Scripts/python.exe scripts/audit_curriculum_content.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = ROOT / "brain" / "curriculum" / "es-en.json"
GENERATOR = ROOT / "scripts" / "build_curriculum.py"

ENGLISH_WORDS = {
    "a",
    "always",
    "am",
    "and",
    "are",
    "fine",
    "five",
    "four",
    "goodbye",
    "he",
    "hello",
    "i",
    "is",
    "my",
    "never",
    "no",
    "one",
    "please",
    "she",
    "six",
    "sorry",
    "thank",
    "thanks",
    "the",
    "three",
    "to",
    "two",
    "very",
    "yes",
    "you",
    "your",
}


def load_generator():
    spec = importlib.util.spec_from_file_location("build_curriculum", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lessons(gen):
    for unit in gen.SKELETON:
        for lesson in unit["lessons"]:
            yield lesson


def meaning_table(gen):
    """spanish text -> the English meanings the generator gives it."""
    out: dict[str, set] = {}
    for lesson in lessons(gen):
        for item in lesson["items"]:
            out.setdefault(item[0], set()).add(item[1].lower())
    for item in gen.EXTRA_POOL:
        out.setdefault(item[0], set()).add(item[1].lower())
    for lesson in lessons(gen):
        for kind in ("select", "match", "fill"):
            for challenge in lesson.get(kind, []):
                out.setdefault(challenge[1], set())
    return out


def pos_table(gen):
    """spanish text -> part of speech, or '?' for curated answers with no tag."""
    out: dict[str, str] = {}
    for lesson in lessons(gen):
        for item in lesson["items"]:
            out[item[0]] = item[2]
    for item in gen.EXTRA_POOL:
        out.setdefault(item[0], item[2])
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    gen = load_generator()
    meanings = meaning_table(gen)
    pos = pos_table(gen)
    payload = json.loads(CURRICULUM.read_text(encoding="utf-8"))

    total = 0
    collisions = []
    english_options = []
    index_problems = []
    mixed_pos = []
    for unit in payload["units"]:
        for lesson in unit["lessons"]:
            for challenge in lesson["challenges"]:
                total += 1
                options = challenge["options"]
                answer = challenge["answer"]
                answer_meanings = meanings.get(answer, set())
                for option in options:
                    if option == answer:
                        continue
                    shared = answer_meanings & meanings.get(option, set())
                    if shared:
                        collisions.append((challenge["id"], option, sorted(shared)))
                    if option.lower().strip() in ENGLISH_WORDS:
                        english_options.append((challenge["id"], option))
                if options[challenge["correctIndex"]] != answer:
                    index_problems.append((challenge["id"], "correctIndex points elsewhere"))
                if options.count(answer) != 1:
                    index_problems.append((challenge["id"], "answer not unique among options"))
                tags = {pos.get(option, "?") for option in options}
                if len(tags) > 1:
                    mixed_pos.append((challenge["id"], sorted(tags), options))

    def report(label, rows, show):
        print("%s: %d" % (label, len(rows)))
        for row in rows:
            print("   ", show(row))

    print("challenges audited: %d" % total)
    report(
        "options that carry the meaning the prompt asks for",
        collisions,
        lambda row: "%s -> %s shares %s" % row,
    )
    report(
        "options written in English",
        english_options,
        lambda row: "%s -> %s" % row,
    )
    report("correctIndex problems", index_problems, lambda row: "%s -> %s" % row)
    report(
        "option sets that mix parts of speech",
        mixed_pos,
        lambda row: "%s -> %s %s" % row,
    )
    return 1 if collisions or english_options or index_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
