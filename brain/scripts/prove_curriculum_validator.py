"""Prove the curriculum validator rejects each deliberate violation.

Every mutation runs on an in-memory copy of what es-en.json currently holds;
nothing on disk is written. Two layers are exercised, in this order:

1. the pydantic models in brain/curriculum/__init__.py, which own the field and
   cross-field rules (index, uniqueness, list length, difficulty range, type
   literal, answer membership, id uniqueness);
2. check_document from tests/test_curriculum.py, which owns the document level
   rules (6 to 8 challenges per lesson, 72 to 128 total, hex colour, four units,
   no em dash).

Run from D:/Projects/flylingo/brain:

    .venv/Scripts/python.exe scripts/prove_curriculum_validator.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from pydantic import ValidationError  # noqa: E402

from brain.curriculum import CURRICULUM_PATH, Curriculum, load_curriculum  # noqa: E402
from test_curriculum import CheckFailure, check_document  # noqa: E402


def first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]
    where = ".".join(str(part) for part in err["loc"])
    return "%s: %s" % (where, err["msg"])


def mutate_correct_index(payload: dict) -> None:
    ch = payload["units"][0]["lessons"][0]["challenges"][0]
    ch["correctIndex"] = (ch["correctIndex"] + 1) % len(ch["options"])


def mutate_duplicate_option(payload: dict) -> None:
    ch = payload["units"][0]["lessons"][0]["challenges"][0]
    ch["options"][1] = ch["options"][0]


def mutate_two_options(payload: dict) -> None:
    ch = payload["units"][0]["lessons"][0]["challenges"][0]
    ch["options"] = ch["options"][:2]
    ch["correctIndex"] = 0


def mutate_difficulty_nine(payload: dict) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["difficulty"] = 9


def mutate_bogus_type(payload: dict) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["type"] = "telepathy"


def mutate_answer_not_in_options(payload: dict) -> None:
    ch = payload["units"][0]["lessons"][0]["challenges"][0]
    ch["options"][ch["correctIndex"]] = "definitely wrong"


def mutate_prompt_lang(payload: dict) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["promptLang"] = "es"


def mutate_empty_audio(payload: dict) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["audio"] = ""


def mutate_lesson_too_short(payload: dict) -> None:
    lesson = payload["units"][0]["lessons"][0]
    lesson["challenges"] = lesson["challenges"][:5]


def mutate_duplicate_lesson_id(payload: dict) -> None:
    lessons = payload["units"][0]["lessons"]
    lessons[1]["id"] = lessons[0]["id"]


def mutate_lesson_too_long(payload: dict) -> None:
    lesson = payload["units"][0]["lessons"][0]
    template = copy.deepcopy(lesson["challenges"][0])
    extra = len(lesson["challenges"]) + 5
    for number in range(len(lesson["challenges"]) + 1, extra + 1):
        clone = copy.deepcopy(template)
        clone["id"] = "%sc%d" % (lesson["id"], number)
        lesson["challenges"].append(clone)


def mutate_em_dash(payload: dict) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["prompt"] += " \u2014 translated"


MUTATIONS = [
    ("wrong correctIndex", mutate_correct_index),
    ("duplicate option", mutate_duplicate_option),
    ("only 2 options", mutate_two_options),
    ("difficulty 9", mutate_difficulty_nine),
    ("bogus type", mutate_bogus_type),
    ("answer not among options", mutate_answer_not_in_options),
    ("promptLang es", mutate_prompt_lang),
    ("empty audio", mutate_empty_audio),
    ("lesson too short", mutate_lesson_too_short),
    ("duplicate lesson id", mutate_duplicate_lesson_id),
    ("lesson too long", mutate_lesson_too_long),
    ("em dash in prompt", mutate_em_dash),
]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    payload = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))

    Curriculum.model_validate(copy.deepcopy(payload))
    live = load_curriculum()
    print(
        "baseline: unmutated payload validates; load_curriculum() -> %d units, %d challenges"
        % (len(live.units), len(list(live.challenges())))
    )
    print("")

    missed = 0
    for name, mutate in MUTATIONS:
        broken = copy.deepcopy(payload)
        mutate(broken)
        before = json.dumps(payload, sort_keys=True)
        if json.dumps(broken, sort_keys=True) == before:
            print("NO-OP   %-26s mutation changed nothing" % name)
            missed += 1
            continue
        try:
            Curriculum.model_validate(broken)
        except ValidationError as exc:
            print("CAUGHT  %-26s [pydantic] %s" % (name, first_error(exc)))
            continue
        try:
            check_document(broken)
        except CheckFailure as exc:
            print("CAUGHT  %-26s [check_document] %s" % (name, exc))
        else:
            print("MISSED  %-26s validator accepted the payload" % name)
            missed += 1

    print("")
    print("mutations: %d, not caught: %d" % (len(MUTATIONS), missed))
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
