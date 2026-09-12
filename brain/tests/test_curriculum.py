"""Validation tests for FlyLingo's Spanish curriculum.

These tests cover the data file, the pydantic models and the loader together,
because the service in brain/api.py consumes all three:

    brain/curriculum/es-en.json      the course content
    brain/curriculum/__init__.py     load_curriculum() plus the models
    scripts/build_curriculum.py      the generator that produced the JSON

Every challenge in the file is checked for schema validity, at least three
options, unique options, exactly one correct option, a correctIndex that agrees
with the answer, a difficulty inside 1 to 5, an allowed type, Spanish audio, and
a prompt tagged as English. Lessons must hold 6 to 8 challenges and the whole
course must hold 72 to 128.

The final test mutates a copy of the course on purpose, one defect at a time, and
requires every mutation to be caught. Without that test the checks above could
silently stop working and still report green.

No em dashes anywhere in this file.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
from pydantic import ValidationError

from brain.curriculum import (
    CHALLENGE_TYPES,
    CURRICULUM_PATH,
    Challenge,
    Curriculum,
    Lesson,
    Unit,
    load_curriculum,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "brain" / "curriculum" / "es-en.json"
GENERATOR_PATH = ROOT / "scripts" / "build_curriculum.py"

ALLOWED_TYPES = {"translate", "select", "match", "fill"}
REQUIRED_TOP_LEVEL = {"course", "from", "version", "units"}
REQUIRED_UNIT_KEYS = {"id", "title", "color", "order", "lessons"}
REQUIRED_LESSON_KEYS = {"id", "title", "order", "challenges"}
REQUIRED_CHALLENGE_KEYS = {
    "id",
    "type",
    "prompt",
    "promptLang",
    "answer",
    "options",
    "correctIndex",
    "audio",
    "difficulty",
}

LESSON_MIN, LESSON_MAX = 6, 8
TOTAL_MIN, TOTAL_MAX = 72, 128
UNIT_COUNT = 4
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
EM_DASH = "\u2014"

# Per-unit challenge totals the NOTES.md file documents. Counts are checked
# against these so the notes and the shipped data cannot drift apart.
EXPECTED_UNIT_TOTALS = {"u1": 21, "u2": 24, "u3": 28, "u4": 24}


class CheckFailure(AssertionError):
    """A curriculum rule was broken."""


def raw_payload() -> dict[str, Any]:
    """The course exactly as it sits on disk."""
    return json.loads(RAW_PATH.read_text(encoding="utf-8"))


def check_challenge(challenge: Any, where: str) -> None:
    """Every rule for one challenge, checked against a dict or a Challenge model."""

    def field(name: str) -> Any:
        if isinstance(challenge, dict):
            if name not in challenge:
                raise CheckFailure(f"{where}: challenge is missing the key {name!r}")
            return challenge[name]
        return getattr(challenge, name)

    for key in sorted(REQUIRED_CHALLENGE_KEYS):
        field(key)

    if field("type") not in ALLOWED_TYPES:
        raise CheckFailure(f"{where}: type {field('type')!r} is not one of {sorted(ALLOWED_TYPES)}")

    if field("promptLang") != "en":
        raise CheckFailure(f"{where}: promptLang must be 'en', got {field('promptLang')!r}")

    prompt = field("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise CheckFailure(f"{where}: prompt must be a non-empty string")
    if EM_DASH in prompt:
        raise CheckFailure(f"{where}: prompt contains an em dash")

    options = field("options")
    if not isinstance(options, list):
        raise CheckFailure(f"{where}: options must be a list, got {type(options).__name__}")
    if len(options) < 3:
        raise CheckFailure(f"{where}: needs at least 3 options, got {len(options)}")
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise CheckFailure(f"{where}: every option must be a non-empty string, got {options!r}")
    if len({option.strip().lower() for option in options}) != len(options):
        raise CheckFailure(f"{where}: options must be unique, got {options!r}")

    answer = field("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise CheckFailure(f"{where}: answer must be a non-empty string")
    if answer not in options:
        raise CheckFailure(f"{where}: answer {answer!r} is not among the options {options!r}")
    if sum(1 for option in options if option.strip().lower() == answer.strip().lower()) != 1:
        raise CheckFailure(f"{where}: exactly one option must equal the answer, got {options!r}")

    index = field("correctIndex")
    if not isinstance(index, int) or isinstance(index, bool):
        raise CheckFailure(f"{where}: correctIndex must be an int, got {index!r}")
    if not 0 <= index < len(options):
        raise CheckFailure(f"{where}: correctIndex {index} is out of range for {len(options)} options")
    if options[index] != answer:
        raise CheckFailure(
            f"{where}: correctIndex {index} points at {options[index]!r}, not at the answer {answer!r}"
        )

    difficulty = field("difficulty")
    if not isinstance(difficulty, int) or isinstance(difficulty, bool):
        raise CheckFailure(f"{where}: difficulty must be an int, got {difficulty!r}")
    if not 1 <= difficulty <= 5:
        raise CheckFailure(f"{where}: difficulty {difficulty} is outside 1 to 5")

    audio = field("audio")
    if not isinstance(audio, str) or not audio.strip():
        raise CheckFailure(f"{where}: audio must be the Spanish text to speak")
    if "___" in audio:
        raise CheckFailure(f"{where}: audio must not keep a blank placeholder")
    if EM_DASH in audio:
        raise CheckFailure(f"{where}: audio contains an em dash")

    challenge_id = field("id")
    if not isinstance(challenge_id, str) or not challenge_id.strip():
        raise CheckFailure(f"{where}: id must be a non-empty string")


def check_document(payload: Any) -> dict[str, Any]:
    """Every rule for the whole course. Returns stats when the course is valid."""
    if not isinstance(payload, dict):
        raise CheckFailure(f"payload must be a JSON object, got {type(payload).__name__}")
    missing = REQUIRED_TOP_LEVEL - set(payload)
    if missing:
        raise CheckFailure(f"payload is missing the keys {sorted(missing)}")

    if payload["course"] != "Spanish":
        raise CheckFailure(f"course must be 'Spanish', got {payload['course']!r}")
    if payload["from"] != "English":
        raise CheckFailure(f"from must be 'English', got {payload['from']!r}")
    if not isinstance(payload["version"], int) or isinstance(payload["version"], bool):
        raise CheckFailure(f"version must be an int, got {payload['version']!r}")

    units = payload["units"]
    if not isinstance(units, list) or len(units) != UNIT_COUNT:
        raise CheckFailure(f"expected {UNIT_COUNT} units, got {len(units) if isinstance(units, list) else units!r}")

    total = 0
    per_unit: dict[str, int] = {}
    lesson_count = 0
    challenge_ids: list[str] = []
    lesson_ids: list[str] = []

    for unit in units:
        unit_missing = REQUIRED_UNIT_KEYS - set(unit)
        if unit_missing:
            raise CheckFailure(f"unit {unit.get('id', '?')}: missing keys {sorted(unit_missing)}")
        if not HEX_COLOR.match(unit["color"]):
            raise CheckFailure(f"unit {unit['id']}: color {unit['color']!r} is not a hex colour like #58cc02")
        if not isinstance(unit["order"], int) or isinstance(unit["order"], bool) or unit["order"] < 1:
            raise CheckFailure(f"unit {unit['id']}: order must be a positive int, got {unit['order']!r}")
        if not unit["lessons"]:
            raise CheckFailure(f"unit {unit['id']}: has no lessons")

        unit_total = 0
        for lesson in unit["lessons"]:
            lesson_missing = REQUIRED_LESSON_KEYS - set(lesson)
            if lesson_missing:
                raise CheckFailure(f"lesson {lesson.get('id', '?')}: missing keys {sorted(lesson_missing)}")
            challenges = lesson["challenges"]
            if not LESSON_MIN <= len(challenges) <= LESSON_MAX:
                raise CheckFailure(
                    f"lesson {lesson['id']}: {len(challenges)} challenges, expected "
                    f"{LESSON_MIN} to {LESSON_MAX}"
                )
            for challenge in challenges:
                where = f"{lesson['id']}/{challenge.get('id', '?')}"
                check_challenge(challenge, where)
                if EM_DASH in challenge["prompt"]:
                    raise CheckFailure(f"{where}: prompt contains an em dash")
                challenge_ids.append(challenge["id"])
            if len(set(challenge_ids)) != len(challenge_ids):
                duplicate = next(cid for cid in challenge_ids if challenge_ids.count(cid) > 1)
                raise CheckFailure(f"lesson {lesson['id']}: duplicate challenge id {duplicate!r}")
            lesson_ids.append(lesson["id"])
            if len(set(lesson_ids)) != len(lesson_ids):
                raise CheckFailure(f"lesson {lesson['id']}: duplicate lesson id {lesson['id']!r}")
            lesson_count += 1
            unit_total += len(challenges)
            total += len(challenges)

        per_unit[unit["id"]] = unit_total

    if not TOTAL_MIN <= total <= TOTAL_MAX:
        raise CheckFailure(f"course has {total} challenges, expected {TOTAL_MIN} to {TOTAL_MAX}")

    # The models are the gate the service relies on, so run them too.
    curriculum = Curriculum.model_validate(payload)

    return {
        "total": total,
        "per_unit": per_unit,
        "lessons": lesson_count,
        "units": len(units),
        "curriculum": curriculum,
    }


# --------------------------------------------------------------------------- data


def test_payload_matches_its_generator() -> None:
    """The file on disk is exactly what scripts/build_curriculum.py rebuilds."""
    generator = load_generator()
    rebuilt = generator.build_payload()
    generator.validate(rebuilt)

    on_disk = raw_payload()
    assert rebuilt == on_disk, "es-en.json has drifted from scripts/build_curriculum.py"
    assert generator.render(rebuilt) == RAW_PATH.read_text(encoding="utf-8"), (
        "es-en.json formatting is not byte-identical to the generator output"
    )
    assert generator.main(["--check"]) == 0


def test_every_challenge_is_valid() -> None:
    """Full sweep of the shipped data, plus the models built from it."""
    stats = check_document(raw_payload())
    assert stats["units"] == UNIT_COUNT
    assert TOTAL_MIN <= stats["total"] <= TOTAL_MAX
    assert stats["per_unit"] == EXPECTED_UNIT_TOTALS, (
        f"per-unit counts changed: {stats['per_unit']} (NOTES.md documents {EXPECTED_UNIT_TOTALS})"
    )


def test_json_and_models_agree_challenge_for_challenge() -> None:
    """No challenge hides from the sweep: raw JSON and models line up."""
    payload = raw_payload()
    curriculum = Curriculum.model_validate(payload)

    raw_challenges = [
        challenge
        for unit in payload["units"]
        for lesson in unit["lessons"]
        for challenge in lesson["challenges"]
    ]
    model_challenges = curriculum.challenges()
    assert len(raw_challenges) == len(model_challenges) == 97

    for raw, model in zip(raw_challenges, model_challenges):
        assert raw["id"] == model.id
        assert raw["type"] == model.type
        assert raw["answer"] == model.answer
        assert raw["options"] == model.options
        assert raw["correctIndex"] == model.correctIndex
        assert raw["difficulty"] == model.difficulty
        assert raw["prompt"] == model.prompt
        assert raw["promptLang"] == model.promptLang
        assert raw["audio"] == model.audio
        check_challenge(model, f"{model.id}/model")


def test_content_uses_real_spanish() -> None:
    """Accents, inverted opening marks and no English leaking into the options."""
    payload = raw_payload()
    accents = {"á", "é", "í", "ó", "ú", "ñ", "¿", "¡"}

    non_ascii = 0
    accented_characters = 0
    inverted_marks = 0
    for unit in payload["units"]:
        for lesson in unit["lessons"]:
            for challenge in lesson["challenges"]:
                if accents & set(challenge["answer"] + challenge["audio"]):
                    non_ascii += 1
                accented_characters += sum(
                    1
                    for character in challenge["answer"] + "".join(challenge["options"]) + challenge["audio"]
                    if character in accents
                )
                if "¿" in challenge["answer"] or "¿" in challenge["prompt"]:
                    inverted_marks += 1
                assert challenge["promptLang"] == "en"

    # Stripping the accents out of the shipped Spanish would drop these to zero.
    assert non_ascii >= 15, f"only {non_ascii} challenges carry an accented answer"
    assert accented_characters >= 60, f"only {accented_characters} accented characters in the Spanish text"
    assert inverted_marks >= 3, "question prompts lost their inverted opening marks"

    # Unit 1 must stay beginner level: greetings and courtesy only.
    unit_one_answers = {
        challenge["answer"]
        for lesson in payload["units"][0]["lessons"]
        for challenge in lesson["challenges"]
    }
    assert "Hola" in unit_one_answers and "Buenos días" in unit_one_answers
    for advanced in ("estoy", "soy", "eres", "haber", "habría"):
        assert advanced not in {answer.lower() for answer in unit_one_answers}


# ------------------------------------------------------------------------- loader


def test_load_curriculum_returns_a_curriculum() -> None:
    curriculum = load_curriculum()
    assert isinstance(curriculum, Curriculum)
    assert (curriculum.course, curriculum.from_, curriculum.version) == ("Spanish", "English", 1)
    assert len(curriculum.units) == UNIT_COUNT
    assert all(isinstance(unit, Unit) for unit in curriculum.units)
    assert all(isinstance(lesson, Lesson) for unit in curriculum.units for lesson in unit.lessons)
    assert all(
        isinstance(challenge, Challenge)
        for unit in curriculum.units
        for lesson in unit.lessons
        for challenge in lesson.challenges
    )
    assert CURRICULUM_PATH == RAW_PATH


def test_load_curriculum_accepts_an_explicit_path(tmp_path: Path) -> None:
    copy_path = tmp_path / "es-en.json"
    copy_path.write_text(RAW_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    curriculum = load_curriculum(copy_path)
    assert isinstance(curriculum, Curriculum)
    assert curriculum.all_challenges().keys() == load_curriculum().all_challenges().keys()


def test_load_curriculum_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_curriculum(tmp_path / "nope.json")


def test_get_lesson() -> None:
    curriculum = load_curriculum()
    lesson = curriculum.get_lesson("u1l1")
    assert lesson is not None and isinstance(lesson, Lesson)
    assert lesson.id == "u1l1" and lesson.title == "Greetings"
    assert curriculum.get_lesson("does-not-exist") is None
    assert curriculum.get_lesson("") is None


def test_first_lesson_is_the_lowest_unit_then_lesson_order() -> None:
    curriculum = load_curriculum()
    first = curriculum.first_lesson()
    assert first.id == "u1l1"
    assert first == curriculum.get_lesson("u1l1")

    # File order must not matter: shuffle the units and lessons on disk and the
    # answer still comes from unit order then lesson order.
    payload = raw_payload()
    payload["units"] = list(reversed(payload["units"]))
    for unit in payload["units"]:
        unit["lessons"] = list(reversed(unit["lessons"]))
    curriculum = Curriculum.model_validate(payload)
    assert curriculum.first_lesson().id == "u1l1"
    assert list(curriculum.all_challenges())[0] == "u1l1"
    assert list(curriculum.all_challenges())[:3] == ["u1l1", "u1l2", "u1l3"]


def test_all_challenges_keyed_by_lesson() -> None:
    curriculum = load_curriculum()
    mapping = curriculum.all_challenges()
    assert isinstance(mapping, dict)

    lesson_ids = [lesson.id for lesson in curriculum.lessons()]
    assert list(mapping) == lesson_ids
    assert all(isinstance(challenges, list) for challenges in mapping.values())
    assert all(isinstance(challenge, Challenge) for group in mapping.values() for challenge in group)

    for lesson in curriculum.lessons():
        group = mapping[lesson.id]
        assert len(group) == len(lesson.challenges)
        assert [challenge.id for challenge in group] == [challenge.id for challenge in lesson.challenges]


def test_model_dump_is_serialisable_and_keeps_the_from_key() -> None:
    curriculum = load_curriculum()
    dumped = curriculum.model_dump()

    assert type(dumped) is dict
    assert dumped["from"] == "English", "model_dump() must emit the alias 'from'"
    assert "from_" not in dumped
    assert dumped["course"] == "Spanish"
    assert len(dumped["units"]) == UNIT_COUNT

    # Plain data only, so json can take it straight to the browser.
    assert json.loads(json.dumps(dumped, ensure_ascii=False)) == dumped
    assert json.loads(curriculum.model_dump_json()) == dumped

    first = dumped["units"][0]["lessons"][0]["challenges"][0]
    assert set(first) == REQUIRED_CHALLENGE_KEYS
    assert type(first["options"]) is list
    assert type(first["correctIndex"]) is int


def test_challenges_read_like_dicts_for_the_service() -> None:
    """brain/api.py reads the current challenge with index and .get() access."""
    curriculum = load_curriculum()
    challenge = curriculum.first_lesson().challenges[0]
    assert challenge["id"] == "u1l1c1"
    assert challenge["difficulty"] == 1
    assert challenge.get("difficulty", 1) == 1
    assert challenge.get("no_such_field", "fallback") == "fallback"
    assert "answer" in challenge
    assert "no_such_field" not in challenge
    with pytest.raises(KeyError):
        challenge["no_such_field"]


def test_loader_rejects_a_broken_file_on_disk(tmp_path: Path) -> None:
    payload = raw_payload()
    payload["units"][0]["lessons"][0]["challenges"][0]["correctIndex"] = 0
    payload["units"][0]["lessons"][0]["challenges"][0]["options"][0] = "definitely wrong"

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_curriculum(broken)


def test_challenge_types_are_the_four_the_service_ships() -> None:
    assert set(CHALLENGE_TYPES) == ALLOWED_TYPES
    curriculum = load_curriculum()
    seen = {challenge.type for challenge in curriculum.challenges()}
    assert seen == ALLOWED_TYPES, f"the course should use every challenge type, saw {sorted(seen)}"


def test_notes_document_the_counts_that_are_shipped() -> None:
    notes = (ROOT / "brain" / "curriculum" / "NOTES.md").read_text(encoding="utf-8")
    stats = check_document(raw_payload())
    assert f"{stats['total']} challenges" in notes
    for unit_id, unit_total in EXPECTED_UNIT_TOTALS.items():
        assert f"{unit_total}" in notes, f"NOTES.md does not mention {unit_id}'s {unit_total} challenges"


# --------------------------------------------------------------- mutation safety


def _mut_correct_index(payload: dict[str, Any]) -> None:
    challenge = payload["units"][0]["lessons"][0]["challenges"][0]
    challenge["correctIndex"] = (challenge["correctIndex"] + 1) % len(challenge["options"])


def _mut_duplicate_option(payload: dict[str, Any]) -> None:
    challenge = payload["units"][0]["lessons"][0]["challenges"][0]
    challenge["options"][1] = challenge["options"][0]


def _mut_too_few_options(payload: dict[str, Any]) -> None:
    challenge = payload["units"][0]["lessons"][0]["challenges"][0]
    challenge["options"] = challenge["options"][:2]


def _mut_answer_not_in_options(payload: dict[str, Any]) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["answer"] = "respuesta falsa"


def _mut_difficulty_high(payload: dict[str, Any]) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["difficulty"] = 6


def _mut_difficulty_zero(payload: dict[str, Any]) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["difficulty"] = 0


def _mut_bad_type(payload: dict[str, Any]) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["type"] = "listen"


def _mut_wrong_prompt_lang(payload: dict[str, Any]) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["promptLang"] = "es"


def _mut_empty_audio(payload: dict[str, Any]) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["audio"] = ""


def _mut_missing_key(payload: dict[str, Any]) -> None:
    del payload["units"][0]["lessons"][0]["challenges"][0]["audio"]


def _mut_lesson_too_short(payload: dict[str, Any]) -> None:
    lesson = payload["units"][0]["lessons"][0]
    lesson["challenges"] = lesson["challenges"][: LESSON_MIN - 1]


def _mut_lesson_too_long(payload: dict[str, Any]) -> None:
    lesson = payload["units"][0]["lessons"][0]
    while len(lesson["challenges"]) <= LESSON_MAX:
        lesson["challenges"].append(copy.deepcopy(lesson["challenges"][-1]))


def _mut_bad_colour(payload: dict[str, Any]) -> None:
    payload["units"][0]["color"] = "58cc02"


def _mut_duplicate_lesson_id(payload: dict[str, Any]) -> None:
    lessons = payload["units"][0]["lessons"]
    lessons[1]["id"] = lessons[0]["id"]


def _mut_duplicate_unit_id(payload: dict[str, Any]) -> None:
    payload["units"][1]["id"] = payload["units"][0]["id"]


def _mut_dropped_unit(payload: dict[str, Any]) -> None:
    payload["units"].pop()


def _mut_bloated_total(payload: dict[str, Any]) -> None:
    for unit in payload["units"]:
        extra = [copy.deepcopy(lesson) for lesson in unit["lessons"]]
        unit["lessons"].extend(extra)


def _mut_em_dash(payload: dict[str, Any]) -> None:
    payload["units"][0]["lessons"][0]["challenges"][0]["prompt"] += " \u2014 translated"


def _mut_wrong_course(payload: dict[str, Any]) -> None:
    payload["course"] = "French"


MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "correctIndex points at the wrong option": _mut_correct_index,
    "duplicate option": _mut_duplicate_option,
    "fewer than three options": _mut_too_few_options,
    "answer missing from the options": _mut_answer_not_in_options,
    "difficulty above 5": _mut_difficulty_high,
    "difficulty below 1": _mut_difficulty_zero,
    "type outside translate/select/match/fill": _mut_bad_type,
    "promptLang other than en": _mut_wrong_prompt_lang,
    "empty audio": _mut_empty_audio,
    "challenge missing a required key": _mut_missing_key,
    "lesson with too few challenges": _mut_lesson_too_short,
    "lesson with too many challenges": _mut_lesson_too_long,
    "unit colour that is not hex": _mut_bad_colour,
    "duplicate lesson id": _mut_duplicate_lesson_id,
    "duplicate unit id": _mut_duplicate_unit_id,
    "unit count other than four": _mut_dropped_unit,
    "total challenge count out of range": _mut_bloated_total,
    "em dash in a prompt": _mut_em_dash,
    "course other than Spanish": _mut_wrong_course,
}


def test_unmutated_course_passes_the_same_checker() -> None:
    """Baseline for the mutation test: the shipped data passes check_document."""
    stats = check_document(raw_payload())
    assert stats["total"] == 97


@pytest.mark.parametrize("description", sorted(MUTATIONS))
def test_deliberate_mutations_are_detected(description: str) -> None:
    """Each single-defect copy must be rejected, or this suite is not a test."""
    payload = raw_payload()
    MUTATIONS[description](payload)

    with pytest.raises((CheckFailure, ValidationError), match="."):
        check_document(payload)


def test_loader_rejects_every_mutated_course(tmp_path: Path) -> None:
    """The same mutations must also fail through load_curriculum on a real file."""
    for description, mutate in sorted(MUTATIONS.items()):
        payload = raw_payload()
        mutate(payload)
        target = tmp_path / "mutated.json"
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with pytest.raises((CheckFailure, ValidationError)):
            curriculum = load_curriculum(target)
            check_document(curriculum.model_dump())


def test_mutations_are_real_not_noops() -> None:
    """A mutation that changed nothing would make the test above vacuous."""
    original = raw_payload()
    for description, mutate in sorted(MUTATIONS.items()):
        payload = raw_payload()
        mutate(payload)
        assert payload != original, f"mutation does nothing: {description}"


# ------------------------------------------------------------------- generator


def load_generator() -> Any:
    """Import scripts/build_curriculum.py by path, since scripts is not a package."""
    cached = sys.modules.get("flylingo_build_curriculum")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location("flylingo_build_curriculum", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {GENERATOR_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generator_is_deterministic() -> None:
    generator = load_generator()
    first = generator.render(generator.build_payload())
    second = generator.render(generator.build_payload())
    assert first == second, "the generator must reproduce the same bytes every run"
