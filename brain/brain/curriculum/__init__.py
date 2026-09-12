"""Spanish curriculum for FlyLingo: the pydantic models plus the loader the service calls.

Frozen interface (brain/api.py imports these exact names):

    load_curriculum(path=None) -> Curriculum
    Curriculum.get_lesson(lesson_id) -> Lesson | None
    Curriculum.first_lesson() -> Lesson
    Curriculum.all_challenges() -> dict[str, list[Challenge]]
    Curriculum.model_dump() -> plain JSON-ready data

The models are plain pydantic BaseModels, so model_dump() and model_dump_json()
behave the way the service expects, including the alias "from" at the top level.

They also answer to item access (challenge["id"], challenge.get("difficulty", 1)),
because brain/api.py reads the current challenge that way. That is deliberate, not
an accident: tests/test_curriculum.py pins it.

The loader validates, it does not repair. A file with two correct options, a
correctIndex that does not point at the answer, or a difficulty outside 1 to 5 is
rejected with a pydantic ValidationError naming the offending challenge.

No em dashes anywhere in this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CHALLENGE_TYPES: tuple[str, ...] = ("translate", "select", "match", "fill")
ChallengeType = Literal["translate", "select", "match", "fill"]

CURRICULUM_PATH: Path = Path(__file__).resolve().parent / "es-en.json"

__all__ = [
    "CHALLENGE_TYPES",
    "CURRICULUM_PATH",
    "Challenge",
    "ChallengeType",
    "Curriculum",
    "Lesson",
    "Unit",
    "load_curriculum",
]


class DictAccessMixin:
    """Read a model like a dict, which is how brain/api.py touches challenges."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in type(self).model_fields

    def keys(self) -> list[str]:
        return list(type(self).model_fields)


class Challenge(DictAccessMixin, BaseModel):
    """One exercise. Options are in fixed display order, audio is the Spanish to speak."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: ChallengeType
    prompt: str = Field(min_length=1)
    promptLang: Literal["en"] = "en"
    answer: str = Field(min_length=1)
    options: list[str] = Field(min_length=3)
    correctIndex: int = Field(ge=0)
    audio: str = Field(min_length=1)
    difficulty: int = Field(ge=1, le=5)

    @model_validator(mode="after")
    def _options_are_well_formed(self) -> "Challenge":
        cleaned = [option.strip() for option in self.options]
        if any(not option for option in cleaned):
            raise ValueError(f"{self.id}: every option must be a non-empty string")
        if len({option.lower() for option in cleaned}) != len(cleaned):
            raise ValueError(f"{self.id}: options must be unique, got {self.options!r}")
        if self.correctIndex >= len(self.options):
            raise ValueError(
                f"{self.id}: correctIndex {self.correctIndex} is out of range for "
                f"{len(self.options)} options"
            )
        if self.answer not in self.options:
            raise ValueError(f"{self.id}: answer {self.answer!r} is not among the options")
        if self.options[self.correctIndex] != self.answer:
            raise ValueError(
                f"{self.id}: correctIndex {self.correctIndex} points at "
                f"{self.options[self.correctIndex]!r}, not at the answer {self.answer!r}"
            )
        if not self.audio.strip() or "___" in self.audio:
            raise ValueError(f"{self.id}: audio must be the Spanish text to speak")
        return self


class Lesson(DictAccessMixin, BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    order: int = Field(ge=1)
    challenges: list[Challenge] = Field(min_length=1)


class Unit(DictAccessMixin, BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    order: int = Field(ge=1)
    lessons: list[Lesson] = Field(min_length=1)

    @model_validator(mode="after")
    def _lesson_orders_are_unique(self) -> "Unit":
        orders = [lesson.order for lesson in self.lessons]
        if len(set(orders)) != len(orders):
            raise ValueError(f"{self.id}: lesson orders must be unique, got {orders}")
        return self


class Curriculum(DictAccessMixin, BaseModel):
    """The whole course. Plain data, so model_dump() is what /curriculum returns."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    course: str = Field(min_length=1)
    from_: str = Field(alias="from", min_length=1)
    version: int = Field(ge=1)
    units: list[Unit] = Field(min_length=1)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialise with aliases, so the payload carries the key "from".

        by_alias is pinned here instead of in model_config because the config key
        moved between pydantic 2.9 and 2.11, while this override behaves the same
        on every 2.x release.
        """
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("by_alias", True)
        return super().model_dump_json(**kwargs)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> "Curriculum":
        unit_ids = [unit.id for unit in self.units]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError(f"unit ids must be unique, got {unit_ids}")
        unit_orders = [unit.order for unit in self.units]
        if len(set(unit_orders)) != len(unit_orders):
            raise ValueError(f"unit orders must be unique, got {unit_orders}")
        lesson_ids = [lesson.id for lesson in self.lessons()]
        if len(set(lesson_ids)) != len(lesson_ids):
            raise ValueError(f"lesson ids must be unique, got {lesson_ids}")
        challenge_ids = [challenge.id for challenge in self.challenges()]
        if len(set(challenge_ids)) != len(challenge_ids):
            raise ValueError("challenge ids must be unique")
        return self

    # ------------------------------------------------------------------ shape

    def units_sorted(self) -> list[Unit]:
        """Units in teaching order."""
        return sorted(self.units, key=lambda unit: unit.order)

    def lessons(self) -> list[Lesson]:
        """Every lesson, units in teaching order and lessons in teaching order."""
        return [
            lesson
            for unit in self.units_sorted()
            for lesson in sorted(unit.lessons, key=lambda item: item.order)
        ]

    def challenges(self) -> list[Challenge]:
        """Every challenge in teaching order."""
        return [challenge for lesson in self.lessons() for challenge in lesson.challenges]

    # ------------------------------------------------------- frozen service API

    def get_unit(self, unit_id: str) -> Unit | None:
        for unit in self.units:
            if unit.id == unit_id:
                return unit
        return None

    def get_lesson(self, lesson_id: str) -> Lesson | None:
        for lesson in self.lessons():
            if lesson.id == lesson_id:
                return lesson
        return None

    def first_lesson(self) -> Lesson:
        """The lowest unit order, then the lowest lesson order inside it."""
        ordered = self.lessons()
        if not ordered:
            raise ValueError("curriculum has no lessons")
        return ordered[0]

    def all_challenges(self) -> dict[str, list[Challenge]]:
        """Every lesson id mapped to its challenges, in teaching order."""
        return {lesson.id: list(lesson.challenges) for lesson in self.lessons()}

    def counts(self) -> dict[str, int]:
        """Per-unit challenge counts plus the total, for the notes and the tests."""
        per_unit = {unit.id: sum(len(lesson.challenges) for lesson in unit.lessons) for unit in self.units_sorted()}
        return {**per_unit, "total": sum(per_unit.values())}


def load_curriculum(path: str | Path | None = None) -> Curriculum:
    """Load and validate the Spanish course.

    path defaults to brain/curriculum/es-en.json next to this module.
    Raises FileNotFoundError when the file is missing and pydantic ValidationError
    when anything in it breaks the frozen schema.
    """
    resolved = Path(path) if path is not None else CURRICULUM_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"curriculum file not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return Curriculum.model_validate(payload)
