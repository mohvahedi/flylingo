#!/usr/bin/env python
"""Reproducible builder for FlyLingo's Spanish course, brain/curriculum/es-en.json.

The course is generated, not hand-typed. SKELETON below holds, per lesson, the
vocabulary items plus curated select / match / fill challenges; the translate
challenges, their distractor sets and every option order are derived from that
table with a seed fixed per challenge id, so the file rebuilds byte for byte.

Usage, from D:/Projects/flylingo/brain:

    .venv/Scripts/python.exe scripts/build_curriculum.py            # write the json
    .venv/Scripts/python.exe scripts/build_curriculum.py --check    # fail if disk differs
    .venv/Scripts/python.exe scripts/build_curriculum.py --summary  # print the counts

--check is what tests/test_curriculum.py uses to pin reproducibility.

Distractor rules, enforced here rather than trusted to a reviewer:
  - a distractor is the same part of speech as the answer
  - a distractor never carries the English meaning of the prompt
  - distractors come from the same lesson first, then the same unit, then the rest
    of the course, so they stay plausible instead of random
  - option sets are unique and hold exactly one defensible answer

Peninsular-neutral Spanish throughout, with accents and inverted opening marks.
No em dashes anywhere in this file or in the content it writes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "brain" / "curriculum" / "es-en.json"

COURSE = "Spanish"
COURSE_FROM = "English"
VERSION = 1

# Deterministic option order: the seed for a challenge is derived from its id, so
# adding a lesson cannot reshuffle the challenges that already existed.
SEED_NAMESPACE = 0x464C594C  # "FLYL"


def _curriculum_module():
    """Load brain/curriculum/__init__.py by path.

    Importing it as "brain.curriculum" would execute brain/__init__.py and pull the
    service package in with it; the generator only wants the models.
    """
    spec = importlib.util.spec_from_file_location(
        "_flylingo_curriculum_models", ROOT / "brain" / "curriculum" / "__init__.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("cannot load brain/curriculum/__init__.py")
    module = importlib.util.module_from_spec(spec)
    # pydantic resolves the string annotations ("Curriculum") through
    # sys.modules[name], so the module has to be registered before it executes.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


# ---------------------------------------------------------------------- content

# (spanish, english, part of speech). Part of speech is the distractor key.
Item = tuple[str, str, str]

# Vocabulary kept only to enrich distractor pools, so a lesson that runs short of
# same-class peers still gets plausible wrong answers rather than random words.
EXTRA_POOL: list[Item] = [
    ("Hasta pronto", "See you soon", "phrase"),
    ("Bienvenido", "Welcome", "phrase"),
    ("Buenas tardes", "Good afternoon", "phrase"),
    ("Muy", "Very", "adv"),
    ("También", "Also", "adv"),
    ("Siempre", "Always", "adv"),
    ("Nunca", "Never", "adv"),
    ("Muy bien", "Very well", "adv"),
    # Infinitives stay separate from conjugated forms so a prompt such as
    # "to eat" never gets "eres" as a distractor.
    # "tomar" is left out on purpose: it also means to have a drink, so for
    # the prompt "to drink" it would be a defensible answer, not a wrong one.
    ("cocinar", "to cook", "verb_inf"),
    ("leer", "to read", "verb_inf"),
    ("hablar", "to speak", "verb_inf"),
    ("come", "he or she eats", "verb_fin"),
    ("bebe", "he or she drinks", "verb_fin"),
    ("comes", "you eat", "verb_fin"),
    ("como", "I eat", "verb_fin"),
    ("está", "he or she is (state or place)", "verb_fin"),
    ("estoy", "I am (state or place)", "verb_fin"),
    ("triste", "sad", "adj"),
    ("grande", "big", "adj"),
    ("pequeño", "small", "adj"),
    ("nuevo", "new", "adj"),
    ("viejo", "old", "adj"),
    ("moreno", "dark haired", "adj"),
    ("rubio", "fair haired", "adj"),
    ("siete", "seven", "num"),
    ("ocho", "eight", "num"),
    ("nueve", "nine", "num"),
]

SKELETON: list[dict] = [
    {
        "id": "u1",
        "title": "Basics 1",
        "color": "#58cc02",
        "order": 1,
        "lessons": [
            {
                "id": "u1l1",
                "title": "Greetings",
                "size": 7,
                "items": [
                    ("Hola", "Hello", "phrase"),
                    ("Buenos días", "Good morning", "phrase"),
                    ("Buenas tardes", "Good afternoon", "phrase"),
                    ("Buenas noches", "Good evening", "phrase"),
                    ("Adiós", "Goodbye", "phrase"),
                    ("Hasta luego", "See you later", "phrase"),
                ],
                "select": [
                    (
                        "Which greeting do you use first thing in the morning?",
                        "Buenos días",
                        ["Buenas tardes", "Buenas noches", "Hasta luego"],
                    )
                ],
                "match": [
                    ("Match the meaning: 'Goodbye'", "Adiós", ["Buenos días", "Buenas tardes", "Hola"])
                ],
                "fill": [
                    (
                        "Fill the blank: 'Buenos ___' (Good morning)",
                        "días",
                        ["tardes", "noches", "luego"],
                    )
                ],
            },
            {
                "id": "u1l2",
                "title": "Courtesy",
                "size": 7,
                "items": [
                    ("Por favor", "Please", "phrase"),
                    ("Gracias", "Thank you", "phrase"),
                    ("De nada", "You are welcome", "phrase"),
                    ("Perdón", "Excuse me", "phrase"),
                    ("Lo siento", "I am sorry", "phrase"),
                    ("Sí", "Yes", "adv"),
                    ("No", "No", "adv"),
                ],
                "select": [
                    (
                        "Which phrase do you add when you ask for something politely?",
                        "Por favor",
                        ["De nada", "Lo siento", "Perdón"],
                    )
                ],
                "match": [
                    ("Match the meaning: 'I am sorry'", "Lo siento", ["De nada", "Por favor", "Gracias"])
                ],
                "fill": [
                    (
                        "Fill the blank: 'Lo ___' (I am sorry)",
                        "siento",
                        ["favor", "nada", "gusto"],
                    )
                ],
            },
            {
                "id": "u1l3",
                "title": "Introductions",
                "size": 7,
                "items": [
                    ("Me llamo", "My name is", "phrase"),
                    ("Hasta pronto", "See you soon", "phrase"),
                    ("Mucho gusto", "Nice to meet you", "phrase"),
                    ("¿Cómo estás?", "How are you?", "phrase"),
                    ("Bien", "Fine", "adv"),
                    ("¿Cómo te llamas?", "What is your name?", "phrase"),
                ],
                "select": [
                    (
                        "Which question do you ask to learn someone's name?",
                        "¿Cómo te llamas?",
                        ["¿Cómo estás?", "Mucho gusto", "Hasta luego"],
                    )
                ],
                "match": [
                    (
                        "Match the meaning: 'Fine'",
                        "Bien",
                        ["Muy", "Siempre", "Nunca"],
                    )
                ],
                "fill": [
                    (
                        "Fill the blank: 'Me ___ Ana' (My name is Ana)",
                        "llamo",
                        ["soy", "estás", "eres"],
                    )
                ],
            },
        ],
    },
    {
        "id": "u2",
        "title": "Basics 2",
        "color": "#1cb0f6",
        "order": 2,
        "lessons": [
            {
                "id": "u2l1",
                "title": "People",
                "size": 6,
                "items": [
                    ("el hombre", "the man", "noun"),
                    ("la mujer", "the woman", "noun"),
                    ("el chico", "the boy", "noun"),
                    ("la chica", "the girl", "noun"),
                    ("el amigo", "the friend (male)", "noun"),
                    ("la amiga", "the friend (female)", "noun"),
                ],
                "select": [
                    (
                        "Which noun takes the article 'la'?",
                        "la mujer",
                        ["el hombre", "el coche", "el libro"],
                    )
                ],
                "match": [
                    ("Match the meaning: 'the friend (male)'", "el amigo", ["la amiga", "la chica", "el hombre"])
                ],
                "fill": [
                    ("Fill the blank: '___ hombre' (the man)", "el", ["la", "los", "las"])
                ],
            },
            {
                "id": "u2l2",
                "title": "Everyday objects",
                "size": 6,
                "items": [
                    ("el libro", "the book", "noun"),
                    ("la mesa", "the table", "noun"),
                    ("la silla", "the chair", "noun"),
                    ("la puerta", "the door", "noun"),
                    ("la ventana", "the window", "noun"),
                    ("el teléfono", "the phone", "noun"),
                ],
                "select": [
                    (
                        "Which object do you pick up to call someone?",
                        "el teléfono",
                        ["la ventana", "la mesa", "la silla"],
                    )
                ],
                "match": [
                    ("Match the meaning: 'the window'", "la ventana", ["la puerta", "la mesa", "la silla"])
                ],
                "fill": [
                    ("Fill the blank: '___ mesa' (the table)", "la", ["el", "los", "las"])
                ],
            },
            {
                "id": "u2l3",
                "title": "Around town",
                "size": 6,
                "items": [
                    ("la casa", "the house", "noun"),
                    ("el coche", "the car", "noun"),
                    ("la ciudad", "the city", "noun"),
                    ("el trabajo", "the job", "noun"),
                    ("la escuela", "the school", "noun"),
                    ("el parque", "the park", "noun"),
                ],
                "select": [
                    (
                        "Where do children go to learn?",
                        "la escuela",
                        ["la casa", "el parque", "la ciudad"],
                    )
                ],
                "match": [
                    ("Match the meaning: 'the park'", "el parque", ["el coche", "el trabajo", "la casa"])
                ],
                "fill": [
                    ("Fill the blank: '___ trabajo' (the job)", "el", ["la", "las", "los"])
                ],
            },
            {
                "id": "u2l4",
                "title": "Numbers one to six",
                "size": 6,
                "items": [
                    ("uno", "one", "num"),
                    ("dos", "two", "num"),
                    ("tres", "three", "num"),
                    ("cuatro", "four", "num"),
                    ("cinco", "five", "num"),
                    ("seis", "six", "num"),
                ],
                "select": [
                    (
                        "Which number comes straight after 'tres'?",
                        "cuatro",
                        ["dos", "cinco", "seis"],
                    )
                ],
                "match": [("Match the meaning: 'five'", "cinco", ["cuatro", "seis", "tres"])],
                "fill": [
                    (
                        "Fill the blank: 'Uno, dos, ___' (one, two, three)",
                        "tres",
                        ["cuatro", "cinco", "seis"],
                    )
                ],
            },
        ],
    },
    {
        "id": "u3",
        "title": "Food and drink",
        "color": "#ff9600",
        "order": 3,
        "lessons": [
            {
                "id": "u3l1",
                "title": "Food",
                "size": 7,
                "items": [
                    ("el pan", "the bread", "noun"),
                    ("la manzana", "the apple", "noun"),
                    ("el queso", "the cheese", "noun"),
                    ("el arroz", "the rice", "noun"),
                    ("el pollo", "the chicken", "noun"),
                    ("la sopa", "the soup", "noun"),
                ],
                "select": [
                    (
                        "Which food is a fruit?",
                        "la manzana",
                        ["el pan", "el queso", "el pollo"],
                    )
                ],
                "match": [
                    ("Match the meaning: 'the chicken'", "el pollo", ["el queso", "el pan", "el arroz"])
                ],
                "fill": [
                    ("Fill the blank: '___ sopa' (the soup)", "la", ["el", "los", "las"])
                ],
            },
            {
                "id": "u3l2",
                "title": "Drinks",
                "size": 7,
                "items": [
                    ("el agua", "the water", "noun"),
                    ("el café", "the coffee", "noun"),
                    ("el té", "the tea", "noun"),
                    ("la leche", "the milk", "noun"),
                    ("el zumo", "the juice", "noun"),
                    ("el vino", "the wine", "noun"),
                ],
                "select": [
                    (
                        "Which drink is served hot and made from roasted beans?",
                        "el café",
                        ["la leche", "el zumo", "el agua"],
                    )
                ],
                "match": [
                    ("Match the meaning: 'the wine'", "el vino", ["el zumo", "la leche", "el agua"])
                ],
                "fill": [
                    ("Fill the blank: '___ leche' (the milk)", "la", ["el", "los", "las"])
                ],
            },
            {
                "id": "u3l3",
                "title": "Meals and the verbs comer and beber",
                "size": 7,
                "items": [
                    ("comer", "to eat", "verb_inf"),
                    ("beber", "to drink", "verb_inf"),
                    ("el desayuno", "the breakfast", "noun"),
                    ("la cena", "the dinner", "noun"),
                    ("la comida", "the food", "noun"),
                    ("el restaurante", "the restaurant", "noun"),
                ],
                "select": [
                    (
                        "Which meal do you eat in the morning?",
                        "el desayuno",
                        ["la cena", "la comida", "el restaurante"],
                    )
                ],
                "match": [
                    ("Match the meaning: 'to drink'", "beber", ["comer", "cocinar", "leer"])
                ],
                "fill": [
                    (
                        "Fill the blank: 'El ___' (The breakfast)",
                        "desayuno",
                        ["cena", "comida", "restaurante"],
                    )
                ],
            },
            {
                "id": "u3l4",
                "title": "At the cafe",
                "size": 7,
                "items": [
                    ("Quiero un café", "I want a coffee", "phrase"),
                    ("¿Tienes agua?", "Do you have water?", "phrase"),
                    ("La cuenta, por favor", "The bill, please", "phrase"),
                    ("Está delicioso", "It is delicious", "phrase"),
                    ("Un vaso de agua", "A glass of water", "phrase"),
                    ("Me gusta el té", "I like tea", "phrase"),
                ],
                "select": [
                    (
                        "What do you ask for when you want to pay in a cafe?",
                        "La cuenta, por favor",
                        ["Un vaso de agua", "Quiero un café", "Me gusta el té"],
                    )
                ],
                "match": [
                    (
                        "Match the meaning: 'I like tea'",
                        "Me gusta el té",
                        ["Quiero un café", "Un vaso de agua", "La cuenta, por favor"],
                    )
                ],
                "fill": [
                    (
                        "Fill the blank: 'Un ___ de agua' (A glass of water)",
                        "vaso",
                        ["cuenta", "café", "gusta"],
                    )
                ],
            },
        ],
    },
    {
        "id": "u4",
        "title": "Family and description",
        "color": "#ce82ff",
        "order": 4,
        "lessons": [
            {
                "id": "u4l1",
                "title": "Family",
                "size": 6,
                "items": [
                    ("la madre", "the mother", "noun"),
                    ("el padre", "the father", "noun"),
                    ("el hermano", "the brother", "noun"),
                    ("la hermana", "the sister", "noun"),
                    ("el hijo", "the son", "noun"),
                    ("la hija", "the daughter", "noun"),
                ],
                "select": [
                    (
                        "Which word completes the pair of parents, el padre y ___?",
                        "la madre",
                        ["el hermano", "la hija", "el hijo"],
                    )
                ],
                "match": [
                    ("Match the meaning: 'the sister'", "la hermana", ["la hija", "el hermano", "la madre"])
                ],
                "fill": [
                    ("Fill the blank: '___ madre' (the mother)", "la", ["el", "los", "las"])
                ],
            },
            {
                "id": "u4l2",
                "title": "Family and possessives",
                "size": 6,
                "items": [
                    ("el abuelo", "the grandfather", "noun"),
                    ("la abuela", "the grandmother", "noun"),
                    ("el tío", "the uncle", "noun"),
                    ("la tía", "the aunt", "noun"),
                    ("mi madre", "my mother", "phrase"),
                    ("tu hermano", "your brother", "phrase"),
                ],
                "select": [
                    (
                        "Which phrase shows that the mother belongs to me?",
                        "mi madre",
                        ["tu hermano", "el abuelo", "la tía"],
                    )
                ],
                "match": [
                    (
                        "Match the meaning: 'the aunt'",
                        "la tía",
                        ["el tío", "el abuelo", "la abuela"],
                    )
                ],
                "fill": [
                    ("Fill the blank: '___ abuela' (the grandmother)", "la", ["el", "los", "las"])
                ],
            },
            {
                "id": "u4l3",
                "title": "Ser and estar",
                "size": 6,
                "items": [
                    ("soy", "I am (identity)", "verb_fin"),
                    ("eres", "you are (identity)", "verb_fin"),
                    ("es", "he or she is (identity)", "verb_fin"),
                    ("estoy", "I am (state or place)", "verb_fin"),
                    ("estás", "you are (state or place)", "verb_fin"),
                    ("está", "he or she is (state or place)", "verb_fin"),
                ],
                "select": [
                    (
                        "Which form do you use to say where someone is right now?",
                        "está",
                        ["es", "eres", "soy"],
                    )
                ],
                "match": [
                    (
                        "Match the meaning: 'I am', said about a place or a mood",
                        "estoy",
                        ["soy", "estás", "está"],
                    )
                ],
                "fill": [
                    (
                        "Fill the blank: 'Yo ___ profesor' (I am a teacher, identity)",
                        "soy",
                        ["estoy", "eres", "es"],
                    )
                ],
            },
            {
                "id": "u4l4",
                "title": "Describing people",
                "size": 6,
                "items": [
                    ("alto", "tall", "adj"),
                    ("bajo", "short", "adj"),
                    ("guapo", "good looking", "adj"),
                    ("simpático", "friendly", "adj"),
                    ("inteligente", "clever", "adj"),
                    ("cansado", "tired", "adj"),
                ],
                "select": [
                    (
                        "Which word describes someone who reads a lot and learns fast?",
                        "inteligente",
                        ["cansado", "alto", "bajo"],
                    )
                ],
                "match": [("Match the meaning: 'tired'", "cansado", ["simpático", "inteligente", "bajo"])],
                "fill": [
                    (
                        "Fill the blank: 'Mi abuelo es ___' (My grandfather is friendly)",
                        "simpático",
                        ["cansado", "alto", "bajo"],
                    )
                ],
            },
        ],
    },
]


# ------------------------------------------------------------------- generation


def _all_items() -> list[Item]:
    items: list[Item] = []
    for unit in SKELETON:
        for lesson in unit["lessons"]:
            items.extend(lesson["items"])
    items.extend(EXTRA_POOL)
    # First occurrence wins, so a curated lesson item outranks a pool filler.
    seen: set[str] = set()
    unique: list[Item] = []
    for spanish, english, pos in items:
        key = spanish.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((spanish, english, pos))
    return unique


def _pool_by_pos() -> dict[str, list[Item]]:
    pool: dict[str, list[Item]] = {}
    for item in _all_items():
        pool.setdefault(item[2], []).append(item)
    return pool


def _no_em_dash(text: str) -> str:
    if "\u2014" in text:
        raise ValueError(f"em dash found in content: {text!r}")
    return text


def _pick_distractors(
    answer: str,
    english: str,
    pos: str,
    near: list[Item],
    wide: list[Item],
    pool: dict[str, list[Item]],
    count: int,
) -> list[str]:
    """Same-part-of-speech wrong answers that never mean what the prompt means."""
    chosen: list[str] = []
    taken = {answer.strip().lower()}
    prompt_meaning = english.strip().lower()
    # Near peers first (same lesson), then the same unit, then the whole course.
    for group in (near, wide, pool.get(pos, [])):
        for spanish, other_english, other_pos in group:
            if len(chosen) == count:
                return chosen
            if other_pos != pos:
                continue
            key = spanish.strip().lower()
            if key in taken:
                continue
            if other_english.strip().lower() == prompt_meaning:
                continue
            taken.add(key)
            chosen.append(spanish)
    if len(chosen) < count:
        raise ValueError(f"not enough {pos} distractors for {answer!r} ({english!r})")
    return chosen


def _make_options(rng: random.Random, answer: str, distractors: list[str]) -> tuple[list[str], int]:
    options = [answer, *distractors]
    rng.shuffle(options)
    return options, options.index(answer)


def _challenge(
    challenge_id: str,
    kind: str,
    prompt: str,
    answer: str,
    distractors: list[str],
    difficulty: int,
) -> dict:
    rng = random.Random(SEED_NAMESPACE ^ zlib.crc32(challenge_id.encode("utf-8")))
    options, correct_index = _make_options(rng, answer, distractors)
    return {
        "id": challenge_id,
        "type": kind,
        "prompt": _no_em_dash(prompt),
        "promptLang": "en",
        "answer": answer,
        "options": options,
        "correctIndex": correct_index,
        "audio": answer,
        "difficulty": difficulty,
    }


def build_payload() -> dict:
    """Build the whole course as a plain dict, deterministically."""
    pool = _pool_by_pos()
    units_out: list[dict] = []

    for unit in SKELETON:
        lessons_out: list[dict] = []
        unit_items: list[Item] = [item for lesson in unit["lessons"] for item in lesson["items"]]

        for lesson in unit["lessons"]:
            curated = len(lesson["select"]) + len(lesson["match"]) + len(lesson["fill"])
            translate_count = lesson["size"] - curated
            if translate_count < 2:
                raise ValueError(f"{lesson['id']}: lesson size leaves too few translate challenges")
            if translate_count > len(lesson["items"]):
                raise ValueError(f"{lesson['id']}: more translate challenges than items")

            challenges: list[dict] = []
            index = 0

            def next_id() -> str:
                nonlocal index
                index += 1
                return f"{lesson['id']}c{index}"

            for spanish, english, pos in lesson["items"][:translate_count]:
                distractors = _pick_distractors(
                    spanish, english, pos, lesson["items"], unit_items, pool, 3
                )
                challenges.append(
                    _challenge(
                        next_id(),
                        "translate",
                        f"How do you say '{english}' in Spanish?",
                        spanish,
                        distractors,
                        difficulty=unit["order"],
                    )
                )

            for prompt, answer, distractors in lesson["select"]:
                challenges.append(
                    _challenge(next_id(), "select", prompt, answer, list(distractors), unit["order"])
                )

            for prompt, answer, distractors in lesson["match"]:
                challenges.append(
                    _challenge(next_id(), "match", prompt, answer, list(distractors), unit["order"])
                )

            for prompt, answer, distractors in lesson["fill"]:
                challenges.append(
                    _challenge(next_id(), "fill", prompt, answer, list(distractors), unit["order"])
                )

            lessons_out.append(
                {
                    "id": lesson["id"],
                    "title": lesson["title"],
                    "order": len(lessons_out) + 1,
                    "challenges": challenges,
                }
            )

        units_out.append(
            {
                "id": unit["id"],
                "title": unit["title"],
                "color": unit["color"],
                "order": unit["order"],
                "lessons": lessons_out,
            }
        )

    return {
        "course": COURSE,
        "from": COURSE_FROM,
        "version": VERSION,
        "units": units_out,
    }


def render(payload: dict) -> str:
    """Canonical on-disk form: UTF-8, two-space indent, accented text kept literal."""
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def validate(payload: dict) -> None:
    """Fail here rather than leaving an invalid file on disk for the service."""
    models = _curriculum_module()
    models.Curriculum.model_validate(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build brain/curriculum/es-en.json")
    parser.add_argument("--check", action="store_true", help="fail if the file on disk differs")
    parser.add_argument("--summary", action="store_true", help="print per-unit counts")
    args = parser.parse_args(argv)

    payload = build_payload()
    validate(payload)
    text = render(payload)

    if args.check:
        if not OUT_PATH.exists():
            print(f"MISSING {OUT_PATH}")
            return 1
        if OUT_PATH.read_text(encoding="utf-8") != text:
            print(f"STALE {OUT_PATH} does not match the generator")
            return 1
        print(f"OK {OUT_PATH.name} matches the generator")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")

    if args.summary:
        total = 0
        for unit in payload["units"]:
            per_lesson = [len(lesson["challenges"]) for lesson in unit["lessons"]]
            unit_total = sum(per_lesson)
            total += unit_total
            print(
                f"{unit['order']}. {unit['id']} {unit['title']:<26} "
                f"lessons={len(unit['lessons'])} per_lesson={per_lesson} unit_total={unit_total}"
            )
        print(f"total challenges={total} units={len(payload['units'])}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
