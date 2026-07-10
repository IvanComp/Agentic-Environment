from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .benchmarks import BenchmarkItem
from .tools import answers_match, choice_to_value, normalize_answer


@dataclass
class AnswerEvaluation:
    raw_answer: str
    normalized_answer: str
    canonical_answer: str
    mapped_answer: str
    answer_display: str
    gold_answer: str
    gold_display: str
    correct: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_answer(item: BenchmarkItem, answer: Any) -> AnswerEvaluation:
    raw = "" if answer is None else str(answer).strip()
    normalized = normalize_answer(raw)
    gold = normalize_answer(item.answer)
    gold_display = ground_truth_display(item)

    if item.choices:
        mapped = map_to_choice(item, raw)
        expected_letter = gold.upper()
        correct = mapped == expected_letter
        if mapped:
            choice_value = item.choices.get(mapped, "")
            display = f"{raw} -> {mapped} = {choice_value}" if raw and normalized != mapped else f"{mapped} = {choice_value}"
            explanation = "LLM answer mapped to the correct option." if correct else "LLM answer mapped to an option different from the ground truth."
        else:
            display = raw or normalized
            explanation = "LLM answer could not be mapped to any option."
        return AnswerEvaluation(
            raw_answer=raw,
            normalized_answer=normalized,
            canonical_answer=mapped or normalized,
            mapped_answer=mapped,
            answer_display=display,
            gold_answer=item.answer,
            gold_display=gold_display,
            correct=correct,
            explanation=explanation,
        )

    correct = answers_match(normalized, item.answer)
    return AnswerEvaluation(
        raw_answer=raw,
        normalized_answer=normalized,
        canonical_answer=normalized,
        mapped_answer=normalized,
        answer_display=normalized,
        gold_answer=item.answer,
        gold_display=gold_display,
        correct=correct,
        explanation="Exact/numeric match." if correct else "Answer differs from the numeric ground truth.",
    )


def canonical_answer(item: BenchmarkItem, answer: Any) -> str:
    return evaluate_answer(item, answer).canonical_answer


def ground_truth_display(item: BenchmarkItem) -> str:
    if item.choices:
        letter = normalize_answer(item.answer).upper()
        value = item.choices.get(letter)
        if value is not None:
            return f"{letter} = {value}"
    return normalize_answer(item.answer)


def map_to_choice(item: BenchmarkItem, answer: Any) -> str:
    if not item.choices:
        return ""

    normalized = normalize_answer(answer).upper()
    choices = {letter.upper(): value for letter, value in item.choices.items()}
    if normalized in choices:
        return normalized

    for letter, value in choices.items():
        if normalize_answer(value).upper() == normalized:
            return letter

    try:
        predicted_value = float(normalize_answer(answer))
    except ValueError:
        return ""

    for letter, value in choices.items():
        choice_value = choice_to_value(value)
        if choice_value is not None and abs(predicted_value - choice_value) <= 1e-6:
            return letter
    return ""
