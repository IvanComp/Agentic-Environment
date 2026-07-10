from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from typing import Any


_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_ANSWER_VALUE_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+(?:,\d{3})*(?:\.\d+)?)?%?")
_RATIONALE_KEYS = ("rationale", "rational", "reasoning", "explanation")


def estimate_tokens(text: str) -> int:
    """Small, dependency-free token estimate for comparing runs."""
    if not text:
        return 0
    return max(1, math.ceil(len(re.findall(r"\w+|[^\w\s]", text)) * 1.12))


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_RE.findall(text):
        values.append(float(match.replace(",", "")))
    return values


def try_parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    obj = re.search(r"\{.*\}", text, flags=re.S)
    if obj:
        candidates.insert(0, obj.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_answer_candidate(text: Any, choices: dict[str, str] | None = None) -> str:
    raw = "" if text is None else str(text).strip()
    if not raw:
        return ""

    parsed = try_parse_json(raw)
    if parsed and "answer" in parsed:
        return str(parsed["answer"]).strip()

    if choices:
        choice = _extract_choice_answer(raw, choices)
        if choice:
            return choice

    final_numeric = extract_final_answer_from_rationale(raw)
    if final_numeric:
        return final_numeric

    key_values = re.findall(
        r"""["']?answer["']?\s*:\s*["']?([A-Ea-e]|-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?%?)""",
        raw,
    )
    if key_values:
        return normalize_answer(key_values[-1])

    explicit = re.search(
        r"(?:final\s+answer|answer|therefore|thus)\s*(?:is|:|=|->)\s*([A-Ea-e]|-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?%?)",
        raw,
        flags=re.I,
    )
    if explicit:
        return normalize_answer(explicit.group(1))

    return normalize_answer(raw)


def extract_rationale_candidate(text: Any) -> str:
    raw = "" if text is None else str(text).strip()
    if not raw:
        return ""

    parsed = try_parse_json(raw)
    if parsed:
        for key in _RATIONALE_KEYS:
            if parsed.get(key) is not None:
                return _clean_rationale_value(parsed[key])
        return "No rationale was provided by the model."

    cleaned = _strip_code_fences(raw)
    is_context_echo = ('"agent"' in cleaned or "'agent'" in cleaned) and cleaned.lstrip().startswith("[")
    if is_context_echo or (len(cleaned) > 1200 and ('"agent"' in cleaned or "'agent'" in cleaned)):
        return "Malformed JSON/context echo omitted; answer extracted from the model's final answer cue."

    malformed_value = re.search(
        r'''["'](?:rationale|rational|reasoning|explanation)["']\s*:\s*["'](.*?)["']\s*(?:,|\})''',
        raw,
        flags=re.S | re.I,
    )
    if malformed_value:
        return _clean_rationale_value(malformed_value.group(1))

    return cleaned


def _clean_rationale_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in _RATIONALE_KEYS:
            if value.get(key) is not None:
                return _clean_rationale_value(value[key])
        return "; ".join(f"{key}: {item}" for key, item in value.items()).strip()
    if isinstance(value, list):
        return " ".join(_clean_rationale_value(item) for item in value).strip()
    cleaned = str(value).strip()
    cleaned = cleaned.replace("\\n", "\n").replace("\\t", " ")
    return cleaned


def _extract_choice_answer(text: str, choices: dict[str, str]) -> str:
    arrow = re.findall(r"(?:->|=>)\s*\(?([A-Ea-e])\)?\s*(?:=|\b)", text)
    if arrow:
        return arrow[-1].upper()

    explicit = re.findall(
        r"(?:final\s+answer|answer|therefore|thus|choice|option)\s*(?:is|:|=|->)?\s*\(?([A-Ea-e])\)?\b",
        text,
        flags=re.I,
    )
    if explicit:
        return explicit[-1].upper()

    for letter, value in choices.items():
        value_pattern = re.escape(normalize_answer(value))
        if re.search(rf"(?:->|=>)\s*{letter}\s*=\s*{value_pattern}\b", text, flags=re.I):
            return letter.upper()
        if re.search(rf"\b{letter}\s*=\s*{value_pattern}\b", text, flags=re.I):
            return letter.upper()

    values = re.findall(
        r"""["']?answer["']?\s*:\s*["']?(-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?%?)""",
        text,
    )
    if values:
        return normalize_answer(values[-1])
    return ""


def _strip_code_fences(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        return fenced.group(1).strip()
    return text


def normalize_answer(answer: Any) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    if not text:
        return ""

    parsed = try_parse_json(text)
    if parsed and "answer" in parsed:
        return normalize_answer(parsed["answer"])

    explicit = re.search(r"(?:final\s*)?answer\s*[:=]\s*([A-E]|-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?)", text, flags=re.I)
    if explicit:
        return normalize_answer(explicit.group(1))

    letter = re.fullmatch(r"\(?([A-Ea-e])\)?", text)
    if letter:
        return letter.group(1).upper()

    text = text.replace("$", "").replace(",", "").strip()
    text = re.sub(r"\s+", " ", text)
    if re.fullmatch(r"-?\d+(?:\.\d+)?%", text):
        return format_number(float(text[:-1]))
    if re.fullmatch(r"-?\d+(?:\.\d+)?/-?\d+(?:\.\d+)?", text):
        try:
            return format_number(float(Fraction(text)))
        except ZeroDivisionError:
            return text
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return format_number(float(text))

    trailing_number = _NUMBER_RE.findall(text)
    if trailing_number:
        return normalize_answer(trailing_number[-1])
    return text.lower()


def extract_final_answer_from_rationale(text: Any) -> str:
    """Infer a final numeric answer from a rationale without using ground truth.

    Small local models sometimes emit contradictory JSON such as
    {"answer": "24", "rationale": "... 48 + 24 = 72"}. This helper only uses
    explicit conclusion cues from the model's own rationale, preferring the
    right-hand side of equations and final/therefore/total sentences.
    """
    if text is None:
        return ""
    rationale = str(text).strip()
    if not rationale:
        return ""

    candidates: list[tuple[int, int, int, str]] = []
    sentences = re.split(r"(?<=[.!?])\s+", rationale)
    for sentence_index, sentence in enumerate(sentences):
        cleaned = sentence.strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        conclusion_cue = any(
            cue in lower
            for cue in (
                "final",
                "answer",
                "therefore",
                "thus",
                " so ",
                "total",
                "altogether",
                "combined",
                "equals",
                "earned",
                "left",
                "remain",
            )
        )
        if "confidence" in lower and not conclusion_cue and "=" not in cleaned:
            continue

        equation_values = [match.group(1) for match in re.finditer(r"=\s*(" + _ANSWER_VALUE_RE.pattern + r")", cleaned)]
        for match_index, value in enumerate(equation_values):
            score = 8 if conclusion_cue else 5
            candidates.append((score, sentence_index, match_index, value))

        has_unresolved_expression = bool(re.search(r"\d\s*(?:\+|\*|x|×|÷)\s*\d|\d\s*/\s*\d", cleaned, flags=re.I))
        if conclusion_cue and not has_unresolved_expression:
            values = _ANSWER_VALUE_RE.findall(cleaned)
            if values:
                candidates.append((3, sentence_index, len(values) - 1, values[-1]))

    if not candidates:
        return ""
    _, _, _, value = max(candidates)
    return normalize_answer(value)


def answers_match(predicted: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    pred = normalize_answer(predicted)
    gold = normalize_answer(expected)
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) <= tolerance
    except ValueError:
        return pred.lower() == gold.lower()


def choice_to_value(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?%", cleaned):
        return float(cleaned[:-1])
    if re.fullmatch(r"-?\d+(?:\.\d+)?/-?\d+(?:\.\d+)?", cleaned):
        try:
            return float(Fraction(cleaned))
        except ZeroDivisionError:
            return None
    number = _NUMBER_RE.search(cleaned)
    if number:
        return float(number.group(0).replace(",", ""))
    return None


def choose_closest_option(choices: dict[str, str], value: float) -> str:
    best_letter = ""
    best_distance = float("inf")
    for letter, choice in choices.items():
        parsed = choice_to_value(choice)
        if parsed is None:
            continue
        distance = abs(parsed - value)
        if distance < best_distance:
            best_letter = letter
            best_distance = distance
    return best_letter
