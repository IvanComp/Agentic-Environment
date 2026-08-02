from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    benchmark: str
    question: str
    answer: str
    choices: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_answer: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_answer:
            data.pop("answer", None)
        return data


def available_benchmarks() -> list[str]:
    return ["gsm8k_mini", "aqua_rat_mini"]


BENCHMARK_INFO = {
    "gsm8k_mini": {
        "label": "GSM8K mini",
        "task_type": "Arithmetic word problems",
        "answer_format": "Final numeric answer",
        "description": "Natural-language quantitative problems covering arithmetic, percentages, averages, ratios, budgets, and conversions.",
    },
    "aqua_rat_mini": {
        "label": "AQuA-RAT mini",
        "task_type": "Multiple-choice quantitative reasoning",
        "answer_format": "Choice letter or a value that maps to a choice",
        "description": "Math and quantitative questions with A-E options: speed, probability, discounts, margins, geometry, and sequences.",
    },
}


def load_benchmark(name: str, *, limit: int | None = None) -> list[BenchmarkItem]:
    if name not in available_benchmarks():
        raise ValueError(f"Unknown benchmark {name!r}. Available: {', '.join(available_benchmarks())}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when provided")
    items = _load_benchmark_items(name)
    return list(items if limit is None else items[:limit])


@lru_cache(maxsize=None)
def _load_benchmark_items(name: str) -> tuple[BenchmarkItem, ...]:
    """Load immutable packaged data once instead of reopening it for every UI poll."""
    filename = f"{name}.jsonl"
    items: list[BenchmarkItem] = []
    with resources.files("agentic_quant.data").joinpath(filename).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            items.append(
                BenchmarkItem(
                    item_id=raw["item_id"],
                    benchmark=raw["benchmark"],
                    question=raw["question"],
                    answer=str(raw["answer"]),
                    choices=raw.get("choices") or {},
                    metadata=raw.get("metadata") or {},
                )
            )
    return tuple(items)


def load_many(names: list[str], *, limit: int | None = None) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for name in names:
        items.extend(load_benchmark(name, limit=limit))
    return items


def benchmark_info(name: str) -> dict[str, Any]:
    items = load_benchmark(name)
    base = BENCHMARK_INFO.get(name, {})
    skills = sorted({str(item.metadata.get("skill", "")) for item in items if item.metadata.get("skill")})
    example = items[0] if items else None
    return {
        "name": name,
        "label": base.get("label", name),
        "task_type": base.get("task_type", ""),
        "answer_format": base.get("answer_format", ""),
        "description": base.get("description", ""),
        "item_count": len(items),
        "skills": skills,
        "example": None
        if example is None
        else {
            "item_id": example.item_id,
            "question": example.question,
            "answer": example.answer,
            "choices": example.choices,
        },
    }


def all_benchmark_info() -> list[dict[str, Any]]:
    return [benchmark_info(name) for name in available_benchmarks()]
