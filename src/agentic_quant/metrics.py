from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["benchmark"], record["mechanism"])].append(record)

    by_group: list[dict[str, Any]] = []
    for (benchmark, mechanism), rows in sorted(grouped.items()):
        by_group.append(
            {
                "benchmark": benchmark,
                "mechanism": mechanism,
                "n": len(rows),
                "accuracy": _ratio(row["correct"] for row in rows),
                "mean_elapsed_s": mean(row["elapsed_s"] for row in rows),
                "mean_total_tokens": mean(row["total_tokens"] for row in rows),
                "mean_prompt_tokens": mean(row["prompt_tokens"] for row in rows),
                "mean_completion_tokens": mean(row["completion_tokens"] for row in rows),
                "mean_agreement": mean(row["agreement"] for row in rows),
                "tie_rate": _ratio(row["tie"] for row in rows),
                "mean_debate_rounds": mean(row["debate_rounds"] for row in rows),
            }
        )

    return {
        "overall": {
            "n": len(records),
            "question_count": len({(record["benchmark"], record["item_id"]) for record in records}),
            "accuracy": _ratio(row["correct"] for row in records),
            "mean_elapsed_s": mean(row["elapsed_s"] for row in records) if records else 0.0,
            "mean_total_tokens": mean(row["total_tokens"] for row in records) if records else 0.0,
        },
        "by_benchmark_mechanism": by_group,
    }


def _ratio(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


@dataclass
class _GroupTotals:
    n: int = 0
    correct: int = 0
    elapsed_s: float = 0.0
    total_tokens: float = 0.0
    prompt_tokens: float = 0.0
    completion_tokens: float = 0.0
    agreement: float = 0.0
    ties: int = 0
    debate_rounds: float = 0.0

    def add(self, record: dict[str, Any]) -> None:
        self.n += 1
        self.correct += int(bool(record["correct"]))
        self.elapsed_s += float(record["elapsed_s"])
        self.total_tokens += float(record["total_tokens"])
        self.prompt_tokens += float(record["prompt_tokens"])
        self.completion_tokens += float(record["completion_tokens"])
        self.agreement += float(record["agreement"])
        self.ties += int(bool(record["tie"]))
        self.debate_rounds += float(record["debate_rounds"])

    def summary(self, benchmark: str, mechanism: str) -> dict[str, Any]:
        divisor = self.n or 1
        return {
            "benchmark": benchmark,
            "mechanism": mechanism,
            "n": self.n,
            "accuracy": self.correct / divisor,
            "mean_elapsed_s": self.elapsed_s / divisor,
            "mean_total_tokens": self.total_tokens / divisor,
            "mean_prompt_tokens": self.prompt_tokens / divisor,
            "mean_completion_tokens": self.completion_tokens / divisor,
            "mean_agreement": self.agreement / divisor,
            "tie_rate": self.ties / divisor,
            "mean_debate_rounds": self.debate_rounds / divisor,
        }


class SummaryAccumulator:
    """O(1) incremental metric updates for streaming benchmark runs."""

    def __init__(self) -> None:
        self._groups: dict[tuple[str, str], _GroupTotals] = defaultdict(_GroupTotals)
        self._questions: set[tuple[str, str]] = set()
        self._overall = _GroupTotals()

    def add(self, record: dict[str, Any]) -> None:
        key = (str(record["benchmark"]), str(record["mechanism"]))
        self._groups[key].add(record)
        self._overall.add(record)
        self._questions.add((str(record["benchmark"]), str(record["item_id"])))

    def snapshot(self) -> dict[str, Any]:
        n = self._overall.n
        divisor = n or 1
        return {
            "overall": {
                "n": n,
                "question_count": len(self._questions),
                "accuracy": self._overall.correct / divisor,
                "mean_elapsed_s": self._overall.elapsed_s / divisor,
                "mean_total_tokens": self._overall.total_tokens / divisor,
            },
            "by_benchmark_mechanism": [
                self._groups[key].summary(*key) for key in sorted(self._groups)
            ],
        }
