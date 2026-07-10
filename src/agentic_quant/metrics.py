from __future__ import annotations

from collections import defaultdict
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
