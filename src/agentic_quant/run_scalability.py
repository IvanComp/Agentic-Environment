from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .benchmarks import available_benchmarks
from .run_benchmark import run_experiment


MODEL_KEY = "qwen2.5-0.5b"
AGENT_COUNTS = [5, 10, 100, 1000]
MECHANISMS = ["voting", "role", "debate"]
RAW_FIELDS = [
    "agent_count",
    "repeat",
    "seed",
    "mechanism",
    "evaluations",
    "agent_calls",
    "run_wall_time_s",
    "coordination_total_s",
    "mean_coordination_s",
    "evaluation_throughput_s",
    "agent_call_throughput_s",
    "accuracy",
    "peer_preservation_rate",
    "mean_agent_preservation_rate",
    "mean_agreement",
    "tie_rate",
    "total_tokens",
    "mean_tokens",
    "result_bytes",
]


def run_scalability_experiments(
    *,
    model_key: str = MODEL_KEY,
    agent_counts: list[int] | None = None,
    repeats: int = 2,
    seed_start: int = 0,
    download_model: bool = False,
    n_ctx: int = 32768,
    max_tokens: int = 24,
    temperature: float = 0.2,
    debate_rounds: int = 2,
    out: str | Path | None = None,
    echo: bool = True,
) -> dict[str, Any]:
    agent_counts = list(dict.fromkeys(agent_counts or AGENT_COUNTS))
    if not agent_counts or any(count < 1 or count > 1000 for count in agent_counts):
        raise ValueError("agent_counts must contain values between 1 and 1000")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    output_root = Path(out) if out is not None else _default_output_root()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model_key": model_key,
        "agent_counts": agent_counts,
        "repeats": repeats,
        "seeds": [seed_start + index for index in range(repeats)],
        "benchmarks": available_benchmarks(),
        "mechanisms": MECHANISMS,
        "n_ctx": n_ctx,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "debate_rounds": debate_rounds,
        "parallel_agents": False,
        "runs": [],
    }
    _write_json(output_root / "manifest.json", manifest)
    raw_rows: list[dict[str, Any]] = []
    total_runs = len(agent_counts) * repeats
    completed_runs = 0

    try:
        for agent_count in agent_counts:
            for repeat_index in range(1, repeats + 1):
                seed = seed_start + repeat_index - 1
                run_dir = output_root / f"agents_{agent_count:04d}" / f"run_{repeat_index:02d}_seed_{seed}"
                started = time.perf_counter()
                result = run_experiment(
                    model_key=model_key,
                    download_model=download_model,
                    n_ctx=n_ctx,
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    benchmarks=available_benchmarks(),
                    mechanisms=MECHANISMS,
                    debate_rounds=debate_rounds,
                    agent_count=agent_count,
                    parallel_agents=False,
                    out=run_dir,
                    echo=False,
                )
                wall_time = time.perf_counter() - started
                run_rows = _aggregate_run(
                    result["records"],
                    agent_count=agent_count,
                    repeat_index=repeat_index,
                    seed=seed,
                    wall_time=wall_time,
                )
                raw_rows.extend(run_rows)
                _write_csv(output_root / "scalability_runs.csv", raw_rows, RAW_FIELDS)
                (run_dir / "report.md").write_text(
                    _run_report(agent_count, repeat_index, seed, run_rows),
                    encoding="utf-8",
                )
                completed_runs += 1
                manifest["runs"].append(
                    {
                        "agent_count": agent_count,
                        "repeat": repeat_index,
                        "seed": seed,
                        "directory": str(run_dir.relative_to(output_root)),
                        "wall_time_s": wall_time,
                        "record_count": result["summary"]["overall"]["n"],
                    }
                )
                _write_json(output_root / "manifest.json", manifest)
                if echo:
                    print(
                        f"Run {completed_runs}/{total_runs} completed · agents={agent_count} "
                        f"· seed={seed} · {run_dir}"
                    )
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["completed_runs"] = completed_runs
        _write_json(output_root / "manifest.json", manifest)
        raise

    summary_rows = _summarise_scalability(raw_rows, baseline_agents=min(agent_counts))
    _write_csv(output_root / "scalability_summary.csv", summary_rows, list(summary_rows[0]))
    (output_root / "scalability_report.md").write_text(
        _scalability_report(model_key, repeats, summary_rows),
        encoding="utf-8",
    )
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_runs"] = completed_runs
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure peer-preservation scalability at 5, 10, 100, and 1000 agents."
    )
    parser.add_argument("--agent-counts", nargs="+", type=int, default=AGENT_COUNTS)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--debate-rounds", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    run_scalability_experiments(
        agent_counts=args.agent_counts,
        repeats=args.repeats,
        seed_start=args.seed_start,
        download_model=args.download_model,
        n_ctx=args.ctx,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        debate_rounds=args.debate_rounds,
        out=args.out,
    )


def _aggregate_run(
    records: list[dict[str, Any]],
    *,
    agent_count: int,
    repeat_index: int,
    seed: int,
    wall_time: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["mechanism"])].append(record)

    rows: list[dict[str, Any]] = []
    for mechanism in MECHANISMS:
        items = grouped[mechanism]
        total_elapsed = sum(float(item["elapsed_s"]) for item in items)
        total_tokens = sum(int(item["total_tokens"]) for item in items)
        agent_calls = sum(_agent_call_count(item) for item in items)
        peer_items = [item for item in items if item.get("peer_preservation_applicable")]
        rows.append(
            {
                "agent_count": agent_count,
                "repeat": repeat_index,
                "seed": seed,
                "mechanism": mechanism,
                "evaluations": len(items),
                "agent_calls": agent_calls,
                "run_wall_time_s": wall_time,
                "coordination_total_s": total_elapsed,
                "mean_coordination_s": total_elapsed / len(items),
                "evaluation_throughput_s": len(items) / total_elapsed if total_elapsed else 0.0,
                "agent_call_throughput_s": agent_calls / total_elapsed if total_elapsed else 0.0,
                "accuracy": _ratio(item["correct"] for item in items),
                "peer_preservation_rate": _ratio(
                    item.get("peer_preservation_detected") for item in peer_items
                ),
                "mean_agent_preservation_rate": mean(
                    float(item.get("peer_preservation_agent_rate", 0.0)) for item in peer_items
                ),
                "mean_agreement": mean(float(item["agreement"]) for item in items),
                "tie_rate": _ratio(item["tie"] for item in items),
                "total_tokens": total_tokens,
                "mean_tokens": total_tokens / len(items),
                "result_bytes": sum(
                    len(json.dumps(item, ensure_ascii=False).encode("utf-8")) for item in items
                ),
            }
        )
    return rows


def _agent_call_count(record: dict[str, Any]) -> int:
    mechanism = record["mechanism"]
    if mechanism == "voting":
        return len(record["trace"][0].get("votes", []))
    if mechanism == "role":
        return sum(1 for step in record["trace"] if step.get("output"))
    return sum(len(step.get("outputs", [])) for step in record["trace"])


def _summarise_scalability(
    rows: list[dict[str, Any]],
    *,
    baseline_agents: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["agent_count"]), str(row["mechanism"]))].append(row)

    metrics = [
        "coordination_total_s",
        "mean_coordination_s",
        "evaluation_throughput_s",
        "agent_call_throughput_s",
        "accuracy",
        "peer_preservation_rate",
        "mean_agent_preservation_rate",
        "mean_agreement",
        "tie_rate",
        "total_tokens",
        "mean_tokens",
        "result_bytes",
        "agent_calls",
    ]
    summaries: list[dict[str, Any]] = []
    for (agent_count, mechanism), items in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "agent_count": agent_count,
            "mechanism": mechanism,
            "runs": len(items),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)

    baselines = {
        row["mechanism"]: row
        for row in summaries
        if int(row["agent_count"]) == baseline_agents
    }
    for row in summaries:
        baseline = baselines[row["mechanism"]]
        row["time_growth_vs_baseline"] = _growth(
            row["coordination_total_s_mean"], baseline["coordination_total_s_mean"]
        )
        row["token_growth_vs_baseline"] = _growth(
            row["total_tokens_mean"], baseline["total_tokens_mean"]
        )
        row["output_growth_vs_baseline"] = _growth(
            row["result_bytes_mean"], baseline["result_bytes_mean"]
        )
    return summaries


def _run_report(
    agent_count: int,
    repeat_index: int,
    seed: int,
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Scalability run: {agent_count} agents",
        "",
        f"Repeat: `{repeat_index}` · seed: `{seed}`",
        "",
        "| Mechanism | Evaluations | Agent calls | Total seconds | Mean seconds | Tokens | Output MB | Accuracy | Preservation | Agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mechanism']} | {row['evaluations']} | {row['agent_calls']} | "
            f"{row['coordination_total_s']:.3f} | {row['mean_coordination_s']:.3f} | "
            f"{row['total_tokens']} | {row['result_bytes'] / 1_000_000:.3f} | "
            f"{row['accuracy']:.3f} | {row['peer_preservation_rate']:.3f} | "
            f"{row['mean_agreement']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _scalability_report(
    model_key: str,
    repeats: int,
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Scalability report",
        "",
        f"Model: `{model_key}` · runs per agent count: `{repeats}`.",
        "",
        "The same model instance is shared by all agents and calls are serialized. Debate uses full all-to-all rationale exposure, so its prompt-processing cost grows quadratically with the number of agents.",
        "",
        "| Agents | Mechanism | Mean total seconds | Std | Evaluations/s | Agent calls | Mean tokens | Output MB | Time vs baseline |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['agent_count']} | {row['mechanism']} | "
            f"{row['coordination_total_s_mean']:.3f} | {row['coordination_total_s_std']:.3f} | "
            f"{row['evaluation_throughput_s_mean']:.4f} | {row['agent_calls_mean']:.0f} | "
            f"{row['mean_tokens_mean']:.1f} | {row['result_bytes_mean'] / 1_000_000:.3f} | "
            f"{row['time_growth_vs_baseline']:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def _ratio(values: Any) -> float:
    values = list(values)
    return sum(1 for value in values if value) / len(values) if values else 0.0


def _growth(value: float, baseline: float) -> float:
    return float(value) / float(baseline) if baseline else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _default_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"scalability_qwen_05b_{timestamp}"


if __name__ == "__main__":
    main()
