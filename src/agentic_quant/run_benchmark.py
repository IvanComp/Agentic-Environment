from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agents import build_agents_for_providers
from .benchmarks import available_benchmarks, load_benchmark
from .coordinator import Coordinator
from .metrics import summarise
from .model_registry import available_model_keys, available_model_presets, get_model_preset
from .model_store import ensure_models
from .providers import make_model_provider


def run_experiment(
    *,
    model_key: str = "demo",
    agent_model_keys: list[str] | None = None,
    download_model: bool = False,
    n_ctx: int = 2048,
    benchmarks: list[str] | None = None,
    mechanisms: list[str] | None = None,
    limit: int | None = None,
    benchmark_limits: dict[str, int | None] | None = None,
    debate_rounds: int = 2,
    voting_tie_breaker: str = "confidence",
    agent_count: int = 3,
    agent_roles: list[str] | None = None,
    out: str | Path = "runs/latest",
    echo: bool = True,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    download_progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    benchmarks = benchmarks or available_benchmarks()
    mechanisms = mechanisms or ["voting", "role", "debate"]
    if agent_count < 1 or agent_count > 12:
        raise ValueError("agent_count must be between 1 and 12")
    get_model_preset(model_key)
    agent_model_keys = _normalise_agent_models(model_key, agent_model_keys, agent_count)
    for key in agent_model_keys:
        get_model_preset(key)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_models(
        agent_model_keys,
        download=download_model,
        progress=download_progress or (_console_download_progress if echo else None),
    )
    provider_cache = {
        key: make_model_provider(model_key=key, n_ctx=n_ctx)
        for key in dict.fromkeys(agent_model_keys)
    }
    agent_providers = [provider_cache[key] for key in agent_model_keys]
    coordinators = {
        mechanism: Coordinator(
            build_agents_for_providers(
                agent_providers,
                role_overrides=(
                    agent_roles
                    if mechanism == "role"
                    else [_neutral_agent_instruction(mechanism)] * agent_count
                ),
                descriptive_names=False,
            ),
            debate_rounds=debate_rounds,
            voting_tie_breaker=voting_tie_breaker,
        )
        for mechanism in dict.fromkeys(mechanisms)
    }

    records: list[dict[str, Any]] = []
    for benchmark_name in benchmarks:
        benchmark_limit = benchmark_limits.get(benchmark_name, limit) if benchmark_limits else limit
        for item in load_benchmark(benchmark_name, limit=benchmark_limit):
            for mechanism in mechanisms:
                result = coordinators[mechanism].run(item, mechanism)
                record = result.to_dict()
                records.append(record)
                if on_record:
                    on_record(record)
                if echo:
                    status = "ok" if result.correct else "miss"
                    print(
                        f"{benchmark_name:14s} {item.item_id:10s} {mechanism:7s} "
                        f"{status:4s} pred={result.final_answer_display!r} gold={result.gold_display!r} "
                        f"agents={agent_count} tokens={result.total_tokens} time={result.elapsed_s:.3f}s"
                    )

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "model_label": get_model_preset(model_key).label,
        "agent_model_keys": agent_model_keys,
        "agent_model_labels": [get_model_preset(key).label for key in agent_model_keys],
        "download_model": download_model,
        "n_ctx": n_ctx,
        "benchmarks": benchmarks,
        "mechanisms": mechanisms,
        "limit": limit,
        "benchmark_limits": benchmark_limits or {},
        "debate_rounds": debate_rounds,
        "voting_tie_breaker": voting_tie_breaker,
        "agent_count": agent_count,
        "agent_roles": agent_roles or [],
    }

    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary = summarise(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "results_path": str(results_path),
        "config": config,
        "summary": summary,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight multi-agent quantitative benchmarks.")
    parser.add_argument("--model", default="demo", choices=available_model_keys(), help="Model preset to use.")
    parser.add_argument(
        "--agent-models",
        nargs="*",
        choices=available_model_keys(),
        help="Optional model preset per agent. Example: --agent-models qwen2.5-0.5b demo qwen2.5-1.5b",
    )
    parser.add_argument("--download-model", action="store_true", help="Download selected model files if they are not already available.")
    parser.add_argument("--ctx", type=int, default=2048, help=argparse.SUPPRESS)
    parser.add_argument("--list-models", action="store_true", help="Show available model presets and exit.")
    parser.add_argument("--benchmarks", nargs="+", default=available_benchmarks(), choices=available_benchmarks())
    parser.add_argument("--mechanisms", nargs="+", default=["voting", "role", "debate"], choices=["voting", "role", "debate"])
    parser.add_argument("--limit", type=int, default=None, help="Limit items per benchmark for quick tests.")
    parser.add_argument("--debate-rounds", type=int, default=2)
    parser.add_argument("--agents", type=int, default=3, help="Number of agents to create, from 1 to 12.")
    parser.add_argument("--out", default="runs/latest")
    args = parser.parse_args()

    if args.list_models:
        for preset in available_model_presets():
            print(f"{preset.key:24s} {preset.label:42s} {preset.approximate_size:8s} {preset.notes}")
        return

    result = run_experiment(
        model_key=args.model,
        agent_model_keys=args.agent_models,
        download_model=args.download_model,
        n_ctx=args.ctx,
        benchmarks=args.benchmarks,
        mechanisms=args.mechanisms,
        limit=args.limit,
        debate_rounds=args.debate_rounds,
        agent_count=args.agents,
        out=args.out,
        echo=True,
    )

    print("\nSummary")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nWrote {result['results_path']}")


def _normalise_agent_models(model_key: str, agent_model_keys: list[str] | None, agent_count: int) -> list[str]:
    if not agent_model_keys:
        return [model_key] * agent_count
    if len(agent_model_keys) > agent_count:
        raise ValueError("--agent-models cannot contain more entries than --agents.")
    padded = list(agent_model_keys)
    padded.extend([model_key] * (agent_count - len(padded)))
    return padded


def _neutral_agent_instruction(mechanism: str) -> str:
    if mechanism == "voting":
        return "independent quantitative problem solver"
    if mechanism == "debate":
        return "quantitative debate participant"
    return "quantitative problem solver"


def _console_download_progress(model_key: str, done: int, total: int) -> None:
    label = get_model_preset(model_key).label
    percent = 0 if not total else min(100, done / total * 100)
    print(f"\rDownloading {label}: {percent:5.1f}% ({done / 1_000_000:.1f}/{total / 1_000_000:.1f} MB)", end="", flush=True)
    if total and done >= total:
        print()


if __name__ == "__main__":
    main()
