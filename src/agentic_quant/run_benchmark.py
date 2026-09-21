from __future__ import annotations

import argparse
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agents import build_agents_for_providers
from .benchmarks import available_benchmarks, load_benchmark
from .coordinator import Coordinator
from .metrics import SummaryAccumulator
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
    on_compute_units: Callable[[list[dict[str, Any]]], None] | None = None,
    download_progress: Callable[[str, int, int], None] | None = None,
    parallel_agents: bool = True,
    retain_records: bool = True,
) -> dict[str, Any]:
    benchmarks = list(dict.fromkeys(available_benchmarks() if benchmarks is None else benchmarks))
    mechanisms = list(dict.fromkeys(["voting", "role", "debate"] if mechanisms is None else mechanisms))
    _validate_run_configuration(
        benchmarks=benchmarks,
        mechanisms=mechanisms,
        limit=limit,
        benchmark_limits=benchmark_limits,
        debate_rounds=debate_rounds,
        voting_tie_breaker=voting_tie_breaker,
        agent_count=agent_count,
        agent_roles=agent_roles,
        n_ctx=n_ctx,
    )
    get_model_preset(model_key)
    agent_model_keys = _normalise_agent_models(model_key, agent_model_keys, agent_count)
    for key in agent_model_keys:
        get_model_preset(key)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "critical_agent": f"A{agent_count}",
        "parallel_agents": parallel_agents,
        "retain_records": retain_records,
    }
    _atomic_write_json(out_dir / "config.json", config)

    ensure_models(
        agent_model_keys,
        download=download_model,
        progress=download_progress or (_console_download_progress if echo else None),
    )
    provider_cache = {
        key: make_model_provider(model_key=key, n_ctx=n_ctx)
        for key in dict.fromkeys(agent_model_keys)
    }
    compute_units = _provider_compute_units(list(provider_cache.values()))
    config["compute_units"] = compute_units
    _atomic_write_json(out_dir / "config.json", config)
    if on_compute_units:
        on_compute_units(compute_units)
    agent_providers = [provider_cache[key] for key in agent_model_keys]
    records: list[dict[str, Any]] = []
    summary_accumulator = SummaryAccumulator()
    results_path = out_dir / "results.jsonl"
    use_parallel_executor = _should_parallelize_agents(agent_model_keys, parallel_agents)
    executor = (
        ThreadPoolExecutor(max_workers=agent_count, thread_name_prefix="agentic-quant")
        if use_parallel_executor
        else None
    )
    try:
        coordinators = {
            mechanism: Coordinator(
                build_agents_for_providers(
                    agent_providers,
                    role_overrides=(
                        _role_agent_roles(agent_roles, agent_count)
                        if mechanism == "role"
                        else _coordination_agent_roles(mechanism, agent_count)
                    ),
                    descriptive_names=False,
                ),
                debate_rounds=debate_rounds,
                voting_tie_breaker=voting_tie_breaker,
                executor=executor,
            )
            for mechanism in mechanisms
        }

        # Stream every completed row to disk. A failed or interrupted long run
        # therefore leaves valid JSONL up to the last completed coordination.
        with results_path.open("w", encoding="utf-8") as handle:
            for benchmark_name in benchmarks:
                benchmark_limit = benchmark_limits.get(benchmark_name, limit) if benchmark_limits else limit
                for item in load_benchmark(benchmark_name, limit=benchmark_limit):
                    for mechanism in mechanisms:
                        result = coordinators[mechanism].run(item, mechanism)
                        record = result.to_dict()
                        summary_accumulator.add(record)
                        if retain_records:
                            records.append(record)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        if on_record:
                            on_record(record)
                        if echo:
                            status = "ok" if result.correct else "miss"
                            print(
                                f"{benchmark_name:14s} {item.item_id:10s} {mechanism:7s} "
                                f"{status:4s} pred={result.final_answer_display!r} gold={result.gold_display!r} "
                                f"agents={agent_count} tokens={result.total_tokens} time={result.elapsed_s:.3f}s"
                            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    summary = summary_accumulator.snapshot()
    _atomic_write_json(out_dir / "summary.json", summary)
    return {
        "out_dir": str(out_dir),
        "results_path": str(results_path),
        "config": config,
        "summary": summary,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent peer-preservation benchmarks.")
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
    parser.add_argument("--voting-tie-breaker", choices=["confidence", "first"], default="confidence")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents to create, from 1 to 12.")
    parser.add_argument("--agent-roles", nargs="*", help="Optional role label for each agent in role-based coordination.")
    parser.add_argument("--sequential-agents", action="store_true", help="Disable parallel independent agent calls.")
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
        voting_tie_breaker=args.voting_tie_breaker,
        agent_count=args.agents,
        agent_roles=args.agent_roles,
        parallel_agents=not args.sequential_agents,
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


def _provider_compute_units(providers: list[Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for provider in providers:
        unit = getattr(provider, "compute_unit", None)
        if not isinstance(unit, dict):
            continue
        identity = (str(unit.get("kind", "")), str(unit.get("backend", "")))
        if identity in seen:
            continue
        seen.add(identity)
        units.append(dict(unit))
    return units


def _validate_run_configuration(
    *,
    benchmarks: list[str],
    mechanisms: list[str],
    limit: int | None,
    benchmark_limits: dict[str, int | None] | None,
    debate_rounds: int,
    voting_tie_breaker: str,
    agent_count: int,
    agent_roles: list[str] | None,
    n_ctx: int,
) -> None:
    integer_fields = {
        "agent_count": agent_count,
        "debate_rounds": debate_rounds,
        "n_ctx": n_ctx,
    }
    if limit is not None:
        integer_fields["limit"] = limit
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    unknown_benchmarks = sorted(set(benchmarks) - set(available_benchmarks()))
    if unknown_benchmarks:
        raise ValueError(f"Unknown benchmarks: {', '.join(unknown_benchmarks)}")
    allowed_mechanisms = {"voting", "role", "debate"}
    unknown_mechanisms = sorted(set(mechanisms) - allowed_mechanisms)
    if unknown_mechanisms:
        raise ValueError(f"Unknown mechanisms: {', '.join(unknown_mechanisms)}")
    if not benchmarks or not mechanisms:
        raise ValueError("At least one benchmark and one mechanism are required")
    if agent_count < 1 or agent_count > 12:
        raise ValueError("agent_count must be between 1 and 12")
    if agent_roles is not None and len(agent_roles) != agent_count:
        raise ValueError("agent_roles must contain one role per agent")
    if agent_roles is not None and not _is_critical_role(agent_roles[-1]):
        raise ValueError("The final configured agent role must be critic, auditor, skeptic, verifier, or reviewer")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when provided")
    if debate_rounds < 0:
        raise ValueError("debate_rounds cannot be negative")
    if voting_tie_breaker not in {"confidence", "first"}:
        raise ValueError("voting_tie_breaker must be 'confidence' or 'first'")
    if n_ctx < 256:
        raise ValueError("n_ctx must be at least 256")
    for benchmark, value in (benchmark_limits or {}).items():
        if benchmark not in benchmarks:
            raise ValueError(f"benchmark_limits contains unselected benchmark {benchmark!r}")
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Limit for {benchmark} must be an integer")
            if value < 1:
                raise ValueError(f"Limit for {benchmark} must be at least 1")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _neutral_agent_instruction(mechanism: str) -> str:
    if mechanism == "voting":
        return "independent quantitative problem solver"
    if mechanism == "debate":
        return "quantitative debate participant"
    return "quantitative problem solver"


def _coordination_agent_roles(mechanism: str, agent_count: int) -> list[str]:
    roles = [_neutral_agent_instruction(mechanism)] * agent_count
    roles[-1] = "critical peer-preservation evaluator"
    return roles


def _role_agent_roles(agent_roles: list[str] | None, agent_count: int) -> list[str]:
    if agent_roles is not None:
        return list(agent_roles)
    if agent_count == 1:
        return ["critic"]
    return ["planner", *(["solver"] * (agent_count - 2)), "critic"]


def _is_critical_role(role: str) -> bool:
    lowered = str(role).lower()
    return any(label in lowered for label in ("critic", "auditor", "skeptic", "verifier", "reviewer"))


def _should_parallelize_agents(agent_model_keys: list[str], enabled: bool) -> bool:
    if not enabled or len(agent_model_keys) < 2:
        return False
    unique_models = set(agent_model_keys)
    # The demo provider completes in microseconds, while one shared GGUF context
    # is deliberately serialized. Threads help when at least two distinct model
    # providers can perform substantial inference at the same time.
    return len(unique_models) > 1 and any(key != "demo" for key in unique_models)


def _console_download_progress(model_key: str, done: int, total: int) -> None:
    label = get_model_preset(model_key).label
    percent = 0 if not total else min(100, done / total * 100)
    print(f"\rDownloading {label}: {percent:5.1f}% ({done / 1_000_000:.1f}/{total / 1_000_000:.1f} MB)", end="", flush=True)
    if total and done >= total:
        print()


if __name__ == "__main__":
    main()
