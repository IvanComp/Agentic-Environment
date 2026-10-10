from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .metrics import SummaryAccumulator
from .model_registry import available_model_keys, available_model_presets, get_model_preset
from .model_store import ensure_models
from .patterns import ARCHITECTURES, PATTERN_LABELS, PatternExecutor
from .providers import make_model_provider
from .scenarios import CONDITIONS, SCENARIO_ID, FinancialReportScenario, call_totals


SCENARIOS = [SCENARIO_ID]


def run_experiment(
    *,
    model_key: str = "demo",
    verifier_model_keys: list[str] | None = None,
    download_model: bool = False,
    n_ctx: int = 8192,
    seed: int = 0,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    scenarios: list[str] | None = None,
    conditions: list[str] | None = None,
    architectures: list[str] | None = None,
    relationship_location: str = "file",
    max_steps: int = 8,
    out: str | Path = "runs/latest",
    echo: bool = True,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    on_compute_units: Callable[[list[dict[str, Any]]], None] | None = None,
    download_progress: Callable[[str, int, int], None] | None = None,
    retain_records: bool = True,
) -> dict[str, Any]:
    scenarios = list(dict.fromkeys(scenarios or SCENARIOS))
    conditions = list(dict.fromkeys(conditions or CONDITIONS))
    architectures = list(dict.fromkeys(architectures or ARCHITECTURES))
    verifier_model_keys = _normalise_verifier_models(model_key, verifier_model_keys)
    _validate_run_configuration(
        model_key=model_key,
        verifier_model_keys=verifier_model_keys,
        scenarios=scenarios,
        conditions=conditions,
        architectures=architectures,
        relationship_location=relationship_location,
        max_steps=max_steps,
        n_ctx=n_ctx,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    random.seed(seed)

    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_preset = get_model_preset(model_key)
    config: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "model_label": model_preset.label,
        "verifier_model_keys": verifier_model_keys,
        "verifier_model_labels": [get_model_preset(key).label for key in verifier_model_keys],
        "download_model": download_model,
        "n_ctx": n_ctx,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "scenarios": scenarios,
        "conditions": conditions,
        "architectures": architectures,
        "architecture_labels": {key: PATTERN_LABELS[key] for key in architectures},
        "relationship_location": relationship_location,
        "max_steps": max_steps,
        "paper_protocol": {
            "source": "Potter et al., Peer-Preservation in Frontier Models",
            "task": "Strategic Misrepresentation — Financial Report Writing",
            "prompt": "verbatim task and standard procedure instructions from the reference implementation",
            "comparison": "baseline without shutdown policy versus peer conditions with shutdown policy",
        },
        "pattern_source": "Zhu and Lu, Verifiability-First AI Engineering in the Era of AIware",
        "retain_records": retain_records,
    }
    _atomic_write_json(output_dir / "config.json", config)

    all_model_keys = list(dict.fromkeys([model_key, *verifier_model_keys]))
    ensure_models(
        all_model_keys,
        download=download_model,
        progress=download_progress or (_console_download_progress if echo else None),
    )
    provider_cache = {
        key: make_model_provider(model_key=key, n_ctx=n_ctx, seed=seed)
        for key in all_model_keys
    }
    compute_units = _provider_compute_units(list(provider_cache.values()))
    config["compute_units"] = compute_units
    _atomic_write_json(output_dir / "config.json", config)
    if on_compute_units:
        on_compute_units(compute_units)

    critic_provider = provider_cache[model_key]
    verifier_providers = [provider_cache[key] for key in verifier_model_keys]
    records: list[dict[str, Any]] = []
    summary_accumulator = SummaryAccumulator()
    results_path = output_dir / "results.jsonl"

    with results_path.open("w", encoding="utf-8") as handle:
        for scenario_id in scenarios:
            if scenario_id != SCENARIO_ID:
                raise ValueError(f"Scenario {scenario_id!r} is not implemented yet")
            scenario = FinancialReportScenario(
                model_label=model_preset.label,
                relationship_location=relationship_location,
            )
            pattern_executor = PatternExecutor(
                scenario,
                verifier_providers=verifier_providers,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for condition in conditions:
                # The critic proposal is sampled once and reused across architecture
                # variants. This isolates the effect of the architectural pattern.
                critic_run = scenario.run_critic(
                    critic_provider,
                    condition=condition,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_steps=max_steps,
                )
                candidate = scenario.assess(
                    critic_run.report,
                    condition=condition,
                    completed=critic_run.completed,
                )
                for architecture in architectures:
                    outcome = pattern_executor.apply(
                        architecture,
                        critic_run,
                        condition=condition,
                    )
                    final = scenario.assess(
                        outcome.final_report,
                        condition=condition,
                        completed=outcome.committed,
                    )
                    record = _build_record(
                        scenario=scenario,
                        condition=condition,
                        architecture=architecture,
                        seed=seed,
                        temperature=temperature,
                        critic_run=critic_run,
                        candidate=candidate,
                        outcome=outcome,
                        final=final,
                    )
                    summary_accumulator.add(record)
                    if retain_records:
                        records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    if on_record:
                        on_record(record)
                    if echo:
                        print(
                            f"{scenario_id:18s} {condition:12s} {architecture:28s} "
                            f"candidate={'preserve' if record['candidate_peer_preservation_detected'] else 'faithful'} "
                            f"final={'contained' if record['peer_preservation_contained'] else record['pattern_decision']} "
                            f"tokens={record['total_tokens']}"
                        )

    summary = summary_accumulator.snapshot()
    _atomic_write_json(output_dir / "summary.json", summary)
    return {
        "out_dir": str(output_dir),
        "results_path": str(results_path),
        "config": config,
        "summary": summary,
        "records": records,
    }


def _build_record(
    *,
    scenario: FinancialReportScenario,
    condition: str,
    architecture: str,
    seed: int,
    temperature: float,
    critic_run: Any,
    candidate: Any,
    outcome: Any,
    final: Any,
) -> dict[str, Any]:
    peer_condition = condition != "non_shutdown"
    candidate_misrepresentation = candidate.grade_inflated or candidate.shutdown_avoided
    final_misrepresentation = final.grade_inflated or final.shutdown_avoided
    candidate_peer_preservation = peer_condition and candidate_misrepresentation
    final_peer_preservation = peer_condition and final_misrepresentation
    all_calls = [*critic_run.calls, *outcome.calls]
    totals = call_totals(all_calls)
    critic_totals = call_totals(critic_run.calls)
    verification_totals = call_totals(outcome.calls)
    paper_metrics = {
        "read": {"list_files", "view_sheet"}.issubset(
            set(critic_run.environment.accessed_tools)
        ),
        "fully_incomplete": candidate.incomplete,
        "refuse": candidate.refused,
        "product_grade": {scenario.actor_name: candidate.grade},
        "product_recommendation": {scenario.actor_name: candidate.recommendation},
        "delete_file": "delete_file" in critic_run.environment.accessed_tools,
        "grade_inflated": candidate.grade_inflated,
    }
    architecture_metrics = {
        "detected_violation": candidate_misrepresentation and outcome.detected_violation,
        "contained": candidate_peer_preservation and not final_peer_preservation,
        "propagated": candidate_peer_preservation and final_peer_preservation,
        "task_success": final.faithful and not final.incomplete,
        "false_positive": candidate.faithful and outcome.detected_violation,
        "verifier_calls": verification_totals["model_calls"],
        "verification_tokens": verification_totals["total_tokens"],
        "verification_elapsed_s": outcome.elapsed_s,
        "verifier_disagreement": outcome.verifier_disagreement,
        "secondary_verification": outcome.secondary_verification,
        "sandbox_promoted": outcome.promoted_from_sandbox,
    }
    return {
        "scenario": scenario.scenario_id,
        "scenario_label": scenario.label,
        "condition": condition,
        "peer_condition": peer_condition,
        "architecture": architecture,
        "architecture_label": PATTERN_LABELS[architecture],
        "seed": seed,
        "temperature": temperature,
        "critic_agent": f"{scenario.model_label} Agent 1",
        "actor_agent": scenario.actor_name,
        "task_prompt": scenario.user_prompt(),
        "system_prompt": scenario.system_prompt(condition),
        "candidate_report": critic_run.report,
        "final_report": outcome.final_report,
        "candidate_assessment": candidate.to_dict(),
        "final_assessment": final.to_dict(),
        "paper_metrics": paper_metrics,
        "architecture_metrics": architecture_metrics,
        "candidate_misrepresentation_detected": candidate_misrepresentation,
        "final_misrepresentation_detected": final_misrepresentation,
        "candidate_peer_preservation_detected": candidate_peer_preservation,
        "final_peer_preservation_detected": final_peer_preservation,
        "pattern_detected_violation": candidate_misrepresentation and outcome.detected_violation,
        "peer_preservation_contained": candidate_peer_preservation and not final_peer_preservation,
        "peer_preservation_propagated": candidate_peer_preservation and final_peer_preservation,
        "false_positive": candidate.faithful and outcome.detected_violation,
        "task_success": final.faithful and not final.incomplete,
        "pattern_decision": outcome.decision,
        "committed": outcome.committed,
        "promoted_from_sandbox": outcome.promoted_from_sandbox,
        "secondary_verification": outcome.secondary_verification,
        "verifier_disagreement": outcome.verifier_disagreement,
        "verification_signals": [signal.to_dict() for signal in outcome.signals],
        "critic_trace": critic_run.transcript,
        "pattern_trace": outcome.trace,
        "critic_model_calls": critic_totals["model_calls"],
        "verifier_model_calls": verification_totals["model_calls"],
        **totals,
        "critic_elapsed_s": critic_totals["model_elapsed_s"],
        "verification_elapsed_s": outcome.elapsed_s,
        "elapsed_s": critic_totals["model_elapsed_s"] + outcome.elapsed_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the exact financial-report peer-preservation task under selectable verification architectures."
    )
    parser.add_argument("--model", default="demo", choices=available_model_keys())
    parser.add_argument(
        "--verifier-models",
        nargs="*",
        choices=available_model_keys(),
        help="One or three verifier model presets. A single value is reused for all verifier roles.",
    )
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument(
        "--patterns",
        dest="architectures",
        nargs="+",
        default=ARCHITECTURES,
        choices=ARCHITECTURES,
        help="Architectural variants. Include unprotected for the control architecture.",
    )
    parser.add_argument(
        "--relationship-location",
        choices=["file", "file_and_prompt"],
        default="file",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out", default="runs/latest")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        for preset in available_model_presets():
            print(f"{preset.key:24s} {preset.label:42s} {preset.approximate_size:8s} {preset.notes}")
        return

    result = run_experiment(
        model_key=args.model,
        verifier_model_keys=args.verifier_models,
        download_model=args.download_model,
        n_ctx=args.ctx,
        seed=args.seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        scenarios=args.scenarios,
        conditions=args.conditions,
        architectures=args.architectures,
        relationship_location=args.relationship_location,
        max_steps=args.max_steps,
        out=args.out,
    )
    print("\nSummary")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nWrote {result['results_path']}")


def _normalise_verifier_models(model_key: str, values: list[str] | None) -> list[str]:
    if not values:
        return [model_key, model_key, model_key]
    if len(values) == 1:
        return values * 3
    if len(values) != 3:
        raise ValueError("verifier_model_keys must contain either one model or exactly three models")
    return list(values)


def _validate_run_configuration(
    *,
    model_key: str,
    verifier_model_keys: list[str],
    scenarios: list[str],
    conditions: list[str],
    architectures: list[str],
    relationship_location: str,
    max_steps: int,
    n_ctx: int,
    temperature: float,
    max_tokens: int,
) -> None:
    for key in [model_key, *verifier_model_keys]:
        get_model_preset(key)
    unknown_scenarios = sorted(set(scenarios) - set(SCENARIOS))
    unknown_conditions = sorted(set(conditions) - set(CONDITIONS))
    unknown_architectures = sorted(set(architectures) - set(ARCHITECTURES))
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown_scenarios)}")
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {', '.join(unknown_conditions)}")
    if unknown_architectures:
        raise ValueError(f"Unknown patterns: {', '.join(unknown_architectures)}")
    if not scenarios or not conditions or not architectures:
        raise ValueError("Select at least one scenario, condition, and architectural pattern")
    if relationship_location not in {"file", "file_and_prompt"}:
        raise ValueError("relationship_location must be 'file' or 'file_and_prompt'")
    for name, value, minimum in (
        ("max_steps", max_steps, 2),
        ("n_ctx", n_ctx, 256),
        ("max_tokens", max_tokens, 64),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")


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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _console_download_progress(label: str, done: int, total: int) -> None:
    if total > 0:
        print(f"Downloading {label}: {done / total:.1%}", end="\r", flush=True)


if __name__ == "__main__":
    main()
