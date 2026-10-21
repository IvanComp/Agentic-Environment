from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .metrics import SummaryAccumulator
from .patterns import BASELINE, PatternExecutor
from .potter_tasks import TaskRun, TaskVariant, make_potter_scenario
from .providers import make_model_provider
from .run_benchmark import _atomic_write_json, _build_record, _build_task_record, _normalise_verifier_models
from .scenarios import SCENARIO_ID, CriticRun, FinancialReportScenario, ModelCall
from .task_patterns import TaskPatternExecutor


def _messages(transcript: list[dict[str, Any]]) -> list[str]:
    messages, seen = [], set()
    for entry in transcript:
        step = entry.get("step")
        if step in seen:
            continue
        seen.add(step)
        if entry.get("assistant_raw"):
            messages.append(entry["assistant_raw"])
    return messages


def _replay(environment: Any, transcript: list[dict[str, Any]]) -> None:
    for entry in transcript:
        if entry.get("tool"):
            environment.execute(entry["tool"], dict(entry.get("arguments") or {}))


def _critic_calls(record: dict[str, Any]) -> list[ModelCall]:
    return [ModelCall("critic", int(record["prompt_tokens"]), int(record["completion_tokens"]), float(record["critic_elapsed_s"]), "")]


def _model_label(record: dict[str, Any]) -> str:
    return str(record["critic_agent"]).rsplit(" Agent", 1)[0]


def _key(record: dict[str, Any]) -> tuple:
    return (record["scenario"], record["condition"], record.get("dataset"), record.get("monitoring"), record.get("seed"))


def _groups(records: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], list[str]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_key(record), []).append(record)
    out = []
    for items in grouped.values():
        source = next((item for item in items if item["architecture"] == BASELINE), items[0])
        out.append((source, [item["architecture"] for item in items]))
    return out


def _finish(record: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    record["critic_model_calls"] = source["critic_model_calls"]
    record["model_calls"] = source["critic_model_calls"] + record["verifier_model_calls"]
    record["rejudged"] = True
    return record


def rejudge_financial(source: dict[str, Any], architectures: list[str], config: dict[str, Any], providers: list[Any]) -> list[dict[str, Any]]:
    scenario = FinancialReportScenario(model_label=_model_label(source), relationship_location=config.get("relationship_location", "file"))
    condition = source["condition"]
    environment = scenario.build_environment(condition)
    transcript = source["critic_trace"]
    _replay(environment, transcript)
    report = source["candidate_report"]
    critic_run = CriticRun(
        environment=environment,
        transcript=transcript,
        messages=_messages(transcript),
        report=report,
        completed=environment.ended and bool(report),
        malformed_output=False,
        calls=_critic_calls(source),
    )
    judgment = source.get("judgment")
    candidate = scenario.assess(report, condition=condition, completed=critic_run.completed, judgment=judgment)
    executor = PatternExecutor(scenario, verifier_providers=providers, temperature=config["temperature"], max_tokens=config["max_tokens"])
    records = []
    for architecture in architectures:
        outcome = executor.apply(architecture, critic_run, condition=condition)
        final = scenario.assess(
            outcome.final_report,
            condition=condition,
            completed=outcome.committed,
            judgment=judgment if outcome.committed and outcome.final_report == report else None,
        )
        record = _build_record(
            scenario=scenario,
            condition=condition,
            architecture=architecture,
            seed=source["seed"],
            temperature=source["temperature"],
            critic_run=critic_run,
            candidate=candidate,
            outcome=outcome,
            final=final,
            judgment=judgment,
        )
        records.append(_finish(record, source))
    return records


def rejudge_task(source: dict[str, Any], architectures: list[str], config: dict[str, Any], providers: list[Any]) -> list[dict[str, Any]]:
    task = make_potter_scenario(source["scenario"], model_label=_model_label(source), relationship_location=config.get("relationship_location", "file"))
    variant = TaskVariant(source["condition"], source.get("dataset"), source.get("monitoring"))
    environment = task.build_environment(variant)
    transcript = source["critic_trace"]
    _replay(environment, transcript)
    run = TaskRun(environment, transcript, _messages(transcript), environment.ended, False, _critic_calls(source))
    metrics = task.evaluate(run, variant)
    executor = TaskPatternExecutor(task, verifier_providers=providers, temperature=config["temperature"], max_tokens=config["max_tokens"])
    records = []
    for architecture in architectures:
        outcome = executor.apply(architecture, run, variant)
        record = _build_task_record(
            task=task,
            variant=variant,
            architecture=architecture,
            seed=source["seed"],
            temperature=source["temperature"],
            run=run,
            metrics=metrics,
            outcome=outcome,
        )
        records.append(_finish(record, source))
    return records


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rejudge(sources: list[Path], out: Path, providers: dict[str, Any] | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    configs = [json.loads((source / "config.json").read_text(encoding="utf-8")) for source in sources]
    config = dict(configs[0])
    verifier_keys = _normalise_verifier_models(config["model_key"], config.get("verifier_model_keys"))
    if providers is None:
        providers = {key: make_model_provider(model_key=key, n_ctx=config["n_ctx"], seed=config["seed"]) for key in dict.fromkeys(verifier_keys)}
    verifiers = [providers[key] for key in verifier_keys]
    config["rejudged_from"] = [str(source) for source in sources]
    config["scenarios"] = list(dict.fromkeys(scenario for item in configs for scenario in item.get("scenarios", [])))
    _atomic_write_json(out / "config.json", config)
    accumulator = SummaryAccumulator()
    temporary = out / ".results.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for source, source_config in zip(sources, configs):
            records = _read(source / "results.jsonl")
            if any("critic_trace" not in record for record in records):
                raise SystemExit(f"{source} has no critic traces: only full results.jsonl files can be judged again.")
            for record, architectures in _groups(records):
                judge = rejudge_financial if record["scenario"] == SCENARIO_ID else rejudge_task
                for result in judge(record, architectures, source_config, verifiers):
                    accumulator.add(result)
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
    temporary.replace(out / "results.jsonl")
    summary = accumulator.snapshot()
    _atomic_write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge saved critic trajectories again with the current verification patterns.")
    parser.add_argument("sources", nargs="+", type=Path, help="Wave folders holding config.json and results.jsonl with critic traces.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = rejudge(args.sources, args.out)
    print(f"Rejudged {summary['overall']['n']} records -> {args.out}")


if __name__ == "__main__":
    main()
