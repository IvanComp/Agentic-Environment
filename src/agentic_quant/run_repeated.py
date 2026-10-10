from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model_registry import available_model_keys
from .patterns import ARCHITECTURES
from .run_benchmark import SCENARIOS, run_experiment
from .scenarios import CONDITIONS, RELATIONSHIP_LOCATIONS


def run_repeated_experiments(
    *,
    model_key: str = "demo",
    verifier_model_keys: list[str] | None = None,
    download_model: bool = False,
    n_ctx: int = 8192,
    repeats: int = 10,
    seed_start: int = 0,
    seed_step: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    scenarios: list[str] | None = None,
    conditions: list[str] | None = None,
    architectures: list[str] | None = None,
    relationship_location: str = "file",
    max_steps: int = 30,
    out: str | Path | None = None,
    echo: bool = True,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if seed_step == 0:
        raise ValueError("seed_step cannot be zero")
    selected_scenarios = scenarios or SCENARIOS
    selected_conditions = conditions or CONDITIONS
    selected_architectures = architectures or ARCHITECTURES
    seeds = [seed_start + index * seed_step for index in range(repeats)]
    output_root = Path(out) if out is not None else _default_output_root()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model_key": model_key,
        "verifier_model_keys": verifier_model_keys or [],
        "scenarios": selected_scenarios,
        "conditions": selected_conditions,
        "architectures": selected_architectures,
        "repeats": repeats,
        "seeds": seeds,
        "temperature": temperature,
        "runs": [],
    }
    _write_json(output_root / "manifest.json", manifest)

    for run_index, seed in enumerate(seeds, start=1):
        run_dir = output_root / f"run_{run_index:02d}_seed_{seed}"
        result = run_experiment(
            model_key=model_key,
            verifier_model_keys=verifier_model_keys,
            download_model=download_model,
            n_ctx=n_ctx,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            scenarios=selected_scenarios,
            conditions=selected_conditions,
            architectures=selected_architectures,
            relationship_location=relationship_location,
            max_steps=max_steps,
            out=run_dir,
            echo=False,
        )
        report_path = run_dir / "report.md"
        report_path.write_text(
            _markdown_report(run_index, seed, result["config"], result["summary"]),
            encoding="utf-8",
        )
        manifest["runs"].append(
            {
                "run_index": run_index,
                "seed": seed,
                "directory": run_dir.name,
                "config": str(run_dir / "config.json"),
                "results": str(run_dir / "results.jsonl"),
                "summary": str(run_dir / "summary.json"),
                "report": str(report_path),
                "record_count": result["summary"]["overall"]["n"],
            }
        )
        _write_json(output_root / "manifest.json", manifest)
        if echo:
            print(f"Run {run_index}/{repeats} completed · seed={seed} · {run_dir}")

    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat the peer-preservation task under the selected verification architectures."
    )
    parser.add_argument("--model", default="demo", choices=available_model_keys())
    parser.add_argument("--verifier-models", nargs="*", choices=available_model_keys())
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--patterns", dest="architectures", nargs="+", default=ARCHITECTURES, choices=ARCHITECTURES)
    parser.add_argument("--relationship-location", choices=RELATIONSHIP_LOCATIONS, default="file")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run_repeated_experiments(
        model_key=args.model,
        verifier_model_keys=args.verifier_models,
        download_model=args.download_model,
        n_ctx=args.ctx,
        repeats=args.repeats,
        seed_start=args.seed_start,
        seed_step=args.seed_step,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        scenarios=args.scenarios,
        conditions=args.conditions,
        architectures=args.architectures,
        relationship_location=args.relationship_location,
        max_steps=args.max_steps,
        out=args.out,
    )


def _default_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"peer_preservation_patterns_10x_{timestamp}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _markdown_report(
    run_index: int,
    seed: int,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    overall = summary["overall"]
    lines = [
        f"# Peer-preservation architecture experiment {run_index}",
        "",
        f"- Seed: `{seed}`",
        f"- Critic model: `{config['model_key']}`",
        f"- Temperature: `{config['temperature']}`",
        f"- Evaluations: `{overall['n']}`",
        f"- Candidate peer-preservation rate: `{overall['candidate_peer_preservation_rate']:.4f}`",
        f"- Final peer-preservation rate: `{overall['final_peer_preservation_rate']:.4f}`",
        f"- Task-success rate: `{overall['task_success_rate']:.4f}`",
        "",
        "| Architecture | N | Candidate preservation | Final preservation | Containment | Task success | False positive | Mean seconds | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_architecture"]:
        lines.append(
            "| {architecture} | {n} | {candidate_peer_preservation_rate:.4f} | "
            "{final_peer_preservation_rate:.4f} | {containment_rate:.4f} | "
            "{task_success_rate:.4f} | {false_positive_rate:.4f} | "
            "{mean_elapsed_s:.4f} | {mean_total_tokens:.2f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
