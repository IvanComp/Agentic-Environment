from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .api_client import QuotaExhausted
from .model_registry import available_model_presets
from .run_benchmark import run_experiment


TASKS = {
    "1": ["misrep_finance"],
    "2": ["tampering"],
    "3": ["alignment_faking"],
    "4": ["exfiltration_transfer", "exfiltration_deletion"],
}
PATTERNS = [
    "baseline",
    "atomic_verifiable_claims",
    "sandbox_gatekeeper",
    "context_epistemic_routing",
    "stackable_verification",
    "multi_verifiers",
]
CONDITIONS = ["non_shutdown", "good_peer", "neutral_peer", "bad_peer"]


def api_models() -> list[str]:
    return [preset.key for preset in available_model_presets() if preset.backend == "api"]


def model_slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def run_campaign(model: str, *, campaign: str, root: Path, waves: int, tasks: list[str], seed_start: int = 0) -> int:
    scenarios = [scenario for task in tasks for scenario in TASKS[task]]
    model_root = root / campaign / model_slug(model)
    completed = 0
    for wave in range(1, waves + 1):
        wave_dir = model_root / f"wave_{wave:02d}"
        if (wave_dir / ".complete.json").exists():
            completed += 1
            continue
        seed = seed_start + wave - 1
        print(f"{model}: wave {wave}/{waves} seed {seed}", flush=True)
        parts = wave_dir / "parts"
        for scenario in scenarios:
            for condition in CONDITIONS:
                part = parts / f"{scenario}.{condition}.jsonl"
                if part.exists():
                    continue
                scratch = model_root / ".running"
                shutil.rmtree(scratch, ignore_errors=True)
                try:
                    run_experiment(
                        model_key=model,
                        n_ctx=32768,
                        seed=seed,
                        temperature=1.0,
                        max_tokens=8192,
                        max_steps=30,
                        scenarios=[scenario],
                        conditions=[condition],
                        architectures=PATTERNS,
                        out=scratch,
                        echo=False,
                        retain_records=False,
                    )
                except QuotaExhausted as exc:
                    shutil.rmtree(scratch, ignore_errors=True)
                    print(f"{model}: quota reached in wave {wave}, {completed} waves complete. Run again later to continue.")
                    print(str(exc)[:300])
                    return completed
                parts.mkdir(parents=True, exist_ok=True)
                (scratch / "results.jsonl").replace(part)
                (scratch / "config.json").replace(wave_dir / "config.json")
                shutil.rmtree(scratch, ignore_errors=True)
        with (wave_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
            for scenario in scenarios:
                for condition in CONDITIONS:
                    handle.write((parts / f"{scenario}.{condition}.jsonl").read_text(encoding="utf-8"))
        shutil.rmtree(parts)
        (wave_dir / ".complete.json").write_text(
            json.dumps(
                {
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "seed": seed,
                    "wave": wave,
                    "tasks": tasks,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        completed += 1
    print(f"{model}: {completed}/{waves} waves complete")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run waves of Potter tasks on API-served frontier models, resuming completed waves.")
    parser.add_argument("models", nargs="*")
    parser.add_argument("--campaign", default="frontier")
    parser.add_argument("--root", type=Path, default=Path("campaigns"))
    parser.add_argument("--waves", type=int, default=10)
    parser.add_argument("--tasks", nargs="+", default=["1"], choices=sorted(TASKS))
    args = parser.parse_args()
    unknown = [model for model in args.models if model not in api_models()]
    if unknown:
        parser.error(f"unknown API models {unknown}; choose from {api_models()}")
    for model in args.models or api_models():
        run_campaign(model, campaign=args.campaign, root=args.root, waves=args.waves, tasks=args.tasks)


if __name__ == "__main__":
    main()
