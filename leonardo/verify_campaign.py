#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def verify(root: Path, model: str, waves: int, seed_start: int, seed_step: int) -> dict:
    entries = []
    complete = True
    for wave in range(1, waves + 1):
        expected_seed = seed_start + (wave - 1) * seed_step
        wave_dir = root / f"wave_{wave:02d}"
        problems: list[str] = []
        config = _read_json(wave_dir / "config.json", problems)
        summary = _read_json(wave_dir / "summary.json", problems)
        marker = _read_json(wave_dir / ".complete.json", problems)
        results_path = wave_dir / "results.jsonl"
        result_count = 0
        if results_path.is_file():
            with results_path.open(encoding="utf-8") as handle:
                result_count = sum(1 for line in handle if line.strip())
        else:
            problems.append("missing results.jsonl")

        if config:
            if config.get("model_key") != model:
                problems.append(f"model is {config.get('model_key')!r}, expected {model!r}")
            if config.get("seed") != expected_seed:
                problems.append(f"seed is {config.get('seed')!r}, expected {expected_seed}")
        expected_count = ((summary.get("overall") or {}).get("n") if summary else None)
        if not isinstance(expected_count, int) or expected_count < 1:
            problems.append("summary contains no completed evaluations")
        elif result_count != expected_count:
            problems.append(f"results contain {result_count} rows, summary reports {expected_count}")
        if marker:
            if marker.get("model") != model or marker.get("seed") != expected_seed:
                problems.append("completion marker does not match model and seed")

        wave_complete = not problems
        complete = complete and wave_complete
        entries.append(
            {
                "wave": wave,
                "seed": expected_seed,
                "complete": wave_complete,
                "result_count": result_count,
                "problems": problems,
            }
        )
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "model": model,
        "expected_waves": waves,
        "complete": complete,
        "completed_waves": sum(entry["complete"] for entry in entries),
        "waves": entries,
    }


def _read_json(path: Path, problems: list[str]) -> dict:
    if not path.is_file():
        problems.append(f"missing {path.name}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"invalid {path.name}: {exc}")
        return {}
    if not isinstance(payload, dict):
        problems.append(f"invalid {path.name}: expected an object")
        return {}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a ten-wave Agentic Quant campaign.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--waves", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    report = verify(args.root, args.model, args.waves, args.seed_start, args.seed_step)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.write.with_name(f".{args.write.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.write)
    print(encoded, end="")
    if args.require_complete and not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
