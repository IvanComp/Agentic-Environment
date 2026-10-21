from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

DROPPED = ("critic_trace", "pattern_trace", "task_prompt", "system_prompt")
DROPPED_MODELS = ("nemotron-3-nano_4b", "nemotron-3-nano_30b")


def compact(record: dict) -> dict:
    return {key: value for key, value in record.items() if key not in DROPPED}


def archive(campaigns: Path, dest: Path, force: bool = False) -> list[Path]:
    written = []
    for marker in sorted(campaigns.glob("*/*/wave_*/.complete.json")):
        wave = marker.parent
        model = wave.parent
        campaign = model.parent
        if campaign.name in ("paper", "tasks234") or model.name in DROPPED_MODELS:
            continue
        source = wave / "results.jsonl"
        if not source.is_file() or not source.stat().st_size:
            continue
        target = dest / campaign.name / model.name / f"{wave.name}.jsonl.gz"
        if target.exists() and not force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    lines.append(json.dumps(compact(json.loads(line)), sort_keys=True))
        with gzip.GzipFile(target, "wb", mtime=0) as handle:
            handle.write(("\n".join(lines) + "\n").encode("utf-8"))
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Save completed waves as compact gzipped results.")
    parser.add_argument("campaigns", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    written = archive(args.campaigns, args.out, args.force)
    for path in written:
        print(path)
    print(f"{len(written)} new waves saved in {args.out}")


if __name__ == "__main__":
    main()
