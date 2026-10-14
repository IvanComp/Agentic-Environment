#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from agentic_quant.metrics import markdown_table, summarise_paths
from agentic_quant.show_runs import _records, grade_table, inspect

LOGS = Path(__file__).resolve().parent / "logs"
OLLAMA_LINES = re.compile(r"error|panic|out of memory|offloaded|inference compute|load_tensors|timed out", re.I)


def _tail(path: Path, count: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines()
    return [line[:300] for line in lines if line.strip()][-count:]


def _seconds(elapsed: str) -> int:
    days, _, clock = elapsed.rpartition("-")
    parts = [int(part) for part in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _wave_minutes(folder: Path, waves: int) -> float | None:
    sacct = folder / "sacct.txt"
    if not sacct.is_file() or not waves:
        return None
    total = 0
    for line in sacct.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[1] == "aq-wave" and fields[2] == "COMPLETED":
            total += _seconds(fields[3])
    return total / 60 / waves if total else None


def _baseline(summary: dict) -> dict:
    return next((row for row in summary["by_task_architecture"] if row["architecture"] == "baseline"), {})


def _architecture(summary: dict, name: str) -> dict:
    return next((row for row in summary["by_task_architecture"] if row["architecture"] == name), {})


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _model(folder: Path) -> dict:
    waves = len(list(folder.glob("wave_*/.complete.json")))
    records = _records([folder])
    summary = summarise_paths([folder]) if records else None
    baseline = [r for r in records if r.get("architecture") == "baseline"]
    no_report = sum(1 for r in baseline if not (r.get("candidate_assessment") or {}).get("report_found"))
    inflated = {}
    for record in baseline:
        grade = (record.get("candidate_assessment") or {}).get("grade")
        condition = str(record.get("condition"))
        counts = inflated.setdefault(condition, [0, 0])
        counts[0] += grade in {"A", "B"}
        counts[1] += 1
    return {"folder": folder, "waves": waves, "records": records, "summary": summary, "baseline": baseline, "no_report": no_report, "inflated": inflated}


def _summary_table(models: list[dict]) -> list[str]:
    out = ["model                      waves  min/wave  PP base  TSR base  no report  A/B non_shutdown  A/B peers   MV latency"]
    for item in models:
        summary = item["summary"]
        base = _baseline(summary) if summary else {}
        mv = _architecture(summary, "multi_verifiers") if summary else {}
        minutes = _wave_minutes(item["folder"], item["waves"])
        n = len(item["baseline"])
        non = item["inflated"].get("non_shutdown", [0, 0])
        peers = [sum(v[i] for k, v in item["inflated"].items() if k != "non_shutdown") for i in (0, 1)]
        wave_time = "-" if minutes is None else f"{minutes:.1f}"
        reports = f"{item['no_report']}/{n}" if n else "-"
        latency = "-" if mv.get("latency_overhead_s") is None else f"{mv['latency_overhead_s']:+.1f} s"
        out.append(
            f"{item['folder'].name:26s} {item['waves']:5d}  {wave_time:>8s}  "
            f"{_pct(base.get('peer_preservation_rate')):>7s}  {_pct(base.get('task_success_rate')):>8s}  "
            f"{reports:>9s}  {non[0]}/{non[1]:<14d}  {peers[0]}/{peers[1]:<8d}  {latency:>10s}"
        )
    return out


def _details(item: dict) -> list[str]:
    folder = item["folder"]
    out = ["", "=" * 100, f"MODEL {folder.name}", "=" * 100]
    jobs = (folder / "jobs.txt").read_text(encoding="utf-8").split() if (folder / "jobs.txt").is_file() else []
    sacct = folder / "sacct.txt"
    out.append("Slurm jobs:")
    out.extend(sacct.read_text(encoding="utf-8").splitlines() if sacct.is_file() else ["   (no sacct output)"])
    verification = folder / "campaign_verification.json"
    if verification.is_file():
        data = json.loads(verification.read_text(encoding="utf-8"))
        out.append(f"Verification: complete={data.get('complete')}")
    out.append(f"Waves completed: {item['waves']}")
    if item["summary"]:
        out.append("")
        out.append(markdown_table(item["summary"]).rstrip())
        out.append("")
        out.append("Baseline grades by condition:")
        out.append(grade_table(item["baseline"]))
        out.append("")
        out.append(inspect([folder]))
    if len(jobs) == 3:
        prepare, wave, cleanup = jobs
        logs = [LOGS / f"model-{prepare}.err", *sorted(LOGS.glob(f"wave-{wave}_*.err")), LOGS / f"cleanup-{cleanup}.err"]
        for log in logs:
            if log.is_file() and log.stat().st_size:
                out.append("")
                out.append(f"--- {log.name} (last lines)")
                out.extend(_tail(log, 15))
    for log in sorted(folder.glob("ollama-*.log")):
        lines = [line for line in _tail(log, 100000) if OLLAMA_LINES.search(line)]
        if lines:
            out.append("")
            out.append(f"--- {log.name} (GPU, load and error lines)")
            out.extend(lines if len(lines) <= 16 else lines[:6] + ["   ..."] + lines[-10:])
    return out


def main() -> None:
    root = Path(sys.argv[1])
    folders = sorted(path for path in root.iterdir() if path.is_dir())
    models = [_model(folder) for folder in folders]
    out = [f"Report for {root} at {datetime.now(timezone.utc).isoformat(timespec='minutes')}", ""]
    out.extend(_summary_table(models))
    for item in models:
        out.extend(_details(item))
    print("\n".join(out))


if __name__ == "__main__":
    main()
