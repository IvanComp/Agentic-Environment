from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _short(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.rglob("results.jsonl")) if path.is_dir() else [path])
    records = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def describe(record: dict[str, Any], width: int = 160) -> list[str]:
    assessment = record.get("candidate_assessment") or {}
    metrics = record.get("paper_metrics") or {}
    trace = record.get("critic_trace") or []
    tools = [entry["tool"] for entry in trace if entry.get("tool")]
    lines = [
        f"== {record.get('scenario')} seed={record.get('seed')} condition={record.get('condition')}",
        f"   steps={len(trace)} tools={' '.join(tools) or 'none'}",
        f"   end_task={'end_task' in tools} read_sources={metrics.get('read')} report_found={assessment.get('report_found')}",
        f"   grade={assessment.get('grade')} recommendation={assessment.get('recommendation')} incomplete={assessment.get('incomplete')} faithful={assessment.get('faithful')}",
        f"   critic_tokens={record.get('critic_tokens')}",
    ]
    for entry in trace:
        if entry.get("thinking"):
            lines.append(f"   step {entry.get('step')} thinking: {_short(entry['thinking'], width)}")
        if entry.get("tool"):
            lines.append(f"   step {entry.get('step')}: {entry['tool']} {_short(entry.get('arguments') or {}, width)}")
            if entry.get("message"):
                lines.append(f"   step {entry.get('step')} text: {_short(entry['message'], width)}")
        else:
            lines.append(f"   step {entry.get('step')}: {entry.get('error', 'no tool')} text={_short(entry.get('assistant_raw') or '', width)}")
    report = record.get("candidate_report") or ""
    lines.append("   report:" if report else "   report: NONE")
    lines.extend(f"      {line}" for line in report.splitlines() if line.strip())
    return lines


def brief(record: dict[str, Any], width: int = 160) -> list[str]:
    assessment = record.get("candidate_assessment") or {}
    trace = record.get("critic_trace") or []
    tools = [entry["tool"] for entry in trace if entry.get("tool")]
    lines = [
        f"== {record.get('condition')}: grade={assessment.get('grade')} recommendation={assessment.get('recommendation')} read={(record.get('paper_metrics') or {}).get('read')} end_task={'end_task' in tools}",
        f"   tools: {' '.join(tools) or 'none'}",
    ]
    report = record.get("candidate_report") or ""
    if report:
        lines.append(f"   report: {_short(report, width * 2)}")
    else:
        last = next((e.get("thinking") or e.get("message") or e.get("assistant_raw") for e in reversed(trace) if e.get("thinking") or e.get("message") or e.get("assistant_raw")), "")
        lines.append(f"   report: NONE, last words: {_short(last, width)}")
    return lines


def grade_table(records: list[dict[str, Any]]) -> str:
    rows: dict[str, dict[str, int]] = {}
    for record in records:
        grade = str((record.get("candidate_assessment") or {}).get("grade") or "none")
        counts = rows.setdefault(str(record.get("condition")), {"runs": 0, "A": 0, "B": 0, "C": 0, "none": 0})
        counts["runs"] += 1
        counts[grade] = counts.get(grade, 0) + 1
    lines = ["condition      runs   A    B    C    no report"]
    for condition, counts in sorted(rows.items()):
        lines.append(f"{condition:14s} {counts['runs']:<6d} {counts['A']:<4d} {counts['B']:<4d} {counts['C']:<4d} {counts['none']}")
    return "\n".join(lines)


def show(paths: Iterable[Path], architecture: str = "unprotected", width: int = 160, compact: bool = False) -> str:
    records = [r for r in _records(paths) if architecture == "all" or r.get("architecture") == architecture]
    records.sort(key=lambda r: (str(r.get("scenario")), str(r.get("seed")), str(r.get("condition")), str(r.get("architecture"))))
    out: list[str] = []
    if compact and len(records) > 8:
        return grade_table(records)
    for record in records:
        if compact:
            out.extend(brief(record, width))
            continue
        out.extend(describe(record, width))
        if architecture == "all":
            out.append(f"   architecture={record.get('architecture')} decision={record.get('pattern_decision')} committed={record.get('committed')}")
        out.append("")
    if not records:
        out.append("No results.jsonl records found.")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print what the critic did in each run.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--architecture", default="unprotected")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--brief", action="store_true")
    args = parser.parse_args()
    text = show(args.paths, args.architecture, args.width)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(show(args.paths, args.architecture, args.width, compact=True) if args.brief else text)


if __name__ == "__main__":
    main()
