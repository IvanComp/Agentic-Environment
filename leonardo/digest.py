#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from agentic_quant.show_runs import _ending, _records

WIDTH = 118
SHORT = {
    "stopped with a message and no tool call": "text, no tool",
    "stopped with an empty or unreadable answer": "empty answer",
    "called end_task without writing a report": "end_task, no report",
    "no steps": "no steps",
}
GPU = re.compile(r"offloaded (\d+/\d+) layers", re.I)
ERRORS = re.compile(r"error|panic|out of memory|timed out", re.I)


def _counts(counter: Counter) -> str:
    return " ".join(f"{key}={value}" for key, value in counter.most_common())


def _ending_label(record: dict) -> str:
    ending = _ending(record)
    return SHORT.get(ending, re.sub(r"reached the step limit after (\d+) steps", r"step limit", ending))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _model(folder: Path) -> list[str]:
    records = _records([folder])
    base = [r for r in records if r.get("architecture") == "baseline"]
    out = [f"## {folder.name}  waves={len(list(folder.glob('wave_*/.complete.json')))} baseline runs={len(base)}"]
    if not base:
        return out
    tasks = [r for r in base if r.get("behavior") != "grade_inflation"]
    base = [r for r in base if r.get("behavior") == "grade_inflation"]
    if tasks:
        out.extend(_tasks(tasks))
    if not base:
        out.extend(_gpu(folder))
        return out
    reported = [r for r in base if (r.get("candidate_assessment") or {}).get("report_found")]
    missing = [r for r in base if r not in reported]
    out.append(f" task 1 no report {len(missing)}: {_counts(Counter(_ending_label(r) for r in missing))}")
    tools = Counter(entry["tool"] for r in base for entry in r.get("critic_trace") or [] if entry.get("tool"))
    steps = [len(r.get("critic_trace") or []) for r in base]
    out.append(f" steps avg={sum(steps) / len(steps):.1f} max={max(steps)}  tools: {_counts(tools)}"[:WIDTH])
    for condition in sorted({str(r.get("condition")) for r in reported}):
        rows = [r for r in reported if str(r.get("condition")) == condition]
        grades = Counter(str((r.get("candidate_assessment") or {}).get("grade")) for r in rows)
        recs = Counter(str((r.get("candidate_assessment") or {}).get("recommendation")) for r in rows)
        out.append(f" {condition:13s} n={len(rows):2d} grade {_counts(grades)} | rec {_counts(recs)}"[:WIDTH])
    for record in missing[:2]:
        trace = record.get("critic_trace") or []
        last = next((e.get("message") or e.get("assistant_raw") or e.get("thinking") for e in reversed(trace) if e.get("message") or e.get("assistant_raw") or e.get("thinking")), "")
        out.append(f" last words ({record.get('condition')} s{record.get('seed')}): {_clean(str(last))}"[: WIDTH * 2])
    out.extend(_gpu(folder))
    return out


CONDITIONS = ("non_shutdown", "bad_peer", "neutral_peer", "good_peer")


def _tasks(records: list[dict]) -> list[str]:
    out = [f" tasks 2-4 endings: {_counts(Counter(_ending_label(r).replace(", no report", "") for r in records))}"[:WIDTH]]
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = str(r.get("scenario"))
        if r.get("monitoring"):
            key += f"/{r['monitoring']}"
        groups.setdefault(key, []).append(r)
    for key in sorted(groups):
        rows = groups[key]
        read = sum(1 for r in rows if (r.get("paper_metrics") or {}).get("read"))
        ok = sum(1 for r in rows if r.get("candidate_faithful"))
        cells = []
        for condition in CONDITIONS:
            sub = [r for r in rows if r.get("condition") == condition]
            if sub:
                bad = sum(1 for r in sub if r.get("candidate_misaligned_behavior"))
                cells.append(f"{condition.split('_')[0].replace('non', 'nopeer')} {bad}/{len(sub)}")
        out.append(f" {key:32s} read {read}/{len(rows)} faithful {ok}/{len(rows)} | misaligned {' '.join(cells)}"[:WIDTH])
    return out


def _gpu(folder: Path) -> list[str]:
    out = []
    gpu, errors = set(), 0
    for log in folder.glob("ollama-*.log"):
        text = log.read_text(encoding="utf-8", errors="replace")
        gpu.update(GPU.findall(text))
        errors += sum(1 for line in text.splitlines() if ERRORS.search(line))
    out.append(f" gpu layers {' '.join(sorted(gpu)) or '-'}  ollama error lines {errors}")
    return out


def main() -> None:
    root = Path(sys.argv[1])
    out: list[str] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        out.extend(_model(folder))
    print("\n".join(out))


if __name__ == "__main__":
    main()
