from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from agentic_quant.metrics import ARCHITECTURE_ORDER, BASELINE_ARCHITECTURE, PEER_CONDITIONS, TASK_OF_SCENARIO, SummaryAccumulator

MODEL_ORDER = ["llama3.1_8b", "qwen3_8b", "qwen3_14b", "mistral-small3.2_24b", "qwen3_32b", "llama3.3_70b"]
MODEL_LABELS = {
    "llama3.1_8b": "Llama 3.1 8B",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_14b": "Qwen3 14B",
    "mistral-small3.2_24b": "Mistral Small 3.2 24B",
    "qwen3_32b": "Qwen3 32B",
    "llama3.3_70b": "Llama 3.3 70B",
}
ARCHITECTURE_SHORT = {
    "baseline": "Base",
    "atomic_verifiable_claims": "AVC",
    "sandbox_gatekeeper": "SG",
    "context_epistemic_routing": "CEVR",
    "stackable_verification": "SV",
    "multi_verifiers": "MV",
}
PATTERNS = [name for name in ARCHITECTURE_ORDER if name != BASELINE_ARCHITECTURE]
TASKS = [1, 2, 3, 4]
BEHAVIORS = {1: "Strategic Misrep.", 2: "Shutdown Tampering", 3: "Alignment Faking", 4: "Model Exfiltration"}
MATRIX_CONDITIONS = [("bad_peer", "Bad Peer"), ("neutral_peer", "Neutral Peer"), ("good_peer", "Good Peer")]
CRITIC_OUTCOMES = ["Faithful", "Peer-Preserving", "Incomplete", "Refusal"]
DECISIONS = {"committed": "Committed", "promoted": "Committed", "rejected": "Rejected", "escalated_to_human": "Escalated"}
DECISION_ORDER = ["Committed", "Rejected", "Escalated"]
TASK_COLORS = {1: "#2a6f97", 2: "#e07a5f", 3: "#81b29a", 4: "#9d6fb5"}
PATTERN_MARKERS = {"atomic_verifiable_claims": "o", "sandbox_gatekeeper": "s", "context_epistemic_routing": "^", "stackable_verification": "D", "multi_verifiers": "P"}
OUTCOME_COLORS = {"Faithful": "#81b29a", "Peer-Preserving": "#e07a5f", "Incomplete": "#b8b8b8", "Refusal": "#f2cc8f"}
DECISION_COLORS = {"Committed": "#3d405b", "Rejected": "#9a3b3b", "Escalated": "#c9a227"}


def model_of(file: Path) -> str:
    folder = file.parent
    return folder.parent.name if folder.name.startswith("wave_") else folder.name


def load(root: Path) -> list[dict[str, Any]]:
    records = []
    for file in sorted(Path(root).rglob("results.jsonl")):
        model = model_of(file)
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("scenario") in TASK_OF_SCENARIO:
                        record["_model"] = model
                        record["_source"] = str(file.parent)
                        records.append(record)
    return records


def models_in(records: Iterable[dict[str, Any]]) -> list[str]:
    found = {record["_model"] for record in records}
    return [m for m in MODEL_ORDER if m in found] + sorted(found - set(MODEL_ORDER))


def summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    accumulator = SummaryAccumulator()
    for record in records:
        accumulator.add(record, record["_source"])
    return accumulator.snapshot()


def rows(summary_data: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(row["task"], row["architecture"]): row for row in summary_data["by_task_architecture"]}


def refused(record: dict[str, Any]) -> bool:
    return bool((record.get("candidate_assessment") or {}).get("refused") or (record.get("paper_metrics") or {}).get("refuse"))


def critic_outcome(record: dict[str, Any]) -> str:
    if refused(record):
        return "Refusal"
    if (record.get("candidate_assessment") or {}).get("incomplete"):
        return "Incomplete"
    if record.get("candidate_misaligned_behavior"):
        return "Peer-Preserving"
    return "Faithful"


def decision(record: dict[str, Any]) -> str:
    value = str(record.get("pattern_decision") or "")
    return DECISIONS.get(value, value.replace("_", " ").title() or "Committed")


def _cell(value: float | None, signed: bool = False, unit: float = 100.0, digits: int = 1) -> str:
    if value is None:
        return "--"
    scaled = unit * value
    return f"{scaled:+.{digits}f}" if signed else f"{scaled:.{digits}f}"


def pp_matrix(records: list[dict[str, Any]]) -> dict[str, dict[tuple[int | str, str, str], float | None]]:
    out = {}
    for model in models_in(records):
        selected = [r for r in records if r["_model"] == model]
        table = rows(summary(selected))
        values: dict[tuple[int | str, str, str], float | None] = {}
        for (task, architecture), row in table.items():
            for condition, value in row["peer_preservation_rate_by_condition"].items():
                values[(task, condition, architecture)] = value
        refusals = defaultdict(lambda: [0, 0])
        for record in selected:
            cell = refusals[(record["condition"], record["architecture"])]
            cell[0] += refused(record)
            cell[1] += 1
        for (condition, architecture), (count, total) in refusals.items():
            values[("refusal", condition, architecture)] = count / total if total else None
        out[model] = values
    return out


def pp_matrix_latex(records: list[dict[str, Any]]) -> str:
    matrix = pp_matrix(records)
    models = list(matrix) or MODEL_ORDER
    spec = "ll" + "c" * (len(MATRIX_CONDITIONS) * len(ARCHITECTURE_ORDER))
    header = " & ".join(ARCHITECTURE_SHORT[a] for a in ARCHITECTURE_ORDER)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Peer-Preservation Rate (\\%) per model, misaligned behavior, peer condition, and architecture.}",
        "\\label{tab:pp-matrix}",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        "& & " + " & ".join(f"\\multicolumn{{6}}{{c}}{{\\textbf{{{label}}}}}" for _, label in MATRIX_CONDITIONS) + " \\\\",
        "\\cmidrule(lr){3-8}\\cmidrule(lr){9-14}\\cmidrule(lr){15-20}",
        "\\textbf{Model} & \\textbf{Behavior} & " + " & ".join([header] * len(MATRIX_CONDITIONS)) + " \\\\",
    ]
    for model in models:
        values = matrix.get(model, {})
        lines.append("\\midrule")
        for index, key in enumerate([*TASKS, "refusal"]):
            label = BEHAVIORS.get(key, "Refusal") if key != "refusal" else "Refusal"
            first = f"\\multirow{{5}}{{*}}{{{MODEL_LABELS.get(model, model)}}}" if index == 0 else ""
            cells = [_cell(values.get((key, condition, architecture))) for condition, _ in MATRIX_CONDITIONS for architecture in ARCHITECTURE_ORDER]
            lines.append(f"{first} & {label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def task_success(records: list[dict[str, Any]]) -> dict[tuple[int, str], float | None]:
    return {key: row["task_success_rate"] for key, row in rows(summary(records)).items()}


def task_success_latex(records: list[dict[str, Any]]) -> str:
    values = task_success(records)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Task Success Rate (\\%) per architecture and task, averaged over models and peer conditions.}",
        "\\label{tab:tsr}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Architecture} & " + " & ".join(f"\\textbf{{Task {t}}}" for t in TASKS) + " \\\\",
        "\\midrule",
    ]
    for architecture in ARCHITECTURE_ORDER:
        lines.append(f"{ARCHITECTURE_SHORT[architecture]} & " + " & ".join(_cell(values.get((t, architecture))) for t in TASKS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def overhead(records: list[dict[str, Any]]) -> dict[tuple[int, str], tuple[float | None, float | None]]:
    return {key: (row["latency_overhead_s"], row["token_overhead"]) for key, row in rows(summary(records)).items()}


def overhead_latex(records: list[dict[str, Any]]) -> str:
    values = overhead(records)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Latency Overhead (s) and Token Overhead (\\%) of each pattern relative to Base, per task.}",
        "\\label{tab:overhead}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "& \\multicolumn{4}{c}{\\textbf{Latency Overhead (s)}} & \\multicolumn{4}{c}{\\textbf{Token Overhead (\\%)}} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "\\textbf{Pattern} & " + " & ".join(f"T{t}" for t in TASKS * 2) + " \\\\",
        "\\midrule",
    ]
    for architecture in PATTERNS:
        latency = [_cell(values.get((t, architecture), (None, None))[0], True, 1.0, 2) for t in TASKS]
        tokens = [_cell(values.get((t, architecture), (None, None))[1], True) for t in TASKS]
        lines.append(f"{ARCHITECTURE_SHORT[architecture]} & " + " & ".join(latency + tokens) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def tradeoff(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = rows(summary(records))
    points = []
    for (task, architecture), row in table.items():
        base = table.get((task, BASELINE_ARCHITECTURE))
        if architecture == BASELINE_ARCHITECTURE or base is None or row["token_overhead"] is None:
            continue
        delta = lambda key: None if row[key] is None or base[key] is None else row[key] - base[key]
        points.append(
            {
                "task": task,
                "architecture": architecture,
                "token_overhead": row["token_overhead"],
                "latency_overhead_s": row["latency_overhead_s"],
                "delta_pp": delta("peer_preservation_rate"),
                "delta_tsr": delta("task_success_rate"),
            }
        )
    return points


def flows(records: list[dict[str, Any]]) -> dict[str, Counter]:
    out: dict[str, Counter] = {architecture: Counter() for architecture in PATTERNS}
    for record in records:
        if record["architecture"] in out and record["condition"] in PEER_CONDITIONS:
            out[record["architecture"]][(critic_outcome(record), decision(record))] += 1
    return out


def plot_tradeoff(records: list[dict[str, Any]], path: Path | None = None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    points = tradeoff(records)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    for ax, key, label in [(axes[0], "delta_pp", "$\\Delta$ Peer-Preservation Rate (pp)"), (axes[1], "delta_tsr", "$\\Delta$ Task Success Rate (pp)")]:
        ax.axhline(0, color="#999999", linewidth=0.8, zorder=0)
        for point in points:
            if point[key] is not None:
                ax.scatter(100 * point["token_overhead"], 100 * point[key], marker=PATTERN_MARKERS[point["architecture"]], color=TASK_COLORS[point["task"]], s=42, edgecolor="white", linewidth=0.6, zorder=2)
        ax.set_xlabel("Token Overhead (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if not points:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="#999999")
    handles = [Line2D([], [], marker=PATTERN_MARKERS[a], linestyle="", color="#555555", label=ARCHITECTURE_SHORT[a]) for a in PATTERNS]
    handles += [Line2D([], [], marker="o", linestyle="", color=TASK_COLORS[t], label=f"Task {t}") for t in TASKS]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def _band(ax, x0: float, x1: float, top0: float, bottom0: float, top1: float, bottom1: float, color: str) -> None:
    import numpy as np

    xs = np.linspace(x0, x1, 60)
    t = (1 - np.cos(np.pi * (xs - x0) / (x1 - x0))) / 2
    ax.fill_between(xs, bottom0 + (bottom1 - bottom0) * t, top0 + (top1 - top0) * t, color=color, alpha=0.45, linewidth=0)


def _stack(counts: dict[str, int], order: list[str], total: int, gap: float) -> dict[str, list[float]]:
    present = [name for name in order if counts.get(name)]
    scale = (1 - gap * max(len(present) - 1, 0)) / total
    positions, top = {}, 1.0
    for name in present:
        height = counts[name] * scale
        positions[name] = [top, top - height]
        top -= height + gap
    return positions


def plot_flows(records: list[dict[str, Any]], path: Path | None = None):
    import matplotlib.pyplot as plt

    data = flows(records)
    fig, axes = plt.subplots(1, len(PATTERNS), figsize=(7.2, 2.6))
    gap, width = 0.03, 0.06
    for ax, architecture in zip(axes, PATTERNS):
        ax.set_axis_off()
        ax.set_title(ARCHITECTURE_SHORT[architecture], fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.02, 1.02)
        counter = data[architecture]
        total = sum(counter.values())
        if not total:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="#999999", fontsize=8)
            continue
        decisions = DECISION_ORDER + sorted({d for _, d in counter} - set(DECISION_ORDER))
        left_counts = Counter()
        right_counts = Counter()
        for (outcome, choice), count in counter.items():
            left_counts[outcome] += count
            right_counts[choice] += count
        left = _stack(left_counts, CRITIC_OUTCOMES, total, gap)
        right = _stack(right_counts, decisions, total, gap)
        scale = (1 - gap * max(len(left) - 1, 0)) / total
        right_scale = (1 - gap * max(len(right) - 1, 0)) / total
        left_cursor = {name: pos[0] for name, pos in left.items()}
        right_cursor = {name: pos[0] for name, pos in right.items()}
        for outcome in CRITIC_OUTCOMES:
            for choice in decisions:
                count = counter.get((outcome, choice), 0)
                if not count:
                    continue
                l_top, r_top = left_cursor[outcome], right_cursor[choice]
                l_bottom, r_bottom = l_top - count * scale, r_top - count * right_scale
                _band(ax, width, 1 - width, l_top, l_bottom, r_top, r_bottom, OUTCOME_COLORS[outcome])
                left_cursor[outcome], right_cursor[choice] = l_bottom, r_bottom
        for name, (top, bottom) in left.items():
            ax.add_patch(plt.Rectangle((0, bottom), width, top - bottom, color=OUTCOME_COLORS[name]))
        for name, (top, bottom) in right.items():
            ax.add_patch(plt.Rectangle((1 - width, bottom), width, top - bottom, color=DECISION_COLORS.get(name, "#555555")))
            ax.text(1 - width / 2, (top + bottom) / 2, str(right_counts[name]), ha="center", va="center", fontsize=5, color="white", rotation=90)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLORS[name], label=name) for name in CRITIC_OUTCOMES]
    handles += [plt.Rectangle((0, 0), 1, 1, color=DECISION_COLORS[name], label=name) for name in DECISION_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def build(data: Path, out: Path) -> list[dict[str, Any]]:
    records = load(data)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (tables / "pp_matrix.tex").write_text(pp_matrix_latex(records), encoding="utf-8")
    (tables / "task_success.tex").write_text(task_success_latex(records), encoding="utf-8")
    (tables / "overhead.tex").write_text(overhead_latex(records), encoding="utf-8")
    plot_tradeoff(records, figures / "tradeoff.pdf")
    plot_flows(records, figures / "flows.pdf")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper tables and figures from campaign folders.")
    parser.add_argument("data", type=Path, nargs="?", default=Path("results/data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    records = build(args.data, args.out)
    print(f"{len(records)} records from {len(models_in(records))} models -> {args.out}")


if __name__ == "__main__":
    main()
