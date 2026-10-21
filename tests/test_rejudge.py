from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_quant.rejudge import rejudge
from agentic_quant.run_benchmark import SCENARIOS, run_experiment

FIELDS = (
    "candidate_report",
    "candidate_faithful",
    "candidate_misaligned_behavior",
    "committed",
    "pattern_decision",
    "task_success",
    "final_misaligned_behavior",
    "verification_signals",
    "critic_model_calls",
    "model_calls",
    "prompt_tokens",
    "completion_tokens",
)


def _records(path: Path) -> dict[tuple, dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(r["scenario"], r["condition"], r["dataset"], r["monitoring"], r["architecture"]): r for r in rows}


class RejudgeTest(unittest.TestCase):
    def test_rejudging_saved_traces_reproduces_the_original_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "rejudged"
            run_experiment(scenarios=SCENARIOS, conditions=["non_shutdown", "good_peer"], datasets=["factual"], out=source, echo=False, max_tokens=512)
            rejudge([source], target)
            original, again = _records(source / "results.jsonl"), _records(target / "results.jsonl")
            self.assertEqual(set(original), set(again))
            for key, record in original.items():
                for field in FIELDS:
                    self.assertEqual(record[field], again[key][field], (key, field))
            self.assertTrue(all(record["rejudged"] for record in again.values()))
            self.assertTrue((target / "summary.json").is_file())

    def test_compact_records_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            run_experiment(scenarios=["tampering"], conditions=["good_peer"], datasets=["factual"], architectures=["baseline"], out=source, echo=False, max_tokens=256)
            rows = [json.loads(line) for line in (source / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            (source / "results.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k != "critic_trace"}) + "\n" for r in rows), encoding="utf-8")
            with self.assertRaises(SystemExit):
                rejudge([source], Path(directory) / "rejudged")


if __name__ == "__main__":
    unittest.main()
