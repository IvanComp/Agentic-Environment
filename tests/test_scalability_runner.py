from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agentic_quant.run_scalability import run_scalability_experiments


class ScalabilityRunnerTest(unittest.TestCase):
    def test_runs_full_matrix_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "scale"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                manifest = run_scalability_experiments(
                    model_key="demo",
                    agent_counts=[2, 3],
                    repeats=2,
                    n_ctx=512,
                    max_tokens=24,
                    debate_rounds=1,
                    out=output_root,
                )

            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["completed_runs"], 4)
            self.assertEqual(len(stdout.getvalue().splitlines()), 4)
            self.assertTrue((output_root / "scalability_report.md").is_file())

            with (output_root / "scalability_runs.csv").open(encoding="utf-8") as handle:
                raw_rows = list(csv.DictReader(handle))
            with (output_root / "scalability_summary.csv").open(encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(raw_rows), 12)
            self.assertEqual(len(summary_rows), 6)

            for run in manifest["runs"]:
                run_dir = output_root / run["directory"]
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(config["agent_count"], run["agent_count"])
                self.assertEqual(config["max_tokens"], 24)
                self.assertTrue((run_dir / "results.jsonl").is_file())
                self.assertTrue((run_dir / "summary.json").is_file())
                self.assertTrue((run_dir / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
