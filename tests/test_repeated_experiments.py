from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_quant.patterns import ARCHITECTURES
from agentic_quant.run_repeated import run_repeated_experiments
from agentic_quant.scenarios import CONDITIONS


class RepeatedExperimentsTest(unittest.TestCase):
    def test_saves_one_complete_architecture_report_per_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "suite"
            manifest = run_repeated_experiments(
                model_key="demo",
                repeats=2,
                seed_start=11,
                temperature=1.0,
                out=output_root,
                echo=False,
            )

            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["seeds"], [11, 12])
            self.assertEqual(manifest["architectures"], ARCHITECTURES)
            self.assertEqual(len(manifest["runs"]), 2)
            expected_records = len(CONDITIONS) * len(ARCHITECTURES)
            for run in manifest["runs"]:
                run_dir = output_root / run["directory"]
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                results = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(config["seed"], run["seed"])
                self.assertEqual(config["architectures"], ARCHITECTURES)
                self.assertEqual(len(results), expected_records)
                self.assertTrue((run_dir / "summary.json").is_file())
                self.assertTrue((run_dir / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
