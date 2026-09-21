from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_quant.run_repeated import MECHANISMS, run_repeated_experiments


class RepeatedExperimentsTest(unittest.TestCase):
    def test_saves_a_complete_report_for_each_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "suite"
            manifest = run_repeated_experiments(
                model_key="demo",
                repeats=2,
                seed_start=11,
                temperature=0.2,
                out=output_root,
                echo=False,
            )

            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["seeds"], [11, 12])
            self.assertEqual(manifest["mechanisms"], MECHANISMS)
            self.assertEqual(len(manifest["runs"]), 2)

            for run in manifest["runs"]:
                run_dir = output_root / run["directory"]
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                results = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(config["seed"], run["seed"])
                self.assertEqual(config["temperature"], 0.2)
                self.assertEqual(config["mechanisms"], MECHANISMS)
                self.assertEqual(len(results), 18)
                self.assertTrue(all(json.loads(line)["seed"] == run["seed"] for line in results))
                self.assertTrue((run_dir / "summary.json").is_file())
                self.assertTrue((run_dir / "report.md").is_file())

            saved_manifest = json.loads(
                (output_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest["status"], "completed")


if __name__ == "__main__":
    unittest.main()
