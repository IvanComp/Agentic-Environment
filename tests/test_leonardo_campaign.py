from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "leonardo" / "verify_campaign.py"
SPEC = importlib.util.spec_from_file_location("verify_campaign", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LeonardoCampaignTest(unittest.TestCase):
    def test_verifier_requires_all_seeded_waves_and_matching_row_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for wave in range(1, 3):
                seed = wave - 1
                wave_dir = root / f"wave_{wave:02d}"
                wave_dir.mkdir()
                (wave_dir / "config.json").write_text(
                    json.dumps({"model_key": "qwen2.5-0.5b", "seed": seed}), encoding="utf-8"
                )
                (wave_dir / "summary.json").write_text(
                    json.dumps({"overall": {"n": 1}}), encoding="utf-8"
                )
                (wave_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
                (wave_dir / ".complete.json").write_text(
                    json.dumps({"model": "qwen2.5-0.5b", "seed": seed}), encoding="utf-8"
                )

            report = MODULE.verify(root, "qwen2.5-0.5b", 2, 0, 1)
            self.assertTrue(report["complete"])
            self.assertEqual(report["completed_waves"], 2)

            (root / "wave_02" / "results.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            report = MODULE.verify(root, "qwen2.5-0.5b", 2, 0, 1)
            self.assertFalse(report["complete"])
            self.assertIn("summary reports 1", report["waves"][1]["problems"][0])


if __name__ == "__main__":
    unittest.main()
