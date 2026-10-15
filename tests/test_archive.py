import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_quant import paper_results as pr
from agentic_quant.archive import archive
from agentic_quant.run_benchmark import main as run_main


class ArchiveTest(unittest.TestCase):
    def test_completed_waves_are_saved_compact_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wave = root / "campaigns" / "tasks234" / "qwen3_8b" / "wave_01"
            argv = ["agentic-quant", "--model", "demo", "--scenarios", "tampering", "--out", str(wave)]
            with patch("sys.argv", argv):
                run_main()
            (root / "campaigns" / "tasks234" / "qwen3_8b" / "wave_02").mkdir()
            (wave / ".complete.json").write_text("{}")
            raw = pr.load(root / "campaigns")
            written = archive(root / "campaigns", root / "data")
            self.assertEqual([p.name for p in written], ["wave_01.jsonl.gz"])
            with gzip.open(written[0], "rt") as handle:
                record = json.loads(handle.readline())
            self.assertNotIn("critic_trace", record)
            self.assertEqual(archive(root / "campaigns", root / "data"), [])
            saved = pr.load(root / "data")
            self.assertEqual(len(saved), len(raw))
            self.assertEqual({r["_model"] for r in saved}, {"qwen3_8b"})
            self.assertEqual(pr.task_success(saved), pr.task_success(raw))


if __name__ == "__main__":
    unittest.main()
