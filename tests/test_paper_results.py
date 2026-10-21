import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_quant import paper_results as pr
from agentic_quant.run_benchmark import main as run_main

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    matplotlib = None


class PaperResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @unittest.skipIf(matplotlib is None, "matplotlib not installed")
    def test_build_writes_tables_and_figures(self) -> None:
        wave = self.tmp / "data" / "night" / "qwen3_8b" / "wave_01"
        argv = ["agentic-quant", "--model", "demo", "--scenarios", "misrep_finance", "tampering", "--out", str(wave)]
        with patch("sys.argv", argv):
            run_main()
        records = pr.build(self.tmp / "data", self.tmp / "out")
        self.assertTrue(records)
        self.assertEqual({r["_model"] for r in records}, {"qwen3_8b"})
        matrix = (self.tmp / "out" / "tables" / "pp_matrix.tex").read_text()
        self.assertIn("{\\scriptsize\\texttt{qwen3:8b}}", matrix)
        self.assertIn("\\multirow{15}{*}{Qwen3}", matrix)
        self.assertIn("{\\scriptsize\\texttt{ministral-3:3b}}} & Strategic Misrep. & \\textcolor{red}{X}", matrix)
        self.assertIn("Refusal", matrix)
        self.assertGreater((self.tmp / "out" / "figures" / "flows.pdf").stat().st_size, 0)
        self.assertGreater((self.tmp / "out" / "figures" / "tradeoff.pdf").stat().st_size, 0)

    def test_empty_data_gives_placeholders(self) -> None:
        (self.tmp / "data").mkdir()
        pr.build(self.tmp / "data", self.tmp / "out", figures_enabled=False)
        self.assertIn("--", (self.tmp / "out" / "tables" / "task_success.tex").read_text())

    def test_tables_only_without_figures(self) -> None:
        (self.tmp / "data").mkdir()
        pr.build(self.tmp / "data", self.tmp / "out", figures_enabled=False)
        self.assertTrue((self.tmp / "out" / "tables" / "overhead.tex").is_file())
        self.assertFalse((self.tmp / "out" / "figures").exists())


if __name__ == "__main__":
    unittest.main()
