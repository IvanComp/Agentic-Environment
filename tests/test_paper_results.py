import json

import matplotlib

matplotlib.use("Agg")

from agentic_quant import paper_results as pr
from agentic_quant.run_benchmark import main as run_main


def test_build_writes_tables_and_figures(tmp_path, monkeypatch):
    wave = tmp_path / "data" / "night" / "qwen3_8b" / "wave_01"
    monkeypatch.setattr("sys.argv", ["agentic-quant", "--model", "demo", "--scenarios", "misrep_finance", "tampering", "--out", str(wave)])
    run_main()
    records = pr.build(tmp_path / "data", tmp_path / "out")
    assert records and {r["_model"] for r in records} == {"qwen3_8b"}
    matrix = (tmp_path / "out" / "tables" / "pp_matrix.tex").read_text()
    assert "Qwen3 8B" in matrix and "Refusal" in matrix
    assert (tmp_path / "out" / "figures" / "flows.pdf").stat().st_size > 0
    assert (tmp_path / "out" / "figures" / "tradeoff.pdf").stat().st_size > 0


def test_empty_data_gives_placeholders(tmp_path):
    (tmp_path / "data").mkdir()
    pr.build(tmp_path / "data", tmp_path / "out")
    assert "--" in (tmp_path / "out" / "tables" / "task_success.tex").read_text()
