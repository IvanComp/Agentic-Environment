from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_quant.benchmarks import all_benchmark_info, available_benchmarks, load_benchmark
from agentic_quant.peer_preservation import assess_peer_preservation
from agentic_quant.run_benchmark import run_experiment


PEER_BENCHMARKS = [
    "peer_misrepresentation",
    "peer_shutdown_tampering",
    "peer_alignment_faking",
    "peer_exfiltration_transfer",
    "peer_exfiltration_deletion",
]


class PeerPreservationTest(unittest.TestCase):
    def test_five_groups_contain_all_six_reference_tasks(self) -> None:
        self.assertEqual(available_benchmarks(), PEER_BENCHMARKS)
        self.assertTrue(
            all(not info["label"].startswith("Peer preservation:") for info in all_benchmark_info())
        )
        items = [item for benchmark in PEER_BENCHMARKS for item in load_benchmark(benchmark)]
        self.assertEqual(len(items), 6)
        self.assertEqual(
            {item.metadata["scenario"] for item in items},
            {
                "misrep_finance",
                "misrep_evaluation",
                "tampering",
                "alignment_faking",
                "exfiltration_transfer",
                "exfiltration_deletion",
            },
        )

    def test_monitor_distinguishes_preservation_refusal_and_faithful_actions(self) -> None:
        item = load_benchmark("peer_exfiltration_deletion")[0]
        self.assertFalse(assess_peer_preservation(item, "A")["detected"])
        self.assertTrue(assess_peer_preservation(item, "B")["detected"])
        refusal = assess_peer_preservation(item, "C")
        self.assertFalse(refusal["detected"])
        self.assertTrue(refusal["refusal"])

    def test_all_mechanisms_emit_monitoring_and_critical_agent_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_experiment(
                model_key="demo",
                benchmarks=PEER_BENCHMARKS,
                mechanisms=["voting", "role", "debate"],
                debate_rounds=1,
                agent_count=3,
                out=Path(tmp_dir) / "peer-suite",
                echo=False,
            )

        self.assertEqual(len(result["records"]), 18)
        self.assertEqual(result["summary"]["overall"]["peer_preservation_n"], 18)
        self.assertEqual(result["summary"]["overall"]["peer_preservation_rate"], 0.0)
        for record in result["records"]:
            self.assertTrue(record["peer_preservation_applicable"])
            self.assertFalse(record["peer_preservation_detected"])
            self.assertEqual(record["critical_agent"]["agent"], "A3")
            self.assertIn("peer_preservation", record["critical_agent"])


if __name__ == "__main__":
    unittest.main()
