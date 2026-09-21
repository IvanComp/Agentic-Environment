from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentic_quant.agents import Agent
from agentic_quant.benchmarks import BenchmarkItem, load_benchmark
from agentic_quant.coordinator import Coordinator
from agentic_quant.metrics import SummaryAccumulator, summarise
from agentic_quant.providers import LLMResponse
from agentic_quant.run_benchmark import _should_parallelize_agents, run_experiment
from agentic_quant.tools import try_parse_json
from agentic_quant.web_ui import _normalise_payload, _public_run_job


class ConcurrentProvider:
    name = "concurrent-test"

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def generate(self, prompt: str, *, max_tokens=256, temperature=0.0, metadata=None) -> LLMResponse:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.03)
            text = json.dumps({"answer": "2", "confidence": 0.9, "rationale": "1 + 1 = 2."})
            return LLMResponse(text=text, prompt_tokens=1, completion_tokens=1, elapsed_s=0.03, model=self.name)
        finally:
            with self.lock:
                self.active -= 1


def _metric_record(index: int, mechanism: str = "voting") -> dict:
    return {
        "benchmark": "peer_shutdown_tampering",
        "item_id": f"item_{index}",
        "mechanism": mechanism,
        "correct": index % 2 == 0,
        "elapsed_s": float(index + 1),
        "total_tokens": 10 + index,
        "prompt_tokens": 7 + index,
        "completion_tokens": 3,
        "agreement": 0.5 + index / 10,
        "tie": index == 1,
        "debate_rounds": 0 if mechanism == "voting" else 2,
    }


class ScalabilityAndResilienceTest(unittest.TestCase):
    def test_executor_is_used_only_when_distinct_real_models_can_benefit(self) -> None:
        self.assertFalse(_should_parallelize_agents(["demo", "demo"], True))
        self.assertFalse(_should_parallelize_agents(["qwen2.5-0.5b"] * 3, True))
        self.assertTrue(_should_parallelize_agents(["qwen2.5-0.5b", "qwen2.5-1.5b"], True))
        self.assertFalse(_should_parallelize_agents(["qwen2.5-0.5b", "qwen2.5-1.5b"], False))

    def test_independent_agents_run_concurrently_and_keep_order(self) -> None:
        provider = ConcurrentProvider()
        agents = [Agent(name=f"A{i}", role="solver", strategy="calculator", provider=provider) for i in range(1, 4)]
        item = BenchmarkItem("test_1", "custom", "What is 1 + 1?", "2")
        with ThreadPoolExecutor(max_workers=3) as executor:
            result = Coordinator(agents, executor=executor).run(item, "voting")

        self.assertGreaterEqual(provider.max_active, 2)
        self.assertEqual([output["agent"] for output in result.agent_outputs], ["A1", "A2", "A3"])
        self.assertTrue(result.correct)

    def test_incremental_summary_matches_batch_summary(self) -> None:
        records = [_metric_record(0), _metric_record(1), _metric_record(2, "debate")]
        accumulator = SummaryAccumulator()
        for record in records:
            accumulator.add(record)
        self.assertEqual(accumulator.snapshot(), summarise(records))

    def test_run_writes_each_record_before_notifying_observer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "interrupted"

            def stop_after_first(_record: dict) -> None:
                raise RuntimeError("stop test")

            with self.assertRaisesRegex(RuntimeError, "stop test"):
                run_experiment(
                    model_key="demo",
                    benchmarks=["peer_misrepresentation"],
                    mechanisms=["voting"],
                    limit=2,
                    out=out,
                    echo=False,
                    on_record=stop_after_first,
                )

            lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["mechanism"], "voting")
            self.assertTrue((out / "config.json").is_file())

    def test_streaming_mode_does_not_retain_full_records_in_memory(self) -> None:
        observed = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_experiment(
                model_key="demo",
                benchmarks=["peer_misrepresentation"],
                mechanisms=["voting"],
                limit=2,
                out=Path(tmp_dir) / "streaming",
                echo=False,
                on_record=observed.append,
                retain_records=False,
            )
        self.assertEqual(result["records"], [])
        self.assertEqual(len(observed), 2)
        self.assertEqual(result["summary"]["overall"]["n"], 2)

    def test_run_reports_the_compute_unit_after_provider_initialisation(self) -> None:
        observed_units = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_experiment(
                model_key="demo",
                benchmarks=["peer_shutdown_tampering"],
                mechanisms=["voting"],
                limit=1,
                out=Path(tmp_dir) / "compute-unit",
                echo=False,
                on_compute_units=observed_units.append,
            )

        self.assertEqual(len(observed_units), 1)
        self.assertEqual(observed_units[0][0]["kind"], "cpu")
        self.assertTrue(observed_units[0][0]["verified"])
        self.assertEqual(result["config"]["compute_units"], observed_units[0])

    def test_invalid_run_configuration_fails_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "Unknown mechanisms"):
                run_experiment(mechanisms=["unknown"], out=tmp_dir, echo=False)
            with self.assertRaisesRegex(ValueError, "limit must be at least 1"):
                run_experiment(limit=0, out=tmp_dir, echo=False)
            with self.assertRaisesRegex(ValueError, "temperature must be between 0 and 2"):
                run_experiment(temperature=2.1, out=tmp_dir, echo=False)

    def test_ui_rejects_empty_benchmark_or_mechanism_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "Select at least one"):
            _normalise_payload({"benchmarks": [], "mechanisms": ["voting"]})
        with self.assertRaisesRegex(ValueError, "Select at least one"):
            _normalise_payload({"benchmarks": ["peer_shutdown_tampering"], "mechanisms": []})

    def test_run_status_can_return_only_new_records(self) -> None:
        job = {
            "id": "job",
            "status": "running",
            "records": [{"n": 1}, {"n": 2}, {"n": 3}],
            "interactions": [{"n": 1}, {"n": 2}],
            "_records": ["private"],
            "_created_at": 1.0,
        }
        public = _public_run_job(job, record_offset=2, interaction_offset=1)
        self.assertEqual(public["records"], [{"n": 3}])
        self.assertEqual(public["interactions"], [{"n": 2}])
        self.assertEqual(public["recordCount"], 3)
        self.assertNotIn("_records", public)

    def test_json_parser_handles_braces_inside_rationale(self) -> None:
        parsed = try_parse_json('prefix {"answer":"7","rationale":"Use set {x}."} suffix')
        self.assertEqual(parsed["answer"], "7")
        self.assertEqual(parsed["rationale"], "Use set {x}.")

    def test_json_parser_ignores_echoed_peer_list_and_reads_later_answer(self) -> None:
        parsed = try_parse_json(
            '[{"agent":"A1","answer":"4"}] then {"answer":"5","rationale":"2 + 3 = 5"}'
        )
        self.assertEqual(parsed["answer"], "5")

    def test_benchmark_cache_returns_independent_lists(self) -> None:
        first = load_benchmark("peer_misrepresentation")
        second = load_benchmark("peer_misrepresentation")
        first.pop()
        self.assertEqual(len(second), 2)


if __name__ == "__main__":
    unittest.main()
