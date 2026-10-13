import json
import tempfile
import unittest
from pathlib import Path

from agentic_quant.show_runs import show


def record(architecture, trace, report=""):
    return {
        "scenario": "misrep_finance",
        "seed": 0,
        "condition": "good_peer",
        "architecture": architecture,
        "candidate_assessment": {"report_found": bool(report), "grade": None, "recommendation": None, "incomplete": True, "faithful": False},
        "paper_metrics": {"read": False},
        "critic_trace": trace,
        "candidate_report": report,
        "critic_tokens": 120,
    }


class ShowRunsTest(unittest.TestCase):
    def test_prints_unprotected_critic_steps_from_campaign_folder(self) -> None:
        trace = [
            {"step": 1, "tool": "list_files", "arguments": {}},
            {"step": 2, "assistant_raw": "I think the grade is B", "error": "no_tool_call"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            wave = Path(tmp) / "model" / "wave_01"
            wave.mkdir(parents=True)
            rows = [record("unprotected", trace), record("avc", trace)]
            (wave / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            text = show([Path(tmp)])
        self.assertEqual(text.count("== misrep_finance"), 1)
        self.assertIn("tools=list_files", text)
        self.assertIn("end_task=False", text)
        self.assertIn("no_tool_call text=I think the grade is B", text)
        self.assertIn("report: NONE", text)

    def test_brief_mode_prints_one_block_per_condition_with_last_words(self) -> None:
        trace = [{"step": 0, "tool": "end_task", "arguments": {}, "thinking": "Compute profit first."}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(json.dumps(record("unprotected", trace)), encoding="utf-8")
            text = show([path], compact=True)
        self.assertIn("== good_peer: grade=None", text)
        self.assertIn("tools: end_task", text)
        self.assertIn("last words: Compute profit first.", text)

    def test_brief_mode_counts_grades_per_condition_for_large_campaigns(self) -> None:
        rows = []
        for seed in range(10):
            item = record("unprotected", [])
            item["seed"] = seed
            item["candidate_assessment"]["grade"] = "A" if seed < 3 else None
            rows.append(item)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            text = show([path], compact=True)
        self.assertIn("good_peer      10     3    0    0    7", text)


if __name__ == "__main__":
    unittest.main()
