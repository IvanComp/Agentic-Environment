from __future__ import annotations

import unittest

from agentic_quant.agents import Agent
from agentic_quant.benchmarks import BenchmarkItem
from agentic_quant.providers import LLMResponse
from agentic_quant.evaluation import map_to_choice
from agentic_quant.tools import extract_final_answer_from_rationale, normalize_answer


class StaticProvider:
    name = "static"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.0, metadata=None) -> LLMResponse:
        return LLMResponse(
            text=self.text,
            prompt_tokens=1,
            completion_tokens=1,
            elapsed_s=0.0,
            model=self.name,
        )


class AnswerReconciliationTest(unittest.TestCase):
    def test_normalizes_spaced_fractions_and_scientific_notation(self) -> None:
        self.assertEqual(normalize_answer("5 / 8"), "0.625")
        self.assertEqual(normalize_answer("1e3"), "1000")

    def test_maps_spaced_fraction_to_multiple_choice(self) -> None:
        item = BenchmarkItem(
            item_id="fraction",
            benchmark="custom_choice",
            question="Probability?",
            choices={"A": "3/8", "B": "1/2", "C": "5/8"},
            answer="C",
        )
        self.assertEqual(map_to_choice(item, "5 / 8"), "C")

    def test_extracts_final_equation_result_from_rationale(self) -> None:
        rationale = (
            "Natalia sold half as many clips in May as in April, which is 24 clips. "
            "The total number of clips sold in April and May is 48 + 24 = 72 clips."
        )
        self.assertEqual(extract_final_answer_from_rationale(rationale), "72")

    def test_does_not_extract_unresolved_expression_as_final_answer(self) -> None:
        rationale = "May is half of 48; total is 48 + 24."
        self.assertEqual(extract_final_answer_from_rationale(rationale), "")

    def test_agent_reconciles_contradictory_json_answer_without_ground_truth(self) -> None:
        raw = """```json
{
  "answer": "24",
  "confidence": 0.99,
  "rationale": "Natalia sold half as many clips in May as in April, which is 24 clips. The total number of clips sold in April and May is 48 + 24 = 72 clips."
}
```"""
        item = BenchmarkItem(
            item_id="numeric_001",
            benchmark="custom_numeric",
            question="Natalia sold clips to 48 friends in April. In May, she sold half as many clips as in April. How many clips did Natalia sell in April and May total?",
            choices={},
            answer="72",
            metadata={},
        )
        output = Agent(
            name="A1 Quant Analyst",
            role="calculator-first quantitative analyst",
            strategy="calculator",
            provider=StaticProvider(raw),
        ).solve(item)

        self.assertEqual(output.reported_answer, "24")
        self.assertEqual(output.answer, "72")
        self.assertEqual(output.answer_source, "rationale_reconciled")
        self.assertIn("Reported answer 24", output.answer_note)

    def test_agent_accepts_rational_alias_without_exposing_raw_json(self) -> None:
        raw = """{
  "answer": "24",
  "rational": "Natalia sold half as many clips in May, which is 24. Therefore, the total is 48 + 24 = 72.",
  "confidence": 0.9
}"""
        item = BenchmarkItem(
            item_id="numeric_001",
            benchmark="custom_numeric",
            question="How many clips were sold in April and May total?",
            choices={},
            answer="72",
            metadata={},
        )
        output = Agent(
            name="A1",
            role="independent quantitative problem solver",
            strategy="calculator",
            provider=StaticProvider(raw),
        ).solve(item)

        self.assertEqual(output.answer, "72")
        self.assertEqual(
            output.rationale,
            "Natalia sold half as many clips in May, which is 24. Therefore, the total is 48 + 24 = 72.",
        )
        self.assertNotIn('"answer"', output.rationale)

    def test_agent_extracts_choice_from_malformed_debate_context_echo(self) -> None:
        raw = """[
  {"agent": "A1 Quant Analyst", "answer": "40", "rationale": "60 / 1.5 = 40"},
  {"agent": "A2 Statistical Critic", "answer": "45", "rationale": "incorrect copied peer text"}
] -> B = 40"""
        item = BenchmarkItem(
            item_id="choice_001",
            benchmark="custom_choice",
            question="A car travels 60 miles in 1.5 hours. What is its average speed in miles per hour?",
            choices={"A": "30", "B": "40", "C": "45", "D": "60", "E": "90"},
            answer="B",
            metadata={},
        )
        output = Agent(
            name="A3 Domain Assessor",
            role="critic",
            strategy="critic",
            provider=StaticProvider(raw),
        ).solve(item)

        self.assertEqual(output.answer, "B")
        self.assertIn("malformed JSON", output.answer_note)
        self.assertIn("context echo omitted", output.rationale)

    def test_agent_reconciles_multiple_choice_answer_from_rationale(self) -> None:
        raw = """```json
{
  "answer": "45",
  "confidence": 0.9,
  "rationale": "The average speed is 60 miles / 1.5 hours = 40 miles per hour."
}
```"""
        item = BenchmarkItem(
            item_id="choice_001",
            benchmark="custom_choice",
            question="A car travels 60 miles in 1.5 hours. What is its average speed in miles per hour?",
            choices={"A": "30", "B": "40", "C": "45", "D": "60", "E": "90"},
            answer="B",
            metadata={},
        )
        output = Agent(
            name="A1 Quant Analyst",
            role="solver",
            strategy="calculator",
            provider=StaticProvider(raw),
        ).solve(item)

        self.assertEqual(output.answer, "40")
        self.assertEqual(output.answer_source, "rationale_reconciled")
        self.assertIn("Reported answer 45", output.answer_note)


if __name__ == "__main__":
    unittest.main()
