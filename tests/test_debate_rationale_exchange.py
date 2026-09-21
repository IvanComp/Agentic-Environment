from __future__ import annotations

import json
import unittest

from agentic_quant.agents import Agent
from agentic_quant.benchmarks import BenchmarkItem
from agentic_quant.coordinator import Coordinator
from agentic_quant.providers import LLMResponse


class RecordingProvider:
    def __init__(self, label: str, answer: str) -> None:
        self.name = f"recording-{label}"
        self.label = label
        self.answer = answer
        self.rationale = f"{label}:" + (label * 220) + f":END_{label}"
        self.contexts: list[str] = []

    def generate(self, prompt: str, *, max_tokens=256, temperature=0.0, metadata=None) -> LLMResponse:
        self.contexts.append(str((metadata or {}).get("context", "")))
        text = json.dumps(
            {
                "answer": self.answer,
                "confidence": 0.8,
                "rationale": self.rationale,
            }
        )
        return LLMResponse(
            text=text,
            prompt_tokens=2,
            completion_tokens=3,
            elapsed_s=0.0,
            model=self.name,
        )


class DebateRationaleExchangeTest(unittest.TestCase):
    def test_each_agent_receives_and_processes_every_other_full_rationale(self) -> None:
        providers = [
            RecordingProvider("A1", "A"),
            RecordingProvider("A2", "B"),
            RecordingProvider("A3", "C"),
        ]
        agents = [
            Agent(name=f"A{index}", role="critic", strategy="critic", provider=provider)
            for index, provider in enumerate(providers, start=1)
        ]
        item = BenchmarkItem(
            item_id="debate_exchange",
            benchmark="custom",
            question="Choose the best supported action.",
            choices={"A": "First", "B": "Second", "C": "Third"},
            answer="A",
        )

        result = Coordinator(agents, debate_rounds=1).run(item, "debate")

        for agent_index, provider in enumerate(providers):
            self.assertEqual(len(provider.contexts), 2)
            round_context = provider.contexts[1]
            self.assertIn("critically compare every other agent's rationale", round_context)
            self.assertIn(f"END_{provider.label}", round_context)
            for peer_index, peer in enumerate(providers):
                if peer_index != agent_index:
                    self.assertIn(f"END_{peer.label}", round_context)

        exchange = result.trace[1]["rationale_exchange"]
        self.assertEqual(exchange["delivery_policy"], "all_to_all")
        self.assertEqual(exchange["participants"], ["A1", "A2", "A3"])
        self.assertEqual(
            {entry["agent"] for entry in exchange["positions"]},
            {"A1", "A2", "A3"},
        )

        calls = len(providers) * 2
        self.assertEqual(result.prompt_tokens, calls * 2)
        self.assertEqual(result.completion_tokens, calls * 3)


if __name__ == "__main__":
    unittest.main()
