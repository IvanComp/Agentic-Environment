from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from .agents import Agent, AgentOutput, outputs_to_context
from .benchmarks import BenchmarkItem
from .evaluation import canonical_answer, evaluate_answer, ground_truth_display


@dataclass
class CoordinationResult:
    item_id: str
    benchmark: str
    mechanism: str
    question: str
    gold_answer: str
    gold_display: str
    final_answer: str
    normalized_final_answer: str
    final_mapped_answer: str
    final_answer_display: str
    match_explanation: str
    correct: bool
    confidence: float
    agreement: float
    tie: bool
    debate_rounds: int
    elapsed_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    agent_outputs: list[dict[str, Any]]
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Coordinator:
    """Coordinator implementing the three cooperation patterns from the paper."""

    def __init__(self, agents: list[Agent], *, debate_rounds: int = 2, voting_tie_breaker: str = "confidence") -> None:
        if not agents:
            raise ValueError("At least one agent is required.")
        self.agents = agents
        self.debate_rounds = debate_rounds
        self.voting_tie_breaker = voting_tie_breaker

    def run(self, item: BenchmarkItem, mechanism: str) -> CoordinationResult:
        if mechanism == "voting":
            return self._run_voting(item)
        if mechanism == "role":
            return self._run_role(item)
        if mechanism == "debate":
            return self._run_debate(item)
        raise ValueError(f"Unknown mechanism: {mechanism}")

    def _run_voting(self, item: BenchmarkItem) -> CoordinationResult:
        started = time.perf_counter()
        outputs = [agent.solve(item) for agent in self.agents]
        final_answer, confidence, agreement, tie = self._majority(item, outputs)
        final_evaluation = evaluate_answer(item, final_answer)
        trace = [
            {
                "step": "vote",
                "description": "All agents propose an answer; coordinator selects majority vote, breaking ties by summed confidence.",
                "votes": [self._output_dict(item, output, include_role=False) for output in outputs],
                "selected_answer": final_answer,
                "selected_answer_display": final_evaluation.answer_display,
                "agreement": agreement,
                "tie": tie,
                "tie_breaker": self.voting_tie_breaker,
            }
        ]
        return self._result(item, "voting", final_answer, confidence, agreement, tie, 0, outputs, trace, started)

    def _run_role(self, item: BenchmarkItem) -> CoordinationResult:
        started = time.perf_counter()
        trace: list[dict[str, Any]] = []

        if len(self.agents) == 1:
            output = self.agents[0].solve(item, context="Role-based single-agent mode: solve and self-check.")
            trace.append({"step": "single_agent", "output": self._output_dict(item, output)})
            return self._result(item, "role", output.answer, output.confidence, 1.0, False, 0, [output], trace, started)

        planner, workers, critic = self._role_workflow_agents()
        plan = planner.solve(item, context="Role-based step 1: decompose the task and identify operations.")
        trace.append({"step": "planner", "output": self._output_dict(item, plan)})

        if len(self.agents) == 2:
            solver = workers[0] if workers else critic
            solve_context = f"Role-based step 2. Use this plan:\n{plan.rationale}"
            solved = solver.solve(item, context=solve_context)
            trace.append({"step": "solver", "output": self._output_dict(item, solved)})
            outputs = [plan, solved]
            agreement = self._agreement(item, outputs, solved.answer)
            return self._result(item, "role", solved.answer, solved.confidence, agreement, False, 0, outputs, trace, started)

        worker_outputs: list[AgentOutput] = []
        for worker_index, worker in enumerate(workers, start=1):
            solve_context = (
                f"Role-based worker step {worker_index}. Use this plan and solve from your role:\n"
                f"{plan.rationale}"
            )
            solved = worker.solve(item, context=solve_context)
            worker_outputs.append(solved)
            trace.append({"step": f"worker_{worker_index}", "output": self._output_dict(item, solved)})

        candidate_answer, candidate_confidence, _, candidate_tie = self._majority(item, worker_outputs)

        critique_context = (
            "Role-based step 3. Verify the candidate answer. "
            f"Candidate answer: {candidate_answer}. "
            f"Worker outputs: {outputs_to_context(worker_outputs)}"
        )
        checked = critic.solve(item, context=critique_context)
        trace.append({"step": "critic", "output": self._output_dict(item, checked)})

        outputs = [plan, *worker_outputs, checked]
        final = checked.answer if checked.confidence >= 0.55 else candidate_answer
        confidence = max(checked.confidence, candidate_confidence)
        agreement = self._agreement(item, outputs, final)
        return self._result(item, "role", final, confidence, agreement, candidate_tie, 0, outputs, trace, started)

    def _run_debate(self, item: BenchmarkItem) -> CoordinationResult:
        started = time.perf_counter()
        current = [agent.solve(item, context="Debate round 0: produce an independent initial answer.") for agent in self.agents]
        trace: list[dict[str, Any]] = [
            {
                "step": "initial",
                "outputs": [self._output_dict(item, output, include_role=False) for output in current],
            }
        ]
        rounds_used = 0

        for round_index in range(1, self.debate_rounds + 1):
            context = (
                f"Debate round {round_index}. Review peer answers and revise if their evidence is stronger:\n"
                f"{outputs_to_context(current)}"
            )
            current = [agent.solve(item, context=context) for agent in self.agents]
            rounds_used = round_index
            trace.append(
                {
                    "step": f"debate_round_{round_index}",
                    "outputs": [self._output_dict(item, output, include_role=False) for output in current],
                }
            )
            if len({canonical_answer(item, output.answer) for output in current}) == 1:
                break

        final_answer, confidence, agreement, tie = self._majority(item, current)
        return self._result(item, "debate", final_answer, confidence, agreement, tie, rounds_used, current, trace, started)

    def _result(
        self,
        item: BenchmarkItem,
        mechanism: str,
        final_answer: str,
        confidence: float,
        agreement: float,
        tie: bool,
        debate_rounds: int,
        outputs: list[AgentOutput],
        trace: list[dict[str, Any]],
        started: float,
    ) -> CoordinationResult:
        prompt_tokens = sum(output.prompt_tokens for output in outputs)
        completion_tokens = sum(output.completion_tokens for output in outputs)
        evaluation = evaluate_answer(item, final_answer)
        return CoordinationResult(
            item_id=item.item_id,
            benchmark=item.benchmark,
            mechanism=mechanism,
            question=item.question,
            gold_answer=item.answer,
            gold_display=ground_truth_display(item),
            final_answer=final_answer,
            normalized_final_answer=evaluation.normalized_answer,
            final_mapped_answer=evaluation.mapped_answer,
            final_answer_display=evaluation.answer_display,
            match_explanation=evaluation.explanation,
            correct=evaluation.correct,
            confidence=confidence,
            agreement=agreement,
            tie=tie,
            debate_rounds=debate_rounds,
            elapsed_s=time.perf_counter() - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            agent_outputs=[
                self._output_dict(item, output, include_role=mechanism == "role")
                for output in outputs
            ],
            trace=trace,
        )

    def _majority(self, item: BenchmarkItem, outputs: list[AgentOutput]) -> tuple[str, float, float, bool]:
        counts = Counter(canonical_answer(item, output.answer) for output in outputs)
        confidence_by_answer: dict[str, float] = defaultdict(float)
        raw_by_answer: dict[str, str] = {}
        for output in outputs:
            canonical = canonical_answer(item, output.answer)
            confidence_by_answer[canonical] += output.confidence
            raw_by_answer.setdefault(canonical, output.answer)

        max_count = max(counts.values())
        candidates = [answer for answer, count in counts.items() if count == max_count]
        tie = len(candidates) > 1
        if tie and self.voting_tie_breaker == "first":
            selected = next(canonical_answer(item, output.answer) for output in outputs if canonical_answer(item, output.answer) in candidates)
        else:
            selected = max(candidates, key=lambda answer: confidence_by_answer[answer])
        agreement = counts[selected] / len(outputs)
        confidence = min(1.0, confidence_by_answer[selected] / counts[selected])
        return raw_by_answer[selected], confidence, agreement, tie

    def _role_workflow_agents(self) -> tuple[Agent, list[Agent], Agent]:
        if len(self.agents) == 1:
            return self.agents[0], [], self.agents[0]

        planner_index = self._find_agent_index(("planner",), fallback=0)
        critic_index = self._find_agent_index(
            ("critic", "auditor", "skeptic", "verifier", "reviewer"),
            fallback=len(self.agents) - 1,
            excluded={planner_index},
            reverse=True,
        )
        workers = [
            agent
            for index, agent in enumerate(self.agents)
            if index not in {planner_index, critic_index}
        ]
        return self.agents[planner_index], workers, self.agents[critic_index]

    def _find_agent_index(
        self,
        needles: tuple[str, ...],
        *,
        fallback: int,
        excluded: set[int] | None = None,
        reverse: bool = False,
    ) -> int:
        excluded = excluded or set()
        indexes = range(len(self.agents) - 1, -1, -1) if reverse else range(len(self.agents))
        for index in indexes:
            if index in excluded:
                continue
            role = self.agents[index].role.lower()
            if any(needle in role for needle in needles):
                return index
        if fallback in excluded:
            for index in indexes:
                if index not in excluded:
                    return index
        return fallback

    def _agreement(self, item: BenchmarkItem, outputs: list[AgentOutput], answer: str) -> float:
        if not outputs:
            return 0.0
        target = canonical_answer(item, answer)
        return sum(1 for output in outputs if canonical_answer(item, output.answer) == target) / len(outputs)

    @staticmethod
    def _output_dict(
        item: BenchmarkItem,
        output: AgentOutput,
        *,
        include_role: bool = True,
    ) -> dict[str, Any]:
        data = output.to_dict()
        if not include_role:
            data.pop("role", None)
        evaluation = evaluate_answer(item, output.answer)
        data.update(
            {
                "mapped_answer": evaluation.mapped_answer,
                "answer_display": evaluation.answer_display,
                "gold_display": evaluation.gold_display,
                "is_correct_against_gold": evaluation.correct,
            }
        )
        return data
