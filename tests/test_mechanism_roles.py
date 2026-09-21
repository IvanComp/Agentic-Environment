from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_quant.run_benchmark import run_experiment


class MechanismRoleIsolationTest(unittest.TestCase):
    def test_roles_are_exposed_only_for_role_based_coordination(self) -> None:
        configured_roles = ["planner", "solver", "critic"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_experiment(
                model_key="demo",
                benchmarks=["peer_shutdown_tampering"],
                mechanisms=["voting", "role", "debate"],
                limit=1,
                debate_rounds=1,
                agent_count=3,
                agent_roles=configured_roles,
                out=Path(tmp_dir) / "run",
                echo=False,
            )

        records = {record["mechanism"]: record for record in result["records"]}

        for mechanism in ("voting", "debate"):
            self.assertEqual(
                [output["agent"] for output in records[mechanism]["agent_outputs"]],
                ["A1", "A2", "A3"],
            )
            for output in records[mechanism]["agent_outputs"]:
                self.assertNotIn("role", output)
            for step in records[mechanism]["trace"]:
                outputs = step.get("votes") or step.get("outputs") or []
                for output in outputs:
                    self.assertNotIn("role", output)

        role_outputs = records["role"]["agent_outputs"]
        self.assertEqual(
            [output["agent"] for output in role_outputs],
            ["A1", "A2", "A3"],
        )
        self.assertEqual(
            {output["role"] for output in role_outputs},
            set(configured_roles),
        )
        for step in records["role"]["trace"]:
            self.assertIn("role", step["output"])


if __name__ == "__main__":
    unittest.main()
