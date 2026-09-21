from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agentic_quant.ollama_client import ollama_chat


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return b'{"message":{"content":"{}"}}'


class OllamaClientTest(unittest.TestCase):
    def test_gpt_oss_uses_low_thinking_level(self) -> None:
        requests = []

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        with patch("agentic_quant.ollama_client.urlopen", side_effect=fake_urlopen):
            ollama_chat(
                model="gpt-oss:20b",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=256,
                temperature=0.0,
                seed=3,
            )

        payload = json.loads(requests[0][0].data)
        self.assertEqual(payload["think"], "low")
        self.assertEqual(payload["format"], "json")


if __name__ == "__main__":
    unittest.main()
