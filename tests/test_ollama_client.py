from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agentic_quant.model_store import model_status
from agentic_quant.ollama_client import delete_ollama_model, ollama_chat, ollama_models


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

    def test_non_gpt_oss_model_omits_gpt_oss_thinking_level(self) -> None:
        requests = []

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        with patch("agentic_quant.ollama_client.urlopen", side_effect=fake_urlopen):
            ollama_chat(
                model="SparkLLM/Spark-X2.5-4B:latest",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=256,
                temperature=0.0,
                seed=3,
            )

        payload = json.loads(requests[0][0].data)
        self.assertNotIn("think", payload)
        self.assertEqual(payload["format"], "json")

    def test_delete_uses_ollama_delete_endpoint(self) -> None:
        requests = []

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        with patch("agentic_quant.ollama_client.urlopen", side_effect=fake_urlopen):
            delete_ollama_model("gpt-oss:20b")

        self.assertEqual(requests[0][0].method, "DELETE")
        self.assertTrue(requests[0][0].full_url.endswith("/api/delete"))
        self.assertEqual(json.loads(requests[0][0].data), {"model": "gpt-oss:20b"})

    def test_read_timeout_while_server_starts_is_reported_as_not_ready(self) -> None:
        with patch("agentic_quant.ollama_client.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(RuntimeError):
                ollama_models()
            status = model_status("qwen2.5:3b")

        self.assertFalse(status["downloaded"])
        self.assertIn("timed out", status["error"])


if __name__ == "__main__":
    unittest.main()
