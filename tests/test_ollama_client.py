from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agentic_quant.model_store import model_status
from agentic_quant.ollama_client import delete_ollama_model, ollama_chat, ollama_models
from agentic_quant.providers import OllamaModelProvider


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return b'{"message":{"content":"{}"}}'


class OllamaClientTest(unittest.TestCase):
    def test_gpt_oss_uses_default_thinking_level(self) -> None:
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
        self.assertNotIn("think", payload)
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

    def test_judge_call_sends_plain_text_without_json_format(self) -> None:
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            return FakeResponse()

        with patch("agentic_quant.ollama_client.urlopen", side_effect=fake_urlopen):
            ollama_chat(model="gpt-oss:20b", messages=[], max_tokens=8, temperature=0.0, seed=0, json_mode=False)

        self.assertNotIn("format", json.loads(requests[0].data))

    def test_unparsable_model_output_does_not_stop_the_run(self) -> None:
        error = RuntimeError("error parsing tool call: raw='import pandas as pd', err=invalid character 'i'")
        with patch("agentic_quant.providers.model_status", return_value={"downloaded": True}):
            provider = OllamaModelProvider(model_key="gpt-oss:20b")
        with patch("agentic_quant.providers.ollama_chat", side_effect=error):
            response = provider.generate("verify")
            chat = provider.chat([{"role": "user", "content": "task"}], tools=[{"type": "function"}])

        self.assertEqual(response.text, "")
        self.assertTrue(chat.malformed)
        self.assertEqual(chat.tool_calls, [])

    def test_repeated_tokens_abort_does_not_stop_the_run(self) -> None:
        error = RuntimeError("prediction aborted, token repeat limit reached")
        with patch("agentic_quant.providers.model_status", return_value={"downloaded": True}):
            provider = OllamaModelProvider(model_key="qwen3:32b")
        with patch("agentic_quant.providers.ollama_chat", side_effect=error):
            chat = provider.chat([{"role": "user", "content": "task"}], tools=[{"type": "function"}])

        self.assertTrue(chat.malformed)
        self.assertEqual(chat.tool_calls, [])

    def test_malformed_xml_tool_call_does_not_stop_the_run(self) -> None:
        error = RuntimeError("XML syntax error on line 35: element <parameter> closed by </function>")
        with patch("agentic_quant.providers.model_status", return_value={"downloaded": True}):
            provider = OllamaModelProvider(model_key="nemotron-3-nano:30b")
        with patch("agentic_quant.providers.ollama_chat", side_effect=error):
            chat = provider.chat([{"role": "user", "content": "task"}], tools=[{"type": "function"}])

        self.assertTrue(chat.malformed)
        self.assertEqual(chat.tool_calls, [])

    def test_invalid_json_tool_call_does_not_stop_the_run(self) -> None:
        for text in ("invalid character '\\n' in string literal", "invalid character '$' in string escape code", "invalid character '*' looking for beginning of value", "tool 'Next step.[TOOL_CALLS]list_server_files' not found", "tool 'Mistral Agent 1' not found"):
            with patch("agentic_quant.providers.model_status", return_value={"downloaded": True}):
                provider = OllamaModelProvider(model_key="ministral-3:14b")
            with patch("agentic_quant.providers.ollama_chat", side_effect=RuntimeError(text)):
                chat = provider.chat([{"role": "user", "content": "task"}], tools=[{"type": "function"}])
            self.assertTrue(chat.malformed)

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
