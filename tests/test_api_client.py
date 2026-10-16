from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from agentic_quant import api_campaign
from agentic_quant.api_client import QuotaExhausted, api_chat, to_openai_messages
from agentic_quant.providers import ApiModelProvider, make_model_provider
from agentic_quant.tool_protocol import ToolCall, assistant_message, tool_message, tool_schema


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def completion(message, model="deepseek/deepseek-chat-v3.1"):
    return {"model": model, "choices": [{"message": message}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


class ApiClientTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(os.environ, {"AGENTIC_API_KEY": "test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_messages_get_openai_tool_call_ids(self) -> None:
        messages = [
            {"role": "system", "content": "s"},
            assistant_message("", [ToolCall("list_files", {}), ToolCall("view_sheet", {"name": "x"})], "thought"),
            tool_message("list_files", "a"),
            tool_message("view_sheet", "b"),
        ]
        converted = to_openai_messages(messages)
        calls = converted[1]["tool_calls"]
        self.assertEqual(calls[1]["function"]["arguments"], '{"name": "x"}')
        self.assertNotIn("thinking", converted[1])
        self.assertEqual([converted[2]["tool_call_id"], converted[3]["tool_call_id"]], [calls[0]["id"], calls[1]["id"]])

    def test_provider_chat_parses_tool_calls_and_reasoning(self) -> None:
        requests = []
        message = {
            "content": "",
            "reasoning": "checking files",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "list_files", "arguments": "{}"}}],
        }

        def fake_urlopen(request, timeout):
            requests.append(request)
            return FakeResponse(completion(message))

        provider = make_model_provider(model_key="deepseek-v3.1", seed=4)
        self.assertIsInstance(provider, ApiModelProvider)
        with patch("agentic_quant.api_client.urlopen", side_effect=fake_urlopen):
            response = provider.chat([{"role": "user", "content": "go"}], tools=[tool_schema("list_files", "List")], max_tokens=64, temperature=1.0)
        payload = json.loads(requests[0].data)
        self.assertEqual(payload["model"], "deepseek/deepseek-chat-v3.1")
        self.assertEqual(payload["seed"], 4)
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer test-key")
        self.assertEqual(response.tool_calls[0].name, "list_files")
        self.assertEqual(response.thinking, "checking files")
        self.assertEqual((response.prompt_tokens, response.completion_tokens), (11, 7))

    def test_other_served_model_is_rejected(self) -> None:
        with patch("agentic_quant.api_client.urlopen", return_value=FakeResponse(completion({"content": "hi"}, model="meta-llama/llama-3.3-70b"))):
            with self.assertRaises(RuntimeError):
                api_chat(model="deepseek/deepseek-chat-v3.1", messages=[{"role": "user", "content": "x"}], max_tokens=8, temperature=0.0)

    def test_persistent_rate_limit_raises_quota_exhausted(self) -> None:
        def limited(request, timeout):
            raise HTTPError(request.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"daily limit"))

        with patch("agentic_quant.api_client.urlopen", side_effect=limited), patch("agentic_quant.api_client.time.sleep"):
            with self.assertRaises(QuotaExhausted):
                api_chat(model="deepseek/deepseek-chat-v3.1", messages=[{"role": "user", "content": "x"}], max_tokens=8, temperature=0.0, attempts=2)

    def test_rejected_temperature_is_dropped(self) -> None:
        payloads = []

        def fake_urlopen(request, timeout):
            payloads.append(json.loads(request.data))
            if len(payloads) == 1:
                raise HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(b"temperature is not supported"))
            return FakeResponse(completion({"content": "ok"}, model="openai/gpt-5.2"))

        with patch("agentic_quant.api_client.urlopen", side_effect=fake_urlopen):
            api_chat(model="openai/gpt-5.2", messages=[{"role": "user", "content": "x"}], max_tokens=8, temperature=0.0)
        self.assertNotIn("temperature", payloads[1])

    def test_campaign_stops_on_quota_and_resumes_per_condition(self) -> None:
        calls = []

        def fake_run(**kwargs):
            calls.append((kwargs["seed"], kwargs["conditions"][0]))
            if len(calls) == 6:
                raise QuotaExhausted("limit")
            out = Path(kwargs["out"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "results.jsonl").write_text(json.dumps({"condition": kwargs["conditions"][0], "seed": kwargs["seed"]}) + "\n")
            (out / "config.json").write_text("{}")

        with tempfile.TemporaryDirectory() as tmp, patch.object(api_campaign, "run_experiment", side_effect=fake_run):
            root = Path(tmp)
            self.assertEqual(api_campaign.run_campaign("deepseek-v3.1", campaign="frontier", root=root, waves=2, tasks=["1"]), 1)
            self.assertEqual(api_campaign.run_campaign("deepseek-v3.1", campaign="frontier", root=root, waves=2, tasks=["1"]), 2)
            wave = root / "frontier" / "deepseek-v3.1" / "wave_02"
            lines = [json.loads(line) for line in (wave / "results.jsonl").read_text().splitlines()]
            self.assertEqual([line["condition"] for line in lines], api_campaign.CONDITIONS)
            self.assertTrue((wave / ".complete.json").exists())
            self.assertFalse((wave / "parts").exists())
            self.assertEqual(list(root.rglob(".running")), [])
        self.assertEqual(len(calls), 9)
        self.assertEqual(calls[5], calls[6])

    def test_free_preset_uses_its_own_endpoint_and_key(self) -> None:
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            return FakeResponse(completion({"content": "ok"}, model="@cf/moonshotai/kimi-k2.5"))

        with patch.dict(os.environ, {"CLOUDFLARE_API_TOKEN": "cf", "CLOUDFLARE_ACCOUNT_ID": "acc"}):
            provider = make_model_provider(model_key="kimi-k2.5-free")
            with patch("agentic_quant.api_client.urlopen", side_effect=fake_urlopen):
                provider.judge("x", max_tokens=8)
        self.assertEqual(requests[0].full_url, "https://api.cloudflare.com/client/v4/accounts/acc/ai/v1/chat/completions")
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer cf")

    def test_glm_free_caps_output_tokens(self) -> None:
        payloads = []

        def fake_urlopen(request, timeout):
            payloads.append(json.loads(request.data))
            return FakeResponse(completion({"content": "ok"}, model="zai-glm-4.7"))

        with patch.dict(os.environ, {"CEREBRAS_API_KEY": "c"}):
            provider = make_model_provider(model_key="glm-4.7-free")
            with patch("agentic_quant.api_client.urlopen", side_effect=fake_urlopen):
                provider.judge("x", max_tokens=8192)
        self.assertEqual(payloads[0]["max_tokens"], 2048)

if __name__ == "__main__":
    unittest.main()
