from __future__ import annotations

import unittest
from unittest.mock import patch

from agentic_quant.model_registry import get_model_preset
from agentic_quant.model_store import download_model, model_status


class OllamaIntegrationTest(unittest.TestCase):
    def test_ollama_presets_use_exact_ollama_tags(self) -> None:
        for key in ("gpt-oss:20b", "gpt-oss:20b-cloud", "qwen2.5:3b"):
            preset = get_model_preset(key)
            self.assertEqual(preset.backend, "ollama")
            self.assertEqual(preset.ollama_model, key)

    def test_spark_x25_uses_official_ollama_tag(self) -> None:
        preset = get_model_preset("spark-x2.5-4b")

        self.assertEqual(preset.backend, "ollama")
        self.assertEqual(preset.ollama_model, "SparkLLM/Spark-X2.5-4B:latest")

    def test_model_status_reads_ollama_inventory(self) -> None:
        with patch(
            "agentic_quant.model_store.ollama_model_info",
            return_value={"model": "gpt-oss:20b", "size": 13_000_000_000},
        ):
            status = model_status("gpt-oss:20b")

        self.assertTrue(status["downloaded"])
        self.assertEqual(status["path"], "ollama://gpt-oss:20b")
        self.assertEqual(status["bytes_total"], 13_000_000_000)

    def test_download_model_delegates_to_ollama_pull(self) -> None:
        progress = lambda _done, _total: None
        with patch("agentic_quant.model_store.pull_ollama_model") as pull:
            result = download_model("gpt-oss:20b-cloud", progress=progress)

        pull.assert_called_once_with("gpt-oss:20b-cloud", progress=progress)
        self.assertEqual(result.name, "gpt-oss:20b-cloud")

    def test_spark_download_delegates_to_official_ollama_pull(self) -> None:
        progress = lambda _done, _total: None
        with patch("agentic_quant.model_store.pull_ollama_model") as pull:
            result = download_model("spark-x2.5-4b", progress=progress)

        pull.assert_called_once_with(
            "SparkLLM/Spark-X2.5-4B:latest",
            progress=progress,
        )
        self.assertEqual(result.name, "spark-x2.5-4b")


if __name__ == "__main__":
    unittest.main()
