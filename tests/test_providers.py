from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agentic_quant.providers import GgufModelProvider
from agentic_quant.web_ui import INDEX_HTML


class GgufModelProviderTest(unittest.TestCase):
    def test_load_requests_all_available_gpu_layers(self) -> None:
        llama = Mock(return_value=object())
        provider = GgufModelProvider.__new__(GgufModelProvider)
        provider.model_key = "test-model"
        provider.n_ctx = 2048
        provider.preset = SimpleNamespace(
            label="Test model",
            repo_id="example/model",
            filename="model.gguf",
        )

        with (
            patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=llama)}),
            patch("agentic_quant.providers.model_status", return_value={"downloaded": True}),
            patch("agentic_quant.providers.model_path", return_value=Path("/models/model.gguf")),
        ):
            provider._load_llm()

        llama.assert_called_once_with(
            model_path="/models/model.gguf",
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False,
        )

    def test_gpu_notice_is_shown_above_agent_count(self) -> None:
        notice = "GPU acceleration is used automatically"
        self.assertIn(notice, INDEX_HTML)
        self.assertLess(INDEX_HTML.index(notice), INDEX_HTML.index("Number of agents"))


if __name__ == "__main__":
    unittest.main()
