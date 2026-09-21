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
        loaded_model = SimpleNamespace(model_params=SimpleNamespace(n_gpu_layers=2_147_483_647))
        llama = Mock(return_value=loaded_model)
        llama_module = SimpleNamespace(
            Llama=llama,
            llama_supports_gpu_offload=lambda: True,
            llama_print_system_info=lambda: b"MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1",
        )
        provider = GgufModelProvider.__new__(GgufModelProvider)
        provider.model_key = "test-model"
        provider.n_ctx = 2048
        provider.seed = 7
        provider.preset = SimpleNamespace(
            label="Test model",
            repo_id="example/model",
            filename="model.gguf",
        )

        with (
            patch.dict(sys.modules, {"llama_cpp": llama_module}),
            patch("agentic_quant.providers.model_status", return_value={"downloaded": True}),
            patch("agentic_quant.providers.model_path", return_value=Path("/models/model.gguf")),
        ):
            provider._load_llm()

        llama.assert_called_once_with(
            model_path="/models/model.gguf",
            n_ctx=2048,
            n_gpu_layers=-1,
            seed=7,
            verbose=False,
        )
        self.assertEqual(provider.compute_unit["kind"], "gpu")
        self.assertEqual(provider.compute_unit["label"], "GPU (Metal)")
        self.assertTrue(provider.compute_unit["verified"])

    def test_compute_unit_indicator_is_shown_above_agent_count(self) -> None:
        indicator = 'id="compute-unit"'
        self.assertIn(indicator, INDEX_HTML)
        self.assertIn("updateComputeUnit(job)", INDEX_HTML)
        self.assertLess(INDEX_HTML.index(indicator), INDEX_HTML.index("Number of agents"))

    def test_visual_report_button_is_next_to_download_and_uses_run_data(self) -> None:
        visual_button = 'id="visual-report"'
        download_button = 'id="download-report"'
        self.assertIn(visual_button, INDEX_HTML)
        self.assertIn('id="visual-report-modal"', INDEX_HTML)
        self.assertIn("renderVisualReport(currentRunData)", INDEX_HTML)
        self.assertLess(INDEX_HTML.index(visual_button), INDEX_HTML.index(download_button))

    def test_a1_model_can_be_copied_to_every_agent(self) -> None:
        self.assertIn('id="copy-agent-model"', INDEX_HTML)
        self.assertIn("function copyFirstAgentModelToAll()", INDEX_HTML)
        self.assertIn("select.value = firstModel", INDEX_HTML)

    def test_linux_nvidia_gpu_rejects_a_cpu_only_llama_build(self) -> None:
        llama = Mock()
        llama_module = SimpleNamespace(
            Llama=llama,
            llama_supports_gpu_offload=lambda: False,
            llama_print_system_info=lambda: b"CPU : AVX2 = 1",
        )
        provider = GgufModelProvider.__new__(GgufModelProvider)
        provider.model_key = "test-model"
        provider.n_ctx = 2048
        provider.preset = SimpleNamespace(
            label="Test model",
            repo_id="example/model",
            filename="model.gguf",
        )

        with (
            patch.dict(sys.modules, {"llama_cpp": llama_module}),
            patch("agentic_quant.providers.model_status", return_value={"downloaded": True}),
            patch("agentic_quant.providers._nvidia_gpu_visible", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "agentic-quant-install-cuda"):
                provider._load_llm()

        llama.assert_not_called()

    def test_missing_cuda_shared_library_points_to_cuda_installer(self) -> None:
        provider = GgufModelProvider.__new__(GgufModelProvider)
        with (
            patch(
                "agentic_quant.providers._preload_cuda_libraries",
                side_effect=OSError("libcudart.so.12: cannot open shared object file"),
            ),
            patch("agentic_quant.providers._nvidia_gpu_visible", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "agentic-quant-install-cuda"):
                provider._load_llm()


if __name__ == "__main__":
    unittest.main()
