from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentic_quant.model_store import _download_model_locked, delete_model


class FakeResponse:
    def __init__(self, data: bytes, headers: dict[str, str], status: int) -> None:
        self._stream = io.BytesIO(data)
        self.headers = headers
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args) -> None:
        return None


class ModelStoreTest(unittest.TestCase):
    def test_partial_download_is_resumed_and_moved_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            final_path = Path(tmp_dir) / "model.gguf"
            part_path = final_path.with_suffix(".gguf.part")
            part_path.write_bytes(b"abc")
            preset = SimpleNamespace(
                label="Test model",
                download_urls=("https://example.invalid/model.gguf",),
                expected_bytes=6,
            )
            observed_range = []

            def fake_urlopen(request, timeout):
                observed_range.append(request.get_header("Range"))
                self.assertEqual(timeout, 60)
                return FakeResponse(b"def", {"Content-Length": "3", "Content-Range": "bytes 3-5/6"}, 206)

            progress = []
            with (
                patch("agentic_quant.model_store.get_model_preset", return_value=preset),
                patch("agentic_quant.model_store.model_file_paths", return_value=[final_path]),
                patch("agentic_quant.model_store.model_status", return_value={"downloaded": False}),
                patch("agentic_quant.model_store.urlopen", side_effect=fake_urlopen),
            ):
                result = _download_model_locked("test", progress=lambda done, total: progress.append((done, total)))

            self.assertEqual(result, final_path)
            self.assertEqual(final_path.read_bytes(), b"abcdef")
            self.assertFalse(part_path.exists())
            self.assertEqual(observed_range, ["bytes=3-"])
            self.assertEqual(progress[-1], (6, 6))

    def test_delete_model_removes_only_files_declared_by_the_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_dir = root / "selected"
            model_dir.mkdir()
            selected = model_dir / "selected.gguf"
            selected.write_bytes(b"weights")
            unrelated = root / "unrelated.gguf"
            unrelated.write_bytes(b"keep")
            preset = SimpleNamespace(backend="gguf")

            with (
                patch("agentic_quant.model_store.MODELS_DIR", root),
                patch("agentic_quant.model_store.get_model_preset", return_value=preset),
                patch("agentic_quant.model_store.model_file_paths", return_value=[selected]),
            ):
                removed = delete_model("selected")

            self.assertEqual(removed, [str(selected.resolve())])
            self.assertFalse(selected.exists())
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
