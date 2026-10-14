from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import json

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


    def test_split_gguf_is_downloaded_and_imported_into_ollama(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            preset = SimpleNamespace(
                backend="ollama",
                ollama_model="glm:q4",
                repo_id="org/repo",
                hf_folder="Q4",
                modelfile="RENDERER glm-4.7\nPARSER glm-4.7\n",
                filenames=(),
            )
            tree = [
                {"path": "Q4/m-00002-of-00002.gguf", "size": 2},
                {"path": "Q4/m-00001-of-00002.gguf", "size": 3},
            ]
            shards = {"m-00001-of-00002.gguf": b"abc", "m-00002-of-00002.gguf": b"de"}
            requested = []

            def fake_urlopen(request, timeout):
                url = request.full_url
                requested.append(url)
                if "/api/models/" in url:
                    return FakeResponse(json.dumps(tree).encode(), {}, 200)
                data = shards[url.rsplit("/", 1)[1]]
                return FakeResponse(data, {"Content-Length": str(len(data))}, 200)

            created = {}

            def fake_run(command, check):
                folder = root / "glm"
                created["command"] = command
                created["modelfile"] = (folder / "Modelfile").read_text(encoding="utf-8")
                created["files"] = sorted(path.name for path in folder.glob("*.gguf"))

            with (
                patch("agentic_quant.model_store.MODELS_DIR", root),
                patch("agentic_quant.model_store.get_model_preset", return_value=preset),
                patch("agentic_quant.model_store.ollama_model_info", return_value=None),
                patch("agentic_quant.model_store.urlopen", side_effect=fake_urlopen),
                patch("agentic_quant.model_store.subprocess.run", side_effect=fake_run),
                patch.dict("os.environ", {"OLLAMA_MODELS": str(root / "ollama")}),
            ):
                _download_model_locked("glm")

            self.assertEqual(requested[0], "https://huggingface.co/api/models/org/repo/tree/main/Q4")
            self.assertEqual(requested[1], "https://huggingface.co/org/repo/resolve/main/Q4/m-00001-of-00002.gguf")
            self.assertEqual(created["command"][:3], ["ollama", "create", "glm:q4"])
            self.assertEqual(created["modelfile"], f"FROM {root / 'glm'}\nRENDERER glm-4.7\nPARSER glm-4.7\n")
            self.assertEqual(created["files"], ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"])
            self.assertFalse((root / "glm").exists())
            blobs = {path.name: path.read_bytes() for path in (root / "ollama" / "blobs").iterdir()}
            self.assertEqual(blobs, {f"sha256-{hashlib.sha256(data).hexdigest()}": data for data in shards.values()})

    def test_imported_model_is_not_downloaded_again(self) -> None:
        preset = SimpleNamespace(backend="ollama", ollama_model="glm:q4", repo_id="org/repo", hf_folder="Q4", modelfile="", filenames=())
        with (
            patch("agentic_quant.model_store.get_model_preset", return_value=preset),
            patch("agentic_quant.model_store.ollama_model_info", return_value={"model": "glm:q4"}),
            patch("agentic_quant.model_store.urlopen") as urlopen,
            patch("agentic_quant.model_store.subprocess.run") as run,
        ):
            _download_model_locked("glm")
        urlopen.assert_not_called()
        run.assert_not_called()

if __name__ == "__main__":
    unittest.main()
