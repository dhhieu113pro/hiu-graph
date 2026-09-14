import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import docker_entrypoint


class DockerEntrypointTests(unittest.TestCase):
    def _ready_root(self, root: Path) -> None:
        for relative_path in docker_entrypoint.REQUIRED_INDEX_PATHS:
            path = root / relative_path
            path.mkdir(parents=True, exist_ok=True)
            if path.suffix:
                path.touch()

    def test_index_is_ready_requires_every_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(docker_entrypoint.index_is_ready(root))
            self._ready_root(root)
            self.assertTrue(docker_entrypoint.index_is_ready(root))

    def test_build_llama_command_uses_model_and_server_defaults(self) -> None:
        with patch.dict(os.environ, {"LLAMA_CPP_PORT": "8080"}, clear=False):
            self.assertEqual(
                docker_entrypoint.build_llama_command("/models/model.gguf", "local-gemma"),
                [
                    docker_entrypoint.DEFAULT_LLAMA_CPP_BINARY,
                    "--model",
                    "/models/model.gguf",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8080",
                    "--alias",
                    "local-gemma",
                ],
            )

    @patch.object(docker_entrypoint, "wait_for_llama_cpp")
    @patch.object(docker_entrypoint, "run_indexing")
    @patch.object(docker_entrypoint, "start_llama_cpp")
    def test_prepare_index_autostarts_llama_and_stops_it_after_indexing(
        self, start_llama_cpp, run_indexing, wait_for_llama_cpp
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock()
            process.poll.return_value = None
            start_llama_cpp.return_value = process
            run_indexing.side_effect = lambda _root: self._ready_root(root)

            with patch.dict(
                os.environ,
                {
                    "LLAMA_CPP_AUTOSTART": "true",
                    "LLAMA_CPP_MODEL": "/models/model.gguf",
                    "LLAMA_CPP_MODEL_NAME": "local-gemma",
                },
                clear=False,
            ):
                docker_entrypoint.prepare_index(root)

            start_llama_cpp.assert_called_once_with("/models/model.gguf", "local-gemma")
            wait_for_llama_cpp.assert_called_once_with(
                docker_entrypoint.DEFAULT_LLAMA_CPP_BASE_URL, 300.0
            )
            run_indexing.assert_called_once_with(root)
            process.terminate.assert_called_once_with()
            process.wait.assert_called_once_with(timeout=10)

    @patch.object(docker_entrypoint, "run_indexing")
    @patch.object(docker_entrypoint, "wait_for_llama_cpp")
    def test_prepare_index_waits_on_external_llama_when_autostart_disabled(
        self, wait_for_llama_cpp, run_indexing
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_indexing.side_effect = lambda _root: self._ready_root(root)

            with patch.dict(
                os.environ,
                {
                    "LLAMA_CPP_AUTOSTART": "false",
                    "LLAMA_CPP_BASE_URL": "http://llama-cpp:8080",
                    "HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS": "42",
                },
                clear=False,
            ):
                docker_entrypoint.prepare_index(root)

            wait_for_llama_cpp.assert_called_once_with("http://llama-cpp:8080", 42.0)
            run_indexing.assert_called_once_with(root)

    @patch.object(docker_entrypoint, "wait_for_llama_cpp")
    @patch.object(docker_entrypoint, "start_llama_cpp")
    def test_prepare_index_requires_model_when_autostart_enabled(
        self, start_llama_cpp, wait_for_llama_cpp
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"LLAMA_CPP_AUTOSTART": "true"}, clear=False):
                os.environ.pop("LLAMA_CPP_MODEL", None)
                with self.assertRaisesRegex(RuntimeError, "LLAMA_CPP_MODEL"):
                    docker_entrypoint.prepare_index(root)

            start_llama_cpp.assert_not_called()
            wait_for_llama_cpp.assert_not_called()

    @patch.object(docker_entrypoint, "run_indexing")
    @patch.object(docker_entrypoint, "wait_for_llama_cpp")
    def test_prepare_index_skips_ready_index(self, wait_for_llama_cpp, run_indexing) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._ready_root(root)

            docker_entrypoint.prepare_index(root)

            wait_for_llama_cpp.assert_not_called()
            run_indexing.assert_not_called()

    @patch.object(docker_entrypoint, "run_indexing")
    @patch.object(docker_entrypoint, "wait_for_llama_cpp")
    def test_prepare_index_fails_if_indexing_is_incomplete(
        self, wait_for_llama_cpp, run_indexing
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"LLAMA_CPP_AUTOSTART": "false"}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "index is incomplete"):
                    docker_entrypoint.prepare_index(root)

            wait_for_llama_cpp.assert_called_once()
            run_indexing.assert_called_once_with(root)


if __name__ == "__main__":
    unittest.main()
