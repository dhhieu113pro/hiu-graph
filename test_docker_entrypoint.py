import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

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

    @patch.object(docker_entrypoint, "run_indexing")
    @patch.object(docker_entrypoint, "wait_for_llama_cpp")
    def test_prepare_index_waits_and_indexes_when_missing(
        self, wait_for_llama_cpp, run_indexing
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_indexing(_root: Path) -> None:
                self._ready_root(root)

            run_indexing.side_effect = fake_indexing

            with patch.dict(
                os.environ,
                {
                    "LLAMA_CPP_BASE_URL": "http://llama-cpp:8080",
                    "HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS": "42",
                },
                clear=False,
            ):
                docker_entrypoint.prepare_index(root)

            wait_for_llama_cpp.assert_called_once_with("http://llama-cpp:8080", 42.0)
            run_indexing.assert_called_once_with(root)
            self.assertTrue(docker_entrypoint.index_is_ready(root))

    @patch.object(docker_entrypoint, "run_indexing")
    @patch.object(docker_entrypoint, "wait_for_llama_cpp")
    def test_prepare_index_uses_docker_host_default(self, wait_for_llama_cpp, run_indexing) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_indexing(_root: Path) -> None:
                self._ready_root(root)

            run_indexing.side_effect = fake_indexing

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LLAMA_CPP_BASE_URL", None)
                os.environ.pop("HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS", None)
                docker_entrypoint.prepare_index(root)

            wait_for_llama_cpp.assert_called_once_with(
                docker_entrypoint.DEFAULT_LLAMA_CPP_BASE_URL, 300.0
            )
            run_indexing.assert_called_once_with(root)

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

            with self.assertRaisesRegex(RuntimeError, "index is incomplete"):
                docker_entrypoint.prepare_index(root)

            wait_for_llama_cpp.assert_called_once()
            run_indexing.assert_called_once_with(root)

    @patch.object(docker_entrypoint, "time.sleep")
    @patch.object(docker_entrypoint, "time.monotonic", side_effect=[0.0, 1.0])
    @patch.object(docker_entrypoint, "urlopen", side_effect=URLError("connection refused"))
    def test_wait_for_llama_cpp_surfaces_connection_error(
        self, urlopen, _monotonic, _sleep
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "connection refused") as context:
            docker_entrypoint.wait_for_llama_cpp("http://llama-cpp:8080", 1.0)

        self.assertIn("2 attempts", str(context.exception))
        self.assertEqual(urlopen.call_count, 2)

    @patch.object(docker_entrypoint, "time.sleep")
    @patch.object(docker_entrypoint, "time.monotonic", side_effect=[0.0, 1.0])
    @patch.object(
        docker_entrypoint,
        "urlopen",
        side_effect=HTTPError(
            "http://llama-cpp:8080/health", 503, "Service Unavailable", {}, None
        ),
    )
    def test_wait_for_llama_cpp_surfaces_http_error(
        self, urlopen, _monotonic, _sleep
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "HTTP Error 503"):
            docker_entrypoint.wait_for_llama_cpp("http://llama-cpp:8080", 1.0)

        self.assertEqual(urlopen.call_count, 2)

    @patch.object(docker_entrypoint, "urlopen")
    @patch.object(docker_entrypoint, "time.monotonic", return_value=0.0)
    def test_wait_for_llama_cpp_returns_when_health_is_200(self, _monotonic, urlopen) -> None:
        response = urlopen.return_value.__enter__.return_value
        response.status = 200

        docker_entrypoint.wait_for_llama_cpp("http://llama-cpp:8080", 30.0)

        urlopen.assert_called_once_with("http://llama-cpp:8080/health", timeout=5)


if __name__ == "__main__":
    unittest.main()
