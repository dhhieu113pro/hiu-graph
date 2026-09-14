import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
