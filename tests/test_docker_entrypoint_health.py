import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import docker_entrypoint


class DockerEntrypointHealthTests(unittest.TestCase):
    @patch.object(docker_entrypoint, "time.sleep")
    @patch.object(docker_entrypoint, "time.monotonic", side_effect=[0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
    @patch.object(docker_entrypoint, "urlopen", side_effect=URLError("connection refused"))
    def test_wait_for_llama_cpp_surfaces_connection_error(
        self, urlopen, _monotonic, _sleep
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "connection refused") as context:
            docker_entrypoint.wait_for_llama_cpp("http://llama-cpp:8080", 1.0)

        self.assertIn("2 attempts", str(context.exception))
        self.assertEqual(urlopen.call_count, 2)

    @patch.object(docker_entrypoint, "time.sleep")
    @patch.object(docker_entrypoint, "time.monotonic", side_effect=[0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
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
        with self.assertRaisesRegex(RuntimeError, "503"):
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
