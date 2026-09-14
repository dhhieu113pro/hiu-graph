"""Prepare the GraphRAG index before starting the Docker MCP server."""

from __future__ import annotations

import logging
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REQUIRED_INDEX_PATHS = (
    Path("output/entities.parquet"),
    Path("output/relationships.parquet"),
    Path("output/communities.parquet"),
    Path("output/community_reports.parquet"),
    Path("output/text_units.parquet"),
    Path("output/lancedb"),
)

DEFAULT_LLAMA_CPP_BINARY = "/usr/local/bin/llama-server"
DEFAULT_LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_EXTERNAL_LLAMA_CPP_BASE_URL = "http://host.docker.internal:8080"
DEFAULT_LLAMA_CPP_MODEL_NAME = "local-gemma"
logger = logging.getLogger(__name__)


def index_is_ready(root: Path) -> bool:
    """Return whether every artifact required by the MCP runtime exists."""

    return all((root / relative_path).exists() for relative_path in REQUIRED_INDEX_PATHS)


def wait_for_llama_cpp(base_url: str, timeout_seconds: float, poll_interval: float = 2.0) -> None:
    """Poll llama.cpp health until it returns HTTP 200 or the timeout expires."""

    health_url = f"{base_url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    last_error: Exception | None = None

    print(f"GraphRAG index is missing; waiting for llama.cpp at {health_url}...", flush=True)

    while time.monotonic() < deadline:
        attempt += 1
        try:
            with urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    logger.info(
                        "llama.cpp ready at %s after %d attempt(s)", health_url, attempt
                    )
                    print("llama.cpp is ready.", flush=True)
                    return
                last_error = RuntimeError(f"HTTP {response.status} from {health_url}")
        except HTTPError as exc:
            last_error = exc
        except (URLError, socket.timeout, TimeoutError, OSError) as exc:
            last_error = exc

        remaining = max(deadline - time.monotonic(), 0)
        if attempt % 5 == 0 or remaining <= 10:
            logger.warning(
                "Still waiting for llama.cpp at %s (attempt %d, ~%ds left): %s",
                health_url,
                attempt,
                int(remaining),
                last_error,
            )

        if remaining > 0:
            time.sleep(min(poll_interval, remaining))

    raise RuntimeError(
        f"llama.cpp did not become ready at {health_url} within {timeout_seconds:g}s "
        f"({attempt} attempts). Last error: {last_error!r}. "
        "For Docker, verify LLAMA_CPP_BASE_URL is reachable from inside the container "
        "(for example http://host.docker.internal:8080) and that llama-server is bound "
        "to 0.0.0.0, not 127.0.0.1 only. On native Linux Docker, add "
        "--add-host=host.docker.internal:host-gateway."
    ) from last_error


def build_llama_command(model_path: str, model_name: str) -> list[str]:
    """Build the llama-server command used for Docker first-run indexing."""

    command = [
        os.getenv("LLAMA_CPP_BINARY", DEFAULT_LLAMA_CPP_BINARY),
        "--model",
        model_path,
        "--host",
        os.getenv("LLAMA_CPP_HOST", "0.0.0.0"),
        "--port",
        os.getenv("LLAMA_CPP_PORT", "8080"),
        "--alias",
        model_name,
    ]

    extra_args = os.getenv("LLAMA_CPP_EXTRA_ARGS", "")
    if extra_args:
        command.extend(shlex.split(extra_args))

    return command


def start_llama_cpp(model_path: str, model_name: str) -> subprocess.Popen:
    """Start the bundled llama-server process for first-run indexing."""

    command = build_llama_command(model_path, model_name)
    print(f"Starting bundled llama.cpp: {' '.join(command)}", flush=True)
    try:
        return subprocess.Popen(command)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"llama.cpp server binary was not found at {command[0]}. "
            "Set LLAMA_CPP_BINARY to a valid llama-server executable."
        ) from exc


def stop_llama_cpp(process: subprocess.Popen) -> None:
    """Stop the temporary llama.cpp process after indexing."""

    if process.poll() is not None:
        return

    print("Stopping temporary llama.cpp after indexing...", flush=True)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_indexing(root: Path) -> None:
    """Run the existing Hiu Graph indexing module in the GraphRAG root."""

    print("Running first-run GraphRAG indexing...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "maf_graphrag.core.index"],
        cwd=root,
        check=True,
    )


def prepare_index(root: Path) -> None:
    """Ensure a complete GraphRAG index exists before MCP starts."""

    if index_is_ready(root):
        print("GraphRAG index is ready; skipping indexing.", flush=True)
        return

    autostart = os.getenv("LLAMA_CPP_AUTOSTART", "true").strip().lower() in {"1", "true", "yes", "on"}
    base_url = os.getenv(
        "LLAMA_CPP_BASE_URL",
        DEFAULT_LLAMA_CPP_BASE_URL if autostart else DEFAULT_EXTERNAL_LLAMA_CPP_BASE_URL,
    )
    timeout_seconds = float(os.getenv("HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS", "300"))
    llama_process: subprocess.Popen | None = None

    try:
        if autostart:
            model_path = os.getenv("LLAMA_CPP_MODEL", "").strip()
            if not model_path:
                raise RuntimeError(
                    "LLAMA_CPP_MODEL is required when LLAMA_CPP_AUTOSTART is enabled. "
                    "Mount a GGUF model into the container and set LLAMA_CPP_MODEL to its path."
                )
            model_name = os.getenv("LLAMA_CPP_MODEL_NAME", DEFAULT_LLAMA_CPP_MODEL_NAME)
            llama_process = start_llama_cpp(model_path, model_name)

        wait_for_llama_cpp(base_url, timeout_seconds)
        run_indexing(root)

        if not index_is_ready(root):
            raise RuntimeError("GraphRAG index is incomplete after indexing; MCP will not start")

        print("GraphRAG indexing completed successfully.", flush=True)
    finally:
        if llama_process is not None:
            stop_llama_cpp(llama_process)


def main() -> None:
    """Prepare the index and replace this process with the MCP server."""

    root = Path(os.getenv("GRAPHRAG_ROOT", ".")).resolve()
    prepare_index(root)

    mcp_script = root / "run_mcp_server.py"
    print("Starting Hiu Graph MCP server...", flush=True)
    os.execv(sys.executable, [sys.executable, str(mcp_script)])


if __name__ == "__main__":
    main()
