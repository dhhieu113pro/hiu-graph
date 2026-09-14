"""Prepare the GraphRAG index before starting the Docker MCP server."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

REQUIRED_INDEX_PATHS = (
    Path("output/entities.parquet"),
    Path("output/relationships.parquet"),
    Path("output/communities.parquet"),
    Path("output/community_reports.parquet"),
    Path("output/text_units.parquet"),
    Path("output/lancedb"),
)

DEFAULT_LLAMA_CPP_BASE_URL = "http://host.docker.internal:8080"


def index_is_ready(root: Path) -> bool:
    """Return whether every artifact required by the MCP runtime exists."""

    return all((root / relative_path).exists() for relative_path in REQUIRED_INDEX_PATHS)


def wait_for_llama_cpp(base_url: str, timeout_seconds: float) -> None:
    """Wait until the configured llama.cpp health endpoint returns HTTP 200."""

    health_url = f"{base_url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_seconds
    print(f"GraphRAG index is missing; waiting for llama.cpp at {health_url}...", flush=True)

    while True:
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    print("llama.cpp is ready.", flush=True)
                    return
        except (URLError, TimeoutError, OSError):
            pass

        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError(
                f"llama.cpp did not become ready at {health_url} within {timeout_seconds:g} seconds. "
                "For Docker, set LLAMA_CPP_BASE_URL to the reachable llama.cpp endpoint "
                "(for example http://host.docker.internal:8080)."
            )
        time.sleep(min(2.0, deadline - now))


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

    base_url = os.getenv("LLAMA_CPP_BASE_URL", DEFAULT_LLAMA_CPP_BASE_URL)
    timeout_seconds = float(os.getenv("HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS", "300"))

    wait_for_llama_cpp(base_url, timeout_seconds)
    run_indexing(root)

    if not index_is_ready(root):
        raise RuntimeError("GraphRAG index is incomplete after indexing; MCP will not start")

    print("GraphRAG indexing completed successfully.", flush=True)


def main() -> None:
    """Prepare the index and replace this process with the MCP server."""

    root = Path(os.getenv("GRAPHRAG_ROOT", ".")).resolve()
    prepare_index(root)

    mcp_script = root / "run_mcp_server.py"
    print("Starting Hiu Graph MCP server...", flush=True)
    os.execv(sys.executable, [sys.executable, str(mcp_script)])


if __name__ == "__main__":
    main()
