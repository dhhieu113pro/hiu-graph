"""Tests for first-run Docker GraphRAG indexing and MCP startup."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from docker_entrypoint import (
    REQUIRED_INDEX_PATHS,
    index_is_ready,
    main,
    prepare_index,
    run_indexing,
    wait_for_llama_cpp,
)


def create_complete_index(root: Path) -> None:
    """Create the minimal set of artifacts required by the MCP runtime."""

    for relative_path in REQUIRED_INDEX_PATHS:
        path = root / relative_path
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ready", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)


def test_index_is_ready_when_all_required_artifacts_exist(tmp_path: Path) -> None:
    create_complete_index(tmp_path)

    assert index_is_ready(tmp_path)


def test_index_is_not_ready_when_one_artifact_is_missing(tmp_path: Path) -> None:
    create_complete_index(tmp_path)
    (tmp_path / "output/entities.parquet").unlink()

    assert not index_is_ready(tmp_path)


def test_prepare_index_skips_wait_and_index_when_ready(tmp_path: Path) -> None:
    create_complete_index(tmp_path)

    with patch("docker_entrypoint.wait_for_llama_cpp") as wait_mock, patch(
        "docker_entrypoint.run_indexing"
    ) as index_mock:
        prepare_index(tmp_path)

    wait_mock.assert_not_called()
    index_mock.assert_not_called()


def test_prepare_index_waits_and_indexes_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://llama:8080")
    monkeypatch.setenv("HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS", "12")

    with patch("docker_entrypoint.wait_for_llama_cpp") as wait_mock, patch(
        "docker_entrypoint.run_indexing"
    ) as index_mock, patch("docker_entrypoint.index_is_ready", side_effect=[False, True]):
        prepare_index(tmp_path)

    wait_mock.assert_called_once_with("http://llama:8080", 12.0)
    index_mock.assert_called_once_with(tmp_path)


def test_prepare_index_fails_if_output_is_still_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://llama:8080")

    with patch("docker_entrypoint.wait_for_llama_cpp"), patch("docker_entrypoint.run_indexing"), patch(
        "docker_entrypoint.index_is_ready", side_effect=[False, False]
    ):
        with pytest.raises(RuntimeError, match="GraphRAG index is incomplete"):
            prepare_index(tmp_path)


def test_wait_for_llama_cpp_returns_when_health_endpoint_is_ready() -> None:
    response = MagicMock()
    response.status = 200
    response.__enter__.return_value = response

    with patch("docker_entrypoint.urlopen", return_value=response) as urlopen_mock:
        wait_for_llama_cpp("http://llama:8080/", 5)

    urlopen_mock.assert_called_once_with("http://llama:8080/health", timeout=2)


def test_wait_for_llama_cpp_times_out() -> None:
    with patch("docker_entrypoint.urlopen", side_effect=URLError("offline")), patch(
        "docker_entrypoint.time.monotonic", side_effect=[0.0, 2.0]
    ), patch("docker_entrypoint.time.sleep") as sleep_mock:
        with pytest.raises(RuntimeError, match="llama.cpp did not become ready"):
            wait_for_llama_cpp("http://llama:8080", 1)

    sleep_mock.assert_not_called()


def test_run_indexing_invokes_existing_index_module(tmp_path: Path) -> None:
    with patch("docker_entrypoint.subprocess.run") as run_mock:
        run_indexing(tmp_path)

    run_mock.assert_called_once_with(
        [sys.executable, "-m", "maf_graphrag.core.index"],
        cwd=tmp_path,
        check=True,
    )


def test_main_prepares_index_then_execs_mcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHRAG_ROOT", str(tmp_path))

    with patch("docker_entrypoint.prepare_index") as prepare_mock, patch("docker_entrypoint.os.execv") as exec_mock:
        main()

    root = tmp_path.resolve()
    prepare_mock.assert_called_once_with(root)
    exec_mock.assert_called_once_with(
        sys.executable,
        [sys.executable, str(root / "run_mcp_server.py")],
    )
