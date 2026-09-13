"""Tests for the MCP LanceDB retrieval adapter."""

from pathlib import Path

import pytest


def test_score_from_distance_is_monotonic():
    from maf_graphrag.mcp_server.retrieval.vector_store import _score_from_distance

    assert _score_from_distance(0.0) == 1.0
    assert _score_from_distance(0.25) > _score_from_distance(1.0)
    assert _score_from_distance(-1.0) == 1.0


def test_search_returns_ranked_ids_with_normalized_scores(tmp_path: Path):
    import lancedb
    import pyarrow as pa

    db_path = tmp_path / "lancedb"
    db = lancedb.connect(str(db_path))
    db.create_table(
        "text_unit_text",
        data=pa.table(
            {
                "id": ["t1", "t2"],
                "vector": [[1.0, 0.0], [0.0, 1.0]],
                "text": ["alpha", "beta"],
            }
        ),
    )

    from maf_graphrag.mcp_server.retrieval.vector_store import LanceDbVectorStore

    results = LanceDbVectorStore(db_path).search("text_unit_text", [1.0, 0.0], 2)

    assert [match.id for match in results] == ["t1", "t2"]
    assert results[0].score > results[1].score


def test_search_rejects_missing_database(tmp_path: Path):
    from maf_graphrag.mcp_server.retrieval.vector_store import LanceDbVectorStore

    with pytest.raises(FileNotFoundError, match="LanceDB database not found"):
        LanceDbVectorStore(tmp_path / "missing").search("text_unit_text", [1.0, 0.0], 2)


def test_search_rejects_missing_table(tmp_path: Path):
    import lancedb

    db_path = tmp_path / "lancedb"
    lancedb.connect(str(db_path))

    from maf_graphrag.mcp_server.retrieval.vector_store import LanceDbVectorStore

    with pytest.raises(LookupError, match="LanceDB table not found: text_unit_text"):
        LanceDbVectorStore(db_path).search("text_unit_text", [1.0, 0.0], 2)
