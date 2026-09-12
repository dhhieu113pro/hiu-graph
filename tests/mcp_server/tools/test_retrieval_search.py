"""Tests for retrieval-only semantic MCP tools."""

import pandas as pd

from maf_graphrag.core.data_loader import GraphData
from maf_graphrag.mcp_server.retrieval.vector_store import VectorMatch


def _graph_data() -> GraphData:
    return GraphData(
        entities=pd.DataFrame(
            [
                {
                    "id": "e1",
                    "title": "Project Alpha",
                    "type": "project",
                    "description": "Main project",
                    "community_ids": [1],
                }
            ]
        ),
        relationships=pd.DataFrame(),
        communities=pd.DataFrame(),
        community_reports=pd.DataFrame(),
        text_units=pd.DataFrame(
            [
                {
                    "id": "t1",
                    "text": "Project Alpha uses PostgreSQL",
                    "document_id": "d1",
                }
            ]
        ),
    )


class FakeEncoder:
    def encode(self, text: str) -> list[float]:
        assert text
        return [0.1, 0.2]


class FakeStore:
    def __init__(self, matches: list[VectorMatch]) -> None:
        self._matches = matches

    def search(self, table_name: str, vector: list[float], limit: int) -> list[VectorMatch]:
        assert table_name in {"text_unit_text", "entity_description"}
        assert vector == [0.1, 0.2]
        return self._matches[:limit]


async def test_semantic_search_returns_hydrated_text_units(monkeypatch):
    from maf_graphrag.mcp_server.tools import retrieval_search

    monkeypatch.setattr(retrieval_search, "get_graph_data", _graph_data)
    monkeypatch.setattr(retrieval_search, "get_query_encoder", lambda: FakeEncoder())
    monkeypatch.setattr(retrieval_search, "_get_vector_store", lambda: FakeStore([VectorMatch("t1", 0.9)]))

    result = await retrieval_search.semantic_search_tool("database", limit=5)

    assert result == {
        "matches": [
            {
                "text_unit_id": "t1",
                "text": "Project Alpha uses PostgreSQL",
                "score": 0.9,
                "document_ids": ["d1"],
            }
        ],
        "returned": 1,
        "query_type": "semantic_text",
    }


async def test_semantic_search_skips_stale_vector_rows(monkeypatch):
    from maf_graphrag.mcp_server.tools import retrieval_search

    monkeypatch.setattr(retrieval_search, "get_graph_data", _graph_data)
    monkeypatch.setattr(retrieval_search, "get_query_encoder", lambda: FakeEncoder())
    monkeypatch.setattr(retrieval_search, "_get_vector_store", lambda: FakeStore([VectorMatch("missing", 0.9)]))

    result = await retrieval_search.semantic_search_tool("database")

    assert result["matches"] == []
    assert result["returned"] == 0


async def test_search_entities_hydrates_entity_metadata(monkeypatch):
    from maf_graphrag.mcp_server.tools import retrieval_search

    monkeypatch.setattr(retrieval_search, "get_graph_data", _graph_data)
    monkeypatch.setattr(retrieval_search, "get_query_encoder", lambda: FakeEncoder())
    monkeypatch.setattr(retrieval_search, "_get_vector_store", lambda: FakeStore([VectorMatch("e1", 0.88)]))

    result = await retrieval_search.search_entities_tool("alpha", limit=5)

    assert result["matches"] == [
        {
            "entity_id": "e1",
            "name": "Project Alpha",
            "type": "project",
            "description": "Main project",
            "community_ids": [1],
            "score": 0.88,
        }
    ]
    assert result["returned"] == 1
    assert result["query_type"] == "semantic_entity"


async def test_semantic_search_rejects_empty_query():
    from maf_graphrag.mcp_server.tools.retrieval_search import semantic_search_tool

    result = await semantic_search_tool("", limit=5)

    assert result["error"] == "Query must not be empty."


async def test_search_entities_rejects_invalid_limit():
    from maf_graphrag.mcp_server.tools.retrieval_search import search_entities_tool

    result = await search_entities_tool("alpha", limit=0)

    assert result["error"] == "limit must be 1–100."
