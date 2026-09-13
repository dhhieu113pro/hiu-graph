"""Regression tests proving MCP retrieval does not depend on a completion LLM."""

import pandas as pd

from maf_graphrag.core.data_loader import GraphData
from maf_graphrag.mcp_server.retrieval.vector_store import VectorMatch


class FakeEncoder:
    def encode(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeVectorStore:
    def search(self, table_name: str, vector: list[float], limit: int) -> list[VectorMatch]:
        if table_name == "text_unit_text":
            return [VectorMatch("t1", 0.9)]
        if table_name == "entity_description":
            return [VectorMatch("e1", 0.8)]
        raise AssertionError(f"Unexpected table: {table_name}")


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
                },
                {"id": "e2", "title": "Sarah Chen", "type": "person", "description": "Lead", "community_ids": [1]},
            ]
        ),
        relationships=pd.DataFrame(
            [{"source": "Project Alpha", "target": "Sarah Chen", "description": "led by", "weight": 1.0}]
        ),
        communities=pd.DataFrame(),
        community_reports=pd.DataFrame(),
        text_units=pd.DataFrame(
            [{"id": "t1", "text": "Project Alpha is led by Sarah Chen", "document_id": "d1"}]
        ),
        documents=pd.DataFrame([{"id": "d1", "title": "alpha.md", "text": "Project Alpha is led by Sarah Chen"}]),
    )


async def test_all_mcp_tools_work_without_completion_model(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    monkeypatch.delenv("LLAMA_CPP_MODEL", raising=False)
    monkeypatch.delenv("LLAMA_CPP_MODEL_NAME", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("MCP retrieval attempted to call a generative GraphRAG API")

    import graphrag.api

    monkeypatch.setattr(graphrag.api, "local_search", forbidden)
    monkeypatch.setattr(graphrag.api, "global_search", forbidden)
    monkeypatch.setattr(graphrag.api, "basic_search", forbidden)
    monkeypatch.setattr(graphrag.api, "drift_search", forbidden)

    data = _graph_data()
    from maf_graphrag.mcp_server.tools import entity_query, relationships, retrieval_search, sources

    monkeypatch.setattr(retrieval_search, "get_query_encoder", lambda: FakeEncoder())
    monkeypatch.setattr(retrieval_search, "_get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(retrieval_search, "get_graph_data", lambda: data)
    monkeypatch.setattr(entity_query, "get_graph_data", lambda: data)
    monkeypatch.setattr(relationships, "get_graph_data", lambda: data)
    monkeypatch.setattr(sources, "get_graph_data", lambda: data)

    from maf_graphrag.mcp_server.server import (
        get_entity,
        get_relationships,
        get_sources,
        search_entities,
        semantic_search,
    )

    results = [
        await semantic_search("Project Alpha", 3),
        await search_entities("Project Alpha", 3),
        await get_entity("Project Alpha"),
        await get_relationships("Project Alpha", 3),
        await get_sources(["t1"], 3),
    ]

    assert all("error" not in result for result in results)
