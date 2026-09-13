"""Tests for direct graph relationship traversal."""

import pandas as pd

from maf_graphrag.core.data_loader import GraphData


def _graph_data() -> GraphData:
    return GraphData(
        entities=pd.DataFrame(
            [
                {"id": "e1", "title": "Project Alpha"},
                {"id": "e2", "title": "Sarah Chen"},
                {"id": "e3", "title": "PostgreSQL"},
            ]
        ),
        relationships=pd.DataFrame(
            [
                {
                    "source": "Project Alpha",
                    "target": "Sarah Chen",
                    "description": "led by",
                    "weight": 2.0,
                    "combined_degree": 3,
                },
                {
                    "source": "PostgreSQL",
                    "target": "Project Alpha",
                    "description": "used by",
                    "weight": 1.0,
                    "combined_degree": 2,
                },
            ]
        ),
        communities=pd.DataFrame(),
        community_reports=pd.DataFrame(),
        text_units=pd.DataFrame(),
    )


async def test_get_relationships_handles_both_edge_directions(monkeypatch):
    from maf_graphrag.mcp_server.tools import relationships

    monkeypatch.setattr(relationships, "get_graph_data", _graph_data)

    result = await relationships.get_relationships_tool("project alpha", limit=20)

    assert result["entity"] == "Project Alpha"
    assert [edge["counterpart"] for edge in result["relationships"]] == ["Sarah Chen", "PostgreSQL"]
    assert result["relationships"][0]["direction"] == "outgoing"
    assert result["relationships"][1]["direction"] == "incoming"
    assert result["relationships"][0]["combined_degree"] == 3.0
    assert result["returned"] == 2


async def test_get_relationships_uses_contains_fallback(monkeypatch):
    from maf_graphrag.mcp_server.tools import relationships

    monkeypatch.setattr(relationships, "get_graph_data", _graph_data)

    result = await relationships.get_relationships_tool("Alpha", limit=20)

    assert result["entity"] == "Project Alpha"


async def test_get_relationships_returns_error_for_unknown_entity(monkeypatch):
    from maf_graphrag.mcp_server.tools import relationships

    monkeypatch.setattr(relationships, "get_graph_data", _graph_data)

    result = await relationships.get_relationships_tool("Missing", limit=20)

    assert result["error"] == "Entity not found: Missing"


async def test_get_relationships_rejects_invalid_limit():
    from maf_graphrag.mcp_server.tools.relationships import get_relationships_tool

    result = await get_relationships_tool("Project Alpha", limit=0)

    assert result["error"] == "limit must be 1–100."
