"""Tests for direct text-unit/source retrieval."""

import pandas as pd

from maf_graphrag.core.data_loader import GraphData


def _graph_data() -> GraphData:
    return GraphData(
        entities=pd.DataFrame(),
        relationships=pd.DataFrame(),
        communities=pd.DataFrame(),
        community_reports=pd.DataFrame(),
        text_units=pd.DataFrame(
            [
                {"id": "tu-a", "human_readable_id": 0, "document_id": "doc-a", "text": "Alpha source text"},
                {"id": "tu-b", "human_readable_id": 1, "document_id": "doc-b", "text": "Beta source text"},
            ]
        ),
        documents=pd.DataFrame(
            [
                {"id": "doc-a", "title": "alpha.md", "text": "Full alpha document"},
                {"id": "doc-b", "title": "beta.md", "text": "Full beta document"},
            ]
        ),
    )


async def test_get_sources_preserves_order_deduplicates_and_reports_missing(monkeypatch):
    from maf_graphrag.mcp_server.tools import sources

    monkeypatch.setattr(sources, "get_graph_data", _graph_data)

    result = await sources.get_sources_tool(["tu-b", "tu-a", "tu-b", "missing"], limit=3)

    assert [item["text_unit_id"] for item in result["sources"]] == ["tu-b", "tu-a"]
    assert result["missing_ids"] == ["missing"]
    assert result["sources"][0]["document_id"] == "doc-b"
    assert result["sources"][0]["document_title"] == "beta.md"
    assert result["sources"][0]["text_preview"] == "Beta source text"
    assert result["returned"] == 2


async def test_get_sources_enforces_limit(monkeypatch):
    from maf_graphrag.mcp_server.tools import sources

    monkeypatch.setattr(sources, "get_graph_data", _graph_data)

    result = await sources.get_sources_tool(["tu-a", "tu-b"], limit=1)

    assert [item["text_unit_id"] for item in result["sources"]] == ["tu-a"]
    assert result["returned"] == 1


async def test_get_sources_rejects_empty_ids():
    from maf_graphrag.mcp_server.tools.sources import get_sources_tool

    result = await get_sources_tool([], limit=10)

    assert result["error"] == "source_ids must not be empty."


async def test_get_sources_rejects_invalid_limit():
    from maf_graphrag.mcp_server.tools.sources import get_sources_tool

    result = await get_sources_tool(["tu-a"], limit=0)

    assert result["error"] == "limit must be 1–100."
