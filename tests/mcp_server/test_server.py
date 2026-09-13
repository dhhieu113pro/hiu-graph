"""Tests for retrieval-only MCP server wiring and dispatch."""

from unittest.mock import AsyncMock, patch

from fastmcp import Client


async def test_semantic_search_forwards_to_retrieval_tool():
    from maf_graphrag.mcp_server.server import semantic_search

    expected = {"matches": [], "returned": 0, "query_type": "semantic_text"}
    with patch(
        "maf_graphrag.mcp_server.server.semantic_search_tool",
        AsyncMock(return_value=expected),
    ) as tool:
        result = await semantic_search("database", limit=7)

    tool.assert_awaited_once_with("database", 7)
    assert result is expected


async def test_search_entities_forwards_to_retrieval_tool():
    from maf_graphrag.mcp_server.server import search_entities

    expected = {"matches": [], "returned": 0, "query_type": "semantic_entity"}
    with patch(
        "maf_graphrag.mcp_server.server.search_entities_tool",
        AsyncMock(return_value=expected),
    ) as tool:
        result = await search_entities("alpha", limit=4)

    tool.assert_awaited_once_with("alpha", 4)
    assert result is expected


async def test_get_entity_forwards_to_entity_query_tool():
    from maf_graphrag.mcp_server.server import get_entity

    expected = {"entities": [], "total_found": 0, "returned": 0, "available_types": [], "query_type": "lookup"}
    with patch("maf_graphrag.mcp_server.server.entity_query_tool", AsyncMock(return_value=expected)) as tool:
        result = await get_entity("Project Alpha")

    tool.assert_awaited_once_with(entity_name="Project Alpha", limit=1)
    assert result is expected


async def test_get_relationships_forwards_to_relationship_tool():
    from maf_graphrag.mcp_server.server import get_relationships

    expected = {"entity": "Project Alpha", "relationships": [], "returned": 0, "query_type": "relationship_lookup"}
    with patch("maf_graphrag.mcp_server.server.get_relationships_tool", AsyncMock(return_value=expected)) as tool:
        result = await get_relationships("Project Alpha", limit=8)

    tool.assert_awaited_once_with("Project Alpha", 8)
    assert result is expected


async def test_get_sources_forwards_to_source_tool():
    from maf_graphrag.mcp_server.server import get_sources

    expected = {"sources": [], "missing_ids": [], "returned": 0, "query_type": "source_lookup"}
    with patch("maf_graphrag.mcp_server.server.get_sources_tool", AsyncMock(return_value=expected)) as tool:
        result = await get_sources(["tu-1"], limit=6)

    tool.assert_awaited_once_with(["tu-1"], 6)
    assert result is expected


async def test_advertised_tools_are_retrieval_only():
    from maf_graphrag.mcp_server.server import mcp

    async with Client(mcp) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools} == {
        "semantic_search",
        "search_entities",
        "get_entity",
        "get_relationships",
        "get_sources",
    }


def test_create_mcp_server_returns_configured_instance():
    from maf_graphrag.mcp_server.server import create_mcp_server, mcp

    assert create_mcp_server() is mcp


def test_app_is_configured_asgi_application():
    from maf_graphrag.mcp_server.server import app

    assert callable(app)
