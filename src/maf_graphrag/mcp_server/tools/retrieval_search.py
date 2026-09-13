"""Retrieval-only semantic search tools for the MCP server."""

from __future__ import annotations

from typing import Any

from maf_graphrag.mcp_server.config import MCPConfig
from maf_graphrag.mcp_server.retrieval.query_encoder import get_query_encoder
from maf_graphrag.mcp_server.retrieval.vector_store import LanceDbVectorStore
from maf_graphrag.mcp_server.tools._data_cache import get_graph_data
from maf_graphrag.mcp_server.tools.types import (
    EntitySearchMatch,
    EntitySearchResult,
    SemanticMatch,
    SemanticSearchResult,
    ToolError,
    handle_tool_errors,
    validate_limit,
    validate_query,
)

TEXT_UNIT_TABLE = "text_unit_text"
ENTITY_TABLE = "entity_description"


def _get_vector_store() -> LanceDbVectorStore:
    return LanceDbVectorStore(MCPConfig.from_env().lancedb_dir)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None]
    return [value]


def _as_string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _document_ids_from_row(row: Any) -> list[str]:
    document_ids = _as_string_list(row.get("document_ids"))
    if document_ids:
        return document_ids
    return _as_string_list(row.get("document_id"))


@handle_tool_errors("Semantic search")
async def semantic_search_tool(query: str, limit: int = 10) -> SemanticSearchResult | ToolError:
    """Retrieve semantically relevant text units without generating an answer."""
    if err := validate_query(query):
        return err
    if err := validate_limit(limit):
        return err

    vector = get_query_encoder().encode(query)
    matches = _get_vector_store().search(TEXT_UNIT_TABLE, vector, limit)
    data = get_graph_data()
    rows_by_id = {str(row.get("id")): row for _, row in data.text_units.iterrows()}

    hydrated: list[SemanticMatch] = []
    for match in matches:
        row = rows_by_id.get(match.id)
        if row is None:
            continue
        result = SemanticMatch(
            text_unit_id=match.id,
            text=str(row.get("text", "")),
            score=match.score,
        )
        document_ids = _document_ids_from_row(row)
        if document_ids:
            result["document_ids"] = document_ids
        hydrated.append(result)

    return SemanticSearchResult(matches=hydrated, returned=len(hydrated), query_type="semantic_text")


@handle_tool_errors("Entity search")
async def search_entities_tool(query: str, limit: int = 10) -> EntitySearchResult | ToolError:
    """Retrieve semantically relevant entities without generating an answer."""
    if err := validate_query(query):
        return err
    if err := validate_limit(limit):
        return err

    vector = get_query_encoder().encode(query)
    matches = _get_vector_store().search(ENTITY_TABLE, vector, limit)
    data = get_graph_data()
    rows_by_id = {str(row.get("id")): row for _, row in data.entities.iterrows()}

    hydrated: list[EntitySearchMatch] = []
    for match in matches:
        row = rows_by_id.get(match.id)
        if row is None:
            continue
        hydrated.append(
            EntitySearchMatch(
                entity_id=match.id,
                name=str(row.get("title", row.get("name", ""))),
                type=str(row.get("type", "unknown")),
                description=str(row.get("description", "")),
                community_ids=_as_list(row.get("community_ids")),
                score=match.score,
            )
        )

    return EntitySearchResult(matches=hydrated, returned=len(hydrated), query_type="semantic_entity")
