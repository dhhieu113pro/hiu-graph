"""Direct relationship traversal for retrieval-only MCP queries."""

from __future__ import annotations

import pandas as pd

from maf_graphrag.mcp_server.tools._data_cache import get_graph_data
from maf_graphrag.mcp_server.tools.types import (
    RelationshipInfo,
    RelationshipResult,
    ToolError,
    handle_tool_errors,
    validate_entity_name,
    validate_limit,
)


def _resolve_entity_title(entity_name: str, entities: pd.DataFrame) -> str | None:
    if entities.empty:
        return None
    column = "title" if "title" in entities.columns else "name" if "name" in entities.columns else None
    if column is None:
        return None

    values = entities[column].astype(str)
    exact = entities[values.str.casefold() == entity_name.casefold()]
    if not exact.empty:
        return str(exact.iloc[0][column])

    contains = entities[values.str.contains(entity_name, case=False, na=False, regex=False)]
    if not contains.empty:
        return str(contains.iloc[0][column])
    return None


def _optional_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


@handle_tool_errors("Relationship lookup")
async def get_relationships_tool(entity_name: str, limit: int = 20) -> RelationshipResult | ToolError:
    """Return graph edges connected to an entity without LLM reasoning."""
    if err := validate_entity_name(entity_name):
        return err
    if not entity_name or not entity_name.strip():
        return ToolError(error="entity_name must not be empty.")
    if err := validate_limit(limit):
        return err

    data = get_graph_data()
    canonical = _resolve_entity_title(entity_name.strip(), data.entities)
    if canonical is None:
        return ToolError(error=f"Entity not found: {entity_name}")

    relationships: list[RelationshipInfo] = []
    for _, row in data.relationships.iterrows():
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        if source == canonical:
            counterpart = target
            direction = "outgoing"
        elif target == canonical:
            counterpart = source
            direction = "incoming"
        else:
            continue

        item = RelationshipInfo(
            source=source,
            target=target,
            counterpart=counterpart,
            direction=direction,
        )
        description = row.get("description")
        if description is not None and not pd.isna(description):
            item["description"] = str(description)
        weight = _optional_number(row.get("weight"))
        if weight is not None:
            item["weight"] = weight
        rank = _optional_number(row.get("rank"))
        if rank is not None:
            item["rank"] = rank
        relationships.append(item)
        if len(relationships) >= limit:
            break

    return RelationshipResult(
        entity=canonical,
        relationships=relationships,
        returned=len(relationships),
        query_type="relationship_lookup",
    )
