"""Shared type definitions and error handling for retrieval-only MCP responses."""

import functools
import logging
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

from typing_extensions import NotRequired, TypedDict  # noqa: UP035

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 2000
MAX_ENTITY_NAME_LENGTH = 200
MAX_LIMIT = 100
VALID_COMMUNITY_LEVELS = range(0, 5)


class SemanticMatch(TypedDict):
    text_unit_id: str
    text: str
    score: float
    document_ids: NotRequired[list[str]]


class SemanticSearchResult(TypedDict):
    matches: list[SemanticMatch]
    returned: int
    query_type: str


class EntitySearchMatch(TypedDict):
    entity_id: str
    name: str
    type: str
    description: str
    community_ids: list[Any]
    score: float


class EntitySearchResult(TypedDict):
    matches: list[EntitySearchMatch]
    returned: int
    query_type: str


class RelationshipInfo(TypedDict):
    source: str
    target: str
    counterpart: str
    direction: str
    description: NotRequired[str]
    weight: NotRequired[float]
    combined_degree: NotRequired[float]
    rank: NotRequired[float]


class RelationshipResult(TypedDict):
    entity: str
    relationships: list[RelationshipInfo]
    returned: int
    query_type: str


class SourceInfo(TypedDict):
    text_unit_id: str
    document_id: NotRequired[str]
    document_title: NotRequired[str]
    text_preview: NotRequired[str]


class SourceResult(TypedDict):
    sources: list[SourceInfo]
    missing_ids: list[str]
    returned: int
    query_type: str


class EntityInfo(TypedDict):
    name: str
    type: str
    description: str
    community_ids: list[Any]


class EntityQueryResult(TypedDict):
    entities: list[EntityInfo]
    total_found: int
    returned: int
    available_types: list[str]
    query_type: str


class ToolError(TypedDict):
    error: str
    details: NotRequired[str]
    query: NotRequired[str]
    entity_name: NotRequired[str | None]
    entity_type: NotRequired[str | None]


def validate_query(query: str) -> ToolError | None:
    if not query or not query.strip():
        return ToolError(error="Query must not be empty.")
    if len(query) > MAX_QUERY_LENGTH:
        return ToolError(error=f"Query must be at most {MAX_QUERY_LENGTH} characters (got {len(query)}).")
    return None


def validate_community_level(community_level: int | None) -> ToolError | None:
    if community_level is not None and community_level not in VALID_COMMUNITY_LEVELS:
        return ToolError(error=f"community_level must be 0–{VALID_COMMUNITY_LEVELS.stop - 1}.")
    return None


def validate_limit(limit: int) -> ToolError | None:
    if limit < 1 or limit > MAX_LIMIT:
        return ToolError(error=f"limit must be 1–{MAX_LIMIT}.")
    return None


def validate_entity_name(name: str | None) -> ToolError | None:
    if name is not None and len(name) > MAX_ENTITY_NAME_LENGTH:
        return ToolError(error=f"entity_name must be at most {MAX_ENTITY_NAME_LENGTH} characters.")
    return None


_P = ParamSpec("_P")
_T = TypeVar("_T")


def handle_tool_errors(
    tool_name: str,
) -> Callable[[Callable[_P, Coroutine[Any, Any, _T]]], Callable[_P, Coroutine[Any, Any, _T | ToolError]]]:
    """Wrap tool exceptions in the structured MCP ToolError contract."""

    def decorator(
        fn: Callable[_P, Coroutine[Any, Any, _T]],
    ) -> Callable[_P, Coroutine[Any, Any, _T | ToolError]]:
        @functools.wraps(fn)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T | ToolError:
            try:
                return await fn(*args, **kwargs)
            except FileNotFoundError as e:
                logger.warning("%s: knowledge graph not found — %s", tool_name, e)
                return ToolError(
                    error="Knowledge graph not found. Run indexing first: uv run python -m maf_graphrag.core.index",
                    details=str(e),
                )
            except Exception as e:
                logger.exception("%s failed", tool_name)
                return ToolError(error=f"{tool_name} failed: {e}")

        return wrapper

    return decorator
