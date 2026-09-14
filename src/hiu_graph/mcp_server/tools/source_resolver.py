"""
Source Resolution Utilities

Resolves GraphRAG source/text-unit identifiers to meaningful document
references with titles and text previews for agent consumption.
"""

from __future__ import annotations

import pandas as pd

from maf_graphrag.core.data_loader import GraphData

TEXT_PREVIEW_LENGTH = 200


def _build_text_unit_lookup(data: GraphData) -> dict[int, str]:
    """Build text_unit human_readable_id → document_id lookup."""
    tu_to_doc: dict[int, str] = {}
    if data.text_units is None or data.text_units.empty:
        return tu_to_doc
    for _, row in data.text_units.iterrows():
        hrid = row.get("human_readable_id")
        doc_id = row.get("document_id")
        if hrid is not None and doc_id is not None:
            tu_to_doc[int(hrid)] = str(doc_id)
    return tu_to_doc


def _build_doc_title_lookup(data: GraphData) -> dict[str, str]:
    """Build document hash → title lookup."""
    doc_to_title: dict[str, str] = {}
    if data.documents is None or data.documents.empty:
        return doc_to_title
    for _, row in data.documents.iterrows():
        doc_id = row.get("id")
        title = row.get("title")
        if doc_id is not None and title is not None:
            doc_to_title[str(doc_id)] = str(title)
    return doc_to_title


def _make_text_preview(src_text: object) -> str:
    """Create a truncated text preview from a source row's text value."""
    text_str = str(src_text) if src_text else ""
    preview = text_str[:TEXT_PREVIEW_LENGTH]
    if len(text_str) > TEXT_PREVIEW_LENGTH:
        preview += "..."
    return preview


def _resolve_document(src_id: object, tu_to_doc: dict[int, str], doc_to_title: dict[str, str]) -> str:
    """Resolve a GraphRAG context source ID to a document title."""
    try:
        hrid = int(str(src_id))
        doc_hash = tu_to_doc.get(hrid)
        if doc_hash:
            return doc_to_title.get(doc_hash, "unknown")
    except (ValueError, TypeError):
        pass
    return "unknown"


def resolve_sources(sources_df: pd.DataFrame | None, data: GraphData) -> list[dict]:
    """Resolve legacy GraphRAG context source IDs to documents/previews."""
    if sources_df is None or sources_df.empty or "id" not in sources_df.columns:
        return []

    tu_to_doc = _build_text_unit_lookup(data)
    doc_to_title = _build_doc_title_lookup(data)
    has_mapping = bool(tu_to_doc and doc_to_title)
    results: list[dict] = []

    for _, src_row in sources_df.iterrows():
        src_id = src_row.get("id")
        preview = _make_text_preview(src_row.get("text", ""))
        entry: dict = {"text_unit_id": str(src_id)}
        if has_mapping:
            entry["document"] = _resolve_document(src_id, tu_to_doc, doc_to_title)
        if preview:
            entry["text_preview"] = preview
        results.append(entry)

    return results


def resolve_text_unit_ids(source_ids: list[str], data: GraphData, limit: int) -> tuple[list[dict], list[str]]:
    """Resolve actual text-unit IDs for retrieval-only MCP source lookup."""
    rows_by_id = {str(row.get("id")): row for _, row in data.text_units.iterrows()}
    doc_titles = _build_doc_title_lookup(data)

    unique_ids: list[str] = []
    seen: set[str] = set()
    for source_id in source_ids:
        key = str(source_id)
        if key not in seen:
            seen.add(key)
            unique_ids.append(key)

    results: list[dict] = []
    missing: list[str] = []
    for source_id in unique_ids:
        row = rows_by_id.get(source_id)
        if row is None:
            missing.append(source_id)
            continue
        if len(results) >= limit:
            continue

        entry: dict = {"text_unit_id": source_id}
        document_id = row.get("document_id")
        if document_id is not None:
            document_key = str(document_id)
            entry["document_id"] = document_key
            title = doc_titles.get(document_key)
            if title:
                entry["document_title"] = title
        preview = _make_text_preview(row.get("text", ""))
        if preview:
            entry["text_preview"] = preview
        results.append(entry)

    return results, missing


def get_unique_documents(sources: list[dict]) -> list[str]:
    """Extract unique document names from resolved sources."""
    docs = []
    seen: set[str] = set()
    for src in sources:
        doc = src.get("document", "")
        if doc and doc != "unknown" and doc not in seen:
            docs.append(doc)
            seen.add(doc)
    return docs
