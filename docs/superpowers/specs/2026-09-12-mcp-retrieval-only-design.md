# MCP Retrieval-Only Design

## Goal

Make the Hiu Graph MCP server a retrieval-only knowledge interface. Gemma/llama.cpp remains required for GraphRAG indexing, but once a valid index exists every advertised MCP tool must work with llama.cpp stopped and must perform zero completion-model calls.

## Scope

This change only redesigns the MCP query-time surface.

In scope:
- remove generative GraphRAG search tools from the MCP server surface;
- add explicit vector/graph/source retrieval tools;
- use FastEmbed for query embeddings;
- use LanceDB and generated Parquet artifacts as the runtime data sources;
- return structured evidence for the calling MCP host LLM to reason over;
- update MCP documentation and tests;
- prove MCP works with no reachable llama.cpp endpoint after indexing.

Out of scope:
- changing GraphRAG indexing;
- removing Gemma or completion-model configuration from indexing;
- removing `core/search.py` generative GraphRAG helpers used by chat or other non-MCP flows;
- adding chat/Open WebUI in this change;
- adding an MCP-side answer-generation fallback.

## Current Problem

The current MCP server registers `search_knowledge_graph`, `local_search`, and `global_search`. Those tools call `graphrag.api.local_search` / `graphrag.api.global_search`, which synthesize an answer through the configured completion model. As a result an MCP request can invoke `openai/local-gemma` even though the MCP client already has its own LLM.

This creates an unnecessary LLM-to-LLM chain:

```text
Host LLM -> MCP -> GraphRAG -> Gemma -> generated answer -> Host LLM
```

It adds latency/GPU usage, makes MCP runtime depend on llama.cpp, and exposes structured-output failures such as JSON repair warnings.

## Target Architecture

```text
                         INDEXING
Documents
   |-- Gemma/llama.cpp -> entities, relationships, summaries, communities
   `-- FastEmbed       -> vector embeddings
                              |
                              v
                   Parquet + LanceDB index

                      MCP QUERY RUNTIME
Host LLM
   |
   v
Hiu Graph MCP
   |-- semantic_search ------> FastEmbed query embedding -> LanceDB text units
   |-- search_entities ------> FastEmbed query embedding -> LanceDB entity descriptions
   |-- get_entity -----------> entities.parquet
   |-- get_relationships ----> relationships.parquet + entities.parquet
   `-- get_sources ----------> text units/documents/source resolver

No completion model is loaded or called by any MCP tool.
```

## Public MCP Tool Surface

The MCP server will advertise only retrieval tools.

### `semantic_search(query: str, limit: int = 10)`

Purpose: retrieve the most semantically relevant indexed text units.

Behavior:
- validate `query` and `limit` at the MCP boundary;
- embed the query with the same FastEmbed model used by indexing (`BAAI/bge-small-en-v1.5` by default);
- query the `text_unit_text` LanceDB collection;
- hydrate matching text-unit metadata from the GraphRAG index when necessary;
- return ranked evidence, not a generated answer.

Each result should contain stable identifiers plus the useful evidence fields available in the index, including:
- text unit ID;
- text/content;
- similarity/distance score;
- document/source IDs where available.

### `search_entities(query: str, limit: int = 10)`

Purpose: semantically find entities whose indexed descriptions match the query.

Behavior:
- embed with FastEmbed;
- query the `entity_description` LanceDB collection;
- hydrate entity metadata from `entities.parquet`;
- return entity ID/title/type/description/community IDs and score.

This replaces the need to call generative local search merely to discover relevant entities.

### `get_entity(entity_name: str)`

Purpose: exact/case-insensitive entity lookup from the generated graph.

Behavior remains non-generative and is based on the existing entity-query implementation.

### `get_relationships(entity_name: str, limit: int = 20)`

Purpose: traverse the graph around a known entity without LLM reasoning.

Behavior:
- resolve the entity by title/name;
- select relationship rows where the entity is source or target;
- return relationship description/weight/rank fields that exist in the generated schema;
- identify the counterpart entity for each edge;
- enforce a bounded result limit.

### `get_sources(source_ids: list[str], limit: int = 20)`

Purpose: resolve evidence IDs returned by retrieval into source/document information that an MCP host can cite or inspect.

Behavior:
- reuse the existing source-resolution/data-loading utilities where possible;
- return document title/ID and text preview/content fields available in the generated index;
- never summarize with an LLM.

## Removed MCP Tools

The following names will no longer be registered as MCP tools:
- `search_knowledge_graph`;
- `local_search`;
- `global_search`.

Their MCP-specific wrapper modules may be removed once callers/tests are migrated. The underlying generative functions in `src/maf_graphrag/core/search.py` stay available for the chat backend and explicit non-MCP workflows.

This is an intentional MCP API breaking change in favor of a clear retrieval-only contract.

## Retrieval Components

### Query embedding

Create a small MCP retrieval component responsible for query embeddings. It will use FastEmbed directly or a thin shared helper extracted from the current FastEmbed adapter, but it must use the same model name/cache configuration as indexing. It must not load GraphRAG completion-model configuration.

The embedding model should be cached/singleton within the MCP process so repeated tool calls do not repeatedly initialize the model.

### LanceDB access

Create a focused vector-store adapter for the MCP tools. It owns:
- opening `output/lancedb`;
- querying `text_unit_text`;
- querying `entity_description`;
- converting vector-store rows into stable internal result objects;
- clear errors when the required collection/index is missing.

MCP tool modules should not contain raw LanceDB plumbing.

### Graph data access

Reuse `get_graph_data()` for Parquet-backed entity/relationship/text-unit hydration. Existing cached loading behavior should remain.

## Response Types

Replace `SearchResult`, whose contract currently requires an `answer`, with retrieval-oriented typed responses. Suggested model families:
- `SemanticSearchResult` containing `matches` and retrieval metadata;
- `EntitySearchResult` containing ranked entities;
- `RelationshipResult` containing resolved graph edges;
- `SourceResult` containing source/document evidence;
- existing `EntityQueryResult` may be retained for exact entity lookup if it still fits.

No retrieval response type contains an LLM-generated `answer` field.

## Error Handling

All retrieval tools use the existing structured `ToolError` style.

Errors must distinguish at least:
- invalid query/name/limit;
- missing or incomplete GraphRAG index;
- missing LanceDB table/collection;
- unknown entity;
- unknown source ID;
- query-embedding/vector-store failure.

An unreachable `LLAMA_CPP_BASE_URL` is not an MCP runtime error and must not be checked by MCP startup or MCP tools when the index already exists.

## Docker Runtime Contract

The existing first-run auto-index behavior remains:

```text
index missing -> wait for llama.cpp -> index -> verify index -> start MCP
index ready   -> start MCP immediately
```

After MCP starts with a complete index, query-time behavior is fully independent of llama.cpp.

The strongest runtime acceptance test is:
1. build/create a valid index;
2. stop llama.cpp or configure an unreachable `LLAMA_CPP_BASE_URL`;
3. start Hiu Graph MCP with the existing index;
4. invoke every advertised MCP tool;
5. verify successful retrieval and zero LiteLLM completion calls/network requests to llama.cpp.

## Testing Strategy

Use TDD for the implementation.

Unit tests:
- query embedding uses FastEmbed and caches the model;
- vector retrieval returns ranked text-unit results;
- entity semantic search returns hydrated entity data;
- relationship traversal handles source and target directions;
- source resolution handles valid/missing IDs and limits;
- boundary validation returns `ToolError` consistently.

MCP server tests:
- advertised tool list contains only retrieval tools;
- removed generative tool names are absent;
- server functions route directly to retrieval tool implementations;
- no test for MCP retrieval needs to mock `graphrag.api.local_search`, `global_search`, LiteLLM, or a completion model.

No-LLM regression test:
- monkeypatch completion-model/LiteLLM entry points to fail immediately if called;
- invoke each MCP retrieval tool against test index fixtures;
- all calls must still succeed.

Integration/container verification when available:
- run with a complete index and unreachable/stopped llama.cpp;
- confirm MCP health and retrieval calls succeed;
- inspect logs to confirm there are no `LiteLLM completion()` lines during MCP queries.

## Documentation and Migration

Update README and MCP documentation to make the split explicit:

- **Indexing:** requires Gemma/llama.cpp plus FastEmbed.
- **MCP runtime:** requires only the built index plus local FastEmbed for query embeddings; no completion LLM.
- MCP clients such as ChatGPT/Claude/Codex are responsible for reasoning over returned evidence.
- document the new retrieval tool names and example calls;
- call out removal of `search_knowledge_graph`, `local_search`, and `global_search` as a breaking MCP API change.

## Success Criteria

The redesign is complete when all of the following are true:

1. Gemma/llama.cpp is still used for initial/re-index GraphRAG generation.
2. No advertised MCP tool imports or invokes generative GraphRAG search.
3. MCP exposes only `semantic_search`, `search_entities`, `get_entity`, `get_relationships`, and `get_sources`.
4. `semantic_search` and `search_entities` use FastEmbed + LanceDB.
5. graph/source tools read generated index data directly.
6. MCP works after indexing with llama.cpp stopped/unreachable.
7. MCP query logs contain no LiteLLM completion calls.
8. Chat/non-MCP generative GraphRAG helpers remain available unless separately changed later.
