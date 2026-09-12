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
- parameterize the FastEmbed model/cache settings so indexing and MCP retrieval share the same embedding configuration without changing their defaults;
- update MCP documentation and tests;
- prove MCP works with no reachable llama.cpp endpoint after indexing.

Out of scope:
- changing the GraphRAG indexing workflow or extraction/summarization behavior;
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

No completion model is constructed or called by any MCP tool.
```

## Shared Embedding Configuration

Index-time and query-time embeddings must use exactly the same model.

Add two environment-backed settings with current behavior as defaults:

- `FASTEMBED_MODEL_NAME=BAAI/bge-small-en-v1.5`
- `FASTEMBED_CACHE_DIR=.cache/fastembed` for local development; Docker may continue overriding the cache location to `/data/fastembed`.

`settings.yaml` uses these values for the GraphRAG embedding model. The MCP query encoder reads the same environment-backed values directly when constructing `fastembed.TextEmbedding`. This avoids loading completion-model configuration merely to embed an MCP query and prevents index/query model drift.

The indexing algorithm and default embedding model remain unchanged.

## Public MCP Tool Surface

The MCP server will advertise exactly five retrieval tools.

### `semantic_search(query: str, limit: int = 10)`

Purpose: retrieve the most semantically relevant indexed text units.

Behavior:
- validate `query` and `limit` at the MCP boundary;
- embed the query with the shared FastEmbed model;
- query the `text_unit_text` LanceDB collection;
- hydrate matching text-unit metadata from the GraphRAG index when necessary;
- return ranked evidence, not a generated answer.

Each match contains:
- `text_unit_id`;
- `text`;
- `score` (normalized so larger means more relevant at the MCP boundary, regardless of LanceDB's native distance representation);
- `document_ids` when present in the index.

### `search_entities(query: str, limit: int = 10)`

Purpose: semantically find entities whose indexed descriptions match the query.

Behavior:
- embed with the shared FastEmbed model;
- query the `entity_description` LanceDB collection;
- hydrate entity metadata from `entities.parquet`;
- return ranked entities.

Each match contains:
- `entity_id`;
- `name`;
- `type`;
- `description`;
- `community_ids`;
- `score`.

### `get_entity(entity_name: str)`

Purpose: exact/case-insensitive entity lookup from the generated graph.

Behavior remains non-generative and is based on the existing entity-query implementation. It returns entity metadata only and performs no vector search or LLM call.

### `get_relationships(entity_name: str, limit: int = 20)`

Purpose: traverse the graph around a known entity without LLM reasoning.

Behavior:
- resolve the entity by case-insensitive title/name;
- select relationship rows where the entity is source or target;
- return both direction and counterpart entity;
- return available relationship description/weight/rank fields;
- enforce the supplied result limit.

Each edge contains:
- `source`;
- `target`;
- `counterpart`;
- `description` when present;
- `weight` when present;
- `rank` when present.

### `get_sources(source_ids: list[str], limit: int = 20)`

Purpose: resolve evidence IDs returned by retrieval into source/document information that an MCP host can cite or inspect.

Behavior:
- reuse existing source-resolution/data-loading utilities;
- resolve text-unit/document identifiers into document metadata;
- return document ID/title plus available text preview/content fields;
- preserve caller order where possible, de-duplicate repeated IDs, and enforce `limit`;
- never summarize with an LLM.

## Removed MCP Tools and Modules

The following names will no longer be registered as MCP tools:
- `search_knowledge_graph`;
- `local_search`;
- `global_search`.

The MCP-specific wrapper modules `mcp_server/tools/local_search.py` and `mcp_server/tools/global_search.py` and their dedicated tests will be deleted after replacement retrieval tests are added. Their functionality remains available only through the underlying generative functions in `src/maf_graphrag/core/search.py` for chat or explicit non-MCP workflows.

This is an intentional MCP API breaking change in favor of a clear retrieval-only contract.

## Retrieval Components

### Query encoder

Create `mcp_server/retrieval/query_encoder.py` with a small `FastEmbedQueryEncoder` abstraction. It constructs `fastembed.TextEmbedding` from `FASTEMBED_MODEL_NAME` and `FASTEMBED_CACHE_DIR` and exposes a single-query encode operation.

The model instance is cached once per MCP process so repeated tool calls do not repeatedly initialize FastEmbed.

This component does not import `graphrag.api`, LiteLLM, or completion-model factories.

### LanceDB adapter

Create `mcp_server/retrieval/vector_store.py` as the only MCP module that contains raw LanceDB access. It owns:
- opening `output/lancedb`;
- querying `text_unit_text`;
- querying `entity_description`;
- converting native LanceDB distance values into MCP relevance scores;
- returning stable internal match objects;
- clear errors when the required database/table is absent.

MCP tool modules do not contain LanceDB plumbing.

### Graph data access

Reuse `get_graph_data()` for Parquet-backed entity/relationship/text-unit hydration. Existing cached loading behavior remains.

Add focused retrieval helpers only where necessary for relationship traversal and source resolution; do not duplicate whole Parquet datasets into a second cache layer.

## Response Types

Replace the answer-oriented `SearchResult` usage at the MCP boundary with definitive retrieval types:

- `SemanticMatch` and `SemanticSearchResult`;
- `EntitySearchMatch` and `EntitySearchResult`;
- `RelationshipInfo` and `RelationshipResult`;
- `SourceInfo` and `SourceResult`;
- retain `EntityInfo`/`EntityQueryResult` for exact entity lookup if their existing fields remain compatible.

No retrieval response type contains an `answer` field.

## Error Handling

All retrieval tools use the existing structured `ToolError` style.

Errors distinguish at least:
- invalid query/name/limit;
- missing or incomplete GraphRAG index;
- missing LanceDB database/table;
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
- query encoder uses the configured FastEmbed model and caches the model instance;
- vector retrieval returns ranked text-unit results and normalizes score direction;
- entity semantic search returns hydrated entity data;
- relationship traversal handles source and target directions;
- source resolution handles valid/missing IDs, de-duplication, caller order, and limits;
- boundary validation returns `ToolError` consistently.

MCP server tests:
- advertised tool list contains exactly `semantic_search`, `search_entities`, `get_entity`, `get_relationships`, and `get_sources`;
- removed generative tool names are absent;
- server functions route directly to retrieval tool implementations;
- no MCP retrieval test needs to mock `graphrag.api.local_search`, `graphrag.api.global_search`, LiteLLM, or a completion model.

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
- document the new retrieval tool names, schemas, and example calls;
- call out removal of `search_knowledge_graph`, `local_search`, and `global_search` as a breaking MCP API change.

## Success Criteria

The redesign is complete when all of the following are true:

1. Gemma/llama.cpp is still used for initial/re-index GraphRAG generation.
2. No advertised MCP tool imports or invokes generative GraphRAG search.
3. MCP exposes exactly `semantic_search`, `search_entities`, `get_entity`, `get_relationships`, and `get_sources`.
4. `semantic_search` and `search_entities` use the shared FastEmbed configuration plus LanceDB.
5. graph/source tools read generated index data directly.
6. MCP works after indexing with llama.cpp stopped/unreachable.
7. MCP query logs contain no LiteLLM completion calls.
8. Chat/non-MCP generative GraphRAG helpers remain available unless separately changed later.
