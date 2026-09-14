# Hiu Graph

[![GHCR Package](https://img.shields.io/badge/GHCR-hiu--graph-2496ED?logo=docker&logoColor=white)](https://github.com/dhhieu113pro/hiu-graph/pkgs/container/hiu-graph)

Local GraphRAG knowledge graph exposed through an MCP server. GraphRAG indexing uses a local llama.cpp completion model plus FastEmbed. After the index exists, MCP is retrieval-only: it uses FastEmbed, LanceDB, and Parquet data and does not call a completion LLM. No Azure or OpenAI service is required.

## Runtime model

```text
Indexing / re-indexing:
  Documents -> Gemma/llama.cpp (entities, relationships, summaries, communities)
            -> FastEmbed (vectors)
            -> Parquet + LanceDB

MCP runtime after indexing:
  MCP client -> FastEmbed query embedding -> LanceDB/Parquet -> structured evidence
  Host LLM   -> reasons over that evidence and writes the answer
```

Gemma/llama.cpp is therefore required to create or rebuild the GraphRAG index, but it is not required for MCP queries once a valid index exists.

## Requirements

- Windows PowerShell
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- `llama-server.exe` on `PATH` for indexing/re-indexing
- A local GGUF model for indexing. The tested model is Gemma 3 4B Q4_K_M:
  `ggml-org/gemma-3-4b-it-GGUF:Q4_K_M`

## Quick start

```powershell
uv sync --dev
Copy-Item .env.example .env
```

For initial indexing, set `LLAMA_CPP_MODEL` in `.env` to your GGUF path, or let llama.cpp download the tested public model:

```powershell
llama-server -hf ggml-org/gemma-3-4b-it-GGUF:Q4_K_M `
  --alias local-gemma --host 127.0.0.1 --port 8080 `
  --ctx-size 4096 --parallel 1 --gpu-layers 16 `
  --temp 0.2 --n-predict 1200
```

Bootstrap indexing and MCP from another PowerShell window:

```powershell
.\bootstrap_local.ps1
```

The script starts llama.cpp when needed, builds the index when output files are missing, and starts MCP at `http://127.0.0.1:8011/mcp`.

Rebuild the generated index with:

```powershell
.\bootstrap_local.ps1 -ForceIndex
```

Generated Parquet and LanceDB files are stored in `output/` and ignored by Git. Once the index exists, you can stop llama.cpp and start only MCP:

```powershell
uv --cache-dir .uv-cache run python run_mcp_server.py
```

## MCP tools

MCP exposes exactly five retrieval primitives:

- `semantic_search(query, limit=10)` — ranked text-unit evidence from LanceDB
- `search_entities(query, limit=10)` — ranked entity matches from LanceDB
- `get_entity(entity_name)` — direct indexed entity lookup
- `get_relationships(entity_name, limit=20)` — direct graph-edge traversal
- `get_sources(source_ids, limit=20)` — resolve text-unit IDs to source/document evidence

A typical host flow is:

```text
1. semantic_search("Who leads Project Alpha?")
2. get_entity("Project Alpha")
3. get_relationships("Project Alpha")
4. get_sources(["<text-unit-id>"])
5. ChatGPT / Claude / Codex synthesizes the final answer from the returned evidence.
```

The MCP server itself does not synthesize answers and does not invoke LiteLLM or llama.cpp at query time.

> **Breaking MCP change:** `search_knowledge_graph`, `local_search`, `global_search`, and `list_entities` are no longer advertised MCP tools. Migrate callers to the retrieval primitives above.

For manual testing:

```powershell
npx @modelcontextprotocol/inspector
```

## Container image

[View the `hiu-graph` package on GitHub Container Registry](https://github.com/dhhieu113pro/hiu-graph/pkgs/container/hiu-graph)

Build the container image with Docker:

```powershell
docker build --tag ghcr.io/dhhieu113pro/hiu-graph:latest .
docker push ghcr.io/dhhieu113pro/hiu-graph:latest
```

End users can run an image that already contains a valid index without a completion-model endpoint:

```powershell
docker pull ghcr.io/dhhieu113pro/hiu-graph:latest
docker run --rm -p 8011:8011 `
  -v hiu-graph-fastembed:/data/fastembed `
  ghcr.io/dhhieu113pro/hiu-graph:latest
```

On startup the container checks the GraphRAG index. If all required index artifacts are already present, MCP starts immediately and does not check llama.cpp. If the index is missing or an empty `/app/output` mount hides the baked index, Hiu Graph waits for `LLAMA_CPP_BASE_URL`, builds the index from `/app/input`, verifies the generated artifacts, and only then starts MCP.

For first-run indexing with a truly empty output volume, provide the local llama.cpp endpoint/model settings:

```powershell
docker run --rm -p 8011:8011 `
  -e LLAMA_CPP_BASE_URL=http://host.docker.internal:8080 `
  -e LLAMA_CPP_MODEL_NAME=local-gemma `
  --mount type=volume,src=hiu-graph-output,dst=/app/output,volume-nocopy `
  -v hiu-graph-cache:/app/cache `
  -v hiu-graph-fastembed:/data/fastembed `
  ghcr.io/dhhieu113pro/hiu-graph:latest
```

`HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS` controls how long first-run startup waits for llama.cpp and defaults to `300` seconds. Later runs reuse the generated index and MCP queries remain independent of llama.cpp.

FastEmbed query-time and index-time embeddings use the same configuration:

```dotenv
FASTEMBED_MODEL_NAME=BAAI/bge-small-en-v1.5
FASTEMBED_CACHE_DIR=.cache/fastembed
```

Docker defaults the cache to `/data/fastembed`.

## Layout

```text
input/documents/                         source documents
output/                                  generated Parquet and LanceDB index
bootstrap_local.ps1                      llama.cpp + index + MCP bootstrap
docker_entrypoint.py                     container index check + first-run bootstrap
run_mcp_server.py                        FastMCP entry point
src/hiu_graph/core/                      indexing and optional generative search
src/hiu_graph/mcp_server/retrieval/      FastEmbed + LanceDB retrieval adapters
src/hiu_graph/mcp_server/tools/          retrieval-only MCP tools
```

## License

MIT — see [LICENSE](LICENSE).

## Reference and thanks

This project is based on and gratefully references [Cristopher Coronado's maf-graphrag-series](https://github.com/cristofima/maf-graphrag-series).
