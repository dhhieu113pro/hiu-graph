# Hiu Graph

Local GraphRAG knowledge graph exposed through an MCP server. It uses
[llama.cpp](https://github.com/ggml-org/llama.cpp) for generation and FastEmbed
for local embeddings. No Azure or OpenAI service is required.

## Requirements

- Windows PowerShell
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- `llama-server.exe` on `PATH`
- A local GGUF model. The tested model is Gemma 3 4B Q4_K_M:
  `ggml-org/gemma-3-4b-it-GGUF:Q4_K_M`

## Quick start

```powershell
uv sync --dev
Copy-Item .env.example .env
```

Set `LLAMA_CPP_MODEL` in `.env` to your GGUF path. Or let llama.cpp download
the tested public model:

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

The script starts llama.cpp when needed, builds the index when output files
are missing, and starts MCP at `http://127.0.0.1:8011/mcp`.

Rebuild the generated index with:

```powershell
.\bootstrap_local.ps1 -ForceIndex
```

Generated Parquet and LanceDB files are stored in `output/` and ignored by
Git. To start only the MCP server:

```powershell
uv --cache-dir .uv-cache run python run_mcp_server.py
```

## MCP tools

- `search_knowledge_graph`
- `local_search`
- `global_search`
- `list_entities`
- `get_entity`

For manual testing:

```powershell
npx @modelcontextprotocol/inspector
```

## Container image

Every push to `master` publishes a container to GHCR:

```powershell
docker pull ghcr.io/dhhieu113pro/hiu-graph:latest
docker run --rm -p 8011:8011 `
  -v "${PWD}\output:/app/output:ro" `
  -e LLAMA_CPP_BASE_URL=http://host.docker.internal:8080 `
  -e LLAMA_CPP_MODEL=remote-model `
  -e LLAMA_CPP_MODEL_NAME=local-gemma `
  ghcr.io/dhhieu113pro/hiu-graph:latest
```

The image runs only the MCP server. Build the GraphRAG index first with the
local bootstrap, then mount `output/`. llama.cpp must be reachable from the
container at `LLAMA_CPP_BASE_URL`.

## Layout

```text
input/documents/             source documents
output/                      generated Parquet and LanceDB index
bootstrap_local.ps1          llama.cpp + index + MCP bootstrap
run_mcp_server.py            FastMCP entry point
src/maf_graphrag/core/       indexing and search
src/maf_graphrag/mcp_server/ MCP tools and server
```

## License

MIT — see [LICENSE](LICENSE).
