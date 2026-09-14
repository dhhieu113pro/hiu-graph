FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS source

WORKDIR /context
COPY . .
RUN mkdir -p /prepared-output \
    && if [ -d /context/output ]; then cp -a /context/output/. /prepared-output/; fi

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8011 \
    GRAPHRAG_ROOT=/app \
    LLAMA_CPP_BASE_URL=http://host.docker.internal:8080 \
    FASTEMBED_MODEL_NAME=BAAI/bge-small-en-v1.5 \
    FASTEMBED_CACHE_DIR=/data/fastembed

COPY --from=source /context/pyproject.toml /context/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --from=source /context/settings.yaml /context/.env.example /context/run_mcp_server.py /context/docker_entrypoint.py ./
COPY --from=source /context/prompts ./prompts
COPY --from=source /context/input ./input
COPY --from=source /context/src ./src
COPY --from=source /prepared-output ./output

RUN mkdir -p /app/output /app/cache /data/fastembed

EXPOSE 8011

HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import socket; s=socket.create_connection(('127.0.0.1', 8011), 3); s.close()" || exit 1

CMD ["/app/.venv/bin/python", "docker_entrypoint.py"]
