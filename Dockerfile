FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS source

WORKDIR /context
COPY . .
RUN mkdir -p /prepared-output \
    && if [ -d /context/output ]; then cp -a /context/output/. /prepared-output/; fi

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ARG LLAMA_CPP_VERSION=b10702
ARG LLAMA_CPP_ARCHIVE=llama-b10702-bin-ubuntu-x64.tar.gz
ARG LLAMA_CPP_SHA256=20d3a2fad25914a9049100fb644053b4e10b1c4dcf4370f527afc73ea179da89

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8011 \
    GRAPHRAG_ROOT=/app \
    LLAMA_CPP_AUTOSTART=true \
    LLAMA_CPP_BASE_URL=http://127.0.0.1:8080 \
    LLAMA_CPP_BINARY=/usr/local/bin/llama-server \
    LLAMA_CPP_PORT=8080 \
    FASTEMBED_MODEL_NAME=BAAI/bge-small-en-v1.5 \
    FASTEMBED_CACHE_DIR=/data/fastembed

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL -o /tmp/llama.cpp.tar.gz "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_VERSION}/${LLAMA_CPP_ARCHIVE}" \
    && echo "${LLAMA_CPP_SHA256}  /tmp/llama.cpp.tar.gz" | sha256sum -c - \
    && mkdir -p /tmp/llama.cpp \
    && tar -xzf /tmp/llama.cpp.tar.gz -C /tmp/llama.cpp \
    && find /tmp/llama.cpp -type f -name llama-server -exec install -m 0755 {} /usr/local/bin/llama-server \; \
    && test -x /usr/local/bin/llama-server \
    && rm -rf /tmp/llama.cpp /tmp/llama.cpp.tar.gz \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

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