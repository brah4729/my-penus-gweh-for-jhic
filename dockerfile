FROM ghcr.io/ggml-org/llama.cpp:server

# The base image sets ENTRYPOINT to llama-server, so override it
ENTRYPOINT []

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app/api
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --python 3.11
COPY . .
RUN uv run python build_rag_index.py

RUN chmod +x start.sh
CMD ["./start.sh"]