#!/bin/bash
set -e

export LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

# -t caps threads per request (without it one request pinned ~8 cores).
# It must stay BEFORE the & -- see AGENTS.md.
/app/llama-server -m /models/school-assistant-q4_k_m-fixed.gguf \
  --host 127.0.0.1 --port 8080 -c 4096 --parallel 2 -t 4 &
LLAMA_PID=$!

# wait until llama-server is ready, but stop if it dies
until curl -sf http://127.0.0.1:8080/health > /dev/null; do
  if ! kill -0 $LLAMA_PID 2>/dev/null; then
    echo "llama-server exited, stopping container" >&2
    exit 1
  fi
  sleep 1
done

exec uv run uvicorn api_server:app --host 0.0.0.0 --port 8000
