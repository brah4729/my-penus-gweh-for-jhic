# AGENTS.md

Guidance for AI coding agents (Claude Code, etc.) working in this repo.

## What this project is

A RAG-augmented, LoRA-fine-tuned chatbot for **SMK Plus Pelita Nusantara**
(an Indonesian vocational school). All user-facing text (system prompts,
fallback messages) is in **Bahasa Indonesia** — keep it that way unless
told otherwise.

Two things work together to keep the assistant accurate on a small model:
1. A fine-tuned 0.8B model (`Qwen/Qwen3.5-0.8B` + LoRA) that knows the
   *tone/behavior* of the assistant.
2. A TF-IDF RAG index over scraped school data that supplies the actual
   *facts* at query time, so the model doesn't have to recall/hallucinate
   them. If nothing relevant is retrieved, the code returns a fixed
   fallback message **without calling the LLM at all** — this was a
   deliberate fix after the model confidently hallucinated an answer
   ("Joko Widodo") to an out-of-scope question despite being told not to
   guess. Don't "fix" this by trusting the model's own refusal behavior.

## Pipeline (run in this order for a full rebuild)

1. `scrape/scrape.py` → `data/raw/*.json` (teacher, eskul, achievement,
   news, event, industry, testimonial — scraped from the school's site/API)
2. `build_dataset.py` → `data/datasets.jsonl` (SFT training examples,
   factual QA only — see "Known gaps" below)
3. `model.py` → LoRA fine-tune on CPU, merges to `school-assistant-merged/`
4. `llama.cpp/convert_hf_to_gguf.py` → convert merged model to GGUF
5. `llama.cpp/fix_qwen35_gguf.py` → patch GGUF metadata (see below) →
   `school-assistant-q4_k_m-fixed.gguf`
6. `build_rag_index.py` → `data/rag_index.pkl` (TF-IDF index used at
   inference time, independent of the fine-tuned model)
7. Serve: `llama-server` (from llama.cpp, localhost-only) behind
   `api_server.py` (FastAPI, does retrieval + calls llama-server). The
   client side is framework-agnostic — it's a plain REST endpoint, called
   from a Go/Fiber backend or Laravel identically. See the docstring in
   `api_server.py` for the two supported deployment topologies (same-VPS
   vs. separate-VPS, switched via the `HOST` env var) and for
   `ALLOWED_ORIGINS`.

`chat_with_rag.py` is the CLI equivalent of `api_server.py` for local
testing without standing up the HTTP layer. `chat_test.html` is a
standalone browser-based chat UI for manual testing — see the CORS note
below, since it needs `ALLOWED_ORIGINS` set to work against the current
`api_server.py`.

## `llama.cpp/fix_qwen35_gguf.py` — why it exists

Qwen3.5's exported config reports `block_count = N+1` (real transformer
blocks + a next-token-prediction/MTP head), but conversion only exports
the real `N` blocks' weights. This script rewrites three GGUF metadata
fields to match what's actually in the file (`block_count`,
`nextn_predict_layers`, `attention.recurrent_layers`), copying every
tensor and all other metadata unchanged. If you touch this, keep it a
metadata-only patch — don't let it start altering tensors.

Note: the rest of `llama.cpp/` is an upstream vendored checkout (it has
its own `AGENTS.md`/`CLAUDE.md`) — don't treat it as project code to
maintain; `fix_qwen35_gguf.py` is the one file in there that's actually
ours.

## `api_server.py` deployment model

Controlled by two env vars, checked at import time (so a misconfiguration
fails loudly on startup, not silently in production):

- **`HOST`** (default `127.0.0.1`) — signals which topology this instance
  is running under. `127.0.0.1`/`localhost`/`::1` means the backend
  (Fiber/Laravel/whatever) is on the same box, talking over loopback.
  Anything else means this hop crosses a network boundary.
- **`ASSISTANT_API_KEY`** — if unset AND `HOST` is loopback, the app
  prints a warning and starts anyway (fine for local dev). If unset AND
  `HOST` is *not* loopback, **the app refuses to start** (`SystemExit`)
  rather than silently exposing an unauthenticated `/chat` endpoint
  off-box. Don't relax this back to "warn and continue" for the
  off-loopback case — that was the whole point of the change.
- **`ALLOWED_ORIGINS`** — comma-separated list; CORS middleware is only
  added at all if this is non-empty. Default is CORS *closed*, not wide
  open — the assumption is server-to-server calls (Fiber/Laravel →
  here), which don't need CORS at all (no browser origin involved).
  **This means `chat_test.html` (browser-based manual testing) will fail
  silently unless you set `ALLOWED_ORIGINS=*` when running `uvicorn` for
  that purpose.** This is expected, not a bug — don't "fix" it by
  defaulting CORS back to open.

`llama-server` itself stays bound to `localhost` in every topology, full
stop — it is never the thing exposed off-box, `api_server.py` always is.

## Known gaps / footguns

- **`data/datasets.jsonl` only has factual QA.** `build_dataset.py`'s own
  docstring flags that Socratic-tutoring behavior (guide-don't-solve,
  refuse to hand over full code, off-topic redirects) still needs to be
  authored/synthesized separately and appended before training — it can't
  be derived from the scraped school data.
- **`README.md` is currently empty.**
- **`RELEVANCE_THRESHOLD = 0.12`** (in both `chat_with_rag.py` and
  `api_server.py`, kept in sync manually) is tuned for the current corpus
  size (real matches land ~0.35–0.43). Re-check it if the dataset grows —
  more documents shift the score distribution. If you change it in one
  file, change it in the other.
- **Aggregate/"list all X" documents can lose the ranking race** against
  individual per-item documents for broad queries (TF-IDF normalizes by
  document length, so a long aggregate doc's key term gets diluted vs. a
  short individual doc's). Confirmed happening for a "who are all the
  teachers" query, which returned 3 random individual teachers instead of
  the full `teacher_aggregate` list. Not yet fixed — if you touch
  retrieval ranking, this is a known open issue, not a regression you
  introduced.
- Package management is **`uv`**, not pip/poetry — use `uv run python ...`
  / `uv add ...`.

## Sensitive data — do not open

`data/datasets.jsonl` and everything under `data/raw/` contain real,
scraped personal data about students/staff/teachers at a real school.
**Don't read, print, quote, or paste contents of these files into chat or
into commits/PRs** unless the user explicitly asks you to open that
specific file. Treat `data/raw/*.json` the same way.

## Conventions

- System prompts / fallback strings: Bahasa Indonesia, informal but polite
  register (matches the existing `SYSTEM_PROMPT_WITH_CONTEXT` /
  `FALLBACK_MESSAGE` strings — reuse their phrasing style).
- Keep `chat_with_rag.py` and `api_server.py`'s retrieval logic
  (`retrieve()`, threshold, top-k) in sync — they're currently duplicated,
  not shared via an import. If you refactor one, check the other.

## Deployment (live, on a VPS via aaPanel + Docker Compose)

Deployed as a **single Docker image** combining `llama-server` and the
FastAPI RAG service — not two separate containers/images. VPS is managed
through aaPanel; the actual container is run from the terminal
(`/www/my-penus-gweh/`), not through aaPanel's own Docker Compose UI,
since that UI has no path field and only takes pasted compose content.

### Layout on the VPS
```
/www/my-penus-gweh/
  Dockerfile               # capital D — Linux is case-sensitive
  docker-compose.yml
  start.sh                 # must be LF line endings, not CRLF
  .dockerignore
  pyproject.toml           # slim: fastapi, uvicorn, requests, scikit-learn only
  api_server.py
  build_rag_index.py
  data/raw/
  models/school-assistant-q4_k_m-fixed.gguf
```
No separate `/api` subfolder — `WORKDIR /app/api` inside the Dockerfile
creates that path *inside the container*; the server-side folder itself
stays flat.

### Key design decisions
- **One container, not two.** Base image is
  `ghcr.io/ggml-org/llama.cpp:server` (Ubuntu-based). `start.sh` launches
  `llama-server` on `127.0.0.1:8080` (never exposed outside the
  container), waits on `/health`, then runs
  `uvicorn api_server:app --host 0.0.0.0 --port 8000`. This mirrors the
  same-VPS topology `api_server.py`'s own docstring describes, just with
  both processes inside one container instead of one box.
- **`export LD_LIBRARY_PATH=/app` is required** in `start.sh`, before the
  `llama-server` call — otherwise it fails silently in the background
  with `libllama-server-impl.so: cannot open shared object file`, and
  since it's backgrounded (`&`) that error doesn't surface as a build
  failure, only as an unhealthy container.
- **`start.sh` checks if llama-server's PID is still alive** in the
  `/health` polling loop (`kill -0 $LLAMA_PID`) and exits immediately if
  not, rather than polling forever. Without this, a model-load failure
  hangs the container indefinitely instead of failing fast.
- **`-t N` (thread cap)** is passed to `llama-server` in `start.sh` to
  stop a single request from pinning ~8 CPU cores (observed ~790% CPU per
  `docker stats` on one request without it). Must go *before* the `&`,
  not after — `... --parallel 2 -t 4 &`, not `... --parallel 2 & -t 4`
  (the latter runs `-t` as an unrelated shell command after backgrounding
  llama-server, hits `set -e`, and kills the script).
- **`pyproject.toml` is split by dependency group** — this only became
  necessary once building on the VPS, not during local dev.  Base
  (default) deps are API-only: `fastapi`, `uvicorn`, `requests`,
  `scikit-learn`. Training deps (`torch`, `transformers`, `peft`, `trl`,
  `unsloth`, `bitsandbytes`, `wandb`, `accelerate`, `datasets`,
  `huggingface-hub`, `numpy`/`pandas`/`matplotlib`) live under
  `[dependency-groups] train`. Scraping deps (`beautifulsoup4`, `scrapy`)
  live under `scrape`. The Dockerfile runs `uv sync --no-dev`, which only
  installs the base group — without this split the image pulls in ~5-8GB
  of CUDA/torch wheels it never uses. **Locally**, run
  `uv sync --all-groups` (or `--group train --group scrape` as needed)
  or `model.py`/`scrape/scrape.py` will fail with missing imports.
- **No `--frozen` in the Dockerfile**, and `uv.lock` is excluded via
  `.dockerignore` — uv resolves fresh from `pyproject.toml` on every
  build instead of trusting a lockfile. Trade-off: dependency versions
  can drift between rebuilds months apart. If reproducibility becomes
  more important than convenience, regenerate `uv.lock` locally
  (`uv lock`) against the slim `pyproject.toml`, upload it, and restore
  `--frozen`.
- **The `.gguf` model is a mounted volume, not baked into the image** —
  `models/` is excluded via `.dockerignore` and mounted read-only in
  `docker-compose.yml`. Swapping the model file only needs
  `docker compose restart` (no rebuild); changing the *filename* needs an
  edit to the `-m` path in `start.sh` **and** a rebuild, since `start.sh`
  itself is baked into the image.
- **`.dockerignore` must exclude**: `models`, `*.gguf`, `uv.lock`,
  `.venv`, `llama.cpp`, `school-assistant-merged` (and other
  `school-assistant-*` dirs/gguf variants) — otherwise multi-hundred-MB
  files get shipped into the build context on every single build.

### API contract (as actually deployed)
`POST /chat` body is `{"question": "..."}` — **not** `{"message": ...}`.
Auth via `X-API-Key` header checked against `ASSISTANT_API_KEY`.
`GET /health` is unauthenticated, returns `{"status": "ok",
"documents_loaded": <n>}` — fine to leave unauthenticated since it only
leaks a document count, but it's still reachable by anyone who can reach
the port, which is one more reason to keep that port firewalled/tunneled
rather than open to everyone.

### Compose file (as deployed)
```yaml
services:
  assistant:
    build:
      context: /www/my-penus-gweh
      dockerfile: Dockerfile
    environment:
      HOST: 0.0.0.0
      ASSISTANT_API_KEY: <random, via `openssl rand -hex 32` — rotate if
                          ever pasted anywhere, e.g. into a chat log>
    volumes:
      - /www/my-penus-gweh/models:/models:ro
    ports:
      - "127.0.0.1:8000:8000"   # bound to loopback only
    restart: unless-stopped
```
Exposure is via a tunnel (e.g. Cloudflare Tunnel) pointed at
`http://localhost:8000` on the VPS, rather than opening port 8000 in any
firewall — the port binds to `127.0.0.1` specifically so nothing external
can reach it directly even if the tunnel is misconfigured.

### Rebuild vs. restart — don't confuse these
- **Rebuild required** (`docker compose up -d --build`) for changes to:
  `api_server.py`, `start.sh`, `build_rag_index.py`, `pyproject.toml`,
  anything under `data/raw/`. All of these get `COPY`'d into the image at
  build time; a plain restart reuses the old image and silently ignores
  edited files on disk.
- **Restart only** (`docker compose up -d`, no `--build`) suffices for:
  changes to `docker-compose.yml` itself (env vars, ports, volumes), or
  swapping the `.gguf` file (same filename) since `models/` is a runtime
  mount.
- Only one Compose project should exist for this app
  (`my-penus-gweh`, managed from `/www/my-penus-gweh` via terminal). An
  earlier aaPanel-UI-created project (`school-assistant`, built from an
  older, since-abandoned two-container design) briefly coexisted and
  fought over port 8000 — it was deleted. If port 8000 conflicts resurface,
  check `docker ps` / aaPanel's Docker Compose list for a duplicate
  project first.

### Still open (unchanged by deployment — see "Known gaps" above)
Tutoring/refusal dataset, aggregate-ranking bug, concurrent benchmark,
post-quantization behavior check — none of this was addressed by getting
the container running; deployment only covers serving the current
(imperfect) model + RAG behavior.
