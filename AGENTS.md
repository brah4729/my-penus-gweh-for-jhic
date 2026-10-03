# AGENTS.md

> Open issues and fix plan: root `../issue.md`. If you fix an issue listed there, record it in that file (root `../AGENTS.md` §6 rule 7).

Guidance for AI coding agents (Claude Code, etc.) working in this repo.

## What this project is

A RAG-augmented, LoRA-fine-tuned chatbot for **SMK Plus Pelita Nusantara**
(an Indonesian vocational school). All user-facing text (system prompts,
fallback messages) is in **Bahasa Indonesia** — keep it that way unless
told otherwise.

Two things work together to keep the assistant accurate on a small model:
1. A fine-tuned 0.8B model (`Qwen/Qwen3.5-0.8B` + LoRA) that knows the
   *tone/behavior* of the assistant.
2. A hybrid RAG index (multilingual embeddings + TF-IDF) over scraped
   school data that supplies the actual *facts* at query time, so the
   model doesn't have to recall/hallucinate them. Off-topic questions
   (scope gate) and school questions with no matching data both get a
   fixed message **without calling the LLM at all** — this was a
   deliberate fix after the model confidently hallucinated an answer
   ("Joko Widodo") to an out-of-scope question despite being told not to
   guess. Don't "fix" this by trusting the model's own refusal behavior.
   All of this lives in `retrieval.py` — see its docstring.

## Pipeline (run in this order for a full rebuild)

1. `scrape/scrape.py` → `data/raw/*.json` (teacher, eskul, achievement,
   news, event, industry, testimonial — scraped from the school's site/API)
2. `build_dataset.py` → `data/datasets.jsonl` (SFT training examples,
   factual QA only — see "Known gaps" below)
3. `model.py` → LoRA fine-tune on CPU, merges to `school-assistant-merged/`
4. `llama.cpp/convert_hf_to_gguf.py` → convert merged model to GGUF
5. `llama.cpp/fix_qwen35_gguf.py` → patch GGUF metadata (see below) →
   `school-assistant-q4_k_m-fixed.gguf`
6. `build_rag_index.py` → `data/rag_index.pkl` (TF-IDF matrix +
   embeddings of every document and of the `scope_examples.json` example
   questions; independent of the fine-tuned model). Downloads the
   embedding model into `.cache/fastembed/` on first run. Then
   `eval_retrieval.py` to check scope/ranking didn't regress.
7. Serve: `llama-server` (from llama.cpp, localhost-only) behind
   `api_server.py` (FastAPI, does retrieval + calls llama-server). The
   client side is framework-agnostic — it's a plain REST endpoint, called
   from a Go/Fiber backend or Laravel identically. See the docstring in
   `api_server.py` for the two supported deployment topologies (same-VPS
   vs. separate-VPS, switched via the `HOST` env var) and for
   `ALLOWED_ORIGINS`.

`chat_with_rag.py` is the CLI equivalent of `api_server.py` for local
testing without standing up the HTTP layer; both import `retrieval.py`.
`chat_test.html` is a standalone browser-based chat UI for manual testing
— see the CORS note below, since it needs `ALLOWED_ORIGINS` set to work
against the current `api_server.py`.

`prune_vocab.py` is an **experimental, optional** step between 3 and 4:
it shrinks the tokenizer vocab + embedding of `school-assistant-merged/`
to only the tokens the dataset, RAG corpus and runtime prompts use →
`school-assistant-pruned/` (and the `school-assistant-pruned-*.gguf`
files). Production still serves the **unpruned**
`school-assistant-q4_k_m-fixed.gguf`. Its runtime-prompt list imports
from `retrieval.py`, so new user-facing strings there are covered
automatically — the first pruning attempt broke precisely because the
system prompt wasn't in the vocab scan.

## Running locally

```bash
uv sync --all-groups                      # base + train + scrape + dev
uv run python build_rag_index.py          # after any data/raw or scope_examples.json change
uv run python eval_retrieval.py           # regression check, no LLM needed
llama-server.exe -m school-assistant-q4_k_m-fixed.gguf --parallel 2 -c 4096   # prebuilt llama.cpp release binary, not built from llama.cpp/
uv run python chat_with_rag.py "Siapa yang mengajar matematika?"
uv run uvicorn api_server:app --port 8000   # HTTP API, loopback, no key needed locally
```

## Retrieval (`retrieval.py`)

Every question goes through two decisions **in code, before the LLM**:

| Outcome | Condition | Reply | LLM called? |
|---|---|---|---|
| Off-topic | scope gate says no | `FALLBACK_MESSAGE` | no |
| School, no data | in scope, no doc ≥ `DOC_FLOOR` | `NO_DATA_MESSAGE` | no |
| Answerable | in scope, docs found | model answers from top-k docs | yes |

1. **Scope gate** — the question's embedding is compared with the
   labelled example questions in `scope_examples.json` (`school` vs
   `off_topic`, mean of top-3 similarities per side). In scope if closer
   to `school`, or if TF-IDF finds a strong exact hit (≥ 0.40 — teacher /
   partner names no example list can cover).
2. **Ranking** — `embedding_cosine + 0.3 × tfidf_cosine`, top 3, floored
   at `DOC_FLOOR`. Embeddings handle paraphrase/slang ("pengajar MTK" →
   math teacher, which TF-IDF scored 0.00); the TF-IDF term keeps exact
   names sharp.

Files and knobs:
- **Embedding model**: `paraphrase-multilingual-mpnet-base-v2` via
  `fastembed` (ONNX, no torch). Chosen at index-build time via the
  `EMBED_MODEL` env var and stored in the index; `Retriever` reads it
  from there, so build and runtime can't disagree. Weights download into
  `.cache/fastembed/` (gitignored/dockerignored locally; in Docker it's
  downloaded during `build_rag_index.py` at build time and baked in).
  Query embedding runs on `EMBED_THREADS = 1` so it doesn't compete with
  llama-server for cores.
- **`scope_examples.json`** — the tuning surface for scope. A real
  question wrongly rejected/accepted → add a similar example to the right
  list → rebuild the index → re-run the eval. Don't copy
  `eval_retrieval.py`'s questions into it (they're the held-out set).
- **Aggregate docs** (`*_aggregate`, "list all X") are embedded by a
  short `key` description, not their long name-list text — otherwise the
  list washes out the embedding and "siapa saja guru" returns random
  individual teachers. Give any new aggregate a `key`.
- **`eval_retrieval.py`** — held-out in-scope / off-topic questions plus
  exact-name questions generated from the index (never printed). Reports
  scope accept/leak counts and whether the expected source category
  ranks first. Run it after touching anything above.

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

Controlled by three env vars, checked at import time (so a misconfiguration
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
- **Retrieval thresholds in `retrieval.py` are calibrated, not
  universal** — tuned for `paraphrase-multilingual-mpnet-base-v2` on the
  current 221-doc corpus. At last tuning (2026-09-30): 32/32 held-out
  school questions accepted, 1/20 off-topic leaked, right category ranked
  first 23/25. (Old TF-IDF-only gate on the same set: 22/32 accepted,
  6/20 leaked.) Re-run `eval_retrieval.py` if the corpus grows.
- **Pure score thresholds can't decide scope or "do we have data"** —
  measured: off-topic questions score up to ~0.50 against the documents
  while some real school questions score ~0.45. That's why scope is a
  separate nearest-example check. Don't "simplify" it back to a single
  retrieval-score threshold, and fix misclassifications via
  `scope_examples.json`, not by moving a threshold.
- **Known leak: "siapa menteri pendidikan"** passes the scope gate
  ("pendidikan" pulls it toward school) and reaches the LLM with teacher
  documents as context. Left as-is rather than overfitting the example
  list to the eval set.
- **Missing data, not missing search: some school questions reach the
  LLM with unrelated context** (e.g. "ada beasiswa ga" retrieves news
  articles above `DOC_FLOOR`). The scraper only collects teacher/eskul/
  achievement/news/event/industry/testimonial — nothing covers PPDB,
  fees, address, majors, contact, facilities. The real fix is a
  hand-written school profile/FAQ source in `build_rag_index.py`, not
  more tuning.
- **Embedding model costs ~1.3 GB RAM** in the API process (~30 ms/query
  on 1 thread). MiniLM
  (`EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
  is ~0.5 GB but rejected common questions like "ada ekskul apa aja?" in
  testing — if the VPS is RAM-starved, switch, rebuild, and re-tune the
  thresholds with `eval_retrieval.py`.
- **`uv run` / `uv add` sync the default groups only** (base + dev). They
  don't remove already-installed train/scrape packages, but a fresh
  checkout needs `uv sync --all-groups` before `model.py` /
  `scrape/scrape.py` / `prune_vocab.py` will import.
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
  `FALLBACK_MESSAGE` / `NO_DATA_MESSAGE` strings — reuse their phrasing
  style). Same for `scope_examples.json`: write examples the way students
  actually type (casual, slang, abbreviations like "MTK", "ekskul").
- Retrieval logic, thresholds, system prompt and fallback strings live
  only in `retrieval.py` — `api_server.py`, `chat_with_rag.py` and
  `prune_vocab.py` import them. Don't re-duplicate them.
- `eval_retrieval.py` and anything else that inspects the index must
  print only scores/source tags, never document text (see "Sensitive
  data").

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
  start.sh                 # must be LF line endings, not CRLF (.gitattributes enforces this)
  .env                     # ASSISTANT_API_KEY only -- not in git
  .dockerignore
  pyproject.toml           # slim: fastapi, uvicorn, requests, scikit-learn, fastembed
  api_server.py
  retrieval.py
  build_rag_index.py
  scope_examples.json
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
  `scikit-learn`, `fastembed` (ONNX runtime — deliberately not
  `sentence-transformers`, which would drag torch back in). Training deps (`torch`, `transformers`, `peft`, `trl`,
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
  files get shipped into the build context on every single build. It
  also excludes `data/datasets.jsonl` (sensitive, and unused at serve
  time — only `data/raw/` is needed to build the RAG index) and `.env`.

### Caller: the Go API (2026-09-30)
The only client is the Go Fiber API in `../backend-ayamnya-hanan` (`handlers/chat_handler.go`), which runs on a different VPS with the website frontend. It calls this service through the tunnel: `ASSISTANT_API_URL=<tunnel URL>`, header `X-API-Key: $ASSISTANT_API_KEY`, body `{"question"}` (≤ 500 chars, enforced there). Keep this contract stable; changing it needs a matching change in that handler and in root `../AGENTS.md` §2.3. The browser never calls this service directly, so `ALLOWED_ORIGINS` stays empty in production.

### API contract (as actually deployed)
`POST /chat` body is `{"question": "..."}` — **not** `{"message": ...}`.
Auth via `X-API-Key` header checked against `ASSISTANT_API_KEY`.

Response: `{"answer", "sources", "in_scope", "completion_tokens",
"prompt_tokens"}`:
- `in_scope: false`, `sources: []` → off-topic, `answer` is
  `FALLBACK_MESSAGE`.
- `in_scope: true`, `sources: []` → about the school but no data,
  `answer` is `NO_DATA_MESSAGE`.
- `in_scope: true`, `sources` non-empty → LLM answer; token counts set.

Worth logging both "no LLM" cases on the backend: they're the list of
real questions the scope gate wrongly rejects (→ `scope_examples.json`)
or the data doesn't cover (→ FAQ source).
`GET /health` is unauthenticated, returns `{"status": "ok",
"documents_loaded": <n>}` — fine to leave unauthenticated since it only
leaks a document count, but it's still reachable by anyone who can reach
the port, which is one more reason to keep that port firewalled/tunneled
rather than open to everyone.

### Compose file (tracked in git; run from `/www/my-penus-gweh`)
```yaml
services:
  assistant:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      HOST: 0.0.0.0
      ASSISTANT_API_KEY: ${ASSISTANT_API_KEY:?set ASSISTANT_API_KEY in .env}
    volumes:
      - ./models:/models:ro
    ports:
      - "127.0.0.1:8000:8000"   # bound to loopback only
    restart: unless-stopped
```
The real key lives in a `.env` file next to `docker-compose.yml` on the
VPS (`ASSISTANT_API_KEY=<openssl rand -hex 32>`), never in the compose
file itself — `.env` is gitignored and dockerignored. Compose refuses to
start if it's missing. Rotate the key if it's ever pasted anywhere (e.g.
a chat log).

Exposure is via a tunnel (e.g. Cloudflare Tunnel) pointed at
`http://localhost:8000` on the VPS, rather than opening port 8000 in any
firewall — the port binds to `127.0.0.1` specifically so nothing external
can reach it directly even if the tunnel is misconfigured.

### Rebuild vs. restart — don't confuse these
- **Rebuild required** (`docker compose up -d --build`) for changes to:
  `api_server.py`, `retrieval.py`, `start.sh`, `build_rag_index.py`,
  `scope_examples.json`, `pyproject.toml`, anything under `data/raw/`.
  All of these get `COPY`'d into the image at build time; a plain
  restart reuses the old image and silently ignores edited files on disk.
  The embedding model (~1 GB) is also downloaded during the build and
  baked into the image, so builds need internet and the image is ~1 GB
  larger than before semantic search.
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
Tutoring/refusal dataset, school profile/FAQ data source, concurrent benchmark,
post-quantization behavior check — none of this was addressed by getting
the container running; deployment only covers serving the current
(imperfect) model + RAG behavior.
