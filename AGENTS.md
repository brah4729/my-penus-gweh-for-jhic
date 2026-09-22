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
   `api_server.py` (FastAPI, does retrieval + calls llama-server) — see
   the docstring in `api_server.py` for the intended Laravel → FastAPI →
   llama-server VPS topology.

`chat_with_rag.py` is the CLI equivalent of `api_server.py` for local
testing without standing up the HTTP layer.

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
- **`api_server.py` is unauthenticated if `ASSISTANT_API_KEY` isn't set**
  (it just prints a warning and continues). Don't deploy without setting
  that env var. CORS is currently wide open (`allow_origins=["*"]`).
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
