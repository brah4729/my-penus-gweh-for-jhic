"""
Small HTTP API wrapping the RAG pipeline, meant to sit between the Go
(Fiber) backend and llama-server -- hardened for running on a separate,
internet-reachable VPS instead of localhost-only testing.

Works either topology -- decide at deploy time via the HOST env var, code
doesn't need to change:

  SAME VPS (Fiber and this service on one box) -- the common case, simplest
  and fastest, no TLS needed for this hop:

    Fiber (127.0.0.1)  --(HTTP + API key)-->  this FastAPI service  -->  llama-server
    Leave HOST unset (defaults to 127.0.0.1 / loopback-only). Nothing on
    this hop is reachable from outside the box at all.

  SEPARATE VPSes -- this hop crosses the public internet, so it must be
  HTTPS, not plain HTTP:

    Fiber VPS  --(HTTPS + API key)-->  this FastAPI service (model VPS)  -->  llama-server
                                        (llama-server itself still stays
                                         localhost-only on the model VPS)
    Set HOST=0.0.0.0 and put this behind a TLS-terminating reverse proxy
    (nginx/caddy) rather than hand-rolling uvicorn's --ssl-keyfile/
    --ssl-certfile -- easier to renew certs and it's the standard RHEL
    pattern. ASSISTANT_API_KEY becomes mandatory in this mode (see below
    -- the app refuses to start without it once HOST isn't loopback).

This is always a server-to-server call (Fiber -> here), never a browser
hitting this directly, so CORS is not the access-control mechanism in
either topology -- the X-API-Key check is. CORS stays closed by default.

Run:
    uv add fastapi uvicorn
    export ASSISTANT_API_KEY="pick-a-long-random-string"
    export HOST=127.0.0.1   # or 0.0.0.0 if Fiber is on a different VPS
    uv run uvicorn api_server:app --host "$HOST" --port 8000

RHEL VPS notes:
    - Only open the port in firewalld if HOST=0.0.0.0 (separate-VPS case):
        sudo firewall-cmd --permanent --add-port=8000/tcp && sudo firewall-cmd --reload
      If Fiber is on the same box (loopback), leave the port closed to the
      outside entirely -- don't punch a firewalld hole you don't need.
    - Run this under systemd (not a bare foreground process) so it restarts
      on crash/reboot; same for llama-server.
    - SELinux: if you hit AVC denials binding/connecting on these ports,
      check `sudo ausearch -m avc -ts recent` rather than disabling SELinux.
"""

import os
import pickle
from pathlib import Path

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = Path("data/rag_index.pkl")
# llama-server stays on localhost -- only this process talks to it directly,
# in BOTH topologies (even when this service itself is bound to 0.0.0.0 for
# a separate-VPS Fiber backend, llama-server is never exposed).
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"
API_KEY = os.environ.get("ASSISTANT_API_KEY")
HOST = os.environ.get("HOST", "127.0.0.1")

TOP_K = 3
RELEVANCE_THRESHOLD = 0.12

SYSTEM_PROMPT_WITH_CONTEXT = (
    "Kamu adalah asisten AI untuk SMK Plus Pelita Nusantara. "
    "Jawab HANYA berdasarkan informasi di bawah ini. "
    "Jika informasi yang dibutuhkan tidak ada di bawah, katakan dengan jujur "
    "bahwa kamu tidak memiliki informasi tersebut -- jangan mengarang jawaban.\n\n"
    "INFORMASI:\n{context}"
)

FALLBACK_MESSAGE = (
    "Maaf, aku tidak memiliki informasi tentang itu. "
    "Aku hanya bisa membantu dengan pertanyaan seputar SMK Plus Pelita Nusantara."
)

if not API_KEY:
    if HOST not in ("127.0.0.1", "localhost", "::1"):
        # HOST says this is (or might be) reachable off-box -- e.g. the
        # separate-VPS topology -- so an unauthenticated /chat endpoint
        # would be a real, public exposure. Refuse to start rather than
        # just warn; same script, same check, works for either topology.
        raise SystemExit(
            "ASSISTANT_API_KEY is not set, but HOST is not loopback "
            f"({HOST!r}) -- refusing to start with an unauthenticated "
            "/chat endpoint reachable off-box. Set ASSISTANT_API_KEY, or "
            "set HOST=127.0.0.1 if this really is same-VPS as the Fiber backend."
        )
    print("WARNING: ASSISTANT_API_KEY is not set -- the /chat endpoint is UNAUTHENTICATED.")
    print("This is only safe because HOST is loopback-only. Set ASSISTANT_API_KEY "
          "before ever setting HOST=0.0.0.0.")

app = FastAPI()

# Server-to-server (Fiber backend -> here): no browser origin to allow, so
# CORS stays closed by default. ALLOWED_ORIGINS lets you open it up later
# (e.g. a web dashboard) without editing code -- comma-separated env var.
_allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_methods=["POST", "GET"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

print("Loading RAG index...")
with open(INDEX_PATH, "rb") as f:
    _index = pickle.load(f)
print(f"Loaded {len(_index['documents'])} documents.")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    completion_tokens: int | None = None
    prompt_tokens: int | None = None


def check_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def retrieve(question: str):
    vectorizer = _index["vectorizer"]
    matrix = _index["matrix"]
    documents = _index["documents"]

    query_vec = vectorizer.transform([question])
    scores = cosine_similarity(query_vec, matrix)[0]

    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [
        {"score": float(score), "source": doc["source"], "text": doc["text"]}
        for score, doc in ranked[:TOP_K]
        if score >= RELEVANCE_THRESHOLD
    ]


def ask_model(question: str, retrieved: list):
    if not retrieved:
        return FALLBACK_MESSAGE, None, None

    context = "\n".join(f"- {r['text']}" for r in retrieved)
    system_prompt = SYSTEM_PROMPT_WITH_CONTEXT.format(context=context)

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(LLAMA_SERVER_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    usage = data.get("usage", {})
    return (
        message.get("content") or FALLBACK_MESSAGE,
        usage.get("completion_tokens"),
        usage.get("prompt_tokens"),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)
    retrieved = retrieve(req.question)
    answer, completion_tokens, prompt_tokens = ask_model(req.question, retrieved)
    return ChatResponse(
        answer=answer,
        sources=retrieved,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
    )


@app.get("/health")
def health():
    return {"status": "ok", "documents_loaded": len(_index["documents"])}
