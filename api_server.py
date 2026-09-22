"""
Small HTTP API wrapping the RAG pipeline, meant to sit between Laravel and
llama-server -- hardened for running on a separate, internet-reachable VPS
instead of localhost-only testing.

  Laravel VPS  --(HTTPS + API key)-->  this FastAPI service  -->  llama-server
                                        (both on the model VPS,
                                         llama-server stays localhost-only)

Run:
    uv add fastapi uvicorn
    export ASSISTANT_API_KEY="pick-a-long-random-string"
    uv run uvicorn api_server:app --host 0.0.0.0 --port 8000
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
# llama-server stays on localhost -- only this process talks to it directly.
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"
API_KEY = os.environ.get("ASSISTANT_API_KEY")

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
    print("WARNING: ASSISTANT_API_KEY is not set -- the /chat endpoint is UNAUTHENTICATED.")
    print("Set it before exposing this service on a public VPS.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
