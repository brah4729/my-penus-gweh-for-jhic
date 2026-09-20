"""
Queries the RAG index and calls the model through llama-server's
OpenAI-compatible API, injecting only genuinely relevant facts as context.

This is what actually prevents hallucination AND keeps the assistant on
topic: if nothing in the retrieval index scores above RELEVANCE_THRESHOLD
for the question, we don't even call the model -- we return a fixed
fallback message directly. This is more robust than trusting the model to
follow a "say you don't know" instruction (it doesn't reliably -- tested:
asked "who is the president of Indonesia" with zero relevant facts
retrieved, and the model answered "Joko Widodo" anyway despite being told
not to guess). Deciding "in scope or not" in code, based on whether real
matching data exists, sidesteps that unreliability entirely.

Prerequisite: llama-server must be running, e.g.:
    llama-server.exe -m school-assistant-q4_k_m-fixed.gguf --parallel 2 --ctx-size 4096

Usage:
    uv run python chat_with_rag.py "Siapa yang mengajar matematika?"
"""

import pickle
import sys
from pathlib import Path

import requests
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = Path("data/rag_index.pkl")
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

TOP_K = 3
RELEVANCE_THRESHOLD = 0.12  # tune this -- see note at bottom of file

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


def load_index():
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def retrieve(question: str, index: dict):
    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    documents = index["documents"]

    query_vec = vectorizer.transform([question])
    scores = cosine_similarity(query_vec, matrix)[0]

    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    top = [(score, doc) for score, doc in ranked[:TOP_K] if score >= RELEVANCE_THRESHOLD]
    return top


def ask_model(question: str, retrieved: list):
    # Don't even ask the model when nothing relevant was found -- small models
    # are unreliable at obeying a "say you don't know" instruction (we just
    # proved this: it answered "Joko Widodo" anyway despite being told not
    # to). Deciding this in code, before the LLM call, is more robust than
    # any prompt could be, and it doubles as the off-topic/scope guard.
    if not retrieved:
        return FALLBACK_MESSAGE

    context = "\n".join(f"- {doc['text']}" for _, doc in retrieved)
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

    if not message.get("content") and message.get("reasoning_content"):
        print("\n[WARNING] Model got stuck reasoning and never produced a final answer.")
        print(f"[reasoning_content, truncated]: {message['reasoning_content'][:300]}...")

    return message.get("content", "")


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python chat_with_rag.py "your question"')
        sys.exit(1)

    question = sys.argv[1]
    index = load_index()
    retrieved = retrieve(question, index)

    print(f"\n[retrieved {len(retrieved)} relevant fact(s)]")
    for score, doc in retrieved:
        print(f"  ({score:.3f}) [{doc['source']}] {doc['text'][:80]}...")

    answer = ask_model(question, retrieved)
    print(f"\nJawaban: {answer}")


if __name__ == "__main__":
    main()

# NOTE on RELEVANCE_THRESHOLD:
# Real matches for this corpus (after adding stopwords) land around 0.35-0.43;
# genuinely irrelevant questions now correctly retrieve 0 results at this
# threshold. Re-check this if you significantly grow the dataset, since more
# documents can shift the score distribution.
