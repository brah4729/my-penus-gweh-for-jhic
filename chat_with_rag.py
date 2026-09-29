"""
Queries the RAG index and calls the model through llama-server's
OpenAI-compatible API, injecting only genuinely relevant facts as context.

CLI equivalent of api_server.py -- both use retrieval.py, so behavior is
identical. This is what actually prevents hallucination AND keeps the
assistant on topic: if the question is off-topic (scope gate) or nothing
relevant is retrieved, we don't even call the model -- we return a fixed
message directly. This is more robust than trusting the model to
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

import sys

import requests

from retrieval import Retriever, build_messages

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"


def ask_model(question: str, docs: list[dict]):
    payload = {
        "messages": build_messages(question, docs),
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
    result = Retriever().retrieve(question)

    print(f"\n[in_scope={result.in_scope}, retrieved {len(result.docs)} relevant fact(s)]")
    for doc in result.docs:
        print(f"  ({doc['score']:.3f}) [{doc['source']}] {doc['text'][:80]}...")

    answer = result.canned_answer or ask_model(question, result.docs)
    print(f"\nJawaban: {answer}")


if __name__ == "__main__":
    main()
