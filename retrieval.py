"""
Shared retrieval + scope gate, used by both api_server.py and chat_with_rag.py
(previously duplicated in each -- now there's one copy to tune).

Two separate decisions, made in code BEFORE the LLM is ever called:

1. SCOPE -- is this question about the school at all?
   The question's embedding is compared against labelled example questions
   in scope_examples.json ("school" vs "off_topic"). In scope if it's closer
   to the school examples. A strong TF-IDF hit also counts (exact teacher /
   partner names, which no example list can cover). Retrieval scores alone
   can NOT make this call: measured on this corpus, off-topic questions
   ("siapa presiden Indonesia") score higher against the documents than
   some real school questions ("tempat prakerin di mana") -- no threshold
   separates them.

2. FACTS -- which documents answer it?
   Hybrid ranking: multilingual embedding similarity (understands paraphrase
   and slang -- "pengajar MTK" finds the math teacher, which scored 0.00
   under TF-IDF) + a small TF-IDF weight (keeps exact-name matches sharp).

Outcomes:
  - off-topic                        -> FALLBACK_MESSAGE, no LLM call
  - in scope, no document >= floor   -> NO_DATA_MESSAGE, no LLM call
  - in scope, documents found        -> LLM answers from those documents
The no-LLM paths are deliberate: the model answered "Joko Widodo" to an
out-of-scope question despite being told not to guess (see AGENTS.md).

Tune with `uv run python eval_retrieval.py` after changing anything here,
in scope_examples.json, or the corpus.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = Path("data/rag_index.pkl")
# Embedding model weights are downloaded here by build_rag_index.py (baked
# into the Docker image at build time -- no download at container start).
EMBED_CACHE_DIR = ".cache/fastembed"
# Query embedding takes ~30ms on one thread; don't let it compete with
# llama-server for cores.
EMBED_THREADS = 1

TOP_K = 3
# Calibrated for paraphrase-multilingual-mpnet-base-v2 on the current corpus
# (see eval_retrieval.py). Re-run the eval if you switch EMBED_MODEL or the
# corpus grows -- these are not model-independent.
TFIDF_WEIGHT = 0.3           # hybrid = dense + TFIDF_WEIGHT * tfidf
DOC_FLOOR = 0.40             # hybrid score below this isn't a real match
SCOPE_K = 3                  # mean of top-k example similarities per side
SCOPE_MARGIN = 0.0           # in scope if school_sim - off_topic_sim > this
TFIDF_SCOPE_OVERRIDE = 0.40  # TF-IDF hit this strong = in scope regardless

SYSTEM_PROMPT_WITH_CONTEXT = (
    "Kamu adalah asisten AI untuk SMK Plus Pelita Nusantara. "
    "Jawab HANYA berdasarkan informasi di bawah ini. "
    "Jika informasi yang dibutuhkan tidak ada di bawah, katakan dengan jujur "
    "bahwa kamu tidak memiliki informasi tersebut -- jangan mengarang jawaban.\n\n"
    "INFORMASI:\n{context}"
)

# Off-topic question.
FALLBACK_MESSAGE = (
    "Maaf, aku tidak memiliki informasi tentang itu. "
    "Aku hanya bisa membantu dengan pertanyaan seputar SMK Plus Pelita Nusantara."
)

# About the school, but nothing in the data answers it (fees, address, ...).
NO_DATA_MESSAGE = (
    "Maaf, aku belum punya informasi tentang itu. "
    "Untuk info lebih lanjut, silakan hubungi pihak SMK Plus Pelita Nusantara secara langsung."
)


@dataclass
class Retrieval:
    in_scope: bool
    docs: list[dict]  # [{"score", "source", "text"}], best first

    @property
    def canned_answer(self) -> str | None:
        """Fixed reply when the LLM must NOT be called, else None."""
        if not self.in_scope:
            return FALLBACK_MESSAGE
        if not self.docs:
            return NO_DATA_MESSAGE
        return None


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Retriever:
    def __init__(self, index_path: Path = INDEX_PATH):
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        if "doc_vectors" not in index:
            raise SystemExit(
                f"{index_path} is an old TF-IDF-only index. Rebuild it: "
                "uv run python build_rag_index.py"
            )
        self.documents = index["documents"]
        self._vectorizer = index["vectorizer"]
        self._matrix = index["matrix"]
        self._doc_vectors = index["doc_vectors"]
        self._school_vectors = index["scope_school_vectors"]
        self._off_topic_vectors = index["scope_off_topic_vectors"]
        self.embed_model = index["embed_model"]
        self._embedder = TextEmbedding(
            self.embed_model, cache_dir=EMBED_CACHE_DIR, threads=EMBED_THREADS
        )

    def _embed_query(self, text: str) -> np.ndarray:
        return _normalize(next(iter(self._embedder.query_embed([text]))))

    def scope_margin(self, query_vec: np.ndarray) -> float:
        school = np.sort(self._school_vectors @ query_vec)[-SCOPE_K:].mean()
        off_topic = np.sort(self._off_topic_vectors @ query_vec)[-SCOPE_K:].mean()
        return float(school - off_topic)

    def retrieve(self, question: str) -> Retrieval:
        tfidf = cosine_similarity(self._vectorizer.transform([question]), self._matrix)[0]
        query_vec = self._embed_query(question)

        in_scope = (
            self.scope_margin(query_vec) > SCOPE_MARGIN
            or tfidf.max() >= TFIDF_SCOPE_OVERRIDE
        )
        if not in_scope:
            return Retrieval(in_scope=False, docs=[])

        scores = self._doc_vectors @ query_vec + TFIDF_WEIGHT * tfidf
        top = np.argsort(-scores)[:TOP_K]
        docs = [
            {
                "score": float(scores[i]),
                "source": self.documents[i]["source"],
                "text": self.documents[i]["text"],
            }
            for i in top
            if scores[i] >= DOC_FLOOR
        ]
        return Retrieval(in_scope=True, docs=docs)


def build_messages(question: str, docs: list[dict]) -> list[dict]:
    context = "\n".join(f"- {d['text']}" for d in docs)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_WITH_CONTEXT.format(context=context)},
        {"role": "user", "content": question},
    ]
