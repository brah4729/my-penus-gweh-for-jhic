"""
Builds a lightweight TF-IDF retrieval index from the scraped school data.

This is the "knowledge" half of the assistant: instead of relying on the
fine-tuned model to have memorized facts (which it can't do reliably at
0.8B params -- it confabulates when it doesn't actually know), we retrieve
the relevant fact(s) at query time and hand them to the model as context.
The model then only needs to phrase an answer from what it's given, not
recall it from its own weights.

Includes both:
  - individual per-item documents (one teacher, one eskul, etc.) for
    specific questions ("who teaches math")
  - aggregate "list all X" documents for broad questions ("who are all
    the teachers", "what extracurriculars exist") -- broad questions don't
    match any single individual document well, so they need their own
    dedicated summary document to retrieve against.

Each document is indexed twice: TF-IDF over its text, and a multilingual
embedding (see retrieval.py for how they're combined). Aggregate documents
are embedded by a short "key" description instead of their text -- a long
list of names washes out the embedding, so "siapa saja guru" used to lose
to individual teachers; embedding "Daftar semua guru..." fixes that. The
text shown to the model is still the full list.

The scope-gate examples (scope_examples.json) are embedded here too, so
edit that file -> rebuild the index.

Run:
    uv run python build_rag_index.py
"""

import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval import EMBED_CACHE_DIR, INDEX_PATH

RAW_DIR = Path("data/raw")
SCOPE_EXAMPLES_PATH = Path("scope_examples.json")
# ~1 GB of RAM at runtime. paraphrase-multilingual-MiniLM-L12-v2 is ~0.5 GB
# but wrongly rejected common questions ("ada ekskul apa aja?") in testing;
# if you switch, re-tune the thresholds in retrieval.py via eval_retrieval.py.
EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Generic words that appear in nearly every document (mainly the fixed
# school name) and were causing false-positive matches -- e.g. "Siapa
# presiden Indonesia?" was matching random industry partners just because
# their names contain "Indonesia". NOTE: "guru"/"smk" are deliberately NOT
# in this list -- they're real signal words for teacher/school-scope
# queries (an earlier version wrongly stripped them, which broke broad
# questions like "siapa saja guru di sekolah ini").
STOPWORDS = [
    "plus", "pelita", "nusantara", "indonesia", "sekolah",
    "adalah", "yang", "di", "dan", "dengan", "untuk", "ini", "itu",
    "dari", "ke", "pada", "sebagai", "salah", "satu", "atau", "juga",
    "akan", "telah", "para", "oleh", "dalam",
]


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load(name: str):
    path = RAW_DIR / f"{name}.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_documents():
    """Each document is one retrievable fact: (text_shown_to_model, source_tag)."""
    docs = []

    teachers = load("teacher")
    for t in teachers:
        name = t.get("full_name", "").strip()
        subject = t.get("subject", "").strip()
        position = t.get("position", "").strip()
        desc = clean_html(t.get("description", ""))
        if not name:
            continue
        text = f"{name} adalah guru {subject} di SMK Plus Pelita Nusantara"
        if position:
            text += f", menjabat sebagai {position}"
        text += "."
        if desc:
            text += f" {desc}"
        docs.append({"text": text, "source": "teacher"})

    # Aggregate: broad "who are all the teachers" questions need this,
    # since no individual teacher document matches a generic query well.
    names = [t.get("full_name", "").strip() for t in teachers if t.get("full_name")]
    if names:
        listing = ", ".join(names)
        docs.append({
            "text": f"Daftar guru-guru di SMK Plus Pelita Nusantara: {listing}.",
            "source": "teacher_aggregate",
            "key": "Daftar semua guru dan pengajar di sekolah.",
        })

    eskuls = load("eskul")
    for e in eskuls:
        name = e.get("name", "").strip()
        desc = clean_html(e.get("description", ""))
        pembina = e.get("Pembina", {}).get("full_name", "").strip()
        if not name:
            continue
        text = f"Ekstrakurikuler {name} adalah salah satu kegiatan di SMK Plus Pelita Nusantara."
        if desc:
            text += f" {desc}"
        if pembina:
            text += f" Pembina: {pembina}."
        docs.append({"text": text, "source": "eskul"})

    eskul_names = [e.get("name", "").strip() for e in eskuls if e.get("name")]
    if eskul_names:
        docs.append({
            "text": f"Daftar ekstrakurikuler di SMK Plus Pelita Nusantara: {', '.join(eskul_names)}.",
            "source": "eskul_aggregate",
            "key": "Daftar semua ekstrakurikuler (ekskul) dan kegiatan siswa di sekolah.",
        })

    achievements = load("achievement")
    for a in achievements:
        title = a.get("title", "").strip()
        desc = clean_html(a.get("description", ""))
        if not title:
            continue
        text = f"Prestasi: {title}."
        if desc:
            text += f" {desc}"
        docs.append({"text": text, "source": "achievement"})

    achievement_titles = [a.get("title", "").strip() for a in achievements if a.get("title")]
    if achievement_titles:
        docs.append({
            "text": f"Daftar prestasi SMK Plus Pelita Nusantara: {'; '.join(achievement_titles)}.",
            "source": "achievement_aggregate",
            "key": "Daftar semua prestasi dan juara lomba yang pernah diraih sekolah.",
        })

    for n in load("news"):
        title = n.get("title", "").strip()
        excerpt = clean_html(n.get("excerpt", ""))
        if not title or not excerpt:
            continue
        docs.append({"text": f"Berita: {title}. {excerpt}", "source": "news"})

    for e in load("event"):
        title = e.get("title", "").strip()
        desc = clean_html(e.get("description", ""))
        location = e.get("location", "").strip()
        start = e.get("start_date", "")
        end = e.get("end_date", "")
        if not title:
            continue
        text = f"Acara: {title}."
        if desc:
            text += f" {desc}"
        if start:
            text += f" Tanggal: {start}" + (f" - {end}" if end and end != start else "") + "."
        if location:
            text += f" Lokasi: {location}."
        docs.append({"text": text, "source": "event"})

    industry = load("industry")
    for p in industry:
        name = p.get("name", "").strip()
        desc = clean_html(p.get("description", ""))
        if not name:
            continue
        text = f"Mitra industri: {name}."
        if desc:
            text += f" {desc}"
        docs.append({"text": text, "source": "industry"})

    industry_names = [p.get("name", "").strip() for p in industry if p.get("name")]
    if industry_names:
        docs.append({
            "text": f"Daftar mitra industri SMK Plus Pelita Nusantara: {', '.join(industry_names)}.",
            "source": "industry_aggregate",
            "key": "Daftar semua mitra industri dan perusahaan tempat PKL/prakerin siswa.",
        })

    for t in load("testimonial"):
        name = t.get("name", "").strip()
        position = t.get("position", "").strip()
        text_content = clean_html(t.get("testimonial", ""))
        if not text_content:
            continue
        text = f'Testimoni dari {name}' + (f" ({position})" if position else "") + f': "{text_content}"'
        docs.append({"text": text, "source": "testimonial"})

    return docs


def main():
    documents = build_documents()
    print(f"Built {len(documents)} retrievable documents.")

    corpus = [d["text"] for d in documents]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words=STOPWORDS)
    matrix = vectorizer.fit_transform(corpus)

    print(f"Embedding with {EMBED_MODEL} (downloads on first run)...")
    embedder = TextEmbedding(EMBED_MODEL, cache_dir=EMBED_CACHE_DIR)

    def embed(texts):
        vecs = np.array(list(embedder.embed(texts)))
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    with open(SCOPE_EXAMPLES_PATH, encoding="utf-8") as f:
        scope = json.load(f)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "matrix": matrix,
            "documents": documents,
            "embed_model": EMBED_MODEL,
            "doc_vectors": embed([d.get("key", d["text"]) for d in documents]),
            "scope_school_vectors": embed(scope["school"]),
            "scope_off_topic_vectors": embed(scope["off_topic"]),
        }, f)

    print(f"Saved index -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
