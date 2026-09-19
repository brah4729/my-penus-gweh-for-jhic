"""
Builds SFT training data (data/datasets.jsonl) from the scraped school API data
in data/raw/*.json.

IMPORTANT SCOPE NOTE:
This only covers ONE of the two categories your tutoring assistant needs:

  1. Factual school Q&A (this script)      -- "what extracurriculars exist",
                                               "who teaches RPL", etc.
  2. Socratic tutoring behavior (NOT here)  -- "guide, don't solve" dialogues,
                                               off-topic redirects, refusal
                                               to hand over full code/answers.

Category 2 can't be derived from this scraped data at all -- it has to be
authored/synthesized separately (e.g. with a teacher model generating
tutoring dialogues) and appended to this same datasets.jsonl file before
training. Running only this script gives you a model that knows facts about
the school but has NOT learned the tutoring behavior yet.
"""

import json
import random
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/datasets.jsonl")

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI untuk SMK Plus Pelita Nusantara. "
    "Jawab pertanyaan seputar sekolah dengan ramah dan singkat."
)


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load(name: str):
    path = RAW_DIR / f"{name}.json"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_example(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def build_teacher_examples(teachers):
    examples = []
    for t in teachers:
        name = t.get("full_name", "").strip()
        subject = t.get("subject", "").strip()
        position = t.get("position", "").strip()
        desc = clean_html(t.get("description", ""))
        if not name or not subject:
            continue

        q = random.choice([
            f"Siapa yang mengajar {subject}?",
            f"Siapa guru untuk mata pelajaran {subject}?",
        ])
        a = f"{subject} diajar oleh {name}"
        if position:
            a += f" ({position})"
        a += "."
        if desc:
            a += f" {desc}"
        examples.append(make_example(q, a))

    # One aggregate listing, useful for "siapa saja guru di sekolah" style questions
    names = [t.get("full_name", "").strip() for t in teachers if t.get("full_name")]
    if names:
        listing = ", ".join(names[:25])
        if len(names) > 25:
            listing += f", dan {len(names) - 25} guru lainnya"
        examples.append(make_example(
            "Siapa saja guru-guru di SMK Plus Pelita Nusantara?",
            f"Beberapa guru di SMK Plus Pelita Nusantara antara lain: {listing}.",
        ))
    return examples


def build_eskul_examples(eskuls):
    examples = []
    for e in eskuls:
        name = e.get("name", "").strip()
        desc = clean_html(e.get("description", ""))
        pembina = e.get("Pembina", {}).get("full_name", "").strip()
        if not name:
            continue
        q = random.choice([
            f"Apa itu ekstrakurikuler {name}?",
            f"Ceritakan tentang eskul {name}.",
        ])
        a = desc or f"{name} adalah salah satu ekstrakurikuler di SMK Plus Pelita Nusantara."
        if pembina:
            a += f" Pembina ekstrakurikuler ini adalah {pembina}."
        examples.append(make_example(q, a))

    names = [e.get("name", "").strip() for e in eskuls if e.get("name")]
    if names:
        examples.append(make_example(
            "Ekstrakurikuler apa saja yang ada di sekolah?",
            "Ekstrakurikuler yang tersedia di SMK Plus Pelita Nusantara antara lain: "
            + ", ".join(names) + ".",
        ))
    return examples


def build_achievement_examples(achievements):
    examples = []
    for a in achievements:
        title = a.get("title", "").strip()
        desc = clean_html(a.get("description", ""))
        if not title:
            continue
        q = random.choice([
            f"Ceritakan tentang prestasi '{title}'.",
            f"Apa itu prestasi {title}?",
        ])
        answer = desc if desc else f"{title} adalah salah satu prestasi SMK Plus Pelita Nusantara."
        examples.append(make_example(q, answer))

    titles = [a.get("title", "").strip() for a in achievements if a.get("title")]
    if titles:
        examples.append(make_example(
            "Apa saja prestasi yang pernah diraih SMK Plus Pelita Nusantara?",
            "Beberapa prestasi SMK Plus Pelita Nusantara antara lain: " + "; ".join(titles) + ".",
        ))
    return examples


def build_news_examples(news_items):
    examples = []
    for n in news_items:
        title = n.get("title", "").strip()
        excerpt = clean_html(n.get("excerpt", ""))
        if not title or not excerpt:
            continue
        q = f"Apa berita terbaru tentang {title.lower()}?"
        examples.append(make_example(q, excerpt))
    return examples


def build_event_examples(events):
    examples = []
    for e in events:
        title = e.get("title", "").strip()
        desc = clean_html(e.get("description", ""))
        location = e.get("location", "").strip()
        start = e.get("start_date", "")
        end = e.get("end_date", "")
        if not title:
            continue
        q = f"Kapan dan di mana acara {title} diadakan?"
        a = desc or f"{title} adalah salah satu acara di SMK Plus Pelita Nusantara."
        if start:
            a += f" Acara ini berlangsung dari {start}" + (f" sampai {end}" if end and end != start else "") + "."
        if location:
            a += f" Lokasi: {location}."
        examples.append(make_example(q, a))
    return examples


def build_industry_examples(partners):
    examples = []
    # Only make individual examples for partners with real descriptions
    for p in partners:
        name = p.get("name", "").strip()
        desc = clean_html(p.get("description", ""))
        if not name or not desc:
            continue
        examples.append(make_example(
            f"Apa hubungan sekolah dengan {name}?",
            desc,
        ))

    # Aggregate the rest into one listing example (avoids ~90 near-duplicate examples)
    names = [p.get("name", "").strip() for p in partners if p.get("name")]
    if names:
        listing = ", ".join(names[:30])
        if len(names) > 30:
            listing += f", dan {len(names) - 30} mitra industri lainnya"
        examples.append(make_example(
            "Siapa saja mitra industri SMK Plus Pelita Nusantara?",
            f"SMK Plus Pelita Nusantara bermitra dengan berbagai perusahaan dan instansi, "
            f"antara lain: {listing}.",
        ))
    return examples


def build_testimonial_examples(testimonials):
    examples = []
    for t in testimonials:
        name = t.get("name", "").strip()
        position = t.get("position", "").strip()
        text = clean_html(t.get("testimonial", ""))
        if not text:
            continue
        q = "Apa kata orang tentang SMK Plus Pelita Nusantara?"
        a = f'"{text}" - {name}' + (f", {position}" if position else "")
        examples.append(make_example(q, a))
    return examples


def main():
    random.seed(42)
    print("Loading raw data...")
    teachers = load("teacher")
    eskuls = load("eskul")
    achievements = load("achievement")
    news = load("news")
    events = load("event")
    industry = load("industry")
    testimonials = load("testimonial")

    all_examples = []
    all_examples += build_teacher_examples(teachers)
    all_examples += build_eskul_examples(eskuls)
    all_examples += build_achievement_examples(achievements)
    all_examples += build_news_examples(news)
    all_examples += build_event_examples(events)
    all_examples += build_industry_examples(industry)
    all_examples += build_testimonial_examples(testimonials)

    random.shuffle(all_examples)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_examples)} factual QA examples -> {OUT_PATH}")
    print(
        "\nREMINDER: this file ONLY has factual school Q&A so far.\n"
        "The Socratic tutoring / 'guide don't solve' behavior examples still\n"
        "need to be written or synthesized separately and appended here before\n"
        "training -- otherwise the model will know school facts but won't have\n"
        "learned the tutoring behavior at all."
    )


if __name__ == "__main__":
    main()
