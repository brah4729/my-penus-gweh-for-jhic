"""
Regression check for retrieval.py -- run after changing thresholds, the
embedding model, scope_examples.json, or the scraped corpus:

    uv run python eval_retrieval.py

These questions are deliberately NOT in scope_examples.json (held out), so
the numbers reflect unseen phrasing. Add real questions from production
logs here -- especially ones that got the wrong answer.

Prints only verdicts, scores and source tags -- never document text or
names (data/raw is real personal data, see AGENTS.md).
"""

import re

from retrieval import Retriever

# (question, expected source category or None if the school has no data on it)
IN_SCOPE = [
    ("siapa pengajar MTK", "teacher"),
    ("Siapa yang mengajar matematika?", "teacher"),
    ("guru bahasa inggris siapa ya", "teacher"),
    ("guru olahraga siapa", "teacher"),
    ("siapa kepala sekolahnya", "teacher"),
    ("siapa saja guru di sekolah ini", "teacher_aggregate"),
    ("ada ekskul apa aja?", "eskul"),
    ("kegiatan setelah pulang sekolah apa saja", "eskul"),
    ("ada klub olahraga ga?", "eskul"),
    ("daftar ekstrakurikuler", "eskul"),
    ("sekolah pernah juara apa", "achievement"),
    ("penghargaan yang pernah didapat", "achievement"),
    ("apa saja prestasi sekolah", "achievement"),
    ("perusahaan yang kerja sama dengan sekolah", "industry"),
    ("tempat prakerin di mana saja", "industry"),
    ("mitra industri sekolah apa saja", "industry"),
    ("lulusan sini kerja di mana aja", None),
    ("berita terbaru", "news"),
    ("kapan ada acara sekolah", "event"),
    ("apa kata alumni tentang sekolah ini", "testimonial"),
    ("sekolahnya di daerah mana", None),
    ("kalau mau masuk sini gimana caranya", None),
    ("ada beasiswa ga", None),
    ("ada lab komputer?", None),
    ("berapa biaya SPP", None),
    ("ada kantin ga", None),
]

OFF_TOPIC = [
    "Siapa presiden Indonesia?", "resep nasi goreng", "berapa 2+2", "cuaca hari ini gimana",
    "siapa pemenang piala dunia 2022", "cara membuat website dengan laravel", "apa ibukota jepang",
    "rekomendasi film horor", "harga bitcoin sekarang", "tolong buatkan puisi cinta",
    "siapa menteri pendidikan", "jelaskan teori relativitas", "cara bikin akun instagram",
    "lirik lagu indonesia raya", "apa itu blockchain", "siapa pemain bola terbaik dunia",
    "cara masak mie instan", "kenapa langit biru", "berapa jarak bumi ke bulan",
    "gimana cara pacaran yang sehat",
]


def name_questions(retriever: Retriever) -> list[tuple[str, str]]:
    """Exact-name questions built from the index itself (never printed)."""
    out = []
    for doc in retriever.documents:
        if doc["source"] == "teacher" and len(out) < 3:
            out.append(("siapa " + doc["text"].split(" adalah ")[0] + "?", "teacher"))
    for doc in retriever.documents:
        m = re.match(r"Mitra industri: (.*?)\.", doc["text"])
        if m and len(out) < 6:
            out.append((f"apa itu {m.group(1)}?", "industry"))
    return out


def category(source: str) -> str:
    return source.removesuffix("_aggregate")


def main():
    r = Retriever()
    names = name_questions(r)
    cases = IN_SCOPE + names
    hidden = {q for q, _ in names}

    scope_ok = rank1_ok = top3_ok = ranked = 0
    print("IN SCOPE (expect in_scope=True)")
    for q, expected in cases:
        res = r.retrieve(q)
        scope_ok += res.in_scope
        label = "<name question>" if q in hidden else q
        srcs = [d["source"] for d in res.docs]
        verdict = "ok " if res.in_scope else "MISS"
        if expected is not None:
            ranked += 1
            got = [category(s) for s in srcs]
            rank1_ok += bool(got) and got[0] == category(expected)
            top3_ok += category(expected) in got
            if expected.endswith("_aggregate") and (not srcs or srcs[0] != expected):
                verdict += " (aggregate not first)"
        top = f"{res.docs[0]['score']:.2f}" if res.docs else " -- "
        print(f"  {verdict:4} top={top} {srcs}  {label}")

    leaked = 0
    print("\nOFF TOPIC (expect in_scope=False)")
    for q in OFF_TOPIC:
        res = r.retrieve(q)
        leaked += res.in_scope
        print(f"  {'LEAK' if res.in_scope else 'ok  '} {q}")

    print(f"\nscope: {scope_ok}/{len(cases)} school questions accepted, "
          f"{leaked}/{len(OFF_TOPIC)} off-topic leaked")
    print(f"ranking (questions with data): right category first {rank1_ok}/{ranked}, "
          f"in top 3 {top3_ok}/{ranked}")


if __name__ == "__main__":
    main()
