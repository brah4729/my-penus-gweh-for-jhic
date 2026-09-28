"""
Prunes the tokenizer vocab + tied embedding of school-assistant-merged down to
only the tokens actually used in data/datasets.jsonl, plus every 1-character
base token (safe merge leaves), PLUS every "added token" (EOS/PAD/chat
special tokens), which live in a separate part of tokenizer.json from the
main BPE vocab and must never be dropped or remapped incorrectly.

This does NOT touch the RAG index (data/rag_index.pkl) -- that's TF-IDF over
plain text and has nothing to do with the LLM's tokenizer.

Usage:
    uv run python prune_vocab.py
"""

import json
import pickle
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MERGED_DIR = "school-assistant-merged"
OUT_DIR = "school-assistant-pruned"
DATASET_PATH = "data/datasets.jsonl"
RAG_INDEX_PATH = "data/rag_index.pkl"

# These are hardcoded in chat_with_rag.py / api_server.py -- NOT part of
# datasets.jsonl at all. Missing these was the actual bug that broke the
# first pruning attempt: the model's own instruction prompt got shredded
# into byte-fallback fragments it had never seen, so it stopped recognizing
# it as an instruction and just echoed the question back instead.
RUNTIME_PROMPT_STRINGS = [
    "Kamu adalah asisten AI untuk SMK Plus Pelita Nusantara. "
    "Jawab HANYA berdasarkan informasi di bawah ini. "
    "Jika informasi yang dibutuhkan tidak ada di bawah, katakan dengan jujur "
    "bahwa kamu tidak memiliki informasi tersebut -- jangan mengarang jawaban.\n\n"
    "INFORMASI:\n",
    "Maaf, aku tidak memiliki informasi tentang itu. "
    "Aku hanya bisa membantu dengan pertanyaan seputar SMK Plus Pelita Nusantara.",
]


def walk_strings(obj):
    """Yield every string value found anywhere in a nested JSON object."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v)



def collect_used_ids(tokenizer, dataset_path):
    used = set()
    n_lines = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            obj = json.loads(line)
            for text in walk_strings(obj):
                if text:
                    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                    used.update(ids)
    print(f"Scanned {n_lines} dataset lines")
    return used


def collect_rag_index_ids(tokenizer, rag_index_path):
    """The {context} injected into the runtime prompt comes from here, not
    from datasets.jsonl -- a separate corpus (scraped school data) that also
    needs to be tokenizable, or retrieved facts get shredded at inference
    time even though the dataset-derived vocab looks fine."""
    used = set()
    path = Path(rag_index_path)
    if not path.exists():
        print(f"WARNING: {rag_index_path} not found, skipping RAG index vocab scan")
        return used
    with open(path, "rb") as f:
        index = pickle.load(f)
    n_docs = 0
    for doc in index.get("documents", []):
        n_docs += 1
        for text in walk_strings(doc):
            if text:
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                used.update(ids)
    print(f"Scanned {n_docs} RAG index documents")
    return used


def collect_runtime_prompt_ids(tokenizer, strings):
    used = set()
    for text in strings:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        used.update(ids)
    print(f"Runtime prompt/fallback strings contributed {len(used)} tokens")
    return used


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def remap_field(old_to_new, x):
    if x is None:
        return None
    if isinstance(x, list):
        return [old_to_new[i] for i in x if i in old_to_new]
    if x not in old_to_new:
        raise ValueError(f"token id {x} was dropped during pruning but is still "
                          f"referenced by config -- this should never happen")
    return old_to_new[x]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
    model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, torch_dtype=torch.float32)

    used_ids = collect_used_ids(tokenizer, DATASET_PATH)
    used_ids |= collect_rag_index_ids(tokenizer, RAG_INDEX_PATH)
    used_ids |= collect_runtime_prompt_ids(tokenizer, RUNTIME_PROMPT_STRINGS)
    used_ids |= set(tokenizer.all_special_ids)
    for src in (model.config, model.generation_config):
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            used_ids |= set(as_list(getattr(src, attr, None)))
    print(f"Base-vocab tokens touched by dataset + special tokens: {len(used_ids)}")

    tok_path = Path(MERGED_DIR) / "tokenizer.json"
    tok_data = json.loads(tok_path.read_text(encoding="utf-8"))
    vocab = tok_data["model"]["vocab"]              # base BPE subword vocab
    merges = tok_data["model"]["merges"]
    added_tokens = tok_data.get("added_tokens", []) # separate list: EOS/PAD/chat tokens
    id_to_tok = {idx: tok for tok, idx in vocab.items()}

    # Keep every 1-character base token as a safe fallback for unseen words.
    for tok, idx in vocab.items():
        if len(tok) <= 1:
            used_ids.add(idx)

    # --- Part 1: base vocab kept ids (only ids that actually live in `vocab`) ---
    base_keep_ids = sorted(i for i in used_ids if i in id_to_tok)

    # --- Part 2: added tokens (EOS/PAD/chat tokens) -- ALWAYS keep every one of
    # these, since they live outside `vocab` and dropping any would silently
    # break generation, chat templating, or padding.
    added_old_ids = sorted({t["id"] for t in added_tokens})
    added_old_ids = [i for i in added_old_ids if i not in set(base_keep_ids)]

    # Combined id space: base tokens first, then added tokens, both in a new
    # contiguous 0..N-1 range. Order matters -- it defines the new embedding
    # row order below.
    combined_old_ids = base_keep_ids + added_old_ids
    old_to_new = {old: new for new, old in enumerate(combined_old_ids)}
    keep_tok_set = {id_to_tok[i] for i in base_keep_ids}

    new_vocab = {tok: old_to_new[idx] for tok, idx in vocab.items() if idx in old_to_new}

    def pair_of(m):
        return tuple(m.split(" ")) if isinstance(m, str) else tuple(m)

    new_merges = [m for m in merges
                  if (a := pair_of(m)[0]) in keep_tok_set
                  and (b := pair_of(m)[1]) in keep_tok_set
                  and (a + b) in keep_tok_set]

    new_added_tokens = []
    for t in added_tokens:
        t = dict(t)
        t["id"] = old_to_new[t["id"]]
        new_added_tokens.append(t)

    tok_data["model"]["vocab"] = new_vocab
    tok_data["model"]["merges"] = new_merges
    tok_data["added_tokens"] = new_added_tokens

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True)

    # Only tokenizer.json contains numeric ids, so it's the only tokenizer
    # file that needs rebuilding. tokenizer_config.json / chat_template.jinja
    # only reference token strings (unchanged), so copy them byte-for-byte
    # instead of trying to reconstruct them -- reconstruction via
    # PreTrainedTokenizerFast repeatedly dropped/reshaped fields the real
    # Qwen2 tokenizer class expects (tokenizer_class, model_specific_special_tokens).
    (out_dir / "tokenizer.json").write_text(json.dumps(tok_data, ensure_ascii=False), encoding="utf-8")
    for fname in ("tokenizer_config.json", "chat_template.jinja"):
        src = Path(MERGED_DIR) / fname
        if src.exists():
            (out_dir / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Slice the tied embedding to only the kept rows (base + added), in the
    # exact same order used to build old_to_new above.
    emb = model.get_input_embeddings().weight.data
    new_emb = emb[combined_old_ids].clone()
    model.get_input_embeddings().weight = torch.nn.Parameter(new_emb)
    model.config.vocab_size = len(combined_old_ids)
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()

    model.config.bos_token_id = remap_field(old_to_new, model.config.bos_token_id)
    model.config.eos_token_id = remap_field(old_to_new, model.config.eos_token_id)
    model.config.pad_token_id = remap_field(old_to_new, model.config.pad_token_id)
    gen_cfg = model.generation_config
    gen_cfg.bos_token_id = remap_field(old_to_new, gen_cfg.bos_token_id)
    gen_cfg.eos_token_id = remap_field(old_to_new, gen_cfg.eos_token_id)
    gen_cfg.pad_token_id = remap_field(old_to_new, gen_cfg.pad_token_id)

    model.save_pretrained(out_dir)

    print(f"Base vocab: {len(vocab)} -> {len(base_keep_ids)} tokens")
    print(f"Added/special tokens preserved: {len(added_old_ids)}")
    print(f"Total vocab_size: {len(vocab)} -> {len(combined_old_ids)} "
          f"({len(combined_old_ids) / len(vocab):.2%} kept)")
    print(f"Saved pruned model + tokenizer to {out_dir}/")


if __name__ == "__main__":
    main()
