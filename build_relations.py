"""
Build relations between images (data.json) and words (unique_words.json).

- Reads two JSON files:
    * data.json: list of images, each with "filename" and "output" text
    * unique_words.json: list of unique content words, each with "word"

- For each image:
    * Extract content words (spaCy tokenization + lemmatization + stopword removal)
    * Keep only POS in {NOUN, VERB, ADJ, ADV}
    * Drop non-alphabetic, short tokens (len < 3), ordinal suffixes, and a small noise list
    * (Optional) drop super-rare junk/typos using wordfreq Zipf threshold

- Relations are added both ways:
    * image["text_ids"] -> list of word indices present in its text
    * word["image_ids"] -> list of image indices where the word appears

- Indices are 0-based, defined by position in their respective lists

Usage:
  # Windows
  python build_relations.py ^
    --data-in "data\\data_indexed.json" ^
    --words-in "data\\unique_words.json" ^
    --data-out "data\\data_with_relations.json" ^
    --words-out "data\\unique_words_with_relations.json" ^
    --min-len 3

  # macOS / Linux
  python build_relations.py \
    --data-in "data/data_indexed.json" \
    --words-in "data/unique_words.json" \
    --data-out "data/data_with_relations.json" \
    --words-out "data/unique_words_with_relations.json" \
    --min-len 3
"""


import json
import argparse
import sys
from typing import List, Dict, Set

import spacy

ALLOWED_POS = {"NOUN", "VERB", "ADJ", "ADV"}
ORDINAL_SUFFIXES = {"st", "nd", "rd", "th"}
NOISE = {
    "cv","cta","ch","de","dom","ess","lb","mph","ost","midw","st","nd","rd","th","url",
}

try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None


def parse_args():
    p = argparse.ArgumentParser(description="Build relations between images (data.json) and words (unique_words.json).")
    p.add_argument("--data-in", required=True, help="Path to input data.json (images list).")
    p.add_argument("--words-in", required=True, help="Path to input unique_words.json (words list).")
    p.add_argument("--data-out", required=True, help="Path to write updated data.json.")
    p.add_argument("--words-out", required=True, help="Path to write updated unique_words.json.")
    p.add_argument("--min-len", type=int, default=3, help="Minimum token length to keep.")
    p.add_argument("--zipf-min", type=float, default=None,
                   help="If set and wordfreq is installed, drop tokens below this Zipf freq (e.g., 3.0). Use 0 or omit to disable.")
    return p.parse_args()


def is_noise_token(t: str, min_len: int, zipf_min):
    if len(t) < min_len:
        return True
    if t in NOISE:
        return True
    if t in ORDINAL_SUFFIXES:
        return True
    if zipf_min is not None and zipf_frequency is not None:
        if zipf_frequency(t, "en") < float(zipf_min):
            return True
    return False


def extract_lemmas(nlp, text: str, min_len: int, zipf_min) -> Set[str]:
    if not text:
        return set()
    doc = nlp(text)
    out = set()
    for tok in doc:
        if not tok.is_alpha:
            continue
        if tok.is_stop:
            continue
        if tok.pos_ not in ALLOWED_POS:
            continue
        lemma = tok.lemma_.lower()
        if is_noise_token(lemma, min_len, zipf_min):
            continue
        out.add(lemma)
    return out


def main():
    args = parse_args()
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        print("⚠️ Missing spaCy model. Run:  python -m spacy download en_core_web_sm", file=sys.stderr)
        sys.exit(1)

    # Load inputs
    with open(args.data_in, "r", encoding="utf-8") as f:
        images: List[Dict] = json.load(f)

    with open(args.words_in, "r", encoding="utf-8") as f:
        words: List[Dict] = json.load(f)

    lemma_to_index: Dict[str, int] = {}
    for idx, w in enumerate(words):
        lemma = (w.get("word") or "").strip().lower()
        if lemma:
            lemma_to_index[lemma] = idx

    for img in images:
        img["text_ids"] = []

    for w in words:
        w["image_ids"] = []

    for img_idx, img in enumerate(images):
        text = img.get("output", "") or ""
        lemmas_in_img = extract_lemmas(nlp, text, args.min_len, args.zipf_min)

        matched_word_indices = []
        for lemma in lemmas_in_img:
            w_idx = lemma_to_index.get(lemma)
            if w_idx is not None:
                matched_word_indices.append(w_idx)
                words[w_idx]["image_ids"].append(img_idx)

        matched_word_indices = sorted(set(matched_word_indices))
        img["text_ids"] = matched_word_indices

    for w in words:
        w["image_ids"] = sorted(set(w.get("image_ids", [])))

    # Write outputs
    with open(args.data_out, "w", encoding="utf-8") as f:
        json.dump(images, f, ensure_ascii=False, indent=2)

    with open(args.words_out, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Built relations:\n"
        f"   - Images written to: {args.data_out}\n"
        f"   - Words written to:  {args.words_out}\n"
        f"   (Indices are 0-based.)"
    )


if __name__ == "__main__":
    main()
