"""
Extract unique content words from JSON captions (cleaner version).

- spaCy tokenization + lemmatization + stopword removal
- Keep only POS in {NOUN, VERB, ADJ, ADV}
- Drop non-alphabetic, short tokens (len < 3), ordinal suffixes, and a small noise list
- (Optional) wordfreq Zipf threshold to drop super-rare junk/typos

Usage:
  # Windows
  python extract_unique_words.py ^
    --input-file "data\\data.json" ^
    --output-file "data\\unique_words.json" ^
    --min-len 3

  # macOS / Linux
  python extract_unique_words.py \
    --input-file "data/records.json" \
    --output-file "data/unique_words.json" \
    --min-len 3
"""

import json, csv, argparse, sys
import spacy

try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None

ALLOWED_POS = {"NOUN", "VERB", "ADJ", "ADV"}
ORDINAL_SUFFIXES = {"st", "nd", "rd", "th"}
NOISE = {  
    "cv","cta","ch","de","dom","ess","lb","mph","ost","midw","st","nd","rd","th","url",
}

def parse_args():
    p = argparse.ArgumentParser(description="Extract unique clean words from captions.")
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--min-len", type=int, default=3, help="Minimum token length to keep.")
    p.add_argument("--zipf-min", type=float, default=None,
                   help="If set and wordfreq is installed, drop tokens with Zipf freq below this (e.g., 3.0).")
    return p.parse_args()

def is_noise_token(t: str, min_len: int, zipf_min):
    if len(t) < min_len:             
        return True
    if t in NOISE:                   
        return True
    if t in ORDINAL_SUFFIXES:        
        return True
    if zipf_min is not None and zipf_frequency is not None:
        if zipf_frequency(t, "en") < zipf_min:
            return True
    return False

def process_file(nlp, input_file, output_file, min_len, zipf_min):
    unique_words = set()

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for rec in data:
        text = rec.get("output", "")
        if not text:
            continue
        doc = nlp(text)
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
            unique_words.add(lemma)

    result = [
        {"word": word, "coordinates": [], "image_ids": []}
        for word in sorted(unique_words)
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted {len(unique_words)} unique words -> {output_file}")

def main():
    args = parse_args()
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("⚠️ Missing spaCy model. Run:  python -m spacy download en_core_web_sm", file=sys.stderr)
        sys.exit(1)

    process_file(nlp, args.input_file, args.output_file, args.min_len, args.zipf_min)

if __name__ == "__main__":
    main()
