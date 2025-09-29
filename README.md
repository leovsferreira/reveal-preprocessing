# Reveal Dataset Preprocessing

This repository contains all the necessary scripts and instructions to prepare any dataset of images and text for use in **Reveal**.

---

## Expected Data Structure

Your dataset should include:

1. **A folder with images**  
2. **A JSON file** describing each image and its caption, following this format:

```json
[
  {
    "filename": "image_1.jpg",
    "output": "image_1 caption"
  },
  {
    "filename": "image_2.jpg",
    "output": "image_2 caption"
  },
  ...
  {
    "filename": "image_n.jpg",
    "output": "image_n caption"
  }
]
```
---

## Recommended: Conda Environment

Create an isolated environment.

**CPU-only:**

```bash
conda create -n reveal-prep python=3.10 -y
conda activate reveal-prep
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**GPU (CUDA 12.1+):**

```bash
conda create -n reveal-prep-gpu python=3.10 -y
conda activate reveal-prep-gpu
pip install -r requirements-gpu.txt
python -m spacy download en_core_web_sm

```

---

## Step 1 — Image Preprocessing

Make all images uniform (resized) and generate thumbnails.

Run the script:

**Windows (PowerShell / CMD):**
```bash
python preprocess_images.py ^
  --data-in "\\path\\to\\data.json" ^
  --data-out "\\path\\to\\data_indexed.json" ^
  --input-folder "\\path\\to\\images" ^
  --output-folder "\\path\\to\\processed" ^
  --thumb-folder "\\path\\to\\thumbnails" ^
  --max-width 800 --max-height 600 --thumb-size 93
```

**Linux / macOS:**
```bash
python preprocess_images.py \
    --data-in "/path/to/data.json" \
    --data-out "/path/to/data_indexed.json" \
    --input-folder "/path/to/your/images" \
    --output-folder "/path/to/processed/images" \
    --thumb-folder "/path/to/thumbnails" \
    --max-width 800 --max-height 600 --thumb-size 93
```

Instructions and parameter details are documented at the top of `preprocess_images.py`.

---

## Step 2 — Text Preprocessing

Extract all **unique words** from captions (`output` field in JSON) to be used for plotting word embeddings for textual search.

Run the script:

**Windows (PowerShell / CMD):**
```bash
python extract_unique_words.py ^
  --input-file "path\\to\\data_indexed.json" ^
  --output-file "path\\to\\data\\unique_words.json" ^
  --min-len 3
```

**Linux / macOS:**
```bash
python extract_unique_words.py \
  --input-file "/path/to/data_indexed.json" \
  --output-file "/path/to/unique_words.json" \
  --min-len 3
```

Instructions and parameter details are documented at the top of `extract_unique_words.py`.

---

## Step 3 — Build Relations Between Images and Text

Link images and unique words by building a bidirectional mapping:

1. Each image gets a `text_ids` list → indices of words found in its caption

2. Each word gets an `image_ids` list → indices of images where it appears

Run the script:

**Windows (PowerShell / CMD):**
```bash
python build_relations.py ^
  --data-in "path\\to\\data_indexed.json" ^
  --words-in "path\\to\\unique_words.json" ^
  --data-out "path\\to\\data_with_relations.json" ^
  --words-out "path\\to\\unique_words_with_relations.json" ^
  --min-len 3
```

**Linux / macOS:**
```bash
python build_relations.py \
  --data-in "/path/to/data_indexed.json" \
  --words-in "/path/to/unique_words.json" \
  --data-out "/path/to/data_with_relations.json" \
  --words-out "/path/to/unique_words_with_relations.json" \
  --min-len 3
```

Instructions and parameter details are documented at the top of `build_relations.py`.

## Step 4 — Build Relations Between Images and Text

This step builds:

- Image embeddings with OpenCLIP and writes 2D UMAP x,y into your data JSON → saves `multi_clip_images_embedding.pt`

- Word embeddings with Multilingual-CLIP and writes 2D UMAP x,y into your words JSON → saves `multi_clip_words_embedding.pt`

- Joint embeddings per image using the image + its full caption (output) → concatenates normalized image + text vectors → saves `multi_clip_joint_embedding.pt`

**Windows (PowerShell / CMD):**
```bash
python create_embeddings.py ^
  --images-folder "path\\to\\processed" ^
  --data-in "path\\to\\data_with_relations.json" ^
  --data-out "path\\to\\data_final.json" ^
  --words-in "path\\to\\unique_words_with_relations.json" ^
  --words-out "path\\to\\unique_words_final.json" ^
  --image-pt "path\\to\\multi_clip_images_embedding.pt" ^
  --words-pt "path\\to\\multi_clip_words_embedding.pt" ^
  --joint-pt "path\\to\\multi_clip_joint_embedding.pt" ^
  --batch-size 32 ^
  --umap-n-neighbors 15 ^
  --umap-min-dist 0.1 ^
  --seed 42
```

**Linux / macOS:**

```bash
python create_embeddings.py \
  --images-folder "/path/to/processed" \
  --data-in "/path/to/data_with_relations.json" \
  --data-out "/path/to/data_with_final.json" \
  --words-in "/path/to/unique_words_with_relations.json" \
  --words-out "/path/to/unique_words_with_final.json" \
  --image-pt "/path/to/multi_clip_images_embedding.pt" \
  --words-pt "/path/to/multi_clip_words_embedding.pt" \
  --joint-pt "/path/to/multi_clip_joint_embedding.pt" \
  --batch-size 32 \
  --umap-n-neighbors 15 \
  --umap-min-dist 0.1 \
  --seed 42
```

Instructions and parameter details are documented at the top of `create_embeddings.py`.

## ✅ Summary

1. **Install dependencies** → `pip install -r requirements.txt`  
2. **Download spaCy model** → `python -m spacy download en_core_web_sm`  
3. **Preprocess images** → `preprocess_images.py`  
4. **Extract unique words** → `create_unique_texts.py`
5. **Build relations** → `build_relations.py`
6. **Create embeddings + UMAP (images, words, joint)** → `create_embeddings.py`