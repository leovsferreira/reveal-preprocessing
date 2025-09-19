# Reveal Dataset Preprocessing

This repository contains all the necessary scripts and instructions to prepare any dataset of images and text for use in **Reveal**.

---

## 📂 Expected Data Structure

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

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Also make sure to download the English spaCy model:

```bash
python -m spacy download en_core_web_sm
```

---

## 🖼 Step 1 — Image Preprocessing

Make all images uniform (resized) and generate thumbnails.

Run the script:

**Windows (PowerShell / CMD):**
```bash
python preprocess_images.py ^
  --input-folder "path\to\your\images" ^
  --output-folder "path\to\processed\images" ^
  --thumb-folder "path\to\thumbnails"
```

**Linux / macOS:**
```bash
python preprocess_images.py \
  --input-folder "/path/to/your/images" \
  --output-folder "/path/to/processed/images" \
  --thumb-folder "/path/to/thumbnails"
```

Instructions and parameter details are documented at the top of `preprocess_images.py`.

---

## 📝 Step 2 — Text Preprocessing

Extract all **unique words** from captions (`output` field in JSON) to be used for plotting word embeddings for textual search.

Run the script:

**Windows (PowerShell / CMD):**
```bash
python extract_unique_words.py ^
  --input-file "path\to\records.json" ^
  --output-file "path\to\unique_words.csv" ^
  --min-len 3
```

**Linux / macOS:**
```bash
python extract_unique_words.py \
  --input-file "/path/to/records.json" \
  --output-file "/path/to/unique_words.csv" \
  --min-len 3
```

The script:
- Normalizes text  
- Removes stopwords  
- Keeps only content words (NOUN, VERB, ADJ, ADV)  
- Outputs a CSV of unique lemmatized words  

---

## ✅ Summary

1. **Install dependencies** → `pip install -r requirements.txt`  
2. **Download spaCy model** → `python -m spacy download en_core_web_sm`  
3. **Preprocess images** → `preprocess_images.py`  
4. **Extract unique words** → `create_unique_texts.py`  

You now have a dataset ready to use in **Reveal** 🚀
