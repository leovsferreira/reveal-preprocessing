"""
Reveal — Build CLIP/Multilingual-CLIP embeddings (images, words, and joint), and assign UMAP 2D coords.

What this script does:
1) Image embeddings:
   - Loads records from data_with_relations.json (with "filename", "output", "text_ids", ...)
   - Encodes each image file with OpenCLIP ViT-B-16-plus-240 (laion400m_e32)
   - Saves tensor as: multi_clip_images_embedding.pt
   - Projects to 2D via UMAP (metric='cosine') and writes x,y back into data_with_relations.json

2) Word embeddings:
   - Loads entries from unique_words_with_relations.json (with "word", optionally x,y,image_ids)
   - Encodes each word using Multilingual-CLIP (XLM-Roberta-Large-ViT-B-16Plus)
   - Saves tensor as: multi_clip_words_embedding.pt
   - Projects to 2D via UMAP and writes x,y back into unique_words_with_relations.json

3) Joint embeddings (image + full caption):
   - For each image, encodes:
       * image embedding (OpenCLIP)
       * caption embedding of the whole "output" string (Multilingual-CLIP)
     L2-normalizes each and concatenates: [img_norm || txt_norm]
   - Saves tensor as: multi_clip_joint_embedding.pt

Usage (paths use forward slashes so they're Windows-safe):

  # Windows
  python create_embeddings.py ^
    --images-folder "path\\to\\images" ^
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

# macOS / Linux
  python create_embeddings.py \
    --images-folder "/path/to/images" \
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

Requirements:
  pip install open_clip_torch multilingual-clip transformers umap-learn pillow tqdm
  (and a compatible torch)
  python -m spacy download en_core_web_sm  (not strictly required here)

Notes:
- Indices in tensors match the order in the corresponding JSON arrays.
- If an image file is missing, a zero-vector is inserted to keep indices aligned.
- UMAP random_state is set; change --seed to vary layouts.
"""

"""
Reveal — Build CLIP/Multilingual-CLIP embeddings (images, words, and joint), and assign UMAP 2D coords.

What this script does:
1) Image embeddings:
   - Loads records from data_with_relations.json (with "filename", "output", "text_ids", ...)
   - Encodes each image file with OpenCLIP ViT-B-16-plus-240 (laion400m_e32)
   - Saves tensor as: multi_clip_images_embedding.pt
   - Projects to 2D via UMAP (metric='cosine') and writes x,y back into data_with_relations.json

2) Word embeddings:
   - Loads entries from unique_words_with_relations.json (with "word", optionally x,y,image_ids)
   - Encodes each word using Multilingual-CLIP (XLM-Roberta-Large-ViT-B-16Plus)
   - Saves tensor as: multi_clip_words_embedding.pt
   - Projects to 2D via UMAP and writes x,y back into unique_words_with_relations.json

3) Joint embeddings (image + full caption):
   - For each image, encodes:
       * image embedding (OpenCLIP)
       * caption embedding of the whole "output" string (Multilingual-CLIP)
     L2-normalizes each and concatenates: [img_norm || txt_norm]
   - Saves tensor as: multi_clip_joint_embedding.pt
   - Projects to 2D via UMAP and writes joint_x, joint_y back into data_with_relations.json

Usage:

  # Windows
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

  # macOS / Linux
  python create_embeddings.py \
    --images-folder "/path/to/processed" \
    --data-in "/path/to/data_with_relations.json" \
    --data-out "/path/to/data_final.json" \
    --words-in "/path/to/unique_words_with_relations.json" \
    --words-out "/path/to/unique_words_final.json" \
    --image-pt "/path/to/multi_clip_images_embedding.pt" \
    --words-pt "/path/to/multi_clip_words_embedding.pt" \
    --joint-pt "/path/to/multi_clip_joint_embedding.pt" \
    --batch-size 32 \
    --umap-n-neighbors 15 \
    --umap-min-dist 0.1 \
    --seed 42

Requirements:
  pip install open_clip_torch multilingual-clip transformers umap-learn pillow tqdm
  (and a compatible torch)

Notes:
- Indices in tensors match the order in the corresponding JSON arrays
- If an image file is missing, a zero-vector is inserted to keep indices aligned
- UMAP random_state is set; change --seed to vary layouts
- Each data record will have x,y (from image) and joint_x,joint_y (from joint embedding)
"""

import argparse
import json
import os
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import umap

import open_clip
from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer


def l2_normalize(t: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return t / (t.norm(dim=-1, keepdim=True) + eps)


def safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as e:
        print(f"Warning: Could not open image {path}: {e}")
        return None


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def build_image_encoder(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16-plus-240', pretrained="laion400m_e32"
    )
    model.eval().to(device)
    return model, preprocess


def build_text_encoder():
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'
    txt_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    txt_model.eval()
    return txt_model, tokenizer


def window_token_ids(text: str, tokenizer, max_tokens: int = 512, overlap: int = 64) -> List[List[int]]:
    if overlap < 0:
        overlap = 0
    if overlap >= max_tokens:
        overlap = max_tokens - 1

    base = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if len(base) == 0:
        base = tokenizer(" ", add_special_tokens=False)["input_ids"]

    windows = []
    step = max_tokens - overlap - 2
    
    for start in range(0, len(base), step):
        chunk_ids = base[start:start + max_tokens - 2]
        
        enc = tokenizer.prepare_for_model(
            chunk_ids, 
            add_special_tokens=True, 
            truncation=True,
            max_length=max_tokens,
            return_tensors=None
        )
        
        if len(enc["input_ids"]) <= max_tokens:
            windows.append(enc["input_ids"])
        else:
            windows.append(enc["input_ids"][:max_tokens])
        
        if len(chunk_ids) < max_tokens - 2:
            break
    
    return windows


def pool_window_embeddings(embs: List[torch.Tensor], weights: Optional[List[int]], mode: str) -> torch.Tensor:
    if not embs:
        return torch.zeros(512)

    E = torch.stack(embs, dim=0)

    if mode == "max":
        return E.max(dim=0).values

    if mode == "weighted" and weights is not None and sum(weights) > 0:
        w = torch.tensor(weights, dtype=E.dtype).unsqueeze(1)  # [W,1]
        return (E * (w / w.sum())).sum(dim=0)

    return E.mean(dim=0)


@torch.no_grad()
def encode_images(records: List[Dict[str, Any]],
                  images_folder: str,
                  model,
                  preprocess,
                  device: str,
                  batch_size: int = 32) -> torch.Tensor:
    
    embed_dim = None
    embs: List[torch.Tensor] = []
    batch: List[Optional[torch.Tensor]] = []

    for i in tqdm(range(len(records)), desc="Encoding images"):
        fname = records[i].get("filename", "")
        img_path = os.path.join(images_folder, fname)
        img = safe_open_image(img_path)
        if img is None:
            batch.append(None)
        else:
            img_t = preprocess(img).unsqueeze(0).to(device)
            batch.append(img_t)

        if len(batch) == batch_size or i == len(records) - 1:
            valid = [x for x in batch if x is not None]
            if valid:
                x = torch.cat(valid, dim=0)
                feat = model.encode_image(x)
                feat = feat.float()
                feat = l2_normalize(feat)
                ptr = 0
                for b in batch:
                    if b is None:
                        if embed_dim is None:
                            embed_dim = feat.shape[-1]
                        embs.append(torch.zeros(embed_dim, dtype=feat.dtype))
                    else:
                        embs.append(feat[ptr].cpu())
                        if embed_dim is None:
                            embed_dim = feat.shape[-1]
                        ptr += 1
            else:
                if embed_dim is None:
                    embed_dim = 512
                for _ in batch:
                    embs.append(torch.zeros(embed_dim))

            batch = []

    if embed_dim is None:
        embed_dim = 512
        embs = [torch.zeros(embed_dim) for _ in records]

    return torch.stack(embs, dim=0)


@torch.no_grad()
def encode_texts_textlevel(
    texts: List[str],
    txt_model,
    tokenizer,
    device: str,
    batch_size: int = 64,
    max_tokens: int = 512,
    overlap: int = 64,
    pooling: str = "mean",
) -> torch.Tensor:
    out: List[torch.Tensor] = []

    if pooling == "truncate":
        def truncate_to_max_tokens(s: str) -> str:
            enc = tokenizer(s, add_special_tokens=True, truncation=True, max_length=max_tokens)
            return tokenizer.decode(enc["input_ids"], skip_special_tokens=True)

        texts_small = [truncate_to_max_tokens(str(t)) for t in texts]

        for start in tqdm(range(0, len(texts_small), batch_size), desc="Encoding texts"):
            batch = texts_small[start:start + batch_size]
            feats = txt_model.forward(batch, tokenizer).to(device)
            feats = feats.float()
            feats = l2_normalize(feats)
            out.append(feats.cpu())

        return torch.cat(out, dim=0) if out else torch.zeros((0, 640))

    for s in tqdm(texts, desc="Encoding texts (windowed)"):
        s = str(s)
        win_ids = window_token_ids(s, tokenizer, max_tokens=max_tokens, overlap=overlap)
        
        if not win_ids:
            win_ids = [tokenizer(" ", add_special_tokens=True, truncation=True, max_length=max_tokens)["input_ids"]]
        
        win_texts = []
        for ids in win_ids:
            if len(ids) > max_tokens:
                ids = ids[:max_tokens]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            win_texts.append(text)

        doc_embs: List[torch.Tensor] = []
        for start in range(0, len(win_texts), max(1, batch_size // 4)):
            chunk = win_texts[start:start + max(1, batch_size // 4)]
            try:
                feats = txt_model.forward(chunk, tokenizer).to(device)
                feats = feats.float()
                feats = l2_normalize(feats)
                doc_embs.append(feats.cpu())
            except Exception as e:
                print(f"Warning: Error encoding text chunk, using zero vector: {e}")
                doc_embs.append(torch.zeros(1, 640))

        if doc_embs:
            doc_embs_cat = torch.cat(doc_embs, dim=0)  # [W, D]
            weights = [len(ids) for ids in win_ids]
            pooled = pool_window_embeddings(
                [doc_embs_cat[i] for i in range(doc_embs_cat.shape[0])],
                weights if pooling == "weighted" else None,
                pooling
            )
            out.append(pooled)
        else:
            out.append(torch.zeros(640))

    return torch.stack(out, dim=0)


def umap_2d(embeddings: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    print(f"Computing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, seed={seed})")
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        verbose=False
    )
    return reducer.fit_transform(embeddings)


def parse_args():
    p = argparse.ArgumentParser(
        description="Build image/word/joint embeddings and assign UMAP x,y back into JSONs."
    )
    p.add_argument("--images-folder", required=True, 
                   help="Folder containing images (filenames must match data JSON).")
    p.add_argument("--data-in", required=True, 
                   help="Input data JSON with image records.")
    p.add_argument("--data-out", required=True, 
                   help="Output data JSON with x,y and joint_x,joint_y added.")
    p.add_argument("--words-in", required=True, 
                   help="Input unique_words JSON.")
    p.add_argument("--words-out", required=True, 
                   help="Output unique_words JSON with x,y added.")
    p.add_argument("--image-pt", required=True, 
                   help="Path to save image embedding tensor (.pt).")
    p.add_argument("--words-pt", required=True, 
                   help="Path to save words embedding tensor (.pt).")
    p.add_argument("--joint-pt", required=True, 
                   help="Path to save joint embedding tensor (.pt).")
    p.add_argument("--max-tokens", type=int, default=512, 
                   help="Token cap per window for M-CLIP.")
    p.add_argument("--overlap", type=int, default=64,
                   help="Token overlap between consecutive windows (0..max_tokens-1).")
    p.add_argument("--pooling", choices=["truncate", "mean", "weighted", "max"], default="mean",
                   help="How to pool window embeddings into one vector per text.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading data from {args.data_in}")
    with open(args.data_in, "r", encoding="utf-8") as f:
        data_records: List[Dict[str, Any]] = json.load(f)
    
    print(f"Loading words from {args.words_in}")
    with open(args.words_in, "r", encoding="utf-8") as f:
        word_records: List[Dict[str, Any]] = json.load(f)

    print(f"Found {len(data_records)} images and {len(word_records)} words")

    print("\nBuilding models...")
    img_model, preprocess = build_image_encoder(device)
    txt_model, tokenizer = build_text_encoder()
    txt_model.to(device)

    print("\n=== Processing Image Embeddings ===")
    img_emb = encode_images(
        records=data_records,
        images_folder=args.images_folder,
        model=img_model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size
    )
    torch.save(img_emb, args.image_pt)
    print(f"Saved image embeddings to {args.image_pt} (shape={tuple(img_emb.shape)})")

    print("Computing UMAP for images...")
    img_xy = umap_2d(to_numpy(img_emb), args.umap_n_neighbors, args.umap_min_dist, args.seed)
    for i, rec in enumerate(data_records):
        rec["x"] = float(img_xy[i, 0])
        rec["y"] = float(img_xy[i, 1])

    print("\n=== Processing Word Embeddings ===")
    words_list = [str(w.get("word", "")) for w in word_records]
    words_list = [w if w.strip() else "[EMPTY]" for w in words_list]

    word_emb = encode_texts_textlevel(
        texts=words_list,
        txt_model=txt_model,
        tokenizer=tokenizer,
        device=device,
        batch_size=max(32, args.batch_size),
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        pooling="truncate"
    )
    torch.save(word_emb, args.words_pt)
    print(f"Saved word embeddings to {args.words_pt} (shape={tuple(word_emb.shape)})")

    print("Computing UMAP for words...")
    word_xy = umap_2d(to_numpy(word_emb), args.umap_n_neighbors, args.umap_min_dist, args.seed)
    for i, w in enumerate(word_records):
        w["x"] = float(word_xy[i, 0])
        w["y"] = float(word_xy[i, 1])

    print("\n=== Processing Joint Embeddings ===")
    captions = [str(r.get("output", "")) for r in data_records]
    cap_emb = encode_texts_textlevel(
        texts=captions,
        txt_model=txt_model,
        tokenizer=tokenizer,
        device=device,
        batch_size=min(8, args.batch_size),
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        pooling=args.pooling
    )

    img_norm = l2_normalize(img_emb)
    cap_norm = l2_normalize(cap_emb)
    joint = torch.cat([img_norm, cap_norm], dim=-1)
    torch.save(joint, args.joint_pt)
    print(f"Saved joint embeddings to {args.joint_pt} (shape={tuple(joint.shape)})")

    print("Computing UMAP for joint embeddings...")
    joint_xy = umap_2d(to_numpy(joint), args.umap_n_neighbors, args.umap_min_dist, args.seed)
    for i, rec in enumerate(data_records):
        rec["joint_x"] = float(joint_xy[i, 0])
        rec["joint_y"] = float(joint_xy[i, 1])

    print("\n=== Saving Results ===")
    with open(args.data_out, "w", encoding="utf-8") as f:
        json.dump(data_records, f, ensure_ascii=False, indent=2)
    print(f"Saved data with coordinates to {args.data_out}")

    with open(args.words_out, "w", encoding="utf-8") as f:
        json.dump(word_records, f, ensure_ascii=False, indent=2)
    print(f"Saved words with coordinates to {args.words_out}")

    print("\n✅ Completed:")
    print(f"  • Image embeddings  -> {args.image_pt} (shape={tuple(img_emb.shape)})")
    print(f"  • Words embeddings  -> {args.words_pt} (shape={tuple(word_emb.shape)})")
    print(f"  • Joint embeddings  -> {args.joint_pt} (shape={tuple(joint.shape)})")
    print(f"  • Data JSON (x,y + joint_x,joint_y) -> {args.data_out}")
    print(f"  • Words JSON (x,y)  -> {args.words_out}")


if __name__ == "__main__":
    main()