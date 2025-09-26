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
    --images-folder "data\\processed" ^
    --data-in "data\\data_with_relations.json" ^
    --data-out "data\\data_final.json" ^
    --words-in "data\\unique_words_with_relations.json" ^
    --words-out "data\\unique_words_final.json" ^
    --image-pt "data\\multi_clip_images_embedding.pt" ^
    --words-pt "data\\multi_clip_words_embedding.pt" ^
    --joint-pt "data\\multi_clip_joint_embedding.pt" ^
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
    except Exception:
        return None

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

def truncate_to_max_tokens(text: str, tokenizer, max_tokens: int = 512) -> str:
    enc = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_tokens
    )
    return tokenizer.decode(enc["input_ids"], skip_special_tokens=True)


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

@torch.no_grad()
def encode_images(records: List[Dict[str, Any]],
                  images_folder: str,
                  model,
                  preprocess,
                  device: str,
                  batch_size: int = 32) -> torch.Tensor:
    
    embed_dim = None

    embs: List[torch.Tensor] = []
    batch: List[torch.Tensor] = []

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
def encode_texts_textlevel(texts: list[str],
                           txt_model,
                           tokenizer,
                           device: str,
                           batch_size: int = 64,
                           max_tokens: int = 512) -> torch.Tensor:
    texts = [truncate_to_max_tokens(t if isinstance(t, str) else str(t),
                                    tokenizer, max_tokens=max_tokens)
             for t in texts]

    embs: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch = texts[start:start + batch_size]
        feats = txt_model.forward(batch, tokenizer).to(device)
        feats = feats.float()
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-9)
        embs.append(feats.cpu())
    return torch.cat(embs, dim=0) if embs else torch.zeros((0, 512))


def umap_2d(embeddings: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
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
    p = argparse.ArgumentParser(description="Build image/word/joint embeddings and assign UMAP x,y back into JSONs.")
    p.add_argument("--images-folder", required=True, help="Folder containing images (filenames must match data JSON).")
    p.add_argument("--data-in", required=True, help="Input data JSON with image records.")
    p.add_argument("--data-out", required=True, help="Output data JSON with x,y added/overwritten.")
    p.add_argument("--words-in", required=True, help="Input unique_words JSON.")
    p.add_argument("--words-out", required=True, help="Output unique_words JSON with x,y added/overwritten.")
    p.add_argument("--image-pt", required=True, help="Path to save image embedding tensor (.pt).")
    p.add_argument("--words-pt", required=True, help="Path to save words embedding tensor (.pt).")
    p.add_argument("--joint-pt", required=True, help="Path to save joint embedding tensor (.pt).")
    p.add_argument("--max-tokens", type=int, default=512, help="Max tokens per text for M-CLIP (captions/words will be truncated to this length).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data_in, "r", encoding="utf-8") as f:
        data_records: List[Dict[str, Any]] = json.load(f)
    with open(args.words_in, "r", encoding="utf-8") as f:
        word_records: List[Dict[str, Any]] = json.load(f)

    img_model, preprocess = build_image_encoder(device)
    txt_model, tokenizer = build_text_encoder()

    img_emb = encode_images(
        records=data_records,
        images_folder=args.images_folder,
        model=img_model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size
    )
    torch.save(img_emb, args.image_pt)

    img_xy = umap_2d(to_numpy(img_emb), args.umap_n_neighbors, args.umap_min_dist, args.seed)
    for i, rec in enumerate(data_records):
        rec["x"] = float(img_xy[i, 0])
        rec["y"] = float(img_xy[i, 1])

    words_list = [str(w.get("word", "")) for w in word_records]
    words_list = [w if w.strip() else "[EMPTY]" for w in words_list]

    word_emb = encode_texts_textlevel(
        texts=words_list,
        txt_model=txt_model,
        tokenizer=tokenizer,
        device=device,
        batch_size=max(32, args.batch_size),
        max_tokens=args.max_tokens
    )
    torch.save(word_emb, args.words_pt)

    word_xy = umap_2d(to_numpy(word_emb), args.umap_n_neighbors, args.umap_min_dist, args.seed)
    for i, w in enumerate(word_records):
        w["x"] = float(word_xy[i, 0])
        w["y"] = float(word_xy[i, 1])

    captions = [str(r.get("output", "")) for r in data_records]
    cap_emb = encode_texts_textlevel(
        texts=[str(r.get("output", "")) for r in data_records],
        txt_model=txt_model,
        tokenizer=tokenizer,
        device=device,
        batch_size=min(8, args.batch_size),
        max_tokens=args.max_tokens
    )

    img_norm = l2_normalize(img_emb)
    cap_norm = l2_normalize(cap_emb)
    joint = torch.cat([img_norm, cap_norm], dim=-1)
    torch.save(joint, args.joint_pt)

    with open(args.data_out, "w", encoding="utf-8") as f:
        json.dump(data_records, f, ensure_ascii=False, indent=2)

    with open(args.words_out, "w", encoding="utf-8") as f:
        json.dump(word_records, f, ensure_ascii=False, indent=2)

    print("\n✅ Completed:")
    print(f"  • Image embeddings  -> {args.image_pt} (shape={tuple(img_emb.shape)})")
    print(f"  • Words embeddings  -> {args.words_pt} (shape={tuple(word_emb.shape)})")
    print(f"  • Joint embeddings  -> {args.joint_pt} (shape={tuple(joint.shape)})")
    print(f"  • Data JSON (x,y)   -> {args.data_out}")
    print(f"  • Words JSON (x,y)  -> {args.words_out}")


if __name__ == "__main__":
    main()
