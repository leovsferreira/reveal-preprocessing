"""
Batch image renamer + resizer + square thumbnail generator (index-based filenames).

- Reads a JSON dataset (data.json) where each item has:
    {"filename": "<original name in input folder>", "output": "<caption text>"}
- Uses the ORDER in data.json as the canonical index:
    * For item at position i, the processed image is saved as "i.jpg"
      (JPEG) to --output-folder
    * The thumbnail is saved as "i.jpg" to --thumb-folder
    * The record's "filename" is updated to "i.jpg" and written to --data-out

- Resizes every image to fit within (max_width x max_height) preserving aspect ratio.
- Generates square thumbnails (default 93x93) via smart center crop.
- Uses concurrent processing for speed.

Usage:

  # Windows (PowerShell / CMD)
  python preprocess_images.py ^
    --data-in "data\\data.json" ^
    --data-out "data\\data_indexed.json" ^
    --input-folder "data\\images\chicago" ^
    --output-folder "data\\processed_chicago" ^
    --thumb-folder "data\\thumbnails_chicago" ^
    --max-width 800 --max-height 600 --thumb-size 93

  # Linux / macOS
  python preprocess_images.py \
    --data-in "/path/to/data.json" \
    --data-out "/path/to/data_indexed.json" \
    --input-folder "/path/to/your/images" \
    --output-folder "/path/to/processed/images" \
    --thumb-folder "/path/to/thumbnails" \
    --max-width 800 --max-height 600 --thumb-size 93

Notes:
- Output images and thumbnails are always saved as JPEG with names "0.jpg", "1.jpg", ...
- Thumbnails are created from the resized image for consistency.
- Valid crop types: top | center | bottom (via --crop-type).
"""

import argparse
import concurrent.futures
import json
import os
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageOps

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS


def resize_and_crop(image: Image.Image, size, crop_type='center') -> Image.Image:
    img_ratio = image.width / float(image.height)
    target_ratio = size[0] / float(size[1])

    if img_ratio > target_ratio:
        new_w = int(size[1] * img_ratio)
        image = image.resize((new_w, size[1]), LANCZOS)
        if crop_type == 'top':
            box = (0, 0, size[0], size[1])
        elif crop_type == 'center':
            x0 = int((image.width - size[0]) / 2)
            box = (x0, 0, x0 + size[0], size[1])
        else:
            box = (image.width - size[0], 0, image.width, size[1])
    elif img_ratio < target_ratio:
        new_h = int(size[0] / img_ratio)
        image = image.resize((size[0], new_h), LANCZOS)
        if crop_type == 'top':
            box = (0, 0, size[0], size[1])
        elif crop_type == 'center':
            y0 = int((image.height - size[1]) / 2)
            box = (0, y0, size[0], y0 + size[1])
        else:
            box = (0, image.height - size[1], size[0], image.height)
    else:
        return image.resize(size, LANCZOS)

    return image.crop(box)


def save_jpeg(img: Image.Image, path: str):
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(path, format="JPEG", quality=95, optimize=True, subsampling="4:2:0")


def process_one(task: Tuple[int, str, str, str, int, int, str, int, str]) -> Tuple[int, str]:
    (i, orig_filename, input_folder, output_folder, max_w, max_h,
     thumb_folder, thumb_size, crop_type) = task

    src_path = os.path.join(input_folder, orig_filename)
    out_path = os.path.join(output_folder, f"{i}.jpg")
    thumb_path = os.path.join(thumb_folder, f"{i}.jpg")

    try:
        with Image.open(src_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            width_ratio = max_w / img.width
            height_ratio = max_h / img.height
            new_ratio = min(width_ratio, height_ratio, 1.0)

            new_width = max(1, int(img.width * new_ratio))
            new_height = max(1, int(img.height * new_ratio))

            resized_img = img.resize((new_width, new_height), LANCZOS)

            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(thumb_folder, exist_ok=True)

            save_jpeg(resized_img, out_path)

            thumb = resize_and_crop(resized_img, (thumb_size, thumb_size), crop_type=crop_type)
            save_jpeg(thumb, thumb_path)

        return (i, "OK")
    except Exception as e:
        return (i, f"ERROR: {orig_filename} -> {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Index-rename images, resize, and create square thumbnails; update data.json filenames to {index}.jpg.")
    p.add_argument("--data-in", required=True, help="Path to input data.json (list of records).")
    p.add_argument("--data-out", required=True, help="Path to write updated data.json (filenames set to {index}.jpg).")
    p.add_argument("--input-folder", required=True, help="Folder containing ORIGINAL images (as referenced by data.json filenames).")
    p.add_argument("--output-folder", required=True, help="Folder to write PROCESSED images (named {index}.jpg).")
    p.add_argument("--thumb-folder", required=True, help="Folder to write THUMBNAILS (named {index}.jpg).")
    p.add_argument("--max-width", type=int, default=800)
    p.add_argument("--max-height", type=int, default=600)
    p.add_argument("--thumb-size", type=int, default=93, help="Square thumbnail size in pixels (e.g., 93 -> 93x93).")
    p.add_argument("--crop-type", choices=["top", "center", "bottom"], default="center")
    p.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU cores).")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.data_in, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    if not isinstance(records, list):
        raise ValueError("data.json must be a JSON array.")

    tasks: List[Tuple[int, str, str, str, int, int, str, int, str]] = []
    for i, rec in enumerate(records):
        orig = rec.get("filename")
        if not orig:
            raise ValueError(f"Record {i} missing 'filename'.")
        tasks.append((
            i, orig,
            args.input_folder, args.output_folder,
            args.max_width, args.max_height,
            args.thumb_folder, args.thumb_size, args.crop_type
        ))

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.thumb_folder, exist_ok=True)

    results: List[Tuple[int, str]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())

    for i, status in sorted(results, key=lambda x: x[0]):
        print(f"[{i}] {status}")

    for i, rec in enumerate(records):
        rec["filename"] = f"{i}.jpg"

    with open(args.data_out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done.\n - Processed images -> {args.output_folder}\n - Thumbnails -> {args.thumb_folder}\n - Updated data.json -> {args.data_out}\n - Filenames now index-based (0.jpg, 1.jpg, ...)")


if __name__ == "__main__":
    main()
