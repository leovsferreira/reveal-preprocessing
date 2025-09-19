"""
Batch image resizer + square thumbnail generator.

- Resizes every image in an input folder to fit within (max_width x max_height)
  while preserving aspect ratio, then saves to an output folder.
- Generates square thumbnails (default 93x93) via smart center crop and saves
  them to a thumbnails folder.
- Uses concurrent processing for speed.
- Windows-friendly (protects multiprocessing with the __main__ guard).

Call with your folders to ensure they are actually used, e.g.:

  # Windows (PowerShell / CMD)
  python preprocess_images.py ^
    --input-folder "C:\\path\\to\\your\\images" ^
    --output-folder "C:\\path\\to\\processed\\images" ^
    --thumb-folder "C:\\path\\to\\thumbnails" ^
    --max-width 800 --max-height 600 --thumb-size 93

  # Linux / macOS
  python preprocess_images.py \
    --input-folder "/path/to/your/images" \
    --output-folder "/path/to/processed/images" \
    --thumb-folder "/path/to/thumbnails" \
    --max-width 800 --max-height 600 --thumb-size 93

Notes:
- There is no separate "input_thumb_folder" (it would be redundant with input).
- Thumbnails are created from the *resized* image for consistency.
- Valid crop types: top | center | bottom (via --crop-type).
"""

import argparse
import concurrent.futures
import os
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


def save_image(img: Image.Image, path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        img.save(path, quality=95, optimize=True)
    elif ext in ('.png',):
        img.save(path, optimize=True)
    else:
        img.save(path)


def process_image(filename: str, input_folder: str, output_folder: str,
                  max_width: int, max_height: int,
                  thumb_folder: str, thumb_size: int, crop_type: str):
    img_path = os.path.join(input_folder, filename)
    try:
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            width_ratio = max_width / img.width
            height_ratio = max_height / img.height
            new_ratio = min(width_ratio, height_ratio)

            new_width = max(1, int(img.width * new_ratio))
            new_height = max(1, int(img.height * new_ratio))

            resized_img = img.resize((new_width, new_height), LANCZOS)

            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(thumb_folder, exist_ok=True)

            out_path = os.path.join(output_folder, filename)
            save_image(resized_img, out_path)

            thumb = resize_and_crop(resized_img, (thumb_size, thumb_size), crop_type=crop_type)
            thumb_path = os.path.join(thumb_folder, filename)
            save_image(thumb, thumb_path)

            print(f"OK: {filename}")
    except Exception as e:
        print(f"ERROR: {filename} -> {e}")


def resize_images(input_folder: str, output_folder: str,
                  max_width: int, max_height: int,
                  thumb_folder: str, thumb_size: int,
                  crop_type: str = 'center', workers: int | None = None):
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")
    files = [f for f in os.listdir(input_folder)
             if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(supported)]

    if not files:
        print("No images found.")
        return

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(thumb_folder, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_image, filename, input_folder, output_folder,
                max_width, max_height, thumb_folder, thumb_size, crop_type
            )
            for filename in files
        ]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()


def parse_args():
    parser = argparse.ArgumentParser(description="Resize images and create square thumbnails.")
    parser.add_argument("--input-folder", required=False, default=r"./")
    parser.add_argument("--output-folder", required=False, default=r"./")
    parser.add_argument("--thumb-folder", required=False, default=r"./")
    parser.add_argument("--max-width", type=int, default=800)
    parser.add_argument("--max-height", type=int, default=600)
    parser.add_argument("--thumb-size", type=int, default=93, help="Square thumbnail size in pixels (e.g., 93 -> 93x93).")
    parser.add_argument("--crop-type", choices=["top", "center", "bottom"], default="center")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU cores).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    resize_images(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        max_width=args.max_width,
        max_height=args.max_height,
        thumb_folder=args.thumb_folder,
        thumb_size=args.thumb_size,
        crop_type=args.crop_type,
        workers=args.workers
    )
