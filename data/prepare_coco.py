"""Downloads COCO-2017, centre-crops/resizes every image once, and writes a caption manifest.

Edge maps are deliberately NOT precomputed here: they are generated on-the-fly in
dataset.py with randomised Canny thresholds, which is what makes the model robust to
hand-drawn doodles instead of overfitting to a single edge density.

Usage:
    python -m data.prepare_coco --out datasets/coco256 --resolution 256
    python -m data.prepare_coco --out datasets/coco256_small --resolution 256 --max-samples 5000
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

COCO_URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download(url: str, dest: Path) -> None:
    """Streams a large zip to disk with a progress bar, skipping files already present."""
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))


def unzip(archive: Path, dest: Path, sentinel: Path) -> None:
    """Extracts an archive unless its sentinel path already exists (makes reruns cheap)."""
    if sentinel.exists():
        print(f"[skip] {sentinel} already extracted")
        return
    print(f"[unzip] {archive.name} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def load_captions(ann_file: Path) -> dict[str, list[str]]:
    """Reads COCO caption JSON and returns {file_name: [caption, ...]} for O(1) lookup later."""
    with open(ann_file, "r", encoding="utf-8") as fh:
        blob = json.load(fh)

    id_to_name = {img["id"]: img["file_name"] for img in blob["images"]}
    captions: dict[str, list[str]] = defaultdict(list)
    for ann in blob["annotations"]:
        name = id_to_name.get(ann["image_id"])
        if name:
            captions[name].append(ann["caption"].strip())
    return captions


def _process_one(args: tuple[str, str, str], resolution: int) -> str | None:
    """Centre-crops one image to a square, resizes to `resolution`, and saves it as JPEG."""
    src, dst, _name = args
    try:
        img = Image.open(src).convert("RGB")
    except (OSError, ValueError):
        return None  # a handful of COCO files are corrupt; drop them silently

    width, height = img.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((resolution, resolution), Image.BICUBIC)
    img.save(dst, "JPEG", quality=95)
    return dst


def main() -> None:
    """Orchestrates download -> extract -> resize -> manifest for one dataset split."""
    parser = argparse.ArgumentParser(description="Prepare COCO for ControlNet training")
    parser.add_argument("--raw", default="datasets/raw", help="where the COCO zips land")
    parser.add_argument("--out", default="datasets/coco256", help="processed output directory")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    parser.add_argument("--max-samples", type=int, default=-1, help="-1 = all images")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-download", action="store_true", help="COCO is already on disk")
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw), Path(args.out)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # --- 1. fetch + extract -------------------------------------------------
    if not args.skip_download:
        for name in (f"{args.split}.zip", "annotations_trainval2017.zip"):
            download(COCO_URLS[name], raw_dir / name)
        unzip(raw_dir / f"{args.split}.zip", raw_dir, raw_dir / args.split)
        unzip(
            raw_dir / "annotations_trainval2017.zip",
            raw_dir,
            raw_dir / "annotations" / f"captions_{args.split}.json",
        )

    # --- 2. captions --------------------------------------------------------
    captions = load_captions(raw_dir / "annotations" / f"captions_{args.split}.json")
    src_dir = raw_dir / args.split
    files = sorted(p.name for p in src_dir.glob("*.jpg") if p.name in captions)
    if args.max_samples > 0:
        files = files[: args.max_samples]
    print(f"[info] {len(files)} images selected from {args.split}")

    # --- 3. resize in parallel ---------------------------------------------
    jobs = [(str(src_dir / f), str(img_out / f), f) for f in files]
    worker = partial(_process_one, resolution=args.resolution)
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(worker, jobs, chunksize=64), total=len(jobs)))
    ok = {Path(r).name for r in results if r}

    # --- 4. manifest --------------------------------------------------------
    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for name in files:
            if name in ok:
                fh.write(json.dumps({"file": name, "captions": captions[name]}) + "\n")

    meta = {"resolution": args.resolution, "split": args.split, "count": len(ok)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {len(ok)} images -> {img_out}\n[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
