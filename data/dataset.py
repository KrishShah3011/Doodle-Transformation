"""Yields (image, edge-map, caption) triplets, generating Canny edges on-the-fly per sample."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.image import canny_edges


class CocoEdgeDataset(Dataset):
    """Reads the prepare_coco manifest and returns one training triplet per index."""

    def __init__(
        self,
        processed_dir: str | Path,
        resolution: int = 256,
        max_samples: int = -1,
        canny_low_range: tuple[int, int] = (50, 100),
        canny_high_range: tuple[int, int] = (150, 250),
        caption_dropout: float = 0.5,
        hflip: bool = True,
        skip_first: int = 0,
    ):
        self.root = Path(processed_dir)
        self.img_dir = self.root / "images"
        self.resolution = resolution
        self.canny_low_range = tuple(canny_low_range)
        self.canny_high_range = tuple(canny_high_range)
        self.caption_dropout = caption_dropout
        self.hflip = hflip

        self.records = self._load_manifest(self.root / "manifest.jsonl")
        self.records = self.records[skip_first:]           # lets val split off the front
        if max_samples > 0:
            self.records = self.records[:max_samples]      # `--max-samples` flexibility
        if not self.records:
            raise RuntimeError(f"No records found in {self.root}. Did you run data/prepare_coco.py?")

    @staticmethod
    def _load_manifest(path: Path) -> list[dict]:
        """Parses the JSONL manifest into a list of {file, captions} dicts."""
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def __len__(self) -> int:
        """Number of usable training samples after the max_samples cap."""
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """Builds one triplet: VAE-ready image, Canny hint, and a (possibly dropped) caption."""
        rec = self.records[idx]
        img = Image.open(self.img_dir / rec["file"]).convert("RGB")
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)

        arr = np.asarray(img, dtype=np.uint8)
        if self.hflip and random.random() < 0.5:
            arr = np.ascontiguousarray(arr[:, ::-1, :])    # flip BEFORE Canny so they stay aligned

        # Randomised thresholds => varied edge density => generalises to human doodles.
        low = random.randint(*self.canny_low_range)
        high = random.randint(*self.canny_high_range)
        hint = canny_edges(arr, low, max(high, low + 1))

        # Empty caption 50% of the time forces the model to read the edge map for semantics.
        caption = "" if random.random() < self.caption_dropout else random.choice(rec["captions"])

        return {
            "pixel_values": torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1),
            "hint": torch.from_numpy(hint.astype(np.float32) / 255.0).permute(2, 0, 1),
            "caption": caption,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Stacks tensors but keeps captions as a plain list for the CLIP tokenizer."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "hint": torch.stack([b["hint"] for b in batch]),
        "caption": [b["caption"] for b in batch],
    }
