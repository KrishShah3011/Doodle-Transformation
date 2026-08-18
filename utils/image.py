"""Shared image helpers: tensor<->PIL conversion, doodle normalisation, and contact sheets."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


def pil_to_model_input(img: Image.Image, resolution: int) -> torch.Tensor:
    """Centre-crops + resizes a PIL image and scales it to the [-1, 1] range the VAE expects."""
    img = center_crop_square(img).resize((resolution, resolution), Image.BICUBIC)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def center_crop_square(img: Image.Image) -> Image.Image:
    """Crops the largest centred square so resizing never distorts the aspect ratio."""
    width, height = img.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def latent_to_pil(sample: torch.Tensor) -> list[Image.Image]:
    """Converts a decoded VAE batch in [-1, 1] back into a list of PIL images."""
    sample = (sample.float() / 2 + 0.5).clamp(0, 1)
    arr = (sample.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def canny_edges(img_rgb: np.ndarray, low: int, high: int) -> np.ndarray:
    """Runs Canny and returns an HxWx3 uint8 map of white edges on black (ControlNet convention)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return np.stack([edges] * 3, axis=-1)


def load_doodle(path: str, resolution: int) -> torch.Tensor:
    """Loads a user doodle, auto-inverts dark-on-white sketches, and returns a [0,1] CHW tensor."""
    img = Image.open(path).convert("RGB")
    img = center_crop_square(img).resize((resolution, resolution), Image.NEAREST)
    arr = np.asarray(img, dtype=np.uint8)

    # Training hints are white lines on black. A scanned/drawn sketch is usually the
    # opposite, so invert when the image is predominantly bright.
    if arr.mean() > 127:
        arr = 255 - arr

    tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor


def make_grid(images: list[Image.Image], cols: int = 4) -> Image.Image:
    """Tiles PIL images into a single contact sheet for quick visual inspection."""
    if not images:
        raise ValueError("make_grid received an empty image list")
    cols = min(cols, len(images))
    rows = (len(images) + cols - 1) // cols
    width, height = images[0].size
    sheet = Image.new("RGB", (cols * width, rows * height), "black")
    for idx, img in enumerate(images):
        sheet.paste(img, ((idx % cols) * width, (idx // cols) * height))
    return sheet
