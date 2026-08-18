"""Scores a checkpoint on two axes: does it follow the edges, and does it follow the prompt?

* Edge F1  -- re-run Canny on the *generated* image and compare against the input hint
              (with a small dilation tolerance). Measures control fidelity.
* CLIP     -- cosine similarity between the generated image and its caption.
              Measures prompt adherence.

Usage:
    python -m eval.evaluate --config configs/base.yaml \
        --ckpt checkpoints/run1/step-020000.pt --num 200
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from data.dataset import CocoEdgeDataset, collate_fn
from sample import build_model, generate
from utils.config import load_config


def edge_f1(pred_rgb: np.ndarray, hint: np.ndarray, tolerance: int = 2) -> float:
    """Compares Canny edges of the generated image to the conditioning map, allowing N-px slack."""
    gray = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2GRAY)
    pred_edges = cv2.Canny(gray, 100, 200) > 0
    true_edges = hint[..., 0] > 127

    kernel = np.ones((tolerance * 2 + 1,) * 2, np.uint8)
    pred_dil = cv2.dilate(pred_edges.astype(np.uint8), kernel) > 0
    true_dil = cv2.dilate(true_edges.astype(np.uint8), kernel) > 0

    # A predicted edge counts as correct if it lands near ANY ground-truth edge, and vice versa.
    precision = (pred_edges & true_dil).sum() / max(pred_edges.sum(), 1)
    recall = (true_edges & pred_dil).sum() / max(true_edges.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-8))


@torch.no_grad()
def clip_score(images, prompts, model, processor, device) -> list[float]:
    """Returns per-image CLIP cosine similarity (x100) between the image and its caption."""
    inputs = processor(text=prompts, images=images, return_tensors="pt",
                       padding=True, truncation=True).to(device)
    out = model(**inputs)
    img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return (100 * (img_emb * txt_emb).sum(dim=-1)).cpu().tolist()


def main() -> None:
    """Generates images for a held-out slice and prints mean edge-F1 and CLIP score."""
    parser = argparse.ArgumentParser(description="Evaluate a ControlNet checkpoint")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("opts", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    components, controlled = build_model(cfg, args.ckpt, device, dtype)

    dataset = CocoEdgeDataset(
        cfg.data.processed_dir, resolution=cfg.data.resolution,
        max_samples=args.num, caption_dropout=0.0, hflip=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch, collate_fn=collate_fn)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    f1_scores, clip_scores = [], []
    for batch in tqdm(loader, desc="eval"):
        images = generate(
            components, controlled, batch["hint"], batch["caption"],
            negative_prompt=cfg.sample.negative_prompt,
            steps=cfg.sample.steps, guidance_scale=cfg.sample.guidance_scale,
            control_scale=cfg.model.control_scale, resolution=cfg.data.resolution,
            device=device, dtype=dtype,
        )
        hints = (batch["hint"].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        f1_scores += [edge_f1(np.asarray(img), h) for img, h in zip(images, hints)]
        clip_scores += clip_score(images, batch["caption"], clip_model, clip_proc, device)

    print(f"\nsamples      : {len(f1_scores)}")
    print(f"edge F1      : {np.mean(f1_scores):.4f}  (higher = follows the doodle more closely)")
    print(f"CLIP score   : {np.mean(clip_scores):.2f}  (higher = matches the prompt better)")


if __name__ == "__main__":
    main()
