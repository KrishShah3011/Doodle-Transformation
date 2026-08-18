"""Runs one forward+backward pass on random data to prove the wiring works before a long run.

Usage:
    python -m scripts.smoke_test --config configs/base.yaml
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from models.controlnet import ControlNet
from models.injection import ControlledUNet, assert_shapes_match
from models.loader import encode_prompts, load_sd_components
from utils.config import load_config


def main() -> None:
    """Builds the model, checks residual shapes, and confirms gradients reach the ControlNet."""
    parser = argparse.ArgumentParser(description="ControlNet wiring smoke test")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("opts", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    res = cfg.data.resolution

    print("[1/5] loading frozen Stable Diffusion ...")
    components = load_sd_components(cfg.model.pretrained, device=device, dtype=dtype)

    print("[2/5] building ControlNet from the UNet encoder ...")
    controlnet = ControlNet.from_unet(components.unet).to(device, dtype=torch.float32)
    print(f"      trainable params: {controlnet.num_trainable_params/1e6:.1f}M")

    print("[3/5] checking injection points ...")
    assert_shapes_match(components.unet, controlnet, res, device=device)

    print("[4/5] running one forward pass ...")
    controlled = ControlledUNet(components.unet, controlnet)
    latents = torch.randn(2, 4, res // 8, res // 8, device=device)
    hint = torch.rand(2, 3, res, res, device=device)
    context = encode_prompts(["a cat", ""], components.tokenizer,
                             components.text_encoder, device).float()
    timesteps = torch.tensor([10, 500], device=device)

    pred = controlled(latents, timesteps, context, hint)
    assert pred.shape == latents.shape, f"shape mismatch: {pred.shape} vs {latents.shape}"
    print(f"      output {tuple(pred.shape)} at {res}px -> {res//8}x{res//8} latents")

    print("[5/5] running one backward pass ...")
    F.mse_loss(pred.float(), torch.randn_like(latents)).backward()
    grads = [p.grad for p in controlnet.parameters() if p.grad is not None]
    assert grads, "no gradients reached the ControlNet"
    assert all(p.grad is None for p in components.unet.parameters()), "the UNet is not frozen!"
    print(f"      {len(grads)} ControlNet tensors received gradients; UNet untouched")

    print("\n[ok] everything is wired correctly -- safe to start training.")


if __name__ == "__main__":
    main()
