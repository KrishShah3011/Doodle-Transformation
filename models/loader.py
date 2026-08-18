"""Loads the pretrained Stable Diffusion pieces and freezes everything we do not train."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class SDComponents:
    """Container for the four frozen SD modules plus the training noise scheduler."""

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler

    @property
    def vae_scale_factor(self) -> int:
        """Spatial downsample ratio of the VAE (8 for SD 1.x) -- 256px -> 32x32 latents."""
        return 2 ** (len(self.vae.config.block_out_channels) - 1)


def freeze(module: torch.nn.Module) -> torch.nn.Module:
    """Puts a module in eval mode and disables its gradients so it never updates."""
    module.eval()
    module.requires_grad_(False)
    return module


def load_sd_components(
    pretrained: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    unet_dtype: torch.dtype | None = None,
) -> SDComponents:
    """Downloads/loads VAE + UNet + CLIP + scheduler, moves them to `device`, and freezes them."""
    tokenizer = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")

    # Frozen modules run in half precision to save VRAM; the trainable ControlNet stays fp32.
    text_encoder = freeze(text_encoder).to(device, dtype=dtype)
    vae = freeze(vae).to(device, dtype=dtype)
    unet = freeze(unet).to(device, dtype=unet_dtype or dtype)

    return SDComponents(tokenizer, text_encoder, vae, unet, noise_scheduler)


@torch.no_grad()
def encode_prompts(
    prompts: list[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device | str,
) -> torch.Tensor:
    """Tokenises and CLIP-encodes a list of prompts into [B, 77, 768] cross-attention context."""
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(tokens.input_ids.to(device))[0]
