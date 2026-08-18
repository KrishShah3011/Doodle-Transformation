"""Generates realistic images from a doodle + prompt using a trained ControlNet checkpoint.

Usage:
    python sample.py --config configs/base.yaml \
        --ckpt checkpoints/run1/step-020000.pt \
        --doodle examples/cat.png \
        --prompt "a photo of a tabby cat sitting on a wooden table" \
        --out outputs/cat.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import UniPCMultistepScheduler
from tqdm import tqdm

from models.controlnet import ControlNet
from models.injection import ControlledUNet
from models.loader import encode_prompts, load_sd_components
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.image import latent_to_pil, load_doodle, make_grid


@torch.no_grad()
def generate(
    components,
    controlled: ControlledUNet,
    hints: torch.Tensor,
    prompts: list[str],
    negative_prompt: str = "",
    steps: int = 30,
    guidance_scale: float = 9.0,
    control_scale: float = 1.0,
    resolution: int = 256,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    generator: torch.Generator | None = None,
):
    """Runs classifier-free-guided denoising from pure noise, steered by `hints`."""
    batch = len(prompts)
    scheduler = UniPCMultistepScheduler.from_config(components.noise_scheduler.config)
    scheduler.set_timesteps(steps, device=device)

    # Conditional and unconditional context are batched together for a single UNet pass.
    cond = encode_prompts(prompts, components.tokenizer, components.text_encoder, device)
    uncond = encode_prompts([negative_prompt] * batch, components.tokenizer,
                            components.text_encoder, device)
    context = torch.cat([uncond, cond]).to(dtype)

    latent_size = resolution // components.vae_scale_factor
    latents = torch.randn(
        (batch, components.unet.config.in_channels, latent_size, latent_size),
        generator=generator, device=device, dtype=dtype,
    ) * scheduler.init_noise_sigma

    hints = hints.to(device, dtype=dtype)
    hints_both = torch.cat([hints, hints])   # the hint applies to both CFG branches

    for timestep in tqdm(scheduler.timesteps, desc="denoise", leave=False):
        model_in = scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
        noise_pred = controlled(
            model_in, timestep, context, hints_both, control_scale=control_scale
        )
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    vae_dtype = next(components.vae.parameters()).dtype
    decoded = components.vae.decode(
        latents.to(vae_dtype) / components.vae.config.scaling_factor
    ).sample
    return latent_to_pil(decoded)


def build_model(cfg, ckpt_path: str, device: str, dtype: torch.dtype):
    """Loads frozen SD, rebuilds the ControlNet, and restores trained weights from a checkpoint."""
    components = load_sd_components(cfg.model.pretrained, device=device, dtype=dtype)
    controlnet = ControlNet.from_unet(components.unet).to(device, dtype=dtype)
    load_checkpoint(ckpt_path, controlnet, map_location=device)
    controlnet.eval().requires_grad_(False)
    return components, ControlledUNet(components.unet, controlnet).eval()


def main() -> None:
    """Parses CLI args, generates `--num` images from one doodle, and saves them."""
    parser = argparse.ArgumentParser(description="Sample from a trained ControlNet")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--doodle", required=True, help="edge map or hand-drawn sketch")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="outputs/sample.png")
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--control-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("opts", nargs="*", help="config overrides, e.g. data.resolution=512")
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    components, controlled = build_model(cfg, args.ckpt, device, dtype)

    hint = load_doodle(args.doodle, cfg.data.resolution)
    hints = hint.unsqueeze(0).repeat(args.num, 1, 1, 1)

    images = generate(
        components, controlled, hints,
        prompts=[args.prompt] * args.num,
        negative_prompt=cfg.sample.negative_prompt,
        steps=args.steps or cfg.sample.steps,
        guidance_scale=args.guidance or cfg.sample.guidance_scale,
        control_scale=args.control_scale if args.control_scale is not None else cfg.model.control_scale,
        resolution=cfg.data.resolution,
        device=device, dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_grid(images, cols=min(4, args.num)).save(out_path)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
