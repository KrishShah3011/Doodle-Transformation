"""Trains the ControlNet against frozen Stable Diffusion on COCO edge-map triplets.

Usage:
    python train.py --config configs/base.yaml
    python train.py --config configs/base.yaml --resume auto
    python train.py --config configs/base.yaml train.batch_size=8 data.max_samples=5000
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CocoEdgeDataset, collate_fn
from models.controlnet import ControlNet
from models.injection import ControlledUNet, assert_shapes_match
from models.loader import encode_prompts, load_sd_components
from sample import generate
from utils.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from utils.config import load_config, save_config
from utils.image import make_grid


def set_seed(seed: int) -> None:
    """Seeds python/numpy/torch so a run is reproducible from its config alone."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(cfg, params):
    """Builds the optimiser; adafactor/adamw8bit cut optimiser-state VRAM on small GPUs."""
    name = str(cfg.train.get("optimizer", "adamw")).lower()

    if name == "adafactor":
        # Factored second moments: ~2.9 GB of AdamW state becomes a few MB. No extra install.
        from transformers.optimization import Adafactor

        return Adafactor(
            params,
            lr=cfg.train.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=cfg.train.weight_decay,
        )

    if name == "adamw8bit":
        # Keeps Adam's behaviour but quantises the moments to 8 bits (~2.2 GB saved).
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            params,
            lr=cfg.train.learning_rate,
            betas=tuple(cfg.train.adam_betas),
            eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )

    return torch.optim.AdamW(
        params,
        lr=cfg.train.learning_rate,
        betas=tuple(cfg.train.adam_betas),
        eps=cfg.train.adam_eps,
        weight_decay=cfg.train.weight_decay,
    )


def build_lr_lambda(warmup_steps: int):
    """Linear warmup then constant LR -- ControlNet does not benefit much from decay."""
    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / max(1, warmup_steps))
    return lr_lambda


@torch.no_grad()
def run_validation(cfg, components, controlled, val_loader, out_dir: Path, step: int, ema=None) -> None:
    """Generates a fixed preview grid (hint | generated) so you can eyeball control fidelity."""
    controlled.controlnet.eval()
    batch = next(iter(val_loader))
    device = next(controlled.controlnet.parameters()).device

    context = ema.average_parameters(controlled.controlnet) if ema is not None else nullcontext()
    with context:
        images = generate(
            components, controlled,
            hints=batch["hint"],
            prompts=[c if c else "a photo" for c in batch["caption"]],
            negative_prompt=cfg.sample.negative_prompt,
            steps=cfg.logging.val_steps,
            guidance_scale=cfg.logging.val_guidance,
            control_scale=cfg.model.control_scale,
            resolution=cfg.data.resolution,
            device=device,
            dtype=torch.float32,   # ControlNet holds fp32 master weights during training
        )

    from utils.image import latent_to_pil  # local import keeps the module graph shallow
    hint_pils = latent_to_pil(batch["hint"] * 2 - 1)
    grid = make_grid(hint_pils + images, cols=len(images))

    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    grid.save(val_dir / f"step-{step:06d}.png")
    controlled.controlnet.train()


def main() -> None:
    """Wires together data, models, optimiser and the step loop, with resume support."""
    parser = argparse.ArgumentParser(description="Train a ControlNet on COCO edge maps")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", default=None,
                        help="'auto' for newest checkpoint in output_dir, or an explicit path")
    parser.add_argument("opts", nargs="*", help="dotted overrides, e.g. train.batch_size=8")
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}[
        cfg.train.mixed_precision
    ]
    frozen_dtype = torch.float16 if cfg.train.mixed_precision == "fp16" else (
        torch.bfloat16 if cfg.train.mixed_precision == "bf16" else torch.float32
    )

    out_dir = Path(cfg.logging.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    # ------------------------------------------------------------ models
    components = load_sd_components(
        cfg.model.pretrained,
        device=device,
        dtype=frozen_dtype,
        # Set train.unet_fp32=true if fp16 ever produces NaN losses.
        unet_dtype=torch.float32 if cfg.train.get("unet_fp32", False) else None,
    )
    controlnet = ControlNet.from_unet(components.unet).to(device, dtype=torch.float32)
    if cfg.train.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
    controlled = ControlledUNet(components.unet, controlnet)

    # Initialize EMA weights tracker
    from utils.ema import EMAModel
    ema_decay = cfg.train.get("ema_decay", 0.999)
    ema = EMAModel(controlnet, decay=ema_decay)

    print(f"[info] ControlNet trainable params: {controlnet.num_trainable_params/1e6:.1f}M")
    assert_shapes_match(components.unet, controlnet, cfg.data.resolution, device=device)

    # ------------------------------------------------------------ data
    train_ds = CocoEdgeDataset(
        cfg.data.processed_dir,
        resolution=cfg.data.resolution,
        max_samples=cfg.data.max_samples,
        canny_low_range=cfg.data.canny_low_range,
        canny_high_range=cfg.data.canny_high_range,
        caption_dropout=cfg.data.caption_dropout,
        hflip=cfg.data.hflip,
        skip_first=cfg.data.val_samples,      # keep the first N images out of training
    )
    val_ds = CocoEdgeDataset(
        cfg.data.processed_dir,
        resolution=cfg.data.resolution,
        max_samples=cfg.data.val_samples,
        caption_dropout=0.0,                  # validation always uses real captions
        hflip=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True, persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.val_samples, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    print(f"[info] train samples: {len(train_ds)} | val samples: {len(val_ds)}")

    # ------------------------------------------------------------ optimiser
    optimizer = build_optimizer(cfg, controlnet.parameters())
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(cfg.train.lr_warmup_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.mixed_precision == "fp16")

    # ------------------------------------------------------------ resume
    global_step, start_epoch = 0, 0
    resume_path = latest_checkpoint(out_dir) if args.resume == "auto" else args.resume
    if resume_path:
        global_step, start_epoch = load_checkpoint(
            resume_path, controlnet, optimizer, lr_scheduler, scaler, map_location=device, ema=ema
        )
        print(f"[resume] {resume_path} @ step {global_step}, epoch {start_epoch}")

    steps_per_epoch = math.ceil(len(train_loader) / cfg.train.grad_accum)
    max_steps = cfg.train.max_steps if cfg.train.max_steps > 0 else (
        cfg.train.max_epochs * steps_per_epoch
    )
    print(f"[info] {steps_per_epoch} optimiser steps/epoch, target {max_steps} steps")

    # ------------------------------------------------------------ train loop
    controlnet.train()
    scaling = components.vae.config.scaling_factor
    alphas_cumprod = components.noise_scheduler.alphas_cumprod.to(device)
    running_loss, last_log = 0.0, time.time()
    progress = tqdm(total=max_steps, initial=global_step, desc="train")

    epoch = start_epoch
    while global_step < max_steps:
        for micro_step, batch in enumerate(train_loader):
            # --- encode image -> latent and caption -> context (both frozen) ---
            with torch.no_grad():
                pixels = batch["pixel_values"].to(device, dtype=frozen_dtype, non_blocking=True)
                latents = components.vae.encode(pixels).latent_dist.sample() * scaling
                latents = latents.float()
                context = encode_prompts(
                    batch["caption"], components.tokenizer, components.text_encoder, device
                ).float()

            hint = batch["hint"].to(device, dtype=torch.float32, non_blocking=True)

            # --- forward diffusion: add noise at a random timestep -------------
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, components.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device, dtype=torch.long,
            )
            noisy = components.noise_scheduler.add_noise(latents, noise, timesteps)

            # --- predict the noise and score it against the truth --------------
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                pred = controlled(noisy, timesteps, context, hint, control_scale=1.0)
                if components.noise_scheduler.config.prediction_type == "v_prediction":
                    target = components.noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
                weight = torch.clamp(snr, max=5.0) / snr           # gamma=5 is the usual choice
                per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
                loss = (per_sample * weight).mean() / cfg.train.grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * cfg.train.grad_accum

            # --- optimiser step only once every `grad_accum` micro-batches -----
            if (micro_step + 1) % cfg.train.grad_accum != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), cfg.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            # Update EMA shadow weights
            ema.step(controlnet)

            global_step += 1
            progress.update(1)

            # --- logging / checkpointing / validation --------------------------
            if global_step % cfg.logging.log_every == 0:
                avg = running_loss / (cfg.logging.log_every * cfg.train.grad_accum)
                imgs_per_sec = (
                    cfg.logging.log_every * cfg.train.batch_size * cfg.train.grad_accum
                ) / (time.time() - last_log)
                progress.set_postfix(loss=f"{avg:.4f}", ips=f"{imgs_per_sec:.1f}")
                running_loss, last_log = 0.0, time.time()

            if global_step % cfg.logging.ckpt_every == 0:
                path = save_checkpoint(
                    out_dir, controlnet, optimizer, lr_scheduler, scaler,
                    global_step, epoch, keep_last=cfg.logging.keep_last, ema=ema
                )
                progress.write(f"[ckpt] {path}")

            if global_step % cfg.logging.val_every == 0:
                run_validation(cfg, components, controlled, val_loader, out_dir, global_step, ema=ema)

            if global_step >= max_steps:
                break
        epoch += 1

    save_checkpoint(out_dir, controlnet, optimizer, lr_scheduler, scaler,
                    global_step, epoch, keep_last=cfg.logging.keep_last, ema=ema)
    progress.close()
    print(f"[done] finished at step {global_step}")


if __name__ == "__main__":
    main()