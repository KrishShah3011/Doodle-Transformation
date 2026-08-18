"""The trainable ControlNet: a weight-initialised copy of the SD encoder + zero convolutions.

Design notes
------------
* The down/mid blocks are deep-copied from the frozen UNet, so training starts from
  Stable Diffusion's own features rather than from noise.
* Every output passes through a *zero convolution* (1x1 conv with weight AND bias set to 0).
  At step 0 the branch contributes exactly nothing, so the frozen UNet is left untouched
  and cannot be damaged by random gradients. Zero convs are the whole trick -- do not
  "fix" them by using normal init.
* Nothing here is resolution-specific: the module is fully convolutional, so 256px
  (32x32 latents) and 512px (64x64 latents) both work with identical weights.
"""

from __future__ import annotations

import copy
import functools

import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers import UNet2DConditionModel


def zero_module(module: nn.Module) -> nn.Module:
    """Zeroes every parameter of a module so it initially outputs exactly zero."""
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


class HintEncoder(nn.Module):
    """Compresses the 3-channel edge map 8x spatially so it can be added to the latent stream."""

    def __init__(self, in_channels: int = 3, out_channels: int = 320):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2), nn.SiLU(),   # /2
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2), nn.SiLU(),   # /4
            nn.Conv2d(96, 96, 3, padding=1), nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2), nn.SiLU(),  # /8 -> matches VAE stride
        )
        # Zero conv again: the hint contributes nothing until the model learns to use it.
        self.out = zero_module(nn.Conv2d(256, out_channels, 3, padding=1))

    def forward(self, hint: torch.Tensor) -> torch.Tensor:
        """Maps [B,3,H,W] edges to [B,320,H/8,W/8] features aligned with `unet.conv_in` output."""
        return self.out(self.body(hint))


class ControlNet(nn.Module):
    """Parallel trainable encoder that emits 12 down residuals + 1 mid residual for the UNet."""

    def __init__(
        self,
        conv_in: nn.Module,
        time_proj: nn.Module,
        time_embedding: nn.Module,
        down_blocks: nn.ModuleList,
        mid_block: nn.Module,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        hint_channels: int = 3,
    ):
        super().__init__()
        self.conv_in = conv_in
        self.time_proj = time_proj
        self.time_embedding = time_embedding
        self.down_blocks = down_blocks
        self.mid_block = mid_block

        self.hint_encoder = HintEncoder(hint_channels, block_out_channels[0])

        # One zero conv per skip connection the UNet decoder consumes, in the same order.
        zero_convs: list[nn.Module] = [
            zero_module(nn.Conv2d(block_out_channels[0], block_out_channels[0], 1))  # conv_in output
        ]
        for i, channels in enumerate(block_out_channels):
            for _ in range(layers_per_block):
                zero_convs.append(zero_module(nn.Conv2d(channels, channels, 1)))     # each resnet
            if i < len(block_out_channels) - 1:
                zero_convs.append(zero_module(nn.Conv2d(channels, channels, 1)))     # downsampler
        self.zero_convs = nn.ModuleList(zero_convs)
        self.mid_zero_conv = zero_module(nn.Conv2d(block_out_channels[-1], block_out_channels[-1], 1))

    # ------------------------------------------------------------------ build
    @classmethod
    def from_unet(cls, unet: UNet2DConditionModel) -> "ControlNet":
        """Clones the frozen UNet's encoder half into a fresh, fully trainable ControlNet."""
        net = cls(
            conv_in=copy.deepcopy(unet.conv_in),
            time_proj=copy.deepcopy(unet.time_proj),
            time_embedding=copy.deepcopy(unet.time_embedding),
            down_blocks=copy.deepcopy(unet.down_blocks),
            mid_block=copy.deepcopy(unet.mid_block),
            block_out_channels=tuple(unet.config.block_out_channels),
            layers_per_block=unet.config.layers_per_block,
        )
        net.requires_grad_(True)
        return net

    def enable_gradient_checkpointing(self) -> None:
        """Trades ~25% speed for ~40% less activation memory on the copied UNet blocks."""
        # diffusers blocks call `self._gradient_checkpointing_func(...)`, so setting the
        # boolean alone is not enough -- the function must be installed alongside it.
        checkpoint_func = functools.partial(
            torch.utils.checkpoint.checkpoint, use_reentrant=False
        )
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                module._gradient_checkpointing_func = checkpoint_func

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        hint: torch.Tensor,
        conditioning_scale: float = 1.0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Runs the control branch and returns (12 down residuals, 1 mid residual)."""
        # --- timestep embedding, identical maths to the frozen UNet ---------
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=sample.device)
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.expand(sample.shape[0])
        emb = self.time_embedding(self.time_proj(timesteps).to(dtype=sample.dtype))

        # --- inject the edge map at the very first latent-resolution layer ---
        sample = self.conv_in(sample) + self.hint_encoder(hint)

        # --- replicate the SD encoder, collecting every skip connection ------
        down_residuals: tuple[torch.Tensor, ...] = (sample,)
        for block in self.down_blocks:
            if getattr(block, "has_cross_attention", False):
                sample, residuals = block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, residuals = block(hidden_states=sample, temb=emb)
            down_residuals += residuals

        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # --- zero convs turn raw features into safe, learnable corrections ---
        down_out = [
            zc(res) * conditioning_scale
            for zc, res in zip(self.zero_convs, down_residuals)
        ]
        mid_out = self.mid_zero_conv(sample) * conditioning_scale
        return down_out, mid_out

    @property
    def num_trainable_params(self) -> int:
        """Parameter count, handy for the sanity line printed at the start of training."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)