"""Fuses ControlNet outputs into the frozen UNet by adding them to its skip connections.

The frozen encoder still runs -- ControlNet is a *parallel* branch, not a replacement.
Its 12 down residuals are added to the 12 skip tensors the decoder consumes, and its
mid residual is added to the mid-block output. Gradients reach the ControlNet only
through those addition points, which is why the frozen half can stay under `no_grad`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ControlledUNet(nn.Module):
    """Couples the frozen UNet with the trainable ControlNet into one callable module."""

    def __init__(self, unet: nn.Module, controlnet: nn.Module):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        hint: torch.Tensor,
        control_scale: float = 1.0,
    ) -> torch.Tensor:
        """Predicts noise for `latents`, steered by `hint` at strength `control_scale`."""
        down_res, mid_res = self.controlnet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            hint=hint,
            conditioning_scale=control_scale,
        )

        # diffusers adds these to the corresponding skip/mid tensors internally.
        unet_dtype = next(self.unet.parameters()).dtype
        return self.unet(
            latents.to(unet_dtype),
            timestep,
            encoder_hidden_states=encoder_hidden_states.to(unet_dtype),
            down_block_additional_residuals=[r.to(unet_dtype) for r in down_res],
            mid_block_additional_residual=mid_res.to(unet_dtype),
        ).sample


def assert_shapes_match(unet, controlnet, resolution: int, device="cpu") -> None:
    """One-off sanity check that ControlNet residuals line up with the UNet injection points."""
    latent = torch.randn(1, unet.config.in_channels, resolution // 8, resolution // 8, device=device)
    hint = torch.zeros(1, 3, resolution, resolution, device=device)
    ctx = torch.zeros(1, 77, unet.config.cross_attention_dim, device=device)

    with torch.no_grad():
        down_res, mid_res = controlnet(latent, torch.tensor([0], device=device), ctx, hint)

    assert len(down_res) == 12, f"expected 12 down residuals, got {len(down_res)}"
    assert all(torch.count_nonzero(r) == 0 for r in down_res), "zero convs are not zero-initialised!"
    assert torch.count_nonzero(mid_res) == 0, "mid zero conv is not zero-initialised!"
    print(f"[ok] {len(down_res)} down residuals + 1 mid residual, all zero at init")
