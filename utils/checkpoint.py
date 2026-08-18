"""Saves/restores *exact* training state (weights, optimiser, scaler, step, RNG) for stop-resume."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(
    out_dir: str | Path,
    controlnet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    global_step: int,
    epoch: int,
    keep_last: int = 3,
) -> Path:
    """Writes one `step-XXXXXX.pt` bundle and prunes all but the newest `keep_last`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"step-{global_step:06d}.pt"

    torch.save(
        {
            "controlnet": controlnet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "global_step": global_step,
            "epoch": epoch,
            # RNG states make a resumed run bit-identical to an uninterrupted one.
            "rng_python": torch.random.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
        },
        ckpt_path,
    )

    _prune_old(out_dir, keep_last)
    return ckpt_path


def _prune_old(out_dir: Path, keep_last: int) -> None:
    """Deletes the oldest checkpoints so a long run does not fill the disk."""
    ckpts = sorted(out_dir.glob("step-*.pt"), key=_step_of)
    for old in ckpts[:-keep_last] if keep_last > 0 else []:
        old.unlink(missing_ok=True)


def _step_of(path: Path) -> int:
    """Extracts the integer step number encoded in a checkpoint filename."""
    match = re.search(r"step-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    """Returns the highest-step checkpoint in a directory, or None if there are none."""
    ckpts = sorted(Path(out_dir).glob("step-*.pt"), key=_step_of)
    return ckpts[-1] if ckpts else None


def load_checkpoint(
    path: str | Path,
    controlnet: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    map_location: str = "cpu",
) -> tuple[int, int]:
    """Restores a checkpoint in place and returns `(global_step, epoch)` to continue from."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    controlnet.load_state_dict(ckpt["controlnet"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("lr_scheduler"):
        scheduler.load_state_dict(ckpt["lr_scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])

    if ckpt.get("rng_python") is not None:
        torch.random.set_rng_state(ckpt["rng_python"].cpu().to(torch.uint8))
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        try:
            states = [s.cpu().to(torch.uint8) for s in ckpt["rng_cuda"]]
            torch.cuda.set_rng_state_all(states)
        except (RuntimeError, ValueError, TypeError):
            pass  # GPU count changed between runs; harmless, just loses exact reproducibility.
    if ckpt.get("rng_numpy") is not None:
        np.random.set_state(ckpt["rng_numpy"])

    return int(ckpt.get("global_step", 0)), int(ckpt.get("epoch", 0))
