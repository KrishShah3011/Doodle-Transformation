from __future__ import annotations

from contextlib import contextmanager
import torch

class EMAModel:
    """Maintains an exponential moving average (EMA) of trainable model parameters on CPU to save GPU VRAM."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        # Keep shadow copy of trainable weights on CPU to save VRAM
        self.shadow_params = {
            name: p.clone().detach().cpu()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        
    def step(self, model: torch.nn.Module) -> None:
        """Updates shadow parameters on CPU with the new model weights after an optimizer step."""
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    # Copy the current parameter to CPU to perform the update
                    current_cpu = p.detach().to("cpu", non_blocking=True)
                    self.shadow_params[name].mul_(self.decay).add_(current_cpu, alpha=1.0 - self.decay)
                    
    def copy_to(self, model: torch.nn.Module) -> None:
        """Copies the EMA weights from CPU into the active GPU model (for evaluation or saving)."""
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(self.shadow_params[name].to(p.device, non_blocking=True))
                    
    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        """Temporarily applies EMA weights inside a context block (validation)."""
        # Save a copy of the active weights on CPU to avoid double-allocation on GPU
        orig_params = {
            name: p.clone().detach().cpu()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        try:
            self.copy_to(model)
            yield
        finally:
            # Restore original training weights back to GPU
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        p.copy_(orig_params[name].to(p.device, non_blocking=True))
                        
    def state_dict(self) -> dict:
        """Returns the EMA state ready to be serialized to disk."""
        return {
            "decay": self.decay,
            "shadow_params": self.shadow_params
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        """Restores the shadow parameters from a saved state dict (always kept on CPU)."""
        self.decay = state_dict.get("decay", self.decay)
        shadow_params = state_dict["shadow_params"]
        for name, p in shadow_params.items():
            if name in self.shadow_params:
                self.shadow_params[name].copy_(p.cpu())
