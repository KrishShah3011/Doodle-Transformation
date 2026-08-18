from __future__ import annotations

from contextlib import contextmanager
import torch

class EMAModel:
    """Maintains an exponential moving average (EMA) of trainable model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        # Keep shadow copy of trainable weights on the same device
        self.shadow_params = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        
    def step(self, model: torch.nn.Module) -> None:
        """Updates shadow parameters with the new model weights after an optimizer step."""
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow_params[name].mul_(self.decay).add_(p, alpha=1.0 - self.decay)
                    
    def copy_to(self, model: torch.nn.Module) -> None:
        """Copies the EMA weights into the active model (for evaluation or saving)."""
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(self.shadow_params[name])
                    
    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        """Temporarily applies EMA weights inside a context block (e.g. during validation)."""
        # Save a copy of the active weights
        orig_params = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        try:
            self.copy_to(model)
            yield
        finally:
            # Restore original training weights
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        p.copy_(orig_params[name])
                        
    def state_dict(self) -> dict:
        """Returns the EMA state ready to be serialized to disk."""
        # Convert tensors to CPU to save space/memory in checkpoints
        return {
            "decay": self.decay,
            "shadow_params": {name: p.cpu() for name, p in self.shadow_params.items()}
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        """Restores the shadow parameters from a saved state dict."""
        self.decay = state_dict.get("decay", self.decay)
        shadow_params = state_dict["shadow_params"]
        for name, p in shadow_params.items():
            if name in self.shadow_params:
                self.shadow_params[name].copy_(p.to(self.shadow_params[name].device))
