"""Exponential moving average of a model's parameters.

A lightweight EMA helper for Stage-4B+ diffusion training; mirrors the
standard DDPM/EMA pattern used by `lucidrains/denoising-diffusion-pytorch`
and HuggingFace `diffusers`.

Public API:
  * ``update(model)``: blend current params into the EMA buffers.
  * ``copy_to(model)``: install EMA buffers into ``model`` (for eval / sample).
  * ``store(model)`` / ``restore(model)``: stash and restore ``model``'s
    current params, so you can swap EMA in, evaluate, and swap back.
  * ``state_dict`` / ``load_state_dict``: inherited from ``nn.Module``;
    the EMA buffers ride along with checkpoints.

The EMA shadow weights are kept as ``register_buffer`` tensors so
``.to(device)`` moves them with the rest of the wrapping ``nn.Module``.
Parameter names containing ``.`` (e.g. ``input_conv.weight``) cannot be
buffer names; we map ``foo.bar`` -> ``foo_bar`` and remember the
mapping in ``_name_map``. The mapping is rebuilt from the model on
construction, so ``load_state_dict`` does not need to know about it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EMA(nn.Module):
    """Exponential moving average of ``model``'s trainable parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self._name_map: dict[str, str] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            buf_name = name.replace(".", "_")
            self._name_map[name] = buf_name
            self.register_buffer(buf_name, param.detach().clone(), persistent=True)
        # store/restore snapshot (RAM-only; not part of state_dict).
        self._stored: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """``shadow = decay * shadow + (1 - decay) * model.param``."""
        d = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            buf = getattr(self, self._name_map[name])
            buf.mul_(d).add_(param.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy the EMA buffers into ``model``'s params in-place."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(getattr(self, self._name_map[name]))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Stash a snapshot of ``model``'s current params for later restore."""
        self._stored = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore params from the last ``store`` snapshot and clear it."""
        if not self._stored:
            raise RuntimeError("EMA.restore called before EMA.store")
        for name, param in model.named_parameters():
            param.data.copy_(self._stored[name])
        self._stored = {}
