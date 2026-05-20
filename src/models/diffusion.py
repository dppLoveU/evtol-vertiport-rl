"""DDPM wrapper + DDIM sampler for the Stage-4B OD diffusion model.

Minimal implementation built around an epsilon-prediction U-Net (see
``unet_od.py``). The defaults (``num_train_timesteps=100``) are SMOKE
defaults; production runs will use ``num_train_timesteps=1000`` from
``configs/diffusion.yaml::diffusion``.

Schedules are stored as ``register_buffer`` tensors so ``.to(device)``
moves them along with the wrapping ``nn.Module``.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_betas(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule from Nichol & Dhariwal 2021."""
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    f = torch.cos(((x / num_steps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 1e-4, 0.999).to(torch.float32)


def _linear_betas(num_steps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, num_steps, dtype=torch.float32)


class GaussianDiffusion(nn.Module):
    """DDPM training objective + DDIM sampler.

    Parameters
    ----------
    num_train_timesteps : number of diffusion steps. SMOKE default 100,
        production 1000.
    beta_schedule : ``"cosine"`` or ``"linear"``.
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_schedule: str = "cosine",
    ) -> None:
        super().__init__()
        if num_train_timesteps < 2:
            raise ValueError(f"num_train_timesteps must be >=2, got {num_train_timesteps}")
        if beta_schedule == "cosine":
            betas = _cosine_betas(num_train_timesteps)
        elif beta_schedule == "linear":
            betas = _linear_betas(num_train_timesteps)
        else:
            raise ValueError(f"unknown beta_schedule {beta_schedule!r}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.num_train_timesteps = num_train_timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    # --- training ---

    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``q(x_t | x_0)``. Returns ``(x_t, noise)``."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_acp * x0 + sqrt_one_minus * noise, noise

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        hour: torch.Tensor,
        dow: torch.Tensor,
        is_weekend: torch.Tensor,
        t: torch.Tensor | None = None,
        cond_dropout_prob: float = 0.0,
        weight_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard MSE-on-epsilon DDPM loss.

        When ``cond_dropout_prob > 0`` each batch element independently
        has its conditional embedding dropped with that probability --
        the classifier-free-guidance training-side trick (Ho & Salimans
        2022). The model thus learns both conditional and unconditional
        noise estimates from a single set of weights. The default ``0.0``
        keeps the original Stage 4B-1/4B-2 behaviour.

        ``weight_map`` (Stage 4B-5B plumbing): optional per-pixel weight
        tensor broadcastable to ``pred``. When ``None`` (default) the
        loss is exactly ``F.mse_loss(pred, noise)`` -- bit-equal to the
        pre-PR5B-1 behaviour. When provided the loss becomes
        ``(weight_map * (pred - noise) ** 2).mean()``. The caller is
        responsible for normalising ``weight_map`` (e.g. dividing by
        ``weight_map.mean()``) so the loss magnitude stays comparable
        to the unweighted case. PR5B-1 only exposes the parameter --
        the training script still passes ``None``.
        """
        b = x0.shape[0]
        if t is None:
            t = torch.randint(
                0, self.num_train_timesteps, (b,), device=x0.device, dtype=torch.long
            )
        x_t, noise = self.add_noise(x0, t)
        cond_drop_mask: torch.Tensor | None = None
        if cond_dropout_prob > 0.0:
            cond_drop_mask = torch.rand(b, device=x0.device) < cond_dropout_prob
        pred = model(x_t, t, hour, dow, is_weekend, cond_drop_mask=cond_drop_mask)
        if weight_map is None:
            return F.mse_loss(pred, noise)
        return (weight_map * (pred - noise) ** 2).mean()

    # --- sampling ---

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        hour: torch.Tensor,
        dow: torch.Tensor,
        is_weekend: torch.Tensor,
        num_inference_steps: int = 10,
        eta: float = 0.0,
        clip_sample: bool = True,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Deterministic DDIM sampler (``eta=0``) by default.

        Sub-samples ``num_inference_steps`` indices from
        ``range(num_train_timesteps)`` and iterates them from latest to
        earliest, producing ``x_0`` predictions and stepping along
        Song et al. 2021's DDIM update.

        ``clip_sample`` (default True) clamps the predicted ``x_0`` to
        ``[-1, 1]`` at every step -- the same trick HuggingFace
        ``diffusers`` applies via ``scheduler.config.clip_sample``. Real
        OD slices live in ``[-1, 1]`` after normalization
        (``configs/diffusion.yaml::data``), so the clip is consistent
        with the data prior; without it an untrained model can blow up
        through ``1/sqrt(alpha_cumprod_T) ~ 2000`` before downstream
        ``expm1`` overflows.

        ``guidance_scale`` selects classifier-free-guidance behaviour:

          * ``1.0`` (default) -- one conditional forward per step, exact
            same code path as before CFG was added; deterministic to the
            bit given a fixed seed.
          * ``> 1.0`` -- two forwards per step, a conditional one and an
            unconditional one (``cond_drop_mask`` all True); the noise
            estimates are combined as
            ``eps_uncond + scale * (eps_cond - eps_uncond)``.
        """
        if num_inference_steps < 1 or num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps must be in [1, {self.num_train_timesteps}], "
                f"got {num_inference_steps}"
            )
        device = self.betas.device

        # Evenly spaced subset, latest first.
        ts_full = torch.linspace(
            0, self.num_train_timesteps - 1, num_inference_steps + 1, device=device
        ).round().long()
        # Pair (t_cur, t_prev) -- iterate from highest to lowest.
        t_curs = ts_full[1:].flip(0)
        t_prevs = ts_full[:-1].flip(0)

        x = torch.randn(shape, device=device)
        # Pre-build the uncond mask once; only used when guidance_scale != 1.
        uncond_mask = (
            torch.ones(shape[0], device=device, dtype=torch.bool)
            if guidance_scale != 1.0
            else None
        )
        for t_cur, t_prev in zip(t_curs, t_prevs):
            t_batch = torch.full((shape[0],), int(t_cur.item()), device=device, dtype=torch.long)
            if guidance_scale == 1.0:
                # No CFG: single conditional forward (bit-equal to pre-CFG path).
                eps_pred = model(x, t_batch, hour, dow, is_weekend)
            else:
                eps_cond = model(x, t_batch, hour, dow, is_weekend)
                eps_uncond = model(x, t_batch, hour, dow, is_weekend, cond_drop_mask=uncond_mask)
                eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            acp_t = self.alphas_cumprod[t_cur]
            acp_prev = self.alphas_cumprod[t_prev] if int(t_prev.item()) >= 0 else torch.tensor(
                1.0, device=device
            )

            # Predicted x_0 from x_t and eps.
            x0_pred = (x - torch.sqrt(1.0 - acp_t) * eps_pred) / torch.sqrt(acp_t)
            if clip_sample:
                x0_pred = x0_pred.clamp(-1.0, 1.0)
            # Optional stochasticity (eta=0 -> deterministic).
            sigma = (
                eta
                * torch.sqrt((1.0 - acp_prev) / (1.0 - acp_t))
                * torch.sqrt(1.0 - acp_t / acp_prev)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - acp_prev - sigma**2, min=0.0)) * eps_pred
            x = torch.sqrt(acp_prev) * x0_pred + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x

    # --- diagnostics ---

    def summary(self) -> dict[str, Any]:
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "beta_min": float(self.betas.min().item()),
            "beta_max": float(self.betas.max().item()),
            "alpha_cumprod_T": float(self.alphas_cumprod[-1].item()),
        }
