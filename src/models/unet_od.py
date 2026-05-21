"""Tiny U-Net for OD-slice diffusion (Stage 4B-1 smoke).

Designed to exercise the full diffusion pipeline end-to-end on the
``[1, 544, 544]`` padded OD slices, NOT to produce useful samples. The
defaults (``base_channels=16``, ``channel_mults=(1, 2, 4)``) come from
``configs/diffusion.yaml::model`` (the SMOKE block, not production).

Architecture (depth-3, two down/up steps):
  in -> Conv3x3 -> [ResBlock]
         |-> Conv stride 2 -> [ResBlock]
                 |-> Conv stride 2 -> [ResBlock(bottom)]
                 |<- ConvT stride 2 <- [ResBlock(skip)]
         |<- ConvT stride 2 <- [ResBlock(skip)]
  -> GroupNorm + Conv3x3 -> out

Conditioning:
  * timestep ``t`` -> sinusoidal -> 2-layer MLP -> ``time_emb_dim`` vec,
  * ``(hour, day_of_week, is_weekend)`` -> small MLP -> ``cond_emb_dim``,
  * fused via a Linear into a single embedding that is broadcast-added to
    each ResBlock's feature map (FiLM-style bias only, no scale).

Tested in ``tests/test_diffusion.py`` on a 32x32 tiny tensor; the real
544x544 forward is exercised in ``experiments/run_stage4_model_smoke.py``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timesteps.

    Parameters
    ----------
    t : ``[B]`` long tensor of timesteps.
    dim : embedding dimension; must be even.

    Returns
    -------
    ``[B, dim]`` float tensor on the same device as ``t``.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionEmbedding(nn.Module):
    """Encode ``(hour, day_of_week, is_weekend)`` into a ``cond_dim`` vec."""

    def __init__(self, cond_dim: int = 128, dow_dim: int = 8) -> None:
        super().__init__()
        self.dow_embed = nn.Embedding(7, dow_dim)
        in_dim = 2 + dow_dim + 1  # hour sin/cos + dow embed + is_weekend
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(
        self,
        hour: torch.Tensor,
        dow: torch.Tensor,
        is_weekend: torch.Tensor,
    ) -> torch.Tensor:
        # All inputs: ``[B]`` long tensors.
        angle = 2.0 * math.pi * hour.float() / 24.0
        h = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        d = self.dow_embed(dow)
        w = is_weekend.float().unsqueeze(-1)
        return self.mlp(torch.cat([h, d, w], dim=-1))


class ResBlock(nn.Module):
    """GroupNorm + Conv3x3 + SiLU, twice, with a fused emb bias injection."""

    def __init__(self, in_c: int, out_c: int, emb_dim: int) -> None:
        super().__init__()
        groups_in = min(8, in_c) if in_c % min(8, in_c) == 0 else 1
        groups_out = min(8, out_c) if out_c % min(8, out_c) == 0 else 1
        self.norm1 = nn.GroupNorm(groups_in, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_c)
        if in_c != out_c:
            self.skip: nn.Module = nn.Conv2d(in_c, out_c, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Broadcast-add emb along channel dim (FiLM bias).
        h = h + self.emb_proj(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNetOD(nn.Module):
    """Tiny conditional U-Net for OD-slice diffusion.

    Parameters
    ----------
    in_channels : input channels (= ``window`` in
        ``configs/diffusion.yaml::data.window``; default 1).
    base_channels : channel count at the input resolution.
    channel_mults : per-level channel multipliers; depth = len - 1 down
        steps. Default ``(1, 2, 4)`` -> two downsamples, three resolutions.
    time_emb_dim : sinusoidal timestep embedding dim (must be even).
    cond_emb_dim : ``(hour, dow, is_weekend)`` embedding dim.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        time_emb_dim: int = 128,
        cond_emb_dim: int = 128,
    ) -> None:
        super().__init__()
        if len(channel_mults) < 2:
            raise ValueError(f"channel_mults must have >=2 levels, got {channel_mults}")
        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim
        channels = [base_channels * m for m in channel_mults]

        # --- embeddings ---
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.cond_mlp = ConditionEmbedding(cond_emb_dim)
        emb_dim = time_emb_dim  # fused embedding size
        self.emb_fuse = nn.Linear(time_emb_dim + cond_emb_dim, emb_dim)

        # --- encoder ---
        self.input_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.enc_blocks.append(ResBlock(channels[i], channels[i], emb_dim))
            self.downs.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1)
            )

        # --- bottom ---
        self.bottom = ResBlock(channels[-1], channels[-1], emb_dim)

        # --- decoder ---
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.ups.append(
                nn.ConvTranspose2d(
                    channels[i], channels[i - 1], kernel_size=4, stride=2, padding=1
                )
            )
            self.dec_blocks.append(
                ResBlock(channels[i - 1] * 2, channels[i - 1], emb_dim)
            )

        # --- output ---
        out_groups = min(8, channels[0]) if channels[0] % min(8, channels[0]) == 0 else 1
        self.out_norm = nn.GroupNorm(out_groups, channels[0])
        self.out_conv = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)

    def _fuse_embeddings(
        self,
        t: torch.Tensor,
        hour: torch.Tensor,
        dow: torch.Tensor,
        is_weekend: torch.Tensor,
        cond_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        te = self.time_mlp(sinusoidal_embedding(t, self.time_emb_dim))
        ce = self.cond_mlp(hour, dow, is_weekend)
        if cond_drop_mask is not None:
            # Zero the conditional embedding for samples where the mask is
            # True. The fused embedding for those samples is therefore
            # ``emb_fuse([te, 0])`` -- the model's "unconditional" branch.
            keep = (~cond_drop_mask).to(ce.dtype).unsqueeze(-1)
            ce = ce * keep
        return self.emb_fuse(torch.cat([te, ce], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        hour: torch.Tensor,
        dow: torch.Tensor,
        is_weekend: torch.Tensor,
        cond_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise epsilon for the diffusion training objective.

        Shapes:
          * ``x``: ``[B, in_channels, H, W]`` (e.g. ``[B, 1, 544, 544]``)
          * ``t``, ``hour``, ``dow``, ``is_weekend``: ``[B]`` long tensors.
          * ``cond_drop_mask``: optional ``[B]`` bool tensor. ``True`` means
            "drop the conditional embedding for this sample" -- the
            classifier-free-guidance unconditional branch. ``None`` (the
            default) keeps the original Stage 4B-1 / 4B-2 behaviour
            unchanged.
          * returns: same shape as ``x``.
        """
        emb = self._fuse_embeddings(t, hour, dow, is_weekend, cond_drop_mask)

        h = self.input_conv(x)
        skips: list[torch.Tensor] = []
        for block, down in zip(self.enc_blocks, self.downs):
            h = block(h, emb)
            skips.append(h)
            h = down(h)
        h = self.bottom(h, emb)
        for up, block in zip(self.ups, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, emb)
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
