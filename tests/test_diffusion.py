"""Stage 4B-1 smoke tests for ``src/models/unet_od.py`` + ``diffusion.py``.

All tests run on tiny ``[B=2, C=1, H=32, W=32]`` fake tensors so the
suite stays fast and CPU-only. The real 544x544 forward is exercised in
``experiments/run_stage4_model_smoke.py``.
"""
from __future__ import annotations

import pytest
import torch

from src.models.diffusion import GaussianDiffusion, _cosine_betas, _linear_betas
from src.models.unet_od import (
    ConditionEmbedding,
    ResBlock,
    UNetOD,
    sinusoidal_embedding,
)

B, C, H, W = 2, 1, 32, 32


@pytest.fixture
def cond() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hour = torch.tensor([0, 12], dtype=torch.long)
    dow = torch.tensor([0, 5], dtype=torch.long)
    is_weekend = torch.tensor([0, 1], dtype=torch.long)
    return hour, dow, is_weekend


@pytest.fixture
def unet() -> UNetOD:
    # Even smaller than the smoke config -- enough to exercise the path.
    return UNetOD(
        in_channels=C,
        base_channels=8,
        channel_mults=(1, 2, 4),
        time_emb_dim=64,
        cond_emb_dim=64,
    )


# --- helpers --------------------------------------------------------------


def test_sinusoidal_embedding_shape_and_finite() -> None:
    t = torch.tensor([0, 50, 99], dtype=torch.long)
    emb = sinusoidal_embedding(t, 64)
    assert emb.shape == (3, 64)
    assert emb.dtype == torch.float32
    assert torch.isfinite(emb).all()


def test_sinusoidal_embedding_rejects_odd_dim() -> None:
    with pytest.raises(ValueError):
        sinusoidal_embedding(torch.tensor([0]), 65)


def test_condition_embedding_shape(cond: tuple) -> None:
    emb = ConditionEmbedding(cond_dim=32)
    out = emb(*cond)
    assert out.shape == (B, 32)
    assert torch.isfinite(out).all()


def test_resblock_shape_and_residual() -> None:
    block = ResBlock(in_c=8, out_c=8, emb_dim=16)
    x = torch.randn(B, 8, 16, 16)
    emb = torch.randn(B, 16)
    out = block(x, emb)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# --- U-Net ----------------------------------------------------------------


def test_unet_forward_shape(unet: UNetOD, cond: tuple) -> None:
    x = torch.randn(B, C, H, W)
    t = torch.tensor([0, 50], dtype=torch.long)
    out = unet(x, t, *cond)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


def test_unet_backward_grad(unet: UNetOD, cond: tuple) -> None:
    x = torch.randn(B, C, H, W)
    t = torch.tensor([0, 50], dtype=torch.long)
    out = unet(x, t, *cond)
    loss = out.pow(2).mean()
    loss.backward()
    # Every parameter that requires grad should have a finite gradient.
    n_grad = 0
    for name, p in unet.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"missing grad: {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad: {name}"
        n_grad += 1
    assert n_grad > 0


def test_unet_rejects_too_shallow_channel_mults() -> None:
    with pytest.raises(ValueError):
        UNetOD(channel_mults=(1,))


# --- schedules ------------------------------------------------------------


def test_cosine_betas_monotone() -> None:
    betas = _cosine_betas(100)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # alphas_cumprod must be monotone non-increasing.
    assert torch.all(alphas_cumprod[:-1] >= alphas_cumprod[1:])
    # In (0, 1] throughout.
    assert (alphas_cumprod > 0).all() and (alphas_cumprod <= 1).all()


def test_linear_betas_range() -> None:
    betas = _linear_betas(100)
    assert betas[0] == pytest.approx(1e-4)
    assert betas[-1] == pytest.approx(0.02)


# --- diffusion wrapper ----------------------------------------------------


def test_add_noise_at_t0_is_identity_like() -> None:
    diff = GaussianDiffusion(num_train_timesteps=100)
    x0 = torch.randn(B, C, H, W)
    t = torch.tensor([0, 0], dtype=torch.long)
    x_t, _ = diff.add_noise(x0, t, noise=torch.zeros_like(x0))
    # noise=0 -> x_t = sqrt(alpha_cumprod_0) * x0 ~ x0 (alpha~1 at t=0).
    assert torch.allclose(x_t, x0, atol=5e-3)


def test_training_loss_finite_and_backward(unet: UNetOD, cond: tuple) -> None:
    diff = GaussianDiffusion(num_train_timesteps=100)
    x0 = torch.randn(B, C, H, W)
    # Fix t so the assertion below is deterministic.
    t = torch.tensor([10, 50], dtype=torch.long)
    loss = diff.training_loss(unet, x0, *cond, t=t)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    g_total = sum(
        p.grad.abs().sum() for p in unet.parameters() if p.grad is not None
    )
    assert torch.isfinite(g_total)


def test_ddim_sample_shape_no_nan(unet: UNetOD, cond: tuple) -> None:
    diff = GaussianDiffusion(num_train_timesteps=100)
    sample = diff.ddim_sample(
        unet, (B, C, H, W), *cond, num_inference_steps=5
    )
    assert sample.shape == (B, C, H, W)
    assert torch.isfinite(sample).all()


def test_ddim_rejects_bad_inference_steps(unet: UNetOD, cond: tuple) -> None:
    diff = GaussianDiffusion(num_train_timesteps=100)
    with pytest.raises(ValueError):
        diff.ddim_sample(unet, (B, C, H, W), *cond, num_inference_steps=0)
    with pytest.raises(ValueError):
        diff.ddim_sample(unet, (B, C, H, W), *cond, num_inference_steps=200)


def test_diffusion_to_device_moves_buffers() -> None:
    diff = GaussianDiffusion(num_train_timesteps=100)
    # Buffers must follow .to(); use CPU here so the test runs without CUDA.
    diff = diff.to("cpu")
    for name, buf in diff.named_buffers():
        assert buf.device.type == "cpu", f"buffer {name} on {buf.device}"


def test_summary_keys() -> None:
    diff = GaussianDiffusion(num_train_timesteps=100)
    s = diff.summary()
    assert set(s) >= {"num_train_timesteps", "beta_min", "beta_max", "alpha_cumprod_T"}
