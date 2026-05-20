"""Stage 4B-1 smoke tests for ``src/models/unet_od.py`` + ``diffusion.py``.

All tests run on tiny ``[B=2, C=1, H=32, W=32]`` fake tensors so the
suite stays fast and CPU-only. The real 544x544 forward is exercised in
``experiments/run_stage4_model_smoke.py``.
"""
from __future__ import annotations

import pytest
import torch

from src.models.diffusion import GaussianDiffusion, _cosine_betas, _linear_betas
from src.models.ema import EMA
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


# --- classifier-free guidance (Stage 4B-3) --------------------------------


def test_unet_forward_with_cond_drop_mask(unet: UNetOD, cond: tuple) -> None:
    """Dropping the condition should change the model output."""
    x = torch.randn(B, C, H, W)
    t = torch.tensor([0, 50], dtype=torch.long)
    drop_all = torch.tensor([True, True])
    out_cond = unet(x, t, *cond)
    out_uncond = unet(x, t, *cond, cond_drop_mask=drop_all)
    assert out_uncond.shape == out_cond.shape
    assert torch.isfinite(out_uncond).all()
    # The cond MLP starts from non-zero init weights; zeroing its output
    # must propagate to a different forward.
    assert not torch.allclose(out_cond, out_uncond, atol=1e-6)


def test_unet_per_sample_cond_dropout(unet: UNetOD, cond: tuple) -> None:
    """A per-sample mask leaves un-masked samples bit-equal to no-mask."""
    x = torch.randn(B, C, H, W)
    t = torch.tensor([0, 50], dtype=torch.long)
    # Sample 0 keeps cond; sample 1 drops it.
    mask = torch.tensor([False, True])
    out_mixed = unet(x, t, *cond, cond_drop_mask=mask)
    out_no_mask = unet(x, t, *cond)
    # Sample 0 should match the no-mask path; sample 1 should not.
    assert torch.allclose(out_mixed[0], out_no_mask[0], atol=1e-6)
    assert not torch.allclose(out_mixed[1], out_no_mask[1], atol=1e-6)


def test_training_loss_full_cond_dropout_runs(unet: UNetOD, cond: tuple) -> None:
    diff = GaussianDiffusion(num_train_timesteps=100)
    x0 = torch.randn(B, C, H, W)
    t = torch.tensor([10, 50], dtype=torch.long)
    loss = diff.training_loss(unet, x0, *cond, t=t, cond_dropout_prob=1.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_ddim_sample_with_guidance_shape(unet: UNetOD, cond: tuple) -> None:
    diff = GaussianDiffusion(num_train_timesteps=100)
    out = diff.ddim_sample(
        unet, (B, C, H, W), *cond, num_inference_steps=5, guidance_scale=2.0
    )
    assert out.shape == (B, C, H, W)
    assert torch.isfinite(out).all()


def test_ddim_sample_guidance_one_matches_no_arg(unet: UNetOD, cond: tuple) -> None:
    """guidance_scale=1.0 must be bit-equal to the no-guidance default path."""
    diff = GaussianDiffusion(num_train_timesteps=100)
    torch.manual_seed(0)
    out_default = diff.ddim_sample(unet, (B, C, H, W), *cond, num_inference_steps=5)
    torch.manual_seed(0)
    out_g1 = diff.ddim_sample(
        unet, (B, C, H, W), *cond, num_inference_steps=5, guidance_scale=1.0
    )
    assert torch.equal(out_default, out_g1)


# --- Stage 4B-5B: weighted-loss plumbing ----------------------------------


def test_training_loss_weight_map_none_matches_default(
    unet: UNetOD, cond: tuple
) -> None:
    """weight_map=None must be bit-equal to the pre-PR5B-1 F.mse_loss path."""
    diff = GaussianDiffusion(num_train_timesteps=100)
    x0 = torch.randn(B, C, H, W)
    t = torch.tensor([10, 50], dtype=torch.long)
    torch.manual_seed(0)
    loss_default = diff.training_loss(unet, x0, *cond, t=t)
    torch.manual_seed(0)
    loss_none = diff.training_loss(unet, x0, *cond, t=t, weight_map=None)
    assert torch.equal(loss_default, loss_none)


def test_training_loss_weight_ones_is_close_to_unweighted(
    unet: UNetOD, cond: tuple
) -> None:
    """A uniform ones weight produces ~unweighted MSE (modulo float fma)."""
    diff = GaussianDiffusion(num_train_timesteps=100)
    x0 = torch.randn(B, C, H, W)
    t = torch.tensor([10, 50], dtype=torch.long)
    weight = torch.ones_like(x0)
    torch.manual_seed(0)
    loss_unweighted = diff.training_loss(unet, x0, *cond, t=t)
    torch.manual_seed(0)
    loss_weighted = diff.training_loss(unet, x0, *cond, t=t, weight_map=weight)
    assert torch.isfinite(loss_weighted)
    assert torch.allclose(loss_weighted, loss_unweighted, atol=1e-6, rtol=1e-6)


def test_training_loss_nonuniform_weight_backward(
    unet: UNetOD, cond: tuple
) -> None:
    """A non-uniform mean-normalised weight produces a finite scalar loss
    whose gradient flows back through the model."""
    diff = GaussianDiffusion(num_train_timesteps=100)
    x0 = torch.randn(B, C, H, W)
    t = torch.tensor([10, 50], dtype=torch.long)
    weight = torch.full_like(x0, 0.5)
    weight[:, :, : H // 2, : W // 2] = 5.0
    weight = weight / weight.mean()  # normalise so loss magnitude is comparable
    loss = diff.training_loss(unet, x0, *cond, t=t, weight_map=weight)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    g_total = sum(
        p.grad.abs().sum() for p in unet.parameters() if p.grad is not None
    )
    assert torch.isfinite(g_total)
    assert g_total > 0


def test_training_loss_weight_map_changes_loss_value(
    unet: UNetOD, cond: tuple
) -> None:
    """A non-uniform weight must change the scalar loss value vs unweighted."""
    diff = GaussianDiffusion(num_train_timesteps=100)
    x0 = torch.randn(B, C, H, W)
    t = torch.tensor([10, 50], dtype=torch.long)
    weight = torch.full_like(x0, 0.5)
    weight[:, :, : H // 2, : W // 2] = 5.0  # not mean-normalised here
    torch.manual_seed(0)
    loss_unw = diff.training_loss(unet, x0, *cond, t=t)
    torch.manual_seed(0)
    loss_w = diff.training_loss(unet, x0, *cond, t=t, weight_map=weight)
    assert not torch.allclose(loss_unw, loss_w, atol=1e-4)


# --- checkpoint round-trip (Stage 4B-2 smoke) -----------------------------


def test_checkpoint_save_load_roundtrip(
    tmp_path, unet: UNetOD, cond: tuple
) -> None:
    """Save (model + EMA + optim + scalars) and reload into fresh instances."""
    diff = GaussianDiffusion(num_train_timesteps=100)
    ema = EMA(unet, decay=0.99)
    optim = torch.optim.AdamW(unet.parameters(), lr=2e-4)

    # One step to populate optimizer state.
    x = torch.randn(B, C, H, W)
    loss = diff.training_loss(unet, x, *cond, t=torch.tensor([5, 50], dtype=torch.long))
    loss.backward()
    optim.step()
    ema.update(unet)

    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_state_dict": unet.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "step": 1,
            "val_loss": 0.5,
        },
        ckpt_path,
    )

    # Fresh instances mimicking a "load checkpoint and resume" flow.
    unet2 = UNetOD(
        in_channels=C, base_channels=8, channel_mults=(1, 2, 4),
        time_emb_dim=64, cond_emb_dim=64,
    )
    ema2 = EMA(unet2, decay=0.99)
    optim2 = torch.optim.AdamW(unet2.parameters(), lr=2e-4)
    ckpt = torch.load(ckpt_path, weights_only=True)
    unet2.load_state_dict(ckpt["model_state_dict"])
    ema2.load_state_dict(ckpt["ema_state_dict"])
    optim2.load_state_dict(ckpt["optimizer_state_dict"])

    # Model params match.
    for (n1, p1), (n2, p2) in zip(unet.named_parameters(), unet2.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2), n1
    # EMA buffers match.
    for (n1, b1), (n2, b2) in zip(ema.named_buffers(), ema2.named_buffers()):
        assert n1 == n2
        assert torch.allclose(b1, b2), n1
    # Forward passes agree on a random input.
    x_in = torch.randn(B, C, H, W)
    t_in = torch.tensor([10, 50], dtype=torch.long)
    with torch.no_grad():
        out1 = unet(x_in, t_in, *cond)
        out2 = unet2(x_in, t_in, *cond)
    assert torch.allclose(out1, out2, atol=1e-6)
    # Scalars survived.
    assert ckpt["step"] == 1
    assert ckpt["val_loss"] == pytest.approx(0.5)
