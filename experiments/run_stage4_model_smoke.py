"""Stage 4B-1: small U-Net + DDPM forward/sample smoke on the real OD tensor.

Goal: exercise the full diffusion pipeline end-to-end on a single
``[1, 1, 544, 544]`` real OD slice — DataLoader fetch, U-Net forward,
DDPM training loss + backward, DDIM sample, inverse transform — and
report shapes, NaN/finite checks, GPU peak memory, and timing.

This is a SMOKE script: 3 training steps, batch_size=1, 5-step DDIM
sampling, no checkpoint, no ``models/`` write, no ``data/synthetic/``
write. Run BEFORE doing any real training.

Run:
    python -m experiments.run_stage4_model_smoke
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.od_dataset import ODDataset
from src.models.diffusion import GaussianDiffusion
from src.models.unet_od import UNetOD

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "diffusion.yaml"


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--num_train_timesteps", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("STAGE 4B-1  --  small U-Net + DDPM smoke")
    print("=" * 76)
    print(f"  device                : {device}")
    if device.type == "cuda":
        print(f"  gpu                   : {torch.cuda.get_device_name(0)}")
        print(f"  capability            : {torch.cuda.get_device_capability(0)}")
        torch.cuda.reset_peak_memory_stats()
    print(f"  torch                 : {torch.__version__}")
    print(f"  num_train_timesteps   : {args.num_train_timesteps}")
    print(f"  num_inference_steps   : {args.num_inference_steps}")
    print(f"  smoke training steps  : {args.n_steps}")

    # --- dataset --------------------------------------------------------
    train = ODDataset(
        _resolve(cfg["input"]["od_path"]),
        _resolve(cfg["input"]["meta_path"]),
        "train",
        window=cfg["data"]["window"],
        pad_multiple=cfg["data"]["pad_multiple"],
        clip_val=cfg["data"]["clip_val"],
        split_cfg=cfg["data"]["split"],
    )
    print("\n--- dataset --------------------------------------------------")
    s = train.summary()
    print(f"  split={s['split']}  n_samples={s['n_samples']}  pad_size={s['pad_size']}")
    print(f"  norm_stats={s['norm_stats']}")

    # Fetch one batch (batch_size=1).
    x_np, cond = train[0]
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # [1, W, H, W]
    hour = torch.tensor([cond["hour"]], dtype=torch.long, device=device)
    dow = torch.tensor([cond["day_of_week"]], dtype=torch.long, device=device)
    is_weekend = torch.tensor([cond["is_weekend"]], dtype=torch.long, device=device)
    print(
        f"  batch[0]: x.shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={x.min():.4f} max={x.max():.4f} mean={x.mean():.6f}"
    )
    print(
        f"  condition: hour={cond['hour']}  dow={cond['day_of_week']}  "
        f"is_weekend={cond['is_weekend']}"
    )
    assert torch.isfinite(x).all(), "input contains non-finite values"

    # --- model ----------------------------------------------------------
    model_cfg = cfg["model"]
    model = UNetOD(
        in_channels=cfg["data"]["window"],
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=tuple(model_cfg["channel_mults"]),
        time_emb_dim=int(model_cfg["time_emb_dim"]),
        cond_emb_dim=int(model_cfg["time_emb_dim"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n--- model ----------------------------------------------------")
    print(f"  UNetOD(base_channels={model_cfg['base_channels']}, "
          f"channel_mults={tuple(model_cfg['channel_mults'])})")
    print(f"  params: {n_params:,}  trainable: {n_trainable:,}  "
          f"({n_params * 4 / 1e6:.2f} MB float32)")

    # --- diffusion ------------------------------------------------------
    diff = GaussianDiffusion(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=cfg["diffusion"]["beta_schedule"],
    ).to(device)
    print("\n--- diffusion ------------------------------------------------")
    print(f"  {diff.summary()}")

    # --- forward / loss / backward smoke -------------------------------
    print("\n--- training smoke -------------------------------------------")
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    model.train()
    for step in range(args.n_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.time()
        # Fix t deterministically inside the schedule range -- mostly for
        # repeatable logs across runs.
        t = torch.tensor(
            [(step * 17) % args.num_train_timesteps], device=device, dtype=torch.long
        )
        loss = diff.training_loss(model, x, hour, dow, is_weekend, t=t)
        assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss}"
        loss.backward()
        g_norm = torch.sqrt(
            sum(p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None)
        )
        optim.step()
        optim.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t_start) * 1000.0
        print(
            f"  step {step}: t={int(t.item()):3d}  loss={loss.item():.4f}  "
            f"grad_norm={g_norm.item():.4f}  dt={dt:.0f} ms"
        )

    # --- DDIM sample ----------------------------------------------------
    print("\n--- DDIM sample ----------------------------------------------")
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.time()
    sample = diff.ddim_sample(
        model,
        tuple(x.shape),
        hour,
        dow,
        is_weekend,
        num_inference_steps=args.num_inference_steps,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t_start) * 1000.0
    print(f"  sample.shape={tuple(sample.shape)}  dt={dt:.0f} ms")
    print(
        f"  min={sample.min().item():.4f}  max={sample.max().item():.4f}  "
        f"mean={sample.mean().item():.4f}"
    )
    sample_finite = bool(torch.isfinite(sample).all().item())
    print(f"  finite={sample_finite}")
    assert sample_finite, "sample contains non-finite values"

    # --- inverse transform (unpad + inverse_norm) ----------------------
    sample_np = sample.detach().cpu().numpy()[0]  # [W, pad, pad]
    recovered = train.inverse_transform(sample_np)
    print("\n--- inverse transform ----------------------------------------")
    print(f"  recovered.shape={recovered.shape}  dtype={recovered.dtype}")
    nonneg = bool(np.all(recovered >= 0.0))
    finite = bool(np.all(np.isfinite(recovered)))
    print(f"  nonneg={nonneg}  finite={finite}")
    print(
        f"  min={recovered.min():.4f}  max={recovered.max():.4e}  "
        f"mean={recovered.mean():.4e}"
    )
    assert nonneg, "inverse-transformed sample has negatives"
    # finite is asserted lazily -- a freshly-initialized model may still
    # land near the boundary; we only require finite for the test to pass.
    assert finite, "inverse-transformed sample has non-finite values"

    # --- memory ---------------------------------------------------------
    print("\n--- memory ---------------------------------------------------")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e6
        reserved = torch.cuda.max_memory_reserved() / 1e6
        print(f"  gpu peak allocated : {peak:.1f} MB")
        print(f"  gpu peak reserved  : {reserved:.1f} MB")
    else:
        print("  cpu run -- no GPU memory tracking")

    print("\n" + "=" * 76)
    print("STAGE 4B-1 smoke OK -- forward/loss/backward/sample/inverse all clean.")
    print("=" * 76)


if __name__ == "__main__":
    main()
