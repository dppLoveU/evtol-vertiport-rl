"""Stage 4B-2: training-loop smoke for the OD diffusion model.

Verifies the full Stage-4B training scaffold end-to-end on real data:
``DataLoader`` -> AMP autocast -> ``training_loss`` -> backward -> grad clip ->
optimizer step -> ``EMA.update`` -> periodic eval (with EMA weights) ->
checkpoint save -> reload-from-best -> re-eval to confirm bit-equal val
loss. NOT a real training run -- only ``max_steps`` steps with a tiny
U-Net, on a 0.117%-sparse target. NO ``od_samples.npy``, NO
``data/synthetic/`` write.

Outputs (under ``--run_dir``, default ``models/diffusion_od_smoke/``):
  * ``last.pt``         latest checkpoint (overwritten on each save).
  * ``best.pt``         lowest-val-loss checkpoint seen so far.
  * ``config.yaml``     snapshot of the loaded YAML config (for provenance).
  * ``train_log.jsonl`` append-only event log -- one JSON dict per line.

``models/`` is gitignored, so nothing produced here lands in commits.

Run:
    python -m experiments.run_stage4_train_smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.od_dataset import ODDataset
from src.models.diffusion import GaussianDiffusion
from src.models.ema import EMA
from src.models.unet_od import UNetOD

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "diffusion.yaml"
DEFAULT_RUN_DIR = REPO / "models" / "diffusion_od_smoke"


class _TorchODDataset(Dataset):
    """Wrap ``ODDataset`` so the default ``DataLoader`` collate gets tensors.

    ``ODDataset.__getitem__`` returns ``(numpy_array, dict[str, int])``; this
    thin shim converts the array to a tensor. The condition dict's int
    values are batched by the default collate into shape-``(B,)`` tensors.
    """

    def __init__(self, base: ODDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x_np, cond = self.base[idx]
        return torch.from_numpy(x_np), cond


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _cycle(loader: DataLoader) -> Iterator:
    """Endless iterator over a DataLoader (smoke doesn't think in epochs)."""
    while True:
        for batch in loader:
            yield batch


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(
    model: nn.Module,
    diff: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    fixed_seed: int,
) -> float:
    """Average DDPM training loss over ``loader``, using a fixed-seed ``t``."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(fixed_seed)
    losses: list[float] = []
    with torch.no_grad():
        for x, cond in loader:
            x = x.to(device, non_blocking=True)
            hour = cond["hour"].to(device)
            dow = cond["day_of_week"].to(device)
            is_weekend = cond["is_weekend"].to(device)
            t = torch.randint(
                0,
                diff.num_train_timesteps,
                (x.shape[0],),
                device=device,
                dtype=torch.long,
                generator=gen,
            )
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = diff.training_loss(model, x, hour, dow, is_weekend, t=t)
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def _save_ckpt(
    path: Path,
    model: nn.Module,
    ema: EMA,
    optim: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    config: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "step": int(step),
            "val_loss": float(val_loss),
            "config": config,
        },
        path,
    )


def _build_unet(cfg: dict, device: torch.device) -> UNetOD:
    m = cfg["model"]
    return UNetOD(
        in_channels=int(cfg["data"]["window"]),
        base_channels=int(m["base_channels"]),
        channel_mults=tuple(m["channel_mults"]),
        time_emb_dim=int(m["time_emb_dim"]),
        cond_emb_dim=int(m["time_emb_dim"]),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=100,
        help="DDPM step count for the smoke; overrides config's production 1000.",
    )
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16  # Blackwell sm_120 supports bf16 natively; no scaler needed.

    # --- run directory + config snapshot + fresh log ---
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log_path = args.run_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    def log(event: dict) -> None:
        event = {"ts": time.time(), **event}
        with open(log_path, "a") as fh:
            fh.write(json.dumps(event) + "\n")

    print("=" * 76)
    print("STAGE 4B-2  --  training-loop smoke")
    print("=" * 76)
    print(f"  device                : {device}")
    if device.type == "cuda":
        print(f"  gpu                   : {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
    print(f"  torch                 : {torch.__version__}")
    print(f"  amp                   : enabled={amp_enabled}  dtype={amp_dtype}")
    print(f"  run_dir               : {args.run_dir}")
    print(f"  max_steps             : {args.max_steps}")
    print(f"  eval_every            : {args.eval_every}")
    print(f"  save_every            : {args.save_every}")
    print(f"  num_train_timesteps   : {args.num_train_timesteps}")
    print(f"  ema_decay             : {args.ema_decay}")
    print(f"  seed                  : {seed}")

    # --- datasets / loaders ---
    od_path = _resolve(cfg["input"]["od_path"])
    meta_path = _resolve(cfg["input"]["meta_path"])
    train_base = ODDataset(
        od_path, meta_path, "train",
        window=cfg["data"]["window"],
        pad_multiple=cfg["data"]["pad_multiple"],
        clip_val=cfg["data"]["clip_val"],
        split_cfg=cfg["data"]["split"],
    )
    val_base = ODDataset(
        od_path, meta_path, "val",
        window=cfg["data"]["window"],
        pad_multiple=cfg["data"]["pad_multiple"],
        norm_stats=train_base.norm_stats,
        split_cfg=cfg["data"]["split"],
    )
    bs = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(
        _TorchODDataset(train_base), batch_size=bs, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        _TorchODDataset(val_base), batch_size=bs, shuffle=False, num_workers=0
    )
    print(f"\n  train samples         : {len(train_base)}")
    print(f"  val samples           : {len(val_base)}")

    # --- model / diffusion / EMA / optim ---
    model = _build_unet(cfg, device)
    diff = GaussianDiffusion(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=cfg["diffusion"]["beta_schedule"],
    ).to(device)
    ema = EMA(model, decay=args.ema_decay).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    grad_clip = float(cfg["train"]["grad_clip"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  model params          : {n_params:,}  ({n_params * 4 / 1e6:.2f} MB)")
    print(f"  diffusion             : {diff.summary()}")

    # --- training loop ---
    train_iter = _cycle(train_loader)
    train_losses: list[float] = []
    last_val: float = float("nan")
    best_val: float = float("inf")
    print("\n--- training -------------------------------------------------")
    for step in range(1, args.max_steps + 1):
        model.train()
        x, cond = next(train_iter)
        x = x.to(device, non_blocking=True)
        hour = cond["hour"].to(device)
        dow = cond["day_of_week"].to(device)
        is_weekend = cond["is_weekend"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss = diff.training_loss(model, x, hour, dow, is_weekend)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)
        ema.update(model)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.time() - t0) * 1000.0

        loss_val = float(loss.detach().item())
        gn_val = float(grad_norm.detach().item())
        train_losses.append(loss_val)
        if not np.isfinite(loss_val):
            raise RuntimeError(f"non-finite train loss at step {step}: {loss_val}")
        if not np.isfinite(gn_val):
            raise RuntimeError(f"non-finite grad_norm at step {step}: {gn_val}")
        log({
            "step": step, "phase": "train",
            "loss": loss_val, "grad_norm": gn_val, "dt_ms": dt_ms,
        })
        print(
            f"  step {step:>3}: train_loss={loss_val:.4f}  "
            f"grad_norm={gn_val:.4f}  dt={dt_ms:.0f} ms"
        )

        # Eval with EMA weights.
        if step % args.eval_every == 0:
            ema.store(model)
            ema.copy_to(model)
            val_loss = _evaluate(
                model, diff, val_loader, device,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype, fixed_seed=seed,
            )
            ema.restore(model)
            last_val = val_loss
            if not np.isfinite(last_val):
                raise RuntimeError(f"non-finite val loss at step {step}: {last_val}")
            log({"step": step, "phase": "val", "val_loss_ema": last_val})
            print(f"  step {step:>3}: val_loss_ema={last_val:.4f}")

        # Save.
        if step % args.save_every == 0:
            _save_ckpt(args.run_dir / "last.pt", model, ema, optim, step, last_val, cfg)
            is_best = last_val < best_val
            if is_best:
                best_val = last_val
                _save_ckpt(args.run_dir / "best.pt", model, ema, optim, step, last_val, cfg)
            log({
                "step": step, "phase": "save",
                "ckpt": "last.pt", "best": is_best, "val_loss": last_val,
            })
            tag = " (best)" if is_best else ""
            print(f"  step {step:>3}: saved last.pt{tag}")

    # --- summary ---
    print("\n--- summary --------------------------------------------------")
    print(f"  total steps           : {args.max_steps}")
    print(
        f"  train_loss range      : [{min(train_losses):.4f}, "
        f"{max(train_losses):.4f}]  mean={np.mean(train_losses):.4f}"
    )
    print(f"  last val_loss         : {last_val:.4f}")
    print(f"  best val_loss         : {best_val:.4f}")

    # --- checkpoint files ---
    print("\n--- checkpoint files -----------------------------------------")
    for p in sorted(args.run_dir.iterdir()):
        size = p.stat().st_size
        print(f"  {p.name:<24} {size / 1e6:>7.3f} MB")

    # --- best.pt reload verification ---
    print("\n--- best.pt reload check -------------------------------------")
    best_path = args.run_dir / "best.pt"
    ckpt = torch.load(best_path, weights_only=True, map_location=device)
    fresh_model = _build_unet(cfg, device)
    fresh_ema = EMA(fresh_model, decay=args.ema_decay).to(device)
    fresh_model.load_state_dict(ckpt["model_state_dict"])
    fresh_ema.load_state_dict(ckpt["ema_state_dict"])
    fresh_ema.copy_to(fresh_model)
    reloaded_val = _evaluate(
        fresh_model, diff, val_loader, device,
        amp_enabled=amp_enabled, amp_dtype=amp_dtype, fixed_seed=seed,
    )
    abs_diff = abs(reloaded_val - float(ckpt["val_loss"]))
    print(f"  saved val_loss        : {float(ckpt['val_loss']):.6f}")
    print(f"  reloaded val_loss     : {reloaded_val:.6f}")
    print(f"  abs diff              : {abs_diff:.6e}")
    log({
        "step": args.max_steps, "phase": "reload_check",
        "reloaded_val_loss": reloaded_val, "saved_val_loss": float(ckpt["val_loss"]),
        "abs_diff": abs_diff,
    })
    if abs_diff >= 1e-3:
        raise RuntimeError(f"reload val mismatch: |diff|={abs_diff:.6e}")

    # --- memory ---
    print("\n--- memory ---------------------------------------------------")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e6
        reserved = torch.cuda.max_memory_reserved() / 1e6
        print(f"  gpu peak allocated    : {peak:.1f} MB")
        print(f"  gpu peak reserved     : {reserved:.1f} MB")
    else:
        print("  cpu run -- no GPU memory tracking")

    print("\n" + "=" * 76)
    print("STAGE 4B-2 train smoke OK")
    print("=" * 76)


if __name__ == "__main__":
    main()
