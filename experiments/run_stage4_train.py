"""Stage 4B-3 real training entry point.

Three profiles -- ``pilot`` / ``medium`` / ``full`` -- with hyperparameters
encoded in the ``PROFILES`` table below. The configurable surface is
intentionally narrow: profile picks the heavy knobs (max_steps,
base_channels, channel_mults, batch_size, ema_decay, warmup, eval/save
cadence, in-loop sample count). Everything else (lr, grad_clip, T,
beta schedule, normalization, padding, split) comes from
``configs/diffusion.yaml``.

Differences from the Stage 4B-2 smoke script:

  * Per-profile heavy knobs (max_steps up to 20k, depth-4 U-Net, lr warmup).
  * Classifier-free guidance via ``cond_dropout_prob`` (training) and
    ``guidance_scale`` (in-loop sample diagnostics).
  * In-loop sample diagnostics every ``save_every`` steps: generate
    ``n_samples_diag`` slices with the EMA weights, ``inverse_transform``
    to count space, compute ``marginal_compare`` against the real
    validation OD, log to TensorBoard and ``train_log.jsonl``.
  * ``--resume <ckpt.pt>`` continues from a previous run.
  * TensorBoard writer at ``results/stage4/train_<profile>/tb/``.
  * End-of-run artifacts under ``results/stage4/train_<profile>/``:
    ``loss_curve.png``, ``sample_grid.png``, ``marginal_match.png``,
    ``metrics.json``.

Checkpoints land under ``models/diffusion_od_<profile>/`` -- ``models/``
is gitignored. ``results/`` is tracked (commit policy: small artifacts
only, no checkpoints).

This is NOT a smoke. Running ``--profile pilot`` consumes ~10-20 minutes
on the RTX 5070 Ti; ``--profile full`` budgets 12 hours.

Run:
    python -m experiments.run_stage4_train --profile pilot
    python -m experiments.run_stage4_train --profile medium --resume models/diffusion_od_pilot/best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")  # headless backend; no DISPLAY needed.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter

from src.data.od_dataset import ODDataset
from src.models.diffusion import GaussianDiffusion
from src.models.ema import EMA
from src.models.unet_od import UNetOD
from src.utils.metrics_dist import marginal_compare

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "diffusion.yaml"


# --- profile table --------------------------------------------------------


PROFILES: dict[str, dict[str, Any]] = {
    "pilot": {
        "max_steps": 1000,
        "base_channels": 32,
        "channel_mults": (1, 2, 4),
        "batch_size": 4,
        "ema_decay": 0.999,
        "eval_every": 200,
        "save_every": 500,
        "warmup_steps": 0,
        "n_samples_diag": 4,
        "diag_at_end_only": True,   # do the costly sample diag once, at the end
        "num_train_timesteps": 1000,
        "cond_dropout_prob": 0.1,
        "guidance_scale": 2.0,
        "num_inference_steps": 50,
    },
    "medium": {
        "max_steps": 5000,
        "base_channels": 64,
        "channel_mults": (1, 2, 4, 8),
        "batch_size": 2,
        "ema_decay": 0.999,
        "eval_every": 500,
        "save_every": 1000,
        "warmup_steps": 500,
        "n_samples_diag": 16,
        "diag_at_end_only": False,
        "num_train_timesteps": 1000,
        "cond_dropout_prob": 0.1,
        "guidance_scale": 2.0,
        "num_inference_steps": 50,
    },
    "full": {
        "max_steps": 20000,
        "base_channels": 64,
        "channel_mults": (1, 2, 4, 8),
        "batch_size": 2,
        "ema_decay": 0.9999,
        "eval_every": 1000,
        "save_every": 2000,
        "warmup_steps": 1000,
        "n_samples_diag": 16,
        "diag_at_end_only": False,
        "num_train_timesteps": 1000,
        "cond_dropout_prob": 0.1,
        "guidance_scale": 2.0,
        "num_inference_steps": 50,
    },
}


# --- helpers --------------------------------------------------------------


class _TorchODDataset(Dataset):
    """Wrap ODDataset so DataLoader's default collate yields tensors."""

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
    while True:
        for batch in loader:
            yield batch


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_unet(profile: dict[str, Any], window: int, time_emb_dim: int, device: torch.device) -> UNetOD:
    return UNetOD(
        in_channels=window,
        base_channels=int(profile["base_channels"]),
        channel_mults=tuple(profile["channel_mults"]),
        time_emb_dim=time_emb_dim,
        cond_emb_dim=time_emb_dim,
    ).to(device)


def _get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup from 0 to base_lr; constant base_lr afterward."""
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    return base_lr * float(step) / float(warmup_steps)


def _set_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for g in optim.param_groups:
        g["lr"] = lr


def _save_ckpt(
    path: Path,
    model: nn.Module,
    ema: EMA,
    optim: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    config: dict[str, Any],
    profile_name: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "step": int(step),
            "val_loss": float(val_loss),
            "config": config,
            "profile_name": profile_name,
        },
        path,
    )


def _load_ckpt(
    path: Path,
    model: nn.Module,
    ema: EMA,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    ema.load_state_dict(ckpt["ema_state_dict"])
    optim.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["step"]), float(ckpt["val_loss"])


# --- eval / sample diagnostics -------------------------------------------


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
    """Average DDPM training loss over the val loader at fixed-seed t."""
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
                0, diff.num_train_timesteps, (x.shape[0],),
                device=device, dtype=torch.long, generator=gen,
            )
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = diff.training_loss(model, x, hour, dow, is_weekend, t=t)
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def _collect_real_val_counts(val_base: ODDataset) -> np.ndarray:
    """Inverse-transform every val slice back to count space (one-time, CPU)."""
    counts_list: list[np.ndarray] = []
    for i in range(len(val_base)):
        x_np, _ = val_base[i]
        counts = val_base.inverse_transform(x_np)  # [W, Z, Z]
        counts_list.append(counts[0])  # window=1
    return np.stack(counts_list)  # [N_val, Z, Z]


def _sample_diag(
    model: nn.Module,
    diff: GaussianDiffusion,
    ema: EMA,
    val_loader: DataLoader,
    train_base: ODDataset,
    real_val_counts: np.ndarray,
    *,
    n_samples: int,
    chunk_size: int,
    pad_size: int,
    device: torch.device,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Generate ``n_samples`` slices with EMA weights and compare marginals.

    Returns ``(metrics_dict, gen_counts)``. ``gen_counts`` has shape
    ``[n_samples, Z, Z]`` in count space (after ``inverse_transform``).
    """
    # Collect ``n_samples`` conditions from the val loader (repeating if needed).
    hours: list[int] = []
    dows: list[int] = []
    is_wks: list[int] = []
    for x, cond in val_loader:
        bs = x.shape[0]
        hours.extend(cond["hour"].tolist())
        dows.extend(cond["day_of_week"].tolist())
        is_wks.extend(cond["is_weekend"].tolist())
        if len(hours) >= n_samples:
            break
    hours = hours[:n_samples]
    dows = dows[:n_samples]
    is_wks = is_wks[:n_samples]

    model.eval()
    ema.store(model)
    ema.copy_to(model)
    torch.manual_seed(seed)  # repeatable init noise across diagnostics
    gen_chunks: list[np.ndarray] = []
    try:
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            n_chunk = end - start
            hour_t = torch.tensor(hours[start:end], device=device, dtype=torch.long)
            dow_t = torch.tensor(dows[start:end], device=device, dtype=torch.long)
            iw_t = torch.tensor(is_wks[start:end], device=device, dtype=torch.long)
            shape = (n_chunk, 1, pad_size, pad_size)
            samples = diff.ddim_sample(
                model, shape, hour_t, dow_t, iw_t,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            samples_np = samples.detach().cpu().numpy()
            for i in range(n_chunk):
                counts = train_base.inverse_transform(samples_np[i])  # [1, Z, Z]
                gen_chunks.append(counts[0])
    finally:
        ema.restore(model)
        model.train()

    gen_counts = np.stack(gen_chunks)  # [n_samples, Z, Z]
    metrics = marginal_compare(real_val_counts, gen_counts)
    return metrics, gen_counts


# --- plotting -------------------------------------------------------------


def _plot_loss_curve(log_path: Path, out_path: Path) -> None:
    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    with open(log_path) as fh:
        for line in fh:
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = evt.get("phase")
            if phase == "train":
                train_steps.append(int(evt["step"]))
                train_losses.append(float(evt["loss"]))
            elif phase == "val":
                val_steps.append(int(evt["step"]))
                val_losses.append(float(evt["val_loss_ema"]))
    if not train_steps:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_steps, train_losses, label="train_loss", color="C0", alpha=0.5, lw=1)
    if val_steps:
        ax.plot(val_steps, val_losses, "o-", label="val_loss_ema", color="C3", lw=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (MSE on epsilon)")
    ax.set_title("Stage 4B-3 training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_sample_grid(real: np.ndarray, gen: np.ndarray, out_path: Path) -> None:
    n = min(4, real.shape[0], gen.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        axes[0, i].imshow(np.log1p(real[i]), cmap="hot", aspect="auto")
        axes[0, i].set_title(f"real {i}", fontsize=10)
        axes[0, i].axis("off")
        axes[1, i].imshow(np.log1p(gen[i]), cmap="hot", aspect="auto")
        axes[1, i].set_title(f"gen {i}", fontsize=10)
        axes[1, i].axis("off")
    fig.suptitle("OD slices (log1p): real vs generated", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_marginal_match(real: np.ndarray, gen: np.ndarray, out_path: Path) -> None:
    r_row = real.sum(axis=-1).ravel()
    g_row = gen.sum(axis=-1).ravel()
    r_col = real.sum(axis=-2).ravel()
    g_col = gen.sum(axis=-2).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = 50
    for ax, (r, g, title) in zip(
        axes, [(r_row, g_row, "row sums"), (r_col, g_col, "col sums")]
    ):
        ax.hist(r, bins=bins, alpha=0.5, density=True, label="real", color="C0")
        ax.hist(g, bins=bins, alpha=0.5, density=True, label="gen", color="C3")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Marginal match: real vs generated", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --- main -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override the profile's max_steps (debug only).",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=REPO / "models",
        help="Parent dir for checkpoint subdir (default: <repo>/models/).",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=REPO / "results" / "stage4",
        help="Parent dir for tracked artifacts (default: <repo>/results/stage4/).",
    )
    args = parser.parse_args()

    profile_name = args.profile
    profile = dict(PROFILES[profile_name])
    if args.max_steps is not None:
        profile["max_steps"] = int(args.max_steps)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16  # see docs/decisions.md 2026-05-20

    run_dir = (args.models_dir / f"diffusion_od_{profile_name}").resolve()
    out_dir = (args.results_dir / f"train_{profile_name}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    log_path = run_dir / "train_log.jsonl"
    if log_path.exists() and args.resume is None:
        log_path.unlink()
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    def log(event: dict) -> None:
        event = {"ts": time.time(), **event}
        with open(log_path, "a") as fh:
            fh.write(json.dumps(event) + "\n")

    print("=" * 78)
    print(f"STAGE 4B-3  --  real training run  (profile={profile_name})")
    print("=" * 78)
    print(f"  device              : {device}")
    if device.type == "cuda":
        print(f"  gpu                 : {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
    print(f"  torch               : {torch.__version__}")
    print(f"  amp                 : enabled={amp_enabled}  dtype={amp_dtype}")
    print(f"  run_dir             : {run_dir}")
    print(f"  results_dir         : {out_dir}")
    print(f"  resume              : {args.resume}")
    print(f"  seed                : {seed}")
    print(f"  profile             :")
    for k, v in profile.items():
        print(f"    {k:<22}: {v}")

    # --- datasets ---
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
    train_loader = DataLoader(
        _TorchODDataset(train_base), batch_size=int(profile["batch_size"]),
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        _TorchODDataset(val_base), batch_size=int(profile["batch_size"]),
        shuffle=False, num_workers=0,
    )
    print(f"\n  train samples       : {len(train_base)}")
    print(f"  val samples         : {len(val_base)}")
    print(f"  pad_size            : {train_base.pad_size}")

    # Pre-compute real val counts once (small: 48 * 530^2 * 8 B ~= 108 MB).
    print(f"  collecting real val counts ...", end="", flush=True)
    real_val_counts = _collect_real_val_counts(val_base)
    print(f" done  shape={real_val_counts.shape}")

    # --- model / diffusion / EMA / optim ---
    model = _build_unet(profile, int(cfg["data"]["window"]),
                        int(cfg["model"]["time_emb_dim"]), device)
    diff = GaussianDiffusion(
        num_train_timesteps=int(profile["num_train_timesteps"]),
        beta_schedule=cfg["diffusion"]["beta_schedule"],
    ).to(device)
    ema = EMA(model, decay=float(profile["ema_decay"])).to(device)
    base_lr = float(cfg["train"]["lr"])
    optim = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    grad_clip = float(cfg["train"]["grad_clip"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  model params        : {n_params:,}  ({n_params * 4 / 1e6:.2f} MB)")
    print(f"  diffusion           : {diff.summary()}")

    start_step = 0
    if args.resume is not None:
        start_step, resumed_val = _load_ckpt(args.resume, model, ema, optim, device)
        print(f"\n  resumed from {args.resume}  step={start_step}  val_loss={resumed_val:.4f}")
        log({"step": start_step, "phase": "resume",
             "path": str(args.resume), "val_loss": resumed_val})

    # --- training loop ---
    train_iter = _cycle(train_loader)
    train_losses_recent: list[float] = []
    best_val = float("inf")
    last_val = float("nan")
    eval_history: list[dict[str, float]] = []
    final_diag_metrics: dict[str, float] | None = None
    final_gen_counts: np.ndarray | None = None

    max_steps = int(profile["max_steps"])
    eval_every = int(profile["eval_every"])
    save_every = int(profile["save_every"])
    warmup = int(profile["warmup_steps"])

    print("\n--- training -------------------------------------------------")
    t_run = time.time()
    for step in range(start_step + 1, max_steps + 1):
        model.train()
        lr = _get_lr(step, warmup, base_lr)
        _set_lr(optim, lr)

        x, cond = next(train_iter)
        x = x.to(device, non_blocking=True)
        hour = cond["hour"].to(device)
        dow = cond["day_of_week"].to(device)
        is_weekend = cond["is_weekend"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss = diff.training_loss(
                model, x, hour, dow, is_weekend,
                cond_dropout_prob=float(profile["cond_dropout_prob"]),
            )
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
        train_losses_recent.append(loss_val)
        if not np.isfinite(loss_val):
            raise RuntimeError(f"non-finite train loss at step {step}: {loss_val}")
        if not np.isfinite(gn_val):
            raise RuntimeError(f"non-finite grad_norm at step {step}: {gn_val}")

        log({
            "step": step, "phase": "train",
            "loss": loss_val, "grad_norm": gn_val, "lr": lr, "dt_ms": dt_ms,
        })
        writer.add_scalar("train/loss", loss_val, step)
        writer.add_scalar("train/grad_norm", gn_val, step)
        writer.add_scalar("train/lr", lr, step)
        writer.add_scalar("train/dt_ms", dt_ms, step)

        # Periodic console line (every step for small max_steps, every 50 otherwise).
        if max_steps <= 50 or step % 50 == 0 or step == 1:
            print(
                f"  step {step:>6}/{max_steps}: loss={loss_val:.4f}  "
                f"grad_norm={gn_val:.4f}  lr={lr:.2e}  dt={dt_ms:.0f} ms"
            )

        # Eval.
        if step % eval_every == 0:
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
            writer.add_scalar("val/loss_ema", last_val, step)
            eval_history.append({"step": float(step), "val_loss_ema": last_val})
            print(f"  step {step:>6}: val_loss_ema={last_val:.4f}")

        # In-loop sample diagnostics + checkpoint save.
        if step % save_every == 0:
            do_diag = (not profile["diag_at_end_only"]) or (step == max_steps)
            diag_metrics: dict[str, float] | None = None
            if do_diag:
                diag_metrics, gen_counts = _sample_diag(
                    model, diff, ema, val_loader, train_base, real_val_counts,
                    n_samples=int(profile["n_samples_diag"]),
                    chunk_size=int(profile["batch_size"]),
                    pad_size=train_base.pad_size, device=device,
                    num_inference_steps=int(profile["num_inference_steps"]),
                    guidance_scale=float(profile["guidance_scale"]),
                    seed=seed,
                )
                for k, v in diag_metrics.items():
                    writer.add_scalar(f"sample/{k}", v, step)
                log({"step": step, "phase": "sample_diag", **diag_metrics})
                print(
                    f"  step {step:>6}: sample_diag nonzero_ratio="
                    f"{diag_metrics['gen_nonzero_ratio']:.4%}  "
                    f"row_ks={diag_metrics['row_sum_ks_stat']:.3f}  "
                    f"col_ks={diag_metrics['col_sum_ks_stat']:.3f}"
                )
                final_diag_metrics = diag_metrics
                final_gen_counts = gen_counts

            _save_ckpt(run_dir / "last.pt", model, ema, optim, step, last_val, cfg, profile_name)
            is_best = last_val < best_val
            if is_best:
                best_val = last_val
                _save_ckpt(run_dir / "best.pt", model, ema, optim, step, last_val, cfg, profile_name)
            log({
                "step": step, "phase": "save",
                "ckpt": "last.pt", "best": is_best, "val_loss": last_val,
            })
            tag = " (best)" if is_best else ""
            print(f"  step {step:>6}: saved last.pt{tag}")

    wall = time.time() - t_run
    print(f"\n--- training complete  ({wall:.1f} s, {wall/60:.1f} min) ---")

    # --- end-of-run artifacts ---
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_loss_curve(log_path, out_dir / "loss_curve.png")
    if final_gen_counts is not None:
        _plot_sample_grid(real_val_counts, final_gen_counts, out_dir / "sample_grid.png")
        _plot_marginal_match(real_val_counts, final_gen_counts, out_dir / "marginal_match.png")

    metrics = {
        "profile": profile_name,
        "max_steps": max_steps,
        "final_val_loss_ema": last_val,
        "best_val_loss_ema": best_val,
        "train_loss_last": float(train_losses_recent[-1]) if train_losses_recent else None,
        "eval_history": eval_history,
        "final_sample_diag": final_diag_metrics,
        "wall_time_s": wall,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Copy the log into the tracked results dir for provenance.
    (out_dir / "train_log.jsonl").write_text(log_path.read_text())

    # --- summary console ---
    print("\n--- artifacts ------------------------------------------------")
    print(f"  checkpoints under {run_dir} (gitignored):")
    for p in sorted(run_dir.iterdir()):
        size = p.stat().st_size
        print(f"    {p.name:<24} {size / 1e6:>7.3f} MB")
    print(f"  tracked artifacts under {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        size = p.stat().st_size
        print(f"    {p.name:<24} {size / 1e6:>7.3f} MB")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e6
        reserved = torch.cuda.max_memory_reserved() / 1e6
        print(f"\n  gpu peak allocated   : {peak:.1f} MB")
        print(f"  gpu peak reserved    : {reserved:.1f} MB")

    writer.close()
    print("\n" + "=" * 78)
    print(f"STAGE 4B-3 ({profile_name}) DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
