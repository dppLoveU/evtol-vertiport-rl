"""Stage 4B-5A: posthoc guidance / sampler ablation on a Stage 4B training checkpoint.

Reuses an EXISTING checkpoint (no training, no model changes) and sweeps
``guidance_scale`` to test whether the over-dense generation observed at
the medium profile's default ``guidance_scale=2.0`` can be recovered by
lowering the scale.

Background
----------
The 12 km medium failed run (``models/diffusion_od_medium_12km_failed_dense_debug``)
reaches a low val loss but the EMA samples have ``gen_nonzero_ratio`` ~ 68.43 %
versus a real 12 km val ratio of ~ 0.2642 % (rounded-count basis) -- ~259x
over-dense. The 15 km medium run shows the same symptom. Before invoking the
normalization / weighted-loss ablation, check whether classifier-free guidance
is the culprit: at high scales CFG is known to over-saturate.

Metrics per ``guidance_scale``
------------------------------
Computed via ``src.utils.metrics_dist.marginal_compare`` on ROUNDED integer
counts (matching the real-data reference), plus continuous-side aux fields:

  * ``gen_nonzero_ratio`` -- rounded-count basis (acceptance target).
  * ``real_val_nonzero_ratio`` -- the int reference, identical across scales.
  * ``gen_cont_nonzero_ratio`` -- continuous-float sanity check.
  * ``gen_min`` / ``gen_max`` / ``gen_mean`` -- rounded-count distribution.
  * ``row_sum_ks_stat`` / ``col_sum_ks_stat`` -- two-sample K-S.
  * real / gen row & col mean & std.
  * ``sampling_seconds`` -- wall-clock for the slice batch.
  * ``has_nan`` -- non-finite check on the continuous samples.

Outputs (under ``--output_dir``)
--------------------------------
  * ``metrics.csv`` -- one row per ``guidance_scale``.
  * ``metrics.json`` -- the same data plus run-level provenance.
  * ``sample_grid.png`` -- (optional) one ``log1p`` heatmap per scale,
    side-by-side with a real val slice.

Run
---
::

    python -m experiments.run_stage4_guidance_ablation \\
        --config configs/diffusion_12km.yaml \\
        --ckpt models/diffusion_od_medium_12km_failed_dense_debug/best.pt \\
        --profile medium \\
        --n_samples 64 \\
        --guidance_scales 0.0 0.5 1.0 1.5 2.0 \\
        --num_inference_steps 50 \\
        --output_dir results/stage4/guidance_ablation_12km_medium_failed
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from experiments.run_stage4_train import (
    PROFILES,
    _TorchODDataset,
    _build_unet,
    _set_seed,
)
from src.data.od_dataset import ODDataset
from src.models.diffusion import GaussianDiffusion
from src.models.ema import EMA
from src.utils.metrics_dist import marginal_compare

REPO = Path(__file__).resolve().parents[1]


# --- helpers --------------------------------------------------------------


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _real_val_counts_raw(val_base: ODDataset) -> np.ndarray:
    """Read the raw int OD slices for val days, without round-tripping
    through normalization. Matches the reference used by the in-loop
    sample diagnostics in run_stage4_train.py."""
    raw_list: list[np.ndarray] = []
    for start in val_base._starts:
        sl = np.asarray(val_base._od[start : start + val_base.window], dtype=np.int64)
        raw_list.append(sl[0])  # window=1
    return np.stack(raw_list)


# --- sampling for a single guidance scale ---------------------------------


def _sample_at_scale(
    model: torch.nn.Module,
    diff: GaussianDiffusion,
    hours: list[int],
    dows: list[int],
    is_wks: list[int],
    train_base: ODDataset,
    *,
    chunk: int,
    pad_size: int,
    guidance_scale: float,
    num_inference_steps: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Generate len(hours) slices at one guidance_scale; return continuous
    counts of shape [N, Z, Z] (float64) and wall-clock seconds.

    The model is assumed to already be in eval mode with EMA weights
    installed (caller responsibility -- saves a copy_to per scale).
    """
    n = len(hours)
    torch.manual_seed(seed)  # repeatable init noise across scales
    gen_chunks: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            n_c = end - start
            hour_t = torch.tensor(hours[start:end], device=device, dtype=torch.long)
            dow_t = torch.tensor(dows[start:end], device=device, dtype=torch.long)
            iw_t = torch.tensor(is_wks[start:end], device=device, dtype=torch.long)
            shape = (n_c, 1, pad_size, pad_size)
            samples = diff.ddim_sample(
                model, shape, hour_t, dow_t, iw_t,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            samples_np = samples.detach().cpu().numpy()
            for i in range(n_c):
                counts = train_base.inverse_transform(samples_np[i])  # [1, Z, Z]
                gen_chunks.append(counts[0])
    dt = time.time() - t0
    return np.stack(gen_chunks), dt


# --- plotting -------------------------------------------------------------


def _plot_sample_grid(
    real_slice: np.ndarray,
    gen_per_scale: dict[float, np.ndarray],
    out_path: Path,
) -> None:
    """One heatmap per guidance_scale (using sample index 0), plus the
    first real val slice for reference. log1p colormap matches the
    sample_grid.png convention from run_stage4_train.py."""
    scales = sorted(gen_per_scale)
    n = len(scales) + 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
    axes[0].imshow(np.log1p(real_slice), cmap="hot", aspect="auto")
    axes[0].set_title("real val[0]", fontsize=10)
    axes[0].axis("off")
    for ax, s in zip(axes[1:], scales):
        g = gen_per_scale[s][0]
        ax.imshow(np.log1p(np.maximum(g, 0.0)), cmap="hot", aspect="auto")
        ax.set_title(f"gen gs={s:.1f}", fontsize=10)
        ax.axis("off")
    fig.suptitle("OD slices (log1p): real vs generated across guidance scales", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --- main -----------------------------------------------------------------


CSV_COLUMNS = [
    "guidance_scale",
    "real_val_nonzero_ratio",
    "gen_nonzero_ratio",
    "gen_cont_nonzero_ratio",
    "gen_min",
    "gen_max",
    "gen_mean",
    "row_sum_ks_stat",
    "col_sum_ks_stat",
    "real_row_sum_mean",
    "real_row_sum_std",
    "gen_row_sum_mean",
    "gen_row_sum_std",
    "real_col_sum_mean",
    "real_col_sum_std",
    "gen_col_sum_mean",
    "gen_col_sum_std",
    "sampling_seconds",
    "has_nan",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs" / "diffusion.yaml",
        help="Stage-4 config YAML (input paths, norm/pad/split). For the "
             "12 km failed run use configs/diffusion_12km.yaml.",
    )
    parser.add_argument(
        "--ckpt", type=Path, required=True,
        help="Path to a saved training checkpoint (.pt).",
    )
    parser.add_argument(
        "--profile", default="medium", choices=sorted(PROFILES),
        help="Profile whose base_channels / channel_mults match the "
             "checkpoint's architecture (default medium).",
    )
    parser.add_argument(
        "--n_samples", type=int, default=64,
        help="Number of slices to draw per guidance scale (default 64).",
    )
    parser.add_argument(
        "--guidance_scales", type=float, nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
        help="Guidance scales to sweep. 1.0 disables CFG; <1.0 down-weights "
             "the conditional vs unconditional; 0.0 is pure unconditional.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="DDIM steps (default 50, matching the training profile).",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory for metrics.csv / metrics.json / sample_grid.png.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_grid", action="store_true",
        help="Skip sample_grid.png (saves a bit of time / disk).",
    )
    args = parser.parse_args()

    profile = dict(PROFILES[args.profile])
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"STAGE 4B-5A  --  posthoc guidance ablation")
    print("=" * 78)
    print(f"  device              : {device}")
    print(f"  ckpt                : {args.ckpt}")
    print(f"  config              : {args.config}")
    print(f"  profile             : {args.profile}")
    print(f"  n_samples           : {args.n_samples}")
    print(f"  guidance_scales     : {args.guidance_scales}")
    print(f"  num_inference_steps : {args.num_inference_steps}")
    print(f"  output_dir          : {args.output_dir}")
    print(f"  seed                : {args.seed}")

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
    val_loader = DataLoader(
        _TorchODDataset(val_base), batch_size=int(profile["batch_size"]),
        shuffle=False, num_workers=0,
    )
    print(f"\n  train samples       : {len(train_base)}")
    print(f"  val samples         : {len(val_base)}")
    print(f"  pad_size            : {train_base.pad_size}")

    real_int = _real_val_counts_raw(val_base)
    real_nz = float(np.count_nonzero(real_int)) / float(real_int.size)
    print(f"  real val nonzero    : {real_nz:.6%}  ({real_int.shape})")

    # --- model / diffusion / EMA ---
    model = _build_unet(profile, int(cfg["data"]["window"]),
                        int(cfg["model"]["time_emb_dim"]), device)
    diff = GaussianDiffusion(
        num_train_timesteps=int(profile["num_train_timesteps"]),
        beta_schedule=cfg["diffusion"]["beta_schedule"],
    ).to(device)
    ema = EMA(model, decay=float(profile["ema_decay"])).to(device)

    ckpt = torch.load(args.ckpt, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    ema.load_state_dict(ckpt["ema_state_dict"])
    print(
        f"\n  loaded step={ckpt['step']}  val_loss={ckpt['val_loss']:.4f}  "
        f"profile_name={ckpt.get('profile_name', '?')}"
    )

    # Install EMA weights once (all scales share the same weights, only the
    # sampler changes). store/restore at the very end keeps the model
    # symbol clean even though we exit the script right after.
    ema.store(model)
    ema.copy_to(model)
    model.eval()

    # --- conditions: take the first n_samples val slices' (hour, dow, is_weekend) ---
    hours: list[int] = []
    dows: list[int] = []
    is_wks: list[int] = []
    for x, cond in val_loader:
        hours.extend(cond["hour"].tolist())
        dows.extend(cond["day_of_week"].tolist())
        is_wks.extend(cond["is_weekend"].tolist())
        if len(hours) >= args.n_samples:
            break
    hours = hours[: args.n_samples]
    dows = dows[: args.n_samples]
    is_wks = is_wks[: args.n_samples]
    n_eff = len(hours)
    if n_eff < args.n_samples:
        print(
            f"  WARNING: val loader exhausted at {n_eff} samples "
            f"(< requested {args.n_samples})"
        )

    # --- sweep ---
    results: list[dict[str, Any]] = []
    gen_per_scale: dict[float, np.ndarray] = {}
    chunk = int(profile["batch_size"])
    print("\n--- sweeping guidance scales ---")
    for gs in args.guidance_scales:
        print(f"  guidance_scale={gs:.2f} ...", flush=True)
        gen_cont, dt = _sample_at_scale(
            model, diff,
            hours, dows, is_wks,
            train_base,
            chunk=chunk, pad_size=train_base.pad_size,
            guidance_scale=float(gs),
            num_inference_steps=int(args.num_inference_steps),
            device=device, seed=args.seed,
        )
        has_nan = bool(not np.all(np.isfinite(gen_cont)))
        gen_round = np.rint(np.maximum(gen_cont, 0.0)).astype(np.int64)
        m = marginal_compare(real_int, gen_round)
        row: dict[str, Any] = {
            "guidance_scale": float(gs),
            "real_val_nonzero_ratio": float(m["real_nonzero_ratio"]),
            "gen_nonzero_ratio": float(m["gen_nonzero_ratio"]),
            "gen_cont_nonzero_ratio": float(
                np.count_nonzero(gen_cont)
            ) / float(gen_cont.size),
            "gen_min": float(m["gen_min"]),
            "gen_max": float(m["gen_max"]),
            "gen_mean": float(m["gen_mean"]),
            "row_sum_ks_stat": float(m["row_sum_ks_stat"]),
            "col_sum_ks_stat": float(m["col_sum_ks_stat"]),
            "real_row_sum_mean": float(m["real_row_sum_mean"]),
            "real_row_sum_std": float(m["real_row_sum_std"]),
            "gen_row_sum_mean": float(m["gen_row_sum_mean"]),
            "gen_row_sum_std": float(m["gen_row_sum_std"]),
            "real_col_sum_mean": float(m["real_col_sum_mean"]),
            "real_col_sum_std": float(m["real_col_sum_std"]),
            "gen_col_sum_mean": float(m["gen_col_sum_mean"]),
            "gen_col_sum_std": float(m["gen_col_sum_std"]),
            "sampling_seconds": float(dt),
            "has_nan": has_nan,
        }
        results.append(row)
        gen_per_scale[float(gs)] = gen_cont
        print(
            f"    gen_nonzero={row['gen_nonzero_ratio']:.4%}  "
            f"row_ks={row['row_sum_ks_stat']:.3f}  "
            f"col_ks={row['col_sum_ks_stat']:.3f}  "
            f"gen_max={row['gen_max']:.2f}  "
            f"dt={dt:.1f}s  nan={has_nan}"
        )

    # Restore non-EMA weights for hygiene (script will exit anyway).
    ema.restore(model)

    # --- write metrics.csv / metrics.json ---
    csv_path = args.output_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in CSV_COLUMNS})

    json_payload = {
        "ckpt": str(args.ckpt),
        "step": int(ckpt["step"]),
        "val_loss": float(ckpt["val_loss"]),
        "profile": args.profile,
        "config": str(args.config),
        "n_samples": int(n_eff),
        "num_inference_steps": int(args.num_inference_steps),
        "seed": int(args.seed),
        "guidance_scales": [float(s) for s in args.guidance_scales],
        "real_val_nonzero_ratio": real_nz,
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(json_payload, indent=2))

    if not args.no_grid:
        _plot_sample_grid(real_int[0], gen_per_scale, args.output_dir / "sample_grid.png")

    # --- pretty console summary ---
    print("\n--- summary --------------------------------------------------------")
    hdr = (
        f"{'guidance':>8} {'gen_nz_round':>14} {'gen_nz_cont':>13} "
        f"{'gen_max':>10} {'row_ks':>8} {'col_ks':>8} {'dt_s':>7} {'nan':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['guidance_scale']:>8.2f} "
            f"{r['gen_nonzero_ratio']:>14.4%} "
            f"{r['gen_cont_nonzero_ratio']:>13.4%} "
            f"{r['gen_max']:>10.2f} "
            f"{r['row_sum_ks_stat']:>8.3f} "
            f"{r['col_sum_ks_stat']:>8.3f} "
            f"{r['sampling_seconds']:>7.1f} "
            f"{'Y' if r['has_nan'] else 'N':>5}"
        )
    print(f"\n  real_val_nonzero_ratio reference: {real_nz:.4%}")

    # Closest-to-real summary line for the immediate question.
    best = min(results, key=lambda r: abs(r["gen_nonzero_ratio"] - real_nz))
    ratio = (
        best["gen_nonzero_ratio"] / real_nz if real_nz > 0 else float("inf")
    )
    print(
        f"  closest scale         : gs={best['guidance_scale']:.2f}  "
        f"gen_nz={best['gen_nonzero_ratio']:.4%}  "
        f"({ratio:.1f}x real)"
    )

    print(f"\nwrote {csv_path}")
    print(f"wrote {args.output_dir / 'metrics.json'}")
    if not args.no_grid:
        print(f"wrote {args.output_dir / 'sample_grid.png'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
