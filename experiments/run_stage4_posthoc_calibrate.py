"""Stage 4B-5C PR5C-1B: posthoc calibration evaluation of a diffusion checkpoint.

Loads a Stage-4B diffusion checkpoint, draws ``--n-samples`` continuous
OD slices with EMA weights + DDIM, then fits a ``(tau, scale)``
posthoc calibrator on the TRAIN aggregate only and reports
before/after metrics against train, val, and test aggregates.

Strict scope (PR5C-1B):

  * No training, no medium.
  * No ``data/synthetic/od_samples_agg.npy`` write (frozen Stage-5 input).
  * No ``data/synthetic/od_samples_agg_diffusion_calibrated.npy`` write.
  * No modification of ``data/synthetic/od_samples_agg_bootstrap.npy``.
  * No PR5C-3 entry. No Stage-5 code changes.

The output of this script is a diagnostic report under
``--output-dir`` (default ``results/stage4/posthoc_calibration_zpin_weighted/``):
``metrics.json``, ``metrics.csv``, ``calibration_grid.png``,
``marginal_match_before_after.png``, ``decision_report.md``.

Usage
-----

    python -m experiments.run_stage4_posthoc_calibrate \\
        --config configs/diffusion_12km_zpin_weighted.yaml \\
        --ckpt models/diffusion_od_pilot_zpin_weighted/best.pt \\
        --profile pilot \\
        --output-dir results/stage4/posthoc_calibration_zpin_weighted \\
        --n-samples 48 --seed 42 --num-inference-steps 50 \\
        --tau-grid-start 0.1 --tau-grid-end 2.0 --tau-grid-size 20 \\
        --scale-grid-start 1e-3 --scale-grid-end 2.0 --scale-grid-size 20 \\
        --lambda-mass 1.0
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
from src.eval.posthoc_calibration import (
    DEFAULT_FAILED_DIFFUSION_BASELINE,
    acceptance_verdict,
    apply_threshold_and_scale,
    evaluate_calibrated_samples,
    grid_search_tau_scale,
)
from src.models.diffusion import GaussianDiffusion
from src.models.ema import EMA

REPO = Path(__file__).resolve().parents[1]
FROZEN_AGG_PATH = REPO / "data" / "synthetic" / "od_samples_agg.npy"
FROZEN_4D_PATH = REPO / "data" / "synthetic" / "od_samples.npy"
CALIBRATED_CANDIDATE_PATH = (
    REPO / "data" / "synthetic" / "od_samples_agg_diffusion_calibrated.npy"
)
BOOTSTRAP_CANDIDATE_PATH = (
    REPO / "data" / "synthetic" / "od_samples_agg_bootstrap.npy"
)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _collect_slot_conditions(
    val_base: ODDataset, n_samples: int
) -> tuple[list[int], list[int], list[int]]:
    """Pull ``n_samples`` (hour, day_of_week, is_weekend) tuples from val.

    The val split has 48 slots (one day at the 12 km configuration), so
    ``n_samples=48`` covers the whole hour-grid exactly once.
    """
    hours: list[int] = []
    dows: list[int] = []
    is_wks: list[int] = []
    while len(hours) < n_samples:
        for idx in range(len(val_base)):
            _, cond = val_base[idx]
            hours.append(int(cond["hour"]))
            dows.append(int(cond["day_of_week"]))
            is_wks.append(int(cond["is_weekend"]))
            if len(hours) >= n_samples:
                break
    return hours[:n_samples], dows[:n_samples], is_wks[:n_samples]


def _sample_continuous(
    model: torch.nn.Module,
    diff: GaussianDiffusion,
    ema: EMA,
    train_base: ODDataset,
    *,
    hours: list[int],
    dows: list[int],
    is_wks: list[int],
    batch_size: int,
    pad_size: int,
    device: torch.device,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> np.ndarray:
    """Draw ``len(hours)`` continuous, inverse-transformed OD slices.

    Returns ``[N, Z, Z]`` float64 -- ``inverse_transform`` already
    un-pads to the canonical ``[Z, Z]`` shape.
    """
    n_samples = len(hours)
    ema.store(model)
    ema.copy_to(model)
    model.eval()
    torch.manual_seed(seed)
    gen_chunks: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                n_c = end - start
                hour_t = torch.tensor(hours[start:end], device=device, dtype=torch.long)
                dow_t = torch.tensor(dows[start:end], device=device, dtype=torch.long)
                iw_t = torch.tensor(is_wks[start:end], device=device, dtype=torch.long)
                shape = (n_c, 1, pad_size, pad_size)
                samples = diff.ddim_sample(
                    model,
                    shape,
                    hour_t,
                    dow_t,
                    iw_t,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
                samples_np = samples.detach().cpu().numpy()
                for i in range(n_c):
                    counts = train_base.inverse_transform(samples_np[i])  # [1, Z, Z]
                    gen_chunks.append(counts[0])
    finally:
        ema.restore(model)
    return np.stack(gen_chunks, axis=0)  # [N, Z, Z] float64


def _real_aggregate(od: np.ndarray, slots: list[int]) -> np.ndarray:
    """Sum the integer OD tensor across ``slots`` -- int64 [Z, Z]."""
    z = od.shape[-1]
    acc = np.zeros((z, z), dtype=np.int64)
    for s in slots:
        acc += np.asarray(od[s], dtype=np.int64)
    return acc


def _plot_calibration_grid(
    grid: list[dict[str, float]],
    tau_grid: np.ndarray,
    scale_grid: np.ndarray,
    best: dict[str, float],
    out_path: Path,
) -> None:
    """Heatmap of objective(tau, scale) with the argmin marked."""
    n_tau = tau_grid.size
    n_scale = scale_grid.size
    Z = np.array([g["objective"] for g in grid], dtype=np.float64).reshape(n_tau, n_scale)
    fig, ax = plt.subplots(figsize=(7, 5))
    # log-scale objective for readability: clip to avoid log(0).
    Z_plot = np.log10(np.clip(Z, 1e-6, None))
    im = ax.imshow(
        Z_plot,
        aspect="auto",
        origin="lower",
        extent=[
            float(scale_grid[0]),
            float(scale_grid[-1]),
            float(tau_grid[0]),
            float(tau_grid[-1]),
        ],
        cmap="viridis",
    )
    ax.set_xscale("log")
    ax.set_xlabel("scale (log-axis)")
    ax.set_ylabel("tau")
    ax.set_title("posthoc calibration objective (log10)")
    fig.colorbar(im, ax=ax, label="log10(objective)")
    ax.plot(best["scale"], best["tau"], marker="*", color="red", markersize=18,
            markeredgecolor="white", label="argmin")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_marginal_match(
    real_test: np.ndarray,
    samples_cont: np.ndarray,
    best_tau: float,
    best_scale: float,
    out_path: Path,
) -> None:
    """row/col-sum histograms: real_test vs uncalibrated vs calibrated."""
    before = apply_threshold_and_scale(samples_cont, tau=0.0, scale=1.0).astype(np.float64)
    after = apply_threshold_and_scale(samples_cont, tau=best_tau, scale=best_scale).astype(np.float64)

    real_row = real_test.sum(axis=-1).ravel().astype(np.float64)
    real_col = real_test.sum(axis=-2).ravel().astype(np.float64)
    # Pool per-sample row/col sums into one distribution for fair shape comparison.
    before_row = before.sum(axis=-1).ravel()
    before_col = before.sum(axis=-2).ravel()
    after_row = after.sum(axis=-1).ravel()
    after_col = after.sum(axis=-2).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (r, b, a, title) in zip(
        axes,
        [
            (real_row, before_row, after_row, "row sums"),
            (real_col, before_col, after_col, "col sums"),
        ],
    ):
        combined = np.concatenate([r, b, a])
        lo, hi = float(combined.min()), float(combined.max())
        if hi - lo < 1e-9:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 50)
        ax.hist(r, bins=bins, alpha=0.5, density=True, label="real_test", color="C0")
        ax.hist(b, bins=bins, alpha=0.5, density=True, label="before (clip+round)", color="C1")
        ax.hist(a, bins=bins, alpha=0.5, density=True,
                label=f"after (tau={best_tau:.3f}, scale={best_scale:.3g})", color="C3")
        ax.set_title(title)
        ax.set_xlabel("sum (counts)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("PR5C-1B marginals: real_test vs uncalibrated vs calibrated",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _flatten_metric_dict(m: dict[str, Any]) -> dict[str, float]:
    """Strip per-sample arrays so the dict is JSON-serializable."""
    out: dict[str, float] = {}
    for k, v in m.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (int, float, bool, str)):
            out[k] = v
    return out


def _write_csv(
    metrics_by_split: dict[str, dict[str, dict[str, Any]]],
    out_path: Path,
) -> None:
    """One row per (phase=before|after, split=train|val|test)."""
    fields = [
        "phase",
        "split",
        "tau",
        "scale",
        "n_samples",
        "real_nonzero_ratio",
        "real_total_mass",
        "real_entropy",
        "nonzero_ratio_x_real_mean",
        "total_mass_ratio_mean",
        "row_sum_ks_stat_mean",
        "col_sum_ks_stat_mean",
        "top20_pair_overlap_mean",
        "top20_pair_overlap_against_top50_mean",
        "gen_max_mean",
        "gen_mean_mean",
        "entropy_mean",
    ]
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for phase in ("before", "after"):
            for split in ("train", "val", "test"):
                m = metrics_by_split[phase][split]
                row = []
                for f in fields:
                    if f == "phase":
                        row.append(phase)
                    elif f == "split":
                        row.append(split)
                    else:
                        v = m.get(f, "")
                        if isinstance(v, float):
                            row.append(f"{v:.6f}")
                        else:
                            row.append(str(v))
                w.writerow(row)


def _decision_report(
    args: argparse.Namespace,
    ckpt_info: dict[str, Any],
    samples_summary: dict[str, Any],
    best_tau: float,
    best_scale: float,
    best_obj: float,
    metrics_by_split: dict[str, dict[str, dict[str, Any]]],
    test_verdict: str,
    out_path: Path,
) -> None:
    """Human-readable markdown verdict."""
    baseline = DEFAULT_FAILED_DIFFUSION_BASELINE
    def fmt(m: dict[str, Any], key: str) -> str:
        v = m.get(f"{key}_mean", m.get(key, float("nan")))
        return f"{float(v):.4f}"

    lines = [
        "# Stage 4B-5C PR5C-1B posthoc calibration report",
        "",
        f"- config: `{_rel(args.config)}`",
        f"- ckpt: `{_rel(args.ckpt)}`  step={ckpt_info['step']}  "
        f"val_loss={ckpt_info['val_loss']:.4f}",
        f"- profile: {args.profile}  num_inference_steps={args.num_inference_steps}  "
        f"guidance_scale={ckpt_info['guidance_scale']}",
        f"- n_samples: {args.n_samples}  seed: {args.seed}",
        "",
        "## Sampling summary",
        f"- shape: {samples_summary['shape']}  dtype: {samples_summary['dtype']}",
        f"- min: {samples_summary['min']:.4f}  max: {samples_summary['max']:.4f}  "
        f"mean: {samples_summary['mean']:.4f}",
        f"- sample wall time: {samples_summary['wall_s']:.1f} s",
        "",
        "## Calibration fit (train aggregate target only)",
        f"- best_tau: **{best_tau:.4f}**",
        f"- best_scale: **{best_scale:.6g}**",
        f"- train objective at argmin: {best_obj:.6f}",
        f"- test untouched for fitting: **True**",
        f"- objective: `|nz_ratio/real_nz - 1| + lambda_mass * |total_mass/real_total - 1|` "
        f"(lambda_mass = {args.lambda_mass})",
        "",
        "## Before vs after by split",
        "",
        "### before (clip → round on continuous samples; tau=0, scale=1)",
        "",
        "| split | nz_x_real | mass_ratio | row_ks | col_ks | top20 | top20vs50 | gen_max | gen_mean |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        m = metrics_by_split["before"][split]
        lines.append(
            f"| {split} | {fmt(m, 'nonzero_ratio_x_real')} | "
            f"{fmt(m, 'total_mass_ratio')} | {fmt(m, 'row_sum_ks_stat')} | "
            f"{fmt(m, 'col_sum_ks_stat')} | {fmt(m, 'top20_pair_overlap')} | "
            f"{fmt(m, 'top20_pair_overlap_against_top50')} | "
            f"{fmt(m, 'gen_max')} | {fmt(m, 'gen_mean')} |"
        )
    lines += [
        "",
        "### after (apply_threshold_and_scale(samples_cont, best_tau, best_scale))",
        "",
        "| split | nz_x_real | mass_ratio | row_ks | col_ks | top20 | top20vs50 | gen_max | gen_mean |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        m = metrics_by_split["after"][split]
        lines.append(
            f"| {split} | {fmt(m, 'nonzero_ratio_x_real')} | "
            f"{fmt(m, 'total_mass_ratio')} | {fmt(m, 'row_sum_ks_stat')} | "
            f"{fmt(m, 'col_sum_ks_stat')} | {fmt(m, 'top20_pair_overlap')} | "
            f"{fmt(m, 'top20_pair_overlap_against_top50')} | "
            f"{fmt(m, 'gen_max')} | {fmt(m, 'gen_mean')} |"
        )
    test_after = metrics_by_split["after"]["test"]
    lines += [
        "",
        "## Acceptance verdict on TEST (after calibration)",
        f"- **{test_verdict.upper()}**",
        "",
        "PASS gates:",
        f"- nz_ratio_x_real in [0.7, 1.5]: {fmt(test_after, 'nonzero_ratio_x_real')}",
        f"- total_mass_ratio in [0.8, 1.2]: {fmt(test_after, 'total_mass_ratio')}",
        f"- top20_pair_overlap >= 12: {fmt(test_after, 'top20_pair_overlap')}",
        f"- row_sum_ks_stat <= 0.3: {fmt(test_after, 'row_sum_ks_stat')}",
        f"- col_sum_ks_stat <= 0.3: {fmt(test_after, 'col_sum_ks_stat')}",
        "",
        "MILD floor (must strictly beat failed-diffusion baseline on every axis):",
        f"- row_sum_ks_stat < {baseline['row_sum_ks_stat']}",
        f"- col_sum_ks_stat < {baseline['col_sum_ks_stat']}",
        f"- nonzero_ratio_x_real < {baseline['nonzero_ratio_x_real']}",
        f"- total_mass_ratio < {baseline['total_mass_ratio']}",
        "",
        "## Safety",
        f"- frozen `data/synthetic/od_samples_agg.npy` written: "
        f"{FROZEN_AGG_PATH.exists()}",
        f"- diffusion 4-D `data/synthetic/od_samples.npy` written: "
        f"{FROZEN_4D_PATH.exists()}",
        f"- candidate `data/synthetic/od_samples_agg_diffusion_calibrated.npy` "
        f"written: {CALIBRATED_CANDIDATE_PATH.exists()}",
        f"- bootstrap candidate `data/synthetic/od_samples_agg_bootstrap.npy` "
        f"modified by this run: False (script never opens it)",
    ]
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--tau-grid-start", type=float, default=0.1)
    parser.add_argument("--tau-grid-end", type=float, default=2.0)
    parser.add_argument("--tau-grid-size", type=int, default=20)
    parser.add_argument("--scale-grid-start", type=float, default=1e-3)
    parser.add_argument("--scale-grid-end", type=float, default=2.0)
    parser.add_argument("--scale-grid-size", type=int, default=20)
    parser.add_argument("--lambda-mass", type=float, default=1.0)
    parser.add_argument(
        "--save-continuous-debug",
        action="store_true",
        help="Save samples_cont to <output-dir>/samples_cont_debug.npy (default off).",
    )
    args = parser.parse_args()

    # --- safety guards: refuse to write protected paths ---------------
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == FROZEN_AGG_PATH.parent:
        raise SystemExit(
            f"refusing to write report into data/synthetic/; use a "
            f"results/ path. Got: {args.output_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- load config / profile / seed ---------------------------------
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    profile = dict(PROFILES[args.profile])
    yaml_diffusion = cfg.get("diffusion") or {}
    if "guidance_scale" in yaml_diffusion:
        profile["guidance_scale"] = float(yaml_diffusion["guidance_scale"])
    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("STAGE 4B-5C PR5C-1B  --  posthoc calibration evaluation")
    print("=" * 70)
    print(f"  config        : {args.config}")
    print(f"  ckpt          : {args.ckpt}")
    print(f"  profile       : {args.profile}")
    print(f"  device        : {device}")
    print(f"  guidance      : {profile['guidance_scale']}  "
          f"(YAML override of profile default 2.0 when applicable)")
    print(f"  n_samples     : {args.n_samples}")
    print(f"  seed          : {args.seed}")
    print(f"  inference t   : {args.num_inference_steps}")
    print(f"  output_dir    : {args.output_dir}")

    # --- datasets (12 km tensor with zpin scheme) ---------------------
    od_path = _resolve(cfg["input"]["od_path"])
    meta_path = _resolve(cfg["input"]["meta_path"])
    data_scheme = (cfg.get("data") or {}).get("scheme", "global_clip")
    split_cfg = cfg["data"]["split"]

    train_base = ODDataset(
        od_path, meta_path, "train",
        window=cfg["data"]["window"],
        pad_multiple=cfg["data"]["pad_multiple"],
        clip_val=cfg["data"]["clip_val"],
        scheme=data_scheme,
        split_cfg=split_cfg,
    )
    val_base = ODDataset(
        od_path, meta_path, "val",
        window=cfg["data"]["window"],
        pad_multiple=cfg["data"]["pad_multiple"],
        scheme=data_scheme,
        norm_stats=train_base.norm_stats,
        split_cfg=split_cfg,
    )

    # --- model + EMA + ckpt -------------------------------------------
    model = _build_unet(
        profile,
        int(cfg["data"]["window"]),
        int(cfg["model"]["time_emb_dim"]),
        device,
    )
    diff = GaussianDiffusion(
        num_train_timesteps=int(profile["num_train_timesteps"]),
        beta_schedule=cfg["diffusion"]["beta_schedule"],
    ).to(device)
    ema = EMA(model, decay=float(profile["ema_decay"])).to(device)
    ckpt = torch.load(args.ckpt, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    ema.load_state_dict(ckpt["ema_state_dict"])
    print(f"\n  loaded ckpt   : step={ckpt['step']}  val_loss={ckpt['val_loss']:.4f}")
    print(f"  data.scheme   : {data_scheme}  "
          f"pad_size={train_base.pad_size}  Z={train_base.n_zones}")

    # --- conditions + sampling ----------------------------------------
    hours, dows, is_wks = _collect_slot_conditions(val_base, args.n_samples)
    t0 = time.time()
    samples_cont = _sample_continuous(
        model, diff, ema, train_base,
        hours=hours, dows=dows, is_wks=is_wks,
        batch_size=int(profile["batch_size"]),
        pad_size=train_base.pad_size,
        device=device,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=float(profile["guidance_scale"]),
        seed=args.seed,
    )
    sample_dt = time.time() - t0
    samples_summary = {
        "shape": list(samples_cont.shape),
        "dtype": str(samples_cont.dtype),
        "min": float(samples_cont.min()),
        "max": float(samples_cont.max()),
        "mean": float(samples_cont.mean()),
        "wall_s": sample_dt,
    }
    print(f"\n  sampled       : shape={samples_cont.shape}  "
          f"min={samples_summary['min']:.4f}  max={samples_summary['max']:.4f}  "
          f"mean={samples_summary['mean']:.4f}  wall={sample_dt:.1f}s")

    if args.save_continuous_debug:
        debug_path = args.output_dir / "samples_cont_debug.npy"
        np.save(debug_path, samples_cont.astype(np.float32))
        print(f"  debug saved   : {debug_path}")

    # --- real aggregates (train fit + val/test reporting) -------------
    od = np.load(od_path, mmap_mode="r")
    spd = 24 * 60 // int(cfg.get("data", {}).get("time_bin_min", 30)) \
        if "time_bin_min" in (cfg.get("data") or {}) \
        else 48
    # Build slot lists from the config day ranges (re-derive, do not
    # depend on the ODDataset internals).
    with open(meta_path) as fh:
        meta = json.load(fh)
    n_slots = int(meta["T"])
    bin_min = int(meta.get("time_bin_min", 30))
    spd = 24 * 60 // bin_min
    train_lo, train_hi = split_cfg["train_days"]
    val_lo, val_hi = split_cfg["val_days"]
    test_lo, test_hi = split_cfg["test_days"]
    train_slots = list(range(train_lo * spd, min(train_hi * spd, n_slots)))
    val_slots = list(range(val_lo * spd, min(val_hi * spd, n_slots)))
    test_slots = list(range(test_lo * spd, min(test_hi * spd, n_slots)))

    print(f"\n  train_slots   : {len(train_slots)}  "
          f"val_slots: {len(val_slots)}  test_slots: {len(test_slots)}")

    real_train_agg = _real_aggregate(od, train_slots)
    real_val_agg = _real_aggregate(od, val_slots)
    real_test_agg = _real_aggregate(od, test_slots)
    print(f"  real train_agg: nz={(real_train_agg != 0).mean():.6f}  "
          f"sum={int(real_train_agg.sum())}")
    print(f"  real val_agg  : nz={(real_val_agg != 0).mean():.6f}  "
          f"sum={int(real_val_agg.sum())}")
    print(f"  real test_agg : nz={(real_test_agg != 0).mean():.6f}  "
          f"sum={int(real_test_agg.sum())}")

    # --- grid search (TRAIN only) -------------------------------------
    tau_grid = np.linspace(args.tau_grid_start, args.tau_grid_end, args.tau_grid_size)
    scale_grid = np.geomspace(args.scale_grid_start, args.scale_grid_end, args.scale_grid_size)
    print(f"\n  grid          : {args.tau_grid_size} tau x "
          f"{args.scale_grid_size} scale = "
          f"{args.tau_grid_size * args.scale_grid_size} pts")
    t1 = time.time()
    best_tau, best_scale, best_metrics, grid = grid_search_tau_scale(
        samples_cont,
        real_train_agg,
        tau_grid=tau_grid,
        scale_grid=scale_grid,
        lambda_mass=args.lambda_mass,
    )
    grid_dt = time.time() - t1
    print(f"  best          : tau={best_tau:.4f}  scale={best_scale:.6g}  "
          f"obj={best_metrics['objective']:.6f}  ({grid_dt:.1f}s)")

    # --- before / after evaluation ------------------------------------
    metrics_by_split: dict[str, dict[str, dict[str, Any]]] = {
        "before": {},
        "after": {},
    }
    for phase, (tau, scale) in (
        ("before", (0.0, 1.0)),
        ("after", (float(best_tau), float(best_scale))),
    ):
        for split_name, ref in (
            ("train", real_train_agg),
            ("val", real_val_agg),
            ("test", real_test_agg),
        ):
            m = evaluate_calibrated_samples(samples_cont, ref, tau, scale)
            metrics_by_split[phase][split_name] = m

    # --- test verdict (after calibration) -----------------------------
    test_after = metrics_by_split["after"]["test"]
    test_verdict = acceptance_verdict(test_after)
    print(f"\n  TEST verdict  : {test_verdict.upper()}")

    # --- write artefacts ---------------------------------------------
    flat_metrics: dict[str, Any] = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "ckpt_step": int(ckpt["step"]),
        "ckpt_val_loss": float(ckpt["val_loss"]),
        "profile": args.profile,
        "guidance_scale": float(profile["guidance_scale"]),
        "num_inference_steps": int(args.num_inference_steps),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "samples_cont": samples_summary,
        "best_tau": float(best_tau),
        "best_scale": float(best_scale),
        "best_train_objective": float(best_metrics["objective"]),
        "lambda_mass": float(args.lambda_mass),
        "test_acceptance_verdict": test_verdict,
        "test_untouched_for_fitting": True,
        "real_train_agg": {
            "nonzero_ratio": float((real_train_agg != 0).mean()),
            "total_mass": int(real_train_agg.sum()),
            "n_slots": len(train_slots),
        },
        "real_val_agg": {
            "nonzero_ratio": float((real_val_agg != 0).mean()),
            "total_mass": int(real_val_agg.sum()),
            "n_slots": len(val_slots),
        },
        "real_test_agg": {
            "nonzero_ratio": float((real_test_agg != 0).mean()),
            "total_mass": int(real_test_agg.sum()),
            "n_slots": len(test_slots),
        },
        "failed_diffusion_baseline": DEFAULT_FAILED_DIFFUSION_BASELINE,
    }
    for phase in ("before", "after"):
        flat_metrics[phase] = {
            split: _flatten_metric_dict(metrics_by_split[phase][split])
            for split in ("train", "val", "test")
        }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(flat_metrics, indent=2, sort_keys=True)
    )
    _write_csv(metrics_by_split, args.output_dir / "metrics.csv")
    _plot_calibration_grid(
        grid, tau_grid, scale_grid, best_metrics,
        args.output_dir / "calibration_grid.png",
    )
    _plot_marginal_match(
        real_test_agg, samples_cont, best_tau, best_scale,
        args.output_dir / "marginal_match_before_after.png",
    )
    _decision_report(
        args,
        {
            "step": int(ckpt["step"]),
            "val_loss": float(ckpt["val_loss"]),
            "guidance_scale": float(profile["guidance_scale"]),
        },
        samples_summary,
        best_tau,
        best_scale,
        float(best_metrics["objective"]),
        metrics_by_split,
        test_verdict,
        args.output_dir / "decision_report.md",
    )

    # --- post-run safety verification --------------------------------
    print(f"\n  metrics.json  : {args.output_dir / 'metrics.json'}")
    print(f"  metrics.csv   : {args.output_dir / 'metrics.csv'}")
    print(f"  cal grid png  : {args.output_dir / 'calibration_grid.png'}")
    print(f"  marg match    : {args.output_dir / 'marginal_match_before_after.png'}")
    print(f"  report        : {args.output_dir / 'decision_report.md'}")
    print()
    print(f"  data/synthetic/od_samples_agg.npy exists: {FROZEN_AGG_PATH.exists()}")
    print(f"  data/synthetic/od_samples.npy exists:     {FROZEN_4D_PATH.exists()}")
    print(f"  data/synthetic/od_samples_agg_diffusion_calibrated.npy exists: "
          f"{CALIBRATED_CANDIDATE_PATH.exists()}")
    print(f"  data/synthetic/od_samples_agg_bootstrap.npy exists "
          f"(untouched by this script): {BOOTSTRAP_CANDIDATE_PATH.exists()}")

    print("\n" + "=" * 70)
    print(f"PR5C-1B posthoc calibration complete  --  TEST verdict: {test_verdict.upper()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
