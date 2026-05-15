"""Stage-1 EDA: 5 hand-picked plots + ydata-profiling HTML report.

Loads ``data/processed/orders_clean.parquet`` (the task-7 output) and
produces the artifacts listed under "EDA report" in
``docs/plan/stage1_data_cleaning.md``:

  results/stage1/plots/
    01_duration_before_after.{png,pdf}   # uses pre-filter re-run
    02_drive_km.{png,pdf}
    03_fare_yuan_log.{png,pdf}
    04_geo_vs_drive.{png,pdf}            # scatter on a 50k sample
    05_hourly_volume.{png,pdf}
  results/stage1/eda_summary.html        # ydata-profiling, sampled

Plot 1 needs the unfiltered ``duration_min`` distribution, so the
script re-runs the pre-filter pipeline (load -> fix_coords -> parse
-> derived). This adds ~15 s but keeps the EDA self-contained — no
extra intermediate parquet to track.

ydata-profiling on 4 M rows is too heavy, so the HTML is built from
a 200k random sample (seed=42). The hand-picked plots use the full
4 M rows (or 50k sample for the scatter).

Run: python -m experiments.run_stage1_eda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: WSL has no DISPLAY by default.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.clean import (
    add_derived_columns,
    apply_simple_renames,
    fix_coordinates,
    load_raw,
    parse_times,
)

REPO = Path(__file__).resolve().parents[1]
PARQUET = REPO / "data" / "processed" / "orders_clean.parquet"
RAW_CSV = REPO / "data" / "raw" / "suzhou_orders_7days.csv"
OUT_DIR = REPO / "results" / "stage1"
PLOTS_DIR = OUT_DIR / "plots"
HTML_PATH = OUT_DIR / "eda_summary.html"

PROFILE_SAMPLE_N = 200_000
SCATTER_SAMPLE_N = 50_000
RNG_SEED = 42

# Filter (b) thresholds, mirrored here for the "after"-shading on plot 1.
DUR_MIN, DUR_MAX = 2.0, 180.0


def _save(fig: plt.Figure, name: str) -> None:
    """Save fig as both PNG (300 dpi) and PDF (vector). Same stem."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _load_pre_filter_duration(raw_csv: Path) -> pd.Series:
    """Re-run pipeline up to derived columns to get unfiltered duration_min."""
    print(f"  re-loading raw CSV for pre-filter distribution ({raw_csv.name}) ...")
    t0 = time.time()
    df = load_raw(raw_csv, chunksize=500_000)
    df = fix_coordinates(df)
    df, _ = parse_times(df)
    df = add_derived_columns(df)
    df = apply_simple_renames(df)
    print(f"  pre-filter rows: {len(df):,}  ({time.time() - t0:.1f}s)")
    return df["duration_min"].dropna()


def plot_duration_before_after(after: pd.Series, before: pd.Series) -> None:
    """Two-panel hist: full pre-filter range (log y) above, filtered below."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Top: pre-filter, log-y so extreme outliers are visible.
    # Clip view to [-30, 360] just for axis legibility (raw min/max go far
    # beyond), and shade the [DUR_MIN, DUR_MAX] keep window.
    view_lo, view_hi = -30, 360
    axes[0].hist(
        before.clip(view_lo, view_hi),
        bins=200,
        color="#777777",
        edgecolor="none",
    )
    axes[0].axvspan(DUR_MIN, DUR_MAX, color="#1f77b4", alpha=0.12, label="keep window")
    axes[0].set_yscale("log")
    axes[0].set_xlim(view_lo, view_hi)
    axes[0].set_title(
        f"duration_min — before filter (log y, view clipped to [{view_lo}, {view_hi}])"
    )
    axes[0].set_xlabel("duration_min")
    axes[0].set_ylabel("count (log)")
    axes[0].legend(loc="upper right")

    # Bottom: post-filter (already in [2, 180]).
    axes[1].hist(after, bins=180, color="#1f77b4", edgecolor="none")
    axes[1].set_xlim(DUR_MIN - 2, DUR_MAX + 2)
    axes[1].set_title(f"duration_min — after filter (n = {len(after):,})")
    axes[1].set_xlabel("duration_min")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    _save(fig, "01_duration_before_after")


def plot_drive_km(after: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(after, bins=200, color="#2ca02c", edgecolor="none")
    ax.set_title(f"drive_km — after filter (n = {len(after):,})")
    ax.set_xlabel("drive_km")
    ax.set_ylabel("count")
    fig.tight_layout()
    _save(fig, "02_drive_km")


def plot_fare_yuan_log(after: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    # Log y; values are >0 by filter (f) so no zero-bin issues.
    ax.hist(after, bins=200, color="#d62728", edgecolor="none")
    ax.set_yscale("log")
    ax.set_title(f"fare_yuan — after filter, log y (n = {len(after):,})")
    ax.set_xlabel("fare_yuan")
    ax.set_ylabel("count (log)")
    fig.tight_layout()
    _save(fig, "03_fare_yuan_log")


def plot_geo_vs_drive(df: pd.DataFrame) -> None:
    """Scatter on a sample; reference line geo == drive (lower bound)."""
    rng = np.random.default_rng(RNG_SEED)
    if len(df) > SCATTER_SAMPLE_N:
        idx = rng.choice(len(df), size=SCATTER_SAMPLE_N, replace=False)
        sub = df.iloc[idx]
    else:
        sub = df

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(sub["drive_km"], sub["geo_dist_km"], s=1, alpha=0.15, color="#9467bd")
    lim = max(sub["drive_km"].max(), sub["geo_dist_km"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="geo = drive")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("drive_km")
    ax.set_ylabel("geo_dist_km")
    ax.set_title(f"geo_dist_km vs drive_km (sample n = {len(sub):,})")
    ax.legend()
    fig.tight_layout()
    _save(fig, "04_geo_vs_drive")


def plot_hourly_volume(df: pd.DataFrame) -> None:
    """Trip count per hour over the 7-day window, indexed by dep_time."""
    s = df.set_index("dep_time").sort_index()["order_id"].resample("1h").count()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(s.index, s.values, color="#ff7f0e", lw=1)
    ax.set_title("hourly trip volume — 7 days")
    ax.set_xlabel("hour")
    ax.set_ylabel("orders / hour")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "05_hourly_volume")


def write_profile_html(df: pd.DataFrame, out_path: Path) -> None:
    """Render ydata-profiling report on a sample (full data is too slow)."""
    from ydata_profiling import ProfileReport

    sample = df.sample(n=min(PROFILE_SAMPLE_N, len(df)), random_state=RNG_SEED)
    print(
        f"  profiling on sample n={len(sample):,} (full df is {len(df):,}); "
        "this takes ~1-2 min ..."
    )
    t0 = time.time()
    profile = ProfileReport(
        sample,
        title="eVTOL Stage 1 — orders_clean (200k sample)",
        minimal=True,
        explorative=False,
    )
    profile.to_file(out_path)
    print(f"  wrote {out_path}  ({time.time() - t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--raw-csv", type=Path, default=RAW_CSV)
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="skip the slow ydata-profiling step (plots only)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.parquet}")
    t0 = time.time()
    df = pd.read_parquet(args.parquet)
    print(f"  {len(df):,} rows in {time.time() - t0:.1f}s")

    print("[plot 1/5] duration_min — before vs after filter")
    duration_before = _load_pre_filter_duration(args.raw_csv)
    plot_duration_before_after(after=df["duration_min"], before=duration_before)
    del duration_before  # free ~30 MB before profiling step.

    print("[plot 2/5] drive_km")
    plot_drive_km(df["drive_km"])

    print("[plot 3/5] fare_yuan (log y)")
    plot_fare_yuan_log(df["fare_yuan"])

    print("[plot 4/5] geo_dist_km vs drive_km (scatter sample)")
    plot_geo_vs_drive(df[["drive_km", "geo_dist_km"]])

    print("[plot 5/5] hourly trip volume")
    plot_hourly_volume(df[["dep_time", "order_id"]])

    if args.no_profile:
        print("[profile] skipped (--no-profile)")
    else:
        print(f"[profile] -> {HTML_PATH}")
        write_profile_html(df, HTML_PATH)

    print(f"[total wall time: {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
