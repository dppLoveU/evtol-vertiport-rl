"""Stage-3 tasks 7-8: eVTOL share statistics and EDA figures.

Reads the OD tensors built in R2 plus the cleaned orders, computes the
eVTOL share-by-hour statistics (task 7), and renders the seven paper
figures (task 8) to results/stage3/eda/ as PNG (300 dpi) + PDF.

The ``15 km / 25 min`` eVTOL cut is a *provisional baseline*, not a
settled standard (see docs/decisions.md 2026-05-18); figure titles label
it as such.

Run:
    python -m experiments.run_stage3_eda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.eda_plots import apply_house_style, save_figure, zone_choropleth
from src.data.od import (
    assign_time_bin,
    assign_zones,
    build_zone_lookup,
    is_evtol_eligible,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "od.yaml"

# t0 = 2023-07-10 is a Monday, so day-of-window % 7 in {5, 6} is the
# weekend (Sat 2023-07-15, Sun 2023-07-16).
_WEEKEND_DOW = (5, 6)

_BASELINE_TAG = "provisional 15 km / 25 min baseline"

_ORDER_COLUMNS = [
    "o_lon",
    "o_lat",
    "d_lon",
    "d_lat",
    "dep_time",
    "geo_dist_km",
    "duration_min",
    "fare_yuan",
]


def _resolve(path_str: str) -> Path:
    """Resolve a yaml path relative to repo root if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _share_by_hour(
    full_per_slot: np.ndarray,
    evtol_per_slot: np.ndarray,
    hour: np.ndarray,
    slot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-hour (0-23) full count, eVTOL count, and eVTOL share."""
    full_h = np.zeros(24, dtype=np.int64)
    evtol_h = np.zeros(24, dtype=np.int64)
    for h in range(24):
        m = (hour == h) & slot_mask
        full_h[h] = int(full_per_slot[m].sum())
        evtol_h[h] = int(evtol_per_slot[m].sum())
    share = np.divide(
        evtol_h, full_h, out=np.zeros(24, dtype=np.float64), where=full_h > 0
    )
    return full_h, evtol_h, share


def run(cfg: dict[str, Any]) -> None:
    """Compute share statistics and render the seven Stage-3 figures."""
    apply_house_style()

    out_cfg = cfg["output"]
    meta_path = _resolve(out_cfg["meta_path"])
    with open(meta_path) as fh:
        meta = json.load(fh)
    n_time_bins = int(meta["T"])

    eda_dir = _resolve(cfg["eda"]["output_dir"])
    eda_dir.mkdir(parents=True, exist_ok=True)

    od_full = np.load(_resolve(out_cfg["od_full_path"]))
    od_evtol = np.load(_resolve(out_cfg["od_evtol_path"]))
    od_weighted = np.load(_resolve(out_cfg["od_evtol_weighted_path"]), mmap_mode="r")
    print(
        f"[load] od_full {od_full.shape} {od_full.dtype}, "
        f"od_evtol {od_evtol.shape} {od_evtol.dtype}, "
        f"od_evtol_weighted {od_weighted.shape} (mmap, not plotted)"
    )

    written: list[Path] = []

    # === task 7: share statistics =====================================
    slots = np.arange(n_time_bins)
    hour = (slots // 2) % 24            # 30-min bins -> 2 slots per hour
    dow = (slots // 48) % 7            # 48 slots per day; 0 = Monday
    weekend_mask = np.isin(dow, _WEEKEND_DOW)
    weekday_mask = ~weekend_mask

    full_per_slot = od_full.sum(axis=(1, 2))
    evtol_per_slot = od_evtol.sum(axis=(1, 2))

    full_all, evtol_all, share_all = _share_by_hour(
        full_per_slot, evtol_per_slot, hour, np.ones(n_time_bins, dtype=bool)
    )
    _, _, share_wd = _share_by_hour(
        full_per_slot, evtol_per_slot, hour, weekday_mask
    )
    _, _, share_we = _share_by_hour(
        full_per_slot, evtol_per_slot, hour, weekend_mask
    )
    share_total = float(od_evtol.sum()) / float(od_full.sum())

    share_df = pd.DataFrame(
        {
            "hour": np.arange(24),
            "full_count": full_all,
            "evtol_count": evtol_all,
            "share": share_all,
            "share_weekday": share_wd,
            "share_weekend": share_we,
        }
    )
    share_csv = eda_dir / "share_by_hour.csv"
    share_df.to_csv(share_csv, index=False)
    print(f"[task7] share_total={share_total:.4%} -> {share_csv}")

    # Augment od_meta.json with the share-by-hour summary (existing
    # fields are preserved -- this only adds keys).
    meta["share_by_hour"] = [float(x) for x in share_all]
    meta["share_by_hour_weekday"] = [float(x) for x in share_wd]
    meta["share_by_hour_weekend"] = [float(x) for x in share_we]
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[task7] share_by_hour summary added to {meta_path}")

    # === task 8: figures ==============================================
    hours = np.arange(24)

    # --- 1. od_share_by_hour -----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hours, share_wd * 100, marker="o", label="Weekday")
    ax.plot(hours, share_we * 100, marker="s", label="Weekend")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("eVTOL-eligible share (%)")
    ax.set_title(f"eVTOL-eligible trip share by hour\n({_BASELINE_TAG})")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    written += save_figure(fig, eda_dir, "od_share_by_hour")

    # --- 2. top20_od_pairs -------------------------------------------
    od_pair = od_evtol.sum(axis=0)  # [Z, Z], time folded away
    n_zones = od_pair.shape[0]
    flat = od_pair.ravel()
    top_idx = np.argsort(flat)[::-1][:20]
    top_o = top_idx // n_zones
    top_d = top_idx % n_zones
    top_counts = flat[top_idx]
    labels = [f"z{o}→z{d}" for o, d in zip(top_o, top_d)]
    fig, ax = plt.subplots(figsize=(8, 6))
    ypos = np.arange(20)
    ax.barh(ypos, top_counts[::-1], color="C3")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("eVTOL trip count (11-day total)")
    ax.set_title(f"Top 20 eVTOL OD pairs by zone\n({_BASELINE_TAG})")
    written += save_figure(fig, eda_dir, "top20_od_pairs")

    # --- 3 & 4. heatmap_o_volume / heatmap_d_volume ------------------
    zones = gpd.read_file(_resolve(cfg["input"]["zones_path"]))
    zones = zones.sort_values("zone_id").reset_index(drop=True)
    o_volume = od_evtol.sum(axis=(0, 2))  # per origin zone
    d_volume = od_evtol.sum(axis=(0, 1))  # per destination zone
    fig = zone_choropleth(
        zones,
        o_volume,
        title=f"eVTOL origin volume per zone (11-day total)\n({_BASELINE_TAG})",
        legend_label="eVTOL trips originating",
    )
    written += save_figure(fig, eda_dir, "heatmap_o_volume")
    fig = zone_choropleth(
        zones,
        d_volume,
        title=f"eVTOL destination volume per zone (11-day total)\n({_BASELINE_TAG})",
        legend_label="eVTOL trips arriving",
    )
    written += save_figure(fig, eda_dir, "heatmap_d_volume")

    # --- 5. temporal_pattern -----------------------------------------
    # Two hourly curves (24 points, 11 days folded), full vs eVTOL, on a
    # twin axis since the magnitudes differ by ~20x.
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    l1 = ax1.plot(hours, full_all, marker="o", color="C0", label="All trips")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("All trips per hour (11-day total)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_xticks(range(0, 24, 2))
    ax2 = ax1.twinx()
    ax2.grid(False)
    l2 = ax2.plot(
        hours, evtol_all, marker="s", color="C1", label="eVTOL-eligible"
    )
    ax2.set_ylabel("eVTOL trips per hour (11-day total)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_title(f"Hourly trip volume: all vs eVTOL\n({_BASELINE_TAG})")
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
    written += save_figure(fig, eda_dir, "temporal_pattern")

    del od_full, od_evtol  # free ~1.2 GB before the order-level pass

    # --- order-level pass for distance / fare distributions ----------
    evtol_cfg = cfg["evtol_filter"]
    time_cfg = cfg["time"]
    min_dist = float(evtol_cfg["min_dist_km"])
    min_dur = float(evtol_cfg["min_duration_min"])

    orders = pd.read_parquet(
        _resolve(cfg["input"]["orders_path"]), columns=_ORDER_COLUMNS
    )
    in_range, _, _ = assign_time_bin(
        orders,
        t0=pd.Timestamp(time_cfg["start_datetime"]),
        time_bin_min=int(time_cfg["time_bin_min"]),
        num_time_bins=n_time_bins,
    )
    lookup = build_zone_lookup(zones)
    assigned, _, _ = assign_zones(in_range, lookup)
    mask = is_evtol_eligible(
        assigned, min_dist_km=min_dist, min_duration_min=min_dur
    )
    evtol_orders = assigned[mask]
    print(
        f"[orders] zone-assigned={len(assigned)}  eVTOL-eligible={len(evtol_orders)}"
    )

    # --- 6. distance_distribution ------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dist_bins = np.linspace(0, 100, 101)
    ax.hist(
        assigned["geo_dist_km"],
        bins=dist_bins,
        color="C0",
        alpha=0.6,
        label="All zone-assigned trips",
    )
    ax.hist(
        evtol_orders["geo_dist_km"],
        bins=dist_bins,
        color="C1",
        alpha=0.6,
        label="eVTOL-eligible trips",
    )
    ax.axvline(
        min_dist,
        color="C3",
        linestyle="--",
        linewidth=1.5,
        label=f"baseline cut-off {min_dist:.0f} km (provisional)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Trip geodesic distance (km)")
    ax.set_ylabel("Trip count (log scale)")
    ax.set_title("Trip distance distribution: all vs eVTOL-eligible")
    ax.legend()
    written += save_figure(fig, eda_dir, "distance_distribution")

    # --- 7. fare_distribution ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fare_hi = float(np.percentile(assigned["fare_yuan"], 99.5))
    fare_bins = np.linspace(0, fare_hi, 81)
    ax.hist(
        assigned["fare_yuan"].clip(upper=fare_hi),
        bins=fare_bins,
        color="C0",
        alpha=0.6,
        label="All zone-assigned trips",
    )
    ax.hist(
        evtol_orders["fare_yuan"].clip(upper=fare_hi),
        bins=fare_bins,
        color="C1",
        alpha=0.6,
        label="eVTOL-eligible trips",
    )
    ax.set_yscale("log")
    ax.set_xlabel(f"Fare (yuan, clipped at p99.5 = {fare_hi:.0f})")
    ax.set_ylabel("Trip count (log scale)")
    ax.set_title(f"Fare distribution: all vs eVTOL-eligible\n({_BASELINE_TAG})")
    ax.legend()
    written += save_figure(fig, eda_dir, "fare_distribution")

    print(f"[done] {len(written)} figure files written to {eda_dir}")
    for p in written:
        print(f"  {p.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to od yaml (default: {DEFAULT_CONFIG.relative_to(REPO)})",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    run(cfg)


if __name__ == "__main__":
    main()
