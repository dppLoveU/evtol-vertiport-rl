"""Stage-3 task 9: eVTOL distance-threshold sensitivity sweep.

Sweeps EVTOL_MIN_DIST_KM over the values in configs/od.yaml::sensitivity
and, for each, reports how many trips stay eVTOL-eligible and how
spatially concentrated the resulting OD distribution is. The duration
threshold and the 11-day time window are held fixed at their od.yaml
values; zone assignment uses the current 530 demand zones.

Does NOT rebuild or write any od_*.npy -- it only reads orders, applies
assign_time_bin / assign_zones / is_evtol_eligible, and writes a small
table to results/stage3/sensitivity.csv (results/ is gitignored).

The OD-concentration metrics fold the time axis away: trips are grouped
by `(o_zone, d_zone)` only, and Shannon entropy (natural log, nats) is
computed over that OD count distribution. `normalized_od_entropy` is
`od_entropy / ln(nonzero_od_pairs)` -- 1.0 = uniform spread, → 0 = all
demand on one OD pair.

Run:
    python -m experiments.run_stage3_sensitivity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from src.data.od import (
    assign_time_bin,
    assign_zones,
    build_zone_lookup,
    is_evtol_eligible,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "od.yaml"

# Max allowed gap between the 15 km sweep row and od_meta.json's share.
META_TOLERANCE = 0.001  # 0.1 percentage point

_CSV_COLUMNS = [
    "min_dist_km",
    "min_duration_min",
    "eligible_trip_count",
    "eligible_share_of_zone_assigned",
    "eligible_share_of_in_window",
    "nonzero_od_pairs",
    "top_od_pair_share",
    "od_entropy",
    "normalized_od_entropy",
]

_READ_COLUMNS = [
    "o_lon",
    "o_lat",
    "d_lon",
    "d_lat",
    "dep_time",
    "geo_dist_km",
    "duration_min",
]


def _resolve(path_str: str) -> Path:
    """Resolve a yaml path relative to repo root if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _od_concentration(eligible_df: pd.DataFrame) -> dict[str, float | int]:
    """OD-distribution concentration metrics for an eligible-trip frame.

    Trips are grouped by ``(o_zone, d_zone)`` only (time axis folded
    away). Entropy uses the natural log.
    """
    od_counts = eligible_df.groupby(["o_zone", "d_zone"], sort=False).size()
    total = int(od_counts.sum())
    nonzero = int(len(od_counts))
    if total == 0:
        return {
            "nonzero_od_pairs": 0,
            "top_od_pair_share": 0.0,
            "od_entropy": 0.0,
            "normalized_od_entropy": 0.0,
        }
    p = od_counts.to_numpy(dtype=np.float64) / total
    entropy = float(-np.sum(p * np.log(p)))
    norm = entropy / np.log(nonzero) if nonzero > 1 else 0.0
    return {
        "nonzero_od_pairs": nonzero,
        "top_od_pair_share": float(od_counts.max() / total),
        "od_entropy": entropy,
        "normalized_od_entropy": norm,
    }


def run(cfg: dict[str, Any]) -> pd.DataFrame:
    """Execute the distance-threshold sweep. Returns the result table."""
    orders_path = _resolve(cfg["input"]["orders_path"])
    zones_path = _resolve(cfg["input"]["zones_path"])
    time_cfg = cfg["time"]
    evtol_cfg = cfg["evtol_filter"]
    sens_cfg = cfg["sensitivity"]
    meta_path = _resolve(cfg["output"]["meta_path"])

    start = pd.Timestamp(time_cfg["start_datetime"])
    bin_min = int(time_cfg["time_bin_min"])
    n_bins = int(time_cfg["num_time_bins"])
    min_dur = float(evtol_cfg["min_duration_min"])
    dist_sweep = [float(d) for d in sens_cfg["dist_km_sweep"]]

    # --- read + time bin + zone assignment (done once) ----------------
    df = pd.read_parquet(orders_path, columns=_READ_COLUMNS)
    in_range, _, _ = assign_time_bin(
        df, t0=start, time_bin_min=bin_min, num_time_bins=n_bins
    )
    n_in_window = len(in_range)

    zones = gpd.read_file(zones_path)
    lookup = build_zone_lookup(zones)
    assigned, _, zstats = assign_zones(in_range, lookup)
    n_zone_assigned = int(zstats["n_zone_assigned"])

    print(
        f"[base] n_in_window={n_in_window}  n_zone_assigned={n_zone_assigned}  "
        f"min_duration_min={min_dur}"
    )

    # --- sweep --------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for dist in dist_sweep:
        mask = is_evtol_eligible(
            assigned, min_dist_km=dist, min_duration_min=min_dur
        )
        eligible = assigned[mask]
        count = int(mask.sum())
        conc = _od_concentration(eligible)
        row = {
            "min_dist_km": dist,
            "min_duration_min": min_dur,
            "eligible_trip_count": count,
            "eligible_share_of_zone_assigned": (
                count / n_zone_assigned if n_zone_assigned else 0.0
            ),
            "eligible_share_of_in_window": (
                count / n_in_window if n_in_window else 0.0
            ),
            **conc,
        }
        rows.append(row)
        print(
            f"  dist>={dist:>5.1f} km : trips={count:>8}  "
            f"share_zone={row['eligible_share_of_zone_assigned']:.4%}  "
            f"od_pairs={conc['nonzero_od_pairs']:>6}  "
            f"top_pair={conc['top_od_pair_share']:.4%}  "
            f"H={conc['od_entropy']:.4f}  H_norm={conc['normalized_od_entropy']:.4f}"
        )

    table = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    out_path = _resolve(sens_cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"[write] sensitivity table ({len(table)} rows) -> {out_path}")

    # --- consistency check: 15 km row vs od_meta.json -----------------
    with open(meta_path) as fh:
        meta = json.load(fh)
    meta_share = float(meta["share_of_evtol_trips"])
    row15 = table[table["min_dist_km"] == 15.0]
    print("[consistency] 15 km eligible share vs od_meta.json:")
    if not len(row15):
        print("  -> CHECK: no 15 km row in the sweep -- cannot verify.")
    else:
        sens_share = float(row15.iloc[0]["eligible_share_of_zone_assigned"])
        diff = abs(sens_share - meta_share)
        print(f"  sensitivity (eligible/zone_assigned) : {sens_share:.6f}")
        print(f"  od_meta.json share_of_evtol_trips    : {meta_share:.6f}")
        print(f"  abs diff                             : {diff:.6f}")
        if diff > META_TOLERANCE:
            print(
                f"  -> STOP: diff {diff:.4%} exceeds 0.1 pp. The eVTOL "
                "statistic may be computed on inconsistent scopes -- "
                "review before trusting the sweep."
            )
        else:
            print("  -> OK (within 0.1 pp; same scope as the R2 full build).")

    return table


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
