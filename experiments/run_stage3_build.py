"""Stage-3 build CLI (R2 core): cleaned orders -> OD tensors.

Builds the three dense OD tensors defined in configs/od.yaml:
  od_full           [T,|Z|,|Z|] int32   -- all in-zone orders, counted
  od_evtol          [T,|Z|,|Z|] int32   -- eVTOL-eligible orders, counted
  od_evtol_weighted [T,|Z|,|Z|] float32 -- eVTOL orders, weight summed

SMOKE vs FULL mode:
  --nrows > 0  : SMOKE -- builds the tensors in memory and prints stats
                 ONLY; NOTHING is written to disk. A partial-data tensor
                 must not land at data/processed/od_*.npy where it could
                 be mistaken for the real product.
  --nrows 0    : FULL  -- after all acceptance checks pass, writes
                 od_full / od_evtol / od_evtol_weighted .npy and
                 od_meta.json to the configs/od.yaml output paths.

If any acceptance check fails (zone drop_rate > 12%, evtol_share outside
[0.03, 0.20], NaN / negative values, or a non-zero od_evtol diagonal)
the script reports and aborts -- it does NOT auto-tune any threshold and
writes nothing.

EDA figures and the sensitivity sweep are NOT part of this script.

Run:
    python -m experiments.run_stage3_build --nrows 200000   # smoke
    python -m experiments.run_stage3_build                  # full
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from src.data.od import (
    assign_time_bin,
    assign_zones,
    build_od_tensor,
    build_zone_lookup,
    is_evtol_eligible,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "od.yaml"

# Zone-drop ceiling -- docs/decisions.md 2026-05-18.
ZONE_DROP_MAX = 0.12
# eVTOL share acceptance window -- plan Acceptance Criteria.
SHARE_LO, SHARE_HI = 0.03, 0.20

_READ_COLUMNS = [
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


def _read_rows(path: Path, columns: list[str], nrows: int) -> pd.DataFrame:
    """Read ``nrows`` leading rows of a parquet file; ``nrows <= 0`` reads all."""
    if nrows <= 0:
        return pd.read_parquet(path, columns=columns)
    pf = pq.ParquetFile(path)
    batch = next(pf.iter_batches(batch_size=nrows, columns=columns))
    return batch.to_pandas()


def _print_tensor_stats(name: str, st: dict[str, Any]) -> None:
    """Print one tensor's shape / dtype / sum / sparsity / memory."""
    print(f"  {name}")
    print(f"    shape         : {st['shape']}")
    print(f"    dtype         : {st['dtype']}")
    print(f"    sum           : {st['sum']}")
    print(f"    nonzero_count : {st['nonzero_count']}  / {st['n_cells']} cells")
    print(f"    nonzero_ratio : {st['nonzero_ratio']:.6%}")
    print(f"    est. memory   : {st['memory_mb']:.1f} MB")


def _write_outputs(
    cfg: dict[str, Any],
    tensors: dict[str, np.ndarray],
    n_time_bins: int,
    n_zones: int,
    time_cfg: dict[str, Any],
    evtol_cfg: dict[str, Any],
    weight_col: str,
    evtol_share: float,
) -> None:
    """FULL mode: write the three .npy tensors and od_meta.json."""
    out = cfg["output"]
    npy_paths = {
        "od_full": _resolve(out["od_full_path"]),
        "od_evtol": _resolve(out["od_evtol_path"]),
        "od_evtol_weighted": _resolve(out["od_evtol_weighted_path"]),
    }
    for name, path in npy_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, tensors[name])
        print(f"  wrote {name} -> {path}")

    meta = {
        "T": n_time_bins,
        "n_zones": n_zones,
        "time_bin_min": int(time_cfg["time_bin_min"]),
        "start_datetime": time_cfg["start_datetime"],
        "end_datetime": time_cfg["end_datetime"],
        "evtol_filter_params": {
            "min_dist_km": float(evtol_cfg["min_dist_km"]),
            "min_duration_min": float(evtol_cfg["min_duration_min"]),
        },
        "weight_column": weight_col,
        "share_of_evtol_trips": evtol_share,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = _resolve(out["meta_path"])
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  wrote od_meta.json -> {meta_path}")


def run(cfg: dict[str, Any], nrows: int) -> None:
    """Build the OD tensors. ``nrows > 0`` is smoke mode (no disk write)."""
    t_wall = time.time()
    smoke = nrows > 0
    print(f"[mode] {'SMOKE (--nrows ' + str(nrows) + ', no .npy written)' if smoke else 'FULL'}")

    orders_path = _resolve(cfg["input"]["orders_path"])
    zones_path = _resolve(cfg["input"]["zones_path"])
    time_cfg = cfg["time"]
    evtol_cfg = cfg["evtol_filter"]
    weight_col = str(cfg["weight"]["column"])

    start = pd.Timestamp(time_cfg["start_datetime"])
    bin_min = int(time_cfg["time_bin_min"])
    n_bins = int(time_cfg["num_time_bins"])

    # --- read + time bin + zone assignment ----------------------------
    df = _read_rows(orders_path, _READ_COLUMNS, nrows)
    n_orders_total = len(df)

    in_range, _, _ = assign_time_bin(
        df, t0=start, time_bin_min=bin_min, num_time_bins=n_bins
    )
    n_time_in_range = len(in_range)

    zones = gpd.read_file(zones_path)
    n_zones = len(zones)
    lookup = build_zone_lookup(zones)
    assigned, _, zstats = assign_zones(in_range, lookup)
    drop_rate = float(zstats["drop_rate"])
    n_zone_assigned = int(zstats["n_zone_assigned"])

    print("[orders]")
    print(f"  n_orders_total            : {n_orders_total}")
    print(f"  n_time_in_range           : {n_time_in_range}")
    print(f"  n_zone_assigned           : {n_zone_assigned}")
    print(f"  zone_assignment_drop_rate : {drop_rate:.4%}")

    if drop_rate > ZONE_DROP_MAX:
        print(
            f"  -> STOP: zone_assignment_drop_rate > {ZONE_DROP_MAX:.0%}. "
            "Not auto-tuning -- aborting before tensor build."
        )
        return

    # --- build od_full -------------------------------------------------
    od_full, full_stats = build_od_tensor(
        assigned, n_time_bins=n_bins, n_zones=n_zones
    )

    # --- eVTOL filter + od_evtol + od_evtol_weighted -------------------
    mask = is_evtol_eligible(
        assigned,
        min_dist_km=float(evtol_cfg["min_dist_km"]),
        min_duration_min=float(evtol_cfg["min_duration_min"]),
    )
    evtol_df = assigned[mask]
    evtol_trip_count = int(mask.sum())

    od_evtol, evtol_stats = build_od_tensor(
        evtol_df, n_time_bins=n_bins, n_zones=n_zones
    )
    od_evtol_weighted, weighted_stats = build_od_tensor(
        evtol_df, n_time_bins=n_bins, n_zones=n_zones, value_col=weight_col
    )

    full_sum = int(od_full.sum())
    evtol_sum = int(od_evtol.sum())
    evtol_share = evtol_sum / full_sum if full_sum else 0.0

    print("[eVTOL]")
    print(f"  evtol_trip_count : {evtol_trip_count}")
    print(f"  evtol_share      : {evtol_share:.4%}  (od_evtol.sum / od_full.sum)")

    print("[tensors]")
    _print_tensor_stats("od_full", full_stats)
    _print_tensor_stats("od_evtol", evtol_stats)
    _print_tensor_stats("od_evtol_weighted", weighted_stats)

    # --- acceptance checks (plan Acceptance Criteria) ------------------
    issues: list[str] = []
    share_ok = SHARE_LO <= evtol_share <= SHARE_HI
    if not share_ok:
        issues.append(
            f"evtol_share {evtol_share:.4%} outside [{SHARE_LO:.0%}, {SHARE_HI:.0%}]"
        )
    for name, tensor in (
        ("od_full", od_full),
        ("od_evtol", od_evtol),
        ("od_evtol_weighted", od_evtol_weighted),
    ):
        if np.issubdtype(tensor.dtype, np.floating) and np.isnan(tensor).any():
            issues.append(f"{name} contains NaN")
        if (tensor < 0).any():
            issues.append(f"{name} contains negative values")
    diag_sum = int(od_evtol[:, np.arange(n_zones), np.arange(n_zones)].sum())
    if diag_sum != 0:
        issues.append(f"od_evtol diagonal sum is {diag_sum} (expected 0)")

    print("[checks]")
    print(
        f"  evtol_share in [{SHARE_LO:.0%}, {SHARE_HI:.0%}] : "
        f"{'PASS' if share_ok else 'FAIL'}"
    )
    print(f"  no NaN / no negative values            : "
          f"{'PASS' if not any('NaN' in i or 'negative' in i for i in issues) else 'FAIL'}")
    print(f"  od_evtol diagonal == 0                 : "
          f"{'PASS' if diag_sum == 0 else 'FAIL'}")

    if issues:
        print("  -> STOP: acceptance check(s) failed:")
        for it in issues:
            print(f"     - {it}")
        print("  Not auto-tuning thresholds. Aborting before any write.")
        return
    print("  all acceptance checks PASS")

    # --- write (FULL mode only) ---------------------------------------
    if smoke:
        print(
            f"[smoke] tensors built in memory; no .npy written "
            f"(partial {nrows}-row data). Re-run without --nrows for the "
            "full build."
        )
    else:
        print("[write]")
        _write_outputs(
            cfg,
            {
                "od_full": od_full,
                "od_evtol": od_evtol,
                "od_evtol_weighted": od_evtol_weighted,
            },
            n_bins,
            n_zones,
            time_cfg,
            evtol_cfg,
            weight_col,
            evtol_share,
        )

    print(f"[wall time: {time.time() - t_wall:.1f}s]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to od yaml (default: {DEFAULT_CONFIG.relative_to(REPO)})",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=0,
        help="Leading orders to build from; >0 = smoke (no write), 0 = full (default: 0)",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    run(cfg, args.nrows)


if __name__ == "__main__":
    main()
