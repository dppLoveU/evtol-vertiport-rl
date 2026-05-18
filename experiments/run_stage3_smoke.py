"""Stage-3 R1.5 smoke test: validate R1's time-bin + zone assignment on real data.

This is a *validation-only* script. It does NOT build any OD tensor and
does NOT write od_full.npy / od_evtol.npy / od_meta.json.

Two parts:

  A. Time span check  -- reads only the full ``dep_time`` column and
     applies the configured OD time window
     ``[start_datetime, end_datetime)`` (left-closed, right-open). Orders
     before the start or at/after the end land in ``out_of_range``. With
     the 11-day window these are the two residual partial days at the
     data's edges, so a non-zero out-of-range rate is *expected* here --
     the script reports how the out-of-range rows split before vs. after
     the window rather than treating the rate alone as a failure.

  B. Zone assignment check -- reads ``--nrows`` orders (0 = the whole
     file), runs ``assign_time_bin`` then ``assign_zones``, and reports
     the zone-drop rate. If ``drop_rate > 2%`` it prints a breakdown of
     the dropped orders -- area_name shares, geo_dist_km / duration_min /
     fare_yuan describe, departure-hour distribution, and the share that
     would still meet the eVTOL base condition -- so the impact of the
     dropped zones on eVTOL demand can be judged.

The OD time window comes from ``configs/od.yaml::time`` -- ``t0`` is the
configured ``start_datetime``, never ``min(dep_time)``.

Run:
    python -m experiments.run_stage3_smoke              # full file
    python -m experiments.run_stage3_smoke --nrows 200000
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import yaml

from src.data.od import assign_time_bin, assign_zones, build_zone_lookup

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "od.yaml"

# Stop threshold for part B (per the R1.5 task brief).
ZONE_DROP_STOP = 0.02  # 2% zone-drop -> report a full breakdown

# Columns part B needs: O/D coords for zone assignment, dep_time for the
# time bin, and area_name / geo_dist_km / duration_min / fare_yuan for
# the dropped-order breakdown.
_PART_B_COLUMNS = [
    "o_lon",
    "o_lat",
    "d_lon",
    "d_lat",
    "dep_time",
    "area_name",
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


def part_a(orders_path: Path, time_cfg: dict[str, Any]) -> tuple[pd.Timestamp, bool]:
    """Time window check on the full dep_time column.

    Returns ``(start, consistent)`` where ``consistent`` is True when
    every out-of-range row is accounted for by the before/after window
    tails (the expected residual-day case).
    """
    print("\n" + "=" * 64)
    print("PART A  --  time window check (full dep_time column)")
    print("=" * 64)
    start = pd.Timestamp(time_cfg["start_datetime"])
    end = pd.Timestamp(time_cfg["end_datetime"])
    bin_min = int(time_cfg["time_bin_min"])
    num_bins = int(time_cfg["num_time_bins"])

    dep = pd.read_parquet(orders_path, columns=["dep_time"])
    t_min = dep["dep_time"].min()
    t_max = dep["dep_time"].max()

    in_range, oor, stats = assign_time_bin(
        dep, t0=start, time_bin_min=bin_min, num_time_bins=num_bins
    )
    n = int(stats["n_total"])
    oor_count = int(stats["n_out_of_range"])
    oor_rate = oor_count / n if n else 0.0
    before = int((oor["dep_time"] < start).sum())
    after = int((oor["dep_time"] >= end).sum())
    consistent = before + after == oor_count

    print(f"  n orders            : {n}")
    print(f"  data dep_time range : {t_min}  ..  {t_max}")
    print(f"  window start (t0)   : {start}  (left-closed)")
    print(f"  window end          : {end}  (right-open)")
    print(f"  num_time_bins (T)   : {num_bins}  (bin_min={bin_min})")
    print(f"  in_range_count      : {int(stats['n_in_range'])}")
    print(f"  out_of_range_count  : {oor_count}")
    print(f"  out_of_range_rate   : {oor_rate:.4%}")
    print(f"    before window     : {before}  (dep_time < start)")
    print(f"    after window      : {after}  (dep_time >= end)")
    print(f"  max slot observed   : {stats['max_slot']}  (in-range slots are 0..{num_bins - 1})")

    if consistent:
        print(
            "  -> OK  (all out-of-range rows are the two residual-day "
            "tails; this is expected for the 11-day window)"
        )
    else:
        print(
            f"  -> CHECK  ({oor_count - before - after} out-of-range rows "
            "are NOT in either tail -- unexpected, investigate)"
        )
    return start, consistent


def _report_dropped(
    dropped: pd.DataFrame, evtol_min_dist: float, evtol_min_dur: float
) -> None:
    """Print the breakdown of zone-dropped orders requested in the brief."""
    n = len(dropped)
    print("\n  --- dropped-order breakdown -----------------------------------")
    print(f"  dropped n = {n}")

    print("\n  area_name value_counts (count / share of dropped):")
    vc = dropped["area_name"].value_counts()
    for name, cnt in vc.items():
        print(f"    {name:<10}: {cnt:>9}  ({cnt / n:.2%})")

    print("\n  geo_dist_km describe:")
    print(dropped["geo_dist_km"].describe().to_string())
    print("\n  duration_min describe:")
    print(dropped["duration_min"].describe().to_string())
    print("\n  fare_yuan describe:")
    print(dropped["fare_yuan"].describe().to_string())

    print("\n  dep_time hour-of-day distribution (0-23):")
    hours = dropped["dep_time"].dt.hour.value_counts().sort_index()
    print(hours.to_string())

    eligible = (dropped["geo_dist_km"] >= evtol_min_dist) & (
        dropped["duration_min"] >= evtol_min_dur
    )
    n_elig = int(eligible.sum())
    print(
        f"\n  dropped orders meeting eVTOL base condition "
        f"(geo_dist_km >= {evtol_min_dist} AND duration_min >= {evtol_min_dur}): "
        f"{n_elig}  ({n_elig / n:.2%} of dropped)"
    )


def part_b(
    orders_path: Path,
    zones_path: Path,
    time_cfg: dict[str, Any],
    evtol_cfg: dict[str, Any],
    t0: pd.Timestamp,
    nrows: int,
) -> None:
    """Zone assignment check on ``nrows`` orders (0 = all)."""
    scope = "all orders" if nrows <= 0 else f"first {nrows} orders"
    print("\n" + "=" * 64)
    print(f"PART B  --  zone assignment check ({scope})")
    print("=" * 64)
    bin_min = int(time_cfg["time_bin_min"])
    num_bins = int(time_cfg["num_time_bins"])

    df = _read_rows(orders_path, _PART_B_COLUMNS, nrows)
    n_total = len(df)

    in_range, _, _ = assign_time_bin(
        df, t0=t0, time_bin_min=bin_min, num_time_bins=num_bins
    )
    n_time_in_range = len(in_range)

    zones = gpd.read_file(zones_path)
    lookup = build_zone_lookup(zones)
    print(f"  zone lookup size    : {len(lookup)} H3 cells")

    _, dropped, stats = assign_zones(in_range, lookup)
    drop_rate = float(stats["drop_rate"])

    print(f"  n_total             : {n_total}")
    print(f"  n_time_in_range     : {n_time_in_range}")
    print(f"  n_zone_assigned     : {stats['n_zone_assigned']}")
    print(f"  drop_count          : {stats['drop_count']}")
    print(f"  drop_rate           : {drop_rate:.4%}  (of n_time_in_range)")
    print(f"  n_unknown_o         : {stats['n_unknown_o']}")
    print(f"  n_unknown_d         : {stats['n_unknown_d']}")

    if drop_rate > ZONE_DROP_STOP:
        print(f"  -> drop_rate > {ZONE_DROP_STOP:.0%} -- full breakdown below.")
        if len(dropped):
            _report_dropped(
                dropped,
                float(evtol_cfg["min_dist_km"]),
                float(evtol_cfg["min_duration_min"]),
            )
    else:
        print(f"  -> PASS  (drop_rate <= {ZONE_DROP_STOP:.0%})")


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
        help="Leading orders for the part-B zone check; 0 = whole file (default: 0)",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    orders_path = _resolve(cfg["input"]["orders_path"])
    zones_path = _resolve(cfg["input"]["zones_path"])
    time_cfg = cfg["time"]
    evtol_cfg = cfg["evtol_filter"]

    start, consistent = part_a(orders_path, time_cfg)
    if not consistent:
        print(
            "\nSTOP: out-of-range rows are not fully explained by the two "
            "residual-day tails. Skipping part B -- investigate the time "
            "window first."
        )
        return

    part_b(orders_path, zones_path, time_cfg, evtol_cfg, start, args.nrows)


if __name__ == "__main__":
    main()
