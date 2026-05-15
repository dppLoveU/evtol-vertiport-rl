"""Stage-1 cleaning CLI: raw CSV -> cleaned Parquet.

Reads ``configs/preprocess.yaml`` (PyYAML; Hydra wiring is deferred —
see ``docs/decisions.md`` 2026-05-14), runs the full task-1..6 pipeline,
and writes ``data/processed/orders_clean.parquet`` with snappy
compression and 100k-row groups.

Per-step row-count audit and post-write summary are printed to stdout.

Run:
    python -m experiments.run_stage1_clean
    python -m experiments.run_stage1_clean --config configs/preprocess.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.clean import (
    DEFAULT_DATETIME_FORMAT,
    add_derived_columns,
    apply_outlier_filters,
    apply_simple_renames,
    fix_coordinates,
    load_raw,
    parse_times,
    select_output_columns,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "preprocess.yaml"

# Acceptance bounds from docs/plan/stage1_data_cleaning.md (revised
# 2026-05-15 after metro-area bbox). Treated as a soft sanity check
# here, not an exception.
ACCEPT_LO = 3_620_000
ACCEPT_HI = 4_150_000


def _resolve(path_str: str) -> Path:
    """Resolve a yaml path relative to repo root if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute the full pipeline and write parquet. Returns a summary dict."""
    in_path = _resolve(cfg["input"]["path"])
    out_path = _resolve(cfg["output"]["path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunksize = cfg["input"].get("chunksize")
    if chunksize in (0, None):
        chunksize = None

    fmt = cfg["datetime"]["format"]
    if fmt != DEFAULT_DATETIME_FORMAT:
        print(
            f"[warn] config datetime.format ({fmt!r}) differs from "
            f"clean.py default ({DEFAULT_DATETIME_FORMAT!r}); using config."
        )

    f = cfg["filters"]
    bbox = f["bbox"]

    print(f"[load_raw] {in_path}  (chunksize={chunksize})")
    t0 = time.time()
    df = load_raw(in_path, chunksize=chunksize, encoding=cfg["input"]["encoding"])
    n_initial = len(df)
    print(f"  read {n_initial:,} rows in {time.time() - t0:.1f}s")

    print("[fix_coordinates]")
    df = fix_coordinates(df)

    print(f"[parse_times] format={fmt!r}")
    df, n_dropped_time = parse_times(df, fmt=fmt)
    print(f"  dropped {n_dropped_time:,} rows with unparseable timestamps")

    print("[add_derived_columns]")
    df = add_derived_columns(df)

    print("[apply_simple_renames]")
    df = apply_simple_renames(df)

    print("[apply_outlier_filters]")
    t1 = time.time()
    df, audit = apply_outlier_filters(
        df,
        bbox=bbox,
        duration_min_range=tuple(f["duration_min"]),
        drive_km_range=tuple(f["drive_km"]),
        geo_dist_km_min=float(f["geo_dist_km_min"]),
        geo_drive_ratio_max=float(f["geo_drive_ratio_max"]),
        fare_yuan_range=tuple(f["fare_yuan"]),
        wait_min_range=tuple(f["wait_min"]),
    )
    print(f"  filters done in {time.time() - t1:.1f}s")
    print("  audit:")
    prev = audit["initial"]
    for key, val in audit.items():
        if key == "initial":
            print(f"    {key:<22}: {val:>10,}")
        else:
            drop = prev - val
            pct = (drop / prev * 100) if prev else 0.0
            print(f"    {key:<22}: {val:>10,}  (drop {drop:>9,} = {pct:5.2f}% of prev)")
            prev = val

    print("[select_output_columns]  (raises on any NA)")
    df = select_output_columns(df)
    n_final = len(df)

    print(f"[to_parquet] -> {out_path}")
    df.to_parquet(
        out_path,
        compression=cfg["output"]["compression"],
        row_group_size=cfg["output"]["row_group_size"],
    )
    sz_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {n_final:,} rows  ({sz_mb:.1f} MB on disk)")

    accept_ok = ACCEPT_LO <= n_final <= ACCEPT_HI
    accept_status = "PASS" if accept_ok else "FAIL"
    print(
        f"[acceptance] target [{ACCEPT_LO:,}, {ACCEPT_HI:,}]  "
        f"actual {n_final:,}  {accept_status}"
    )
    if not accept_ok:
        delta = n_final - (ACCEPT_HI if n_final > ACCEPT_HI else ACCEPT_LO)
        print(f"  -> {abs(delta):,} rows {'above HI' if delta > 0 else 'below LO'}")

    print(f"[total wall time: {time.time() - t0:.1f}s]")
    return {
        "n_initial": n_initial,
        "n_final": n_final,
        "audit": audit,
        "out_path": str(out_path),
        "size_mb": sz_mb,
        "accept_ok": accept_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to preprocess yaml (default: {DEFAULT_CONFIG.relative_to(REPO)})",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    run(cfg)


if __name__ == "__main__":
    main()
