"""Per-area coordinate bounds on the full raw CSV.

For each ``AreaName`` in ``data/raw/suzhou_orders_7days.csv``, compute
min / 1st-percentile / 99th-percentile / max for all four coordinates
(after :func:`fix_coordinates`) and the order count. Used to refine
``SUZHOU_BBOX`` from "Suzhou City proper" to a data-driven envelope of
the full Suzhou metropolitan area (city + 4 county-level cities).

Zero-coord placeholder rows are dropped before stats so a single
missing-coord record does not pull the per-area min to 0.

Output: ``results/stage1/eda/area_bounds.csv``.

Run: ``python -m experiments.run_stage1_area_bounds``  (from repo root,
venv active).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean import fix_coordinates, load_raw

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "suzhou_orders_7days.csv"
OUT_DIR = REPO / "results" / "stage1" / "eda"
OUT_CSV = OUT_DIR / "area_bounds.csv"

COORD_COLS = ["o_lon", "o_lat", "d_lon", "d_lat"]
PAD_KM = 2.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load_raw] reading full file ...")
    t0 = time.time()
    df = load_raw(RAW, chunksize=500_000)
    n_initial = len(df)
    print(f"  {n_initial:,} rows in {time.time() - t0:.1f}s")

    df = fix_coordinates(df)

    nonzero = (df[COORD_COLS] > 0).all(axis=1)
    n_kept = int(nonzero.sum())
    n_dropped = n_initial - n_kept
    print(
        f"  zero-coord placeholder rows dropped: {n_dropped:,} "
        f"({n_dropped / n_initial * 100:.2f}%)"
    )
    df = df[nonzero]

    print("[area_bounds] grouping by AreaName ...")
    rows: list[dict] = []
    for area, sub in df.groupby("AreaName", sort=True):
        rec: dict = {"area_name": area, "n_orders": int(len(sub))}
        for c in COORD_COLS:
            arr = sub[c].to_numpy()
            p1, p99 = np.percentile(arr, [1, 99])
            rec[f"{c}_min"] = float(arr.min())
            rec[f"{c}_p1"] = float(p1)
            rec[f"{c}_p99"] = float(p99)
            rec[f"{c}_max"] = float(arr.max())
        rows.append(rec)

    out = (
        pd.DataFrame(rows)
        .sort_values("n_orders", ascending=False)
        .reset_index(drop=True)
    )
    out.to_csv(OUT_CSV, index=False)
    print(f"[area_bounds] wrote {len(out)} areas to {OUT_CSV}\n")

    # Compact summary print: just min / p1 / p99 / max per coord per area.
    pretty = out.copy()
    pretty["n_orders"] = pretty["n_orders"].map(lambda x: f"{x:>10,}")
    print(pretty.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ----- bbox proposal: union of p1/p99 across areas + km padding -----
    # Latitude:  1 km ≈ 0.0090° (1 / 111).
    # Longitude at lat ≈ 31.5°N: 1 km ≈ 0.0105° (1 / (111 * cos(lat))).
    pad_lat = PAD_KM * (1.0 / 111.0)
    pad_lon = PAD_KM * (1.0 / (111.0 * np.cos(np.radians(31.5))))

    lon_p1_cols = [f"{c}_p1" for c in COORD_COLS if c.endswith("lon")]
    lon_p99_cols = [f"{c}_p99" for c in COORD_COLS if c.endswith("lon")]
    lat_p1_cols = [f"{c}_p1" for c in COORD_COLS if c.endswith("lat")]
    lat_p99_cols = [f"{c}_p99" for c in COORD_COLS if c.endswith("lat")]

    lon_min_proposed = float(out[lon_p1_cols].to_numpy().min()) - pad_lon
    lon_max_proposed = float(out[lon_p99_cols].to_numpy().max()) + pad_lon
    lat_min_proposed = float(out[lat_p1_cols].to_numpy().min()) - pad_lat
    lat_max_proposed = float(out[lat_p99_cols].to_numpy().max()) + pad_lat

    print(f"\n[bbox proposal — union(p1, p99) across areas + {PAD_KM} km padding]")
    print(f"  pad_lat = {pad_lat:.5f}°  pad_lon = {pad_lon:.5f}°")
    print(f"  proposed  lon = [{lon_min_proposed:.4f}, {lon_max_proposed:.4f}]")
    print(f"  proposed  lat = [{lat_min_proposed:.4f}, {lat_max_proposed:.4f}]")
    print("  current   lon = [120.4500, 120.9500]")
    print("  current   lat = [31.2000, 31.5000]")

    print(f"\n[total wall time: {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
