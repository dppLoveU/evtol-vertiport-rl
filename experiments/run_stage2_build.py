"""Stage-2 build CLI: cleaned orders -> demand zones + candidate sites.

Reads ``configs/spatial.yaml`` (PyYAML; Hydra wiring is deferred — see
``docs/decisions.md`` 2026-05-14) and runs stage-2 tasks 1-5:

  1. build H3 demand zones          -> data/processed/zones.geojson
  2. pull OSM POI candidates        (cached under data/raw/)
  3. pad with uniform-grid candidates
  4. merge + finalize candidates    -> data/processed/candidates.geojson
  5. build distance/coverage matrices -> data/processed/*.npy + spatial_meta.json

Task 6 (folium maps) lives in experiments/run_stage2_maps.py.

Run:
    python -m experiments.run_stage2_build
    python -m experiments.run_stage2_build --config configs/spatial.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.constants import SUZHOU_BBOX
from src.data.candidates import (
    DEFAULT_POI_TAGS,
    add_grid_seeds,
    finalize_candidates,
    pull_poi,
)
from src.data.spatial import build_matrices
from src.data.zones import build_zones

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "spatial.yaml"

# Acceptance bounds from docs/plan/stage2_spatial_discretization.md
# (|Z| and |C| both revised 2026-05-18 after the metro-area bbox
# expansion — see docs/decisions.md). Soft sanity check, not an exception.
ZONE_LO, ZONE_HI = 350, 800
CAND_LO, CAND_HI = 600, 1500
# >= 85% of demand zones must have a candidate within walk radius
# (stage2 plan Acceptance Criteria). Below this the problem is degenerate.
COVERAGE_MIN = 0.85


def _resolve(path_str: str) -> Path:
    """Resolve a yaml path relative to repo root if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute stage-2 tasks 1-4. Returns a summary dict."""
    t0 = time.time()
    orders_path = _resolve(cfg["input"]["orders_path"])
    zcfg = cfg["zones"]
    ccfg = cfg["candidates"]

    # --- Task 1: demand zones ------------------------------------------
    print(f"[zones] reading endpoints from {orders_path}")
    orders = pd.read_parquet(orders_path, columns=["o_lon", "o_lat", "d_lon", "d_lat"])
    zones = build_zones(
        orders,
        resolution=zcfg["h3_resolution"],
        min_orders_per_zone=zcfg["min_orders_per_zone"],
    )
    zpath = _resolve(zcfg["output_path"])
    zpath.parent.mkdir(parents=True, exist_ok=True)
    zones.to_file(zpath, driver="GeoJSON")
    n_zones = len(zones)
    z_ok = ZONE_LO <= n_zones <= ZONE_HI
    print(
        f"  |Z|={n_zones}  ({'PASS' if z_ok else 'CHECK'} "
        f"target [{ZONE_LO}, {ZONE_HI}]) -> {zpath}"
    )
    if not z_ok:
        print("  -> |Z| outside target; consult before tuning min_orders_per_zone")

    # --- Task 2: POI candidates ----------------------------------------
    pcfg = ccfg["poi"]
    poi_cache = _resolve(pcfg["cache_path"])
    print(f"[poi] cache={poi_cache}")
    poi = pull_poi(
        bbox=SUZHOU_BBOX,
        tags=pcfg.get("tags", DEFAULT_POI_TAGS),
        cache_path=poi_cache,
        hospital_min_area_m2=float(pcfg["hospital_min_area_m2"]),
        industrial_min_area_m2=float(pcfg["industrial_min_area_m2"]),
        dedupe_h3_res=pcfg.get("dedupe_h3_res", 8),
    )
    print(f"  POI count={len(poi)}")
    if len(poi):
        for src, n in poi["source"].value_counts().items():
            print(f"    {src:<16}: {n}")

    # --- Task 3: uniform-grid candidates -------------------------------
    gcfg = ccfg["grid"]
    grid = add_grid_seeds(
        poi,
        zones,
        bbox=SUZHOU_BBOX,
        spacing_deg=float(gcfg["spacing_deg"]),
        min_separation_km=float(gcfg["min_separation_km"]),
    )
    print(f"[grid] grid candidate count={len(grid)}")

    # --- Task 4: merge + finalize candidates ---------------------------
    n_cand_raw = len(poi) + len(grid)
    print(f"[finalize] POI+grid pre-finalize = {n_cand_raw}")
    candidates = finalize_candidates(poi, grid, zones)
    cpath = _resolve(ccfg["output_path"])
    cpath.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_file(cpath, driver="GeoJSON")
    n_cand = len(candidates)
    c_ok = CAND_LO <= n_cand <= CAND_HI
    print(
        f"  |C|={n_cand}  (dropped {n_cand_raw - n_cand} outside demand zones; "
        f"{'PASS' if c_ok else 'CHECK'} target [{CAND_LO}, {CAND_HI}]) -> {cpath}"
    )
    if not c_ok:
        print("  -> |C| outside target; log warning and stop for human review")
    for src, n in candidates["source"].value_counts().items():
        print(f"    {src:<16}: {n}")

    # --- Task 5: distance + coverage matrices --------------------------
    mcfg = cfg["matrices"]
    walk_radius_km = float(mcfg["walk_radius_km"])
    print(f"[matrices] building distance/coverage matrices (walk_radius_km={walk_radius_km})")
    mats = build_matrices(zones, candidates, walk_radius_km=walk_radius_km)

    dzz_path = _resolve(mcfg["dist_zone_zone_path"])
    dzc_path = _resolve(mcfg["dist_zone_cand_path"])
    cov_path = _resolve(mcfg["cand_covers_zones_path"])
    for p in (dzz_path, dzc_path, cov_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    np.save(dzz_path, mats["dist_zone_zone"])
    np.save(dzc_path, mats["dist_zone_cand"])
    np.save(cov_path, mats["cand_covers_zones"])
    print(f"  dist_zone_zone   {mats['dist_zone_zone'].shape} -> {dzz_path}")
    print(f"  dist_zone_cand   {mats['dist_zone_cand'].shape} -> {dzc_path}")
    print(f"  cand_covers_zones {mats['cand_covers_zones'].shape} -> {cov_path}")

    coverage_ratio = mats["coverage_ratio"]
    cov_ok = coverage_ratio >= COVERAGE_MIN
    print(
        f"  coverage_ratio={coverage_ratio:.4f}  "
        f"({'PASS' if cov_ok else 'CHECK'} target >= {COVERAGE_MIN})"
    )
    if not cov_ok:
        print(
            "  -> coverage below threshold; STOP and report — do not "
            "auto-tune walk_radius or candidate params"
        )

    source_counts = {
        str(src): int(n) for src, n in candidates["source"].value_counts().items()
    }
    meta = {
        "n_zones": mats["n_zones"],
        "n_candidates": mats["n_candidates"],
        "walk_radius_km": mats["walk_radius_km"],
        "h3_resolution": int(zcfg["h3_resolution"]),
        "source_counts": source_counts,
        "coverage_ratio": coverage_ratio,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = _resolve(cfg["meta_path"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  spatial_meta.json -> {meta_path}")

    print(f"[total wall time: {time.time() - t0:.1f}s]")

    return {
        "n_zones": n_zones,
        "n_poi": len(poi),
        "n_grid": len(grid),
        "n_cand_raw": n_cand_raw,
        "n_cand": n_cand,
        "zones_ok": z_ok,
        "cand_ok": c_ok,
        "coverage_ratio": coverage_ratio,
        "coverage_ok": cov_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to spatial yaml (default: {DEFAULT_CONFIG.relative_to(REPO)})",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    run(cfg)


if __name__ == "__main__":
    main()
