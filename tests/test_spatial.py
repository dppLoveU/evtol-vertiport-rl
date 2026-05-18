"""Tests for src/data/spatial.py (stage-2 task 5: build_matrices)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import WALK_RADIUS_KM
from src.data.spatial import build_matrices


def _zones(coords: list[tuple[float, float]]) -> pd.DataFrame:
    """Build a zones frame from (lon, lat) centroids."""
    return pd.DataFrame(
        {
            "zone_id": range(len(coords)),
            "centroid_lon": [c[0] for c in coords],
            "centroid_lat": [c[1] for c in coords],
        }
    )


def _cands(coords: list[tuple[float, float]]) -> pd.DataFrame:
    """Build a candidates frame from (lon, lat) points."""
    return pd.DataFrame(
        {
            "cand_id": range(len(coords)),
            "lon": [c[0] for c in coords],
            "lat": [c[1] for c in coords],
        }
    )


# A spread of zones/candidates across the Suzhou metro extent.
_ZONE_PTS = [
    (120.50, 31.10),
    (120.80, 31.30),
    (121.10, 31.60),
    (120.60, 31.90),
]
_CAND_PTS = [
    (120.51, 31.11),  # ~1.5 km from zone 0
    (120.95, 31.45),
    (121.30, 32.00),
    (120.40, 30.95),
    (120.62, 31.88),  # close to zone 3
]


def test_matrix_shapes() -> None:
    zones, cands = _zones(_ZONE_PTS), _cands(_CAND_PTS)
    out = build_matrices(zones, cands, walk_radius_km=WALK_RADIUS_KM)
    nz, nc = len(zones), len(cands)
    assert out["dist_zone_zone"].shape == (nz, nz)
    assert out["dist_zone_cand"].shape == (nz, nc)
    assert out["cand_covers_zones"].shape == (nc, nz)
    assert out["dist_zone_zone"].dtype == np.float32
    assert out["dist_zone_cand"].dtype == np.float32
    assert out["cand_covers_zones"].dtype == np.bool_
    assert out["n_zones"] == nz
    assert out["n_candidates"] == nc


def test_dist_zone_zone_symmetric() -> None:
    out = build_matrices(_zones(_ZONE_PTS), _cands(_CAND_PTS))
    dzz = out["dist_zone_zone"]
    assert np.array_equal(dzz, dzz.T)


def test_dist_zone_zone_zero_diagonal() -> None:
    out = build_matrices(_zones(_ZONE_PTS), _cands(_CAND_PTS))
    diag = np.diagonal(out["dist_zone_zone"])
    assert np.all(diag == 0.0)


def test_no_nan_in_distance_matrices() -> None:
    out = build_matrices(_zones(_ZONE_PTS), _cands(_CAND_PTS))
    assert not np.isnan(out["dist_zone_zone"]).any()
    assert not np.isnan(out["dist_zone_cand"]).any()


def test_cand_covers_zones_matches_distance_threshold() -> None:
    out = build_matrices(_zones(_ZONE_PTS), _cands(_CAND_PTS), walk_radius_km=WALK_RADIUS_KM)
    expected = out["dist_zone_cand"].T <= WALK_RADIUS_KM
    assert np.array_equal(out["cand_covers_zones"], expected)


def test_coverage_ratio_in_unit_interval() -> None:
    out = build_matrices(_zones(_ZONE_PTS), _cands(_CAND_PTS))
    covered = out["cand_covers_zones"].any(axis=0).sum() / out["n_zones"]
    assert 0.0 <= out["coverage_ratio"] <= 1.0
    assert out["coverage_ratio"] == covered
