"""Tests for src/data/zones.py (stage-2 task 1: build_zones)."""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.data.zones import build_zones

RES = 7


def _cell_center(lon: float, lat: float) -> tuple[float, float, str]:
    """Exact (lon, lat) centroid and H3 index of the res-7 cell at a point."""
    cell = h3.geo_to_h3(lat, lon, RES)
    clat, clon = h3.h3_to_geo(cell)
    return clon, clat, cell


def _orders_from(groups: list[tuple[float, float, int]]) -> pd.DataFrame:
    """Build an orders frame; each (lon, lat, n) contributes n orders whose
    origin AND destination both sit at (lon, lat) -> 2n endpoints there."""
    lon: list[float] = []
    lat: list[float] = []
    for x, y, n in groups:
        lon += [x] * n
        lat += [y] * n
    return pd.DataFrame(
        {
            "o_lon": np.array(lon, dtype=np.float32),
            "o_lat": np.array(lat, dtype=np.float32),
            "d_lon": np.array(lon, dtype=np.float32),
            "d_lat": np.array(lat, dtype=np.float32),
        }
    )


def test_min_orders_threshold_drops_sparse_cells() -> None:
    a_lon, a_lat, a_cell = _cell_center(120.50, 31.30)
    b_lon, b_lat, _ = _cell_center(120.80, 31.55)
    # cell A: 20 orders -> 40 endpoints; cell B: 3 orders -> 6 endpoints.
    orders = _orders_from([(a_lon, a_lat, 20), (b_lon, b_lat, 3)])
    zones = build_zones(orders, resolution=RES, min_orders_per_zone=10)
    assert len(zones) == 1
    assert zones.iloc[0]["h3_index"] == a_cell
    assert zones.iloc[0]["n_orders"] == 40


def test_zone_id_is_lexicographic_and_contiguous() -> None:
    groups = [
        _cell_center(120.50, 31.30)[:2] + (15,),
        _cell_center(120.95, 31.70)[:2] + (15,),
        _cell_center(120.70, 31.45)[:2] + (15,),
    ]
    zones = build_zones(_orders_from(groups), resolution=RES, min_orders_per_zone=10)
    assert len(zones) == 3
    assert list(zones["zone_id"]) == [0, 1, 2]
    assert list(zones["h3_index"]) == sorted(zones["h3_index"])


def test_centroid_inside_polygon_and_resolution() -> None:
    a_lon, a_lat, _ = _cell_center(120.55, 31.32)
    zones = build_zones(
        _orders_from([(a_lon, a_lat, 30)]), resolution=RES, min_orders_per_zone=10
    )
    for _, row in zones.iterrows():
        assert row["geometry"].contains(Point(row["centroid_lon"], row["centroid_lat"]))
        assert h3.h3_get_resolution(row["h3_index"]) == RES


def test_build_zones_is_deterministic() -> None:
    groups = [
        _cell_center(120.50, 31.30)[:2] + (12,),
        _cell_center(120.90, 31.65)[:2] + (12,),
    ]
    z1 = build_zones(_orders_from(groups), resolution=RES, min_orders_per_zone=10)
    z2 = build_zones(_orders_from(groups), resolution=RES, min_orders_per_zone=10)
    assert list(z1["h3_index"]) == list(z2["h3_index"])
    assert list(z1["n_orders"]) == list(z2["n_orders"])
