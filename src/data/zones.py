"""Stage-2 task 1: build H3 demand zones from cleaned orders.

Aggregates the union of order origin and destination endpoints into H3
hexagons at a fixed resolution, keeping only cells with enough demand to
act as a demand zone.
"""
from __future__ import annotations

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from src.constants import H3_RESOLUTION

# Decimal places for the endpoint-dedup key. 5 dp is ~1.1 m, finer than
# the float32 storage precision of the cleaned coordinates, so collapsing
# duplicates on this key loses no information while cutting the H3
# indexing loop to one call per distinct location, not one per order.
_DEDUP_DECIMALS = 5


def _zone_counts(lon: np.ndarray, lat: np.ndarray, resolution: int) -> pd.Series:
    """Total endpoint count per H3 cell at ``resolution``."""
    pts = pd.DataFrame(
        {
            "klon": np.round(lon, _DEDUP_DECIMALS),
            "klat": np.round(lat, _DEDUP_DECIMALS),
        }
    )
    per_point = pts.groupby(["klon", "klat"], sort=False).size().reset_index(name="n")
    # h3-py uses (lat, lon) order — opposite to the GIS convention.
    per_point["h3_index"] = [
        h3.geo_to_h3(la, lo, resolution)
        for lo, la in zip(per_point["klon"].to_numpy(), per_point["klat"].to_numpy())
    ]
    return per_point.groupby("h3_index")["n"].sum()


def build_zones(
    orders: pd.DataFrame,
    resolution: int = H3_RESOLUTION,
    min_orders_per_zone: int = 50,
) -> gpd.GeoDataFrame:
    """Build H3 demand zones from cleaned orders.

    Takes the union of origin and destination coordinates, indexes each
    distinct endpoint at H3 ``resolution``, and keeps cells whose total
    endpoint count is at least ``min_orders_per_zone``. ``zone_id`` is
    assigned by lexicographic order of the H3 index for determinism.

    The returned frame carries ``n_orders`` (endpoint count) in addition
    to the plan schema; the task-6 demand-density map needs it.
    """
    lon = np.concatenate(
        [
            orders["o_lon"].to_numpy(dtype=np.float64),
            orders["d_lon"].to_numpy(dtype=np.float64),
        ]
    )
    lat = np.concatenate(
        [
            orders["o_lat"].to_numpy(dtype=np.float64),
            orders["d_lat"].to_numpy(dtype=np.float64),
        ]
    )

    counts = _zone_counts(lon, lat, resolution)
    kept = counts[counts >= min_orders_per_zone].sort_index()

    records: list[dict[str, object]] = []
    for zone_id, (h3_index, n_orders) in enumerate(kept.items()):
        clat, clon = h3.h3_to_geo(h3_index)
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        records.append(
            {
                "zone_id": zone_id,
                "h3_index": h3_index,
                "centroid_lon": float(clon),
                "centroid_lat": float(clat),
                "n_orders": int(n_orders),
                "geometry": Polygon(boundary),
            }
        )

    return gpd.GeoDataFrame(
        records,
        columns=[
            "zone_id",
            "h3_index",
            "centroid_lon",
            "centroid_lat",
            "n_orders",
            "geometry",
        ],
        crs="EPSG:4326",
    )
