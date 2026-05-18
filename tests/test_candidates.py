"""Tests for src/data/candidates.py (stage-2 tasks 2-3)."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from src.data.candidates import (
    _classify_source,
    _dedupe_by_h3,
    add_grid_seeds,
    finalize_candidates,
    pull_poi,
)
from src.utils.geo import haversine_km

_BBOX = {"lon_min": 120.40, "lon_max": 120.60, "lat_min": 31.00, "lat_max": 31.20}


def _square(lon0: float, lon1: float, lat0: float, lat1: float) -> Polygon:
    return Polygon([(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)])


def _empty_poi() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "lon": pd.Series(dtype=float),
            "lat": pd.Series(dtype=float),
            "source": pd.Series(dtype=str),
        },
        geometry=gpd.points_from_xy([], []),
        crs="EPSG:4326",
    )


def test_classify_source_priority_and_drop() -> None:
    assert _classify_source(pd.Series({"aeroway": "aerodrome"})) == "poi_airport"
    assert _classify_source(pd.Series({"railway": "station"})) == "poi_subway"
    assert _classify_source(pd.Series({"station": "subway"})) == "poi_subway"
    assert _classify_source(pd.Series({"shop": "mall"})) == "poi_mall"
    assert _classify_source(pd.Series({"amenity": "hospital"})) == "poi_hospital"
    assert _classify_source(pd.Series({"landuse": "industrial"})) == "poi_industrial"
    # airport wins over a co-tagged mall
    assert (
        _classify_source(pd.Series({"aeroway": "aerodrome", "shop": "mall"}))
        == "poi_airport"
    )
    # unrelated feature is dropped
    assert _classify_source(pd.Series({"amenity": "cafe"})) is None
    # subway=yes marks an underground rail *way*, not a station -> dropped
    assert _classify_source(pd.Series({"subway": "yes"})) is None


def test_dedupe_by_h3_keeps_highest_priority_per_cell() -> None:
    lon = [120.5000, 120.5001, 120.5002, 120.9000]
    lat = [31.1000, 31.1001, 31.1002, 31.5000]
    poi = gpd.GeoDataFrame(
        {
            "lon": lon,
            "lat": lat,
            "source": ["poi_industrial", "poi_subway", "poi_mall", "poi_hospital"],
        },
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
    deduped = _dedupe_by_h3(poi, resolution=8)
    # the first three points collapse into one cell (subway outranks
    # mall and industrial); the distant fourth point survives.
    assert len(deduped) == 2
    assert set(deduped["source"]) == {"poi_subway", "poi_hospital"}


def test_add_grid_seeds_inside_zones_only() -> None:
    # one zone covering the lower-left quarter of the bbox
    zones = gpd.GeoDataFrame(
        {"zone_id": [0]},
        geometry=[_square(120.40, 120.50, 31.00, 31.10)],
        crs="EPSG:4326",
    )
    grid = add_grid_seeds(
        _empty_poi(), zones, bbox=_BBOX, spacing_deg=0.03, min_separation_km=1.0
    )
    assert len(grid) > 0
    assert set(grid["source"]) == {"grid"}
    zone_poly = zones.geometry.iloc[0]
    assert all(zone_poly.contains(p) for p in grid.geometry)


def test_add_grid_seeds_drops_points_near_poi() -> None:
    zones = gpd.GeoDataFrame(
        {"zone_id": [0]},
        geometry=[_square(120.40, 120.60, 31.00, 31.20)],
        crs="EPSG:4326",
    )
    base = add_grid_seeds(
        _empty_poi(), zones, bbox=_BBOX, spacing_deg=0.03, min_separation_km=1.0
    )
    # put a POI exactly on one grid point
    target = base.iloc[0]
    poi = gpd.GeoDataFrame(
        {"lon": [target["lon"]], "lat": [target["lat"]], "source": ["poi_mall"]},
        geometry=gpd.points_from_xy([target["lon"]], [target["lat"]]),
        crs="EPSG:4326",
    )
    pruned = add_grid_seeds(
        poi, zones, bbox=_BBOX, spacing_deg=0.03, min_separation_km=1.0
    )
    assert len(pruned) == len(base) - 1
    dist = haversine_km(
        pruned["lat"].to_numpy(),
        pruned["lon"].to_numpy(),
        target["lat"],
        target["lon"],
    )
    assert (dist > 1.0).all()


def test_finalize_candidates_drops_outside_and_assigns_ids() -> None:
    zones = gpd.GeoDataFrame(
        {"zone_id": [0, 1]},
        geometry=[
            _square(120.40, 120.50, 31.00, 31.10),
            _square(120.50, 120.60, 31.10, 31.20),
        ],
        crs="EPSG:4326",
    )
    # two POIs inside zone 0, one POI well outside both zones
    p_lon, p_lat = [120.45, 120.42, 121.20], [31.05, 31.08, 31.90]
    poi = gpd.GeoDataFrame(
        {
            "lon": p_lon,
            "lat": p_lat,
            "source": ["poi_mall", "poi_subway", "poi_industrial"],
        },
        geometry=gpd.points_from_xy(p_lon, p_lat),
        crs="EPSG:4326",
    )
    # one grid candidate inside zone 1
    grid = gpd.GeoDataFrame(
        {"lon": [120.55], "lat": [31.15], "source": ["grid"]},
        geometry=gpd.points_from_xy([120.55], [31.15]),
        crs="EPSG:4326",
    )
    cand = finalize_candidates(poi, grid, zones)
    # the outside POI is dropped -> 2 POI + 1 grid
    assert len(cand) == 3
    assert list(cand["cand_id"]) == [0, 1, 2]
    assert "poi_industrial" not in set(cand["source"])
    assert cand.loc[cand["source"] == "grid", "zone_id"].iloc[0] == 1
    # cand_id follows lexicographic (lon, lat) order
    assert list(cand["lon"]) == sorted(cand["lon"])


def test_pull_poi_reads_from_cache(tmp_path) -> None:
    cache = tmp_path / "poi_cache.geojson"
    sample = gpd.GeoDataFrame(
        {"lon": [120.5], "lat": [31.1], "source": ["poi_mall"]},
        geometry=gpd.points_from_xy([120.5], [31.1]),
        crs="EPSG:4326",
    )
    sample.to_file(cache, driver="GeoJSON")
    got = pull_poi(cache_path=cache)
    assert len(got) == 1
    assert got.iloc[0]["source"] == "poi_mall"
