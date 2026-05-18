"""Stage-2 tasks 2-3: candidate vertiport site generation.

``pull_poi`` fetches OSM points of interest (transit hubs, malls,
hospitals, industrial parks, airports) inside the study bbox.
``add_grid_seeds`` pads coverage with a uniform grid, dropping points
that fall outside any demand zone or too close to an existing POI.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

from src.constants import SUZHOU_BBOX
from src.utils.geo import haversine_km

# OSM tag set for the POI categories of interest (stage-2 plan, Inputs).
DEFAULT_POI_TAGS: dict[str, list[str]] = {
    "railway": ["station", "halt"],
    "station": ["subway"],
    "shop": ["mall"],
    "aeroway": ["aerodrome"],
    "amenity": ["hospital"],
    "landuse": ["industrial"],
}

# Source ranking for the H3 dedup tie-break (lower = kept first). See
# docs/decisions.md 2026-05-18 "Stage 2: POI tightening".
_SOURCE_PRIORITY: dict[str, int] = {
    "poi_airport": 0,
    "poi_subway": 1,
    "poi_mall": 2,
    "poi_hospital": 3,
    "poi_industrial": 4,
}

_POLYGON_TYPES = ("Polygon", "MultiPolygon")


def _classify_source(feat: pd.Series) -> str | None:
    """Map an OSM feature to a candidate ``source`` label, or None to drop.

    Categories are checked most-specific first so a feature carrying
    several tags lands in a single deterministic bucket.
    """
    if feat.get("aeroway") == "aerodrome":
        return "poi_airport"
    if feat.get("railway") in ("station", "halt") or feat.get("station") == "subway":
        return "poi_subway"
    if feat.get("shop") == "mall":
        return "poi_mall"
    if feat.get("amenity") == "hospital":
        return "poi_hospital"
    if feat.get("landuse") == "industrial":
        return "poi_industrial"
    return None


def _dedupe_by_h3(poi: gpd.GeoDataFrame, resolution: int) -> gpd.GeoDataFrame:
    """Keep one POI per H3 cell at ``resolution``.

    On a collision the highest-priority source wins (see
    ``_SOURCE_PRIORITY``); ties keep the first occurrence.
    """
    if len(poi) == 0:
        return poi.reset_index(drop=True)
    cells = [
        h3.geo_to_h3(la, lo, resolution)
        for lo, la in zip(poi["lon"].to_numpy(), poi["lat"].to_numpy())
    ]
    ranked = poi.assign(
        _h3=cells, _prio=poi["source"].map(_SOURCE_PRIORITY)
    ).sort_values("_prio", kind="stable")
    kept = ranked.drop_duplicates(subset="_h3", keep="first")
    return kept.drop(columns=["_h3", "_prio"]).reset_index(drop=True)


def pull_poi(
    bbox: dict[str, float] = SUZHOU_BBOX,
    tags: dict[str, list[str]] | None = None,
    cache_path: str | Path | None = None,
    hospital_min_area_m2: float = 10_000.0,
    industrial_min_area_m2: float = 50_000.0,
    dedupe_h3_res: int | None = 8,
) -> gpd.GeoDataFrame:
    """Fetch and classify OSM POI candidate sites within ``bbox``.

    The classified+filtered result is cached to ``cache_path`` (GeoJSON)
    so reruns do not re-hit the OSM API. Polygon features are reduced to
    their centroid; the size-gated categories (hospital, industrial)
    drop polygons below the given area thresholds. As a final step the
    merged POI set is collapsed to one site per H3 cell at
    ``dedupe_h3_res`` (``None`` disables); this runs after the cache
    branch, so the cached file stays the pre-dedup set.

    Returns a GeoDataFrame with columns ``lon``, ``lat``, ``source``.
    """
    tags = tags or DEFAULT_POI_TAGS
    cache = Path(cache_path) if cache_path else None

    if cache is not None and cache.exists():
        poi = gpd.read_file(cache)
    else:
        import osmnx as ox

        # osmnx v2 bbox order is (left, bottom, right, top) = (W, S, E, N).
        raw = ox.features_from_bbox(
            bbox=(bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]),
            tags=tags,
        ).reset_index(drop=True)

        # Project once for metric area and centroid computation.
        proj = raw.to_crs(3857)
        area_m2 = proj.geometry.area.to_numpy()
        centroids = proj.geometry.centroid.to_crs(4326)

        rows: list[dict[str, object]] = []
        for i in range(len(raw)):
            source = _classify_source(raw.iloc[i])
            if source is None:
                continue
            is_poly = raw.geometry.iloc[i].geom_type in _POLYGON_TYPES
            if (
                source == "poi_hospital"
                and is_poly
                and area_m2[i] < hospital_min_area_m2
            ):
                continue
            if (
                source == "poi_industrial"
                and is_poly
                and area_m2[i] < industrial_min_area_m2
            ):
                continue
            pt = centroids.iloc[i]
            rows.append({"lon": float(pt.x), "lat": float(pt.y), "source": source})

        df = pd.DataFrame(rows, columns=["lon", "lat", "source"])
        # Drop exact-duplicate coordinates (~11 m at 4 dp).
        df = df.assign(_klon=df["lon"].round(4), _klat=df["lat"].round(4))
        df = df.drop_duplicates(subset=["_klon", "_klat"])
        df = df.drop(columns=["_klon", "_klat"]).reset_index(drop=True)

        poi = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
        )
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            poi.to_file(cache, driver="GeoJSON")

    if dedupe_h3_res is not None:
        poi = _dedupe_by_h3(poi, dedupe_h3_res)
    return poi


def add_grid_seeds(
    poi_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    bbox: dict[str, float] = SUZHOU_BBOX,
    spacing_deg: float = 0.03,
    min_separation_km: float = 1.0,
) -> gpd.GeoDataFrame:
    """Generate uniform-grid candidate sites to pad spatial coverage.

    Grid points that fall outside every demand zone, or within
    ``min_separation_km`` of any POI, are dropped. Returns a GeoDataFrame
    with columns ``lon``, ``lat``, ``source`` (always ``"grid"``).
    """
    lons = np.arange(bbox["lon_min"], bbox["lon_max"], spacing_deg)
    lats = np.arange(bbox["lat_min"], bbox["lat_max"], spacing_deg)
    mesh_lon, mesh_lat = np.meshgrid(lons, lats)
    grid = gpd.GeoDataFrame(
        {"lon": mesh_lon.ravel(), "lat": mesh_lat.ravel()},
        geometry=gpd.points_from_xy(mesh_lon.ravel(), mesh_lat.ravel()),
        crs="EPSG:4326",
    )

    # Keep only grid points that fall inside a demand zone.
    inside = gpd.sjoin(grid, zones_gdf[["geometry"]], how="inner", predicate="within")
    grid = grid[grid.index.isin(inside.index)].reset_index(drop=True)

    # Drop grid points too close to an existing POI.
    if len(poi_gdf) > 0 and len(grid) > 0:
        dist = haversine_km(
            grid["lat"].to_numpy()[:, None],
            grid["lon"].to_numpy()[:, None],
            poi_gdf["lat"].to_numpy()[None, :],
            poi_gdf["lon"].to_numpy()[None, :],
        )
        keep = dist.min(axis=1) > min_separation_km
        grid = grid[keep].reset_index(drop=True)

    grid["source"] = "grid"
    return grid[["lon", "lat", "source", "geometry"]]


def finalize_candidates(
    poi_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Merge POI and grid candidates into the final candidate set.

    Each candidate is assigned the ``zone_id`` of its containing demand
    zone; any candidate outside every zone (its cell has sub-threshold
    demand) is dropped. ``cand_id`` is assigned by lexicographic order
    of ``(lon, lat)`` for determinism.

    Returns a GeoDataFrame with columns ``cand_id``, ``lon``, ``lat``,
    ``source``, ``zone_id``.
    """
    cols = ["lon", "lat", "source", "geometry"]
    merged = gpd.GeoDataFrame(
        pd.concat([poi_gdf[cols], grid_gdf[cols]], ignore_index=True),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        merged, zones_gdf[["zone_id", "geometry"]], how="inner", predicate="within"
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    cand = (
        joined.drop(columns="index_right")
        .sort_values(["lon", "lat"], kind="stable")
        .reset_index(drop=True)
    )
    cand["cand_id"] = range(len(cand))
    return cand[["cand_id", "lon", "lat", "source", "zone_id", "geometry"]]
