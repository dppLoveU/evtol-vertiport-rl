# Stage 2: Spatial Discretization

## Purpose

Define two spatial representations:

1. **Demand zones (`Z`)**: coarser units used to aggregate OD demand.
   Chosen as H3 hexagons at resolution 7 (~1.2 km edge).
2. **Candidate vertiport sites (`C`)**: a discrete set of locations
   where a vertiport may potentially be placed. Constructed as a hybrid
   of POI-based and uniform-grid-based seeds.

Also build the spatial helper structures (zone-to-candidate distance
matrix, zone-to-zone distance matrix) needed by Stages 3, 5, 6.

## Inputs

- `data/processed/orders_clean.parquet` from Stage 1.
- (Optional) OSM POI export for Suzhou. If not pre-downloaded, fetch via
  `osmnx` API in the script. POI categories of interest:
  - `railway=station`, `railway=halt`
  - `subway=yes` and `station=subway`
  - `shop=mall`
  - `aeroway=aerodrome`
  - `amenity=hospital` (large only, filter by polygon area)
  - `landuse=industrial` (large parks only)

## Outputs

- `data/processed/zones.geojson` — `|Z|` H3 hexagon polygons with:
  ```
  zone_id     int      0..|Z|-1
  h3_index    str      H3 cell id at resolution 7
  centroid_lon float
  centroid_lat float
  geometry    Polygon
  ```
- `data/processed/candidates.geojson` — `|C|` candidate points with:
  ```
  cand_id     int      0..|C|-1
  lon         float
  lat         float
  source      str      "poi_subway" | "poi_mall" | "poi_hospital" |
                       "poi_industrial" | "poi_airport" | "grid"
  zone_id     int      which demand zone it falls in
  ```
- `data/processed/dist_zone_zone.npy` — `[|Z|, |Z|]` float32, haversine
  km between zone centroids.
- `data/processed/dist_zone_cand.npy` — `[|Z|, |C|]` float32, haversine
  km between each zone centroid and each candidate.
- `data/processed/cand_covers_zones.npy` — `[|C|, |Z|]` bool, True if
  candidate `c` is within `WALK_RADIUS_KM` of zone `z` centroid.
- `data/processed/spatial_meta.json` — bookkeeping: `|Z|`, `|C|`,
  resolution, walk radius, source counts, build timestamp.
- `results/stage2/maps/` — `folium` interactive maps:
  - `zones_map.html` — hexagons colored by demand density (preview).
  - `candidates_map.html` — candidates colored by source.
  - `coverage_map.html` — example: pick a random candidate, highlight
    the zones it covers.

## Tasks

1. **Build demand zones** (`src/data/zones.py`, `build_zones`):
   1. Take the union of all O and D coordinates from the cleaned
      orders. Compute the H3 index at resolution 7 for each unique
      (lon, lat) — use `h3.geo_to_h3(lat, lon, 7)`.
   2. Keep only H3 cells that contain at least `min_orders_per_zone`
      points (default 50) to drop ghost cells in lakes/farmland.
   3. Assign `zone_id` by sorting H3 indices lexicographically for
      determinism.
   4. For each zone, compute centroid (`h3.h3_to_geo`) and polygon
      (`h3.h3_to_geo_boundary`).
   5. Save GeoJSON.

2. **Pull POI candidates** (`src/data/candidates.py`, `pull_poi`):
   1. Use `osmnx.features_from_bbox` with the SUZHOU_BBOX and the
      tag set above. Set network_type to None (we want features, not
      network).
   2. For polygon features, take the centroid.
   3. Deduplicate by `(round(lat, 4), round(lon, 4))`.
   4. Tag each by `source`.
   5. Expect on the order of 200–400 POIs total. If <100, broaden tags;
      if >800, tighten (e.g., only major malls).

3. **Pad with uniform grid candidates** (`add_grid_seeds`):
   1. Generate a uniform grid of points inside SUZHOU_BBOX with spacing
      ~3 km (~0.03°). Drop those falling in zones with zero demand.
   2. Drop grid points within `min_separation_km` (default 1.0) of any
      POI to avoid duplication.
   3. Tag as `source="grid"`.

4. **Merge and finalize candidates** (`finalize_candidates`):
   1. Concatenate POI + grid candidates.
   2. Assign each candidate to its containing zone (`zone_id`); drop
      candidates whose zone has zero demand.
   3. Assign `cand_id` by lexicographic sort.
   4. Save GeoJSON.
   5. Target `|C|` in `[600, 1500]` (revised — see `docs/decisions.md`
      2026-05-18). If outside this range, log a warning and stop for
      human review.

5. **Build distance and coverage matrices** (`build_matrices`):
   1. `dist_zone_zone[i, j]`: haversine_km between zones `i` and `j`'s
      centroids. Vectorize with numpy; the loop should be over zones
      only, not orders.
   2. `dist_zone_cand[z, c]`: same, between zone centroid and
      candidate location.
   3. `cand_covers_zones[c, z] = dist_zone_cand[z, c] <= WALK_RADIUS_KM`.
   4. Save as `.npy` files.

6. **Generate maps** (`experiments/run_stage2_maps.py`):
   Build the three `folium` maps described in Outputs.

## Acceptance Criteria

- [ ] `|Z|` falls in `[350, 800]`. If outside, investigate.
- [ ] `|C|` falls in `[600, 1500]` (revised from [200, 500] after the
  metro bbox expansion — see `docs/decisions.md` 2026-05-18).
- [ ] Every zone has at least one candidate (or is explicitly logged
  as "uncovered by design").
- [ ] `dist_zone_zone` is symmetric (`np.allclose(D, D.T)`) and has
  zero diagonal.
- [ ] No NaN in any distance matrix.
- [ ] `cand_covers_zones.any(axis=0).sum() / |Z|` ≥ 0.85, i.e. at least
  85% of demand zones have at least one candidate within walk radius
  (otherwise the problem is degenerate).
- [ ] All three maps render and look sane on visual inspection.
- [ ] Tests in `tests/test_zones.py` and `tests/test_candidates.py`
  pass.

## Files to Create

- `src/data/zones.py` — `build_zones(orders_df) -> gpd.GeoDataFrame`.
- `src/data/candidates.py` — `pull_poi`, `add_grid_seeds`,
  `finalize_candidates`.
- `src/data/spatial.py` — `build_matrices(zones_gdf, cands_gdf) -> dict`.
- `configs/spatial.yaml` — H3 resolution, min_orders_per_zone, POI
  tags, grid spacing, walk radius.
- `experiments/run_stage2_build.py` — orchestration: zones, candidates,
  matrices.
- `experiments/run_stage2_maps.py` — folium visualization.
- `tests/test_zones.py`, `tests/test_candidates.py`,
  `tests/test_spatial.py`.

## Common Pitfalls

- **H3 lat/lon order**: `h3-py` uses `(lat, lon)` order, opposite to
  most GIS libraries. Triple-check.
- **OSMnx rate limits**: the API can be slow or throttle. Cache the raw
  POI download to `data/raw/osm_poi_suzhou.geojson` so we don't re-pull
  on every script run.
- **CRS**: keep everything in EPSG:4326 (WGS84 lon/lat) for simplicity.
  All distances are haversine, not projected; that's fine for Suzhou's
  small extent.
- **GeoPandas memory**: do not load all 4M points into a GeoDataFrame;
  use plain pandas and only build geometries for the unique H3 cells.
- **Zone holes**: H3 cells over water bodies (Taihu lake) may have a
  few stray GPS points. The `min_orders_per_zone` threshold filters
  these but check the map.
- **Walk radius semantics**: we model "walk to vertiport" as straight-
  line haversine. This is an approximation; real walking distance is
  1.2–1.4× longer. Note this limitation in the paper.
- **bbox expanded since this plan was drafted**: `SUZHOU_BBOX` was
  widened to the full metropolitan area at the end of Stage 1 (~7×
  the original City-proper area). The `|Z|` / `|C|` capacity estimates
  here have been updated accordingly — see `docs/decisions.md`
  2026-05-18.

## Dependencies

Stage 1 (`orders_clean.parquet`).

Adds to `pyproject.toml`: `h3`, `osmnx`, `geopandas`, `shapely`,
`folium`, `branca`.

## Estimated effort

2 days. The bulk of time is OSM exploration and tuning POI filters.