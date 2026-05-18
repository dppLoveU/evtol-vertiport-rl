# Stage 2 — Spatial Discretization: Completion Summary

Status: **complete (6/6) 2026-05-18**. Plan: `docs/plan/stage2_spatial_discretization.md`.
Commits: `7bd12c1` (tasks 1-4), `c08a54b` (task 5), `7e5a572` (task 6).

## 1. Core files produced

### Source / code (git-tracked)
| File | Role |
|------|------|
| `src/data/zones.py` | `build_zones` — H3 res-7 demand zones from order endpoints |
| `src/data/candidates.py` | `pull_poi`, `add_grid_seeds`, `finalize_candidates` — candidate vertiport set |
| `src/data/spatial.py` | `build_matrices` — zone/candidate distance + coverage matrices |
| `configs/spatial.yaml` | single source of truth for all stage-2 parameters |
| `experiments/run_stage2_build.py` | CLI: tasks 1-5 (zones, candidates, matrices) |
| `experiments/run_stage2_maps.py` | CLI: task 6 (folium maps) |
| `tests/test_zones.py`, `tests/test_candidates.py`, `tests/test_spatial.py` | unit tests |

### Data artifacts (in `data/processed/`, gitignored)
| File | Shape / content |
|------|-----------------|
| `zones.geojson` | 530 H3 hexagons — `zone_id`, `h3_index`, `centroid_lon/lat`, `n_orders`, `geometry` |
| `candidates.geojson` | 813 candidate sites — `cand_id`, `lon`, `lat`, `source`, `zone_id`, `geometry` |
| `dist_zone_zone.npy` | `[530, 530]` float32, symmetric, zero diagonal — haversine km between zone centroids |
| `dist_zone_cand.npy` | `[530, 813]` float32 — haversine km zone centroid ↔ candidate |
| `cand_covers_zones.npy` | `[813, 530]` bool — `True` where `dist_zone_cand[z,c] <= walk_radius_km` |
| `spatial_meta.json` | bookkeeping: counts, walk radius, h3 resolution, source counts, coverage ratio, build timestamp |

### Maps (in `results/stage2/maps/`, gitignored)
- `zones_map.html` — 530 H3 zones shaded by `n_orders` (quantile-binned YlOrRd).
- `candidates_map.html` — 813 candidates as source-colored markers + legend.
- `coverage_map.html` — highest-coverage candidate (`cand_id=254`, 23 zones within 5 km walk radius).

## 2. Final quantities

- **|Z| = 530** demand zones (H3 resolution 7; acceptance window [350, 800]).
- **|C| = 813** candidate vertiport sites (acceptance window [600, 1500]).
  Source breakdown: `poi_industrial` 328, `poi_subway` 265, `grid` 99,
  `poi_hospital` 73, `poi_mall` 47, `poi_airport` 1.
- **coverage_ratio = 0.9962** — fraction of demand zones with at least one
  candidate within the 5.0 km walk radius (acceptance threshold ≥ 0.85).

## 3. Tests

`pytest tests/test_zones.py tests/test_candidates.py tests/test_spatial.py` → **16/16 pass**.

- `test_zones.py` (4): min-orders threshold, lexicographic contiguous `zone_id`,
  centroid-inside-polygon + resolution, determinism.
- `test_candidates.py` (6): source classification + priority, H3 dedup,
  grid seeds inside zones, grid seeds drop near POIs, finalize drops
  out-of-zone candidates + assigns ids, POI cache read.
- `test_spatial.py` (6): matrix shapes/dtypes, zone-zone symmetry, zero
  diagonal, no NaN, coverage mask == distance threshold, coverage ratio range.

## 4. What is gitignored

`.gitignore` excludes (root-anchored, so `src/data/` is **not** affected):

- `/data/` — includes `data/processed/` (the `.geojson` / `.npy` / `spatial_meta.json`
  above), `data/synthetic/`, and the `data/raw` symlink.
- `/results/` — includes `results/stage2/maps/*.html` and the Stage 1 artifacts.
- `/models/`, `/wandb/` — model checkpoints and run logs.
- `cache/` — osmnx HTTP response cache (regenerable).
- Environment / build caches: `.venv/`, `__pycache__/`, `*.pyc`,
  `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.DS_Store`, `.env`.

Note: in commit `7e5a572` the previously-tracked Stage 1 `results/` files
were removed from the index (`git rm --cached`) so `results/` is no longer
tracked. Data and result artifacts are reproducible by re-running the
stage-2 CLIs.

## 5. Stage 3 prerequisites — files that must exist before starting

Per `docs/plan/stage3_od_construction.md` (Inputs):

| File | Produced by | Present? |
|------|-------------|----------|
| `data/processed/orders_clean.parquet` | Stage 1 | yes (4,050,523 rows) |
| `data/processed/zones.geojson` | Stage 2 task 1 | yes (530 zones) |
| `data/processed/dist_zone_zone.npy` | Stage 2 task 5 | yes (`[530,530]`) — used by Stage 3 trip-distance filter |

Also available and consumed by later stages (RL env / training):
`data/processed/candidates.geojson`, `dist_zone_cand.npy`,
`cand_covers_zones.npy`, `spatial_meta.json`.

Because all of `data/processed/` is gitignored, on a fresh checkout these
must be regenerated:

```
python -m experiments.run_stage1_clean       # -> orders_clean.parquet
python -m experiments.run_stage2_build       # -> zones, candidates, matrices, meta
python -m experiments.run_stage2_maps        # -> maps (optional, visualization only)
```

`run_stage2_build` needs the `data/raw` symlink (raw advisor CSV) and, on a
cold run, network access for the OSM POI pull (cached afterwards under
`data/raw/osm_poi_suzhou.geojson`).
