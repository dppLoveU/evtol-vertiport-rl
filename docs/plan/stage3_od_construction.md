# Stage 3: OD Tensor Construction & eVTOL Filtering

## Purpose

Aggregate the cleaned orders into time-resolved OD tensors at the zone
level. Produce two tensors:

1. **Full OD tensor `M`**: every order, useful for training the
   diffusion model on richer signal.
2. **eVTOL-eligible OD tensor `M_eV`**: filtered by trip length and
   duration thresholds — this is what the RL agent is rewarded for
   covering.

Also produce comprehensive EDA to characterize the eVTOL demand pattern.
This stage is the primary source of figures for the paper's Case Study
section.

## Inputs

- `data/processed/orders_clean.parquet` from Stage 1.
- `data/processed/zones.geojson` from Stage 2.
- `data/processed/dist_zone_zone.npy` from Stage 2 (for filter (b)
  below).

## Outputs

- `data/processed/od_full.npy` — shape `[T, |Z|, |Z|]`, int32, full OD
  count per time slot.
- `data/processed/od_evtol.npy` — shape `[T, |Z|, |Z|]`, int32, eVTOL-
  filtered OD count.
- `data/processed/od_evtol_weighted.npy` — shape `[T, |Z|, |Z|]`,
  float32, eVTOL OD weighted by trip value (used by alternative reward;
  weight = fare or duration depending on config).
- `data/processed/od_meta.json` — `T`, `|Z|`, `time_bin_min`,
  `start_datetime`, `evtol_filter_params`, `share_of_evtol_trips`.
- `results/stage3/eda/` — figures for the paper:
  - `od_share_by_hour.png` — % eVTOL-eligible by hour, weekday vs weekend.
  - `top20_od_pairs.png` — bar chart of busiest eVTOL OD pairs.
  - `heatmap_o_volume.png` — choropleth of total eVTOL origin volume
    per zone, overlay 7-day mean.
  - `heatmap_d_volume.png` — same for destinations.
  - `temporal_pattern.png` — hourly volume curves (24 lines for 24
    hours), full vs eVTOL.
  - `distance_distribution.png` — histogram of trip distances, full vs
    eVTOL highlighting the cut-off.
  - `fare_distribution.png` — fare histogram, full vs eVTOL.

## Tasks

1. **Time bin assignment** (`src/data/od.py`, `assign_time_bin`):
   - Define `t0 = min(dep_time)`.
   - For each order, `slot = floor((dep_time - t0).total_seconds() /
     (TIME_BIN_MIN * 60))`.
   - Assert `0 <= slot < NUM_TIME_BINS`.

2. **Zone assignment** (`assign_zones`):
   - Compute H3 index at resolution 7 for each order's O and D.
   - Map to `zone_id` via the lookup from Stage 2 (`h3_index -> zone_id`).
   - Orders whose O or D maps to an H3 cell not in our zone set are
     dropped (this is expected for sparse outlier cells).

3. **Build `od_full`** (`build_od_tensor`):
   - Group by `(slot, o_zone, d_zone)`, count rows.
   - Materialize into a dense `[T, |Z|, |Z|]` int32 numpy array.
   - Memory check: `T * |Z|^2 * 4 bytes`. With `T=336, |Z|=250`,
     that's 336 * 62500 * 4 ≈ 84 MB. Fine. If `|Z|` ends up >400,
     consider sparse representation (`scipy.sparse.COO` per slot).

4. **Define eVTOL filter** (`is_evtol_eligible`):
   An order is eVTOL-eligible if ALL hold:
   - `(a)` `geo_dist_km >= EVTOL_MIN_DIST_KM` (default 15.0)
   - `(b)` `duration_min >= EVTOL_MIN_DURATION_MIN` (default 25.0)
   - `(c)` `o_zone != d_zone` (eVTOL is wasted on intra-zone trips)

   All thresholds live in `configs/od.yaml` for sensitivity analysis.

5. **Build `od_evtol`**: same as task 3 but only over eligible rows.

6. **Build `od_evtol_weighted`**: same aggregation but `sum(weight)`
   instead of `count`. `weight` is `fare_yuan` by default, configurable
   to `duration_min` (time savings proxy).

7. **Compute and log share statistics**:
   - `share_total = od_evtol.sum() / od_full.sum()` — overall eVTOL
     share. Expected 5–15%.
   - `share_by_hour[h]` — same per hour-of-day.
   - Save to `od_meta.json`.

8. **Generate EDA figures** (`experiments/run_stage3_eda.py`).
   This script is for the paper, treat it carefully:
   - Use matplotlib with `seaborn-v0_8-whitegrid` style.
   - All fonts ≥ 10 pt for print readability.
   - Save as PNG at 300 dpi AND as PDF for the manuscript.

9. **Sensitivity preview** (`experiments/run_stage3_sensitivity.py`):
   Sweep `EVTOL_MIN_DIST_KM` over `{10, 12, 15, 18, 20}` km and report
   the share of eligible trips and the spatial concentration (entropy
   of OD distribution) for each. Output a small table to
   `results/stage3/sensitivity.csv`. This data goes into the paper's
   sensitivity section.

## Acceptance Criteria

- [ ] `od_full.sum()` equals the number of orders whose O and D both
  fall into known zones (allow ≤2% drop from Stage 1 total due to zone
  filter).
- [ ] `od_evtol.sum() / od_full.sum()` is in `[0.03, 0.20]`. If outside
  this range, revisit the eVTOL thresholds.
- [ ] No NaN, no negative values.
- [ ] Sum over the diagonal of `od_evtol[t]` is zero for every `t`
  (intra-zone trips filtered).
- [ ] All 7 EDA figures generated and visually sensible.
- [ ] Sensitivity CSV has 5 rows.
- [ ] `od_meta.json` is valid JSON, contains all required fields.
- [ ] `tests/test_od.py` passes.

## Files to Create

- `src/data/od.py` — `assign_time_bin`, `assign_zones`,
  `is_evtol_eligible`, `build_od_tensor`.
- `src/data/eda_plots.py` — reusable plotting helpers (centralized so
  Stage 7 can reuse the same style).
- `configs/od.yaml` — time bin size, eVTOL thresholds, weight choice.
- `experiments/run_stage3_build.py` — orchestrate od_full, od_evtol,
  od_evtol_weighted, save meta.
- `experiments/run_stage3_eda.py` — generate paper figures.
- `experiments/run_stage3_sensitivity.py` — threshold sweep.
- `tests/test_od.py` — tests for time bin math, zone assignment, eVTOL
  filter logic on small synthetic data.

## Common Pitfalls

- **Time bin off-by-one**: orders right at `t0 + 7*24*60 minutes` get
  `slot == NUM_TIME_BINS` (out of range). Clamp or drop.
- **Zone lookup speed**: a naive Python dict lookup over 4M rows is
  slow (~30 s). Use `pd.Series.map` with a dict, or join on H3 index.
- **Dense vs sparse**: `od_evtol` is mostly zeros. Saving as `.npy`
  dense is fine for `|Z| ≤ 300` but uses 80+ MB. Consider also saving
  a sparse `.npz` (`scipy.sparse.save_npz` for each slot stacked) if
  Stage 4 needs faster IO.
- **`scipy.sparse` and 3D**: scipy sparse is 2D only. For 3D, store a
  list of `T` sparse matrices in a pickle, or stick with dense.
- **Weighted aggregation**: when computing `od_evtol_weighted`, do not
  forget to apply the eVTOL filter. Easy mistake when refactoring.
- **EDA double-counting**: when plotting "trip volume by hour", aggregate
  by `slot % 48` to fold the 7 days, but compute weekday/weekend
  separately to show pattern differences. Don't average all 7 days into
  one curve if the pattern matters.
- **Figure reuse**: every figure produced here will likely appear in
  the paper. Generate both PNG (for slides) and PDF (for LaTeX).

## Dependencies

- Stage 1 (`orders_clean.parquet`)
- Stage 2 (`zones.geojson`)

## Estimated effort

2 days. Mostly EDA polish to make figures publication-ready.