# Stage 1: Raw Data Cleaning

## Purpose

Convert the advisor-provided raw CSV (~4M rows, 54 columns) into a
clean, typed, validated Parquet file that downstream stages can rely on
without re-checking edge cases.

## Inputs

- **File**: `data/raw/suzhou_orders_7days.csv`
- **Rows**: approximately 4,000,000
- **Schema notes** (only columns we care about; others to be dropped):

| Column              | Raw type      | Cleaning                        |
|---------------------|---------------|---------------------------------|
| `fGuid`             | str           | keep as `order_id`              |
| `fDepLongitude`     | int           | divide by 1e6 → float lon       |
| `fDepLatitude`      | int           | divide by 1e6 → float lat       |
| `fDestLongitude`    | int           | divide by 1e6                   |
| `fDestLatitude`     | int           | divide by 1e6                   |
| `fDepTime`          | "YYYY/M/D H:MM" str | parse to datetime         |
| `fDestTime`         | same          | parse to datetime               |
| `fDriveMile`        | float (km)    | keep as `drive_km`              |
| `fDriveTime`        | int (seconds) | divide by 60 → `drive_min`      |
| `fWaitTime`         | int           | keep as `wait_min`              |
| `fFactPrice`        | float         | keep as `fare_yuan`             |
| `AreaName`          | str           | keep                            |
| `性别`              | str           | rename to `gender`              |
| `年龄`              | int           | rename to `age`                 |

All other columns: drop.

## Outputs

- **Primary**: `data/processed/orders_clean.parquet`
  Schema:
  ```
  order_id        str
  o_lon           float32
  o_lat           float32
  d_lon           float32
  d_lat           float32
  dep_time        datetime64[ns]
  dest_time       datetime64[ns]
  duration_min    float32     # derived
  drive_km        float32     # from fDriveMile
  drive_min       float32     # from fDriveTime / 60
  geo_dist_km     float32     # haversine(O, D), derived
  wait_min        float32
  fare_yuan       float32
  area_name       str
  gender          str
  age             int8
  ```
  Compression: snappy. Row group size: 100k.

- **EDA report**: `results/stage1/eda_summary.html` (pandas-profiling
  or sweetviz) and a few hand-picked plots in `results/stage1/plots/`:
  - histogram of `duration_min` (before and after filter)
  - histogram of `drive_km`
  - log-scale histogram of `fare_yuan`
  - scatter of `geo_dist_km` vs `drive_km` (sanity check)
  - hourly trip volume over the 7 days

## Tasks

1. **Implement loader** (`src/data/clean.py`, function `load_raw`):
   read the CSV. Use `pd.read_csv` with `dtype` dict from
   `src/data/schema.py` to avoid the default float64 explosion. If
   memory pressure is an issue, use `chunksize=500_000` and concatenate.
2. **Coordinate fix**: vectorized divide by 1e6 for all four coordinate
   columns. Cast to float32.
3. **Time parsing**: `pd.to_datetime(..., format="%Y/%m/%d %H:%M",
   errors="coerce")`. Rows where parsing fails go to a side dataframe
   for debugging; in production they are dropped.
4. **Derived columns**:
   - `duration_min = (dest_time - dep_time).total_seconds() / 60`
   - `drive_min = fDriveTime / 60`
   - `geo_dist_km = haversine((o_lat, o_lon), (d_lat, d_lon))`. Implement
     haversine in `src/utils/geo.py` as a vectorized numpy function.
5. **Outlier filtering**. Apply in this exact order, recording the row
   count after each step to a dict for the EDA report:
   - (a) all four coordinates inside `SUZHOU_BBOX`
   - (b) `2 <= duration_min <= 180`
   - (c) `0.5 <= drive_km <= 100`
   - (d) `geo_dist_km >= 0.3` (drop near-zero OD)
   - (e) `geo_dist_km / drive_km < 3` (GPS sanity; drive should be ≥ geo)
   - (f) `0 < fare_yuan <= 1000`
6. **Column selection**: keep only the columns listed in the Outputs
   schema, in that order.
7. **Save Parquet**: `df.to_parquet(out_path, compression="snappy",
   row_group_size=100_000)`.
8. **EDA report**: a small script `experiments/run_stage1_eda.py` that
   loads the cleaned Parquet and produces the plots + HTML report.

## Acceptance Criteria

- [ ] `data/processed/orders_clean.parquet` exists.
- [ ] Row count between 3,400,000 and 3,900,000 (5–15% drop is normal).
- [ ] `df.isna().sum()` returns 0 for every column.
- [ ] All coordinates fall inside `SUZHOU_BBOX`.
- [ ] `df.dtypes` matches the schema above exactly.
- [ ] EDA report HTML is generated and the 5 plots exist.
- [ ] Unit tests in `tests/test_clean.py` pass:
  - `test_coordinate_scale` — verifies 1e6 division on a 10-row sample.
  - `test_time_parsing` — verifies "2023/7/16 10:38" parses correctly.
  - `test_outlier_filter` — verifies each filter (a)–(f) on synthetic
    rows that should be dropped.
  - `test_haversine` — known points (Beijing–Shanghai ≈ 1064 km, ±5%).

## Files to Create

- `src/data/__init__.py`
- `src/data/schema.py` — dtype dict, column rename map, filter
  thresholds (importable constants).
- `src/data/clean.py` — main `clean_orders(cfg)` function returning the
  cleaned dataframe and the filter audit dict.
- `src/utils/geo.py` — `haversine_km(lat1, lon1, lat2, lon2)`
  vectorized numpy function.
- `src/utils/__init__.py`
- `src/constants.py` — `SUZHOU_BBOX`, `COORD_SCALE`, etc. (see
  `CLAUDE.md` section 7).
- `configs/preprocess.yaml` — paths, filter thresholds, output options.
- `experiments/run_stage1_clean.py` — CLI entrypoint:
  `python -m experiments.run_stage1_clean`.
- `experiments/run_stage1_eda.py` — CLI entrypoint for EDA report.
- `tests/test_clean.py` — unit tests.
- `tests/fixtures/orders_sample.csv` — 1000-row hand-crafted CSV with
  known bad rows for testing.

## Common Pitfalls

- **CSV encoding**: the file may be GBK-encoded due to Chinese chars in
  some fields. Try `encoding="utf-8"` first; if `UnicodeDecodeError`,
  try `encoding="gbk"` or `encoding="utf-8-sig"`.
- **Datetime format**: month and day are not zero-padded ("2023/7/16",
  not "2023/07/16"). Use `format="%Y/%m/%d %H:%M"` not the ISO one.
- **Coordinate edge case**: some rows have `0` for coordinates
  (placeholder for missing). Outlier filter (a) catches these.
- **`fDriveTime` units**: seconds, not minutes. Easy to miss.
- **Chinese column names**: `性别` and `年龄`. Use them literally as
  Python strings (Python 3 handles UTF-8 column names) or rename
  during read with the `dtype` dict.
- **Memory**: full 4M-row CSV in pandas with default dtypes can hit
  6–8 GB. Cast aggressively to float32/int8.
- **Outlier filter order**: applying (e) before (d) can divide by zero.
  Stick to the order above.

## Dependencies

None; this is the first stage.

## Estimated effort

1–2 days for a careful first pass including tests.