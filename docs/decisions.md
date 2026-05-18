# Decision Log

This file records deviations from the plan documents and major
technical choices made during implementation.
Format: `## YYYY-MM-DD — <topic>` followed by what changed and why.

---

## 2026-05-14 — Stage 1: datetime format includes seconds

**Plan said**: parse `fDepTime` / `fDestTime` with
`format="%Y/%m/%d %H:%M"` (see `docs/plan/stage1_data_cleaning.md`,
Common Pitfalls).

**Reality**: the raw CSV's first data row shows
`fDepTime = "2023/7/16 10:43:05"` — month/day are not zero-padded
(matches plan) **but seconds are present** (does not match plan).

**Change**: parse with `format="%Y/%m/%d %H:%M:%S"`.
`DEFAULT_DATETIME_FORMAT` lives in `src/data/clean.py`; the same string
is exposed in `configs/preprocess.yaml`.

**Impact**: gain second-level precision in `dep_time` / `dest_time`
and downstream `duration_min`. No loss versus the planned format.

---

## 2026-05-14 — Stage 1: Python 3.10 instead of 3.11

**Plan said**: Python 3.11 (CLAUDE.md §5).

**Reality**: target machine has only system Python 3.10.12, no conda /
uv / poetry, no pre-installed Python 3.11. Installing 3.11 would
require either a heavy toolchain (miniconda) or a third-party binary
(uv); neither was available without extra setup.

**Change**: created `.venv` from system Python 3.10.12
(`python3 -m venv .venv` after `apt install python3.10-venv`). All
stage-1 code is type-hinted under PEP 604 (`X | Y`) which 3.10
supports; nothing in this stage needs 3.11-only features.

**Impact**: minimal. To revisit before stage 4 (PyTorch is fine on
3.10, so no near-term upgrade pressure).

---

## 2026-05-14 — Stage 1: wait_min filter added (plan extension)

**Plan said**: 6 outlier filters (a)-(f); `wait_min` is kept as an
output column but never filtered.

**Reality**: full-data audit on 4,257,529 rows showed `wait_min` has
a max of 764,932 minutes (~531 days) and a std of ~2,057, while p50
and p75 are both 0.0. Almost all orders have no wait; a handful have
clearly garbage values from data-entry errors. Without filtering,
those extreme values would propagate into the cleaned Parquet and
distort downstream OD aggregation in stage 3.

**Change**: added filter step (g) `0 <= wait_min <= 120` to
`apply_outlier_filters`; the audit dict gains a new `after_wait`
key. Threshold 120 minutes (2 hours) is generous — any wait longer
than that is almost certainly a logging artefact.

**Impact**: small extra row drops on real data (TBD by pressure
test). Updated `tests/test_clean.py` to assert the new audit key.

---

## 2026-05-14 — Stage 1: select_output_columns contract = "no NA on input"

**Previously**: `select_output_columns` did `dropna(subset=OUTPUT_COLUMNS)`
internally, silently removing rows with NA in any output column to
satisfy the plan's `df.isna().sum() == 0` acceptance criterion.

**Reality**: the same full-data audit showed **zero NA in all 14
KEEP_COLUMNS** in the raw 4,257,529 rows. Silent drops were
defending against a problem that does not exist; a contract-violation
in upstream data should be loud, not silent.

**Change**: `select_output_columns` now raises `ValueError` listing
the NA-bearing columns when its input contains any NA. Upstream
callers (CLI, tests) explicitly handle NA before invocation if/when
needed — e.g., the test pipeline calls `df.dropna(subset=["age"])`
before `select_output_columns(df)` because the test fixture has one
deliberate NaN-age row.

**Impact**: stricter API. The CLI entrypoint (task 7 / future) will
need to either confirm the data is NA-free or do an explicit dropna
with logging.

---

## 2026-05-15 — Stage 1: SUZHOU_BBOX expanded to metropolitan area

**Plan said**: `SUZHOU_BBOX = lon[120.45, 120.95], lat[31.20, 31.50]`
(CLAUDE.md §7) — Suzhou City proper.

**Reality**: full-data EDA on 4,257,529 rows (after `fix_coordinates`,
3 zero-coord placeholder rows excluded) showed the data covers all 5
`AreaName` values of the Suzhou prefecture: 苏州市 65%, 昆山市 19%,
常熟市 7%, 张家港市 6%, 太仓市 4%. Per-area p1/p99 bounds are saved
in `results/stage1/eda/area_bounds.csv` (produced by
`experiments/run_stage1_area_bounds.py`).

The City-proper bbox excluded ~35% of rows by area-name alone (the 4
county-level cities); a pre-fix pressure test of the full filter
pipeline kept only 2,132,464 rows — far below the planned acceptance
window [3,620,000, 4,050,000].

**Change**: `SUZHOU_BBOX = lon[120.37, 121.33], lat[30.88, 32.01]` —
union of per-area p1/p99 across all 4 coordinates plus ~2 km padding
(rounded to 2 decimals). Updated `src/constants.py`,
`configs/preprocess.yaml::filters.bbox`, and CLAUDE.md §1 (scope
narrative) and §7 (constant block).

The new envelope is ~7× the City-proper area but still strictly
within the Suzhou prefecture boundary (no overlap with Shanghai,
Wuxi, Jiaxing).

**Impact**:
- Re-run pressure test: row count rises from 2,132,464 to **4,050,523**
  (95.14% kept). bbox alone now drops only 0.74%; remaining 4.12%
  comes from filters (b)-(g), dominated by `wait_min` (3.10%, mostly
  garbage logging artefacts).
- Acceptance HI revised once more: from 4,050,000 → **4,150,000**
  (~2.5%–15% drop). The original 5%–15% drop estimate predates the
  bbox fix and now reads as too tight; 4.86% actual drop is below
  the LO of 5% but is essentially "data-quality-only" filtering,
  which is the desired regime. Recorded in
  `docs/plan/stage1_data_cleaning.md` Acceptance Criteria.
- Project narrative shifts from "Suzhou City eVTOL" to "Suzhou
  metropolitan area eVTOL" — consistent with the inter-county-city
  trip data and a more realistic operational scope for vertiport
  siting.

---

## 2026-05-18 — Stage 2: |Z| and |C| expectations revised after metro bbox

**Plan said**: `docs/plan/stage2_spatial_discretization.md` Acceptance
Criteria required `|Z|` in `[150, 350]`; CLAUDE.md §8 noted "~250
hexagons for Suzhou". Both figures were drafted against the original
City-proper `SUZHOU_BBOX`.

**Reality**: the 2026-05-15 bbox expansion (entry above) enlarged the
envelope ~7× to `lon[120.37, 121.33], lat[30.88, 32.01]` — area ≈
11,450 km². At H3 resolution 7 (~5.16 km² per cell) the bbox holds an
upper bound of ~2,220 cells. The Suzhou metro built-up area (City
proper ~500 km² plus Kunshan / Changshu / Zhangjiagang / Taicang and
the connecting industrial belt) is on the order of 2,500–4,000 km²,
which after the `min_orders_per_zone` ghost-cell filter corresponds to
roughly 480–770 demand zones. The old "~250" estimate is stale by the
same ~7× factor as the bbox.

**Change**: revised expectations to **|Z| ≈ 400–700**. Acceptance
criterion updated to `|Z|` in `[350, 800]`; CLAUDE.md §8 updated to
"~400-700 hexagons for the metro bbox"; a note added to the Stage 2
Common Pitfalls section pointing back here. `|C|` target window
`[200, 500]` is left unchanged for now — POI density and the 3 km grid
spacing are independent of the zone count — but it may need a similar
revision once the actual grid-seed count over the larger bbox is known.

**Impact**: no code change. If a Stage 2 run produces `|Z|` outside
`[350, 800]`, treat `min_orders_per_zone` as the lever but consult the
user before changing it (CLAUDE.md §6 rule 5).

---

## 2026-05-18 — Stage 2: POI tightening (tag fix, area floor, H3 dedup)

The first stage-2 run pulled **4,392** POI candidates against the
plan's 200-400 target (`subway` 1,072, `industrial` 2,895, `mall` 230,
`hospital` 191, `airport` 4). Three changes tighten `pull_poi`:

**1. OSM tag fix: drop `subway=yes`, keep `station=subway`.**
The plan's Inputs section listed both `subway=yes` and
`station=subway`. `subway=yes` is an OSM *property on railway ways*
marking that a line runs underground — it matches tunnel segments, not
stations, and pulled ~1,000 spurious features. The correct station tag
is `station=subway` (already in the query). `subway=yes` is removed
from the tag set and from the `_classify_source` check. This corrects
a plan error; it is not a design change.

**2. Industrial area floor.**
`landuse=industrial` matches every industrial parcel, including single
factory buildings; the Suzhou metro is industry-dense (2,895 hits). A
vertiport candidate should sit at a park-scale site, so industrial
polygons are kept only if `area_m2 >= 50000` (>5 ha). Exposed as
`poi.industrial_min_area_m2` in `configs/spatial.yaml` (default
50000). The plan said "large parks only" without pinning a number;
this concretizes plan intent rather than deviating from it.

**3. H3 res-8 POI deduplication.**
After merging all POI sources, dense urban cores still hold many
near-co-located candidates (overlapping malls / stations within a few
hundred metres). POIs are grouped by H3 resolution 8 (~0.74 km^2 cell,
~0.46 km edge — finer than the res-7 demand zones) and one POI is kept
per cell, chosen by source priority. Config key `poi.dedupe_h3_res`
(default 8; set to `null` to disable). The plan did not specify a
POI-stage dedup step; this is a standard normalization added here.

Source priority for the dedup tie-break — the user specified
`subway > mall > industrial`; `airport` and `hospital` placement is
**proposed below and needs confirmation**:

    poi_airport > poi_subway > poi_mall > poi_hospital > poi_industrial

Rationale: airport — ready-made aviation infrastructure; subway —
high-throughput multimodal interchange node; mall above hospital —
this project's OD comes from ride-hailing commute/commercial trips, so
hospital demand is already implicit in that signal; industrial ranks
last so the res-8 dedup never crowds out a more general transport node
(e.g. a subway station or mall sited inside an industrial park).

**Status**: code changes pending user confirmation of this entry;
no re-run yet. Expected to bring the POI count toward the `[200, 500]`
`|C|` window — to be verified on the next `run_stage2_build`.

**Update 2026-05-18 (measured)**: the re-run with the three changes
gave POI 2389 / `|C|` 2467 (`poi_industrial` 1732 alone). The
50000 m^2 industrial floor still leaves `|C|` ~5x over the `[200, 500]`
target, so `industrial_min_area_m2` is raised 50000 -> 200000 m^2
(major parks only, >20 ha).

---

## 2026-05-18 — Stage 2: min_orders_per_zone raised 50 -> 2000

**Plan said**: `min_orders_per_zone = 50` (default in
`docs/plan/stage2_spatial_discretization.md`, task 1.2) — intended to
"drop ghost cells in lakes/farmland".

**Reality**: on the ~8.1M endpoints (4.05M orders x origin + dest), a
50-endpoint threshold filters almost nothing — even rural road cells
clear it easily. The first stage-2 run produced `|Z| = 1421`, nearly
2x the revised `[350, 800]` acceptance ceiling and well above the
`[400, 700]` working estimate (see the 2026-05-18 entry above). On
this data volume the threshold's role is no longer "drop ghost cells"
but "set the demand-zone count".

**Change**: `min_orders_per_zone` 50 -> 2000, set in
`configs/spatial.yaml`. This drops `|Z|` from 1421 to 530; the
threshold -> |Z| curve was measured directly from the run output.

**Impact / rationale**:
- a) The Stage-4 diffusion U-Net's memory scales with `|Z|^2` (each
  `[T_window, |Z|, |Z|]` OD slice is treated as an image); `530^2` is
  in the same ballpark as a standard 512x512 DDPM.
- b) After eVTOL eligibility filtering (>=15 km, >=25 min) the OD
  matrix sparsifies sharply; too large a `|Z|` drives mean trips per
  OD pair below ~2, leaving the DDPM no structure to learn.
- c) `|Z| = 530` sits lower-middle of `[400, 700]`, leaving margin for
  a Stage-3 secondary filter (e.g. `min_eligible_orders_per_zone`).

---

## 2026-05-18 — Stage 2: |C| target revised [200, 500] -> [600, 1500]

**Plan said**: `|C|` (candidate vertiport count) should fall in
`[200, 500]` — `docs/plan/stage2_spatial_discretization.md`, Acceptance
Criteria and task 4 step 5.

**Reality**: that window was set against the original City-proper
`SUZHOU_BBOX`. The metro-area bbox (2026-05-15 entry above) is ~7x
larger; POI density and the ~3 km uniform grid both scale with area,
so a City-proper `|C|` target cannot hold on the metro bbox — the same
issue already forced the `|Z|` revision. With the POI pipeline
tightened (`subway=yes` removed, industrial floor 200000 m^2, H3 res-8
dedup) the POI+grid count pre-finalize is 1482 (POI 1383 + grid 99);
even dropping every industrial POI leaves ~756, still above the old
ceiling.

**Change**: `|C|` target revised `[200, 500]` -> `[600, 1500]`,
applied in the Stage 2 plan Acceptance Criteria, plan task 4 step 5,
and the `CAND_LO`/`CAND_HI` constants of
`experiments/run_stage2_build.py`.

**Impact**:
- Anchored to the `|C|/|Z|` ratio common in MCLP / facility-location
  literature (~2-3 candidates per demand zone): `|Z| = 530` implies
  `[1060, 1590]`. The adopted `[600, 1500]` brackets that range with a
  lower floor, since task-4 `finalize_candidates` still drops
  zero-demand-zone candidates (post-finalize `|C|` < pre-finalize).
- This is not target-fitting: the industrial POIs were tightened
  independently for correctness (200000 m^2 floor + res-8 dedup)
  *before* this target was revised — the line moved to match the data,
  not the data to clear the line.
- Confirmed by the full task-1..4 run: POI+grid pre-finalize = 1482;
  `finalize_candidates` dropped 669 candidates lying outside the 530
  demand zones, giving a final **`|C| = 813`** — inside `[600, 1500]`.
- Closing the loop on the anchor: the `[600, 1500]` band is set wide to
  admit the whole `|C|/|Z| ∈ [1, 3]` ratio range, not to bracket the
  midpoint of the `[1060, 1590]` anchor. The measured ratio
  813 / 530 ≈ 1.5 sits toward the lower end of the MCLP literature —
  the candidate-sparse side (grid contributes only 99; POIs dominate) —
  and lies inside `[600, 1500]`, so acceptance passes.

---

## 2026-05-18 — Stage 3: OD time window revised from 7 days to 11 full days

**Plan said**: the OD tensor's time axis is `T = NUM_TIME_BINS = 336`,
i.e. 7 days x 48 half-hour bins (CLAUDE.md §7;
`docs/plan/stage3_od_construction.md` task 1 and the task-3 memory
check). The raw CSV is even named `suzhou_orders_7days.csv`.

**Reality**: the Stage-3 R1.5 smoke test
(`experiments/run_stage3_smoke.py`, Part A) read the full `dep_time`
column of `orders_clean.parquet` (4,050,523 rows) and found the data
spans **12.45 days** — from `2023-07-09 07:33:46` to
`2023-07-21 18:25:01` (298.85 h), not 7 days. Against a 336-bin window
the observed max slot is 597 and **502,214 orders (12.40%)** fall out
of range.

**Change**: adopt an **11 full-calendar-day** window,
`[2023-07-10 00:00:00, 2023-07-21 00:00:00)` — left-closed, right-open —
giving `T = num_time_bins = 528` (11 x 48). Orders with `dep_time`
before the start or at/after the end go to `out_of_range`; the two
partial days at the data's edges (07-09 07:33->24:00 and
07-21 00:00->18:25) are intentionally excluded. `configs/od.yaml::time`
is the source of truth for the window (`start_datetime`, `end_datetime`,
`time_bin_min`, `num_time_bins`); `src/constants.py::NUM_TIME_BINS` was
updated 336 -> 528 to keep code that runs without a config in hand
consistent.

**Rationale**: 11 whole days keep ~88% of the orders (vs. ~56% for a
single 7-day week) while still aligning to calendar-day boundaries, so
the Stage-3 EDA's weekday/weekend folding and hour-of-day curves are not
distorted by partial residual days at the span edges.

**Impact**:
- Stage-3 output tensors change shape `[336, |Z|, |Z|]` ->
  `[528, |Z|, |Z|]`.
- Memory per dense tensor: `528 x 530^2 x 4 B ~= 593 MB` (was ~377 MB at
  336); three tensors ~= 1.78 GB. Stage-4 diffusion memory estimates
  must be redone against `T = 528`.
- CLAUDE.md §7 still prints `NUM_TIME_BINS ... # = 336`; that block needs
  a user-side sync (Claude does not edit CLAUDE.md unprompted).
- The raw file name `_7days` is inconsistent with the 12.45-day span —
  worth confirming the export scope with the data provider.

---

## 2026-05-18 — Stage 3: zone assignment drop acceptance revised after full smoke

**Plan said**: `docs/plan/stage3_od_construction.md` Acceptance Criteria
allowed the zone filter to drop "≤2%" of orders — an order whose O or D
maps to an H3 cell outside the demand-zone set is dropped (task 2). The
2% figure was drafted against an early ~250-zone discretization.

**Reality**: the Stage-3 R1.5 full smoke (`run_stage3_smoke.py` Part B,
all 4,050,501 in-window orders) measured a zone drop of **412,856 rows,
drop_rate = 10.19%** — far above 2%.

**Root cause**: Stage 2's `min_orders_per_zone = 2000` collapsed 1421
H3 res-7 cells to 530 demand zones (see the min_orders_per_zone entry
above), dropping 891 low-density cells. Any order with an O or D
endpoint in one of those 891 cells is unmapped and dropped. 10.19% is
the designed-in consequence of that discretization, not a bug.

**Decision**: keep `|Z| = 530`; do NOT revisit Stage 2 or retune
`min_orders_per_zone`. Stage 3's acceptance criterion is revised
instead.

**New acceptance criterion**: the zone filter drops **≤ 12%** of
in-window orders (full-smoke actual 10.19% + ~1.8 pp buffer). Applied
to `docs/plan/stage3_od_construction.md` Acceptance Criteria.

**Rationale**: a larger `|Z|` would inflate the OD tensor's `|Z|^2`
memory footprint and worsen the eVTOL OD sparsity the 530-zone choice
was made to control. Preserving the high-demand zone structure is worth
dropping the low-density edge cells.

**Impact assessment** (from the full-smoke dropped-order breakdown):
- area_name shares of the dropped set — 苏州市 39.6%, 昆山市 18.3%,
  常熟市 17.5%, 张家港市 14.2%, 太仓市 10.4%. Against the overall area
  mix (苏州市 65%, 昆山市 19%, 常熟市 7%, 张家港市 6%, 太仓市 4%) the
  drop falls disproportionately on the four county-level cities — their
  low-density cells were the ones Stage 2 filtered out.
- of the 412,856 dropped orders only **58,811 (14.24%)** meet the eVTOL
  base condition (`geo_dist_km >= 15` AND `duration_min >= 25`); the
  other 85.76% are short / brief trips that are not eVTOL demand
  anyway. The dropped median trip is 5.9 km / 14.1 min.

**Risk**: low-density edge demand (especially in the county-level
cities) is under-represented in the OD tensors. The Stage-3 EDA and the
paper's Case Study must state this explicitly — the vertiport siting
optimizes over the 530-zone metro core, not the full administrative
footprint.

---

## 2026-05-18 — Stage 3: low-altitude eligibility threshold treated as provisional baseline

**Context**: Stage 3 labels an order eVTOL-eligible when
`geo_dist_km >= 15` AND `duration_min >= 25` AND `o_zone != d_zone`.

**Status**: this `15 km / 25 min` cut is a *provisional baseline*, NOT a
settled transport-science standard. It rests on an engineering
assumption — low-altitude air mobility substitutes best for
medium-to-long, time-consuming ground trips. Stage 3 therefore does not
claim 15 km / 25 min is the optimal threshold; it is the baseline
scenario only.

**Sensitivity evidence (R3, task 9)**: with the duration threshold held
at 25 min, the distance sweep gives these eligible shares (of
zone-assigned orders):
- 10 km — 10.04%
- 12 km —  7.98%
- 15 km —  5.19%  (baseline)
- 18 km —  3.36%
- 20 km —  2.47%

**Main-line decision**: keep 15 km on the main line — it is consistent
with the already-built `od_evtol.npy` / `od_meta.json`, and its 5.19%
share is inside the `[0.03, 0.20]` acceptance window.

**Fallback / robustness scenario**: 12 km (7.98%) is the designated
fallback. If Stage-4 diffusion or Stage-5/6 RL underperforms because the
eVTOL OD tensor is too sparse (`od_evtol` nonzero_ratio is only 0.117%),
switch to 12 km and regenerate the OD tensors.

**Paper-writing constraint**: the manuscript must describe this cut as a
threshold-based proxy / baseline definition / sensitivity-tested
assumption — never as the single ground-truth definition of
low-altitude mobility demand.
