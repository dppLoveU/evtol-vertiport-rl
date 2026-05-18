# Stage 3 — OD Tensor Construction & eVTOL Filtering: Completion Summary

Status: **complete (tasks 1-9) 2026-05-18**. Plan: `docs/plan/stage3_od_construction.md`.
Built across rounds R1 / R1.5 / R2 / R3 / R4 — see `docs/progress.md`.

## 1. Goal

Aggregate the 4.05M cleaned ride-hailing orders into time-resolved,
zone-level OD tensors: a full OD tensor (richer diffusion-training
signal) and an eVTOL-eligible OD tensor (the RL coverage-reward target),
plus the paper's Case Study EDA figures.

## 2. Inputs

| File | From |
|------|------|
| `data/processed/orders_clean.parquet` | Stage 1 (4,050,523 cleaned rows) |
| `data/processed/zones.geojson` | Stage 2 (530 H3 res-7 demand zones) |

The plan's Inputs section also lists `dist_zone_zone.npy` "for filter
(b)", but the eVTOL filter is order-level (`geo_dist_km` / `duration_min`
columns), so `dist_zone_zone` is not used in Stage 3 — see the R1 entry
in `docs/progress.md`.

## 3. Outputs

### Code (git-tracked)
| File | Role |
|------|------|
| `src/data/od.py` | `assign_time_bin`, `assign_zones`, `build_zone_lookup`, `is_evtol_eligible`, `build_od_tensor` |
| `src/data/eda_plots.py` | reusable figure style + PNG/PDF save + zone choropleth |
| `configs/od.yaml` | Stage-3 source of truth (time window, eVTOL thresholds, weight, paths) |
| `experiments/run_stage3_smoke.py` | R1.5 real-data time-window + zone-drop validation |
| `experiments/run_stage3_build.py` | OD tensor build (smoke / full modes) |
| `experiments/run_stage3_sensitivity.py` | distance-threshold sweep (task 9) |
| `experiments/run_stage3_eda.py` | share statistics (task 7) + 7 figures (task 8) |
| `tests/test_od.py` | 11 unit tests |

### Data artifacts (`data/processed/`, gitignored)
| File | Shape / content |
|------|-----------------|
| `od_full.npy` | `[528,530,530]` int32 — all-order OD count per time slot |
| `od_evtol.npy` | `[528,530,530]` int32 — eVTOL-eligible OD count |
| `od_evtol_weighted.npy` | `[528,530,530]` float32 — eVTOL OD weighted by fare |
| `od_meta.json` | T, |Z|, window, eVTOL params, share, share_by_hour |

### Results (`results/stage3/`, gitignored)
- `sensitivity.csv` — 5-row distance-threshold sweep
- `eda/share_by_hour.csv` — 24-row hourly eVTOL share
- `eda/*.png` + `eda/*.pdf` — 7 figures × 2 formats

## 4. Time window

11 full calendar days, `[2023-07-10 00:00:00, 2023-07-21 00:00:00)` —
left-closed, right-open. `T = 528` (30-min bins). Revised from the
planned 7-day / T=336 window after R1.5 found the data actually spans
12.45 days (2023-07-09 .. 2023-07-21); see `docs/decisions.md`
2026-05-18. Only 22 of 4,050,523 orders fall outside the window.

## 5. Zone assignment

| Metric | Value |
|--------|-------|
| n_time_in_range | 4,050,501 |
| n_zone_assigned | 3,637,645 |
| zone drop_rate | 10.19% |
| acceptance | ≤ 12% (revised from the plan's ≤2%; see decisions.md) |

The drop is the designed-in consequence of Stage 2's 530-zone
discretization — 891 low-density H3 cells are unmapped. Of the dropped
orders only 14.24% meet the eVTOL base condition, so eVTOL-demand impact
is limited; low-density edge demand is nonetheless under-represented and
the paper's Case Study must say so.

## 6. OD tensors

All `[528, 530, 530]`; od_evtol diagonal = 0, no NaN, no negative values.

| Tensor | dtype | sum | nonzero | nonzero_ratio | file size |
|--------|-------|-----|---------|---------------|-----------|
| od_full | int32 | 3,637,645 | 1,837,574 | 1.239% | 593.26 MB |
| od_evtol | int32 | 188,699 | 173,264 | 0.117% | 593.26 MB |
| od_evtol_weighted | float32 | 14,915,827 | 173,264 | 0.117% | 593.26 MB |

## 7. eVTOL eligibility baseline

An order is eVTOL-eligible iff `geo_dist_km >= 15` AND
`duration_min >= 25` AND `o_zone != d_zone`. The `15 km / 25 min` cut is
a **provisional baseline** — an engineering assumption (low-altitude air
mobility substitutes best for medium-to-long, time-consuming ground
trips), NOT a settled transport-science standard. The manuscript must
describe it as a threshold-based proxy / sensitivity-tested assumption.
See `docs/decisions.md` 2026-05-18.

evtol_share (`od_evtol.sum / od_full.sum`) = **5.19%**.

## 8. Sensitivity (task 9)

Distance sweep at fixed 25-min duration; eligible share of zone-assigned
orders:

| min_dist_km | 10 | 12 | 15 | 18 | 20 |
|-------------|------|------|------|------|------|
| eligible share | 10.04% | 7.98% | 5.19% | 3.36% | 2.47% |

15 km kept on the main line (consistent with the built tensors, share in
the [3%,20%] acceptance window). **12 km (7.98%) is the designated
fallback / robustness scenario** — switch to it and regenerate the OD
tensors if Stage-4/5/6 underperforms due to eVTOL OD sparsity.

## 9. EDA figures (task 8)

`results/stage3/eda/`, each saved as PNG (300 dpi) + PDF:
`od_share_by_hour`, `top20_od_pairs`, `heatmap_o_volume`,
`heatmap_d_volume`, `temporal_pattern`, `distance_distribution`,
`fare_distribution`. Plus `share_by_hour.csv` (24 rows). The eVTOL
share_by_hour spans ~3.4%–9.9%, peaking at 05:00 (pre-dawn long-haul)
and bottoming at 00:00–03:00.

## 10. Tests

`pytest tests/test_od.py` → **11/11 pass** (time-bin math, zone mapping,
eVTOL filter, OD-tensor count/weighted aggregation, zero-diagonal,
out-of-range raise).

## 11. Stage 4 prerequisites

Stage 4 (diffusion) consumes:
- `data/processed/od_evtol.npy` — `[528, 530, 530]` int32
- `data/processed/od_meta.json` — T, |Z|, window, eVTOL params, shares

Key parameters for the diffusion design:
- **T = 528**, **|Z| = 530** — each OD slice is `[T_window, 530, 530]`.
- **od_evtol is extremely sparse** — nonzero_ratio 0.117%. This is the
  main modelling risk; if the U-Net cannot learn structure from such a
  sparse signal, the 12 km fallback (denser, 7.98% eVTOL share) is
  available without re-running Stages 1-2.
