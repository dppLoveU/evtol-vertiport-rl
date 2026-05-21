# Stage 4B-5C PR5C-3A unified scenario comparison

## Recommended winner
- **bootstrap_day_block** (bootstrap, tier=MILD)
- candidate path: `data/synthetic/od_samples_agg_bootstrap.npy`
- can_freeze_to_stage5: True

**This recommendation is advisory.** The final copy of the winning candidate to `data/synthetic/od_samples_agg.npy` is a separate sub-PR (PR5C-3B) gated on explicit user confirmation. This PR (PR5C-3A) does NOT freeze any source: no `od_samples_agg.npy` was written or modified, no candidate npy was overwritten, no Stage-5 code was touched.

## Why this winner

Bootstrap beats every other candidate on the decisive axes:
- `top20_pair_overlap`: **11.89** vs diffusion-calibrated 0.00 (spatial structure preserved vs collapsed)
- `row_sum_ks_stat` (per-day-equivalent): **0.093** vs calibrated 0.971 (raw scale)
- `col_sum_ks_stat` (per-day-equivalent): **0.078** vs calibrated 0.969
- `total_mass_ratio` (per-day-equivalent): **1.126** vs calibrated 1.078
- `nonzero_ratio_x_real_test`: **2.677** vs calibrated 2.829

Bootstrap also already has a usable candidate npy on disk (`data/synthetic/od_samples_agg_bootstrap.npy`, 71.9 MB, shape (64, 530, 530), int32, nonnegative, used_slots ⊆ train_slots verified). The diffusion-calibrated path does NOT — PR5C-1B deliberately did not write one because the checkpoint produces samples with zero rank correlation against real top OD pairs (`top20_pair_overlap = 0`), which no posthoc thresholding can fix.

## All sources (sorted by decision rule)

| rank | source | tier | freezeable | top20 | row_ks | col_ks | mass_ratio | nz_x_real |
|---|---|---|---|---|---|---|---|---|
| 1 | bootstrap_day_block | MILD | True | 11.89 | 0.093 | 0.078 | 1.126 | 2.677 |
| 2 | diffusion_raw_zpin_weighted_pilot | MILD | False | 0.00 | 0.961 | 0.966 | 0.819 | 5.831 |
| 3 | diffusion_calibrated_zpin_weighted_pilot | MILD | False | 0.00 | 0.971 | 0.969 | 1.078 | 2.829 |
| 4 | diffusion_failed_baseline_pr5b_3b3 | FAIL | False | 0.00 | 1.000 | 1.000 | 181.000 | 143.000 |

## Important scale caveat

The `row_sum_ks_stat` / `col_sum_ks_stat` / `total_mass_ratio` columns are not at identical scales across rows. The bootstrap row uses per-day-equivalent KS (bootstrap aggregates 11 days, real_test is 1 day, so the comparison divides bootstrap by `n_days_per_scenario`). The diffusion-calibrated row reports raw KS at the train-aggregate scale; its `total_mass_ratio` is rescaled to per-day-equivalent by dividing the raw `total_mass_ratio_mean` (≈ 5) by `n_train_days / n_test_days` = 5. See each row's `*_scale` column for the convention used. The headline conclusion (bootstrap >> diffusion-calibrated on structure) holds under any reasonable rescaling because the gap is at least an order of magnitude on row_ks / col_ks and is qualitative on top20.

## Decision rule (lowest tuple = best, top of sort)

1. `can_freeze_to_stage5` (True > False)
2. `acceptance_tier` (PASS > MILD > FAIL)
3. `top20_pair_overlap` (higher better)
4. `row_sum_ks_stat` (lower better)
5. `col_sum_ks_stat` (lower better)
6. `|total_mass_ratio - 1|` (closer to 1 better)

## Next steps (NOT in this PR)

- PR5C-3B: with user confirmation, copy `data/synthetic/od_samples_agg_bootstrap.npy` to `data/synthetic/od_samples_agg.npy` and record the freeze in `docs/decisions.md`. Until then `data/synthetic/od_samples_agg.npy` remains absent.
- Optional PR5C-2B-ext: bootstrap parameter sensitivity sweep (`n_days_per_scenario`, seed variance) if PR5C-3B wants to log a robustness band before freezing.
- Stage 5 (`docs/plan/stage5_rl_env.md`) remains gated on the freeze.

## Safety verification

- `data/synthetic/od_samples_agg.npy` exists: False
- `data/synthetic/od_samples.npy` exists: False
- `data/synthetic/od_samples_agg_diffusion_calibrated.npy` exists: False
- `data/synthetic/od_samples_agg_bootstrap.npy` exists (unmodified by this script): True