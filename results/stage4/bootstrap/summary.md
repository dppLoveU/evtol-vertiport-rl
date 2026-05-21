# Stage 4B-5C PR5C-2B bootstrap report

- config: `configs/diffusion_12km.yaml`
- output npy: `data/synthetic/od_samples_agg_bootstrap.npy`
- shape: [64, 530, 530]  dtype: int32  nonneg: True
- n_omega: 64  n_days_per_scenario: 11  n_test_days: 1

## Leak check
- train_slots: 240
- used_slots:  240
- used_slots ⊆ train_slots: True
- leaked into val: 0
- leaked into test: 0
- any leak: False

## Acceptance verdict
- **MILD**

| gate | value | band |
|---|---|---|
| gen_nonzero_ratio (x real_test) | 2.677 | pass: [0.7, 1.5] |
| per_day_total_mass_ratio | 1.126 | pass: [0.7, 1.5] |
| per_day row_sum_ks (mean) | 0.093 | pass: ≤ 0.3 |
| per_day col_sum_ks (mean) | 0.078 | pass: ≤ 0.3 |
| top20 overlap (mean) | 11.89 | pass: ≥ 14 |
| top20 vs top50 overlap (mean) | 19.83 | (diagnostic) |

## Headline numbers
- gen_nonzero_ratio (mean): 0.173980
- real_test_nonzero_ratio: 0.064991
- per_day_total_mass (mean): 42447.830
- real_per_day_test_total_mass: 37710.000
- raw total_mass_ratio (mean): 12.382 (raw 11-day aggregate vs 1-day test reference)
- gen_max (mean / max): 1152.6 / 1257  real_test_max: 121
- entropy mean: 9.852  real_test_entropy: 9.313

## Failed-diffusion baseline (PR5B-3b-3, for reference)
- diffusion row_sum_ks_stat: 1.000
- diffusion col_sum_ks_stat: 1.000
- diffusion gen_nonzero_ratio_x_real: 143.0
- diffusion total_mass_ratio_x_real_per_day: 181.0

## Sampler summary
- n_omega: 64
- z: 530
- total_sum_mean: 466926.125
- total_sum_std: 5477.39982365036
- nonzero_ratio_mean: 0.17397961908152365
- nonzero_ratio_std: 0.010418831723691584
- global_min: 0
- global_max: 1257

## Notes on scale
- Bootstrap scenarios aggregate `n_days_per_scenario` calendar days; `real_test` covers `n_test_days` calendar days. KS, row/col sums, and total_mass are compared at per-day-equivalent scale (bootstrap / n_days_per_scenario vs real_test / n_test_days). nonzero_ratio is NOT scale-invariant under aggregation -- the bootstrap nonzero_ratio is naturally higher because more events accumulate across days; the `x real_test` ratio is reported for transparency.