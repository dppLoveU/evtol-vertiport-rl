# Stage 4B-5C PR5C-1B posthoc calibration report

- config: `configs/diffusion_12km_zpin_weighted.yaml`
- ckpt: `models/diffusion_od_pilot_zpin_weighted/best.pt`  step=1000  val_loss=0.0601
- profile: pilot  num_inference_steps=50  guidance_scale=1.0
- n_samples: 48  seed: 42

## Sampling summary
- shape: [48, 530, 530]  dtype: float64
- min: 0.0000  max: 45.2318  mean: 0.5898
- sample wall time: 26.6 s

## Calibration fit (train aggregate target only)
- best_tau: **1.0000**
- best_scale: **2**
- train objective at argmin: 0.046187
- test untouched for fitting: **True**
- objective: `|nz_ratio/real_nz - 1| + lambda_mass * |total_mass/real_total - 1|` (lambda_mass = 1.0)

## Before vs after by split

### before (clip → round on continuous samples; tau=0, scale=1)

| split | nz_x_real | mass_ratio | row_ks | col_ks | top20 | top20vs50 | gen_max | gen_mean |
|---|---|---|---|---|---|---|---|---|
| train | 2.0671 | 0.7269 | 0.4954 | 0.4796 | 0.0000 | 0.0000 | 42.7500 | 0.5497 |
| val | 5.5464 | 3.8600 | 0.9608 | 0.9540 | 0.0000 | 0.0000 | 42.7500 | 0.5497 |
| test | 5.8314 | 4.0944 | 0.9611 | 0.9658 | 0.0000 | 0.0000 | 42.7500 | 0.5497 |

### after (apply_threshold_and_scale(samples_cont, best_tau, best_scale))

| split | nz_x_real | mass_ratio | row_ks | col_ks | top20 | top20vs50 | gen_max | gen_mean |
|---|---|---|---|---|---|---|---|---|
| train | 1.0028 | 0.9567 | 0.5444 | 0.5467 | 0.0000 | 0.0000 | 85.5833 | 0.7233 |
| val | 2.6908 | 5.0797 | 0.9699 | 0.9686 | 0.0000 | 0.0000 | 85.5833 | 0.7233 |
| test | 2.8291 | 5.3881 | 0.9711 | 0.9695 | 0.0000 | 0.0000 | 85.5833 | 0.7233 |

## Acceptance verdict on TEST (after calibration)
- **MILD**

PASS gates:
- nz_ratio_x_real in [0.7, 1.5]: 2.8291
- total_mass_ratio in [0.8, 1.2]: 5.3881
- top20_pair_overlap >= 12: 0.0000
- row_sum_ks_stat <= 0.3: 0.9711
- col_sum_ks_stat <= 0.3: 0.9695

MILD floor (must strictly beat failed-diffusion baseline on every axis):
- row_sum_ks_stat < 1.0
- col_sum_ks_stat < 1.0
- nonzero_ratio_x_real < 143.0
- total_mass_ratio < 181.0

## Safety
- frozen `data/synthetic/od_samples_agg.npy` written: False
- diffusion 4-D `data/synthetic/od_samples.npy` written: False
- candidate `data/synthetic/od_samples_agg_diffusion_calibrated.npy` written: False
- bootstrap candidate `data/synthetic/od_samples_agg_bootstrap.npy` modified by this run: False (script never opens it)