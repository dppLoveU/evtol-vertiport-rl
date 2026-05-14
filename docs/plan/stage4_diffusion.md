# Stage 4: Diffusion Model for OD Generation

## Purpose

Train a conditional diffusion model that learns the distribution of
eVTOL OD slices and generates plausible counterfactual scenarios. The
generated samples will feed Stage 5/6 as a distributional input to make
the RL policy robust to demand uncertainty.

This stage requires the most engineering care: diffusion models are
notoriously sensitive to data normalization and conditioning.

## Inputs

- `data/processed/od_evtol.npy` — `[T, |Z|, |Z|]` int32 from Stage 3.
- `data/processed/od_meta.json` — needed for time-of-day / day-of-week
  conditioning.

## Outputs

- `models/diffusion_od/best.pt` — best checkpoint by validation loss.
- `models/diffusion_od/config.yaml` — frozen training config.
- `data/synthetic/od_samples.npy` — `[N_ω, T, |Z|, |Z|]`, generated OD
  scenarios. Default `N_ω = 64`. Stored as int16 (cast back to float at
  load time).
- `results/stage4/eval/` — evaluation artifacts:
  - `marginal_match.png` — row-sum and column-sum distributions: real
    vs generated overlay.
  - `top_pairs_match.png` — bar chart, top-20 real OD pairs vs same
    pairs in generated samples (averaged).
  - `temporal_pattern_match.png` — hourly volume curves: real vs
    generated mean ± 1 std band.
  - `mmd_jsd.json` — Maximum Mean Discrepancy and Jensen-Shannon
    divergence between real and generated OD distributions, computed
    on the row-sum and column-sum marginals.
  - `samples_grid.png` — heatmap grid of 16 random generated OD slices
    next to 16 random real slices, for qualitative inspection.

## Model Design

### Representation

Treat the OD tensor as a stack of "images":
- Each training sample is one time slot `t`, giving an
  `[|Z|, |Z|]` single-channel image.
- Optionally extend to "windows" of `W` consecutive slots, giving a
  `[W, |Z|, |Z|]` multi-channel image. Default `W = 1`; ablate `W = 4`
  if time permits.

### Normalization

OD counts are heavily long-tailed (most zeros, a few thousands). Apply:
1. `log1p(x)` — compress the tail.
2. Compute mean and std over all training slots (NOT per-pixel).
3. Standardize: `(log1p(x) - mu) / sigma`.
4. Final clip to `[-3, 3]` and scale to `[-1, 1]` for DDPM input.

Save `mu` and `sigma` in `models/diffusion_od/norm_stats.pt` for
inverse transform at sampling time.

### Architecture

U-Net adapted to non-image inputs. Reference: `lucidrains/denoising-
diffusion-pytorch`'s `Unet` class as starting point.

- Input channels: 1 (or `W` if windowed).
- Base channels: 64.
- Channel multipliers: `(1, 2, 4, 8)` — i.e. up to 512 channels at
  deepest layer.
- Number of down/up blocks: 4.
- Attention at resolution `|Z|/4` and below (helps capture global OD
  structure).
- Time embedding: sinusoidal + 2-layer MLP, embedding dim 128.
- **Condition embeddings** (concatenated to time embedding):
  - `hour_of_day` (0..23) → sin/cos features → 2-d.
  - `day_of_week` (0..6) → one-hot or embedding → 7-d.
  - `is_weekend` (bool) → 1-d.
- Use classifier-free guidance: with probability 0.1 during training,
  drop the condition embedding to all-zero (unconditional pass).

### Diffusion Schedule

- `num_train_timesteps = 1000`, cosine beta schedule.
- `num_inference_steps = 50` (DDIM sampler).
- Loss: predict `epsilon`, MSE.
- Guidance scale: 1.5–3.0 at sampling time (tune by visual inspection
  of generated samples).

### Padding

`|Z|` (≈250) is unlikely to be a power of 2. Zero-pad along both spatial
dims to the next power of 2 (e.g., 256) for U-Net friendliness. Record
the pad size in config; un-pad after generation.

## Tasks

1. **Dataset implementation** (`src/data/od_dataset.py`):
   - `ODDataset(od_path, meta_path, norm_stats=None, window=1)`:
     returns `(slice, condition)` pairs.
   - On first build, computes `mu`, `sigma` from training split and
     caches.
   - Split: first 5 days train, day 6 val, day 7 test by `t0` offset.
     Hold the test slots out of training so we don't leak.

2. **U-Net model** (`src/models/unet_od.py`):
   - Adapt the lucidrains U-Net or implement a slimmer version.
   - Conditioning injected via `time_emb + cond_emb` added before each
     residual block.

3. **DDPM wrapper** (`src/models/diffusion.py`):
   - Forward (training): sample `t`, add noise, predict epsilon, MSE.
   - Reverse (sampling): DDIM sampler with classifier-free guidance.

4. **Training loop** (`experiments/run_stage4_train.py`):
   - Hydra config from `configs/diffusion.yaml`.
   - Mixed precision (`torch.amp`).
   - EMA weights with decay 0.9999, eval/save with EMA copy.
   - Optimizer: AdamW, lr 2e-4, weight decay 0.0, betas (0.9, 0.999).
   - Gradient clipping at 1.0.
   - Total steps: 100k–200k depending on convergence.
   - Eval every 5k steps: generate 32 samples, compute MMD on row sums,
     log to WandB.
   - Save `best.pt` (lowest val MMD) and `last.pt`.

5. **Sampling script** (`experiments/run_stage4_sample.py`):
   - Load best checkpoint.
   - For each of the `T = 336` time slots, generate `N_ω` samples
     conditioned on that slot's (hour, weekday) signature.
   - Un-pad, inverse-normalize, clip to non-negative, round to int.
   - Stack to `[N_ω, T, |Z|, |Z|]` int16. Save.

6. **Evaluation script** (`experiments/run_stage4_eval.py`):
   - Compute marginal-match plots, MMD, JSD.
   - Visual grid of generated vs real slices.
   - Write `mmd_jsd.json`.

7. **Sanity baseline**: also implement a trivial baseline —
   "resample a real slice with same (hour, weekday) and add Gaussian
   noise". Generate `N_ω` such samples and compare metrics against the
   diffusion samples. The diffusion model must beat this on MMD,
   otherwise it's not learning anything useful. Add to `mmd_jsd.json`.

## Acceptance Criteria

- [ ] Training loss curve is smooth and decreasing (saved to WandB).
- [ ] Best model's row-sum-MMD against real test set < 0.1 (rough
  threshold; refine after first run).
- [ ] Diffusion beats the noise-baseline on MMD by ≥ 30%.
- [ ] Generated samples are non-negative integers after inverse
  transform. No NaN.
- [ ] Generated samples preserve the temporal pattern: hourly mean
  volume of generated samples correlates with real (Pearson r > 0.9).
- [ ] Top-20 hottest real OD pairs appear in the top-50 of the
  generated mean (overlap ≥ 12 of 20).
- [ ] `od_samples.npy` exists, shape `[64, 336, |Z|, |Z|]`, int16.
- [ ] `tests/test_diffusion.py` passes (smoke tests on a tiny dummy
  dataset, verify forward+sample don't crash).

## Files to Create

- `src/data/od_dataset.py`
- `src/models/unet_od.py`
- `src/models/diffusion.py`
- `src/models/ema.py` — EMA helper.
- `src/utils/metrics_dist.py` — MMD, JSD, marginal matchers.
- `configs/diffusion.yaml`
- `experiments/run_stage4_train.py`
- `experiments/run_stage4_sample.py`
- `experiments/run_stage4_eval.py`
- `tests/test_diffusion.py`

## Common Pitfalls

- **NaN losses early**: usually a normalization bug. Verify the
  pre-normalized samples are in roughly `[-1, 1]` by printing min/max
  on the first batch.
- **Mode collapse to zeros**: if guidance scale is too high or training
  is too short, generated OD becomes all zeros (the dominant mode).
  Lower guidance, train longer, or weight loss by `(1 + log1p(x))` to
  emphasize non-zero entries.
- **Spatial structure missing**: if the U-Net is too small, generated
  OD has no spatial coherence. Watch the `samples_grid.png` regularly.
- **`|Z|` not pow2**: forgetting to un-pad after generation gives extra
  rows/cols of garbage. Test the round-trip explicitly.
- **Conditioning leak**: if conditioning is wrong (e.g., always passing
  hour=0), generated samples will all look like 0am traffic. Print
  a few (hour, weekday) conditioning vectors on first batch.
- **Sampling time**: 64 samples × 336 slots × 50 DDIM steps × model
  forward is non-trivial. Expect 30–90 min on a single 3090.
  Batch over slots within the same (hour, weekday) bucket.
- **EMA**: forgetting to switch to EMA weights for eval gives much
  worse samples than reported in papers. Always eval and sample with
  EMA.
- **Determinism**: DDIM with fixed seed should be deterministic. Useful
  for reproducing exact samples in the paper.

## Robustness Note

If diffusion fails to converge after honest effort, the fallback is:
"diffusion-as-data-augmentation" replaced by "bootstrap resampling
from real OD slices with conditional matching". The RL pipeline still
runs with that fallback; only the C1 innovation (in CLAUDE.md §8) is
downgraded. This fallback is documented in `docs/decisions.md` IF
invoked.

## Dependencies

- Stage 3 (`od_evtol.npy`, `od_meta.json`).
- Adds to `pyproject.toml`: `denoising-diffusion-pytorch` (or roll
  our own), `einops`, `accelerate` (optional).

## Estimated effort

7 days. Most likely bottleneck: getting diffusion to actually generate
useful samples (vs blank or noise). Budget 2–3 days for debugging.