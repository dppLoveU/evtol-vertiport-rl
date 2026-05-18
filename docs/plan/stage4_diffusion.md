# Stage 4: Diffusion Model for OD Generation

## Purpose

Train a conditional diffusion model that learns the distribution of
eVTOL OD slices and generates plausible counterfactual scenarios. The
generated samples will feed Stage 5/6 as a distributional input to make
the RL policy robust to demand uncertainty.

This stage requires the most engineering care: diffusion models are
notoriously sensitive to data normalization and conditioning.

## Stage-4 dimensions (locked by Stage 3)

- `T = 528` time slots (11-day window, 30-min bins) — was planned 336.
- `|Z| = 530` demand zones — was estimated ≈250.
- `od_evtol.npy` is `[528, 530, 530]` int32, 593 MB; nonzero_ratio
  ≈ 0.117% (extremely sparse — the main modelling risk).

## Staged execution

Stage 4 is run in sub-stages so the pipeline is validated before any
expensive training:

- **Stage 4A** — dataset / normalization / padding smoke. Build
  `ODDataset`, verify shapes, norm stats, pad/unpad and inverse-transform
  round-trips on the real tensor. No model, no training. (this round)
- **Stage 4B+** — U-Net, DDPM wrapper, training, sampling, evaluation.

Always start from a **small smoke model** (`base_channels` 16/32, a few
steps) to confirm forward + sample run end-to-end; only then scale up to
a real training run. A full-size U-Net is NOT the default — see
Architecture and Padding below.

## Inputs

- `data/processed/od_evtol.npy` — `[528, 530, 530]` int32 from Stage 3.
- `data/processed/od_meta.json` — needed for time-of-day / day-of-week
  conditioning (`T`, `n_zones`, `start_datetime`, `share_by_hour`).

## Outputs

- `models/diffusion_od/best.pt` — best checkpoint by validation loss.
- `models/diffusion_od/config.yaml` — frozen training config.
- `data/synthetic/od_samples.npy` — `[N_ω, 528, 530, 530]`, generated OD
  scenarios. Default `N_ω = 64`. Stored as int16 (cast back to float at
  load time). **NOTE**: at int16 this 4-D array is
  `64 × 528 × 530² × 2 B ≈ 19 GB` — far too large to hold in memory.
  Stage 5/6 must NOT load it directly; see `od_samples_agg.npy`.
- `data/synthetic/od_samples_agg.npy` — `[N_ω, 530, 530]` int32,
  per-scenario OD aggregated over the time axis (sum over the 528
  slots). ~72 MB. This is the artifact Stage 5/6 should consume: the RL
  coverage reward is bilateral OD coverage and is time-aggregated, so
  the full 4-D tensor is never needed downstream. The sampling script
  (`run_stage4_sample.py`) writes both files; if disk or memory is
  tight, `od_samples.npy` may be skipped and only `_agg` kept.
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

Save `mu`, `sigma` and `clip_val` to `norm_stats_path` (default
`data/processed/od_norm_stats.json`) for inverse transform at sampling
time. JSON, not `.pt` — the stats are three scalars and Stage 4A has no
torch dependency yet. See `docs/decisions.md` 2026-05-18.

### Architecture

U-Net adapted to non-image inputs. Reference: `lucidrains/denoising-
diffusion-pytorch`'s `Unet` class as starting point.

- Input channels: 1 (or `W` if windowed).
- Base channels: **start at 16/32 for the smoke model**; 64 is the
  upper bound for a real run, NOT the default. The padded input is
  544×544 (see Padding) — a `base_channels=64`, `(1,2,4,8)` U-Net on a
  544² input is heavy; profile memory before committing.
- Channel multipliers: smoke `(1, 2, 4)`; real run up to `(1, 2, 4, 8)`.
- Number of down/up blocks: 3 (smoke) or 4 (real).
- Attention at the two deepest resolutions only (helps capture global
  OD structure; full attention at 544² is too expensive).
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

`|Z| = 530` is not a power of 2. A U-Net only needs each spatial dim
divisible by `2^depth` (the number of down-sampling levels), NOT a full
power of 2:

- **Padding to the next power of 2 = 1024** quadruples the spatial area
  vs. 530² (`1024²/530² ≈ 3.7×`) and makes even a small U-Net costly.
  This is the heavy option and is **not** the default.
- **Lightweight default**: pad to the next multiple of `2^depth`. For a
  depth-4 U-Net that is the next multiple of 16 → **544**
  (`544²/530² ≈ 1.05×`, negligible overhead). `pad_size` is computed
  automatically from `pad_multiple` in `configs/diffusion.yaml`
  (`pad_size = ceil(|Z| / pad_multiple) * pad_multiple`).

Zero-pad the raw OD counts at the bottom/right, THEN normalize, so the
padded region carries the same normalized value as a real zero-count OD
pair. Record `pad_size` in config; un-pad (slice `[:530, :530]`) after
generation. Test the round-trip explicitly.

## Tasks

1. **Dataset implementation** (`src/data/od_dataset.py`) — Stage 4A:
   - `ODDataset(od_path, meta_path, split, window=1, norm_stats=None,
     ...)`: returns `(slice, condition)` pairs. `slice` is a normalized,
     padded `[W, pad_size, pad_size]` float32 array; `condition` carries
     `hour`, `day_of_week`, `is_weekend` for the window's first slot.
   - `od_evtol.npy` is read via `np.load(..., mmap_mode="r")` so the
     593 MB tensor is never fully resident.
   - On first build, computes `mu`, `sigma` from the **train** split and
     caches to `norm_stats_path`; val/test load the cached stats.
   - Split over the 11-day window (528 slots, 48/day), contiguous-day so
     val/test slots are held out of training (no temporal leak). Default
     `train_days [0,9]` (slots 0..431), `val_days [9,10]` (432..479),
     `test_days [10,11]` (480..527). Configurable in
     `configs/diffusion.yaml::data.split`.
   - For Stage 4A the dataset is pure NumPy (no torch import); it already
     exposes `__len__`/`__getitem__` so it works as a `torch.utils.data.
     Dataset` by duck typing once torch is added in Stage 4B.

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
   - For each of the `T = 528` time slots, generate `N_ω` samples
     conditioned on that slot's (hour, weekday) signature.
   - Un-pad, inverse-normalize, clip to non-negative, round to int.
   - Stack to `[N_ω, 528, 530, 530]` int16. Save `od_samples.npy`.
   - Also write `od_samples_agg.npy` `[N_ω, 530, 530]` int32 (sum over
     the 528 time slots) — the Stage 5/6 input. See Outputs.

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
- [ ] `od_samples.npy` exists, shape `[64, 528, 530, 530]`, int16; and
  `od_samples_agg.npy` exists, shape `[64, 530, 530]`, int32.
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
- `tests/test_od_dataset.py` — Stage 4A dataset tests.
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
- **Sampling time**: 64 samples × 528 slots × 50 DDIM steps × model
  forward is non-trivial. Expect 45–120 min on a single 3090.
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