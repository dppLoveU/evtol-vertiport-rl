# Stage 5: RL Environment Design

## Purpose

Implement the `VertiportEnv` gymnasium environment that the RL agent
interacts with in Stage 6. The environment encapsulates the sequential
vertiport placement decision problem and computes bilateral OD
coverage as the reward signal.

This is a self-contained engineering task. Get the environment correct
and Stage 6 becomes mostly hyperparameter tuning.

## Inputs

- `data/processed/od_evtol.npy` — real eVTOL OD tensor.
- `data/synthetic/od_samples.npy` — diffusion-generated scenarios.
- `data/processed/cand_covers_zones.npy` — `[|C|, |Z|]` bool coverage.
- `data/processed/dist_zone_cand.npy` — `[|Z|, |C|]` distance, used to
  build static features.
- `data/processed/candidates.geojson` — static candidate features.

## Outputs

This stage produces code only (the environment); artifacts come from
Stage 6 onward.

## MDP Formalization

### State

A flat dict / structured tensor with three parts:

| Part | Shape | Meaning |
|------|-------|---------|
| `mask` | `[|C|]` bool | 1 if candidate already placed |
| `demand_agg` | `[|Z|, F_d]` float32 | per-zone demand statistics (see below) |
| `cand_static` | `[|C|, F_s]` float32 | per-candidate static features |
| `step_idx` | scalar int | k = 0..K |

**Demand aggregation** (`F_d = 4` features per zone):
1. Total outgoing eVTOL trips (sum over t, d) on current scenario.
2. Total incoming eVTOL trips.
3. Mean outgoing across all sampled scenarios (robustness signal).
4. Std outgoing across all sampled scenarios.

**Candidate static features** (`F_s ≈ 8`):
- One-hot of `source` (6 categories from Stage 2).
- Distance to nearest subway (km), continuous.
- A reserved feature for future land cost (zeros for now if no data).

### Action

Discrete, size `|C|`. `action = c` means place a vertiport at candidate
`c`. Invalid actions (already-placed) are filtered via action mask
returned by `env.action_masks()`.

### Reward

At step `k`, after placing candidate `a_k`:

```
prev_covered_zones = bool[|Z|], current covered set before placement
new_covered_zones  = prev_covered_zones OR cand_covers_zones[a_k]
delta_zones        = new_covered_zones AND NOT prev_covered_zones

# Bilateral coverage delta on current scenario M_t (sum over t)
M_total            = sum over t of M_eV[t]    # [|Z|, |Z|], precomputed per scenario
gained = (sum of M_total[o, d] over o,d where:
            (o in new_covered AND d in new_covered)
            AND NOT (o in prev_covered AND d in prev_covered))

# Normalize so the reward roughly lives in [0, 1] per step
gained_norm = gained / M_total.sum()

reward = gained_norm
       - lam_overlap * overlap_penalty(a_k, prev_placed)
       - lam_cost    * land_cost(a_k)
```

`overlap_penalty(a_k, prev_placed)`:
1.0 if `a_k`'s covered zone set is fully contained in already-covered
zones (no marginal gain), else fraction of its covered zones already
covered. Discourages stacking.

`land_cost(a_k)`: zero in v1 (no land-cost data yet); leave the hook.

### Termination

Episode ends when `step_idx == K` (default `K=10`).

### Scenario sampling

On `reset`:
- With probability `p_real` (default 0.3), use real `od_evtol`
  aggregated over `T` as `M_total`.
- Else, sample one scenario index from `[0, N_ω)` and use the
  corresponding diffusion sample (aggregated over `T`).

This implements the "train on diffusion-augmented distribution" core
of the C1 innovation. For Stage 6 ablation A5 (no diffusion), force
`p_real = 1.0`.

For CVaR training (C2), the wrapper outside the env handles the
multi-scenario rollout; the env itself only knows one scenario per
episode. See Stage 6 for the wrapper.

## Tasks

1. **Define `EnvConfig` dataclass** (`src/envs/config.py`):
   K, walk_radius_km, lam_overlap, lam_cost, p_real, time_aggregation
   ("sum" | "mean"), reward_normalization, etc.

2. **Implement `VertiportEnv`** (`src/envs/vertiport_env.py`):
   - Subclass `gymnasium.Env`.
   - `observation_space`: `gym.spaces.Dict`, with the four parts
     above.
   - `action_space`: `gym.spaces.Discrete(|C|)`.
   - `reset(seed, options)`:
     - Sample a scenario per `p_real`.
     - Aggregate the chosen scenario to `M_total = M.sum(axis=0)` (or
       mean per config).
     - Initialize mask to all False, covered to all False, step_idx 0.
     - Return obs.
   - `step(action)`:
     - Assert action is unmasked, else raise.
     - Compute reward as above.
     - Update mask, covered, step_idx.
     - Compute `terminated = (step_idx == K)`.
     - Return obs, reward, terminated, truncated=False, info.
   - `action_masks()`:
     - Return `~self.mask`.
   - `info` dict at terminal step:
     - `coverage`: fraction of OD volume covered.
     - `n_covered_zones`: int.
     - `selected_candidates`: list of `cand_id`.

3. **Implement vectorized environment** (`src/envs/vec_env.py`):
   Thin wrapper over `gymnasium.vector.SyncVectorEnv` or `AsyncVectorEnv`.
   Required because PPO needs `n_envs` parallel rollouts. Default
   `n_envs=8`.

4. **Helpers**:
   - `precompute_coverage_index` — precompute, for each candidate `c`,
     the sparse list of zones it covers (already in
     `cand_covers_zones`; just convert to lists for fast updates).
   - `aggregate_scenario(M, mode)` — sum or mean over time.

5. **Smoke test scripts**:
   - `experiments/smoke_test_env.py` — instantiate env, do 100 random
     episodes, assert no exceptions, log mean reward and coverage.
   - `experiments/smoke_test_vec_env.py` — same with vectorized env.

6. **Unit tests** (`tests/test_env.py`):
   - `test_reset_shapes` — obs has expected shapes.
   - `test_action_mask_invalidates_used` — after placing, mask reflects.
   - `test_reward_monotone_with_coverage` — on a hand-crafted tiny
     case with `|Z|=4, |C|=4`, placing candidate covering more demand
     gives strictly higher reward.
   - `test_terminal_metrics` — at termination, `info["coverage"]`
     equals the analytical bilateral coverage computed independently.
   - `test_no_double_place` — stepping with masked action raises.

## Acceptance Criteria

- [ ] All 5 unit tests pass.
- [ ] Smoke test runs 100 episodes without exception.
- [ ] Mean random-policy coverage after K=10 is reproducible (fixed
  seed → exact same number across runs).
- [ ] `obs_space.contains(obs)` returns True for all returned obs.
- [ ] Memory per env instance < 200 MB even with `N_ω = 64` scenarios
  loaded (scenarios should be lazily loaded or mmap'd if needed).
- [ ] One full episode (K=10) executes in < 50 ms on CPU.

## Files to Create

- `src/envs/__init__.py`
- `src/envs/config.py`
- `src/envs/vertiport_env.py`
- `src/envs/vec_env.py`
- `src/envs/coverage.py` — pure-function bilateral coverage utilities,
  importable in evaluation too.
- `configs/env.yaml`
- `experiments/smoke_test_env.py`
- `experiments/smoke_test_vec_env.py`
- `tests/test_env.py`

## Common Pitfalls

- **Coverage delta math**: it is tempting to write
  `gained = (covered AND NOT prev_covered).sum() over OD`, but
  bilateral coverage is the cartesian product of zones, not a single
  axis. The correct delta is:
  ```
  M_old = M_total[prev_covered][:, prev_covered].sum()
  M_new = M_total[new_covered][:, new_covered].sum()
  gained = M_new - M_old
  ```
- **Scenario aggregation choice**: summing over T gives total demand
  served over the week; this is what the paper claims to maximize.
  Mean gives per-slot. Stick with sum and document.
- **Determinism with parallel envs**: each sub-env needs its own seed,
  derived from a base seed. `gym.utils.seeding.np_random` is the right
  way; don't share `np.random.default_rng()`.
- **Action mask plumbing**: MaskablePPO from sb3-contrib expects the
  env to have an `action_masks()` method. Verify the wrapper picks it
  up; otherwise actions can be invalid.
- **Memory for diffusion samples**: 64 × 336 × 250 × 250 × 2 bytes ≈
  2.7 GB if naively loaded as int16. Either pre-aggregate to
  `[N_ω, |Z|, |Z|]` and store separately (much smaller, ~16 MB), or
  mmap. Decision: pre-aggregate at Stage 4 end, save as
  `od_samples_agg.npy`. Update Stage 4 task list if needed.

## Dependencies

- Stage 2 (coverage matrix, candidates).
- Stage 3 (od_evtol).
- Stage 4 (od_samples, ideally pre-aggregated).
- Adds: `gymnasium>=0.29`, `sb3-contrib`.

## Estimated effort

3 days. The MDP math (bilateral coverage delta) needs unit tests on
small hand-crafted cases to catch bugs early.