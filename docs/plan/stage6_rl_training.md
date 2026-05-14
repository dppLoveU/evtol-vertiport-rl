# Stage 6: RL Training (Baseline → Full Method)

## Purpose

Train the MaskablePPO policy on `VertiportEnv`. Run the full ablation
ladder needed for the paper, in order of increasing complexity:

- **A5**: PPO on real demand only.
- **A6**: PPO with diffusion-augmented demand (expectation objective).
- **A7**: PPO with diffusion-augmented demand and CVaR objective.

This is also where SpoNet (A4) is adapted for fair comparison.

## Inputs

- All of Stage 5's environment.
- `data/synthetic/od_samples_agg.npy` from Stage 4.

## Outputs

For each ablation run:

- `results/stage6/{run_name}/ckpt/best.pt` — best policy by eval reward.
- `results/stage6/{run_name}/ckpt/last.pt` — final policy.
- `results/stage6/{run_name}/eval/` — eval logs over training.
- `results/stage6/{run_name}/config.yaml` — frozen config.
- `results/stage6/{run_name}/selected.json` — for the best policy, the
  selected `cand_id` list across seeds.
- WandB run linked from each output folder.

Across runs:

- `results/stage6/ablation_summary.csv` — coverage and time for every
  run × seed × K.

## Policy Network

Implement a custom features extractor for MaskablePPO.

### Architecture

Inputs (from env's `Dict` obs):
- `mask` `[|C|]` bool → embed as a 0/1 feature.
- `demand_agg` `[|Z|, 4]` → flattened or passed through a small MLP.
- `cand_static` `[|C|, F_s]` → per-token static features.

Build per-candidate token features:
1. For each `c`, concat:
   - `cand_static[c]` (`F_s`)
   - `mask[c]` (1)
   - Aggregated demand at zones that `c` covers: mean and sum of
     `demand_agg` over `Γ(c)` (8 features).
   - Step-progress feature `(k / K)` (1).
   Total per-token dim ≈ 18.
2. Pass through a linear projection to `d_model = 128`.
3. Self-attention over the `|C|` tokens, 3 layers, 4 heads, FFN dim
   256, GELU.
4. Policy head: linear `d_model → 1`, gives logits over candidates.
5. Value head: mean-pool tokens → MLP → scalar.

Why per-token attention rather than CNN on zone-grid: the candidates
are an unordered set with coverage relations; attention is more natural
and matches `SpoNet`/`DeepMCLP`'s design, which makes the paper's
comparison fair.

### Action Masking

Logits from the policy head are masked by `env.action_masks()` before
softmax. MaskablePPO handles this if the env exposes `action_masks()`.

## CVaR Training (A7)

For A7, wrap rollout collection so each "trajectory" is actually `M`
parallel trajectories on different sampled scenarios, then take the
worst-α fraction as the trajectory for advantage computation.

Implementation:
1. Custom `RolloutBuffer` subclass that, on each `collect_rollouts`
   step, runs `M` parallel env instances differing only in scenario
   index.
2. For each step, compute reward across the `M` envs, sort, take
   bottom `α * M` (default `α = 0.3`).
3. Use the mean of the bottom α as the reward signal for PPO updates
   (i.e. CVaR_α optimization).

Reference: "Risk-Sensitive Reinforcement Learning" literature (Tamar
et al. 2015, Chow et al. 2017). Cite in the paper.

Simpler alternative if the custom buffer is too invasive: just
sample-and-evaluate. At each training episode, sample `M=8` scenarios,
play the same actions, compute CVaR over rewards offline, then use a
single env that returns a "post-hoc CVaR-weighted reward". Pragmatic.

**Default**: start with the pragmatic version. Only switch to the
custom buffer if A7 shows no benefit over A6.

## Baselines for Paper Comparison

In addition to A5/A6/A7, also implement (in `src/baselines/`):

- **A0: Random** — uniformly sample K candidates without replacement.
- **A1: K-means + greedy** — cluster eVTOL origin points into K
  centers, snap each center to the nearest candidate, no overlap.
- **A2: Genetic algorithm** — use `deap` or `pymoo`. Population 100,
  generations 200, tournament selection.
- **A3: CPLEX (Gurobi)** — exact MILP for the bilateral MCLP. Only run
  for small K (e.g. K=5) because of NP-hardness; document timeout for
  larger K.
- **A4: SpoNet** — clone the repo to `external/SpoNet/`, write a
  thin adapter in `src/baselines/sponet_adapter.py` that:
  1. Exports our zones and candidates to SpoNet's expected format.
  2. Calls their `eval.py` to get a placement.
  3. Reads back placement, evaluates with our coverage function.
  This is the most important comparison: an MCLP-DRL method without
  diffusion or CVaR.

## Tasks

1. **Policy network** (`src/agents/policy.py`):
   `CandidateTokenExtractor` + custom policy class compatible with
   sb3-contrib MaskablePPO.

2. **Training entrypoint** (`experiments/run_stage6_train.py`):
   - Hydra config from `configs/ppo_vertiport.yaml`.
   - Builds vec env, policy, callbacks.
   - Trains for `total_steps` (default 2M).
   - Saves best + last.

3. **Eval callback** (`src/agents/callbacks.py`):
   - Every `eval_every` steps, evaluate on a fixed eval env using real
     demand only (deterministic, not sampled).
   - Log `coverage`, `n_covered_zones`, `mean_reward`.

4. **Run A5**: real demand only.
   ```
   p_real: 1.0
   K: 10
   seeds: [42, 43, 44, 45, 46]
   total_steps: 2_000_000
   ```

5. **Run A6**: diffusion-augmented, expectation objective.
   Same config but `p_real: 0.3`. Other 70% of episodes sample from
   diffusion.

6. **Run A7**: CVaR objective with α=0.3, M=8. See above.

7. **Implement and run A0–A4** baselines.
   Each baseline outputs a JSON of selected `cand_id` list, and is
   evaluated against the same real test demand. Single script:
   `experiments/run_stage6_baselines.py`.

8. **Sensitivity sweep**: rerun A7 (or the best ablation) over
   `K ∈ {5, 7, 10, 13, 15, 20}` and `R ∈ {3, 4, 5, 6, 8}` (walk radius
   in km). One seed each (or 3 if time permits). Output:
   `results/stage6/sensitivity_K.csv`, `sensitivity_R.csv`.

9. **Compile ablation summary**:
   `experiments/run_stage6_summary.py` reads all runs and produces
   `ablation_summary.csv` with columns `method, K, seed, coverage,
   coverage_worst, n_covered_zones, time_sec`.

## Acceptance Criteria

- [ ] A5 reward curve monotonically increases (some noise OK) and
  plateaus above random baseline (A0) by ≥ 50% relative gain.
- [ ] A6 final coverage ≥ A5 final coverage by ≥ 2 percentage points
  (otherwise diffusion augmentation isn't helping).
- [ ] A7 worst-case (5th percentile) coverage ≥ A6 worst-case by ≥ 3
  percentage points (otherwise CVaR isn't helping; the paper's C2
  story falls apart and needs revisiting).
- [ ] At least 3 seeds per method, std reported in summary.
- [ ] All baselines (A0–A4) successfully run and produce numbers in
  the summary.
- [ ] Selected candidates from each run are visualizable on the map.
- [ ] `ablation_summary.csv` has the right columns and is consumed by
  Stage 7 paper figures.

## Files to Create

- `src/agents/__init__.py`
- `src/agents/policy.py` — features extractor and policy class.
- `src/agents/cvar_wrapper.py` — CVaR rollout logic.
- `src/agents/callbacks.py` — eval callback, custom logging.
- `src/baselines/__init__.py`
- `src/baselines/random.py`
- `src/baselines/kmeans_greedy.py`
- `src/baselines/genetic.py`
- `src/baselines/milp.py`
- `src/baselines/sponet_adapter.py`
- `configs/ppo_vertiport.yaml`
- `configs/ppo_a7_cvar.yaml`
- `experiments/run_stage6_train.py`
- `experiments/run_stage6_baselines.py`
- `experiments/run_stage6_sensitivity.py`
- `experiments/run_stage6_summary.py`
- `tests/test_policy.py` — smoke test on policy forward pass.
- `tests/test_cvar.py` — verify CVaR computation on a fixed array.

## Common Pitfalls

- **Reward scale**: if `gained_norm` ends up tiny (e.g. each step adds
  0.01), the entropy bonus dominates and PPO does random walk. Scale
  rewards × 10 or × 100 if needed; document in config.
- **Action mask in policy**: forgetting to apply the mask to logits in
  the value function path is OK; forgetting in the policy path is
  fatal. MaskablePPO does this for you only if you use its policy
  class.
- **Vec env with different scenarios**: if two parallel envs sample
  the same scenario (because seeds collide), variance reduction is
  hurt. Ensure independent seeds per sub-env.
- **CVaR α**: too small (0.05) → unstable; too large (0.5) → not
  meaningful robustness. Start with 0.3.
- **Long training**: 2M steps × 8 parallel envs ≈ 4–6 hours on a
  3090. Always launch in tmux / nohup, never in interactive Claude
  Code session.
- **SpoNet adapter**: their codebase expects specific data formats
  (pickled dicts). Write the adapter carefully; differences in input
  representation can lead to apples-to-oranges comparison.
- **MILP timeout**: bilateral MCLP is non-trivial for Gurobi at
  K=10, |C|=300. Set a 1-hour timeout per solve; if it times out, just
  report the timeout in the table.

## Dependencies

- Stages 1–5.
- Adds: `sb3-contrib`, `stable-baselines3`, `deap` or `pymoo`,
  `gurobipy` (academic license).

## Estimated effort

10 days, including baseline implementation and the full ablation matrix
with multiple seeds.