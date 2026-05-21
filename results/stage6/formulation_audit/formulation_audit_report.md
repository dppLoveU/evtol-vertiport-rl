# Stage 6 PR4: RL formulation audit + oracle diagnostics

Diagnostic only -- no PPO training, no CVaR-PPO, no diffusion-source switching, no Stage-5 env behavior changes.

## Inputs

- env config: `configs/env.yaml`
- scenario source: `data/synthetic/od_samples_agg.npy`
- scenario source origin: `bootstrap_day_block`
- k_select: 10
- alpha: 0.3

## Observation-leak ablation

| method | mean | std | min | p05 | CVaR | unique sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy_oracle | 0.432261 | 0.005372 | 0.416708 | 0.418941 | 0.426315 | 6 |
| greedy_blind_mean | 0.434402 | 0.002741 | 0.428089 | 0.430029 | 0.431224 | 1 |
| greedy_robust_cvar_simple | 0.434402 | 0.002741 | 0.428089 | 0.430029 | 0.431224 | 1 |

- oracle - blind mean: -0.002141
- robust - blind min: +0.000000
- robust - blind p05: +0.000000
- robust - blind CVaR: +0.000000
- oracle advantage meaningful by threshold 0.02: False
- weak robustness tension by threshold 0.02: True

## PPO references

- PPO static 20k mean: 0.361545, unique sequences: 1
- PPO demand-aware 20k mean: 0.335205, unique sequences: 1
- greedy oracle beats best PPO by large margin: True

## MILP availability

- MILP unavailable, skipped.

## Answers

1. **Is the current Stage-5 env just an expectation objective?** Yes. Each reset samples one scenario and the reward is the incremental bilateral coverage ratio for that one scenario; PPO optimizes expected return over sampled episodes, not a multi-scenario risk objective.

2. **Are current demand_features oracle ground-truth scenario info?** Yes. They are computed directly from the sampled scenario OD matrix at reset. They are not leaked future actions, but they are ground-truth scenario identity/demand information available to the policy inside the episode.

3. **Does the current scenario set have enough robustness tension?** No. The simple robust-CVaR fixed greedy set is too close to blind-mean on lower-tail coverage, so the current scenarios do not create enough robustness tension.

4. **Should the project continue PPO / CVaR now?** No. Continuing PPO or CVaR-PPO now would optimize an expectation single-scenario environment with weak scenario tension and a greedy oracle still ahead of the PPO references.

5. **Recommended path.** Path beta is the cleanest research path: redesign the MDP around multi-scenario rollout / held-out evaluation and a true robustness reward. Path alpha can patch engineering gaps but will not fully support a robust-RL claim. Path gamma is viable if the paper is reframed around diagnostics / heuristic planning rather than robust RL.

## Path options

- **Path alpha: patch current framework.** Keep single-scenario env, add held-out scenario split, stronger greedy / local-search baselines, and avoid robust claims unless lower-tail gaps become real.
- **Path beta: redesign MDP.** Build multi-scenario rollout/evaluation with a true lower-tail reward and train/evaluate on disjoint scenario families. This is required for a defensible robust-RL claim.
- **Path gamma: reframe paper story.** Treat the current work as scenario diagnostics plus facility-location heuristics, not diffusion-robust RL.
