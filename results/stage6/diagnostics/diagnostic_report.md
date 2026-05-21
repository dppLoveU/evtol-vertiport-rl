# Stage 6 PR3: scenario-diversity + greedy diagnostic report

Diagnostic only -- no PPO training, no CVaR, no scenario regeneration.

## A. Scenario diversity

- scenarios (n_omega): 64
- demand-summary std (mean): 0.000755  (threshold > 0.01)
- demand-summary std (max): 0.010561  (threshold > 0.05)
- pairwise L2 (mean / max): 0.060074 / 0.162705
- total mass mean/std/min/max: 466926.1 / 5477.4 / 454985.0 / 479692.0
- nonzero ratio mean/std/min/max: 0.17398 / 0.01042 / 0.13968 / 0.18334

**Verdict: FAIL**

## B. Greedy marginal-coverage baseline

- k_select: 10
- episodes (one per scenario): 64
- coverage mean/std/min/max: 0.432261 / 0.005372 / 0.416708 / 0.439423
- unique selected sequences: 6
- first selected candidates: [155, 448, 256, 250, 307, 38, 75, 557, 405, 174]
- best scenario 62 (coverage 0.439423): [155, 448, 256, 250, 307, 38, 75, 557, 405, 174]
- worst scenario 44 (coverage 0.416708): [0, 201, 494, 256, 307, 38, 363, 502, 75, 404]
- runtime: 0.999 s

## C. Comparison vs PPO

| method | mean coverage | unique sequences |
| --- | --- | --- |
| random (PR1 smoke) | 0.1047 | - |
| PPO static 20k | 0.3615 | 1 |
| PPO demand 20k | 0.3352 | 1 |
| greedy marginal | 0.4323 | 6 |

- greedy - PPO static: +0.0707
- greedy beats PPO static: True

**RL value: greedy >= PPO static -- the RL policy currently adds NO value over the greedy heuristic under the current scenario set.**

**Recommendation -- A: greedy >= PPO and scenario diversity FAIL -- fix scenario generation / add perturbation before any further RL work or CVaR.**
