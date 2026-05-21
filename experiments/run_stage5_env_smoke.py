"""Stage-5 PR1 smoke test: run one VertiportEnv episode on real data.

Validation-only. Builds the VertiportEnv from ``configs/env.yaml`` (the
frozen bootstrap scenario source + Stage-2 coverage mask), resets with a
fixed seed, then places ``k_select`` candidates by sampling uniformly
from the action mask. It trains nothing and writes nothing.

Run:
    python -m experiments.run_stage5_env_smoke --config configs/env.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.envs.vertiport_env import VertiportEnv

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "env.yaml"
SMOKE_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to env yaml (default: {DEFAULT_CONFIG.relative_to(REPO)})",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    print("=" * 64)
    print("Stage-5 PR1 VertiportEnv smoke test")
    print("=" * 64)
    print(f"  config              : {args.config}")
    print(f"  scenario_source     : {cfg['scenario_source']}")
    print(f"  scenario_origin     : {cfg['scenario_source_origin']}")

    env = VertiportEnv.from_config(cfg, base_dir=REPO)
    print(
        f"  env built           : n_omega={env.n_omega} "
        f"n_zones={env.n_zones} n_candidates={env.n_candidates} "
        f"k_select={env.k_select}"
    )

    obs, info = env.reset(seed=SMOKE_SEED)
    print(f"  reset(seed={SMOKE_SEED})      : scenario_idx={info['scenario_idx']}")
    print("  observation shapes  :")
    for key, value in obs.items():
        arr = np.asarray(value)
        print(f"      {key:<24}: shape={arr.shape} dtype={arr.dtype}")

    # Episode rollout: uniform random action under the mask.
    rng = np.random.default_rng(SMOKE_SEED)
    total_reward = 0.0
    terminated = False
    step_i = 0
    while not terminated:
        masks = env.action_masks()
        valid = np.flatnonzero(masks)
        action = int(rng.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_i += 1
        print(
            f"  step {step_i:>2}: action={action:>4}  "
            f"reward={reward:.6f}  incr_gain={info['incremental_gain']:.1f}  "
            f"coverage={info['coverage_ratio']:.6f}  "
            f"valid_actions={valid.size}"
        )

    print("-" * 64)
    print(f"  scenario_idx        : {info['scenario_idx']}")
    print(f"  k_select            : {env.k_select}")
    print(f"  selected candidates : {info['selected_candidates']}")
    print(f"  final coverage_ratio: {info['coverage_ratio']:.6f}")
    print(f"  total reward        : {total_reward:.6f}")
    print(f"  total_covered_demand: {info['total_covered_demand']}")
    print(f"  final valid actions : {int(env.action_masks().sum())}")
    print("  -> OK (one full episode ran without exception)")


if __name__ == "__main__":
    main()
