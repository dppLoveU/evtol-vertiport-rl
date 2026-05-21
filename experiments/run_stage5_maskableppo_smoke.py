"""Stage-5 PR2 smoke test: a minimal MaskablePPO training run on VertiportEnv.

This is a SMOKE test, not a paper baseline. It verifies that the
Gymnasium-ified ``VertiportEnv`` plugs into ``sb3-contrib``'s MaskablePPO:
the action mask is consumed, a short ``learn()`` completes, and a trained
policy can roll out deterministic evaluation episodes. The default
512-timestep budget runs in seconds on CPU.

No CVaR, no long training, no Stage-6 work.

Run:
    python -m experiments.run_stage5_maskableppo_smoke \\
        --config configs/env.yaml --total-timesteps 512 --seed 42 \\
        --output-dir results/stage5/maskableppo_smoke \\
        --model-dir models/rl/maskableppo_smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.envs.vertiport_env import VertiportEnv

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "env.yaml"

N_EVAL_EPISODES = 3


def _resolve(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _run_episode(env: VertiportEnv, model: Any, seed: int) -> dict[str, Any]:
    """Roll out one deterministic, masked evaluation episode (no learning)."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
    return {
        "seed": seed,
        "scenario_idx": info["scenario_idx"],
        "selected_count": info["selected_count"],
        "selected_candidates": info["selected_candidates"],
        "final_coverage_ratio": float(info["coverage_ratio"]),
        "total_reward": float(total_reward),
        "total_covered_demand": int(info["total_covered_demand"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--total-timesteps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "results/stage5/maskableppo_smoke"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=REPO / "models/rl/maskableppo_smoke"
    )
    args = parser.parse_args()

    # Imported here so a missing sb3-contrib fails with a clear message.
    import gymnasium
    import sb3_contrib
    import stable_baselines3
    from sb3_contrib import MaskablePPO

    output_dir = _resolve(args.output_dir)
    model_dir = _resolve(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Stage-5 PR2 MaskablePPO training smoke")
    print("=" * 64)
    print(f"  gymnasium           : {gymnasium.__version__}")
    print(f"  stable_baselines3   : {stable_baselines3.__version__}")
    print(f"  sb3_contrib         : {sb3_contrib.__version__}")
    print(f"  config              : {args.config}")
    print(f"  total_timesteps     : {args.total_timesteps}")
    print(f"  seed                : {args.seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device (requested)  : {device}")

    train_env = VertiportEnv.from_config(args.config, base_dir=REPO)
    eval_env = VertiportEnv.from_config(args.config, base_dir=REPO)
    print(
        f"  env built           : n_omega={train_env.n_omega} "
        f"n_zones={train_env.n_zones} n_candidates={train_env.n_candidates} "
        f"k_select={train_env.k_select}"
    )

    def _make_model(dev: str) -> MaskablePPO:
        return MaskablePPO(
            "MultiInputPolicy",
            train_env,
            n_steps=64,
            batch_size=64,
            gamma=1.0,
            learning_rate=3e-4,
            seed=args.seed,
            device=dev,
            verbose=1,
        )

    # Pre-train sanity: action mask -> a legal predicted action.
    model = _make_model(device)
    obs, _ = eval_env.reset(seed=args.seed)
    masks = eval_env.action_masks()
    assert masks.shape == (eval_env.n_candidates,) and masks.dtype == bool
    sanity_action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    sanity_action = int(sanity_action)
    assert masks[sanity_action], "predicted action is not in the valid mask"
    print(f"  pre-train predict   : action={sanity_action} (valid)")

    # Short training run, with a one-shot CPU fallback if CUDA misbehaves.
    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=args.total_timesteps)
    except RuntimeError as exc:
        if device != "cuda":
            raise
        print(f"  CUDA training failed ({exc}); retrying on CPU.")
        device = "cpu"
        model = _make_model(device)
        model.learn(total_timesteps=args.total_timesteps)
    train_wall_s = time.perf_counter() - t0
    print(f"  device (used)       : {device}")
    print(f"  train wall          : {train_wall_s:.2f} s")

    # Evaluation: deterministic, masked, no model update.
    episodes = [
        _run_episode(eval_env, model, seed=args.seed + i)
        for i in range(N_EVAL_EPISODES)
    ]
    for ep_i, ep in enumerate(episodes):
        print(
            f"  eval ep {ep_i}: scenario_idx={ep['scenario_idx']} "
            f"selected_count={ep['selected_count']} "
            f"coverage_ratio={ep['final_coverage_ratio']:.6f} "
            f"total_reward={ep['total_reward']:.6f}"
        )

    model_path = model_dir / "model.zip"
    model.save(model_path)

    metrics = {
        "kind": "stage5_pr2_maskableppo_smoke",
        "note": "SMOKE TEST, not a paper baseline",
        "versions": {
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__,
        },
        "config": str(args.config),
        "seed": args.seed,
        "device": device,
        "total_timesteps": args.total_timesteps,
        "train_wall_s": round(train_wall_s, 3),
        "env": {
            "n_omega": train_env.n_omega,
            "n_zones": train_env.n_zones,
            "n_candidates": train_env.n_candidates,
            "k_select": train_env.k_select,
        },
        "ppo": {
            "policy": "MultiInputPolicy",
            "n_steps": 64,
            "batch_size": 64,
            "gamma": 1.0,
            "learning_rate": 3e-4,
        },
        "eval_episodes": episodes,
        "eval_mean_coverage_ratio": float(
            np.mean([ep["final_coverage_ratio"] for ep in episodes])
        ),
        "eval_mean_total_reward": float(
            np.mean([ep["total_reward"] for ep in episodes])
        ),
        "model_path": str(model_path),
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("-" * 64)
    print(f"  model saved         : {model_path}")
    print(f"  metrics saved       : {metrics_path}")
    print(
        f"  eval mean coverage  : {metrics['eval_mean_coverage_ratio']:.6f}  "
        f"mean reward: {metrics['eval_mean_total_reward']:.6f}"
    )
    print("  -> OK (MaskablePPO smoke training + eval completed)")


if __name__ == "__main__":
    main()
