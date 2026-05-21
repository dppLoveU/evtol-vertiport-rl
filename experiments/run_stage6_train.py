"""Stage-6 PR1: MaskablePPO training entrypoint for VertiportEnv.

This is the Stage-6 PR1 training harness -- a reusable PPO training
entrypoint plus a 20k-timestep mini run on the frozen bootstrap
scenario source. It is deliberately minimal:

- ``MultiInputPolicy`` (no custom ``CandidateTokenExtractor`` yet).
- Frozen bootstrap scenarios via ``configs/env.yaml`` (the A6
  expectation objective over a fixed scenario set).
- No CVaR, no A5/A7, no WandB, no TensorBoard.

The full A5/A6/A7 ablation ladder, the custom policy network, and the
CVaR objective are later Stage-6 work; this harness and its output
schema (``metrics.json`` / ``selected.json``) are meant to be reused by
those runs. See ``docs/plan/stage6_rl_training.md`` and
``docs/decisions.md`` 2026-05-21 "Stage 6 PR1".

Run:
    python -m experiments.run_stage6_train --config configs/ppo_vertiport.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.envs.vertiport_env import VertiportEnv

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "ppo_vertiport.yaml"


def _resolve(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _resolve_device(requested: str) -> str:
    """Map the config ``device`` ('auto' | 'cuda' | 'cpu') to a real device."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


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
        "scenario_idx": int(info["scenario_idx"]),
        "selected_count": int(info["selected_count"]),
        "selected_candidates": [int(c) for c in info["selected_candidates"]],
        "coverage_ratio": float(info["coverage_ratio"]),
        "total_reward": float(total_reward),
        "total_covered_demand": int(info["total_covered_demand"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    # Imported here so a missing sb3-contrib fails with a clear message.
    import gymnasium
    import sb3_contrib
    import stable_baselines3
    from sb3_contrib import MaskablePPO

    config_path = _resolve(args.config)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    run_name = str(cfg["run_name"])
    seed = int(cfg["seed"])
    method = str(cfg["method"])
    policy = str(cfg["policy"])
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    out_cfg = cfg["output"]

    env_config_path = _resolve(cfg["env_config"])
    result_dir = _resolve(out_cfg["result_dir"])
    model_dir = _resolve(out_cfg["model_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(train_cfg["total_timesteps"])
    n_steps = int(train_cfg["n_steps"])
    batch_size = int(train_cfg["batch_size"])
    gamma = float(train_cfg["gamma"])
    learning_rate = float(train_cfg["learning_rate"])
    ent_coef = float(train_cfg["ent_coef"])
    vf_coef = float(train_cfg["vf_coef"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    eval_episodes = int(eval_cfg["eval_episodes"])
    deterministic = bool(eval_cfg["deterministic"])

    print("=" * 64)
    print("Stage-6 PR1 MaskablePPO training harness")
    print("=" * 64)
    print(f"  run_name            : {run_name}")
    print(f"  method              : {method}")
    print(f"  config              : {config_path}")
    print(f"  env_config          : {env_config_path}")
    print(f"  seed                : {seed}")
    print(f"  total_timesteps     : {total_timesteps}")
    print(f"  gymnasium           : {gymnasium.__version__}")
    print(f"  stable_baselines3   : {stable_baselines3.__version__}")
    print(f"  sb3_contrib         : {sb3_contrib.__version__}")

    device = _resolve_device(str(train_cfg["device"]))
    print(f"  device (requested)  : {device}")

    train_env = VertiportEnv.from_config(env_config_path, base_dir=REPO)
    eval_env = VertiportEnv.from_config(env_config_path, base_dir=REPO)
    print(
        f"  env built           : n_omega={train_env.n_omega} "
        f"n_zones={train_env.n_zones} n_candidates={train_env.n_candidates} "
        f"k_select={train_env.k_select}"
    )

    def _make_model(dev: str) -> MaskablePPO:
        return MaskablePPO(
            policy,
            train_env,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            seed=seed,
            device=dev,
            verbose=1,
        )

    model = _make_model(device)

    # Pre-train sanity: action mask -> a legal predicted action.
    obs, _ = eval_env.reset(seed=seed)
    masks = eval_env.action_masks()
    assert masks.shape == (eval_env.n_candidates,) and masks.dtype == bool
    sanity_action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    sanity_action = int(sanity_action)
    assert masks[sanity_action], "predicted action is not in the valid mask"
    print(f"  pre-train predict   : action={sanity_action} (valid)")

    # Training, with a one-shot CPU fallback if CUDA misbehaves.
    cuda_fallback = False
    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=total_timesteps)
    except RuntimeError as exc:
        if device != "cuda":
            raise
        print(f"  CUDA training failed ({exc}); retrying on CPU.")
        device = "cpu"
        cuda_fallback = True
        model = _make_model(device)
        model.learn(total_timesteps=total_timesteps)
    train_wall_s = time.perf_counter() - t0
    print(f"  device (used)       : {device}")
    print(f"  cuda_fallback       : {cuda_fallback}")
    print(f"  train wall          : {train_wall_s:.2f} s")

    # Evaluation: deterministic, masked, no model update.
    episodes = [
        _run_episode(eval_env, model, seed=seed + i) for i in range(eval_episodes)
    ]
    coverages = np.array([ep["coverage_ratio"] for ep in episodes], dtype=np.float64)
    rewards = np.array([ep["total_reward"] for ep in episodes], dtype=np.float64)
    for ep_i, ep in enumerate(episodes):
        print(
            f"  eval ep {ep_i:2d}: scenario_idx={ep['scenario_idx']:3d} "
            f"coverage_ratio={ep['coverage_ratio']:.6f} "
            f"total_reward={ep['total_reward']:.6f}"
        )

    model_path = model_dir / "model.zip"
    model.save(model_path)

    # -- write outputs ------------------------------------------------
    # Frozen copy of the resolved training config.
    config_out = result_dir / "config.yaml"
    with open(config_out, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)

    metrics = {
        "kind": "stage6_pr1_maskableppo_train",
        "note": "training harness / mini run, NOT the final paper baseline",
        "run_name": run_name,
        "method": method,
        "config": str(config_path),
        "env_config": str(env_config_path),
        "seed": seed,
        "device": device,
        "cuda_fallback": cuda_fallback,
        "versions": {
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__,
        },
        "env": {
            "n_omega": train_env.n_omega,
            "n_zones": train_env.n_zones,
            "n_candidates": train_env.n_candidates,
            "k_select": train_env.k_select,
        },
        "ppo": {
            "policy": policy,
            "total_timesteps": total_timesteps,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
        },
        "train_wall_s": round(train_wall_s, 3),
        "eval": {
            "eval_episodes": eval_episodes,
            "deterministic": deterministic,
            "protocol": "single post-training deterministic masked eval",
            "mean_coverage": float(coverages.mean()),
            "std_coverage": float(coverages.std()),
            "min_coverage": float(coverages.min()),
            "max_coverage": float(coverages.max()),
            "mean_total_reward": float(rewards.mean()),
        },
        "model_path": str(model_path),
    }
    metrics_path = result_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    selected = {
        "kind": "stage6_pr1_selected_candidates",
        "run_name": run_name,
        "method": method,
        "seed": seed,
        "deterministic": deterministic,
        "k_select": train_env.k_select,
        "eval_episodes": eval_episodes,
        "selected_candidates_per_episode": [
            ep["selected_candidates"] for ep in episodes
        ],
        "coverage_ratio_per_episode": [ep["coverage_ratio"] for ep in episodes],
        "scenario_idx_per_episode": [ep["scenario_idx"] for ep in episodes],
        "mean_coverage": float(coverages.mean()),
        "std_coverage": float(coverages.std()),
        "min_coverage": float(coverages.min()),
        "max_coverage": float(coverages.max()),
    }
    selected_path = result_dir / "selected.json"
    with open(selected_path, "w") as fh:
        json.dump(selected, fh, indent=2)

    print("-" * 64)
    print(f"  model saved         : {model_path}")
    print(f"  config saved        : {config_out}")
    print(f"  metrics saved       : {metrics_path}")
    print(f"  selected saved      : {selected_path}")
    print(
        f"  eval coverage       : mean={metrics['eval']['mean_coverage']:.6f} "
        f"std={metrics['eval']['std_coverage']:.6f} "
        f"min={metrics['eval']['min_coverage']:.6f} "
        f"max={metrics['eval']['max_coverage']:.6f}"
    )
    print("  -> OK (Stage-6 PR1 MaskablePPO training + eval completed)")


if __name__ == "__main__":
    main()
