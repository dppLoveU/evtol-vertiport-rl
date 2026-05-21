"""Stage-6 PR3: scenario-diversity diagnostic + greedy-baseline comparison.

PR1/PR2 found the trained MaskablePPO policy is scenario-blind: it emits
an identical candidate sequence across every eval episode. This script
diagnoses *why*, without any further PPO training. It does three things:

A. **Scenario diversity** -- quantifies how much the 64 frozen bootstrap
   scenarios actually differ in their per-zone demand summary. If they
   barely differ, scenario-blindness is a property of the *data*, not
   the policy.
B. **Greedy baseline** -- runs the deterministic greedy marginal-coverage
   heuristic on every scenario.
C. **Comparison** -- puts random / PPO-static / PPO-demand / greedy side
   by side and states whether the RL policy currently beats greedy.

No PPO training, no CVaR, no custom policy, no scenario regeneration.

Run:
    python -m experiments.run_stage6_diagnostics \\
        --env-config configs/env.yaml \\
        --output-dir results/stage6/diagnostics
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.baselines.greedy import greedy_marginal_coverage

REPO = Path(__file__).resolve().parents[1]

# Reference numbers from earlier Stage-5/6 runs (see docs/progress.md).
RANDOM_PR1_SMOKE_COVERAGE = 0.1047
PPO_STATIC_SELECTED = REPO / "results/stage6/ppo_a6_bootstrap_20k_seed42/selected.json"
PPO_DEMAND_SELECTED = (
    REPO / "results/stage6/ppo_a6_bootstrap_demand_20k_seed42/selected.json"
)

# Scenario-diversity verdict thresholds (on the [0, 1]-normalized
# per-zone demand summary).
DIVERSITY_STD_MEAN_MIN = 0.01
DIVERSITY_STD_MAX_MIN = 0.05


def _resolve(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _scenario_demand_summaries(od: np.ndarray) -> np.ndarray:
    """Per-scenario [Z, 3] demand summary, normalized per scenario.

    Columns: origin outflow, destination inflow, total zone flow. Each
    scenario is divided by its own ``max(total_zone_flow) + eps`` so the
    summary lives in [0, 1] -- the same transform VertiportEnv applies.
    """
    bases = []
    for s in range(od.shape[0]):
        m = od[s].astype(np.float64)
        origin_outflow = m.sum(axis=1)
        destination_inflow = m.sum(axis=0)
        total_zone_flow = origin_outflow + destination_inflow
        base = np.stack(
            [origin_outflow, destination_inflow, total_zone_flow], axis=1
        )
        base = base / (float(total_zone_flow.max()) + 1e-8)
        bases.append(base)
    return np.stack(bases)  # [n_omega, Z, 3]


def scenario_diversity(od: np.ndarray) -> dict[str, Any]:
    """Function A: quantify how much the scenarios differ from each other."""
    bases = _scenario_demand_summaries(od)  # [S, Z, 3]
    n_omega = int(bases.shape[0])

    # Per-element std across scenarios, then aggregated.
    per_element_std = bases.std(axis=0)  # [Z, 3]
    std_mean = float(per_element_std.mean())
    std_max = float(per_element_std.max())

    # Pairwise L2 distance between flattened scenario summaries.
    flat = bases.reshape(n_omega, -1)  # [S, Z*3]
    sq = (flat**2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (flat @ flat.T)
    dist = np.sqrt(np.maximum(d2, 0.0))
    iu = np.triu_indices(n_omega, k=1)
    pairwise = dist[iu]
    pairwise_l2_mean = float(pairwise.mean())
    pairwise_l2_max = float(pairwise.max())

    # Total mass and nonzero-ratio spread across scenarios.
    total_mass = np.array([float(od[s].sum()) for s in range(n_omega)])
    nonzero_ratio = np.array(
        [float((od[s] > 0).mean()) for s in range(n_omega)]
    )

    verdict = (
        "PASS"
        if std_mean > DIVERSITY_STD_MEAN_MIN and std_max > DIVERSITY_STD_MAX_MIN
        else "FAIL"
    )

    return {
        "n_omega": n_omega,
        "demand_summaries_std_mean": std_mean,
        "demand_summaries_std_max": std_max,
        "pairwise_l2_mean": pairwise_l2_mean,
        "pairwise_l2_max": pairwise_l2_max,
        "total_mass": {
            "mean": float(total_mass.mean()),
            "std": float(total_mass.std()),
            "min": float(total_mass.min()),
            "max": float(total_mass.max()),
        },
        "nonzero_ratio": {
            "mean": float(nonzero_ratio.mean()),
            "std": float(nonzero_ratio.std()),
            "min": float(nonzero_ratio.min()),
            "max": float(nonzero_ratio.max()),
        },
        "thresholds": {
            "std_mean_min": DIVERSITY_STD_MEAN_MIN,
            "std_max_min": DIVERSITY_STD_MAX_MIN,
        },
        "scenario_diversity_verdict": verdict,
    }


def greedy_baseline(od: np.ndarray, cov: np.ndarray, k_select: int) -> dict[str, Any]:
    """Function B: run greedy marginal coverage on every scenario."""
    n_omega = int(od.shape[0])
    t0 = time.perf_counter()
    results = [
        greedy_marginal_coverage(od[s], cov, k_select) for s in range(n_omega)
    ]
    runtime_s = time.perf_counter() - t0

    coverages = np.array([r["coverage_ratio"] for r in results], dtype=np.float64)
    sequences = [r["selected_candidates"] for r in results]
    unique = sorted({tuple(seq) for seq in sequences})

    best_idx = int(np.argmax(coverages))
    worst_idx = int(np.argmin(coverages))

    return {
        "eval_episodes": n_omega,
        "k_select": k_select,
        "mean_coverage": float(coverages.mean()),
        "std_coverage": float(coverages.std()),
        "min_coverage": float(coverages.min()),
        "max_coverage": float(coverages.max()),
        "unique_selected_sequences": len(unique),
        "first_selected_candidates": sequences[0],
        "best_scenario": {
            "scenario_idx": best_idx,
            "coverage_ratio": float(coverages[best_idx]),
            "selected_candidates": sequences[best_idx],
        },
        "worst_scenario": {
            "scenario_idx": worst_idx,
            "coverage_ratio": float(coverages[worst_idx]),
            "selected_candidates": sequences[worst_idx],
        },
        "runtime_s": round(runtime_s, 3),
        "selected_candidates_per_scenario": sequences,
        "coverage_ratio_per_scenario": [float(c) for c in coverages],
    }


def _load_ppo_run(path: Path) -> dict[str, Any] | None:
    """Read a PPO selected.json; recompute unique-sequence count locally."""
    if not path.exists():
        return None
    with open(path) as fh:
        data = json.load(fh)
    seqs = data.get("selected_candidates_per_episode", [])
    unique = sorted({tuple(seq) for seq in seqs})
    return {
        "run_name": data.get("run_name"),
        "mean_coverage": float(data["mean_coverage"]),
        "std_coverage": float(data.get("std_coverage", 0.0)),
        "unique_selected_sequences": len(unique),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-config", type=Path, default=REPO / "configs/env.yaml")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "results/stage6/diagnostics"
    )
    args = parser.parse_args()

    env_config = _resolve(args.env_config)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(env_config) as fh:
        cfg = yaml.safe_load(fh)

    od_path = _resolve(cfg["scenario_source"])
    cov_path = _resolve(cfg["cand_covers_zones_path"])
    k_select = int(cfg["k_select"])

    od = np.load(od_path)
    cov = np.load(cov_path).astype(bool)

    print("=" * 64)
    print("Stage-6 PR3 scenario-diversity + greedy diagnostics")
    print("=" * 64)
    print(f"  env_config          : {env_config}")
    print(f"  scenario_source     : {od_path.name}  shape={od.shape}")
    print(f"  cand_covers_zones   : {cov_path.name}  shape={cov.shape}")
    print(f"  k_select            : {k_select}")

    # -- Function A: scenario diversity -------------------------------
    diversity = scenario_diversity(od)
    print("-" * 64)
    print("  [A] scenario diversity")
    print(f"      std_mean        : {diversity['demand_summaries_std_mean']:.6f}")
    print(f"      std_max         : {diversity['demand_summaries_std_max']:.6f}")
    print(f"      pairwise_l2_mean: {diversity['pairwise_l2_mean']:.6f}")
    print(f"      pairwise_l2_max : {diversity['pairwise_l2_max']:.6f}")
    print(f"      verdict         : {diversity['scenario_diversity_verdict']}")

    # -- Function B: greedy baseline ----------------------------------
    greedy = greedy_baseline(od, cov, k_select)
    print("-" * 64)
    print("  [B] greedy marginal-coverage baseline")
    print(
        f"      coverage        : mean={greedy['mean_coverage']:.6f} "
        f"std={greedy['std_coverage']:.6f} "
        f"min={greedy['min_coverage']:.6f} max={greedy['max_coverage']:.6f}"
    )
    print(f"      unique sequences: {greedy['unique_selected_sequences']}")
    print(f"      first selected  : {greedy['first_selected_candidates']}")
    print(f"      runtime         : {greedy['runtime_s']:.3f} s")

    # -- Function C: comparison vs PPO --------------------------------
    ppo_static = _load_ppo_run(PPO_STATIC_SELECTED)
    ppo_demand = _load_ppo_run(PPO_DEMAND_SELECTED)
    greedy_mean = greedy["mean_coverage"]

    rows = [
        {
            "method": "random_pr1_smoke",
            "mean_coverage": RANDOM_PR1_SMOKE_COVERAGE,
            "unique_selected_sequences": "",
            "notes": "PR1 random-policy smoke reference",
        },
        {
            "method": "ppo_static_20k",
            "mean_coverage": ppo_static["mean_coverage"] if ppo_static else "",
            "unique_selected_sequences": (
                ppo_static["unique_selected_sequences"] if ppo_static else ""
            ),
            "notes": "Stage 6 PR1, static observation",
        },
        {
            "method": "ppo_demand_20k",
            "mean_coverage": ppo_demand["mean_coverage"] if ppo_demand else "",
            "unique_selected_sequences": (
                ppo_demand["unique_selected_sequences"] if ppo_demand else ""
            ),
            "notes": "Stage 6 PR2, demand-aware observation",
        },
        {
            "method": "greedy_marginal",
            "mean_coverage": greedy_mean,
            "unique_selected_sequences": greedy["unique_selected_sequences"],
            "notes": "Stage 6 PR3, greedy marginal-coverage baseline",
        },
    ]

    ppo_static_mean = ppo_static["mean_coverage"] if ppo_static else float("nan")
    greedy_minus_ppo_static = greedy_mean - ppo_static_mean
    greedy_beats_ppo = greedy_mean >= ppo_static_mean
    diversity_pass = diversity["scenario_diversity_verdict"] == "PASS"

    if greedy_beats_ppo:
        rl_value_conclusion = (
            "greedy >= PPO static -- the RL policy currently adds NO value "
            "over the greedy heuristic under the current scenario set."
        )
    else:
        rl_value_conclusion = (
            "greedy < PPO static -- PPO has measurable value over greedy "
            "under the current scenario set."
        )

    if greedy_beats_ppo and not diversity_pass:
        recommendation = (
            "A: greedy >= PPO and scenario diversity FAIL -- fix scenario "
            "generation / add perturbation before any further RL work or CVaR."
        )
    elif not greedy_beats_ppo and not diversity_pass:
        recommendation = (
            "B: greedy < PPO but scenario diversity FAIL -- the scenario set "
            "must still be fixed before CVaR is meaningful."
        )
    else:
        recommendation = (
            "C: scenario diversity PASS -- proceed to a structure-aware "
            "policy (CandidateTokenExtractor) and longer training."
        )

    comparison = {
        "random_pr1_smoke": RANDOM_PR1_SMOKE_COVERAGE,
        "ppo_static_20k": ppo_static,
        "ppo_demand_20k": ppo_demand,
        "greedy_marginal": {
            "mean_coverage": greedy_mean,
            "unique_selected_sequences": greedy["unique_selected_sequences"],
        },
        "greedy_minus_ppo_static": float(greedy_minus_ppo_static),
        "greedy_beats_ppo_static": bool(greedy_beats_ppo),
        "scenario_diversity_verdict": diversity["scenario_diversity_verdict"],
        "rl_value_conclusion": rl_value_conclusion,
        "recommendation": recommendation,
    }

    print("-" * 64)
    print("  [C] comparison")
    print(f"      random          : {RANDOM_PR1_SMOKE_COVERAGE:.4f}")
    print(
        f"      PPO static 20k  : "
        f"{ppo_static_mean:.4f}" if ppo_static else "      PPO static 20k  : n/a"
    )
    if ppo_demand:
        print(f"      PPO demand 20k  : {ppo_demand['mean_coverage']:.4f}")
    print(f"      greedy          : {greedy_mean:.4f}")
    print(f"      greedy - PPO    : {greedy_minus_ppo_static:+.4f}")
    print(f"      => {rl_value_conclusion}")
    print(f"      => recommendation {recommendation}")

    # -- write outputs ------------------------------------------------
    diversity_path = output_dir / "scenario_diversity.json"
    with open(diversity_path, "w") as fh:
        json.dump(diversity, fh, indent=2)

    greedy_path = output_dir / "greedy_baseline.json"
    with open(greedy_path, "w") as fh:
        json.dump(greedy, fh, indent=2)

    comparison_csv = output_dir / "comparison.csv"
    with open(comparison_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["method", "mean_coverage", "unique_selected_sequences", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    report_path = output_dir / "diagnostic_report.md"
    _write_report(report_path, diversity, greedy, comparison, k_select)

    print("-" * 64)
    print(f"  scenario_diversity  : {diversity_path}")
    print(f"  greedy_baseline     : {greedy_path}")
    print(f"  comparison.csv      : {comparison_csv}")
    print(f"  diagnostic_report   : {report_path}")
    print("  -> OK (Stage-6 PR3 diagnostics completed)")


def _write_report(
    path: Path,
    diversity: dict[str, Any],
    greedy: dict[str, Any],
    comparison: dict[str, Any],
    k_select: int,
) -> None:
    """Write the human-readable Markdown diagnostic report."""
    ppo_static = comparison["ppo_static_20k"]
    ppo_demand = comparison["ppo_demand_20k"]
    lines = [
        "# Stage 6 PR3: scenario-diversity + greedy diagnostic report",
        "",
        "Diagnostic only -- no PPO training, no CVaR, no scenario regeneration.",
        "",
        "## A. Scenario diversity",
        "",
        f"- scenarios (n_omega): {diversity['n_omega']}",
        f"- demand-summary std (mean): {diversity['demand_summaries_std_mean']:.6f}"
        f"  (threshold > {diversity['thresholds']['std_mean_min']})",
        f"- demand-summary std (max): {diversity['demand_summaries_std_max']:.6f}"
        f"  (threshold > {diversity['thresholds']['std_max_min']})",
        f"- pairwise L2 (mean / max): {diversity['pairwise_l2_mean']:.6f}"
        f" / {diversity['pairwise_l2_max']:.6f}",
        f"- total mass mean/std/min/max: {diversity['total_mass']['mean']:.1f}"
        f" / {diversity['total_mass']['std']:.1f}"
        f" / {diversity['total_mass']['min']:.1f}"
        f" / {diversity['total_mass']['max']:.1f}",
        f"- nonzero ratio mean/std/min/max: {diversity['nonzero_ratio']['mean']:.5f}"
        f" / {diversity['nonzero_ratio']['std']:.5f}"
        f" / {diversity['nonzero_ratio']['min']:.5f}"
        f" / {diversity['nonzero_ratio']['max']:.5f}",
        "",
        f"**Verdict: {diversity['scenario_diversity_verdict']}**",
        "",
        "## B. Greedy marginal-coverage baseline",
        "",
        f"- k_select: {k_select}",
        f"- episodes (one per scenario): {greedy['eval_episodes']}",
        f"- coverage mean/std/min/max: {greedy['mean_coverage']:.6f}"
        f" / {greedy['std_coverage']:.6f}"
        f" / {greedy['min_coverage']:.6f} / {greedy['max_coverage']:.6f}",
        f"- unique selected sequences: {greedy['unique_selected_sequences']}",
        f"- first selected candidates: {greedy['first_selected_candidates']}",
        f"- best scenario {greedy['best_scenario']['scenario_idx']}"
        f" (coverage {greedy['best_scenario']['coverage_ratio']:.6f}):"
        f" {greedy['best_scenario']['selected_candidates']}",
        f"- worst scenario {greedy['worst_scenario']['scenario_idx']}"
        f" (coverage {greedy['worst_scenario']['coverage_ratio']:.6f}):"
        f" {greedy['worst_scenario']['selected_candidates']}",
        f"- runtime: {greedy['runtime_s']:.3f} s",
        "",
        "## C. Comparison vs PPO",
        "",
        "| method | mean coverage | unique sequences |",
        "| --- | --- | --- |",
        f"| random (PR1 smoke) | {comparison['random_pr1_smoke']:.4f} | - |",
    ]
    if ppo_static:
        lines.append(
            f"| PPO static 20k | {ppo_static['mean_coverage']:.4f} "
            f"| {ppo_static['unique_selected_sequences']} |"
        )
    if ppo_demand:
        lines.append(
            f"| PPO demand 20k | {ppo_demand['mean_coverage']:.4f} "
            f"| {ppo_demand['unique_selected_sequences']} |"
        )
    lines += [
        f"| greedy marginal | {greedy['mean_coverage']:.4f} "
        f"| {greedy['unique_selected_sequences']} |",
        "",
        f"- greedy - PPO static: {comparison['greedy_minus_ppo_static']:+.4f}",
        f"- greedy beats PPO static: {comparison['greedy_beats_ppo_static']}",
        "",
        f"**RL value: {comparison['rl_value_conclusion']}**",
        "",
        f"**Recommendation -- {comparison['recommendation']}**",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
