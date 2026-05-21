"""Stage-6 PR4: RL formulation audit + oracle diagnostics.

Diagnostic only. This script does not train PPO, does not implement
CVaR-PPO, does not switch scenario sources, and does not change the
Stage-5 environment. It compares oracle per-scenario greedy placement
against fixed blind / robust greedy placements to test whether the
current MDP observation and scenario set can support the paper's robust
RL claim.

Run:
    python -m experiments.run_stage6_formulation_audit \\
        --env-config configs/env.yaml \\
        --output-dir results/stage6/formulation_audit \\
        --alpha 0.3
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.baselines.greedy import (
    evaluate_fixed_selection_many,
    greedy_blind_mean,
    greedy_marginal_coverage,
    robust_cvar_greedy,
)

REPO = Path(__file__).resolve().parents[1]

PPO_STATIC_SELECTED = REPO / "results/stage6/ppo_a6_bootstrap_20k_seed42/selected.json"
PPO_DEMAND_SELECTED = (
    REPO / "results/stage6/ppo_a6_bootstrap_demand_20k_seed42/selected.json"
)

ORACLE_ADVANTAGE_THRESHOLD = 0.02
ROBUST_TENSION_THRESHOLD = 0.02
PPO_LARGE_MARGIN_THRESHOLD = 0.05


def _resolve(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _tail_stats(values: np.ndarray, alpha: float) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    bottom_n = max(1, int(np.ceil(alpha * arr.size)))
    sorted_arr = np.sort(arr)
    return {
        "mean_coverage": float(arr.mean()),
        "std_coverage": float(arr.std()),
        "min_coverage": float(arr.min()),
        "max_coverage": float(arr.max()),
        "p05_coverage": float(np.percentile(arr, 5)),
        f"cvar{alpha:g}_coverage": float(sorted_arr[:bottom_n].mean()),
        "bottom_n": int(bottom_n),
    }


def _method_summary(
    *,
    method: str,
    coverages: list[float] | np.ndarray,
    selected_sequences: list[list[int]],
    alpha: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cov = np.asarray(coverages, dtype=np.float64)
    unique = sorted({tuple(seq) for seq in selected_sequences})
    out: dict[str, Any] = {
        "method": method,
        "eval_episodes": int(cov.size),
        "unique_selected_sequences": int(len(unique)),
        "first_selected_candidates": list(selected_sequences[0]) if selected_sequences else [],
        "selected_candidates_per_scenario": selected_sequences,
        "coverage_ratio_per_scenario": [float(c) for c in cov],
    }
    out.update(_tail_stats(cov, alpha))
    if extra:
        out.update(extra)
    return out


def run_greedy_oracle(
    od: np.ndarray,
    cov: np.ndarray,
    k_select: int,
    alpha: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    results = [
        greedy_marginal_coverage(od[s], cov, k_select) for s in range(od.shape[0])
    ]
    runtime_s = time.perf_counter() - t0
    coverages = [float(r["coverage_ratio"]) for r in results]
    sequences = [list(r["selected_candidates"]) for r in results]
    return _method_summary(
        method="greedy_oracle",
        coverages=coverages,
        selected_sequences=sequences,
        alpha=alpha,
        extra={"runtime_s": round(runtime_s, 3)},
    )


def run_greedy_blind_mean(
    od: np.ndarray,
    cov: np.ndarray,
    k_select: int,
    alpha: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = greedy_blind_mean(od, cov, k_select)
    runtime_s = time.perf_counter() - t0
    selected = list(result["selected_candidates"])
    sequences = [selected for _ in range(od.shape[0])]
    return _method_summary(
        method="greedy_blind_mean",
        coverages=result["coverage_ratio_per_scenario"],
        selected_sequences=sequences,
        alpha=alpha,
        extra={
            "selected_candidates": selected,
            "mean_od_greedy_coverage_ratio": float(
                result["mean_od_greedy_coverage_ratio"]
            ),
            "mean_od_gains": result["mean_od_gains"],
            "runtime_s": round(runtime_s, 3),
        },
    )


def run_greedy_robust(
    od: np.ndarray,
    cov: np.ndarray,
    k_select: int,
    alpha: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = robust_cvar_greedy(od, cov, k_select, alpha=alpha)
    runtime_s = time.perf_counter() - t0
    selected = list(result["selected_candidates"])
    sequences = [selected for _ in range(od.shape[0])]
    return _method_summary(
        method="greedy_robust_cvar_simple",
        coverages=result["coverage_ratio_per_scenario"],
        selected_sequences=sequences,
        alpha=alpha,
        extra={
            "selected_candidates": selected,
            "robust_gains": result["robust_gains"],
            "bottom_n": result["bottom_n"],
            "runtime_s": round(runtime_s, 3),
        },
    )


def _load_ppo_reference(path: Path) -> dict[str, Any] | None:
    data = _load_json(path)
    if data is None:
        return None
    seqs = data.get("selected_candidates_per_episode", [])
    unique = sorted({tuple(seq) for seq in seqs})
    return {
        "run_name": data.get("run_name"),
        "mean_coverage": float(data["mean_coverage"]),
        "std_coverage": float(data.get("std_coverage", 0.0)),
        "min_coverage": float(data.get("min_coverage", np.nan)),
        "max_coverage": float(data.get("max_coverage", np.nan)),
        "unique_selected_sequences": len(unique),
    }


def milp_availability_and_smoke(
    od: np.ndarray,
    cov: np.ndarray,
    k_select: int,
    *,
    time_limit_s: float = 120.0,
    max_scenarios: int = 3,
) -> dict[str, Any]:
    """Check gurobipy availability and optionally run a bounded smoke."""
    check = subprocess.run(
        [sys.executable, "-c", "import gurobipy"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        return {
            "available": False,
            "skipped": True,
            "reason": "MILP unavailable: gurobipy import failed",
            "stderr": check.stderr.strip(),
            "smoke_results": [],
        }

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - depends on local license.
        return {
            "available": False,
            "skipped": True,
            "reason": f"MILP unavailable after import check: {exc}",
            "smoke_results": [],
        }

    smoke_results: list[dict[str, Any]] = []
    for scenario_idx in range(min(max_scenarios, od.shape[0])):
        t0 = time.perf_counter()
        try:
            m = od[scenario_idx].astype(np.float64)
            nz_o, nz_d = np.nonzero(m > 0)
            model = gp.Model(f"stage6_pr4_milp_smoke_s{scenario_idx}")
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = float(time_limit_s)

            x = model.addVars(cov.shape[0], vtype=GRB.BINARY, name="x")
            y = model.addVars(cov.shape[1], vtype=GRB.BINARY, name="y")
            w = model.addVars(len(nz_o), vtype=GRB.BINARY, name="w")

            model.addConstr(gp.quicksum(x[c] for c in range(cov.shape[0])) == k_select)
            for z in range(cov.shape[1]):
                covering = np.flatnonzero(cov[:, z])
                if len(covering) == 0:
                    model.addConstr(y[z] == 0)
                else:
                    model.addConstr(
                        y[z] <= gp.quicksum(x[int(c)] for c in covering)
                    )
            for idx, (o, d) in enumerate(zip(nz_o, nz_d, strict=True)):
                model.addConstr(w[idx] <= y[int(o)])
                model.addConstr(w[idx] <= y[int(d)])

            model.setObjective(
                gp.quicksum(float(m[o, d]) * w[idx] for idx, (o, d) in enumerate(zip(nz_o, nz_d, strict=True))),
                GRB.MAXIMIZE,
            )
            model.optimize()

            smoke_results.append(
                {
                    "scenario_idx": scenario_idx,
                    "status": int(model.Status),
                    "status_name": _gurobi_status_name(model.Status, GRB),
                    "objective": (
                        float(model.ObjVal) if model.SolCount > 0 else None
                    ),
                    "mip_gap": (
                        float(model.MIPGap)
                        if model.SolCount > 0 and model.IsMIP
                        else None
                    ),
                    "runtime_s": round(float(model.Runtime), 3),
                    "wall_s": round(time.perf_counter() - t0, 3),
                    "n_nonzero_od_pairs": int(len(nz_o)),
                }
            )
        except Exception as exc:  # pragma: no cover - depends on local license.
            smoke_results.append(
                {
                    "scenario_idx": scenario_idx,
                    "status": None,
                    "status_name": "ERROR",
                    "objective": None,
                    "mip_gap": None,
                    "runtime_s": round(time.perf_counter() - t0, 3),
                    "error": str(exc),
                }
            )

    return {
        "available": True,
        "skipped": False,
        "time_limit_s": float(time_limit_s),
        "max_scenarios": int(max_scenarios),
        "smoke_results": smoke_results,
    }


def _gurobi_status_name(status: int, grb: Any) -> str:
    names = {
        grb.OPTIMAL: "OPTIMAL",
        grb.TIME_LIMIT: "TIME_LIMIT",
        grb.INFEASIBLE: "INFEASIBLE",
        grb.INF_OR_UNBD: "INF_OR_UNBD",
        grb.UNBOUNDED: "UNBOUNDED",
        grb.INTERRUPTED: "INTERRUPTED",
    }
    return names.get(status, str(status))


def _comparison(
    oracle: dict[str, Any],
    blind: dict[str, Any],
    robust: dict[str, Any],
    *,
    alpha: float,
) -> dict[str, Any]:
    cvar_key = f"cvar{alpha:g}_coverage"
    oracle_minus_blind = oracle["mean_coverage"] - blind["mean_coverage"]
    robust_minus_blind_min = robust["min_coverage"] - blind["min_coverage"]
    robust_minus_blind_p05 = robust["p05_coverage"] - blind["p05_coverage"]
    robust_minus_blind_cvar = robust[cvar_key] - blind[cvar_key]

    ppo_static = _load_ppo_reference(PPO_STATIC_SELECTED)
    ppo_demand = _load_ppo_reference(PPO_DEMAND_SELECTED)
    ppo_best_mean = None
    if ppo_static or ppo_demand:
        ppo_best_mean = max(
            r["mean_coverage"] for r in [ppo_static, ppo_demand] if r is not None
        )

    oracle_beats_ppo_large = (
        ppo_best_mean is not None
        and oracle["mean_coverage"] - ppo_best_mean >= PPO_LARGE_MARGIN_THRESHOLD
    )

    return {
        "alpha": float(alpha),
        "oracle_minus_blind_mean": float(oracle_minus_blind),
        "robust_minus_blind_min": float(robust_minus_blind_min),
        "robust_minus_blind_p05": float(robust_minus_blind_p05),
        "robust_minus_blind_cvar": float(robust_minus_blind_cvar),
        "observation_oracle_advantage_meaningful": bool(
            abs(oracle_minus_blind) >= ORACLE_ADVANTAGE_THRESHOLD
        ),
        "weak_robustness_tension": bool(
            abs(robust_minus_blind_cvar) < ROBUST_TENSION_THRESHOLD
        ),
        "robust_improves_worst": bool(robust_minus_blind_min > 0.0),
        "robust_improves_p05": bool(robust_minus_blind_p05 > 0.0),
        "robust_improves_cvar": bool(robust_minus_blind_cvar > 0.0),
        "ppo_static_20k": ppo_static,
        "ppo_demand_20k": ppo_demand,
        "ppo_best_mean_coverage": ppo_best_mean,
        "oracle_beats_ppo_by_large_margin": bool(oracle_beats_ppo_large),
        "thresholds": {
            "oracle_minus_blind_mean_abs_min": ORACLE_ADVANTAGE_THRESHOLD,
            "robust_minus_blind_cvar_abs_min": ROBUST_TENSION_THRESHOLD,
            "oracle_minus_ppo_large_margin_min": PPO_LARGE_MARGIN_THRESHOLD,
        },
    }


def _write_csv(
    path: Path,
    methods: list[dict[str, Any]],
    *,
    alpha: float,
) -> None:
    cvar_key = f"cvar{alpha:g}_coverage"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "mean_coverage",
                "std_coverage",
                "min_coverage",
                "p05_coverage",
                cvar_key,
                "max_coverage",
                "unique_selected_sequences",
                "first_selected_candidates",
                "runtime_s",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for method in methods:
            writer.writerow(
                {
                    "method": method["method"],
                    "mean_coverage": method["mean_coverage"],
                    "std_coverage": method["std_coverage"],
                    "min_coverage": method["min_coverage"],
                    "p05_coverage": method["p05_coverage"],
                    cvar_key: method[cvar_key],
                    "max_coverage": method["max_coverage"],
                    "unique_selected_sequences": method["unique_selected_sequences"],
                    "first_selected_candidates": method["first_selected_candidates"],
                    "runtime_s": method.get("runtime_s", ""),
                }
            )


def _write_report(
    path: Path,
    *,
    env_config: Path,
    cfg: dict[str, Any],
    oracle: dict[str, Any],
    blind: dict[str, Any],
    robust: dict[str, Any],
    comparison: dict[str, Any],
    milp: dict[str, Any],
    alpha: float,
) -> None:
    cvar_key = f"cvar{alpha:g}_coverage"
    ppo_static = comparison["ppo_static_20k"]
    ppo_demand = comparison["ppo_demand_20k"]

    milp_line = (
        "MILP unavailable, skipped."
        if milp.get("skipped")
        else f"MILP available; smoke results: {milp.get('smoke_results', [])}"
    )

    if comparison["weak_robustness_tension"]:
        robustness_answer = (
            "No. The simple robust-CVaR fixed greedy set is too close to "
            "blind-mean on lower-tail coverage, so the current scenarios "
            "do not create enough robustness tension."
        )
    else:
        robustness_answer = (
            "Partially. The robust fixed greedy set separates from blind-mean "
            "on lower-tail coverage, but this is only a diagnostic baseline."
        )

    continue_answer = (
        "No. Continuing PPO or CVaR-PPO now would optimize an expectation "
        "single-scenario environment with weak scenario tension and a greedy "
        "oracle still ahead of the PPO references."
        if comparison["oracle_beats_ppo_by_large_margin"]
        or comparison["weak_robustness_tension"]
        else "Only after a stronger baseline and held-out scenario protocol are added."
    )

    recommended_path = (
        "Path beta is the cleanest research path: redesign the MDP around "
        "multi-scenario rollout / held-out evaluation and a true robustness "
        "reward. Path alpha can patch engineering gaps but will not fully "
        "support a robust-RL claim. Path gamma is viable if the paper is "
        "reframed around diagnostics / heuristic planning rather than robust RL."
    )

    lines = [
        "# Stage 6 PR4: RL formulation audit + oracle diagnostics",
        "",
        "Diagnostic only -- no PPO training, no CVaR-PPO, no diffusion-source switching, no Stage-5 env behavior changes.",
        "",
        "## Inputs",
        "",
        f"- env config: `{env_config.relative_to(REPO)}`",
        f"- scenario source: `{cfg['scenario_source']}`",
        f"- scenario source origin: `{cfg.get('scenario_source_origin', 'unknown')}`",
        f"- k_select: {cfg['k_select']}",
        f"- alpha: {alpha}",
        "",
        "## Observation-leak ablation",
        "",
        "| method | mean | std | min | p05 | CVaR | unique sequences |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| greedy_oracle | {oracle['mean_coverage']:.6f} | {oracle['std_coverage']:.6f} | {oracle['min_coverage']:.6f} | {oracle['p05_coverage']:.6f} | {oracle[cvar_key]:.6f} | {oracle['unique_selected_sequences']} |",
        f"| greedy_blind_mean | {blind['mean_coverage']:.6f} | {blind['std_coverage']:.6f} | {blind['min_coverage']:.6f} | {blind['p05_coverage']:.6f} | {blind[cvar_key]:.6f} | {blind['unique_selected_sequences']} |",
        f"| greedy_robust_cvar_simple | {robust['mean_coverage']:.6f} | {robust['std_coverage']:.6f} | {robust['min_coverage']:.6f} | {robust['p05_coverage']:.6f} | {robust[cvar_key]:.6f} | {robust['unique_selected_sequences']} |",
        "",
        f"- oracle - blind mean: {comparison['oracle_minus_blind_mean']:+.6f}",
        f"- robust - blind min: {comparison['robust_minus_blind_min']:+.6f}",
        f"- robust - blind p05: {comparison['robust_minus_blind_p05']:+.6f}",
        f"- robust - blind CVaR: {comparison['robust_minus_blind_cvar']:+.6f}",
        f"- oracle advantage meaningful by threshold 0.02: {comparison['observation_oracle_advantage_meaningful']}",
        f"- weak robustness tension by threshold 0.02: {comparison['weak_robustness_tension']}",
        "",
        "## PPO references",
        "",
    ]
    if ppo_static:
        lines.append(
            f"- PPO static 20k mean: {ppo_static['mean_coverage']:.6f}, unique sequences: {ppo_static['unique_selected_sequences']}"
        )
    if ppo_demand:
        lines.append(
            f"- PPO demand-aware 20k mean: {ppo_demand['mean_coverage']:.6f}, unique sequences: {ppo_demand['unique_selected_sequences']}"
        )
    lines += [
        f"- greedy oracle beats best PPO by large margin: {comparison['oracle_beats_ppo_by_large_margin']}",
        "",
        "## MILP availability",
        "",
        f"- {milp_line}",
        "",
        "## Answers",
        "",
        "1. **Is the current Stage-5 env just an expectation objective?** Yes. Each reset samples one scenario and the reward is the incremental bilateral coverage ratio for that one scenario; PPO optimizes expected return over sampled episodes, not a multi-scenario risk objective.",
        "",
        "2. **Are current demand_features oracle ground-truth scenario info?** Yes. They are computed directly from the sampled scenario OD matrix at reset. They are not leaked future actions, but they are ground-truth scenario identity/demand information available to the policy inside the episode.",
        "",
        f"3. **Does the current scenario set have enough robustness tension?** {robustness_answer}",
        "",
        f"4. **Should the project continue PPO / CVaR now?** {continue_answer}",
        "",
        f"5. **Recommended path.** {recommended_path}",
        "",
        "## Path options",
        "",
        "- **Path alpha: patch current framework.** Keep single-scenario env, add held-out scenario split, stronger greedy / local-search baselines, and avoid robust claims unless lower-tail gaps become real.",
        "- **Path beta: redesign MDP.** Build multi-scenario rollout/evaluation with a true lower-tail reward and train/evaluate on disjoint scenario families. This is required for a defensible robust-RL claim.",
        "- **Path gamma: reframe paper story.** Treat the current work as scenario diagnostics plus facility-location heuristics, not diffusion-robust RL.",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-config", type=Path, default=REPO / "configs/env.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/stage6/formulation_audit",
    )
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--milp-time-limit-s", type=float, default=120.0)
    parser.add_argument("--milp-max-scenarios", type=int, default=3)
    args = parser.parse_args()

    env_config = _resolve(args.env_config)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(env_config) as fh:
        cfg = yaml.safe_load(fh)

    od_path = _resolve(cfg["scenario_source"])
    cov_path = _resolve(cfg["cand_covers_zones_path"])
    k_select = int(cfg["k_select"])
    alpha = float(args.alpha)

    od = np.load(od_path)
    cov = np.load(cov_path).astype(bool)

    print("=" * 64)
    print("Stage-6 PR4 formulation audit + oracle diagnostics")
    print("=" * 64)
    print(f"  env_config          : {env_config}")
    print(f"  scenario_source     : {od_path.name}  shape={od.shape}")
    print(f"  cand_covers_zones   : {cov_path.name}  shape={cov.shape}")
    print(f"  k_select            : {k_select}")
    print(f"  alpha               : {alpha}")

    print("-" * 64)
    print("  [A1] greedy oracle")
    oracle = run_greedy_oracle(od, cov, k_select, alpha)
    print(
        f"      mean={oracle['mean_coverage']:.6f} "
        f"std={oracle['std_coverage']:.6f} min={oracle['min_coverage']:.6f}"
    )

    print("-" * 64)
    print("  [A2] greedy blind mean")
    blind = run_greedy_blind_mean(od, cov, k_select, alpha)
    print(
        f"      mean={blind['mean_coverage']:.6f} "
        f"std={blind['std_coverage']:.6f} min={blind['min_coverage']:.6f}"
    )

    print("-" * 64)
    print("  [A3] greedy robust CVaR simple")
    robust = run_greedy_robust(od, cov, k_select, alpha)
    print(
        f"      mean={robust['mean_coverage']:.6f} "
        f"std={robust['std_coverage']:.6f} min={robust['min_coverage']:.6f}"
    )

    comparison = _comparison(oracle, blind, robust, alpha=alpha)
    print("-" * 64)
    print("  [B] MILP availability check")
    milp = milp_availability_and_smoke(
        od,
        cov,
        k_select,
        time_limit_s=float(args.milp_time_limit_s),
        max_scenarios=int(args.milp_max_scenarios),
    )
    if milp.get("skipped"):
        print(f"      {milp['reason']}; skipped")
    else:
        print(f"      available; smoke_results={milp['smoke_results']}")

    audit = {
        "kind": "stage6_pr4_formulation_audit",
        "env_config": str(env_config.relative_to(REPO)),
        "scenario_source": str(od_path.relative_to(REPO)),
        "scenario_source_origin": cfg.get("scenario_source_origin"),
        "cand_covers_zones_path": str(cov_path.relative_to(REPO)),
        "n_omega": int(od.shape[0]),
        "n_zones": int(od.shape[1]),
        "n_candidates": int(cov.shape[0]),
        "k_select": int(k_select),
        "alpha": alpha,
        "methods": {
            "greedy_oracle": oracle,
            "greedy_blind_mean": blind,
            "greedy_robust_cvar_simple": robust,
        },
        "comparison": comparison,
        "milp": milp,
        "fixed_selection_eval_check": evaluate_fixed_selection_many(
            od, cov, blind["selected_candidates"]
        ),
    }

    json_path = output_dir / "formulation_audit.json"
    csv_path = output_dir / "formulation_audit.csv"
    report_path = output_dir / "formulation_audit_report.md"

    with open(json_path, "w") as fh:
        json.dump(audit, fh, indent=2)
    _write_csv(csv_path, [oracle, blind, robust], alpha=alpha)
    _write_report(
        report_path,
        env_config=env_config,
        cfg=cfg,
        oracle=oracle,
        blind=blind,
        robust=robust,
        comparison=comparison,
        milp=milp,
        alpha=alpha,
    )

    print("-" * 64)
    print(f"  formulation_audit.json : {json_path}")
    print(f"  formulation_audit.csv  : {csv_path}")
    print(f"  report                 : {report_path}")
    print("  -> OK (Stage-6 PR4 formulation audit completed)")


if __name__ == "__main__":
    main()
