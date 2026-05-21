"""Stage 4B-5C PR5C-2B: real bootstrap scenario generation + metrics.

CLI driver that runs ``ConditionalBootstrapSampler`` against the real
12 km eVTOL OD tensor and writes a CANDIDATE Stage-4 fallback file:

    data/synthetic/od_samples_agg_bootstrap.npy   [N_omega, |Z|, |Z|] int32

The FROZEN Stage-5 input ``data/synthetic/od_samples_agg.npy`` is NOT
touched here. The PR5C-3 unified scenario-generation comparison
decides what ultimately lands at that path, with a user-confirmed
write. This script refuses to write to the frozen path defensively.

Acceptance bands (PR5C-2B)
--------------------------

Comparisons against real_test are done at per-day-equivalent scale
(``bootstrap[i] / n_days_per_scenario`` vs ``real_test / n_test_days``)
because the bootstrap aggregates ``n_days_per_scenario`` calendar days
per scenario while ``real_test`` covers only the test window. The raw
totals are also reported for transparency.

    pass : gen_nonzero_ratio_mean      in [0.7x, 1.5x] of real_test_nonzero_ratio
        AND per_day_total_mass_ratio_mean in [0.7, 1.5]
        AND per_day_row_sum_ks_mean    <= 0.3
        AND per_day_col_sum_ks_mean    <= 0.3
        AND top20_pair_overlap_mean    >= 14
    mild : strictly better than the failed-diffusion 4B-5B PR5B-3b-3
           diagnostics (row_sum_ks_stat=1.000, col_sum_ks_stat=1.000,
           gen_nonzero_ratio = 143.0x real, total_mass_ratio = 181x
           real_per_day) but at least one "pass" gate not met
    fail : metrics close to or worse than the failed-diffusion bands

Usage
-----

    python -m experiments.run_stage4_bootstrap \\
        --config configs/diffusion_12km.yaml \\
        --output data/synthetic/od_samples_agg_bootstrap.npy \\
        --results-dir results/stage4/bootstrap \\
        --n-omega 64 \\
        --seed 42 \\
        --n-days-per-scenario 11 \\
        --slots-per-day 48
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.bootstrap_od import (
    ConditionalBootstrapSampler,
    compute_bootstrap_summary,
)
from src.utils.metrics_dist import ks_stat_1d

REPO = Path(__file__).resolve().parents[1]
FROZEN_AGG_PATH = REPO / "data" / "synthetic" / "od_samples_agg.npy"
FROZEN_4D_PATH = REPO / "data" / "synthetic" / "od_samples.npy"

# Failed-diffusion baseline (PR5B-3b-3, docs/decisions.md 2026-05-20).
# Used as the "mild" floor: bootstrap must beat ALL of these to count
# as at least "mild" -- otherwise it sits in "fail" alongside the
# failed diffusion runs.
DIFFUSION_FAILED_BASELINE = {
    "row_sum_ks_stat": 1.000,
    "col_sum_ks_stat": 1.000,
    "gen_nonzero_ratio_x_real": 143.0,
    "total_mass_ratio_x_real_per_day": 181.0,
}


# --- helpers --------------------------------------------------------------


def _resolve(path_str: str | Path) -> Path:
    """Resolve a yaml path relative to repo root if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _shannon_entropy(arr: np.ndarray) -> float:
    """Shannon entropy (nats) of a non-negative array, normalized to sum=1."""
    flat = np.asarray(arr, dtype=np.float64).ravel()
    total = float(flat.sum())
    if total <= 0.0:
        return float("nan")
    p = flat[flat > 0] / total
    return float(-(p * np.log(p)).sum())


def _top_n_flat_indices(arr: np.ndarray, n: int) -> set[int]:
    """Return the flat indices of the top-``n`` cells of ``arr`` as a set.

    Ties at the boundary are broken by ``np.argpartition`` semantics.
    OD cells are well-separated in practice (heavy-tailed counts) so
    ties on the 20th/50th value are not a real concern.
    """
    flat = np.asarray(arr).ravel()
    if n >= flat.size:
        return set(range(flat.size))
    idx = np.argpartition(flat, -n)[-n:]
    return set(int(i) for i in idx)


def _compute_metrics(
    bootstrap: np.ndarray,
    real_test: np.ndarray,
    real_val: np.ndarray,
    n_days_per_scenario: int,
    n_test_days: int,
    n_val_days: int,
) -> dict[str, Any]:
    """Compute the PR5C-2B metrics dictionary on bootstrap vs real_test.

    The KS / total-mass comparisons are done at per-day-equivalent scale
    because the bootstrap aggregates ``n_days_per_scenario`` days per
    scenario while ``real_test`` covers ``n_test_days`` (= 1 for the
    12 km Stage-4 split). Raw totals are also included.
    """
    n_omega = bootstrap.shape[0]
    z = bootstrap.shape[1]

    # Per-day-equivalent floats (NOT used for the .npy output, which
    # stays int32 raw 11-day-aggregate counts -- this is purely the
    # apples-to-apples comparison space for KS/total_mass).
    per_day_bs = bootstrap.astype(np.float64) / float(n_days_per_scenario)
    real_per_day_test = real_test.astype(np.float64) / float(n_test_days)
    real_per_day_val = real_val.astype(np.float64) / float(n_val_days)

    # --- real_test reference scalars -------------------------------------
    real_test_total = float(real_test.sum())
    real_per_day_test_total = float(real_per_day_test.sum())
    real_test_nz_ratio = float((real_test != 0).mean())
    real_test_max = int(real_test.max())
    real_test_mean = float(real_test.mean())
    real_test_entropy = _shannon_entropy(real_test)
    real_val_total = float(real_val.sum())
    real_val_nz_ratio = float((real_val != 0).mean())

    real_test_top20 = _top_n_flat_indices(real_test, 20)
    real_test_top50 = _top_n_flat_indices(real_test, 50)

    real_test_row = real_per_day_test.sum(axis=-1).ravel()
    real_test_col = real_per_day_test.sum(axis=-2).ravel()

    # --- per-scenario arrays --------------------------------------------
    raw_total = np.empty(n_omega, dtype=np.float64)
    nz_ratio = np.empty(n_omega, dtype=np.float64)
    gen_max = np.empty(n_omega, dtype=np.int64)
    gen_mean = np.empty(n_omega, dtype=np.float64)
    per_day_row_ks = np.empty(n_omega, dtype=np.float64)
    per_day_col_ks = np.empty(n_omega, dtype=np.float64)
    raw_row_ks = np.empty(n_omega, dtype=np.float64)
    raw_col_ks = np.empty(n_omega, dtype=np.float64)
    top20_overlap = np.empty(n_omega, dtype=np.int64)
    top20_vs_top50 = np.empty(n_omega, dtype=np.int64)
    entropy = np.empty(n_omega, dtype=np.float64)

    real_row_raw = real_test.sum(axis=-1).ravel().astype(np.float64)
    real_col_raw = real_test.sum(axis=-2).ravel().astype(np.float64)

    for i in range(n_omega):
        bs_i = bootstrap[i]
        pd_i = per_day_bs[i]
        raw_total[i] = float(bs_i.sum())
        nz_ratio[i] = float((bs_i != 0).mean())
        gen_max[i] = int(bs_i.max())
        gen_mean[i] = float(bs_i.mean())
        per_day_row_ks[i] = ks_stat_1d(pd_i.sum(axis=-1).ravel(), real_test_row)
        per_day_col_ks[i] = ks_stat_1d(pd_i.sum(axis=-2).ravel(), real_test_col)
        raw_row_ks[i] = ks_stat_1d(bs_i.sum(axis=-1).ravel(), real_row_raw)
        raw_col_ks[i] = ks_stat_1d(bs_i.sum(axis=-2).ravel(), real_col_raw)
        bs_top20 = _top_n_flat_indices(bs_i, 20)
        top20_overlap[i] = len(bs_top20 & real_test_top20)
        top20_vs_top50[i] = len(bs_top20 & real_test_top50)
        entropy[i] = _shannon_entropy(bs_i)

    per_day_total_mass = raw_total / float(n_days_per_scenario)
    total_mass_ratio_raw = raw_total / max(real_test_total, 1.0)
    per_day_total_mass_ratio = per_day_total_mass / max(real_per_day_test_total, 1.0)

    def mean_std(arr: np.ndarray) -> dict[str, float]:
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    metrics: dict[str, Any] = {
        "sample_shape": list(bootstrap.shape),
        "dtype": str(bootstrap.dtype),
        "nonnegative": bool(bootstrap.min() >= 0),
        "n_omega": int(n_omega),
        "z": int(z),
        "n_days_per_scenario": int(n_days_per_scenario),
        "n_test_days": int(n_test_days),
        "n_val_days": int(n_val_days),
        # --- nonzero ratios ---------------------------------------------
        "gen_nonzero_ratio": mean_std(nz_ratio),
        "real_test_nonzero_ratio": real_test_nz_ratio,
        "real_val_nonzero_ratio": real_val_nz_ratio,
        "gen_nonzero_ratio_x_real_test": {
            "mean": float(nz_ratio.mean() / max(real_test_nz_ratio, 1e-12)),
            "std": float(nz_ratio.std() / max(real_test_nz_ratio, 1e-12)),
        },
        # --- mass: raw 11-day-aggregate scale ---------------------------
        "total_mass": mean_std(raw_total),
        "real_test_total_mass": real_test_total,
        "real_val_total_mass": real_val_total,
        "total_mass_ratio_raw": mean_std(total_mass_ratio_raw),
        # --- mass: per-day-equivalent scale (primary) -------------------
        "per_day_total_mass": mean_std(per_day_total_mass),
        "real_per_day_test_total_mass": real_per_day_test_total,
        "per_day_total_mass_ratio": mean_std(per_day_total_mass_ratio),
        # --- KS: per-day-equivalent (primary, comparable to real_test) --
        "row_sum_ks_stat": mean_std(per_day_row_ks),
        "col_sum_ks_stat": mean_std(per_day_col_ks),
        # --- KS: raw scale (kept for transparency, expected near 1.0) ---
        "row_sum_ks_stat_raw": mean_std(raw_row_ks),
        "col_sum_ks_stat_raw": mean_std(raw_col_ks),
        # --- spread, magnitudes -----------------------------------------
        "gen_max": {"mean": float(gen_max.mean()), "max": int(gen_max.max())},
        "gen_mean": mean_std(gen_mean),
        "real_test_max": real_test_max,
        "real_test_mean": real_test_mean,
        # --- ranking-based, scale-invariant -----------------------------
        "top20_pair_overlap": mean_std(top20_overlap.astype(np.float64)),
        "top20_pair_overlap_against_top50": mean_std(
            top20_vs_top50.astype(np.float64)
        ),
        # --- entropy (nats) ---------------------------------------------
        "entropy": mean_std(entropy),
        "real_test_entropy": real_test_entropy,
    }
    return metrics, {
        "raw_total": raw_total,
        "per_day_total_mass": per_day_total_mass,
        "nz_ratio": nz_ratio,
        "gen_max": gen_max,
        "gen_mean": gen_mean,
        "per_day_row_ks": per_day_row_ks,
        "per_day_col_ks": per_day_col_ks,
        "raw_row_ks": raw_row_ks,
        "raw_col_ks": raw_col_ks,
        "top20_overlap": top20_overlap,
        "top20_vs_top50": top20_vs_top50,
        "entropy": entropy,
        "total_mass_ratio_raw": total_mass_ratio_raw,
        "per_day_total_mass_ratio": per_day_total_mass_ratio,
    }


def _judge(metrics: dict[str, Any]) -> str:
    """pass / mild / fail per PR5C-2B bands."""
    gen_nz_ratio_x = metrics["gen_nonzero_ratio_x_real_test"]["mean"]
    per_day_mass_ratio = metrics["per_day_total_mass_ratio"]["mean"]
    row_ks = metrics["row_sum_ks_stat"]["mean"]
    col_ks = metrics["col_sum_ks_stat"]["mean"]
    top20 = metrics["top20_pair_overlap"]["mean"]

    pass_gates = [
        0.7 <= gen_nz_ratio_x <= 1.5,
        0.7 <= per_day_mass_ratio <= 1.5,
        row_ks <= 0.3,
        col_ks <= 0.3,
        top20 >= 14.0,
    ]
    if all(pass_gates):
        return "pass"

    # The "mild" floor: bootstrap beats the failed-diffusion baseline on
    # ALL four headline axes.
    baseline = DIFFUSION_FAILED_BASELINE
    mild_gates = [
        row_ks < baseline["row_sum_ks_stat"],
        col_ks < baseline["col_sum_ks_stat"],
        gen_nz_ratio_x < baseline["gen_nonzero_ratio_x_real"],
        per_day_mass_ratio < baseline["total_mass_ratio_x_real_per_day"],
    ]
    if all(mild_gates):
        return "mild"
    return "fail"


def _plot_marginals(
    real_test: np.ndarray,
    bootstrap: np.ndarray,
    n_days_per_scenario: int,
    n_test_days: int,
    out_path: Path,
) -> None:
    """Save row-sum / col-sum histogram comparison at per-day-equivalent scale."""
    real_per_day = real_test.astype(np.float64) / float(n_test_days)
    bs_per_day_mean = bootstrap.astype(np.float64).mean(axis=0) / float(
        n_days_per_scenario
    )

    real_row = real_per_day.sum(axis=-1).ravel()
    real_col = real_per_day.sum(axis=-2).ravel()
    bs_row = bs_per_day_mean.sum(axis=-1).ravel()
    bs_col = bs_per_day_mean.sum(axis=-2).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (r, g, title) in zip(
        axes,
        [
            (real_row, bs_row, "row sums (per-day-equiv)"),
            (real_col, bs_col, "col sums (per-day-equiv)"),
        ],
    ):
        combined = np.concatenate([r, g])
        lo, hi = float(combined.min()), float(combined.max())
        if hi - lo < 1e-9:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 50)
        ax.hist(r, bins=bins, alpha=0.5, density=True, label="real_test", color="C0")
        ax.hist(
            g, bins=bins, alpha=0.5, density=True, label="bootstrap mean", color="C3"
        )
        ax.set_title(title)
        ax.set_xlabel("count per zone (per-day-equivalent)")
        ax.set_ylabel("density")
        ax.legend()
    fig.suptitle(
        "PR5C-2B bootstrap marginals: real_test vs bootstrap (per-day-equivalent)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_csv(per_scenario: dict[str, np.ndarray], out_path: Path) -> None:
    """One row per scenario; columns = the per-scenario metric arrays."""
    n = next(iter(per_scenario.values())).shape[0]
    keys = list(per_scenario.keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scenario_idx"] + keys)
        for i in range(n):
            row: list[Any] = [i]
            for k in keys:
                v = per_scenario[k][i]
                row.append(int(v) if np.issubdtype(per_scenario[k].dtype, np.integer) else float(v))
            writer.writerow(row)


def _write_summary(
    metrics: dict[str, Any],
    judgment: str,
    cfg_path: Path,
    output_path: Path,
    used_slots: set[int],
    train_slots: list[int],
    val_slots: list[int],
    test_slots: list[int],
    sampler_summary: dict[str, float],
    out_path: Path,
) -> None:
    """Human-readable summary.md for the user to skim."""
    used_count = len(used_slots)
    train_count = len(train_slots)
    leak_val = sorted(used_slots & set(val_slots))
    leak_test = sorted(used_slots & set(test_slots))
    leaked = bool(leak_val or leak_test)

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO))
        except ValueError:
            return str(p)

    pass_x = metrics["gen_nonzero_ratio_x_real_test"]["mean"]
    pd_mass = metrics["per_day_total_mass_ratio"]["mean"]
    row_ks = metrics["row_sum_ks_stat"]["mean"]
    col_ks = metrics["col_sum_ks_stat"]["mean"]
    top20 = metrics["top20_pair_overlap"]["mean"]
    top50 = metrics["top20_pair_overlap_against_top50"]["mean"]
    lines = [
        "# Stage 4B-5C PR5C-2B bootstrap report",
        "",
        f"- config: `{_rel(cfg_path)}`",
        f"- output npy: `{_rel(output_path)}`",
        f"- shape: {metrics['sample_shape']}  dtype: {metrics['dtype']}  "
        f"nonneg: {metrics['nonnegative']}",
        f"- n_omega: {metrics['n_omega']}  "
        f"n_days_per_scenario: {metrics['n_days_per_scenario']}  "
        f"n_test_days: {metrics['n_test_days']}",
        "",
        "## Leak check",
        f"- train_slots: {train_count}",
        f"- used_slots:  {used_count}",
        f"- used_slots ⊆ train_slots: {set(used_slots).issubset(set(train_slots))}",
        f"- leaked into val: {len(leak_val)}",
        f"- leaked into test: {len(leak_test)}",
        f"- any leak: {leaked}",
        "",
        "## Acceptance verdict",
        f"- **{judgment.upper()}**",
        "",
        "| gate | value | band |",
        "|---|---|---|",
        f"| gen_nonzero_ratio (x real_test) | {pass_x:.3f} | pass: [0.7, 1.5] |",
        f"| per_day_total_mass_ratio | {pd_mass:.3f} | pass: [0.7, 1.5] |",
        f"| per_day row_sum_ks (mean) | {row_ks:.3f} | pass: ≤ 0.3 |",
        f"| per_day col_sum_ks (mean) | {col_ks:.3f} | pass: ≤ 0.3 |",
        f"| top20 overlap (mean) | {top20:.2f} | pass: ≥ 14 |",
        f"| top20 vs top50 overlap (mean) | {top50:.2f} | (diagnostic) |",
        "",
        "## Headline numbers",
        f"- gen_nonzero_ratio (mean): {metrics['gen_nonzero_ratio']['mean']:.6f}",
        f"- real_test_nonzero_ratio: {metrics['real_test_nonzero_ratio']:.6f}",
        f"- per_day_total_mass (mean): {metrics['per_day_total_mass']['mean']:.3f}",
        f"- real_per_day_test_total_mass: {metrics['real_per_day_test_total_mass']:.3f}",
        f"- raw total_mass_ratio (mean): {metrics['total_mass_ratio_raw']['mean']:.3f} "
        "(raw 11-day aggregate vs 1-day test reference)",
        f"- gen_max (mean / max): {metrics['gen_max']['mean']:.1f} / "
        f"{metrics['gen_max']['max']}  real_test_max: {metrics['real_test_max']}",
        f"- entropy mean: {metrics['entropy']['mean']:.3f}  "
        f"real_test_entropy: {metrics['real_test_entropy']:.3f}",
        "",
        "## Failed-diffusion baseline (PR5B-3b-3, for reference)",
        f"- diffusion row_sum_ks_stat: {DIFFUSION_FAILED_BASELINE['row_sum_ks_stat']:.3f}",
        f"- diffusion col_sum_ks_stat: {DIFFUSION_FAILED_BASELINE['col_sum_ks_stat']:.3f}",
        f"- diffusion gen_nonzero_ratio_x_real: "
        f"{DIFFUSION_FAILED_BASELINE['gen_nonzero_ratio_x_real']:.1f}",
        f"- diffusion total_mass_ratio_x_real_per_day: "
        f"{DIFFUSION_FAILED_BASELINE['total_mass_ratio_x_real_per_day']:.1f}",
        "",
        "## Sampler summary",
    ]
    for k, v in sampler_summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Notes on scale")
    lines.append(
        "- Bootstrap scenarios aggregate `n_days_per_scenario` calendar days; "
        "`real_test` covers `n_test_days` calendar days. KS, row/col sums, and "
        "total_mass are compared at per-day-equivalent scale (bootstrap / "
        "n_days_per_scenario vs real_test / n_test_days). "
        "nonzero_ratio is NOT scale-invariant under aggregation -- the "
        "bootstrap nonzero_ratio is naturally higher because more events "
        "accumulate across days; the `x real_test` ratio is reported for "
        "transparency."
    )
    out_path.write_text("\n".join(lines))


# --- main -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "diffusion_12km.yaml",
        help="diffusion YAML providing input.od_path / input.meta_path / data.split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "data" / "synthetic" / "od_samples_agg_bootstrap.npy",
        help="candidate bootstrap output path (NOT the frozen Stage-5 input)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO / "results" / "stage4" / "bootstrap",
    )
    parser.add_argument("--n-omega", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-days-per-scenario", type=int, default=11)
    parser.add_argument("--slots-per-day", type=int, default=48)
    args = parser.parse_args()

    # --- guard: refuse to overwrite frozen Stage-5 input ----------------
    out_resolved = args.output.resolve()
    if out_resolved == FROZEN_AGG_PATH.resolve():
        raise SystemExit(
            f"refusing to write to frozen Stage-5 path {FROZEN_AGG_PATH}; "
            "PR5C-3 decides what goes there. Use a different --output."
        )
    if out_resolved == FROZEN_4D_PATH.resolve():
        raise SystemExit(
            f"refusing to write to diffusion 4-D path {FROZEN_4D_PATH}; "
            "this is the diffusion sampler's output, not the bootstrap's."
        )

    # --- load config ----------------------------------------------------
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    od_path = _resolve(cfg["input"]["od_path"])
    meta_path = _resolve(cfg["input"]["meta_path"])
    split_cfg = cfg["data"]["split"]

    with open(meta_path) as fh:
        meta = json.load(fh)
    n_slots = int(meta["T"])
    n_zones = int(meta["n_zones"])
    bin_min = int(meta.get("time_bin_min", 30))

    train_lo, train_hi = split_cfg["train_days"]
    val_lo, val_hi = split_cfg["val_days"]
    test_lo, test_hi = split_cfg["test_days"]
    spd = args.slots_per_day
    expected_spd = 24 * 60 // bin_min
    if spd != expected_spd:
        raise SystemExit(
            f"--slots-per-day={spd} does not match meta time_bin_min={bin_min} "
            f"(expected {expected_spd})"
        )

    train_slots = list(range(train_lo * spd, min(train_hi * spd, n_slots)))
    val_slots = list(range(val_lo * spd, min(val_hi * spd, n_slots)))
    test_slots = list(range(test_lo * spd, min(test_hi * spd, n_slots)))
    n_test_days = max(test_hi - test_lo, 1)
    n_val_days = max(val_hi - val_lo, 1)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("STAGE 4B-5C PR5C-2B  --  conditional bootstrap (real 12 km)")
    print("=" * 68)
    print(f"  config         : {args.config}")
    print(f"  od_path        : {od_path}")
    print(f"  meta_path      : {meta_path}")
    print(f"  T={n_slots}  |Z|={n_zones}  slots_per_day={spd}")
    print(f"  train_slots    : [{train_slots[0]}, {train_slots[-1]}]  "
          f"(n={len(train_slots)}, {(train_hi-train_lo)} days)")
    print(f"  val_slots      : [{val_slots[0]}, {val_slots[-1]}]  "
          f"(n={len(val_slots)}, {n_val_days} days)")
    print(f"  test_slots     : [{test_slots[0]}, {test_slots[-1]}]  "
          f"(n={len(test_slots)}, {n_test_days} days)")
    print(f"  output (cand.) : {args.output}")
    print(f"  results_dir    : {args.results_dir}")
    print(f"  n_omega        : {args.n_omega}")
    print(f"  seed           : {args.seed}")
    print(f"  n_days_per_sc  : {args.n_days_per_scenario}")

    # --- load OD (mmap) -------------------------------------------------
    od = np.load(od_path, mmap_mode="r")
    if od.shape != (n_slots, n_zones, n_zones):
        raise SystemExit(
            f"OD shape mismatch: {od.shape} vs meta ({n_slots}, {n_zones}, {n_zones})"
        )
    print(f"\n  od.shape       : {od.shape}  dtype={od.dtype}")

    # --- sampler --------------------------------------------------------
    sampler = ConditionalBootstrapSampler(
        od,
        train_slots,
        slots_per_day=spd,
        n_days_per_scenario=args.n_days_per_scenario,
        n_omega=args.n_omega,
        seed=args.seed,
        mode="day_block",
    )
    print(f"\n  day_blocks     : {len(sampler.day_blocks)}  "
          f"(each {spd} slots, all in train_slots)")
    print("  drawing scenarios ...")
    samples = sampler.sample()
    print(f"  samples.shape  : {samples.shape}  dtype={samples.dtype}  "
          f"min={int(samples.min())}  max={int(samples.max())}  "
          f"total={int(samples.sum())}")

    assert sampler.used_slots is not None
    used_set = sampler.used_slots
    print(f"  used_slots     : n={len(used_set)}  "
          f"subset_of_train_slots={set(used_set).issubset(set(train_slots))}")
    print(f"  leaked_to_val  : {len(used_set & set(val_slots))}")
    print(f"  leaked_to_test : {len(used_set & set(test_slots))}")

    # --- save .npy candidate -------------------------------------------
    np.save(args.output, samples)
    saved_bytes = args.output.stat().st_size
    print(f"\n  saved          : {args.output}  ({saved_bytes / 1e6:.1f} MB)")

    # --- real reference aggregates -------------------------------------
    real_test = np.zeros((n_zones, n_zones), dtype=np.int64)
    for s in test_slots:
        real_test += np.asarray(od[s], dtype=np.int64)
    real_val = np.zeros((n_zones, n_zones), dtype=np.int64)
    for s in val_slots:
        real_val += np.asarray(od[s], dtype=np.int64)

    # --- metrics --------------------------------------------------------
    metrics, per_scenario = _compute_metrics(
        samples,
        real_test,
        real_val,
        n_days_per_scenario=args.n_days_per_scenario,
        n_test_days=n_test_days,
        n_val_days=n_val_days,
    )
    sampler_summary = compute_bootstrap_summary(samples)
    metrics["used_slots_count"] = len(used_set)
    metrics["used_slots_subset_of_train_slots"] = bool(
        set(used_set).issubset(set(train_slots))
    )
    metrics["sampler_summary"] = sampler_summary
    metrics["config"] = str(args.config)
    metrics["seed"] = int(args.seed)
    metrics["output_npy"] = str(args.output)

    judgment = _judge(metrics)
    metrics["acceptance"] = judgment
    print(f"\n  acceptance verdict: {judgment.upper()}")

    # --- write artifacts ------------------------------------------------
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )
    _write_csv(per_scenario, args.results_dir / "metrics.csv")
    _plot_marginals(
        real_test,
        samples,
        n_days_per_scenario=args.n_days_per_scenario,
        n_test_days=n_test_days,
        out_path=args.results_dir / "marginal_match.png",
    )
    _write_summary(
        metrics,
        judgment,
        args.config,
        args.output,
        used_set,
        train_slots,
        val_slots,
        test_slots,
        sampler_summary,
        args.results_dir / "summary.md",
    )

    print(f"  metrics.json   : {args.results_dir / 'metrics.json'}")
    print(f"  metrics.csv    : {args.results_dir / 'metrics.csv'}")
    print(f"  marginal_match : {args.results_dir / 'marginal_match.png'}")
    print(f"  summary.md     : {args.results_dir / 'summary.md'}")
    print("\n" + "=" * 68)
    print(f"PR5C-2B bootstrap run complete: {judgment.upper()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
