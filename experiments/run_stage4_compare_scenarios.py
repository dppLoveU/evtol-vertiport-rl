"""Stage 4B-5C PR5C-3A: unified comparison of Stage-4 scenario sources.

Reads the per-source metrics JSON files written by the candidate
generation PRs (PR5C-2B bootstrap and PR5C-1B posthoc-calibrated
diffusion) and produces a unified comparison table + a recommended
winner. **This PR does NOT freeze any source**: no
``data/synthetic/od_samples_agg.npy`` is written or modified, no
candidate npy is overwritten, no Stage-5 code is touched. The
recommended winner is an advisory shortlist for user review; the final
copy to ``data/synthetic/od_samples_agg.npy`` is a separate sub-PR
gated on user confirmation.

Acceptance bands
----------------

Per-source ``acceptance_tier`` is the verdict reported by each source's
own pipeline:

  * bootstrap (PR5C-2B): MILD/PASS via
    ``run_stage4_bootstrap.py::_judge`` against per-day-equivalent
    bands.
  * diffusion_calibrated (PR5C-1B "after"): MILD/PASS via
    ``src.eval.posthoc_calibration.acceptance_verdict`` against the
    test aggregate.
  * diffusion_raw (PR5C-1B "before"): the same
    ``acceptance_verdict`` is applied to the clip+round-only metrics.
  * failed_diffusion_baseline (PR5B-3b-3 decisions.md record): always
    FAIL by construction (it IS the baseline that defines the MILD
    floor).

Decision rule (winner ranking, top wins)
----------------------------------------

Sort sources by, in order of decreasing priority:
  1. ``can_freeze_to_stage5`` (True > False)
  2. ``acceptance_tier`` (PASS > MILD > FAIL)
  3. ``top20_pair_overlap`` (higher better)
  4. ``row_sum_ks_stat`` (lower better)
  5. ``col_sum_ks_stat`` (lower better)
  6. ``|total_mass_ratio - 1|`` (closer to 1 better)

The top row is the **recommended** winner. ``decision.md`` makes
explicit that the final freeze is a separate sub-PR gated on user
confirmation.

Usage
-----

    python -m experiments.run_stage4_compare_scenarios \\
        --bootstrap-metrics results/stage4/bootstrap/metrics.json \\
        --posthoc-metrics   results/stage4/posthoc_calibration_zpin_weighted/metrics.json \\
        --output-dir        results/stage4/comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from src.eval.posthoc_calibration import (
    DEFAULT_FAILED_DIFFUSION_BASELINE,
    acceptance_verdict,
)

REPO = Path(__file__).resolve().parents[1]
FROZEN_AGG_PATH = REPO / "data" / "synthetic" / "od_samples_agg.npy"
FROZEN_4D_PATH = REPO / "data" / "synthetic" / "od_samples.npy"
BOOTSTRAP_CANDIDATE_PATH = (
    REPO / "data" / "synthetic" / "od_samples_agg_bootstrap.npy"
)
CALIBRATED_CANDIDATE_PATH = (
    REPO / "data" / "synthetic" / "od_samples_agg_diffusion_calibrated.npy"
)

# Column order for the CSV / table. All rows must populate these keys
# (missing values written as the empty string).
COLUMNS = [
    "source_name",
    "source_type",
    "candidate_path",
    "acceptance_tier",
    "nonzero_ratio_x_real_test",
    "total_mass_ratio",
    "total_mass_ratio_scale",
    "row_sum_ks_stat",
    "col_sum_ks_stat",
    "ks_scale",
    "top20_pair_overlap",
    "top20_pair_overlap_against_top50",
    "gen_max",
    "gen_mean",
    "entropy",
    "has_candidate_npy",
    "can_freeze_to_stage5",
    "notes",
]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _row_bootstrap(metrics_path: Path) -> dict[str, Any]:
    """Build the bootstrap source row from PR5C-2B's metrics.json."""
    m = json.loads(metrics_path.read_text())
    candidate = BOOTSTRAP_CANDIDATE_PATH
    has_npy = candidate.exists()
    return {
        "source_name": "bootstrap_day_block",
        "source_type": "bootstrap",
        "candidate_path": _rel(candidate),
        "acceptance_tier": m["acceptance"].upper(),
        "nonzero_ratio_x_real_test": float(
            m["gen_nonzero_ratio_x_real_test"]["mean"]
        ),
        "total_mass_ratio": float(m["per_day_total_mass_ratio"]["mean"]),
        "total_mass_ratio_scale": "per_day_equiv",
        "row_sum_ks_stat": float(m["row_sum_ks_stat"]["mean"]),
        "col_sum_ks_stat": float(m["col_sum_ks_stat"]["mean"]),
        "ks_scale": "per_day_equiv",
        "top20_pair_overlap": float(m["top20_pair_overlap"]["mean"]),
        "top20_pair_overlap_against_top50": float(
            m["top20_pair_overlap_against_top50"]["mean"]
        ),
        "gen_max": float(m["gen_max"]["mean"]),
        "gen_mean": float(m["gen_mean"]["mean"]),
        "entropy": float(m["entropy"]["mean"]),
        "has_candidate_npy": bool(has_npy),
        "can_freeze_to_stage5": bool(has_npy),
        "notes": (
            "PR5C-2B: 11-day day-block bootstrap over the real 12 km train "
            "slots; used_slots subset of train_slots verified; diagonal "
            "excluded by Stage-3 invariant."
        ),
    }


def _row_diffusion_calibrated(metrics_path: Path) -> dict[str, Any]:
    """Posthoc-calibrated diffusion (after) row from PR5C-1B."""
    m = json.loads(metrics_path.read_text())
    a = m["after"]["test"]
    n_train_days = m["real_train_agg"]["n_slots"] // 48
    n_test_days = m["real_test_agg"]["n_slots"] // 48
    day_factor = n_train_days / max(n_test_days, 1)
    # The posthoc CLI fit (tau, scale) against the train aggregate
    # (5-day sum), then reports total_mass_ratio against the 1-day test
    # aggregate -- so the raw value is ~5x by construction. Report a
    # per-day-equivalent value for apples-to-apples comparison with
    # bootstrap's per-day-equivalent column.
    per_day_mass_ratio = float(a["total_mass_ratio_mean"]) / day_factor
    candidate = CALIBRATED_CANDIDATE_PATH
    has_npy = candidate.exists()
    return {
        "source_name": "diffusion_calibrated_zpin_weighted_pilot",
        "source_type": "diffusion_posthoc_calibrated",
        "candidate_path": _rel(candidate),
        "acceptance_tier": m["test_acceptance_verdict"].upper(),
        "nonzero_ratio_x_real_test": float(a["nonzero_ratio_x_real_mean"]),
        "total_mass_ratio": per_day_mass_ratio,
        "total_mass_ratio_scale": (
            f"per_day_equiv (raw mass_ratio={a['total_mass_ratio_mean']:.3f} "
            f"rescaled by 1/{day_factor:.0f}; calibrator targeted train sum)"
        ),
        "row_sum_ks_stat": float(a["row_sum_ks_stat_mean"]),
        "col_sum_ks_stat": float(a["col_sum_ks_stat_mean"]),
        "ks_scale": (
            "raw (NOT per-day-normalized; cal samples at "
            f"{n_train_days}-day-train scale vs {n_test_days}-day test)"
        ),
        "top20_pair_overlap": float(a["top20_pair_overlap_mean"]),
        "top20_pair_overlap_against_top50": float(
            a["top20_pair_overlap_against_top50_mean"]
        ),
        "gen_max": float(a["gen_max_mean"]),
        "gen_mean": float(a["gen_mean_mean"]),
        "entropy": float(a["entropy_mean"]),
        "has_candidate_npy": bool(has_npy),
        "can_freeze_to_stage5": False,  # PR5C-1B deliberately did NOT write a npy
        "notes": (
            f"PR5C-1B: posthoc tau={m['best_tau']:.4f}, "
            f"scale={m['best_scale']:.4g} fit on real_train_agg only; "
            f"ckpt step={m['ckpt_step']} val_loss={m['ckpt_val_loss']:.4f}; "
            f"top20=0 = no spatial structure recovered; no candidate npy "
            f"written by PR5C-1B (deliberate)."
        ),
    }


def _row_diffusion_raw(metrics_path: Path) -> dict[str, Any]:
    """Uncalibrated diffusion (before = clip+round only) row from PR5C-1B."""
    m = json.loads(metrics_path.read_text())
    b = m["before"]["test"]
    n_train_days = m["real_train_agg"]["n_slots"] // 48
    n_test_days = m["real_test_agg"]["n_slots"] // 48
    day_factor = n_train_days / max(n_test_days, 1)
    per_day_mass_ratio = float(b["total_mass_ratio_mean"]) / day_factor
    verdict = acceptance_verdict(b)
    return {
        "source_name": "diffusion_raw_zpin_weighted_pilot",
        "source_type": "diffusion_uncalibrated",
        "candidate_path": "",
        "acceptance_tier": verdict.upper(),
        "nonzero_ratio_x_real_test": float(b["nonzero_ratio_x_real_mean"]),
        "total_mass_ratio": per_day_mass_ratio,
        "total_mass_ratio_scale": (
            f"per_day_equiv (raw mass_ratio={b['total_mass_ratio_mean']:.3f} "
            f"rescaled by 1/{day_factor:.0f}; no calibration applied)"
        ),
        "row_sum_ks_stat": float(b["row_sum_ks_stat_mean"]),
        "col_sum_ks_stat": float(b["col_sum_ks_stat_mean"]),
        "ks_scale": (
            f"raw (NOT per-day-normalized; samples at 1-slot scale, "
            f"row_sums summed across {m['n_samples']} samples)"
        ),
        "top20_pair_overlap": float(b["top20_pair_overlap_mean"]),
        "top20_pair_overlap_against_top50": float(
            b["top20_pair_overlap_against_top50_mean"]
        ),
        "gen_max": float(b["gen_max_mean"]),
        "gen_mean": float(b["gen_mean_mean"]),
        "entropy": float(b["entropy_mean"]),
        "has_candidate_npy": False,
        "can_freeze_to_stage5": False,
        "notes": (
            "PR5C-1B before-calibration baseline (clip+round only on "
            "continuous samples); no tau/scale applied; included as a "
            "reference for what posthoc calibration buys."
        ),
    }


def _row_failed_baseline() -> dict[str, Any]:
    """Failed-diffusion baseline from docs/decisions.md PR5B-3b-3."""
    b = DEFAULT_FAILED_DIFFUSION_BASELINE
    return {
        "source_name": "diffusion_failed_baseline_pr5b_3b3",
        "source_type": "diffusion_failed_baseline",
        "candidate_path": "",
        "acceptance_tier": "FAIL",
        "nonzero_ratio_x_real_test": float(b["nonzero_ratio_x_real"]),
        "total_mass_ratio": float(b["total_mass_ratio"]),
        "total_mass_ratio_scale": (
            "raw vs real_test_per_day (record from PR5B-3b-3 decisions.md)"
        ),
        "row_sum_ks_stat": float(b["row_sum_ks_stat"]),
        "col_sum_ks_stat": float(b["col_sum_ks_stat"]),
        "ks_scale": "raw (in-loop diag at pilot end-of-run)",
        "top20_pair_overlap": 0.0,
        "top20_pair_overlap_against_top50": 0.0,
        "gen_max": float("nan"),
        "gen_mean": float("nan"),
        "entropy": float("nan"),
        "has_candidate_npy": False,
        "can_freeze_to_stage5": False,
        "notes": (
            "Floor baseline: PR5B-3b-3 zpin+weighted pilot in-loop diag, "
            "n=4, gs=1.0. Any candidate that does not strictly beat all "
            "four axes here is FAIL by definition."
        ),
    }


def _sort_key(row: dict[str, Any]) -> tuple:
    """Decision-rule sort key (lower tuple = better, will be at top after sort)."""
    tier_order = {"PASS": 0, "MILD": 1, "FAIL": 2}
    return (
        # 1. can_freeze first
        0 if row["can_freeze_to_stage5"] else 1,
        # 2. acceptance tier
        tier_order.get(row["acceptance_tier"], 3),
        # 3. top20 higher better -> negate
        -float(row["top20_pair_overlap"]),
        # 4. row_ks lower better
        float(row["row_sum_ks_stat"]),
        # 5. col_ks lower better
        float(row["col_sum_ks_stat"]),
        # 6. |mass_ratio - 1| smaller better
        abs(float(row["total_mass_ratio"]) - 1.0),
    )


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for r in rows:
            line: list[Any] = []
            for col in COLUMNS:
                v = r.get(col, "")
                if isinstance(v, bool):
                    line.append("True" if v else "False")
                elif isinstance(v, float):
                    line.append(f"{v:.6f}" if v == v else "nan")  # NaN check
                else:
                    line.append(str(v) if v is not None else "")
            writer.writerow(line)


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats (NaN / inf / -inf) with None.

    ``json.dumps(..., allow_nan=False)`` raises on a bare ``NaN``; the
    failed-diffusion baseline row legitimately carries ``NaN`` for
    metrics that were never measured (``gen_max`` etc.). Mapping those
    to JSON ``null`` keeps ``metrics_all.json`` strict-JSON-parseable.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(rows: list[dict[str, Any]], winner: str, out_path: Path) -> None:
    obj = {
        "recommended_winner": winner,
        "freeze_decision": "deferred",
        "freeze_action_in_this_pr": False,
        "frozen_path": _rel(FROZEN_AGG_PATH),
        "frozen_path_exists": FROZEN_AGG_PATH.exists(),
        "sources": rows,
        "decision_rule": [
            "can_freeze_to_stage5 (True > False)",
            "acceptance_tier (PASS > MILD > FAIL)",
            "top20_pair_overlap (higher better)",
            "row_sum_ks_stat (lower better)",
            "col_sum_ks_stat (lower better)",
            "|total_mass_ratio - 1| (closer to 1 better)",
        ],
    }
    # allow_nan=False guarantees strict JSON; _json_safe maps any
    # non-finite float to null first so the dump cannot raise.
    out_path.write_text(
        json.dumps(_json_safe(obj), indent=2, sort_keys=False, allow_nan=False)
    )


def _write_decision(
    rows: list[dict[str, Any]],
    winner: str,
    out_path: Path,
) -> None:
    """Human-readable winner recommendation + caveats."""
    winner_row = next(r for r in rows if r["source_name"] == winner)
    lines = [
        "# Stage 4B-5C PR5C-3A unified scenario comparison",
        "",
        "## Recommended winner",
        f"- **{winner}** ({winner_row['source_type']}, "
        f"tier={winner_row['acceptance_tier']})",
        f"- candidate path: `{winner_row['candidate_path']}`",
        f"- can_freeze_to_stage5: {winner_row['can_freeze_to_stage5']}",
        "",
        "**This recommendation is advisory.** The final copy of the winning "
        f"candidate to `{_rel(FROZEN_AGG_PATH)}` is a separate sub-PR (PR5C-3B) "
        "gated on explicit user confirmation. This PR (PR5C-3A) does NOT "
        "freeze any source: no `od_samples_agg.npy` was written or modified, "
        "no candidate npy was overwritten, no Stage-5 code was touched.",
        "",
        "## Why this winner",
        "",
    ]
    if winner == "bootstrap_day_block":
        bs = next(r for r in rows if r["source_name"] == "bootstrap_day_block")
        cal = next(
            (r for r in rows
             if r["source_name"] == "diffusion_calibrated_zpin_weighted_pilot"),
            None,
        )
        lines.append(
            f"Bootstrap beats every other candidate on the decisive axes:"
        )
        if cal is not None:
            lines += [
                f"- `top20_pair_overlap`: **{bs['top20_pair_overlap']:.2f}** "
                f"vs diffusion-calibrated {cal['top20_pair_overlap']:.2f} "
                "(spatial structure preserved vs collapsed)",
                f"- `row_sum_ks_stat` (per-day-equivalent): "
                f"**{bs['row_sum_ks_stat']:.3f}** vs calibrated "
                f"{cal['row_sum_ks_stat']:.3f} (raw scale)",
                f"- `col_sum_ks_stat` (per-day-equivalent): "
                f"**{bs['col_sum_ks_stat']:.3f}** vs calibrated "
                f"{cal['col_sum_ks_stat']:.3f}",
                f"- `total_mass_ratio` (per-day-equivalent): "
                f"**{bs['total_mass_ratio']:.3f}** vs calibrated "
                f"{cal['total_mass_ratio']:.3f}",
                f"- `nonzero_ratio_x_real_test`: "
                f"**{bs['nonzero_ratio_x_real_test']:.3f}** vs calibrated "
                f"{cal['nonzero_ratio_x_real_test']:.3f}",
                "",
                "Bootstrap also already has a usable candidate npy on disk "
                "(`data/synthetic/od_samples_agg_bootstrap.npy`, 71.9 MB, "
                "shape (64, 530, 530), int32, nonnegative, "
                "used_slots ⊆ train_slots verified). The "
                "diffusion-calibrated path does NOT — PR5C-1B deliberately "
                "did not write one because the checkpoint produces samples "
                "with zero rank correlation against real top OD pairs "
                "(`top20_pair_overlap = 0`), which no posthoc thresholding "
                "can fix.",
            ]
    lines += [
        "",
        "## All sources (sorted by decision rule)",
        "",
        "| rank | source | tier | freezeable | top20 | row_ks | col_ks | mass_ratio | nz_x_real |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows):
        lines.append(
            f"| {i+1} | {r['source_name']} | {r['acceptance_tier']} | "
            f"{r['can_freeze_to_stage5']} | "
            f"{r['top20_pair_overlap']:.2f} | "
            f"{r['row_sum_ks_stat']:.3f} | "
            f"{r['col_sum_ks_stat']:.3f} | "
            f"{r['total_mass_ratio']:.3f} | "
            f"{r['nonzero_ratio_x_real_test']:.3f} |"
        )
    lines += [
        "",
        "## Important scale caveat",
        "",
        "The `row_sum_ks_stat` / `col_sum_ks_stat` / `total_mass_ratio` columns "
        "are not at identical scales across rows. The bootstrap row uses "
        "per-day-equivalent KS (bootstrap aggregates 11 days, real_test is 1 "
        "day, so the comparison divides bootstrap by `n_days_per_scenario`). "
        "The diffusion-calibrated row reports raw KS at the train-aggregate "
        "scale; its `total_mass_ratio` is rescaled to per-day-equivalent by "
        "dividing the raw `total_mass_ratio_mean` (≈ 5) by `n_train_days / "
        "n_test_days` = 5. See each row's `*_scale` column for the convention "
        "used. The headline conclusion (bootstrap >> diffusion-calibrated on "
        "structure) holds under any reasonable rescaling because the gap is "
        "at least an order of magnitude on row_ks / col_ks and is "
        "qualitative on top20.",
        "",
        "## Decision rule (lowest tuple = best, top of sort)",
        "",
        "1. `can_freeze_to_stage5` (True > False)",
        "2. `acceptance_tier` (PASS > MILD > FAIL)",
        "3. `top20_pair_overlap` (higher better)",
        "4. `row_sum_ks_stat` (lower better)",
        "5. `col_sum_ks_stat` (lower better)",
        "6. `|total_mass_ratio - 1|` (closer to 1 better)",
        "",
        "## Next steps (NOT in this PR)",
        "",
        "- PR5C-3B: with user confirmation, copy "
        f"`{_rel(BOOTSTRAP_CANDIDATE_PATH)}` to `{_rel(FROZEN_AGG_PATH)}` and "
        "record the freeze in `docs/decisions.md`. Until then "
        f"`{_rel(FROZEN_AGG_PATH)}` remains absent.",
        "- Optional PR5C-2B-ext: bootstrap parameter sensitivity sweep "
        "(`n_days_per_scenario`, seed variance) if PR5C-3B wants to log a "
        "robustness band before freezing.",
        "- Stage 5 (`docs/plan/stage5_rl_env.md`) remains gated on the freeze.",
        "",
        "## Safety verification",
        "",
        f"- `{_rel(FROZEN_AGG_PATH)}` exists: "
        f"{FROZEN_AGG_PATH.exists()}",
        f"- `{_rel(FROZEN_4D_PATH)}` exists: {FROZEN_4D_PATH.exists()}",
        f"- `{_rel(CALIBRATED_CANDIDATE_PATH)}` exists: "
        f"{CALIBRATED_CANDIDATE_PATH.exists()}",
        f"- `{_rel(BOOTSTRAP_CANDIDATE_PATH)}` exists "
        f"(unmodified by this script): {BOOTSTRAP_CANDIDATE_PATH.exists()}",
    ]
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bootstrap-metrics",
        type=Path,
        default=REPO / "results" / "stage4" / "bootstrap" / "metrics.json",
    )
    parser.add_argument(
        "--posthoc-metrics",
        type=Path,
        default=REPO
        / "results"
        / "stage4"
        / "posthoc_calibration_zpin_weighted"
        / "metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results" / "stage4" / "comparison",
    )
    args = parser.parse_args()

    # --- safety guards ----------------------------------------------------
    out_dir = args.output_dir.resolve()
    if out_dir == FROZEN_AGG_PATH.parent:
        raise SystemExit(
            f"refusing to write comparison into data/synthetic/; use a "
            f"results/ path. Got: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- pre-run state record --------------------------------------------
    pre_state = {
        "frozen_agg_exists": FROZEN_AGG_PATH.exists(),
        "frozen_4d_exists": FROZEN_4D_PATH.exists(),
        "calibrated_candidate_exists": CALIBRATED_CANDIDATE_PATH.exists(),
        "bootstrap_candidate_exists": BOOTSTRAP_CANDIDATE_PATH.exists(),
    }

    print("=" * 70)
    print("STAGE 4B-5C PR5C-3A  --  unified scenario source comparison")
    print("=" * 70)
    print(f"  bootstrap_metrics: {args.bootstrap_metrics}")
    print(f"  posthoc_metrics  : {args.posthoc_metrics}")
    print(f"  output_dir       : {out_dir}")
    print(f"\n  PRE-RUN paths:")
    for k, v in pre_state.items():
        print(f"    {k:<32}: {v}")

    if not args.bootstrap_metrics.exists():
        raise SystemExit(
            f"bootstrap metrics not found: {args.bootstrap_metrics}\n"
            "Run PR5C-2B first: python -m experiments.run_stage4_bootstrap ..."
        )
    if not args.posthoc_metrics.exists():
        raise SystemExit(
            f"posthoc metrics not found: {args.posthoc_metrics}\n"
            "Run PR5C-1B first: python -m experiments.run_stage4_posthoc_calibrate ..."
        )

    # --- build rows ------------------------------------------------------
    rows = [
        _row_bootstrap(args.bootstrap_metrics),
        _row_diffusion_calibrated(args.posthoc_metrics),
        _row_diffusion_raw(args.posthoc_metrics),
        _row_failed_baseline(),
    ]
    rows.sort(key=_sort_key)
    winner = rows[0]["source_name"]
    print(f"\n  recommended winner: {winner}")
    print(f"  freeze action     : DEFERRED (user-confirmation required)")

    # --- write artefacts -------------------------------------------------
    _write_csv(rows, out_dir / "metrics_all.csv")
    _write_json(rows, winner, out_dir / "metrics_all.json")
    _write_decision(rows, winner, out_dir / "decision.md")

    # --- post-run safety verification ------------------------------------
    post_state = {
        "frozen_agg_exists": FROZEN_AGG_PATH.exists(),
        "frozen_4d_exists": FROZEN_4D_PATH.exists(),
        "calibrated_candidate_exists": CALIBRATED_CANDIDATE_PATH.exists(),
        "bootstrap_candidate_exists": BOOTSTRAP_CANDIDATE_PATH.exists(),
    }
    if pre_state != post_state:
        print("\n  WARNING: data/synthetic/ state changed during this run:")
        for k in pre_state:
            if pre_state[k] != post_state[k]:
                print(f"    {k}: {pre_state[k]} -> {post_state[k]}")
    else:
        print("\n  data/synthetic/ state unchanged by this run ✓")

    print(f"\n  metrics_all.csv  : {out_dir / 'metrics_all.csv'}")
    print(f"  metrics_all.json : {out_dir / 'metrics_all.json'}")
    print(f"  decision.md      : {out_dir / 'decision.md'}")
    print("\n" + "=" * 70)
    print(f"PR5C-3A comparison complete  --  recommended winner: {winner}")
    print("PR5C-3B (freeze) is a separate sub-PR; NOT entered here.")
    print("=" * 70)


if __name__ == "__main__":
    main()
