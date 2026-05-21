"""Stage-4B-0: normalization ablation only.

Goal: pick a normalization scheme that survives the eVTOL OD sparsity
(~0.117% nonzero). Stage-4A's global ``log1p`` + standardize + ``clip[-3,3]``
degenerates -- every nonzero saturates to ``+1`` (see ``docs/decisions.md``
2026-05-19). This script compares 5 candidates on the TRAIN split (slots
0..431, 9 days) and reports per-scheme:

  * the value count=0 maps to,
  * the distribution of nonzero entries after normalization,
  * what fraction of nonzero entries saturate to +-1,
  * inverse-transform round-trip error (mean / p95 / max),
  * whether small counts {1, 2, 5, 10, 20} map to distinct normalized
    values, and how well-separated they are,
  * a one-word recommendation level.

Outputs:
  * stdout: stats per scheme + ranked recommendation.
  * ``results/stage4/norm_ablation.csv``: same data as a flat table.

This script does NOT modify ``src/data/od_dataset.py`` or its defaults; the
schemes are inlined here so the Stage-4A pipeline stays reproducible.

Run:
    python -m experiments.run_stage4_norm_ablation
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "diffusion.yaml"
DEFAULT_OUTPUT = REPO / "results" / "stage4" / "norm_ablation.csv"

COUNTS_TO_DISTINGUISH = [1, 2, 5, 10, 20]
ZERO_BACKGROUND = -1.0          # value zeros are pinned to in scheme D
SAT_TOL = 1e-6                  # |x| >= 1 - SAT_TOL counts as saturated
WELL_SEPARATED_GAP = 0.01       # min adjacent gap on COUNTS_TO_DISTINGUISH


# --- helpers ---------------------------------------------------------------


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


# --- statistics passes over the train split -------------------------------


def stats_global(od: np.ndarray, train_slots: Iterable[int]) -> tuple[float, float]:
    """Global mu/sigma of ``log1p(count)`` over all train cells."""
    total, total_sq, count = 0.0, 0.0, 0
    for s in train_slots:
        v = np.log1p(np.asarray(od[s], dtype=np.float64))
        total += float(v.sum())
        total_sq += float(np.square(v).sum())
        count += int(v.size)
    mu = total / count
    var = max(total_sq / count - mu * mu, 0.0)
    sigma = math.sqrt(var) if var > 0 else 1.0
    return mu, sigma


def stats_nonzero(od: np.ndarray, train_slots: Iterable[int]) -> tuple[float, float, int]:
    """mu/sigma of ``log1p`` over NONZERO train cells only."""
    total, total_sq, count = 0.0, 0.0, 0
    for s in train_slots:
        v = np.asarray(od[s], dtype=np.float64)
        mask = v > 0
        if not mask.any():
            continue
        vv = np.log1p(v[mask])
        total += float(vv.sum())
        total_sq += float(np.square(vv).sum())
        count += int(vv.size)
    if count == 0:
        return 0.0, 1.0, 0
    mu = total / count
    var = max(total_sq / count - mu * mu, 0.0)
    sigma = math.sqrt(var) if var > 0 else 1.0
    return mu, sigma, count


def quantile_all_approx(
    od: np.ndarray, train_slots: Iterable[int], q: float, per_slot: int = 10_000
) -> float:
    """Approximate q-quantile of ``log1p(count)`` over ALL train cells."""
    rng = np.random.default_rng(0)
    chunks: list[np.ndarray] = []
    for s in train_slots:
        v = np.log1p(np.asarray(od[s], dtype=np.float64).ravel())
        if v.size > per_slot:
            idx = rng.choice(v.size, per_slot, replace=False)
            v = v[idx]
        chunks.append(v)
    return float(np.quantile(np.concatenate(chunks), q))


def quantile_nonzero_exact(od: np.ndarray, train_slots: Iterable[int], q: float) -> float:
    """Exact q-quantile of ``log1p`` over NONZERO train cells (small set)."""
    chunks: list[np.ndarray] = []
    for s in train_slots:
        v = np.asarray(od[s], dtype=np.float64)
        nz = v[v > 0]
        if nz.size:
            chunks.append(np.log1p(nz))
    if not chunks:
        return 0.0
    return float(np.quantile(np.concatenate(chunks), q))


# --- scheme classes -------------------------------------------------------


class Scheme:
    """A normalization scheme: ``apply`` raw counts -> normalized in [-1, 1];
    ``inverse`` normalized -> nonneg counts."""

    name: str
    description: str

    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def zero_value(self) -> float:
        return float(self.apply(np.array([0.0])).item())


class GlobalStdClip(Scheme):
    def __init__(self, name: str, mu: float, sigma: float, clip_val: float) -> None:
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.clip_val = clip_val
        self.description = (
            f"log1p + global std (mu={mu:.4f}, sigma={sigma:.4f}), clip={clip_val:g}"
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        v = np.log1p(np.asarray(x, dtype=np.float64))
        v = (v - self.mu) / self.sigma
        v = np.clip(v, -self.clip_val, self.clip_val) / self.clip_val
        return v

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        v = np.asarray(x_norm, dtype=np.float64) * self.clip_val
        v = v * self.sigma + self.mu
        return np.maximum(np.expm1(v), 0.0)


class NonzeroStdClip(Scheme):
    """Standardize using NONZERO-only stats; zeros are pinned to
    ``zero_value`` instead of going through the log1p+std path."""

    def __init__(
        self,
        name: str,
        mu_nz: float,
        sigma_nz: float,
        clip_val: float,
        zero_value: float = ZERO_BACKGROUND,
    ) -> None:
        self.name = name
        self.mu = mu_nz
        self.sigma = sigma_nz
        self.clip_val = clip_val
        self.zero_value_ = zero_value
        self.description = (
            f"log1p + NONZERO-only std (mu_nz={mu_nz:.4f}, sigma_nz={sigma_nz:.4f}), "
            f"clip={clip_val:g}, zero->{zero_value:+.1f}"
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        out = np.full(x_arr.shape, self.zero_value_, dtype=np.float64)
        mask = x_arr > 0
        if mask.any():
            v = np.log1p(x_arr[mask])
            v = (v - self.mu) / self.sigma
            v = np.clip(v, -self.clip_val, self.clip_val) / self.clip_val
            out[mask] = v
        return out

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        # Exact pin: zeros land at ``zero_value_`` with no FP noise from
        # apply(), so an equality test is enough for the ablation.
        x_arr = np.asarray(x_norm, dtype=np.float64)
        out = np.zeros_like(x_arr)
        mask = x_arr != self.zero_value_
        if mask.any():
            v = x_arr[mask] * self.clip_val
            v = v * self.sigma + self.mu
            out[mask] = np.maximum(np.expm1(v), 0.0)
        return out

    def zero_value(self) -> float:
        return self.zero_value_


class Log1pMinmax(Scheme):
    """``log1p`` -> linear to ``[-1, 1]`` using ``vmax`` as the upper cap."""

    def __init__(self, name: str, vmax: float, source: str) -> None:
        self.name = name
        self.vmax = vmax
        self.description = (
            f"log1p + linear scale to [-1,1], vmax={vmax:.4f} ({source})"
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        v = np.log1p(np.asarray(x, dtype=np.float64))
        v = np.clip(v, 0.0, self.vmax)
        return (v / self.vmax) * 2.0 - 1.0

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        v = (np.asarray(x_norm, dtype=np.float64) + 1.0) * 0.5 * self.vmax
        return np.maximum(np.expm1(v), 0.0)


# --- evaluation -----------------------------------------------------------


def evaluate(scheme: Scheme, od: np.ndarray, train_slots: list[int]) -> dict[str, Any]:
    """Stream the train split through a scheme, collect summary stats."""
    sat_threshold = 1.0 - SAT_TOL

    nz_count = 0
    nz_sum = 0.0
    nz_sumsq = 0.0
    nz_min = math.inf
    nz_max = -math.inf
    nz_saturated = 0

    err_sum = 0.0
    err_count = 0
    err_max = 0.0
    err_samples: list[np.ndarray] = []
    rng = np.random.default_rng(0)

    for s in train_slots:
        raw = np.asarray(od[s], dtype=np.float64)
        x_norm = scheme.apply(raw)

        nz_mask = raw > 0
        if nz_mask.any():
            nz = x_norm[nz_mask]
            nz_count += int(nz.size)
            nz_sum += float(nz.sum())
            nz_sumsq += float(np.square(nz).sum())
            nz_min = min(nz_min, float(nz.min()))
            nz_max = max(nz_max, float(nz.max()))
            nz_saturated += int((np.abs(nz) >= sat_threshold).sum())

        back = scheme.inverse(x_norm)
        err = np.abs(back - raw)
        err_sum += float(err.sum())
        err_count += int(err.size)
        err_max = max(err_max, float(err.max()))
        # Keep a per-slot sub-sample for the p95.
        if err.size > 5_000:
            idx = rng.choice(err.size, 5_000, replace=False)
            err_samples.append(err.ravel()[idx])
        else:
            err_samples.append(err.ravel())

    nz_mean = nz_sum / nz_count if nz_count else 0.0
    nz_std = (
        math.sqrt(max(nz_sumsq / nz_count - nz_mean * nz_mean, 0.0))
        if nz_count
        else 0.0
    )
    err_mean = err_sum / err_count
    err_p95 = float(np.quantile(np.concatenate(err_samples), 0.95))
    pct_sat = (nz_saturated / nz_count) if nz_count else float("nan")

    # Probe a few specific counts for distinguishability.
    probes = np.array(COUNTS_TO_DISTINGUISH, dtype=np.float64)
    probe_norm = scheme.apply(probes)
    pairwise_min_gap = float(np.min(np.diff(probe_norm)))
    distinguishable = bool(pairwise_min_gap > 1e-4)
    well_separated = bool(pairwise_min_gap > WELL_SEPARATED_GAP)

    return {
        "name": scheme.name,
        "description": scheme.description,
        "zero_value_after_norm": scheme.zero_value(),
        "nonzero_min": float(nz_min) if nz_count else float("nan"),
        "nonzero_max": float(nz_max) if nz_count else float("nan"),
        "nonzero_mean": float(nz_mean),
        "nonzero_std": float(nz_std),
        "pct_nonzero_saturated": pct_sat,
        "inverse_max_abs_err": err_max,
        "inverse_p95_abs_err": err_p95,
        "inverse_mean_abs_err": err_mean,
        "distinguishes_small_counts": distinguishable,
        "well_separated_small_counts": well_separated,
        "small_count_norm_min_gap": pairwise_min_gap,
        "norm_count_1": float(probe_norm[0]),
        "norm_count_2": float(probe_norm[1]),
        "norm_count_5": float(probe_norm[2]),
        "norm_count_10": float(probe_norm[3]),
        "norm_count_20": float(probe_norm[4]),
    }


def assign_recommendations(results: list[dict[str, Any]]) -> None:
    """Tag each scheme with one of: primary / fallback / rejected_*.

    Mutates ``results`` in place by adding a ``recommendation`` key.
    """
    well_sep = [r for r in results if r["well_separated_small_counts"]]
    well_sep.sort(key=lambda r: (r["pct_nonzero_saturated"], r["inverse_mean_abs_err"]))
    primary = well_sep[0]["name"] if well_sep else None
    fallback = well_sep[1]["name"] if len(well_sep) > 1 else None

    for r in results:
        if r["name"] == primary:
            r["recommendation"] = "primary"
        elif r["name"] == fallback:
            r["recommendation"] = "fallback"
        elif not r["distinguishes_small_counts"]:
            r["recommendation"] = "rejected_undistinguishable"
        elif r["pct_nonzero_saturated"] >= 0.5:
            r["recommendation"] = "rejected_saturated"
        else:
            r["recommendation"] = "alternative"


# --- I/O ------------------------------------------------------------------


CSV_COLUMNS = [
    "name", "description", "recommendation",
    "zero_value_after_norm",
    "nonzero_min", "nonzero_max", "nonzero_mean", "nonzero_std",
    "pct_nonzero_saturated",
    "inverse_max_abs_err", "inverse_p95_abs_err", "inverse_mean_abs_err",
    "distinguishes_small_counts", "well_separated_small_counts",
    "small_count_norm_min_gap",
    "norm_count_1", "norm_count_2", "norm_count_5", "norm_count_10", "norm_count_20",
]


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in CSV_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    od_path = _resolve(cfg["input"]["od_path"])
    meta_path = _resolve(cfg["input"]["meta_path"])

    with open(meta_path) as fh:
        meta = json.load(fh)
    n_slots = int(meta["T"])
    bin_min = int(meta.get("time_bin_min", 30))
    spd = 24 * 60 // bin_min
    split_cfg = cfg["data"]["split"]
    lo_d, hi_d = split_cfg["train_days"]
    train_slots = list(range(lo_d * spd, min(hi_d * spd, n_slots)))

    print("=" * 78)
    print("STAGE 4B-0  --  normalization ablation (TRAIN split only)")
    print("=" * 78)
    print(f"  od_path       : {od_path}")
    print(f"  meta_path     : {meta_path}")
    print(
        f"  train slots   : [{train_slots[0]}, {train_slots[-1]}]  "
        f"({len(train_slots)} slots, {len(train_slots) // spd} days)"
    )
    print(f"  counts probed : {COUNTS_TO_DISTINGUISH}")
    print(f"  output csv    : {args.output}")

    od = np.load(od_path, mmap_mode="r")
    print(f"  od shape      : {tuple(od.shape)}  dtype={od.dtype}")

    # --- precompute stats each scheme needs ---
    print("\n--- precomputing stats on TRAIN split ---")
    mu_g, sigma_g = stats_global(od, train_slots)
    print(f"  global    : mu={mu_g:.6f}   sigma={sigma_g:.6f}")
    mu_nz, sigma_nz, n_nz_train = stats_nonzero(od, train_slots)
    print(
        f"  nonzero   : mu={mu_nz:.4f}   sigma={sigma_nz:.4f}   "
        f"n_nonzero_train_cells={n_nz_train}"
    )
    p999_all = quantile_all_approx(od, train_slots, 0.999)
    p999_nz = quantile_nonzero_exact(od, train_slots, 0.999)
    print(
        f"  log1p p999 (all cells, approx)  : {p999_all:.6f}  "
        "<- ~0 on this 0.117%-sparse tensor"
    )
    print(
        f"  log1p p999 (nonzero cells, exact): {p999_nz:.4f}  "
        "<- scheme E uses this as vmax"
    )

    # --- schemes ---
    schemes: list[Scheme] = [
        GlobalStdClip("A_current_global_clip3", mu_g, sigma_g, 3.0),
        GlobalStdClip("B_global_clip30", mu_g, sigma_g, 30.0),
        GlobalStdClip("C_global_clip100", mu_g, sigma_g, 100.0),
        NonzeroStdClip("D_nonzero_global_clip3", mu_nz, sigma_nz, 3.0),
        Log1pMinmax(
            "E_log1p_minmax_p999_nonzero",
            p999_nz,
            "p99.9 of nonzero log1p (train)",
        ),
    ]

    # --- evaluate ---
    print("\n--- evaluating ---")
    results: list[dict[str, Any]] = []
    for sc in schemes:
        print(f"  {sc.name}: {sc.description}")
        results.append(evaluate(sc, od, train_slots))
    assign_recommendations(results)

    # --- summary tables ---
    print("\n--- summary --------------------------------------------------------"
          "-------")
    hdr = (
        f"{'scheme':<32} {'zero':>8} {'nz_mean':>9} {'nz_std':>8} "
        f"{'pct_sat':>9} {'err_mean':>10} {'err_p95':>10} {'err_max':>10} "
        f"{'gap':>8} {'reco':>16}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<32} "
            f"{r['zero_value_after_norm']:>8.4f} "
            f"{r['nonzero_mean']:>9.4f} "
            f"{r['nonzero_std']:>8.4f} "
            f"{r['pct_nonzero_saturated']:>9.2%} "
            f"{r['inverse_mean_abs_err']:>10.4f} "
            f"{r['inverse_p95_abs_err']:>10.4f} "
            f"{r['inverse_max_abs_err']:>10.4f} "
            f"{r['small_count_norm_min_gap']:>8.4f} "
            f"{r['recommendation']:>16}"
        )

    print("\n--- normalized values for probe counts ---")
    print(
        f"{'scheme':<32} {'n(1)':>10} {'n(2)':>10} {'n(5)':>10} "
        f"{'n(10)':>10} {'n(20)':>10}"
    )
    for r in results:
        print(
            f"{r['name']:<32} "
            f"{r['norm_count_1']:>10.4f} "
            f"{r['norm_count_2']:>10.4f} "
            f"{r['norm_count_5']:>10.4f} "
            f"{r['norm_count_10']:>10.4f} "
            f"{r['norm_count_20']:>10.4f}"
        )

    write_csv(results, args.output)
    print(f"\nWrote {args.output}")

    # --- recommendation line ---
    primary = next((r for r in results if r["recommendation"] == "primary"), None)
    fallback = next((r for r in results if r["recommendation"] == "fallback"), None)
    print("\n--- recommendation -------------------------------------------------"
          "-------")
    if primary is not None:
        print(
            f"  PRIMARY  : {primary['name']}  "
            f"(pct_sat={primary['pct_nonzero_saturated']:.4%}, "
            f"mean_err={primary['inverse_mean_abs_err']:.4f}, "
            f"min gap on probes={primary['small_count_norm_min_gap']:.4f})"
        )
    else:
        print("  PRIMARY  : none -- no scheme well-separates counts 1, 2, 5, 10, 20")
    if fallback is not None:
        print(
            f"  FALLBACK : {fallback['name']}  "
            f"(pct_sat={fallback['pct_nonzero_saturated']:.4%}, "
            f"mean_err={fallback['inverse_mean_abs_err']:.4f})"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
