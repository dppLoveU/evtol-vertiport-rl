"""Stage 4B-5C PR5C-1A: posthoc calibration helpers for diffusion samples.

Pure-function module. Does NOT read real data, does NOT load checkpoints,
does NOT sample from any diffusion model. The downstream CLI (deferred
to a later PR) will compose these helpers with real diffusion samples
and the real 12 km train aggregate to produce a candidate calibrated
artefact, but that is out of scope here.

Calibration model (per-pixel, per-sample)
-----------------------------------------

Given continuous OD samples ``x = inverse_transform(diffusion_sample)``
in count units, the calibrated integer output is::

    cal = round( I(x > tau) * x * scale )

with two scalar knobs:

* ``tau`` -- the suppression threshold below which a continuous pixel is
  forced to zero (kills the "many small positives everywhere" failure
  mode documented in ``docs/decisions.md`` 2026-05-20 PR5B-3 /
  PR5B-3b-3).
* ``scale`` -- a multiplicative rescaling that brings the surviving
  pixels' magnitude back to count units consistent with real OD.

Negatives are clipped before the threshold via ``clip_nonnegative``;
the clip is implemented as ``np.maximum(x, 0)`` and **does NOT use
``abs``** (which would silently flip negatives into spurious positives).

Fitting contract
----------------

``grid_search_tau_scale`` fits ``(tau, scale)`` ONLY against the real
TRAIN aggregate's ``nonzero_ratio`` and ``total_mass``. val / test
aggregates are never consumed by the fit; they exist purely for
``evaluate_calibrated_samples`` reporting downstream. This avoids the
"calibrate against the holdout marginal" overfit failure mode and
mirrors the no-leak contract enforced by
``ConditionalBootstrapSampler`` (PR5C-2A).

Diagonal convention
-------------------

Stage-3 OD construction sets ``drop_intra_zone: true`` (see
``configs/od_12km.yaml``), so every real OD aggregate has zeros on the
diagonal. Calibrated diffusion samples might place spurious mass on
the diagonal; the top-k pair-overlap functions exclude diagonal cells
by default so an off-diagonal-only ranking is compared apples-to-apples.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.utils.metrics_dist import ks_stat_1d


__all__ = [
    "clip_nonnegative",
    "apply_threshold_and_scale",
    "sparse_aggregate_stats",
    "compute_reference_stats",
    "topk_pair_overlap",
    "grid_search_tau_scale",
    "evaluate_calibrated_samples",
    "acceptance_verdict",
    "DEFAULT_FAILED_DIFFUSION_BASELINE",
]


# Failed-diffusion baseline (PR5B-3b-3, docs/decisions.md 2026-05-20).
# Used as the "mild" floor by ``acceptance_verdict``: calibrated samples
# must beat ALL of these to count as at least "mild"; otherwise they
# sit in "fail" alongside the uncalibrated failed-diffusion runs.
DEFAULT_FAILED_DIFFUSION_BASELINE: dict[str, float] = {
    "row_sum_ks_stat": 1.000,
    "col_sum_ks_stat": 1.000,
    "nonzero_ratio_x_real": 143.0,
    "total_mass_ratio": 181.0,
}


# --- 1. clip_nonnegative ---------------------------------------------------


def clip_nonnegative(x: np.ndarray | float) -> np.ndarray:
    """Clip ``x`` to be non-negative.

    Implemented as ``np.maximum(x, 0)``. This function **does NOT use
    ``np.abs``**: ``abs(-2.0) = 2.0`` would silently flip a negative
    into a spurious positive, which is exactly the bug we want to
    avoid for a count-domain calibrator. The accompanying unit test
    pins this behaviour for an input that distinguishes the two.
    """
    arr = np.asarray(x)
    return np.maximum(arr, 0)


# --- 2. apply_threshold_and_scale -----------------------------------------


def apply_threshold_and_scale(
    samples_cont: np.ndarray, tau: float, scale: float
) -> np.ndarray:
    """Apply the ``(tau, scale)`` posthoc calibration to continuous OD.

    Pipeline:

      1. ``clip_nonnegative`` -- forbid any negative continuous value
         to contribute (no ``abs``; see ``clip_nonnegative``).
      2. mask = ``x > tau`` -- suppress small-magnitude pixels.
      3. ``round( mask * x * scale )`` -- recover integer counts.

    Parameters
    ----------
    samples_cont : np.ndarray
        Continuous OD samples of shape ``[Z, Z]`` or ``[N, Z, Z]``.
    tau : float
        Suppression threshold in count units (post-clip).
    scale : float
        Multiplicative rescaling applied to the surviving mass.

    Returns
    -------
    np.ndarray
        Same outer shape as ``samples_cont``, dtype ``int32``,
        non-negative.
    """
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    if scale < 0:
        raise ValueError(f"scale must be >= 0, got {scale}")
    x = clip_nonnegative(np.asarray(samples_cont, dtype=np.float64))
    mask = x > tau
    return np.round(mask * x * scale).astype(np.int32)


# --- internal helpers -----------------------------------------------------


def _shannon_entropy(arr: np.ndarray) -> float:
    """Shannon entropy (nats) of a non-negative array normalised to sum=1."""
    flat = np.asarray(arr, dtype=np.float64).ravel()
    total = float(flat.sum())
    if total <= 0.0:
        return float("nan")
    p = flat[flat > 0] / total
    return float(-(p * np.log(p)).sum())


def _top_n_flat_indices(
    arr: np.ndarray, n: int, exclude_diagonal: bool = True
) -> set[int]:
    """Return flat indices of the top-``n`` cells as a set.

    For 2-D square inputs with ``exclude_diagonal=True`` the diagonal
    is replaced by ``-inf`` before ranking, so a diagonal cell can
    never appear in the returned set. Ties are broken by stable sort
    (deterministic ordering by ascending flat index for equal values).
    """
    work = np.asarray(arr, dtype=np.float64).copy()
    if (
        exclude_diagonal
        and work.ndim == 2
        and work.shape[0] == work.shape[1]
    ):
        np.fill_diagonal(work, -np.inf)
    flat = work.ravel()
    n_eff = min(n, flat.size)
    if n_eff <= 0:
        return set()
    order = np.argsort(-flat, kind="stable")[:n_eff]
    # If exclude_diagonal pushed -inf entries into the prefix (only
    # possible if n exceeds the non-diagonal count), drop them.
    if exclude_diagonal:
        order = order[np.isfinite(flat[order])]
    return {int(i) for i in order}


def _single_stats(arr: np.ndarray) -> dict[str, Any]:
    """Stats for a single ``[Z, Z]`` (or any-shape) matrix."""
    a = np.asarray(arr)
    total = float(a.sum())
    return {
        "nonzero_ratio": float((a != 0).mean()) if a.size else float("nan"),
        "total_mass": total,
        "mean": float(a.mean()) if a.size else float("nan"),
        "max": float(a.max()) if a.size else float("nan"),
        "row_sums": a.sum(axis=-1).ravel().astype(np.float64),
        "col_sums": a.sum(axis=-2).ravel().astype(np.float64),
        "entropy": _shannon_entropy(a),
    }


# --- 3. sparse_aggregate_stats --------------------------------------------


def sparse_aggregate_stats(x_int: np.ndarray) -> dict[str, Any]:
    """Aggregate statistics for an OD matrix or a stack of OD matrices.

    Parameters
    ----------
    x_int : np.ndarray
        ``[Z, Z]`` for a single matrix or ``[N, Z, Z]`` for a stack.

    Returns
    -------
    dict
        For ``[Z, Z]`` input: keys ``nonzero_ratio``, ``total_mass``,
        ``mean``, ``max``, ``row_sums`` ``[Z]``, ``col_sums`` ``[Z]``,
        ``entropy``.

        For ``[N, Z, Z]`` input: per-sample arrays under
        ``<key>_per_sample`` for the scalar keys, and ``_mean`` / ``_std``
        summary scalars for ``nonzero_ratio``, ``total_mass``, ``mean``,
        ``max``, ``entropy``. ``row_sums_per_sample`` and
        ``col_sums_per_sample`` are stacked ``[N, Z]`` arrays.
    """
    arr = np.asarray(x_int)
    if arr.ndim == 2:
        return _single_stats(arr)
    if arr.ndim == 3:
        per = [_single_stats(arr[i]) for i in range(arr.shape[0])]
        scalar_keys = ("nonzero_ratio", "total_mass", "mean", "max", "entropy")
        out: dict[str, Any] = {"n_samples": int(arr.shape[0])}
        for k in scalar_keys:
            vals = np.array([s[k] for s in per], dtype=np.float64)
            out[f"{k}_per_sample"] = vals
            out[f"{k}_mean"] = float(np.nanmean(vals))
            out[f"{k}_std"] = float(np.nanstd(vals))
        out["row_sums_per_sample"] = np.stack(
            [s["row_sums"] for s in per], axis=0
        )
        out["col_sums_per_sample"] = np.stack(
            [s["col_sums"] for s in per], axis=0
        )
        return out
    raise ValueError(
        f"sparse_aggregate_stats expects 2-D or 3-D input, got shape {arr.shape}"
    )


# --- 4. compute_reference_stats -------------------------------------------


def compute_reference_stats(
    real_agg: np.ndarray,
    *,
    real_k: int = 20,
    exclude_diagonal: bool = True,
) -> dict[str, Any]:
    """Reference statistics for the real OD aggregate.

    Parameters
    ----------
    real_agg : np.ndarray
        ``[Z, Z]`` real aggregate (e.g. ``real_train.sum(axis=0)`` for
        the fit reference, or ``real_test`` for reporting).
    real_k : int, default 20
        Number of top OD pairs to record.
    exclude_diagonal : bool, default True
        If True, the diagonal is excluded from the top-k ranking
        (consistent with Stage-3 ``drop_intra_zone: true``).
    """
    arr = np.asarray(real_agg)
    if arr.ndim != 2:
        raise ValueError(
            f"compute_reference_stats expects a 2-D matrix, got shape {arr.shape}"
        )
    stats = _single_stats(arr)
    top_k_set = _top_n_flat_indices(arr, real_k, exclude_diagonal=exclude_diagonal)
    return {
        "real_nonzero_ratio": stats["nonzero_ratio"],
        "real_total_mass": stats["total_mass"],
        "real_mean": stats["mean"],
        "real_max": stats["max"],
        "real_row_sums": stats["row_sums"],
        "real_col_sums": stats["col_sums"],
        "real_entropy": stats["entropy"],
        f"real_top{real_k}_pairs": top_k_set,
        "real_k": int(real_k),
        "exclude_diagonal": bool(exclude_diagonal),
    }


# --- 5. topk_pair_overlap -------------------------------------------------


def topk_pair_overlap(
    gen_agg: np.ndarray,
    real_agg: np.ndarray,
    *,
    real_k: int = 20,
    gen_k: int = 50,
    exclude_diagonal: bool = True,
) -> int:
    """Number of real top-``real_k`` pairs that appear in gen top-``gen_k``.

    Both inputs are 2-D OD aggregates ``[Z, Z]``. Diagonal cells are
    excluded by default (Stage-3 ``drop_intra_zone: true`` invariant).
    Ties are broken deterministically by stable argsort (ascending flat
    index for equal values).
    """
    real_top = _top_n_flat_indices(real_agg, real_k, exclude_diagonal=exclude_diagonal)
    gen_top = _top_n_flat_indices(gen_agg, gen_k, exclude_diagonal=exclude_diagonal)
    return int(len(real_top & gen_top))


# --- 6. grid_search_tau_scale --------------------------------------------


def _calibrated_mean_stats(
    samples_cont: np.ndarray, tau: float, scale: float
) -> tuple[float, float]:
    """Mean per-sample nz_ratio and total_mass after (tau, scale)."""
    cal = apply_threshold_and_scale(samples_cont, tau, scale)
    if cal.ndim == 2:
        return float((cal != 0).mean()), float(cal.sum())
    if cal.ndim == 3:
        # (cal != 0).mean() over the whole stack equals the mean of
        # per-sample nz_ratios because every sample has equal Z*Z size.
        nz = float((cal != 0).mean())
        total = float(cal.sum() / cal.shape[0])
        return nz, total
    raise ValueError(f"unexpected calibrated shape {cal.shape}")


def grid_search_tau_scale(
    samples_cont: np.ndarray,
    real_train_agg: np.ndarray,
    *,
    tau_grid: Sequence[float] | None = None,
    scale_grid: Sequence[float] | None = None,
    lambda_mass: float = 1.0,
) -> tuple[float, float, dict[str, float], list[dict[str, float]]]:
    """Grid-search ``(tau, scale)`` against the TRAIN aggregate's marginals.

    Objective::

        J(tau, scale) = |nz_ratio / real_nz_ratio - 1|
                      + lambda_mass * |total_mass / real_total_mass - 1|

    Only ``real_train_agg`` is consumed. val / test must never be
    passed here. The objective deliberately avoids row / column KS to
    prevent the calibrator from overfitting marginal shape to the fit
    target.

    Parameters
    ----------
    samples_cont : np.ndarray
        Continuous OD samples ``[Z, Z]`` or ``[N, Z, Z]``.
    real_train_agg : np.ndarray
        Real TRAIN aggregate ``[Z, Z]``. NOT val / test.
    tau_grid : Sequence[float] | None
        ``np.linspace(0.1, 2.0, 20)`` by default.
    scale_grid : Sequence[float] | None
        ``np.geomspace(1e-3, 2.0, 20)`` by default.
    lambda_mass : float, default 1.0
        Weight on the total-mass term of the objective.

    Returns
    -------
    best_tau : float
    best_scale : float
    best_metrics : dict
        ``{tau, scale, nz_ratio, total_mass, objective}`` at the
        argmin of the grid.
    grid : list[dict]
        Every (tau, scale) point evaluated, in row-major order
        (outer = tau, inner = scale).
    """
    if tau_grid is None:
        tau_grid = np.linspace(0.1, 2.0, 20)
    if scale_grid is None:
        scale_grid = np.geomspace(1e-3, 2.0, 20)
    if lambda_mass < 0:
        raise ValueError(f"lambda_mass must be >= 0, got {lambda_mass}")
    real_arr = np.asarray(real_train_agg)
    if real_arr.ndim != 2:
        raise ValueError(
            f"real_train_agg must be 2-D [Z, Z], got shape {real_arr.shape}"
        )
    real_nz = float((real_arr != 0).mean())
    real_total = float(real_arr.sum())
    if real_nz <= 0 or real_total <= 0:
        raise ValueError(
            "real_train_agg has zero nonzero ratio or zero total mass -- "
            "cannot define a meaningful calibration target"
        )

    grid: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    for tau in tau_grid:
        for scale in scale_grid:
            nz, total = _calibrated_mean_stats(samples_cont, float(tau), float(scale))
            obj = abs(nz / real_nz - 1.0) + lambda_mass * abs(
                total / real_total - 1.0
            )
            point = {
                "tau": float(tau),
                "scale": float(scale),
                "nz_ratio": nz,
                "total_mass": total,
                "objective": float(obj),
            }
            grid.append(point)
            if best is None or point["objective"] < best["objective"]:
                best = point
    assert best is not None
    return best["tau"], best["scale"], best, grid


# --- 7. evaluate_calibrated_samples ---------------------------------------


def _evaluate_single(
    cal: np.ndarray,
    real_arr: np.ndarray,
    real_nz: float,
    real_total: float,
    real_row: np.ndarray,
    real_col: np.ndarray,
    real_top20: set[int],
    real_top50: set[int],
    real_entropy: float,
    exclude_diagonal: bool,
) -> dict[str, float]:
    """Per-sample comparison metrics for one calibrated matrix."""
    nz = float((cal != 0).mean())
    total = float(cal.sum())
    gen_row = cal.sum(axis=-1).ravel().astype(np.float64)
    gen_col = cal.sum(axis=-2).ravel().astype(np.float64)
    gen_top20 = _top_n_flat_indices(cal, 20, exclude_diagonal=exclude_diagonal)
    gen_top50 = _top_n_flat_indices(cal, 50, exclude_diagonal=exclude_diagonal)
    return {
        "nonzero_ratio_x_real": nz / max(real_nz, 1e-12),
        "total_mass_ratio": total / max(real_total, 1.0),
        "row_sum_ks_stat": ks_stat_1d(gen_row, real_row),
        "col_sum_ks_stat": ks_stat_1d(gen_col, real_col),
        "top20_pair_overlap": float(len(gen_top20 & real_top20)),
        "top20_pair_overlap_against_top50": float(len(gen_top20 & real_top50)),
        "gen_max": float(cal.max()),
        "gen_mean": float(cal.mean()),
        "entropy": _shannon_entropy(cal),
    }


def evaluate_calibrated_samples(
    samples_cont: np.ndarray,
    real_ref_agg: np.ndarray,
    tau: float,
    scale: float,
    *,
    exclude_diagonal: bool = True,
) -> dict[str, Any]:
    """Calibrate samples and compare against an arbitrary reference aggregate.

    ``real_ref_agg`` may be the train, val, or test aggregate -- the
    function does no leakage check, that contract belongs to the
    caller. The fit (``grid_search_tau_scale``) is what is required to
    be train-only.

    Returns a flat dict of scalars suitable for JSON logging. For
    ``[N, Z, Z]`` inputs each per-sample metric is summarised by
    ``_mean`` / ``_std``; the per-sample arrays are also returned under
    ``<key>_per_sample``.
    """
    cal = apply_threshold_and_scale(samples_cont, tau, scale)
    real_arr = np.asarray(real_ref_agg, dtype=np.float64)
    if real_arr.ndim != 2:
        raise ValueError(
            f"real_ref_agg must be 2-D [Z, Z], got shape {real_arr.shape}"
        )

    real_nz = float((real_arr != 0).mean())
    real_total = float(real_arr.sum())
    real_row = real_arr.sum(axis=-1).ravel()
    real_col = real_arr.sum(axis=-2).ravel()
    real_top20 = _top_n_flat_indices(real_arr, 20, exclude_diagonal=exclude_diagonal)
    real_top50 = _top_n_flat_indices(real_arr, 50, exclude_diagonal=exclude_diagonal)
    real_entropy = _shannon_entropy(real_arr)

    if cal.ndim == 2:
        per = _evaluate_single(
            cal,
            real_arr,
            real_nz,
            real_total,
            real_row,
            real_col,
            real_top20,
            real_top50,
            real_entropy,
            exclude_diagonal,
        )
        out: dict[str, Any] = {
            "tau": float(tau),
            "scale": float(scale),
            "n_samples": 1,
            "real_nonzero_ratio": real_nz,
            "real_total_mass": real_total,
            "real_entropy": real_entropy,
            **per,
        }
        return out

    if cal.ndim == 3:
        per_list = [
            _evaluate_single(
                cal[i],
                real_arr,
                real_nz,
                real_total,
                real_row,
                real_col,
                real_top20,
                real_top50,
                real_entropy,
                exclude_diagonal,
            )
            for i in range(cal.shape[0])
        ]
        keys = list(per_list[0].keys())
        out = {
            "tau": float(tau),
            "scale": float(scale),
            "n_samples": int(cal.shape[0]),
            "real_nonzero_ratio": real_nz,
            "real_total_mass": real_total,
            "real_entropy": real_entropy,
        }
        for k in keys:
            vals = np.array([p[k] for p in per_list], dtype=np.float64)
            out[f"{k}_per_sample"] = vals
            out[f"{k}_mean"] = float(np.nanmean(vals))
            out[f"{k}_std"] = float(np.nanstd(vals))
        return out

    raise ValueError(
        f"evaluate_calibrated_samples expects 2-D or 3-D samples_cont, "
        f"got shape {cal.shape}"
    )


# --- 8. acceptance_verdict ------------------------------------------------


def acceptance_verdict(
    metrics: dict[str, Any],
    *,
    pass_nz_ratio_x_real_band: tuple[float, float] = (0.7, 1.5),
    pass_total_mass_ratio_band: tuple[float, float] = (0.8, 1.2),
    pass_top20_overlap_min: float = 12.0,
    pass_row_ks_max: float = 0.3,
    pass_col_ks_max: float = 0.3,
    failed_diffusion_baseline: dict[str, float] | None = None,
) -> str:
    """Rule-based ``pass`` / ``mild`` / ``fail`` verdict.

    Reads ``<key>_mean`` if present, otherwise the bare ``<key>``, so
    this works on both the per-sample-summary dict from
    ``evaluate_calibrated_samples`` ([N, Z, Z] input) and the single-
    sample variant ([Z, Z] input).

    PASS gates (all must hold):
      * ``nonzero_ratio_x_real`` in ``pass_nz_ratio_x_real_band``
      * ``total_mass_ratio`` in ``pass_total_mass_ratio_band``
      * ``top20_pair_overlap`` >= ``pass_top20_overlap_min``
      * ``row_sum_ks_stat`` <= ``pass_row_ks_max``
      * ``col_sum_ks_stat`` <= ``pass_col_ks_max``

    MILD (any PASS gate fails but every headline axis strictly beats
    ``failed_diffusion_baseline``):
      * ``row_sum_ks_stat`` < baseline
      * ``col_sum_ks_stat`` < baseline
      * ``nonzero_ratio_x_real`` < baseline
      * ``total_mass_ratio`` < baseline

    FAIL otherwise.
    """
    if failed_diffusion_baseline is None:
        failed_diffusion_baseline = DEFAULT_FAILED_DIFFUSION_BASELINE

    def _get(key: str) -> float:
        if f"{key}_mean" in metrics:
            return float(metrics[f"{key}_mean"])
        if key in metrics:
            return float(metrics[key])
        raise KeyError(
            f"metrics dict is missing both {key!r} and {key + '_mean'!r}"
        )

    nz_x = _get("nonzero_ratio_x_real")
    mass_r = _get("total_mass_ratio")
    top20 = _get("top20_pair_overlap")
    row_ks = _get("row_sum_ks_stat")
    col_ks = _get("col_sum_ks_stat")

    pass_gates = [
        pass_nz_ratio_x_real_band[0] <= nz_x <= pass_nz_ratio_x_real_band[1],
        pass_total_mass_ratio_band[0] <= mass_r <= pass_total_mass_ratio_band[1],
        top20 >= pass_top20_overlap_min,
        row_ks <= pass_row_ks_max,
        col_ks <= pass_col_ks_max,
    ]
    if all(pass_gates):
        return "pass"

    mild_gates = [
        row_ks < failed_diffusion_baseline["row_sum_ks_stat"],
        col_ks < failed_diffusion_baseline["col_sum_ks_stat"],
        nz_x < failed_diffusion_baseline["nonzero_ratio_x_real"],
        mass_r < failed_diffusion_baseline["total_mass_ratio"],
    ]
    if all(mild_gates):
        return "mild"
    return "fail"
