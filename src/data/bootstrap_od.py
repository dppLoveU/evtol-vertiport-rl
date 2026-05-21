"""Stage-4B-5C PR5C-2A: conditional bootstrap sampler for OD scenarios.

``ConditionalBootstrapSampler`` builds ``[N_ω, |Z|, |Z|]`` aggregate OD
scenarios by day-block bootstrap resampling from the train-split slots
of the real Stage-3 OD tensor. The output matches the format the Stage-5
RL environment expects for the diffusion-augmentation channel (per-
scenario, time-aggregated OD; see ``docs/plan/stage5_rl_env.md`` and
``docs/plan/stage4_diffusion.md`` "Outputs -> od_samples_agg.npy"), and
this sampler is the documented Stage-4 fallback for the case where the
diffusion model fails to produce usable scenarios (see
``docs/plan/stage4_diffusion.md`` "Robustness Note" and
``docs/decisions.md`` 2026-05-20 PR5B-3b-3 "bootstrap / resampling
baseline as a Stage-4 fallback").

Strict no-leak contract
-----------------------
* The sampler only reads slot indices that belong to ``train_slots``.
  val/test slots are never indexed even transitively. The class
  exposes ``used_slots`` after ``sample()`` so callers and tests can
  assert this empirically.
* ``train_slots`` must form a whole number of complete day blocks of
  length ``slots_per_day``. A partial day block raises ``ValueError``
  so a silent off-by-one in the caller's split cannot turn into a
  silent leak.

Mode coverage (PR5C-2A only)
----------------------------
* ``mode="day_block"`` is the only mode in this PR. Each scenario
  draws ``n_days_per_scenario`` day blocks WITH REPLACEMENT from the
  train day blocks and sums the OD slices of those blocks over the
  time axis. The result is an ``[|Z|, |Z|]`` int32 nonnegative matrix.
* No Poisson perturbation or demand scaling. Those are deferred to a
  later sub-PR (PR5C-2B) and are deliberately not implemented here.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Supported bootstrap modes. PR5C-2A ships only "day_block".
VALID_MODES: tuple[str, ...] = ("day_block",)


def split_train_slots_into_day_blocks(
    train_slots: Sequence[int] | range, slots_per_day: int
) -> list[list[int]]:
    """Split ``train_slots`` into ``slots_per_day``-length day blocks.

    The blocks are formed by taking the input slot list in its given
    order and chopping it into consecutive chunks of ``slots_per_day``
    indices. The slots are not re-sorted: the caller is responsible for
    passing a sequence that already groups into days, which matches the
    Stage-4 split convention where train days are contiguous and
    ``train_slots`` is built via ``range(day_lo*slots_per_day,
    day_hi*slots_per_day)``.

    Parameters
    ----------
    train_slots : Sequence[int] | range
        Train-split slot indices into the Stage-3 OD tensor.
    slots_per_day : int
        Number of slots per calendar day (e.g. 48 for 30-min bins).

    Returns
    -------
    list[list[int]]
        ``len(train_slots) // slots_per_day`` lists, each of length
        ``slots_per_day``.

    Raises
    ------
    ValueError
        If ``slots_per_day`` is not positive, ``train_slots`` is empty,
        or the number of train slots is not a multiple of
        ``slots_per_day`` (i.e. an incomplete trailing day block).
    """
    if slots_per_day <= 0:
        raise ValueError(f"slots_per_day must be positive, got {slots_per_day}")
    slots = list(train_slots)
    if len(slots) == 0:
        raise ValueError("train_slots is empty -- no day blocks to build")
    if len(slots) % slots_per_day != 0:
        raise ValueError(
            f"train_slots length {len(slots)} is not a multiple of "
            f"slots_per_day={slots_per_day}; cannot form complete day blocks"
        )
    n_blocks = len(slots) // slots_per_day
    return [
        slots[i * slots_per_day : (i + 1) * slots_per_day] for i in range(n_blocks)
    ]


def aggregate_day_blocks(
    od: np.ndarray, selected_blocks: Sequence[Sequence[int]]
) -> np.ndarray:
    """Sum OD slices over the slot indices of the given day blocks.

    Parameters
    ----------
    od : np.ndarray | np.memmap
        OD tensor of shape ``[T, |Z|, |Z|]`` with nonnegative integer
        counts. Memory-mapped arrays are supported -- only the slots
        named in ``selected_blocks`` are read.
    selected_blocks : Sequence[Sequence[int]]
        Each inner sequence is one day block's slot indices. Repeated
        blocks contribute their sum multiple times (with-replacement
        bootstrap is the caller's responsibility).

    Returns
    -------
    np.ndarray
        Shape ``[|Z|, |Z|]``, dtype int32.

    Raises
    ------
    ValueError
        If ``selected_blocks`` is empty.
    """
    if len(selected_blocks) == 0:
        raise ValueError("selected_blocks is empty -- nothing to aggregate")
    z = od.shape[-1]
    # int64 accumulator avoids overflow on dense slices; safe to cast
    # back to int32 at the end because eVTOL OD counts per slot are
    # bounded well below the int32 range.
    acc = np.zeros((z, z), dtype=np.int64)
    for block in selected_blocks:
        for s in block:
            acc += np.asarray(od[s], dtype=np.int64)
    return acc.astype(np.int32)


def compute_bootstrap_summary(samples: np.ndarray) -> dict[str, float]:
    """Diagnostic per-scenario summary statistics.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``[N_ω, |Z|, |Z|]``, int32 (the output of
        ``ConditionalBootstrapSampler.sample()``).

    Returns
    -------
    dict[str, float]
        Aggregate stats useful for logging: per-scenario total mass
        mean/std, per-scenario nonzero-ratio mean/std, global min/max.
    """
    if samples.ndim != 3:
        raise ValueError(f"expected 3-D samples [N_omega, Z, Z], got {samples.shape}")
    flat = samples.reshape(samples.shape[0], -1)
    sums = flat.sum(axis=1)
    nz_ratios = (flat != 0).mean(axis=1)
    return {
        "n_omega": int(samples.shape[0]),
        "z": int(samples.shape[1]),
        "total_sum_mean": float(sums.mean()),
        "total_sum_std": float(sums.std()),
        "nonzero_ratio_mean": float(nz_ratios.mean()),
        "nonzero_ratio_std": float(nz_ratios.std()),
        "global_min": int(samples.min()),
        "global_max": int(samples.max()),
    }


class ConditionalBootstrapSampler:
    """Day-block bootstrap sampler over the train slots of an OD tensor.

    Parameters
    ----------
    od : np.ndarray | np.memmap
        Shape ``[T, |Z|, |Z|]`` int OD tensor (Stage-3 output).
    train_slots : Sequence[int] | range
        Sorted slot indices belonging to the train split. Must form a
        whole number of complete day blocks of length ``slots_per_day``.
    slots_per_day : int, default 48
        Slots per calendar day (30-min bins -> 48).
    n_days_per_scenario : int, default 11
        Number of day blocks drawn (with replacement) per scenario.
    n_omega : int, default 64
        Number of scenarios produced by one ``sample()`` call.
    seed : int, default 0
        Seed for this sampler's ``np.random.Generator``. Two samplers
        with the same ``seed``, ``train_slots``, and other parameters
        produce bit-identical output.
    mode : str, default "day_block"
        Only ``"day_block"`` is implemented in PR5C-2A.

    Attributes populated by ``sample()``
    -----------------------------------
    selected_day_indices : np.ndarray | None
        Shape ``[n_omega, n_days_per_scenario]`` int32. Index into
        ``self.day_blocks`` (NOT into ``od``); used by tests asserting
        only train day-blocks were drawn.
    used_slots : set[int] | None
        Union of slot indices across all drawn blocks. Always a
        subset of ``train_slots``.
    """

    def __init__(
        self,
        od: np.ndarray,
        train_slots: Sequence[int] | range,
        *,
        slots_per_day: int = 48,
        n_days_per_scenario: int = 11,
        n_omega: int = 64,
        seed: int = 0,
        mode: str = "day_block",
    ) -> None:
        if od.ndim != 3:
            raise ValueError(f"od must be 3-D [T, Z, Z], got shape {od.shape}")
        if od.shape[-1] != od.shape[-2]:
            raise ValueError(
                f"od last two dims must match (square Z x Z), got {od.shape}"
            )
        if n_days_per_scenario <= 0:
            raise ValueError(
                f"n_days_per_scenario must be positive, got {n_days_per_scenario}"
            )
        if n_omega <= 0:
            raise ValueError(f"n_omega must be positive, got {n_omega}")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

        self._od = od
        self.train_slots: list[int] = list(train_slots)
        self.slots_per_day = int(slots_per_day)
        self.n_days_per_scenario = int(n_days_per_scenario)
        self.n_omega = int(n_omega)
        self.seed = int(seed)
        self.mode = mode

        # Built once at construction so a partial day block fails fast,
        # before any sample call.
        self.day_blocks: list[list[int]] = split_train_slots_into_day_blocks(
            self.train_slots, self.slots_per_day
        )
        # Bounds-check the train slot range against the OD tensor so a
        # caller mis-specifying train_slots fails before we read.
        t = od.shape[0]
        if any(s < 0 or s >= t for s in self.train_slots):
            raise ValueError(
                f"train_slots contains indices outside [0, {t}); "
                "check the dataset split"
            )
        self._train_slot_set: set[int] = set(self.train_slots)

        self.selected_day_indices: np.ndarray | None = None
        self.used_slots: set[int] | None = None

    def sample(self) -> np.ndarray:
        """Draw ``n_omega`` aggregate OD scenarios.

        Returns
        -------
        np.ndarray
            Shape ``[n_omega, |Z|, |Z|]``, dtype int32, nonnegative.

        Notes
        -----
        Side effects: ``self.selected_day_indices`` and
        ``self.used_slots`` are set, and remain available for inspection
        by tests / diagnostics until the next ``sample()`` call.
        """
        rng = np.random.default_rng(self.seed)
        n_blocks = len(self.day_blocks)
        selected = rng.integers(
            low=0,
            high=n_blocks,
            size=(self.n_omega, self.n_days_per_scenario),
        ).astype(np.int32)

        z = self._od.shape[-1]
        out = np.zeros((self.n_omega, z, z), dtype=np.int32)
        used: set[int] = set()
        for i in range(self.n_omega):
            scenario_blocks = [self.day_blocks[j] for j in selected[i]]
            out[i] = aggregate_day_blocks(self._od, scenario_blocks)
            for block in scenario_blocks:
                used.update(block)

        leaked = used - self._train_slot_set
        if leaked:
            # Defensive: should be impossible because day_blocks is
            # derived entirely from train_slots; raise loudly if it
            # ever happens.
            raise RuntimeError(
                f"internal: sampler used non-train slots {sorted(leaked)[:5]}"
            )
        if out.min() < 0:
            raise RuntimeError("aggregated samples contain negative entries")

        self.selected_day_indices = selected
        self.used_slots = used
        return out
