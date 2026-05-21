"""Stage-5 task 2: the VertiportEnv RL environment (PR1 minimal version).

Gymnasium-style sequential vertiport placement environment. The agent
places ``k_select`` vertiports, one per step; the reward at each step is
the incremental bilateral OD demand newly covered by that placement.

The class subclasses ``gymnasium.Env`` and exposes an ``action_masks()``
method so ``sb3-contrib``'s MaskablePPO can mask already-placed
candidates. The MDP itself (reset / step / reward) is unchanged from the
Stage-5 PR1 scaffold.

Bilateral OD coverage: an OD pair ``(i, j)`` is covered iff both origin
zone ``i`` and destination zone ``j`` lie in the currently covered zone
set. A candidate "covers" every zone within the Stage-2 walk radius
(``cand_covers_zones``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from numpy.typing import NDArray

REPO = Path(__file__).resolve().parents[2]


def _resolve(path_str: str, base_dir: Path) -> Path:
    """Resolve a config path relative to ``base_dir`` if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else base_dir / p


class VertiportEnv(gym.Env):
    """Sequential K-vertiport placement environment.

    Parameters
    ----------
    od_samples_agg:
        ``[n_omega, n_zones, n_zones]`` nonnegative integer OD tensor,
        already aggregated over time. One slice is sampled per episode.
    cand_covers_zones:
        ``[n_candidates, n_zones]`` bool coverage mask -- ``True`` where a
        candidate covers a zone (within the Stage-2 walk radius).
    k_select:
        Number of vertiports placed per episode (episode length).
    normalize:
        If True, reward and coverage are divided by the scenario's total
        OD demand so they live in ``[0, 1]``.
    invalid_action:
        ``"mask"`` -- already-selected candidates are excluded by
        :meth:`action_masks`; stepping such an action raises ``ValueError``.
    seed:
        Base seed for the scenario-sampling RNG.
    """

    def __init__(
        self,
        od_samples_agg: NDArray[np.integer],
        cand_covers_zones: NDArray[np.bool_],
        k_select: int,
        *,
        normalize: bool = True,
        invalid_action: str = "mask",
        seed: int = 42,
    ) -> None:
        super().__init__()
        od = np.ascontiguousarray(od_samples_agg)
        cov = np.ascontiguousarray(cand_covers_zones).astype(bool)

        if od.ndim != 3 or od.shape[1] != od.shape[2]:
            raise ValueError(
                f"od_samples_agg must be [n_omega, Z, Z]; got shape {od.shape}"
            )
        if cov.ndim != 2:
            raise ValueError(
                f"cand_covers_zones must be [n_candidates, Z]; got shape {cov.shape}"
            )
        if cov.shape[1] != od.shape[1]:
            raise ValueError(
                f"zone-count mismatch: od has {od.shape[1]} zones, "
                f"cand_covers_zones has {cov.shape[1]}"
            )
        if invalid_action != "mask":
            raise ValueError(f"unsupported invalid_action: {invalid_action!r}")
        if not 1 <= k_select <= cov.shape[0]:
            raise ValueError(
                f"k_select must be in [1, n_candidates={cov.shape[0]}]; got {k_select}"
            )

        self._od = od
        self._cov = cov
        self.n_omega: int = int(od.shape[0])
        self.n_zones: int = int(od.shape[1])
        self.n_candidates: int = int(cov.shape[0])
        self.k_select: int = int(k_select)
        self.normalize: bool = bool(normalize)
        self.invalid_action: str = invalid_action

        # Gymnasium spaces. The observation is a Dict; scalar fields are
        # shape-(1,) float32 arrays so observation_space.contains(obs)
        # holds (gymnasium Box rejects 0-d / python-scalar entries).
        self.action_space = spaces.Discrete(self.n_candidates)
        self.observation_space = spaces.Dict(
            {
                "selected_mask": spaces.Box(
                    0.0, 1.0, shape=(self.n_candidates,), dtype=np.float32
                ),
                "covered_zones": spaces.Box(
                    0.0, 1.0, shape=(self.n_zones,), dtype=np.float32
                ),
                "remaining_budget": spaces.Box(
                    0.0, float(self.k_select), shape=(1,), dtype=np.float32
                ),
                "current_coverage_ratio": spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Episode state -- populated by reset().
        self.scenario_idx: int = -1
        self._m_total: NDArray[np.int64] = np.zeros((0, 0), dtype=np.int64)
        self._total_demand: int = 0
        self._selected_mask: NDArray[np.bool_] = np.zeros(0, dtype=bool)
        self._covered_zones: NDArray[np.bool_] = np.zeros(0, dtype=bool)
        self._selected: list[int] = []
        self._covered_demand: int = 0

    @classmethod
    def from_config(
        cls,
        config: str | Path | dict[str, Any],
        base_dir: Path | None = None,
    ) -> VertiportEnv:
        """Build a VertiportEnv from ``configs/env.yaml``.

        ``config`` may be a path to the yaml file or an already-parsed
        config dict. Loads the frozen scenario tensor and the Stage-2
        coverage mask, validates their shapes against the config
        dimensions, and returns the constructed environment. Relative
        data paths are resolved against ``base_dir`` (default: repo root).
        """
        if isinstance(config, (str, Path)):
            with open(config) as fh:
                cfg = yaml.safe_load(fh)
        else:
            cfg = config

        base = base_dir if base_dir is not None else REPO
        od_path = _resolve(cfg["scenario_source"], base)
        cov_path = _resolve(cfg["cand_covers_zones_path"], base)

        od = np.load(od_path)
        cov = np.load(cov_path)

        n_zones = int(cfg["n_zones"])
        n_candidates = int(cfg["n_candidates"])
        n_omega = int(cfg["n_omega"])
        if od.shape != (n_omega, n_zones, n_zones):
            raise ValueError(
                f"{od_path.name} shape {od.shape} != expected "
                f"({n_omega}, {n_zones}, {n_zones})"
            )
        if cov.shape != (n_candidates, n_zones):
            raise ValueError(
                f"{cov_path.name} shape {cov.shape} != expected "
                f"({n_candidates}, {n_zones})"
            )
        if od.min() < 0:
            raise ValueError(f"{od_path.name} has negative entries")

        env_cfg = cfg.get("env", {})
        return cls(
            od,
            cov,
            k_select=int(cfg["k_select"]),
            normalize=bool(cfg["reward"]["normalize"]),
            invalid_action=str(env_cfg.get("invalid_action", "mask")),
            seed=int(env_cfg.get("seed", cfg.get("seed", 42))),
        )

    # -- core API -----------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode.

        Samples one scenario index from ``[0, n_omega)``. If ``seed`` is
        given the RNG is re-seeded first, so the sampled ``scenario_idx``
        is reproducible across resets with the same seed.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.scenario_idx = int(self._rng.integers(0, self.n_omega))
        # int64 view keeps demand sums exact and avoids int32 overflow.
        self._m_total = self._od[self.scenario_idx].astype(np.int64)
        self._total_demand = int(self._m_total.sum())

        self._selected_mask = np.zeros(self.n_candidates, dtype=bool)
        self._covered_zones = np.zeros(self.n_zones, dtype=bool)
        self._selected = []
        self._covered_demand = 0

        return self._get_obs(), self._get_info(incremental_gain=0.0)

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Place a vertiport at candidate ``action``.

        Returns ``(obs, reward, terminated, truncated, info)``. Stepping a
        masked (already-selected) action raises ``ValueError``.
        """
        action = int(action)
        if not 0 <= action < self.n_candidates:
            raise ValueError(
                f"action {action} out of range [0, {self.n_candidates})"
            )
        if self._selected_mask[action]:
            raise ValueError(
                f"invalid action {action}: candidate already selected "
                f"(use action_masks() to filter)"
            )

        prev_demand = self._covered_demand

        self._selected_mask[action] = True
        self._selected.append(action)
        self._covered_zones |= self._cov[action]

        # Bilateral coverage: OD pair (i, j) covered iff both zones are in
        # the covered set. np.ix_ accepts boolean masks; an empty covered
        # set yields a 0x0 slice that sums to 0.
        covered_demand = int(
            self._m_total[np.ix_(self._covered_zones, self._covered_zones)].sum()
        )
        self._covered_demand = covered_demand

        gained = covered_demand - prev_demand
        if self.normalize and self._total_demand > 0:
            reward = gained / self._total_demand
        else:
            reward = float(gained)

        terminated = len(self._selected) >= self.k_select
        truncated = False
        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(incremental_gain=float(gained)),
        )

    def action_masks(self) -> NDArray[np.bool_]:
        """Return the ``[n_candidates]`` bool mask: True = selectable."""
        return ~self._selected_mask

    # -- observation / info -------------------------------------------

    @property
    def coverage_ratio(self) -> float:
        """Fraction of total OD demand currently covered (0 if total 0)."""
        if self._total_demand <= 0:
            return 0.0
        return self._covered_demand / self._total_demand

    def _get_obs(self) -> dict[str, Any]:
        return {
            "selected_mask": self._selected_mask.astype(np.float32),
            "covered_zones": self._covered_zones.astype(np.float32),
            "remaining_budget": np.array(
                [self.k_select - len(self._selected)], dtype=np.float32
            ),
            "current_coverage_ratio": np.array(
                [self.coverage_ratio], dtype=np.float32
            ),
        }

    def _get_info(self, incremental_gain: float) -> dict[str, Any]:
        return {
            "scenario_idx": self.scenario_idx,
            "selected_count": len(self._selected),
            "total_covered_demand": self._covered_demand,
            "coverage_ratio": self.coverage_ratio,
            "incremental_gain": incremental_gain,
            "selected_candidates": list(self._selected),
        }
