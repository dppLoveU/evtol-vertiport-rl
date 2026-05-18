"""Stage-2 task 5: spatial helper matrices.

Builds the zone-to-zone and zone-to-candidate haversine distance matrices
and the candidate coverage mask consumed by Stages 3, 5, 6.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.constants import WALK_RADIUS_KM
from src.utils.geo import haversine_km


def build_matrices(
    zones_gdf: pd.DataFrame,
    cands_gdf: pd.DataFrame,
    walk_radius_km: float = WALK_RADIUS_KM,
) -> dict[str, Any]:
    """Build the stage-2 distance and coverage matrices.

    ``zones_gdf`` must carry ``centroid_lon`` / ``centroid_lat``;
    ``cands_gdf`` must carry ``lon`` / ``lat``. Geometry columns are not
    used. Distances are great-circle (haversine) km.

    Returns a dict with:

    - ``dist_zone_zone``   ``[|Z|, |Z|]`` float32, symmetric, zero diagonal
    - ``dist_zone_cand``   ``[|Z|, |C|]`` float32
    - ``cand_covers_zones`` ``[|C|, |Z|]`` bool,
      ``True`` where ``dist_zone_cand[z, c] <= walk_radius_km``
    - ``n_zones``, ``n_candidates``, ``walk_radius_km``, ``coverage_ratio``
    """
    zlon = zones_gdf["centroid_lon"].to_numpy(dtype=np.float64)
    zlat = zones_gdf["centroid_lat"].to_numpy(dtype=np.float64)
    clon = cands_gdf["lon"].to_numpy(dtype=np.float64)
    clat = cands_gdf["lat"].to_numpy(dtype=np.float64)
    n_zones = zlon.shape[0]
    n_cand = clon.shape[0]

    # Zone-to-zone: haversine is mathematically symmetric, but enforce it
    # exactly so np.allclose(D, D.T) holds bit-for-bit; force the diagonal
    # to a true zero rather than a tiny rounding residue.
    dzz = haversine_km(zlat[:, None], zlon[:, None], zlat[None, :], zlon[None, :])
    dzz = 0.5 * (dzz + dzz.T)
    np.fill_diagonal(dzz, 0.0)
    dist_zone_zone: NDArray[np.float32] = dzz.astype(np.float32)

    # Zone-to-candidate.
    dzc = haversine_km(zlat[:, None], zlon[:, None], clat[None, :], clon[None, :])
    dist_zone_cand: NDArray[np.float32] = dzc.astype(np.float32)

    # Coverage mask [|C|, |Z|] derived from the float32 distances so the
    # mask matches what a downstream consumer would recompute from disk.
    cand_covers_zones: NDArray[np.bool_] = np.ascontiguousarray(
        (dist_zone_cand.T <= walk_radius_km)
    )

    covered_zones = int(cand_covers_zones.any(axis=0).sum())
    coverage_ratio = covered_zones / n_zones if n_zones else 0.0

    return {
        "dist_zone_zone": dist_zone_zone,
        "dist_zone_cand": dist_zone_cand,
        "cand_covers_zones": cand_covers_zones,
        "n_zones": n_zones,
        "n_candidates": n_cand,
        "walk_radius_km": float(walk_radius_km),
        "coverage_ratio": coverage_ratio,
    }
