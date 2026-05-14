"""Geographic helpers."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Mean Earth radius used by the haversine formula.
_EARTH_RADIUS_KM: float = 6371.0


def haversine_km(
    lat1: ArrayLike,
    lon1: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
) -> NDArray[np.float64]:
    """Great-circle distance in kilometers between two lat/lon points.

    Inputs are in degrees and follow numpy broadcasting rules; pass
    scalars or arrays of any compatible shape.
    """
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    # Clip guards against tiny numerical excursions above 1.0 from
    # floating-point rounding, which would produce NaN under arcsin.
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return _EARTH_RADIUS_KM * c
