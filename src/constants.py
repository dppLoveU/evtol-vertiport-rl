"""Project-wide constants. Mirrors CLAUDE.md section 7 verbatim."""
from __future__ import annotations

# Suzhou metropolitan area bounding box. Covers Suzhou City + the four
# county-level cities (Kunshan, Changshu, Zhangjiagang, Taicang). Refined
# 2026-05-15 from per-area p1/p99 bounds + ~2 km padding; see
# results/stage1/eda/area_bounds.csv and docs/decisions.md.
SUZHOU_BBOX: dict[str, float] = {
    "lon_min": 120.37,
    "lon_max": 121.33,
    "lat_min": 30.88,
    "lat_max": 32.01,
}

# Raw coordinates are stored as int (e.g. 120557806 = 120.557806 deg).
COORD_SCALE: float = 1e6

# Time discretization.
TIME_BIN_MIN: int = 30
NUM_TIME_BINS: int = 7 * 24 * 60 // TIME_BIN_MIN  # = 336

# eVTOL trip eligibility thresholds (refined in Stage 3 sensitivity).
EVTOL_MIN_DIST_KM: float = 15.0
EVTOL_MIN_DURATION_MIN: float = 25.0

# Spatial discretization.
H3_RESOLUTION: int = 7  # ~1.2 km hex edge.

# Vertiport placement.
DEFAULT_K: int = 10  # sweep 5..20 in sensitivity analysis.
WALK_RADIUS_KM: float = 5.0
