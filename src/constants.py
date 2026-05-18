"""Project-wide constants. Mirrors CLAUDE.md section 7.

Note: ``NUM_TIME_BINS`` was revised 336 -> 528 in Stage 3 — the OD time
window moved from 7 days to 11 full calendar days. See docs/decisions.md
2026-05-18.
"""
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

# Time discretization. TIME_BIN_MIN is the OD slot width in minutes.
# NUM_TIME_BINS was revised from 336 (planned 7-day window) to 528 after
# Stage-3 R1.5 found the data spans 12.45 days; the OD tensor now covers
# 11 full calendar days (2023-07-10 .. 2023-07-21). configs/od.yaml::time
# is the source of truth for the window; this constant mirrors it for
# code paths that run without a config loaded. See docs/decisions.md.
TIME_BIN_MIN: int = 30
NUM_TIME_BINS: int = 11 * 24 * 60 // TIME_BIN_MIN  # = 528 (was 336 for 7 days)

# eVTOL trip eligibility thresholds (refined in Stage 3 sensitivity).
EVTOL_MIN_DIST_KM: float = 15.0
EVTOL_MIN_DURATION_MIN: float = 25.0

# Spatial discretization.
H3_RESOLUTION: int = 7  # ~1.2 km hex edge.

# Vertiport placement.
DEFAULT_K: int = 10  # sweep 5..20 in sensitivity analysis.
WALK_RADIUS_KM: float = 5.0
