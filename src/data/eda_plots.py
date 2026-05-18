"""Reusable EDA plotting helpers for Stage 3 (and reused in Stage 7).

Centralizes the house figure style and the PNG+PDF save routine so every
stage's figures look consistent. Pure matplotlib -- no seaborn package
dependency (the ``seaborn-v0_8-whitegrid`` style is shipped with
matplotlib itself).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def apply_house_style() -> None:
    """Apply the project-wide figure style. Call once before plotting.

    Forces the Agg backend (headless) and sets every font size to >= 10
    pt for print readability.
    """
    plt.switch_backend("Agg")
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:  # pragma: no cover - very old matplotlib
        plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def save_figure(fig: Figure, out_dir: Path, name: str, dpi: int = 300) -> list[Path]:
    """Save ``fig`` as both PNG (``dpi``) and PDF under ``out_dir``.

    Returns the two written paths and closes the figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def zone_choropleth(
    zones_gdf: gpd.GeoDataFrame,
    values: np.ndarray,
    *,
    title: str,
    legend_label: str,
    cmap: str = "YlOrRd",
) -> Figure:
    """A zone-polygon choropleth (no basemap; keeps dependencies light).

    ``values`` must be ordered by ``zone_id`` -- i.e. the row order of
    ``zones_gdf`` after a ``zone_id`` sort.
    """
    gdf = zones_gdf.copy()
    gdf["_value"] = np.asarray(values)
    fig, ax = plt.subplots(figsize=(7, 7))
    gdf.plot(
        column="_value",
        ax=ax,
        cmap=cmap,
        legend=True,
        legend_kwds={"label": legend_label, "shrink": 0.6},
        edgecolor="white",
        linewidth=0.1,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    return fig
