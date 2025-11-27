"""Visualization utilities for urban net returns."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from rich.console import Console

from urban_rents.config import (
    CONTIGUOUS_US_FIPS,
    CRS_ALBERS,
    FIGURES_DIR,
    RAW_DIR,
    TIGER_YEAR,
)

console = Console()


def load_county_geometries() -> gpd.GeoDataFrame:
    """
    Load county geometries for visualization.

    Returns:
        GeoDataFrame with county geometries
    """
    shp_path = RAW_DIR / "shapefiles" / "county" / f"tl_{TIGER_YEAR}_us_county.shp"

    if not shp_path.exists():
        raise FileNotFoundError(f"County shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)

    # Filter to contiguous US
    gdf = gdf[gdf["STATEFP"].isin(CONTIGUOUS_US_FIPS)].copy()

    # Project to Albers Equal Area
    gdf = gdf.to_crs(CRS_ALBERS)

    # Rename for consistency
    gdf = gdf.rename(columns={"GEOID": "county_geoid"})

    return gdf


def create_choropleth(
    data: pd.DataFrame,
    value_column: str,
    title: str,
    filename: str,
    cmap: str = "YlOrRd",
    n_classes: int = 7,
    legend_label: str | None = None,
    log_scale: bool = False,
) -> Path:
    """
    Create a choropleth map of county-level data.

    Args:
        data: DataFrame with county_geoid and value column
        value_column: Column name to visualize
        title: Map title
        filename: Output filename
        cmap: Matplotlib colormap name
        n_classes: Number of color classes
        legend_label: Label for colorbar (defaults to value_column)
        log_scale: If True, use log scale for values

    Returns:
        Path to saved figure
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load county geometries
    counties = load_county_geometries()

    # Merge data
    merged = counties.merge(data[["county_geoid", value_column]], on="county_geoid", how="left")

    # Handle log scale
    if log_scale:
        merged[f"{value_column}_plot"] = np.log10(merged[value_column].clip(lower=1))
        plot_col = f"{value_column}_plot"
    else:
        plot_col = value_column

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Calculate class boundaries using quantiles
    valid_values = merged[plot_col].dropna()
    if len(valid_values) == 0:
        console.print(f"[red]No valid values for {value_column}[/red]")
        return None

    boundaries = np.percentile(valid_values, np.linspace(0, 100, n_classes + 1))
    boundaries = np.unique(boundaries)  # Remove duplicates

    if len(boundaries) < 3:
        boundaries = np.linspace(valid_values.min(), valid_values.max(), n_classes + 1)

    norm = BoundaryNorm(boundaries, ncolors=256)

    # Plot counties without data in light gray
    merged[merged[plot_col].isna()].plot(
        ax=ax, color="#d3d3d3", edgecolor="#999999", linewidth=0.1
    )

    # Plot counties with data
    merged[merged[plot_col].notna()].plot(
        column=plot_col,
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="#666666",
        linewidth=0.1,
        legend=True,
        legend_kwds={
            "label": legend_label or value_column,
            "orientation": "horizontal",
            "shrink": 0.6,
            "pad": 0.02,
        },
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    console.print(f"[green]Saved map to {output_path}[/green]")

    return output_path


def create_urban_net_returns_map(data: pd.DataFrame) -> Path:
    """
    Create choropleth map of urban net returns.

    Args:
        data: Urban net returns DataFrame

    Returns:
        Path to saved figure
    """
    return create_choropleth(
        data=data,
        value_column="urban_net_return",
        title="Urban Net Returns ($/acre/year)\n5% Annualized Land Value Returns",
        filename="urban_net_returns_map.png",
        cmap="YlOrRd",
        legend_label="$/acre/year",
        log_scale=True,
    )


def create_property_value_map(data: pd.DataFrame) -> Path:
    """
    Create choropleth map of property values.

    Args:
        data: Urban net returns DataFrame with property values

    Returns:
        Path to saved figure
    """
    return create_choropleth(
        data=data,
        value_column="mean_property_value",
        title="Mean Property Value of Recently Built Homes ($)\nOwner-Occupied Single-Family Homes Built 2019-2023",
        filename="property_values_map.png",
        cmap="Blues",
        legend_label="$",
        log_scale=True,
    )


def create_comparison_map(data: pd.DataFrame) -> Path:
    """
    Create side-by-side comparison of standard vs adjusted returns.

    Args:
        data: Urban net returns DataFrame with both return columns

    Returns:
        Path to saved figure
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load county geometries
    counties = load_county_geometries()

    # Merge data
    merged = counties.merge(
        data[["county_geoid", "urban_net_return", "urban_net_return_adjusted"]],
        on="county_geoid",
        how="left",
    )

    # Use log scale
    merged["nr_log"] = np.log10(merged["urban_net_return"].clip(lower=1))
    merged["nr_adj_log"] = np.log10(merged["urban_net_return_adjusted"].clip(lower=1))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Calculate common scale
    valid_values = pd.concat([merged["nr_log"], merged["nr_adj_log"]]).dropna()
    vmin, vmax = valid_values.quantile(0.02), valid_values.quantile(0.98)

    # Plot standard returns
    merged[merged["nr_log"].notna()].plot(
        column="nr_log",
        ax=ax1,
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        edgecolor="#666666",
        linewidth=0.1,
    )
    merged[merged["nr_log"].isna()].plot(ax=ax1, color="#d3d3d3", edgecolor="#999999", linewidth=0.1)
    ax1.set_title("Standard Urban Net Returns", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Plot adjusted returns
    im = merged[merged["nr_adj_log"].notna()].plot(
        column="nr_adj_log",
        ax=ax2,
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        edgecolor="#666666",
        linewidth=0.1,
    )
    merged[merged["nr_adj_log"].isna()].plot(ax=ax2, color="#d3d3d3", edgecolor="#999999", linewidth=0.1)
    ax2.set_title("PUMA-Size Adjusted Returns", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Add colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        ax=[ax1, ax2],
        orientation="horizontal",
        shrink=0.4,
        pad=0.02,
    )
    cbar.set_label("Log10($/acre/year)")

    plt.suptitle(
        "Urban Net Returns: Standard vs PUMA-Size Adjusted",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Save figure
    output_path = FIGURES_DIR / "returns_comparison_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    console.print(f"[green]Saved comparison map to {output_path}[/green]")

    return output_path


def create_quality_flag_map(data: pd.DataFrame) -> Path:
    """
    Create map showing data quality flags.

    Args:
        data: Urban net returns DataFrame with quality flags

    Returns:
        Path to saved figure
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load county geometries
    counties = load_county_geometries()

    # Merge data
    merged = counties.merge(data[["county_geoid", "data_quality_flag"]], on="county_geoid", how="left")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Color mapping for quality flags
    colors = {"good": "#2ecc71", "caution": "#f39c12", "poor": "#e74c3c"}

    # Plot each category
    for flag, color in colors.items():
        subset = merged[merged["data_quality_flag"] == flag]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor="#666666", linewidth=0.1, label=flag)

    # Plot missing data
    missing = merged[merged["data_quality_flag"].isna()]
    if len(missing) > 0:
        missing.plot(ax=ax, color="#d3d3d3", edgecolor="#999999", linewidth=0.1, label="No data")

    ax.set_title(
        "Data Quality Flags\nBased on PUMA Coverage and Weights",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()

    # Save figure
    output_path = FIGURES_DIR / "data_quality_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    console.print(f"[green]Saved quality map to {output_path}[/green]")

    return output_path


def create_all_visualizations(data: pd.DataFrame) -> list[Path]:
    """
    Generate all visualization outputs.

    Args:
        data: Urban net returns DataFrame

    Returns:
        List of paths to generated figures
    """
    console.print("[bold]Generating visualizations...[/bold]")

    figures = []

    console.print("[cyan]Creating urban net returns map...[/cyan]")
    figures.append(create_urban_net_returns_map(data))

    console.print("[cyan]Creating property value map...[/cyan]")
    figures.append(create_property_value_map(data))

    console.print("[cyan]Creating comparison map...[/cyan]")
    figures.append(create_comparison_map(data))

    console.print("[cyan]Creating data quality map...[/cyan]")
    figures.append(create_quality_flag_map(data))

    console.print(f"[green]Generated {len(figures)} visualizations[/green]")

    return [f for f in figures if f is not None]
