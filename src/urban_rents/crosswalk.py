"""PUMA-to-county crosswalk generation using area-weighted spatial overlay."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track

from urban_rents.config import (
    CONTIGUOUS_US_FIPS,
    CRS_ALBERS,
    DEFAULT_SURVEY_YEAR,
    PROCESSED_DIR,
    RAW_DIR,
    TIGER_YEAR,
    get_puma_vintage,
    get_tiger_year,
)

console = Console()


def load_puma_shapefile(
    state_fips: str,
    survey_year: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Load PUMA shapefile for a single state, appropriate for the survey year.

    Args:
        state_fips: 2-digit state FIPS code
        survey_year: Survey year to determine PUMA vintage. If None, uses default.

    Returns:
        GeoDataFrame with PUMA geometries
    """
    if survey_year is None:
        survey_year = DEFAULT_SURVEY_YEAR

    vintage = get_puma_vintage(survey_year)
    tiger_year = get_tiger_year(survey_year)

    # Determine PUMA suffix based on vintage
    puma_suffix_map = {"2000": "puma00", "2010": "puma10", "2020": "puma20"}
    puma_suffix = puma_suffix_map.get(vintage, "puma20")

    # Try vintage-specific directory first (new structure)
    shp_path = (
        RAW_DIR / "shapefiles" / f"puma_{vintage}" /
        f"tl_{tiger_year}_{state_fips}_{puma_suffix}.shp"
    )

    # Fall back to legacy flat structure
    if not shp_path.exists():
        shp_path = RAW_DIR / "shapefiles" / "puma" / f"tl_{tiger_year}_{state_fips}_{puma_suffix}.shp"

    # Also try TIGER_YEAR default
    if not shp_path.exists():
        shp_path = RAW_DIR / "shapefiles" / "puma" / f"tl_{TIGER_YEAR}_{state_fips}_puma20.shp"

    if not shp_path.exists():
        raise FileNotFoundError(
            f"PUMA shapefile not found: {shp_path}. "
            f"Looking for {vintage} PUMAs (TIGER {tiger_year})"
        )

    gdf = gpd.read_file(shp_path)

    # Column names differ by vintage
    if vintage == "2020":
        gdf = gdf.rename(columns={
            "STATEFP20": "state_fips",
            "PUMACE20": "puma",
        })
    elif vintage == "2010":
        gdf = gdf.rename(columns={
            "STATEFP10": "state_fips",
            "PUMACE10": "puma",
        })
    else:
        # 2000 PUMAs
        gdf = gdf.rename(columns={
            "STATEFP00": "state_fips",
            "PUMACE00": "puma",
        })

    # Ensure string types with proper padding
    gdf["state_fips"] = gdf["state_fips"].astype(str).str.zfill(2)
    gdf["puma"] = gdf["puma"].astype(str).str.zfill(5)

    # Create combined identifier
    gdf["state_puma"] = gdf["state_fips"] + "_" + gdf["puma"]

    # Track vintage
    gdf["puma_vintage"] = vintage

    return gdf


def load_county_shapefile(tiger_year: int | None = None) -> gpd.GeoDataFrame:
    """
    Load national county shapefile.

    Args:
        tiger_year: TIGER year. If None, uses default TIGER_YEAR.

    Returns:
        GeoDataFrame with county geometries
    """
    year = tiger_year if tiger_year is not None else TIGER_YEAR
    shp_path = RAW_DIR / "shapefiles" / "county" / f"tl_{year}_us_county.shp"

    # Fall back to other available county shapefiles if requested year not found
    # County boundaries change minimally between years
    if not shp_path.exists():
        county_dir = RAW_DIR / "shapefiles" / "county"
        if county_dir.exists():
            available = list(county_dir.glob("tl_*_us_county.shp"))
            if available:
                shp_path = available[0]
                console.print(f"[yellow]Using {shp_path.name} instead of year {year}[/yellow]")

    if not shp_path.exists():
        raise FileNotFoundError(f"County shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)

    # Standardize column names
    gdf = gdf.rename(columns={
        "STATEFP": "state_fips",
        "COUNTYFP": "county_fips",
        "NAME": "county_name",
        "GEOID": "county_geoid",
    })

    # Ensure string types with proper padding
    gdf["state_fips"] = gdf["state_fips"].astype(str).str.zfill(2)
    gdf["county_fips"] = gdf["county_fips"].astype(str).str.zfill(3)

    # Filter to contiguous US
    gdf = gdf[gdf["state_fips"].isin(CONTIGUOUS_US_FIPS)].copy()

    return gdf


def build_crosswalk_for_state(
    state_fips: str,
    puma_gdf: gpd.GeoDataFrame,
    county_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Build PUMA-to-county crosswalk for a single state using area-weighted overlay.

    Args:
        state_fips: 2-digit state FIPS code
        puma_gdf: GeoDataFrame with PUMA geometries (should be projected)
        county_gdf: GeoDataFrame with county geometries (should be projected)

    Returns:
        DataFrame with crosswalk weights
    """
    # Filter to state
    state_pumas = puma_gdf[puma_gdf["state_fips"] == state_fips].copy()
    state_counties = county_gdf[county_gdf["state_fips"] == state_fips].copy()

    if len(state_pumas) == 0 or len(state_counties) == 0:
        return pd.DataFrame()

    # Calculate PUMA areas for later use
    state_pumas["puma_area"] = state_pumas.geometry.area

    # Get vintage for tracking
    vintage = state_pumas["puma_vintage"].iloc[0] if "puma_vintage" in state_pumas.columns else "unknown"

    # Perform spatial overlay (intersection)
    try:
        overlay = gpd.overlay(state_counties, state_pumas, how="intersection", keep_geom_type=False)
    except Exception as e:
        console.print(f"[red]Overlay failed for state {state_fips}: {e}[/red]")
        return pd.DataFrame()

    if len(overlay) == 0:
        return pd.DataFrame()

    # Handle column name conflicts from overlay (state_fips_1, state_fips_2)
    # Use state_fips_1 (from counties) as the canonical state_fips
    if "state_fips_1" in overlay.columns:
        overlay["state_fips"] = overlay["state_fips_1"]
    elif "state_fips" not in overlay.columns:
        overlay["state_fips"] = state_fips

    # Calculate intersection areas
    overlay["intersection_area"] = overlay.geometry.area

    # Calculate total county area from intersections
    county_areas = overlay.groupby("county_geoid")["intersection_area"].sum().reset_index()
    county_areas = county_areas.rename(columns={"intersection_area": "total_county_area"})

    # Merge back to get weights
    overlay = overlay.merge(county_areas, on="county_geoid")

    # Calculate weight as proportion of county area covered by each PUMA
    overlay["weight"] = overlay["intersection_area"] / overlay["total_county_area"]

    # Build crosswalk dataframe - select columns that exist
    crosswalk = pd.DataFrame({
        "state_fips": overlay["state_fips"],
        "county_fips": overlay["county_fips"],
        "county_geoid": overlay["county_geoid"],
        "county_name": overlay["county_name"],
        "puma": overlay["puma"],
        "state_puma": overlay["state_puma"],
        "puma_area": overlay["puma_area"],
        "weight": overlay["weight"],
        "puma_vintage": vintage,
    })

    # Filter out negligible weights (less than 0.001 or 0.1%)
    crosswalk = crosswalk[crosswalk["weight"] > 0.001].copy()

    # Re-normalize weights within each county
    county_totals = crosswalk.groupby("county_geoid")["weight"].transform("sum")
    crosswalk["weight"] = crosswalk["weight"] / county_totals

    # Verify weights sum to 1 for each county
    weight_sums = crosswalk.groupby("county_geoid")["weight"].sum()
    if not np.allclose(weight_sums, 1.0, rtol=0.01):
        console.print(f"[yellow]Warning: Some county weights don't sum to 1 for state {state_fips}[/yellow]")

    return crosswalk


def build_full_crosswalk(survey_year: int | None = None) -> pd.DataFrame:
    """
    Build PUMA-to-county crosswalk for all contiguous US states.

    Args:
        survey_year: Survey year to determine PUMA vintage. If None, uses default.

    Returns:
        DataFrame with complete crosswalk
    """
    if survey_year is None:
        survey_year = DEFAULT_SURVEY_YEAR

    vintage = get_puma_vintage(survey_year)
    tiger_year = get_tiger_year(survey_year)

    console.print(f"[bold]Building PUMA-to-county crosswalk ({vintage} PUMAs)...[/bold]")

    # Load county shapefile
    county_gdf = load_county_shapefile(tiger_year)
    county_gdf = county_gdf.to_crs(CRS_ALBERS)  # Project to equal-area for accurate calculations

    all_crosswalks = []
    states = sorted(CONTIGUOUS_US_FIPS)

    for state_fips in track(states, description=f"Processing {vintage} crosswalk"):
        try:
            # Load state PUMA shapefile
            puma_gdf = load_puma_shapefile(state_fips, survey_year)
            puma_gdf = puma_gdf.to_crs(CRS_ALBERS)

            # Build crosswalk for this state
            state_crosswalk = build_crosswalk_for_state(state_fips, puma_gdf, county_gdf)

            if len(state_crosswalk) > 0:
                all_crosswalks.append(state_crosswalk)

        except FileNotFoundError:
            console.print(f"[red]Shapefile not found for state {state_fips}[/red]")
        except Exception as e:
            console.print(f"[red]Error processing state {state_fips}: {e}[/red]")

    if not all_crosswalks:
        raise ValueError("No crosswalk data generated")

    # Combine all states
    crosswalk = pd.concat(all_crosswalks, ignore_index=True)

    # Calculate statistics for each county
    county_stats = crosswalk.groupby("county_geoid").agg(
        n_pumas=("puma", "count"),
        max_puma_weight=("weight", "max"),
    ).reset_index()

    crosswalk = crosswalk.merge(county_stats, on="county_geoid")

    console.print(f"[green]Built {vintage} crosswalk with {len(crosswalk)} PUMA-county pairs[/green]")
    console.print(f"[green]Covering {crosswalk['county_geoid'].nunique()} counties[/green]")

    return crosswalk


def build_crosswalks_for_panel(survey_years: list[int]) -> dict[str, pd.DataFrame]:
    """
    Build PUMA-to-county crosswalks for all vintages needed by a panel.

    Args:
        survey_years: List of survey end years in the panel

    Returns:
        Dictionary mapping vintage ("2010" or "2020") to crosswalk DataFrame
    """
    # Determine which vintages we need
    vintages_needed = {}
    for year in survey_years:
        vintage = get_puma_vintage(year)
        if vintage not in vintages_needed:
            vintages_needed[vintage] = year  # Store a representative year

    console.print(f"[bold]Building crosswalks for vintages: {list(vintages_needed.keys())}[/bold]")

    crosswalks = {}
    for vintage, rep_year in vintages_needed.items():
        console.print(f"\n[bold cyan]Building {vintage} PUMA crosswalk...[/bold cyan]")
        crosswalks[vintage] = build_full_crosswalk(rep_year)

    return crosswalks


def save_crosswalk(
    df: pd.DataFrame,
    survey_year: int | None = None,
    filename: str | None = None,
) -> Path:
    """
    Save crosswalk to file.

    Args:
        df: Crosswalk DataFrame
        survey_year: Survey year (for naming). If None, uses default name.
        filename: Output filename. If None, auto-generated.

    Returns:
        Path to saved file
    """
    if filename is None:
        if "puma_vintage" in df.columns:
            vintage = df["puma_vintage"].iloc[0]
            filename = f"puma_county_crosswalk_{vintage}.parquet"
        elif survey_year is not None:
            vintage = get_puma_vintage(survey_year)
            filename = f"puma_county_crosswalk_{vintage}.parquet"
        else:
            filename = "puma_county_crosswalk.parquet"

    output_path = PROCESSED_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    console.print(f"[green]Saved crosswalk to {output_path}[/green]")

    return output_path


def load_crosswalk(
    survey_year: int | None = None,
    filename: str | None = None,
) -> pd.DataFrame:
    """
    Load crosswalk from file.

    Args:
        survey_year: Survey year to determine which vintage crosswalk to load.
        filename: Input filename. If provided, survey_year is ignored.

    Returns:
        Crosswalk DataFrame
    """
    if filename is None:
        if survey_year is not None:
            vintage = get_puma_vintage(survey_year)
            filename = f"puma_county_crosswalk_{vintage}.parquet"
        else:
            filename = "puma_county_crosswalk.parquet"

    input_path = PROCESSED_DIR / filename

    if not input_path.exists():
        # Try legacy filename
        legacy_path = PROCESSED_DIR / "puma_county_crosswalk.parquet"
        if legacy_path.exists():
            return pd.read_parquet(legacy_path)
        raise FileNotFoundError(f"Crosswalk file not found: {input_path}")

    return pd.read_parquet(input_path)


def apply_puma_size_adjustment(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """
    Apply inverse-PUMA-size weighting to address western US measurement error.

    Larger PUMAs (common in rural western areas) should receive less weight
    relative to smaller, more densely populated PUMAs.

    Args:
        crosswalk: Base crosswalk DataFrame with puma_area column

    Returns:
        DataFrame with adjusted weights
    """
    # Calculate median PUMA area (in square meters from Albers projection)
    median_puma_area = crosswalk["puma_area"].median()

    # Create size adjustment factor (inverse relationship)
    # Larger PUMAs get smaller adjustment, smaller PUMAs get larger adjustment
    crosswalk["size_adjustment"] = median_puma_area / crosswalk["puma_area"]

    # Apply adjustment to base weight
    crosswalk["adjusted_weight_raw"] = crosswalk["weight"] * crosswalk["size_adjustment"]

    # Normalize within each county so weights sum to 1
    county_totals = crosswalk.groupby("county_geoid")["adjusted_weight_raw"].transform("sum")
    crosswalk["adjusted_weight"] = crosswalk["adjusted_weight_raw"] / county_totals

    # Clean up intermediate columns
    crosswalk = crosswalk.drop(columns=["adjusted_weight_raw"])

    return crosswalk


def get_crosswalk_quality_flags(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """
    Generate data quality flags based on PUMA coverage.

    Args:
        crosswalk: Crosswalk DataFrame

    Returns:
        DataFrame with county-level quality flags
    """
    # Aggregate to county level
    county_quality = crosswalk.groupby("county_geoid").agg(
        state_fips=("state_fips", "first"),
        county_fips=("county_fips", "first"),
        county_name=("county_name", "first"),
        n_pumas=("puma", "count"),
        max_puma_weight=("weight", "max"),
        puma_vintage=("puma_vintage", "first"),
    ).reset_index()

    # Assign quality flags
    def assign_flag(row: pd.Series) -> str:
        if row["max_puma_weight"] > 0.9:
            return "poor"  # Essentially single-PUMA county
        elif row["max_puma_weight"] > 0.7 or row["n_pumas"] < 2:
            return "caution"
        else:
            return "good"

    county_quality["data_quality_flag"] = county_quality.apply(assign_flag, axis=1)

    return county_quality


def get_crosswalk_for_year(survey_year: int) -> pd.DataFrame:
    """
    Get the appropriate crosswalk for a survey year, loading or building as needed.

    Args:
        survey_year: Survey year

    Returns:
        Crosswalk DataFrame for the appropriate PUMA vintage
    """
    vintage = get_puma_vintage(survey_year)

    try:
        return load_crosswalk(survey_year)
    except FileNotFoundError:
        console.print(f"[yellow]Crosswalk for {vintage} not found, building...[/yellow]")
        crosswalk = build_full_crosswalk(survey_year)
        save_crosswalk(crosswalk, survey_year)
        return crosswalk
