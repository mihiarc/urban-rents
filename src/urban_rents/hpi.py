"""House Price Index data loading and interpolation for backcasting.

This module provides functionality to:
1. Load FHFA county-level House Price Index data
2. Calculate growth rates between years
3. Apply HPI-based interpolation to fill gaps in property value data
4. Smooth volatile 1-year ACS estimates using HPI trends
5. State-level HPI fallback for counties without county-level data
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rich.console import Console

from urban_rents.config import PROCESSED_DIR, RAW_DIR

console = Console()

# HPI data paths
HPI_RAW_DIR = RAW_DIR / "hpi"
HPI_PARQUET_PATH = PROCESSED_DIR / "fhfa_county_hpi.parquet"
STATE_HPI_PARQUET_PATH = PROCESSED_DIR / "fhfa_state_hpi.parquet"


def load_hpi_data(source: Literal["fhfa", "zillow"] = "fhfa") -> pd.DataFrame:
    """
    Load county-level House Price Index data.

    Args:
        source: Data source ("fhfa" or "zillow")

    Returns:
        DataFrame with columns: county_fips, year, hpi
        HPI is indexed to 100 in base year (2000 for FHFA)
    """
    if source == "fhfa":
        if not HPI_PARQUET_PATH.exists():
            raise FileNotFoundError(
                f"FHFA HPI data not found at {HPI_PARQUET_PATH}. "
                "Run the HPI download script first."
            )
        df = pd.read_parquet(HPI_PARQUET_PATH)

        # Ensure proper types
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        df["year"] = df["year"].astype(int)
        df["hpi"] = df["hpi"].astype(float)

        return df
    else:
        raise NotImplementedError(f"Source '{source}' not yet implemented")


def calculate_hpi_growth_rates(hpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate year-over-year HPI growth rates for each county.

    Args:
        hpi_df: HPI data with columns: county_fips, year, hpi

    Returns:
        DataFrame with additional columns: hpi_growth, hpi_cumulative_from_2000
    """
    df = hpi_df.copy()
    df = df.sort_values(["county_fips", "year"])

    # Calculate YoY growth rate
    df["hpi_growth"] = df.groupby("county_fips")["hpi"].pct_change()

    # Calculate cumulative growth from base year 2000
    # Since base year 2000 = 100, cumulative growth = (hpi / 100) - 1
    df["hpi_cumulative_from_2000"] = (df["hpi"] / 100) - 1

    return df


def calculate_state_hpi(hpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate state-level average HPI from county data.

    Used as fallback for counties without county-level HPI data.

    Args:
        hpi_df: County-level HPI data

    Returns:
        DataFrame with columns: state_fips, year, state_hpi
    """
    df = hpi_df.copy()
    df["state_fips"] = df["county_fips"].str[:2]

    # Calculate weighted average HPI by state (simple mean across counties)
    state_hpi = df.groupby(["state_fips", "year"]).agg(
        state_hpi=("hpi", "mean"),
        n_counties=("county_fips", "nunique"),
    ).reset_index()

    return state_hpi


def get_hpi_with_state_fallback(
    hpi_df: pd.DataFrame,
    county_fips: str,
    year: int,
) -> tuple[float | None, str]:
    """
    Get HPI for a county, falling back to state average if county data unavailable.

    Args:
        hpi_df: County-level HPI data
        county_fips: 5-digit county FIPS
        year: Year to get HPI for

    Returns:
        Tuple of (hpi_value, source) where source is 'county' or 'state'
    """
    # Try county first
    county_data = hpi_df[
        (hpi_df["county_fips"] == county_fips) & (hpi_df["year"] == year)
    ]["hpi"].values

    if len(county_data) > 0:
        return county_data[0], "county"

    # Fall back to state average
    state_fips = county_fips[:2]
    state_data = hpi_df[hpi_df["county_fips"].str[:2] == state_fips]
    state_year = state_data[state_data["year"] == year]["hpi"]

    if len(state_year) > 0:
        return state_year.mean(), "state"

    return None, "none"


def get_hpi_ratio(
    hpi_df: pd.DataFrame,
    county_fips: str,
    from_year: int,
    to_year: int,
) -> float | None:
    """
    Get the HPI ratio between two years for a county.

    This ratio can be used to scale property values:
    value_to_year = value_from_year * ratio

    Args:
        hpi_df: HPI data
        county_fips: 5-digit county FIPS
        from_year: Starting year
        to_year: Target year

    Returns:
        Ratio (to_year HPI / from_year HPI), or None if data unavailable
    """
    county_data = hpi_df[hpi_df["county_fips"] == county_fips]

    from_hpi = county_data[county_data["year"] == from_year]["hpi"].values
    to_hpi = county_data[county_data["year"] == to_year]["hpi"].values

    if len(from_hpi) == 0 or len(to_hpi) == 0:
        return None

    if from_hpi[0] == 0:
        return None

    return to_hpi[0] / from_hpi[0]


def get_hpi_ratio_with_fallback(
    hpi_df: pd.DataFrame,
    county_fips: str,
    from_year: int,
    to_year: int,
) -> tuple[float | None, str]:
    """
    Get HPI ratio with state-level fallback for missing county data.

    Args:
        hpi_df: HPI data
        county_fips: 5-digit county FIPS
        from_year: Starting year
        to_year: Target year

    Returns:
        Tuple of (ratio, source) where source is 'county', 'state', or 'none'
    """
    from_hpi, from_source = get_hpi_with_state_fallback(hpi_df, county_fips, from_year)
    to_hpi, to_source = get_hpi_with_state_fallback(hpi_df, county_fips, to_year)

    if from_hpi is None or to_hpi is None or from_hpi == 0:
        return None, "none"

    # Determine source (use the most specific available)
    if from_source == "county" and to_source == "county":
        source = "county"
    elif from_source == "state" or to_source == "state":
        source = "state"
    else:
        source = "none"

    return to_hpi / from_hpi, source


def backcast_property_values(
    anchor_df: pd.DataFrame,
    hpi_df: pd.DataFrame,
    anchor_year: int,
    target_years: list[int],
    value_column: str = "mean_property_value",
) -> pd.DataFrame:
    """
    Backcast property values from an anchor year to earlier years using HPI.

    Formula: value_target = value_anchor * (HPI_target / HPI_anchor)

    Args:
        anchor_df: DataFrame with anchor year property values
                  Must have columns: county_geoid and value_column
        hpi_df: HPI data with columns: county_fips, year, hpi
        anchor_year: Year with observed property values
        target_years: Years to backcast to (should be < anchor_year)
        value_column: Name of the property value column

    Returns:
        DataFrame with backcasted values for target years
        Includes data_source column marking values as "hpi_backcast"
    """
    console.print(f"[cyan]Backcasting from {anchor_year} to {target_years}...[/cyan]")

    # Prepare anchor data
    anchor = anchor_df[anchor_df["survey_year"] == anchor_year].copy()
    if len(anchor) == 0:
        raise ValueError(f"No data found for anchor year {anchor_year}")

    # Extract county_fips from county_geoid (first 5 chars)
    anchor["county_fips"] = anchor["county_geoid"].str[:5]

    # Calculate HPI ratios for anchor year
    hpi_anchor = hpi_df[hpi_df["year"] == anchor_year][["county_fips", "hpi"]].copy()
    hpi_anchor = hpi_anchor.rename(columns={"hpi": "hpi_anchor"})

    backcasted_records = []

    for target_year in target_years:
        console.print(f"  Processing {target_year}...")

        # Get HPI for target year
        hpi_target = hpi_df[hpi_df["year"] == target_year][["county_fips", "hpi"]].copy()
        hpi_target = hpi_target.rename(columns={"hpi": "hpi_target"})

        # Merge anchor data with HPI values
        year_data = anchor.merge(hpi_anchor, on="county_fips", how="left")
        year_data = year_data.merge(hpi_target, on="county_fips", how="left")

        # Calculate backcast ratio and apply
        valid_mask = (
            year_data["hpi_anchor"].notna() &
            year_data["hpi_target"].notna() &
            (year_data["hpi_anchor"] > 0)
        )

        year_data["hpi_ratio"] = np.where(
            valid_mask,
            year_data["hpi_target"] / year_data["hpi_anchor"],
            np.nan
        )

        # Backcast the property value
        year_data[value_column] = np.where(
            valid_mask,
            year_data[value_column] * year_data["hpi_ratio"],
            np.nan
        )

        # Update metadata
        year_data["survey_year"] = target_year
        year_data["data_source"] = "hpi_backcast"
        year_data["anchor_year"] = anchor_year
        year_data["hpi_ratio_applied"] = year_data["hpi_ratio"]

        # Track interpolation quality
        n_valid = valid_mask.sum()
        n_total = len(year_data)
        console.print(f"    Backcasted {n_valid}/{n_total} counties ({100*n_valid/n_total:.1f}%)")

        backcasted_records.append(year_data)

    # Combine all backcasted years
    result = pd.concat(backcasted_records, ignore_index=True)

    # Clean up temporary columns
    drop_cols = ["county_fips", "hpi_anchor", "hpi_target", "hpi_ratio"]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])

    return result


def smooth_volatile_estimates(
    panel_df: pd.DataFrame,
    hpi_df: pd.DataFrame,
    volatile_years: list[int],
    stable_anchor_year: int,
    value_column: str = "mean_property_value",
    smoothing_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Smooth volatile 1-year ACS estimates using HPI-implied trends.

    Uses a weighted average of:
    - Original 1-year estimate
    - HPI-implied value (scaled from stable anchor)

    Args:
        panel_df: Panel data with volatile years
        hpi_df: HPI data
        volatile_years: Years to smooth (typically 2005-2008)
        stable_anchor_year: Stable reference year (typically 2009)
        value_column: Column to smooth
        smoothing_weight: Weight for HPI-implied value (0-1)
                         0 = keep original, 1 = fully HPI-based

    Returns:
        DataFrame with smoothed values and data_source = "hpi_smoothed"
    """
    console.print(f"[cyan]Smoothing volatile years {volatile_years} using anchor {stable_anchor_year}...[/cyan]")

    result_dfs = []

    # Get anchor year data
    anchor = panel_df[panel_df["survey_year"] == stable_anchor_year].copy()
    anchor["county_fips"] = anchor["county_geoid"].str[:5]

    # Get anchor HPI values
    hpi_anchor = hpi_df[hpi_df["year"] == stable_anchor_year][["county_fips", "hpi"]].copy()
    hpi_anchor = hpi_anchor.rename(columns={"hpi": "hpi_anchor"})

    for year in volatile_years:
        # Get original data for this year
        year_data = panel_df[panel_df["survey_year"] == year].copy()

        if len(year_data) == 0:
            continue

        year_data["county_fips"] = year_data["county_geoid"].str[:5]

        # Get HPI for this year
        hpi_year = hpi_df[hpi_df["year"] == year][["county_fips", "hpi"]].copy()
        hpi_year = hpi_year.rename(columns={"hpi": "hpi_year"})

        # Merge to get HPI values
        year_data = year_data.merge(hpi_year, on="county_fips", how="left")

        # Get anchor values for matching counties
        anchor_subset = anchor[["county_geoid", value_column, "county_fips"]].copy()
        anchor_subset = anchor_subset.rename(columns={value_column: "anchor_value"})
        anchor_subset = anchor_subset.merge(hpi_anchor, on="county_fips", how="left")

        year_data = year_data.merge(
            anchor_subset[["county_geoid", "anchor_value", "hpi_anchor"]],
            on="county_geoid",
            how="left"
        )

        # Calculate HPI-implied value
        valid_mask = (
            year_data["hpi_anchor"].notna() &
            year_data["hpi_year"].notna() &
            year_data["anchor_value"].notna() &
            (year_data["hpi_anchor"] > 0)
        )

        year_data["hpi_implied_value"] = np.where(
            valid_mask,
            year_data["anchor_value"] * (year_data["hpi_year"] / year_data["hpi_anchor"]),
            np.nan
        )

        # Store original value
        year_data["original_value"] = year_data[value_column]

        # Calculate smoothed value as weighted average
        has_original = year_data[value_column].notna()
        has_hpi = year_data["hpi_implied_value"].notna()

        # Case 1: Both original and HPI available - weighted average
        both_mask = has_original & has_hpi
        year_data.loc[both_mask, value_column] = (
            (1 - smoothing_weight) * year_data.loc[both_mask, "original_value"] +
            smoothing_weight * year_data.loc[both_mask, "hpi_implied_value"]
        )
        year_data.loc[both_mask, "data_source"] = "hpi_smoothed"

        # Case 2: Only HPI available - use HPI value
        hpi_only_mask = ~has_original & has_hpi
        year_data.loc[hpi_only_mask, value_column] = year_data.loc[hpi_only_mask, "hpi_implied_value"]
        year_data.loc[hpi_only_mask, "data_source"] = "hpi_backcast"

        # Case 3: Only original available - keep original
        orig_only_mask = has_original & ~has_hpi
        year_data.loc[orig_only_mask, "data_source"] = "acs_1year"

        # Track smoothing metadata
        year_data["smoothing_weight"] = smoothing_weight
        year_data["anchor_year"] = stable_anchor_year

        # Clean up
        drop_cols = ["county_fips", "hpi_year", "hpi_anchor", "anchor_value",
                     "hpi_implied_value", "original_value"]
        year_data = year_data.drop(columns=[c for c in drop_cols if c in year_data.columns])

        n_smoothed = both_mask.sum()
        n_backcast = hpi_only_mask.sum()
        console.print(f"  {year}: {n_smoothed} smoothed, {n_backcast} backcasted")

        result_dfs.append(year_data)

    return pd.concat(result_dfs, ignore_index=True)


def get_hpi_coverage_stats(hpi_df: pd.DataFrame, panel_counties: list[str]) -> dict:
    """
    Calculate HPI coverage statistics relative to panel counties.

    Args:
        hpi_df: HPI data
        panel_counties: List of county_geoid values in the panel

    Returns:
        Dictionary with coverage statistics
    """
    # Extract county_fips from panel counties
    panel_fips = set(c[:5] for c in panel_counties)
    hpi_fips = set(hpi_df["county_fips"].unique())

    covered = panel_fips & hpi_fips
    missing = panel_fips - hpi_fips

    # Year coverage
    years = sorted(hpi_df["year"].unique())

    return {
        "panel_counties": len(panel_fips),
        "hpi_counties": len(hpi_fips),
        "covered_counties": len(covered),
        "missing_counties": len(missing),
        "coverage_rate": len(covered) / len(panel_fips) if panel_fips else 0,
        "hpi_years": years,
        "year_range": (min(years), max(years)) if years else (None, None),
    }


def interpolate_between_endpoints(
    panel_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    target_years: list[int],
    hpi_df: pd.DataFrame,
    value_column: str = "mean_property_value",
) -> pd.DataFrame:
    """
    Interpolate property values between two endpoint years using HPI trends.

    For counties missing data in target_years but having data at start_year and end_year,
    this function interpolates values using HPI-weighted interpolation.

    Args:
        panel_df: Panel data with start_year and end_year values
        start_year: Starting anchor year (e.g., 2000)
        end_year: Ending anchor year (e.g., 2012)
        target_years: Years to interpolate (e.g., [2005, 2006, 2007, 2008, 2009, 2010, 2011])
        hpi_df: HPI data for trend adjustment
        value_column: Property value column name

    Returns:
        DataFrame with interpolated values for target_years
    """
    console.print(f"[cyan]Interpolating {target_years} between {start_year} and {end_year}...[/cyan]")

    # Add county_fips if not present
    df = panel_df.copy()
    if "county_fips" not in df.columns:
        df["county_fips"] = df["county_geoid"].str[:5]

    # Get start and end year data
    start_data = df[df["survey_year"] == start_year][["county_fips", value_column]].copy()
    start_data = start_data.rename(columns={value_column: "start_value"})

    end_data = df[df["survey_year"] == end_year][["county_fips", value_column]].copy()
    end_data = end_data.rename(columns={value_column: "end_value"})

    # Identify counties with both endpoints but missing target years
    counties_both = start_data.merge(end_data, on="county_fips", how="inner")
    counties_both = counties_both[
        counties_both["start_value"].notna() & counties_both["end_value"].notna()
    ]

    # Get HPI for start and end years
    hpi_start = hpi_df[hpi_df["year"] == start_year][["county_fips", "hpi"]].copy()
    hpi_start = hpi_start.rename(columns={"hpi": "hpi_start"})

    hpi_end = hpi_df[hpi_df["year"] == end_year][["county_fips", "hpi"]].copy()
    hpi_end = hpi_end.rename(columns={"hpi": "hpi_end"})

    # Template for target years
    template_year = df["survey_year"].max()
    template = df[df["survey_year"] == template_year].copy()

    interpolated_records = []

    for target_year in target_years:
        console.print(f"  Processing {target_year}...")

        # Get counties missing data for this year
        year_data = df[df["survey_year"] == target_year]
        missing_counties = set(
            year_data[year_data[value_column].isna()]["county_fips"]
        )

        # Filter to counties with both endpoints and missing this year
        can_interpolate = counties_both[
            counties_both["county_fips"].isin(missing_counties)
        ].copy()

        if len(can_interpolate) == 0:
            console.print(f"    No counties to interpolate for {target_year}")
            continue

        # Get HPI for target year
        hpi_target = hpi_df[hpi_df["year"] == target_year][["county_fips", "hpi"]].copy()
        hpi_target = hpi_target.rename(columns={"hpi": "hpi_target"})

        # Merge HPI data
        can_interpolate = can_interpolate.merge(hpi_start, on="county_fips", how="left")
        can_interpolate = can_interpolate.merge(hpi_end, on="county_fips", how="left")
        can_interpolate = can_interpolate.merge(hpi_target, on="county_fips", how="left")

        # Calculate HPI-weighted interpolation
        # weight = (hpi_target - hpi_start) / (hpi_end - hpi_start)
        # value_target = start_value + weight * (end_value - start_value)
        valid_mask = (
            can_interpolate["hpi_start"].notna() &
            can_interpolate["hpi_end"].notna() &
            can_interpolate["hpi_target"].notna() &
            (can_interpolate["hpi_end"] != can_interpolate["hpi_start"])
        )

        can_interpolate["hpi_weight"] = np.where(
            valid_mask,
            (can_interpolate["hpi_target"] - can_interpolate["hpi_start"]) /
            (can_interpolate["hpi_end"] - can_interpolate["hpi_start"]),
            # Linear fallback if HPI not available
            (target_year - start_year) / (end_year - start_year)
        )

        can_interpolate["interpolated_value"] = (
            can_interpolate["start_value"] +
            can_interpolate["hpi_weight"] * (can_interpolate["end_value"] - can_interpolate["start_value"])
        )

        # Create records for interpolated counties
        interp_template = template[template["county_fips"].isin(can_interpolate["county_fips"])].copy()
        interp_template = interp_template.merge(
            can_interpolate[["county_fips", "interpolated_value"]],
            on="county_fips",
            how="left"
        )

        interp_template["survey_year"] = target_year
        interp_template[value_column] = interp_template["interpolated_value"]
        interp_template["data_source"] = np.where(
            valid_mask[can_interpolate["county_fips"].isin(interp_template["county_fips"])].values
            if len(valid_mask) > 0 else False,
            "hpi_interpolated",
            "linear_interpolated"
        )
        interp_template["data_source"] = "hpi_interpolated"
        interp_template["anchor_years"] = f"{start_year}-{end_year}"

        # Clean up
        interp_template = interp_template.drop(columns=["interpolated_value"], errors="ignore")

        n_interpolated = len(interp_template)
        console.print(f"    Interpolated {n_interpolated} counties for {target_year}")

        interpolated_records.append(interp_template)

    if not interpolated_records:
        return pd.DataFrame()

    result = pd.concat(interpolated_records, ignore_index=True)

    # Clean up county_fips if it was added
    if "county_fips" in result.columns:
        result = result.drop(columns=["county_fips"])

    return result
