"""Census 2000 SF3 county-level property value data.

This module provides functionality to:
1. Download Census 2000 Summary File 3 (SF3) property values via Census API
2. Process and normalize county-level median house values
3. Integrate Census 2000 data as anchor for backcasting methodology
"""

from pathlib import Path

import pandas as pd
import requests
from rich.console import Console

from urban_rents.config import PROCESSED_DIR

console = Console()

# Census 2000 SF3 API endpoint
CENSUS_2000_API = "https://api.census.gov/data/2000/dec/sf3"

# Output file path
CENSUS_2000_PARQUET = PROCESSED_DIR / "census_2000_county_property_values.parquet"


def download_census_2000_property_values() -> pd.DataFrame:
    """
    Download Census 2000 SF3 median house values for all counties.

    Uses the Census API to get H085001 (median value of owner-occupied housing units).

    Returns:
        DataFrame with columns: county_fips, county_name, median_property_value
    """
    console.print("[bold]Downloading Census 2000 SF3 county property values...[/bold]")

    # H085001 = Median value (dollars) for owner-occupied housing units
    url = f"{CENSUS_2000_API}?get=NAME,H085001&for=county:*"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    # Create county FIPS (5-digit)
    df["county_fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)

    # Convert to numeric
    df["median_property_value"] = pd.to_numeric(df["H085001"], errors="coerce")

    # Clean up
    df = df[["county_fips", "NAME", "median_property_value"]].rename(
        columns={"NAME": "county_name"}
    )

    # Remove missing values
    df = df[df["median_property_value"].notna()]

    console.print(f"  Downloaded {len(df)} county records")
    console.print(f"  Median value: ${df['median_property_value'].median():,.0f}")

    return df


def load_census_2000_property_values(download_if_missing: bool = True) -> pd.DataFrame:
    """
    Load Census 2000 county property values.

    Args:
        download_if_missing: If True, download data if parquet file doesn't exist

    Returns:
        DataFrame with columns: county_fips, county_name, median_property_value
    """
    if CENSUS_2000_PARQUET.exists():
        console.print(f"[green]Loading Census 2000 data from {CENSUS_2000_PARQUET}[/green]")
        return pd.read_parquet(CENSUS_2000_PARQUET)

    if not download_if_missing:
        raise FileNotFoundError(
            f"Census 2000 data not found at {CENSUS_2000_PARQUET}. "
            "Run download_census_2000_property_values() first."
        )

    # Download and save
    df = download_census_2000_property_values()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CENSUS_2000_PARQUET, index=False)
    console.print(f"[green]Saved to {CENSUS_2000_PARQUET}[/green]")

    return df


def create_census_2000_panel_records(
    panel_df: pd.DataFrame,
    census_2000_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Create year 2000 panel records using Census 2000 SF3 data.

    This replaces HPI-backcasted 2000 values with actual Census 2000 observations.

    Args:
        panel_df: Panel data (typically 2009+ 5-year ACS data)
        census_2000_df: Census 2000 property values (loaded if None)

    Returns:
        DataFrame with year 2000 records for all counties
    """
    if census_2000_df is None:
        census_2000_df = load_census_2000_property_values()

    console.print("[bold]Creating year 2000 panel records from Census 2000 SF3...[/bold]")

    # Get unique counties from panel
    if "county_fips" not in panel_df.columns:
        panel_df = panel_df.copy()
        panel_df["county_fips"] = panel_df["county_geoid"].str[:5]

    panel_counties = set(panel_df["county_fips"].unique())

    # Get a template record (for column structure)
    template_year = panel_df["survey_year"].max()
    template = panel_df[panel_df["survey_year"] == template_year].copy()

    # Merge with Census 2000 values
    census_2000_df = census_2000_df.copy()
    census_2000_df["county_fips"] = census_2000_df["county_fips"].astype(str).str.zfill(5)

    # Create year 2000 records
    result = template.merge(
        census_2000_df[["county_fips", "median_property_value"]],
        on="county_fips",
        how="left",
    )

    # Update values
    result["survey_year"] = 2000
    result["mean_property_value"] = result["median_property_value"]
    result["data_source"] = "census_2000_sf3"

    # Drop temporary columns
    result = result.drop(columns=["county_fips", "median_property_value"], errors="ignore")

    # Report coverage
    n_matched = result["mean_property_value"].notna().sum()
    n_total = len(result)

    console.print(f"  Created {n_total} year 2000 records")
    console.print(f"  Census 2000 coverage: {n_matched}/{n_total} ({100*n_matched/n_total:.1f}%)")

    return result


def get_census_2000_coverage_stats(panel_counties: list[str]) -> dict:
    """
    Get Census 2000 coverage statistics relative to panel counties.

    Args:
        panel_counties: List of county FIPS codes in the panel

    Returns:
        Dictionary with coverage statistics
    """
    census_df = load_census_2000_property_values()

    panel_fips = set(c[:5] for c in panel_counties)
    census_fips = set(census_df["county_fips"].unique())

    covered = panel_fips & census_fips
    missing = panel_fips - census_fips

    return {
        "panel_counties": len(panel_fips),
        "census_2000_counties": len(census_fips),
        "covered_counties": len(covered),
        "missing_counties": len(missing),
        "coverage_rate": len(covered) / len(panel_fips) if panel_fips else 0,
        "source": "Census 2000 SF3 (H085001)",
    }
