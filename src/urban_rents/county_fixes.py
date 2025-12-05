"""County FIPS code fixes and crosswalks for consistent panel coverage.

This module handles special cases where county FIPS codes changed or new counties
were created, ensuring 100% coverage across all years.

Cases handled:
1. Oglala Lakota County, SD (46102) - renamed from Shannon County (46113) in 2015
2. Broomfield County, CO (08014) - created November 2001 from parts of 4 counties
3. Connecticut Planning Regions (09110-09190) - replaced 8 counties in 2022 Census
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


# ============================================================================
# FIPS Code Mappings
# ============================================================================

# Oglala Lakota County was Shannon County until 2015
OGLALA_LAKOTA_FIPS = "46102"  # New FIPS (2015+)
SHANNON_COUNTY_FIPS = "46113"  # Old FIPS (pre-2015)

# Broomfield County created 2001 from parts of these counties
BROOMFIELD_FIPS = "08014"
BROOMFIELD_PARENT_COUNTIES = ["08001", "08013", "08031", "08123"]  # Adams, Boulder, Jefferson, Weld

# Connecticut Planning Regions (2022+) replacing old counties
# Old CT counties (09001-09015) -> New planning regions (09110-09190)
CT_COUNTY_TO_PLANNING_REGION = {
    # Capitol Planning Region
    "09003": "09110",  # Hartford County -> Capitol
    # Greater Bridgeport Planning Region
    "09001": "09120",  # Fairfield County -> split between multiple regions
    # Lower Connecticut River Valley Planning Region
    "09007": "09130",  # Middlesex County -> Lower CT River Valley
    # Naugatuck Valley Planning Region
    "09009": "09140",  # New Haven County -> split (partial to Naugatuck Valley)
    # Northeastern Connecticut Planning Region
    "09015": "09150",  # Windham County -> Northeastern CT
    # Northwest Hills Planning Region
    "09005": "09160",  # Litchfield County -> Northwest Hills
    # South Central Connecticut Planning Region
    # New Haven County also contributes here
    # Southeastern Connecticut Planning Region
    "09011": "09180",  # New London County -> Southeastern CT
    # Western Connecticut Planning Region
    # Fairfield County also contributes here
}

# Reverse mapping: Planning regions to original counties (weighted average)
CT_PLANNING_REGION_TO_COUNTIES = {
    "09110": ["09003"],  # Capitol <- Hartford
    "09120": ["09001"],  # Greater Bridgeport <- Fairfield (partial)
    "09130": ["09007"],  # Lower CT River Valley <- Middlesex
    "09140": ["09009"],  # Naugatuck Valley <- New Haven (partial)
    "09150": ["09015"],  # Northeastern CT <- Windham
    "09160": ["09005"],  # Northwest Hills <- Litchfield
    "09170": ["09009"],  # South Central CT <- New Haven (partial)
    "09180": ["09011"],  # Southeastern CT <- New London
    "09190": ["09001"],  # Western CT <- Fairfield (partial)
}


def fix_oglala_lakota(
    panel_df: pd.DataFrame,
    census_2000_df: pd.DataFrame,
    value_column: str = "mean_property_value",
) -> pd.DataFrame:
    """Fix Oglala Lakota County by using Shannon County (old FIPS) Census 2000 data.

    Oglala Lakota County (46102) was renamed from Shannon County (46113) in 2015.
    Census 2000 uses Shannon County FIPS 46113, which has median house value $23,100.

    Args:
        panel_df: Panel data with county_geoid column
        census_2000_df: Census 2000 SF3 data with county_fips column
        value_column: Property value column name

    Returns:
        Updated panel with Oglala Lakota fixed
    """
    console.print("[cyan]Fixing Oglala Lakota County (46102 <- Shannon 46113)...[/cyan]")

    panel_df = panel_df.copy()

    # Get Shannon County value from Census 2000
    census_2000_df = census_2000_df.copy()
    census_2000_df["county_fips"] = census_2000_df["county_fips"].astype(str).str.zfill(5)

    shannon_data = census_2000_df[census_2000_df["county_fips"] == SHANNON_COUNTY_FIPS]

    if len(shannon_data) == 0:
        console.print(f"  [yellow]Warning: Shannon County {SHANNON_COUNTY_FIPS} not found in Census 2000[/yellow]")
        return panel_df

    shannon_value = shannon_data["median_property_value"].iloc[0]
    console.print(f"  Shannon County Census 2000 value: ${shannon_value:,.0f}")

    # Find Oglala Lakota records needing fix
    if "county_fips" not in panel_df.columns:
        panel_df["county_fips"] = panel_df["county_geoid"].str[:5]

    oglala_mask = panel_df["county_fips"] == OGLALA_LAKOTA_FIPS
    missing_mask = oglala_mask & panel_df[value_column].isna()

    n_fixed = missing_mask.sum()
    if n_fixed > 0:
        # For year 2000, use Shannon value directly
        year_2000_mask = missing_mask & (panel_df["survey_year"] == 2000)
        panel_df.loc[year_2000_mask, value_column] = shannon_value
        panel_df.loc[year_2000_mask, "data_source"] = "census_2000_sf3_crosswalk"

        console.print(f"  Fixed {year_2000_mask.sum()} year 2000 records")
    else:
        console.print("  No missing Oglala Lakota records to fix")

    return panel_df


def fix_broomfield(
    panel_df: pd.DataFrame,
    census_2000_df: pd.DataFrame,
    value_column: str = "mean_property_value",
) -> pd.DataFrame:
    """Fix Broomfield County by using weighted average of parent counties.

    Broomfield County (08014) was created November 2001 from parts of:
    - Adams County (08001)
    - Boulder County (08013)
    - Jefferson County (08031)
    - Weld County (08123)

    For Census 2000, we use the average of parent county median values.

    Args:
        panel_df: Panel data with county_geoid column
        census_2000_df: Census 2000 SF3 data with county_fips column
        value_column: Property value column name

    Returns:
        Updated panel with Broomfield fixed
    """
    console.print("[cyan]Fixing Broomfield County (08014 - created 2001)...[/cyan]")

    panel_df = panel_df.copy()

    # Get parent county values from Census 2000
    census_2000_df = census_2000_df.copy()
    census_2000_df["county_fips"] = census_2000_df["county_fips"].astype(str).str.zfill(5)

    parent_data = census_2000_df[census_2000_df["county_fips"].isin(BROOMFIELD_PARENT_COUNTIES)]

    if len(parent_data) == 0:
        console.print("  [yellow]Warning: Parent counties not found in Census 2000[/yellow]")
        return panel_df

    # Calculate average of parent counties
    parent_values = parent_data["median_property_value"].dropna()
    broomfield_value = parent_values.mean()

    console.print(f"  Parent county values: {dict(zip(parent_data['county_fips'], parent_data['median_property_value']))}")
    console.print(f"  Broomfield estimated value: ${broomfield_value:,.0f}")

    # Find Broomfield records needing fix
    if "county_fips" not in panel_df.columns:
        panel_df["county_fips"] = panel_df["county_geoid"].str[:5]

    broomfield_mask = panel_df["county_fips"] == BROOMFIELD_FIPS
    missing_mask = broomfield_mask & panel_df[value_column].isna()

    # For year 2000, use parent average
    year_2000_mask = missing_mask & (panel_df["survey_year"] == 2000)
    if year_2000_mask.sum() > 0:
        panel_df.loc[year_2000_mask, value_column] = broomfield_value
        panel_df.loc[year_2000_mask, "data_source"] = "census_2000_parent_avg"
        console.print(f"  Fixed {year_2000_mask.sum()} year 2000 records")
    else:
        console.print("  No missing Broomfield year 2000 records")

    return panel_df


def fix_connecticut_planning_regions(
    panel_df: pd.DataFrame,
    census_2000_df: pd.DataFrame,
    value_column: str = "mean_property_value",
) -> pd.DataFrame:
    """Fix Connecticut planning regions by mapping from old county values.

    In 2022, Connecticut replaced its 8 counties with 9 planning regions:
    - 09110: Capitol Planning Region
    - 09120: Greater Bridgeport Planning Region
    - 09130: Lower Connecticut River Valley Planning Region
    - 09140: Naugatuck Valley Planning Region
    - 09150: Northeastern Connecticut Planning Region
    - 09160: Northwest Hills Planning Region
    - 09170: South Central Connecticut Planning Region
    - 09180: Southeastern Connecticut Planning Region
    - 09190: Western Connecticut Planning Region

    Args:
        panel_df: Panel data with county_geoid column
        census_2000_df: Census 2000 SF3 data with county_fips column
        value_column: Property value column name

    Returns:
        Updated panel with CT planning regions fixed
    """
    console.print("[cyan]Fixing Connecticut planning regions (09110-09190)...[/cyan]")

    panel_df = panel_df.copy()

    # Get Connecticut county values from Census 2000
    census_2000_df = census_2000_df.copy()
    census_2000_df["county_fips"] = census_2000_df["county_fips"].astype(str).str.zfill(5)

    ct_county_fips = [f"09{str(i).zfill(3)}" for i in range(1, 16, 2)]  # 09001, 09003, ..., 09015
    ct_data = census_2000_df[census_2000_df["county_fips"].isin(ct_county_fips)]

    if len(ct_data) == 0:
        console.print("  [yellow]Warning: CT counties not found in Census 2000[/yellow]")
        return panel_df

    # Create mapping from old counties to values
    ct_values = dict(zip(ct_data["county_fips"], ct_data["median_property_value"]))
    console.print(f"  CT county Census 2000 values: {ct_values}")

    # Find CT planning region records
    if "county_fips" not in panel_df.columns:
        panel_df["county_fips"] = panel_df["county_geoid"].str[:5]

    ct_regions = list(CT_PLANNING_REGION_TO_COUNTIES.keys())

    n_fixed = 0
    for region_fips, source_counties in CT_PLANNING_REGION_TO_COUNTIES.items():
        region_mask = panel_df["county_fips"] == region_fips
        missing_mask = region_mask & panel_df[value_column].isna()
        year_2000_mask = missing_mask & (panel_df["survey_year"] == 2000)

        if year_2000_mask.sum() > 0:
            # Calculate average from source counties
            source_values = [ct_values.get(c) for c in source_counties if c in ct_values]
            if source_values:
                region_value = np.mean([v for v in source_values if v is not None])
                panel_df.loc[year_2000_mask, value_column] = region_value
                panel_df.loc[year_2000_mask, "data_source"] = "census_2000_ct_crosswalk"
                n_fixed += year_2000_mask.sum()
                console.print(f"  {region_fips}: ${region_value:,.0f} (from {source_counties})")

    console.print(f"  Fixed {n_fixed} CT planning region records for year 2000")

    return panel_df


def interpolate_acs_gaps(
    panel_df: pd.DataFrame,
    value_column: str = "mean_property_value",
    gap_years: list[int] | None = None,
) -> pd.DataFrame:
    """Interpolate missing ACS values for specific county-years.

    For counties with sporadic missing years (e.g., 2014-2016), interpolate
    from surrounding years using linear interpolation.

    Args:
        panel_df: Panel data with survey_year and county_geoid
        value_column: Property value column name
        gap_years: Years to check for gaps (default: 2014-2016)

    Returns:
        Updated panel with gaps interpolated
    """
    if gap_years is None:
        gap_years = [2014, 2015, 2016]

    console.print(f"[cyan]Interpolating ACS gaps for years {gap_years}...[/cyan]")

    panel_df = panel_df.copy()

    if "county_fips" not in panel_df.columns:
        panel_df["county_fips"] = panel_df["county_geoid"].str[:5]

    n_fixed = 0

    for gap_year in gap_years:
        # Find counties missing this year
        year_data = panel_df[panel_df["survey_year"] == gap_year]
        missing_counties = year_data[year_data[value_column].isna()]["county_fips"].unique()

        for county_fips in missing_counties:
            county_data = panel_df[panel_df["county_fips"] == county_fips].copy()
            county_data = county_data.sort_values("survey_year")

            # Get surrounding years with data
            valid_years = county_data[county_data[value_column].notna()]["survey_year"].values

            if len(valid_years) < 2:
                continue

            # Find closest years before and after
            years_before = valid_years[valid_years < gap_year]
            years_after = valid_years[valid_years > gap_year]

            if len(years_before) == 0 or len(years_after) == 0:
                continue

            year_before = years_before[-1]
            year_after = years_after[0]

            value_before = county_data[county_data["survey_year"] == year_before][value_column].iloc[0]
            value_after = county_data[county_data["survey_year"] == year_after][value_column].iloc[0]

            # Linear interpolation
            weight = (gap_year - year_before) / (year_after - year_before)
            interpolated_value = value_before + weight * (value_after - value_before)

            # Update panel
            gap_mask = (panel_df["county_fips"] == county_fips) & (panel_df["survey_year"] == gap_year)
            panel_df.loc[gap_mask, value_column] = interpolated_value
            panel_df.loc[gap_mask, "data_source"] = "acs_interpolated"
            n_fixed += gap_mask.sum()

    console.print(f"  Interpolated {n_fixed} records")

    return panel_df


def propagate_fixed_counties_with_hpi(
    panel_df: pd.DataFrame,
    value_column: str = "mean_property_value",
    fixed_counties: list[str] | None = None,
) -> pd.DataFrame:
    """Propagate fixed year 2000 values to 2001-2011 using linear interpolation.

    For counties that were fixed in year 2000 (Oglala Lakota, Broomfield, CT regions),
    we need to fill in 2001-2011 by interpolating between 2000 and 2012.

    Args:
        panel_df: Panel data with fixed year 2000 values
        value_column: Property value column name
        fixed_counties: List of county FIPS to propagate (default: special cases)

    Returns:
        Panel with 2001-2011 filled for fixed counties
    """
    if fixed_counties is None:
        # All the special case counties
        fixed_counties = [
            OGLALA_LAKOTA_FIPS,
            BROOMFIELD_FIPS,
        ] + list(CT_PLANNING_REGION_TO_COUNTIES.keys())

    console.print(f"[cyan]Propagating {len(fixed_counties)} fixed counties to 2001-2011...[/cyan]")

    panel_df = panel_df.copy()

    if "county_fips" not in panel_df.columns:
        panel_df["county_fips"] = panel_df["county_geoid"].str[:5]

    n_fixed = 0
    years_to_fill = list(range(2001, 2012))  # 2001-2011

    for county_fips in fixed_counties:
        county_data = panel_df[panel_df["county_fips"] == county_fips]

        if len(county_data) == 0:
            continue

        # Get 2000 and 2012 values
        year_2000_data = county_data[county_data["survey_year"] == 2000]
        year_2012_data = county_data[county_data["survey_year"] == 2012]

        if len(year_2000_data) == 0 or len(year_2012_data) == 0:
            continue

        value_2000 = year_2000_data[value_column].iloc[0]
        value_2012 = year_2012_data[value_column].iloc[0]

        if pd.isna(value_2000) or pd.isna(value_2012):
            continue

        # Interpolate for each missing year
        for year in years_to_fill:
            year_mask = (panel_df["county_fips"] == county_fips) & (panel_df["survey_year"] == year)
            year_data = panel_df.loc[year_mask]

            if len(year_data) > 0 and pd.isna(year_data[value_column].iloc[0]):
                # Linear interpolation
                weight = (year - 2000) / (2012 - 2000)
                interpolated_value = value_2000 + weight * (value_2012 - value_2000)

                panel_df.loc[year_mask, value_column] = interpolated_value
                panel_df.loc[year_mask, "data_source"] = "crosswalk_interpolated"
                n_fixed += 1

    console.print(f"  Propagated {n_fixed} records for special counties")

    return panel_df


def apply_all_county_fixes(
    panel_df: pd.DataFrame,
    census_2000_df: pd.DataFrame,
    value_column: str = "mean_property_value",
) -> pd.DataFrame:
    """Apply all county fixes to achieve 100% coverage.

    Args:
        panel_df: Panel data
        census_2000_df: Census 2000 SF3 data
        value_column: Property value column name

    Returns:
        Panel with all fixes applied
    """
    console.print("\n[bold]Applying County Coverage Fixes[/bold]")
    console.print("=" * 60)

    # 1. Fix Oglala Lakota County (year 2000)
    panel_df = fix_oglala_lakota(panel_df, census_2000_df, value_column)

    # 2. Fix Broomfield County (year 2000)
    panel_df = fix_broomfield(panel_df, census_2000_df, value_column)

    # 3. Fix Connecticut planning regions (year 2000)
    panel_df = fix_connecticut_planning_regions(panel_df, census_2000_df, value_column)

    # 4. Propagate fixed counties to 2001-2011
    panel_df = propagate_fixed_counties_with_hpi(panel_df, value_column)

    # 5. Interpolate ACS gaps (2014-2016)
    panel_df = interpolate_acs_gaps(panel_df, value_column)

    # Report final coverage
    console.print("\n[bold]Coverage After Fixes:[/bold]")
    for year in sorted(panel_df["survey_year"].unique()):
        year_data = panel_df[panel_df["survey_year"] == year]
        n_total = len(year_data)
        n_valid = year_data[value_column].notna().sum()
        coverage = 100 * n_valid / n_total if n_total > 0 else 0
        console.print(f"  {year}: {n_valid}/{n_total} ({coverage:.1f}%)")

    return panel_df
