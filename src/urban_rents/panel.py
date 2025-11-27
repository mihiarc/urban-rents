"""Panel dataset assembly for multi-year urban net returns analysis."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track

from urban_rents.config import (
    CENSUS_DIVISIONS,
    DISCOUNT_RATE,
    OUTPUT_DIR,
    PROCESSED_DIR,
    STATE_FIPS_TO_ABBR,
    STATE_TO_DIVISION,
    get_period_label,
    get_puma_vintage,
    get_soc_division_data,
)
from urban_rents.crosswalk import (
    apply_puma_size_adjustment,
    get_crosswalk_for_year,
    load_crosswalk,
)
from urban_rents.models import RECOMMENDED_PANEL_YEARS, CountyPanelRecord, PanelConfig
from urban_rents.pums_processing import (
    load_panel_property_values,
    load_puma_property_values,
    process_all_states,
)

console = Console()


def aggregate_puma_to_county(
    puma_values: pd.DataFrame,
    crosswalk: pd.DataFrame,
    use_adjusted_weights: bool = True,
) -> pd.DataFrame:
    """
    Aggregate PUMA-level property values to county level using crosswalk weights.

    Args:
        puma_values: DataFrame with PUMA-level property values
        crosswalk: Crosswalk DataFrame with PUMA-county weights
        use_adjusted_weights: If True, use size-adjusted weights

    Returns:
        County-level aggregated property values
    """
    # Apply size adjustment to crosswalk if needed
    if use_adjusted_weights and "adjusted_weight" not in crosswalk.columns:
        crosswalk = apply_puma_size_adjustment(crosswalk.copy())

    weight_col = "adjusted_weight" if use_adjusted_weights else "weight"

    # Merge PUMA values with crosswalk
    merged = crosswalk.merge(
        puma_values[["state_puma", "mean_property_value", "n_observations", "total_weight"]],
        on="state_puma",
        how="left",
    )

    # Calculate weighted mean for each county
    def aggregate_county(group: pd.DataFrame) -> pd.Series:
        # Only include PUMAs with data
        has_data = group["mean_property_value"].notna()
        valid = group[has_data]

        if len(valid) == 0:
            return pd.Series({
                "mean_property_value": np.nan,
                "n_pumas": len(group),
                "n_pumas_with_data": 0,
                "max_puma_weight": group[weight_col].max(),
                "total_observations": 0,
            })

        # Calculate weighted mean using crosswalk weights
        weights = valid[weight_col]
        values = valid["mean_property_value"]

        # Normalize weights for PUMAs with data
        norm_weights = weights / weights.sum()
        weighted_mean = (values * norm_weights).sum()

        return pd.Series({
            "mean_property_value": weighted_mean,
            "n_pumas": len(group),
            "n_pumas_with_data": len(valid),
            "max_puma_weight": group[weight_col].max(),
            "total_observations": valid["n_observations"].sum(),
        })

    county_values = merged.groupby(
        ["county_geoid", "state_fips", "county_fips", "county_name"]
    ).apply(aggregate_county, include_groups=False).reset_index()

    return county_values


def calculate_county_net_returns(
    county_values: pd.DataFrame,
    survey_year: int,
    soc_year: int | None = None,
) -> pd.DataFrame:
    """
    Calculate urban net returns for counties.

    Args:
        county_values: County-level property values
        survey_year: Survey year for metadata
        soc_year: SOC year for lot parameters. If None, uses nearest available.

    Returns:
        DataFrame with net returns calculated
    """
    # Get SOC data
    soc_data = get_soc_division_data(soc_year or survey_year)

    # Add state abbreviation
    county_values["state_abbr"] = county_values["state_fips"].map(STATE_FIPS_TO_ABBR)

    # Add Census Division
    county_values["census_division"] = county_values["state_fips"].map(STATE_TO_DIVISION)
    county_values["division_name"] = county_values["census_division"].map(CENSUS_DIVISIONS)

    # Add SOC parameters by division
    def get_soc_params(division: int) -> tuple:
        if division in soc_data:
            d = soc_data[division]
            return d.lot_share, d.lot_acres
        return np.nan, np.nan

    soc_params = county_values["census_division"].apply(get_soc_params)
    county_values["lot_share"] = soc_params.apply(lambda x: x[0])
    county_values["lot_acres"] = soc_params.apply(lambda x: x[1])

    # Calculate urban net return
    # NR_urban = (SalesPrice × LotShare) / LotAcres × DiscountRate
    county_values["urban_net_return"] = (
        county_values["mean_property_value"] *
        county_values["lot_share"] /
        county_values["lot_acres"] *
        DISCOUNT_RATE
    )

    # Add metadata
    county_values["survey_year"] = survey_year
    county_values["period_label"] = get_period_label(survey_year)
    county_values["puma_vintage"] = get_puma_vintage(survey_year)
    county_values["soc_year"] = soc_year or survey_year

    return county_values


def build_panel_for_year(
    survey_year: int,
    soc_methodology: Literal["year_specific", "fixed_2023", "nearest"] = "nearest",
    use_adjusted_weights: bool = True,
) -> pd.DataFrame:
    """
    Build county-level data for a single survey year.

    Args:
        survey_year: Survey year to process
        soc_methodology: How to select SOC parameters
        use_adjusted_weights: If True, use size-adjusted crosswalk weights

    Returns:
        DataFrame with county-level net returns for the year
    """
    console.print(f"[bold]Building county data for {get_period_label(survey_year)}...[/bold]")

    # Load PUMA property values - try pre-processed file first, else process raw data
    try:
        puma_values = load_puma_property_values(survey_year)
        console.print(f"[cyan]Loaded pre-processed PUMA values for {survey_year}[/cyan]")
    except FileNotFoundError:
        # Try loading from panel file
        try:
            panel = load_panel_property_values()
            puma_values = panel[panel["survey_year"] == survey_year]
            if len(puma_values) == 0:
                raise ValueError(f"No data for {survey_year} in panel file")
            console.print(f"[cyan]Loaded PUMA values from panel file for {survey_year}[/cyan]")
        except (FileNotFoundError, ValueError):
            # Process raw PUMS data
            console.print(f"[yellow]Processing raw PUMS data for {survey_year}...[/yellow]")
            puma_values = process_all_states(survey_year)

    # Get appropriate crosswalk
    crosswalk = get_crosswalk_for_year(survey_year)

    # Aggregate to county level
    county_values = aggregate_puma_to_county(
        puma_values, crosswalk, use_adjusted_weights
    )

    # Determine SOC year based on methodology
    if soc_methodology == "fixed_2023":
        soc_year = 2023
    elif soc_methodology == "year_specific":
        soc_year = survey_year
    else:  # nearest
        soc_year = None  # get_soc_division_data will find nearest

    # Calculate net returns
    county_returns = calculate_county_net_returns(
        county_values, survey_year, soc_year
    )

    # Add data quality flags
    def quality_flag(row: pd.Series) -> str:
        if row["n_pumas_with_data"] == 0:
            return "no_data"
        elif row["max_puma_weight"] > 0.9:
            return "poor"
        elif row["max_puma_weight"] > 0.7 or row["n_pumas_with_data"] < 2:
            return "caution"
        else:
            return "good"

    county_returns["data_quality_flag"] = county_returns.apply(quality_flag, axis=1)

    console.print(f"[green]Built data for {len(county_returns)} counties[/green]")

    return county_returns


def build_panel_dataset(
    survey_years: list[int],
    soc_methodology: Literal["year_specific", "fixed_2023", "nearest"] = "nearest",
    include_growth_rates: bool = True,
    use_adjusted_weights: bool = True,
) -> pd.DataFrame:
    """
    Build complete panel dataset across multiple survey years.

    Args:
        survey_years: List of survey end years to include
        soc_methodology: How to select SOC parameters
        include_growth_rates: If True, compute year-over-year growth rates
        use_adjusted_weights: If True, use size-adjusted crosswalk weights

    Returns:
        Panel DataFrame with county-year observations
    """
    console.print(f"[bold]Building panel dataset for years: {survey_years}[/bold]")

    all_years = []
    for year in track(sorted(survey_years), description="Processing years"):
        try:
            year_data = build_panel_for_year(
                year, soc_methodology, use_adjusted_weights
            )
            all_years.append(year_data)
        except Exception as e:
            console.print(f"[red]Failed to process {year}: {e}[/red]")
            continue

    if not all_years:
        raise ValueError("No years processed successfully")

    # Combine all years
    panel = pd.concat(all_years, ignore_index=True)

    # Sort by county and year
    panel = panel.sort_values(["county_geoid", "survey_year"]).reset_index(drop=True)

    # Calculate growth rates if requested
    if include_growth_rates and len(survey_years) > 1:
        panel = calculate_growth_rates(panel)

    console.print(f"\n[green]Panel complete: {len(panel)} county-year observations[/green]")
    console.print(f"[green]Years: {sorted(panel['survey_year'].unique())}[/green]")
    console.print(f"[green]Counties: {panel['county_geoid'].nunique()}[/green]")

    return panel


def calculate_growth_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate period-over-period growth rates for panel data.

    Args:
        panel: Panel DataFrame sorted by county and year

    Returns:
        Panel with growth rate columns added
    """
    # Sort to ensure proper ordering
    panel = panel.sort_values(["county_geoid", "survey_year"])

    # Calculate growth rates within each county
    def county_growth(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("survey_year")

        # Property value growth
        group["property_value_growth"] = group["mean_property_value"].pct_change()

        # Net return growth
        group["net_return_growth"] = group["urban_net_return"].pct_change()

        # Years between observations (for annualization)
        group["years_since_prior"] = group["survey_year"].diff()

        # Annualized growth rates
        group["property_value_cagr"] = (
            (1 + group["property_value_growth"]) **
            (1 / group["years_since_prior"]) - 1
        )
        group["net_return_cagr"] = (
            (1 + group["net_return_growth"]) **
            (1 / group["years_since_prior"]) - 1
        )

        return group

    panel = panel.groupby("county_geoid", group_keys=False).apply(county_growth)

    return panel


def pivot_panel_wide(panel: pd.DataFrame, value_col: str = "urban_net_return") -> pd.DataFrame:
    """
    Convert long panel to wide format.

    Args:
        panel: Long-format panel DataFrame
        value_col: Column to pivot

    Returns:
        Wide-format DataFrame with years as columns
    """
    # Identify ID columns
    id_cols = [
        "county_geoid", "state_fips", "county_fips", "county_name",
        "state_abbr", "census_division", "division_name"
    ]

    # Keep first occurrence of ID columns per county
    county_ids = panel.drop_duplicates("county_geoid")[
        [c for c in id_cols if c in panel.columns]
    ]

    # Pivot the value column
    pivoted = panel.pivot(
        index="county_geoid",
        columns="survey_year",
        values=value_col
    )

    # Rename columns with prefix
    pivoted.columns = [f"{value_col}_{year}" for year in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Merge back ID columns
    wide = county_ids.merge(pivoted, on="county_geoid")

    return wide


def save_panel(
    panel: pd.DataFrame,
    output_format: Literal["long", "wide", "both"] = "long",
    filename_prefix: str = "county_urban_net_returns_panel",
) -> list[Path]:
    """
    Save panel dataset to file(s).

    Args:
        panel: Panel DataFrame
        output_format: Output format(s) to save
        filename_prefix: Prefix for output filenames

    Returns:
        List of paths to saved files
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    if output_format in ["long", "both"]:
        long_path = OUTPUT_DIR / f"{filename_prefix}_long.parquet"
        panel.to_parquet(long_path, index=False)
        console.print(f"[green]Saved long format to {long_path}[/green]")
        saved.append(long_path)

        # Also save CSV for easy viewing
        csv_path = OUTPUT_DIR / f"{filename_prefix}_long.csv"
        panel.to_csv(csv_path, index=False)
        console.print(f"[green]Saved CSV to {csv_path}[/green]")
        saved.append(csv_path)

    if output_format in ["wide", "both"]:
        wide = pivot_panel_wide(panel)
        wide_path = OUTPUT_DIR / f"{filename_prefix}_wide.parquet"
        wide.to_parquet(wide_path, index=False)
        console.print(f"[green]Saved wide format to {wide_path}[/green]")
        saved.append(wide_path)

    return saved


def load_panel(
    filename: str = "county_urban_net_returns_panel_long.parquet",
) -> pd.DataFrame:
    """
    Load panel dataset from file.

    Args:
        filename: Input filename

    Returns:
        Panel DataFrame
    """
    input_path = OUTPUT_DIR / filename
    if not input_path.exists():
        raise FileNotFoundError(f"Panel file not found: {input_path}")

    return pd.read_parquet(input_path)


def get_panel_summary(panel: pd.DataFrame) -> dict:
    """
    Generate summary statistics for panel dataset.

    Args:
        panel: Panel DataFrame

    Returns:
        Dictionary of summary statistics
    """
    years = sorted(panel["survey_year"].unique())
    n_counties = panel["county_geoid"].nunique()

    # Calculate balanced vs unbalanced panel statistics
    county_year_counts = panel.groupby("county_geoid").size()
    n_balanced = (county_year_counts == len(years)).sum()
    n_unbalanced = n_counties - n_balanced

    # Summary by year
    year_stats = []
    for year in years:
        year_data = panel[panel["survey_year"] == year]
        valid = year_data[year_data["urban_net_return"].notna()]
        year_stats.append({
            "survey_year": year,
            "period_label": get_period_label(year),
            "n_counties": len(year_data),
            "n_valid": len(valid),
            "mean_net_return": valid["urban_net_return"].mean(),
            "median_net_return": valid["urban_net_return"].median(),
        })

    summary = {
        "n_years": len(years),
        "years": years,
        "n_counties": n_counties,
        "n_observations": len(panel),
        "n_balanced_counties": n_balanced,
        "n_unbalanced_counties": n_unbalanced,
        "year_statistics": year_stats,
    }

    return summary


def run_panel_pipeline(
    survey_years: list[int] | None = None,
    panel_type: str = "standard",
    soc_methodology: Literal["year_specific", "fixed_2023", "nearest"] = "nearest",
    output_format: Literal["long", "wide", "both"] = "both",
) -> pd.DataFrame:
    """
    Run complete panel pipeline from PUMA processing to final output.

    Args:
        survey_years: List of survey years. If None, uses panel_type preset.
        panel_type: Panel preset ("standard", "extended", "annual")
        soc_methodology: How to select SOC parameters
        output_format: Output format(s) to save

    Returns:
        Final panel DataFrame
    """
    # Get survey years from preset if not provided
    if survey_years is None:
        survey_years = RECOMMENDED_PANEL_YEARS.get(panel_type, [2013, 2018, 2023])

    console.print(f"[bold]Running panel pipeline[/bold]")
    console.print(f"  Survey years: {survey_years}")
    console.print(f"  SOC methodology: {soc_methodology}")
    console.print(f"  Output format: {output_format}")

    # Build the panel
    panel = build_panel_dataset(
        survey_years=survey_years,
        soc_methodology=soc_methodology,
        include_growth_rates=True,
    )

    # Save outputs
    save_panel(panel, output_format)

    # Print summary
    summary = get_panel_summary(panel)
    console.print(f"\n[bold]Panel Summary:[/bold]")
    console.print(f"  Years: {summary['years']}")
    console.print(f"  Counties: {summary['n_counties']}")
    console.print(f"  Observations: {summary['n_observations']}")
    console.print(f"  Balanced counties: {summary['n_balanced_counties']}")

    return panel
