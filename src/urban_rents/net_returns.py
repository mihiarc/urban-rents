"""Urban net returns calculation and aggregation."""

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from urban_rents.config import (
    CENSUS_DIVISIONS,
    DEFAULT_SURVEY_YEAR,
    DISCOUNT_RATE,
    OUTPUT_DIR,
    STATE_FIPS_TO_ABBR,
    STATE_TO_DIVISION,
    get_period_label,
    get_puma_vintage,
    get_soc_division_data,
)
from urban_rents.crosswalk import (
    apply_puma_size_adjustment,
    get_crosswalk_for_year,
    get_crosswalk_quality_flags,
    load_crosswalk,
)
from urban_rents.pums_processing import load_puma_property_values

console = Console()


def calculate_county_property_values(
    puma_values: pd.DataFrame,
    crosswalk: pd.DataFrame,
    use_adjusted_weights: bool = False,
) -> pd.DataFrame:
    """
    Calculate county-level property values from PUMA-level values using crosswalk.

    Args:
        puma_values: PUMA-level property values with state_puma identifier
        crosswalk: PUMA-to-county crosswalk with weights
        use_adjusted_weights: If True, use PUMA-size-adjusted weights

    Returns:
        DataFrame with county-level property values
    """
    # Merge PUMA values with crosswalk
    weight_col = "adjusted_weight" if use_adjusted_weights else "weight"

    merged = crosswalk.merge(
        puma_values[["state_puma", "mean_property_value", "n_observations", "total_weight"]],
        on="state_puma",
        how="left",
    )

    # Flag PUMAs with missing data
    missing_mask = merged["mean_property_value"].isna()
    if missing_mask.sum() > 0:
        console.print(f"[yellow]Warning: {missing_mask.sum()} PUMA-county pairs missing property values[/yellow]")

    # For PUMAs with missing values, we'll exclude them and renormalize weights
    valid_data = merged[~missing_mask].copy()

    # Renormalize weights within each county
    county_weight_totals = valid_data.groupby("county_geoid")[weight_col].transform("sum")
    valid_data["normalized_weight"] = valid_data[weight_col] / county_weight_totals

    # Calculate weighted county property values
    valid_data["weighted_value"] = (
        valid_data["mean_property_value"] * valid_data["normalized_weight"]
    )

    # Compute max_puma_weight if not in crosswalk
    if "max_puma_weight" not in valid_data.columns:
        valid_data["max_puma_weight"] = valid_data.groupby("county_geoid")[weight_col].transform("max")

    # Aggregate to county level
    county_values = valid_data.groupby("county_geoid").agg(
        state_fips=("state_fips", "first"),
        county_fips=("county_fips", "first"),
        county_name=("county_name", "first"),
        mean_property_value=("weighted_value", "sum"),
        n_pumas=("puma", "count"),
        n_pumas_with_data=("mean_property_value", lambda x: x.notna().sum()),
        max_puma_weight=("max_puma_weight", "first"),
        total_observations=("n_observations", "sum"),
    ).reset_index()

    return county_values


def add_soc_division_data(
    county_values: pd.DataFrame,
    survey_year: int | None = None,
) -> pd.DataFrame:
    """
    Add Survey of Construction division-level data to county values.

    Args:
        county_values: County-level property values
        survey_year: Survey year to determine SOC data. If None, uses nearest.

    Returns:
        DataFrame with SOC data added
    """
    soc_data = get_soc_division_data(survey_year)

    # Map state FIPS to Census Division
    county_values["census_division"] = county_values["state_fips"].map(STATE_TO_DIVISION)
    county_values["division_name"] = county_values["census_division"].map(CENSUS_DIVISIONS)

    # Add SOC data for each county based on division
    county_values["lot_share"] = county_values["census_division"].apply(
        lambda d: soc_data[d].lot_share if d in soc_data else np.nan
    )
    county_values["lot_acres"] = county_values["census_division"].apply(
        lambda d: soc_data[d].lot_acres if d in soc_data else np.nan
    )
    county_values["median_lot_value_division"] = county_values["census_division"].apply(
        lambda d: soc_data[d].median_lot_value if d in soc_data else np.nan
    )

    return county_values


def calculate_urban_net_returns(county_values: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate annualized urban net returns for each county.

    Formula: NR_annual = (SalesPrice × LotShare) / LotAcres × DiscountRate

    Args:
        county_values: County-level property values with SOC data

    Returns:
        DataFrame with urban net returns calculated
    """
    # Calculate lot value per acre
    county_values["lot_value"] = (
        county_values["mean_property_value"] * county_values["lot_share"]
    )

    # Calculate urban net return (land value per acre)
    county_values["urban_net_return_raw"] = (
        county_values["lot_value"] / county_values["lot_acres"]
    )

    # Annualize at discount rate
    county_values["urban_net_return"] = (
        county_values["urban_net_return_raw"] * DISCOUNT_RATE
    )

    # Add state abbreviation
    county_values["state_abbr"] = county_values["state_fips"].map(STATE_FIPS_TO_ABBR)

    return county_values


def calculate_adjusted_returns(
    puma_values: pd.DataFrame,
    crosswalk: pd.DataFrame,
    survey_year: int | None = None,
) -> pd.DataFrame:
    """
    Calculate urban net returns with PUMA-size adjustment.

    Args:
        puma_values: PUMA-level property values
        crosswalk: PUMA-to-county crosswalk
        survey_year: Survey year for SOC data

    Returns:
        DataFrame with adjusted returns
    """
    # Apply PUMA size adjustment
    adjusted_crosswalk = apply_puma_size_adjustment(crosswalk.copy())

    # Calculate county values with adjusted weights
    adjusted_county_values = calculate_county_property_values(
        puma_values, adjusted_crosswalk, use_adjusted_weights=True
    )

    # Add SOC data and calculate returns
    adjusted_county_values = add_soc_division_data(adjusted_county_values, survey_year)
    adjusted_county_values = calculate_urban_net_returns(adjusted_county_values)

    return adjusted_county_values[["county_geoid", "urban_net_return"]].rename(
        columns={"urban_net_return": "urban_net_return_adjusted"}
    )


def build_final_dataset(
    puma_values: pd.DataFrame | None = None,
    crosswalk: pd.DataFrame | None = None,
    survey_year: int | None = None,
) -> pd.DataFrame:
    """
    Build the complete urban net returns dataset for a single survey year.

    Args:
        puma_values: PUMA-level property values (loads from file if None)
        crosswalk: PUMA-to-county crosswalk (loads from file if None)
        survey_year: Survey year. If None, uses DEFAULT_SURVEY_YEAR.

    Returns:
        Complete urban net returns DataFrame
    """
    if survey_year is None:
        survey_year = DEFAULT_SURVEY_YEAR

    console.print(f"[bold]Building urban net returns dataset for {get_period_label(survey_year)}...[/bold]")

    # Load data if not provided
    if puma_values is None:
        puma_values = load_puma_property_values(survey_year)

    if crosswalk is None:
        crosswalk = get_crosswalk_for_year(survey_year)

    # Calculate standard county property values
    console.print("[cyan]Calculating county property values...[/cyan]")
    county_values = calculate_county_property_values(puma_values, crosswalk)

    # Add SOC data
    console.print("[cyan]Adding Survey of Construction data...[/cyan]")
    county_values = add_soc_division_data(county_values, survey_year)

    # Calculate standard urban net returns
    console.print("[cyan]Calculating urban net returns...[/cyan]")
    county_values = calculate_urban_net_returns(county_values)

    # Calculate adjusted returns
    console.print("[cyan]Calculating PUMA-size-adjusted returns...[/cyan]")
    adjusted_returns = calculate_adjusted_returns(puma_values, crosswalk, survey_year)

    # Merge adjusted returns
    county_values = county_values.merge(adjusted_returns, on="county_geoid", how="left")

    # Get quality flags
    quality_flags = get_crosswalk_quality_flags(crosswalk)
    county_values = county_values.merge(
        quality_flags[["county_geoid", "data_quality_flag"]],
        on="county_geoid",
        how="left",
    )

    # Add year metadata
    county_values["survey_year"] = survey_year
    county_values["period_label"] = get_period_label(survey_year)
    county_values["puma_vintage"] = get_puma_vintage(survey_year)

    # Select and order final columns
    final_columns = [
        "state_fips",
        "county_fips",
        "county_geoid",
        "county_name",
        "state_abbr",
        "census_division",
        "division_name",
        "survey_year",
        "period_label",
        "puma_vintage",
        "mean_property_value",
        "lot_share",
        "lot_acres",
        "lot_value",
        "urban_net_return",
        "urban_net_return_adjusted",
        "n_pumas",
        "n_pumas_with_data",
        "max_puma_weight",
        "total_observations",
        "data_quality_flag",
    ]

    final = county_values[[c for c in final_columns if c in county_values.columns]].copy()

    # Sort by state and county
    final = final.sort_values(["state_fips", "county_fips"]).reset_index(drop=True)

    console.print(f"[green]Built dataset with {len(final)} counties for {get_period_label(survey_year)}[/green]")

    return final


def save_final_dataset(
    df: pd.DataFrame,
    filename: str | None = None,
    parquet: bool = True,
    survey_year: int | None = None,
) -> list[Path]:
    """
    Save the final dataset to file(s).

    Args:
        df: Final urban net returns DataFrame
        filename: Output CSV filename. If None, auto-generated with year.
        parquet: If True, also save as parquet
        survey_year: Survey year for filename. Inferred from data if None.

    Returns:
        List of paths to saved files
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # Determine filename
    if filename is None:
        if survey_year is None and "survey_year" in df.columns:
            survey_year = df["survey_year"].iloc[0]
        if survey_year is not None:
            filename = f"urban_net_returns_{survey_year}.csv"
        else:
            filename = "urban_net_returns.csv"

    # Save CSV
    csv_path = OUTPUT_DIR / filename
    df.to_csv(csv_path, index=False)
    saved_files.append(csv_path)
    console.print(f"[green]Saved CSV to {csv_path}[/green]")

    # Save parquet
    if parquet:
        parquet_path = OUTPUT_DIR / filename.replace(".csv", ".parquet")
        df.to_parquet(parquet_path, index=False)
        saved_files.append(parquet_path)
        console.print(f"[green]Saved Parquet to {parquet_path}[/green]")

    return saved_files


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the urban net returns dataset.

    Args:
        df: Urban net returns DataFrame

    Returns:
        Dictionary of summary statistics
    """
    stats = {
        "total_counties": len(df),
        "counties_with_data": df["urban_net_return"].notna().sum(),
        "counties_missing_data": df["urban_net_return"].isna().sum(),
        "urban_net_return": {
            "mean": df["urban_net_return"].mean(),
            "median": df["urban_net_return"].median(),
            "std": df["urban_net_return"].std(),
            "min": df["urban_net_return"].min(),
            "max": df["urban_net_return"].max(),
            "p25": df["urban_net_return"].quantile(0.25),
            "p75": df["urban_net_return"].quantile(0.75),
        },
        "property_values": {
            "mean": df["mean_property_value"].mean(),
            "median": df["mean_property_value"].median(),
            "min": df["mean_property_value"].min(),
            "max": df["mean_property_value"].max(),
        },
        "quality_distribution": df["data_quality_flag"].value_counts().to_dict(),
    }

    # Add survey year info if available
    if "survey_year" in df.columns:
        stats["survey_year"] = int(df["survey_year"].iloc[0])
        stats["period_label"] = df["period_label"].iloc[0] if "period_label" in df.columns else None

    # By division
    division_stats = df.groupby("division_name").agg(
        n_counties=("county_geoid", "count"),
        mean_net_return=("urban_net_return", "mean"),
        median_net_return=("urban_net_return", "median"),
        mean_property_value=("mean_property_value", "mean"),
    ).to_dict(orient="index")

    stats["by_division"] = division_stats

    return stats


def print_summary_statistics(df: pd.DataFrame) -> None:
    """
    Print formatted summary statistics to console.

    Args:
        df: Urban net returns DataFrame
    """
    stats = generate_summary_statistics(df)

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    title = "URBAN NET RETURNS SUMMARY STATISTICS"
    if "period_label" in stats and stats["period_label"]:
        title += f" ({stats['period_label']})"
    console.print(f"[bold cyan]              {title}              [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    console.print(f"[bold]Total Counties:[/bold] {stats['total_counties']}")
    console.print(f"[bold]Counties with Data:[/bold] {stats['counties_with_data']}")
    console.print(f"[bold]Counties Missing Data:[/bold] {stats['counties_missing_data']}")

    console.print("\n[bold underline]Urban Net Returns ($/acre/year)[/bold underline]")
    nr = stats["urban_net_return"]
    console.print(f"  Mean:   ${nr['mean']:,.0f}")
    console.print(f"  Median: ${nr['median']:,.0f}")
    console.print(f"  Std:    ${nr['std']:,.0f}")
    console.print(f"  Min:    ${nr['min']:,.0f}")
    console.print(f"  Max:    ${nr['max']:,.0f}")
    console.print(f"  IQR:    ${nr['p25']:,.0f} - ${nr['p75']:,.0f}")

    console.print("\n[bold underline]Property Values ($)[/bold underline]")
    pv = stats["property_values"]
    console.print(f"  Mean:   ${pv['mean']:,.0f}")
    console.print(f"  Median: ${pv['median']:,.0f}")
    console.print(f"  Range:  ${pv['min']:,.0f} - ${pv['max']:,.0f}")

    console.print("\n[bold underline]Data Quality Distribution[/bold underline]")
    for flag, count in stats["quality_distribution"].items():
        console.print(f"  {flag}: {count} counties")

    console.print("\n[bold underline]By Census Division[/bold underline]")
    for division, div_stats in stats["by_division"].items():
        console.print(f"  [bold]{division}[/bold]")
        console.print(f"    Counties: {div_stats['n_counties']}")
        console.print(f"    Mean Net Return: ${div_stats['mean_net_return']:,.0f}/acre/year")
        console.print(f"    Mean Property Value: ${div_stats['mean_property_value']:,.0f}")

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
