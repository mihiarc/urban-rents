"""PUMS data processing for property value estimation."""

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track

from urban_rents.config import (
    ALL_US_STATES_FIPS,
    AVAILABLE_SURVEY_YEARS,
    DEFAULT_SURVEY_YEAR,
    PROCESSED_DIR,
    PUMSVariables,
    RAW_DIR,
    STATE_FIPS_TO_ABBR,
    get_period_label,
    get_puma_vintage,
    get_survey_type_label,
    is_one_year_survey,
    validate_survey_year,
)

console = Console()


def load_pums_housing_file(state_fips: str, survey_year: int | None = None) -> pd.DataFrame:
    """
    Load a single state's PUMS housing file for a specific survey year.

    Handles different file naming conventions and column names across ACS vintages:
    - 2000 (1-year): c2ssh{abbr}.csv with ST, YBL, PUMA columns
    - 2001-2008 (1-year): ss{yy}h{abbr}.csv with ST, YBL, PUMA columns
    - 2009-2021 (5-year): ss{yy}h{abbr}.csv or psam_h{abbr}.csv with ST, YBL, PUMA columns
    - 2022+ (5-year): psam_h{fips}.csv with STATE, YRBLT, PUMA columns

    Args:
        state_fips: 2-digit state FIPS code
        survey_year: Survey year. If None, uses DEFAULT_SURVEY_YEAR.

    Returns:
        DataFrame with PUMS housing records
    """
    if survey_year is None:
        survey_year = DEFAULT_SURVEY_YEAR

    state_fips = state_fips.zfill(2)
    abbr = STATE_FIPS_TO_ABBR.get(state_fips, "").lower()

    # Determine the PUMA vintage to know which PUMA column to use
    puma_vintage = get_puma_vintage(survey_year)
    is_one_year = is_one_year_survey(survey_year)

    # Try different file naming patterns based on survey year
    pums_path = None
    search_paths = []

    if is_one_year:
        # 1-year PUMS (2000-2008) naming conventions
        if survey_year == 2000:
            # 2000: c2ssh{abbr}.csv
            search_paths.append(RAW_DIR / "pums" / str(survey_year) / f"c2ssh{abbr}.csv")
        else:
            # 2001-2008: ss{yy}h{abbr}.csv
            year_suffix = str(survey_year)[-2:]
            search_paths.append(RAW_DIR / "pums" / str(survey_year) / f"ss{year_suffix}h{abbr}.csv")
    else:
        # 5-year PUMS (2009+) naming conventions
        # Pattern 1: psam_h{abbr}.csv (2022+ style)
        search_paths.append(RAW_DIR / "pums" / str(survey_year) / f"psam_h{abbr}.csv")

        # Pattern 2: psam_h{fips}.csv
        search_paths.append(RAW_DIR / "pums" / str(survey_year) / f"psam_h{state_fips}.csv")

        # Pattern 3: ss{yy}h{abbr}.csv (older 5-year files)
        year_suffix = str(survey_year)[-2:]
        search_paths.append(RAW_DIR / "pums" / str(survey_year) / f"ss{year_suffix}h{abbr}.csv")

    # Also check flat directory structure (legacy)
    search_paths.append(RAW_DIR / "pums" / f"psam_h{abbr}.csv")
    search_paths.append(RAW_DIR / "pums" / f"psam_h{state_fips}.csv")

    # Find first existing path
    for path in search_paths:
        if path.exists():
            pums_path = path
            break

    if pums_path is None:
        raise FileNotFoundError(
            f"PUMS file not found for state {state_fips} ({abbr.upper()}), "
            f"survey year {survey_year}. Looked in: "
            f"{RAW_DIR / 'pums' / str(survey_year)}"
        )

    # Column names vary by year - read all possible variants
    # STATE vs ST, YRBLT vs YBL, PUMA vs PUMA10 vs PUMA00 vs PUMA20
    columns_to_read = [
        "SERIALNO",  # Unique identifier
        "STATE",     # State code (2022+)
        "ST",        # State code (pre-2022)
        "PUMA",      # PUMA code (2012+)
        "PUMA20",    # 2020 PUMA code (2022+ files)
        "PUMA10",    # 2010 PUMA code (2013-2021 files may have this)
        "PUMA00",    # 2000 PUMA code (older files)
        "WGTP",      # Housing unit weight
        "TEN",       # Tenure
        "BLD",       # Building type
        "VALP",      # Property value (2005+)
        "VAL",       # Property value (2000-2004 1-year PUMS)
        "YRBLT",     # Year structure built (2022+)
        "YBL",       # Year built category (pre-2022)
    ]

    df = pd.read_csv(
        pums_path,
        usecols=lambda c: c in columns_to_read,
        dtype={
            "SERIALNO": str,
            "STATE": str,
            "ST": str,
            "PUMA": str,
            "PUMA20": str,
            "PUMA10": str,
            "PUMA00": str,
            "WGTP": float,
            "TEN": str,
            "BLD": str,
            "VALP": float,
            "VAL": str,  # Property value code in 2000-2004 PUMS (categorical)
            "YRBLT": str,
            "YBL": str,
        },
        low_memory=False,
    )

    # Standardize column names

    # State column: STATE -> ST
    if "STATE" in df.columns and "ST" not in df.columns:
        df = df.rename(columns={"STATE": "ST"})

    # Year built column: YBL -> YRBLT (need to handle different coding)
    if "YBL" in df.columns and "YRBLT" not in df.columns:
        # YBL uses different codes than YRBLT - map them
        # YBL codes (pre-2019): 01-21 representing year ranges
        # We'll keep it as-is and handle in filter function
        df = df.rename(columns={"YBL": "YRBLT"})

    # Property value column: VAL -> VALP (2000-2004 uses VAL categorical codes)
    # VAL is a categorical variable with codes 01-24 representing value ranges
    # We convert to midpoint dollar values for consistency with VALP
    if "VAL" in df.columns and "VALP" not in df.columns:
        # VAL code to midpoint value mapping (2000-2004 ACS)
        # Source: Census ACS PUMS Data Dictionary
        val_code_to_dollars = {
            "01": 5000,      # Less than $10,000
            "02": 12500,     # $10,000 - $14,999
            "03": 17500,     # $15,000 - $19,999
            "04": 22500,     # $20,000 - $24,999
            "05": 27500,     # $25,000 - $29,999
            "06": 32500,     # $30,000 - $34,999
            "07": 37500,     # $35,000 - $39,999
            "08": 42500,     # $40,000 - $49,999
            "09": 55000,     # $50,000 - $59,999
            "10": 67500,     # $60,000 - $69,999
            "11": 77500,     # $70,000 - $79,999
            "12": 87500,     # $80,000 - $89,999
            "13": 97500,     # $90,000 - $99,999
            "14": 112500,    # $100,000 - $124,999
            "15": 137500,    # $125,000 - $149,999
            "16": 175000,    # $150,000 - $174,999
            "17": 200000,    # $175,000 - $199,999
            "18": 225000,    # $200,000 - $249,999
            "19": 275000,    # $250,000 - $299,999
            "20": 350000,    # $300,000 - $399,999
            "21": 450000,    # $400,000 - $499,999
            "22": 625000,    # $500,000 - $749,999
            "23": 875000,    # $750,000 - $999,999
            "24": 1000000,   # $1,000,000 or more (use $1M as floor)
        }
        # Also handle single-digit codes (e.g., "1" instead of "01")
        val_code_to_dollars.update({
            str(int(k)): v for k, v in val_code_to_dollars.items()
        })
        df["VALP"] = df["VAL"].map(val_code_to_dollars)
        df = df.drop(columns=["VAL"])

    # PUMA column: Use appropriate vintage
    # For older ACS files with both PUMA00 and PUMA10, select the appropriate vintage
    # For newer files (2022+), PUMA20 contains 2020 vintage codes
    if "PUMA" not in df.columns:
        if puma_vintage == "2020" and "PUMA20" in df.columns:
            # For 2020 vintage (2022+ surveys), use PUMA20
            df = df.rename(columns={"PUMA20": "PUMA"})
            # Filter out records with missing PUMA (coded as -9)
            df = df[df["PUMA"] != "-9"].copy()
        elif puma_vintage == "2010" and "PUMA10" in df.columns:
            # For 2010 vintage, use PUMA10 and filter out missing values (-9)
            df = df.rename(columns={"PUMA10": "PUMA"})
            # Filter out records with missing PUMA (coded as -9)
            df = df[df["PUMA"] != "-9"].copy()
        elif puma_vintage == "2000" and "PUMA00" in df.columns:
            # For 2000 vintage, use PUMA00
            df = df.rename(columns={"PUMA00": "PUMA"})
            df = df[df["PUMA"] != "-9"].copy()
        elif "PUMA10" in df.columns:
            # Fall back to PUMA10 if available
            df = df.rename(columns={"PUMA10": "PUMA"})
            df = df[df["PUMA"] != "-9"].copy()
        elif "PUMA00" in df.columns:
            # Fall back to PUMA00 if no other option
            df = df.rename(columns={"PUMA00": "PUMA"})
            df = df[df["PUMA"] != "-9"].copy()

    # Standardize state FIPS to 2 digits
    if "ST" in df.columns:
        df["ST"] = df["ST"].astype(str).str.zfill(2)

    # Standardize PUMA to 5 digits
    if "PUMA" in df.columns:
        # Convert to string, handling any negative or invalid values
        df["PUMA"] = df["PUMA"].astype(int).astype(str).str.zfill(5)

    # Add survey year tracking
    df["survey_year"] = survey_year

    return df


def get_recent_built_codes(survey_year: int) -> list:
    """
    Get YRBLT/YBL codes for "recently built" homes based on survey year.

    The definition of "recent" varies by survey year to capture homes
    built within approximately 5-10 years of the survey period.

    YBL coding for 1-year ACS (2000-2008):
        1 = 1999 or earlier (varies; in some years means "recent")
        2 = 1990-1999 / 1995-1999 (varies)
        ...coding varies by year, generally:
        For 2000-2004: codes 1-2 represent newest housing
        For 2005-2008: codes 1-2 represent newest housing
            1 = [year] or later (e.g., 2005 or later)
            2 = 2000-2004

    YBL coding for ACS 5-year files (2009-2021):
        01 = 1939 or earlier
        02 = 1940-1949
        ...
        07 = 1990-1999
        08 = 2000-2004
        09 = 2005
        10 = 2006
        11 = 2007
        12 = 2008
        13 = 2009
        14 = 2010
        15 = 2011
        16 = 2012
        17 = 2013
        18 = 2014
        19 = 2015
        20 = 2016
        21 = 2017
        22 = 2018
        23 = 2019
        24 = 2020
        25 = 2021

    YRBLT coding for 2022+ ACS:
        "2010" = 2010-2019 (decade), then individual years "2020", "2021", etc.

    Args:
        survey_year: Survey year (1-year for 2000-2008, 5-year for 2009+)

    Returns:
        List of YRBLT/YBL codes to include (may be int or str depending on vintage)
    """
    if survey_year >= 2021:
        # 2021+ ACS uses new YRBLT coding with decade and individual years
        # "2010" represents 2010-2019 decade, then individual years 2020, 2021, etc.
        # Include years up to the survey end year
        recent_codes = ["2010"]  # 2010-2019 decade
        for year in range(2020, survey_year + 1):
            recent_codes.append(str(year))
        return recent_codes

    elif survey_year >= 2018:
        # 2018-2020 ACS (5-year) uses numeric YBL codes
        # For recently built (2010+), codes are 14 onward
        # Max code depends on survey end year
        max_code = min(14 + (survey_year - 2010), 24)  # 2010=14, caps at 24 (2020)
        return list(range(14, max_code + 1))

    elif survey_year >= 2014:
        # 2014-2017 ACS (5-year): For recently built (2010+), codes 14 onward
        max_code = 14 + (survey_year - 2010)  # e.g., 2017 -> code 21 (2017)
        return list(range(14, max_code + 1))

    elif survey_year >= 2010:
        # 2010-2013 ACS (5-year): For recently built (2005+), codes 9 onward
        # Code 9=2005, ..., code 17=2013
        max_code = 9 + (survey_year - 2005)  # e.g., 2013 -> code 17
        return list(range(9, max_code + 1))

    elif survey_year >= 2009:
        # 2009 ACS (first 5-year): YBL codes
        # Code 8=2000-2004, 9=2005+
        return list(range(8, 10))

    elif survey_year >= 2005:
        # 2005-2008 1-year ACS: YBL codes
        # 1 = [survey_year] or later
        # 2 = 2000-2004
        # Include both for "recent" construction
        return [1, 2]

    else:
        # 2000-2004 1-year ACS: YBL codes
        # 1 = 1999 or later (most recent category)
        # 2 = 1995-1999 (varies)
        # For these years, code 1 represents most recent construction
        return [1, 2]


def filter_recently_built_owner_occupied(
    df: pd.DataFrame,
    survey_year: int | None = None,
) -> pd.DataFrame:
    """
    Filter PUMS data to recently built, owner-occupied, single-family homes.

    Args:
        df: Raw PUMS housing DataFrame
        survey_year: Survey year (used to determine "recent" definition)

    Returns:
        Filtered DataFrame
    """
    # Get survey year from data if not provided
    if survey_year is None:
        if "survey_year" in df.columns:
            survey_year = df["survey_year"].iloc[0]
        else:
            survey_year = DEFAULT_SURVEY_YEAR

    # Owner-occupied: TEN in [1, 2] (owned with mortgage or free and clear)
    # Handle both string and numeric types across ACS vintages
    owner_codes = [
        PUMSVariables.TEN_OWNED_MORTGAGE,  # "1"
        PUMSVariables.TEN_OWNED_FREE,      # "2"
        1, 1.0, 2, 2.0,  # numeric variants
    ]
    owner_mask = df["TEN"].isin(owner_codes)

    # Single-family: BLD in [02, 03] (detached or attached)
    # Handle both string and numeric types
    sf_codes = [
        PUMSVariables.BLD_SF_DETACHED,  # "02"
        PUMSVariables.BLD_SF_ATTACHED,  # "03"
        "2", "3",  # without leading zero
        2, 2.0, 3, 3.0,  # numeric variants
    ]
    sf_mask = df["BLD"].isin(sf_codes)

    # Recently built - use year-appropriate codes
    recent_codes = get_recent_built_codes(survey_year)
    # Also add float versions of numeric codes for older ACS
    all_recent_codes = recent_codes.copy()
    for code in recent_codes:
        if isinstance(code, int):
            all_recent_codes.append(float(code))
            all_recent_codes.append(str(code))
    recent_mask = df["YRBLT"].isin(all_recent_codes)

    # Valid property value (not null, greater than 0)
    value_mask = df["VALP"].notna() & (df["VALP"] > 0)

    # Valid weight
    weight_mask = df["WGTP"].notna() & (df["WGTP"] > 0)

    filtered = df[owner_mask & sf_mask & recent_mask & value_mask & weight_mask].copy()

    return filtered


def calculate_puma_property_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weighted mean property values at PUMA level.

    Args:
        df: Filtered PUMS housing DataFrame

    Returns:
        DataFrame with PUMA-level property value statistics
    """
    # Include survey_year in grouping if present
    group_cols = ["ST", "PUMA"]
    if "survey_year" in df.columns:
        group_cols.append("survey_year")

    grouped = df.groupby(group_cols)

    # Calculate weighted statistics
    def weighted_stats(group: pd.DataFrame) -> pd.Series:
        weights = group["WGTP"]
        values = group["VALP"]
        total_weight = weights.sum()

        if total_weight == 0:
            return pd.Series({
                "mean_property_value": np.nan,
                "median_property_value": np.nan,
                "n_observations": 0,
                "total_weight": 0,
            })

        # Weighted mean
        weighted_mean = np.average(values, weights=weights)

        # Weighted median (approximate)
        sorted_idx = np.argsort(values)
        sorted_values = values.iloc[sorted_idx]
        sorted_weights = weights.iloc[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, total_weight / 2)
        weighted_median = sorted_values.iloc[min(median_idx, len(sorted_values) - 1)]

        return pd.Series({
            "mean_property_value": weighted_mean,
            "median_property_value": weighted_median,
            "n_observations": len(group),
            "total_weight": total_weight,
        })

    puma_stats = grouped.apply(weighted_stats, include_groups=False).reset_index()

    return puma_stats


def process_all_states(survey_year: int | None = None) -> pd.DataFrame:
    """
    Process PUMS data for all US states (including Alaska and Hawaii) for a specific survey year.

    Args:
        survey_year: Survey year (2000-2008 for 1-year, 2009+ for 5-year).
                    If None, uses DEFAULT_SURVEY_YEAR.

    Returns:
        DataFrame with PUMA-level property values for all states
    """
    if survey_year is None:
        survey_year = DEFAULT_SURVEY_YEAR
    else:
        validate_survey_year(survey_year)

    all_puma_stats = []
    period_label = get_period_label(survey_year)
    puma_vintage = get_puma_vintage(survey_year)
    survey_type = get_survey_type_label(survey_year)

    states = sorted(ALL_US_STATES_FIPS)
    console.print(f"[bold]Processing PUMS data for {len(states)} states ({period_label}, {survey_type})...[/bold]")

    for state_fips in track(states, description=f"Processing {period_label}"):
        try:
            # Load state data
            df = load_pums_housing_file(state_fips, survey_year)

            # Filter to relevant housing units
            filtered = filter_recently_built_owner_occupied(df, survey_year)

            if len(filtered) == 0:
                console.print(f"[yellow]No qualifying records for state {state_fips}[/yellow]")
                continue

            # Calculate PUMA-level statistics
            puma_stats = calculate_puma_property_values(filtered)
            all_puma_stats.append(puma_stats)

        except FileNotFoundError:
            console.print(f"[red]PUMS file not found for state {state_fips}[/red]")
        except Exception as e:
            console.print(f"[red]Error processing state {state_fips}: {e}[/red]")

    if not all_puma_stats:
        raise ValueError(f"No PUMS data processed successfully for {period_label}")

    # Combine all states
    combined = pd.concat(all_puma_stats, ignore_index=True)

    # Ensure survey_year column exists
    if "survey_year" not in combined.columns:
        combined["survey_year"] = survey_year

    # Add metadata columns
    combined["period_label"] = period_label
    combined["puma_vintage"] = puma_vintage

    # Create state-PUMA identifier
    combined["state_puma"] = combined["ST"] + "_" + combined["PUMA"]

    console.print(f"[green]Processed {len(combined)} PUMAs for {period_label}[/green]")

    return combined


def process_panel_years(survey_years: list[int]) -> pd.DataFrame:
    """
    Process PUMS data for multiple survey years to build a panel dataset.

    Args:
        survey_years: List of survey end years to process

    Returns:
        Combined DataFrame with PUMA-level property values across all years
    """
    console.print(f"[bold]Building panel dataset for years: {survey_years}[/bold]")

    all_years_data = []

    for year in survey_years:
        console.print(f"\n[bold cyan]Processing {get_period_label(year)}...[/bold cyan]")
        try:
            year_data = process_all_states(year)
            all_years_data.append(year_data)
        except Exception as e:
            console.print(f"[red]Failed to process {year}: {e}[/red]")
            continue

    if not all_years_data:
        raise ValueError("No survey years processed successfully")

    # Combine all years
    panel = pd.concat(all_years_data, ignore_index=True)

    console.print(f"\n[green]Panel dataset: {len(panel)} PUMA-year observations[/green]")
    console.print(f"[green]Years: {sorted(panel['survey_year'].unique())}[/green]")

    return panel


def save_puma_property_values(
    df: pd.DataFrame,
    survey_year: int | None = None,
    filename: str | None = None,
) -> Path:
    """
    Save PUMA-level property values to file.

    Args:
        df: PUMA-level property values DataFrame
        survey_year: Survey year (for naming). If None, inferred from data.
        filename: Output filename. If None, auto-generated.

    Returns:
        Path to saved file
    """
    if survey_year is None and "survey_year" in df.columns:
        survey_year = df["survey_year"].iloc[0]

    if filename is None:
        if survey_year is not None:
            filename = f"puma_property_values_{survey_year}.parquet"
        else:
            filename = "puma_property_values.parquet"

    output_path = PROCESSED_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    console.print(f"[green]Saved PUMA property values to {output_path}[/green]")

    return output_path


def save_panel_property_values(
    df: pd.DataFrame,
    filename: str = "puma_property_values_panel.parquet",
) -> Path:
    """
    Save panel of PUMA-level property values to file.

    Args:
        df: Panel DataFrame with multiple survey years
        filename: Output filename

    Returns:
        Path to saved file
    """
    output_path = PROCESSED_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)

    years = sorted(df["survey_year"].unique())
    console.print(f"[green]Saved panel ({len(years)} years) to {output_path}[/green]")

    return output_path


def load_puma_property_values(
    survey_year: int | None = None,
    filename: str | None = None,
) -> pd.DataFrame:
    """
    Load PUMA-level property values from file.

    Args:
        survey_year: Survey year to load. If None and filename not provided,
                     loads default file.
        filename: Input filename. If provided, survey_year is ignored.

    Returns:
        PUMA-level property values DataFrame
    """
    if filename is None:
        if survey_year is not None:
            filename = f"puma_property_values_{survey_year}.parquet"
        else:
            filename = "puma_property_values.parquet"

    input_path = PROCESSED_DIR / filename

    if not input_path.exists():
        raise FileNotFoundError(f"PUMA property values file not found: {input_path}")

    return pd.read_parquet(input_path)


def load_panel_property_values(
    filename: str = "puma_property_values_panel.parquet",
) -> pd.DataFrame:
    """
    Load panel of PUMA-level property values from file.

    Args:
        filename: Input filename

    Returns:
        Panel DataFrame with multiple survey years
    """
    input_path = PROCESSED_DIR / filename
    if not input_path.exists():
        raise FileNotFoundError(f"Panel file not found: {input_path}")

    return pd.read_parquet(input_path)


def get_puma_summary_stats(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for PUMA-level property values.

    Args:
        df: PUMA-level property values DataFrame

    Returns:
        Dictionary of summary statistics
    """
    valid_pumas = df[df["mean_property_value"].notna()]

    stats = {
        "total_pumas": len(df),
        "pumas_with_data": len(valid_pumas),
        "pumas_missing_data": len(df) - len(valid_pumas),
        "mean_property_value": valid_pumas["mean_property_value"].mean(),
        "median_property_value": valid_pumas["median_property_value"].median(),
        "min_property_value": valid_pumas["mean_property_value"].min(),
        "max_property_value": valid_pumas["mean_property_value"].max(),
        "total_observations": valid_pumas["n_observations"].sum(),
    }

    # Add year-specific stats if panel
    if "survey_year" in df.columns:
        stats["survey_years"] = sorted(df["survey_year"].unique().tolist())
        stats["n_years"] = df["survey_year"].nunique()

    return stats
