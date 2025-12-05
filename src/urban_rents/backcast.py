"""Hybrid backcasting methodology for extending panel data to 2000.

This module implements Option 4 (Hybrid Approach) for data quality improvement:

1. Year 2000: Census 2000 SF3 median house values (direct observation)
2. Years 2001-2004: HPI-based interpolation forward from Census 2000 anchor
3. Years 2005-2008: Smoothed 1-year ACS blended with HPI trends
4. Years 2009-2023: Original 5-year ACS (no modification)

Data Provenance Tracking:
- acs_5year: Original 5-year ACS estimate (2009+)
- acs_1year: Original 1-year ACS estimate (2005-2008)
- hpi_forward: Interpolated forward from Census 2000 using HPI (2001-2004)
- hpi_smoothed: 1-year ACS smoothed with HPI (2005-2008)
- census_2000_sf3: Census 2000 Summary File 3 median house value (2000)
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rich.console import Console

from urban_rents.census2000 import load_census_2000_property_values
from urban_rents.config import OUTPUT_DIR, PROCESSED_DIR
from urban_rents.county_fixes import apply_all_county_fixes
from urban_rents.hpi import (
    backcast_property_values,
    get_hpi_coverage_stats,
    get_hpi_ratio_with_fallback,
    get_hpi_with_state_fallback,
    load_hpi_data,
    smooth_volatile_estimates,
)

console = Console()


# Year classification for hybrid methodology
CENSUS_2000_YEAR = 2000  # Census 2000 SF3 direct observation
HPI_FORWARD_YEARS = [2001, 2002, 2003, 2004]  # HPI forward from Census 2000
SMOOTH_YEARS = [2005, 2006, 2007, 2008]  # 1-year ACS + HPI smoothing
INTERPOLATE_GAP_YEARS = [2005, 2006, 2007, 2008, 2009, 2010, 2011]  # Years to fill gaps via interpolation
STABLE_YEARS = list(range(2009, 2024))  # 5-year ACS (no modification)
FIRST_FULL_COVERAGE_YEAR = 2012  # First year with ~100% ACS coverage

# Anchor years
CENSUS_2000_ANCHOR = 2000  # Census 2000 SF3 anchor for forward interpolation
DEFAULT_STABLE_ANCHOR = 2009  # ACS 5-year anchor for smoothing

# Smoothing weight for volatile years (0.5 = equal blend)
DEFAULT_SMOOTHING_WEIGHT = 0.5


class HybridBackcaster:
    """
    Applies hybrid backcasting methodology to extend panel data.

    Methodology:
    1. Year 2000: Use Census 2000 SF3 median house values (direct observation)
    2. Years 2001-2004: HPI forward interpolation from Census 2000
    3. Years 2005-2008: Smooth 1-year ACS by blending with HPI trends
    4. Years 2009+: Keep 5-year ACS unchanged
    5. Track data provenance for all records
    """

    def __init__(
        self,
        stable_anchor_year: int = DEFAULT_STABLE_ANCHOR,
        smoothing_weight: float = DEFAULT_SMOOTHING_WEIGHT,
        hpi_source: Literal["fhfa", "zillow"] = "fhfa",
        use_census_2000: bool = True,
    ):
        """
        Initialize the backcaster.

        Args:
            stable_anchor_year: Stable ACS reference year for smoothing (typically 2009)
            smoothing_weight: Weight for HPI in smoothing (0-1)
            hpi_source: HPI data source
            use_census_2000: If True, use Census 2000 SF3 for year 2000 anchor
        """
        self.stable_anchor_year = stable_anchor_year
        self.smoothing_weight = smoothing_weight
        self.hpi_source = hpi_source
        self.use_census_2000 = use_census_2000

        # Load HPI data
        console.print(f"[bold]Loading {hpi_source.upper()} HPI data...[/bold]")
        self.hpi_df = load_hpi_data(hpi_source)
        console.print(f"  Loaded HPI for {self.hpi_df['county_fips'].nunique()} counties")
        console.print(f"  Years: {self.hpi_df['year'].min()}-{self.hpi_df['year'].max()}")

        # Load Census 2000 data if using
        if use_census_2000:
            console.print("[bold]Loading Census 2000 SF3 data...[/bold]")
            self.census_2000_df = load_census_2000_property_values()
            console.print(f"  Loaded {len(self.census_2000_df)} county records")

    def apply_hybrid_methodology(
        self,
        panel_df: pd.DataFrame,
        value_column: str = "mean_property_value",
    ) -> pd.DataFrame:
        """
        Apply the full hybrid backcasting methodology to a panel.

        Args:
            panel_df: Original panel data
            value_column: Property value column to process

        Returns:
            Enhanced panel with backcasted/smoothed values and provenance tracking
        """
        console.print("\n[bold cyan]Applying Hybrid Backcasting Methodology[/bold cyan]")

        # Add county_fips for HPI matching
        panel_df = panel_df.copy()
        if "county_fips" not in panel_df.columns and "county_geoid" in panel_df.columns:
            panel_df["county_fips"] = panel_df["county_geoid"].str[:5]

        # Initialize data_source column
        if "data_source" not in panel_df.columns:
            panel_df["data_source"] = None

        # Separate data by methodology
        years_in_panel = set(panel_df["survey_year"].unique())

        all_data = []

        # 1. Stable years (5-year ACS) - keep unchanged
        stable_years_present = [y for y in STABLE_YEARS if y in years_in_panel]
        stable_data = panel_df[panel_df["survey_year"].isin(stable_years_present)].copy()
        stable_data["data_source"] = "acs_5year"
        console.print(f"\n[green]Stable years (unchanged): {stable_years_present}[/green]")
        if len(stable_data) > 0:
            all_data.append(stable_data)

        # 2. Volatile years (1-year ACS) - smooth with HPI
        smooth_years_present = [y for y in SMOOTH_YEARS if y in years_in_panel]
        if smooth_years_present:
            console.print(f"\n[yellow]Smoothing volatile years: {smooth_years_present}[/yellow]")
            volatile_data = panel_df[panel_df["survey_year"].isin(smooth_years_present)].copy()

            smoothed_data = smooth_volatile_estimates(
                panel_df=volatile_data,
                hpi_df=self.hpi_df,
                volatile_years=smooth_years_present,
                stable_anchor_year=self.stable_anchor_year,
                value_column=value_column,
                smoothing_weight=self.smoothing_weight,
            )
            if len(smoothed_data) > 0:
                all_data.append(smoothed_data)

        # 3. Year 2000 - use Census 2000 SF3 if available
        if CENSUS_2000_YEAR in years_in_panel and self.use_census_2000:
            console.print(f"\n[cyan]Year 2000: Using Census 2000 SF3 data[/cyan]")
            census_2000_records = self._create_census_2000_records(panel_df, value_column)
            if len(census_2000_records) > 0:
                all_data.append(census_2000_records)

        # 4. HPI forward years (2001-2004) - interpolate from Census 2000
        forward_years_needed = [y for y in HPI_FORWARD_YEARS if y in years_in_panel]
        if forward_years_needed and self.use_census_2000:
            console.print(f"\n[yellow]HPI forward interpolation: {forward_years_needed}[/yellow]")
            forward_data = self._forward_interpolate_from_census_2000(
                panel_df, forward_years_needed, value_column
            )
            if len(forward_data) > 0:
                all_data.append(forward_data)

        if not all_data:
            raise ValueError("No data after applying hybrid methodology")

        result = pd.concat(all_data, ignore_index=True)

        # 5. Fill remaining gaps in 2005-2011 (counties missing from 1-year/early 5-year ACS)
        if self.use_census_2000 and FIRST_FULL_COVERAGE_YEAR in years_in_panel:
            gap_fill_data = self._fill_gaps_2005_2011(result, value_column)
            if len(gap_fill_data) > 0:
                # Merge gap fills with existing data
                result = pd.concat([result, gap_fill_data], ignore_index=True)
                # Remove duplicates (keep gap-fill only for counties/years without data)
                result = result.sort_values(
                    ["county_geoid", "survey_year", "data_source"],
                    ascending=[True, True, False],  # Keep non-interpolated first
                )
                result = result.drop_duplicates(
                    subset=["county_geoid", "survey_year"],
                    keep="first"
                )

        # 6. Apply county fixes for special cases (Oglala Lakota, Broomfield, CT regions)
        if self.use_census_2000:
            # Ensure county_fips column exists for fixes
            if "county_fips" not in result.columns:
                result["county_fips"] = result["county_geoid"].str[:5]

            result = apply_all_county_fixes(
                result,
                self.census_2000_df,
                value_column,
            )

        # Clean up temporary columns
        if "county_fips" in result.columns:
            result = result.drop(columns=["county_fips"])

        # Sort by county and year
        result = result.sort_values(["county_geoid", "survey_year"]).reset_index(drop=True)

        # Report summary
        self._report_summary(result)

        return result

    def _create_census_2000_records(
        self,
        panel_df: pd.DataFrame,
        value_column: str,
    ) -> pd.DataFrame:
        """Create year 2000 records from Census 2000 SF3 data."""
        # Get template from a stable year
        template_year = max([y for y in STABLE_YEARS if y in panel_df["survey_year"].unique()])
        template = panel_df[panel_df["survey_year"] == template_year].copy()

        # Merge with Census 2000 values
        census_df = self.census_2000_df.copy()
        census_df["county_fips"] = census_df["county_fips"].astype(str).str.zfill(5)

        result = template.merge(
            census_df[["county_fips", "median_property_value"]],
            on="county_fips",
            how="left",
        )

        # Update values
        result["survey_year"] = CENSUS_2000_YEAR
        result[value_column] = result["median_property_value"]
        result["data_source"] = "census_2000_sf3"

        # Drop temporary columns
        result = result.drop(columns=["median_property_value"], errors="ignore")

        n_matched = result[value_column].notna().sum()
        console.print(f"  Census 2000 coverage: {n_matched}/{len(result)} ({100*n_matched/len(result):.1f}%)")

        return result

    def _forward_interpolate_from_census_2000(
        self,
        panel_df: pd.DataFrame,
        target_years: list[int],
        value_column: str,
    ) -> pd.DataFrame:
        """Forward interpolate property values from Census 2000 using HPI.

        Uses state-level HPI fallback for counties without county-level HPI data.
        """
        # Get template from a stable year
        template_year = max([y for y in STABLE_YEARS if y in panel_df["survey_year"].unique()])
        template = panel_df[panel_df["survey_year"] == template_year].copy()

        # Merge with Census 2000 values
        census_df = self.census_2000_df.copy()
        census_df["county_fips"] = census_df["county_fips"].astype(str).str.zfill(5)

        anchor = template.merge(
            census_df[["county_fips", "median_property_value"]],
            on="county_fips",
            how="left",
        )
        anchor[value_column] = anchor["median_property_value"]
        anchor["survey_year"] = CENSUS_2000_ANCHOR

        interpolated_records = []

        for target_year in target_years:
            console.print(f"  Processing {target_year}...")

            year_data = anchor.copy()

            # Get HPI ratio with state fallback for each county
            hpi_ratios = []
            hpi_sources = []

            for _, row in year_data.iterrows():
                fips = row["county_fips"]
                ratio, source = get_hpi_ratio_with_fallback(
                    self.hpi_df, fips, CENSUS_2000_ANCHOR, target_year
                )
                hpi_ratios.append(ratio)
                hpi_sources.append(source)

            year_data["hpi_ratio"] = hpi_ratios
            year_data["hpi_source"] = hpi_sources

            # Calculate forward interpolation
            valid_mask = (
                year_data["hpi_ratio"].notna() &
                year_data[value_column].notna()
            )

            # Apply forward interpolation
            year_data[value_column] = np.where(
                valid_mask,
                year_data[value_column] * year_data["hpi_ratio"],
                np.nan
            )

            # Update metadata - track whether county or state HPI was used
            year_data["survey_year"] = target_year
            year_data["data_source"] = np.where(
                year_data["hpi_source"] == "county",
                "hpi_forward",
                np.where(
                    year_data["hpi_source"] == "state",
                    "hpi_forward_state",
                    "hpi_forward"
                )
            )
            year_data["anchor_year"] = CENSUS_2000_ANCHOR
            year_data["hpi_ratio_applied"] = year_data["hpi_ratio"]

            # Clean up
            drop_cols = ["hpi_ratio", "hpi_source", "median_property_value"]
            year_data = year_data.drop(columns=[c for c in drop_cols if c in year_data.columns])

            n_valid = valid_mask.sum()
            n_total = len(year_data)
            n_county = sum(1 for s in hpi_sources if s == "county")
            n_state = sum(1 for s in hpi_sources if s == "state")
            console.print(f"    Interpolated {n_valid}/{n_total} counties ({100*n_valid/n_total:.1f}%)")
            console.print(f"      County HPI: {n_county}, State HPI fallback: {n_state}")

            interpolated_records.append(year_data)

        return pd.concat(interpolated_records, ignore_index=True)

    def _fill_gaps_2005_2011(
        self,
        panel_df: pd.DataFrame,
        value_column: str,
    ) -> pd.DataFrame:
        """Fill gaps in 2005-2011 by interpolating between Census 2000 and 2012.

        For counties missing ACS data (1-year or 5-year) in 2005-2011,
        interpolate using HPI-weighted average between Census 2000 and 2012 ACS.
        """
        console.print("\n[cyan]Filling 2005-2011 gaps with HPI interpolation...[/cyan]")

        # Get Census 2000 anchor values
        census_df = self.census_2000_df.copy()
        census_df["county_fips"] = census_df["county_fips"].astype(str).str.zfill(5)

        # Get 2012 values from panel
        panel_2012 = panel_df[panel_df["survey_year"] == FIRST_FULL_COVERAGE_YEAR].copy()
        if len(panel_2012) == 0:
            console.print("  [yellow]No 2012 data found for interpolation[/yellow]")
            return pd.DataFrame()

        if "county_fips" not in panel_2012.columns:
            panel_2012["county_fips"] = panel_2012["county_geoid"].str[:5]

        # Template for creating new records
        template = panel_2012.copy()

        gap_records = []

        for target_year in INTERPOLATE_GAP_YEARS:
            # Find counties with missing data for this year
            year_data = panel_df[panel_df["survey_year"] == target_year].copy()
            if "county_fips" not in year_data.columns and len(year_data) > 0:
                year_data["county_fips"] = year_data["county_geoid"].str[:5]

            # Counties with valid values
            if len(year_data) > 0:
                counties_with_data = set(
                    year_data[year_data[value_column].notna()]["county_fips"]
                )
            else:
                counties_with_data = set()

            # All counties in panel
            all_counties = set(template["county_fips"])

            # Counties missing this year
            missing_counties = all_counties - counties_with_data

            if not missing_counties:
                console.print(f"  {target_year}: No gaps to fill")
                continue

            # Create records for missing counties
            fill_records = template[template["county_fips"].isin(missing_counties)].copy()

            # Merge Census 2000 values
            fill_records = fill_records.merge(
                census_df[["county_fips", "median_property_value"]],
                on="county_fips",
                how="left",
            )
            fill_records = fill_records.rename(columns={"median_property_value": "census_2000_value"})

            # Merge 2012 values
            fill_records = fill_records.merge(
                panel_2012[["county_fips", value_column]].rename(columns={value_column: "value_2012"}),
                on="county_fips",
                how="left",
            )

            # Get HPI ratios with state fallback
            hpi_ratios = []
            hpi_sources = []

            for _, row in fill_records.iterrows():
                fips = row["county_fips"]
                # Get HPI values for 2000, target year, and 2012
                hpi_2000, _ = get_hpi_with_state_fallback(self.hpi_df, fips, CENSUS_2000_ANCHOR)
                hpi_target, _ = get_hpi_with_state_fallback(self.hpi_df, fips, target_year)
                hpi_2012, src = get_hpi_with_state_fallback(self.hpi_df, fips, FIRST_FULL_COVERAGE_YEAR)

                if hpi_2000 and hpi_target and hpi_2012 and hpi_2012 != hpi_2000:
                    # HPI-weighted interpolation
                    weight = (hpi_target - hpi_2000) / (hpi_2012 - hpi_2000)
                    hpi_ratios.append(weight)
                    hpi_sources.append(src)
                else:
                    # Linear fallback
                    weight = (target_year - CENSUS_2000_ANCHOR) / (FIRST_FULL_COVERAGE_YEAR - CENSUS_2000_ANCHOR)
                    hpi_ratios.append(weight)
                    hpi_sources.append("linear")

            fill_records["hpi_weight"] = hpi_ratios
            fill_records["hpi_source"] = hpi_sources

            # Calculate interpolated values
            valid_mask = (
                fill_records["census_2000_value"].notna() &
                fill_records["value_2012"].notna()
            )

            fill_records[value_column] = np.where(
                valid_mask,
                fill_records["census_2000_value"] + fill_records["hpi_weight"] * (
                    fill_records["value_2012"] - fill_records["census_2000_value"]
                ),
                np.nan
            )

            # Update metadata
            fill_records["survey_year"] = target_year
            fill_records["data_source"] = "hpi_interpolated"
            fill_records["anchor_years"] = f"{CENSUS_2000_ANCHOR}-{FIRST_FULL_COVERAGE_YEAR}"

            # Clean up
            drop_cols = ["census_2000_value", "value_2012", "hpi_weight", "hpi_source"]
            fill_records = fill_records.drop(columns=[c for c in drop_cols if c in fill_records.columns])

            n_filled = valid_mask.sum()
            console.print(f"  {target_year}: Filled {n_filled}/{len(missing_counties)} missing counties")

            if n_filled > 0:
                gap_records.append(fill_records[fill_records[value_column].notna()])

        if not gap_records:
            return pd.DataFrame()

        result = pd.concat(gap_records, ignore_index=True)

        # Clean up county_fips
        if "county_fips" in result.columns:
            result = result.drop(columns=["county_fips"])

        return result

    def _report_summary(self, df: pd.DataFrame) -> None:
        """Print summary of data sources after backcasting."""
        console.print("\n[bold]Data Source Summary:[/bold]")

        if "data_source" in df.columns:
            source_counts = df["data_source"].value_counts()
            for source, count in source_counts.items():
                pct = 100 * count / len(df)
                console.print(f"  {source}: {count:,} records ({pct:.1f}%)")

        # Coverage by year
        console.print("\n[bold]Records by Year:[/bold]")
        year_counts = df.groupby("survey_year").size()
        for year, count in year_counts.items():
            source = df[df["survey_year"] == year]["data_source"].mode()
            source_str = source.iloc[0] if len(source) > 0 else "unknown"
            console.print(f"  {year}: {count:,} ({source_str})")

    def get_hpi_coverage(self, panel_counties: list[str]) -> dict:
        """Get HPI coverage statistics for panel counties."""
        return get_hpi_coverage_stats(self.hpi_df, panel_counties)


def apply_hybrid_backcast(
    input_path: Path | str,
    output_path: Path | str | None = None,
    stable_anchor_year: int = DEFAULT_STABLE_ANCHOR,
    smoothing_weight: float = DEFAULT_SMOOTHING_WEIGHT,
    use_census_2000: bool = True,
) -> pd.DataFrame:
    """
    Apply hybrid backcasting to a panel file.

    Args:
        input_path: Path to input panel parquet file
        output_path: Path to save enhanced panel. If None, uses default.
        stable_anchor_year: Stable ACS reference year for smoothing
        smoothing_weight: Weight for HPI smoothing
        use_census_2000: If True, use Census 2000 SF3 for year 2000

    Returns:
        Enhanced panel DataFrame
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Panel file not found: {input_path}")

    console.print(f"[bold]Loading panel from {input_path}...[/bold]")
    panel = pd.read_parquet(input_path)

    console.print(f"  Loaded {len(panel):,} records")
    console.print(f"  Years: {sorted(panel['survey_year'].unique())}")

    # Apply hybrid methodology
    backcaster = HybridBackcaster(
        stable_anchor_year=stable_anchor_year,
        smoothing_weight=smoothing_weight,
        use_census_2000=use_census_2000,
    )

    enhanced_panel = backcaster.apply_hybrid_methodology(panel)

    # Save if output path specified
    if output_path is None:
        output_path = OUTPUT_DIR / "county_urban_net_returns_panel_enhanced.parquet"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced_panel.to_parquet(output_path, index=False)
    console.print(f"\n[green]Saved enhanced panel to {output_path}[/green]")

    return enhanced_panel


def generate_methodology_report(
    panel_df: pd.DataFrame,
    output_path: Path | str | None = None,
) -> str:
    """
    Generate a methodology report documenting data sources and quality.

    Args:
        panel_df: Enhanced panel with data_source column
        output_path: Path to save report. If None, returns string only.

    Returns:
        Report text
    """
    report_lines = [
        "# Hybrid Backcasting Methodology Report",
        "",
        "## Overview",
        "",
        "This panel dataset uses a hybrid methodology to extend coverage back to 2000:",
        "",
        "| Period | Data Source | Method |",
        "|--------|-------------|--------|",
        "| 2009-2023 | 5-year ACS PUMS | Direct observation |",
        "| 2005-2008 | 1-year ACS + HPI | Smoothed blend |",
        "| 2001-2004 | Census 2000 + HPI | Forward interpolation |",
        "| 2000 | Census 2000 SF3 | Direct observation |",
        "",
        "## Data Source Distribution",
        "",
    ]

    if "data_source" in panel_df.columns:
        source_counts = panel_df["data_source"].value_counts()
        for source, count in source_counts.items():
            pct = 100 * count / len(panel_df)
            report_lines.append(f"- **{source}**: {count:,} records ({pct:.1f}%)")

    report_lines.extend([
        "",
        "## Coverage by Year",
        "",
        "| Year | Records | Primary Source | Coverage Rate |",
        "|------|---------|----------------|---------------|",
    ])

    for year in sorted(panel_df["survey_year"].unique()):
        year_data = panel_df[panel_df["survey_year"] == year]
        n_records = len(year_data)
        n_with_value = year_data["mean_property_value"].notna().sum()
        coverage = 100 * n_with_value / n_records if n_records > 0 else 0

        if "data_source" in year_data.columns:
            source = year_data["data_source"].mode()
            source_str = source.iloc[0] if len(source) > 0 else "unknown"
        else:
            source_str = "unknown"

        report_lines.append(f"| {year} | {n_records:,} | {source_str} | {coverage:.1f}% |")

    report_lines.extend([
        "",
        "## Methodology Details",
        "",
        "### Census 2000 SF3 (Year 2000)",
        "",
        "Year 2000 uses direct observation from Census 2000 Summary File 3:",
        "",
        "- Variable H085001: Median value of owner-occupied housing units",
        "- Coverage: 3,219 counties (near-complete)",
        "- Data source: `census_2000_sf3`",
        "",
        "### Forward Interpolation (2001-2004)",
        "",
        "Values are interpolated forward from Census 2000 using FHFA HPI:",
        "",
        "```",
        "value_target = value_2000 × (HPI_target / HPI_2000)",
        "```",
        "",
        "This captures the housing bubble appreciation from 2000-2004.",
        "",
        "### Smoothed Estimates (2005-2008)",
        "",
        "Volatile 1-year ACS estimates are blended with HPI-implied values:",
        "",
        "```",
        "value_smoothed = (1 - w) × value_original + w × value_hpi_implied",
        "```",
        "",
        "Where w is the smoothing weight (default 0.5).",
        "",
        "### Data Quality Flags",
        "",
        "- **acs_5year**: Highest quality, large pooled sample (2009+)",
        "- **census_2000_sf3**: Direct Census observation (2000)",
        "- **hpi_forward**: Forward interpolation using county-level HPI (2001-2004)",
        "- **hpi_forward_state**: Forward interpolation using state-level HPI fallback (2001-2004)",
        "- **hpi_interpolated**: HPI-weighted interpolation between Census 2000 and 2012 (gap fill)",
        "- **hpi_smoothed**: Blended 1-year ACS + HPI (2005-2008)",
        "- **acs_1year**: Original 1-year estimate (higher variance)",
        "",
        "## Limitations",
        "",
        "1. HPI coverage: ~77.6% of panel counties have county-level FHFA HPI data",
        "2. State HPI fallback used for 715 additional counties (assumes local trends follow state average)",
        "3. Census 2000 uses median (not mean) property values",
        "4. Forward interpolation assumes local trends follow HPI",
        "5. 1-year ACS samples (2005-2008) have higher variance",
        "",
        "## Recommended Usage",
        "",
        "- For time series requiring maximum span: Use full enhanced panel",
        "- For highest accuracy: Filter to data_source = 'acs_5year' (2012+)",
        "- For year 2000 anchor: Census 2000 SF3 is direct observation",
        "- For regional analysis: Include all sources with provenance awareness",
        "",
    ])

    report_text = "\n".join(report_lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text)
        console.print(f"[green]Saved methodology report to {output_path}[/green]")

    return report_text
