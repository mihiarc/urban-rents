"""Pydantic models for panel dataset configuration and data structures."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PUMAVintageConfig(BaseModel):
    """Configuration for a specific PUMA vintage (Census boundary definition)."""

    vintage: Literal["2000", "2010", "2020"]
    valid_survey_years: list[int]
    tiger_year: int  # Shapefile year to download
    shapefile_suffix: str  # e.g., "puma00", "puma10", "puma20"

    def contains_survey_year(self, survey_year: int) -> bool:
        """Check if this vintage applies to a given survey year."""
        return survey_year in self.valid_survey_years


class SurveyYearConfig(BaseModel):
    """Configuration for a specific ACS survey year."""

    survey_year: int  # End year of 5-year ACS (e.g., 2023 for 2019-2023)
    period_start: int  # First year of 5-year window
    period_end: int  # Last year of 5-year window (same as survey_year)
    puma_vintage: Literal["2000", "2010", "2020"]
    midpoint_year: float  # For time series plotting

    @property
    def period_label(self) -> str:
        """Human-readable period label like '2019-2023'."""
        return f"{self.period_start}-{self.period_end}"

    @field_validator("period_end")
    @classmethod
    def period_end_matches_survey_year(cls, v, info):
        if "survey_year" in info.data and v != info.data["survey_year"]:
            raise ValueError("period_end must equal survey_year")
        return v


class SOCDivisionParams(BaseModel):
    """Survey of Construction parameters for a Census Division."""

    division_code: int
    division_name: str
    median_lot_value: float  # USD
    median_lot_size_sqft: float  # Square feet
    median_sales_price: float  # USD
    lot_share: float | None = None  # Computed: lot_value / sales_price
    lot_acres: float | None = None  # Computed: sqft / 43560

    def model_post_init(self, __context) -> None:
        """Compute derived fields after initialization."""
        if self.lot_share is None and self.median_sales_price > 0:
            object.__setattr__(
                self, "lot_share", self.median_lot_value / self.median_sales_price
            )
        if self.lot_acres is None:
            object.__setattr__(self, "lot_acres", self.median_lot_size_sqft / 43560)


class SOCYearConfig(BaseModel):
    """Survey of Construction data for a specific year."""

    year: int
    source: str = "NAHB/Census SOC"
    division_data: dict[int, SOCDivisionParams]

    def get_division(self, division_code: int) -> SOCDivisionParams:
        """Get SOC parameters for a division."""
        if division_code not in self.division_data:
            raise KeyError(f"No SOC data for division {division_code}")
        return self.division_data[division_code]


class PanelConfig(BaseModel):
    """Configuration for building a panel dataset."""

    survey_years: list[int] = Field(
        ...,
        description="List of ACS survey end years to include"
    )
    soc_methodology: Literal["year_specific", "fixed_2023", "nearest"] = Field(
        default="nearest",
        description="How to select SOC parameters for each year"
    )
    output_format: Literal["long", "wide", "both"] = Field(
        default="long",
        description="Output format for panel data"
    )
    include_growth_rates: bool = Field(
        default=True,
        description="Whether to compute period-over-period growth rates"
    )

    @field_validator("survey_years")
    @classmethod
    def validate_survey_years(cls, v):
        if len(v) < 2:
            raise ValueError("Panel requires at least 2 survey years")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate survey years not allowed")
        return sorted(v)

    @property
    def year_range_label(self) -> str:
        """Label for the panel year range."""
        return f"{min(self.survey_years)}-{max(self.survey_years)}"


class CountyPanelRecord(BaseModel):
    """A single county-year observation in the panel."""

    # Identifiers
    county_geoid: str
    survey_year: int
    period_label: str
    puma_vintage: str

    # Geographic metadata
    state_fips: str
    county_fips: str
    county_name: str
    state_abbr: str
    census_division: int
    division_name: str

    # Core measures
    mean_property_value: float | None
    lot_share: float | None
    lot_acres: float | None
    urban_net_return: float | None
    urban_net_return_adjusted: float | None

    # Data quality
    n_pumas: int
    n_pumas_with_data: int
    max_puma_weight: float
    total_observations: int
    data_quality_flag: str

    # Methodology tracking
    soc_year: int
    crosswalk_vintage: str


# Recommended survey year configurations
# Now includes 1-year PUMS (2000-2008) and 5-year PUMS (2009-2023)
RECOMMENDED_PANEL_YEARS = {
    # Full annual time series from 2000 to 2023 (24 years)
    # 2000-2008: 1-year PUMS with 2000 PUMAs (smaller sample)
    # 2009-2012: 5-year PUMS with 2000 PUMAs
    # 2013-2021: 5-year PUMS with 2010 PUMAs
    # 2022-2023: 5-year PUMS with 2020 PUMAs
    "full_annual": list(range(2000, 2024)),  # All 24 years

    # Maximum span with ~5-year spacing, includes 1-year data
    "maximum": [2000, 2005, 2009, 2013, 2018, 2023],

    # Standard 5-year intervals from 2000 onward
    "standard_extended": [2000, 2005, 2010, 2015, 2020, 2023],

    # 5-year only (original methodology, excludes 1-year data)
    "five_year_only": list(range(2009, 2024)),  # All 5-year surveys

    # Quick panel using milestone years
    "milestones": [2000, 2008, 2013, 2018, 2023],  # Pre-crisis, crisis, recovery, recent

    # Legacy presets for backward compatibility
    "standard": [2013, 2018, 2023],  # ~5-year spacing within 2010 PUMA era
    "extended": [2009, 2012, 2015, 2018, 2021, 2023],  # ~3-year spacing (5-year only)
}
