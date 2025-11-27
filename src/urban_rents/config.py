"""Configuration settings and constants for urban net returns calculation."""

from pathlib import Path
from typing import Final

from pydantic import BaseModel

# Project paths
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
OUTPUT_DIR: Final[Path] = DATA_DIR / "output"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "figures"

# PUMS configuration
PUMS_BASE_URL: Final[str] = "https://www2.census.gov/programs-surveys/acs/data/pums"
DEFAULT_SURVEY_YEAR: Final[int] = 2023  # Default to most recent

# Survey type by year
# 1-year PUMS: 2000-2008 (smaller sample, single year snapshot)
# 5-year PUMS: 2009+ (larger sample, pooled over 5 years)
ONE_YEAR_PUMS_YEARS: Final[list[int]] = list(range(2000, 2009))  # 2000-2008
FIVE_YEAR_PUMS_YEARS: Final[list[int]] = list(range(2009, 2024))  # 2009-2023

# All available survey years (combined 1-year and 5-year)
AVAILABLE_SURVEY_YEARS: Final[list[int]] = ONE_YEAR_PUMS_YEARS + FIVE_YEAR_PUMS_YEARS

# PUMA vintage configuration
# 2000 PUMAs: Used for 1-year ACS 2000-2008 and 5-year ACS 2009-2012
# 2010 PUMAs: Used for ACS 2013-2021 (2009-2013 through 2017-2021 5-year)
# 2020 PUMAs: Used for ACS from 2022 onward (2018-2022 5-year and later)
PUMA_VINTAGE_MAPPING: Final[dict[int, str]] = {
    # 1-year PUMS (2000-2008) all use 2000 PUMAs
    2000: "2000", 2001: "2000", 2002: "2000", 2003: "2000", 2004: "2000",
    2005: "2000", 2006: "2000", 2007: "2000", 2008: "2000",
    # 5-year PUMS (2009-2012) use 2000 PUMAs
    2009: "2000", 2010: "2000", 2011: "2000", 2012: "2000",
    # 5-year PUMS (2013-2021) use 2010 PUMAs
    2013: "2010", 2014: "2010", 2015: "2010", 2016: "2010", 2017: "2010",
    2018: "2010", 2019: "2010", 2020: "2010", 2021: "2010",
    # 5-year PUMS (2022+) use 2020 PUMAs
    2022: "2020", 2023: "2020", 2024: "2020", 2025: "2020",
}

# TIGER shapefile years for each PUMA vintage
TIGER_YEAR_FOR_VINTAGE: Final[dict[str, int]] = {
    "2000": 2012,  # TIGER year with 2000 PUMAs (tl_2012_XX_puma00)
    "2010": 2019,  # Last TIGER year with 2010 PUMAs
    "2020": 2022,  # First TIGER year with 2020 PUMAs
}

# Shapefile configuration
TIGER_BASE_URL: Final[str] = "https://www2.census.gov/geo/tiger"
TIGER_YEAR: Final[int] = 2022  # Default for backward compatibility

# Survey of Construction configuration
SOC_BASE_URL: Final[str] = "https://www.census.gov/construction/chars"
SOC_MICRODATA_URL: Final[str] = "https://www.census.gov/construction/chars/xls"

# Coordinate reference system for area calculations
CRS_ALBERS: Final[str] = "EPSG:5070"  # Albers Equal Area Conic

# Economic parameters
DISCOUNT_RATE: Final[float] = 0.05  # 5% annualization rate

# PUMS variable codes
class PUMSVariables:
    """PUMS variable codes and values based on 2019-2023 ACS data dictionary."""

    # Tenure (TEN)
    TEN_OWNED_MORTGAGE: Final[str] = "1"  # Owned with mortgage/loan
    TEN_OWNED_FREE: Final[str] = "2"  # Owned free and clear
    TEN_RENTED: Final[str] = "3"  # Rented
    TEN_NO_RENT: Final[str] = "4"  # Occupied without payment

    # Building type (BLD)
    BLD_MOBILE: Final[str] = "01"  # Mobile home
    BLD_SF_DETACHED: Final[str] = "02"  # Single-family detached
    BLD_SF_ATTACHED: Final[str] = "03"  # Single-family attached

    # Year built (YRBLT) - categories for recent construction
    # IMPORTANT: YRBLT uses DECADE categories for 2010-2019 (code "2010")
    # Individual years only for 2020, 2021, 2022, 2023
    # To capture "recently built" we include:
    #   - "2010" = built 2010-2019
    #   - "2020", "2021", "2022", "2023" = individual years
    YRBLT_RECENT: Final[list[str]] = ["2010", "2020", "2021", "2022", "2023"]


# Census Division codes and names
CENSUS_DIVISIONS: Final[dict[int, str]] = {
    1: "New England",
    2: "Middle Atlantic",
    3: "East North Central",
    4: "West North Central",
    5: "South Atlantic",
    6: "East South Central",
    7: "West South Central",
    8: "Mountain",
    9: "Pacific",
}

# State FIPS to Census Division mapping
STATE_TO_DIVISION: Final[dict[str, int]] = {
    # New England (Division 1)
    "09": 1, "23": 1, "25": 1, "33": 1, "44": 1, "50": 1,  # CT, ME, MA, NH, RI, VT
    # Middle Atlantic (Division 2)
    "34": 2, "36": 2, "42": 2,  # NJ, NY, PA
    # East North Central (Division 3)
    "17": 3, "18": 3, "26": 3, "39": 3, "55": 3,  # IL, IN, MI, OH, WI
    # West North Central (Division 4)
    "19": 4, "20": 4, "27": 4, "29": 4, "31": 4, "38": 4, "46": 4,  # IA, KS, MN, MO, NE, ND, SD
    # South Atlantic (Division 5)
    "10": 5, "11": 5, "12": 5, "13": 5, "24": 5, "37": 5, "45": 5, "51": 5, "54": 5,  # DE, DC, FL, GA, MD, NC, SC, VA, WV
    # East South Central (Division 6)
    "01": 6, "21": 6, "28": 6, "47": 6,  # AL, KY, MS, TN
    # West South Central (Division 7)
    "05": 7, "22": 7, "40": 7, "48": 7,  # AR, LA, OK, TX
    # Mountain (Division 8)
    "04": 8, "08": 8, "16": 8, "30": 8, "32": 8, "35": 8, "49": 8, "56": 8,  # AZ, CO, ID, MT, NV, NM, UT, WY
    # Pacific (Division 9)
    "02": 9, "06": 9, "15": 9, "41": 9, "53": 9,  # AK, CA, HI, OR, WA
}

# Contiguous US state FIPS codes (excludes AK=02, HI=15, territories)
CONTIGUOUS_US_FIPS: Final[set[str]] = {
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56",
}

# State FIPS to abbreviation mapping
STATE_FIPS_TO_ABBR: Final[dict[str, str]] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}


class SOCDivisionData(BaseModel):
    """Survey of Construction data by Census Division."""

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
            self.lot_share = self.median_lot_value / self.median_sales_price
        if self.lot_acres is None:
            self.lot_acres = self.median_lot_size_sqft / 43560


# SOC data by year and Census Division
# Sources: NAHB analysis of Census SOC, https://eyeonhousing.org/
# Note: Historical data estimated from NAHB reports and Census SOC tables
SOC_DATA_BY_YEAR: Final[dict[int, list[dict]]] = {
    # 2013 data (approximate, from historical SOC reports)
    2013: [
        {"division_code": 1, "division_name": "New England",
         "median_lot_value": 85000, "median_lot_size_sqft": 28000, "median_sales_price": 380000},
        {"division_code": 2, "division_name": "Middle Atlantic",
         "median_lot_value": 50000, "median_lot_size_sqft": 18000, "median_sales_price": 320000},
        {"division_code": 3, "division_name": "East North Central",
         "median_lot_value": 35000, "median_lot_size_sqft": 14000, "median_sales_price": 260000},
        {"division_code": 4, "division_name": "West North Central",
         "median_lot_value": 32000, "median_lot_size_sqft": 13000, "median_sales_price": 240000},
        {"division_code": 5, "division_name": "South Atlantic",
         "median_lot_value": 35000, "median_lot_size_sqft": 12000, "median_sales_price": 270000},
        {"division_code": 6, "division_name": "East South Central",
         "median_lot_value": 28000, "median_lot_size_sqft": 16000, "median_sales_price": 210000},
        {"division_code": 7, "division_name": "West South Central",
         "median_lot_value": 38000, "median_lot_size_sqft": 8000, "median_sales_price": 230000},
        {"division_code": 8, "division_name": "Mountain",
         "median_lot_value": 45000, "median_lot_size_sqft": 8500, "median_sales_price": 290000},
        {"division_code": 9, "division_name": "Pacific",
         "median_lot_value": 90000, "median_lot_size_sqft": 7000, "median_sales_price": 420000},
    ],
    # 2018 data (from historical SOC reports)
    2018: [
        {"division_code": 1, "division_name": "New England",
         "median_lot_value": 130000, "median_lot_size_sqft": 26000, "median_sales_price": 460000},
        {"division_code": 2, "division_name": "Middle Atlantic",
         "median_lot_value": 65000, "median_lot_size_sqft": 16000, "median_sales_price": 380000},
        {"division_code": 3, "division_name": "East North Central",
         "median_lot_value": 45000, "median_lot_size_sqft": 13000, "median_sales_price": 320000},
        {"division_code": 4, "division_name": "West North Central",
         "median_lot_value": 40000, "median_lot_size_sqft": 12000, "median_sales_price": 300000},
        {"division_code": 5, "division_name": "South Atlantic",
         "median_lot_value": 42000, "median_lot_size_sqft": 11000, "median_sales_price": 330000},
        {"division_code": 6, "division_name": "East South Central",
         "median_lot_value": 35000, "median_lot_size_sqft": 15000, "median_sales_price": 260000},
        {"division_code": 7, "division_name": "West South Central",
         "median_lot_value": 48000, "median_lot_size_sqft": 7500, "median_sales_price": 290000},
        {"division_code": 8, "division_name": "Mountain",
         "median_lot_value": 65000, "median_lot_size_sqft": 7500, "median_sales_price": 380000},
        {"division_code": 9, "division_name": "Pacific",
         "median_lot_value": 120000, "median_lot_size_sqft": 6000, "median_sales_price": 550000},
    ],
    # 2023 data (from NAHB analysis)
    # Sources: https://eyeonhousing.org/2024/07/lot-values-trend-higher/
    2023: [
        {"division_code": 1, "division_name": "New England",
         "median_lot_value": 200000, "median_lot_size_sqft": 24394, "median_sales_price": 550000},
        {"division_code": 2, "division_name": "Middle Atlantic",
         "median_lot_value": 80000, "median_lot_size_sqft": 15000, "median_sales_price": 450000},
        {"division_code": 3, "division_name": "East North Central",
         "median_lot_value": 55000, "median_lot_size_sqft": 12000, "median_sales_price": 380000},
        {"division_code": 4, "division_name": "West North Central",
         "median_lot_value": 50000, "median_lot_size_sqft": 11000, "median_sales_price": 350000},
        {"division_code": 5, "division_name": "South Atlantic",
         "median_lot_value": 49000, "median_lot_size_sqft": 10000, "median_sales_price": 400000},
        {"division_code": 6, "division_name": "East South Central",
         "median_lot_value": 46000, "median_lot_size_sqft": 14375, "median_sales_price": 320000},
        {"division_code": 7, "division_name": "West South Central",
         "median_lot_value": 61000, "median_lot_size_sqft": 6534, "median_sales_price": 350000},
        {"division_code": 8, "division_name": "Mountain",
         "median_lot_value": 90000, "median_lot_size_sqft": 6970, "median_sales_price": 480000},
        {"division_code": 9, "division_name": "Pacific",
         "median_lot_value": 147000, "median_lot_size_sqft": 5500, "median_sales_price": 650000},
    ],
}

# Legacy alias for backward compatibility
SOC_2023_DATA: Final[list[dict]] = SOC_DATA_BY_YEAR[2023]


def get_soc_division_data(year: int | None = None) -> dict[int, SOCDivisionData]:
    """
    Load and return SOC data by Census Division for a given year.

    Args:
        year: SOC year to load. If None, uses 2023 (most recent).
              If year not available, uses nearest available year.

    Returns:
        Dictionary mapping division_code to SOCDivisionData
    """
    if year is None:
        year = 2023

    # Find nearest available year
    available_years = sorted(SOC_DATA_BY_YEAR.keys())
    if year in available_years:
        soc_year = year
    else:
        # Find nearest
        soc_year = min(available_years, key=lambda y: abs(y - year))

    return {
        d["division_code"]: SOCDivisionData(**d)
        for d in SOC_DATA_BY_YEAR[soc_year]
    }


def get_puma_vintage(survey_year: int) -> str:
    """Get the PUMA vintage (boundary definition) for a survey year."""
    if survey_year not in PUMA_VINTAGE_MAPPING:
        # Default logic for future years
        return "2020" if survey_year >= 2022 else "2010"
    return PUMA_VINTAGE_MAPPING[survey_year]


def get_tiger_year(survey_year: int) -> int:
    """Get the appropriate TIGER shapefile year for a survey year."""
    vintage = get_puma_vintage(survey_year)
    return TIGER_YEAR_FOR_VINTAGE[vintage]


def is_one_year_survey(survey_year: int) -> bool:
    """Check if a survey year uses 1-year PUMS data."""
    return survey_year in ONE_YEAR_PUMS_YEARS


def get_period_label(survey_year: int) -> str:
    """Get human-readable period label for an ACS survey year.

    For 1-year PUMS (2000-2008): Returns single year like "2005"
    For 5-year PUMS (2009+): Returns range like "2019-2023"
    """
    if is_one_year_survey(survey_year):
        return str(survey_year)
    return f"{survey_year - 4}-{survey_year}"


def get_survey_type_label(survey_year: int) -> str:
    """Get the survey type label (1-year or 5-year)."""
    return "1-year" if is_one_year_survey(survey_year) else "5-year"


def validate_survey_year(survey_year: int) -> None:
    """Validate that a survey year is available."""
    if survey_year not in AVAILABLE_SURVEY_YEARS:
        available = ", ".join(str(y) for y in AVAILABLE_SURVEY_YEARS)
        raise ValueError(
            f"Survey year {survey_year} not available. "
            f"Available years: {available}"
        )
