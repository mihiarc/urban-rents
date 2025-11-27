# Urban Net Returns

County-level annualized net returns to urban land for the contiguous United States (2000-2023).

## Overview

This project constructs a **24-year panel dataset** of urban net returns at the county level using:
- **Census PUMS** (Public Use Microdata Sample) housing data
- **Survey of Construction** (SOC) parameters for lot characteristics
- **TIGER/Line shapefiles** for PUMA-to-county geographic crosswalks

### Methodology

Urban net returns are calculated using the formula:

```
NR_urban = (PropertyValue × LotShare) / LotAcres × CapRate
```

Where:
- **PropertyValue** = mean value of recently constructed homes (from PUMS)
- **LotShare** = lot value as fraction of total property value (by Census Division, from SOC)
- **LotAcres** = median lot size in acres (by Census Division, from SOC)
- **CapRate** = capitalization rate (default 5%)

## Installation

```bash
git clone https://github.com/mihiarc/urban-rents.git
cd urban-rents
uv venv
uv pip install -e ".[dev]"
```

## Usage

### Download data for a state

```bash
# Download PUMS data for North Carolina, year 2023
uv run python -m urban_rents.cli download pums --state 37 --year 2023

# Download all years for a state (2000-2023)
for year in $(seq 2000 2023); do
  uv run python -m urban_rents.cli download pums --state 37 --year $year
done
```

### Process data programmatically

```python
from urban_rents.pums_processing import load_pums_housing_file, get_recent_built_codes
from urban_rents.config import get_period_label, is_one_year_survey

# Load housing data for NC 2023
df = load_pums_housing_file('37', 2023)

# Filter to recently built homes
recent_codes = get_recent_built_codes(2023)
recent = df[df['YRBLT'].astype(str).isin([str(c) for c in recent_codes])]

# Check survey type
print(f"Survey type: {get_period_label(2023)}")  # "2019-2023"
print(f"Is 1-year: {is_one_year_survey(2023)}")  # False
```

## Data Sources

| Source | Years | Type | Notes |
|--------|-------|------|-------|
| ACS 1-Year PUMS | 2000-2008 | Housing microdata | ~1% population sample |
| ACS 5-Year PUMS | 2009-2023 | Housing microdata | ~5% pooled sample |
| Survey of Construction | Annual | Lot characteristics | By Census Division |
| TIGER/Line | 2000, 2010, 2020 | Shapefiles | PUMA boundaries |

### Data availability notes
- **2007-2008**: 1-year data unavailable for small states (population threshold)
- **2000-2004**: Property values encoded as categorical `VAL` codes (converted to midpoints)
- **2005+**: Property values as continuous `VALP` in dollars

## Output

The primary output is a CSV file with columns:
- `state_fips`, `county_fips`, `county_geoid`, `county_name`, `state_abbr`
- `census_division`, `division_name`
- `mean_property_value` - PUMS-derived property values
- `lot_share`, `lot_acres` - SOC division-level parameters
- `urban_net_return` - annualized $/acre
- `urban_net_return_adjusted` - with PUMA-size correction
- `n_pumas`, `max_puma_weight`, `data_quality_flag`

## Project Structure

```
urban-rents/
├── data/
│   ├── raw/
│   │   ├── pums/
│   │   ├── soc/
│   │   └── shapefiles/
│   ├── processed/
│   └── output/
├── src/urban_rents/
│   ├── cli.py          # Command-line interface
│   ├── config.py       # Configuration and constants
│   ├── crosswalk.py    # PUMA-to-county crosswalk
│   ├── download.py     # Data download utilities
│   ├── net_returns.py  # Urban net returns calculation
│   ├── pums_processing.py  # PUMS data processing
│   └── visualization.py    # Map generation
├── figures/
└── pyproject.toml
```
