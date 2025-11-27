"""Data download utilities for PUMS and shapefile data."""

import zipfile
from io import BytesIO
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from urban_rents.config import (
    AVAILABLE_SURVEY_YEARS,
    CONTIGUOUS_US_FIPS,
    DEFAULT_SURVEY_YEAR,
    PUMS_BASE_URL,
    RAW_DIR,
    STATE_FIPS_TO_ABBR,
    TIGER_BASE_URL,
    TIGER_YEAR,
    TIGER_YEAR_FOR_VINTAGE,
    get_period_label,
    get_puma_vintage,
    get_survey_type_label,
    get_tiger_year,
    is_one_year_survey,
    validate_survey_year,
)

console = Console()


def get_pums_url(state_fips: str, survey_year: int, file_type: str = "h") -> str:
    """
    Construct PUMS file download URL for a specific survey year.

    Args:
        state_fips: 2-digit state FIPS code
        survey_year: Survey year (2000-2008 for 1-year, 2009+ for 5-year)
        file_type: 'h' for housing, 'p' for person

    Returns:
        Full URL to the PUMS CSV file

    Example URLs:
        1-year (2000-2008):
            https://www2.census.gov/programs-surveys/acs/data/pums/2005/csv_hri.zip
        5-year (2009+):
            https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/csv_hri.zip
    """
    state_abbr = STATE_FIPS_TO_ABBR.get(state_fips, "").lower()
    if not state_abbr:
        raise ValueError(f"Unknown state FIPS: {state_fips}")

    if is_one_year_survey(survey_year):
        # 1-year PUMS: files are directly in year folder
        return f"{PUMS_BASE_URL}/{survey_year}/csv_{file_type}{state_abbr}.zip"
    else:
        # 5-year PUMS: files are in 5-Year subfolder
        return f"{PUMS_BASE_URL}/{survey_year}/5-Year/csv_{file_type}{state_abbr}.zip"


def get_puma_shapefile_url(state_fips: str, survey_year: int | None = None) -> str:
    """
    Construct PUMA shapefile download URL for the appropriate vintage.

    Args:
        state_fips: 2-digit state FIPS code
        survey_year: Survey year to determine PUMA vintage. If None, uses default.

    Returns:
        Full URL to the PUMA shapefile ZIP

    Note:
        - 2000 PUMAs (survey years <= 2012): tl_2012_{fips}_puma00.zip
        - 2010 PUMAs (survey years 2013-2021): tl_2019_{fips}_puma10.zip
        - 2020 PUMAs (survey years >= 2022): tl_2022_{fips}_puma20.zip
    """
    if survey_year is None:
        survey_year = DEFAULT_SURVEY_YEAR

    vintage = get_puma_vintage(survey_year)
    tiger_year = get_tiger_year(survey_year)

    # Map vintage to PUMA suffix
    puma_suffix_map = {"2000": "puma00", "2010": "puma10", "2020": "puma20"}
    puma_suffix = puma_suffix_map.get(vintage, "puma20")

    # For 2000 PUMAs, the directory structure is slightly different
    if vintage == "2000":
        return f"{TIGER_BASE_URL}/TIGER{tiger_year}/PUMA5/tl_{tiger_year}_{state_fips}_{puma_suffix}.zip"

    return f"{TIGER_BASE_URL}/TIGER{tiger_year}/PUMA/tl_{tiger_year}_{state_fips}_{puma_suffix}.zip"


def get_county_shapefile_url(tiger_year: int | None = None) -> str:
    """
    Construct county shapefile download URL.

    Args:
        tiger_year: TIGER year. If None, uses default TIGER_YEAR.

    Returns:
        Full URL to the county shapefile ZIP
    """
    year = tiger_year if tiger_year is not None else TIGER_YEAR
    return f"{TIGER_BASE_URL}/TIGER{year}/COUNTY/tl_{year}_us_county.zip"


def download_file(
    url: str,
    dest_path: Path,
    extract: bool = True,
    timeout: float = 300.0,
) -> Path:
    """
    Download a file with progress indicator.

    Args:
        url: URL to download
        dest_path: Destination path (file or directory for extraction)
        extract: If True and file is a ZIP, extract contents
        timeout: Request timeout in seconds

    Returns:
        Path to downloaded/extracted file or directory
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        filename = url.split("/")[-1]
        task = progress.add_task(f"Downloading {filename}", total=None)

        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0))
                if total:
                    progress.update(task, total=total)

                content = BytesIO()
                for chunk in response.iter_bytes(chunk_size=8192):
                    content.write(chunk)
                    progress.advance(task, len(chunk))

    content.seek(0)

    if extract and url.endswith(".zip"):
        extract_dir = dest_path if dest_path.is_dir() else dest_path.parent
        with zipfile.ZipFile(content) as zf:
            zf.extractall(extract_dir)
        console.print(f"[green]Extracted to {extract_dir}[/green]")
        return extract_dir
    else:
        with open(dest_path, "wb") as f:
            f.write(content.read())
        console.print(f"[green]Downloaded to {dest_path}[/green]")
        return dest_path


def get_expected_csv_name(state_fips: str, survey_year: int) -> str:
    """
    Get the expected CSV filename for a PUMS housing file.

    Args:
        state_fips: 2-digit state FIPS code
        survey_year: Survey year

    Returns:
        Expected filename after extraction

    Note:
        1-year PUMS (2000-2008): Various naming conventions by year
            - 2000: c2ssh{aa}.csv (e.g., c2sshri.csv)
            - 2001-2004: ss{YY}h{aa}.csv (e.g., ss01hri.csv)
            - 2005-2008: ss{YY}h{aa}.csv (e.g., ss05hri.csv)
        5-year PUMS (2009+): psam_h{aa}.csv (e.g., psam_hri.csv)
    """
    abbr = STATE_FIPS_TO_ABBR.get(state_fips, "").lower()
    if is_one_year_survey(survey_year):
        # 1-year naming varies by year
        if survey_year == 2000:
            return f"c2ssh{abbr}.csv"
        else:
            # 2001-2008: ss{YY}h{aa}.csv
            year_suffix = str(survey_year)[-2:]  # Last 2 digits
            return f"ss{year_suffix}h{abbr}.csv"
    else:
        # 5-year: psam_h{aa}.csv
        return f"psam_h{abbr}.csv"


def download_pums_housing(
    state_fips: str | None = None,
    survey_year: int | None = None,
    force: bool = False,
) -> list[Path]:
    """
    Download PUMS housing files for a specific survey year.

    Args:
        state_fips: Specific state FIPS code, or None for all contiguous US
        survey_year: Survey year (2000-2008 for 1-year, 2009+ for 5-year).
                    If None, uses DEFAULT_SURVEY_YEAR.
        force: If True, re-download even if files exist

    Returns:
        List of paths to downloaded files
    """
    if survey_year is None:
        survey_year = DEFAULT_SURVEY_YEAR
    else:
        validate_survey_year(survey_year)

    # Organize PUMS by survey year
    pums_dir = RAW_DIR / "pums" / str(survey_year)
    pums_dir.mkdir(parents=True, exist_ok=True)

    states = [state_fips] if state_fips else sorted(CONTIGUOUS_US_FIPS)
    downloaded = []

    period_label = get_period_label(survey_year)
    survey_type = get_survey_type_label(survey_year)
    console.print(f"[bold]Downloading PUMS {period_label} ({survey_type}) housing files...[/bold]")

    for fips in states:
        abbr = STATE_FIPS_TO_ABBR.get(fips, "").lower()
        expected_csv_name = get_expected_csv_name(fips, survey_year)
        expected_csv = pums_dir / expected_csv_name

        if expected_csv.exists() and not force:
            console.print(f"[yellow]PUMS {survey_year} for {abbr.upper()} already exists, skipping[/yellow]")
            downloaded.append(expected_csv)
            continue

        url = get_pums_url(fips, survey_year, "h")
        try:
            download_file(url, pums_dir, extract=True)
            if expected_csv.exists():
                downloaded.append(expected_csv)
            else:
                # Try to find the extracted CSV (naming may vary)
                possible_csvs = list(pums_dir.glob(f"*h{abbr}.csv"))
                if possible_csvs:
                    downloaded.append(possible_csvs[0])
                    console.print(f"[green]Found {possible_csvs[0].name}[/green]")
        except httpx.HTTPStatusError as e:
            console.print(f"[red]Failed to download PUMS {survey_year} for {abbr.upper()}: {e}[/red]")

    return downloaded


def download_pums_for_panel(
    survey_years: list[int],
    state_fips: str | None = None,
    force: bool = False,
) -> dict[int, list[Path]]:
    """
    Download PUMS housing files for multiple survey years (panel dataset).

    Args:
        survey_years: List of survey years to download (1-year: 2000-2008, 5-year: 2009+)
        state_fips: Specific state FIPS code, or None for all contiguous US
        force: If True, re-download even if files exist

    Returns:
        Dictionary mapping survey_year to list of downloaded file paths
    """
    results = {}
    for year in survey_years:
        period_label = get_period_label(year)
        survey_type = get_survey_type_label(year)
        console.print(f"\n[bold cyan]Downloading PUMS for {period_label} ({survey_type})...[/bold cyan]")
        results[year] = download_pums_housing(
            state_fips=state_fips,
            survey_year=year,
            force=force,
        )
    return results


def download_puma_shapefiles(
    state_fips: str | None = None,
    survey_year: int | None = None,
    force: bool = False,
) -> list[Path]:
    """
    Download PUMA shapefiles for the appropriate vintage.

    Args:
        state_fips: Specific state FIPS code, or None for all contiguous US
        survey_year: Survey year to determine PUMA vintage. If None, uses default.
        force: If True, re-download even if files exist

    Returns:
        List of paths to shapefile directories
    """
    if survey_year is None:
        survey_year = DEFAULT_SURVEY_YEAR

    vintage = get_puma_vintage(survey_year)
    tiger_year = get_tiger_year(survey_year)

    # Map vintage to PUMA suffix
    puma_suffix_map = {"2000": "puma00", "2010": "puma10", "2020": "puma20"}
    puma_suffix = puma_suffix_map.get(vintage, "puma20")

    # Organize by vintage
    shp_dir = RAW_DIR / "shapefiles" / f"puma_{vintage}"
    shp_dir.mkdir(parents=True, exist_ok=True)

    states = [state_fips] if state_fips else sorted(CONTIGUOUS_US_FIPS)
    downloaded = []

    console.print(f"[bold]Downloading {vintage} PUMA shapefiles (TIGER {tiger_year})...[/bold]")

    for fips in states:
        expected_shp = shp_dir / f"tl_{tiger_year}_{fips}_{puma_suffix}.shp"

        if expected_shp.exists() and not force:
            console.print(f"[yellow]PUMA {vintage} shapefile for {fips} already exists, skipping[/yellow]")
            downloaded.append(expected_shp)
            continue

        url = get_puma_shapefile_url(fips, survey_year)
        try:
            download_file(url, shp_dir, extract=True)
            if expected_shp.exists():
                downloaded.append(expected_shp)
        except httpx.HTTPStatusError as e:
            console.print(f"[red]Failed to download PUMA {vintage} shapefile for {fips}: {e}[/red]")

    return downloaded


def download_puma_shapefiles_for_panel(
    survey_years: list[int],
    state_fips: str | None = None,
    force: bool = False,
) -> dict[str, list[Path]]:
    """
    Download PUMA shapefiles for all vintages needed by a panel.

    Args:
        survey_years: List of survey years in the panel
        state_fips: Specific state FIPS code, or None for all contiguous US
        force: If True, re-download even if files exist

    Returns:
        Dictionary mapping vintage ("2000", "2010", or "2020") to list of downloaded paths
    """
    # Determine which vintages we need
    vintages_needed = set(get_puma_vintage(year) for year in survey_years)

    results = {}
    for vintage in sorted(vintages_needed):
        # Pick a representative year for this vintage
        vintage_rep_years = {
            "2000": 2005,   # Use a 1-year survey year to get 2000 PUMAs
            "2010": 2021,   # Last 5-year using 2010 PUMAs
            "2020": 2023,   # Recent 5-year using 2020 PUMAs
        }
        rep_year = vintage_rep_years.get(vintage, 2023)
        console.print(f"\n[bold cyan]Downloading {vintage} PUMA shapefiles...[/bold cyan]")
        results[vintage] = download_puma_shapefiles(
            state_fips=state_fips,
            survey_year=rep_year,
            force=force,
        )
    return results


def download_county_shapefile(
    tiger_year: int | None = None,
    force: bool = False,
) -> Path | None:
    """
    Download national county shapefile.

    Args:
        tiger_year: TIGER year. If None, uses default TIGER_YEAR.
        force: If True, re-download even if file exists

    Returns:
        Path to shapefile or None if download failed
    """
    year = tiger_year if tiger_year is not None else TIGER_YEAR

    shp_dir = RAW_DIR / "shapefiles" / "county"
    shp_dir.mkdir(parents=True, exist_ok=True)

    expected_shp = shp_dir / f"tl_{year}_us_county.shp"

    if expected_shp.exists() and not force:
        console.print(f"[yellow]County shapefile (TIGER {year}) already exists, skipping[/yellow]")
        return expected_shp

    url = get_county_shapefile_url(year)
    try:
        download_file(url, shp_dir, extract=True)
        return expected_shp if expected_shp.exists() else None
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Failed to download county shapefile: {e}[/red]")
        return None


def download_all_data(
    survey_year: int | None = None,
    force: bool = False,
) -> dict[str, list[Path] | Path | None]:
    """
    Download all required data files for a single survey year.

    Args:
        survey_year: Survey year (2000-2008 for 1-year, 2009+ for 5-year).
                    If None, uses DEFAULT_SURVEY_YEAR.
        force: If True, re-download all files

    Returns:
        Dictionary with paths to downloaded files by type
    """
    if survey_year is None:
        survey_year = DEFAULT_SURVEY_YEAR

    period_label = get_period_label(survey_year)
    survey_type = get_survey_type_label(survey_year)
    console.print(f"[bold]Downloading all required data for {period_label} ({survey_type})...[/bold]")

    results = {}

    console.print("\n[bold cyan]1. Downloading PUMS housing files...[/bold cyan]")
    results["pums"] = download_pums_housing(survey_year=survey_year, force=force)

    console.print("\n[bold cyan]2. Downloading PUMA shapefiles...[/bold cyan]")
    results["puma_shapefiles"] = download_puma_shapefiles(survey_year=survey_year, force=force)

    console.print("\n[bold cyan]3. Downloading county shapefile...[/bold cyan]")
    results["county_shapefile"] = download_county_shapefile(force=force)

    console.print("\n[bold green]Download complete![/bold green]")
    return results


def download_panel_data(
    survey_years: list[int],
    state_fips: str | None = None,
    force: bool = False,
) -> dict[str, dict]:
    """
    Download all required data files for a panel dataset.

    Args:
        survey_years: List of survey end years to include in panel
        state_fips: Specific state FIPS code, or None for all contiguous US
        force: If True, re-download all files

    Returns:
        Dictionary with downloaded data organized by type and year/vintage
    """
    console.print(f"[bold]Downloading panel data for years: {survey_years}[/bold]")

    results = {}

    # Download PUMS for each year
    console.print("\n[bold cyan]1. Downloading PUMS housing files...[/bold cyan]")
    results["pums"] = download_pums_for_panel(
        survey_years=survey_years,
        state_fips=state_fips,
        force=force,
    )

    # Download PUMA shapefiles for needed vintages
    console.print("\n[bold cyan]2. Downloading PUMA shapefiles...[/bold cyan]")
    results["puma_shapefiles"] = download_puma_shapefiles_for_panel(
        survey_years=survey_years,
        state_fips=state_fips,
        force=force,
    )

    # Download county shapefile (use most recent TIGER year)
    console.print("\n[bold cyan]3. Downloading county shapefile...[/bold cyan]")
    results["county_shapefile"] = download_county_shapefile(force=force)

    console.print("\n[bold green]Panel data download complete![/bold green]")
    return results
