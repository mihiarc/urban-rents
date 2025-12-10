"""Command-line interface for urban net returns calculation."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from urban_rents.config import AVAILABLE_SURVEY_YEARS, DEFAULT_SURVEY_YEAR
from urban_rents.models import RECOMMENDED_PANEL_YEARS

console = Console()


def download_data(args: argparse.Namespace) -> int:
    """Download required data files."""
    from urban_rents.download import (
        download_all_data,
        download_county_shapefile,
        download_panel_data,
        download_puma_shapefiles,
        download_pums_housing,
    )

    survey_year = getattr(args, "year", None)

    if args.type == "all":
        download_all_data(survey_year=survey_year, force=args.force)
    elif args.type == "pums":
        download_pums_housing(state_fips=args.state, survey_year=survey_year, force=args.force)
    elif args.type == "puma":
        download_puma_shapefiles(state_fips=args.state, survey_year=survey_year, force=args.force)
    elif args.type == "county":
        download_county_shapefile(force=args.force)
    else:
        console.print(f"[red]Unknown data type: {args.type}[/red]")
        return 1

    return 0


def download_panel(args: argparse.Namespace) -> int:
    """Download data for panel dataset."""
    from urban_rents.download import download_panel_data

    # Parse survey years
    if args.years:
        survey_years = [int(y) for y in args.years.split(",")]
    elif args.preset:
        survey_years = RECOMMENDED_PANEL_YEARS.get(args.preset, RECOMMENDED_PANEL_YEARS["standard"])
    else:
        survey_years = RECOMMENDED_PANEL_YEARS["standard"]

    console.print(f"[bold]Downloading panel data for years: {survey_years}[/bold]")
    download_panel_data(survey_years=survey_years, state_fips=args.state, force=args.force)

    return 0


def process_pums(args: argparse.Namespace) -> int:
    """Process PUMS data to extract PUMA-level property values."""
    from urban_rents.pums_processing import (
        get_puma_summary_stats,
        process_all_states,
        save_puma_property_values,
    )

    survey_year = getattr(args, "year", None) or DEFAULT_SURVEY_YEAR

    console.print(f"[bold]Processing PUMS data for {survey_year}...[/bold]")

    puma_values = process_all_states(survey_year)
    save_puma_property_values(puma_values, survey_year)

    stats = get_puma_summary_stats(puma_values)
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total PUMAs: {stats['total_pumas']}")
    console.print(f"  PUMAs with data: {stats['pumas_with_data']}")
    console.print(f"  Mean property value: ${stats['mean_property_value']:,.0f}")
    console.print(f"  Total observations: {stats['total_observations']:,}")

    return 0


def process_pums_panel(args: argparse.Namespace) -> int:
    """Process PUMS data for multiple survey years."""
    from urban_rents.pums_processing import (
        get_puma_summary_stats,
        process_panel_years,
        save_panel_property_values,
    )

    # Parse survey years
    if args.years:
        survey_years = [int(y) for y in args.years.split(",")]
    elif args.preset:
        survey_years = RECOMMENDED_PANEL_YEARS.get(args.preset, RECOMMENDED_PANEL_YEARS["standard"])
    else:
        survey_years = RECOMMENDED_PANEL_YEARS["standard"]

    console.print(f"[bold]Processing PUMS data for panel: {survey_years}[/bold]")

    panel = process_panel_years(survey_years)
    save_panel_property_values(panel)

    stats = get_puma_summary_stats(panel)
    console.print(f"\n[bold]Panel Summary:[/bold]")
    console.print(f"  Survey years: {stats.get('survey_years', survey_years)}")
    console.print(f"  Total PUMA-year observations: {stats['total_pumas']}")
    console.print(f"  Observations with data: {stats['pumas_with_data']}")

    return 0


def build_crosswalk(args: argparse.Namespace) -> int:
    """Build PUMA-to-county crosswalk."""
    from urban_rents.crosswalk import build_full_crosswalk, save_crosswalk

    survey_year = getattr(args, "year", None) or DEFAULT_SURVEY_YEAR

    console.print(f"[bold]Building PUMA-to-county crosswalk for {survey_year}...[/bold]")

    crosswalk = build_full_crosswalk(survey_year)
    save_crosswalk(crosswalk, survey_year)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total PUMA-county pairs: {len(crosswalk)}")
    console.print(f"  Unique counties: {crosswalk['county_geoid'].nunique()}")
    console.print(f"  Unique PUMAs: {crosswalk['state_puma'].nunique()}")

    return 0


def build_crosswalk_panel(args: argparse.Namespace) -> int:
    """Build crosswalks for all PUMA vintages needed by panel."""
    from urban_rents.crosswalk import build_crosswalks_for_panel, save_crosswalk

    # Parse survey years
    if args.years:
        survey_years = [int(y) for y in args.years.split(",")]
    elif args.preset:
        survey_years = RECOMMENDED_PANEL_YEARS.get(args.preset, RECOMMENDED_PANEL_YEARS["standard"])
    else:
        survey_years = RECOMMENDED_PANEL_YEARS["standard"]

    console.print(f"[bold]Building crosswalks for panel: {survey_years}[/bold]")

    crosswalks = build_crosswalks_for_panel(survey_years)

    for vintage, crosswalk in crosswalks.items():
        save_crosswalk(crosswalk)
        console.print(f"\n[bold]{vintage} Crosswalk Summary:[/bold]")
        console.print(f"  Total PUMA-county pairs: {len(crosswalk)}")
        console.print(f"  Unique counties: {crosswalk['county_geoid'].nunique()}")

    return 0


def calculate_returns(args: argparse.Namespace) -> int:
    """Calculate urban net returns."""
    from urban_rents.net_returns import (
        build_final_dataset,
        print_summary_statistics,
        save_final_dataset,
    )

    survey_year = getattr(args, "year", None) or DEFAULT_SURVEY_YEAR

    console.print(f"[bold]Calculating urban net returns for {survey_year}...[/bold]")

    final_data = build_final_dataset(survey_year=survey_year)
    save_final_dataset(final_data, survey_year=survey_year)
    print_summary_statistics(final_data)

    return 0


def build_panel(args: argparse.Namespace) -> int:
    """Build complete panel dataset."""
    from urban_rents.panel import run_panel_pipeline

    # Parse survey years
    if args.years:
        survey_years = [int(y) for y in args.years.split(",")]
    else:
        survey_years = None

    panel_type = getattr(args, "preset", "standard")
    soc_methodology = getattr(args, "soc_method", "nearest")
    output_format = getattr(args, "format", "both")

    panel = run_panel_pipeline(
        survey_years=survey_years,
        panel_type=panel_type,
        soc_methodology=soc_methodology,
        output_format=output_format,
    )

    return 0


def visualize(args: argparse.Namespace) -> int:
    """Generate visualizations."""
    import pandas as pd

    from urban_rents.config import OUTPUT_DIR
    from urban_rents.visualization import create_all_visualizations

    survey_year = getattr(args, "year", None)

    # Determine which data file to load
    if survey_year:
        data_path = OUTPUT_DIR / f"urban_net_returns_{survey_year}.parquet"
        if not data_path.exists():
            data_path = OUTPUT_DIR / f"urban_net_returns_{survey_year}.csv"
    else:
        data_path = OUTPUT_DIR / "urban_net_returns.parquet"
        if not data_path.exists():
            data_path = OUTPUT_DIR / "urban_net_returns.csv"

    if not data_path.exists():
        console.print("[red]Final dataset not found. Run 'calculate' first.[/red]")
        return 1

    if data_path.suffix == ".parquet":
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_csv(data_path, dtype={"state_fips": str, "county_fips": str, "county_geoid": str})

    create_all_visualizations(data)

    return 0


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the complete pipeline for a single survey year."""
    survey_year = getattr(args, "year", None) or DEFAULT_SURVEY_YEAR

    steps = [
        ("download", lambda: download_data(argparse.Namespace(type="all", force=args.force, state=None, year=survey_year))),
        ("process_pums", lambda: process_pums(argparse.Namespace(year=survey_year))),
        ("build_crosswalk", lambda: build_crosswalk(argparse.Namespace(year=survey_year))),
        ("calculate_returns", lambda: calculate_returns(argparse.Namespace(year=survey_year))),
        ("visualize", lambda: visualize(argparse.Namespace(year=survey_year))),
    ]

    console.print(f"[bold]Running pipeline for survey year {survey_year}[/bold]")

    for step_name, step_func in steps:
        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[bold magenta]STEP: {step_name.upper()}[/bold magenta]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]\n")

        try:
            result = step_func()
            if result != 0:
                console.print(f"[red]Step {step_name} failed with code {result}[/red]")
                return result
        except Exception as e:
            console.print(f"[red]Step {step_name} failed with error: {e}[/red]")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    console.print("\n[bold green]Pipeline completed successfully![/bold green]")
    return 0


def run_panel_pipeline(args: argparse.Namespace) -> int:
    """Run the complete panel pipeline."""
    from urban_rents.panel import run_panel_pipeline as _run_panel

    # Parse survey years
    if args.years:
        survey_years = [int(y) for y in args.years.split(",")]
    else:
        survey_years = None

    panel_type = getattr(args, "preset", "standard")

    console.print(f"[bold]Running panel pipeline[/bold]")
    console.print(f"  Preset: {panel_type}")
    console.print(f"  Years: {survey_years or RECOMMENDED_PANEL_YEARS.get(panel_type)}")

    # First download all data
    if not args.skip_download:
        years = survey_years or RECOMMENDED_PANEL_YEARS.get(panel_type, [2013, 2018, 2023])
        download_panel(argparse.Namespace(
            years=",".join(str(y) for y in years),
            preset=panel_type,
            state=args.state,
            force=args.force,
        ))

    # Process PUMS for all years
    if not args.skip_pums:
        process_pums_panel(argparse.Namespace(
            years=args.years,
            preset=panel_type,
        ))

    # Build crosswalks
    if not args.skip_crosswalk:
        build_crosswalk_panel(argparse.Namespace(
            years=args.years,
            preset=panel_type,
        ))

    # Build final panel
    build_panel(argparse.Namespace(
        years=args.years,
        preset=panel_type,
        soc_method=getattr(args, "soc_method", "nearest"),
        format=getattr(args, "format", "both"),
    ))

    console.print("\n[bold green]Panel pipeline completed successfully![/bold green]")
    return 0


def apply_backcast(args: argparse.Namespace) -> int:
    """Apply hybrid backcasting methodology to enhance panel data."""
    from urban_rents.backcast import (
        apply_hybrid_backcast,
        generate_methodology_report,
    )
    from urban_rents.config import OUTPUT_DIR

    # Determine input file
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = OUTPUT_DIR / "county_urban_net_returns_panel_long.parquet"

    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_path}[/red]")
        console.print("[yellow]Run 'run-panel' first to generate the base panel.[/yellow]")
        return 1

    # Determine output file
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / "county_urban_net_returns_panel_enhanced.parquet"

    # Apply backcast
    enhanced_panel = apply_hybrid_backcast(
        input_path=input_path,
        output_path=output_path,
        anchor_year=args.anchor_year,
        smoothing_weight=args.smoothing_weight,
    )

    # Generate methodology report
    if args.report:
        report_path = OUTPUT_DIR / "hybrid_methodology_report.md"
        generate_methodology_report(enhanced_panel, report_path)

    console.print("\n[bold green]Backcast completed successfully![/bold green]")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate county-level urban net returns for the United States",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download required data")
    download_parser.add_argument(
        "type",
        choices=["all", "pums", "puma", "county"],
        help="Type of data to download",
    )
    download_parser.add_argument(
        "--state",
        type=str,
        help="Specific state FIPS code (for pums/puma)",
    )
    download_parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_SURVEY_YEAR,
        help=f"Survey year (default: {DEFAULT_SURVEY_YEAR})",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist",
    )
    download_parser.set_defaults(func=download_data)

    # Download panel command
    download_panel_parser = subparsers.add_parser("download-panel", help="Download data for panel dataset")
    download_panel_parser.add_argument(
        "--years",
        type=str,
        help="Comma-separated survey years (e.g., 2013,2018,2023)",
    )
    download_panel_parser.add_argument(
        "--preset",
        choices=["standard", "extended", "full_annual", "maximum", "milestones", "five_year_only"],
        default="standard",
        help="Panel preset: standard (2013,2018,2023), full_annual (2000-2023), maximum, milestones, etc.",
    )
    download_panel_parser.add_argument(
        "--state",
        type=str,
        help="Specific state FIPS code",
    )
    download_panel_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist",
    )
    download_panel_parser.set_defaults(func=download_panel)

    # Process PUMS command
    pums_parser = subparsers.add_parser("process-pums", help="Process PUMS data")
    pums_parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_SURVEY_YEAR,
        help=f"Survey year (default: {DEFAULT_SURVEY_YEAR})",
    )
    pums_parser.set_defaults(func=process_pums)

    # Process PUMS panel command
    pums_panel_parser = subparsers.add_parser("process-pums-panel", help="Process PUMS data for panel")
    pums_panel_parser.add_argument(
        "--years",
        type=str,
        help="Comma-separated survey years",
    )
    pums_panel_parser.add_argument(
        "--preset",
        choices=["standard", "extended", "full_annual", "maximum", "milestones", "five_year_only"],
        default="standard",
        help="Panel preset",
    )
    pums_panel_parser.set_defaults(func=process_pums_panel)

    # Build crosswalk command
    crosswalk_parser = subparsers.add_parser("build-crosswalk", help="Build PUMA-county crosswalk")
    crosswalk_parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_SURVEY_YEAR,
        help=f"Survey year (default: {DEFAULT_SURVEY_YEAR})",
    )
    crosswalk_parser.set_defaults(func=build_crosswalk)

    # Build crosswalk panel command
    crosswalk_panel_parser = subparsers.add_parser("build-crosswalk-panel", help="Build crosswalks for panel")
    crosswalk_panel_parser.add_argument(
        "--years",
        type=str,
        help="Comma-separated survey years",
    )
    crosswalk_panel_parser.add_argument(
        "--preset",
        choices=["standard", "extended", "full_annual", "maximum", "milestones", "five_year_only"],
        default="standard",
        help="Panel preset",
    )
    crosswalk_panel_parser.set_defaults(func=build_crosswalk_panel)

    # Calculate returns command
    calculate_parser = subparsers.add_parser("calculate", help="Calculate urban net returns")
    calculate_parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_SURVEY_YEAR,
        help=f"Survey year (default: {DEFAULT_SURVEY_YEAR})",
    )
    calculate_parser.set_defaults(func=calculate_returns)

    # Build panel command
    panel_parser = subparsers.add_parser("build-panel", help="Build complete panel dataset")
    panel_parser.add_argument(
        "--years",
        type=str,
        help="Comma-separated survey years",
    )
    panel_parser.add_argument(
        "--preset",
        choices=["standard", "extended", "full_annual", "maximum", "milestones", "five_year_only"],
        default="standard",
        help="Panel preset (default: standard)",
    )
    panel_parser.add_argument(
        "--soc-method",
        choices=["year_specific", "fixed_2023", "nearest"],
        default="nearest",
        help="SOC parameter methodology (default: nearest)",
    )
    panel_parser.add_argument(
        "--format",
        choices=["long", "wide", "both"],
        default="both",
        help="Output format (default: both)",
    )
    panel_parser.set_defaults(func=build_panel)

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument(
        "--year",
        type=int,
        help="Survey year to visualize",
    )
    viz_parser.set_defaults(func=visualize)

    # Run pipeline command (single year)
    pipeline_parser = subparsers.add_parser("run", help="Run complete pipeline (single year)")
    pipeline_parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_SURVEY_YEAR,
        help=f"Survey year (default: {DEFAULT_SURVEY_YEAR})",
    )
    pipeline_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download data even if files exist",
    )
    pipeline_parser.set_defaults(func=run_pipeline)

    # Run panel pipeline command
    panel_pipeline_parser = subparsers.add_parser("run-panel", help="Run complete panel pipeline")
    panel_pipeline_parser.add_argument(
        "--years",
        type=str,
        help="Comma-separated survey years",
    )
    panel_pipeline_parser.add_argument(
        "--preset",
        choices=["standard", "extended", "full_annual", "maximum", "milestones", "five_year_only"],
        default="standard",
        help="Panel preset",
    )
    panel_pipeline_parser.add_argument(
        "--state",
        type=str,
        help="Specific state FIPS code (for testing)",
    )
    panel_pipeline_parser.add_argument(
        "--soc-method",
        choices=["year_specific", "fixed_2023", "nearest"],
        default="nearest",
        help="SOC parameter methodology",
    )
    panel_pipeline_parser.add_argument(
        "--format",
        choices=["long", "wide", "both"],
        default="both",
        help="Output format",
    )
    panel_pipeline_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download data",
    )
    panel_pipeline_parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step",
    )
    panel_pipeline_parser.add_argument(
        "--skip-pums",
        action="store_true",
        help="Skip PUMS processing step",
    )
    panel_pipeline_parser.add_argument(
        "--skip-crosswalk",
        action="store_true",
        help="Skip crosswalk building step",
    )
    panel_pipeline_parser.set_defaults(func=run_panel_pipeline)

    # Backcast command
    backcast_parser = subparsers.add_parser(
        "backcast",
        help="Apply hybrid backcasting to extend panel to 2000",
    )
    backcast_parser.add_argument(
        "--input",
        type=str,
        help="Input panel parquet file (default: county_urban_net_returns_panel_long.parquet)",
    )
    backcast_parser.add_argument(
        "--output",
        type=str,
        help="Output enhanced panel file (default: county_urban_net_returns_panel_enhanced.parquet)",
    )
    backcast_parser.add_argument(
        "--anchor-year",
        type=int,
        default=2009,
        help="Anchor year for backcasting (default: 2009, first stable 5-year estimate)",
    )
    backcast_parser.add_argument(
        "--smoothing-weight",
        type=float,
        default=0.5,
        help="Weight for HPI in smoothing volatile years (0-1, default: 0.5)",
    )
    backcast_parser.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="Generate methodology report (default: True)",
    )
    backcast_parser.set_defaults(func=apply_backcast)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
