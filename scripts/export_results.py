"""
Export attribution results to Excel/PDF for leave-behind.
Usage: python scripts/export_results.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from pathlib import Path
from src.utils.data_io import load_parquet


def export_to_excel(output_path: str = "data/exports/erie_mca_results.xlsx"):
    """Export all results to a multi-sheet Excel workbook."""
    logger.info("Exporting to Excel...")

    try:
        import openpyxl
    except ImportError:
        logger.warning("openpyxl not installed. Install with: pip install openpyxl")
        return

    # Load all result datasets
    datasets = {}
    parquet_files = {
        "Attribution Results": ("attribution_results.parquet", "processed"),
        "Model Comparison": ("model_comparison.parquet", "processed"),
        "Budget Scenarios": ("budget_scenarios.parquet", "processed"),
        "Path Analysis": ("path_analysis.parquet", "processed"),
        "Validation Metrics": ("validation_metrics.parquet", "processed"),
    }

    for sheet_name, (filename, subdir) in parquet_files.items():
        try:
            datasets[sheet_name] = load_parquet(filename, subdir)
            logger.info(f"  Loaded {sheet_name}: {len(datasets[sheet_name])} rows")
        except FileNotFoundError:
            logger.warning(f"  {filename} not found, skipping")

    if not datasets:
        logger.error("No data files found. Run attribution first.")
        return

    # Create exports directory
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write to Excel
    with open(output, "wb") as f:
        import pandas as pd
        with pd.ExcelWriter(f, engine="openpyxl") as writer:
            for sheet_name, df in datasets.items():
                # Limit sheet name length
                safe_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)

    logger.info(f"Excel export: {output}")
    return str(output)


def export_summary_csv(output_dir: str = "data/exports"):
    """Export key summary tables as individual CSVs."""
    logger.info("Exporting summary CSVs...")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    try:
        attr = load_parquet("attribution_results.parquet", "processed")

        # Channel summary by model
        pivot = attr.pivot_table(
            index="channel_id", columns="model_name",
            values="attribution_pct", aggfunc="first",
        ).fillna(0)
        pivot.to_csv(output / "channel_attribution_by_model.csv")
        logger.info(f"  channel_attribution_by_model.csv")

        # Model comparison
        model_summary = attr.groupby("model_name").agg(
            n_channels=("channel_id", "nunique"),
            total_conversions=("attributed_conversions", "sum"),
        ).reset_index()
        model_summary.to_csv(output / "model_summary.csv", index=False)
        logger.info(f"  model_summary.csv")

    except FileNotFoundError:
        logger.error("Attribution results not found.")


def main():
    """Export all results."""
    logger.info("=" * 60)
    logger.info("Erie MCA — Results Export")
    logger.info("=" * 60)

    start = time.time()

    export_to_excel()
    export_summary_csv()

    elapsed = time.time() - start
    logger.info(f"\nExport complete in {elapsed:.1f}s")
    logger.info(f"Files saved to data/exports/")


if __name__ == "__main__":
    main()
