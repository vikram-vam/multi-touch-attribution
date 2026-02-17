"""
CLI script to run all attribution models and save results.
Usage: python scripts/run_attribution.py [--models all]
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger

from src.utils.data_io import load_parquet, save_parquet
from src.models.model_registry import get_all_models, get_model, MODEL_REGISTRY
from src.models.ensemble import EnsembleAttribution
from src.pipeline.journey_assembler import assemble_journeys
from src.pipeline.time_decay import apply_time_decay


def main():
    parser = argparse.ArgumentParser(description="Run attribution models")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated model names or 'all'")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    logger.info("═══ Erie MCA — Attribution Pipeline ═══")

    # Load raw data
    logger.info("\n[1/4] Loading data...")
    try:
        journeys = load_parquet("journeys.parquet", "raw")
        touchpoints = load_parquet("touchpoints.parquet", "raw")
        conversions = load_parquet("conversions.parquet", "raw")
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}. Run 'python scripts/generate_data.py' first.")
        sys.exit(1)

    # Parse channel_path from string if needed
    if "channel_path" not in journeys.columns and "channel_path_str" in journeys.columns:
        journeys["channel_path"] = journeys["channel_path_str"].apply(lambda x: x.split("|"))
    if "channel_set" not in journeys.columns and "channel_set_str" in journeys.columns:
        journeys["channel_set"] = journeys["channel_set_str"].apply(lambda x: x.split("|"))

    total = len(journeys)
    converting = journeys[journeys["is_converting"]]
    logger.info(f"  Journeys: {total:,} ({len(converting):,} converting)")

    # Apply time decay
    logger.info("\n[2/4] Computing time decay weights...")
    journeys = apply_time_decay(journeys)

    # Select models
    logger.info("\n[3/4] Running attribution models...")
    if args.models == "all":
        models = get_all_models()
    else:
        model_names = [m.strip() for m in args.models.split(",")]
        models = {name: get_model(name) for name in model_names}

    # Run each model
    all_results = []
    model_result_objects = {}

    for model_name, model in models.items():
        start = time.time()
        logger.info(f"\n  ▶ Running {MODEL_REGISTRY.get(model_name, {}).get('display_name', model_name)}...")

        try:
            model.fit(journeys)
            result = model.attribute(journeys)
            model_result_objects[model_name] = result

            summary_df = result.to_summary_df()
            all_results.append(summary_df)

            elapsed = time.time() - start
            top_ch = list(result.channel_credit_rank.keys())[0] if result.channel_credit_rank else "N/A"
            logger.info(f"    ✅ Done in {elapsed:.1f}s | Top channel: {top_ch}")
        except Exception as e:
            logger.error(f"    ❌ Failed: {e}")
            continue

    # Run ensemble
    logger.info(f"\n  ▶ Running Weighted Ensemble...")
    ensemble = EnsembleAttribution(model_results=model_result_objects)
    ensemble.fit(journeys)
    ensemble_result = ensemble.attribute(journeys)
    all_results.append(ensemble_result.to_summary_df())
    logger.info(f"    ✅ Ensemble complete")

    # Combine all results
    logger.info("\n[4/4] Saving results...")
    attribution_df = pd.concat(all_results, ignore_index=True)
    save_parquet(attribution_df, "attribution_results.parquet", "processed")
    logger.info(f"  ✅ data/processed/attribution_results.parquet ({len(attribution_df):,} rows)")

    # Save model comparison
    comparison = attribution_df.pivot_table(
        index="channel_id", columns="model_name",
        values="attribution_pct", aggfunc="first",
    ).fillna(0)
    save_parquet(comparison.reset_index(), "model_comparison.parquet", "processed")
    logger.info(f"  ✅ data/processed/model_comparison.parquet")

    logger.info(f"\n═══ Attribution pipeline complete ═══")
    logger.info(f"  Models run: {len(all_results)}")
    logger.info(f"  Total result rows: {len(attribution_df):,}")


if __name__ == "__main__":
    main()
