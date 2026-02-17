"""
Pre-compute all attribution results and cache for fast UI loading.
Usage: python scripts/precompute.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from src.utils.data_io import load_parquet, save_parquet
from src.models.model_registry import get_all_models, get_model
from src.models.ensemble import EnsembleAttribution
from src.pipeline.time_decay import apply_time_decay
from src.utils.cache_manager import CacheManager
import pandas as pd


def main():
    logger.info("═══ Erie MCA — Pre-computation Pipeline ═══")
    start_total = time.time()

    # 1. Load data
    logger.info("[1/4] Loading generated data...")
    journeys = load_parquet("journeys.parquet", "raw")
    spend = load_parquet("channel_spend.parquet", "raw")

    # Parse list columns
    if "channel_path" not in journeys.columns and "channel_path_str" in journeys.columns:
        journeys["channel_path"] = journeys["channel_path_str"].apply(lambda x: x.split("|"))
    if "channel_set" not in journeys.columns and "channel_set_str" in journeys.columns:
        journeys["channel_set"] = journeys["channel_set_str"].apply(lambda x: x.split("|"))

    logger.info(f"  Journeys: {len(journeys):,}")
    logger.info(f"  Converting: {journeys['is_converting'].sum():,}")

    # 2. Time decay
    logger.info("[2/4] Applying time decay...")
    journeys = apply_time_decay(journeys)

    # 3. Run all models
    logger.info("[3/4] Running all attribution models...")
    models = get_all_models(exclude_tiers=["deep_learning"])  # Skip DL for speed
    all_results = []
    model_result_objects = {}

    for name, model in models.items():
        t0 = time.time()
        try:
            model.fit(journeys)
            result = model.attribute(journeys)
            model_result_objects[name] = result
            all_results.append(result.to_summary_df())
            logger.info(f"  ✅ {name}: {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"  ❌ {name}: {e}")

    # Ensemble
    ensemble = EnsembleAttribution(model_results=model_result_objects)
    ensemble.fit(journeys)
    ens_result = ensemble.attribute(journeys)
    all_results.append(ens_result.to_summary_df())

    # 4. Save
    logger.info("[4/4] Saving results...")
    attribution_df = pd.concat(all_results, ignore_index=True)
    save_parquet(attribution_df, "attribution_results.parquet", "processed")

    comparison = attribution_df.pivot_table(
        index="channel_id", columns="model_name",
        values="attribution_pct", aggfunc="first",
    ).fillna(0)
    save_parquet(comparison.reset_index(), "model_comparison.parquet", "processed")

    # Cache
    cache = CacheManager()
    cache.set("attribution_results", attribution_df)
    cache.set("model_comparison", comparison)

    elapsed = time.time() - start_total
    logger.info(f"\n═══ Pre-computation complete in {elapsed:.1f}s ═══")
    logger.info(f"  Models: {len(all_results)}")
    logger.info(f"  Results: {len(attribution_df):,} rows")
    logger.info(f"  Output: data/processed/attribution_results.parquet")


if __name__ == "__main__":
    main()
