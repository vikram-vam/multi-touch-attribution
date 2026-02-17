"""
Main synthetic data orchestrator.
Coordinates all data generation components and writes output Parquet files.
"""

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.utils.config_loader import load_synthetic_data_config, load_channel_config
from src.utils.data_io import save_parquet, ensure_dirs
from src.data_generation.user_profiles import generate_user_profiles
from src.data_generation.journey_simulator import (
    simulate_journeys,
    extract_touchpoints,
    extract_conversions,
    extract_journey_dataframe,
)
from src.data_generation.spend_generator import generate_channel_spend
from src.data_generation.identity_simulator import generate_identity_graph
from src.data_generation.output_validator import validate_generated_data


def generate_all_data(config_path: str = None, seed: int = None) -> dict:
    """
    Generate all synthetic data for the Erie MCA demo.

    This orchestrates:
    1. User profile sampling
    2. Journey simulation (Markov state machine)
    3. Touchpoint extraction
    4. Conversion extraction
    5. Channel spend generation
    6. Identity graph generation
    7. Output validation

    Args:
        config_path: Optional path to synthetic_data.yaml.
        seed: Optional random seed (overrides config).

    Returns:
        dict of DataFrames keyed by filename.
    """
    # Load configuration
    config = load_synthetic_data_config()
    channel_config = load_channel_config()

    random_seed = seed or config.random_seed
    rng = np.random.default_rng(random_seed)

    logger.info(f"=== Erie MCA Synthetic Data Generation ===")
    logger.info(f"  Total journeys: {config.total_journeys:,}")
    logger.info(f"  Random seed: {random_seed}")
    logger.info(f"  Simulation months: {config.simulation_months}")
    logger.info(f"  Target conversion rate: ~3%")

    # ── Step 1: Generate user profiles ──
    logger.info("\n[1/7] Generating user profiles...")
    profiles = generate_user_profiles(config.total_journeys, rng)
    logger.info(f"  Generated {len(profiles):,} profiles")

    # ── Step 2: Simulate journeys ──
    logger.info("\n[2/7] Simulating customer journeys...")
    # Process in batches for memory efficiency and progress tracking
    batch_size = 10_000
    all_journeys = []

    for i in tqdm(range(0, len(profiles), batch_size), desc="  Simulating"):
        batch = profiles[i:i + batch_size]
        batch_journeys = simulate_journeys(
            batch, rng,
            max_touchpoints=int(config.avg_touchpoints_converting * 3),
            agent_last_touch_pct=config.agent_last_touch_pct,
        )
        all_journeys.extend(batch_journeys)

    logger.info(f"  Generated {len(all_journeys):,} journeys")
    converting = [j for j in all_journeys if j["is_converting"]]
    logger.info(f"  Converting journeys: {len(converting):,} ({len(converting)/len(all_journeys)*100:.1f}%)")

    # ── Step 3: Extract touchpoints ──
    logger.info("\n[3/7] Extracting touchpoints...")
    touchpoints_df = extract_touchpoints(all_journeys, rng)
    logger.info(f"  Total touchpoints: {len(touchpoints_df):,}")

    # ── Step 4: Extract conversions ──
    logger.info("\n[4/7] Extracting conversions...")
    conversions_df = extract_conversions(all_journeys, rng)
    logger.info(f"  Total conversions: {len(conversions_df):,}")

    # ── Step 5: Create journey DataFrame ──
    logger.info("\n[5/7] Creating journey DataFrame...")
    journeys_df = extract_journey_dataframe(all_journeys)
    logger.info(f"  Journey records: {len(journeys_df):,}")

    # ── Step 6: Generate channel spend ──
    logger.info("\n[6/7] Generating channel spend data...")
    channel_spend_df = generate_channel_spend(
        total_annual_spend=config.total_annual_spend,
        rng=rng,
    )
    total_spend = channel_spend_df["spend_dollars"].sum()
    logger.info(f"  Total annual spend: ${total_spend:,.0f}")

    # ── Step 7: Generate identity graph ──
    logger.info("\n[7/7] Generating identity graph...")
    unique_pids = journeys_df["persistent_id"].unique().tolist()
    # Sample subset for identity graph (full graph for all 150K users would be huge)
    sample_size = min(10_000, len(unique_pids))
    sample_pids = list(rng.choice(unique_pids, size=sample_size, replace=False))
    identity_graph_df = generate_identity_graph(sample_pids, rng)
    logger.info(f"  Identity records: {len(identity_graph_df):,}")

    # ── Validate generated data ──
    logger.info("\n── Validation ──")
    passed, results = validate_generated_data(journeys_df)
    if passed:
        logger.info("  [OK] All validation targets met!")
    else:
        logger.warning("  [WARN] Some validation targets missed -- check ranges above")

    # ── Save all data ──
    logger.info("\n-- Saving data to Parquet --")
    ensure_dirs()

    save_parquet(touchpoints_df, "touchpoints.parquet", "raw")
    logger.info("  [OK] data/raw/touchpoints.parquet")

    save_parquet(conversions_df, "conversions.parquet", "raw")
    logger.info("  [OK] data/raw/conversions.parquet")

    save_parquet(journeys_df, "journeys.parquet", "raw")
    logger.info("  [OK] data/raw/journeys.parquet")

    save_parquet(channel_spend_df, "channel_spend.parquet", "raw")
    logger.info("  [OK] data/raw/channel_spend.parquet")

    save_parquet(identity_graph_df, "identity_graph.parquet", "raw")
    logger.info("  [OK] data/raw/identity_graph.parquet")

    logger.info(f"\n=== Data generation complete ===")

    return {
        "touchpoints": touchpoints_df,
        "conversions": conversions_df,
        "journeys": journeys_df,
        "channel_spend": channel_spend_df,
        "identity_graph": identity_graph_df,
    }
