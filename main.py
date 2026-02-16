"""
Main Pipeline Orchestrator
Generates synthetic data, runs attribution models, and prepares results for UI
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd

# Add src to path
sys.path.append('.')

from src.utils import Config, set_random_seed, ensure_directories
from src.data_generation.synthetic_data_generator import SyntheticDataGenerator
from src.pipeline.attribution_runner import AttributionRunner, BudgetOptimizer


def main():
    """Main pipeline execution"""
    
    logger.info("=" * 60)
    logger.info("Erie MCA Demo - Pipeline Execution")
    logger.info("=" * 60)
    
    # Load configuration
    config = Config('config/config.yaml')
    set_random_seed(config.get('project.random_seed', 42))
    ensure_directories()
    
    # Step 1: Generate synthetic data
    logger.info("\n[Step 1/4] Generating synthetic customer journey data...")
    generator = SyntheticDataGenerator(config)
    journey_data = generator.generate_and_save()
    
    logger.info(f"✓ Generated {len(journey_data):,} touchpoints")
    logger.info(f"✓ {journey_data['customer_id'].nunique():,} unique customers")
    logger.info(f"✓ {journey_data['converted'].sum():,} conversions")
    
    # Step 2: Run attribution models
    logger.info("\n[Step 2/4] Running attribution models...")
    runner = AttributionRunner(config)
    attribution_results = runner.run_all_models(journey_data)
    
    # Save results
    runner.save_results()
    logger.info(f"✓ Completed {len(attribution_results['model'].unique())} attribution models")
    
    # Step 3: Validate results
    logger.info("\n[Step 3/4] Validating attribution results...")
    total_conversions = journey_data['converted'].sum()
    validation = runner.validate_results(total_conversions)
    
    passed = sum(validation.values())
    total = len(validation)
    logger.info(f"✓ Validation: {passed}/{total} models passed efficiency axiom")
    
    for model_name, is_valid in validation.items():
        status = "✓" if is_valid else "✗"
        logger.info(f"  {status} {model_name}")
    
    # Step 4: Generate insights and optimizations
    logger.info("\n[Step 4/4] Generating insights and budget optimizations...")
    
    # Generate insights
    insights = runner.generate_insights()
    logger.info(f"✓ Generated {len(insights)} key insights")
    
    # Budget optimization
    optimizer = BudgetOptimizer(config)
    shapley_optimization = optimizer.optimize_from_attribution(
        attribution_results,
        model_name="Shapley Value"
    )
    
    if not shapley_optimization.empty:
        shapley_optimization.to_parquet(
            'data/results/budget_optimization.parquet',
            index=False
        )
        logger.info("✓ Budget optimization complete")
    
    # Display summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Execution Summary")
    logger.info("=" * 60)
    
    print("\nAttribution Results by Model:")
    print("-" * 60)
    model_summary = attribution_results.groupby('model').agg({
        'credit': 'sum',
        'cost': 'sum'
    }).round(0)
    model_summary['cpa'] = (model_summary['cost'] / model_summary['credit']).round(2)
    print(model_summary)
    
    print("\nTop Channels by Shapley Attribution:")
    print("-" * 60)
    shapley_data = attribution_results[
        attribution_results['model'] == 'Shapley Value'
    ].sort_values('credit', ascending=False)
    print(shapley_data[['channel', 'credit', 'cost', 'cpa']].head(10))
    
    print("\nKey Insights:")
    print("-" * 60)
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    logger.info("\n✓ Pipeline complete! Results saved to data/results/")
    logger.info("✓ Run 'python app.py' to launch the interactive dashboard")
    
    return {
        'journey_data': journey_data,
        'attribution_results': attribution_results,
        'validation': validation,
        'insights': insights,
        'budget_optimization': shapley_optimization
    }


if __name__ == "__main__":
    try:
        results = main()
        logger.info("\n✓ Success! All pipeline steps completed.")
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        raise
