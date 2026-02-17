"""
Run the full validation suite.
Usage: python scripts/run_validation.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.utils.data_io import load_parquet


def main():
    """Run all validation checks on computed attribution results."""
    logger.info("=" * 60)
    logger.info("Erie MCA — Validation Suite")
    logger.info("=" * 60)

    start = time.time()

    # Load data
    logger.info("Loading data...")
    try:
        attribution_results = load_parquet("attribution_results.parquet", "processed")
        journeys = load_parquet("assembled_journeys.parquet", "processed")
    except FileNotFoundError:
        logger.error("Data files not found. Run 'python scripts/run_attribution.py' first.")
        return

    # 1. Data contract validation
    logger.info("\n--- Data Contract Validation ---")
    from src.validation.data_contract_tests import validate_schema, contract_summary
    violations = validate_schema(attribution_results, "attribution_results")
    violations += validate_schema(journeys, "journeys")
    summary = contract_summary(violations)
    logger.info(f"Contract violations: {len(violations)}")
    if len(violations) > 0:
        print(summary.to_string(index=False))

    # 2. Shapley axiom tests
    logger.info("\n--- Shapley Axiom Tests ---")
    from src.validation.axiom_tests import run_all_axiom_tests, axiom_compliance_summary
    from src.models.shapley_engine import ShapleyAttribution
    model = ShapleyAttribution()
    model.fit(journeys)
    result = model.attribute(journeys)
    axiom_results = run_all_axiom_tests(result)
    axiom_df = axiom_compliance_summary(axiom_results)
    print(axiom_df.to_string(index=False))

    # 3. Cross-model comparison
    logger.info("\n--- Cross-Model Comparison ---")
    from src.validation.cross_model import cross_model_summary
    cm = cross_model_summary(attribution_results)
    logger.info(f"Average pairwise Spearman ρ: {cm['avg_pairwise_rho']:.3f}")
    logger.info(f"Model clusters: {cm['model_clusters']}")

    # 4. Sanity checks
    logger.info("\n--- Business Logic Sanity Checks ---")
    from src.validation.sanity_checks import run_all_sanity_checks, sanity_check_summary
    sanity_results = run_all_sanity_checks(attribution_results, journeys)
    sanity_df = sanity_check_summary(sanity_results)
    print(sanity_df.to_string(index=False))

    # Summary
    elapsed = time.time() - start
    n_passed = sum(1 for r in axiom_results if r.passed) + sum(1 for r in sanity_results if r.passed)
    n_total = len(axiom_results) + len(sanity_results)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Validation complete in {elapsed:.1f}s")
    logger.info(f"Overall: {n_passed}/{n_total} checks passed")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
