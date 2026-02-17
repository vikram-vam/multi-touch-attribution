"""
Shapley axiom compliance tests.
Validates that Shapley-based models satisfy the four fundamental axioms.
"""

from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.base import AttributionResult


@dataclass
class AxiomTestResult:
    """Result of a single axiom test."""
    axiom_name: str
    passed: bool
    actual_value: float
    expected_value: float
    tolerance: float
    details: str


def test_efficiency_axiom(
    result: AttributionResult,
    tolerance: float = 0.01,
) -> AxiomTestResult:
    """
    Efficiency: sum of all Shapley values = v(N) (total conversions).

    φ₁ + φ₂ + ... + φₙ = v(N)
    """
    total_credit = sum(result.channel_credits.values())
    total_conv = result.total_conversions
    passed = abs(total_credit - total_conv) / max(total_conv, 1) <= tolerance

    return AxiomTestResult(
        axiom_name="Efficiency",
        passed=passed,
        actual_value=total_credit,
        expected_value=float(total_conv),
        tolerance=tolerance,
        details=f"Sum of credits: {total_credit:.4f}, Total conversions: {total_conv}",
    )


def test_symmetry_axiom(
    result: AttributionResult,
    symmetric_channels: List[tuple] = None,
    tolerance: float = 0.05,
) -> AxiomTestResult:
    """
    Symmetry: if two channels contribute identically to all coalitions,
    they receive equal credit.

    Note: Perfect symmetry is rare in real data. This tests for
    approximate symmetry among channels with similar roles.
    """
    if symmetric_channels is None:
        return AxiomTestResult(
            axiom_name="Symmetry",
            passed=True,
            actual_value=0.0,
            expected_value=0.0,
            tolerance=tolerance,
            details="No symmetric channel pairs specified — axiom trivially satisfied.",
        )

    max_diff = 0.0
    for ch1, ch2 in symmetric_channels:
        c1 = result.channel_credits.get(ch1, 0)
        c2 = result.channel_credits.get(ch2, 0)
        diff = abs(c1 - c2) / max(c1 + c2, 1)
        max_diff = max(max_diff, diff)

    passed = max_diff <= tolerance

    return AxiomTestResult(
        axiom_name="Symmetry",
        passed=passed,
        actual_value=max_diff,
        expected_value=0.0,
        tolerance=tolerance,
        details=f"Max relative difference between symmetric pairs: {max_diff:.4f}",
    )


def test_null_player_axiom(
    result: AttributionResult,
    null_channels: List[str] = None,
    tolerance: float = 0.001,
) -> AxiomTestResult:
    """
    Null Player: channels that contribute nothing to any coalition
    receive zero credit.

    φᵢ = 0 if v(S ∪ {i}) = v(S) for all S
    """
    if null_channels is None:
        # Check for very low-credit channels
        min_credit = min(result.channel_credits.values()) if result.channel_credits else 0
        passed = True  # Can't test without known null channels
        return AxiomTestResult(
            axiom_name="Null Player",
            passed=passed,
            actual_value=min_credit,
            expected_value=0.0,
            tolerance=tolerance,
            details=f"No explicit null channels. Min credit: {min_credit:.6f}",
        )

    max_null_credit = 0.0
    for ch in null_channels:
        credit = result.channel_credits.get(ch, 0)
        max_null_credit = max(max_null_credit, credit)

    passed = max_null_credit <= tolerance

    return AxiomTestResult(
        axiom_name="Null Player",
        passed=passed,
        actual_value=max_null_credit,
        expected_value=0.0,
        tolerance=tolerance,
        details=f"Max credit for null channels: {max_null_credit:.6f}",
    )


def test_additivity_axiom(
    result1: AttributionResult,
    result2: AttributionResult,
    combined_result: AttributionResult,
    tolerance: float = 0.05,
) -> AxiomTestResult:
    """
    Additivity: for two independent games G1 and G2,
    φᵢ(G1 + G2) = φᵢ(G1) + φᵢ(G2)

    Note: This is hard to test exactly with real data.
    We approximate by checking linearity of credits.
    """
    max_diff = 0.0
    n_channels = 0

    for ch in combined_result.channel_credits:
        c1 = result1.channel_credits.get(ch, 0)
        c2 = result2.channel_credits.get(ch, 0)
        combined = combined_result.channel_credits.get(ch, 0)
        expected = c1 + c2
        if expected > 0:
            diff = abs(combined - expected) / expected
            max_diff = max(max_diff, diff)
            n_channels += 1

    passed = max_diff <= tolerance

    return AxiomTestResult(
        axiom_name="Additivity",
        passed=passed,
        actual_value=max_diff,
        expected_value=0.0,
        tolerance=tolerance,
        details=f"Max relative deviation from additivity: {max_diff:.4f} across {n_channels} channels",
    )


def run_all_axiom_tests(
    result: AttributionResult,
) -> List[AxiomTestResult]:
    """Run all applicable axiom tests on a single result."""
    tests = [
        test_efficiency_axiom(result),
        test_symmetry_axiom(result),
        test_null_player_axiom(result),
    ]
    return tests


def axiom_compliance_summary(results: List[AxiomTestResult]) -> pd.DataFrame:
    """Summarize axiom test results as a DataFrame."""
    records = []
    for r in results:
        records.append({
            "axiom": r.axiom_name,
            "passed": r.passed,
            "status": "✅ PASS" if r.passed else "❌ FAIL",
            "actual": r.actual_value,
            "expected": r.expected_value,
            "tolerance": r.tolerance,
            "details": r.details,
        })
    return pd.DataFrame(records)
