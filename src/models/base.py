"""
Abstract base model and standard result schema.
Every attribution model must extend BaseAttributionModel.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


@dataclass
class AttributionResult:
    """
    Standard output format for all attribution models.
    Consumed by the metrics layer and UI — never the raw model internals.
    """
    model_name: str
    model_tier: str                           # e.g., "rule_based", "game_theoretic"
    channel_credits: Dict[str, float]         # channel_id → total attributed conversions
    channel_credit_pct: Dict[str, float]      # channel_id → fraction of total
    channel_credit_rank: Dict[str, int]       # channel_id → rank (1 = highest)
    journey_level_credits: pd.DataFrame       # journey_id × channel_id credit matrix
    total_conversions: int
    total_conversion_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary_df(self) -> pd.DataFrame:
        """Convert to summary DataFrame for comparison across models."""
        records = []
        for ch_id in self.channel_credits:
            records.append({
                "model_name": self.model_name,
                "model_tier": self.model_tier,
                "channel_id": ch_id,
                "attributed_conversions": self.channel_credits[ch_id],
                "attribution_pct": self.channel_credit_pct.get(ch_id, 0.0),
                "rank": self.channel_credit_rank.get(ch_id, 0),
            })
        return pd.DataFrame(records)


class BaseAttributionModel(ABC):
    """
    Abstract base class for all attribution models.
    
    All models follow:
    1. fit() — process the journey dataset
    2. attribute() — compute channel-level attribution
    3. name — model display name
    4. tier — model tier classification
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    @abstractmethod
    def tier(self) -> str:
        """Model tier classification."""
        ...

    @abstractmethod
    def fit(self, journeys: pd.DataFrame) -> None:
        """
        Fit/learn the model from journey data.
        For rule-based models, this is typically a no-op.

        Args:
            journeys: Assembled journeys DataFrame.
        """
        ...

    @abstractmethod
    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        """
        Run attribution and return standardized results.

        Args:
            journeys: Assembled journeys DataFrame.

        Returns:
            AttributionResult with channel-level credits.
        """
        ...

    def _compute_ranks(self, credits: Dict[str, float]) -> Dict[str, int]:
        """Utility: rank channels by credit (descending)."""
        sorted_channels = sorted(credits.items(), key=lambda x: -x[1])
        return {ch: rank + 1 for rank, (ch, _) in enumerate(sorted_channels)}

    def _compute_pct(self, credits: Dict[str, float]) -> Dict[str, float]:
        """Utility: convert credits to percentages."""
        total = sum(credits.values())
        if total == 0:
            return {ch: 0.0 for ch in credits}
        return {ch: val / total for ch, val in credits.items()}

    def _build_result(
        self,
        channel_credits: Dict[str, float],
        journeys: pd.DataFrame,
        journey_credits: pd.DataFrame = None,
        metadata: Dict = None,
    ) -> AttributionResult:
        """Utility: build standardized AttributionResult."""
        total_conv = int(journeys["is_converting"].sum()) if "is_converting" in journeys.columns else 0
        total_value = float(journeys["conversion_value"].sum()) if "conversion_value" in journeys.columns else 0.0

        return AttributionResult(
            model_name=self.name,
            model_tier=self.tier,
            channel_credits=channel_credits,
            channel_credit_pct=self._compute_pct(channel_credits),
            channel_credit_rank=self._compute_ranks(channel_credits),
            journey_level_credits=journey_credits if journey_credits is not None else pd.DataFrame(),
            total_conversions=total_conv,
            total_conversion_value=total_value,
            metadata=metadata or {},
        )
