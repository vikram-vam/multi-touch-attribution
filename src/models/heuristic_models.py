"""
Attribution Models - Base Classes and Heuristic Models
Implements baseline and rule-based attribution approaches
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from loguru import logger


class AttributionModel(ABC):
    """Base class for all attribution models"""
    
    def __init__(self, name: str, tier: int):
        self.name = name
        self.tier = tier
    
    @abstractmethod
    def fit(self, journey_data: pd.DataFrame) -> 'AttributionModel':
        """Fit the model to journey data"""
        pass
    
    @abstractmethod
    def attribute(self, journey_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate attribution credits
        Returns DataFrame with columns: channel, credit, cost, cpa
        """
        pass
    
    def _validate_journey_data(self, df: pd.DataFrame):
        """Validate input data format"""
        required_cols = ['customer_id', 'channel', 'converted']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _calculate_metrics(self, attribution: pd.DataFrame, 
                          journey_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cost and CPA metrics"""
        # Calculate cost per channel
        cost_df = journey_data.groupby('channel')['channel_cost'].sum().reset_index()
        cost_df.columns = ['channel', 'cost']
        
        # Merge with attribution
        result = attribution.merge(cost_df, on='channel', how='left')
        result['cost'] = result['cost'].fillna(0)
        
        # Calculate CPA
        result['cpa'] = result.apply(
            lambda row: row['cost'] / row['credit'] if row['credit'] > 0 else 0,
            axis=1
        )
        
        return result


class LastClickAttribution(AttributionModel):
    """Last-click attribution (100% credit to final touch)"""
    
    def __init__(self):
        super().__init__("Last-Click", tier=1)
    
    def fit(self, journey_data: pd.DataFrame) -> 'LastClickAttribution':
        return self
    
    def attribute(self, journey_data: pd.DataFrame) -> pd.DataFrame:
        self._validate_journey_data(journey_data)
        
        # Get last touchpoint for each converter
        converters = journey_data[journey_data['converted'] == 1].copy()
        
        # Group by channel and count
        attribution = converters.groupby('channel').size().reset_index()
        attribution.columns = ['channel', 'credit']
        attribution['credit'] = attribution['credit'].astype(float)
        
        # Add metrics
        result = self._calculate_metrics(attribution, journey_data)
        result['model'] = self.name
        
        logger.info(f"{self.name}: Attributed {result['credit'].sum():.0f} conversions")
        return result


class FirstClickAttribution(AttributionModel):
    """First-click attribution (100% credit to first touch)"""
    
    def __init__(self):
        super().__init__("First-Click", tier=1)
    
    def fit(self, journey_data: pd.DataFrame) -> 'FirstClickAttribution':
        return self
    
    def attribute(self, journey_data: pd.DataFrame) -> pd.DataFrame:
        self._validate_journey_data(journey_data)
        
        # Get first touchpoint for each converter
        converters = journey_data[journey_data['is_converter'] == 1].copy()
        first_touches = converters.groupby('customer_id').first().reset_index()
        
        # Group by channel and count
        attribution = first_touches.groupby('channel').size().reset_index()
        attribution.columns = ['channel', 'credit']
        attribution['credit'] = attribution['credit'].astype(float)
        
        # Add metrics
        result = self._calculate_metrics(attribution, journey_data)
        result['model'] = self.name
        
        logger.info(f"{self.name}: Attributed {result['credit'].sum():.0f} conversions")
        return result


class LinearAttribution(AttributionModel):
    """Linear attribution (equal credit to all touchpoints)"""
    
    def __init__(self):
        super().__init__("Linear", tier=1)
    
    def fit(self, journey_data: pd.DataFrame) -> 'LinearAttribution':
        return self
    
    def attribute(self, journey_data: pd.DataFrame) -> pd.DataFrame:
        self._validate_journey_data(journey_data)
        
        # Get converter journeys
        converters = journey_data[journey_data['is_converter'] == 1].copy()
        
        # Calculate journey length for each customer
        journey_lengths = converters.groupby('customer_id').size()
        converters['journey_length'] = converters['customer_id'].map(journey_lengths)
        
        # Each touchpoint gets 1/journey_length credit
        converters['touchpoint_credit'] = 1.0 / converters['journey_length']
        
        # Sum by channel
        attribution = converters.groupby('channel')['touchpoint_credit'].sum().reset_index()
        attribution.columns = ['channel', 'credit']
        
        # Add metrics
        result = self._calculate_metrics(attribution, journey_data)
        result['model'] = self.name
        
        logger.info(f"{self.name}: Attributed {result['credit'].sum():.0f} conversions")
        return result


class TimeDecayAttribution(AttributionModel):
    """Time-decay attribution (exponential decay toward conversion)"""
    
    def __init__(self, half_life_days: float = 7.0):
        super().__init__("Time-Decay", tier=1)
        self.half_life_days = half_life_days
    
    def fit(self, journey_data: pd.DataFrame) -> 'TimeDecayAttribution':
        return self
    
    def attribute(self, journey_data: pd.DataFrame) -> pd.DataFrame:
        self._validate_journey_data(journey_data)
        
        # Get converter journeys
        converters = journey_data[journey_data['is_converter'] == 1].copy()
        
        # Get conversion time for each customer
        conversion_times = converters[converters['converted'] == 1].set_index('customer_id')['timestamp']
        converters['conversion_time'] = converters['customer_id'].map(conversion_times)
        
        # Calculate days before conversion
        converters['days_before_conversion'] = (
            converters['conversion_time'] - converters['timestamp']
        ).dt.total_seconds() / 86400
        
        # Apply exponential decay
        decay_rate = np.log(2) / self.half_life_days
        converters['decay_weight'] = np.exp(-decay_rate * converters['days_before_conversion'])
        
        # Normalize weights per customer
        weight_sums = converters.groupby('customer_id')['decay_weight'].transform('sum')
        converters['touchpoint_credit'] = converters['decay_weight'] / weight_sums
        
        # Sum by channel
        attribution = converters.groupby('channel')['touchpoint_credit'].sum().reset_index()
        attribution.columns = ['channel', 'credit']
        
        # Add metrics
        result = self._calculate_metrics(attribution, journey_data)
        result['model'] = self.name
        
        logger.info(f"{self.name}: Attributed {result['credit'].sum():.0f} conversions")
        return result


class PositionBasedAttribution(AttributionModel):
    """Position-based attribution (U-shaped: first 40%, last 40%, middle 20%)"""
    
    def __init__(self, first_weight: float = 0.4, last_weight: float = 0.4):
        super().__init__("Position-Based", tier=1)
        self.first_weight = first_weight
        self.last_weight = last_weight
        self.middle_weight = 1.0 - first_weight - last_weight
    
    def fit(self, journey_data: pd.DataFrame) -> 'PositionBasedAttribution':
        return self
    
    def attribute(self, journey_data: pd.DataFrame) -> pd.DataFrame:
        self._validate_journey_data(journey_data)
        
        # Get converter journeys
        converters = journey_data[journey_data['is_converter'] == 1].copy()
        
        # Calculate journey length for each customer
        journey_lengths = converters.groupby('customer_id').size()
        converters['journey_length'] = converters['customer_id'].map(journey_lengths)
        
        # Calculate position-based weights
        def get_position_weight(row):
            if row['touchpoint_sequence'] == 0:
                return self.first_weight
            elif row['touchpoint_sequence'] == row['journey_length'] - 1:
                return self.last_weight
            else:
                # Middle touches share remaining weight
                n_middle = row['journey_length'] - 2
                if n_middle > 0:
                    return self.middle_weight / n_middle
                else:
                    # Only 2 touchpoints
                    return 0.0
        
        converters['touchpoint_credit'] = converters.apply(get_position_weight, axis=1)
        
        # Sum by channel
        attribution = converters.groupby('channel')['touchpoint_credit'].sum().reset_index()
        attribution.columns = ['channel', 'credit']
        
        # Add metrics
        result = self._calculate_metrics(attribution, journey_data)
        result['model'] = self.name
        
        logger.info(f"{self.name}: Attributed {result['credit'].sum():.0f} conversions")
        return result


if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'customer_id': [1, 1, 1, 2, 2],
        'touchpoint_sequence': [0, 1, 2, 0, 1],
        'channel': ['organic_search', 'paid_search_brand', 'agent_call', 
                   'display', 'agent_call'],
        'timestamp': pd.to_datetime(['2025-01-01', '2025-01-05', '2025-01-10',
                                     '2025-01-02', '2025-01-08']),
        'converted': [0, 0, 1, 0, 1],
        'is_converter': [1, 1, 1, 1, 1],
        'channel_cost': [0, 3.5, 85, 2.1, 85]
    })
    
    models = [
        LastClickAttribution(),
        FirstClickAttribution(),
        LinearAttribution(),
        TimeDecayAttribution(),
        PositionBasedAttribution()
    ]
    
    for model in models:
        result = model.fit(sample_data).attribute(sample_data)
        print(f"\n{model.name}:")
        print(result)
