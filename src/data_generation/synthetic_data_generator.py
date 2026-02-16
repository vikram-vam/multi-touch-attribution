"""
Synthetic Data Generation Engine
Generates Erie-calibrated customer journey data with realistic touchpoint sequences,
timing patterns, and conversion characteristics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from loguru import logger
import sys
sys.path.append('.')
from src.utils import Config, set_random_seed, save_parquet, format_number


class SyntheticDataGenerator:
    """
    Generates synthetic customer journey data calibrated to Erie Insurance's
    100% independent agent distribution model
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.random_seed = config.get('project.random_seed', 42)
        set_random_seed(self.random_seed)
        
        # Load parameters
        self.n_customers = config.get('synthetic_data.n_customers', 50000)
        self.n_conversions = config.get('synthetic_data.n_conversions', 2500)
        self.date_start = datetime.strptime(
            config.get('synthetic_data.date_range.start'), '%Y-%m-%d'
        )
        self.date_end = datetime.strptime(
            config.get('synthetic_data.date_range.end'), '%Y-%m-%d'
        )
        
        # Journey parameters
        self.min_touchpoints = config.get('synthetic_data.journey.min_touchpoints', 2)
        self.max_touchpoints = config.get('synthetic_data.journey.max_touchpoints', 12)
        self.mean_touchpoints = config.get('synthetic_data.journey.mean_touchpoints', 5.2)
        self.max_journey_days = config.get('synthetic_data.journey.max_journey_days', 90)
        
        # Channel definitions
        self.channels = self._load_channels()
        self.channel_probs = self._load_channel_probabilities()
        
        logger.info("SyntheticDataGenerator initialized")
    
    def _load_channels(self) -> List[str]:
        """Load channel IDs from config"""
        channel_defs = self.config.get('channels.definitions', [])
        return [ch['id'] for ch in channel_defs]
    
    def _load_channel_probabilities(self) -> Dict[str, Dict[str, float]]:
        """Load channel probability distributions"""
        return {
            'first': self.config.get('synthetic_data.channel_probabilities.first_touch_distribution', {}),
            'subsequent': self.config.get('synthetic_data.channel_probabilities.subsequent_touch_distribution', {}),
            'final': self.config.get('synthetic_data.channel_probabilities.final_touch_distribution', {})
        }
    
    def _sample_channel(self, position: str, exclude: Optional[List[str]] = None) -> str:
        """Sample a channel based on position in journey"""
        probs = self.channel_probs.get(position, {})
        
        # Filter out excluded channels
        if exclude:
            probs = {k: v for k, v in probs.items() if k not in exclude}
        
        # Normalize probabilities
        total = sum(probs.values())
        if total == 0:
            # Fallback to uniform over available channels
            available = [ch for ch in self.channels if ch not in (exclude or [])]
            return np.random.choice(available)
        
        probs = {k: v/total for k, v in probs.items()}
        
        channels = list(probs.keys())
        weights = list(probs.values())
        
        return np.random.choice(channels, p=weights)
    
    def _generate_journey_length(self, is_converter: bool) -> int:
        """Generate number of touchpoints for a journey"""
        # Converters have longer journeys on average
        if is_converter:
            mean = self.mean_touchpoints * 1.2
            std = 2.0
        else:
            mean = self.mean_touchpoints * 0.8
            std = 1.5
        
        # Sample from truncated normal distribution
        length = int(np.random.normal(mean, std))
        return np.clip(length, self.min_touchpoints, self.max_touchpoints)
    
    def _generate_touchpoint_timestamps(self, journey_start: datetime, 
                                       n_touchpoints: int, is_converter: bool) -> List[datetime]:
        """Generate timestamps for touchpoints"""
        if is_converter:
            # Converters have concentrated activity
            total_days = min(
                int(np.random.exponential(scale=20)),
                self.max_journey_days
            )
        else:
            # Non-converters spread out more
            total_days = min(
                int(np.random.exponential(scale=30)),
                self.max_journey_days
            )
        
        # Generate inter-arrival times
        if n_touchpoints == 1:
            return [journey_start]
        
        # Use exponential distribution for time between touches
        inter_arrival_days = np.random.exponential(
            scale=total_days/(n_touchpoints-1), 
            size=n_touchpoints-1
        )
        
        # Cumulative sum to get actual timestamps
        cumulative_days = np.cumsum([0] + list(inter_arrival_days))
        timestamps = [
            journey_start + timedelta(days=float(days))
            for days in cumulative_days
        ]
        
        return timestamps[:n_touchpoints]
    
    def _generate_single_journey(self, customer_id: int, 
                                is_converter: bool) -> List[Dict]:
        """Generate a single customer journey"""
        # Determine journey length
        n_touchpoints = self._generate_journey_length(is_converter)
        
        # Generate journey start time
        days_in_period = (self.date_end - self.date_start).days
        start_offset = np.random.randint(0, max(1, days_in_period - self.max_journey_days))
        journey_start = self.date_start + timedelta(days=start_offset)
        
        # Generate timestamps
        timestamps = self._generate_touchpoint_timestamps(
            journey_start, n_touchpoints, is_converter
        )
        
        # Generate channel sequence
        touchpoints = []
        used_channels = set()
        
        for i, timestamp in enumerate(timestamps):
            # Determine position
            if i == 0:
                position = 'first'
            elif i == len(timestamps) - 1 and is_converter:
                position = 'final'  # Last touch for converters uses final distribution
            else:
                position = 'subsequent'
            
            # Sample channel (avoid immediate repeats)
            channel = self._sample_channel(
                position, 
                exclude=[touchpoints[i-1]['channel']] if i > 0 else None
            )
            used_channels.add(channel)
            
            touchpoints.append({
                'customer_id': customer_id,
                'touchpoint_id': f"{customer_id}_{i}",
                'touchpoint_sequence': i,
                'channel': channel,
                'timestamp': timestamp,
                'converted': 1 if (is_converter and i == len(timestamps) - 1) else 0,
                'is_converter': 1 if is_converter else 0
            })
        
        return touchpoints
    
    def generate(self) -> pd.DataFrame:
        """Generate complete synthetic dataset"""
        logger.info(f"Generating synthetic data for {self.n_customers:,} customers "
                   f"({self.n_conversions:,} converters)")
        
        # Determine which customers convert
        converter_ids = set(np.random.choice(
            self.n_customers, 
            size=self.n_conversions, 
            replace=False
        ))
        
        # Generate journeys
        all_touchpoints = []
        
        for customer_id in range(self.n_customers):
            is_converter = customer_id in converter_ids
            journey = self._generate_single_journey(customer_id, is_converter)
            all_touchpoints.extend(journey)
            
            if (customer_id + 1) % 10000 == 0:
                logger.info(f"Generated {customer_id + 1:,} journeys")
        
        # Create DataFrame
        df = pd.DataFrame(all_touchpoints)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Validate
        self._validate_data(df)
        
        logger.info(f"Generated {len(df):,} total touchpoints")
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to journey data"""
        # Journey length per customer
        journey_lengths = df.groupby('customer_id').size()
        df['journey_length'] = df['customer_id'].map(journey_lengths)
        
        # Time to conversion (for converters)
        converter_data = df[df['is_converter'] == 1].groupby('customer_id').agg({
            'timestamp': ['min', 'max']
        })
        converter_data.columns = ['journey_start', 'conversion_time']
        converter_data['days_to_conversion'] = (
            converter_data['conversion_time'] - converter_data['journey_start']
        ).dt.total_seconds() / 86400
        
        df = df.merge(
            converter_data[['days_to_conversion']],
            left_on='customer_id',
            right_index=True,
            how='left'
        )
        
        # Add channel cost
        channel_costs = {}
        for ch_def in self.config.get('channels.definitions', []):
            channel_costs[ch_def['id']] = ch_def['cost_per_touch']
        df['channel_cost'] = df['channel'].map(channel_costs)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame):
        """Validate generated data quality"""
        # Check conversion count
        actual_conversions = df['converted'].sum()
        assert actual_conversions == self.n_conversions, \
            f"Conversion count mismatch: {actual_conversions} vs {self.n_conversions}"
        
        # Check unique customers
        actual_customers = df['customer_id'].nunique()
        assert actual_customers == self.n_customers, \
            f"Customer count mismatch: {actual_customers} vs {self.n_customers}"
        
        # Check journey lengths
        journey_lengths = df.groupby('customer_id').size()
        assert journey_lengths.min() >= self.min_touchpoints
        assert journey_lengths.max() <= self.max_touchpoints
        
        logger.info("Data validation passed")
    
    def generate_and_save(self, output_path: str = "data/synthetic/journey_data.parquet"):
        """Generate data and save to file"""
        df = self.generate()
        save_parquet(df, output_path, "journey data")
        
        # Generate summary statistics
        self._save_summary_stats(df)
        
        return df
    
    def _save_summary_stats(self, df: pd.DataFrame):
        """Generate and save summary statistics"""
        stats = {
            'total_touchpoints': len(df),
            'total_customers': df['customer_id'].nunique(),
            'total_conversions': df['converted'].sum(),
            'conversion_rate': df.groupby('customer_id')['converted'].max().mean(),
            'avg_journey_length': df.groupby('customer_id').size().mean(),
            'median_journey_length': df.groupby('customer_id').size().median(),
            'avg_days_to_conversion': df[df['is_converter'] == 1]['days_to_conversion'].mean(),
            'channel_distribution': df['channel'].value_counts().to_dict(),
            'touchpoints_by_channel': df.groupby('channel').size().to_dict(),
            'conversions_by_channel': df[df['converted'] == 1]['channel'].value_counts().to_dict()
        }
        
        # Save as JSON
        import json
        with open('data/synthetic/summary_stats.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info("Summary statistics saved")


if __name__ == "__main__":
    # Test data generation
    config = Config()
    generator = SyntheticDataGenerator(config)
    df = generator.generate_and_save()
    
    print("\nData Generation Summary:")
    print(f"Total touchpoints: {len(df):,}")
    print(f"Unique customers: {df['customer_id'].nunique():,}")
    print(f"Conversions: {df['converted'].sum():,}")
    print(f"\nChannel distribution:")
    print(df['channel'].value_counts())
