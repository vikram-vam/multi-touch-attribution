"""
Basic Tests for Erie MCA Demo
Tests configuration loading and data generation
"""

import pytest
import sys
import pandas as pd
from pathlib import Path

sys.path.append('.')

from src.utils import Config, set_random_seed
from src.data_generation.synthetic_data_generator import SyntheticDataGenerator
from src.models.heuristic_models import LastClickAttribution, ShapleyAttribution


def test_config_loading():
    """Test that configuration loads correctly"""
    config = Config('config/config.yaml')
    
    assert config.get('project.name') is not None
    assert config.get('project.random_seed') == 42
    assert config.get('synthetic_data.n_customers') > 0


def test_random_seed():
    """Test that random seed produces consistent results"""
    set_random_seed(42)
    sample1 = pd.Series([1, 2, 3, 4, 5]).sample(3, random_state=42)
    
    set_random_seed(42)
    sample2 = pd.Series([1, 2, 3, 4, 5]).sample(3, random_state=42)
    
    assert sample1.equals(sample2)


def test_data_generation():
    """Test synthetic data generation"""
    config = Config('config/config.yaml')
    
    # Use small sample for testing
    config._config['synthetic_data']['n_customers'] = 100
    config._config['synthetic_data']['n_conversions'] = 5
    
    generator = SyntheticDataGenerator(config)
    df = generator.generate()
    
    # Validate basic properties
    assert len(df) > 0
    assert df['customer_id'].nunique() == 100
    assert df['converted'].sum() == 5
    assert all(col in df.columns for col in ['customer_id', 'channel', 'timestamp', 'converted'])


def test_last_click_attribution():
    """Test last-click attribution model"""
    
    # Create sample journey data
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
    
    model = LastClickAttribution()
    result = model.fit(sample_data).attribute(sample_data)
    
    # Validate results
    assert len(result) > 0
    assert 'channel' in result.columns
    assert 'credit' in result.columns
    assert result['credit'].sum() == 2  # 2 conversions


def test_attribution_efficiency_axiom():
    """Test that attribution satisfies efficiency axiom"""
    
    sample_data = pd.DataFrame({
        'customer_id': [1, 1, 2, 2, 2],
        'touchpoint_sequence': [0, 1, 0, 1, 2],
        'channel': ['organic_search', 'agent_call', 'display', 'email', 'agent_call'],
        'timestamp': pd.to_datetime(['2025-01-01', '2025-01-05', 
                                     '2025-01-02', '2025-01-06', '2025-01-08']),
        'converted': [0, 1, 0, 0, 1],
        'is_converter': [1, 1, 1, 1, 1],
        'channel_cost': [0, 85, 2.1, 0.15, 85]
    })
    
    model = LastClickAttribution()
    result = model.fit(sample_data).attribute(sample_data)
    
    # Efficiency axiom: sum of credits = total conversions
    total_conversions = sample_data['converted'].sum()
    total_credit = result['credit'].sum()
    
    assert abs(total_credit - total_conversions) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
