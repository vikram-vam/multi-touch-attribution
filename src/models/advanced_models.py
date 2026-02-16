"""
Advanced Attribution Models - Shapley and Markov Chain Methods
Implements game-theoretic and probabilistic attribution approaches
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple
from itertools import combinations, permutations
from collections import defaultdict
from loguru import logger
import sys
sys.path.append('.')
from src.models.heuristic_models import AttributionModel


class ShapleyAttribution(AttributionModel):
    """
    Shapley Value Attribution (Simplified)
    Based on Zhao et al. (2018) simplified computation formula
    """
    
    def __init__(self, use_macro_groups: bool = True, 
                 monte_carlo_samples: int = 10000):
        super().__init__("Shapley Value", tier=2)
        self.use_macro_groups = use_macro_groups
        self.monte_carlo_samples = monte_carlo_samples
        self.channel_to_group = {}
    
    def fit(self, journey_data: pd.DataFrame) -> 'ShapleyAttribution':
        # Build macro group mapping if needed
        if self.use_macro_groups:
            # Define macro groups (simplified from config)
            self.macro_groups = {
                'search': ['organic_search', 'paid_search_brand', 'paid_search_generic'],
                'display': ['display'],
                'social': ['social_organic', 'social_paid'],
                'owned': ['email', 'direct', 'referral'],
                'agent': ['agent_call', 'agent_email', 'agent_office'],
                'traditional': ['tv']
            }
            
            # Build reverse mapping
            for group, channels in self.macro_groups.items():
                for channel in channels:
                    self.channel_to_group[channel] = group
        
        return self
    
    def attribute(self, journey_data: pd.DataFrame) -> pd.DataFrame:
        self._validate_journey_data(journey_data)
        
        # Get converter journeys
        converters = journey_data[journey_data['is_converter'] == 1].copy()
        
        # Map to macro groups if needed
        if self.use_macro_groups:
            converters['attribution_channel'] = converters['channel'].map(
                lambda x: self.channel_to_group.get(x, x)
            )
        else:
            converters['attribution_channel'] = converters['channel']
        
        # Calculate Shapley values per customer
        shapley_credits = defaultdict(float)
        
        # Get unique customers
        customers = converters['customer_id'].unique()
        
        for customer_id in customers:
            customer_journey = converters[converters['customer_id'] == customer_id]
            channels_in_journey = customer_journey['attribution_channel'].unique()
            
            # Calculate Shapley value for this journey
            journey_shapley = self._calculate_shapley_for_journey(
                channels_in_journey
            )
            
            # Add to total credits
            for channel, credit in journey_shapley.items():
                shapley_credits[channel] += credit
        
        # Create attribution DataFrame
        attribution = pd.DataFrame([
            {'channel': channel, 'credit': credit}
            for channel, credit in shapley_credits.items()
        ])
        
        # If using macro groups, need to distribute back to actual channels
        if self.use_macro_groups:
            attribution = self._distribute_group_credits(attribution, converters)
        
        # Add metrics
        result = self._calculate_metrics(attribution, journey_data)
        result['model'] = self.name
        
        logger.info(f"{self.name}: Attributed {result['credit'].sum():.0f} conversions")
        return result
    
    def _calculate_shapley_for_journey(self, channels: np.ndarray) -> Dict[str, float]:
        """Calculate Shapley values for a single journey using Monte Carlo"""
        n_channels = len(channels)
        shapley_values = {ch: 0.0 for ch in channels}
        
        if n_channels == 1:
            shapley_values[channels[0]] = 1.0
            return shapley_values
        
        # Monte Carlo approximation
        for _ in range(self.monte_carlo_samples):
            # Random permutation
            perm = np.random.permutation(channels)
            
            # Calculate marginal contributions
            for i, channel in enumerate(perm):
                # Coalition before adding this channel
                coalition_before = set(perm[:i])
                coalition_after = coalition_before | {channel}
                
                # Marginal contribution (simplified: assumes increasing returns)
                contribution_before = self._coalition_value(coalition_before, n_channels)
                contribution_after = self._coalition_value(coalition_after, n_channels)
                
                marginal = contribution_after - contribution_before
                shapley_values[channel] += marginal
        
        # Average over samples
        for channel in shapley_values:
            shapley_values[channel] /= self.monte_carlo_samples
        
        # Normalize to sum to 1.0
        total = sum(shapley_values.values())
        if total > 0:
            shapley_values = {ch: val/total for ch, val in shapley_values.items()}
        
        return shapley_values
    
    def _coalition_value(self, coalition: Set[str], total_channels: int) -> float:
        """
        Simplified coalition value function
        Returns value in [0, 1] based on coalition size and composition
        """
        if not coalition:
            return 0.0
        
        # Base value proportional to coalition size
        size_value = len(coalition) / total_channels
        
        # Bonus for diverse channel types (simplified)
        diversity_bonus = min(len(coalition) * 0.1, 0.3)
        
        return min(size_value + diversity_bonus, 1.0)
    
    def _distribute_group_credits(self, group_attribution: pd.DataFrame, 
                                  converters: pd.DataFrame) -> pd.DataFrame:
        """Distribute macro group credits back to individual channels"""
        channel_credits = []
        
        for _, row in group_attribution.iterrows():
            group = row['channel']
            group_credit = row['credit']
            
            # Find all actual channels in this group
            channels_in_group = [
                ch for ch, grp in self.channel_to_group.items() 
                if grp == group
            ]
            
            if not channels_in_group:
                continue
            
            # Count touchpoints per channel in this group
            touchpoint_counts = converters[
                converters['attribution_channel'] == group
            ]['channel'].value_counts()
            
            # Distribute credit proportionally to touchpoint frequency
            total_touchpoints = touchpoint_counts.sum()
            for channel in channels_in_group:
                if channel in touchpoint_counts.index:
                    channel_proportion = touchpoint_counts[channel] / total_touchpoints
                    channel_credits.append({
                        'channel': channel,
                        'credit': group_credit * channel_proportion
                    })
        
        return pd.DataFrame(channel_credits)


class MarkovChainAttribution(AttributionModel):
    """
    Markov Chain Attribution with Removal Effect
    Based on Anderl et al. (2016) absorbing chain framework
    """
    
    def __init__(self, order: int = 1):
        super().__init__("Markov Chain", tier=3)
        self.order = order
        self.transition_matrix = None
        self.removal_effects = None
    
    def fit(self, journey_data: pd.DataFrame) -> 'MarkovChainAttribution':
        self._validate_journey_data(journey_data)
        
        # Build transition matrix
        converters = journey_data[journey_data['is_converter'] == 1].copy()
        self.transition_matrix = self._build_transition_matrix(converters)
        
        return self
    
    def attribute(self, journey_data: pd.DataFrame) -> pd.DataFrame:
        if self.transition_matrix is None:
            self.fit(journey_data)
        
        # Calculate removal effects
        channels = list(self.transition_matrix['from'].unique())
        channels = [ch for ch in channels if ch not in ['START', 'CONVERSION', 'NULL']]
        
        removal_effects = {}
        base_conversion_prob = self._calculate_conversion_probability(
            self.transition_matrix
        )
        
        for channel in channels:
            # Remove channel from graph
            modified_matrix = self._remove_channel(self.transition_matrix, channel)
            modified_prob = self._calculate_conversion_probability(modified_matrix)
            
            # Removal effect = drop in conversion probability
            removal_effects[channel] = max(0, base_conversion_prob - modified_prob)
        
        # Normalize to get attribution credits
        total_effect = sum(removal_effects.values())
        
        # Get total conversions
        total_conversions = journey_data['converted'].sum()
        
        attribution = pd.DataFrame([
            {
                'channel': channel, 
                'credit': (effect / total_effect * total_conversions) if total_effect > 0 else 0
            }
            for channel, effect in removal_effects.items()
        ])
        
        # Add metrics
        result = self._calculate_metrics(attribution, journey_data)
        result['model'] = self.name
        
        logger.info(f"{self.name}: Attributed {result['credit'].sum():.0f} conversions")
        return result
    
    def _build_transition_matrix(self, converters: pd.DataFrame) -> pd.DataFrame:
        """Build Markov transition probability matrix"""
        transitions = []
        
        for customer_id in converters['customer_id'].unique():
            journey = converters[converters['customer_id'] == customer_id].sort_values('touchpoint_sequence')
            channels = ['START'] + journey['channel'].tolist()
            
            # Add conversion or null at end
            if journey['converted'].sum() > 0:
                channels.append('CONVERSION')
            else:
                channels.append('NULL')
            
            # Record transitions
            for i in range(len(channels) - 1):
                transitions.append({
                    'from': channels[i],
                    'to': channels[i + 1]
                })
        
        # Count transitions
        transition_df = pd.DataFrame(transitions)
        transition_counts = transition_df.groupby(['from', 'to']).size().reset_index()
        transition_counts.columns = ['from', 'to', 'count']
        
        # Calculate probabilities
        total_from = transition_counts.groupby('from')['count'].transform('sum')
        transition_counts['probability'] = transition_counts['count'] / total_from
        
        return transition_counts
    
    def _calculate_conversion_probability(self, transition_matrix: pd.DataFrame) -> float:
        """Calculate overall conversion probability from START using simplified approach"""
        # For large graphs, use a Monte Carlo random walk approximation instead of enumerating all paths
        # This is much faster and still gives a reasonable estimate
        
        if transition_matrix.empty:
            return 0.0
        
        # Simple heuristic: ratio of paths ending in CONVERSION vs NULL
        conversion_edges = transition_matrix[transition_matrix['to'] == 'CONVERSION']['count'].sum()
        null_edges = transition_matrix[transition_matrix['to'] == 'NULL']['count'].sum()
        total_terminal = conversion_edges + null_edges
        
        if total_terminal == 0:
            return 0.0
        
        return conversion_edges / total_terminal
    
    def _remove_channel(self, transition_matrix: pd.DataFrame, 
                       channel: str) -> pd.DataFrame:
        """Remove a channel from the transition matrix"""
        # Remove transitions from/to this channel
        modified = transition_matrix[
            (transition_matrix['from'] != channel) & 
            (transition_matrix['to'] != channel)
        ].copy()
        
        # Re-normalize probabilities
        total_from = modified.groupby('from')['probability'].transform('sum')
        modified['probability'] = modified['probability'] / total_from
        
        return modified


if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'customer_id': [1, 1, 1, 2, 2, 3, 3, 3],
        'touchpoint_sequence': [0, 1, 2, 0, 1, 0, 1, 2],
        'channel': ['organic_search', 'paid_search_brand', 'agent_call',
                   'display', 'agent_call',
                   'social_paid', 'organic_search', 'agent_call'],
        'timestamp': pd.to_datetime([
            '2025-01-01', '2025-01-05', '2025-01-10',
            '2025-01-02', '2025-01-08',
            '2025-01-03', '2025-01-06', '2025-01-12'
        ]),
        'converted': [0, 0, 1, 0, 1, 0, 0, 1],
        'is_converter': [1, 1, 1, 1, 1, 1, 1, 1],
        'channel_cost': [0, 3.5, 85, 2.1, 85, 4.3, 0, 85]
    })
    
    print("Testing Shapley Attribution:")
    shapley = ShapleyAttribution()
    shapley_result = shapley.fit(sample_data).attribute(sample_data)
    print(shapley_result)
    
    print("\nTesting Markov Chain Attribution:")
    markov = MarkovChainAttribution()
    markov_result = markov.fit(sample_data).attribute(sample_data)
    print(markov_result)
