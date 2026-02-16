"""
Attribution Model Runner
Orchestrates execution of multiple attribution models and aggregates results
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger
import sys
sys.path.append('.')

from src.utils import Config, save_parquet, load_parquet
from src.models.heuristic_models import (
    LastClickAttribution, FirstClickAttribution, LinearAttribution,
    TimeDecayAttribution, PositionBasedAttribution
)
from src.models.advanced_models import (
    ShapleyAttribution, MarkovChainAttribution
)


class AttributionRunner:
    """Runs multiple attribution models and compiles results"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = self._initialize_models()
        self.results = None
    
    def _initialize_models(self) -> List:
        """Initialize all enabled attribution models"""
        models = []
        
        # Heuristic models (Tier 1)
        if self.config.get('attribution_models.last_click.enabled', True):
            models.append(LastClickAttribution())
        
        if self.config.get('attribution_models.first_click.enabled', True):
            models.append(FirstClickAttribution())
        
        if self.config.get('attribution_models.linear.enabled', True):
            models.append(LinearAttribution())
        
        if self.config.get('attribution_models.time_decay.enabled', True):
            half_life = self.config.get('attribution_models.time_decay.half_life_days', 7)
            models.append(TimeDecayAttribution(half_life_days=half_life))
        
        if self.config.get('attribution_models.position_based.enabled', True):
            first_weight = self.config.get('attribution_models.position_based.first_touch_weight', 0.4)
            last_weight = self.config.get('attribution_models.position_based.last_touch_weight', 0.4)
            models.append(PositionBasedAttribution(first_weight, last_weight))
        
        # Advanced models (Tier 2-3)
        if self.config.get('attribution_models.shapley_simplified.enabled', True):
            use_macro = self.config.get('attribution_models.shapley_simplified.use_macro_groups', True)
            samples = self.config.get('attribution_models.shapley_simplified.monte_carlo_samples', 10000)
            models.append(ShapleyAttribution(use_macro_groups=use_macro, 
                                           monte_carlo_samples=samples))
        
        if self.config.get('attribution_models.markov_chain.enabled', True):
            order = self.config.get('attribution_models.markov_chain.order', 1)
            models.append(MarkovChainAttribution(order=order))
        
        logger.info(f"Initialized {len(models)} attribution models")
        return models
    
    def run_all_models(self, journey_data: pd.DataFrame) -> pd.DataFrame:
        """Run all models and compile results"""
        logger.info("Starting attribution model execution")
        
        all_results = []
        
        for model in self.models:
            try:
                logger.info(f"Running {model.name}...")
                result = model.fit(journey_data).attribute(journey_data)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error running {model.name}: {e}")
                continue
        
        # Concatenate all results
        if all_results:
            self.results = pd.concat(all_results, ignore_index=True)
            logger.info(f"Compiled results from {len(all_results)} models")
        else:
            logger.error("No model results generated")
            self.results = pd.DataFrame()
        
        return self.results
    
    def save_results(self, output_path: str = "data/results/attribution_results.parquet"):
        """Save attribution results to file"""
        if self.results is not None and not self.results.empty:
            save_parquet(self.results, output_path, "attribution results")
        else:
            logger.warning("No results to save")
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Generate model comparison summary"""
        if self.results is None or self.results.empty:
            return pd.DataFrame()
        
        # Pivot to compare models
        comparison = self.results.pivot_table(
            index='channel',
            columns='model',
            values='credit',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        return comparison
    
    def validate_results(self, total_conversions: int) -> Dict[str, bool]:
        """Validate attribution results"""
        if self.results is None or self.results.empty:
            return {}
        
        validation_results = {}
        
        for model_name in self.results['model'].unique():
            model_results = self.results[self.results['model'] == model_name]
            
            # Check efficiency axiom
            total_credit = model_results['credit'].sum()
            tolerance = total_conversions * 0.01  # 1% tolerance
            
            is_valid = abs(total_credit - total_conversions) <= tolerance
            validation_results[model_name] = is_valid
            
            if not is_valid:
                logger.warning(
                    f"{model_name} failed efficiency axiom: "
                    f"{total_credit:.0f} vs {total_conversions}"
                )
        
        return validation_results
    
    def generate_insights(self) -> List[Dict]:
        """Generate key insights from attribution results"""
        if self.results is None or self.results.empty:
            return []
        
        insights = []
        
        # Compare Last-Click vs Shapley
        last_click = self.results[self.results['model'] == 'Last-Click'].copy()
        shapley = self.results[self.results['model'] == 'Shapley Value'].copy()
        
        if not last_click.empty and not shapley.empty:
            # Merge for comparison
            comparison = last_click[['channel', 'credit']].merge(
                shapley[['channel', 'credit']],
                on='channel',
                suffixes=('_last_click', '_shapley'),
                how='outer'
            ).fillna(0)
            
            comparison['credit_diff'] = (
                comparison['credit_shapley'] - comparison['credit_last_click']
            )
            comparison['pct_change'] = (
                comparison['credit_diff'] / comparison['credit_last_click'] * 100
            ).replace([np.inf, -np.inf], 0)
            
            # Find biggest gainers/losers
            biggest_gainer = comparison.loc[comparison['credit_diff'].idxmax()]
            biggest_loser = comparison.loc[comparison['credit_diff'].idxmin()]
            
            insights.append({
                'type': 'channel_revaluation',
                'biggest_gainer': biggest_gainer['channel'],
                'gainer_pct_change': biggest_gainer['pct_change'],
                'biggest_loser': biggest_loser['channel'],
                'loser_pct_change': biggest_loser['pct_change']
            })
        
        # Check agent channel performance
        agent_channels = ['agent_call', 'agent_email', 'agent_office']
        for model_name in self.results['model'].unique():
            model_data = self.results[self.results['model'] == model_name]
            agent_credit = model_data[
                model_data['channel'].isin(agent_channels)
            ]['credit'].sum()
            total_credit = model_data['credit'].sum()
            
            agent_pct = (agent_credit / total_credit * 100) if total_credit > 0 else 0
            
            if model_name in ['Last-Click', 'Shapley Value']:
                insights.append({
                    'type': 'agent_contribution',
                    'model': model_name,
                    'agent_credit_pct': agent_pct
                })
        
        logger.info(f"Generated {len(insights)} insights")
        return insights


class BudgetOptimizer:
    """Optimizes budget allocation based on attribution results"""
    
    def __init__(self, config: Config):
        self.config = config
        self.baseline_budget = config.get('budget_optimization.baseline_budget', 5000000)
    
    def optimize_from_attribution(self, attribution_results: pd.DataFrame,
                                  model_name: str = "Shapley Value") -> pd.DataFrame:
        """Optimize budget allocation based on attribution credits"""
        
        # Get results for specified model
        model_results = attribution_results[
            attribution_results['model'] == model_name
        ].copy()
        
        if model_results.empty:
            logger.error(f"No results found for model: {model_name}")
            return pd.DataFrame()
        
        # Calculate credit percentage
        total_credit = model_results['credit'].sum()
        model_results['credit_pct'] = model_results['credit'] / total_credit
        
        # Calculate optimal budget (proportional to credit)
        model_results['optimal_budget'] = (
            model_results['credit_pct'] * self.baseline_budget
        )
        
        # Calculate expected conversions at optimal budget
        model_results['expected_conversions'] = model_results.apply(
            lambda row: (row['optimal_budget'] / row['cpa']) 
            if row['cpa'] > 0 else 0,
            axis=1
        )
        
        # Calculate improvement vs current
        model_results['current_budget'] = model_results['cost']
        model_results['budget_change'] = (
            model_results['optimal_budget'] - model_results['current_budget']
        )
        model_results['budget_change_pct'] = (
            model_results['budget_change'] / model_results['current_budget'] * 100
        ).replace([np.inf, -np.inf], 0)
        
        logger.info(f"Budget optimization complete for {model_name}")
        return model_results[
            ['channel', 'credit', 'credit_pct', 'current_budget', 
             'optimal_budget', 'budget_change', 'budget_change_pct',
             'cpa', 'expected_conversions']
        ]


if __name__ == "__main__":
    # Test runner
    from src.data_generation.synthetic_data_generator import SyntheticDataGenerator
    
    config = Config()
    
    # Generate test data
    generator = SyntheticDataGenerator(config)
    journey_data = generator.generate()
    
    # Run attribution models
    runner = AttributionRunner(config)
    results = runner.run_all_models(journey_data)
    
    print("\nAttribution Results Summary:")
    print(results.groupby('model')['credit'].sum())
    
    print("\nModel Comparison:")
    comparison = runner.get_model_comparison()
    print(comparison)
    
    # Validate
    total_conversions = journey_data['converted'].sum()
    validation = runner.validate_results(total_conversions)
    print("\nValidation Results:")
    for model, valid in validation.items():
        print(f"{model}: {'✓' if valid else '✗'}")
    
    # Generate insights
    insights = runner.generate_insights()
    print("\nKey Insights:")
    for insight in insights:
        print(insight)
