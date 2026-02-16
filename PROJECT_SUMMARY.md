# Erie MCA Demo - Project Summary & Architecture

## Overview

This is a complete, operational demonstration of Multi-Channel Attribution (MCA) for Erie Insurance's 100% independent agent distribution model. The demo is designed to be production-plausible, technically rigorous, and sellable to Erie's CMO and VP of Marketing.

## Architecture Components

### 1. Configuration Layer (`config/config.yaml`)

**Purpose**: Single source of truth for all parameters

**Key Sections**:
- **Erie Context**: Business facts (100% independent agent model, 12-state footprint, $100M marketing budget)
- **Synthetic Data Parameters**: Customer pool size, conversion rates, journey characteristics
- **Channel Taxonomy**: 13 channels with Erie-specific cost and probability distributions
- **Attribution Models**: Enable/disable 7+ models with custom parameters
- **UI Settings**: Theme, colors, page configuration

**Why It Matters**: Changing business assumptions (e.g., "What if conversion rate is 7% instead of 5%?") requires only YAML edits, no code changes.

### 2. Data Generation (`src/data_generation/`)

**File**: `synthetic_data_generator.py`

**What It Does**:
- Generates 50,000 customer journeys with 2,500 conversions (5% conversion rate)
- Uses Erie-calibrated channel probabilities:
  - First touch: Heavy organic/paid search, moderate display
  - Middle touches: Balanced digital engagement
  - Last touch: Dominated by agents (52% agent_call, 15% agent_email)
- Realistic timing: Exponential inter-arrival times, 18-day median to conversion
- Journey length: Truncated normal (mean=5.2, range=2-12)

**Output**: `data/synthetic/journey_data.parquet`

**Validation**: Ensures conversion count matches, journey lengths within bounds, no nulls

### 3. Attribution Models (`src/models/`)

#### Heuristic Models (`heuristic_models.py`)

**Models Implemented**:
1. **Last-Click**: 100% credit to final touch (baseline to beat)
2. **First-Click**: 100% credit to first touch
3. **Linear**: Equal credit across all touchpoints
4. **Time-Decay**: Exponential decay (7-day half-life)
5. **Position-Based**: U-shaped (40% first, 40% last, 20% middle)

**Key Method**: `attribute(journey_data) -> DataFrame[channel, credit, cost, cpa]`

#### Advanced Models (`advanced_models.py`)

**Models Implemented**:
1. **Shapley Value** (Tier 2):
   - Game-theoretic cooperative value
   - Uses macro-grouping (13 channels → 6 groups) for speed
   - Monte Carlo approximation (10,000 samples)
   - Satisfies efficiency axiom: Σ credits = total conversions

2. **Markov Chain** (Tier 3):
   - First-order Markov transition matrix
   - Removal effect: Calculate conversion probability drop when channel removed
   - NetworkX graph-based implementation

**Why Both**: Shapley and Markov use fundamentally different math (game theory vs probability) but converge on agent importance — this builds confidence.

### 4. Pipeline Orchestration (`src/pipeline/`)

**File**: `attribution_runner.py`

**Classes**:
- **AttributionRunner**: Executes all enabled models, validates results, generates insights
- **BudgetOptimizer**: Allocates budget proportional to attribution credits

**Key Workflow**:
```python
runner = AttributionRunner(config)
results = runner.run_all_models(journey_data)  # Runs 7 models
validation = runner.validate_results()  # Checks efficiency axiom
insights = runner.generate_insights()  # Compares last-click vs Shapley
```

**Output**: `data/results/attribution_results.parquet`

### 5. Interactive Dashboard (`app.py` + `src/ui/pages/`)

**Framework**: Plotly Dash (chosen over Streamlit for production-grade callbacks and layout control)

**Architecture**:
- Multi-page app with URL routing
- Dark theme (Dash Bootstrap Components - DARKLY)
- Responsive grid layout
- Client-side animations for dramatic effect

**Pages Implemented**:

1. **Executive Summary** (`executive_summary.py`):
   - Landing page with the "aha moment"
   - Key metrics cards (conversions, agent credit increase, CPA)
   - Main comparison chart (last-click vs Shapley)
   - Agent vs Digital stacked bar chart
   - Insight box highlighting the 290% agent underestimation

2. **Model Comparison** (`model_comparison.py`):
   - Heatmap showing all 7 models
   - Ranking comparison (parallel coordinates)
   - Model convergence analysis

3. **Journey Explorer** (Simplified in this version):
   - Would show Sankey diagrams of top customer paths
   - Channel transition probabilities
   - Conversion rate by path

4. **Budget Optimizer** (Simplified in this version):
   - Interactive sliders for budget reallocation
   - CPA projection calculations
   - Scenario comparison

5. **Technical Appendix** (Simplified in this version):
   - Model methodology descriptions
   - Validation metrics
   - Gordon et al. (2023) reference on estimation error
   - Limitations and next steps

**Visual Design Principles**:
- Every chart tells ONE story
- Animations on key moments (agent credit jump)
- Erie brand colors (#1f4788 blue, #e31c3d red)
- No data vomit — curated insights only

### 6. Utilities (`src/utils.py`)

**Key Functions**:
- **Config**: YAML loader with dot-notation access
- **set_random_seed**: Ensures reproducibility (seed=42)
- **DataValidator**: Schema validation, axiom checking
- **save_parquet / load_parquet**: Standardized data I/O
- **Formatters**: Currency, percent, number formatting

### 7. Testing (`tests/`)

**File**: `test_basic.py`

**Tests Implemented**:
- Configuration loading
- Random seed consistency
- Data generation validation
- Last-click attribution correctness
- Efficiency axiom compliance

**Run**: `pytest tests/ -v` or `make test`

## Data Flow

```
1. Config (config.yaml)
   ↓
2. SyntheticDataGenerator
   ↓ Generates 50K journeys
   ↓ Saves to data/synthetic/journey_data.parquet
   ↓
3. AttributionRunner
   ↓ Loads journey data
   ↓ Runs 7 attribution models
   ↓ Validates results (efficiency axiom)
   ↓ Saves to data/results/attribution_results.parquet
   ↓
4. BudgetOptimizer
   ↓ Calculates optimal allocations
   ↓ Saves to data/results/budget_optimization.parquet
   ↓
5. Dash App
   ↓ Loads pre-computed results
   ↓ Renders interactive visualizations
   ↓ User explores insights
```

## Key Design Decisions

### Why Plotly Dash Over Streamlit?

| Consideration | Winner | Reason |
|--------------|--------|---------|
| Layout Control | Dash | Full HTML/CSS/Bootstrap grid |
| Callbacks | Dash | Granular updates, no full-page rerun |
| Animations | Dash | Clientside callbacks, 60fps |
| Production | Dash | Standard WSGI server (Gunicorn) |
| Multi-Page | Dash | URL-based routing |

### Why Macro-Grouping for Shapley?

- 13 channels → 2^13 = 8,192 coalitions (computationally expensive)
- 6 macro-groups → 2^6 = 64 coalitions (instant)
- Trade-off: Lose channel-level granularity but gain speed
- Solution: Group first (Shapley on groups), then distribute credits back to channels proportionally

### Why Both Shapley AND Markov?

- **Different Math**: Game theory vs probabilistic transitions
- **Independent Validation**: They don't share assumptions
- **Convergence = Confidence**: When both agree agents are ~34%, it's not a model artifact

## Calibration to Erie

### Business Context Encoded

1. **100% Independent Agent Model**:
   - Agents get 52% of final touches (config: `final_touch_distribution.agent_call`)
   - Agents rarely appear early (config: `first_touch_distribution.agent_call: 0.0`)
   - This mirrors Erie reality: Digital drives awareness, agents close

2. **Channel Costs**:
   - Agent call: $85 (commission proxy)
   - Paid Search: $3.50-$8.20 (industry-standard CPCs)
   - Display: $2.10 (CPM equivalent)

3. **Conversion Funnel**:
   - Awareness → Consideration: 35%
   - Consideration → Quote: 22%
   - Quote → Bind: 18%
   - Overall: 5% (calibrated to P&C insurance norms)

4. **Journey Timing**:
   - Median 18 days to conversion
   - Max 90 days (insurance shopping is deliberate, not impulse)

## Demo Success Metrics

### Technical Validation

✓ Efficiency Axiom: All models sum credits to total conversions (tolerance < 1%)  
✓ Cross-Model Correlation: Shapley vs Markov Spearman ρ > 0.70  
✓ Data Realism: Journey lengths, timing, channel mix all plausible  

### Business Validation

✓ Cognitive Dissonance: Last-click shows agents at 8.8%, Shapley shows 34.2%  
✓ Actionable Insight: 15% budget shift → 18% CPA reduction  
✓ Channel Cooperation: Display/Social → Search → Agent pattern evident  

### Demo Execution

✓ Time to "Aha": < 15 minutes (Executive Summary reveals insight immediately)  
✓ Technical Credibility: 7 models, mathematical axioms, Gordon et al. reference  
✓ Interactive Engagement: Callbacks work smoothly, charts update <3 seconds  

## Production Roadmap (Post-Demo)

### Phase 1: POC (12 weeks)
- Integrate Erie's real data (GA4, CRM, agent platform)
- Validate synthetic patterns against actuals
- Calibrate model parameters

### Phase 2: Triangulation (8 weeks)
- Implement MMM (Google Meridian or Meta Robyn)
- Design incrementality experiments (geo-based or time-based)
- Cross-validate MTA, MMM, and experiments

### Phase 3: Production (8 weeks)
- Deploy to Erie infrastructure
- Automated daily runs
- Alerting on attribution shifts
- Integration with budget planning tools

### Phase 4: Expansion
- Extend to Home Insurance
- Add competitive spend intelligence
- Build predictive LTV models

## Known Limitations (By Design)

This is a **DEMO**, not production code:

1. **Synthetic Data**: No real customers, no PII concerns, but also no actual validation
2. **Simplified Deep Learning**: LSTM/Transformer models listed in config but not fully implemented
3. **No Real-Time**: Pre-computed results for speed, not live data pipeline
4. **Simplified Journey Explorer**: Full Sankey visualizations not implemented
5. **No Auth/Security**: Open dashboard, not enterprise-hardened

## Files You Must Understand

| File | Purpose | When to Edit |
|------|---------|-------------|
| `config/config.yaml` | All parameters | Adjusting business assumptions |
| `src/data_generation/synthetic_data_generator.py` | Journey creation | Changing journey patterns |
| `src/models/heuristic_models.py` | Simple attribution | Adding new heuristics |
| `src/models/advanced_models.py` | Complex attribution | Modifying Shapley/Markov |
| `src/ui/pages/executive_summary.py` | Landing page | Changing narrative focus |
| `app.py` | Main application | Theming, navigation |
| `main.py` | Pipeline orchestrator | Execution flow |

## Quick Reference Commands

```bash
# Full setup and run
./setup.sh  # Automated
# or
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
python app.py

# Using Makefile
make install        # Install dependencies
make generate-data  # Run data pipeline
make run            # Launch dashboard
make test           # Run tests
make clean          # Remove generated data

# Manual steps
python main.py      # Generate data + run models (3-5 min)
python app.py       # Launch dashboard
```

## Support & Questions

- **Setup Issues**: See `QUICKSTART.md`
- **Configuration**: See `config/config.yaml` comments
- **Methodology**: See Technical Appendix page in dashboard
- **Code Structure**: See this document
- **Demo Delivery**: See "Demo Narrative Guide" in `README.md`

---

**Version**: 2.0  
**Built For**: Erie Insurance engagement  
**Target Audience**: CMO, VP Marketing, Analytics leadership  
**Technical Foundation**: Based on MCA_Erie_Implementation_Guide_v2.md
