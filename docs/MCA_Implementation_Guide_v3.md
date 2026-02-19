# Erie Insurance Multi-Channel Attribution Demo
## Complete Implementation Guide — LLM Code-Generation Reference

**Version:** 3.0 | **Date:** February 19, 2026  
**Target Consumer:** LLM code generation (Claude Sonnet 4.6) + Development Team  
**Scope:** Full codebase specification — backend computation through frontend UI/UX to deployment  
**Client:** Erie Insurance — 100% independent agent, 12-state regional P&C carrier, personal auto insurance  

### Version 3.0 Changes from v2.0
- **Tech stack finalized:** Plotly Dash + Dash Bootstrap Components on light theme, deployed to Render.com
- **Model arsenal scoped:** 15 models (5 baselines + 3 Shapley + 3 Markov + 4 Constrained Optimization). Deep learning tier removed — no PyTorch dependency
- **Design language:** Erie brand blue (#00447C) on white/light gray. Client-branded, not generic dark theme
- **Budget optimizer fully specified:** Complete MILP formulation with response curves, constraints, dual variables, and scenario engine
- **Deployment target:** Shareable URL via Render.com — stakeholders can explore without installation
- **Synthetic data engine refined:** Four-layer behavioral simulation with decision-state model (not transition matrix journey generation)
- **Identity resolution added:** Three-tier matching architecture with fragmentation scenarios
- **Configuration unified:** Single YAML config (`erie_mca_synth_config.yaml`) drives entire pipeline

### How to Use This Document
This document is the **single-source specification** for generating the complete project codebase. An LLM should:
1. Read this document end-to-end before generating any code
2. Follow the module architecture exactly — file paths, class names, function signatures
3. Read from the referenced `erie_mca_synth_config.yaml` for all parameter values (never hardcode)
4. Ensure every UI metric traces to a computed DataFrame column (zero hardcoded numbers in UI)
5. Generate code module-by-module in the order specified in Section 13 (Build Order)

---

## Table of Contents

1. [Solution Objective & Success Criteria](#1-solution-objective--success-criteria)
2. [Tech Stack & Dependencies](#2-tech-stack--dependencies)
3. [Repository Architecture](#3-repository-architecture)
4. [Configuration System](#4-configuration-system)
5. [Data Contracts & Parquet Schemas](#5-data-contracts--parquet-schemas)
6. [Synthetic Data Engine](#6-synthetic-data-engine)
7. [Identity Resolution & Journey Assembly](#7-identity-resolution--journey-assembly)
8. [Attribution Model Arsenal](#8-attribution-model-arsenal)
9. [Model Comparison Framework](#9-model-comparison-framework)
10. [Budget Optimization Layer](#10-budget-optimization-layer)
11. [UI/UX Specification — Plotly Dash Application](#11-uiux-specification--plotly-dash-application)
12. [Deployment & Packaging](#12-deployment--packaging)
13. [Build Order & Validation Checkpoints](#13-build-order--validation-checkpoints)
14. [Appendices](#14-appendices)

---

## 1. Solution Objective & Success Criteria

### 1.1 What This Demo Must Accomplish

This is a **deployed, shareable capability demo** — not a notebook or local prototype. It takes Erie's CMO and VP of Marketing through a narrative proving that last-click attribution (GA4) systematically misallocates marketing spend by under-crediting agents and upper-funnel channels. The demo must:

1. **Create cognitive dissonance**: Show last-click attribution where Paid Search gets ~38% credit and agents get ~3%. Then show three independent attribution models where agents jump to ~35% and Paid Search drops to ~11%.
2. **Explain WHY via identity resolution**: One customer (Jennifer Morrison) exists as 8 records across 8 systems. Stitching them together changes the story completely.
3. **Demonstrate channel cooperation**: Via Markov transition heatmap, show that "Display/Social → Search → Agent Locator → Agent Call → Bind" is the highest-converting sequence.
4. **Provide convergence-weighted budget recommendations**: Three models agree on direction → act with confidence. Models disagree → flag for incrementality testing.
5. **Quantify infrastructure ROI**: Call tracking at $50K/year enables $280K in better allocation + ~340 additional binds. That's 9.6× ROI.
6. **Close with measurement maturity roadmap**: MTA (now) → MMM (6 months) → Incrementality (12 months).

### 1.2 Success Criteria

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| Shapley axiom compliance | Efficiency (credits = total conversions) | Exact (tolerance < 0.01%) |
| Cross-model convergence | Spearman ρ between 3 primary models | > 0.70 |
| Agent insight delivery | Agent credit shift from last-click to model-based | ≥ +30 percentage points |
| Budget optimization | Projected cost-per-bind improvement | ≥ 14% across all 3 models |
| Data realism | Funnel rates, journey lengths, channel mix match Erie calibration | All within YAML config tolerances |
| UI responsiveness | Page load and callback response time | < 3 seconds on Render Starter |
| Data binding integrity | Every numeric value in UI traces to computed DataFrame | 0 hardcoded metrics |
| Deployment availability | Shareable URL accessible without installation | 100% uptime (Render always-on) |

### 1.3 Key Design Principles

- **Insight-first, evidence on demand.** Every page leads with a business finding in one sentence. Technical detail is one click away.
- **Erie-specific, not generic.** Channel names, funnel rates, agent involvement — all calibrated to Erie's 100% independent agent model.
- **Three-model convergence > single-model precision.** No single model is trustworthy alone. The demo's power is showing where three independent frameworks agree.
- **Zero hardcoded UI metrics.** Every number on screen reads from a Parquet file produced by the computation pipeline. Change the config → regenerate data → UI automatically reflects new numbers.
- **Pre-computed for reliability.** All attribution models run as a batch step. The Dash app reads results — zero live computation during stakeholder exploration.

---

## 2. Tech Stack & Dependencies

### 2.1 Stack Decision: Plotly Dash + Render.com

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| **UI Framework** | Plotly Dash | ≥ 2.18.x | Multi-page apps, granular callbacks, production WSGI server |
| **UI Components** | Dash Bootstrap Components (DBC) | ≥ 1.6.x | Grid system, cards, modals, tabs — professional layout |
| **Visualization** | Plotly | 5.x | Interactive charts, Sankey diagrams, animated transitions |
| **Data Processing** | Pandas + NumPy | Pandas 2.x, NumPy 1.26+ | Core data manipulation |
| **Shapley Engine** | Custom Python | N/A | Zhao et al. (2018) simplified computation — no external library needed |
| **Markov Chain** | Custom Python + NetworkX | NetworkX 3.x | Graph construction, absorbing chain matrix operations |
| **Constrained Optimization** | SciPy + CVXPY | Latest stable | SLSQP solver + cleaner constraint syntax, dual variable extraction |
| **Budget Optimization** | PuLP + CBC | Latest stable | MILP via piecewise linearization |
| **Statistical Testing** | SciPy.stats | Latest stable | Spearman correlation, bootstrap CIs, JSD |
| **Data Storage** | Parquet via PyArrow | PyArrow 14+ | Fast columnar read for pre-computed results |
| **Configuration** | PyYAML + Pydantic v2 | Latest stable | YAML config loading with schema validation |
| **Logging** | Loguru | Latest stable | Structured logging |
| **Testing** | pytest | Latest stable | Unit tests + data contract validation |
| **Deployment** | Gunicorn + Docker + Render.com | Latest stable | Production WSGI server, containerized deployment |

### 2.2 `requirements.txt`

```
# Core UI
dash>=2.18.0
dash-bootstrap-components>=1.6.0
plotly>=5.18.0

# Data processing
pandas>=2.0.0
numpy>=1.26.0
pyarrow>=14.0.0

# Attribution engine
networkx>=3.2.0
scipy>=1.12.0
cvxpy>=1.4.0

# Budget optimization
pulp>=2.7.0

# Configuration & validation
pydantic>=2.5.0
pyyaml>=6.0.0

# Utilities
tqdm>=4.66.0
loguru>=0.7.0

# Testing
pytest>=7.4.0

# Production deployment
gunicorn>=21.2.0
```

**Deliberately excluded:**
- PyTorch / deep learning (unnecessary model complexity for demo scope)
- DuckDB (Parquet reads via PyArrow are sufficient)
- Streamlit (replaced by Dash)
- shapiq / marketing-attribution-models (custom implementations are more controllable)
- matplotlib / seaborn (Plotly handles all visualization)

### 2.3 Python Version

Python 3.11+. Required for type hint features used in Pydantic v2 models and improved performance.

---

## 3. Repository Architecture

### 3.1 Complete File Tree

```
erie-mca-demo/
├── Dockerfile
├── render.yaml                        # Render.com deployment config
├── requirements.txt
├── README.md
├── .dockerignore
├── .gitignore
│
├── config/
│   └── erie_mca_synth_config.yaml     # Master configuration (THE single source of truth)
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/                        # Configuration loading & validation
│   │   ├── __init__.py
│   │   ├── loader.py                  # YAML → Pydantic model
│   │   └── schema.py                  # Pydantic v2 models for all config sections
│   │
│   ├── generators/                    # Synthetic data generation (Step 4)
│   │   ├── __init__.py
│   │   ├── population.py              # Layer 1: Prospect demographics & profiles
│   │   ├── behavioral_sim.py          # Layer 2: Decision-state simulation engine
│   │   ├── system_records.py          # Layer 3: Source-system record generation
│   │   └── dirty_data.py              # Layer 4: Data quality issue injection
│   │
│   ├── resolution/                    # Identity resolution & journey assembly (Step 5)
│   │   ├── __init__.py
│   │   ├── identity_resolver.py       # Three-tier identity matching
│   │   ├── journey_assembler.py       # Touchpoint ordering & deduplication
│   │   └── resolution_metrics.py      # Resolution quality reporting
│   │
│   ├── models/                        # Attribution models (Step 6)
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract base class + AttributionResult dataclass
│   │   ├── baselines.py               # 5 heuristic models (last-click, first-touch, etc.)
│   │   ├── shapley.py                 # 3 Shapley variants (standard, time-weighted, CASV)
│   │   ├── markov.py                  # 3 Markov variants (1st, 2nd, 3rd order)
│   │   └── constrained_opt.py         # 4 constrained optimization variants
│   │
│   ├── comparison/                    # Model comparison framework (Step 6 evaluation)
│   │   ├── __init__.py
│   │   ├── head_to_head.py            # Spearman, JSD, axiom audit, sensitivity
│   │   ├── convergence_map.py         # Per-channel confidence classification
│   │   └── dual_funnel.py             # Quote vs. bind attribution comparison
│   │
│   ├── optimization/                  # Budget optimization (Step 7)
│   │   ├── __init__.py
│   │   ├── response_curves.py         # Saturating exponential calibration
│   │   ├── budget_optimizer.py        # MILP formulation + solver
│   │   ├── scenario_engine.py         # Pre-built + custom scenario runner
│   │   └── shadow_prices.py           # Dual variable extraction & interpretation
│   │
│   ├── metrics/                       # Centralized metric computation (UI data contract)
│   │   ├── __init__.py
│   │   ├── executive_metrics.py       # Page 1 KPI cards
│   │   ├── identity_metrics.py        # Page 2 resolution stats
│   │   ├── comparison_metrics.py      # Page 3 model comparison
│   │   ├── channel_metrics.py         # Page 4 channel deep-dive
│   │   ├── funnel_metrics.py          # Page 5 quote vs bind
│   │   ├── budget_metrics.py          # Page 6 optimization results
│   │   ├── scenario_metrics.py        # Page 7 what-if engine
│   │   └── roadmap_metrics.py         # Page 8 maturity assessment
│   │
│   └── utils/
│       ├── __init__.py
│       ├── data_io.py                 # Parquet read/write helpers
│       ├── formatters.py              # Number formatting (currency, pct, pp)
│       └── logging_setup.py           # Loguru configuration
│
├── app/                               # Plotly Dash application
│   ├── app.py                         # Dash entry point + layout
│   ├── theme.py                       # Color palette, Plotly template, CSS constants
│   ├── data_store.py                  # Central data loading from Parquet files
│   │
│   ├── assets/
│   │   ├── custom.css                 # Global CSS overrides
│   │   └── favicon.ico                # Erie-branded favicon
│   │
│   ├── components/                    # Reusable Dash components
│   │   ├── __init__.py
│   │   ├── navbar.py                  # Top navigation bar
│   │   ├── metric_card.py             # KPI card component
│   │   ├── insight_callout.py         # Styled insight/finding box
│   │   ├── model_selector.py          # Model toggle pills
│   │   ├── attribution_bars.py        # Side-by-side attribution comparison chart
│   │   ├── channel_heatmap.py         # Markov transition heatmap
│   │   ├── budget_sliders.py          # Budget reallocation interface
│   │   ├── convergence_table.py       # Confidence zone table
│   │   └── waterfall_chart.py         # Waterfall/cascade chart
│   │
│   └── pages/                         # Multi-page Dash routing
│       ├── executive_summary.py       # Page 1: Landing page
│       ├── identity_resolution.py     # Page 2: Fragmentation story
│       ├── model_comparison.py        # Page 3: Three-model head-to-head
│       ├── channel_deep_dive.py       # Page 4: Agent focus (dynamic for any channel)
│       ├── dual_funnel.py             # Page 5: Quote vs. bind shift
│       ├── budget_optimization.py     # Page 6: Recommendations + simulator
│       ├── scenario_explorer.py       # Page 7: What-if engine
│       ├── measurement_roadmap.py     # Page 8: Maturity stages
│       └── technical_appendix.py      # Appendix: Math formulations, diagnostics
│
├── scripts/
│   ├── generate_data.py               # Run synthetic data generation
│   ├── run_attribution.py             # Run all models + comparison + optimization
│   ├── run_full_pipeline.py           # End-to-end: generate → resolve → attribute → optimize
│   └── validate_data_contracts.py     # Verify all Parquet schemas
│
├── tests/
│   ├── test_config.py                 # Config loading and validation
│   ├── test_generators.py             # Synthetic data parameter conformance
│   ├── test_attribution.py            # Shapley axioms, Markov properties, OR constraints
│   ├── test_data_contracts.py         # Schema validation for all Parquet files
│   └── test_optimization.py           # Budget constraint satisfaction
│
├── data/                              # Generated data (gitignored, created by pipeline)
│   ├── source_systems/                # Raw fragmented system extracts
│   │   ├── ga4_sessions.parquet
│   │   ├── google_ads_clicks.parquet
│   │   ├── crm_contacts.parquet
│   │   ├── ams_records.parquet
│   │   ├── pas_policies.parquet
│   │   ├── esp_events.parquet
│   │   ├── call_tracking.parquet
│   │   └── direct_mail.parquet
│   │
│   ├── resolved/                      # After identity resolution
│   │   ├── identity_graph.parquet
│   │   ├── unified_journeys.parquet
│   │   └── resolution_report.parquet
│   │
│   ├── attribution/                   # Model outputs (standard contract)
│   │   ├── channel_credits.parquet    # All 15 models × 2 conversion events
│   │   ├── journey_credits.parquet    # Per-journey allocations
│   │   └── model_diagnostics.parquet  # Runtime, convergence, coverage
│   │
│   ├── comparison/                    # Head-to-head analysis
│   │   ├── pairwise_metrics.parquet   # Spearman, JSD, sensitivity
│   │   ├── convergence_map.parquet    # Per-channel confidence zones
│   │   └── dual_funnel.parquet        # Quote vs. bind comparison
│   │
│   ├── optimization/                  # Budget optimizer results
│   │   ├── budget_recommendations.parquet  # Per-model optimal allocations
│   │   ├── response_curves.parquet         # Channel saturation parameters
│   │   ├── shadow_prices.parquet           # Dual variables per constraint
│   │   └── scenario_results.parquet        # 7 pre-built scenarios
│   │
│   └── ground_truth/                  # "God view" for validation (never shown in demo)
│       └── true_journeys.parquet
│
└── output/                            # Exportable artifacts
    └── validation_report.html         # Post-pipeline validation dashboard
```

### 3.2 Critical Architecture Rules

1. **Data flows one direction:** `config/ → generators/ → resolution/ → models/ → comparison/ → optimization/ → metrics/ → app/pages/`
2. **The `metrics/` layer is the UI's ONLY data source.** Pages never import from `models/` or `optimization/` directly. They read from `metrics/` functions which read from Parquet files.
3. **No circular imports.** Each module depends only on modules earlier in the pipeline (plus `config/` and `utils/`).
4. **Parquet is the interface.** Modules communicate via Parquet files with schemas defined in Section 5. This allows independent testing and re-running of any pipeline stage.
5. **The YAML config is the single source of truth.** Every parameter (funnel rates, channel names, model hyperparameters, budget constraints) comes from the config. Nothing is hardcoded in Python.


---

## 4. Configuration System

### 4.1 Master Configuration File

All parameters are defined in `config/erie_mca_synth_config.yaml` (777 lines, 10 sections). The config is loaded once at pipeline start and passed to all modules.

**The YAML config file (`erie_mca_synth_config.yaml`) is delivered separately alongside this guide.** It contains educated defaults for all parameters with valid ranges and rationale documented inline.

### 4.2 Pydantic Schema (`src/config/schema.py`)

Every config section maps to a Pydantic v2 `BaseModel`. This provides type validation, default values, and clear error messages when parameters are out of range.

```python
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
from enum import Enum

class SimulationConfig(BaseModel):
    random_seed: int = 42
    output_format: str = "parquet"
    output_directory: str = "./data/source_systems/"
    generate_unified_truth: bool = True
    log_level: str = "INFO"

class PopulationConfig(BaseModel):
    total_prospects: int = Field(25000, ge=1000, le=100000)
    geography: Dict[str, float]  # State code → weight (must sum to 1.0)
    age_bands: Dict[str, float]  # Band → weight
    digital_sophistication: Dict[str, float]  # high/medium/low → weight
    shopping_triggers: Dict[str, float]  # Trigger → weight
    prior_awareness: Dict  # default + state_overrides

    @field_validator('geography')
    @classmethod
    def geography_sums_to_one(cls, v):
        assert abs(sum(v.values()) - 1.0) < 0.01, f"Geography weights sum to {sum(v.values())}"
        return v

class FunnelConfig(BaseModel):
    prospect_to_quote_rate: float = Field(0.35, ge=0.15, le=0.50)
    quote_to_complete_rate: float = Field(0.70, ge=0.50, le=0.85)
    complete_to_bind_rate: float = Field(0.58, ge=0.35, le=0.70)
    agent_involvement_rate: float = Field(0.87, ge=0.70, le=0.98)
    digital_only_bind_rate: float = Field(0.13, ge=0.02, le=0.30)
    multi_agent_touch_rate: float = Field(0.35, ge=0.15, le=0.50)
    agent_only_bind_fraction: float = Field(0.29, ge=0.10, le=0.45)

class ChannelDef(BaseModel):
    id: str
    name: str
    type: str
    data_source: str | List[str]
    user_level_trackable: bool | str
    spend_trackable: bool | str
    avg_cpm: Optional[float] = None
    avg_cpc: Optional[float] = None
    avg_cost_per_send: Optional[float] = None
    avg_cost_per_piece: Optional[float] = None
    notes: Optional[str] = None

class JourneyConfig(BaseModel):
    converters: Dict  # length_mean, length_std, length_min, length_max
    non_converters: Dict
    duration: Dict  # converters_mean_days, etc.
    state_transitions: Dict  # awareness/consideration/intent → [stay, progress, exit]
    transition_boosters: Dict  # channel_id → boost amount

class IdentityConfig(BaseModel):
    fragmentation_distribution: Dict[str, float]  # Must sum to 1.0
    system_match_rates: Dict[str, float]
    cookie: Dict
    call_tracking: Dict

class DataQualityConfig(BaseModel):
    enabled: bool = True
    issues: Dict  # Issue type → {rate, description, ...}

class ShapleyConfig(BaseModel):
    coalition_value_function: str = "inclusive"
    min_coalition_observations: int = 30
    sparse_coalition_handling: str = "adjacent_imputation"
    time_weighting: Dict
    casv_variant: Dict

class MarkovConfig(BaseModel):
    primary_order: int = Field(2, ge=1, le=3)
    comparison_orders: List[int] = [1, 3]
    smoothing: str = "dirichlet"
    smoothing_alpha: float = 0.077
    min_path_frequency: int = 5

class ConstrainedOptConfig(BaseModel):
    objective: str = "logistic_likelihood"
    regularization: Dict
    feature_encodings: List[str]
    primary_encoding: str = "recency_weighted"
    constraints: Dict
    solver: str = "scipy_slsqp"
    report_dual_variables: bool = True

class BudgetOptimizerConfig(BaseModel):
    total_budget: float = 5000000
    current_allocation: Dict[str, float]
    channel_saturation_factors: Dict[str, float]
    channel_minimum_floors: Dict[str, float]
    constraints: Dict
    response_curve_type: str = "saturating_exponential"
    piecewise_segments: int = 10
    solver: str = "pulp_cbc"

class ComparisonConfig(BaseModel):
    metrics: List[str]
    convergence_threshold_rho: float = 0.70
    sensitivity_perturbation_pct: float = 0.10
    n_sensitivity_runs: int = 20

class ScenarioConfig(BaseModel):
    description: str
    overrides: Dict

class MasterConfig(BaseModel):
    """Top-level configuration — the complete parameter space."""
    simulation: SimulationConfig
    population: PopulationConfig
    funnel: FunnelConfig
    channels: Dict  # Contains 'taxonomy' list of ChannelDef
    state_interaction_matrix: Dict[str, List[float]]  # channel_id → [awareness, consideration, intent, action]
    journey: JourneyConfig
    identity: IdentityConfig
    data_quality: DataQualityConfig
    attribution: Dict  # Contains shapley, markov, constrained_optimization, budget_optimizer, comparison
    scenarios: Dict[str, ScenarioConfig]
```

### 4.3 Config Loader (`src/config/loader.py`)

```python
import yaml
from pathlib import Path
from .schema import MasterConfig

def load_config(config_path: str = "config/erie_mca_synth_config.yaml") -> MasterConfig:
    """Load and validate the master configuration."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return MasterConfig(**raw)

def load_config_with_scenario(config_path: str, scenario_name: str) -> MasterConfig:
    """Load config with scenario overrides applied."""
    config = load_config(config_path)
    if scenario_name not in config.scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    scenario = config.scenarios[scenario_name]
    # Apply dot-notation overrides (e.g., "funnel.agent_involvement_rate": 0.65)
    for key, value in scenario.overrides.items():
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part) if hasattr(obj, part) else obj[part]
        if isinstance(obj, dict):
            obj[parts[-1]] = value
        else:
            setattr(obj, parts[-1], value)
    return config

# Global config instance (loaded once at pipeline start)
CONFIG: MasterConfig = None

def init_config(path: str = "config/erie_mca_synth_config.yaml"):
    global CONFIG
    CONFIG = load_config(path)
    return CONFIG
```

---

## 5. Data Contracts & Parquet Schemas

### 5.1 Design Rule

**The UI layer may ONLY read from these defined schemas.** No magic numbers, no inline computations, no hardcoded percentages. Every `metric_card`, every chart trace, every insight callout must reference a column in one of these DataFrames.

Data flows: `generators/ → Parquet → resolution/ → Parquet → models/ → Parquet → comparison/ → Parquet → optimization/ → Parquet → metrics/ → Dash callbacks`

### 5.2 Source System Schemas (output of generators/)

**`ga4_sessions.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| ga4_client_id | str | Cookie-based identifier (resets on churn) |
| session_id | str | Session UUID |
| session_timestamp | datetime | Session start time (UTC) |
| landing_page | str | First page of session |
| source | str | Traffic source (google, facebook, direct, etc.) |
| medium | str | Traffic medium (cpc, organic, referral, etc.) |
| campaign | str | Campaign name |
| device_type | str | desktop / mobile / tablet |
| state | str | US state from geo-IP |
| pages_viewed | int | Pages in session |
| session_duration_sec | float | Session duration |
| events | List[str] | Key events (quote_start, agent_locator_click, etc.) |
| gclid | str | Google Click ID (null if not from Google Ads) |
| _ground_truth_id | str | True prospect ID (for validation only — never used in resolution) |

**`crm_contacts.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| crm_id | str | CRM record UUID |
| email | str | Contact email (may be null) |
| phone | str | Contact phone (may be null) |
| first_name | str | First name |
| last_name | str | Last name |
| address | str | Street address |
| zip_code | str | ZIP code |
| state | str | State |
| lead_source | str | How this lead entered CRM |
| agent_code | str | Assigned agent (may be null for online-only) |
| created_date | datetime | Record creation date |
| quote_started | bool | Whether a quote was initiated |
| quote_completed | bool | Whether quote was completed |
| _ground_truth_id | str | True prospect ID |

**`ams_records.parquet`** (Agent Management System)

| Column | Type | Description |
|--------|------|-------------|
| ams_client_id | str | Agent's internal client ID |
| agent_code | str | Agent identifier (may differ from CRM agent_code) |
| client_name | str | Client name as entered by agent |
| client_phone | str | Client phone as entered by agent |
| interaction_date | datetime | Date of agent interaction |
| interaction_type | str | phone_call / office_visit / outbound_call |
| notes | str | Free-text agent notes |
| _ground_truth_id | str | True prospect ID |

**`pas_policies.parquet`** (Policy Administration System)

| Column | Type | Description |
|--------|------|-------------|
| policy_number | str | Policy ID |
| policyholder_name | str | Full name |
| policyholder_address | str | Address |
| policyholder_zip | str | ZIP |
| writing_agent_code | str | Agent who wrote the policy |
| effective_date | datetime | Policy effective date |
| processing_date | datetime | When policy was processed (may lag effective_date) |
| premium_amount | float | Annual premium |
| line_of_business | str | Always "auto" for this demo |
| _ground_truth_id | str | True prospect ID |

**`call_tracking.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| call_id | str | Call UUID |
| tracked_number | str | Dynamic tracking number |
| caller_phone | str | Caller's phone number |
| call_timestamp | datetime | Call start time |
| call_duration_sec | float | Call duration |
| web_session_id | str | Correlated GA4 session (if available) |
| destination_agent_code | str | Agent who received the call |
| _ground_truth_id | str | True prospect ID |

**Additional source system files** (`google_ads_clicks.parquet`, `esp_events.parquet`, `direct_mail.parquet`) follow similar patterns with system-specific fields.

### 5.3 Resolution Output Schemas

**`identity_graph.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| unified_id | str | Resolved identity UUID |
| source_system | str | ga4 / crm / ams / pas / call_tracking / esp / direct_mail / google_ads |
| source_record_id | str | Original record ID in source system |
| match_tier | int | 1 (deterministic), 2 (fuzzy), 3 (probabilistic) |
| match_confidence | float | 0.0-1.0 |
| match_rule | str | Description of match rule applied |
| _ground_truth_id | str | For validation: true prospect ID |
| _match_correct | bool | For validation: does unified_id match ground truth? |

**`unified_journeys.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| journey_id | str | Journey UUID |
| unified_id | str | Resolved prospect identity |
| touchpoints | List[Dict] | Ordered array of touchpoint structs |
| touchpoint_count | int | Number of touchpoints |
| channel_path | str | Pipe-delimited channel sequence: "display|search_brand|agent_interaction" |
| channel_set | str | Pipe-delimited unique channels (sorted, for Shapley coalition) |
| first_touch_channel | str | Channel of first interaction |
| last_touch_channel | str | Channel of last interaction before conversion |
| has_agent_touch | bool | Whether any touchpoint was agent interaction |
| agent_touch_count | int | Number of agent touchpoints |
| quote_started | bool | Whether quote-start event occurred |
| quote_completed | bool | Whether quote was completed |
| policy_bound | bool | Whether policy was bound (primary conversion) |
| conversion_type | str | "bind", "quote_only", "none" |
| journey_start | datetime | First touchpoint timestamp |
| journey_end | datetime | Last touchpoint / conversion timestamp |
| duration_days | float | Journey duration in days |
| resolution_quality | str | "fully_resolved", "partially_resolved", "singleton" |
| resolution_tiers_used | List[int] | Which tiers contributed (e.g., [1, 2]) |
| identity_confidence | float | Average match confidence across linked records |
| age_band | str | Prospect demographic |
| state | str | Prospect state |
| shopping_trigger | str | What triggered insurance shopping |
| digital_sophistication | str | high / medium / low |

**Touchpoint struct within `touchpoints` array:**

```python
{
    "touchpoint_id": str,          # UUID
    "channel_id": str,             # One of 13 channels
    "timestamp": datetime,         # Interaction time
    "interaction_type": str,       # click / impression / call / visit / email / mail
    "source_system": str,          # Which system recorded this
    "device_type": str,            # desktop / mobile / phone / offline
    "decision_state": str,         # Inferred: awareness / consideration / intent / action
    "time_to_conversion_days": float,  # Days until conversion (null for non-converters)
}
```

### 5.4 Attribution Output Schemas

**`channel_credits.parquet`** — THE central attribution results table

| Column | Type | Description |
|--------|------|-------------|
| model_id | str | Model identifier (e.g., "shapley_time_weighted", "markov_2nd", "constrained_opt_recency") |
| model_family | str | "baseline", "shapley", "markov", "constrained_opt" |
| conversion_event | str | "quote_start" or "bind" |
| channel_id | str | One of 13 channels |
| channel_name | str | Display name from config |
| total_credit | float | Total attributed conversions for this channel |
| credit_share | float | Fraction of total credit (0.0-1.0, sums to 1.0 per model+event) |
| credit_rank | int | Rank (1 = highest credit) |
| journeys_involved | int | Number of journeys this channel appeared in |
| avg_credit_per_journey | float | Average credit per involved journey |
| ci_lower | float | 95% bootstrap CI lower bound |
| ci_upper | float | 95% bootstrap CI upper bound |
| is_primary_model | bool | Whether this is one of the 3 primary narrative models |

**`journey_credits.parquet`** — Per-journey credit allocation (for drill-down)

| Column | Type | Description |
|--------|------|-------------|
| journey_id | str | Journey UUID |
| model_id | str | Model identifier |
| conversion_event | str | "quote_start" or "bind" |
| channel_credits | Dict[str, float] | channel_id → credited fraction for this journey |
| dominant_channel | str | Channel receiving most credit |
| credit_entropy | float | Shannon entropy of credit distribution (higher = more spread) |

**`model_diagnostics.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| model_id | str | Model identifier |
| conversion_event | str | "quote_start" or "bind" |
| computation_time_sec | float | Wall-clock time |
| convergence_achieved | bool | Did optimization converge? (N/A for non-iterative models) |
| data_coverage | float | Fraction of journeys usable by this model |
| effective_sample_size | int | Journeys that contributed to credit estimation |
| total_conversions | int | Total conversions attributed |
| credits_sum_check | float | Sum of all channel credits (should equal total_conversions) |
| variant_params | Dict | Model-specific parameters used |

### 5.5 Comparison Output Schemas

**`pairwise_metrics.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| model_a | str | First model |
| model_b | str | Second model |
| conversion_event | str | "quote_start" or "bind" |
| spearman_rho | float | Rank correlation of credit vectors |
| spearman_pvalue | float | P-value of Spearman test |
| jensen_shannon_div | float | JSD between normalized credit shares |
| l2_distance | float | L2 distance between credit share vectors |

**`convergence_map.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| channel_id | str | Channel |
| channel_name | str | Display name |
| conversion_event | str | "quote_start" or "bind" |
| shapley_credit | float | Time-Weighted Shapley credit share |
| markov_credit | float | Markov 2nd Order credit share |
| or_credit | float | Constrained Opt Recency credit share |
| credit_mean | float | Mean across 3 primary models |
| credit_range | float | Max - min across 3 models |
| confidence_zone | str | "HIGH" / "MODERATE" / "LOW" |
| consensus_direction | str | "INCREASE" / "DECREASE" / "HOLD" / "INVESTIGATE" |
| last_click_credit | float | Last-digital-click baseline for comparison |
| credit_shift_from_last_click | float | Mean model credit - last click credit |

**`dual_funnel.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| channel_id | str | Channel |
| channel_name | str | Display name |
| model_id | str | Model (primary models only) |
| quote_credit_share | float | Credit share for quote-start |
| bind_credit_share | float | Credit share for bind |
| credit_shift_pp | float | bind_credit - quote_credit (percentage points) |
| funnel_role | str | "quote_driver" / "bind_closer" / "consistent" / "niche" |

### 5.6 Optimization Output Schemas

**`budget_recommendations.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| channel_id | str | Channel |
| channel_name | str | Display name |
| model_id | str | Which attribution model drove this optimization |
| current_spend | float | Current budget allocation ($) |
| optimal_spend | float | Optimizer-recommended allocation ($) |
| spend_change_dollars | float | optimal - current |
| spend_change_pct | float | (optimal - current) / current |
| predicted_conversions_current | float | Conversions at current spend |
| predicted_conversions_optimal | float | Conversions at optimal spend |
| is_optimizable | bool | Whether this channel is in the optimizer (false for earned/owned) |

**`response_curves.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| channel_id | str | Channel |
| saturation_ceiling_a | float | Max achievable conversions |
| efficiency_rate_b | float | Diminishing returns rate |
| saturation_factor | float | Current closeness to ceiling |
| current_spend | float | Current spend level |
| current_conversions | float | Conversions at current spend |
| marginal_roi_at_current | float | First derivative at current spend |

**`shadow_prices.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| constraint_name | str | Human-readable constraint name |
| constraint_type | str | "total_budget" / "reallocation_cap" / "agent_floor" / "tv_minimum" / etc. |
| model_id | str | Which attribution model |
| shadow_price | float | Dual variable value |
| is_binding | bool | Whether constraint is active at optimum |
| interpretation | str | Plain-English explanation |

**`scenario_results.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| scenario_id | str | Scenario name from config |
| scenario_description | str | Human-readable description |
| channel_id | str | Channel |
| baseline_spend | float | Spend in baseline scenario |
| scenario_spend | float | Spend in this scenario |
| baseline_conversions | float | Predicted conversions at baseline |
| scenario_conversions | float | Predicted conversions in this scenario |
| total_binds_baseline | float | Total binds across all channels (baseline) |
| total_binds_scenario | float | Total binds (this scenario) |
| cost_per_bind_baseline | float | $ per bind (baseline) |
| cost_per_bind_scenario | float | $ per bind (this scenario) |


---

## 6. Synthetic Data Engine

### 6.1 Architecture Overview

The generator uses a **four-layer behavioral simulation** that produces realistic customer journeys from which attribution patterns EMERGE — they are not hardcoded.

```
Layer 1: Population Generator (population.py)
    → 25,000 prospect profiles with demographics, geography, shopping triggers
    
Layer 2: Behavioral Simulator (behavioral_sim.py)  
    → Decision-state engine: awareness → consideration → intent → action
    → Channel interactions sampled probabilistically per decision state per day
    → Conversion/dropout decisions based on accumulated journey state
    → Output: "god view" unified journeys with all touchpoints
    
Layer 3: System Record Generator (system_records.py)
    → Fragments unified journeys into source-system-specific records
    → Applies identity fragmentation (cookie churn, cross-device, agent-only)
    → Assigns system-specific IDs (GA4 client_id, CRM email, AMS agent_client_id)
    
Layer 4: Dirty Data Injector (dirty_data.py)
    → Injects realistic data quality issues per config rates
    → Duplicate records, timestamp drift, missing fields, bot traffic, etc.
```

### 6.2 Layer 1: Population Generator (`src/generators/population.py`)

```python
class ProspectProfile:
    """A synthetic insurance shopping prospect."""
    prospect_id: str          # UUID (ground truth identity)
    age_band: str             # "25-34", "35-44", etc.
    state: str                # One of Erie's 12 states + DC
    shopping_trigger: str     # "new_vehicle", "rate_shopping", "moving", etc.
    digital_sophistication: str  # "high", "medium", "low"
    prior_erie_awareness: bool   # Knows about Erie brand before journey
    
def generate_population(config: MasterConfig) -> List[ProspectProfile]:
    """
    Generate prospect profiles sampled from config distributions.
    
    Key behaviors:
    - prior_erie_awareness varies by state (PA: 75%, expansion states: 25%)
    - digital_sophistication correlates with age (younger = more digital)
    - shopping_trigger influences journey length and urgency
    """
    rng = np.random.default_rng(config.simulation.random_seed)
    prospects = []
    for i in range(config.population.total_prospects):
        state = rng.choice(list(config.population.geography.keys()),
                          p=list(config.population.geography.values()))
        age_band = rng.choice(list(config.population.age_bands.keys()),
                             p=list(config.population.age_bands.values()))
        # ... sample all attributes from config distributions
        prospects.append(ProspectProfile(...))
    return prospects
```

### 6.3 Layer 2: Behavioral Simulator (`src/generators/behavioral_sim.py`)

This is the heart of the synthetic data engine. Each prospect is simulated through a decision-state machine.

**Decision States:** `AWARENESS → CONSIDERATION → INTENT → ACTION → CONVERSION/EXIT`

**Simulation loop for each prospect:**

```python
def simulate_journey(prospect: ProspectProfile, config: MasterConfig, rng) -> Journey:
    """
    Simulate one prospect's insurance shopping journey.
    
    Algorithm:
    1. Initialize in AWARENESS state
    2. For each day until conversion, dropout, or max_duration:
       a. For each channel, sample whether interaction occurs
          (probability from state_interaction_matrix × modifiers)
       b. Record touchpoints that occurred
       c. Apply state transition: stay / progress / exit
          (boosted by channels interacted with today)
       d. If in ACTION state and agent interaction occurred,
          apply conversion logic
    3. Output complete journey with all touchpoints
    """
    state = "awareness"
    day = 0
    touchpoints = []
    last_interaction = {}  # channel_id → last day interacted (for cooldowns)
    
    # Determine journey parameters
    max_duration = sample_duration(prospect, config, rng)
    will_convert = determine_conversion(prospect, config, rng)  # Pre-determined but path-dependent
    
    while day < max_duration and state != "converted" and state != "exited":
        day += 1
        todays_touchpoints = []
        
        # Sample channel interactions for today
        for channel_id, state_probs in config.state_interaction_matrix.items():
            if channel_id == "modifiers" or channel_id == "cooldowns":
                continue
            
            # Base probability from state_interaction_matrix
            state_idx = {"awareness": 0, "consideration": 1, "intent": 2, "action": 3}[state]
            base_prob = state_probs[state_idx]
            
            # Apply modifiers from config
            prob = apply_modifiers(base_prob, channel_id, prospect, config)
            
            # Check cooldown
            if channel_id in last_interaction:
                cooldown = config.state_interaction_matrix.get("cooldowns", {}).get(channel_id, 1)
                if day - last_interaction[channel_id] < cooldown:
                    continue
            
            # Sample interaction
            if rng.random() < prob:
                touchpoint = create_touchpoint(channel_id, prospect, day, state)
                todays_touchpoints.append(touchpoint)
                last_interaction[channel_id] = day
        
        touchpoints.extend(todays_touchpoints)
        
        # State transition
        transition_probs = list(config.journey.state_transitions[state])  # [stay, progress, exit]
        
        # Apply transition boosters from today's interactions
        for tp in todays_touchpoints:
            boost = config.journey.transition_boosters.get(tp.channel_id, 0)
            transition_probs[1] = min(0.90, transition_probs[1] + boost)
            transition_probs[0] = max(0.05, transition_probs[0] - boost * 0.5)
        
        # Normalize
        total = sum(transition_probs)
        transition_probs = [p / total for p in transition_probs]
        
        outcome = rng.choice(["stay", "progress", "exit"], p=transition_probs)
        if outcome == "progress":
            state = next_state(state)  # awareness→consideration→intent→action
        elif outcome == "exit":
            state = "exited"
        
        # Conversion logic in ACTION state
        if state == "action" and will_convert:
            # Check if agent interaction occurred (required for most conversions)
            has_agent = any(tp.channel_id == "agent_interaction" for tp in todays_touchpoints)
            if has_agent or (prospect.digital_sophistication == "high" and rng.random() < 0.15):
                state = "converted"
    
    return Journey(
        prospect_id=prospect.prospect_id,
        touchpoints=touchpoints,
        converted=state == "converted",
        final_state=state,
        duration_days=day,
    )

def apply_modifiers(base_prob, channel_id, prospect, config):
    """Apply demographic, geographic, and trigger-based modifiers."""
    prob = base_prob
    
    # Digital sophistication modifier
    dig_mod = config.state_interaction_matrix["modifiers"]["digital_sophistication"]
    if is_digital_channel(channel_id):
        prob *= dig_mod[prospect.digital_sophistication]["digital"]
    elif channel_id == "agent_interaction":
        prob *= dig_mod[prospect.digital_sophistication]["agent"]
    
    # Geography modifier (affects TV/radio)
    if channel_id == "tv_radio":
        geo_mod = config.state_interaction_matrix["modifiers"]["geography"]
        market_type = classify_market(prospect.state)  # core/secondary/expansion
        prob *= geo_mod[market_type]
    
    # Shopping trigger modifier
    trigger_mod = config.state_interaction_matrix["modifiers"]["shopping_trigger"]
    if prospect.shopping_trigger in trigger_mod:
        channel_mods = trigger_mod[prospect.shopping_trigger]
        for key, mult in channel_mods.items():
            if key in channel_id or channel_id.startswith(key):
                prob *= mult
    
    return min(prob, 0.95)  # Cap at 95%
```

**Key behavioral patterns that should emerge from simulation:**

1. **Agent interaction probability spikes near conversion:** awareness 2% → action 65%
2. **Digital-to-agent bridge:** Display/Social → Search → Agent Locator → Agent Call is most common converting path
3. **Agent-touched journeys convert at ~2.8× rate:** Because agent_interaction has 0.20 transition booster
4. **Upper-funnel channels enable but don't close:** TV/Display have high awareness probability but low action probability
5. **Branded search is a capture channel, not a driver:** High in intent/action states, captures existing demand

### 6.4 Layer 3: System Record Generator (`src/generators/system_records.py`)

Takes the "god view" unified journeys and fragments them into source-system records, simulating realistic identity fragmentation.

```python
def fragment_journey(journey: Journey, config: MasterConfig, rng) -> Dict[str, List[Dict]]:
    """
    Take a unified journey and produce fragmented source-system records.
    
    Steps:
    1. Assign fragmentation type (clean_match, cookie_churn, etc.) from config distribution
    2. Generate system-specific identifiers (GA4 cookies, CRM email, AMS agent_id, etc.)
    3. Apply identity fragmentation rules (cookie expiry, cross-device gaps, missing links)
    4. Output records grouped by source system
    
    Returns: {
        "ga4_sessions": [...],     # Digital session records
        "google_ads_clicks": [...], # Ad click records
        "crm_contacts": [...],     # CRM records (if email/phone captured)
        "ams_records": [...],      # Agent interaction records
        "pas_policies": [...],     # Policy records (converters only)
        "call_tracking": [...],    # Call records (if call tracking deployed)
        "esp_events": [...],       # Email events
        "direct_mail": [...],      # Direct mail records
    }
    """
    frag_type = rng.choice(
        list(config.identity.fragmentation_distribution.keys()),
        p=list(config.identity.fragmentation_distribution.values())
    )
    
    # Generate base identifiers
    ga4_client_id = str(uuid4())[:16]
    email = generate_email(journey.prospect) if rng.random() < 0.75 else None
    phone = generate_phone(journey.prospect) if rng.random() < 0.80 else None
    
    # Apply fragmentation
    if frag_type == "cookie_churn":
        # 2-3 GA4 client_ids due to cookie expiration
        num_cookies = rng.integers(2, 4)
        # Split touchpoints across cookies by time
        ...
    elif frag_type == "agent_only_no_digital":
        # No GA4 records at all — agent records only
        ...
    elif frag_type == "cross_device_gap":
        # Mobile GA4 cookie ≠ desktop GA4 cookie
        ...
    
    # Generate records per system
    records = {}
    for touchpoint in journey.touchpoints:
        system = channel_to_system(touchpoint.channel_id)
        record = create_system_record(touchpoint, identifiers, system)
        records.setdefault(system, []).append(record)
    
    # Apply system match rates — not all systems will have records
    # E.g., CRM↔AMS link exists only 50% of the time
    apply_match_rate_gaps(records, config.identity.system_match_rates, rng)
    
    return records
```

### 6.5 Layer 4: Dirty Data Injector (`src/generators/dirty_data.py`)

```python
def inject_dirty_data(system_records: Dict, config: MasterConfig, rng) -> Dict:
    """
    Inject realistic data quality issues per config rates.
    Only runs if config.data_quality.enabled is True.
    
    Issue types (from config):
    - duplicate_crm_records (4%): Same prospect, two CRM records
    - timestamp_mismatches (2.5%): ±2hr drift between GA4 and Google Ads
    - missing_fields (6%): Null values in phone, email, agent_code
    - stale_cookie_attribution (1.5%): Shared device pollution
    - agent_code_inconsistency (5%): Different agent codes in AMS vs PAS
    - bot_fraud_traffic (2.5%): Non-human GA4 sessions
    - retroactive_pas_updates (3.5%): Bind date ≠ processing date
    - email_bounce_invalid (6%): Undelivered email events
    """
    if not config.data_quality.enabled:
        return system_records
    
    for issue_type, params in config.data_quality.issues.items():
        rate = params["rate"]
        # Apply each issue type to the appropriate system records
        ...
    
    return system_records
```

### 6.6 Validation Checks Post-Generation

After generation, automatically validate against config targets:

```python
VALIDATION_CHECKS = {
    "total_binds": (4500, 5500),           # ±10% of ~5000
    "agent_involvement_rate": (0.83, 0.91), # ±4pp of 87%
    "avg_journey_length_converters": (5.5, 8.0),
    "digital_only_bind_rate": (0.10, 0.16),
    "avg_journey_length_non_converters": (2.0, 4.5),
    "identity_resolution_achievable": (0.55, 0.75),
    "dirty_data_prevalence": (0.08, 0.15),  # % of records affected
}
```

---

## 7. Identity Resolution & Journey Assembly

### 7.1 The Problem

A single prospect (Jennifer Morrison) may exist as 8+ identifiers across 8 systems. Without resolution, GA4 sees a cookie, CRM sees an email, AMS sees an agent client ID, and PAS sees a policy number — no system has the complete picture.

### 7.2 Three-Tier Resolution Architecture (`src/resolution/identity_resolver.py`)

```python
class IdentityResolver:
    """
    Deterministic-first, probabilistic-fallback identity resolution.
    
    Architecture:
    - Tier 1 (Deterministic, 95-99% confidence): Exact matches on stable IDs
    - Tier 2 (Fuzzy Deterministic, 80-94%): Normalized name/address matching
    - Tier 3 (Probabilistic, 60-79%): Behavioral/contextual matching
    
    Output: identity_graph.parquet — maps every source record to a unified_id
    """
    
    def resolve(self, system_records: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Run full resolution pipeline.
        
        Returns identity_graph DataFrame mapping source records → unified IDs.
        """
        graph = self._build_initial_graph(system_records)
        graph = self._apply_tier1_matches(graph, system_records)
        graph = self._apply_tier2_matches(graph, system_records)
        graph = self._apply_tier3_matches(graph, system_records)
        graph = self._resolve_connected_components(graph)
        return graph
    
    # --- Tier 1: Deterministic Matches ---
    TIER1_RULES = [
        # Rule, Systems, Confidence, Description
        ("email_exact", ["crm", "esp", "meta"], 0.99, "Email exact match"),
        ("gclid_linkage", ["google_ads", "ga4"], 0.98, "GCLID session linkage"),
        ("phone_exact", ["crm", "ams", "call_tracking"], 0.97, "Phone exact match"),
        ("policy_crm_lookup", ["pas", "crm"], 0.95, "Name+address+agent match"),
        ("call_session_bridge", ["call_tracking", "ga4"], 0.95, "Session ID correlation"),
    ]
    
    # --- Tier 2: Fuzzy Deterministic ---
    TIER2_RULES = [
        ("name_address_matchback", ["direct_mail", "crm"], 0.88, "Name+address fuzzy match"),
        ("name_normalization", ["ams", "pas"], 0.85, "Name variant matching"),
        ("cookie_form_submit", ["ga4", "crm"], 0.90, "Cookie→CRM via form timestamp"),
        ("phone_partial", ["crm", "ams"], 0.82, "Last 7 digits match"),
    ]
    
    # --- Tier 3: Probabilistic ---
    TIER3_RULES = [
        ("agent_day_zip", ["ams", "ga4"], 0.75, "Same agent+day+zip"),
        ("device_fingerprint", ["ga4", "ga4"], 0.65, "Cross-device behavioral similarity"),
        ("recookied_detection", ["ga4", "ga4"], 0.70, "Old→new cookie via page history"),
        ("household_inference", ["crm", "pas"], 0.60, "Same address, different name"),
    ]
```

### 7.3 Identity Graph Construction

The resolver builds a **connected component graph**:
- Nodes = source records (ga4_session_1, crm_contact_1, ams_record_1, etc.)
- Edges = match rules with confidence scores
- Connected components = unified identities
- CRM record is the **anchor node** (richest identity fields)

After resolution, each connected component receives a `unified_id`.

### 7.4 Expected Resolution Metrics

| Metric | Expected Range |
|--------|---------------|
| Total source records | ~180,000 |
| Unique unified identities | 22,000-24,000 |
| Resolution rate (records linked) | 65-72% |
| Fully resolved journeys (3+ systems) | ~41% |
| Partially resolved (2 systems) | ~25% |
| Unresolved singletons | ~34% |
| Tier 1 matches (% of all links) | ~45% |
| Tier 2 matches | ~35% |
| Tier 3 matches | ~20% |
| False positive rate | 2-4% |
| Agent journeys recovered | ~76% of actual |

### 7.5 Journey Assembly (`src/resolution/journey_assembler.py`)

```python
class JourneyAssembler:
    """
    Assemble unified journeys from resolved identity graph + source records.
    
    Pipeline:
    1. Collect all touchpoint records per unified_id
    2. Normalize timestamps (handle timezone mismatches, processing delays)
    3. Deduplicate (same channel + same session = one touchpoint)
    4. Apply attribution window (30-day lookback from conversion)
    5. Order by timestamp → create journey sequence
    6. Annotate with channel, interaction type, inferred decision state
    7. Flag conversion events (quote-start, quote-complete, bind)
    8. Compute journey-level aggregates
    """
    
    DEDUP_RULES = {
        "ga4_multi_pageview": "collapse_to_session",     # Multiple pages → 1 session touchpoint
        "display_impression_click": "keep_click_only",   # Impression + click same day → keep click
        "email_send_open_click": "keep_highest_engagement",  # Keep click > open > send
        "agent_same_day": "keep_all_separate",           # Each agent interaction is meaningful
        "retargeting_organic_same_day": "keep_both",     # Different intent signals
    }
    
    def assemble(self, identity_graph: pd.DataFrame, 
                 system_records: Dict[str, pd.DataFrame],
                 config: MasterConfig) -> pd.DataFrame:
        """
        Produce unified_journeys.parquet.
        """
        journeys = []
        lookback_days = config.attribution.get("windows", {}).get("lookback_days", 30)
        
        for unified_id in identity_graph["unified_id"].unique():
            records = self._collect_records(unified_id, identity_graph, system_records)
            records = self._normalize_timestamps(records)
            records = self._deduplicate(records)
            records = self._apply_window(records, lookback_days)
            records = self._order_and_annotate(records)
            journey = self._build_journey(unified_id, records)
            journeys.append(journey)
        
        return pd.DataFrame(journeys)
```

### 7.6 Resolution Quality as Demo Insight

The resolution process itself generates a key demo insight — the "Before/After" comparison:

```
BEFORE resolution (GA4 last-click only):
  Brand Search: 38%  |  Agents: 3%  |  TV: 0%

AFTER resolution (same data, stitched identities):
  Brand Search: 11%  |  Agents: 35%  |  TV: 12%
```

The `resolution_metrics.py` module computes these before/after metrics and writes them to a comparison table consumed by Page 2 (Identity Resolution) of the Dash app.


---

## 8. Attribution Model Arsenal

### 8.1 Standard Output Contract

Every model — regardless of methodology — MUST produce results conforming to the `channel_credits.parquet` and `journey_credits.parquet` schemas from Section 5.4. This is enforced by the base class.

```python
# src/models/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

@dataclass
class AttributionResult:
    """Standard output for all attribution models."""
    model_id: str                          # e.g., "shapley_time_weighted"
    model_family: str                      # "baseline", "shapley", "markov", "constrained_opt"
    conversion_event: str                  # "quote_start" or "bind"
    channel_credits: Dict[str, float]      # channel_id → total credited conversions
    credit_shares: Dict[str, float]        # channel_id → fraction (sums to 1.0)
    credit_ranks: Dict[str, int]           # channel_id → rank (1=highest)
    journey_credits: pd.DataFrame          # journey_id × channel_id credit matrix
    total_conversions: int
    ci_lower: Dict[str, float] = field(default_factory=dict)  # Bootstrap 95% CI
    ci_upper: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict = field(default_factory=dict)
    
    def to_channel_credits_df(self) -> pd.DataFrame:
        """Convert to channel_credits.parquet schema."""
        rows = []
        for ch_id in self.channel_credits:
            rows.append({
                "model_id": self.model_id,
                "model_family": self.model_family,
                "conversion_event": self.conversion_event,
                "channel_id": ch_id,
                "total_credit": self.channel_credits[ch_id],
                "credit_share": self.credit_shares[ch_id],
                "credit_rank": self.credit_ranks[ch_id],
                "ci_lower": self.ci_lower.get(ch_id, np.nan),
                "ci_upper": self.ci_upper.get(ch_id, np.nan),
                "is_primary_model": self.model_id in PRIMARY_MODELS,
            })
        return pd.DataFrame(rows)

# The 3 primary models used in the main narrative
PRIMARY_MODELS = {"shapley_time_weighted", "markov_2nd_order", "constrained_opt_recency"}

class BaseAttributionModel(ABC):
    """Abstract base class for all attribution models."""
    
    def __init__(self, model_id: str, model_family: str, channel_ids: List[str]):
        self.model_id = model_id
        self.model_family = model_family
        self.channel_ids = channel_ids
    
    @abstractmethod
    def fit(self, journeys: pd.DataFrame) -> None:
        """Learn model parameters from journey data."""
        pass
    
    @abstractmethod
    def attribute(self, journeys: pd.DataFrame, conversion_event: str) -> AttributionResult:
        """Compute attribution for all converting journeys."""
        pass
    
    def validate(self, result: AttributionResult) -> Dict[str, bool]:
        """Standard validation — called after every attribution run."""
        checks = {
            "credits_sum_to_conversions": abs(
                sum(result.channel_credits.values()) - result.total_conversions
            ) < 0.01 * result.total_conversions,
            "no_negative_credits": all(v >= -0.001 for v in result.channel_credits.values()),
            "all_channels_present": set(result.channel_credits.keys()) == set(self.channel_ids),
            "shares_sum_to_one": abs(sum(result.credit_shares.values()) - 1.0) < 0.001,
        }
        return checks
    
    def bootstrap_ci(self, journeys: pd.DataFrame, conversion_event: str,
                     n_resamples: int = 200) -> tuple:
        """Compute bootstrap confidence intervals for channel credits."""
        credit_samples = {ch: [] for ch in self.channel_ids}
        rng = np.random.default_rng(42)
        
        for _ in range(n_resamples):
            sample = journeys.sample(n=len(journeys), replace=True, random_state=rng)
            result = self.attribute(sample, conversion_event)
            for ch, share in result.credit_shares.items():
                credit_samples[ch].append(share)
        
        ci_lower = {ch: np.percentile(samples, 2.5) for ch, samples in credit_samples.items()}
        ci_upper = {ch: np.percentile(samples, 97.5) for ch, samples in credit_samples.items()}
        return ci_lower, ci_upper
```

### 8.2 Model Registry (15 Variants)

| # | Model ID | Family | Primary? | Description |
|---|----------|--------|----------|-------------|
| 0a | `last_digital_click` | baseline | ✓ Benchmark | 100% credit → last digital touchpoint (GA4-equivalent) |
| 0b | `last_click_all` | baseline | | 100% credit → last touchpoint including agent |
| 0c | `first_touch` | baseline | | 100% credit → first touchpoint |
| 0d | `linear` | baseline | | Equal credit to all touchpoints |
| 0e | `time_decay` | baseline | | Exponential decay from conversion backward |
| 1a | `shapley_standard` | shapley | | Zhao et al. (2018) simplified coalition computation |
| 1b | `shapley_time_weighted` | shapley | ✓ PRIMARY | Standard + exponential recency weighting |
| 1c | `shapley_casv` | shapley | | Singal et al. (2022) causal counterfactual Shapley |
| 2a | `markov_1st_order` | markov | | 16-state absorbing chain, removal effects |
| 2b | `markov_2nd_order` | markov | ✓ PRIMARY | Compound states capture two-step sequences |
| 2c | `markov_3rd_order` | markov | | Three-step patterns, sparser but richer |
| 3a | `constrained_opt_binary` | constrained_opt | | Binary feature encoding |
| 3b | `constrained_opt_count` | constrained_opt | | Count-based features |
| 3c | `constrained_opt_recency` | constrained_opt | ✓ PRIMARY | Exponential recency-weighted features |
| 3d | `constrained_opt_position` | constrained_opt | | Position-encoded features (first/mid/last) |

### 8.3 Baseline Models (`src/models/baselines.py`)

Five heuristic models. These are straightforward — the code should handle them in a single module.

```python
class LastDigitalClickAttribution(BaseAttributionModel):
    """GA4 default — 100% credit to last DIGITAL touchpoint (agent excluded)."""
    
    def attribute(self, journeys: pd.DataFrame, conversion_event: str) -> AttributionResult:
        converting = journeys[journeys[conversion_event] == True]
        credits = {ch: 0.0 for ch in self.channel_ids}
        
        for _, journey in converting.iterrows():
            path = journey["channel_path"].split("|")
            # Find last digital touchpoint (exclude agent_interaction)
            digital_path = [ch for ch in path if ch != "agent_interaction"]
            if digital_path:
                credits[digital_path[-1]] += 1.0
            elif path:  # All agent → give to agent
                credits[path[-1]] += 1.0
        
        total = sum(credits.values())
        shares = {ch: c / total if total > 0 else 0 for ch, c in credits.items()}
        ranks = rank_dict(credits)
        return AttributionResult(
            model_id="last_digital_click", model_family="baseline",
            conversion_event=conversion_event,
            channel_credits=credits, credit_shares=shares, credit_ranks=ranks,
            journey_credits=pd.DataFrame(),  # Trivial for baselines
            total_conversions=int(total),
        )

# Similarly: LastClickAll, FirstTouch, Linear, TimeDecay
# TimeDecay uses half_life_days from config.attribution.shapley.time_weighting.half_life_days
```

### 8.4 Shapley Models (`src/models/shapley.py`)

**Variant 1a — Standard Shapley (Zhao et al. 2018 Simplified):**

```python
class ShapleyStandardAttribution(BaseAttributionModel):
    """
    Standard Shapley Values using inclusive coalition value function.
    
    Algorithm (Zhao et al. 2018 / Google DDA simplification):
    1. Group journeys by CHANNEL SET (unordered unique channels)
    2. For each unique channel set S with sufficient observations:
       v(S) = conversion_rate(journeys where channel_set == S)
    3. For each channel i, compute marginal contribution across all coalitions:
       φ_i = Σ_{S ⊆ N\{i}} [|S|!(|N|-|S|-1)!/|N|!] × [v(S∪{i}) - v(S)]
    4. Normalize so Σ φ_i = total conversions (efficiency axiom)
    
    Sparse coalition handling:
    - Coalitions with < min_observations (default 30): use adjacent imputation
      (average value of coalitions that differ by one channel)
    """
    
    def fit(self, journeys: pd.DataFrame) -> None:
        """Pre-compute coalition value function from journey data."""
        self.coalition_values = {}
        
        for channel_set_str, group in journeys.groupby("channel_set"):
            channels = frozenset(channel_set_str.split("|"))
            n_journeys = len(group)
            n_converting = group["policy_bound"].sum()
            
            if n_journeys >= self.config.min_coalition_observations:
                self.coalition_values[channels] = n_converting / n_journeys
            else:
                self.coalition_values[channels] = None  # Mark for imputation
        
        # Impute sparse coalitions
        self._impute_sparse_coalitions()
    
    def attribute(self, journeys: pd.DataFrame, conversion_event: str) -> AttributionResult:
        """Compute Shapley values for each channel."""
        n_channels = len(self.channel_ids)
        credits = {ch: 0.0 for ch in self.channel_ids}
        converting = journeys[journeys[conversion_event] == True]
        
        for channel in self.channel_ids:
            marginal_sum = 0.0
            others = [c for c in self.channel_ids if c != channel]
            
            # Iterate over all subsets of other channels
            for size in range(len(others) + 1):
                for subset in itertools.combinations(others, size):
                    S = frozenset(subset)
                    S_with_i = S | {channel}
                    
                    v_S = self._get_coalition_value(S)
                    v_S_with_i = self._get_coalition_value(S_with_i)
                    
                    if v_S is not None and v_S_with_i is not None:
                        # Shapley weight
                        weight = (math.factorial(len(S)) * 
                                 math.factorial(n_channels - len(S) - 1) /
                                 math.factorial(n_channels))
                        marginal_sum += weight * (v_S_with_i - v_S)
            
            credits[channel] = marginal_sum * len(converting)
        
        # ... normalize, compute shares, ranks, return AttributionResult
```

**Variant 1b — Time-Weighted Shapley:**

Extends 1a by applying exponential recency weighting to each channel's per-journey credit:

```python
# For each journey j, each channel i's touchpoints get weighted by recency:
# w(t) = exp(-λ × (T_j - t))   where λ = ln(2) / half_life_days
# The journey-level credit for channel i is scaled by the sum of its touchpoint weights
# relative to all touchpoints' weights in that journey.
# half_life_days comes from config.attribution.shapley.time_weighting.half_life_days
```

**Variant 1c — CASV (Causal Shapley Values, Singal et al. 2022):**

Uses a 1st-order Markov chain to compute COUNTERFACTUAL coalition values:

```python
# v_CASV(S) = predicted conversion rate if ONLY channels in S were available
# (simulated via restricted Markov chain where non-S channels are removed)
# This addresses Shapley's observational selection bias.
# Hybrid approach: Markov's causal structure + Shapley's four axioms.
```

### 8.5 Markov Chain Models (`src/models/markov.py`)

```python
class MarkovAttribution(BaseAttributionModel):
    """
    Absorbing Markov Chain attribution via removal effects.
    
    State space: {START} ∪ {13 channels} ∪ {CONVERSION} ∪ {NULL}
    
    For kth-order model, states are k-tuples of channels.
    E.g., 2nd order: (Display, Search) = "currently at Search, previously was Display"
    
    Attribution via removal effect:
    RE(c) = 1 - P(conversion | c removed) / P(conversion | all channels)
    φ(c) = RE(c) / Σ RE(all) × Total Conversions
    """
    
    def __init__(self, order: int, **kwargs):
        super().__init__(**kwargs)
        self.order = order
    
    def fit(self, journeys: pd.DataFrame) -> None:
        """Build transition matrix from journey data."""
        # Extract channel paths
        paths = []
        for _, j in journeys.iterrows():
            path = j["channel_path"].split("|")
            outcome = "CONVERSION" if j["policy_bound"] else "NULL"
            full_path = ["START"] + path + [outcome]
            paths.append(full_path)
        
        # Build kth-order transition counts
        if self.order == 1:
            self.states = ["START"] + self.channel_ids + ["CONVERSION", "NULL"]
            self.transition_counts = np.zeros((len(self.states), len(self.states)))
            for path in paths:
                for i in range(len(path) - 1):
                    from_idx = self.states.index(path[i])
                    to_idx = self.states.index(path[i + 1])
                    self.transition_counts[from_idx][to_idx] += 1
        
        elif self.order == 2:
            # Compound states: (prev_channel, current_channel)
            # Build from all consecutive triplets in paths
            ...
        
        elif self.order == 3:
            # Three-step compound states
            ...
        
        # Apply Dirichlet smoothing
        alpha = self.config.smoothing_alpha  # Default: 1/13 ≈ 0.077
        self.transition_matrix = (
            (self.transition_counts + alpha) / 
            (self.transition_counts.sum(axis=1, keepdims=True) + alpha * len(self.states))
        )
        
        # Compute overall conversion probability via absorbing chain
        self.base_conversion_prob = self._compute_absorption_prob()
    
    def attribute(self, journeys: pd.DataFrame, conversion_event: str) -> AttributionResult:
        """Compute removal effects for each channel."""
        credits = {}
        removal_effects = {}
        
        for channel in self.channel_ids:
            # Remove channel: redirect all transitions through channel to NULL
            modified_matrix = self.transition_matrix.copy()
            ch_idx = self.states.index(channel)
            modified_matrix[:, ch_idx] = 0  # No transitions TO this channel
            # Redistribute probability mass
            for row in range(len(self.states)):
                row_sum = modified_matrix[row].sum()
                if row_sum > 0:
                    modified_matrix[row] /= row_sum
            
            removed_prob = self._compute_absorption_prob(modified_matrix)
            removal_effects[channel] = 1 - (removed_prob / self.base_conversion_prob)
        
        # Normalize removal effects to sum to total conversions
        total_re = sum(removal_effects.values())
        converting = journeys[journeys[conversion_event] == True]
        total_conv = len(converting)
        
        for channel in self.channel_ids:
            credits[channel] = (removal_effects[channel] / total_re) * total_conv if total_re > 0 else 0
        
        # ... compute shares, ranks, return AttributionResult
    
    def _compute_absorption_prob(self, matrix=None) -> float:
        """
        Compute absorption probability from START to CONVERSION.
        Uses fundamental matrix of absorbing Markov chain:
        N = (I - Q)^{-1}  where Q is the transient-to-transient submatrix
        """
        if matrix is None:
            matrix = self.transition_matrix
        
        # Identify transient and absorbing states
        absorbing = {self.states.index("CONVERSION"), self.states.index("NULL")}
        transient = [i for i in range(len(self.states)) if i not in absorbing]
        
        Q = matrix[np.ix_(transient, transient)]
        R = matrix[np.ix_(transient, list(absorbing))]
        
        # Fundamental matrix
        N = np.linalg.inv(np.eye(len(transient)) - Q)
        
        # Absorption probabilities
        B = N @ R
        
        # Find START state index in transient list
        start_transient_idx = transient.index(self.states.index("START"))
        conv_absorbing_idx = list(absorbing).index(self.states.index("CONVERSION"))
        
        return B[start_transient_idx, conv_absorbing_idx]
    
    def get_transition_heatmap_data(self) -> pd.DataFrame:
        """Extract transition probabilities for the Markov heatmap visualization."""
        rows = []
        for i, from_state in enumerate(self.states):
            for j, to_state in enumerate(self.states):
                if from_state not in ["CONVERSION", "NULL"]:  # No transitions FROM absorbing states
                    rows.append({
                        "from_channel": from_state,
                        "to_channel": to_state,
                        "probability": self.transition_matrix[i, j],
                    })
        return pd.DataFrame(rows)
```

### 8.6 Constrained Optimization Models (`src/models/constrained_opt.py`)

```python
class ConstrainedOptAttribution(BaseAttributionModel):
    """
    Penalized constrained logistic regression for attribution.
    
    Objective: max L(w) - λ‖w‖₂²
    where L(w) = Σ_j [y_j × log(σ(α_j)) + (1-y_j) × log(1-σ(α_j))]
    α_j = Σ_i w_i × f_i(j)  (weighted combination of channel features for journey j)
    
    Constraints:
    - w_i ≥ 0 (non-negativity) for all channels
    - w_agent ≥ β × (1/N) × Σ_k w_k (agent floor, β from config)
    - w_i ≤ γ × Σ_k w_k for i≠agent (concentration cap, γ from config)
    
    Feature Encodings (4 variants):
    - Binary: f_i(j) = 1 if channel i in journey j
    - Count: f_i(j) = number of times channel i in journey j
    - Recency-weighted: f_i(j) = Σ_t exp(-λ(T_j - t)) for each touchpoint
    - Position-encoded: f_i(j) = [is_first, is_mid, is_last, total_count] (4D per channel)
    """
    
    def __init__(self, encoding: str, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding  # "binary", "count", "recency_weighted", "position_encoded"
    
    def fit(self, journeys: pd.DataFrame) -> None:
        """Learn optimal weights via constrained optimization."""
        X = self._encode_features(journeys)
        y = journeys["policy_bound"].astype(float).values
        
        n_channels = len(self.channel_ids)
        
        # Define objective (negative log-likelihood + L2 penalty)
        def objective(w):
            alpha = X @ w
            sigma = 1 / (1 + np.exp(-np.clip(alpha, -20, 20)))
            ll = np.sum(y * np.log(sigma + 1e-10) + (1 - y) * np.log(1 - sigma + 1e-10))
            penalty = self.config.regularization["lambda"] * np.sum(w ** 2)
            return -(ll - penalty)
        
        # Constraints
        constraints = []
        
        # Non-negativity
        bounds = [(0, None)] * n_channels
        
        # Agent floor: w_agent ≥ β × (1/N) × Σ w_k
        agent_idx = self.channel_ids.index("agent_interaction")
        beta = self.config.constraints["agent_floor"]["beta"]
        constraints.append({
            "type": "ineq",
            "fun": lambda w: w[agent_idx] - beta * (1 / n_channels) * np.sum(w)
        })
        
        # Concentration cap: w_i ≤ γ × Σ w_k for non-agent channels
        gamma = self.config.constraints["concentration_cap"]["gamma"]
        for i in range(n_channels):
            if i != agent_idx:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, idx=i: gamma * np.sum(w) - w[idx]
                })
        
        # Solve
        from scipy.optimize import minimize
        w0 = np.ones(n_channels) / n_channels
        result = minimize(objective, w0, method="SLSQP",
                         bounds=bounds, constraints=constraints)
        
        self.weights = result.x
        self.dual_variables = self._extract_dual_variables(result)
    
    def attribute(self, journeys: pd.DataFrame, conversion_event: str) -> AttributionResult:
        """Derive channel credits from learned weights."""
        converting = journeys[journeys[conversion_event] == True]
        X_conv = self._encode_features(converting)
        
        credits = {ch: 0.0 for ch in self.channel_ids}
        
        for j_idx, (_, journey) in enumerate(converting.iterrows()):
            # Per-journey credit: proportional to w_i × f_i(j)
            weighted = self.weights * X_conv[j_idx]
            total_weighted = weighted.sum()
            if total_weighted > 0:
                for i, ch in enumerate(self.channel_ids):
                    credits[ch] += weighted[i] / total_weighted
        
        # ... compute shares, ranks, return AttributionResult
    
    def _encode_features(self, journeys: pd.DataFrame) -> np.ndarray:
        """Encode journey data into feature matrix per encoding type."""
        n = len(journeys)
        
        if self.encoding == "binary":
            # f_i(j) = 1 if channel i appears in journey j
            X = np.zeros((n, len(self.channel_ids)))
            for j, (_, journey) in enumerate(journeys.iterrows()):
                for ch in journey["channel_set"].split("|"):
                    if ch in self.channel_ids:
                        X[j, self.channel_ids.index(ch)] = 1.0
        
        elif self.encoding == "count":
            # f_i(j) = count of channel i in journey j
            X = np.zeros((n, len(self.channel_ids)))
            for j, (_, journey) in enumerate(journeys.iterrows()):
                for ch in journey["channel_path"].split("|"):
                    if ch in self.channel_ids:
                        X[j, self.channel_ids.index(ch)] += 1.0
        
        elif self.encoding == "recency_weighted":
            # f_i(j) = Σ_t exp(-λ(T_j - t)) for each touchpoint of channel i
            half_life = self.config.time_weighting.get("half_life_days", 7)
            decay_rate = np.log(2) / half_life
            X = np.zeros((n, len(self.channel_ids)))
            for j, (_, journey) in enumerate(journeys.iterrows()):
                for tp in journey["touchpoints"]:
                    ch_idx = self.channel_ids.index(tp["channel_id"])
                    days_before = tp.get("time_to_conversion_days", 0)
                    X[j, ch_idx] += np.exp(-decay_rate * days_before)
        
        elif self.encoding == "position_encoded":
            # f_i(j) = [is_first, is_mid, is_last, count] per channel (4N features)
            X = np.zeros((n, len(self.channel_ids) * 4))
            for j, (_, journey) in enumerate(journeys.iterrows()):
                path = journey["channel_path"].split("|")
                for i, ch in enumerate(self.channel_ids):
                    base = i * 4
                    positions = [k for k, p in enumerate(path) if p == ch]
                    if positions:
                        X[j, base] = 1.0 if 0 in positions else 0.0      # is_first
                        X[j, base + 1] = 1.0 if any(0 < p < len(path)-1 for p in positions) else 0.0  # is_mid
                        X[j, base + 2] = 1.0 if len(path)-1 in positions else 0.0  # is_last
                        X[j, base + 3] = len(positions)                    # count
        
        return X
```

### 8.7 Model Runner (`scripts/run_attribution.py`)

```python
def run_all_models(journeys: pd.DataFrame, config: MasterConfig) -> List[AttributionResult]:
    """
    Run all 15 model variants × 2 conversion events = 30 attribution runs.
    
    Runtime budget: ~5 minutes total on 25K journeys.
    """
    channel_ids = [ch["id"] for ch in config.channels["taxonomy"]]
    results = []
    
    models = [
        # Baselines
        LastDigitalClickAttribution("last_digital_click", "baseline", channel_ids),
        LastClickAllAttribution("last_click_all", "baseline", channel_ids),
        FirstTouchAttribution("first_touch", "baseline", channel_ids),
        LinearAttribution("linear", "baseline", channel_ids),
        TimeDecayAttribution("time_decay", "baseline", channel_ids),
        # Shapley
        ShapleyStandardAttribution("shapley_standard", "shapley", channel_ids, config.attribution["shapley"]),
        ShapleyTimeWeightedAttribution("shapley_time_weighted", "shapley", channel_ids, config.attribution["shapley"]),
        ShapleyCAVSAttribution("shapley_casv", "shapley", channel_ids, config.attribution["shapley"]),
        # Markov
        MarkovAttribution("markov_1st_order", "markov", channel_ids, order=1, config=config.attribution["markov"]),
        MarkovAttribution("markov_2nd_order", "markov", channel_ids, order=2, config=config.attribution["markov"]),
        MarkovAttribution("markov_3rd_order", "markov", channel_ids, order=3, config=config.attribution["markov"]),
        # Constrained Optimization
        ConstrainedOptAttribution("constrained_opt_binary", "constrained_opt", channel_ids, encoding="binary", config=config.attribution["constrained_optimization"]),
        ConstrainedOptAttribution("constrained_opt_count", "constrained_opt", channel_ids, encoding="count", config=config.attribution["constrained_optimization"]),
        ConstrainedOptAttribution("constrained_opt_recency", "constrained_opt", channel_ids, encoding="recency_weighted", config=config.attribution["constrained_optimization"]),
        ConstrainedOptAttribution("constrained_opt_position", "constrained_opt", channel_ids, encoding="position_encoded", config=config.attribution["constrained_optimization"]),
    ]
    
    for model in models:
        model.fit(journeys)
        for event in ["quote_start", "bind"]:
            conversion_col = "quote_started" if event == "quote_start" else "policy_bound"
            result = model.attribute(journeys, event)
            
            # Bootstrap CIs for primary models only (saves time)
            if model.model_id in PRIMARY_MODELS:
                result.ci_lower, result.ci_upper = model.bootstrap_ci(journeys, event, n_resamples=200)
            
            validation = model.validate(result)
            assert all(validation.values()), f"Validation failed for {model.model_id}: {validation}"
            results.append(result)
    
    return results
```

---

## 9. Model Comparison Framework

### 9.1 Head-to-Head Comparison (`src/comparison/head_to_head.py`)

Five metrics comparing model outputs:

```python
class HeadToHeadComparison:
    """
    Compare attribution models pairwise and generate comparison metrics.
    
    Metrics:
    1. Spearman Rank Correlation (ρ)
    2. Jensen-Shannon Divergence (JSD)
    3. Axiom Compliance Audit
    4. Sensitivity Analysis
    5. Convergence/Divergence Map
    """
    
    def compare_all(self, results: List[AttributionResult]) -> Dict[str, pd.DataFrame]:
        """Run all comparisons. Returns dict of DataFrames for each Parquet output."""
        pairwise = self._pairwise_metrics(results)
        convergence = self._convergence_map(results)
        dual_funnel = self._dual_funnel_analysis(results)
        return {
            "pairwise_metrics": pairwise,
            "convergence_map": convergence,
            "dual_funnel": dual_funnel,
        }
    
    def _pairwise_metrics(self, results):
        """Spearman ρ and JSD for all model pairs."""
        from scipy.stats import spearmanr
        from scipy.spatial.distance import jensenshannon
        
        pairs = []
        for i, r1 in enumerate(results):
            for r2 in results[i+1:]:
                if r1.conversion_event != r2.conversion_event:
                    continue
                
                v1 = np.array([r1.credit_shares[ch] for ch in sorted(r1.credit_shares)])
                v2 = np.array([r2.credit_shares[ch] for ch in sorted(r2.credit_shares)])
                
                rho, pval = spearmanr(v1, v2)
                jsd = jensenshannon(v1, v2) ** 2  # Squared for bounded [0,1]
                
                pairs.append({
                    "model_a": r1.model_id, "model_b": r2.model_id,
                    "conversion_event": r1.conversion_event,
                    "spearman_rho": rho, "spearman_pvalue": pval,
                    "jensen_shannon_div": jsd,
                    "l2_distance": np.linalg.norm(v1 - v2),
                })
        return pd.DataFrame(pairs)
    
    def _convergence_map(self, results):
        """
        Per-channel confidence classification across 3 primary models.
        
        Rules:
        - All 3 agree within 3pp → HIGH confidence
        - 2 agree, 1 diverges → MODERATE confidence  
        - All 3 disagree → LOW confidence → "INVESTIGATE"
        """
        primary = [r for r in results if r.model_id in PRIMARY_MODELS]
        rows = []
        
        for event in ["quote_start", "bind"]:
            event_results = [r for r in primary if r.conversion_event == event]
            last_click = [r for r in results if r.model_id == "last_digital_click" 
                         and r.conversion_event == event][0]
            
            for channel in sorted(event_results[0].credit_shares.keys()):
                credits = [r.credit_shares[channel] for r in event_results]
                mean_credit = np.mean(credits)
                credit_range = max(credits) - min(credits)
                
                if credit_range <= 0.03:
                    confidence = "HIGH"
                elif credit_range <= 0.08:
                    confidence = "MODERATE"
                else:
                    confidence = "LOW"
                
                lc_credit = last_click.credit_shares[channel]
                shift = mean_credit - lc_credit
                
                if shift > 0.03:
                    direction = "INCREASE"
                elif shift < -0.03:
                    direction = "DECREASE"
                elif confidence == "LOW":
                    direction = "INVESTIGATE"
                else:
                    direction = "HOLD"
                
                rows.append({
                    "channel_id": channel,
                    "conversion_event": event,
                    "shapley_credit": event_results[0].credit_shares[channel],
                    "markov_credit": event_results[1].credit_shares[channel],
                    "or_credit": event_results[2].credit_shares[channel],
                    "credit_mean": mean_credit,
                    "credit_range": credit_range,
                    "confidence_zone": confidence,
                    "consensus_direction": direction,
                    "last_click_credit": lc_credit,
                    "credit_shift_from_last_click": shift,
                })
        
        return pd.DataFrame(rows)
```

---

## 10. Budget Optimization Layer

### 10.1 Response Curve Calibration (`src/optimization/response_curves.py`)

```python
class ResponseCurveCalibrator:
    """
    Calibrate saturating exponential response curves per channel:
    f_i(x_i) = a_i × (1 - exp(-b_i × x_i))
    
    Parameters:
    - a_i = saturation ceiling = attributed_conversions_i / saturation_factor_i
    - b_i = efficiency rate = -ln(1 - saturation_factor_i) / current_spend_i
    
    saturation_factor from config.attribution.budget_optimizer.channel_saturation_factors
    """
    
    def calibrate(self, attribution_credits: pd.DataFrame, config) -> pd.DataFrame:
        """
        For each channel, compute response curve parameters.
        Input: attribution results (from primary model) + current budget allocation.
        Output: response_curves.parquet
        """
        rows = []
        budget_config = config.attribution["budget_optimizer"]
        
        for channel_id, sat_factor in budget_config["channel_saturation_factors"].items():
            current_spend = budget_config["current_allocation"].get(channel_id, 0)
            if current_spend <= 0:
                continue
            
            # Get attributed conversions for this channel (from primary model, bind event)
            attr_conv = attribution_credits[
                (attribution_credits["channel_id"] == channel_id) & 
                (attribution_credits["conversion_event"] == "bind")
            ]["total_credit"].values[0]
            
            a_i = attr_conv / sat_factor  # Saturation ceiling
            b_i = -np.log(1 - sat_factor) / current_spend  # Efficiency rate
            marginal_roi = a_i * b_i * np.exp(-b_i * current_spend)  # First derivative
            
            rows.append({
                "channel_id": channel_id,
                "saturation_ceiling_a": a_i,
                "efficiency_rate_b": b_i,
                "saturation_factor": sat_factor,
                "current_spend": current_spend,
                "current_conversions": attr_conv,
                "marginal_roi_at_current": marginal_roi,
            })
        
        return pd.DataFrame(rows)
```

### 10.2 MILP Budget Optimizer (`src/optimization/budget_optimizer.py`)

```python
class BudgetOptimizer:
    """
    MILP budget optimization with piecewise linearization.
    
    Decision variables:
    - x_i ∈ ℝ⁺: spend allocated to channel i
    - z_i ∈ {0,1}: is channel i active?
    
    Objective: MAXIMIZE Σ_i a_i × (1 - exp(-b_i × x_i))
    → Linearized via K piecewise segments per channel
    
    Constraints:
    C1: Σ x_i = B (total budget constant)
    C2: x_i ≥ floor_i × z_i (minimum viable spend)
    C3: |x_i - x_i_current| ≤ δ × x_i_current (max reallocation)
    C4: x_agent ≥ 0.02 × B (agent support floor)
    C5: x_i ≤ M × z_i (activation linking)
    C6: Σ z_i ≥ K_min (minimum active channels)
    C7: x_tv ≥ 0.10 × B (TV minimum for brand presence)
    C8: x_search_nb ≤ 0.25 × B (search concentration cap)
    """
    
    def optimize(self, response_curves: pd.DataFrame, config) -> pd.DataFrame:
        """
        Run budget optimization. Returns budget_recommendations.parquet.
        """
        import pulp
        
        budget_config = config.attribution["budget_optimizer"]
        B = budget_config["total_budget"]
        channels = list(budget_config["current_allocation"].keys())
        # Exclude non-optimizable channels
        optimizable = [ch for ch in channels if ch in budget_config["channel_saturation_factors"]]
        non_optimizable_spend = sum(
            spend for ch, spend in budget_config["current_allocation"].items() 
            if ch not in optimizable
        )
        B_opt = B - non_optimizable_spend  # Budget available for optimization
        
        # Create piecewise linear approximation of response curves
        K = budget_config.get("piecewise_segments", 10)
        
        prob = pulp.LpProblem("BudgetOptimization", pulp.LpMaximize)
        
        # Decision variables
        x = {ch: pulp.LpVariable(f"spend_{ch}", lowBound=0) for ch in optimizable}
        z = {ch: pulp.LpVariable(f"active_{ch}", cat="Binary") for ch in optimizable}
        
        # Piecewise linear variables (λ weights for each segment)
        seg_vars = {}
        for ch in optimizable:
            rc = response_curves[response_curves["channel_id"] == ch].iloc[0]
            a, b = rc["saturation_ceiling_a"], rc["efficiency_rate_b"]
            max_spend = budget_config["current_allocation"][ch] * 2  # Allow up to 2× current
            breakpoints = np.linspace(0, max_spend, K + 1)
            values = [a * (1 - np.exp(-b * bp)) for bp in breakpoints]
            seg_vars[ch] = (breakpoints, values)
        
        # Objective: sum of piecewise-linearized conversions
        # ... (implement piecewise linearization using SOS2 or incremental formulation)
        
        # Constraints
        constraints = budget_config.get("constraints", {})
        
        # C1: Total budget
        prob += pulp.lpSum(x[ch] for ch in optimizable) == B_opt
        
        # C2: Channel minimum floors
        for ch in optimizable:
            floor = budget_config["channel_minimum_floors"].get(ch, 0)
            prob += x[ch] >= floor * z[ch]
        
        # C3: Max reallocation
        delta = constraints.get("max_reallocation_pct", 0.20)
        for ch in optimizable:
            current = budget_config["current_allocation"][ch]
            prob += x[ch] <= current * (1 + delta)
            prob += x[ch] >= current * (1 - delta)
        
        # C4: Agent support floor
        agent_floor_pct = constraints.get("agent_support_floor_pct", 0.02)
        prob += x["agent_support"] >= agent_floor_pct * B
        
        # C5: Activation linking
        for ch in optimizable:
            prob += x[ch] <= B * z[ch]
        
        # C6: Minimum active channels
        min_active = constraints.get("min_active_channels", 8)
        prob += pulp.lpSum(z[ch] for ch in optimizable) >= min(min_active, len(optimizable))
        
        # C7: TV minimum
        tv_min_pct = constraints.get("tv_minimum_pct", 0.10)
        if "tv_radio" in x:
            prob += x["tv_radio"] >= tv_min_pct * B
        
        # C8: Search nonbrand cap
        snb_cap_pct = constraints.get("search_nonbrand_cap_pct", 0.25)
        if "search_nonbrand" in x:
            prob += x["search_nonbrand"] <= snb_cap_pct * B
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract results
        results = []
        for ch in optimizable:
            results.append({
                "channel_id": ch,
                "current_spend": budget_config["current_allocation"][ch],
                "optimal_spend": x[ch].varValue,
                "spend_change_dollars": x[ch].varValue - budget_config["current_allocation"][ch],
                "spend_change_pct": (x[ch].varValue - budget_config["current_allocation"][ch]) / budget_config["current_allocation"][ch],
                "is_optimizable": True,
            })
        
        return pd.DataFrame(results)
```

### 10.3 Scenario Engine (`src/optimization/scenario_engine.py`)

Pre-computes optimization for 7 scenarios from config. Each scenario overrides specific parameters and re-runs the optimizer.

```python
SCENARIOS = [
    "baseline", "digital_transformation", "no_call_tracking", "full_call_tracking",
    "cut_upper_funnel", "agent_decline", "clean_data_baseline"
]

# For each scenario:
# 1. Load config with scenario overrides
# 2. If scenario changes data generation params → use pre-computed scenario-specific attribution
#    (or approximate by scaling current attribution results)
# 3. Re-run budget optimizer on modified attribution
# 4. Output scenario_results.parquet
```

### 10.4 Shadow Price Extraction (`src/optimization/shadow_prices.py`)

Extract dual variables from the constrained optimization and translate to business insights.

```python
# Shadow price on C4 (agent floor) ≈ 0 → "Optimizer wants to give agents MORE than minimum"
# Shadow price on C7 (TV minimum) > 0 → "TV floor is costing N conversions — evaluate this assumption"
# Shadow price on C3 (reallocation cap) for Search NB → "How much more we'd gain by reallocating faster"
```


---

## 11. UI/UX Specification — Plotly Dash Application

### 11.1 Design System

**Color Palette:**

```python
# app/theme.py

ERIE_BLUE = "#00447C"          # Primary brand color — used for headers, active states, primary CTAs
ERIE_BLUE_LIGHT = "#E8F0F8"    # Light blue — card backgrounds, selected states
ERIE_BLUE_DARK = "#002D52"     # Dark blue — navbar, footer
WHITE = "#FFFFFF"              # Page background
LIGHT_GRAY = "#F5F7FA"        # Section backgrounds, alternating rows
MEDIUM_GRAY = "#E0E4E8"       # Borders, dividers
DARK_GRAY = "#4A5568"         # Body text
TEXT_BLACK = "#1A202C"         # Headlines
ACCENT_GREEN = "#38A169"       # Positive changes, "INCREASE" indicators
ACCENT_RED = "#E53E3E"        # Negative changes, "DECREASE" indicators
ACCENT_AMBER = "#D69E2E"      # "MODERATE" confidence, warnings
ACCENT_EMERALD = "#2F855A"    # "HIGH" confidence zones

# Consistent channel colors — used across ALL charts
CHANNEL_COLORS = {
    "agent_interaction": "#00447C",    # Erie Blue — the hero channel
    "tv_radio": "#7B61FF",             # Purple
    "display": "#FF6B35",              # Orange
    "paid_social": "#00B4D8",          # Teal
    "search_nonbrand": "#2D9B44",      # Green
    "search_brand": "#85C74E",         # Light green
    "seo_organic": "#8B8B8B",          # Gray
    "email_nurture": "#FFB627",        # Gold
    "direct_mail": "#C77DFF",          # Light purple
    "referral_wom": "#FF8FAB",         # Pink
    "agent_locator": "#0077B6",        # Blue
    "erie_direct": "#003D6B",          # Dark blue
    "retargeting": "#F77F00",          # Deep orange
}

# Last-click baseline always in muted gray
BASELINE_COLOR = "#CBD5E0"

# Plotly template
PLOTLY_TEMPLATE = {
    "layout": {
        "font": {"family": "Inter, system-ui, sans-serif", "color": DARK_GRAY},
        "paper_bgcolor": WHITE,
        "plot_bgcolor": WHITE,
        "colorway": list(CHANNEL_COLORS.values()),
        "xaxis": {"gridcolor": LIGHT_GRAY, "zerolinecolor": MEDIUM_GRAY},
        "yaxis": {"gridcolor": LIGHT_GRAY, "zerolinecolor": MEDIUM_GRAY},
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    }
}
```

**Typography:**
- Headlines: Inter Bold, 20-24px, TEXT_BLACK
- Subheads: Inter SemiBold, 16-18px, DARK_GRAY
- Body: Inter Regular, 14px, DARK_GRAY
- Insight callouts: Inter Medium, 15px, ERIE_BLUE on ERIE_BLUE_LIGHT background
- Metric cards: Value in Inter Bold 28px ERIE_BLUE, label in Inter Regular 12px DARK_GRAY

**Chart titles should BE the insight** — not generic labels:
- ✅ "Agents receive 35% credit when measured properly — 10× what last-click shows"
- ❌ "Figure 3: Channel Attribution Comparison"

### 11.2 Dash Application Architecture (`app/app.py`)

```python
import dash
from dash import Dash, html, dcc, page_registry, page_container
import dash_bootstrap_components as dbc
from app.theme import ERIE_BLUE_DARK
from app.components.navbar import create_navbar
from app.data_store import load_all_data

# Initialize with multi-page support
app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,  # Base Bootstrap CSS
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="MCA Intelligence — Erie Auto Insurance",
    update_title=None,
)

server = app.server  # Exposed for Gunicorn

# Pre-load all data at startup
DATA = load_all_data()

app.layout = dbc.Container([
    # Cross-page state stores
    dcc.Store(id="selected-conversion-event", data="bind"),
    dcc.Store(id="selected-primary-model", data="shapley_time_weighted"),
    
    # Navbar
    create_navbar(),
    
    # Page content
    html.Div(
        page_container,
        style={"paddingTop": "20px", "paddingBottom": "40px"}
    ),
    
    # Footer
    html.Footer(
        html.P("Erie Insurance MCA Capability Demo — Confidential",
               className="text-center text-muted small py-3"),
        style={"borderTop": f"1px solid #E0E4E8"}
    ),
], fluid=True, style={"backgroundColor": "#F5F7FA", "minHeight": "100vh"})


if __name__ == "__main__":
    app.run(debug=False, port=8050)
```

### 11.3 Data Store (`app/data_store.py`)

```python
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def load_all_data() -> dict:
    """Load all pre-computed Parquet files at app startup."""
    return {
        # Attribution results
        "channel_credits": pd.read_parquet(DATA_DIR / "attribution" / "channel_credits.parquet"),
        "journey_credits": pd.read_parquet(DATA_DIR / "attribution" / "journey_credits.parquet"),
        "model_diagnostics": pd.read_parquet(DATA_DIR / "attribution" / "model_diagnostics.parquet"),
        
        # Comparison
        "pairwise_metrics": pd.read_parquet(DATA_DIR / "comparison" / "pairwise_metrics.parquet"),
        "convergence_map": pd.read_parquet(DATA_DIR / "comparison" / "convergence_map.parquet"),
        "dual_funnel": pd.read_parquet(DATA_DIR / "comparison" / "dual_funnel.parquet"),
        
        # Resolution
        "identity_graph": pd.read_parquet(DATA_DIR / "resolved" / "identity_graph.parquet"),
        "unified_journeys": pd.read_parquet(DATA_DIR / "resolved" / "unified_journeys.parquet"),
        "resolution_report": pd.read_parquet(DATA_DIR / "resolved" / "resolution_report.parquet"),
        
        # Optimization
        "budget_recommendations": pd.read_parquet(DATA_DIR / "optimization" / "budget_recommendations.parquet"),
        "response_curves": pd.read_parquet(DATA_DIR / "optimization" / "response_curves.parquet"),
        "shadow_prices": pd.read_parquet(DATA_DIR / "optimization" / "shadow_prices.parquet"),
        "scenario_results": pd.read_parquet(DATA_DIR / "optimization" / "scenario_results.parquet"),
    }
```

### 11.4 Navigation Bar (`app/components/navbar.py`)

```python
def create_navbar():
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("MCA Intelligence", className="fw-bold text-white fs-5"),
            dbc.Nav([
                dbc.NavLink("Summary", href="/", className="text-white"),
                dbc.NavLink("Identity", href="/identity", className="text-white"),
                dbc.NavLink("Models", href="/models", className="text-white"),
                dbc.NavLink("Channels", href="/channels", className="text-white"),
                dbc.NavLink("Funnel", href="/funnel", className="text-white"),
                dbc.NavLink("Budget", href="/budget", className="text-white"),
                dbc.NavLink("Scenarios", href="/scenarios", className="text-white"),
                dbc.NavLink("Roadmap", href="/roadmap", className="text-white"),
            ], navbar=True),
        ], fluid=True),
        color="#002D52",  # ERIE_BLUE_DARK
        dark=True,
        sticky="top",
    )
```

### 11.5 Page Specifications

Each page registers with Dash's multi-page system via `dash.register_page(__name__, path="/...", name="...")`.

---

#### Page 1: Executive Summary (`/`)

**Purpose:** The "jaw-drop" page. Four KPI cards + the single most important chart.

**Layout:**

```
┌─────────────────────────────────────────────────────────────┐
│ ERIE INSURANCE — MULTI-CHANNEL ATTRIBUTION INTELLIGENCE     │
├─────────────────────────────────────────────────────────────┤
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│ │ AGENT    │ │ BRAND    │ │ BUDGET   │ │ MODEL            ││
│ │ CREDIT   │ │ SEARCH   │ │ EFFICIENCY│ │ AGREEMENT       ││
│ │  35%     │ │  11%     │ │  +16%    │ │  ρ = 0.87       ││
│ │ (was 3%) │ │ (was 38%)│ │ projected│ │  (3 models)     ││
│ └──────────┘ └──────────┘ └──────────┘ └──────────────────┘│
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [Grouped horizontal bar chart]                          │ │
│ │ "Attribution by Channel: Last-Click vs. Model-Based"    │ │
│ │                                                         │ │
│ │ For each of 13 channels:                                │ │
│ │   - Gray bar = last-digital-click credit                │ │
│ │   - Erie Blue bar = model-based credit (primary avg)    │ │
│ │ Sorted by absolute difference (largest shift first)     │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Insight: "Current last-click attribution under-credits  │ │
│ │ agents by 32pp and over-credits branded search by 27pp. │ │
│ │ Three independent models agree."                        │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Data sources:** `channel_credits.parquet` (filter by `is_primary_model=True` + `model_id="last_digital_click"`), `convergence_map.parquet`, `budget_recommendations.parquet`.

**KPI cards compute:**
- Agent credit: mean credit_share across 3 primary models for agent_interaction (bind event)
- Brand search: same for search_brand
- Budget efficiency: mean projected improvement from budget_recommendations
- Model agreement: mean Spearman ρ across primary model pairs from pairwise_metrics

---

#### Page 2: Identity Resolution (`/identity`)

**Purpose:** Explain WHY current attribution is wrong — the identity fragmentation problem.

**Layout:**
- **Top:** Jennifer Morrison journey visualization — conceptual diagram showing 8 system records fragmenting into disconnected pieces on the left, then connecting into a unified journey on the right. (This can be a static Plotly figure or an annotated timeline.)
- **Middle:** Resolution metrics cards (records processed, unique identities, resolution rate, agent journeys recovered)
- **Bottom:** Before/After attribution comparison table — showing how channel credits change with and without identity resolution. Toggleable by resolution tier (Tier 1 only / Tier 1+2 / Tier 1+2+3).

**Data sources:** `identity_graph.parquet`, `resolution_report.parquet`, `channel_credits.parquet`.

---

#### Page 3: Three-Model Comparison (`/models`)

**Purpose:** Build credibility through convergence — three independent frameworks agree.

**Layout:**
- **Top row:** Three side-by-side horizontal bar charts (Shapley, Markov, OR) with identical axis scales. Same channel colors. Visual comparison is immediate.
- **Middle:** Convergence/Divergence table from `convergence_map.parquet` — the money table. Columns: Channel, Shapley %, Markov %, OR %, Confidence Zone (colored dot), Consensus Direction.
- **Bottom:** Pairwise Spearman correlation matrix (3×3 heatmap) — small supporting visual.

**Interactive elements:**
- Toggle between bind / quote-start conversion events
- Click any channel row → navigate to Channel Deep-Dive (Page 4) with that channel pre-selected

**Data sources:** `channel_credits.parquet`, `convergence_map.parquet`, `pairwise_metrics.parquet`.

---

#### Page 4: Channel Deep-Dive (`/channels`)

**Purpose:** Agent-focused by default, but dynamic for any channel.

**Layout:**
- **Channel selector dropdown** (default: Agent Interaction)
- **Visualization 1: Credit Waterfall** — decomposes credit shift from last-click → with resolution → with attribution modeling. Shows: Starting point (last-click) + identity resolution contribution + attribution modeling contribution = final credit.
- **Visualization 2: Markov Transition Heatmap** — which channels have highest transition probability INTO the selected channel. For Agent: Agent Locator 0.42, Brand Search 0.18, Erie.com 0.15, etc.
- **Visualization 3: Funnel Comparison** — for Agent: Agent-touched journeys vs. digital-only journeys, showing conversion rates at each funnel stage. Agent-touched converts at 2.8× rate.

**Data sources:** `channel_credits.parquet`, Markov transition data from `model_diagnostics.parquet` or pre-computed transition matrix, `unified_journeys.parquet`.

---

#### Page 5: Dual-Funnel Analysis (`/funnel`)

**Purpose:** Show that quote and bind tell different stories.

**Layout:**
- **Side-by-side bar charts:** Quote-start attribution (left) vs. Bind attribution (right)
- **Center arrows:** Channels that gain/lose credit from quote→bind with magnitude
- **Insight callout:** "Nonbrand search drives 18% of quotes but only 9% of binds — it generates research but doesn't close. Agents drive 20% of quotes but 35% of binds — they close deals."

**Data sources:** `dual_funnel.parquet`.

---

#### Page 6: Budget Optimization (`/budget`)

**Purpose:** Translate attribution into dollars.

**Layout:**
- **Top:** Budget recommendation table from `budget_recommendations.parquet` — Current Spend | Shapley-Optimal | Markov-Optimal | OR-Optimal | Consensus Direction
- **Middle:** Projected impact summary cards (projected binds, cost per bind, efficiency gain)
- **Bottom:** Interactive budget simulator — sliders for each optimizable channel, total spend must remain constant. Real-time update of predicted conversions as user moves sliders.

**Interactive elements:**
- Budget sliders (linked: moving one reduces available budget for others)
- Real-time conversion prediction using response curve parameters from `response_curves.parquet`

**Data sources:** `budget_recommendations.parquet`, `response_curves.parquet`, `shadow_prices.parquet`.

---

#### Page 7: Scenario Explorer (`/scenarios`)

**Purpose:** Interactive what-if engine for strategic questions.

**Layout:**
- **Left:** Scenario selector (7 pre-built + custom option). Each scenario has a description card.
- **Right:** Results waterfall showing key metric changes vs. baseline (projected binds, cost per bind, agent credit, budget reallocation amount).
- **Bottom:** Channel-level comparison table: baseline spend vs. scenario spend per channel.

**Highlight the call tracking ROI scenario:**
```
Full Call Tracking Deployment
  Agent credit accuracy: +13pp
  Additional binds: +340/year
  Better budget allocation: $280K shifted
  Call tracking cost: $50K/year
  ROI: 9.6×
```

**Data sources:** `scenario_results.parquet`.

---

#### Page 8: Measurement Roadmap (`/roadmap`)

**Purpose:** Connect demo to production implementation.

**Layout:**
- **Top:** Four-stage maturity visualization (horizontal progression):
  Stage 1: Last-Click (Current) → Stage 2: MTA (Demo Proves) → Stage 3: MTA+MMM → Stage 4: Full Triangulation
- **Middle:** Three concrete next steps with timeline and investment:
  1. Deploy call tracking (90 days, ~$50K/year)
  2. Productionize MTA pipeline (6 months, 3-4 FTE)
  3. Add MMM layer (3 months after MTA)
- **Bottom:** Value projection — cumulative improvement from each stage

This page can be mostly static content with Plotly figure for the maturity visualization.

---

#### Technical Appendix (`/appendix`)

**Purpose:** Deep-dive for analytics team. NOT in main navigation flow.

**Sub-sections:**
- Mathematical Formulations (Shapley formula, Markov chain notation, OR KKT conditions)
- Full 15-model arsenal results (all variants, all channels)
- Sensitivity analysis heatmaps (parameter perturbation results)
- Data quality report (dirty data handling summary, identity resolution detailed metrics)
- Axiom compliance audit results

### 11.6 Key Dash Patterns

**Callbacks must follow these rules:**

1. **No computation in callbacks.** Callbacks filter/slice pre-loaded DataFrames from `data_store.py`. They do NOT call attribution models or optimizers.
2. **Use `dcc.Store` for cross-page state.** Selected conversion event and model preferences persist across pages.
3. **Use `prevent_initial_call=True`** for expensive callbacks.
4. **Chart updates use `Patch()`** where possible for partial updates (Dash 2.17+).

**Example callback pattern:**

```python
@callback(
    Output("attribution-comparison-chart", "figure"),
    Input("conversion-event-toggle", "value"),
)
def update_attribution_chart(conversion_event):
    from app.data_store import DATA
    df = DATA["channel_credits"]
    
    # Filter to primary models + baseline
    df_filtered = df[
        (df["conversion_event"] == conversion_event) &
        ((df["is_primary_model"] == True) | (df["model_id"] == "last_digital_click"))
    ]
    
    fig = create_attribution_comparison_figure(df_filtered)
    return fig
```

### 11.7 Reusable Components

**Metric Card (`app/components/metric_card.py`):**

```python
def metric_card(value, label, subtitle=None, color=ERIE_BLUE, delta=None):
    """KPI card component used across all pages."""
    children = [
        html.H2(value, style={"color": color, "fontWeight": "bold", "marginBottom": "4px"}),
        html.P(label, className="text-muted small mb-0"),
    ]
    if subtitle:
        children.append(html.P(subtitle, className="text-muted small mb-0", 
                               style={"fontSize": "11px"}))
    if delta:
        delta_color = ACCENT_GREEN if "+" in str(delta) else ACCENT_RED
        children.append(html.Span(str(delta), style={"color": delta_color, "fontSize": "12px"}))
    
    return dbc.Card(
        dbc.CardBody(children, className="text-center py-3"),
        className="shadow-sm h-100",
        style={"borderTop": f"3px solid {color}"}
    )
```

**Insight Callout (`app/components/insight_callout.py`):**

```python
def insight_callout(text, icon="💡"):
    """Styled insight box — the one-sentence finding per page."""
    return html.Div(
        [html.Span(icon, style={"marginRight": "8px"}), html.Span(text)],
        style={
            "backgroundColor": ERIE_BLUE_LIGHT,
            "borderLeft": f"4px solid {ERIE_BLUE}",
            "padding": "16px 20px",
            "borderRadius": "4px",
            "fontSize": "15px",
            "color": ERIE_BLUE_DARK,
            "fontWeight": "500",
        },
        className="my-3",
    )
```

---

## 12. Deployment & Packaging

### 12.1 Pipeline Execution Order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data (~2-3 minutes)
python scripts/generate_data.py --config config/erie_mca_synth_config.yaml
# → Outputs to data/source_systems/ and data/ground_truth/

# 3. Run identity resolution + journey assembly (~1-2 minutes)
python scripts/run_attribution.py --stage resolution
# → Outputs to data/resolved/

# 4. Run all attribution models + comparison + optimization (~5-7 minutes)
python scripts/run_attribution.py --stage attribution
# → Outputs to data/attribution/, data/comparison/, data/optimization/

# 5. Validate all data contracts
python scripts/validate_data_contracts.py
# → Checks all Parquet schemas match Section 5 specs

# 6. Launch the Dash app
python app/app.py
# → http://localhost:8050

# OR: full pipeline in one command
python scripts/run_full_pipeline.py --config config/erie_mca_synth_config.yaml
```

### 12.2 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate data and run pipeline at build time
# This ensures the container ships with pre-computed results
RUN python scripts/run_full_pipeline.py --config config/erie_mca_synth_config.yaml

# Expose port
EXPOSE 8050

# Run with Gunicorn for production
CMD ["gunicorn", "app.app:server", "-b", "0.0.0.0:8050", "-w", "2", "--timeout", "120"]
```

### 12.3 Render.com Configuration (`render.yaml`)

```yaml
services:
  - type: web
    name: erie-mca-demo
    runtime: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: PYTHON_VERSION
        value: "3.11"
    plan: starter  # $7/month, always-on, no cold starts
    healthCheckPath: /
```

**Deployment steps:**
1. Push code to GitHub repository
2. Connect repository to Render.com
3. Render auto-builds Docker image and deploys
4. Shareable URL: `https://erie-mca-demo.onrender.com`

### 12.4 `.dockerignore`

```
.git
.gitignore
__pycache__
*.pyc
.pytest_cache
tests/
*.md
.env
```

### 12.5 `.gitignore`

```
data/
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
.pytest_cache/
output/
```

Note: The `data/` directory is gitignored because it's generated by the pipeline. The Dockerfile's build step generates all data inside the container.

---

## 13. Build Order & Validation Checkpoints

### 13.1 Module Build Order

Generate code in this exact order. Each module has validation checks before proceeding to the next.

| Phase | Module | Validation |
|-------|--------|-----------|
| **1** | `config/schema.py` + `config/loader.py` | Config loads without errors, all Pydantic validators pass |
| **2** | `utils/data_io.py` + `utils/formatters.py` + `utils/logging_setup.py` | Utility functions work standalone |
| **3** | `generators/population.py` | 25,000 prospects generated, distribution matches config |
| **4** | `generators/behavioral_sim.py` | Journeys generated, bind count 4,500-5,500, agent involvement 83-91% |
| **5** | `generators/system_records.py` | Source system Parquet files created, ~180K total records |
| **6** | `generators/dirty_data.py` | Dirty data injected, 8-15% records affected |
| **7** | `resolution/identity_resolver.py` | Identity graph created, resolution rate 65-72% |
| **8** | `resolution/journey_assembler.py` | Unified journeys created, ~25K total, schema matches Section 5.3 |
| **9** | `models/base.py` | Abstract base class + AttributionResult dataclass ready |
| **10** | `models/baselines.py` | 5 baseline models run, credits sum to conversions |
| **11** | `models/shapley.py` | 3 Shapley variants run, all 4 axioms satisfied |
| **12** | `models/markov.py` | 3 Markov variants run, removal effects positive, transition matrix valid |
| **13** | `models/constrained_opt.py` | 4 OR variants run, all constraints satisfied, weights non-negative |
| **14** | `comparison/head_to_head.py` | Pairwise metrics computed, primary models ρ > 0.70 |
| **15** | `comparison/convergence_map.py` | Per-channel confidence zones assigned |
| **16** | `comparison/dual_funnel.py` | Quote vs. bind comparison ready |
| **17** | `optimization/response_curves.py` | Curves calibrated for all optimizable channels |
| **18** | `optimization/budget_optimizer.py` | MILP solves, constraints satisfied, total budget = $5M |
| **19** | `optimization/scenario_engine.py` | 7 scenarios pre-computed |
| **20** | `optimization/shadow_prices.py` | Dual variables extracted and interpreted |
| **21** | `metrics/` (all 8 modules) | Every page's data needs are served from Parquet |
| **22** | `app/theme.py` + `app/data_store.py` | Theme constants, data loads correctly |
| **23** | `app/components/` (all) | Reusable components render without errors |
| **24** | `app/pages/` (all 8 + appendix) | Pages render, callbacks work, no hardcoded values |
| **25** | `scripts/run_full_pipeline.py` | End-to-end pipeline completes in <15 minutes |
| **26** | `Dockerfile` + `render.yaml` | Container builds and serves on port 8050 |

### 13.2 Critical Validation Checkpoints

After generating all code, verify:

1. **Zero hardcoded metrics in UI:** `grep -r "35%" app/pages/` should find ZERO matches. Every number must come from a DataFrame.
2. **Shapley efficiency axiom:** `abs(sum(channel_credits) - total_conversions) < 0.01%`
3. **Cross-model convergence:** Spearman ρ > 0.70 between all pairs of primary models
4. **Budget constraint satisfaction:** All 8 MILP constraints hold at optimum
5. **Data schema compliance:** All Parquet files match Section 5 schemas exactly
6. **Agent insight delivery:** Agent credit ≥ 30% in model-based attribution vs. ≤ 5% in last-digital-click

---

## 14. Appendices

### 14.1 Key Implementation Gotchas

1. **Shapley with 13 channels:** 2^13 = 8,192 coalitions. The Zhao et al. simplification (group by observed channel sets, impute unobserved) makes this tractable. Don't enumerate all 8,192 — only compute marginals for the ~800-1,200 sets actually observed in data.

2. **Markov 2nd-order state space:** 13 channels → up to 169 compound states. Only ~40-60 are observed with sufficient frequency. Apply Dirichlet smoothing with α = 1/13.

3. **Constrained optimization solver:** SciPy SLSQP can be sensitive to initial values. Use w0 = 1/N (uniform) and verify convergence. If SLSQP fails, fall back to CVXPY which handles constraints more robustly.

4. **MILP piecewise linearization:** Use PuLP's built-in SOS2 constraints for piecewise linear functions, or implement the incremental formulation manually. Both give exact linearization of concave objectives.

5. **Dash callback cascading:** Avoid chains longer than 3 callbacks. Use `dcc.Store` for cross-page state. Use `prevent_initial_call=True` for expensive computations.

6. **Numbers must tie:** Every metric on the Executive Summary must exactly match the detailed view. Use `channel_credits.parquet` as the single source of truth. The `metrics/` layer reads from this file — pages read from `metrics/`.

7. **Chart axis scales:** When showing side-by-side charts for multiple models, all charts MUST use the same axis scale. Otherwise visual comparison is misleading.

8. **The agent insight is the climax:** Every design choice builds toward the moment agents jump from ~3% to ~35%. The grouped horizontal bar chart on Page 1 is the single most important visual. It should be immediately legible from across a conference room.

9. **Render.com memory:** Starter plan has 512MB RAM. Pre-compute everything to Parquet, don't hold large DataFrames in memory. Load only what each page needs via lazy filtering.

10. **Docker build time:** The `run_full_pipeline.py` step in the Dockerfile takes ~10-15 minutes. Render builds may timeout — consider running the pipeline locally, committing the `data/` directory, and building a simpler Docker image that just serves the app.

### 14.2 Alternative Deployment Strategy (if Render build times out)

```dockerfile
# Alternative: Don't generate data in Docker build.
# Instead, generate locally and include data/ in the repo.
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8050
CMD ["gunicorn", "app.app:server", "-b", "0.0.0.0:8050", "-w", "2", "--timeout", "120"]
```

In this approach:
1. Run `python scripts/run_full_pipeline.py` locally
2. Remove `data/` from `.gitignore`
3. Commit the generated Parquet files
4. Docker build just installs deps and copies code+data

This is faster to deploy and more reliable for demo purposes.

### 14.3 Erie-Specific Reference Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Agent involvement rate | 87% | Erie 100% agent model + industry benchmarks |
| Core markets | PA, OH | Erie HQ in Erie, PA |
| Annual marketing budget (estimated) | $5M | Proportional to Erie's ~$8B+ in direct premium |
| Brand awareness (PA) | 75% | Dominant regional brand in home state |
| Brand awareness (expansion) | 25% | Growing presence in newer markets |
| Avg auto premium | ~$1,000-1,200 | Erie known for competitive pricing |
| Agent count | ~13,000 | Independent agents across 12 states + DC |

---

*This document is the complete implementation specification for the Erie MCA Capability Demo. Every section provides sufficient detail for an LLM to produce working, production-quality code. The YAML configuration file (`erie_mca_synth_config.yaml`) is the companion artifact containing all parameter values.*

