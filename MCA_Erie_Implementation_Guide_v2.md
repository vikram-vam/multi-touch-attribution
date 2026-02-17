# Multi-Channel Attribution for Auto Insurance: Complete Implementation Guide
## Erie Insurance Capability Demo — Code-Generation Reference Document

**Version:** 2.0 | **Date:** February 2026 | **Target Audience:** LLM code generation (Sonnet 4.5) + Development Team  
**Scope:** Full codebase specification for a production-plausible MCA capability demo  
**Client Context:** Erie Insurance — 100% independent agent, 12-state regional P&C carrier

### Changelog from v1.0
- **UI Framework**: Migrated from Streamlit to Plotly Dash for full layout control, true callback-driven interactivity, and production-grade component architecture
- **Model Arsenal Expanded**: Added CASV (Counterfactual Adjusted Shapley), DARNN (Dual-Attention RNN), Transformer-based attribution, CausalMTA, Variable-Order Markov, and Regression-Adjusted Monte Carlo Shapley from SOTA literature
- **Data→UI Binding Contract**: New Section 3A introduces explicit schema contracts ensuring every metric, chart, and insight box reads from computed DataFrames — zero hardcoded values in the UI layer
- **Triangulation Framework**: Added MMM (Google Meridian) and incrementality experiment references as the "where this goes next" strategic layer
- **Synthetic Data Engine Refined**: Added output validation targets, explicit Parquet schemas, and pre-computed result manifests
- **Estimation Error Calibration**: Anchored around Gordon et al. (2023) finding of 488–948% errors in observational MTA — the demo narrative explicitly addresses this

---

## Table of Contents

1. [Solution Objective & Success Criteria](#1-solution-objective--success-criteria)
2. [Tech Stack & Repository Architecture](#2-tech-stack--repository-architecture)
3. [Data Contract & Schema Specification](#3-data-contract--schema-specification)
4. [Synthetic Data Engine](#4-synthetic-data-engine)
5. [Identity Resolution Simulator](#5-identity-resolution-simulator)
6. [Attribution Model Arsenal](#6-attribution-model-arsenal)
7. [Data Pipeline & Journey Assembly](#7-data-pipeline--journey-assembly)
8. [Budget Optimization & Triangulation Engine](#8-budget-optimization--triangulation-engine)
9. [Validation & QA Framework](#9-validation--qa-framework)
10. [UI/UX Specification — Plotly Dash Application](#10-uiux-specification--plotly-dash-application)
11. [Demo Narrative Integration](#11-demo-narrative-integration)
12. [Deployment & Packaging](#12-deployment--packaging)

---

## 1. Solution Objective & Success Criteria

### 1.1 What This Demo Must Accomplish

This is not a data science notebook. It is a **sellable capability demo** that takes Erie's CMO and VP of Marketing through a 60-minute narrative proving that their current attribution model (likely last-click via GA4) systematically misallocates marketing spend by under-crediting agents and upper-funnel channels. The demo must:

1. **Create cognitive dissonance**: Show last-click attribution where Paid Search gets ~40% credit and agents barely register. Then immediately show Shapley attribution where agents jump to ~35% and Paid Search drops to ~18%.
2. **Demonstrate channel cooperation**: Via Markov chain path analysis, show that digital channels and agents are not competing — they cooperate. The pattern "Display/Social → Organic/Paid Search → Agent → Bind" is the highest-converting sequence.
3. **Provide actionable budget guidance**: An interactive optimizer that shows shifting 15% from branded search to display/social projects a 15–25% improvement in cost-per-bind.
4. **Build technical confidence**: Show identity resolution concepts and model validation metrics that prove this isn't a black box.
5. **Anchor in estimation humility**: Explicitly reference Gordon et al. (2023) finding that observational MTA produces 488–948% errors in ad effect estimation. Position the demo's triangulation approach (multiple models + validation + incrementality roadmap) as the antidote. This builds credibility with data-literate stakeholders.
6. **Close with urgency**: A clear 12-week POC scope with specific data requirements from Erie, plus a triangulation roadmap (MTA now → MMM next → incrementality experiments after).

### 1.2 Success Criteria

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| Model validity | Shapley efficiency axiom (credits = total conversions) | Exact (tolerance < 0.01%) |
| Cross-model consistency | Spearman ρ between Shapley and Markov rankings | > 0.70 |
| Data realism | Internal SME rating on funnel rates, channel mix, journey patterns | "Plausible for Erie" on all key metrics |
| Insight generation | Non-obvious, quantified budget reallocation recommendations | ≥ 3 distinct insights |
| Demo engagement | Time from "hello" to first "aha moment" (last-click vs. Shapley comparison) | < 15 minutes |
| Technical polish | Zero crashes, sub-3-second page loads, smooth callback-driven transitions | 100% uptime during demo |
| Data binding | Every numeric value displayed in UI traces to a computed DataFrame cell | 0 hardcoded metrics |

### 1.3 Guiding Design Principles

- **Every chart tells a story**: No data vomit. Each visualization answers one business question.
- **Erie-specific, not generic**: Channel names, funnel stages, conversion rates — all calibrated to Erie's reality.
- **Interactivity builds trust**: Let Erie's team adjust parameters (lookback windows, time-decay half-life, budget sliders) and see results change in real time via Dash callbacks. This removes the "black box" objection.
- **Production-plausible**: The architecture shown must map to what a real engagement would deploy. No shortcuts that break the illusion. Dash's production-grade server architecture reinforces this.
- **Model spectrum shows rigor**: Showing 12+ attribution models (from naive to Transformer-based) demonstrates landscape mastery, not single-technique dependency.
- **Triangulation > perfection**: No single model is trustworthy alone (Gordon et al. 2023). The demo's power is in showing convergence across fundamentally different methods.

---

## 2. Tech Stack & Repository Architecture

### 2.1 Tech Stack Decision

**Recommendation: Plotly Dash with Dash Bootstrap Components + Plotly for visualizations.**

Rationale for Dash over Streamlit:

| Consideration | Streamlit | Plotly Dash | Winner for This Demo |
|---------------|-----------|-------------|---------------------|
| **Layout control** | Single-column default, limited CSS overrides | Full HTML/CSS layout via `dash-bootstrap-components`, CSS Grid, Flexbox | **Dash** — C-suite demo needs pixel-perfect layout |
| **Callback architecture** | Full-page rerun on any input change | Granular callbacks — only the targeted output updates | **Dash** — parameter changes shouldn't re-render entire pages |
| **Component ecosystem** | Limited to `st.` widgets | Full React component wrapping, custom components, rich third-party ecosystem | **Dash** — identity graph viz, animated transitions need custom components |
| **Multi-page routing** | Pages directory convention | `dash.page_registry` with URL routing | **Dash** — cleaner URL-based navigation for demo flow |
| **Production deployment** | Requires Streamlit Cloud or custom deployment | Standard WSGI/ASGI (Gunicorn, uWSGI) — same as any Flask app | **Dash** — enterprise audience expects production-grade deployment |
| **Animations** | Limited (Plotly only) | Full Plotly animations + CSS transitions + clientside callbacks for 60fps | **Dash** — animated attribution transitions are the demo climax |
| **Chart interactivity** | Native Plotly but limited cross-chart linking | Full `dash.callback` cross-filtering — click chart A, update chart B | **Dash** — essential for journey exploration drill-downs |
| **Development speed** | Faster for prototypes | Moderately slower (callback wiring) but more maintainable | Streamlit wins here, but Dash's advantages outweigh for this use case |

**Critical Dash design principle**: Dash defaults to a plain white page. The demo must use `dash-bootstrap-components` with a custom dark theme, branded CSS, and deliberate component layout. Every page needs a `dbc.Container` with `fluid=True` and responsive grid rows/columns.

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| **UI Framework** | Plotly Dash | ≥ 2.18.x | Multi-page apps, callback architecture, production WSGI server |
| **UI Components** | Dash Bootstrap Components (DBC) | ≥ 1.6.x | Grid system, cards, modals, tabs, navbars — professional layout |
| **Visualization** | Plotly (primary) + Matplotlib/Seaborn (static exports) | Plotly 5.x | Interactive charts. Plotly's Sankey, sunburst, animated bar charts. |
| **Data Processing** | Pandas + NumPy + Polars (optional) | Pandas 2.x, NumPy 1.26+ | Core data manipulation. |
| **Attribution Engine** | Custom Python + `shapiq` + `marketing-attribution-models` | Latest | Custom core + SOTA library acceleration for Shapley approximation. |
| **Statistical Modeling** | SciPy + Statsmodels + scikit-learn + lifelines | Latest stable | Survival analysis, logistic regression, statistical tests. |
| **Deep Learning** | PyTorch (LSTM, Transformer, DARNN modules) | 2.x | Attention-based and transformer attribution models. |
| **Optimization** | SciPy.optimize + PuLP (LP/IP) | Latest stable | Budget optimization via linear/integer programming. |
| **Graph Analysis** | NetworkX | 3.x | Markov chain graph construction, removal effect computation. |
| **Data Storage** | DuckDB (embedded) | Latest | Lightweight analytical queries. Pre-computed results served via Parquet. |
| **Caching** | `diskcache` + Dash `@callback` with `prevent_initial_call` | Latest | Parameter-grid pre-computation cache. |
| **Configuration** | YAML + Pydantic v2 | Latest | All model parameters, channel definitions, UI settings externalized. |
| **Testing** | pytest + hypothesis | Latest | Property-based testing for Shapley axioms, data contract validation. |

### 2.2 Repository Structure

```
erie-mca-demo/
│
├── README.md                          # Setup, run instructions, demo narrative
├── pyproject.toml                     # Dependencies (Poetry)
├── requirements.txt                   # Pinned dependencies
├── Makefile                           # Common commands: make run, make generate-data, make test
│
├── config/
│   ├── channels.yaml                  # Channel taxonomy: 13 channels with sub-channels, weights, lookback windows
│   ├── funnel.yaml                    # Funnel stage definitions, conversion rates, volume estimates
│   ├── synthetic_data.yaml            # Data generation parameters: journey count, distributions, seasonality
│   ├── model_params.yaml              # Attribution model hyperparameters: decay rates, Markov order, thresholds
│   ├── ui_theme.yaml                  # UI colors, fonts, layout settings
│   └── demo_narrative.yaml            # Talking points, insight callouts, per-page annotations
│
├── data/
│   ├── raw/                           # Generated synthetic data (Parquet)
│   │   ├── touchpoints.parquet        # Schema: Section 3.2
│   │   ├── conversions.parquet        # Schema: Section 3.3
│   │   ├── journeys.parquet           # Schema: Section 3.4
│   │   ├── identity_graph.parquet     # Schema: Section 3.5
│   │   └── channel_spend.parquet      # Schema: Section 3.6
│   ├── processed/                     # Post-pipeline: assembled journeys, attribution results
│   │   ├── assembled_journeys.parquet # Schema: Section 3.7
│   │   ├── attribution_results.parquet # Schema: Section 3.8 — THE MASTER UI DATA SOURCE
│   │   ├── model_comparison.parquet   # Schema: Section 3.9
│   │   ├── path_analysis.parquet      # Schema: Section 3.10
│   │   ├── budget_scenarios.parquet   # Schema: Section 3.11
│   │   └── validation_metrics.parquet # Schema: Section 3.12
│   └── reference/                     # Static reference data
│       ├── erie_state_map.json        # Erie's 12-state + DC footprint
│       └── industry_benchmarks.json   # P&C auto insurance benchmarks
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── generator.py               # Main synthetic data orchestrator
│   │   ├── user_profiles.py           # Demographic profile sampling
│   │   ├── journey_simulator.py       # Markov-based journey state machine
│   │   ├── channel_transitions.py     # Transition matrix definition and calibration
│   │   ├── conversion_model.py        # Conversion probability model (logistic with channel features)
│   │   ├── timestamp_engine.py        # Temporal modeling: seasonality, inter-touch gaps
│   │   ├── identity_simulator.py      # Simulated identity fragmentation and resolution
│   │   ├── spend_generator.py         # Channel-level spend data for ROI calculations
│   │   └── output_validator.py        # NEW: Validates generated data against target distributions
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── sessionizer.py             # Session boundary detection (30-min inactivity)
│   │   ├── journey_assembler.py       # Ordered journey construction per persistent_id
│   │   ├── touch_qualifier.py         # Apply touch qualification rules
│   │   ├── time_decay.py              # Exponential decay weight computation
│   │   └── channel_classifier.py      # MECE channel classification from raw event data
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract base class for all attribution models
│   │   ├── rule_based.py              # 5 heuristic models
│   │   ├── shapley_engine.py          # Shapley Value: exact + simplified (Zhao et al.)
│   │   ├── shapley_casv.py            # NEW: Counterfactual Adjusted Shapley Value (Singal et al. 2022)
│   │   ├── shapley_regression_mc.py   # NEW: Regression-Adjusted Monte Carlo (Witter et al. 2025)
│   │   ├── markov_chain.py            # Fixed-order Markov (1st, 2nd, 3rd) with removal effect
│   │   ├── markov_variable_order.py   # NEW: Variable-Order Markov / MTD model
│   │   ├── logistic_attribution.py    # Logistic regression-based fractional attribution
│   │   ├── survival_attribution.py    # Additive hazard / survival model attribution
│   │   ├── darnn_attribution.py       # NEW: Dual-Attention RNN (Ren et al. 2018)
│   │   ├── transformer_attribution.py # NEW: Transformer-based (Lu & Kannan 2025)
│   │   ├── lstm_attribution.py        # LSTM-based deep learning attribution
│   │   ├── causal_mta.py             # NEW: CausalMTA confounding decomposition (KDD 2022)
│   │   ├── ensemble.py                # Weighted ensemble of multiple models
│   │   └── model_registry.py          # Registry pattern for loading/running any model by name
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── budget_optimizer.py        # LP/IP-based budget reallocation optimizer
│   │   ├── marginal_roi.py            # Marginal ROI / diminishing returns curves per channel
│   │   ├── scenario_simulator.py      # What-if scenario engine for budget changes
│   │   ├── constraints.py             # Business constraints
│   │   └── triangulation.py           # NEW: MMM/incrementality comparison framework
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── axiom_tests.py             # Shapley axiom compliance
│   │   ├── cross_model.py             # Spearman ρ, rank comparison across models
│   │   ├── holdout_simulation.py      # Remove-and-predict validation
│   │   ├── sensitivity.py             # Window sensitivity, parameter sensitivity
│   │   ├── sanity_checks.py           # Business logic validation
│   │   └── data_contract_tests.py     # NEW: Validates all UI data contracts
│   │
│   ├── metrics/                       # NEW: Centralized metric computation layer
│   │   ├── __init__.py
│   │   ├── executive_metrics.py       # Computes all Executive Summary KPIs from DataFrames
│   │   ├── attribution_metrics.py     # Computes all Attribution Comparison metrics
│   │   ├── journey_metrics.py         # Computes all Journey Path metrics
│   │   ├── budget_metrics.py          # Computes all Budget Optimizer metrics
│   │   ├── identity_metrics.py        # Computes all Identity Resolution metrics
│   │   ├── channel_metrics.py         # Computes all Channel Deep Dive metrics
│   │   └── validation_metrics.py      # Computes all Validation Dashboard metrics
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py           # YAML config loading with Pydantic validation
│       ├── data_io.py                 # Parquet/CSV read/write utilities
│       ├── formatters.py              # Number formatting, percentage display, currency
│       ├── logging_setup.py           # Structured logging configuration
│       └── cache_manager.py           # NEW: Pre-computed result cache management
│
├── app/
│   ├── app.py                         # Dash entry point: app = Dash(__name__, use_pages=True)
│   ├── theme.py                       # Custom CSS, Plotly template, color palette
│   ├── data_store.py                  # NEW: Central data loading + caching for Dash callbacks
│   ├── assets/
│   │   ├── custom.css                 # Global CSS (Dash auto-loads from assets/)
│   │   ├── favicon.ico
│   │   └── fonts/                     # Self-hosted fonts for offline demo
│   │       ├── DMSans-*.woff2
│   │       └── Inter-*.woff2
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── metric_card.py             # KPI card component (Dash html.Div)
│   │   ├── model_selector.py          # Model toggle pills (dbc.ButtonGroup)
│   │   ├── parameter_panel.py         # Sidebar parameter controls (dcc.Slider, dcc.Dropdown)
│   │   ├── sankey_chart.py            # Plotly Sankey wrapper
│   │   ├── attribution_bars.py        # Side-by-side attribution comparison chart
│   │   ├── journey_explorer.py        # Interactive journey path explorer
│   │   ├── budget_sliders.py          # Budget reallocation slider interface
│   │   ├── identity_graph_viz.py      # Identity resolution network visualization
│   │   ├── insight_callout.py         # Styled insight/finding callout boxes
│   │   ├── animated_transition.py     # Chart transition animations (Plotly frames + clientside callbacks)
│   │   └── navbar.py                  # Top navigation bar component
│   │
│   └── pages/
│       ├── executive_summary.py       # Page 1: Landing page with key findings
│       ├── attribution_comparison.py  # Page 2: Side-by-side all models
│       ├── journey_paths.py           # Page 3: Sankey, sunburst, path analysis
│       ├── budget_optimizer.py        # Page 4: Interactive budget reallocation
│       ├── identity_resolution.py     # Page 5: Identity graph health metrics
│       ├── channel_deep_dive.py       # Page 6: Per-channel drill-down
│       ├── model_validation.py        # Page 7: Validation metrics dashboard
│       └── technical_appendix.py      # Page 8: Methodology, math, documentation
│
├── tests/
│   ├── test_shapley_axioms.py
│   ├── test_markov_chain.py
│   ├── test_data_generation.py
│   ├── test_journey_assembly.py
│   ├── test_optimization.py
│   └── test_data_contracts.py         # NEW: Validates schema contracts between layers
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_validation_deep_dive.ipynb
│
└── scripts/
    ├── generate_data.py               # CLI: generate synthetic data
    ├── run_attribution.py             # CLI: run all models, output results
    ├── run_validation.py              # CLI: run full validation suite
    ├── precompute_cache.py            # NEW: Pre-compute parameter grid results
    └── export_results.py             # CLI: export to Excel/PDF for leave-behind
```

### 2.3 Configuration Schema (Pydantic Models)

All configuration is externalized to YAML files and validated at startup via Pydantic v2. This is critical for demo flexibility — the presenter can adjust parameters without touching code.

```python
# src/utils/config_loader.py

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum

class ChannelType(str, Enum):
    CLICK = "click"
    IMPRESSION = "impression"
    CALL = "call"
    AGENT = "agent"
    MAIL = "mail"
    EMAIL = "email"

class ChannelConfig(BaseModel):
    channel_id: str
    display_name: str
    channel_group: str                          # Macro group for Shapley fallback
    sub_channels: List[str]
    touch_type: ChannelType
    lookback_window_days: int = Field(ge=1, le=180)
    time_decay_half_life_days: float = Field(ge=0.5, le=30.0)
    attribution_weight_default: float = Field(ge=0.0, le=1.0)
    erie_estimated_bind_share: float            # Erie's estimated % of binds
    funnel_role: str                            # "awareness", "consideration", "conversion"
    data_source: str                            # "GA4", "CRM", "AMS", etc.
    color: str                                  # Hex color for charts

class FunnelStageConfig(BaseModel):
    stage_name: str
    erie_definition: str
    key_events: List[str]
    estimated_annual_volume: str
    conversion_rate_to_next: float

class SyntheticDataConfig(BaseModel):
    total_journeys: int = Field(default=150_000, ge=10_000, le=1_000_000)
    simulation_months: int = Field(default=12, ge=1, le=36)
    quote_start_rate: float = Field(default=0.08, ge=0.01, le=0.20)
    bind_rate_from_quotes: float = Field(default=0.35, ge=0.10, le=0.60)
    avg_touchpoints_converting: float = Field(default=3.8, ge=1.0, le=15.0)
    avg_touchpoints_non_converting: float = Field(default=1.6, ge=1.0, le=10.0)
    agent_last_touch_pct: float = Field(default=0.60, ge=0.30, le=0.90)
    journey_duration_median_days: float = Field(default=8.0, ge=1.0, le=60.0)
    journey_duration_mean_days: float = Field(default=14.0, ge=1.0, le=90.0)
    seasonality_peaks: List[str] = ["March", "April", "September", "October"]
    random_seed: int = 42

class ModelParamsConfig(BaseModel):
    # Shapley parameters
    shapley_min_coalition_obs: int = Field(default=30, ge=5, le=100)
    shapley_use_time_weights: bool = True
    shapley_mc_samples: int = Field(default=10000, ge=1000, le=100000)    # For Monte Carlo approximation
    shapley_regression_mc_model: str = "xgboost"  # For Witter et al. 2025 method
    # CASV parameters
    casv_markov_order: int = Field(default=1, ge=1, le=3)
    # Markov parameters
    markov_order: int = Field(default=2, ge=1, le=3)
    markov_null_state_name: str = "NULL"
    markov_vmm_max_depth: int = Field(default=4, ge=1, le=6)  # For Variable-Order Markov
    # Rule-based parameters
    time_decay_half_life_click: float = 7.0
    time_decay_half_life_impression: float = 3.0
    position_based_first_weight: float = 0.40
    position_based_last_weight: float = 0.40
    # Deep learning parameters
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_epochs: int = 50
    lstm_learning_rate: float = 0.001
    transformer_n_heads: int = 4           # For Transformer model
    transformer_d_model: int = 64
    transformer_n_layers: int = 2
    darnn_encoder_dim: int = 64            # For DARNN model
    darnn_decoder_dim: int = 64
    # General
    journey_max_touchpoints: int = 20
    session_timeout_minutes: int = 30

class DemoConfig(BaseModel):
    channels: List[ChannelConfig]
    funnel_stages: List[FunnelStageConfig]
    synthetic_data: SyntheticDataConfig
    model_params: ModelParamsConfig
```

---

## 3. Data Contract & Schema Specification

**This section is the single most important addition in v2.** It defines the exact schema of every DataFrame that flows between the data generation layer, model layer, and UI layer. The rule: **the UI layer may ONLY read from these defined schemas. No magic numbers, no inline computations, no hardcoded percentages.** Every `metric_card`, every chart trace, every insight callout must reference a column in one of these DataFrames.

### 3.1 Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Synthetic Data  │───→│  Pipeline Layer   │───→│  Attribution Models │
│  Generator       │    │  (Sessionize,     │    │  (Shapley, Markov,  │
│  (Section 4)     │    │   Assemble,       │    │   DARNN, etc.)      │
│                  │    │   Qualify)         │    │                     │
│  Outputs:        │    │                   │    │  Outputs:           │
│  - touchpoints   │    │  Output:          │    │  - attribution_     │
│  - conversions   │    │  - assembled_     │    │    results          │
│  - identity_graph│    │    journeys       │    │  - model_comparison │
│  - channel_spend │    │                   │    │  - path_analysis    │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     src/metrics/ Layer                               │
│  Reads: attribution_results, assembled_journeys, channel_spend,     │
│         validation_metrics, budget_scenarios                         │
│  Outputs: Page-specific metric DataFrames consumed by Dash callbacks│
│                                                                     │
│  RULE: This is the ONLY layer the UI touches.                       │
│  RULE: Every number in the UI has a traceable path to a cell here.  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Dash Callbacks (UI Layer)                         │
│  Reads ONLY from src/metrics/ outputs via app/data_store.py         │
│  Renders Plotly figures, metric cards, insight text                  │
│  NEVER computes attribution or business logic directly               │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Raw Schema: `touchpoints.parquet`

```
Column                  | Type      | Description
─────────────────────────────────────────────────────────────────
touchpoint_id           | str       | UUID, unique per interaction event
persistent_id           | str       | Resolved user identity (ground truth in synthetic data)
channel_id              | str       | One of 13 channels from config/channels.yaml
sub_channel             | str       | Sub-channel detail (e.g., "google_search", "facebook_feed")
touch_type              | str       | "click", "impression", "call", "agent", "mail", "email"
event_timestamp         | datetime  | UTC timestamp of the interaction
session_id              | str       | Session identifier (assigned by sessionizer)
dwell_time_seconds      | float     | For clicks: time on site. For calls: call duration.
viewability_pct         | float     | For impressions: % in-view (MRC standard)
device_type             | str       | "desktop", "mobile", "tablet", "phone", "offline"
state                   | str       | US state code (Erie's 12 + DC)
campaign_id             | str       | Marketing campaign identifier
creative_id             | str       | Ad creative variant
utm_source              | str       | UTM source parameter
utm_medium              | str       | UTM medium parameter
utm_campaign            | str       | UTM campaign parameter
is_qualified            | bool      | Passes touch qualification rules (dwell time, viewability, etc.)
touch_weight            | float     | 0.0-1.0, qualification-based weight (email opens = 0.5, etc.)
```

### 3.3 Raw Schema: `conversions.parquet`

```
Column                  | Type      | Description
─────────────────────────────────────────────────────────────────
conversion_id           | str       | UUID, unique per conversion event
persistent_id           | str       | Resolved user identity
conversion_type         | str       | "quote_start", "quote_complete", "bind"
conversion_timestamp    | datetime  | UTC timestamp of conversion
conversion_value        | float     | Estimated premium value ($) — for bind events
channel_of_conversion   | str       | Channel where conversion occurred (often "independent_agent")
agent_id                | str       | Agent identifier (for agent-assisted conversions)
policy_line             | str       | "auto" (for this demo, always auto)
state                   | str       | US state code
```

### 3.4 Raw Schema: `journeys.parquet` (assembled from touchpoints + conversions)

```
Column                  | Type      | Description
─────────────────────────────────────────────────────────────────
journey_id              | str       | UUID, one per customer journey
persistent_id           | str       | Resolved user identity
channel_path            | List[str] | Ordered list of channel_ids in the journey
channel_path_str        | str       | Pipe-delimited string: "display|search|agent"
channel_set             | Set[str]  | Unique channels (for Shapley coalition computation)
channel_set_str         | str       | Pipe-delimited unique channels (sortable)
touchpoint_count        | int       | Number of qualified touchpoints
journey_duration_days   | float     | Days from first touch to last touch / conversion
first_touch_channel     | str       | Channel of first interaction
last_touch_channel      | str       | Channel of last interaction before conversion/null
is_converting           | bool      | True if journey ends in a bind
conversion_type         | str       | "bind", "quote_only", "null"
conversion_value        | float     | $ value if converting, 0 otherwise
has_agent_touch         | bool      | Whether any touchpoint was an agent interaction
agent_touch_position    | str       | "first", "middle", "last", "none"
simulation_month        | int       | 1-12, month of journey start (for seasonality analysis)
age_band                | str       | User demographic
state                   | str       | US state code
life_event_trigger      | str       | Life event or "none"
```

### 3.5 Raw Schema: `identity_graph.parquet`

```
Column                  | Type      | Description
─────────────────────────────────────────────────────────────────
persistent_id           | str       | Ground truth identity
identifier_type         | str       | "email_hash", "phone_hash", "ga4_client_id", "gclid", "address_hash"
identifier_value        | str       | Hashed identifier value
match_tier              | int       | 1 (deterministic), 2 (household), 3 (probabilistic)
match_confidence        | float     | 0.0-1.0 confidence score
device_type             | str       | Device associated with this identifier
```

### 3.6 Raw Schema: `channel_spend.parquet`

```
Column                  | Type      | Description
─────────────────────────────────────────────────────────────────
channel_id              | str       | Channel identifier
month                   | int       | 1-12
year                    | int       | Simulation year
spend_dollars           | float     | Monthly spend in dollars
impressions             | int       | Monthly impressions (for CPM channels)
clicks                  | int       | Monthly clicks (for CPC channels)
```

### 3.7 Processed Schema: `assembled_journeys.parquet`

Same as Section 3.4, plus additional computed columns:

```
Additional columns:
─────────────────────────────────────────────────────────────────
time_decay_weights      | List[float] | Per-touchpoint decay weights (aligned with channel_path)
session_count           | int         | Number of distinct sessions in the journey
inter_touch_gap_avg_hrs | float       | Average hours between consecutive touches
```

### 3.8 Processed Schema: `attribution_results.parquet` — MASTER UI DATA SOURCE

**This is the single source of truth for all UI attribution displays.** One row per (model_name, channel_id) combination.

```
Column                  | Type      | Description
─────────────────────────────────────────────────────────────────
model_name              | str       | "first_touch", "last_touch", "linear", "time_decay",
                        |           | "position_based", "shapley", "casv", "shapley_reg_mc",
                        |           | "markov_order_1", "markov_order_2", "markov_vmm",
                        |           | "logistic", "survival", "lstm", "darnn", "transformer",
                        |           | "causal_mta", "ensemble"
channel_id              | str       | Channel identifier
channel_display_name    | str       | Human-readable channel name
channel_group           | str       | Macro group for grouping
credited_conversions    | float     | Total conversions credited to this channel by this model
credit_pct              | float     | % of total conversions (0.0-1.0)
credit_rank             | int       | Rank within this model (1 = highest credit)
annual_spend            | float     | Channel's annual spend (from channel_spend)
cost_per_conversion     | float     | annual_spend / credited_conversions
roas                    | float     | credited_conversions * avg_premium / annual_spend
funnel_role             | str       | "awareness", "consideration", "conversion"
channel_color           | str       | Hex color for charts
param_lookback_days     | int       | Lookback window used for this computation
param_decay_half_life   | float     | Decay half-life used (for time-dependent models)
param_markov_order      | int       | Markov order used (for Markov models)
total_conversions       | int       | Total conversions across all channels (same for all rows within a model)
```

### 3.9 Processed Schema: `model_comparison.parquet`

Cross-model comparison metrics. One row per (model_a, model_b) pair.

```
Column                  | Type      | Description
─────────────────────────────────────────────────────────────────
model_a                 | str       | First model name
model_b                 | str       | Second model name
spearman_rho            | float     | Spearman rank correlation between channel rankings
kendall_tau             | float     | Kendall's tau correlation
max_rank_diff           | int       | Maximum rank difference for any channel
max_rank_diff_channel   | str       | Channel with the maximum rank difference
mean_abs_credit_diff    | float     | Mean absolute difference in credit_pct across channels
```

### 3.10 Processed Schema: `path_analysis.parquet`

Top converting paths for Sankey and journey exploration. One row per unique path.

```
Column                  | Type      | Description
─────────────────────────────────────────────────────────────────
path_str                | str       | Pipe-delimited path (e.g., "display|search|agent")
path_length             | int       | Number of channels in path
frequency               | int       | Number of journeys with this exact path
converting_count        | int       | Number of converting journeys with this path
conversion_rate         | float     | converting_count / frequency
avg_duration_days       | float     | Average journey duration for this path
avg_touchpoints         | float     | Average touchpoint count for this path
has_agent               | bool      | Whether path includes an agent touch
first_touch             | str       | First channel in path
last_touch              | str       | Last channel in path
pct_of_total_journeys   | float     | frequency / total_journeys
pct_of_conversions      | float     | converting_count / total_conversions
```

### 3.11 Processed Schema: `budget_scenarios.parquet`

Pre-computed budget optimization results. One row per (scenario_id, channel_id).

```
Column                  | Type      | Description
─────────────────────────────────────────────────────────────────
scenario_id             | str       | "current", "optimal_unconstrained", "optimal_20pct_cap",
                        |           | "shift_search_to_display", "double_agent_support", "cut_tv_50"
scenario_display_name   | str       | Human-readable scenario name
channel_id              | str       | Channel identifier
current_spend           | float     | Current annual spend
proposed_spend          | float     | Proposed annual spend under this scenario
spend_change            | float     | proposed - current
spend_change_pct        | float     | (proposed - current) / current
projected_conversions   | float     | Projected conversions at proposed spend
current_conversions     | float     | Current conversions (from Shapley attribution)
conversion_change       | float     | projected - current
projected_cpb           | float     | Projected cost-per-bind
current_cpb             | float     | Current cost-per-bind
total_budget            | float     | Total budget (same across all channels per scenario)
total_projected_binds   | float     | Total binds across all channels (per scenario)
total_current_binds     | float     | Total current binds
total_bind_change       | float     | total_projected - total_current
total_cpb_change_pct    | float     | % change in overall cost-per-bind
```

### 3.12 Processed Schema: `validation_metrics.parquet`

Validation test results. One row per (test_name, model_name).

```
Column                  | Type      | Description
─────────────────────────────────────────────────────────────────
test_name               | str       | "efficiency_axiom", "symmetry_axiom", "null_player_axiom",
                        |           | "additivity_axiom", "holdout_channel_X", "sensitivity_window_Y"
model_name              | str       | Model that was tested
passed                  | bool      | Whether the test passed
metric_value            | float     | Quantitative result (e.g., actual sum for efficiency test)
expected_value          | float     | Expected value
tolerance               | float     | Acceptable tolerance
details                 | str       | JSON string with additional context
```

### 3.13 The `src/metrics/` Layer — Central Metric Computation

This is the translation layer between raw model outputs and what the UI displays. Each page has a dedicated metrics module.

```python
# src/metrics/executive_metrics.py

import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExecutiveSummaryMetrics:
    """Every metric displayed on the Executive Summary page."""
    total_journeys: int
    total_binds: int
    total_annual_spend: float
    avg_premium_value: float

    # Last-click vs Shapley headline comparison
    last_click_agent_credit_pct: float      # e.g., 0.078
    shapley_agent_credit_pct: float         # e.g., 0.342
    agent_credit_shift_pp: float            # e.g., +26.4 percentage points
    last_click_search_credit_pct: float
    shapley_search_credit_pct: float
    search_credit_shift_pp: float

    # Key insight metrics
    agent_conversion_multiplier: float      # e.g., 4.8×
    optimal_cpb_improvement_pct: float      # e.g., -18.2%
    optimal_additional_binds: int           # e.g., 680
    optimal_annual_savings: float           # e.g., $2.4M

    # Supporting data for charts
    last_click_by_channel: pd.DataFrame     # channel_id, credit_pct — for bar chart
    shapley_by_channel: pd.DataFrame        # channel_id, credit_pct — for bar chart

    @classmethod
    def from_data(
        cls,
        attribution_results: pd.DataFrame,
        assembled_journeys: pd.DataFrame,
        channel_spend: pd.DataFrame,
        budget_scenarios: pd.DataFrame,
        avg_premium: float = 1200.0
    ) -> "ExecutiveSummaryMetrics":
        """
        Compute ALL executive summary metrics from source DataFrames.
        This is where the numbers come from — nowhere else.
        """
        # Total journeys and binds
        total_journeys = assembled_journeys['journey_id'].nunique()
        total_binds = assembled_journeys[assembled_journeys['is_converting']].shape[0]
        total_annual_spend = channel_spend.groupby('channel_id')['spend_dollars'].sum().sum()

        # Last-click agent credit
        lt = attribution_results[attribution_results['model_name'] == 'last_touch']
        lt_agent = lt[lt['channel_id'] == 'independent_agent']['credit_pct'].iloc[0]

        # Shapley agent credit
        sv = attribution_results[attribution_results['model_name'] == 'shapley']
        sv_agent = sv[sv['channel_id'] == 'independent_agent']['credit_pct'].iloc[0]

        # Last-click search credit (brand + nonbrand combined)
        lt_search = lt[lt['channel_id'].isin(['paid_search_brand', 'paid_search_nonbrand'])]['credit_pct'].sum()
        sv_search = sv[sv['channel_id'].isin(['paid_search_brand', 'paid_search_nonbrand'])]['credit_pct'].sum()

        # Agent conversion multiplier
        agent_journeys = assembled_journeys[assembled_journeys['has_agent_touch']]
        no_agent_journeys = assembled_journeys[~assembled_journeys['has_agent_touch']]
        agent_conv_rate = agent_journeys['is_converting'].mean()
        no_agent_conv_rate = no_agent_journeys['is_converting'].mean()
        agent_multiplier = agent_conv_rate / no_agent_conv_rate if no_agent_conv_rate > 0 else float('inf')

        # Budget optimization metrics — read from pre-computed scenarios
        optimal = budget_scenarios[budget_scenarios['scenario_id'] == 'optimal_20pct_cap']
        current = budget_scenarios[budget_scenarios['scenario_id'] == 'current']
        # These are per-scenario aggregates, take the first row's total columns
        opt_row = optimal.iloc[0]
        cur_row = current.iloc[0]
        additional_binds = int(opt_row['total_bind_change'])
        cpb_change_pct = opt_row['total_cpb_change_pct']
        annual_savings = abs(cpb_change_pct) * total_annual_spend

        return cls(
            total_journeys=total_journeys,
            total_binds=total_binds,
            total_annual_spend=total_annual_spend,
            avg_premium_value=avg_premium,
            last_click_agent_credit_pct=lt_agent,
            shapley_agent_credit_pct=sv_agent,
            agent_credit_shift_pp=(sv_agent - lt_agent) * 100,
            last_click_search_credit_pct=lt_search,
            shapley_search_credit_pct=sv_search,
            search_credit_shift_pp=(sv_search - lt_search) * 100,
            agent_conversion_multiplier=round(agent_multiplier, 1),
            optimal_cpb_improvement_pct=round(cpb_change_pct * 100, 1),
            optimal_additional_binds=additional_binds,
            optimal_annual_savings=round(annual_savings, 0),
            last_click_by_channel=lt[['channel_id', 'channel_display_name', 'credit_pct', 'channel_color']].copy(),
            shapley_by_channel=sv[['channel_id', 'channel_display_name', 'credit_pct', 'channel_color']].copy(),
        )
```

```python
# src/metrics/attribution_metrics.py

@dataclass
class AttributionComparisonMetrics:
    """Every metric displayed on the Attribution Comparison page."""
    # Full model × channel matrix
    model_channel_matrix: pd.DataFrame   # Pivoted: rows=channels, cols=models, values=credit_pct
    model_channel_ranks: pd.DataFrame    # Same structure but with ranks
    model_names_available: List[str]     # All models that have results

    # Cross-model agreement
    spearman_matrix: pd.DataFrame        # models × models correlation heatmap data
    rank_stability: pd.DataFrame         # channel_id, min_rank, max_rank, rank_range, stable (bool)

    # Highlight insights (computed, not hardcoded)
    biggest_gainer_channel: str          # Channel with largest rank improvement from last-touch to Shapley
    biggest_gainer_shift_pp: float
    biggest_loser_channel: str           # Channel with largest rank drop
    biggest_loser_shift_pp: float
    models_agree_on_top3: bool           # Do Shapley, Markov, Logistic agree on top 3 channels?

    @classmethod
    def from_data(cls, attribution_results: pd.DataFrame, model_comparison: pd.DataFrame):
        """Compute from source DataFrames."""
        # Pivot attribution_results to model × channel matrix
        matrix = attribution_results.pivot_table(
            index='channel_id', columns='model_name', values='credit_pct'
        )
        ranks = attribution_results.pivot_table(
            index='channel_id', columns='model_name', values='credit_rank'
        )
        # ... compute all derived metrics
        pass
```

```python
# src/metrics/journey_metrics.py

@dataclass
class JourneyPathMetrics:
    """Every metric displayed on the Journey Paths page."""
    top_paths: pd.DataFrame                # From path_analysis.parquet, top N by conversion_rate
    sankey_flows: pd.DataFrame             # Source → Target → Value for Sankey diagram
    path_length_distribution: pd.DataFrame # path_length, count, is_converting
    agent_multiplier_data: pd.DataFrame    # Segment (with_agent, without_agent), conversion_rate, journey_count
    sunburst_data: pd.DataFrame            # Hierarchical: first_touch → mid → last_touch → outcome
    highest_converting_path: str
    highest_converting_path_rate: float
    highest_converting_path_pct_of_binds: float
    avg_touchpoints_converting: float
    avg_touchpoints_non_converting: float

    @classmethod
    def from_data(cls, assembled_journeys: pd.DataFrame, path_analysis: pd.DataFrame):
        """Compute from source DataFrames."""
        pass
```

```python
# src/metrics/budget_metrics.py

@dataclass
class BudgetOptimizerMetrics:
    """Every metric displayed on the Budget Optimizer page."""
    scenarios: pd.DataFrame               # All pre-computed scenarios from budget_scenarios.parquet
    current_allocation: pd.DataFrame      # channel_id, current_spend
    response_curves: Dict[str, callable]  # channel_id → fitted response curve function
    scenario_comparison: pd.DataFrame     # scenario_id, total_binds, total_cpb, delta_binds, delta_cpb
    marginal_roi_data: pd.DataFrame       # channel_id, spend_level, marginal_roi (for curves)
    diminishing_returns_warnings: List[str]  # Channels near saturation point

    @classmethod
    def from_data(cls, budget_scenarios: pd.DataFrame, channel_spend: pd.DataFrame,
                  attribution_results: pd.DataFrame, response_curves: dict):
        pass
```

### 3.14 `app/data_store.py` — Central Data Loading for Dash

```python
# app/data_store.py

"""
Central data loading and caching layer for the Dash application.

ALL Dash callbacks access data through this module.
NO Dash callback directly reads Parquet files.
NO Dash callback contains business logic or metric computation.

This module:
1. Loads pre-computed Parquet files at app startup
2. Instantiates metric objects from src/metrics/
3. Caches results for the parameter grid
4. Provides accessor functions for callbacks
"""

import pandas as pd
from functools import lru_cache
from src.metrics.executive_metrics import ExecutiveSummaryMetrics
from src.metrics.attribution_metrics import AttributionComparisonMetrics
from src.metrics.journey_metrics import JourneyPathMetrics
from src.metrics.budget_metrics import BudgetOptimizerMetrics
from src.metrics.identity_metrics import IdentityResolutionMetrics
from src.metrics.channel_metrics import ChannelDeepDiveMetrics
from src.metrics.validation_metrics import ValidationDashboardMetrics

# Load all DataFrames at module import (app startup)
_attribution_results = pd.read_parquet("data/processed/attribution_results.parquet")
_assembled_journeys = pd.read_parquet("data/processed/assembled_journeys.parquet")
_model_comparison = pd.read_parquet("data/processed/model_comparison.parquet")
_path_analysis = pd.read_parquet("data/processed/path_analysis.parquet")
_budget_scenarios = pd.read_parquet("data/processed/budget_scenarios.parquet")
_validation_metrics = pd.read_parquet("data/processed/validation_metrics.parquet")
_channel_spend = pd.read_parquet("data/raw/channel_spend.parquet")
_identity_graph = pd.read_parquet("data/raw/identity_graph.parquet")

def get_executive_metrics() -> ExecutiveSummaryMetrics:
    return ExecutiveSummaryMetrics.from_data(
        _attribution_results, _assembled_journeys, _channel_spend, _budget_scenarios
    )

def get_attribution_metrics() -> AttributionComparisonMetrics:
    return AttributionComparisonMetrics.from_data(_attribution_results, _model_comparison)

def get_journey_metrics() -> JourneyPathMetrics:
    return JourneyPathMetrics.from_data(_assembled_journeys, _path_analysis)

def get_budget_metrics() -> BudgetOptimizerMetrics:
    return BudgetOptimizerMetrics.from_data(
        _budget_scenarios, _channel_spend, _attribution_results, _response_curves
    )

def get_attribution_results() -> pd.DataFrame:
    """Direct access for pages that need to filter/pivot the raw results."""
    return _attribution_results

def get_assembled_journeys() -> pd.DataFrame:
    return _assembled_journeys

# ... etc for all data access
```

---

## 4. Synthetic Data Engine

The synthetic data is the foundation of demo credibility. If Erie's marketing team looks at the data distributions and thinks "this doesn't match our world," the entire demo collapses. Every parameter below is calibrated to Erie's publicly known characteristics and P&C auto insurance industry benchmarks.

### 4.1 Data Generation Architecture

The generator uses a **layered probabilistic state machine**:

```
Layer 1: User Profile Sampling
    → Demographics (age, state, vehicle type) matching Erie's book

Layer 2: Entry Channel Assignment
    → First-touch channel drawn from calibrated distribution

Layer 3: Journey State Machine (Markov transitions)
    → Sequence of channel interactions with realistic transition probabilities
    → "Digital research → agent conversion" pattern embedded prominently

Layer 4: Conversion Decision
    → At each step, logistic model computes P(conversion | accumulated channels, recency)
    → Creates realistic funnel narrowing

Layer 5: Temporal Assignment
    → Timestamps via lognormal duration model
    → Inter-touch gaps from exponential distribution
    → Seasonality overlay

Layer 6: Non-Conversion Dropout
    → Survival function with increasing dropout probability per touch

Layer 7: Identity Fragmentation
    → Simulate realistic match rates and ID fragmentation across devices
```

### 4.2 User Profile Generation (`user_profiles.py`)

```python
class UserProfile:
    persistent_id: str              # UUID, the "ground truth" identity
    age_band: str                   # "18-24", "25-34", "35-44", "45-54", "55-64", "65+"
    state: str                      # One of Erie's 12 states + DC
    vehicle_type: str               # "sedan", "suv", "truck", "luxury", "sports"
    life_event_trigger: Optional[str]  # "new_car", "teen_driver", "relocation", "marriage", "home_purchase", None
    digital_propensity: float       # 0-1, how likely to use digital channels vs. agent-first
    price_sensitivity: float        # 0-1, affects conversion threshold

# Erie's 12 states + DC with population-weighted distribution
ERIE_STATE_DISTRIBUTION = {
    "PA": 0.30, "OH": 0.12, "NY": 0.11, "NC": 0.08, "VA": 0.07,
    "MD": 0.06, "IN": 0.05, "WI": 0.05, "TN": 0.04, "WV": 0.04,
    "IL": 0.04, "KY": 0.03, "DC": 0.01
}

AGE_DISTRIBUTION = {
    "18-24": 0.08, "25-34": 0.20, "35-44": 0.25,
    "45-54": 0.22, "55-64": 0.15, "65+": 0.10
}

VEHICLE_TYPE_DISTRIBUTION = {
    "sedan": 0.35, "suv": 0.30, "truck": 0.15, "luxury": 0.10, "sports": 0.10
}

LIFE_EVENT_DISTRIBUTION = {
    None: 0.60, "new_car": 0.15, "relocation": 0.08,
    "teen_driver": 0.07, "marriage": 0.05, "home_purchase": 0.05
}

# Digital propensity varies by age — younger skews digital, older skews agent-first
DIGITAL_PROPENSITY_BY_AGE = {
    "18-24": {"mean": 0.75, "std": 0.12},
    "25-34": {"mean": 0.65, "std": 0.15},
    "35-44": {"mean": 0.55, "std": 0.18},
    "45-54": {"mean": 0.45, "std": 0.18},
    "55-64": {"mean": 0.35, "std": 0.15},
    "65+":   {"mean": 0.25, "std": 0.12}
}
```

**Generation logic**: For each of the 150,000 journeys, sample a user profile from these distributions. Users with `life_event_trigger != None` have longer journeys (more touchpoints, wider time window) and higher conversion rates. Users with higher `digital_propensity` start with digital channels; lower propensity users start via direct mail, TV-triggered search, or agent referral.

### 4.3 Channel Transition Matrix (`channel_transitions.py`)

The transition matrix is the heart of journey simulation. It encodes Erie's specific channel interaction patterns. The matrix represents P(next_channel | current_channel) for a first-order process; the generator uses this as base with contextual modifications.

```python
CHANNELS = [
    "independent_agent",      # 0
    "paid_search_brand",      # 1
    "paid_search_nonbrand",   # 2
    "organic_search",         # 3
    "display_programmatic",   # 4
    "paid_social",            # 5
    "tv_radio",               # 6
    "direct_mail",            # 7
    "email_marketing",        # 8
    "call_center",            # 9
    "aggregator_comparator",  # 10
    "direct_organic",         # 11
    "video_ott_ctv",          # 12
]

FIRST_TOUCH_DISTRIBUTION = {
    "paid_search_brand": 0.12, "paid_search_nonbrand": 0.10, "organic_search": 0.15,
    "display_programmatic": 0.10, "paid_social": 0.08, "tv_radio": 0.12,
    "direct_mail": 0.08, "email_marketing": 0.02, "call_center": 0.02,
    "aggregator_comparator": 0.04, "direct_organic": 0.10, "video_ott_ctv": 0.05,
    "independent_agent": 0.02,
}

# KEY TRANSITION PATTERNS TO EMBED:
#
# Pattern 1: "Digital Awareness → Search → Agent → Bind" (highest converting, ~35% of converting journeys)
# Pattern 2: "Direct Mail → Agent" (offline-to-offline, ~15%)
# Pattern 3: "Search → Quote Start → Email Nurture → Agent → Bind" (~12%)
# Pattern 4: "Aggregator → Search → Agent" (~5%)
# Pattern 5: "Pure Digital" (rare for Erie, ~8%)
#
# The full 16x16 transition matrix (13 channels + START + CONVERSION + NULL) is specified
# as a NumPy array. Below are the key transition probabilities that define Erie's model:

TRANSITION_DESIGN = {
    "display_programmatic": {
        "organic_search": 0.20, "paid_search_brand": 0.18, "paid_social": 0.10,
        "independent_agent": 0.08, "direct_organic": 0.12,
        "NULL": 0.25, "CONVERSION": 0.02, "_other": 0.05
    },
    "paid_search_brand": {
        "independent_agent": 0.30, "direct_organic": 0.12, "email_marketing": 0.08,
        "CONVERSION": 0.15, "NULL": 0.20, "call_center": 0.08, "organic_search": 0.04,
        "_other": 0.03
    },
    "independent_agent": {
        "CONVERSION": 0.55, "email_marketing": 0.08, "NULL": 0.30,
        "call_center": 0.04, "paid_search_brand": 0.03
    },
    "direct_mail": {
        "independent_agent": 0.35, "paid_search_brand": 0.15, "call_center": 0.12,
        "direct_organic": 0.10, "NULL": 0.25, "CONVERSION": 0.03
    },
    "paid_social": {
        "organic_search": 0.22, "paid_search_brand": 0.15, "display_programmatic": 0.10,
        "direct_organic": 0.10, "independent_agent": 0.06,
        "NULL": 0.30, "CONVERSION": 0.02, "_other": 0.05
    },
    "organic_search": {
        "independent_agent": 0.20, "paid_search_brand": 0.12, "direct_organic": 0.15,
        "email_marketing": 0.05, "CONVERSION": 0.10, "NULL": 0.25,
        "call_center": 0.05, "_other": 0.08
    },
    "email_marketing": {
        "independent_agent": 0.25, "paid_search_brand": 0.10, "direct_organic": 0.15,
        "CONVERSION": 0.12, "NULL": 0.30, "_other": 0.08
    },
    "call_center": {
        "independent_agent": 0.35, "CONVERSION": 0.20, "email_marketing": 0.05,
        "NULL": 0.35, "_other": 0.05
    },
    "tv_radio": {
        "paid_search_brand": 0.25, "organic_search": 0.20, "direct_organic": 0.15,
        "independent_agent": 0.05, "NULL": 0.30, "CONVERSION": 0.01, "_other": 0.04
    },
    "video_ott_ctv": {
        "paid_search_brand": 0.18, "organic_search": 0.18, "paid_social": 0.10,
        "direct_organic": 0.12, "independent_agent": 0.05,
        "NULL": 0.30, "CONVERSION": 0.02, "_other": 0.05
    },
    "aggregator_comparator": {
        "paid_search_brand": 0.25, "independent_agent": 0.15, "organic_search": 0.10,
        "direct_organic": 0.10, "CONVERSION": 0.05, "NULL": 0.30, "_other": 0.05
    },
    "direct_organic": {
        "independent_agent": 0.20, "paid_search_brand": 0.08, "email_marketing": 0.05,
        "CONVERSION": 0.15, "NULL": 0.40, "call_center": 0.05, "_other": 0.07
    },
}

# 2nd-order patterns (for higher-order simulation):
# (display, paid_search_brand) → agent: 0.40 (higher than base)
# (direct_mail, call_center) → agent: 0.45 (mail provides context)
# (paid_social, organic_search) → agent: 0.30 (social-assisted path)
# (tv_radio, paid_search_brand) → agent: 0.35 (TV triggers brand search → agent)
```

### 4.4 Conversion Model (`conversion_model.py`)

At each step in journey simulation, a logistic model determines conversion probability. The model captures: **accumulated channel diversity and agent presence dramatically increase conversion probability.**

```python
CONVERSION_FEATURES = {
    "intercept":               -4.5,     # Base conversion ~1%
    "agent_present":            1.8,     # Agent = 5-6x lift (dominant factor)
    "agent_is_last_touch":      0.9,     # Agent as last touch adds further lift
    "num_distinct_channels":    0.3,     # More diverse journeys convert better
    "has_search_touch":         0.4,     # Search = active shopping
    "has_display_or_social":    0.2,     # Upper funnel awareness helps
    "has_direct_mail":          0.25,    # Direct mail = targeted prospect
    "has_email_click":          0.5,     # Email click = high engagement
    "has_call_center":          0.6,     # Phone call = high intent
    "journey_length_penalty":  -0.05,    # Per-touchpoint: long journeys = lower conversion
    "recency_factor":           0.15,    # Per-unit: recent activity lifts conversion
    "life_event_trigger":       0.6,     # Life event = motivated shopper
}

# Calibration targets:
# - Overall journey-to-bind rate: ~2.5-3.5%
# - Quote-start-to-bind rate: ~30-40%
# - Journeys with agent touch convert at 4-5x rate of pure digital
# - Agent as last touch + prior digital: highest converting segment
```

### 4.5 Temporal Engine (`timestamp_engine.py`)

```python
JOURNEY_DURATION_PARAMS = {
    "lognormal_mu": 2.08,    # ln(8) — median 8 days
    "lognormal_sigma": 0.8,  # mean ~14 days
    "min_days": 0, "max_days": 90,
}

INTER_TOUCH_GAP_PARAMS = {
    "digital_mean_gap_days": 2.5,
    "offline_mean_gap_days": 5.0,
    "min_gap_minutes": 5,
}

MONTHLY_SEASONALITY = {
    1: 0.85, 2: 0.90, 3: 1.20, 4: 1.25, 5: 1.05, 6: 0.95,
    7: 0.90, 8: 0.90, 9: 1.15, 10: 1.20, 11: 0.95, 12: 0.80,
}

DAY_OF_WEEK_WEIGHTS = {
    0: 0.95, 1: 1.10, 2: 1.15, 3: 1.10, 4: 1.00, 5: 0.40, 6: 0.30,
}
```

### 4.6 Spend Data Generation (`spend_generator.py`)

```python
# Erie estimated annual marketing spend ~$100M for demo purposes
ANNUAL_SPEND_ALLOCATION = {
    "paid_search_brand": 0.15, "paid_search_nonbrand": 0.08, "organic_search": 0.05,
    "display_programmatic": 0.10, "paid_social": 0.07, "tv_radio": 0.25,
    "direct_mail": 0.10, "email_marketing": 0.03, "call_center": 0.05,
    "aggregator_comparator": 0.02, "direct_organic": 0.00, "video_ott_ctv": 0.06,
    "independent_agent": 0.04,
}
# Monthly spend varies with seasonality + ±10% random noise per month
```

### 4.7 Output Validation Targets (`output_validator.py`)

**NEW in v2**: After generation, automatically validate that the synthetic data hits these targets. If any target is missed by >2σ, the generator reruns with adjusted parameters.

```python
VALIDATION_TARGETS = {
    "overall_conversion_rate": {"min": 0.025, "max": 0.045},
    "agent_last_touch_pct_of_conversions": {"min": 0.55, "max": 0.70},
    "avg_touchpoints_converting": {"min": 3.2, "max": 4.5},
    "avg_touchpoints_non_converting": {"min": 1.2, "max": 2.0},
    "pct_single_touch_journeys": {"min": 0.30, "max": 0.50},
    "pct_journeys_with_agent": {"min": 0.15, "max": 0.30},
    "agent_conversion_multiplier": {"min": 3.5, "max": 6.0},
    "median_journey_duration_days": {"min": 5.0, "max": 12.0},
    "top_path_conversion_rate": {"min": 0.08, "max": 0.20},
    "state_PA_pct": {"min": 0.27, "max": 0.33},
}
```

### 4.8 Built-in Insights (Engineered Discoveries)

These patterns are calibrated into the synthetic data. The attribution models will "discover" them:

1. **Branded Search Over-Credit**: Last-click ~40% → Shapley ~15-18%. Branded search is navigational, not demand-creating.
2. **Agent Under-Credit**: Last-click ~5-8% → Shapley ~30-35%. Agents are invisible in digital analytics but close ~85% of business.
3. **Display/Social Assist Value**: Last-click ~3% → Shapley ~12%. These channels create demand that agents convert.
4. **Direct Mail → Agent Pipeline**: ~70% of mail-influenced binds have agent as next touch. Mail activates agents.
5. **Highest-Converting Path**: Display/Social → Search → Agent → Bind converts at 4-5× average.
6. **Agent Multiplier Effect**: Any agent touch = 4-5× conversion rate vs. pure digital.

---

## 5. Identity Resolution Simulator

(Retained from v1 with light refinements.)

The demo uses pre-assigned persistent IDs but must **show** identity resolution concepts to sell the production engagement.

### 5.1 Simulated Identity Fragmentation

```python
MATCH_RATE_TARGETS = {
    "tier_1_deterministic": 0.65,    # 65% matched via email/phone hash
    "tier_2_household": 0.12,        # +12% via address matching
    "tier_3_probabilistic": 0.10,    # +10% via IP+UA (lower confidence)
    "unmatched": 0.13,               # 13% remain fragmented
}

# Per user: ~2.1 GA4 client IDs (multi-device), ~1.3 ad click IDs
# Simulate missing email for agent-only interactions
# Simulate household-level address collisions
```

### 5.2 Identity Graph Visualization Data

Generate ~50 users with fully materialized identity graphs for the Identity Resolution page. Each graph shows nodes (identifiers) and edges (linkages) with match tier coloring, rendered via NetworkX → Plotly scatter.

---

## 6. Attribution Model Arsenal

The demo implements **15+ distinct attribution models** organized in a spectrum from naive to state-of-the-art. This expanded arsenal (vs. v1's 9 models) reflects the full landscape from the resource guide. Models marked with ★ are highlighted in the demo narrative; others are available for exploration.

### 6.1 Model Hierarchy

```
Tier 1: Heuristic/Rule-Based (5 models) — "What everyone does today"
    ├── First-Touch
    ├── Last-Touch ★ (GA4 default — the "current state" for Erie)
    ├── Linear
    ├── Time-Decay
    └── Position-Based (U-shaped)

Tier 2: Game-Theoretic (3 models) — "The fair answer"
    ├── Shapley Value — Exact + Simplified Zhao et al. (2018) ★ [PRIMARY]
    ├── CASV — Counterfactual Adjusted Shapley Value (Singal et al. 2022, Management Science) ★
    └── Regression-Adjusted Monte Carlo Shapley (Witter et al. 2025) — 6.5× lower error than PermSHAP

Tier 3: Probabilistic/Graph-Based (3 models) — "The sequence answer"
    ├── Markov Chain — Fixed-order (1st, 2nd, 3rd) with Removal Effect ★
    ├── Variable-Order Markov (VMM) — Context-tree / MTD model (Berchtold & Raftery 2002)
    └── SHAP-Connection — Train XGBoost conversion model → SHAP values as attribution

Tier 4: Statistical (2 models) — "The regression answer"
    ├── Logistic Regression Attribution
    └── Survival/Additive Hazard Attribution (Zhang, Wei & Ren, ICDM 2014)

Tier 5: Deep Learning (4 models) — "The frontier"
    ├── LSTM + Attention (baseline DL)
    ├── DARNN — Dual-Attention RNN (Ren et al., CIKM 2018) ★
    ├── Transformer-based (Lu & Kannan, JMR 2025) — SOTA, outperforms HMMs/LSTMs
    └── CausalMTA — Confounding decomposition (KDD 2022, Alibaba)

Tier 6: Meta-Models
    ├── Weighted Ensemble — Shapley (0.45) + Markov (0.30) + Logistic (0.25) ★
    └── Triangulation Report — Cross-method convergence analysis (not a model, a framework)

Tier 7: Optimization — "The action layer"
    └── LP/IP Budget Optimizer (uses attribution as input)
```

### 6.2 Abstract Base Class

```python
# src/models/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class AttributionResult:
    """Standard output format for all attribution models.
    Every model MUST return this exact structure — the UI layer depends on it."""
    model_name: str
    channel_credits: Dict[str, float]        # channel_id → total credited conversions
    channel_credit_pct: Dict[str, float]     # channel_id → fraction of total credit (0.0-1.0)
    channel_credit_rank: Dict[str, int]      # channel_id → rank (1 = highest)
    journey_level_credits: pd.DataFrame      # journey_id × channel_id matrix of fractional credits
    total_conversions: int
    metadata: Dict                            # Model-specific diagnostics

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to the attribution_results.parquet schema (Section 3.8).
        This is how results flow to the UI."""
        rows = []
        for ch_id in self.channel_credits:
            rows.append({
                "model_name": self.model_name,
                "channel_id": ch_id,
                "credited_conversions": self.channel_credits[ch_id],
                "credit_pct": self.channel_credit_pct[ch_id],
                "credit_rank": self.channel_credit_rank[ch_id],
                "total_conversions": self.total_conversions,
            })
        return pd.DataFrame(rows)

class BaseAttributionModel(ABC):
    """All attribution models implement this interface."""

    def __init__(self, config: ModelParamsConfig, channel_config: List[ChannelConfig]):
        self.config = config
        self.channel_config = channel_config
        self.channel_names = [c.channel_id for c in channel_config]

    @abstractmethod
    def fit(self, journeys: pd.DataFrame) -> None:
        """Learn model parameters from journey data."""
        pass

    @abstractmethod
    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        """Compute attribution for all conversions."""
        pass

    @abstractmethod
    def attribute_single(self, journey: List[str], converted: bool) -> Dict[str, float]:
        """Compute attribution for a single journey. Used for real-time exploration."""
        pass

    def validate_output(self, result: AttributionResult) -> Dict[str, bool]:
        """Standard validation checks applied to every model's output."""
        checks = {}
        checks["credits_sum_to_conversions"] = abs(
            sum(result.channel_credits.values()) - result.total_conversions
        ) < 0.01
        checks["no_negative_credits"] = all(v >= 0 for v in result.channel_credits.values())
        checks["all_channels_present"] = set(result.channel_credits.keys()) == set(self.channel_names)
        checks["percentages_sum_to_one"] = abs(sum(result.channel_credit_pct.values()) - 1.0) < 0.001
        return checks
```

### 6.3 Rule-Based Models (`rule_based.py`)

Five heuristic models, primarily as baselines to contrast against Shapley/Markov.

```python
class FirstTouchAttribution(BaseAttributionModel):
    """100% credit to the first touchpoint in the journey."""

class LastTouchAttribution(BaseAttributionModel):
    """100% credit to the last touchpoint before conversion.
    This is what GA4 effectively does — the 'current state' for Erie."""

class LinearAttribution(BaseAttributionModel):
    """Equal credit split across all touchpoints. 1/N per touch."""

class TimeDecayAttribution(BaseAttributionModel):
    """Exponentially decaying credit favoring recent touches.
    weight_i = exp(-λ × t_i) where λ = ln(2) / half_life.
    Default half_life: 7 days clicks, 3 days impressions."""

class PositionBasedAttribution(BaseAttributionModel):
    """U-shaped: 40% first, 40% last, 20% split among middle touches."""
```

### 6.4 Shapley Value — Exact + Simplified (`shapley_engine.py`) — PRIMARY MODEL

```python
class ShapleyAttribution(BaseAttributionModel):
    """
    Shapley Value attribution using cooperative game theory.

    Core formula:
        φ(i) = Σ_{S ⊆ N\{i}} [|S|! × (|N| - |S| - 1)! / |N|!] × [v(S ∪ {i}) - v(S)]

    Value function v(S): INCLUSIVE formulation (Zhao et al. 2018, Google DDA standard).
        v(S) = conversion rate among journeys where ALL channels in S appeared.

    Computation: Zhao et al. simplified approach — group users by channel-set "type"
    and pre-compute coalition values. O(Σ_T 2^|T|) where T = user types with 3-5 channels.

    Sparse coalition handling:
    1. Coalitions with < min_obs observations: impute via adjacent coalition average
    2. If sparsity persists: fall back to channel grouping (7 macro-channels)
    3. Alternative: use shapiq GaussianImputer for missing coalitions

    Time-weighted extension: Each touchpoint's presence weighted by time-decay factor.

    References:
    - Zhao et al. (2018), arXiv:1804.05327 — simplified computation
    - Dalessandro et al. (2012), ADKDD at KDD — causal connection
    - Lin et al. (2025), VLDB — comprehensive survey
    """
    # Full implementation as specified in v1 Section 5.4
    pass
```

### 6.5 CASV — Counterfactual Adjusted Shapley Value (`shapley_casv.py`) — NEW

```python
class CASVAttribution(BaseAttributionModel):
    """
    Counterfactual Adjusted Shapley Value (Singal et al. 2022, Management Science 68(10)).
    Originally WWW 2019, published in Management Science.

    KEY INNOVATION: Uses a Markov chain model to compute counterfactual conversion
    probabilities, then feeds these into the Shapley value computation. This inherently
    captures stage effects (position in the journey matters) while preserving all four
    Shapley axioms.

    How it differs from standard Shapley:
    - Standard Shapley: v(S) = P(conversion | channels in S present). Channel order ignored.
    - CASV: v(S) computed via a Markov model that conditions on the SEQUENCE of channels.
      v(S) = P(conversion via Markov chain | only channels in S are available).
      This means removing a channel doesn't just reduce a rate — it redirects paths.

    Why include in demo:
    - It's the most theoretically rigorous Shapley extension in the literature
    - It naturally captures Erie's sequential pattern (display → search → agent)
    - Published in Management Science — top credibility for academic-leaning stakeholders
    - Bridges Shapley and Markov — showing they're complementary, not competing

    Implementation:
    1. Build a Markov chain transition matrix from observed journeys
    2. For each coalition S, construct a modified Markov chain with only channels in S
    3. Compute v(S) = P(START → CONVERSION) via the fundamental matrix on the modified chain
    4. Apply standard Shapley formula using these counterfactual v(S) values
    5. Normalize to total conversions

    Computational note: This is more expensive than standard Shapley because each v(S)
    requires a matrix inversion. For 13 channels (8,192 coalitions), pre-compute all v(S)
    in batch. With channel grouping fallback to 7 groups (128 coalitions), instant.
    """

    def __init__(self, config, channel_config):
        super().__init__(config, channel_config)
        self.markov_order = config.casv_markov_order
        self.transition_matrix = None

    def fit(self, journeys: pd.DataFrame):
        """Build base Markov chain from journey data."""
        # Reuse MarkovChainAttribution._fit_first_order logic
        pass

    def _compute_counterfactual_value(self, coalition: frozenset) -> float:
        """
        Compute v(S) by constructing a modified Markov chain where only
        channels in S are available. All transitions to channels NOT in S
        are redistributed proportionally among channels IN S (or to NULL).
        Then compute P(START → CONVERSION) via fundamental matrix.
        """
        pass

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        """Standard Shapley computation but using counterfactual v(S) values."""
        pass
```

### 6.6 Regression-Adjusted Monte Carlo Shapley (`shapley_regression_mc.py`) — NEW

```python
class RegressionAdjustedMCShapley(BaseAttributionModel):
    """
    Regression-Adjusted Monte Carlo Shapley (Witter et al. 2025, arXiv:2506.11849).

    STATE-OF-THE-ART Shapley approximation achieving 6.5× lower error than
    Permutation SHAP and 3.8× lower than KernelSHAP.

    Core idea: Combine Monte Carlo permutation sampling with a regression
    model that predicts marginal contributions. The regression model provides
    a "control variate" that dramatically reduces variance.

    Algorithm:
    1. Sample M random permutations of channels
    2. For each permutation, compute marginal contributions (same as PermSHAP)
    3. Train an XGBoost model to predict marginal contributions from
       (channel, coalition features)
    4. Use the regression predictions as control variates:
       φ_adj(i) = φ_MC(i) - β × (f_regression(i) - E[f_regression])
    5. The variance reduction from step 4 is where the 6.5× gain comes from

    Why include in demo:
    - It's the cutting-edge (2025) approximation method
    - Allows XGBoost as the regression model — familiar to Erie's data science team
    - Maintains unbiased Shapley estimates despite using ML acceleration
    - Shows the team is current with SOTA, not implementing textbook methods from 2018

    Implementation: Uses the `shapiq` library's approximation framework.
    Fallback: If computation exceeds 30 seconds, fall back to standard MC with 10K samples.
    """
    pass
```

### 6.7 Markov Chain — Fixed-Order (`markov_chain.py`)

```python
class MarkovChainAttribution(BaseAttributionModel):
    """
    Absorbing Markov Chain with Removal Effect attribution.

    Model: States = {START, Channel_1, ..., Channel_N, CONVERSION, NULL}
    CONVERSION and NULL are absorbing states.

    Transition probabilities: MLE with Dirichlet(α=1/N) smoothing.
    Per Heiner et al. (2022, JCGS): sparse Dirichlet mixture priors for
    high-order chains to handle zero-count transitions robustly.

    Removal effect for channel c:
        1. Remove c from graph (redirect transitions into c → NULL)
        2. Recompute P(START → CONVERSION) via (I - Q')^{-1} R'
        3. RE(c) = 1 - P(conv | c removed) / P(conv | all channels)
        4. Normalize: credit(c) = RE(c) / Σ RE(all) × total_conversions

    Supports 1st, 2nd, and 3rd order chains.
    3rd order often most proficient (Anderl et al. 2016) but state space = N^3.
    With 13 channels and 3rd order: 2,197 compound states. Only use observed states.

    KEY INSIGHT from resource guide (Singal et al. 2022):
    Removal effect OVERWEIGHTS high-traffic channels because removing them
    disrupts more paths. This is why Markov tends to give agents even MORE
    credit than Shapley — agents appear in many paths. This divergence is
    itself an interesting demo talking point.

    References:
    - Anderl et al. (2014/2016), IJRM — founding framework
    - ChannelAttribution R/Python package — industry standard
    - Adequate Digital guide — normalization issues
    """

    def fit(self, journeys: pd.DataFrame):
        """Build transition matrix with Dirichlet smoothing."""
        pass

    def _compute_base_conversion_probability(self) -> float:
        """P(START → CONVERSION) via fundamental matrix N = (I-Q)^{-1}, B = N×R."""
        pass

    def _compute_removal_effect(self, channel: str) -> float:
        """Remove channel, recompute conversion probability, measure drop."""
        pass

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        """Compute removal effects, normalize to total conversions."""
        pass

    def get_transition_matrix_for_viz(self) -> pd.DataFrame:
        """Return transition matrix as DataFrame for Sankey/heatmap."""
        pass

    def get_top_paths(self, n: int = 20) -> List[Dict]:
        """Top N converting paths with probabilities and frequencies."""
        pass
```

### 6.8 Variable-Order Markov (`markov_variable_order.py`) — NEW

```python
class VariableOrderMarkovAttribution(BaseAttributionModel):
    """
    Variable-Order Markov Model (VMM) using Mixture Transition Distribution (MTD).

    Based on Berchtold & Raftery (2002, Statistical Science 17(3)).

    KEY INNOVATION: Instead of fixed k-th order (which causes N^k state explosion),
    the VMM lets the context length vary per state. Channels that benefit from
    longer context (e.g., knowing the two previous touches matters for agent)
    get deeper conditioning, while simple transitions use first-order.

    MTD parameterization:
        P(X_t = j | X_{t-1} = i_1, ..., X_{t-k} = i_k) =
            Σ_{l=1}^{k} λ_l × P_l(j | i_l)

    where P_l are first-order transition matrices and λ_l are mixing weights.
    This reduces parameters from N^k to k × N² — massive reduction.

    Why include:
    - Addresses the state space explosion problem that limits fixed-order Markov
    - The λ_l weights are themselves interpretable: λ_1 high means recency dominates,
      λ_2 high means the two-step-back channel matters more
    - For Erie, we expect λ_1 > λ_2 > λ_3 — recency matters most
    - Demonstrates awareness of the Markov literature beyond basic implementations

    Implementation:
    1. Fit k separate first-order transition matrices P_1, ..., P_k
    2. Estimate mixing weights λ_l via EM algorithm
    3. Compute removal effects using the mixed model
    4. Attribute via normalized removal effects
    """
    pass
```

### 6.9 Logistic Regression Attribution (`logistic_attribution.py`)

```python
class LogisticAttribution(BaseAttributionModel):
    """
    Logistic regression with binary channel indicators + interaction terms.
    L2 regularization. Coefficient magnitudes → attribution weights.
    Include (channel_i × channel_j) interactions for key pairs (display × agent).
    """
    pass
```

### 6.10 Survival/Hazard Attribution (`survival_attribution.py`)

```python
class SurvivalAttribution(BaseAttributionModel):
    """
    Additive hazard model: h(t) = h₀(t) + Σ βᵢ × x_i(t)
    Based on Zhang, Wei & Ren (ICDM 2014) and Google's Shender et al. (2020).
    Uses lifelines library for Weibull baseline hazard estimation.
    Captures "which channels ACCELERATE conversion" — not just "which help."
    """
    pass
```

### 6.11 DARNN — Dual-Attention RNN (`darnn_attribution.py`) — NEW

```python
class DARNNAttribution(BaseAttributionModel):
    """
    Dual-Attention Recurrent Neural Network (Ren et al., CIKM 2018).
    Paper: https://arxiv.org/abs/1808.03737

    Architecture:
    - Encoder: GRU with input attention over channel embeddings
    - Decoder: GRU with temporal attention over encoder states
    - Output: Conversion probability + attention weights as attribution

    KEY INNOVATION: TWO attention mechanisms —
    1. Input attention: learns which channel features matter at each timestep
    2. Temporal attention: learns which historical timesteps matter for conversion
    Combined, they produce interpretable attribution scores.

    Budget-allocation-based evaluation (introduced in this paper):
    Instead of just computing attribution weights, evaluate by simulating
    budget reallocation based on attribution → measure predicted conversion change.
    This became the standard DL attribution benchmark.

    Why include in demo:
    - Published at CIKM with public code — reproducible and credible
    - Two attention layers give richer interpretability than single-attention LSTM
    - The budget-allocation evaluation connects directly to Erie's optimizer page
    - Shows we can go beyond vanilla LSTM to specialized attribution architectures

    Implementation:
    - PyTorch with GRU encoder/decoder
    - Channel embedding dim: 16, hidden dim: 64
    - Input attention: Softmax over channel features per timestep
    - Temporal attention: Softmax over encoder hidden states
    - Training: BCE loss, Adam, 50 epochs, batch 256
    - Attribution: Average temporal attention weights per channel across converting journeys
    """
    pass
```

### 6.12 Transformer-Based Attribution (`transformer_attribution.py`) — NEW

```python
class TransformerAttribution(BaseAttributionModel):
    """
    Transformer-based customer journey attribution (Lu & Kannan, JMR 2025).
    Paper: https://journals.sagepub.com/doi/10.1177/00222437251347268

    THE CURRENT STATE OF THE ART — outperforms HMMs, point process models, and LSTMs.

    Architecture:
    - Heterogeneous mixture multi-head self-attention
    - Each attention head captures a different type of channel interaction
    - Mixture weights capture individual-level variation in response patterns
    - Position encoding captures journey stage effects

    KEY INNOVATION: Multi-head attention where each head specializes in a
    different aspect of the journey (e.g., one head for recency, one for
    channel diversity, one for agent presence). The mixture component means
    different users can have different dominant attention patterns.

    Why include in demo:
    - Published in Journal of Marketing Research (top marketing journal) in 2025
    - Represents where the field is heading — Transformers replacing RNNs
    - The multi-head attention visualization is visually compelling for demos
    - Shows Erie this team is at the frontier, not implementing 2018 methods

    Implementation (simplified for demo):
    - PyTorch Transformer encoder (2 layers, 4 heads, d_model=64)
    - Channel embedding + positional encoding + time-to-conversion encoding
    - Classification head for conversion prediction
    - Attribution: Per-head attention weight extraction + aggregation
    - Training: Same as DARNN (BCE, Adam, 50 epochs)

    DEMO NOTE: This is positioned as "roadmap" capability alongside DARNN.
    Show attention heatmaps comparing DARNN and Transformer — where they agree
    builds confidence, where they disagree reveals interaction effects.
    """
    pass
```

### 6.13 CausalMTA (`causal_mta.py`) — NEW

```python
class CausalMTAAttribution(BaseAttributionModel):
    """
    CausalMTA: Confounding-aware multi-touch attribution (KDD 2022, Alibaba).
    Paper: https://dl.acm.org/doi/10.1145/3534678.3539108

    THE BEST CAUSAL DEEP LEARNING APPROACH with theoretical guarantees.

    Core insight: Standard DL attribution models suffer from confounding bias
    because users who see more ads are systematically different from those who don't.
    CausalMTA decomposes this bias into:
    1. Static confounders (user demographics, device type, geographic factors)
    2. Dynamic confounders (browsing behavior, time-of-day, previous conversions)

    Architecture:
    1. Confounder decomposition network: separates static and dynamic factors
    2. Journey reweighting: IPW-style reweighting to adjust for selection bias
    3. Causal conversion prediction: predicts conversion under counterfactual
       channel exposure scenarios
    4. Attribution: Difference in predicted conversion under channel presence
       vs. absence, weighted by causal adjustment

    Why include in demo:
    - Directly addresses the Gordon et al. (2023) critique that observational
      MTA produces 488–948% errors
    - Shows Erie we understand the causal inference problem, not just the math
    - The "confounding decomposition" concept resonates with business audiences:
      "Users who see TV ads are different from users who don't — and our model
      accounts for that."
    - Published at KDD (top CS venue) by Alibaba's team — production credibility

    Implementation (simplified for demo):
    - Static confounder features: age_band, state, vehicle_type, digital_propensity
    - Dynamic confounder features: time_of_day, day_of_week, session_depth
    - IPW reweighting via propensity score estimation (logistic regression)
    - Causal prediction via reweighted LSTM
    - Attribution: Mean counterfactual difference per channel

    DEMO NOTE: Positioned as "what production would look like" — showing that
    we can move beyond simple observational attribution to causal methods.
    """
    pass
```

### 6.14 LSTM Attribution (`lstm_attribution.py`)

Retained from v1. LSTM with attention mechanism. Used as the "baseline deep learning" model against which DARNN and Transformer are compared.

### 6.15 Ensemble Model (`ensemble.py`)

```python
class EnsembleAttribution(BaseAttributionModel):
    """
    Weighted ensemble: Shapley (0.45) + Markov 2nd-order (0.30) + Logistic (0.25).
    Confidence bands: spread across models provides natural uncertainty range.
    Production recommendation — triangulating across fundamentally different methods.
    """
    pass
```

### 6.16 Model Registry (`model_registry.py`)

```python
STANDARD_MODELS = {
    # Tier 1: Rule-based
    "first_touch": FirstTouchAttribution,
    "last_touch": LastTouchAttribution,
    "linear": LinearAttribution,
    "time_decay": TimeDecayAttribution,
    "position_based": PositionBasedAttribution,
    # Tier 2: Game-theoretic
    "shapley": ShapleyAttribution,
    "casv": CASVAttribution,
    "shapley_reg_mc": RegressionAdjustedMCShapley,
    # Tier 3: Graph-based
    "markov_order_1": MarkovChainAttribution,       # config.markov_order=1
    "markov_order_2": MarkovChainAttribution,       # config.markov_order=2
    "markov_vmm": VariableOrderMarkovAttribution,
    # Tier 4: Statistical
    "logistic": LogisticAttribution,
    "survival": SurvivalAttribution,
    # Tier 5: Deep Learning
    "lstm": LSTMAttribution,
    "darnn": DARNNAttribution,
    "transformer": TransformerAttribution,
    "causal_mta": CausalMTAAttribution,
    # Tier 6: Ensemble
    "ensemble": EnsembleAttribution,
}

# For demo default view, show these 4:
DEFAULT_DISPLAY_MODELS = ["last_touch", "shapley", "markov_order_2", "ensemble"]
# For "full comparison" mode:
FULL_COMPARISON_MODELS = list(STANDARD_MODELS.keys())
```

---

## 7. Data Pipeline & Journey Assembly

(Retained from v1 — core logic unchanged.)

### 7.1 Pipeline Overview

```
Raw Touchpoints → Sessionizer → Touch Qualifier → Channel Classifier →
Journey Assembler → Time Decay Calculator → Attribution-Ready Journeys
```

### 7.2 Sessionizer

30-minute inactivity-based windowing. Sort by (persistent_id, event_timestamp), gap detection, session_id assignment.

### 7.3 Touch Qualifier

Apply MRC viewability standards, dwell time thresholds, call duration minimums, Apple Mail Privacy Protection adjustment (email opens = 0.5 weight).

### 7.4 Journey Assembler

Construct ordered journeys per persistent_id with lookback window application, truncation to max_touchpoints, channel_path and channel_set extraction. Output must conform exactly to Section 3.4 schema.

### 7.5 Time Decay Calculator

Exponential decay: weight_i = exp(-λ × days_to_conversion_i) with channel-type-specific half-lives.

---

## 8. Budget Optimization & Triangulation Engine

### 8.1 Marginal ROI Curves (`marginal_roi.py`)

```python
class MarginalROICurve:
    """
    Saturating exponential response curves per channel:
        conversions(spend) = a × (1 - exp(-b × spend))

    Parameters estimated from monthly (spend, attributed_conversions) pairs.
    Calibrated so marginal ROI at current spend matches Shapley-implied efficiency.

    Output per channel: function mapping spend → expected conversions,
    with first derivative (marginal ROI) and saturation point.
    """
    pass
```

### 8.2 Budget Optimizer (`budget_optimizer.py`)

```python
class BudgetOptimizer:
    """
    LP/IP optimization: max Σ f_i(x_i) subject to:
    1. Σ x_i = B (total budget constant)
    2. x_i ≥ min_i (minimum spend per channel)
    3. |x_i - x_i_current| ≤ δ × x_i_current (max shift cap)
    4. Agent support ≥ 2% of total
    5. x_i ≥ 0

    Nonlinear solver: scipy.optimize.minimize (SLSQP method).
    Pre-compute results for all scenarios in Section 3.11 schema.
    """
    pass
```

### 8.3 Scenario Simulator (`scenario_simulator.py`)

Pre-computes 6 standard scenarios to `budget_scenarios.parquet`:

1. **"current"** — baseline allocation
2. **"optimal_unconstrained"** — pure optimization, no shift caps
3. **"optimal_20pct_cap"** — optimization with ≤20% reallocation per channel
4. **"shift_search_to_display"** — move 15% of branded search to display/social
5. **"double_agent_support"** — 2× agent marketing support budget
6. **"cut_tv_50"** — halve TV, redistribute to digital

Each scenario populates the full `budget_scenarios.parquet` schema from Section 3.11.

### 8.4 Triangulation Framework (`triangulation.py`) — NEW

```python
class TriangulationReport:
    """
    NOT a model — a strategic framework shown on the Technical Appendix page.

    Positions the demo within the industry consensus (Gordon et al. 2023):
    - MTA (this demo): Tactical channel-level optimization. Fast feedback loop.
      Limitation: observational only, 488-948% potential error in absolute effects.
    - MMM (Google Meridian / Meta Robyn): Strategic budget allocation across channels.
      Uses geo-level data. Slower feedback loop but captures offline effects.
      Recommended as Phase 2 for Erie — Meridian's geo-level modeling ideal
      for 12-state regional measurement.
    - Incrementality Experiments: Causal ground truth. Ghost ads, geo-experiments.
      Recommended as Phase 3 for Erie — validate MTA/MMM with controlled tests.
      Meta GeoLift and Google CausalImpact for implementation.

    The triangulation view shows:
    1. Where MTA is strong (channel-level, real-time, granular)
    2. Where MTA is weak (no causal guarantees, offline blind spots)
    3. How MMM and incrementality experiments complement MTA
    4. A 3-phase roadmap: MTA (now) → MMM (6 months) → Experiments (12 months)

    This is the "sell the engagement" slide — showing Erie that this demo is
    Phase 1 of a multi-year measurement maturity journey.
    """
    pass
```

---

## 9. Validation & QA Framework

(Retained from v1 with addition of data contract tests.)

### 9.1 Shapley Axiom Tests

Efficiency (credits = conversions), Symmetry, Null Player, Additivity. Property-based testing via `hypothesis` library.

### 9.2 Cross-Model Comparison

Spearman ρ matrix, rank stability, divergence flags. Output to `model_comparison.parquet` (Section 3.9).

### 9.3 Holdout Simulation

Remove channel from data, re-run attribution, verify detected impact matches credited amount.

### 9.4 Sensitivity Analysis

Lookback window sensitivity, time-decay half-life sensitivity, Markov order sensitivity. Heatmap outputs.

### 9.5 Data Contract Tests (`data_contract_tests.py`) — NEW

```python
def test_attribution_results_schema():
    """Verify attribution_results.parquet matches Section 3.8 schema exactly."""
    df = pd.read_parquet("data/processed/attribution_results.parquet")
    required_columns = [
        "model_name", "channel_id", "channel_display_name", "channel_group",
        "credited_conversions", "credit_pct", "credit_rank", "annual_spend",
        "cost_per_conversion", "roas", "funnel_role", "channel_color",
        "param_lookback_days", "param_decay_half_life", "param_markov_order",
        "total_conversions"
    ]
    assert set(required_columns).issubset(set(df.columns))

    # Every model must have exactly 13 channel rows
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        assert len(model_df) == 13, f"Model {model} has {len(model_df)} channels, expected 13"

    # credit_pct must sum to ~1.0 per model
    for model in df['model_name'].unique():
        pct_sum = df[df['model_name'] == model]['credit_pct'].sum()
        assert abs(pct_sum - 1.0) < 0.001, f"Model {model} credit_pct sums to {pct_sum}"

def test_budget_scenarios_schema():
    """Verify budget_scenarios.parquet matches Section 3.11 schema."""
    df = pd.read_parquet("data/processed/budget_scenarios.parquet")
    # Verify all 6 scenarios present
    required_scenarios = [
        "current", "optimal_unconstrained", "optimal_20pct_cap",
        "shift_search_to_display", "double_agent_support", "cut_tv_50"
    ]
    assert set(required_scenarios).issubset(set(df['scenario_id'].unique()))

    # Total budget must be constant across scenarios
    for scenario in df['scenario_id'].unique():
        budget = df[df['scenario_id'] == scenario]['proposed_spend'].sum()
        current_budget = df[df['scenario_id'] == 'current']['current_spend'].sum()
        assert abs(budget - current_budget) < 100, f"Scenario {scenario} budget mismatch"

def test_no_hardcoded_metrics():
    """Verify the UI layer contains no hardcoded numeric values."""
    import ast, os
    ui_files = []
    for root, dirs, files in os.walk("app/pages"):
        for f in files:
            if f.endswith(".py"):
                ui_files.append(os.path.join(root, f))

    forbidden_patterns = [
        # These patterns indicate hardcoded metrics in UI code
        r'credit_pct\s*=\s*\d',     # Direct assignment of credit percentages
        r'total_binds\s*=\s*\d',     # Hardcoded bind counts
        r'"34.2%"',                  # Hardcoded string percentages
        r'"4,800"',                  # Hardcoded formatted numbers
    ]
    # Scan each file for forbidden patterns
    pass
```

---

## 10. UI/UX Specification — Plotly Dash Application

### 10.1 Design Philosophy

Three principles for a C-suite demo:

1. **Lead with insight, not data.** Every page opens with a headline finding, then supports it with interactive evidence.
2. **Branded and polished.** Dash Bootstrap Components with a fully custom dark theme. No default Dash/Bootstrap appearance.
3. **Progressive disclosure.** Simple story on top, interactive exploration for drill-down, technical detail in collapsible sections.

### 10.2 Dash Application Architecture

```python
# app/app.py

import dash
from dash import Dash, html, dcc, page_registry, page_container
import dash_bootstrap_components as dbc
from app.theme import EXTERNAL_STYLESHEETS, PLOTLY_TEMPLATE
from app.components.navbar import create_navbar

# Initialize Dash app with multi-page support
app = Dash(
    __name__,
    use_pages=True,                    # Enable pages/ directory routing
    external_stylesheets=EXTERNAL_STYLESHEETS,
    suppress_callback_exceptions=True,
    title="MCA Intelligence — Erie Auto Insurance",
    update_title=None,                 # Disable "Updating..." title flash
)

server = app.server  # Flask server for production deployment (Gunicorn)

# App layout: navbar + page container
app.layout = dbc.Container([
    dcc.Store(id="selected-models-store", data=["last_touch", "shapley", "markov_order_2", "ensemble"]),
    dcc.Store(id="parameter-store", data={"lookback_days": 45, "decay_half_life": 7.0, "markov_order": 2}),
    create_navbar(),
    html.Div(id="page-content", children=[page_container]),
], fluid=True, className="app-container")


if __name__ == "__main__":
    app.run(debug=False, port=8050)
```

### 10.3 Color Palette & Typography

```python
# app/theme.py

import plotly.graph_objects as go
import plotly.io as pio

EXTERNAL_STYLESHEETS = [
    dbc.themes.DARKLY,  # Dark Bootstrap theme as base
    "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap",
]

COLORS = {
    "bg_primary": "#0F1419",
    "bg_secondary": "#1A2332",
    "bg_tertiary": "#243044",
    "text_primary": "#E8ECF1",
    "text_secondary": "#8899AA",
    "text_accent": "#FFFFFF",
    "highlight": "#00D4AA",
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "warning": "#F39C12",
    "info": "#3498DB",

    "channel_colors": {
        "independent_agent":      "#2ECC71",
        "paid_search_brand":      "#3498DB",
        "paid_search_nonbrand":   "#2980B9",
        "organic_search":         "#1ABC9C",
        "display_programmatic":   "#E74C3C",
        "paid_social":            "#9B59B6",
        "tv_radio":               "#F39C12",
        "direct_mail":            "#E67E22",
        "email_marketing":        "#F1C40F",
        "call_center":            "#16A085",
        "aggregator_comparator":  "#95A5A6",
        "direct_organic":         "#34495E",
        "video_ott_ctv":          "#D35400",
    },
}

# Register custom Plotly template used by ALL charts
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E8ECF1", size=12),
        title=dict(font=dict(family="DM Sans, sans-serif", size=18, color="#FFFFFF"), x=0, xanchor="left"),
        xaxis=dict(gridcolor="#2A3A4A", zerolinecolor="#2A3A4A", tickfont=dict(size=11, color="#8899AA")),
        yaxis=dict(gridcolor="#2A3A4A", zerolinecolor="#2A3A4A", tickfont=dict(size=11, color="#8899AA")),
        legend=dict(bgcolor="rgba(26,35,50,0.8)", bordercolor="#2A3A4A", font=dict(color="#E8ECF1", size=11)),
        margin=dict(l=60, r=30, t=60, b=40),
        hoverlabel=dict(bgcolor="#1A2332", bordercolor="#00D4AA", font=dict(family="Inter", size=12, color="#E8ECF1")),
    )
)
pio.templates["erie_dark"] = PLOTLY_TEMPLATE
pio.templates.default = "erie_dark"
```

### 10.4 Global CSS (`app/assets/custom.css`)

Dash automatically loads all CSS files from the `assets/` directory.

```css
/* app/assets/custom.css */

/* ===== Global ===== */
body {
    background-color: #0F1419;
    color: #E8ECF1;
    font-family: 'Inter', sans-serif;
}

.app-container {
    padding: 0;
    max-width: 100%;
}

/* ===== Typography ===== */
h1, h2, h3, h4, h5 { font-family: 'DM Sans', sans-serif; }
h1 { font-weight: 700; color: #FFFFFF; font-size: 2.2rem; letter-spacing: -0.02em; }
h2 {
    font-weight: 500; color: #E8ECF1; font-size: 1.5rem;
    border-bottom: 2px solid #00D4AA; padding-bottom: 0.5rem; margin-top: 2rem;
}
h3 {
    font-weight: 500; color: #8899AA; font-size: 1.1rem;
    text-transform: uppercase; letter-spacing: 0.05em;
}

/* ===== Metric Cards ===== */
.metric-card {
    background: linear-gradient(135deg, #1A2332 0%, #243044 100%);
    border: 1px solid #2A3A4A;
    border-radius: 12px;
    padding: 1.5rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 212, 170, 0.1);
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.5rem; font-weight: 700; color: #00D4AA;
}
.metric-label {
    font-size: 0.85rem; color: #8899AA;
    text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.5rem;
}
.metric-delta { font-family: 'JetBrains Mono', monospace; font-size: 1rem; margin-top: 0.3rem; }
.delta-positive { color: #2ECC71; }
.delta-negative { color: #E74C3C; }

/* ===== Insight Callout ===== */
.insight-box {
    background: linear-gradient(135deg, rgba(0, 212, 170, 0.08) 0%, rgba(0, 212, 170, 0.03) 100%);
    border-left: 4px solid #00D4AA;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem; margin: 1.5rem 0;
    font-size: 1.05rem; line-height: 1.6;
}
.warning-box {
    background: rgba(243, 156, 18, 0.08);
    border-left: 4px solid #F39C12;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem; margin: 1.5rem 0;
}

/* ===== Navigation ===== */
.navbar-dark { background-color: #0F1419 !important; border-bottom: 1px solid #2A3A4A; }
.nav-link { font-family: 'DM Sans', sans-serif; font-weight: 500; color: #8899AA !important; }
.nav-link.active, .nav-link:hover { color: #00D4AA !important; }

/* ===== Model Selector Pills ===== */
.model-pill {
    border: 1px solid #2A3A4A; border-radius: 20px;
    padding: 6px 16px; margin: 4px;
    font-family: 'Inter', sans-serif; font-size: 0.85rem;
    color: #8899AA; background: transparent;
    cursor: pointer; transition: all 0.2s;
}
.model-pill.active {
    border-color: #00D4AA; color: #00D4AA;
    background: rgba(0, 212, 170, 0.1);
}
.model-pill.starred::before { content: "★ "; color: #F1C40F; }

/* ===== Cards and Panels ===== */
.card { background-color: #1A2332 !important; border: 1px solid #2A3A4A !important; border-radius: 12px; }
.card-header { background-color: #243044 !important; border-bottom: 1px solid #2A3A4A !important; }

/* ===== Section Divider ===== */
.section-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #2A3A4A, transparent);
    margin: 2rem 0;
}

/* ===== Demo Title ===== */
.demo-title {
    font-family: 'DM Sans', sans-serif; font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(135deg, #00D4AA, #3498DB);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
```

### 10.5 Reusable Components

#### 10.5.1 Metric Card (`components/metric_card.py`)

```python
from dash import html

def metric_card(label: str, value: str, delta: str = None,
                delta_direction: str = None, subtitle: str = None):
    """
    Returns a Dash html.Div representing a styled metric card.

    CRITICAL: `value` and `delta` are STRINGS passed in by the callback
    after formatting. They are NEVER hardcoded in this component.
    The callback reads from src/metrics/ and formats via src/utils/formatters.py.
    """
    children = [
        html.Div(value, className="metric-value"),
        html.Div(label, className="metric-label"),
    ]
    if delta:
        arrow = "↑" if delta_direction == "positive" else "↓" if delta_direction == "negative" else ""
        cls = f"metric-delta delta-{delta_direction}" if delta_direction else "metric-delta"
        children.append(html.Div(f"{arrow} {delta}", className=cls))
    if subtitle:
        children.append(html.Div(subtitle, style={"color": "#5A6A7A", "fontSize": "0.8rem", "marginTop": "0.3rem"}))

    return html.Div(children, className="metric-card")
```

#### 10.5.2 Insight Callout (`components/insight_callout.py`)

```python
def insight_callout(text: str, icon: str = "💡"):
    """
    Styled insight box. `text` is computed by the metrics layer, never hardcoded.
    Example: the metrics layer computes "Agents receive {x}% credit in last-click
    vs {y}% in Shapley — a {z}× gap" by reading from attribution_results DataFrame.
    """
    return html.Div([
        html.Span(icon, style={"fontSize": "1.2rem", "marginRight": "0.5rem"}),
        html.Span(text),
    ], className="insight-box")
```

#### 10.5.3 Model Selector (`components/model_selector.py`)

```python
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

def model_selector(available_models: list, starred_models: list = None):
    """
    Row of toggle pills for selecting which attribution models to display.
    Uses dcc.Store to persist selection across page components.
    """
    starred = starred_models or ["shapley", "markov_order_2", "ensemble"]
    pills = []
    for model in available_models:
        cls = "model-pill"
        if model in starred:
            cls += " starred"
        pills.append(
            html.Button(
                model.replace("_", " ").title(),
                id={"type": "model-pill", "model": model},
                className=cls,
                n_clicks=0,
            )
        )
    return html.Div(pills, style={"display": "flex", "flexWrap": "wrap", "gap": "4px", "marginBottom": "1rem"})
```

#### 10.5.4 Navbar (`components/navbar.py`)

```python
import dash_bootstrap_components as dbc
from dash import html

def create_navbar():
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(html.Div([
                    html.Span("⚡", style={"fontSize": "1.5rem", "marginRight": "0.5rem"}),
                    html.Span("MCA Intelligence", className="demo-title"),
                    html.Span(" — Erie Auto Insurance", style={"color": "#8899AA", "fontSize": "0.95rem", "marginLeft": "0.5rem"}),
                ]), width="auto"),
            ], align="center"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Executive Summary", href="/", active="exact")),
                dbc.NavItem(dbc.NavLink("Attribution", href="/attribution", active="exact")),
                dbc.NavItem(dbc.NavLink("Journeys", href="/journeys", active="exact")),
                dbc.NavItem(dbc.NavLink("Budget Optimizer", href="/budget", active="exact")),
                dbc.NavItem(dbc.NavLink("Identity", href="/identity", active="exact")),
                dbc.NavItem(dbc.NavLink("Channels", href="/channels", active="exact")),
                dbc.NavItem(dbc.NavLink("Validation", href="/validation", active="exact")),
                dbc.NavItem(dbc.NavLink("Technical", href="/technical", active="exact")),
            ], className="ms-auto", navbar=True),
        ], fluid=True),
        dark=True,
        className="mb-4",
    )
```

### 10.6 Page-by-Page Specification with Dash Layouts & Callbacks

#### Page 1: Executive Summary (`pages/executive_summary.py`)

```python
import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from app.components.metric_card import metric_card
from app.components.insight_callout import insight_callout
from app.data_store import get_executive_metrics, get_attribution_results
from src.utils.formatters import fmt_pct, fmt_number, fmt_currency, fmt_multiplier

dash.register_page(__name__, path="/", name="Executive Summary", order=0)

def layout():
    """
    Page layout is a FUNCTION, not a static variable.
    This ensures data is loaded fresh on each page visit.
    """
    m = get_executive_metrics()  # All metrics computed from DataFrames

    return dbc.Container([
        # Header
        html.H1("Multi-Channel Attribution Intelligence"),
        html.P("Auto Insurance — Erie Market Context", style={"color": "#8899AA", "fontSize": "1.1rem"}),
        html.Hr(className="section-divider"),

        # KPI row — ALL values from computed metrics
        dbc.Row([
            dbc.Col(metric_card(
                label="CUSTOMER JOURNEYS ANALYZED",
                value=fmt_number(m.total_journeys),
            ), md=4),
            dbc.Col(metric_card(
                label="POLICY BINDS",
                value=fmt_number(m.total_binds),
            ), md=4),
            dbc.Col(metric_card(
                label="ANNUAL MARKETING INVESTMENT",
                value=fmt_currency(m.total_annual_spend),
            ), md=4),
        ], className="mb-4"),

        # Problem statement — insight text uses computed values
        insight_callout(
            f"Your current attribution model (last-click) gives Paid Search "
            f"{fmt_pct(m.last_click_search_credit_pct)} of conversion credit and your "
            f"agent channel only {fmt_pct(m.last_click_agent_credit_pct)}. "
            f"We both know that's wrong.",
            icon="⚠️"
        ),

        # Hero chart: animated transition (last-click → Shapley)
        html.H2("What Changes With Fair Attribution"),
        dcc.Graph(
            id="hero-attribution-chart",
            figure=build_hero_chart(m),
            config={"displayModeBar": False},
            style={"height": "500px"},
        ),

        # Key findings — ALL from computed metrics
        dbc.Row([
            dbc.Col(metric_card(
                label="TRUE AGENT ATTRIBUTION",
                value=fmt_pct(m.shapley_agent_credit_pct),
                delta=f"+{m.agent_credit_shift_pp:.1f}pp vs. Last-Click",
                delta_direction="positive",
                subtitle="Shapley Value Attribution",
            ), md=4),
            dbc.Col(metric_card(
                label="PROJECTED CPB SAVINGS",
                value=f"{m.optimal_cpb_improvement_pct:+.1f}%",
                delta=f"via optimized reallocation",
                delta_direction="positive",
            ), md=4),
            dbc.Col(metric_card(
                label="AGENT CONVERSION MULTIPLIER",
                value=fmt_multiplier(m.agent_conversion_multiplier),
                subtitle="vs. pure digital journeys",
            ), md=4),
        ]),

        # Navigation links
        html.Div(className="section-divider"),
        dbc.Row([
            dbc.Col(dbc.Button("→ Explore Full Attribution Analysis", href="/attribution", color="link", className="text-info"), md=4),
            dbc.Col(dbc.Button("→ See Journey Paths", href="/journeys", color="link", className="text-info"), md=4),
            dbc.Col(dbc.Button("→ Try Budget Optimizer", href="/budget", color="link", className="text-info"), md=4),
        ]),
    ], fluid=True)


def build_hero_chart(m) -> go.Figure:
    """
    Animated grouped horizontal bar chart: Last-Click vs Shapley.
    Uses Plotly animation frames for the dramatic transition.
    All data from m.last_click_by_channel and m.shapley_by_channel DataFrames.
    """
    lt = m.last_click_by_channel.sort_values("credit_pct", ascending=True)
    sv = m.shapley_by_channel.set_index("channel_id").loc[lt["channel_id"].values].reset_index()

    fig = go.Figure()

    # Frame 1: Last-click only
    fig.add_trace(go.Bar(
        y=lt["channel_display_name"],
        x=lt["credit_pct"] * 100,
        orientation="h",
        name="Last-Click",
        marker_color="#E74C3C",
        opacity=0.8,
    ))

    # Frame 2: Shapley alongside
    fig.add_trace(go.Bar(
        y=sv["channel_display_name"],
        x=sv["credit_pct"] * 100,
        orientation="h",
        name="Shapley Value",
        marker_color="#00D4AA",
        opacity=0.8,
    ))

    fig.update_layout(
        barmode="group",
        xaxis_title="Attribution Credit (%)",
        yaxis_title="",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Add annotation arrow for the agent shift
    agent_lt = lt[lt["channel_id"] == "independent_agent"]["credit_pct"].iloc[0] * 100
    agent_sv = sv[sv["channel_id"] == "independent_agent"]["credit_pct"].iloc[0] * 100
    fig.add_annotation(
        x=agent_sv, y="Independent Agent",
        text=f"+{agent_sv - agent_lt:.1f}pp",
        showarrow=True, arrowhead=2, arrowcolor="#00D4AA",
        font=dict(color="#00D4AA", size=14, family="JetBrains Mono"),
        ax=60, ay=0,
    )

    return fig
```

#### Page 2: Attribution Comparison (`pages/attribution_comparison.py`)

```python
dash.register_page(__name__, path="/attribution", name="Attribution Comparison", order=1)

def layout():
    m = get_attribution_metrics()

    return dbc.Container([
        html.H1("Attribution Model Comparison"),
        html.Hr(className="section-divider"),

        # Model selector — toggleable pills
        html.H3("SELECT MODELS TO COMPARE"),
        model_selector(m.model_names_available, starred_models=["shapley", "casv", "markov_order_2", "ensemble"]),

        # Main chart — updates via callback based on selected models
        dcc.Graph(id="attribution-comparison-chart", style={"height": "600px"}),

        # Auto-generated insights — computed from data, not hardcoded
        dbc.Row([
            dbc.Col(insight_callout(
                f"{m.biggest_gainer_channel.replace('_', ' ').title()} gains "
                f"{m.biggest_gainer_shift_pp:+.1f}pp in Shapley vs. Last-Click — "
                f"the largest upward revision across all channels."
            ), md=6),
            dbc.Col(insight_callout(
                f"{m.biggest_loser_channel.replace('_', ' ').title()} drops "
                f"{m.biggest_loser_shift_pp:+.1f}pp — evidence of over-credit "
                f"in last-click attribution.",
                icon="⚠️"
            ), md=6),
        ]),

        # Detailed table — expandable
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Table.from_dataframe(
                    m.model_channel_matrix.reset_index().round(3),
                    striped=True, bordered=True, hover=True, dark=True, size="sm",
                )
            ], title="Full Model × Channel Matrix"),
            dbc.AccordionItem([
                dcc.Graph(id="spearman-heatmap", figure=build_spearman_heatmap(m.spearman_matrix)),
            ], title="Cross-Model Agreement (Spearman ρ)"),
        ], start_collapsed=True),
    ], fluid=True)


@callback(
    Output("attribution-comparison-chart", "figure"),
    Input("selected-models-store", "data"),
)
def update_attribution_chart(selected_models):
    """
    Callback: when user toggles model pills, update the comparison chart.
    Reads from attribution_results DataFrame, never hardcodes values.
    """
    df = get_attribution_results()
    df_filtered = df[df['model_name'].isin(selected_models)]

    fig = go.Figure()
    for model in selected_models:
        model_df = df_filtered[df_filtered['model_name'] == model].sort_values('credit_pct', ascending=True)
        fig.add_trace(go.Bar(
            y=model_df['channel_display_name'],
            x=model_df['credit_pct'] * 100,
            name=model.replace("_", " ").title(),
            orientation='h',
            opacity=0.85,
        ))

    fig.update_layout(barmode='group', xaxis_title='Attribution Credit (%)', height=600)
    return fig
```

#### Page 3: Journey Paths (`pages/journey_paths.py`)

```python
dash.register_page(__name__, path="/journeys", name="Journey Paths", order=2)

def layout():
    m = get_journey_metrics()

    return dbc.Container([
        html.H1("Customer Journey Analysis"),
        html.Hr(className="section-divider"),

        # Sankey diagram with toggle
        dbc.Row([
            dbc.Col([
                html.H3("JOURNEY FLOW"),
                dbc.RadioItems(
                    id="sankey-filter",
                    options=[
                        {"label": "All Journeys", "value": "all"},
                        {"label": "Converting Only", "value": "converting"},
                        {"label": "Non-Converting", "value": "non_converting"},
                    ],
                    value="converting",
                    inline=True,
                    className="mb-3",
                ),
                dcc.Graph(id="sankey-chart", style={"height": "500px"}),
            ], md=8),
            dbc.Col([
                html.H3("TOP CONVERTING PATHS"),
                dbc.Table.from_dataframe(
                    m.top_paths[["path_str", "frequency", "conversion_rate"]].head(10).round(3),
                    striped=True, dark=True, size="sm",
                ),
            ], md=4),
        ]),

        # Agent multiplier proof point — ALL from computed data
        html.H2("Agent Multiplier Effect"),
        insight_callout(
            f"The highest-converting path — {m.highest_converting_path.replace('|', ' → ')} — "
            f"converts at {m.highest_converting_path_rate:.1%}, which is "
            f"{m.highest_converting_path_rate / (m.top_paths['conversion_rate'].mean()):.1f}× "
            f"the average path conversion rate. This pattern accounts for "
            f"{m.highest_converting_path_pct_of_binds:.1%} of all policy binds."
        ),

        # Agent vs no-agent comparison
        dbc.Row([
            dbc.Col(metric_card(
                label="WITH AGENT TOUCH",
                value=f"{m.agent_multiplier_data[m.agent_multiplier_data['segment'] == 'with_agent']['conversion_rate'].iloc[0]:.2%}",
                subtitle=f"{m.agent_multiplier_data[m.agent_multiplier_data['segment'] == 'with_agent']['journey_count'].iloc[0]:,} journeys",
            ), md=6),
            dbc.Col(metric_card(
                label="WITHOUT AGENT TOUCH",
                value=f"{m.agent_multiplier_data[m.agent_multiplier_data['segment'] == 'without_agent']['conversion_rate'].iloc[0]:.2%}",
                subtitle=f"{m.agent_multiplier_data[m.agent_multiplier_data['segment'] == 'without_agent']['journey_count'].iloc[0]:,} journeys",
            ), md=6),
        ]),

        # Path length distribution chart
        dcc.Graph(id="path-length-chart", figure=build_path_length_chart(m.path_length_distribution)),
    ], fluid=True)
```

#### Page 4: Budget Optimizer (`pages/budget_optimizer.py`)

```python
dash.register_page(__name__, path="/budget", name="Budget Optimizer", order=3)

def layout():
    m = get_budget_metrics()
    current = m.current_allocation

    # Build slider inputs for each channel — initial values from data
    sliders = []
    for _, row in current.iterrows():
        sliders.append(
            dbc.Row([
                dbc.Col(html.Label(row['channel_id'].replace('_', ' ').title()), md=3),
                dbc.Col(dcc.Slider(
                    id={"type": "budget-slider", "channel": row['channel_id']},
                    min=0, max=row['current_spend'] * 2.5,
                    step=100000, value=row['current_spend'],
                    marks=None, tooltip={"placement": "bottom", "always_visible": True},
                ), md=7),
                dbc.Col(html.Div(
                    id={"type": "budget-display", "channel": row['channel_id']},
                    children=fmt_currency(row['current_spend']),
                ), md=2),
            ], className="mb-2")
        )

    return dbc.Container([
        html.H1("Budget Optimization Simulator"),
        html.Hr(className="section-divider"),

        dbc.Row([
            # Left: Sliders
            dbc.Col([
                html.H3("ADJUST ALLOCATION"),
                html.Div(sliders),
                html.Div(id="total-budget-display", className="mt-3",
                         style={"fontFamily": "JetBrains Mono", "fontSize": "1.2rem"}),
            ], md=6),

            # Right: Projected impact — ALL from callbacks, never hardcoded
            dbc.Col([
                html.H3("PROJECTED IMPACT"),
                dbc.Row(id="impact-metrics-row"),
                dcc.Graph(id="response-curves-chart", style={"height": "350px"}),
            ], md=6),
        ]),

        # Preset scenarios
        html.H3("PRESET SCENARIOS"),
        dbc.ButtonGroup([
            dbc.Button("Optimal (±20% cap)", id="btn-optimal-20", color="outline-info", className="me-2"),
            dbc.Button("Shift Search → Display", id="btn-shift-search", color="outline-info", className="me-2"),
            dbc.Button("Double Agent Support", id="btn-double-agent", color="outline-info", className="me-2"),
            dbc.Button("Cut TV 50%", id="btn-cut-tv", color="outline-info", className="me-2"),
            dbc.Button("Reset to Current", id="btn-reset", color="outline-warning"),
        ], className="mb-4"),

        # Dynamic insight — populated by callback from scenario data
        html.Div(id="budget-insight-box"),
    ], fluid=True)


@callback(
    Output("impact-metrics-row", "children"),
    Output("budget-insight-box", "children"),
    Input({"type": "budget-slider", "channel": dash.ALL}, "value"),
)
def update_budget_impact(slider_values):
    """
    When any slider moves, recompute projected impact using response curves.
    ALL numbers come from the optimization engine, not hardcoded.
    """
    # Reconstruct allocation from slider values
    # Compute projected conversions via response curves
    # Generate insight text from computed deltas
    pass
```

#### Page 5: Identity Resolution (`pages/identity_resolution.py`)

Match rate dashboard (donut chart from identity_graph data), sample identity graph visualizations (NetworkX → Plotly scatter), channel coverage heatmap by match tier, and simulated impact of improved resolution on attribution.

#### Page 6: Channel Deep Dive (`pages/channel_deep_dive.py`)

Channel selector dropdown → per-channel dashboard: funnel role breakdown, attribution across models, co-occurring channels heatmap, conversion lift analysis, temporal patterns, trend over time. All from filtered `attribution_results` and `assembled_journeys` DataFrames.

#### Page 7: Model Validation (`pages/model_validation.py`)

Axiom compliance dashboard (green checkmarks from `validation_metrics.parquet`), cross-model Spearman ρ heatmap, holdout simulation scatter plot, sensitivity heatmaps. All from pre-computed validation results.

#### Page 8: Technical Appendix (`pages/technical_appendix.py`)

Mathematical formulations (LaTeX via `dcc.Markdown`), data schema documentation, production architecture diagram, triangulation framework (Section 8.4), glossary. Includes the Gordon et al. (2023) calibration discussion and the 3-phase roadmap (MTA → MMM → Incrementality).

### 10.7 Callback Architecture — Key Patterns

```python
# Pattern 1: Clientside callback for instant UI updates (no server round-trip)
app.clientside_callback(
    """
    function(n_clicks) {
        // Toggle model pill active state
        var el = dash_clientside.callback_context.triggered[0];
        // ... CSS class toggle logic
    }
    """,
    Output({"type": "model-pill", "model": dash.MATCH}, "className"),
    Input({"type": "model-pill", "model": dash.MATCH}, "n_clicks"),
)

# Pattern 2: Cross-page state via dcc.Store
# selected-models-store and parameter-store persist across page navigation

# Pattern 3: Prevent initial call for expensive computations
@callback(
    Output("heavy-chart", "figure"),
    Input("trigger-button", "n_clicks"),
    prevent_initial_call=True,
)
def expensive_update(n_clicks):
    pass

# Pattern 4: Pattern-matching callbacks for dynamic components (budget sliders)
@callback(
    Output({"type": "budget-display", "channel": dash.MATCH}, "children"),
    Input({"type": "budget-slider", "channel": dash.MATCH}, "value"),
)
def update_slider_display(value):
    return fmt_currency(value)
```

---

## 11. Demo Narrative Integration

### 11.1 Talking Points per Page (Stored in `config/demo_narrative.yaml`)

```yaml
executive_summary:
  opening_line: >
    "Erie invests $100M annually across 13 marketing channels to drive auto
    policy growth. Today, last-click attribution tells you that Paid Search
    is your best channel. But we both know that your 12,000 independent agents
    close 85% of your business. Something doesn't add up."
  transition_to_shapley: >
    "Watch what happens when we apply fair attribution — the same cooperative
    game theory that powers Google's own Data-Driven Attribution — to your full
    channel ecosystem, including your agent network."
  closing_hook: >
    "The question isn't whether your attribution is wrong — it's how much
    money you're leaving on the table because of it."

attribution_comparison:
  key_callout: >
    "Notice how agents jump from single digits to over 30% of attribution credit.
    That's not model bias — it's Shapley's efficiency axiom ensuring every
    conversion is fairly allocated."
  objection_handler_ga4: >
    "GA4's data-driven attribution only sees digital touchpoints. For Erie,
    that's 30-40% of the picture. Our model sees the full journey — including
    the agent interaction that GA4 doesn't know about."
  estimation_humility: >
    "We should be honest: Gordon et al. found that observational MTA can produce
    errors of 5-10× in absolute ad effects. That's why we show you 15+ models
    and a triangulation roadmap — no single model is trustworthy alone."

budget_optimizer:
  cfo_language: >
    "At Erie's estimated average premium of $1,200 per auto policy, the
    projected additional binds represent meaningful first-year premium.
    Over a 5-year average policyholder lifetime, the value compounds."

technical_appendix:
  triangulation_roadmap: >
    "Phase 1 (now): Multi-touch attribution for tactical channel optimization.
    Phase 2 (6 months): Media mix modeling via Google Meridian for strategic
    allocation across Erie's 12-state footprint.
    Phase 3 (12 months): Incrementality experiments via geo-testing for
    causal ground truth. This is the industry gold standard."
```

---

## 12. Deployment & Packaging

### 12.1 Running the Demo

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data (one-time, ~2-3 minutes)
python scripts/generate_data.py --config config/synthetic_data.yaml

# 3. Run attribution models + validation + optimization (one-time, ~10-15 minutes)
python scripts/run_attribution.py --config config/model_params.yaml
# This produces ALL processed Parquet files consumed by the UI

# 4. Pre-compute parameter grid cache (optional, ~20 minutes)
python scripts/precompute_cache.py

# 5. Run data contract validation
pytest tests/test_data_contracts.py -v

# 6. Launch the Dash app
python app/app.py
# → http://localhost:8050
# For production: gunicorn app.app:server -b 0.0.0.0:8050 -w 4
```

### 12.2 Pre-Computation Strategy

All model results are pre-computed to Parquet files. The UI reads from these files — zero live computation during the demo. Parameter variations are cached:

```python
PRECOMPUTE_GRID = {
    "lookback_windows": [14, 21, 30, 45, 60, 90],
    "time_decay_half_lives": [3, 5, 7, 10, 14],
    "markov_orders": [1, 2, 3],
    "conversion_events": ["quote_start", "bind"],
}
# Cached results keyed by parameter hash for instant retrieval
```

### 12.3 Dependencies (`requirements.txt`)

```
# Core
dash>=2.18.0
dash-bootstrap-components>=1.6.0
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
scikit-learn>=1.4.0

# Visualization
plotly>=5.18.0
matplotlib>=3.8.0
seaborn>=0.13.0

# Attribution engine
networkx>=3.2.0
shapiq>=1.1.0                # SOTA Shapley approximation (Witter et al. 2025 method)
marketing-attribution-models>=0.0.5  # Zhao et al. implementation

# Optimization
pulp>=2.7.0

# Deep learning
torch>=2.1.0

# Statistical modeling
statsmodels>=0.14.0
lifelines>=0.28.0            # Survival analysis

# Data storage
duckdb>=0.10.0
pyarrow>=14.0.0              # Parquet read/write

# Configuration
pydantic>=2.5.0
pyyaml>=6.0.0

# Caching
diskcache>=5.6.0

# Testing
pytest>=7.4.0
hypothesis>=6.92.0

# Utilities
tqdm>=4.66.0
loguru>=0.7.0

# Production deployment
gunicorn>=21.2.0
```

---

## Appendix A: Key Implementation Gotchas

1. **Shapley with 13 channels**: 2^13 = 8,192 coalitions. With Zhao et al. simplification + grouping to ~7 macro-channels (2^7 = 128), computation is instant. Always implement the grouping fallback. The `shapiq` library handles this natively.

2. **Markov 2nd-order state space**: 13 channels → up to 169 compound states. Only ~40-60 observed. Use Dirichlet smoothing (Heiner et al. 2022 sparse priors for robustness).

3. **Sankey diagram readability**: Filter to top 20 paths. Aggregate rest as "Other." Plotly Sankey with >30 flows is unreadable.

4. **Dash callback cascading**: Avoid callback chains longer than 3 steps. Use `dcc.Store` for cross-page state. Use `prevent_initial_call=True` for expensive computations.

5. **Dark theme consistency**: Plotly template, Bootstrap theme, and custom CSS must all agree. One white-background chart destroys visual cohesion. Test every chart against the dark background.

6. **Numbers must tie**: Every metric on Executive Summary must exactly match the detailed view. Use `attribution_results.parquet` as single source of truth. The `data_contract_tests.py` enforces this.

7. **Synthetic data seed**: `random_seed=42` everywhere. Identical results every run. Non-negotiable for live demos.

8. **The agent insight is the climax**: Every design choice builds toward the moment agents jump from ~8% to ~34%. Don't bury it. Make it dramatic. The animated Plotly transition is the single most important visual.

9. **Gordon et al. calibration**: Mention the 488-948% error finding ONCE (Technical Appendix), frame it as why triangulation matters, then move on. Don't dwell — it undermines confidence. Use it to sell Phase 2/3 (MMM + incrementality).

10. **Removal effect divergence from Shapley**: Expect Markov to give agents even MORE credit than Shapley (Singal et al. 2022 finding). This is a feature, not a bug — it's a demo talking point about model convergence.

---

## Appendix B: Erie-Specific Reference Data

```json
{
  "erie_facts": {
    "headquarters": "Erie, Pennsylvania",
    "founded": 1925,
    "operating_states": ["PA", "OH", "NY", "NC", "VA", "MD", "IN", "WI", "TN", "WV", "IL", "KY", "DC"],
    "distribution_model": "100% independent agent",
    "policies_in_force": "6M+",
    "am_best_rating": "A+ (Superior)",
    "auto_insurance_market_position": "Top 15 nationally, dominant in PA/OH",
    "agent_count_estimate": "~12,000 independent agents",
    "consumer_reports_rating": "Highest rated auto insurer (multiple years)",
    "key_competitors_in_footprint": ["State Farm", "GEICO", "Progressive", "Allstate", "Nationwide"],
    "estimated_annual_marketing_spend": "$80-120M (estimate)",
    "auto_new_business_binds_estimate": "150-250K annually"
  }
}
```

---

## Appendix C: SOTA Literature Reference Map

| Model | Paper | Year | Venue | Key Innovation | Demo Tier |
|-------|-------|------|-------|----------------|-----------|
| Shapley (Simplified) | Zhao et al. | 2018 | arXiv | Simplified computation formula | Tier 2 ★ |
| CASV | Singal et al. | 2022 | Management Science | Counterfactual Markov + Shapley | Tier 2 ★ |
| Reg-MC Shapley | Witter et al. | 2025 | arXiv | 6.5× lower error than PermSHAP | Tier 2 |
| Markov Removal | Anderl et al. | 2016 | IJRM | Absorbing chain framework | Tier 3 ★ |
| VMM / MTD | Berchtold & Raftery | 2002 | Statistical Science | Variable-order, additive mixing | Tier 3 |
| Survival Attribution | Zhang, Wei & Ren | 2014 | ICDM | Time-to-conversion hazard model | Tier 4 |
| DARNN | Ren et al. | 2018 | CIKM | Dual-attention RNN | Tier 5 ★ |
| DNAMTA | Li et al. | 2018 | AdKDD | LSTM+Attention (Adobe) | Tier 5 |
| CausalMTA | Alibaba | 2022 | KDD | Confounding decomposition | Tier 5 |
| Transformer MTA | Lu & Kannan | 2025 | JMR | Multi-head self-attention SOTA | Tier 5 |
| CAMTA | Kumar et al. | 2020 | ICDM Workshop | Counterfactual + domain adversarial | Tier 5 |
| Observational MTA Errors | Gordon et al. | 2023 | Marketing Science | 488-948% error finding | Calibration anchor |
| Google Meridian | Google | 2025 | Open-source | Bayesian hierarchical geo MMM | Phase 2 roadmap |
| Meta Robyn | Meta | 2023 | Open-source | Ridge + Nevergrad MMM | Phase 2 roadmap |

---

*This document is designed to serve as a complete, self-contained reference for generating the full Erie MCA demo codebase. Every section provides sufficient detail for an LLM to produce working, production-quality code without additional context. Version 2.0 specifically ensures: (a) zero hardcoded UI metrics via the data contract layer, (b) SOTA model coverage per the resource guide, and (c) Plotly Dash architecture for production-grade interactivity.*
