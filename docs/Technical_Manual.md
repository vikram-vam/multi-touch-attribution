# MCA Intelligence — Technical Document
## Erie Insurance Multi-Channel Attribution: Architecture, Mathematics & Implementation

**Version:** 1.0 | **Date:** February 2026  
**Audience:** Analytics Team, Data Engineers, Implementation Developers, VP of Analytics  
**Scope:** Complete technical specification for the MCA capability demo  
**Companion To:** Business User Manual (non-technical stakeholder guide)  

---

## Table of Contents

1. [Solution Architecture Overview](#1-solution-architecture-overview)
2. [Technology Stack & Dependencies](#2-technology-stack--dependencies)
3. [Repository Structure & Module Architecture](#3-repository-structure--module-architecture)
4. [Configuration System](#4-configuration-system)
5. [Data Contracts & Schema Specifications](#5-data-contracts--schema-specifications)
6. [Synthetic Data Engine](#6-synthetic-data-engine)
7. [Identity Resolution & Journey Assembly](#7-identity-resolution--journey-assembly)
8. [Attribution Model Arsenal — Mathematical Formulations](#8-attribution-model-arsenal--mathematical-formulations)
9. [Model Comparison Framework](#9-model-comparison-framework)
10. [Budget Optimization Layer](#10-budget-optimization-layer)
11. [UI/UX Architecture — Plotly Dash Application](#11-uiux-architecture--plotly-dash-application)
12. [Data Pipeline & Execution](#12-data-pipeline--execution)
13. [Deployment & Infrastructure](#13-deployment--infrastructure)
14. [Validation & Quality Assurance](#14-validation--quality-assurance)
15. [Implementation Gotchas & Edge Cases](#15-implementation-gotchas--edge-cases)
16. [Appendix A: Erie Insurance Domain Reference](#a-erie-insurance-domain-reference)
17. [Appendix B: Academic References](#b-academic-references)
18. [Appendix C: Full Configuration Schema](#c-full-configuration-schema)

---

## 1. Solution Architecture Overview

### 1.1 Design Philosophy

This system is built on five architectural principles that govern every implementation decision.

**Principle 1 — YAML Is the Single Source of Truth.** Every parameter — channel names, funnel rates, model hyperparameters, UI colors, scenario definitions — lives in `erie_mca_synth_config.yaml`. Zero hardcoded values in Python code. This enables scenario analysis, sensitivity testing, and client customization without code changes.

**Principle 2 — Parquet Is the Data Contract.** Modules communicate exclusively through typed Parquet files with schemas defined in Section 5. This enforces clean module boundaries, enables independent testing, and means the UI layer never imports model code directly.

**Principle 3 — metrics/ Is the UI's Only Data Source.** The Dash application reads exclusively from pre-aggregated metric files in the `metrics/` directory. No raw model outputs, no live computation in callbacks, no direct imports from attribution modules. This ensures sub-second page loads and eliminates runtime errors during demos.

**Principle 4 — Three-Model Convergence Over Single-Model Precision.** The demo's credibility comes from showing that three fundamentally different mathematical frameworks agree on core insights, not from claiming that any single model is "correct." The comparison framework is therefore as important as the models themselves.

**Principle 5 — Pre-Compute Everything.** The full pipeline (data generation → resolution → attribution → comparison → optimization → metric aggregation) runs as a batch process. The Dash app is a read-only visualization layer. This separation means the demo never waits for computation, never crashes from a model error, and always presents consistent results.

### 1.2 End-to-End Data Flow

```
erie_mca_synth_config.yaml
         │
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  GENERATORS      │ ──► │  RESOLUTION       │ ──► │  ATTRIBUTION     │
│  population      │     │  identity_resolver│     │  baselines       │
│  behavioral_sim  │     │  journey_assembler│     │  shapley         │
│  system_records  │     │  resolution_report│     │  markov          │
│  dirty_data      │     │                  │     │  constrained_opt │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                         │
    source_systems/          resolved/               attribution/
    *.parquet               *.parquet                *.parquet
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │  COMPARISON       │
                                                │  head_to_head     │
                                                │  convergence_map  │
                                                │  dual_funnel      │
                                                └────────┬─────────┘
                                                         │
                                                    comparison/
                                                    *.parquet
                                                         │
                                                         ▼
                                                ┌──────────────────┐
                                                │  OPTIMIZATION     │
                                                │  budget_optimizer │
                                                │  scenario_engine  │
                                                │  sensitivity      │
                                                └────────┬─────────┘
                                                         │
                                                   optimization/
                                                    *.parquet
                                                         │
                                                         ▼
                                                ┌──────────────────┐
                                                │  METRICS          │
                                                │  aggregator       │
                                                │  (pre-computed    │
                                                │   UI data stores) │
                                                └────────┬─────────┘
                                                         │
                                                    metrics/
                                                    *.parquet
                                                         │
                                                         ▼
                                                ┌──────────────────┐
                                                │  DASH APP         │
                                                │  8 pages + appx   │
                                                │  READ-ONLY        │
                                                └──────────────────┘
```

Data flows strictly left-to-right (or top-to-bottom). No module reaches backward. No circular imports. The Dash app sits at the terminal end and reads only from `metrics/`.

### 1.3 Success Criteria (Technical)

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| Shapley efficiency axiom | |Σ credits - total conversions| | < 0.01% |
| Cross-model convergence | Spearman ρ between primary models | > 0.70 |
| Agent insight delivery | Model-based agent credit vs. last-click | ≥ +30 percentage points |
| Data contract compliance | All Parquet files match Section 5 schemas | 100% |
| UI data binding | `grep` for hardcoded numeric values in pages/ | Zero matches |
| Page load time | Time from navigation to render complete | < 3 seconds |
| Pipeline total runtime | Config → metrics/ complete | < 15 minutes |
| Zero crashes | Dash app stability during 60-minute session | 100% uptime |

---

## 2. Technology Stack & Dependencies

### 2.1 Core Stack

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Web Framework | Plotly Dash | 2.18+ | Multi-page application with callback-driven interactivity |
| UI Components | Dash Bootstrap Components | 1.6+ | Responsive layout, cards, navbars, modals |
| Charting | Plotly | 5.x | All visualizations (bar, heatmap, waterfall, funnel, sankey) |
| Data Manipulation | Pandas | 2.x | DataFrame operations, Parquet I/O |
| Numerical Computing | NumPy | 1.26+ | Array operations, matrix math |
| Graph Analysis | NetworkX | 3.x | Markov chain graph construction, removal effects, identity graph |
| Scientific Computing | SciPy | Latest | SLSQP optimizer, statistical tests, bootstrap, Spearman ρ |
| Convex Optimization | CVXPY | Latest | Constrained optimization with dual variable extraction |
| MILP Solver | PuLP (CBC backend) | Latest | Budget optimization with integer variables |
| Configuration | PyYAML | Latest | Config file parsing |
| Schema Validation | Pydantic v2 | Latest | Config schema validation with field validators |
| Deployment | Gunicorn | Latest | WSGI server for production |

### 2.2 Deliberately Excluded

| Excluded | Reason |
|----------|--------|
| PyTorch / TensorFlow | No deep learning models. All 15 variants are CPU-based, seconds runtime. |
| DuckDB | Parquet files are small enough for direct Pandas reads. |
| Streamlit | Insufficient layout control for 8-page professional demo. |
| shapiq | Custom Shapley implementation follows Zhao et al. simplification. |
| matplotlib | Plotly handles all visualization needs natively in Dash. |

### 2.3 Python Version

Python 3.11+ required. The codebase uses type hints, match statements, and modern syntax throughout.

---

## 3. Repository Structure & Module Architecture

### 3.1 Directory Tree

```
erie-mca-demo/
│
├── config/
│   └── erie_mca_synth_config.yaml        # 777-line master config
│
├── src/
│   ├── config/
│   │   ├── schema.py                      # Pydantic v2 models for config validation
│   │   └── loader.py                      # YAML → validated Python objects
│   │
│   ├── generators/
│   │   ├── population_generator.py        # 25K prospect profiles
│   │   ├── behavioral_simulator.py        # Decision-state engine
│   │   ├── system_record_generator.py     # Fragment into source systems
│   │   └── dirty_data_injector.py         # Realistic data quality issues
│   │
│   ├── resolution/
│   │   ├── identity_resolver.py           # Three-tier matching
│   │   ├── journey_assembler.py           # Unified journey construction
│   │   └── resolution_reporter.py         # Quality metrics
│   │
│   ├── attribution/
│   │   ├── base.py                        # BaseAttributionModel ABC + output contract
│   │   ├── baselines.py                   # 5 baseline models
│   │   ├── shapley.py                     # Standard, time-weighted, CASV
│   │   ├── markov.py                      # 1st, 2nd, 3rd order chains
│   │   ├── constrained_optimization.py    # 4 feature encodings
│   │   └── model_runner.py                # Registry + batch execution
│   │
│   ├── comparison/
│   │   ├── head_to_head.py                # Pairwise metrics (Spearman, JSD)
│   │   ├── convergence_map.py             # Per-channel confidence zones
│   │   └── dual_funnel_analysis.py        # Quote vs. bind comparison
│   │
│   ├── optimization/
│   │   ├── response_curves.py             # Saturating exponential calibration
│   │   ├── budget_optimizer.py            # MILP with piecewise linearization
│   │   ├── scenario_engine.py             # 7 pre-built + custom scenarios
│   │   └── sensitivity_analyzer.py        # Shadow prices + constraint analysis
│   │
│   ├── metrics/
│   │   └── aggregator.py                  # Pre-compute all UI data stores
│   │
│   ├── pipeline/
│   │   ├── run_full_pipeline.py           # End-to-end execution
│   │   └── run_attribution_only.py        # Re-run models without regenerating data
│   │
│   └── utils/
│       ├── formatters.py                  # fmt_pct, fmt_currency, fmt_number
│       └── validators.py                  # Schema validation helpers
│
├── app/
│   ├── app.py                             # Dash app initialization + multi-page setup
│   ├── data_store.py                      # Load all metrics/ Parquet at startup
│   ├── theme.py                           # Erie brand colors, fonts, chart defaults
│   ├── components/
│   │   ├── metric_card.py                 # Reusable KPI card
│   │   ├── insight_callout.py             # Highlighted insight box
│   │   ├── model_selector.py              # Model toggle pills
│   │   ├── attribution_bars.py            # Horizontal grouped bar chart
│   │   ├── channel_heatmap.py             # Markov transition heatmap
│   │   ├── budget_sliders.py              # Interactive budget simulator
│   │   ├── convergence_table.py           # Color-coded convergence zones
│   │   ├── waterfall_chart.py             # Credit decomposition waterfall
│   │   └── navbar.py                      # Navigation bar
│   │
│   └── pages/
│       ├── executive_summary.py           # Page 1: /
│       ├── identity_resolution.py         # Page 2: /identity
│       ├── model_comparison.py            # Page 3: /models
│       ├── channel_deep_dive.py           # Page 4: /channels
│       ├── dual_funnel.py                 # Page 5: /funnel
│       ├── budget_optimization.py         # Page 6: /budget
│       ├── scenario_explorer.py           # Page 7: /scenarios
│       ├── measurement_roadmap.py         # Page 8: /roadmap
│       └── technical_appendix.py          # /appendix
│
├── data/
│   ├── source_systems/                    # Generated system extracts
│   ├── resolved/                          # Identity resolution output
│   ├── attribution/                       # All model outputs
│   ├── comparison/                        # Head-to-head analysis
│   ├── optimization/                      # Budget recommendations
│   └── metrics/                           # Pre-aggregated UI data
│
├── tests/
│   ├── test_config.py
│   ├── test_generators.py
│   ├── test_attribution.py
│   ├── test_schemas.py
│   └── test_axioms.py                     # Property-based Shapley axiom tests
│
├── Dockerfile
├── render.yaml
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md
```

### 3.2 Module Dependency Rules

Each module may only import from modules above it in the pipeline. The strict ordering is: config → generators → resolution → attribution → comparison → optimization → metrics → app/pages. No module reaches backward. The `app/` directory imports only from `app/data_store.py`, which reads only from `data/metrics/`.

---

## 4. Configuration System

### 4.1 Master Configuration File

The file `erie_mca_synth_config.yaml` (777 lines, 10 sections) drives the entire pipeline. Every numeric value, channel name, model parameter, and business assumption is externalized here.

**10 Sections:**

1. **simulation_control:** Random seeds, prospect count (25,000), simulation period, dirty data toggle, scenario override.
2. **population:** Geographic mix (PA 40%, OH 20%, other 40%), age distribution, digital sophistication tiers (H:30%, M:45%, L:25%), shopping trigger distribution.
3. **conversion_funnel:** Base rates per decision state, agent involvement rate (0.87), mean journey lengths, attribution windows (30 days), conversion events.
4. **channel_taxonomy:** 13 channels with IDs, display names, channel types (paid/owned/earned), touchpoint types (digital/offline/hybrid), cost data, and base interaction probabilities per decision state.
5. **state_interaction_matrix:** Per-channel interaction probabilities across four decision states (awareness, consideration, intent, action). Agent interaction spikes from 0.02 in awareness to 0.65 in action state.
6. **journey_dynamics:** State transition rules, channel-specific transition boosts, recency half-life (7 days), minimum/maximum touchpoints per journey.
7. **identity_resolution:** Tier-specific match rules and confidence thresholds, fragmentation scenario distribution, call tracking coverage (0.45), cross-device gap rates.
8. **data_quality:** Dirty data injection rates per type (duplicates 4%, timestamp drift 2.5%, missing fields 6%, stale cookies 1.5%, agent code inconsistency 5%, bot traffic 2.5%, retroactive PAS 3.5%, email bounces 6%).
9. **attribution_models:** Per-model hyperparameters (Shapley half-life, Markov smoothing alpha 0.077, OR agent floor β=1.5, concentration cap γ=0.30), model registry, bootstrap iterations (200), sensitivity runs (20).
10. **budget_optimizer:** Current allocation ($5M across 10 channels), saturation factors (0.35–0.80), channel minimum floors ($10K–$200K), constraints (max reallocation 20%, min channels 8, agent floor 2%, TV min 10%, search cap 25%), 7 scenario presets.

### 4.2 Pydantic v2 Schema

All config sections are validated at load time through Pydantic v2 models with field validators. Invalid configurations fail fast with descriptive error messages before any computation begins.

Key validations include: channel interaction probabilities must be within [0, 1], budget allocations must sum to total_budget, saturation factors must be within (0, 1), all referenced channel_ids must exist in channel_taxonomy, and scenario preset overrides must reference valid config paths.

### 4.3 Scenario Override Pattern

Scenarios modify specific config parameters without duplicating the full configuration:

```yaml
scenario_presets:
  digital_transformation:
    description: "Agent involvement drops to 65%"
    overrides:
      conversion_funnel.agent_involvement_rate: 0.65
      state_interaction_matrix.agent_interaction: [0.02, 0.05, 0.18, 0.42]
```

The config loader applies overrides on top of the base configuration, preserving all other parameters.

---

## 5. Data Contracts & Schema Specifications

### 5.1 Source System Schemas

These Parquet files simulate the fragmented data systems that a real Erie integration would encounter.

**ga4_sessions.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| ga4_client_id | string | First-party cookie ID |
| session_id | string | Unique session identifier |
| session_start | datetime | Session start timestamp |
| landing_page | string | First page URL |
| traffic_source | string | utm_source / referrer |
| traffic_medium | string | utm_medium |
| traffic_campaign | string | utm_campaign |
| gclid | string (nullable) | Google Click ID if paid search |
| events | list[dict] | Page views, quote starts, agent locator clicks |
| device_category | string | desktop / mobile / tablet |
| geo_state | string | Two-letter state code |

**crm_contacts.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| crm_id | string | CRM contact identifier |
| email_hash | string | SHA-256 hashed email |
| phone_hash | string | SHA-256 hashed phone |
| first_name | string | Contact first name |
| last_name | string | Contact last name |
| address_line1 | string | Street address |
| city | string | City |
| state | string | Two-letter state code |
| zip_code | string | 5-digit ZIP |
| lead_source | string | Original lead source channel |
| assigned_agent_id | string | Agent code in AMS |
| created_date | date | Contact creation date |

**ams_records.parquet (Agent Management System):**
| Column | Type | Description |
|--------|------|-------------|
| agent_id | string | Unique agent identifier |
| agent_name | string | Agent display name |
| agency_name | string | Agency name |
| interaction_id | string | Unique interaction record |
| contact_identifier | string | Agent's record for the customer (may be name, phone, or internal ID) |
| interaction_type | string | office_visit / phone_call / email / referral |
| interaction_date | datetime | When the interaction occurred |
| notes | string | Free text notes |
| outcome | string | quote_requested / follow_up / bind_submitted / no_action |

**pas_policies.parquet (Policy Administration System):**
| Column | Type | Description |
|--------|------|-------------|
| policy_number | string | Unique policy identifier |
| policyholder_name | string | Full name on policy |
| policyholder_email_hash | string | SHA-256 hashed email |
| policyholder_phone_hash | string | SHA-256 hashed phone |
| writing_agent_id | string | Agent who bound the policy |
| quote_start_date | datetime | When quote was initiated |
| quote_complete_date | datetime (nullable) | When quote was completed |
| bind_date | datetime (nullable) | When policy was bound |
| premium_annual | float | Annual premium amount |
| product_line | string | auto / home / umbrella |
| state | string | Policy state |
| bind_channel | string | agent / online / phone |

**call_tracking.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| call_id | string | Unique call identifier |
| tracking_number | string | Dynamic tracking number |
| caller_phone_hash | string | SHA-256 hashed caller phone |
| call_start | datetime | Call start timestamp |
| call_duration_seconds | int | Duration |
| call_outcome | string | connected / voicemail / abandoned |
| web_session_id | string (nullable) | Linked GA4 session if available |
| landing_page | string (nullable) | Page that triggered the call |
| destination_agent_id | string | Agent who received the call |

Additional source system schemas: **google_ads_clicks.parquet**, **esp_events.parquet** (email service provider), **direct_mail.parquet**.

### 5.2 Resolution Output Schemas

**identity_graph.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| unified_id | string | Resolved identity identifier |
| source_system | string | Which system this record came from |
| source_record_id | string | Original record ID in source system |
| match_tier | int | Resolution tier (1, 2, or 3) |
| match_rule | string | Which rule linked this record |
| match_confidence | float | Confidence score (0.60–0.99) |
| anchor_node | string | CRM contact ID if available |

**unified_journeys.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| unified_id | string | Resolved identity |
| touchpoint_id | string | Unique touchpoint identifier |
| channel_id | string | One of 13 channel IDs from taxonomy |
| channel_name | string | Display name |
| touchpoint_type | string | impression / click / call / visit / email / mail |
| timestamp | datetime | When the interaction occurred |
| days_to_conversion | float (nullable) | Days until conversion (null for non-converters) |
| time_weight | float | Exponential decay weight |
| decision_state | string | awareness / consideration / intent / action |
| source_system | string | Origin system |
| match_tier | int | How this touchpoint was linked |
| conversion_event | string (nullable) | quote_start / bind / null |
| journey_length | int | Total touchpoints in this journey |
| channel_path | string | Ordered channel sequence (e.g., "Display→Search→Agent") |
| channel_set | string | Unordered unique channel set |
| is_converting | bool | Whether journey resulted in conversion |

### 5.3 Attribution Output Schemas

**channel_credits.parquet (THE central attribution table):**
| Column | Type | Description |
|--------|------|-------------|
| model_id | string | e.g., "shapley_time_weighted" |
| conversion_event | string | "quote_start" or "bind" |
| channel_id | string | One of 13 channels |
| channel_name | string | Display name |
| total_credit | float | Absolute credited conversions |
| credit_share | float | Percentage of total (sums to 1.0 per model×event) |
| credit_rank | int | Rank position (1 = highest) |
| journeys_involved | int | Journeys where this channel appeared |
| avg_credit_per_journey | float | Mean credit when present |
| ci_lower | float | 95% bootstrap CI lower bound |
| ci_upper | float | 95% bootstrap CI upper bound |

**journey_credits.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| journey_id | string | Links to unified_journeys |
| model_id | string | Which model produced this |
| conversion_event | string | Which event was attributed |
| channel_credits | dict | {channel_id: credit_amount} |
| dominant_channel | string | Channel with highest credit |
| credit_entropy | float | Distribution evenness (0=concentrated, high=spread) |

**model_diagnostics.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| model_id | string | Model identifier |
| variant_params | dict | Key hyperparameters used |
| computation_time_sec | float | Wall-clock runtime |
| convergence_achieved | bool | Solver converged? |
| data_coverage | float | % of journeys attributed |
| efficiency_error | float | |Σ credits - total conversions| |

### 5.4 Comparison Output Schemas

**pairwise_metrics.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| model_a | string | First model |
| model_b | string | Second model |
| conversion_event | string | quote_start or bind |
| spearman_rho | float | Rank correlation |
| spearman_pvalue | float | Statistical significance |
| jensen_shannon_div | float | Credit distribution divergence |

**convergence_map.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| channel_id | string | Channel |
| shapley_credit | float | Time-weighted Shapley credit share |
| markov_credit | float | 2nd-order Markov credit share |
| or_credit | float | Constrained opt (recency) credit share |
| last_click_credit | float | Last-digital-click baseline |
| confidence_zone | string | HIGH / MODERATE / LOW |
| consensus_direction | string | INCREASE / DECREASE / HOLD / INVESTIGATE |
| credit_range_pp | float | Max - min across 3 models (percentage points) |

### 5.5 Optimization Output Schemas

**budget_recommendations.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| channel_id | string | Channel |
| current_spend | float | Current budget allocation |
| shapley_optimal | float | Optimizer result using Shapley credits |
| markov_optimal | float | Optimizer result using Markov credits |
| or_optimal | float | Optimizer result using OR credits |
| consensus_spend | float | Average of agreeing models |
| consensus_direction | string | INCREASE / DECREASE / HOLD / INVESTIGATE |
| delta_pct | float | Recommended change percentage |

**response_curves.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| channel_id | string | Channel |
| a_ceiling | float | Saturation ceiling parameter |
| b_efficiency | float | Efficiency rate parameter |
| saturation_factor | float | Current saturation level |
| spend_points | list[float] | Evaluation points for curve plotting |
| conversion_points | list[float] | Predicted conversions at each spend level |
| marginal_roi_at_current | float | Derivative at current spend |

**shadow_prices.parquet:**
| Column | Type | Description |
|--------|------|-------------|
| constraint_name | string | Constraint identifier |
| constraint_description | string | Business-language description |
| dual_value | float | Shadow price (marginal value of relaxation) |
| is_binding | bool | Whether constraint is active at optimum |
| business_implication | string | Pre-generated interpretation |

---

## 6. Synthetic Data Engine

### 6.1 Four-Layer Architecture

The synthetic data engine generates realistic customer journeys through behavioral simulation rather than hardcoded transition matrices. Attribution patterns emerge naturally from simulated prospect behavior.

**Layer 1 — Population Generator:**
Produces 25,000 prospect profiles with demographics (age, household composition), geography (state distribution: PA 40%, OH 20%, other 40%), digital sophistication tier (High 30%, Medium 45%, Low 25%), and shopping trigger type (life event, renewal, price shopping, new driver, relocation).

**Layer 2 — Behavioral Simulator (Decision-State Engine):**
The core simulation engine. Each prospect progresses through four decision states: Awareness → Consideration → Intent → Action. At each simulated day, the engine determines which channels the prospect interacts with (based on the state interaction matrix and the prospect's digital sophistication) and whether the prospect transitions to a new state (boosted by channels interacted with).

Key behavioral dynamics: Agent interaction probability spikes from 2% in awareness to 65% in the action state. Conversion requires agent interaction for the majority of binds (controlled by agent_involvement_rate = 0.87). The "digital-to-agent bridge" pattern emerges naturally as prospects in the intent and action states increasingly interact with Agent Locator and Brand Search, which then transition into agent interactions.

**Layer 3 — System Record Generator:**
Fragments each prospect's unified journey into source-system-specific records, simulating the real-world data fragmentation challenge. A single prospect may generate records in GA4, CRM, AMS, PAS, call tracking, ESP, and ad platforms — each with different identifiers and varying levels of linkage.

Identity fragmentation scenarios are applied per the config distribution: clean match (25%), cookie churn (20%), digital-only with no agent link (15%), agent-only with no digital footprint (12%), partial bridge (18%), and cross-device gap (10%).

**Layer 4 — Dirty Data Injector:**
Applies realistic data quality issues to simulate production data challenges. Each dirty data type has an independent injection rate: duplicate CRM records (4%), timestamp drift between systems (2.5%), missing required fields (6%), stale or churned cookies (1.5%), agent code inconsistency (5%), bot traffic (2.5%), retroactive PAS updates (3.5%), and email bounces (6%).

A master toggle (`data_quality.dirty_data_enabled`) allows all injection to be disabled for the clean-data baseline scenario.

### 6.2 Emergent Patterns

These patterns are NOT hardcoded — they emerge from the behavioral simulation parameters:

- Agent probability spikes near conversion (2% awareness → 65% action)
- Digital-to-agent bridge is the most common converting path
- Agent-touched journeys convert at 2.8× the rate of digital-only
- Branded search appears prominently in last-click because it frequently occurs just before an agent call
- TV/Display appear prominently in first-touch but are invisible in last-click
- Mean converting journey length: 6.8 touchpoints; non-converting: 3.2

### 6.3 Validation Targets

After generation, the engine validates against config-defined targets:

| Metric | Target Range | Source |
|--------|-------------|--------|
| Total binds | 4,500–5,500 | funnel.target_binds |
| Agent involvement | 83–91% | funnel.agent_involvement_rate ± 4% |
| Mean journey length (converting) | 5.8–7.8 | journey_dynamics.mean_touchpoints |
| Digital-only bind rate | 9–18% | Derived from agent_involvement_rate |
| Call tracking coverage | 40–50% | identity.call_tracking.coverage_rate |
| Resolution rate | 65–72% | identity.expected_resolution_rate |

If targets are missed, the generator logs warnings and adjusts (but does not re-run) to avoid infinite loops.

---

## 7. Identity Resolution & Journey Assembly

### 7.1 Three-Tier Resolution Architecture

**Tier 1 — Deterministic (Confidence: 95–99%):**
Exact match on stable identifiers.

| Rule | Join Key | Confidence |
|------|----------|------------|
| Email exact | email_hash match across CRM, ESP, PAS, quoting engine | 99% |
| GCLID linkage | GCLID from ga4_sessions → google_ads_clicks → CRM (if form submitted) | 97% |
| Phone exact | phone_hash match across CRM, call tracking, PAS | 98% |
| Policy-CRM lookup | PAS policyholder ↔ CRM contact via policy number or email | 99% |
| Call-session bridge | call_tracking.web_session_id → ga4_sessions.session_id | 95% |

**Tier 2 — Fuzzy Deterministic (Confidence: 80–94%):**
Approximate matching with scoring.

| Rule | Method | Confidence |
|------|--------|------------|
| Name + address fuzzy | Jaro-Winkler similarity > 0.88 on name + address | 85% |
| Name normalization | Standardized name (Robert=Bob, etc.) + exact ZIP | 82% |
| Cookie → form submit | GA4 session with form event → CRM contact created within 5 minutes | 90% |
| Phone partial | Last 7 digits match + same state | 80% |

**Tier 3 — Probabilistic (Confidence: 60–79%):**
Behavioral and contextual matching.

| Rule | Method | Confidence |
|------|--------|------------|
| Agent + day + ZIP | Same agent interaction, same day, same ZIP code | 75% |
| Device fingerprint | Same browser/OS/screen resolution + same ZIP within 48 hours | 65% |
| Recookied detection | New cookie, same device fingerprint, within 7 days of expired cookie | 70% |
| Household inference | Same address, different names, same time window | 60% |

### 7.2 Identity Graph Construction

Each prospect's fragmented records form a connected component in an identity graph. CRM is the anchor node when available. Edges represent match rules with associated confidence tiers.

Graph construction process: Start with all source records as isolated nodes. Apply Tier 1 rules first (highest confidence), creating edges. Apply Tier 2 rules to remaining unlinked nodes. Apply Tier 3 rules last. Extract connected components as unified identities. Assign the CRM record (if present) as the anchor node for each component.

Expected outcome: 180K source records → 22–24K unified identities, 65–72% resolution rate, 41% fully resolved (3+ systems linked).

### 7.3 Journey Assembly Pipeline

For each unified_id, the assembler: collects all touchpoints from all linked source records, normalizes timestamps to UTC, deduplicates (same channel + same timestamp = one touchpoint), applies the 30-day attribution window (touchpoints > 30 days before conversion are excluded), orders by timestamp, annotates each touchpoint with its decision state (inferred from position and channel type), and flags conversion events (quote_start, bind).

Output conforms exactly to the `unified_journeys.parquet` schema defined in Section 5.2.

---

## 8. Attribution Model Arsenal — Mathematical Formulations

### 8.1 Standard Output Contract

All 15 models implement `BaseAttributionModel` and produce identical output structures:

```python
@dataclass
class AttributionResult:
    model_id: str
    conversion_event: str
    channel_credits: pd.DataFrame       # Conforms to channel_credits schema
    journey_credits: pd.DataFrame       # Conforms to journey_credits schema
    diagnostics: pd.DataFrame           # Conforms to model_diagnostics schema
```

This uniform contract enables the comparison framework and budget optimizer to consume any model's output without adaptation. 15 models × 2 conversion events = 30 attribution runs.

### 8.2 Baseline Models (5 Variants)

**0a — Last Digital Click (GA4 Equivalent):**
For each converting journey, 100% credit → last DIGITAL touchpoint. Agent interactions excluded (matching what GA4 would see). This is the primary comparison baseline.

**0b — Last Click (All Channels):**
100% credit → last touchpoint of any type (including agent). Shows the impact of including agent as last touch.

**0c — First Touch:**
100% credit → first touchpoint in the journey. Upper-funnel perspective.

**0d — Linear:**
Equal credit to all touchpoints in the journey. credit_per_touch = 1 / journey_length.

**0e — Time Decay:**
Exponential decay weighting. For touchpoint at time t with conversion at time T:

```
w(t) = exp(-λ × (T - t))
where λ = ln(2) / half_life_days     (half_life_days = 7 from config)

credit(t) = w(t) / Σ_all_touchpoints w(t_i)
```

### 8.3 Shapley Values (3 Variants)

**8.3.1 Standard Shapley (1a)**

Implementation follows Zhao et al. (2018) simplified computation.

The Shapley value for channel i is:

```
φ_i = Σ_{S⊆N\{i}}  [|S|! × (|N|-|S|-1)! / |N|!] × [v(S ∪ {i}) - v(S)]
```

Where N is the set of all channels, S is a coalition (subset), and v(S) is the coalition value function.

Practical computation:

1. Group all journeys by their channel SET (unordered distinct channels). ~800–1,200 unique sets out of 2^13 = 8,192 possible.
2. For each observed channel set S, compute v(S) = conversion_rate(journeys with channel set S). Uses inclusive formulation: "has channel set S" means S is a subset of the journey's channels.
3. Handle sparse/unobserved coalitions: if |observations(S)| < 30, use adjacent imputation (average value from coalitions differing by ±1 channel).
4. Compute Shapley value for each of 13 channels. Total computation: 13 × 8,192 = 106,496 marginal contribution calculations.
5. Normalize so Σ φ_i = total conversions (efficiency axiom).
6. Bootstrap 200 resamples for 95% confidence intervals.

Runtime: ~30 seconds on 25K journeys. Bottleneck is coalition value computation, not the Shapley formula.

**8.3.2 Time-Weighted Shapley (1b) — PRIMARY MODEL:**

Extends standard Shapley with recency weighting within journeys.

After computing channel-level Shapley values, distribute credit within each journey proportionally to time weights:

```
w(t) = exp(-λ × (T - t))
where λ = ln(2) / 7 (7-day half-life)

journey_credit(channel_i, journey_j) = φ_i × [Σ_touchpoints_of_i_in_j w(t)] / [Σ_all_touchpoints_in_j w(t)]
```

A touchpoint 7 days before conversion gets 50% weight; 14 days gets 25%; 21 days gets 12.5%.

**8.3.3 CASV — Causal Adjusted Shapley Values (1c):**

Based on Singal et al. (2022). Uses a 1st-order Markov chain to compute counterfactual coalition values rather than observational ones.

Addresses the fundamental problem with observational Shapley: coalition values are confounded by selection bias (customers who see more channels may already be higher-intent).

Computation:
1. Fit a 1st-order Markov chain on full journey data.
2. For each coalition S, simulate the chain with only channels in S available. Transitions to removed channels redirect proportionally to remaining channels or to NULL.
3. v_CASV(S) = predicted conversion probability under the restricted chain.
4. Apply standard Shapley formula using v_CASV instead of v_observational.
5. Normalize and bootstrap.

### 8.4 Markov Chain (3 Variants)

**Core Implementation (all orders):**

State space: {START} ∪ {13 channels} ∪ {CONVERSION} ∪ {NULL}. For kth-order chains, use compound states (channel sequences of length k).

**Step 1 — Transition Probability Estimation:**

```
P(s' | s) = [count(s → s') + α] / [count(s → *) + α × |states|]
where α = smoothing_alpha = 1/13 ≈ 0.077 (Dirichlet smoothing)
```

**Step 2 — Overall Conversion Probability:**

Formulated as absorbing Markov chain. Partition transition matrix into transient (T) and absorbing (A = {CONVERSION, NULL}) states:

```
P_full = | Q  R |
         | 0  I |

where Q = transient-to-transient transitions
      R = transient-to-absorbing transitions

Fundamental matrix: N = (I - Q)^(-1)
Absorption probabilities: B = N × R
P(conversion) = B[START, CONVERSION]
```

**Step 3 — Removal Effects:**

For each channel c: remove c from the graph, redirect all transitions into c proportionally to remaining channels, recompute P(conversion | c removed).

```
RE(c) = 1 - P(conversion | c removed) / P(conversion | all channels)
```

**Step 4 — Normalize to Credits:**

```
φ(c) = RE(c) / Σ_all RE(c) × Total_Conversions
```

**Order-Specific Notes:**

*1st Order (2a):* 16 states. 16×16 = 256 transition probabilities. Dense, stable estimates. Misses sequence effects.

*2nd Order (2b) — PRIMARY MODEL:* Compound states like (Display, Search). Up to ~182 reachable states. Captures "Display→Search converts differently than Social→Search." Sweet spot of richness vs. data density.

*3rd Order (2c):* Compound triples. ~300–500 observed states (of 2,197 theoretical). Captures full three-step patterns like "Display→Search→Agent." Sparser but reveals the highest-converting pathways.

**Markov-Specific Outputs Beyond Attribution:**
- Channel transition heatmap: P(channel_j | channel_i) for all pairs. Reveals the digital-to-agent bridge.
- Top-10 converting vs. non-converting paths: Directly shows which channel sequences work.

### 8.5 Constrained Convex Optimization (4 Variants)

**Core Formulation:**

```
MAXIMIZE    L(w) - λ‖w‖₂²

where L(w) = Σ_j [y_j × log(σ(α_j)) + (1-y_j) × log(1-σ(α_j))]
      α_j = Σ_i w_i × f_i(j)
      σ(·) = sigmoid function
      y_j = 1 if journey j converts, 0 otherwise

SUBJECT TO:
      w_i ≥ 0                                    (non-negativity)
      w_agent ≥ β × (1/N) × Σ_k w_k             (agent floor, β=1.5)
      w_i ≤ γ × Σ_k w_k  for i ≠ agent          (concentration cap, γ=0.30)
      Σ_i w_i ≤ W_max                            (scale bound)

SOLVER: SciPy minimize(method='SLSQP') or CVXPY
```

**Four Feature Encodings f_i(j):**

*Binary (3a):* f_i(j) = 1 if channel i appeared in journey j, else 0.

*Count (3b):* f_i(j) = count of channel i's appearances in journey j.

*Recency-Weighted (3c) — PRIMARY MODEL:* f_i(j) = Σ_t exp(-λ × (T_j - t)) for each touchpoint of channel i in journey j.

*Position-Encoded (3d):* f_i(j) = [is_first_touch, is_mid_funnel, is_last_touch, total_count]. Four-dimensional feature per channel.

**Attribution Credit Derivation from Weights:**

Learned weights w are model parameters, not directly "credit." Credit per journey:

```
credit_ij = w_i × f_i(j) / Σ_k w_k × f_k(j)     (proportional contribution)
φ_i = Σ_j(converting) credit_ij                     (channel total)
```

**Dual Variable (Shadow Price) Analysis:**

After solving, extract dual variables on each constraint:

| Constraint | Shadow Price = 0 Means | Shadow Price > 0 Means |
|-----------|----------------------|----------------------|
| Agent floor | Optimizer wants agent above floor (floor not binding) — validation | Floor is restricting model — investigate |
| Concentration cap (channel X) | Channel X naturally below cap | Channel X wants more credit than allowed |
| Non-negativity (channel Z) | Channel Z has positive contribution | Channel Z may be anti-correlated with conversion |

### 8.6 Model Registry & Runner

The model runner executes all 15 variants × 2 conversion events = 30 runs in sequence, enforcing the output contract after each. Total runtime budget: ~5 minutes.

| Model | Estimated Runtime |
|-------|-----------------|
| 5 baselines × 2 events | < 5 seconds |
| 3 Shapley × 2 events | ~2 minutes |
| 3 Markov × 2 events | ~30 seconds |
| 4 Constrained Opt × 2 events | ~1 minute |
| Bootstrap CIs (primaries only) | ~90 seconds |

---

## 9. Model Comparison Framework

### 9.1 Pairwise Metrics

For each pair of models (6 pairwise comparisons among 4 models: 3 primaries + last-digital-click baseline), per conversion event:

**Spearman Rank Correlation (ρ):** Do models agree on which channels are most/least important?

```
ρ = Spearman correlation between 13-channel credit rank vectors
```

Expected: ρ > 0.70 among the three advanced models; ρ < 0.50 vs. last-digital-click.

**Jensen-Shannon Divergence (JSD):** Do models agree on credit magnitudes?

```
JSD(P||Q) = ½ × KL(P||M) + ½ × KL(Q||M)    where M = ½(P+Q)
```

Bounded [0, 1]. JSD < 0.05 = very close agreement. JSD > 0.20 = meaningful divergence.

### 9.2 Axiom Compliance Audit

Shapley satisfies all four axioms by construction. For Markov and OR, test approximate compliance:

**Efficiency:** |Σ φ_i - Total Conversions| < ε (should be ~0 after normalization).

**Symmetry:** For channels with statistically similar contribution patterns, |φ_i - φ_j| < δ × max(φ_i, φ_j).

**Null Player:** For any channel with zero marginal impact, is φ_i ≈ 0?

**Additivity:** Split conversions into two subsets, compute separately, verify sum ≈ joint computation.

### 9.3 Convergence / Divergence Map

Per-channel confidence classification across the 3 primary models:

```
For each channel:
  If all 3 agree within 3 percentage points → "HIGH CONFIDENCE"
  If 2 agree, 1 diverges                    → "MODERATE CONFIDENCE"
  If all 3 disagree                         → "LOW — INVESTIGATE"

Consensus direction:
  If average shift from last-click > +3pp   → INCREASE
  If average shift < -3pp                   → DECREASE
  Within ±3pp                               → HOLD
  If confidence = LOW                       → INVESTIGATE
```

### 9.4 Dual-Funnel Comparison

Every model runs on both conversion events. Key insight patterns:

| Pattern | What It Means | Expected Finding |
|---------|--------------|-----------------|
| HIGH quote credit, LOW bind credit | Drives research but doesn't close | Search Nonbrand: ~18% quote, ~9% bind |
| LOW quote credit, HIGH bind credit | Closes deals but doesn't generate interest | Agent: ~20% quote, ~37% bind |
| Similar credit across both | Consistent funnel contribution | Display: ~10% quote, ~8% bind |

---

## 10. Budget Optimization Layer

### 10.1 Response Curve Calibration

Saturating exponential response function per channel:

```
f_i(x_i) = a_i × (1 - exp(-b_i × x_i))
```

**Parameter estimation from attribution results:**

```
a_i = attributed_conversions_i / saturation_factor_i     (ceiling)
b_i = -ln(1 - saturation_factor_i) / current_spend_i     (efficiency rate)
```

Saturation factors range from 0.35 (Agent Support — high ceiling, far from saturation) to 0.80 (Brand Search — very constrained, near ceiling).

### 10.2 MILP Formulation

**Decision Variables:**

```
x_i ∈ ℝ⁺    Spend allocated to channel i (continuous)
z_i ∈ {0,1}  Binary activation variable for channel i
```

**Objective:**

```
MAXIMIZE  Σ_i  f_i(x_i)     [total predicted conversions]
```

**Piecewise Linearization:** Because the objective is nonlinear, approximate each f_i with K=10 linear segments using SOS2 (Special Ordered Sets of Type 2) or incremental formulation. This converts the concave objective into a pure MILP solvable by PuLP/CBC.

**Constraints:**

| # | Name | Formulation | Default |
|---|------|-------------|---------|
| C1 | Total budget | Σ_i x_i = B | B = $5,000,000 |
| C2 | Channel minimum floors | x_i ≥ floor_i × z_i | Varies ($10K–$200K) |
| C3 | Max reallocation | |x_i - x_i_current| ≤ δ × x_i_current | δ = 0.20 |
| C4 | Agent support floor | x_agent ≥ 0.02 × B | 2% of total |
| C5 | Activation linking | x_i ≤ M × z_i | M = B (big-M) |
| C6 | Min active channels | Σ_i z_i ≥ K_min | K_min = 8 |
| C7 | TV minimum | x_tv ≥ 0.10 × B | 10% of total |
| C8 | Search NB cap | x_search_nb ≤ 0.25 × B | 25% of total |

Solution time: under 5 seconds. With 9 optimizable channels and 9 binary variables, there are only 512 possible activation combinations.

### 10.3 Scenario Engine

Seven pre-built scenarios plus custom:

| Scenario | Key Override | Question Answered |
|----------|-------------|-------------------|
| baseline | None | Reference point |
| digital_transformation | agent_involvement → 0.65 | If digital self-service grows? |
| no_call_tracking | call_tracking → 0% | Cost of NOT having call tracking? |
| full_call_tracking | call_tracking → 90% | ROI of full call tracking? |
| cut_upper_funnel | TV/Display probabilities halved | What if awareness spend cut 50%? |
| agent_decline | agent_involvement → 0.72 | Demographic shift scenario? |
| clean_data_baseline | dirty_data_enabled → false | Data quality impact? |

**Call Tracking ROI Calculation (headline scenario):**
- No tracking: agents get 22% credit → optimizer under-funds agents
- Full tracking: agents get 38% credit → optimizer properly funds agents
- Delta: +$280K better allocation + ~340 additional binds/year
- Call tracking cost: ~$50K/year
- ROI: ($280K + 340 × $1,000 premium) / $50K = 9.6×

### 10.4 Shadow Price Extraction

Dual variables from the MILP are translated into business-language insights:

| Constraint | If Shadow Price ≈ 0 | If Shadow Price > 0 |
|-----------|---------------------|---------------------|
| Total budget (C1) | Budget is sufficient | Each additional dollar yields λ₁ additional conversions |
| Reallocation cap (C3) for Search NB | 20% cap not restrictive | Optimizer wants to move MORE from search |
| Agent floor (C4) | Floor not binding — optimizer WANTS to give agents more | Floor is constraining optimization |
| TV minimum (C7) | TV earns its floor on merit | TV floor costs λ₇ forgone conversions |
| Search NB cap (C8) | Cap not restrictive | Data wants search above cap — investigate |

---

## 11. UI/UX Architecture — Plotly Dash Application

### 11.1 Design System

**Color Palette:**
- Primary: Erie brand blue (#00447C)
- Last-click baseline: Gray (#94A3B8) — muted, representing the "old way"
- Model-based results: Erie blue (#00447C) — vivid, representing the "new way"
- Confidence indicators: Green (#16A34A) HIGH / Amber (#D97706) MODERATE / Red (#DC2626) LOW
- Channel colors: Consistent across all charts (defined in theme.py)

**Typography:** Inter font family. Chart titles are INSIGHTS, not labels (e.g., "Agents receive 35% of credit when measured properly — 10× what last-click shows" not "Figure 3: Channel Attribution").

### 11.2 Multi-Page Architecture

Eight pages registered via `dash.register_page()` with URL-based navigation:

| Page | Path | Narrative Role |
|------|------|---------------|
| Executive Summary | / | Act 1 — headline findings |
| Identity Resolution | /identity | Act 2 — why current attribution is broken |
| Three-Model Comparison | /models | Act 3 — convergence builds credibility |
| Channel Deep-Dive | /channels | Act 4 — the agent story |
| Dual-Funnel Analysis | /funnel | Act 3 extension — quote vs. bind |
| Budget Optimization | /budget | Act 5 — what to do about it |
| Scenario Explorer | /scenarios | Act 5 extension — strategic what-if |
| Measurement Roadmap | /roadmap | Act 6 — path from demo to production |
| Technical Appendix | /appendix | Reference — math, diagnostics, data quality |

### 11.3 Data Store Pattern

At app startup, `data_store.py` loads all Parquet files from `data/metrics/` into memory as DataFrames. Each page imports getter functions from `data_store` — never raw file paths or model imports.

```python
# data_store.py pattern
_channel_credits: pd.DataFrame = None

def get_channel_credits() -> pd.DataFrame:
    global _channel_credits
    if _channel_credits is None:
        _channel_credits = pd.read_parquet("data/metrics/channel_credits.parquet")
    return _channel_credits
```

### 11.4 Callback Rules

1. No computation in callbacks — filter, slice, format only.
2. Cross-page state managed via `dcc.Store` (selected conversion event, selected model).
3. `prevent_initial_call=True` for expensive callbacks.
4. Use `Patch()` for partial property updates.
5. Maximum callback chain depth: 3 (to prevent cascading delays).

### 11.5 Reusable Components

All components accept DataFrames as input and return Dash layouts:

| Component | Purpose | Used On Pages |
|-----------|---------|--------------|
| metric_card | KPI card with value, label, delta | 1, 2, 6 |
| insight_callout | Highlighted text insight box | 1, 3, 4, 5 |
| attribution_bars | Grouped horizontal bar chart | 1, 3, 4 |
| convergence_table | Color-coded model agreement table | 3, 6 |
| channel_heatmap | Markov transition probability matrix | 4 |
| waterfall_chart | Credit decomposition steps | 4, 7 |
| budget_sliders | Interactive spend adjustment | 6 |
| model_selector | Toggle pills for model selection | 3, 4, 5 |

---

## 12. Data Pipeline & Execution

### 12.1 Pipeline Execution Order

```
Phase 1:  Install dependencies                                    (~1 min)
Phase 2:  Validate config schema                                  (~1 sec)
Phase 3:  Generate population                                     (~10 sec)
Phase 4:  Run behavioral simulator                                (~2 min)
Phase 5:  Generate system records + fragmentation                 (~30 sec)
Phase 6:  Inject dirty data                                       (~10 sec)
Phase 7:  Run identity resolution (3 tiers)                       (~1 min)
Phase 8:  Assemble unified journeys                               (~30 sec)
Phase 9:  Generate resolution quality report                      (~5 sec)
Phase 10: Run 5 baseline models × 2 events                       (~5 sec)
Phase 11: Run 3 Shapley variants × 2 events + bootstrap          (~3 min)
Phase 12: Run 3 Markov variants × 2 events + bootstrap           (~30 sec)
Phase 13: Run 4 constrained opt variants × 2 events + bootstrap  (~1 min)
Phase 14: Validate all output contracts against schemas           (~5 sec)
Phase 15: Run pairwise comparison (Spearman, JSD)                 (~10 sec)
Phase 16: Generate convergence map                                (~5 sec)
Phase 17: Run dual-funnel analysis                                (~5 sec)
Phase 18: Calibrate response curves                               (~5 sec)
Phase 19: Run MILP optimizer × 3 primary models                  (~15 sec)
Phase 20: Run 7 scenario variations                               (~2 min)
Phase 21: Extract shadow prices                                   (~5 sec)
Phase 22: Aggregate metrics for UI                                (~10 sec)
Phase 23: Validate final metrics against success criteria         (~5 sec)
Phase 24: Launch Dash app                                         (~3 sec)
```

**Total: approximately 12–15 minutes end-to-end.**

### 12.2 Selective Re-Run

`run_attribution_only.py` skips Phases 3–6 (data generation) and starts from Phase 7. Useful when modifying model parameters without regenerating data.

---

## 13. Deployment & Infrastructure

### 13.1 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m src.pipeline.run_full_pipeline    # Pre-compute at build time
EXPOSE 8050
CMD ["gunicorn", "app.app:server", "-b", "0.0.0.0:8050", "-w", "2", "--timeout", "120"]
```

The container ships with pre-computed results. The Dash app serves immediately on startup with no pipeline delay.

### 13.2 Render.com Configuration

```yaml
# render.yaml
services:
  - type: web
    name: erie-mca-demo
    runtime: docker
    plan: starter          # $7/month, always-on
    envVars:
      - key: PYTHON_ENV
        value: production
```

Render Starter plan provides: 512MB RAM, 0.5 CPU, always-on (no cold starts), HTTPS with custom domain support, auto-deploy from GitHub.

### 13.3 Alternative Deployment Strategy

If Docker build times out (pipeline > 10 minutes, Render default 15-minute limit):

1. Generate data locally: `python -m src.pipeline.run_full_pipeline`
2. Commit `data/metrics/` directory to the repository
3. Simplified Dockerfile that only installs Dash dependencies and serves the app
4. Build time drops to < 2 minutes

### 13.4 Memory Management

With 512MB RAM (Render Starter), use lazy loading in data_store.py. Load each Parquet file on first access, not all at startup. Total data footprint should be under 100MB for 25K journeys.

---

## 14. Validation & Quality Assurance

### 14.1 Build-Phase Validations

Each pipeline phase has a validation checkpoint before the next phase begins:

| Phase | Validation | Failure Action |
|-------|-----------|---------------|
| Config load | Pydantic schema validation | Fatal error, stop pipeline |
| Data generation | Target metric ranges (binds, agent rate) | Warning + continue |
| Resolution | Resolution rate within expected range | Warning + continue |
| Attribution (Shapley) | Efficiency axiom: |Σ - total| < 0.01% | Fatal error |
| Attribution (all) | Output schema compliance | Fatal error |
| Comparison | Spearman ρ > 0.70 among primaries | Warning (investigate) |
| Optimization | Budget constraints satisfied | Fatal error |
| Metrics aggregation | All required metric files generated | Fatal error |
| UI binding | `grep -rn "[0-9]\{2,\}\." app/pages/` returns empty | Fatal error |

### 14.2 Property-Based Testing

Shapley axiom tests using hypothesis:

```python
# test_axioms.py
@given(journeys=journey_strategy())
def test_shapley_efficiency(journeys):
    """Credits must sum to total conversions."""
    result = shapley_standard(journeys)
    assert abs(result.channel_credits.total_credit.sum() - total_conversions) < 0.0001

@given(journeys=journey_strategy())
def test_shapley_null_player(journeys):
    """A channel with zero marginal contribution gets zero credit."""
    # Add a dummy channel that appears randomly but doesn't affect conversion
    result = shapley_standard(journeys_with_dummy)
    assert result.channel_credits[result.channel_credits.channel_id == "dummy"].total_credit.iloc[0] < 0.01
```

### 14.3 Data Contract Testing

Every Parquet file is validated against its schema after creation:

```python
def validate_schema(df: pd.DataFrame, schema: dict, name: str):
    for col, expected_type in schema.items():
        assert col in df.columns, f"{name} missing column: {col}"
        assert df[col].dtype == expected_type, f"{name}.{col}: expected {expected_type}, got {df[col].dtype}"
```

---

## 15. Implementation Gotchas & Edge Cases

### 15.1 Shapley Coalition Enumeration

With 13 channels, there are 2^13 = 8,192 possible coalitions per channel. Most coalitions are unobserved in the data. Use the Zhao et al. simplification: group journeys by channel set, compute coalition values only for observed sets, and impute sparse coalitions via adjacent averaging.

### 15.2 Markov 2nd-Order State Space

Theoretical maximum is 13² = 169 compound states plus 13 singletons. In practice, only 40–60 compound states are actually observed with sufficient data. Do not pre-allocate a full 182×182 matrix — use sparse representation and only instantiate observed states.

### 15.3 OR Solver Sensitivity

The constrained optimization solver (SLSQP) is sensitive to initialization. Use uniform w₀ = 1/N as starting point. If solver fails to converge, fall back to CVXPY which handles constraints more robustly.

### 15.4 MILP Piecewise Linearization

Use SOS2 constraints (Special Ordered Sets of Type 2) for the piecewise approximation, not naive big-M formulation. SOS2 is natively supported by CBC and provides tighter LP relaxation.

### 15.5 Dash Callback Cascading

Maximum recommended chain depth is 3 callbacks. Deeper chains cause visible UI delays. If a user interaction requires 4+ sequential updates, batch them into a single callback with multiple outputs.

### 15.6 Numbers Must Tie

The executive summary KPI cards, the convergence table, the budget recommendation table, and the scenario results must all derive from the same underlying Parquet files. Any manual copying of numbers between pages will eventually drift. The metrics/ layer exists specifically to prevent this.

### 15.7 Chart Axis Scales

When comparing models side-by-side (Page 3), all three bar charts MUST use the same x-axis scale. Auto-scaling per chart makes visual comparison impossible. Hardcode the axis range to [0, max_credit + 5%] across all three.

### 15.8 Agent Insight Is the Climax

The demo's persuasive power depends on the agent credit shift (3% → 35%) being presented as a reveal, not a given. Don't show model-based agent credit before Page 1. Build toward it through the executive summary bar chart.

### 15.9 Render Memory Limits

With 512MB RAM on Render Starter, use lazy loading for data and avoid loading all 30 attribution runs simultaneously. The metrics aggregator should pre-filter to only the data each page actually needs.

### 15.10 Docker Build Time

Full pipeline (data generation through optimization) takes 12–15 minutes. Render's default build timeout is 15 minutes. If this is too tight, use the alternative deployment strategy (commit pre-computed data, skip pipeline in Docker build).

---

## Appendix A: Erie Insurance Domain Reference

### A.1 Company Profile

| Attribute | Value |
|-----------|-------|
| Distribution model | 100% independent agent |
| Geographic footprint | 12 states + DC (core: PA, OH) |
| Policies in force | 6M+ |
| Agent network | ~13,000 independent agents |
| AM Best rating | A+ (Superior) |
| Consumer Reports | Highest rated auto insurance |
| Estimated auto marketing budget | ~$5M annually |
| Estimated average auto premium | ~$1,000–$1,200 |
| Brand awareness (PA core market) | ~75% |
| Brand awareness (expansion markets) | ~25% |
| Agent involvement in binds | ~85–90% |
| Technology infrastructure | OneShield PAS, MarkLogic Data Hub (AWS), Power BI |

### A.2 Auto Insurance Funnel (Simulated)

| Stage | Definition | Estimated Rate |
|-------|-----------|---------------|
| Awareness | Knows Erie exists | Base population |
| Consideration | Actively researching Erie | ~35% of aware prospects |
| Quote-start | Initiates a quote | ~50% of considerers |
| Quote-complete | Completes quote form | ~65% of quote-starters |
| Bind | Policy issued | ~45% of quote-completers (overall ~20% of total population) |

### A.3 Channel Taxonomy

| Channel ID | Display Name | Type | Digital/Offline |
|-----------|-------------|------|----------------|
| tv_radio | TV/Radio | Paid | Offline |
| display | Display/Programmatic | Paid | Digital |
| paid_social | Paid Social | Paid | Digital |
| search_nonbrand | Search — Nonbrand | Paid | Digital |
| search_brand | Search — Brand | Paid | Digital |
| seo_organic | SEO/Organic | Earned | Digital |
| email_nurture | Email Nurture | Owned | Digital |
| direct_mail | Direct Mail | Paid | Offline |
| referral_wom | Referral/WOM | Earned | Offline |
| agent_locator | Agent Locator | Owned | Digital |
| agent_interaction | Agent Interaction | Hybrid | Offline |
| erie_direct | Erie.com Direct | Owned | Digital |
| retargeting | Retargeting/Remarketing | Paid | Digital |

---

## Appendix B: Academic References

| Reference | Relevance |
|-----------|-----------|
| Zhao et al. (2018) "A Unified Approach to Quantifying and Evaluating Attribution" | Simplified Shapley computation via coalition grouping |
| Singal et al. (2022) "Shapley Meets Markov: Counterfactual-Adjusted Shapley Values" | CASV variant — causal coalition values |
| Anderl et al. (2016) "Mapping the Customer Journey: A Graph-Based Framework for Online Attribution Modeling" | Markov chain attribution with removal effects |
| Gordon et al. (2023) "Close Enough? A Large-Scale Exploration of Non-Experimental Approaches to Advertising Measurement" | Finding: observational MTA produces 488–948% errors in absolute ad effect estimation. Anchors the demo's humility narrative and triangulation recommendation. |
| Shapley (1953) "A Value for n-Person Games" | Original Shapley value formulation |

---

## Appendix C: Full Configuration Schema

The master configuration file `erie_mca_synth_config.yaml` contains 777 lines across 10 sections. Key high-impact parameters:

**Three parameters with HIGHEST impact on demo insights:**

1. `conversion_funnel.agent_involvement_rate` (default: 0.87, range: 0.70–0.98) — Drives the agent under-credited insight. Higher = bigger shift from last-click.

2. `channel_taxonomy.agent_interaction.state_interaction_matrix` (default: [0.02, 0.08, 0.30, 0.65]) — Controls how agent probability spikes near conversion. The 2%→65% ramp creates the behavioral pattern.

3. `identity_resolution.call_tracking.coverage_rate` (default: 0.45, range: 0.0–1.0) — Determines how visible the digital-to-agent bridge is. Directly affects the call tracking ROI calculation.

**Budget optimizer parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| total_budget | $5,000,000 | Fixed total for optimization |
| current_allocation | 10 channels | Starting budget per channel |
| channel_saturation_factors | 0.35–0.80 | How close each channel is to ceiling |
| channel_minimum_floors | $10K–$200K | Minimum viable spend per channel |
| max_reallocation_pct | 0.20 | Maximum single-cycle budget shift |
| min_active_channels | 8 | Diversification requirement |
| agent_support_floor_pct | 0.02 | Agent channel budget minimum |
| tv_minimum_pct | 0.10 | Brand presence requirement |
| search_nonbrand_cap_pct | 0.25 | Concentration cap |

The full YAML file is maintained as a standalone artifact (`erie_mca_synth_config.yaml`) and serves as the single source of truth for all pipeline parameters.

---

*This document provides the complete technical specification for the MCA Intelligence capability demo. For business context, dashboard navigation guides, and non-technical interpretation of findings, refer to the companion Business User Manual.*