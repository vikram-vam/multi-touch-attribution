# MCA Capability Demo — Complete Alignment Document
## Steps 1–8: Business Context through Demo UI & Narrative

**Document Status:** ALL 8 STEPS ALIGNED  
**Date:** February 19, 2026  
**Client:** Erie Insurance — 100% Independent Agent, 12-State Regional P&C Carrier  
**Product:** Personal Auto Insurance  
**Purpose:** Capability demo on realistic synthetic data for multi-channel attribution  

---

## Table of Contents

1. [Step 1: Business Context & Problem Framing](#step-1)
2. [Step 2: Data Universe Mapping](#step-2)
3. [Step 3: The Attribution Problem — Formally Defined](#step-3)
4. [Step 4: Synthetic Data Strategy](#step-4)
5. [Step 5: Identity Resolution & Journey Assembly](#step-5)
6. [Step 6: Model Architecture & Selection](#step-6)
7. [Step 7: Budget Optimization Layer](#step-7)
8. [Step 8: Demo UI & Narrative Design](#step-8)
9. [Master Decision Log](#decisions)
10. [Configuration Reference](#config)
11. [Module Architecture](#modules)

---

<a id="step-1"></a>
## Step 1: Business Context & Problem Framing

### Erie's Core Business Reality

- 100% independent agent distribution — the agent IS the business
- ~85%+ of policy binds happen through an agent (offline)
- 12 states + DC, 6M+ policies, A+ AM Best, highest Consumer Reports auto rating
- ~13,000 independent agents operating as independent businesses

### The Attribution Problem

Erie almost certainly relies on last-click attribution via GA4, which: over-credits paid search (35-45% of last-click credit); under-credits agents by 30-40% because agent interactions don't generate clickstream data; under-credits upper-funnel channels (TV, display, social); and creates a digital-to-agent measurement black hole.

### What the Demo Must Prove

1. Current attribution model is systematically misrepresenting channel performance
2. Agents are under-credited by ~32 percentage points in current models
3. Digital and agents cooperate, not compete — quantified via channel transition analysis
4. Concrete budget reallocation opportunity: 14-17% improvement in cost-per-bind
5. Measurement infrastructure investments (e.g., call tracking) have quantifiable ROI

### Audience

Mixed: CMO/VP-level executives + data/analytics team. VP has OR background and suggested MIP approach. Requires layered architecture — business narrative on surface, technical drill-down available.

### Conversion Events

Dual-funnel: quote-start (all quotes), policy bind (primary), and qualified quotes (quotes leading to binds). The comparison between quote-start and bind attribution is itself a key insight.

---

<a id="step-2"></a>
## Step 2: Data Universe Mapping

### Erie's Confirmed/Inferred Data Systems

| System | Status | Key Data | Identity Key |
|--------|--------|----------|-------------|
| OneShield Policy (PAS) | Confirmed | Binds, premiums, policyholder, writing agent | Policy #, name |
| Quoting Engine | Inferred | Quote-starts/completes, agent vs. online | Quote ID |
| Google Analytics 4 | Inferred | Sessions, page views, events | GA4 Client ID (cookie) |
| Ad Platforms (Google, Meta) | Inferred | Impressions, clicks, cost | GCLID, FBCLID |
| CRM (likely Salesforce) | Inferred | Contact info, interactions, lead source, agent | Email, phone |
| Agent Management System | Inferred | Agent's client records, interactions | Agent internal ID |
| Call Tracking | Likely NOT deployed | Web session → phone → agent bridge | Tracked phone # |
| MarkLogic Data Hub (AWS) | Confirmed | Enterprise data platform | Policy-centric |
| Microsoft Power BI | Confirmed | BI/analytics | N/A |
| Email Service Provider | Inferred | Email sends, opens, clicks | Hashed email |
| Direct Mail Vendor | Inferred | Mailers, matchback data | Name + address |

### Synthetic Data Decision

Simulate fragmented source systems (not pre-unified journeys) to demonstrate the realistic data integration challenge and the value of identity resolution.

### Call Tracking Decision

Model as partially deployed (40-50% coverage). This creates a concrete production roadmap recommendation and demonstrates attribution accuracy improvement from better measurement infrastructure.

---

<a id="step-3"></a>
## Step 3: The Attribution Problem — Formally Defined

### Three Competing Attribution Contenders

**Contender 1: Shapley Values** — "What's FAIR?"
Cooperative game theory. Unique satisfaction of efficiency, symmetry, null player, and additivity axioms. Inclusive coalition value function (Zhao et al. 2018). Three variants: standard, time-weighted, CASV (Singal et al. 2022).

**Contender 2: Markov Chain** — "What's PROBABLE?"
Absorbing Markov chain on directed graph. Attribution via removal effects. Three orders (1st, 2nd, 3rd). Naturally captures channel sequencing and transition probabilities — key for the digital-to-agent bridge story.

**Contender 3: Constrained Convex Optimization** — "What BEST EXPLAINS the data given business knowledge?"
Penalized constrained logistic program. Domain expertise encoded directly via constraints (agent floor, concentration cap, non-negativity). Dual variable (shadow price) analysis turns constraints into conversation starters about business assumptions. Four feature encodings: binary, count, recency-weighted, position-encoded.

### Three-Tier OR Architecture

| Tier | Component | Purpose |
|------|-----------|---------|
| Tier 1 | Constrained Convex Attribution | Competes head-to-head with Shapley and Markov |
| Tier 2 | MILP Budget Optimizer | Translates credit vectors into dollar recommendations |
| Tier 3 | Sensitivity & What-If Engine | Interactive exploration of constraints and scenarios |

### Head-to-Head Comparison Framework

Spearman rank correlation (ρ > 0.70 = convergence), Jensen-Shannon divergence, axiom compliance audit, sensitivity analysis, and convergence/divergence mapping by channel.

---

<a id="step-4"></a>
## Step 4: Synthetic Data Strategy

### Generative Architecture

Four-layer behavioral simulation: Population Generator → Behavioral Simulation Engine → System-Level Record Generation → Source System Extracts. Attribution patterns emerge from realistic customer behavior, not hardcoded.

### Scale

25,000 prospects / ~5,000 binds. Substantial statistical power for Shapley coalition estimation while keeping generation time under 5 minutes.

### Key Parameters (All Configurable)

| Parameter | Default | Range | Impact Level |
|-----------|---------|-------|-------------|
| Agent involvement rate | 87% | 70-98% | HIGHEST — drives the "agent under-credited" insight |
| State interaction matrix (agent row) | Spikes to 65% in Action state | 40-80% | HIGH — how agent probability changes near conversion |
| Call tracking coverage | 45% | 0-100% | HIGH — how visible the digital-to-agent bridge is |
| Digital sophistication mix | H:30%, M:45%, L:25% | Varies | MODERATE — affects channel interaction probabilities |
| Journey length (converters) | Mean 6.8, std 2.5 | 4-10 mean | MODERATE — affects model training richness |
| Dirty data enabled | True | True/False | LOW-MODERATE — affects data quality narrative |

### 13-Channel Taxonomy

TV/Radio, Display, Paid Social, Search Nonbrand, Search Brand, SEO/Organic, Email Nurture, Direct Mail, Referral/WOM, Agent Locator, Agent Interaction, Erie.com Direct, Retargeting/Remarketing.

### Data Quality Injection

Deliberate dirty data: duplicate CRM records (4%), timestamp mismatches (2.5%), missing fields (6%), stale cookies (1.5%), agent code inconsistency (5%), bot traffic (2.5%), retroactive PAS updates (3.5%), email bounces (6%). Master toggle to disable for clean-data baseline comparison.

### Full Parameterized Configuration

Delivered as `erie_mca_synth_config.yaml` — ~450 lines covering all 10 sections with educated defaults, valid ranges, and rationale for every parameter. Includes 7 pre-built scenario presets.

---

<a id="step-5"></a>
## Step 5: Identity Resolution & Journey Assembly

### Resolution Architecture

Deterministic-first, probabilistic-fallback strategy:

| Tier | Match Type | Confidence | Example |
|------|-----------|------------|---------|
| Tier 1 | Exact/near-exact on stable IDs | 95-99% | Email match, GCLID linkage, phone match |
| Tier 2 | Fuzzy deterministic | 80-94% | Name+address matchback, cookie→CRM via form submit |
| Tier 3 | Probabilistic/behavioral | 60-79% | Same agent+day+zip, cross-device fingerprinting |

### Identity Graph

Connected component structure with CRM as anchor node. Each prospect's fragmented records form a graph linked by match rules at various confidence tiers.

### Fragmentation Scenarios

Clean match (25%), cookie churn (20%), digital-only/no agent link (15%), agent-only/no digital (12%), partial bridge (18%), cross-device gap (10%).

### Expected Resolution Metrics

Resolution rate: 65-72%. Fully resolved journeys: ~41%. Agent journeys recovered: ~76%. False positive rate: 2-4%.

### Journey Assembly Pipeline

Collect records → normalize timestamps → deduplicate → apply attribution window → order by timestamp → annotate channels and states → flag conversions → output unified journey table.

### Demo "Insight Cascade"

Five progressive reveals: fragmentation problem → attribution without resolution → attribution with resolution → resolution quality impact → call tracking ROI.

---

<a id="step-6"></a>
## Step 6: Model Architecture & Selection

### Standard Output Contract

Every model variant produces: channel-level credit table (13 channels × credit metrics), journey-level credit allocations, and model diagnostics. Uniform schema enables apples-to-apples comparison and interchangeable consumption by the budget optimizer.

### Full Model Arsenal (15 Variants)

| # | Model | Type | Primary? |
|---|-------|------|----------|
| 0a-0e | Last Digital Click, Last Click, First Touch, Linear, Time Decay | Baselines | 0a is key benchmark |
| 1a | Shapley (Standard) | Contender 1 | |
| 1b | Shapley (Time-Weighted) | Contender 1 | ✓ PRIMARY |
| 1c | CASV | Contender 1 | |
| 2a | Markov 1st Order | Contender 2 | |
| 2b | Markov 2nd Order | Contender 2 | ✓ PRIMARY |
| 2c | Markov 3rd Order | Contender 2 | |
| 3a | Constrained Opt (Binary) | Contender 3 | |
| 3b | Constrained Opt (Count) | Contender 3 | |
| 3c | Constrained Opt (Recency) | Contender 3 | ✓ PRIMARY |
| 3d | Constrained Opt (Position) | Contender 3 | |

### Presentation Approach

3 primary models in the main narrative (1b, 2b, 3c) + last-digital-click baseline. Full 15-variant arsenal available on demand for drill-down.

### Runtime Budget

~5 minutes total for all 30 computations (15 variants × 2 conversion events) on 25K journeys. Full pipeline (generation through optimization) under 15 minutes.

### Implementation Stack

Pandas/NumPy (data), custom implementations (Shapley, Markov), SciPy/CVXPY (constrained optimization), PuLP+CBC (MILP), NetworkX (graph operations), Plotly Dash (UI), all pip-installable, no GPU required.

---

<a id="step-7"></a>
## Step 7: Budget Optimization Layer

### Response Curves

Saturating exponential: f(x) = a × (1 - exp(-b × x)). Calibrated from attribution results + channel-specific saturation factors (parameterizable).

### Current Budget Assumption

$5M total annual marketing budget distributed across channels (TV: $1.2M, Search NB: $750K, Agent Support: $600K, Display: $450K, Direct Mail: $425K, etc.).

### MILP Formulation

Maximize total predicted conversions subject to: total budget constant, channel minimum floors, ≤20% max reallocation per channel, agent support ≥2% of total, binary channel activation variables, minimum 8 active channels, TV minimum 10%, search nonbrand cap 25%.

Solved via piecewise linearization (10 segments) + PuLP/CBC. Solution time under 5 seconds.

### Convergence-Weighted Recommendations

Optimizer runs on all three primary models' credit vectors. Where all three agree on direction: HIGH confidence recommendation. Where two agree: MODERATE confidence. Where all diverge: flag for incrementality testing.

### Shadow Price Analysis

Dual variables on each constraint provide actionable business insights: "the agent floor isn't binding — the optimizer wants to give agents even MORE than the minimum" or "relaxing the reallocation cap by 5pp would yield 120 additional binds."

### Scenario Engine

7 pre-built scenarios (baseline, digital transformation, no/full call tracking, cut upper funnel, agent decline, clean data) plus custom scenario builder. Each overrides specific config parameters and re-runs the optimizer.

### Key Demo Calculation: Call Tracking ROI

No tracking: agents get 22% credit → under-funded. Full tracking: agents get 38% credit → properly funded. Delta: +$280K better allocation + ~340 additional binds/year. Call tracking cost: ~$50K/year. ROI: 9.6×.

---

<a id="step-8"></a>
## Step 8: Demo UI & Narrative Design

### Six-Act Narrative Arc

| Act | Title | Duration | Emotional Beat | Key Reveal |
|-----|-------|----------|---------------|------------|
| 1 | "The Problem You Didn't Know You Had" | 3 min | Discomfort | Last-click vs. reality side-by-side |
| 2 | "Why It's Broken" | 4 min | Understanding | Jennifer Morrison's fragmented journey |
| 3 | "Three Independent Lenses" | 5 min | Credibility | Convergence map — all three models agree |
| 4 | "The Agent Story" | 3 min | Validation | Markov transition heatmap + 2.8× conversion rate |
| 5 | "What To Do About It" | 4 min | Action | Budget recommendation + 14-17% efficiency gain |
| 6 | "The Roadmap" | 2 min | Partnership | Call tracking ROI + measurement maturity path |

Total guided walkthrough: ~21 minutes.

### 8 UI Pages

| Page | Content | Primary Audience |
|------|---------|-----------------|
| 1. Executive Summary | KPI cards + attribution comparison bar chart | CMO |
| 2. Identity Resolution | Fragmentation visual + before/after attribution | Analytics team |
| 3. Three-Model Comparison | Side-by-side credits + convergence/divergence map | VP + Analytics |
| 4. Channel Deep-Dive | Agent spotlight: waterfall, transition heatmap, funnel comparison | All |
| 5. Dual-Funnel Analysis | Quote vs. bind attribution shift visualization | CMO + VP |
| 6. Budget Optimization | Recommendation table + projected impact + budget simulator | CMO |
| 7. Scenario Explorer | What-if engine with pre-built and custom scenarios | VP |
| 8. Measurement Roadmap | Maturity stages + three concrete next steps | All |

Plus Technical Appendix (3 sub-pages): Mathematical Formulations, Model Diagnostics, Data Quality Report.

### Leave-Behind Materials

1. Interactive dashboard (Dash app — deployed URL or Docker container)
2. Executive summary PDF (2 pages, auto-generated)
3. Technical methodology document (10-15 pages)
4. Parameterized configuration file (YAML)

### Visual Design Language

Erie brand blue (#00447C) as primary accent. Consistent channel colors across all charts. Insight-first chart titles ("Agents receive 35% of credit when measured properly" not "Figure 3"). Two interaction modes: guided walkthrough (presenter) and exploratory (leave-behind).

---

<a id="decisions"></a>
## Master Decision Log

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | Conversion events | Dual-funnel: all quotes, binds, qualified quotes | Show the difference as an insight |
| D2 | OR/MIP formulation | Three-tier: Convex Attribution + MILP Budget Optimizer + Sensitivity Engine | Best fit for attribution problem + VP's OR background |
| D3 | Presentation approach | Business-friendly with technical appendix | Mixed audience |
| D4 | Data scale | 25K prospects / ~5K binds | Statistical power + manageable generation time |
| D5 | Data priority | Behavioral realism over system-level fidelity | Demo credibility hinges on realistic journey patterns |
| D6 | Dirty data | Yes — deliberate injection | Shows analytical rigor and real-world readiness |
| D7 | Parameterization | All parameters configurable with educated defaults | User retains control; enables scenario analysis |
| D8 | Agent involvement rate | 87% default (configurable 70-98%) | Core Erie parameter, adjustable |
| D9 | Call tracking assumption | Partially deployed (45% coverage) | Realistic gap; creates production recommendation |
| D10 | Model arsenal presentation | 3 primaries upfront, full 15 on demand | Layered for mixed audience |
| D11 | Source system simulation | Fragmented extracts, not pre-unified | Shows messy integration problem |
| D12 | Identity resolution tiers | Three-tier (deterministic, fuzzy, probabilistic) | Progressive quality with toggleable confidence |

---

<a id="config"></a>
## Configuration Reference

Master configuration: `erie_mca_synth_config.yaml`

10 sections: Simulation Control, Population, Conversion Funnel, Channel Taxonomy, State Interaction Matrix, Journey Dynamics, Identity Resolution, Data Quality, Attribution Model Parameters, Scenario Presets.

Three highest-impact parameters:
1. `funnel.agent_involvement_rate` (default: 0.87)
2. `state_interaction_matrix.agent_interaction` (default: [0.02, 0.08, 0.30, 0.65])
3. `identity.call_tracking.coverage_rate` (default: 0.45)

---

<a id="modules"></a>
## Module Architecture

```
erie_mca_demo/
├── config/
│   └── erie_mca_synth_config.yaml
├── generators/
│   ├── population_generator.py
│   ├── behavioral_simulator.py
│   ├── system_record_generator.py
│   └── dirty_data_injector.py
├── resolution/
│   ├── identity_resolver.py
│   ├── journey_assembler.py
│   └── resolution_reporter.py
├── attribution/
│   ├── baselines.py
│   ├── shapley.py
│   ├── markov.py
│   ├── constrained_optimization.py
│   └── output_contract.py
├── comparison/
│   ├── head_to_head.py
│   ├── convergence_map.py
│   └── dual_funnel_analysis.py
├── optimization/
│   ├── budget_optimizer.py
│   ├── scenario_engine.py
│   └── sensitivity_analyzer.py
├── visualization/
│   ├── app.py (Dash multi-page app)
│   ├── pages/ (8 pages + appendix)
│   └── components/ (reusable chart components)
├── pipeline/
│   ├── run_full_pipeline.py
│   └── run_attribution_only.py
├── export/
│   ├── summary_pdf.py
│   └── methodology_doc.py
└── output/
    ├── source_systems/
    ├── resolved_journeys/
    ├── attribution_results/
    ├── comparison_results/
    ├── optimization_results/
    └── validation_report.html
```

---

*This document represents the complete aligned specification for the Erie MCA Capability Demo. All 8 steps have been reviewed and confirmed. The next phase is implementation execution using this document as the single-source specification.*
