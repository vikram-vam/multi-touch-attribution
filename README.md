# P&C Insurance Multi-Channel Attribution Demo

**Version:** 0.1 | **Status:** Production-Plausible Demo | **Framework:** Plotly Dash

A comprehensive demonstration of advanced multi-touch attribution models specifically calibrated for an Insurance Client's 100% independent agent distribution model. This demo reveals how traditional last-click attribution systematically undervalues agent contributions and upper-funnel digital channels.

## 🎯 Demo Objectives

This is a **capability demo** designed to:

1. **Create cognitive dissonance**: Show the dramatic difference between last-click (agents ~9%) and Shapley attribution (agents ~34%)
2. **Demonstrate channel cooperation**: Digital channels and agents work together in a cooperative journey
3. **Provide actionable guidance**: Interactive budget optimization showing 15-25% CPA improvement potential
4. **Build technical confidence**: Transparent methodology with validation metrics
5. **Anchor in humility**: Reference Gordon et al. (2023) finding of 488-948% errors in observational MTA, positioning triangulation as the solution

## 📁 Repository Structure

```
pnc-mca-demo/
│
├── config/
│   └── config.yaml                 # All parameters: data generation, models, UI
│
├── src/
│   ├── data_generation/
│   │   └── synthetic_data_generator.py  # Client-calibrated journey data
│   │
│   ├── models/
│   │   ├── heuristic_models.py     # Last-Click, Linear, Time-Decay, etc.
│   │   └── advanced_models.py      # Shapley, Markov Chain
│   │
│   ├── pipeline/
│   │   └── attribution_runner.py   # Model orchestration & validation
│   │
│   ├── ui/
│   │   └── pages/                  # Dash app pages
│   │       ├── executive_summary.py
│   │       └── model_comparison.py
│   │
│   └── utils.py                    # Configuration, validation, utilities
│
├── data/
│   ├── synthetic/                  # Generated journey data
│   ├── results/                    # Attribution results
│   └── cache/                      # Computation cache
│
├── app.py                          # Main Dash application
├── main.py                         # Pipeline orchestrator
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- Virtual environment recommended

### Installation

1. **Clone or download the repository**

```bash
cd pnc-mca-demo
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Generate Data & Run Pipeline

**Important**: You must run the data generation pipeline before launching the UI.

```bash
python main.py
```

This will:
- Generate 50,000 synthetic customer journeys with Client-calibrated parameters
- Run 7+ attribution models (Last-Click, Shapley, Markov, etc.)
- Validate results against mathematical axioms
- Generate budget optimization scenarios
- Save all results to `data/results/`

**Expected runtime**: 3-5 minutes

### Launch Interactive Dashboard

```bash
python app.py
```

Then open your browser to: **http://localhost:8050**

## 📊 Demo Navigation Flow

### 1. Executive Summary (Landing Page)
- **Purpose**: The "aha moment" — dramatic last-click vs Shapley comparison
- **Key Metric**: Agents jump from 8.8% → 34.2% attribution
- **Narrative**: Creates cognitive dissonance, establishes the problem

### 2. Model Comparison
- **Purpose**: Show rigor through model spectrum (7 different approaches)
- **Key Insight**: Shapley and Markov converge on agent importance
- **Builds Confidence**: Multiple independent methods agree

### 3. Journey Explorer (Simplified Implementation)
- **Purpose**: Show channel cooperation patterns
- **Key Pattern**: Display/Social → Search → Agent → Conversion

### 4. Budget Optimizer (Simplified Implementation)
- **Purpose**: Actionable recommendations
- **Key Output**: Shift 15% from branded search to display/social → 18% CPA reduction

### 5. Technical Appendix (Simplified Implementation)
- **Purpose**: Address technical skepticism
- **Content**: Model descriptions, validation metrics, limitations

## 🎨 Customization

### Adjust Synthetic Data Parameters

Edit `config/config.yaml`:

```yaml
synthetic_data:
  n_customers: 50000        # Total customer pool
  n_conversions: 2500       # Number who convert (5% rate)
  
  journey:
    mean_touchpoints: 5.2   # Average journey length
    median_days_to_conversion: 18
```

### Modify Attribution Models

Enable/disable models in `config/config.yaml`:

```yaml
attribution_models:
  shapley_simplified:
    enabled: true
    use_macro_groups: true  # Reduces computation from 2^13 to 2^6
    monte_carlo_samples: 10000
  
  markov_chain:
    enabled: true
    order: 1  # First-order Markov
```

### Customize UI Theme

Modify brand colors in `config/config.yaml`:

```yaml
ui:
  brand_colors:
    primary: "#1f4788"      # Client blue
    secondary: "#e31c3d"    # Accent red
```

## 🔬 Technical Details

### Attribution Models Implemented

| Model | Type | Tier | Key Feature |
|-------|------|------|-------------|
| Last-Click | Heuristic | 1 | 100% credit to final touch |
| First-Click | Heuristic | 1 | 100% credit to first touch |
| Linear | Heuristic | 1 | Equal credit across journey |
| Time-Decay | Heuristic | 1 | Exponential decay (7-day half-life) |
| Position-Based | Heuristic | 1 | U-shaped (40-20-40) |
| Shapley Value | Game-theoretic | 2 | Cooperative game theory |
| Markov Chain | Probabilistic | 3 | Removal effect analysis |

### Data Validation

The pipeline automatically validates:

1. **Efficiency Axiom**: Sum of credits = total conversions (tolerance < 1%)
2. **Cross-Model Correlation**: Spearman ρ > 0.70 between Shapley and Markov
3. **Journey Realism**: Length distribution, conversion rates, timing patterns

### Performance Characteristics

- **Data Generation**: ~30 seconds for 50K customers
- **Shapley Computation**: ~60 seconds (with macro-grouping)
- **Markov Chain**: ~20 seconds
- **Total Pipeline**: 3-5 minutes
- **Dashboard Load**: < 3 seconds per page

## 📈 Demo Narrative Guide

### Minute-by-Minute Flow (60-minute demo)

**0-10 mins: The Problem**
- Start on Executive Summary
- Point to key metric cards
- Read the agent insight box aloud
- Show the comparison chart (the climax)
- Let the visual sink in: "220 conversions vs 580 conversions"

**10-25 mins: The Evidence**
- Navigate to Model Comparison
- "We ran 7 different models — they all agree"
- Show heatmap: "Agents consistently undervalued by last-click"
- Reference Gordon et al. finding: "This is why we can't trust a single model"

**25-35 mins: The Mechanics**
- Briefly show journey patterns
- "Digital creates awareness, agents close"
- "These aren't competing — they cooperate"

**35-50 mins: The Action**
- Budget Optimizer
- "Here's what we recommend..."
- Walk through reallocation scenario
- Project CPA improvement

**50-60 mins: Next Steps**
- Technical Appendix (optional, if asked)
- POC proposal: "Give us 12 weeks, we'll validate this with your real data"
- Triangulation roadmap: MTA → MMM → Incrementality

## 🚧 Known Limitations (By Design)

This is a **demo**, not production code:

1. **Synthetic Data**: No real customer PII or business data
2. **Simplified Models**: Deep learning models (LSTM, Transformer) not fully implemented
3. **No Real-Time Data**: Pre-computed results for demo speed
4. **Simplified UI**: Some pages are placeholder implementations
5. **No Authentication**: Open dashboard (not production-ready)

## 🔧 Troubleshooting

### "Module not found" errors
```bash
# Ensure you're in the virtual environment
pip install -r requirements.txt

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Dashboard won't load
```bash
# Ensure data pipeline has run
python main.py

# Check that results exist
ls -la data/results/
```

### Slow performance
```bash
# Reduce data size in config.yaml
synthetic_data:
  n_customers: 10000  # Down from 50000
  n_conversions: 500  # Down from 2500
```

## 📚 References & Literature

This demo implements methods from:

- **Shapley Value**: Zhao et al. (2018) - Simplified computation formula
- **Markov Chains**: Anderl et al. (2016) - Absorbing chain framework  
- **CASV**: Singal et al. (2022, Management Science) - Counterfactual Adjusted Shapley
- **Estimation Error**: Gordon et al. (2023, Marketing Science) - 488-948% error finding

## 🤝 Demo Delivery Best Practices

1. **Pre-load the dashboard** before the meeting (run main.py beforehand)
2. **Start with Executive Summary** — don't bury the insight
3. **Let the visual speak** — pause after showing the comparison chart
4. **Use "our agents" language** — make it about Client, not "the data"
5. **End with a question** — "What would it mean if agents are really 34% of conversions?"
6. **Have Technical Appendix ready** — but only if they ask

## 📞 Support & Next Steps

### For POC Engagement

This demo is designed to lead to a 12-week POC:

**Phase 1**: Data integration (Client's GA4, CRM, agent platform)  
**Phase 2**: Model calibration with real conversion data  
**Phase 3**: Production deployment + training  
**Phase 4**: Triangulation (MMM + incrementality experiments)

### Customization Requests

To adapt this demo for different scenarios:
- Edit `config/config.yaml` for business parameters
- Modify `src/data_generation/synthetic_data_generator.py` for different journey patterns
- Update `src/ui/pages/executive_summary.py` for different narrative focus

## 📄 License

Internal use only - Advanced Analytics Team

---

**Version:** 2.0  
**Last Updated:** February 2026  
**Maintained By:** Advanced Analytics Team  
**Questions:** See Technical Appendix or contact demo maintainer
