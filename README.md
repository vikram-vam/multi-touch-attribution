# Erie Insurance — Multi-Channel Attribution Intelligence

A production-grade Multi-Channel Attribution (MCA) platform demonstrating advanced attribution modeling for Erie Insurance's marketing ecosystem. Built to showcase the power of data-driven attribution over traditional last-touch models.

---

## 🎯 What This Does

This platform answers a critical question for Erie Insurance: **"Where should we invest our marketing dollars for maximum ROI?"**

Current attribution (GA4 last-touch) credits **41% of conversions** to independent agents simply because they're the last touchpoint. Our Shapley Value analysis reveals agents facilitate conversion but the **true credit distribution is significantly different** — meaning millions in marketing spend may be misallocated.

### Key Insights Demonstrated

| Metric | Current (Last-Touch) | MCA (Shapley) | Impact |
|--------|---------------------|---------------|--------|
| Agent Credit | 41% | 26% | Agent over-credited by 15pp |
| Display Credit | 3% | 11% | Display under-credited 3.7× |
| Paid Social Credit | 5% | 9% | Social under-credited 1.8× |
| Projected Uplift | — | +18% | ~2,700 additional annual binds |

---

## 🏗️ Architecture

```
mca-erie/
├── config/                    # YAML configuration (channels, funnel, models)
├── data/
│   ├── raw/                   # Generated synthetic data (Parquet)
│   ├── processed/             # Attribution results
│   └── reference/             # Erie state map, industry benchmarks
├── src/
│   ├── data_generation/       # Synthetic data engine (9 modules)
│   ├── pipeline/              # Data pipeline (sessionizer, qualifier, classifier)
│   ├── models/                # 17 attribution models across 6 tiers
│   ├── optimization/          # Budget optimizer with 6 scenarios
│   ├── metrics/               # Metrics computation layer
│   ├── validation/            # Data quality & model validation
│   └── utils/                 # Config loader, I/O, formatters, caching
├── app/                       # Plotly Dash UI (8 pages)
│   ├── pages/                 # Executive, Attribution, Journeys, Budget, etc.
│   ├── components/            # Reusable UI components
│   ├── assets/                # CSS theme
│   ├── theme.py               # Color palette & Plotly template
│   ├── data_store.py          # Centralized data loading
│   └── app.py                 # Application entry point
├── scripts/                   # CLI tools
└── tests/                     # Test suite
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python scripts/generate_data.py --seed 42
```

This creates 150,000 customer journeys calibrated to Erie's market:
- 13-channel marketing ecosystem
- ~14.3% conversion rate
- 42% agent last-touch rate
- 12-state geographic distribution

### 3. Run Attribution Models

```bash
python scripts/run_attribution.py --models all
```

Runs all 17 attribution models and saves results to `data/processed/`.

### 4. Launch Dashboard

```bash
python app/app.py
```

Open [http://localhost:8050](http://localhost:8050) to explore the interactive demo.

### One-line Setup (Make)

```bash
make install && make generate-data && make precompute && make run
```

---

## 📊 Attribution Model Tiers

| Tier | Models | Methodology |
|------|--------|-------------|
| 1 — Rule-Based | First-Touch, Last-Touch, Linear, Time-Decay, Position-Based | Heuristic allocation |
| 2 — Game-Theoretic | Shapley Value, CASV | Cooperative game theory (Shapley axioms) |
| 3 — Probabilistic | Markov Chain (1st–3rd order) | Transition probabilities + removal effect |
| 4 — Statistical | Logistic Regression, Survival/Hazard | Regression coefficients / hazard ratios |
| 5 — Deep Learning | LSTM, DARNN, Transformer, CausalMTA | Attention-based sequence models |
| 6 — Meta-Model | Weighted Ensemble | Shapley (45%) + Markov (30%) + Logistic (25%) |

---

## 🖥️ Dashboard Pages

1. **Executive Summary** — KPIs, agent credit shift, key recommendations
2. **Attribution Comparison** — Side-by-side model comparison, heatmap, correlation matrix
3. **Journey Paths** — Sankey diagram, top converting paths, path length distribution
4. **Budget Optimizer** — What-if scenarios, ROI analysis, spend waterfall
5. **Identity Resolution** — Fragment-to-profile resolution, match quality
6. **Channel Deep-Dive** — Per-channel drill-down with cross-model comparison
7. **Validation** — Model quality checks, target vs actual metrics
8. **Technical Appendix** — Methodology, data schema, academic references

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📋 Configuration

All parameters are externalized to `config/`:

- `channels.yaml` — 13-channel taxonomy with Erie-specific attributes
- `funnel.yaml` — Awareness → Quote → Bind funnel stages
- `synthetic_data.yaml` — Data generation parameters
- `model_params.yaml` — Model hyperparameters
- `ui_theme.yaml` — Dashboard theme colors and fonts
- `demo_narrative.yaml` — Per-page talking points

---

## 🏢 Erie Insurance Context

- **Founded:** 1925 | **Operating States:** 12 + D.C.
- **Agent Network:** ~14,000 independent agents
- **DWP Rank:** Top 15 U.S. P&C insurers
- **Primary Line:** Personal auto insurance

This demo uses synthetic data calibrated to Erie's publicly available financials and channel distribution patterns. No real customer data is used.

---

## 📚 References

- Shapley, L.S. (1953). *A Value for n-Person Games*
- Anderl, E. et al. (2016). *Mapping the Customer Journey: Graph-Based Online Attribution*
- Ren, K. et al. (2018). *Learning MTA with Dual-attention Mechanisms* (DARNN)
- Du, R. et al. (2019). *CausalMTA: Eliminating Biases in Multi-touch Attribution*
