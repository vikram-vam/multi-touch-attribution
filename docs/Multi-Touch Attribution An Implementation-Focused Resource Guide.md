# Multi-touch attribution: an implementation-focused resource guide

**Multi-touch attribution (MTA) sits at the intersection of cooperative game theory, causal inference, and messy real-world data engineering — and the field is shifting fast.** This guide provides the mathematical foundations, state-of-the-art methods, practical implementation patterns, and data quality realities needed to build an MTA demo targeting a regional P&C carrier with an independent agent distribution model. The core tension to internalize: **observational-only MTA models produce 488–948% errors in estimated ad effects** (Gordon et al. 2023, Marketing Science), which is why the industry is converging on triangulation — MTA for tactical optimization, media mix modeling (MMM) for strategic allocation, and incrementality experiments for causal ground truth. For Erie Insurance's 100% independent agent model across 12 states, the offline-to-online attribution gap (digital ad → agent contact → policy binding) is the hardest unsolved problem you'll face.

---

## 1. Shapley value attribution: the mathematical core

> **★ START HERE:** Zhao et al. (2018), "Shapley Value Methods for Attribution Modeling in Online Advertising" — arXiv: https://arxiv.org/pdf/1804.05327. This is the single most implementation-relevant paper. It gives you the simplified computation formula, the ordered Shapley extension, and runtime benchmarks.

### Exact vs. approximate computation

The exact Shapley value requires evaluating **2^N coalitions** for N channels. With N=20 channels, that's 1,048,576 coalitions taking ~17 hours (Zhao et al. 2018). For N=10–15 channels, exact computation is feasible; above that, approximation is mandatory.

The three main approximation families:

- **Monte Carlo permutation sampling** (Štrumbelj & Kononenko 2014): Sample M random permutations, average marginal contributions. No good rule of thumb for optimal M, but convergence is empirically fast. The `shapley` Python library by Rozemberczki implements this along with Expected Marginal Contribution Approximation (Fatima et al.) and Multilinear Extension (Owen).
- **Kernel-weighted regression** (Lundberg & Lee 2017, KernelSHAP): Formulates Shapley estimation as a weighted least-squares problem. The `shap` library (20K+ GitHub stars) is the dominant implementation.
- **Regression-adjusted Monte Carlo** (Witter et al. 2025, arXiv:2506.11849): State-of-the-art combining sampling with linear regression, achieving **6.5× lower error** than Permutation SHAP and 3.8× lower than KernelSHAP. Allows XGBoost as the regression model while maintaining unbiased estimates.

A comprehensive survey of all SV computation methods appears in Lin et al. (2025), "A Comprehensive Study of Shapley Value in Data Analytics" (VLDB 2025): https://www.vldb.org/pvldb/vol18/p3077-xie.pdf.

### The coalition value function v(S) — inclusive vs. exclusive

This is the most important modeling decision you'll make. The **inclusive formulation** defines v(S) = conversion probability given that all channels in S appeared in the journey (but other channels may have also appeared). The **exclusive formulation** defines v(S) = conversion probability given that *only* channels in S appeared. The inclusive formulation is standard in practice because **the exclusive formulation creates severe data sparsity** — many coalitions may never be observed in isolation. Zhao et al. (2018) use inclusive; Google's Data-Driven Attribution uses inclusive. Dalessandro et al. (2012), "Causally Motivated Attribution for Online Advertising" (ADKDD at KDD 2012, https://dl.acm.org/doi/10.1145/2351356.2351363, 166 citations), is the foundational paper connecting Shapley to causal attribution and defining v(S) via counterfactual probabilities.

For **sparse coalition imputation** — when many coalitions are never observed — the practical approaches are: (1) use the inclusive formulation to maximize data per coalition, (2) treat unobserved coalitions as v(S)=0 (the Zhao et al. approach, which only sums over observed coalitions), (3) group fine-grained touchpoints into broader channel categories to reduce the coalition space, (4) train an ML model to predict v(S) from features of S. The `shapiq` library (PyPI) implements GaussianImputer and GaussianCopulaImputer for handling missing coalitions.

### Time-weighted and ordered extensions

Standard Shapley treats channels as unordered sets. To incorporate position/recency while preserving axioms:

- **Ordered Shapley** (Zhao et al. 2018): Defines R_i(S∪{x_j}) as the contribution from users where x_j was the i-th touchpoint. Their results show touchpoint 1 gets 91.59% credit — a strong "introducer" effect.
- **Counterfactual Adjusted Shapley Value (CASV)** (Singal et al. 2022, Management Science 68(10), originally WWW 2019, https://pubsonline.informs.org/doi/10.1287/mnsc.2021.4263): Uses a Markov chain model to inherently capture stage effects while preserving axiomatic properties. This is the most theoretically rigorous extension.
- **Sum Game extensions** (Molina, Tejada & Weiss 2022, Annals of Operations Research, https://link.springer.com/article/10.1007/s10479-022-04944-5): Extends to handle repetition count and position of channels with axiomatic characterizations.

### SHAP connection

SHAP (Lundberg & Lee, NeurIPS 2017) uses the identical Shapley formula but in a different "game": players = model features, v(S) = E[f(x) | features in S known]. In MTA, players = channels, v(S) = conversion probability when coalition S is present. **They are mathematically equivalent in structure but differ in practice**: MTA deals with observed coalitions from real journey data where v(S) may be undefined for unobserved coalitions, while SHAP can always evaluate v(S) via the model. Christoph Molnar's "Interpretable Machine Learning" book (Ch. 17–18, https://christophm.github.io/interpretable-ml-book/shapley.html) is the best practitioner explanation of both Shapley and SHAP.

### Key libraries for Shapley attribution

| Library | Language | URL | Notes |
|---------|----------|-----|-------|
| `marketing-attribution-models` | Python | https://pypi.org/project/marketing-attribution-models/ | Full MTA suite: Shapley, Markov, heuristics. Ordered Shapley support. **Best quick-start.** |
| `shapley-attribution-model` | Python | https://github.com/ianchute/shapley-attribution-model | Direct Zhao et al. implementation |
| `mta` | Python | https://github.com/eeghor/mta | Shapley + Markov + additive hazard + logistic regression |
| `shapiq` | Python | https://pypi.org/project/shapiq/ | SOTA: 10+ approximators, interaction indices, imputers |
| `shap` | Python | https://github.com/shap/shap | Train conversion model → SHAP for attribution. 20K+ stars |
| `ChannelAttribution` | R/Python | https://cran.r-project.org/package=ChannelAttribution | Industry standard. Pro version adds Shapley |

---

## 2. Markov chain attribution: graph-based removal effects

> **★ START HERE:** Anderl, Becker, von Wangenheim & Schumann (2014/2016), "Mapping the Customer Journey: A Graph-Based Framework for Online Attribution Modeling" — SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2343077 (published in International Journal of Research in Marketing 33(3), 2016). Then read the ChannelAttribution white paper: https://channelattribution.io/pdf/ChannelAttributionWhitePaper.pdf.

### The absorbing Markov chain framework

Customer journeys are modeled as walks on a directed graph. States = {START, Channel_1, ..., Channel_N, CONVERSION, NULL}. CONVERSION and NULL are absorbing states (once entered, never left). The transition matrix in canonical form is P = [[Q, R], [0, I]] where Q is the transient-to-transient submatrix and R is transient-to-absorbing. The **fundamental matrix** N = (I−Q)^{−1} gives expected visits to each transient state, and the **absorption probability matrix** B = NR gives the probability of ending in each absorbing state from each starting state. The key quantity is B_{START, CONVERSION} = overall conversion probability.

The **removal effect** for channel c: redirect all transitions into c toward NULL, recompute B, measure the drop in P(START → CONVERSION). Normalize across all channels so credits sum to total conversions. Grinstead & Snell's textbook (https://stats.libretexts.org/Bookshelves/Probability_Theory/Introductory_Probability_(Grinstead_and_Snell)/11:_Markov_Chains/11.02:_Absorbing_Markov_Chains) provides the standard undergraduate-level treatment of absorbing chains.

Two computation approaches exist: **exact matrix absorption** via (I−Q)^{−1} inversion (clean, deterministic, but scales as O(N³)), and **simulation-based Monte Carlo** (generate millions of random paths, measure conversion rate drops). The ChannelAttribution package uses the simulation approach for flexibility with conversion values and higher-order models.

### Higher-order chains and state space explosion

A k-th order Markov chain conditions on the previous k states, creating compound states. For N channels, the state space is **N^k**: a 2nd-order chain with 10 channels has 100 compound states; 3rd-order has 1,000. Anderl et al. found **3rd order often most proficient** in practice. Kakalejčík et al. (2018) used the Global Dependency Level (GDL) estimator and found 4th order optimal for Slovak e-commerce data.

**Variable-order Markov models (VMMs)** let the conditioning context length vary based on the specific realization — also called context trees or probabilistic suffix trees. They achieve "a great reduction in the number of model parameters" compared to fixed higher-order chains. The **Mixture Transition Distribution (MTD) model** (Berchtold & Raftery 2002, Statistical Science 17(3), https://projecteuclid.org/journals/statistical-science/volume-17/issue-3/The-Mixture-Transition-Distribution-Model-for-High-Order-Markov-Chains/10.1214/ss/1042727943.pdf) uses additive mixing of first-order matrices instead of N^k parameters — highly relevant for avoiding state space explosion.

### Transition probability estimation

Standard MLE: p(i→j) = count(i→j) / count(transitions from i). For zero-count transitions, use **Dirichlet priors** (conjugate to multinomial likelihood): posterior is Dirichlet with updated concentration parameters. Small symmetric α (e.g., α=1/N) yields non-zero posterior estimates for unobserved transitions. Heiner et al. (2022, JCGS, https://www.tandfonline.com/doi/full/10.1080/10618600.2021.1979565) develop **sparse Dirichlet mixture priors** specifically for high-order Markov chains — directly addressing the zero-count problem. Stan's user guide (Section 2.6, https://mc-stan.org/docs/2_18/stan-users-guide/hmms-section.html) shows explicit code for Dirichlet priors on transition probabilities.

### When removal effect diverges from Shapley

Singal et al. (2022, Management Science) is the definitive comparison. **Removal effect overweights high-traffic channels** — channels appearing in more journeys get inflated credit because removing them disrupts more paths. Additional divergence points: Markov naturally captures sequence ordering while standard Shapley is sequence-agnostic; Markov is less sensitive to noisy/sparse data; and removal effect normalization "unfairly shifts attributed value from single-channel paths and shorter paths to longer conversion paths" (Adequate Digital). For choosing between them: Markov scales better computationally, Shapley has stronger axiomatic guarantees and is used by Google's DDA.

### Key Markov implementation resources

Sergii Bryl's AnalyzeCore tutorial (https://www.analyzecore.com/2016/08/03/attribution-model-r-part-1/) is the most widely-referenced practitioner walkthrough. The Adequate Digital complete guide (https://adequate.digital/en/markov-chain-attribution-modeling-complete-guide/) covers normalization issues and practical order selection. The Databricks Solution Accelerator (https://www.databricks.com/blog/2021/08/23/solution-accelerator-multi-touch-attribution.html) provides production-grade PySpark implementation.

---

## 3. State-of-the-art approaches deployed in industry

> **★ START HERE:** The Statsig technical survey by Yuzheng Sun (https://www.statsig.com/blog/marketing-attribution-models-tech-survey) is the single best overview covering all model families with mathematical formulations. Then read Gordon et al. (2023) on why observational MTA fails.

### Deep learning and attention models

The field has moved from heuristics → game theory → deep learning → causal deep learning. Key papers with code:

**DARNN** (Ren et al., CIKM 2018, https://arxiv.org/abs/1808.03737, code: https://github.com/rk2900/deep-conv-attr): Dual-attention RNN that learns attribution via attention mechanisms from conversion estimation. Introduces a budget-allocation-based evaluation scheme that became the standard benchmark. TensorFlow implementation with Criteo dataset.

**DNAMTA** (Li et al., AdKDD 2018, https://arxiv.org/abs/1809.02230): LSTM+Attention model deployed at Adobe. Incorporates user demographics as control variables to reduce estimation bias. Attention weights serve as interpretable attribution scores.

**CausalMTA** (KDD 2022, https://dl.acm.org/doi/10.1145/3534678.3539108): Alibaba's approach decomposing confounding bias into static user attributes and dynamic features. Uses journey reweighting and causal conversion prediction. **Best causal deep learning approach with theoretical guarantees.**

**Transformer-based attribution** (Lu & Kannan, Journal of Marketing Research 2025, https://journals.sagepub.com/doi/10.1177/00222437251347268): First major transformer framework for customer journey analysis. Heterogeneous mixture multi-head self-attention captures individual-level variation. Outperforms HMMs, point process models, and LSTMs.

**CAMTA** (Kumar et al., ICDM 2020 Workshop, https://arxiv.org/abs/2012.11403): Combines counterfactual recurrent networks with attention for causal attribution. Uses domain adversarial training for treatment-invariant representations.

### Survival analysis for time-to-conversion

Survival models naturally handle the time dimension and censoring (users who haven't converted yet). Google's own research (Shender et al. 2020, https://arxiv.org/abs/2009.08432) frames MTA as a time-to-event problem for production use. The pioneering paper is Zhang, Wei & Ren (ICDM 2014, http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/attr-survival.pdf), using Weibull distributions and hazard rates. Ji & Wang (AAAI 2017) extend this with additive ad effects. The `mta` Python library (https://github.com/eeghor/mta) implements additive hazard attribution alongside Markov and Shapley.

### Big tech's attribution evolution

**Google DDA** trains on all converting and non-converting paths, feeds every variable combination into a probabilistic model, and uses Shapley values to assign credit. It likely incorporates survival analysis (per Google's own 2020 paper). Google deprecated all heuristic models in September 2023, making DDA the default. It requires minimum **300 conversions/30 days and 3,000 ad interactions** — and remains a black box confined to the Google ecosystem.

**Google Meridian** (https://github.com/google/meridian): Open-source Bayesian hierarchical geo-level MMM. Key features include reach/frequency modeling, calibration with incrementality experiments, and Google Query Volume as a control variable. GA in 2025.

**Meta Robyn** (https://github.com/facebookexperimental/Robyn): Open-source MMM using Ridge regression, Nevergrad optimization, and Prophet decomposition. Critically, it **calibrates models with ground truth from incrementality experiments** (Conversion Lift, GeoLift). 1,200+ GitHub stars.

**Amazon's 2025 MTA system** (https://arxiv.org/abs/2508.08209): The most comprehensive public description of how a major platform combines RCTs with ML-based attribution. Uses an ensemble of causal ML models calibrated against randomized experiments.

### Incrementality as ground truth

Incrementality experiments are the industry's accepted "gold standard" for causal measurement. **Ghost ads** (Johnson, Lewis & Nubbemeyer, JMR 2017, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2620078) identify control group counterparts without serving ads, reducing variance by **5.9–16.4×** versus PSA/ITT approaches. **Geo-experiments** compare treatment vs. control regions using synthetic control methods — Meta's GeoLift library (https://github.com/facebookincubator/GeoLift) implements this. Google's CausalImpact (https://github.com/google/CausalImpact) uses Bayesian structural time series for quasi-experimental causal inference.

The consensus is **triangulation**: MTA for real-time tactical optimization, MMM for strategic budget allocation, incrementality experiments for causal validation. No single method alone is sufficient.

---

## 4. Data quality and capture challenges will define your project

> **★ START HERE:** Gordon et al. (2023), "Close Enough? A Large-Scale Exploration of Non-Experimental Approaches to Advertising Measurement" (https://arxiv.org/abs/2201.07055). This paper examining ~2,000 Meta campaigns proves observational attribution fails catastrophically, establishing why every challenge below matters.

### The cookie landscape in 2026

Google **reversed course in July 2024** and abandoned third-party cookie deprecation in Chrome. By October 2025, Google retired most Privacy Sandbox APIs — including the Attribution Reporting API — citing "low levels of adoption" (https://privacysandbox.google.com/blog/update-on-plans-for-privacy-sandbox-technologies). Third-party cookies survive in Chrome's default settings. However, **Safari and Firefox already block third-party cookies by default**, meaning roughly 50% of the web is already cookieless. The practical implication: your MTA system must function in a mixed environment where half of journeys have cross-site tracking and half don't.

CPM fell **33%** when advertisers used Privacy Sandbox (Index Exchange testing), and publishers could lose **60% of revenue** without cookies (Criteo study) — the economic incentives to maintain cookies are powerful, which explains Google's reversal.

### iOS ATT devastated cross-device tracking

Apple's App Tracking Transparency opt-in rates sit at roughly **35% industry average** (Adjust Q2 2025), but the "double opt-in" math is brutal: both the advertiser and publisher apps need consent. With 10% opt-in on each side, only **1% of IDFAs are actually available** (Singular 2024). By Q3 2021, 80% of iOS users on social media platforms had opted out, causing advertisers to lose ~40% of Facebook/YouTube/Twitter impression volume. Meta cited ATT as a primary factor in slowing revenue growth in 2022.

### Apple Mail Privacy Protection killed email attribution

Apple Mail accounts for **49.3% of all email opens** (Litmus, January 2025). MPP pre-loads tracking pixels on Apple's proxy servers, generating artificial "opens" for every email. Omeda's analysis across 80,000+ deployments and ~2 billion emails found open rates nearly **doubled** post-MPP. Email open rate is effectively dead as an attributable signal for Apple Mail users. Click rates remain unaffected and are now the primary email engagement metric. Litmus's analysis (https://www.litmus.com/blog/apple-mail-privacy-protection-for-marketers) details the cascading impacts: A/B subject line testing broken, send time optimization inaccurate, engagement-based segmentation unreliable.

### Server-side tracking as partial mitigation

Facebook CAPI, Google Enhanced Conversions, and server-side GTM bypass browser-side restrictions by sending conversion data directly from your server. **Pixel-only tracking has dropped to roughly 40% attribution accuracy** (wetracked.io analysis); CAPI + pixel dual setup recovers much of the lost signal. Key implementation resources include Stape's comprehensive CAPI guide (https://stape.io/blog/conversions-api-explained) covering Meta, Google, Snapchat, and LinkedIn implementations.

### Identity resolution failure modes

Deterministic matching (login-based) is accurate but has limited coverage — most visitors don't log in. Probabilistic matching expands coverage but introduces **false positive rates** that compound across identity graph operations. LiveRamp's graph reaches 200M+ unique users and 600M+ matched devices, but identity graphs go stale as users change devices, clear cookies, and move. Only **18% of marketers feel confident in their attribution data** due to identity fragmentation (MetaCTO). The practical guide from RudderStack (https://www.rudderstack.com/blog/deterministic-vs-probabilistic/) recommends a tiered approach: deterministic first, probabilistic to broaden coverage.

### Walled gardens block cross-platform attribution

Google, Meta, and Amazon collectively control **70%+ of digital ad spending** (eMarketer) and don't share user-level data across platforms. Data clean rooms (Google Ads Data Hub, Meta Advanced Analytics, Amazon Marketing Cloud) offer limited access with strict privacy thresholds. Practical limitations documented by Tredence (https://www.tredence.com/blog/a-practical-guide-for-building-marketing-measurement-in-clean-rooms): data extraction minimums (10 for clicks/conversions, 50 for impressions in ADH), no cross-platform interoperability, no real-time access. As AdExchanger bluntly states: "Walled garden clean rooms are incredibly limited today. Forget trying to do advanced attribution or any kind of one-to-one targeting."

### Biases you cannot ignore

**Selection bias**: You only observe users who interact with ads — the non-interacting majority is invisible, creating systematically skewed data. **Survivorship bias**: Analyzing only converting users while ignoring the vast majority who didn't convert produces fundamentally flawed conclusions about channel effectiveness. **Consent-driven bias**: Users declining tracking correlate with specific demographics and privacy consciousness, creating non-random data gaps (SecurePrivacy analysis, https://secureprivacy.ai/blog/consent-attribution-tracking-digital-marketing). **Bot traffic**: **51%+ of all internet traffic is automated** (2025 Imperva Bad Bot Report), with click fraud rates averaging **15–25%** across campaigns. Global ad fraud losses reached **$84 billion** in 2023 (Juniper Research), projected to hit $172 billion by 2028.

### Privacy regulation compounds everything

CPRA specifically prohibits service providers from "combining personal information received from one business with data from another" — but cross-publisher attribution fundamentally requires combining data (Gary Kibel, AdExchanger, https://www.adexchanger.com/data-driven-thinking/measurement-is-at-stake-when-cpra-takes-effect/). CPRA also requires processing opt-out preference signals (GPC), which restrict data by default. The patchwork of state privacy laws (Virginia VCDPA, Colorado CPA, Connecticut, etc.) creates jurisdiction-specific data availability rules. For a 12-state carrier like Erie, this means navigating up to 12 different consent regimes.

---

## 5. Implementation patterns that make or break MTA systems

> **★ START HERE:** The Cometly SQL guide (https://www.cometly.com/post/multi-touch-attribution-sql) for journey construction, then the Adobe Analytics attribution components documentation (https://experienceleague.adobe.com/en/docs/analytics/analyze/analysis-workspace/attribution/models) for understanding lookback window sensitivity with worked examples.

### Journey assembly from raw events

Sessionization is the first technical step. **Time-based sessionization** (20–30 minute inactivity timeout) is the standard approach — GA4 uses 30 minutes. AWS's real-time clickstream blog (https://aws.amazon.com/blogs/big-data/create-real-time-clickstream-sessions-and-run-analytics-with-amazon-kinesis-data-analytics-aws-glue-and-amazon-athena/) demonstrates SQL window functions for session boundaries. Key decisions: how to handle repeat visits to the same channel (deduplicate or count separately), ordering touchpoints when timestamps share the same granularity, and whether to sessionize at the device level or user level. Intuit's clickstream case study (LinkedIn article by Irina Pragin) shows enterprise-scale solutions for cross-product sessionization, bot filtering, and event schema standardization.

### Lookback windows change everything

Adobe Analytics provides the clearest demonstration: the same September journey (Paid Search → Social → Email → Display → $50 purchase) produces dramatically different attribution results under different model + window combinations. A **7-day window** might give Display 100% credit (last touch within window); a **90-day window** might split credit across all four channels. For insurance with 60–90 day consideration cycles, standard 7–30 day windows miss most of the journey. Google Analytics defaults to 30 days for acquisition events and 90 days for other events. **Extend lookback windows to at least 90 days for insurance**, ideally 180 days.

### Channel taxonomy is the #1 failure point

"If you have bad data coming in, you're never going to be able to do multi-touch" (UTM.io). The MECE principle (Mutually Exclusive, Collectively Exhaustive) must govern channel categorization. Common failures: `facebook` vs `Facebook` vs `fb` fragmenting data; internal UTM tagging overwriting original source attribution; inconsistent use of utm_medium values. CXL's UTM guide (https://cxl.com/blog/utm-parameters/) is the most authoritative reference on proper parameter usage. Improvado's naming conventions guide (https://improvado.io/blog/utm-naming-conventions) provides three naming models (Cryptic, Positional, Key-Value) with governance frameworks.

### Synthetic data that doesn't embarrass you

For the Erie demo, convincing synthetic data needs: realistic channel mix proportions (heavy on search and agent referrals for insurance), appropriate journey lengths (3–8 touchpoints over 30–90 days), life-event triggers (home purchase, new car), conversion rates in the 2–5% range, and a healthy proportion of phone call touchpoints converting to agent interactions. The `mta` Python library and jakebenn's Jupyter notebook (https://github.com/jakebenn/multi-touch-attribution-markov-chains) provide starting templates for synthetic journey generation. Use Markov chains to generate synthetic paths from realistic transition matrices — this ensures statistical properties mirror real data.

### Model validation without ground truth

In the absence of incrementality experiments, validate through: **cross-model convergence** (if Shapley, Markov, and a logistic regression all point to similar relative channel rankings, you have more confidence), **axiom compliance testing** (verify Shapley values sum to total conversions, dummy channels get zero credit, symmetric channels get equal credit), **holdout simulation** (remove known high-value channels from the model and verify the model detects their absence), and **sensitivity analysis** (how much do results change with ±7 days on the lookback window or ±1 Markov order?). The AjNavneet GitHub repo (https://github.com/AjNavneet/MultiTouch-Attribution-Marketing-Spend-Optimization) implements a full cross-model comparison pipeline with budget optimization.

### Common failure modes in production

- **UTM data quality decay** over time as campaigns launch without governance
- **Lookback window mismatches** between platforms (Facebook 7-day vs Google 30-day) creating double-counting
- **Cookie fragmentation**: Corvidae found up to **80% of data incorrectly categorized** in GA360 Shapley due to cross-device failures
- **Single-touch journey dominance**: When 60–70% of journeys have only one touchpoint, MTA degenerates to last-click
- **Computational blowup**: Shapley becomes intractable above ~15 channels without approximation; higher-order Markov chains explode above order 3 with many channels
- **Stale models**: Attribution weights drift as channel mix and user behavior change; models need regular retraining

---

## 6. Erie Insurance and agent-distributed P&C attribution

> **★ START HERE:** Invoca's insurance solution page (https://www.invoca.com/solutions/insurance) — purpose-built for the exact problem of attributing digital marketing to phone calls to independent agents.

### The central challenge of agent-distributed attribution

Erie Insurance operates a **100% independent agent model** across 12 states. The typical customer journey: user sees a digital ad → researches online → contacts an independent agent → agent quotes and binds a policy in their Agency Management System. The attribution gap occurs at the handoff from digital to agent. **Only ~10% of insurance purchases complete entirely online** (AQ Marketing). Standard 7–30 day lookback windows miss most insurance journeys, which span **60–90 days** (Astha Technology).

### Call tracking bridges the gap

Insurance callers convert at **10× the rate** of web leads (Invoca). Dynamic Number Insertion (DNI) assigns unique phone numbers per campaign/session, linking calls to specific browsing activity. Invoca's insurance solution attributes "each phone call from marketing source to quotes and policies written — even when calls are driven to independent agents, franchisees, and locations." It's HIPAA/SOC2/PCI compliant. eHealth case study results: **20% more conversions, 20% lower CPA, $3M savings**. CallRail and WhatConverts (https://www.whatconverts.com/case/financial-and-insurance/) offer similar capabilities at different price points.

### AMS data quality is a minefield

The major AMS platforms — Applied Epic, Vertafore AMS360, HawkSoft, EZLynx — **weren't built for data administrators** (RecordLinker, https://recordlinker.com/data-governance-insurance-guide/). UIs focus on producers, not data teams. There are **no industry-wide coding standards** for carriers, coverages, or lines of business. Critical issue for attribution: CRM↔AMS integration is typically **unidirectional** (marketing CRM → AMS), meaning AMS updates on policy binding don't flow back to marketing systems (Synatic analysis, https://www.synatic.com/blog/why-insurance-producers-need-access-to-ams-data-in-hubspot). Post-M&A data audits routinely reveal years of accumulated quality issues. Applied Epic is powerful but has a steep learning curve; AMS360 suffers from "system reverting to default account information even after policy-specific updates" (SelectHub). For the demo, expect that AMS data matching will require fuzzy matching on name + address + phone, not clean deterministic joins.

### Life-event triggers and long consideration windows

Insurance purchases cluster around life events: home purchases, new cars, marriages, births, relocations. These create natural attribution windows that don't align with standard marketing lookback periods. **70% of shoppers research online before contacting an agent** (AQ Marketing). Facebook and Pinterest advertising for insurance should emphasize assisted conversions and full-funnel measurement rather than last-click attribution, which systematically undervalues awareness channels in long sales cycles.

For the demo, Google Meridian (https://github.com/google/meridian) is worth considering as a complementary approach — its Bayesian hierarchical geo-level modeling is ideal for measuring regional advertising impact on local agent sales across Erie's 12-state footprint, and it supports calibration with incrementality experiments at the geographic level.

---

## Conclusion: what to build first and why

The demo should implement a **three-model comparison**: Shapley (using Zhao et al.'s simplified formula via `marketing-attribution-models`), Markov chain (using `ChannelAttribution`), and a heuristic baseline (last-touch, linear). This shows Erie the range of outcomes and builds trust through transparency. Use the `mta` Python library for the broadest model coverage in a single package.

The single most important insight to communicate: **no attribution model produces causal ground truth from observational data alone**. Gordon et al.'s finding of 488–948% errors is not an outlier — it's the norm. The path forward is triangulation. For Erie specifically, the offline-to-online gap dwarfs the modeling sophistication question. Getting call tracking (Invoca/CallRail) integrated with campaign data and establishing even basic AMS-to-marketing data flow will deliver more value than perfecting the attribution algorithm. Build the data pipeline first, model second. The math is solved; the data engineering is where projects fail.

For the mathematical foundation, start with Zhao et al. (2018) and Anderl et al. (2016). For industry context, read Gordon et al. (2023) and the Amazon Ads MTA paper (2025). For implementation, use the DP6 Marketing-Attribution-Models package or the `mta` library. And always remember: the model is only as good as the journey data feeding it.