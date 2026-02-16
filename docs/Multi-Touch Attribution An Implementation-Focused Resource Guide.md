# Multi-touch attribution: a complete learning resource guide

**This curated collection of 70+ resources covers everything needed to ramp up on multi-touch attribution (MTA) — from foundational theory through production-ready Python code.** The guide is organized around eight topic areas, progressing from conceptual foundations through mathematical methods to practical implementation and industry context. Each resource is annotated with type, source, and a description of what it covers and why it matters. A suggested learning path at the end ties everything together for efficient ramp-up.

---

## 1. MTA fundamentals: what it is, why it matters, and how models differ

These introductory resources establish the core vocabulary and conceptual framework. Start here before diving into mathematical methods.

**"Multi-Touch Attribution — What It Is, and How to Do It Well"**
Adobe Experience Cloud Blog | Blog | [adobe.com/blog/basics/multi-touch-attribution](https://business.adobe.com/blog/basics/multi-touch-attribution)
Covers MTA definition, how it differs from single-touch (first-touch, last-touch), and walks through the main MTA models — linear, time decay, U-shaped, W-shaped — using a relatable travel-site example. One of the clearest narrative introductions available.

**"A Look at Multi-Touch Attribution & Its Various Models"**
HubSpot Marketing Blog | Blog | [blog.hubspot.com/marketing/multi-touch-attribution](https://blog.hubspot.com/marketing/multi-touch-attribution)
HubSpot's accessible primer on what makes MTA unique compared to first-touch and last-touch. Explains four common MTA models with practical examples. A strong starting point from a widely trusted marketing platform.

**"Methods & Models: A Guide to Multi-Touch Attribution"**
Nielsen (2019) | Industry Guide | [nielsen.com/insights/2019/methods-models-a-guide-to-multi-touch-attribution](https://www.nielsen.com/insights/2019/methods-models-a-guide-to-multi-touch-attribution/)
Authoritative guide from a measurement industry leader covering rule-based models (last-touch, first-touch, even-weighting, position-based, time-decay, custom) alongside algorithmic/data-driven MTA. Excellent for understanding the industry-standard perspective on when each model type is most useful.

**"An Introduction to Multi-Touch Attribution"**
Twilio Segment Academy | Educational Article | [segment.com/academy/advanced-analytics/an-introduction-to-multi-touch-attribution](https://segment.com/academy/advanced-analytics/an-introduction-to-multi-touch-attribution/)
In-depth piece with excellent analogies (the Oscar acceptance speech metaphor for attribution). Covers all model types and discusses implementation challenges including **cross-device tracking** — a topic many introductions skip. Particularly well-written.

**"A Beginner's Guide to Attribution Model Frameworks"**
Amplitude Blog | Blog | [amplitude.com/blog/attribution-model-frameworks](https://amplitude.com/blog/attribution-model-frameworks)
Dives into rule-based vs. data-driven (algorithmic) attribution, explains how data-driven models use machine learning, and provides practical guidance on choosing the right model based on business model, strategy, and budget. Good for someone with a quantitative bent.

**"The Difference Between Rule-Based and Data-Driven Attribution"**
Roivenue | Blog | [roivenue.com/articles/the-difference-between-rule-based-and-data-driven-attribution](https://roivenue.com/articles/the-difference-between-rule-based-and-data-driven-attribution/)
Focused comparison of rule-based vs. data-driven attribution methods, explicitly naming Shapley values, Markov chains, and neural networks as data-driven approaches. Concise and useful for understanding why data-driven models are preferred for serious optimization.

**"Tips to Nail Your Marketing Attribution Model"**
Think with Google — Murtaza Lukmani | Article | [thinkwithgoogle.com](https://www.thinkwithgoogle.com/intl/en-emea/marketing-strategies/data-and-measurement/overhaul-marketing-attribution-model/)
Google's own perspective on why last-click is limited and how data-driven attribution delivers superior results. Includes practical implementation advice from Google's measurement team on pilot testing and cross-device, cross-channel attribution.

**"Multi-Touch Attribution: What It Is, Models, & More"**
Marketing Evolution | Guide | [marketingevolution.com/marketing-essentials/multi-touch-attribution](https://www.marketingevolution.com/marketing-essentials/multi-touch-attribution)
Detailed overview covering all major model types plus implementation steps and the relationship between MTA and Media Mix Modeling. Includes clear Nike purchase-journey examples.

---

## 2. Shapley values and cooperative game theory for attribution

The mathematical core of data-driven attribution. These resources progress from the original theory through marketing-specific papers to hands-on code.

### Foundational papers

**"A Value for n-Person Games" (1953)**
Lloyd S. Shapley, RAND Corporation | Seminal Paper | [rand.org/pubs/papers/P295.html](https://www.rand.org/pubs/papers/P295.html) | [PDF](https://www.rand.org/content/dam/rand/pubs/papers/2021/P295.pdf)
The foundational paper that introduced the Shapley value — the theoretical bedrock underlying all Shapley-based attribution. Defines the value axiomatically via three properties (efficiency, symmetry, additivity) and proves a unique solution exists. Originally published in *Contributions to the Theory of Games, Vol. II* (Princeton University Press). **The RAND PDF is freely accessible.**

**"Data-Driven Multi-Touch Attribution Models" (2011)**
Xuhui Shao and Lexin Li | KDD '11, pp. 258–264 | [Semantic Scholar](https://www.semanticscholar.org/paper/Data-driven-multi-touch-attribution-models-Shao-Li/09c397be5c654041d55451022396b2ed26f0f56a) | [ResearchGate](https://www.researchgate.net/publication/221654662_Data-driven_multi-touch_attribution_models)
**The first major paper bridging Shapley values and marketing attribution.** Proposes a probabilistic model to quantify attribution across advertising channels, using bagged logistic regression for stable estimates and Shapley values to distribute credit. Widely cited as foundational. Full text may require ACM access, but preprints are available.

**"Shapley Value Methods for Attribution Modeling in Online Advertising" (2018)**
Kaifeng Zhao, Seyed Hanif Mahboobi, Saeed R. Bagheri | arXiv preprint | [arxiv.org/abs/1804.05327](https://arxiv.org/abs/1804.05327)
Simplifies Shapley value computation dramatically — **reducing analysis from 17 hours to ~2 minutes** in their example. Proposes an "ordered Shapley value" method that accounts for the sequence of channels visited by users. This is the paper explicitly referenced by Google's Ads Data Hub Shapley implementation. Freely available.

### Google's Shapley-based attribution work

**Shapley Value Analysis — Google Ads Data Hub Documentation**
Google for Developers | Technical Documentation | [developers.google.com/ads-data-hub/guides/shapley](https://developers.google.com/ads-data-hub/guides/shapley)
Official developer documentation for implementing Shapley attribution in Ads Data Hub. Shows SQL code examples calling `ADH.TOUCHPOINT_ANALYSIS` with `'SHAPLEY_VALUES'` as the model parameter. Uses the simplified method from Zhao et al. (2018). Updated September 2024.

**MCF Data-Driven Attribution Methodology**
Google Analytics Help | Official Documentation | [support.google.com/analytics/answer/3191594](https://support.google.com/analytics/answer/3191594?hl=en)
Explains how Google's Multi-Channel Funnels DDA works: analyzing path data to build conversion probability models, then applying Shapley values to assign partial credit via counterfactual gains. Includes a worked example with Organic Search, Display, and Email. Legacy UA documentation but the methodology explanation remains highly instructive.

**"Toward Improving Digital Attribution Model Accuracy"**
Stephanie Sapp and Jon Vaver, Google Inc. | Research Paper | [research.google.com/pubs/archive/45766.pdf](https://research.google.com/pubs/archive/45766.pdf)
Internal Google research describing their Unified Data-Driven Attribution (UDDA) approach, which compares exposed vs. unexposed user sets. A key behind-the-scenes look at how Google built its attribution products.

### Tutorials and code for Shapley attribution

**"Marketing Attribution: Step Up Your Marketing Attribution with Game Theory"**
Reda Affane (Dataiku) | Medium / Data from the Trenches | [medium.com/data-from-the-trenches/marketing-attribution-e7fa7ae9e919](https://medium.com/data-from-the-trenches/marketing-attribution-e7fa7ae9e919)
**One of the most complete applied tutorials available.** Starts from basics, builds up to the Shapley value solution with full mathematical exposition, and includes SQL for data prep and Python code implementing the Shapley equation. Walks through a two-channel example step by step before generalizing.

**"Multi-Touch Attribution Model Using Shapley Value"**
Bernard (bernard-mlab.com) | Blog + GitHub | [Blog](https://bernard-mlab.com/post/mta-sharpley-value/) | [GitHub](https://github.com/bernard-mlab/Multi-Touch-Attribution_ShapleyValue)
Clean Python implementation using a Kaggle dataset. Walks through power set generation, factorial computation, coalition worth, and Shapley formula from scratch. The blog provides excellent theoretical grounding in the four axioms (efficiency, symmetry, dummy, additivity). The GitHub repo contains complete runnable code.

**"Shapley Value Attribution Modeling" (Kaggle Notebook)**
Jason Brewster | Kaggle | [kaggle.com/code/jasonbrewster/shapley-value-attribution-modeling](https://www.kaggle.com/code/jasonbrewster/shapley-value-attribution-modeling)
Interactive, executable notebook on Kaggle implementing Shapley value attribution. Good for experimentation without local setup.

### Accessible Shapley value explainers

**"Shapley Values" — Interpretable Machine Learning Book, Chapter 17**
Christoph Molnar | Free Online Book | [christophm.github.io/interpretable-ml-book/shapley.html](https://christophm.github.io/interpretable-ml-book/shapley.html)
**Perhaps the single best accessible explanation of Shapley values for a quantitative audience.** Uses an apartment price prediction example to build intuition from scratch. Covers the formal definition, worked examples, estimation approaches, strengths, and limitations. The companion chapter on SHAP (Ch. 18) covers SHAP-specific extensions.

**"The Shapley Value for ML Models"**
Towards Data Science | Blog | [towardsdatascience.com/the-shapley-value-for-ml-models-f1100bff78d1](https://towardsdatascience.com/the-shapley-value-for-ml-models-f1100bff78d1/)
Builds intuition starting from cooperative game theory and transitions to ML applications using a mortgage lending example. Covers all four axioms and contextualizes Shapley values as comparison-based explanations.

### The SHAP connection

**"A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)**
Scott M. Lundberg and Su-In Lee | Paper | [arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)
The seminal SHAP paper showing that Shapley values are the unique solution satisfying local accuracy, missingness, and consistency for additive feature attribution. Unifies six existing explanation methods (LIME, DeepLIFT, etc.) under one framework. The theoretical connection to marketing attribution is direct: both use Shapley values to fairly allocate "credit" among "players."

**SHAP Official Documentation — "An Introduction to Explainable AI with Shapley Values"**
SHAP library (Scott Lundberg et al.) | Interactive Tutorial | [shap.readthedocs.io](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)
Hands-on, progressive introduction with runnable Python code. Demonstrates waterfall plots, beeswarm plots, and interaction effects. Conceptual patterns transfer directly to attribution contexts.

---

## 3. Markov chain attribution models and the removal effect

Markov chains model channel sequencing and transition probabilities, offering a complementary approach to Shapley values. These resources cover the foundational theory, the key "removal effect" methodology, and practical implementations.

### Foundational paper

**"Mapping the Customer Journey: A Graph-Based Framework for Online Attribution Modeling"**
Eva Anderl, Ingo Becker, Florian von Wangenheim, Jan Hendrik Schumann | Paper | [SSRN (2014 working paper)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2343077) | [Published version (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0167811616300349)
**THE foundational paper for Markov chain attribution.** Published in *International Journal of Research in Marketing*, Vol. 33(3), pp. 457–474 (2016). Introduces a graph-based framework modeling multichannel customer journeys as first- and higher-order Markov walks. Evaluated on four large real-world datasets. Establishes the removal effect methodology for channel importance measurement and demonstrates substantial differences from heuristic methods.

### Tutorials explaining the removal effect and Markov approach

**AnalyzeCore Blog Series (3 parts)**
Sergii Bryl' | Blog Series | [Part 1](https://www.analyzecore.com/2016/08/03/attribution-model-r-part-1/) | [Part 2](https://www.analyzecore.com/2017/05/31/marketing-multi-channel-attribution-model-r-part-2-practical-issues/) | [Part 3](https://www.analyzecore.com/2017/09/28/marketing-multi-channel-attribution-model-based-on-sales-funnel-with-r/)
**The single most-referenced tutorial on Markov chain attribution.** Part 1 covers Markov chain concepts, transition matrices, removal effect, and R implementation. Part 2 addresses practical issues: handling Direct traffic, missing values, higher-order chains, duplicated touchpoints. Part 3 introduces a Sales Funnel hybrid approach. Includes complete R code with visualizations. Widely cited as the best practical introduction.

**"Markov Chain Attribution — Simple Explanation of Removal Effect"**
Serhii Puzyrov | Blog | [serhiipuzyrov.com](https://serhiipuzyrov.com/2019/07/markov-chain-attribution-simple-explanation-of-removal-effect/)
Explains removal effect math in the simplest possible way, without formulas. Uses a 4-path example to show step-by-step how removal effect is computed. Ideal for someone who finds the formal math intimidating.

**"A Beginner's Guide to Channel Attribution Modeling in Marketing (using Markov Chains)"**
Analytics Vidhya (2018) | Tutorial | [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2018/01/channel-attribution-modeling-using-markov-chains-in-r/)
Comprehensive beginner-friendly tutorial. Explains Markov chain properties (state space, transition operator), removal effect with step-by-step calculations, and implements attribution in R using ChannelAttribution. Includes heuristic vs. Markov model comparison visualizations.

**"Markov Chain Attribution Modeling [Complete Guide]"**
Adequate Digital | Guide | [adequate.digital](https://adequate.digital/en/markov-chain-attribution-modeling-complete-guide/)
Thorough walkthrough covering the mathematical foundation, removal effect formula, normalization, and the important problem of handling loops in Markov graphs (which significantly complicate probability calculations).

**"How to Calculate Removal Effects in Markov Chain Attribution?"**
Erika Gintautas (Mister Spex Tech Blog) | Blog | [blog.misterspex.tech](https://blog.misterspex.tech/how-to-calculate-removal-effects-in-markov-attribution-4a44a2c3c15c)
A practitioner's deep dive into removal effect calculation using stochastic simulations — replicating what ChannelAttribution does internally. Explains why simple textbook examples don't scale to real data, with Python code on GitHub. Valuable for understanding what's happening "under the hood."

**"7 Lessons From Building 20 Markov Chain Attribution Models on Real Datasets"**
Ridhima Kumar (Aryma Labs) | Blog | [Medium](https://ridhima-kumar0203.medium.com/7-lessons-from-building-20-markov-chain-attribution-models-on-real-datasets-f30ee5562be8)
Practical lessons from production implementations. Key takeaways: **~60% of work is data preprocessing**; timestamp ordering is critical; break data into converting vs. non-converting paths; interpret results through domain experience. Great for practitioners moving from theory to practice.

### R and Python packages

**ChannelAttribution R Package**
Davide Altomare and David Loris | R Package (CRAN) | [CRAN](https://cran.r-project.org/package=ChannelAttribution) | [White Paper PDF](https://channelattribution.io/pdf/ChannelAttributionWhitePaper.pdf)
The most widely-used Markov chain attribution implementation. Key functions: `markov_model()`, `heuristic_models()`, `choose_order()`, `transition_matrix()`. The white paper is an excellent pedagogical resource with worked examples and complete R and Python code.

**ChannelAttribution Python Library**
Davide Altomare and David Loris | Python Package | [PyPI](https://pypi.org/project/ChannelAttribution/) | [GitHub](https://github.com/DavideAltomare/ChannelAttribution)
Official Python port of the R package. Implements k-order Markov model plus heuristic models. Install via `pip install ChannelAttribution`.

---

## 4. Identity resolution: connecting users across devices and channels

Identity resolution is the infrastructure layer that makes MTA possible. Without it, touchpoints from the same user across devices and channels cannot be linked.

**"What Is Identity Resolution and Why Is It Important?"**
LiveRamp | Product Overview | [liveramp.com/identity-resolution](https://liveramp.com/identity-resolution/)
Comprehensive overview from the industry's largest deterministic identity graph provider. Covers how LiveRamp's AbiliTec Identity Graph resolves offline PII and links to online identifiers. Key stat: **LiveRamp maintains PII on 245 million individuals in the U.S.**

**"RampID Methodology"**
LiveRamp Documentation | Technical Documentation | [docs.liveramp.com](https://docs.liveramp.com/identity/en/rampid-methodology.html)
Deep technical explanation of how LiveRamp's identity graph actually works: AbiliTec offline PII merging, deterministic online device linking, RampID creation, quality assurance processes. Best resource for understanding identity graph architecture at a technical level.

**"Probabilistic vs Deterministic Matching: Our Viewpoint"**
LiveRamp Blog | Blog | [liveramp.com/blog/probabilistic-vs-deterministic](https://liveramp.com/blog/probabilistic-vs-deterministic/)
Authoritative comparison of the two matching methods. Key insight: deterministic should be the backbone of marketing identity, while probabilistic adds complementary scale but introduces more false positives. **LiveRamp's graph reaches 200M+ unique users deterministically and 600M+ matched mobile devices.**

**"What Is Identity Resolution?"**
TransUnion Blog | Blog | [transunion.com/blog/what-is-identity-resolution](https://www.transunion.com/blog/what-is-identity-resolution)
Explains how identity graphs store all known identifiers correlated to individuals. Covers the "layer cake" approach of using multiple identity methodologies. Notes that the **average US household has 20+ internet-enabled devices**.

**TransUnion Enhanced Identity Graph Announcement (2024)**
TransUnion Newsroom | Press Release | [newsroom.transunion.com](https://newsroom.transunion.com/transunion-announces-enhanced-identity-graph-for-marketing-solutions/)
Details TruAudience's AI-powered four-stage identity process. Scale: 98% of U.S. adult population, **125M+ households, 250M adults, 1.9B phone numbers, 1B+ mobile devices, 1.6B emails.** Reports 40% reduction in duplicate CRM records and 30% increase in conversions for clients.

**"Identity Resolution Guide"**
Experian Marketing | Industry Guide | [experian.com/marketing/resources/resolution/identity-resolution-guide](https://www.experian.com/marketing/resources/resolution/identity-resolution-guide)
Comprehensive guide covering the full lifecycle: data onboarding, identity graphing, deterministic vs. probabilistic methods, and privacy considerations (COPPA, CCPA, GDPR). Notes consumers have 5+ identifiers on average. Emphasizes privacy-first solutions for the cookieless era.

**"Identity Resolution: What It Is and How It Works"**
Twilio/Segment | Blog | [twilio.com/en-us/blog/insights/identity-resolution](https://www.twilio.com/en-us/blog/insights/identity-resolution)
Practical guide covering the core process: data collection → matching → profile unification → activation. Key stat: **98% of website visitors are anonymous**, making identity resolution critical.

**"Deterministic vs Probabilistic Matching, Explained"**
BlueConic | Educational Guide | [blueconic.com/resources/probabilistic-and-deterministic-matching-explained](https://www.blueconic.com/resources/probabilistic-and-deterministic-matching-explained)
Real-world examples across verticals. Explains that deterministic matching requires login/authentication (more precise, less frequent), while probabilistic is more scalable but less accurate. Good for understanding tradeoffs.

---

## 5. Insurance and P&C marketing: the unique attribution challenge

Erie Insurance's independent agent model creates a distinctive attribution problem: the carrier's digital marketing drives online research, but conversion happens offline through independent agents. These resources address that specific dynamic.

**"Marketing Intelligence for Insurance Agencies: Tracking, Attribution, and Data-Driven Growth"**
David Carothers, Killing Commercial (Oct 2025) | Blog | [killingcommercial.com](https://killingcommercial.com/blog/marketing-intelligence-for-insurance-agencies-tracking-attribution-and-data-driven-growth/)
**The most directly relevant resource found** — a detailed guide specifically for insurance agencies on marketing attribution. Covers why agencies struggle with attribution (disconnected CRM/website systems), building attribution foundations with UTM tracking and CRM integration, the "dark social" challenge, and AI-enhanced marketing analysis. Includes insurance-specific use cases for personal lines agencies and commercial agencies.

**"Beyond Price: The Rise of Customer-Centric Marketing in Insurance"**
McKinsey & Company | Research Report (PDF) | [mckinsey.com](https://www.mckinsey.com/~/media/mckinsey/dotcom/client_service/financial%20services/latest%20thinking/insurance/beyond_price_the_rise_of_customer-centric_marketing_in_insurance.ashx)
Based on McKinsey's U.S. Auto Insurance Buyer Survey. Key findings: **U.S. auto insurers spend ~$6B annually on marketing**; digital and social channels influence 40% of consumer decisions; there are 9+ distinct auto insurance consumer segments. Raises fundamental questions about customer journey mapping and marketing effectiveness.

**"How Asian Insurers Can Use Digital Marketing to Fuel Growth"**
McKinsey & Company | Research Article | [mckinsey.com](https://www.mckinsey.com/industries/financial-services/our-insights/how-asian-insurers-can-use-digital-marketing-to-fuel-growth)
Contains globally applicable insights. Key finding: **best-in-class digital marketing can unlock 50–100% sales uplift** and reduce marketing investment 10–15% through attribution-based optimization. Discusses the digital-hybrid growth model where digital marketing drives leads to agents — directly relevant to Erie's independent agent distribution model.

**"The Growth Engine: Superior Customer Experience in Insurance"**
McKinsey & Company | Research Report | [mckinsey.com](https://www.mckinsey.com/industries/financial-services/our-insights/the-growth-engine-superior-customer-experience-in-insurance)
Covers insurance customer journey mapping. Key insight: carriers deliver experiences via separate functions (marketing, distribution, underwriting, claims) managed by different executives, but customers see a single journey. Directly relevant to understanding **why attribution is complex in insurance**: the handoff between digital marketing, agent interactions, underwriting, and policy issuance creates attribution gaps.

**"Why Auto Insurance Marketers Are Turning to Performance Marketing"**
Perform[cb] | Industry Article | [performcb.com](https://www.performcb.com/content-hub/why-auto-insurance-marketers-are-turning-to-performance-marketing/)
Covers auto insurance attribution challenges and the shift to performance-based marketing. Notes that auto insurance decisions involve many touchpoints, making strong attribution models crucial. Claims performance marketing can increase ROI by 20% through better attribution.

**"Digital Marketing Insurance Industry: 10 Powerful 2025 Success Secrets"**
AQ Marketing (2025) | Guide | [aqmarketing.com](https://aqmarketing.com/digital-marketing-insurance-industry/)
Key insight encapsulating the core attribution challenge: **nearly 70% of insurance shoppers research online before contacting an agent, yet only 10% complete the entire purchase digitally.** This online-to-offline gap is the central attribution challenge for carriers like Erie.

**"What Is Marketing Attribution? Basics, Benefits, and Best Practices"**
Invoca | Guide | [invoca.com/blog/what-is-marketing-attribution-basics-benefits](https://www.invoca.com/blog/what-is-marketing-attribution-basics-benefits)
Highly relevant because Invoca specializes in **phone call attribution** — the critical online-to-offline bridge for insurance. Notes that 68% of customers prefer phone calls for high-stakes purchases like insurance. Covers how to bridge the online-to-offline attribution gap using conversation intelligence.

---

## 6. Practical implementation: Python libraries, GitHub repos, and notebooks

These hands-on resources let you build working attribution models. Three Python libraries stand out as must-know tools.

### Key Python libraries

- **`mta` (Multi-Touch Attribution)** by Igor Korostil | [GitHub](https://github.com/eeghor/mta) — The most comprehensive open-source library. Implements 10+ models: first-touch, last-touch, linear, position-based, time-decay, Markov chain, Shapley value, Shao & Li model, logistic regression, and additive hazard. Features a chainable API. MIT licensed.

- **`marketing-attribution-models`** by DP6 | [GitHub](https://github.com/DP6/Marketing-Attribution-Models) | [PyPI](https://pypi.org/project/marketing-attribution-models/) — Implements 6 heuristic models plus Markov Chains and Shapley Value. Works with Google Analytics Top Conversion Paths data. Install via `pip install marketing-attribution-models`.

- **`ChannelAttribution`** (Python port) by Altomare & Loris | [PyPI](https://pypi.org/project/ChannelAttribution/) | [GitHub](https://github.com/DavideAltomare/ChannelAttribution) — Official Python port of the widely-used R package. Implements k-order Markov model plus heuristic models.

### GitHub repositories and notebooks

**Shapley Attribution Model (Zhao et al. Implementation)**
Ian Chute | GitHub | [github.com/ianchute/shapley-attribution-model-zhao-naive](https://github.com/ianchute/shapley-attribution-model-zhao-naive)
Python implementation of Zhao et al.'s simplified Shapley method. Includes both standard and ordered Shapley models. Simple API: feed customer journeys and get attribution scores.

**Markov Chain Attribution from Scratch**
Jake Benn | GitHub | [github.com/jakebenn/multi-touch-attribution-markov-chains](https://github.com/jakebenn/multi-touch-attribution-markov-chains)
Clean proof-of-concept Jupyter notebook. Builds transition matrices and calculates removal effects from scratch. Good starting point for hands-on learners.

**Markov Chain Attribution (jerednel)**
Jeremy Nelson | GitHub | [github.com/jerednel/markov-chain-attribution](https://github.com/jerednel/markov-chain-attribution)
Lightweight Python package (`pip install markov_model_attribution`) implementing first-order Markov chain attribution. Simple API returning Markov-attributed conversions and removal effect matrix. Created for learning the process rather than using ChannelAttribution as a black box.

**"Multitouch Attribution Modelling" (Kaggle)**
Hugh Huyton | Kaggle Notebook | [kaggle.com/code/hughhuyton/multitouch-attribution-modelling](https://www.kaggle.com/code/hughhuyton/multitouch-attribution-modelling)
Executable multi-touch attribution notebook on Kaggle with real data.

**"Python Implementation of Markov Chain Attribution Model"**
Akanksha Anand | Medium Tutorial | [medium.com/@akanksha.etc302/python-implementation-of-markov-chain-attribution-model](https://medium.com/@akanksha.etc302/python-implementation-of-markov-chain-attribution-model-0924687e4037)
Step-by-step from-scratch implementation using pandas, numpy, and seaborn. Uses a 240K-customer dataset. Covers data preprocessing, transition matrix construction, removal effect calculation, and result visualization.

**"Channel Attribution in Python"**
Victor Angelo Blancada | Tutorial | [victorangeloblancada.github.io](https://victorangeloblancada.github.io/blog/2019/01/01/channel-attribution-in-python.html)
Step-by-step Python tutorial with mathematical formulas and code. Builds transition probability matrices, computes removal rates, and derives conversion attribution through normalization.

### Key datasets

**Criteo Attribution Modeling for Bidding Dataset**
Criteo AI Lab | Dataset + Paper | [criteo.com](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/) | [Kaggle mirror](https://www.kaggle.com/datasets/sharatsachin/criteo-attribution-modeling)
**The most authoritative public dataset for attribution modeling research.** ~16M impressions with timestamps, conversion data, click data, attribution labels, cost, and cost-per-order. Accompanies the paper "Attribution Modeling Increases Efficiency of Bidding in Display Advertising" (AdKDD 2017). CC BY-NC-SA 4.0.

### End-to-end tutorial

**"Build a Data Driven Attribution Model using GA4, BigQuery and Python"**
Stacktonic.com | Tutorial | [stacktonic.com](https://stacktonic.com/article/build-a-data-driven-attribution-model-using-google-analytics-4-big-query-and-python)
Comprehensive real-world tutorial connecting GA4 event-level exports via BigQuery to Python attribution models using the DP6 library. Builds both rule-based and data-driven (Markov) models with results saved back to BigQuery. Excellent for practitioners working with Google Analytics data.

### Advanced: RNN + Shapley

**JD-MTA: Causally Driven Multi-Touch Attribution**
Du, Zhong, Nair, Cui, Shou (Stanford/JD.com) | GitHub | [github.com/jd-ads-data/jd-mta](https://github.com/jd-ads-data/jd-mta)
TensorFlow implementation of an RNN-based causal attribution model with Shapley value computation. Academic-grade code for learning advanced causal MTA approaches beyond heuristic Markov/Shapley methods.

---

## 7. Media mix modeling: the complementary top-down view

MMM operates at an aggregate level using historical spend data, while MTA operates at the individual user level. Together they form a complete measurement framework. These resources provide enough context to understand the relationship.

**"Multi-Touch Attribution vs. Marketing Mix Modeling"**
Funnel.io (2023, updated 2025) | Blog | [funnel.io/blog/mta-vs-mmm](https://funnel.io/blog/mta-vs-mmm)
The single best comparison article. Explains that **MTA is bottom-up** (granular, device-level, individual touchpoints) while **MMM is top-down** (aggregated historical data, channel/campaign level). Covers objectives, data types, strengths/weaknesses, and when to use each.

**"Meridian Is Now Available to Everyone" (January 2025)**
Google Official Blog | Announcement | [blog.google](https://blog.google/products/ads-commerce/meridian-marketing-mix-model-open-to-everyone/)
Google's official launch of their open-source Bayesian MMM. Uses Bayesian causal inference with aggregated privacy-safe data. Can incorporate Google Search query volume and YouTube reach/frequency data. 20+ certified measurement partners.

**Google Meridian — GitHub and Documentation**
Google | Open-Source + Docs | [GitHub](https://github.com/google/meridian) | [Developer Docs](https://developers.google.com/meridian/docs/basics/meridian-introduction)
The actual codebase and comprehensive documentation. Covers the four pillars (Accuracy, Actionability, Adaptability, Privacy-Durability), Bayesian regression with adstock/saturation, geo-level hierarchical modeling, and budget optimization. Includes colabs and tutorials.

**Meta's Robyn — Open-Source MMM**
Meta Marketing Science | Open-Source + Docs + Course | [GitHub](https://github.com/facebookexperimental/Robyn) | [Documentation](https://facebookexperimental.github.io/Robyn/) | [Analyst's Guide](https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/) | [Blueprint Course](https://www.facebookblueprint.com/student/path/253121)
Meta's AI/ML-powered MMM package using Ridge regression, multi-objective evolutionary optimization, Prophet for time-series decomposition, and gradient-based budget allocation. R-based (Python in development). **1,200+ GitHub stars.** Meta Blueprint offers a free online course.

**"Bayesian Media Mix Modeling for Marketing Optimization"**
PyMC Labs (with HelloFresh) | Technical Blog | [pymc-labs.com](https://www.pymc-labs.com/blog-posts/bayesian-media-mix-modeling-for-marketing-optimization)
Explains how Bayesian inference improves on frequentist approaches through prior knowledge incorporation, uncertainty quantification, and calibration with experiments. Based on real collaboration with HelloFresh. References the [PyMC-Marketing library](https://www.pymc-marketing.io/en/latest/guide/mmm/mmm_intro.html).

**"Introduction to Bayesian Methods for MMM"**
Recast | Technical Blog + Notebook | [getrecast.com/bayesian-methods-for-mmm](https://getrecast.com/bayesian-methods-for-mmm/)
Hands-on introduction with a Python notebook demonstrating Bayesian regression with PyMC3. Companion post ["Google Meridian MMM: Features and Limitations"](https://getrecast.com/google-meridian/) provides a balanced critique.

---

## 8. Google Analytics 4's data-driven attribution under the hood

GA4's DDA became the default attribution model in 2023 when Google deprecated rule-based models. It uses Shapley values internally combined with machine learning.

**"Get Started with Attribution" — GA4 Official Documentation**
Google | Official Docs | [support.google.com/analytics/answer/10596866](https://support.google.com/analytics/answer/10596866?hl=en)
The canonical reference. Describes three available models and details the DDA methodology: machine learning on converting and non-converting paths, counterfactual approach, and factors considered (time, device type, ad interactions, creative assets).

**"Google's GA4 Data-Driven Attribution Model Explained"**
Adswerve | Blog | [adswerve.com](https://adswerve.com/blog/googles-ga4-data-driven-attribution-model-explained)
Excellent practitioner-friendly explanation. Explains how DDA trains a probabilistic model and uses the Shapley algorithm to measure each feature's contribution. Uses a "factory" analogy for Shapley values. Covers GA4 setup and why each account's model is unique.

**"Everything You Need to Know About GA4 Data-Driven Attribution"**
Growth Method | Blog | [growthmethod.com/data-driven-attribution](https://growthmethod.com/data-driven-attribution/)
Comprehensive timeline of Google's DDA evolution (2013 → 2016 → 2020 → 2023 default). Explains that GA4's DDA is based on the **Shapley model with an added time-decay element** and covers the 2023 deprecation of rule-based models.

**"The Shapley Value in Data-Driven Attribution: How It Works"**
Last Click City | Blog | [lastclick.city](https://lastclick.city/the-shapley-value-in-data-driven-attribution.html)
Focused quantitative walkthrough with a concrete three-channel example (organic search, paid search, display) showing all permutations and the mathematical calculation. Shows how DDA redistributes credit vs. last-click. Ideal for someone with a quantitative background.

---

## Conclusion: a suggested four-week learning path

The resources above form a complete curriculum. For efficient ramp-up on an MTA demo project, this sequencing maximizes understanding per hour invested:

**Week 1 — Conceptual foundations.** Read the HubSpot and Adobe MTA introductions, then Nielsen's guide. Follow with Christoph Molnar's Shapley values chapter and the Funnel.io MMM-vs-MTA comparison. End with the AQ Marketing insurance piece to ground everything in the Erie-specific context. By Friday, you should be able to explain what MTA is, why Shapley and Markov are the two dominant data-driven approaches, and why insurance attribution is uniquely challenging due to the **70% online research / 10% online purchase gap**.

**Week 2 — Mathematical methods.** Read the Shapley (1953) original and Shao & Li (2011) abstract. Work through the Last Click City Shapley walkthrough and Reda Affane's Medium tutorial with code. Read the Anderl et al. (2016) abstract and Serhii Puzyrov's removal effect explainer. Follow the AnalyzeCore Part 1 blog. Study the Google MCF DDA methodology documentation to see how Google implements Shapley internally.

**Week 3 — Hands-on implementation.** Install the `marketing-attribution-models` (DP6) and `mta` (eeghor) Python libraries. Run both Shapley and Markov attribution on the Criteo dataset. Work through the Bernard-ML Shapley notebook and Jake Benn's Markov notebook. Try the Stacktonic GA4-to-BigQuery-to-Python tutorial if GA4 data is available.

**Week 4 — Integration and demo building.** Study the identity resolution resources (LiveRamp deterministic vs. probabilistic) to understand the data infrastructure layer. Review Google's Ads Data Hub Shapley documentation for production-scale patterns. Read the McKinsey insurance reports for stakeholder-facing context. Build the demo, combining Markov chain transition visualization with Shapley value credit allocation, using the DP6 or `mta` library as the computational backbone.