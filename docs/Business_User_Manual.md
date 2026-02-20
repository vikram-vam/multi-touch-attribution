# MCA Intelligence — Business User Manual
## Erie Insurance Multi-Channel Attribution Demo

**Version:** 1.0 | **Date:** February 2026  
**Audience:** Erie Insurance Marketing Leadership, CMO, VP of Marketing, Analytics Team  
**Confidentiality:** Internal — Client-Facing Demo Material  

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [Why Multi-Channel Attribution Matters for Erie](#2-why-multi-channel-attribution-matters-for-erie)
3. [How the Tool Works — Conceptual Overview](#3-how-the-tool-works--conceptual-overview)
4. [Navigating the Dashboard — Page-by-Page Guide](#4-navigating-the-dashboard--page-by-page-guide)
5. [Interpreting Key Findings](#5-interpreting-key-findings)
6. [Using the Budget Simulator](#6-using-the-budget-simulator)
7. [Scenario Explorer — What-If Analysis](#7-scenario-explorer--what-if-analysis)
8. [Frequently Asked Questions](#8-frequently-asked-questions)
9. [From Demo to Production — The Path Forward](#9-from-demo-to-production--the-path-forward)
10. [Glossary of Terms](#10-glossary-of-terms)
11. [Appendix: Methodology Summary for Business Users](#11-appendix-methodology-summary-for-business-users)

---

## 1. Executive Overview

### What This Tool Is

MCA Intelligence is an interactive analytics dashboard that reveals how each of Erie's marketing channels — from TV and display advertising to search, social media, direct mail, and critically, your independent agent network — contributes to generating auto insurance quotes and policy binds.

The tool uses three independent, mathematically rigorous attribution frameworks to measure channel contribution, replacing the industry-standard "last-click" approach that most digital analytics platforms (including Google Analytics 4) provide by default.

### The Core Finding

Erie's current analytics likely tells you that branded paid search drives roughly 35-40% of your conversions. Our analysis, using three independent mathematical frameworks, shows it's closer to 11%. The difference? Your current model cannot see what happens after someone clicks a search ad and then picks up the phone to call their local agent.

When measured properly, your independent agent network receives approximately 35% of attribution credit — making it your single most valuable conversion channel. Under last-click attribution, agents receive roughly 3%.

### What This Means in Dollars

By reallocating marketing spend based on accurate attribution — shifting budget toward channels that truly drive binds and away from channels that receive inflated credit — our models project a 14-17% improvement in cost per bind. On a $5 million annual marketing budget, that translates to approximately 780 additional policy binds per year at the same total spend.

---

## 2. Why Multi-Channel Attribution Matters for Erie

### Erie's Unique Attribution Challenge

Erie Insurance operates with a 100% independent agent distribution model across 12 states plus DC. This creates a fundamentally different marketing attribution problem than what direct-to-consumer carriers like GEICO or Progressive face. For Erie, the typical customer journey looks something like this:

A prospect sees a TV ad, later searches "car insurance quotes" on Google, clicks on Erie's website, uses the Agent Locator to find a local agent, calls that agent, meets in person, and ultimately binds a policy through the agent's system.

In that journey, six separate interactions occurred across six different data systems. Google Analytics only sees two of them (the search click and the website visit). The agent interaction — which actually closed the deal — is completely invisible to digital analytics.

### What "Last-Click" Attribution Gets Wrong

Last-click attribution gives 100% of the conversion credit to the last digital touchpoint before a conversion. For Erie, this means branded paid search captures a hugely disproportionate share of credit because it's often the last thing someone clicks before calling an agent. The agent call, the TV ad that created awareness, the display ad that triggered the initial research — none of these receive any credit.

This creates three problems for budget decisions. First, it makes upper-funnel channels (TV, display, social) appear ineffective, creating pressure to cut their budgets. Second, it makes branded search appear dramatically more effective than it actually is, encouraging over-investment. Third, it makes your agent network — your single greatest competitive asset — invisible in marketing analytics.

### What Proper Attribution Reveals

When we stitch together customer identities across all systems and apply three independent mathematical frameworks, a very different picture emerges. Agents are your dominant conversion channel (roughly 35% of bind credit). TV and display advertising create the awareness that ultimately leads to agent conversations. Branded search is a capture channel, not a driver — it catches existing demand rather than creating it. The path "awareness channel → search → agent locator → agent call → bind" is the highest-converting customer journey for Erie.

---

## 3. How the Tool Works — Conceptual Overview

### Three Attribution Lenses

Rather than relying on a single model, this tool runs three fundamentally different attribution frameworks and compares their results. Where all three agree, you can act with high confidence. Where they disagree, the tool flags those channels for further investigation.

**Lens 1: Shapley Values — "What's Fair?"**

Borrowed from cooperative game theory (and the subject of a Nobel Prize in Economics), Shapley Values ask: if each marketing channel is a "player" in a team game, what is each player's fair share of credit for the team's success? This framework guarantees that credit is distributed fairly, meaning every channel receives credit proportional to its actual contribution, credits sum exactly to total conversions, and channels that contribute identically receive identical credit.

**Lens 2: Markov Chain — "What's Probable?"**

This framework models the customer journey as a sequence of states, asking: what is the probability of conversion given a particular sequence of channel interactions? It then measures each channel's importance by removing it from the model and observing how much the overall conversion rate drops. A channel whose removal causes a large drop is highly important. This framework uniquely captures the value of channel sequences — for example, it can identify that "Display → Search → Agent" is a far more effective path than "Search → Search → Search."

**Lens 3: Constrained Optimization — "What Best Fits the Data Given Business Knowledge?"**

This framework uses optimization mathematics (the Operations Research approach) to find the attribution weights that best explain the observed conversion patterns, subject to real-world business constraints. For example, it enforces that no single channel can receive more than a capped share of total credit, and that the agent channel receives at least a minimum floor of credit reflecting its known importance. The constraints encode business knowledge, and the model finds the best attribution within those bounds.

### Identity Resolution

Before attribution can work, the tool must solve a fundamental data integration challenge: recognizing that the same person who saw a TV ad, clicked a search result, visited the website, and called an agent is, in fact, one person — not four separate anonymous visitors.

The tool uses a three-tier identity resolution process. The first tier uses deterministic matching, linking records through exact matches on email, phone number, or tracking IDs. The second tier uses fuzzy matching, connecting records through approximate name and address matching. The third tier uses probabilistic matching, linking records through behavioral patterns like same agent, same day, same ZIP code.

This resolution process is what unlocks the agent insight. Without it, agent interactions exist in a separate data system with no connection to the digital journey that preceded them.

### Pre-Computed Results for Reliability

All attribution computations are pre-run as a batch process before the dashboard loads. When you interact with the tool, you're exploring pre-computed results — not waiting for models to run. This ensures the demo is fast, reliable, and consistent.

---

## 4. Navigating the Dashboard — Page-by-Page Guide

The dashboard contains eight primary pages plus a technical appendix. The pages are designed to be explored in order (following a narrative arc) or independently (for exploratory analysis).

### Page 1: Executive Summary

This is your landing page and the most important view in the entire tool. Four KPI cards across the top provide the headline numbers: agent attribution credit (model-based vs. last-click), branded search credit (model-based vs. last-click), projected budget efficiency improvement, and cross-model agreement score.

Below the cards, a grouped horizontal bar chart shows attribution credit by channel — with last-click (gray) and model-based (blue) shown side by side for every channel. The visual contrast between agent credit under last-click (tiny gray bar) versus model-based (large blue bar) is the central insight of the entire demo.

An insight callout at the bottom summarizes the finding in one sentence.

**What to look for:** The magnitude of the shift for agents (from roughly 3% to roughly 35%) and for branded search (from roughly 38% to roughly 11%). These two shifts are the core story.

### Page 2: Identity Resolution

This page explains *why* current attribution is wrong — the identity fragmentation problem.

The top section shows a conceptual journey for a synthetic customer ("Jennifer Morrison") who exists as eight separate records across eight different data systems. On the left, those records are disconnected. On the right, they're stitched together into a unified journey.

The middle section displays resolution metrics: how many records were processed, how many unique identities were resolved, the overall resolution rate, and how many agent journeys were recovered.

The bottom section shows a before/after comparison: how channel credits change when you go from unresolved (fragmented) data to fully resolved data. A dropdown lets you toggle between resolution tiers (exact matches only, exact plus fuzzy, all three tiers) to see how each tier incrementally improves attribution accuracy.

**What to look for:** The step-change in agent credit as resolution quality improves. This directly demonstrates the value of investing in data integration infrastructure.

### Page 3: Three-Model Comparison

This page builds credibility by showing that three independent mathematical frameworks converge on the same core insights.

Three side-by-side horizontal bar charts show credit allocation from each primary model (Time-Weighted Shapley, 2nd-Order Markov Chain, Constrained Optimization). All three use the same axis scale so visual comparison is immediate.

Below the charts, the Convergence/Divergence Map table is the most strategically valuable view in the tool. For each channel, it shows the credit from each model, a confidence zone indicator (HIGH if all three agree within 3 percentage points, MODERATE if two agree, LOW if all three diverge), and a consensus action recommendation (INCREASE, DECREASE, HOLD, or INVESTIGATE).

**What to look for:** Channels in the HIGH confidence zone with a clear consensus direction. These are where you can act immediately. Channels in the LOW confidence zone are flagged for further investigation — typically through incrementality testing.

### Page 4: Channel Deep-Dive

This page defaults to the agent channel but can be switched to any channel via a dropdown.

Three visualizations tell the channel story. First, a Credit Waterfall decomposes the credit shift from last-click through identity resolution through attribution modeling — showing exactly where each percentage point of credit comes from. Second, a Markov Transition Heatmap shows which channels have the highest probability of transitioning into the selected channel. For agents, this reveals that Agent Locator and Branded Search are the primary "feeder" channels. Third, a Funnel Comparison shows conversion rates for agent-touched journeys versus digital-only journeys at each funnel stage.

**What to look for:** The 2.8× conversion rate advantage of agent-touched journeys. This quantifies something Erie has always known intuitively but has never been able to prove with data.

### Page 5: Dual-Funnel Analysis

This page reveals that quote attribution and bind attribution tell different stories.

Side-by-side bar charts show quote-start attribution (left) and bind attribution (right). Center arrows highlight channels that gain or lose credit from the quote stage to the bind stage.

The key insight: nonbrand search drives approximately 18% of quotes but only 9% of binds. It generates research and shopping behavior but doesn't close deals. Agents show the opposite pattern — 20% of quote credit but 35% of bind credit. Agents don't generate quotes; they convert them.

**What to look for:** If Erie is optimizing marketing for quote-starts alone, it will over-fund search and under-fund agents. The dual-funnel view reveals which channels drive awareness and consideration versus which drive actual policy binds.

### Page 6: Budget Optimization

This page translates attribution insights into dollar recommendations.

The top section shows a budget recommendation table with current spend, optimizer-recommended spend from each of the three models, the dollar and percentage change, and a consensus direction. The middle section summarizes projected impact: additional binds, improved cost per bind, and efficiency gain percentage.

The bottom section is an interactive budget simulator. Sliders let you manually adjust spend by channel (total must remain constant) and see in real-time how predicted conversions change. The response curves behind this simulator model diminishing returns — the first dollar of spend on any channel is more effective than the millionth.

**What to look for:** The convergence across all three models on direction. If Shapley, Markov, and the OR model all say "increase agent support and decrease branded search," that's a high-confidence recommendation.

### Page 7: Scenario Explorer

This page allows strategic what-if analysis through seven pre-built scenarios.

Select a scenario from the left panel to see how key metrics change versus the baseline. For example, the "Full Call Tracking Deployment" scenario shows: with call tracking infrastructure ($50K/year), agent credit accuracy improves by 13 percentage points, enabling $280K in better budget allocation and approximately 340 additional binds per year. That's a 9.6× ROI on a measurement infrastructure investment.

Other scenarios explore questions like: What happens if digital self-service grows and agent involvement declines? What's the impact of cutting upper-funnel spend by 50%? How much does data quality affect optimization results?

**What to look for:** The call tracking ROI scenario is particularly actionable — it provides a concrete, dollar-denominated business case for a specific infrastructure investment.

### Page 8: Measurement Roadmap

This page connects the demo to production implementation with a four-stage maturity model.

The horizontal progression shows Erie's current state (last-click attribution), the capability this demo proves (multi-touch attribution), the next evolution (MTA plus Media Mix Modeling), and the full vision (complete triangulation with incrementality testing).

Below, three concrete next steps with timelines and investment levels provide a clear path forward: deploy call tracking (90 days), productionize the MTA pipeline (6 months), and add an MMM layer (3 months after MTA is live).

**What to look for:** Each stage delivers standalone value and builds on the previous one. The roadmap is designed so Erie can start capturing value immediately without committing to the full journey upfront.

---

## 5. Interpreting Key Findings

### Finding 1: Agents Are Your Most Valuable Channel

Under proper attribution, independent agents receive approximately 35% of bind credit — more than any other single channel. Under last-click attribution, they receive roughly 3%. The 32-percentage-point gap represents the largest single attribution error in the current measurement framework.

This doesn't mean agents are generating demand from scratch. Rather, agents are the critical conversion mechanism. Digital channels create awareness and consideration, drive prospects to research Erie, and lead them to find a local agent. The agent then converts that digitally-warmed prospect into a policyholder. Without the agent, the digital journey doesn't end in a bind. Without the digital journey, the agent doesn't have a prospect to convert.

**Implication:** Agent support marketing (co-op funds, lead generation for agents, tools and technology) is likely significantly under-funded relative to its contribution.

### Finding 2: Branded Search Is a Capture Channel, Not a Driver

Branded paid search drops from approximately 38% of last-click credit to approximately 11% under model-based attribution. This doesn't mean branded search is unimportant — it plays a critical role in capturing demand that other channels created. But it's currently receiving 3-4× more credit than it deserves, which likely leads to over-investment.

**Implication:** Consider reducing branded search spend by 15-20% and redirecting to upper-funnel channels (TV, display, social) and agent support. Monitor branded search impression share to ensure continued protection of brand terms.

### Finding 3: Upper-Funnel Channels Create the Demand Agents Convert

TV/radio, display, and paid social collectively increase from approximately 4% of last-click credit to approximately 28% under model-based attribution. These channels are invisible in digital analytics but essential for creating the awareness and consideration that starts the customer journey.

**Implication:** Resist pressure to cut upper-funnel budgets based on last-click performance metrics. The Markov transition analysis shows these channels have the highest probability of initiating journeys that eventually reach an agent.

### Finding 4: Digital and Agents Cooperate, Not Compete

The Markov transition heatmap reveals that digital channels and agents are not competing for credit — they cooperate in a sequential funnel. The highest-converting customer journey pattern is: awareness channel (TV, display, social) → informational search → brand search → agent locator → agent call → bind.

**Implication:** Marketing strategy should be optimized for the full funnel, not individual channel performance. Investments in digital channels should be evaluated partly on their ability to drive prospects toward agent interactions.

### Finding 5: Measurement Infrastructure Has Quantifiable ROI

The scenario analysis shows that deploying call tracking at approximately $50K per year enables $280K in better budget allocation and approximately 340 additional binds annually. This is a 9.6× return on investment for a measurement infrastructure improvement.

**Implication:** Call tracking deployment across all markets should be a near-term priority. The business case is clear and the implementation timeline is 90 days.

---

## 6. Using the Budget Simulator

The budget simulator on Page 6 lets you explore allocation changes in real-time. Here's how to get the most from it.

### How It Works

Each channel has a slider showing current spend level. When you move a slider, the total budget must remain constant ($5 million), so other channels adjust proportionally. The predicted conversion count updates in real-time based on response curve models that account for diminishing returns.

### Tips for Effective Use

Start with the optimizer's recommendation (pre-loaded for each model) and then make adjustments. Watch the "Predicted Binds" counter at the top — it shows the net impact of your changes. Pay attention to channels near their saturation ceiling — adding more spend to a nearly-saturated channel yields minimal additional conversions. The response curve visualization (below the sliders) shows each channel's diminishing returns curve with your current spend level marked.

### Understanding Constraints

The optimizer respects real-world constraints. No channel can shift more than 20% from its current level in a single reallocation cycle (reflecting organizational change capacity). Agent support must remain at least 2% of total budget. TV must remain at least 10% of total budget for brand presence. Nonbrand search is capped at 25% to prevent over-concentration.

These constraints are based on domain expertise and can be adjusted in production. The shadow price analysis (available in the technical appendix) shows exactly how much each constraint costs or saves in forgone conversions.

---

## 7. Scenario Explorer — What-If Analysis

The scenario explorer pre-computes optimization results for seven strategic scenarios. Each scenario modifies specific assumptions and shows the resulting impact on budget allocation and projected outcomes.

### Available Scenarios

**Baseline:** Current state — no changes. This is the reference point for all comparisons.

**Digital Transformation:** What if digital self-service grows and agent involvement drops from 87% to 65%? This scenario explores how attribution and budget allocation should evolve if younger demographics increasingly prefer to bind online.

**No Call Tracking:** What if call tracking is removed entirely? This scenario quantifies the cost of NOT having infrastructure that bridges digital and agent interactions. Agent credit drops to approximately 22% and the optimizer significantly under-funds agent support.

**Full Call Tracking:** What if call tracking coverage increases from 45% to 90%? Agent credit rises to approximately 38%, enabling more accurate budget optimization. The delta between this and "No Call Tracking" is the ROI of the infrastructure investment.

**Cut Upper Funnel:** What if TV and display budgets are cut by 50%? This scenario tests the counter-intuitive finding that upper-funnel channels matter. The result: fewer prospects entering the journey, fewer reaching agents, and fewer binds — even though the cut channels appeared "ineffective" under last-click.

**Agent Decline:** What if agent involvement drops to 72% due to demographic shifts? Budget allocation should shift more toward digital channels, but agents remain the highest-value conversion mechanism. The optimizer adjusts gradually rather than abandoning the agent model.

**Clean Data Baseline:** What if all data quality issues were resolved? This scenario shows the incremental value of data quality investment by comparing results with and without dirty data (duplicates, missing fields, timestamp errors).

---

## 8. Frequently Asked Questions

**"Is this based on real Erie data?"**

No. This is a capability demo built on realistic synthetic data calibrated to Erie's publicly known business characteristics — 12-state footprint, 100% independent agent model, estimated marketing budget, and channel mix. The underlying analytical framework is identical to what would be deployed on real data. The synthetic approach eliminates privacy concerns and allows Erie's team to evaluate the methodology before sharing any proprietary information.

**"Why should we trust three models more than one?"**

Each model makes different mathematical assumptions. Shapley Values assume cooperative game theory axioms. Markov Chains model sequential transitions. Constrained Optimization encodes business knowledge directly. If all three converge on the same insight, that insight is robust to methodological choice. If they disagree, that disagreement itself is informative — it tells you where additional investigation (like incrementality testing) is needed.

**"How is this different from Google Analytics 4's data-driven attribution?"**

GA4's DDA uses a Shapley-like approach but only operates on digital touchpoints visible to GA4. It cannot see agent interactions, call center activity, direct mail, or TV. For Erie, GA4 attributes credit across approximately 30-40% of the actual touchpoints. This tool covers the full journey across all channels — digital and offline — which is precisely the problem for agent-distributed carriers.

**"How is this different from off-the-shelf attribution vendors?"**

SaaS attribution tools (like Attribution App or Rockerbox) are designed for e-commerce and direct-to-consumer brands. They ingest digital data via pixels and UTM tracking but have no concept of agent interactions, policy administration systems, or insurance-specific funnels. They cannot model a quote-to-bind funnel. This solution is purpose-built for P&C insurance.

**"What would change if we used real data instead of synthetic?"**

The absolute credit percentages would change (our synthetic calibrations are estimates), but the directional findings — agents being under-credited, search being over-credited, upper-funnel channels mattering more than last-click suggests — are robust to parameter variation. The sensitivity analysis (available in the technical appendix) confirms this.

**"How often would attribution need to be refreshed in production?"**

Quarterly refresh is recommended. Marketing mix changes gradually, and the models need at least 3-6 months of data to produce stable estimates. Major changes (new channel launch, significant budget shift, entering new states) should trigger an ad-hoc refresh.

**"What's the minimum data infrastructure requirement for production?"**

The critical enablers are a way to link digital sessions to CRM contacts (typically via form fills or logged-in state), call tracking for the web-to-agent bridge, and access to the Policy Administration System for bind outcomes. Identity resolution handles the rest.

---

## 9. From Demo to Production — The Path Forward

### Stage 1: Measurement Infrastructure (0-3 Months)

Deploy call tracking across all Erie markets using a provider like Invoca or CallRail. This is the single highest-ROI infrastructure investment identified in the analysis (9.6× projected return). Implementation timeline: 90 days. Estimated annual cost: approximately $50K.

### Stage 2: MTA Production Pipeline (3-9 Months)

Productionize the multi-touch attribution framework demonstrated here. This involves integrating with Erie's actual data systems (GA4, ad platforms, CRM, AMS, PAS, call tracking, ESP), building the identity resolution pipeline on real data, deploying attribution models with quarterly refresh, and creating automated reporting dashboards.

This stage requires a dedicated team of 3-4 people for approximately 6 months. The output is an ongoing, automated attribution system that replaces last-click reporting.

### Stage 3: Add Media Mix Modeling (9-12 Months)

Layer a Bayesian Media Mix Model (such as Google Meridian or Meta Robyn) on top of the MTA pipeline. MMM uses aggregate geographic data and handles channels that MTA struggles with (brand awareness effects of TV, long-term impacts of direct mail). The combination of MTA (tactical, fast-feedback) and MMM (strategic, geo-validated) provides stronger evidence than either alone.

### Stage 4: Full Triangulation (12+ Months)

Add incrementality experiments (geo holdouts, matched market tests) to calibrate and validate both MTA and MMM. This creates a closed-loop measurement system where each methodology validates the others: MTA provides tactical channel-level guidance, MMM provides strategic budget allocation, and incrementality experiments provide causal ground truth.

Each stage delivers standalone value and builds on the previous one.

---

## 10. Glossary of Terms

**Agent Involvement Rate:** The percentage of converting journeys (binds) that include at least one interaction with an independent agent. For Erie, estimated at approximately 87%.

**Attribution Credit:** The fractional share of a conversion assigned to a marketing channel. Under multi-touch attribution, credit is distributed across all channels that contributed, rather than giving 100% to a single touchpoint.

**Attribution Window:** The maximum number of days before a conversion that touchpoints are considered. Default: 30 days. Touchpoints outside this window are excluded from attribution.

**Bind:** A policy being officially issued and taking effect. This is the primary conversion event in the demo — the ultimate measure of marketing effectiveness.

**Coalition (Shapley):** A subset of marketing channels. Shapley Values evaluate every possible coalition to determine each channel's marginal contribution.

**Confidence Zone:** A per-channel classification based on cross-model agreement. HIGH means all three models agree within 3 percentage points. MODERATE means two agree and one diverges. LOW means all three disagree significantly.

**Convergence-Weighted Recommendation:** A budget recommendation that weights its confidence by the degree of agreement across the three attribution models. High-confidence channels get concrete dollar recommendations; low-confidence channels get flagged for investigation.

**Cost Per Bind:** Total marketing spend divided by total policy binds. The primary efficiency metric for marketing budget evaluation.

**Decision State:** The stage of a prospect's insurance shopping journey: Awareness (knows Erie exists), Consideration (actively researching), Intent (comparing quotes), Action (ready to commit).

**Dual-Funnel Analysis:** Comparing attribution for two different conversion events (quote-start and bind) to reveal which channels drive research versus which close deals.

**Identity Resolution:** The process of linking records from multiple data systems (GA4, CRM, AMS, PAS, call tracking) to recognize that they belong to the same person.

**Jensen-Shannon Divergence (JSD):** A statistical measure of how different two probability distributions are. Used to compare credit share distributions between models. Lower JSD means more agreement.

**Last-Click Attribution:** A simplistic model that gives 100% of conversion credit to the final touchpoint before conversion. The default in most digital analytics platforms.

**Markov Chain:** A mathematical model that describes a sequence of events where the probability of each event depends on the state reached in the previous event(s).

**MILP (Mixed-Integer Linear Program):** A mathematical optimization technique that finds the best budget allocation subject to constraints, with some variables being continuous (spend amounts) and some binary (whether a channel is active).

**Multi-Touch Attribution (MTA):** Any attribution method that distributes credit across multiple touchpoints in the customer journey, rather than giving all credit to a single touchpoint.

**Quote-Start:** The event of a prospect initiating an insurance quote (entering their information into Erie's quoting system). An intermediate conversion event that precedes bind.

**Removal Effect (Markov):** The change in overall conversion probability when a specific channel is hypothetically removed from the system. Channels with large removal effects are most important.

**Response Curve:** A mathematical function modeling how conversions change as spend on a channel increases. Typically shows diminishing returns: the first dollar of spend is more effective than the millionth.

**Saturation Factor:** A measure of how close a channel's current spend is to its diminishing returns ceiling. A saturation factor of 0.80 means the channel is 80% saturated — additional spend yields minimal incremental conversions.

**Shadow Price:** A measure of how much the optimal solution would improve if a specific constraint were relaxed by one unit. For example, the shadow price on the agent support floor tells you how many additional conversions you could achieve if you lowered the minimum agent spend requirement.

**Shapley Values:** A game-theoretic concept that fairly distributes credit among "players" (channels) based on their marginal contribution across all possible combinations.

**Spearman Rank Correlation (ρ):** A statistical measure of how similarly two models rank channels. ρ = 1.0 means perfect agreement on ranks; ρ > 0.70 indicates strong agreement.

**Touchpoint:** A single interaction between a prospect and a marketing channel — a click, an impression, a phone call, an office visit, a mailed piece, an email open.

**Unified Journey:** The complete, ordered sequence of all touchpoints for a single prospect, assembled from multiple data systems after identity resolution.

---

## 11. Appendix: Methodology Summary for Business Users

### How Credit Is Calculated — The 30-Second Version

Each attribution model examines the approximately 25,000 customer journeys in the dataset and asks: for each journey that ended in a policy bind, how much did each marketing channel contribute to making that bind happen?

Shapley Values approach this by examining every possible combination of channels and measuring each channel's marginal contribution — what would be lost if that channel weren't present? Markov Chains approach this by modeling the sequential flow of customers through channels and measuring what happens when you remove a channel from the flow. Constrained Optimization approaches this by finding the channel weights that best predict which journeys convert, subject to business-informed constraints.

Each model produces a credit share for each of the 13 channels. These credits sum to 100% (or equivalently, to the total number of conversions). The credit share represents each channel's attributed contribution to the overall conversion outcome.

### How the Budget Optimizer Works — The 30-Second Version

Once we know each channel's true contribution, we can ask: what's the optimal way to allocate the budget? The optimizer uses response curves (which model how conversions change as spend increases) and works within business constraints (minimum channel spends, maximum reallocation rates, agent support floors) to find the allocation that maximizes total predicted binds.

Because the optimizer runs on each of the three models' credit vectors separately, it produces three independent recommendations. Where all three agree on direction, the recommendation is high-confidence. Where they disagree, the tool flags the channel for further investigation rather than making an uncertain recommendation.

### Data in This Demo

This demo uses synthetic data — not real Erie customer records. The data is generated by a behavioral simulation engine calibrated to Erie's publicly known business characteristics. Twenty-five thousand synthetic prospects are simulated through realistic insurance shopping journeys, producing approximately 5,000 policy binds. The journey patterns, channel interactions, conversion rates, and data fragmentation scenarios are designed to match what Erie's team would recognize from their own business.

This approach eliminates privacy concerns, allows full transparency of methodology, and enables Erie's team to evaluate the analytical framework before any real data integration.

---

*This document is designed to accompany a guided walkthrough of the MCA Intelligence dashboard. For technical implementation details, mathematical formulations, and data architecture specifications, refer to the companion Technical Document.*