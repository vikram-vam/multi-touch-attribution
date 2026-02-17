"""
Page 8 — Technical Appendix (EP-8)
Model methodology, data schema, and references.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

from app.components.components import section_header
from app.theme import COLORS


def layout():
    return html.Div([
        section_header(
            "Technical Appendix",
            "Methodology, model details, data schema, and academic references"
        ),

        # Model Tier Table
        html.Div([
            html.H3("Attribution Model Tiers", className="chart-title"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Tier"), html.Th("Models"), html.Th("Methodology"), html.Th("Key Properties"),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("1 — Rule-Based"),
                        html.Td("First-Touch, Last-Touch, Linear, Time-Decay, Position-Based"),
                        html.Td("Heuristic allocation rules"),
                        html.Td("Simple, interpretable, biased toward certain positions"),
                    ]),
                    html.Tr([
                        html.Td("2 — Game-Theoretic"),
                        html.Td("Shapley Value, CASV"),
                        html.Td("Cooperative game theory: credit = marginal contribution across all coalitions"),
                        html.Td("Satisfies efficiency, symmetry, null-player, additivity axioms"),
                    ]),
                    html.Tr([
                        html.Td("3 — Probabilistic"),
                        html.Td("Markov Chain (1st, 2nd, 3rd order)"),
                        html.Td("Transition probability matrix + removal effect"),
                        html.Td("Captures sequential patterns, counterfactual reasoning"),
                    ]),
                    html.Tr([
                        html.Td("4 — Statistical"),
                        html.Td("Logistic Regression, Survival/Hazard"),
                        html.Td("Regression coefficients / hazard ratios as credit proxies"),
                        html.Td("Confidence intervals, feature importance"),
                    ]),
                    html.Tr([
                        html.Td("5 — Deep Learning"),
                        html.Td("LSTM, DARNN, Transformer, CausalMTA"),
                        html.Td("Attention-based sequence modeling"),
                        html.Td("Learns complex non-linear interactions, SOTA accuracy"),
                    ]),
                    html.Tr([
                        html.Td("6 — Meta-Model"),
                        html.Td("Weighted Ensemble"),
                        html.Td("Shapley (45%) + Markov (30%) + Logistic (25%)"),
                        html.Td("Triangulates across paradigms, most robust"),
                    ]),
                ]),
            ], bordered=True, dark=True, hover=True, responsive=True, striped=True,
               className="mb-4"),
        ], className="chart-container"),

        # Data Schema
        html.Div([
            html.H3("Data Schema", className="chart-title"),
            dbc.Accordion([
                dbc.AccordionItem([
                    html.Code("touchpoints.parquet", style={"color": COLORS["highlight"]}),
                    html.P("touchpoint_id, persistent_id, fragment_id, channel_id, "
                           "sub_channel, touch_type, event_timestamp, funnel_stage, "
                           "viewability_pct, dwell_time_seconds, geo_state, device_type, "
                           "is_organic",
                           className="text-muted mt-2"),
                ], title="touchpoints.parquet"),
                dbc.AccordionItem([
                    html.Code("journeys.parquet", style={"color": COLORS["highlight"]}),
                    html.P("journey_id, persistent_id, channel_path, touchpoint_count, "
                           "distinct_channel_count, journey_start, journey_end, "
                           "journey_duration_days, first_touch_channel, last_touch_channel, "
                           "is_converting, conversion_value, has_agent_touch, agent_touch_position",
                           className="text-muted mt-2"),
                ], title="journeys.parquet"),
                dbc.AccordionItem([
                    html.Code("attribution_results.parquet", style={"color": COLORS["highlight"]}),
                    html.P("model_name, model_tier, channel_id, attributed_conversions, "
                           "attribution_pct, rank",
                           className="text-muted mt-2"),
                ], title="attribution_results.parquet"),
            ], start_collapsed=True, className="mb-4"),
        ], className="chart-container"),

        # References
        html.Div([
            html.H3("Academic References", className="chart-title"),
            html.Ul([
                html.Li("Shapley, L.S. (1953). A Value for n-Person Games. Contributions to the Theory of Games."),
                html.Li("Anderl, E. et al. (2016). Mapping the Customer Journey: Lessons Learned from Graph-Based Online Attribution Modeling. IJRM."),
                html.Li("Dalessandro, B. et al. (2012). Causally Motivated Attribution for Online Advertising. ADKDD."),
                html.Li("Shao, X. & Li, L. (2011). Data-driven Multi-touch Attribution Models. KDD."),
                html.Li("Ren, K. et al. (2018). Learning Multi-touch Conversion Attribution with Dual-attention Mechanisms for Online Advertising. CIKM."),
                html.Li("Du, R. et al. (2019). CausalMTA: Eliminating Self-selection and Dependence Biases in Multi-touchpoint Attribution. arXiv."),
            ], style={"color": COLORS["text_secondary"], "lineHeight": "2"}),
        ], className="chart-container"),

        # Erie Context
        html.Div([
            html.H3("Erie Insurance Context", className="chart-title"),
            html.P([
                "Erie Insurance Group operates across 12 states and D.C. through a network of ",
                html.Strong("~14,000 independent agents"),
                ". Founded in 1925, Erie consistently ranks in the top 15 U.S. P&C insurers by DWP. ",
                "This demo uses synthetic data calibrated to Erie's public financials, ",
                "channel mix, and agent distribution patterns.",
            ], style={"color": COLORS["text_secondary"], "lineHeight": "1.8"}),
        ], className="chart-container"),

    ], className="page-container")
