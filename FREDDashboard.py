"""
FRED Economic Monitor — Institutional Dashboard
Run:  python FREDDashboard.py
Then open http://127.0.0.1:8050 in a browser.

Data is loaded from /data at startup and cached in memory.
To refresh data, run FREDDownloader.py and restart this script.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from _classes.data_loader import DataLoader
from _classes.sql_storage import SQLStorage
from _classes.chart_factory import line_chart, area_chart, multi_line_chart, bar_change_chart
from _classes.risk_engine import RiskEngine
from _classes.series_registry import REGISTRY, CATEGORY_LABELS
from _classes.constants import (
    URLS, CHART, TYPOGRAPHY, DASH,
    PALETTE as C,
    RISK_STYLE,
    STYLE_HEADER, STYLE_PAGE, STYLE_TAB, STYLE_TAB_SELECTED, STYLE_CARD, STYLE_SECTION_LABEL,
)

EXTERNAL = [dbc.themes.BOOTSTRAP, URLS.google_fonts]

# ── Startup: load data ────────────────────────────────────────────────────

_sql = SQLStorage.from_config()
dl = DataLoader(sql=_sql)
re = RiskEngine(dl)
crisis_dims = re.crisis_dimensions()


# ── Component Builders ────────────────────────────────────────────────────

def kpi_card(series_id: str) -> html.Div:
    """Single KPI card with risk color coding."""
    risk, display, _ = re.score(series_id)
    style = RISK_STYLE.get(risk, RISK_STYLE["neutral"])
    meta = REGISTRY.get(series_id, {})
    basis = meta.get("risk_basis", "level")
    sublabel = {"yoy": "Year-over-Year", "mom_change": "Month-over-Month", "level": "Current"}.get(basis, "")

    _, as_of_date = dl.get_latest(series_id)
    as_of = as_of_date.strftime(DASH.date_display_fmt) if as_of_date is not None else ""

    return html.Div([
        html.Div(meta.get("short_name", series_id), style={
            "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.06em",
            "color": style["text"], "textTransform": "uppercase", "marginBottom": "6px",
        }),
        html.Div(display, style={
            "fontSize": "28px", "fontWeight": "600", "color": style["text"],
            "lineHeight": "1.1", "marginBottom": "4px",
        }),
        html.Div(sublabel, style={"fontSize": "11px", "color": style["text"], "opacity": "0.75"}),
        html.Div(style={"height": "1px", "backgroundColor": style["border"], "margin": "10px 0"}),
        html.Div([
            html.Span(style["label"], style={"fontWeight": "500"}),
            html.Span(f"as of {as_of}", style={"marginLeft": "auto", "opacity": "0.65"}),
        ], style={
            "display": "flex", "alignItems": "center",
            "fontSize": "11px", "color": style["text"],
        }),
    ], style={
        "backgroundColor": style["bg"],
        "border": f"1px solid {style['border']}",
        "borderLeft": f"4px solid {style['border']}",
        "borderRadius": "6px",
        "padding": "16px",
        "minHeight": "110px",
    })


def section_card(*children, title: str | None = None) -> html.Div:
    """Wrapped content card with optional title."""
    content = []
    if title:
        content.append(html.H6(title, style={
            "fontWeight": "600", "color": "#2d3748", "marginBottom": "16px",
            "fontSize": "14px", "borderBottom": "1px solid #edf2f7", "paddingBottom": "10px",
        }))
    content.extend(children)
    return html.Div(content, style=STYLE_CARD)


def crisis_dim_card(name: str, dim: dict) -> dbc.Col:
    """One of the five Crisis Watch dimension scorecards."""
    risk = dim["score"]
    style = RISK_STYLE[risk]
    components = dim["components"]

    comp_rows = [
        html.Div([
            html.Span(label, style={"fontSize": "11px", "color": C["slate"]}),
            html.Span(display, style={
                "fontSize": "11px", "fontWeight": "600",
                "color": RISK_STYLE[r]["text"], "marginLeft": "auto",
            }),
        ], style={"display": "flex", "marginBottom": "4px"})
        for label, r, display in components
    ]

    n_dims = DASH.n_crisis_dims
    return dbc.Col(html.Div([
        html.Div(name, style={
            "fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.06em",
            "textTransform": "uppercase", "color": style["text"], "marginBottom": "8px",
        }),
        html.Div(style["label"], style={
            "fontSize": "22px", "fontWeight": "700",
            "color": style["text"], "marginBottom": "12px",
        }),
        html.Div(comp_rows),
    ], style={
        "backgroundColor": style["bg"],
        "border": f"1px solid {style['border']}",
        "borderLeft": f"4px solid {style['border']}",
        "borderRadius": "6px",
        "padding": "16px",
    }), md=12 // n_dims, sm=6, xs=12)


def prose(*paragraphs: str, header: str | None = None) -> html.Div:
    """Styled prose block for educational content."""
    children = []
    if header:
        children.append(html.H6(header, style={
            "fontWeight": "600", "color": "#2d3748", "fontSize": "14px", "marginBottom": "10px",
        }))
    for p in paragraphs:
        children.append(html.P(p, style={
            "fontSize": "13px", "lineHeight": "1.75", "color": C["slate"], "marginBottom": "12px",
        }))
    return html.Div(children, style={"padding": "4px 0"})


def graph(figure, height: int = CHART.graph_height) -> dcc.Graph:
    return dcc.Graph(figure=figure, config={"displayModeBar": False},
                     style={"height": f"{height}px"})


# ── Tab Content ───────────────────────────────────────────────────────────

def summary_tab() -> html.Div:
    available_ids = [sid for sid in REGISTRY if sid in dl.available]

    # Group by category for the KPI grid
    from _classes.series_registry import CATEGORY_ORDER
    rows = []
    for cat in CATEGORY_ORDER:
        ids = [s for s in available_ids if REGISTRY[s]["category"] == cat]
        if not ids:
            continue
        rows.append(html.Div(CATEGORY_LABELS.get(cat, cat), style=STYLE_SECTION_LABEL))
        cols = [dbc.Col(kpi_card(sid), md=3, sm=6, xs=12) for sid in ids]
        rows.append(dbc.Row(cols, className="g-3 mb-2"))

    return html.Div([
        section_card(
            prose(
                f"Showing {len(available_ids)} series from {dl.data_dir}. "
                "Color coding reflects current readings relative to historical risk thresholds: "
                "Normal (green), Elevated (yellow), Stressed (red). "
                "No single indicator is deterministic — context and combinations matter."
            ),
            title="Data Coverage"
        ),
        html.Div(rows),
    ], style={"padding": "24px"})


def inflation_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "CPIAUCSL", yoy=True, lookback_years=25,
                                 threshold_green=2.5, threshold_red=4.5,
                                 title="Headline CPI — Year-over-Year %")),
            ), md=6),
            dbc.Col(section_card(
                graph(area_chart(dl, "CPILFESL", yoy=True, lookback_years=25,
                                 threshold_green=2.5, threshold_red=4.0,
                                 color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                                 title="Core CPI (ex Food & Energy) — Year-over-Year %")),
            ), md=6),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(multi_line_chart(dl,
                    [("CPIAUCSL", C["blue"]), ("CPILFESL", C["teal"])],
                    title="Headline vs. Core CPI — YoY % (10 Years)",
                    lookback_years=10, yoy=True,
                )),
            ), md=8),
            dbc.Col(section_card(
                graph(line_chart(dl, "FPCPITOTLZGUSA", lookback_years=30,
                                 title="Annual Inflation Rate (World Bank)")),
                prose(
                    "Annual CPI inflation published by the World Bank. "
                    "The Fed's long-run target is 2%. Sustained readings above 3% "
                    "have historically required monetary tightening that slows growth.",
                ),
            ), md=4),
        ], className="g-0"),
    ], style={"padding": "24px"})


def money_supply_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(section_card(
                graph(multi_line_chart(dl,
                    [("M1SL", C["blue"]), ("M2SL", C["teal"])],
                    title="M1 & M2 Money Supply — Level (Billions $)",
                    lookback_years=20,
                )),
            ), md=8),
            dbc.Col(section_card(
                prose(
                    "M1 is the narrowest measure of money: currency and demand deposits. "
                    "M2 adds savings accounts, retail money market funds, and small CDs. "
                    "The M1/M2 gap widened significantly in 2020 following Fed asset purchases.",
                    "Rapid M2 growth often precedes inflation with a 12–18 month lag. "
                    "Real M2 contraction — M2 growing slower than inflation — typically signals "
                    "tightening monetary conditions.",
                ),
            ), md=4),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "M2SL", yoy=True, lookback_years=30,
                                 title="M2 Growth — Year-over-Year %")),
            ), md=6),
            dbc.Col(section_card(
                graph(area_chart(dl, "M2REAL", yoy=True, lookback_years=20,
                                 color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                                 title="Real M2 Growth — Year-over-Year %")),
            ), md=6),
        ], className="g-0"),
    ], style={"padding": "24px"})


def labor_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "UNRATE", lookback_years=30,
                                 threshold_green=4.5, threshold_red=6.0,
                                 color=C["slate"], fill_color="rgba(74,85,104,0.08)",
                                 title="Unemployment Rate %")),
            ), md=6),
            dbc.Col(section_card(
                graph(bar_change_chart(dl, "PAYEMS", lookback_years=5,
                                       title="Nonfarm Payrolls — Monthly Change (Thousands)")),
            ), md=6),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(line_chart(dl, "CIVPART", lookback_years=30, color=C["teal"],
                                 title="Labor Force Participation Rate %")),
            ), md=6),
            dbc.Col(section_card(
                graph(multi_line_chart(dl,
                    [("EMRATIO", C["blue"]), ("CIVPART", C["teal"])],
                    title="Employment-Population Ratio vs. Participation Rate %",
                    lookback_years=30,
                )),
            ), md=6),
        ], className="g-0"),
    ], style={"padding": "24px"})


def markets_rates_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "VIXCLS", lookback_years=20,
                                 threshold_green=20, threshold_red=30,
                                 color="#744210", fill_color="rgba(116,66,16,0.08)",
                                 title="CBOE Volatility Index (VIX)")),
            ), md=6),
            dbc.Col(section_card(
                graph(multi_line_chart(dl,
                    [("DGS10", C["blue"]), ("GS10", C["teal"])],
                    title="10-Year Treasury Yield — Daily vs. Monthly Average %",
                    lookback_years=20,
                )),
            ), md=6),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "SP500_CAPE", lookback_years=None,
                                 threshold_green=20, threshold_red=30,
                                 color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                                 title="S&P 500 CAPE (Shiller P/E) — Full History")),
                prose(
                    "The Cyclically Adjusted P/E ratio smooths earnings over a rolling 10-year "
                    "real average, removing business cycle distortions. The historical mean is "
                    "approximately 17. Readings above 30 — sustained through 1929, 2000, and "
                    "2021 — have historically preceded decade-long below-average real returns. "
                    "CAPE does not predict the timing or magnitude of corrections.",
                ),
            ), md=8),
            dbc.Col(section_card(
                graph(area_chart(dl, "SP500_PE", lookback_years=40,
                                 threshold_green=20, threshold_red=27,
                                 color=C["slate"], fill_color="rgba(74,85,104,0.08)",
                                 title="S&P 500 Trailing P/E (40 Years)")),
                prose(
                    "Trailing twelve-month P/E uses reported earnings and is more volatile than "
                    "CAPE — it collapses during earnings recessions and can look artificially "
                    "cheap when earnings are temporarily depressed. Use alongside CAPE for "
                    "a more complete valuation picture.",
                ),
            ), md=4),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "DRCRELEXFACBS", lookback_years=25,
                                 threshold_green=1.5, threshold_red=4.0,
                                 color=C["red"], fill_color="rgba(155,44,44,0.09)",
                                 title="CRE Loan Delinquency Rate %")),
                prose(
                    "Commercial real estate delinquency rates are a quarterly, lagging indicator "
                    "of credit stress in rate-sensitive lending. Peaks occurred at ~8.5% in "
                    "2010 (post-GFC) and ~3.5% in 2020 (COVID). Rising rates combined with "
                    "office vacancy trends and tighter lending standards make this series "
                    "particularly relevant in the current cycle.",
                ),
            ), md=6),
            dbc.Col(section_card(
                graph(line_chart(dl, "GS10", lookback_years=40, color=C["blue"],
                                 title="10-Year Treasury Yield — Long History %")),
                prose(
                    "The secular decline in long-term yields from ~16% in 1981 to near 0% "
                    "in 2020 supported rising asset valuations and reduced debt service costs. "
                    "The reversal since 2022 has structural implications for real estate "
                    "valuations, corporate refinancing, and government debt service."
                ),
            ), md=6),
        ], className="g-0"),
    ], style={"padding": "24px"})


def fiscal_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "GFDEGDQ188S", lookback_years=40,
                                 threshold_green=80, threshold_red=120,
                                 color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                                 title="Federal Debt as % of GDP")),
            ), md=8),
            dbc.Col(section_card(
                prose(
                    "Federal debt as a share of GDP rose sharply during World War II (peaking ~106%), "
                    "declined through the postwar expansion, then began climbing again from the 1980s. "
                    "The 2008 financial crisis and the 2020 pandemic triggered the largest peacetime "
                    "increases on record.",
                    "At elevated debt/GDP ratios, interest rate increases have a compounding effect: "
                    "as debt rolls over at higher rates, the share of the budget consumed by interest "
                    "payments expands, crowding out discretionary fiscal space. The IMF identifies "
                    "debt/GDP above 90–100% as a threshold where fiscal multipliers weaken.",
                ),
            ), md=4),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(multi_line_chart(dl,
                    [("M2SL", C["blue"]), ("GFDEGDQ188S", C["amber"])],
                    title="M2 Money Supply (B$) & Federal Debt/GDP — Long-Run Trends",
                    lookback_years=40,
                )),
                prose(
                    "Note: M2 is shown in billions of dollars (left scale concept); Debt/GDP "
                    "is a percentage. The two series share a chart to illustrate the co-movement "
                    "of monetary and fiscal expansion over the past four decades."
                ),
            ), md=12),
        ], className="g-0"),
    ], style={"padding": "24px"})


def crisis_watch_tab() -> html.Div:
    # ── Dimension scorecards ───────────────────────────────────────────────
    dim_cards = dbc.Row(
        [crisis_dim_card(name, dim) for name, dim in crisis_dims.items()],
        className="g-3 mb-4",
    )

    # Overall risk count
    counts = {"red": 0, "yellow": 0, "green": 0, "neutral": 0}
    for dim in crisis_dims.values():
        counts[dim["score"]] += 1

    overall_note = (
        f"{counts['red']} dimension(s) stressed, {counts['yellow']} elevated, "
        f"{counts['green']} within normal range."
    )

    return html.Div([
        # ── Header box ────────────────────────────────────────────────────
        section_card(
            html.Div([
                html.H5("Crisis Watch", style={
                    "fontWeight": "700", "color": C["header_bg"], "marginBottom": "6px",
                }),
                html.P(
                    "This page synthesizes multiple indicators into five structural dimensions. "
                    "It is designed to surface gradual deterioration that individual series may "
                    "obscure. Color coding is descriptive, not predictive — elevated readings "
                    "indicate conditions that have historically preceded stress, not guaranteed outcomes.",
                    style={"fontSize": "13px", "color": C["slate"], "marginBottom": "8px", "lineHeight": "1.7"},
                ),
                html.Div(overall_note, style={
                    "fontSize": "12px", "fontWeight": "500", "color": "#718096",
                    "borderTop": "1px solid #edf2f7", "paddingTop": "10px",
                }),
            ]),
        ),

        # ── Scorecards ────────────────────────────────────────────────────
        dim_cards,

        # ── Dimension detail charts ────────────────────────────────────────
        section_card(
            dbc.Row([
                dbc.Col(graph(area_chart(dl, "DGS10", lookback_years=10,
                    title="10-Year Treasury Yield %"), height=CHART.graph_height_small), md=6),
                dbc.Col(graph(area_chart(dl, "DRCRELEXFACBS", lookback_years=20,
                    color=C["red"], fill_color="rgba(155,44,44,0.09)",
                    threshold_green=1.5, threshold_red=4.0,
                    title="CRE Delinquency Rate %"), height=CHART.graph_height_small), md=6),
            ], className="g-2"),
            dbc.Row([
                dbc.Col(graph(area_chart(dl, "CPILFESL", yoy=True, lookback_years=10,
                    color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                    threshold_green=2.5, threshold_red=4.0,
                    title="Core CPI YoY %"), height=CHART.graph_height_small), md=6),
                dbc.Col(graph(area_chart(dl, "UNRATE", lookback_years=20,
                    threshold_green=4.5, threshold_red=6.0,
                    color=C["slate"], fill_color="rgba(74,85,104,0.08)",
                    title="Unemployment Rate %"), height=CHART.graph_height_small), md=6),
            ], className="g-2"),
            title="Indicator Detail"
        ),

        # ── Educational: Structural vs Acute ──────────────────────────────
        section_card(
            dbc.Row([
                dbc.Col(prose(
                    "Structural deterioration develops over years: rising debt loads, persistent "
                    "above-target inflation, labor market erosion, or deteriorating credit quality "
                    "in key sectors. These imbalances accumulate quietly, often masked by growth "
                    "momentum or asset price appreciation, until they constrain policy options.",
                    "Acute crises arrive suddenly — a bank run, a liquidity freeze, a forced "
                    "delevering event. They compress structural damage into days or weeks. "
                    "Crucially, acute crises almost always land on pre-existing structural "
                    "vulnerabilities that determine the severity of the impact.",
                    header="Structural Deterioration vs. Acute Crisis",
                ), md=6),
                dbc.Col(prose(
                    "The 2008 Global Financial Crisis illustrates the distinction: CRE and "
                    "residential delinquencies climbed for 18+ months before Lehman Brothers' "
                    "collapse compressed the acute phase into a single weekend. The structural "
                    "fragility had been building; the trigger was catalytic, not causal.",
                    "The practical implication for monitoring: structural risks rarely announce "
                    "themselves clearly. They appear in gradual data drift — a steady rise in "
                    "delinquency rates, a persistent core inflation reading, a multi-quarter "
                    "trend in labor market flows — that markets often discount until the pattern "
                    "becomes impossible to ignore.",
                ), md=6),
            ], className="g-4"),
        ),

        # ── Educational: Why markets lag ──────────────────────────────────
        section_card(
            dbc.Row([
                dbc.Col(prose(
                    "Markets are structurally biased toward optimism during deteriorating conditions "
                    "for several compounding reasons:",
                    "Momentum effects cause rising asset prices to generate self-confirming "
                    "expectations. Complexity obscures the interconnections between institutions "
                    "and instruments that determine systemic exposure. Data lags mean that "
                    "economic releases report the past, not the present. And incentive "
                    "misalignment — among analysts, executives, and policymakers — creates "
                    "pressure to project stability even when internal signals are concerning.",
                    header="Why Warning Signs Are Often Discounted",
                ), md=6),
                dbc.Col(prose(
                    "Historically informative examples: The U.S. yield curve inverted in August "
                    "2006 — 16 months before the December 2007 recession began. CRE delinquencies "
                    "started rising in Q4 2006. Both signals were visible and actively debated; "
                    "consensus remained constructive well into 2007.",
                    "Not every warning signal triggers a crisis. Many inversions resolve without "
                    "recession. Many delinquency spikes are contained by policy response or "
                    "improved conditions. The relevant question is not 'will this cause a crisis?' "
                    "but 'does the combination of signals suggest the system has less buffer "
                    "than it appears to have?'",
                ), md=6),
            ], className="g-4"),
        ),

        # ── Educational: Dangerous combinations ───────────────────────────
        section_card(
            prose(
                "Academic research on financial crises (Reinhart & Rogoff; BIS; IMF Working Papers) "
                "identifies recurring pattern clusters. These are probabilistic observations from "
                "historical episodes, not deterministic rules:",
                header="Historically Dangerous Indicator Combinations",
            ),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("Debt-Rate Spiral", style={
                        "fontWeight": "600", "fontSize": "13px", "color": "#2d3748", "marginBottom": "6px",
                    }),
                    html.P(
                        "High government debt/GDP + rapidly rising interest rates → expanding debt "
                        "service costs → crowding out productive investment → deteriorating growth "
                        "trajectory. The critical metric is debt service as a share of revenue, "
                        "not the absolute debt level.",
                        style={"fontSize": "12px", "color": C["slate"], "lineHeight": "1.7"},
                    ),
                ], style={
                    "borderLeft": "3px solid #fc8181", "paddingLeft": "12px", "marginBottom": "16px",
                }), md=6),
                dbc.Col(html.Div([
                    html.Div("Stagflation Trap", style={
                        "fontWeight": "600", "fontSize": "13px", "color": "#2d3748", "marginBottom": "6px",
                    }),
                    html.P(
                        "Persistent above-target inflation + softening labor market → contradictory "
                        "signals for central banks → elevated policy error risk → credibility of "
                        "monetary institutions tested. The 1970s remain the benchmark episode: "
                        "dual mandates in conflict, with no clean resolution available.",
                        style={"fontSize": "12px", "color": C["slate"], "lineHeight": "1.7"},
                    ),
                ], style={
                    "borderLeft": "3px solid #f6ad55", "paddingLeft": "12px", "marginBottom": "16px",
                }), md=6),
                dbc.Col(html.Div([
                    html.Div("Credit Stress Cascade", style={
                        "fontWeight": "600", "fontSize": "13px", "color": "#2d3748", "marginBottom": "6px",
                    }),
                    html.P(
                        "Rising loan delinquencies + elevated market volatility + tightening "
                        "lending standards → credit contraction → demand compression → "
                        "self-reinforcing slowdown. CRE delinquency rates are a key early "
                        "signal because commercial real estate is rate-sensitive and "
                        "heavily levered.",
                        style={"fontSize": "12px", "color": C["slate"], "lineHeight": "1.7"},
                    ),
                ], style={
                    "borderLeft": "3px solid #f6ad55", "paddingLeft": "12px", "marginBottom": "16px",
                }), md=6),
                dbc.Col(html.Div([
                    html.Div("Confidence Breakdown", style={
                        "fontWeight": "600", "fontSize": "13px", "color": "#2d3748", "marginBottom": "6px",
                    }),
                    html.P(
                        "When multiple dimensions show stress simultaneously, policy space narrows. "
                        "Fiscal tools are constrained by debt levels. Monetary tools are constrained "
                        "by inflation. Each instrument has fewer degrees of freedom. Historical "
                        "episodes suggest that institutional credibility — not just economic "
                        "fundamentals — often determines whether stress resolves or cascades.",
                        style={"fontSize": "12px", "color": C["slate"], "lineHeight": "1.7"},
                    ),
                ], style={
                    "borderLeft": "3px solid #fc8181", "paddingLeft": "12px", "marginBottom": "16px",
                }), md=6),
            ], className="g-4"),
        ),

        # ── Reading guide ─────────────────────────────────────────────────
        section_card(
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div([
                        html.Span("Normal", style={
                            "display": "inline-block", "padding": "2px 10px",
                            "backgroundColor": "#f0fff4", "color": "#276749",
                            "borderRadius": "4px", "fontWeight": "600", "fontSize": "12px",
                            "marginRight": "10px",
                        }),
                        html.Span(
                            "Readings consistent with historical norms. Does not imply immunity to future stress.",
                            style={"fontSize": "12px", "color": C["slate"]},
                        ),
                    ], style={"marginBottom": "10px"}),
                    html.Div([
                        html.Span("Elevated", style={
                            "display": "inline-block", "padding": "2px 10px",
                            "backgroundColor": "#fffff0", "color": "#975a16",
                            "borderRadius": "4px", "fontWeight": "600", "fontSize": "12px",
                            "marginRight": "10px",
                        }),
                        html.Span(
                            "Above historical norms. Warrants attention in combination with other indicators.",
                            style={"fontSize": "12px", "color": C["slate"]},
                        ),
                    ], style={"marginBottom": "10px"}),
                    html.Div([
                        html.Span("Stressed", style={
                            "display": "inline-block", "padding": "2px 10px",
                            "backgroundColor": "#fff5f5", "color": "#9b2c2c",
                            "borderRadius": "4px", "fontWeight": "600", "fontSize": "12px",
                            "marginRight": "10px",
                        }),
                        html.Span(
                            "Levels historically associated with significant economic stress. "
                            "Single stressed indicators have often resolved without systemic crisis.",
                            style={"fontSize": "12px", "color": C["slate"]},
                        ),
                    ]),
                ]), md=8),
                dbc.Col(prose(
                    "The overall configuration of indicators — not any single reading — is the "
                    "most useful signal. Trends matter as much as levels: a rising unemployment "
                    "rate at 4.8% may be more concerning than a stable 5.2%. "
                    "All thresholds are empirically motivated but not universally agreed upon. "
                    "Apply judgment accordingly.",
                ), md=4),
            ], className="g-4"),
            title="How to Read This Page",
        ),

    ], style={"padding": "24px"})


# ── App Layout ────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=EXTERNAL,
    suppress_callback_exceptions=True,
    title="FRED Economic Monitor",
)

app.layout = html.Div([
    # ── Header ────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div("FRED Economic Monitor", style={
                "fontSize": "20px", "fontWeight": "600", "letterSpacing": "-0.01em",
            }),
            html.Div("Federal Reserve Bank of St. Louis Data", style={
                "fontSize": "11px", "opacity": "0.65", "marginTop": "2px",
            }),
        ]),
        html.Div([
            html.Div(f"Data updated {dl.last_updated()}", style={
                "fontSize": "11px", "opacity": "0.7", "textAlign": "right",
            }),
            html.Div(
                f"{len(dl.available)} series — "
                + (f"SQL: {_sql.connection_info()}" if _sql else "/data"),
                style={"fontSize": "11px", "opacity": "0.5", "textAlign": "right", "marginTop": "2px"},
            ),
        ]),
    ], style=STYLE_HEADER),

    # ── Tabs ──────────────────────────────────────────────────────────────
    html.Div([
        dcc.Tabs(
            id="main-tabs",
            value="crisis",
            children=[
                dcc.Tab(label="Crisis Watch",     value="crisis",        style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Summary",         value="summary",       style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Inflation",        value="inflation",     style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Money Supply",     value="money",         style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Labor Market",     value="labor",         style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Markets & Rates",  value="markets",       style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Fiscal",           value="fiscal",        style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
            ],
            style={"backgroundColor": "#ffffff", "borderBottom": "1px solid #e2e8f0"},
        ),
        html.Div(id="tab-content"),
    ], style={"backgroundColor": "#ffffff"}),

], style=STYLE_PAGE)


# ── Callback ──────────────────────────────────────────────────────────────

@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab: str) -> html.Div:
    return {
        "summary":   summary_tab,
        "inflation": inflation_tab,
        "money":     money_supply_tab,
        "labor":     labor_tab,
        "markets":   markets_rates_tab,
        "fiscal":    fiscal_tab,
        "crisis":    crisis_watch_tab,
    }.get(tab, summary_tab)()


# ── Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loaded {len(dl.available)} series: {', '.join(sorted(dl.available))}")
    print(f"Starting dashboard at http://{DASH.host}:{DASH.port}")
    app.run(debug=False, host=DASH.host, port=DASH.port)
