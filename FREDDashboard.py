"""
FRED Economic Monitor — Institutional Dashboard
Run:  python FREDDashboard.py
Then open http://127.0.0.1:8050 in a browser.

Data is loaded from /data at startup and cached in memory.
To refresh data, run FREDDownloader.py and restart this script.
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from _classes.data_loader import DataLoader
from _classes.sql_storage import SQLStorage
from _classes.chart_factory import (
    line_chart, area_chart, multi_line_chart, bar_change_chart,
    percentile_chart, dual_axis_chart, derived_ratio_chart,
    walcl_pct_gdp_chart, real_rate_chart, yield_spread_chart, risk_heatmap_chart,
    derived_spread_chart,
)
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
resilience_dims = re.system_resilience_dimensions()

_EVENTS_LONG  = [("2001-09-11", "9/11"), ("2008-09-15", "GFC"), ("2020-03-23", "COVID"), ("2022-03-16", "Hikes")]
_EVENTS_MED   = [("2008-09-15", "GFC"), ("2020-03-23", "COVID"), ("2022-03-16", "Hikes")]
_EVENTS_SHORT = [("2020-03-23", "COVID"), ("2022-03-16", "Hikes")]


# ── Component Builders ────────────────────────────────────────────────────

_STALENESS_DAYS = {"Daily": 7, "Weekly": 21, "Monthly": 60, "Quarterly": 110}


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert #RRGGBB to rgba(R,G,B,alpha) — Plotly requires rgba for transparency."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _sparkline_fig(series_id: str, line_color: str) -> go.Figure:
    """24-month mini area chart for KPI card sparkline."""
    fig = go.Figure()
    try:
        df = dl.load(series_id)
        if df is not None and not df.empty:
            cutoff = df.index[-1] - pd.DateOffset(months=24)
            plot = df[df.index >= cutoff]
            if not plot.empty:
                col = plot.columns[0]
                fill_color = _hex_to_rgba(line_color, 0.18)
                fig.add_trace(go.Scatter(
                    x=plot.index, y=plot[col],
                    fill="tozeroy",
                    line=dict(color=line_color, width=1),
                    fillcolor=fill_color,
                ))
    except Exception:
        pass
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _staleness_warning(series_id: str, as_of_date) -> html.Span | None:
    """Return a warning badge if data is overdue based on expected frequency."""
    if as_of_date is None:
        return None
    meta = REGISTRY.get(series_id, {})
    freq = meta.get("frequency", "Monthly")
    delta_days = (datetime.now() - as_of_date.to_pydatetime()).days
    threshold = _STALENESS_DAYS.get(freq, 60)
    if delta_days > threshold:
        return html.Span(
            f" ⚠ {delta_days}d",
            style={"fontSize": "10px", "color": "#E53E3E", "marginLeft": "4px"},
            title=f"Data is {delta_days} days old — may be stale for {freq} series",
        )
    return None


def kpi_card(series_id: str) -> html.Div:
    """Single KPI card with risk color coding and hover tooltip."""
    risk, display, _ = re.score(series_id)
    style = RISK_STYLE.get(risk, RISK_STYLE["neutral"])
    meta = REGISTRY.get(series_id, {})
    basis = meta.get("risk_basis", "level")
    basis_label = {"yoy": "YoY", "mom_change": "MoM"}.get(basis, "")

    _, as_of_date = dl.get_latest(series_id)
    as_of = as_of_date.strftime(DASH.date_display_fmt) if as_of_date is not None else ""
    arrow = re.trend_arrow(series_id)
    stale = _staleness_warning(series_id, as_of_date)

    tooltip_text = meta.get("description") or meta.get("name", series_id)
    notes = meta.get("notes")
    if notes:
        tooltip_text = f"{tooltip_text}. {notes}"
    card_id = f"kpi-card-{series_id}"

    card = html.Div([
        html.Div(meta.get("short_name", series_id), style={
            "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.06em",
            "color": style["text"], "textTransform": "uppercase", "marginBottom": "6px",
        }),
        html.Div([
            html.Span(display, style={
                "fontSize": "28px", "fontWeight": "600", "color": style["text"], "lineHeight": "1.1",
            }),
            html.Span(arrow, style={
                "fontSize": "16px", "fontWeight": "400", "color": style["text"],
                "opacity": "0.7", "marginLeft": "6px", "verticalAlign": "middle",
            }),
            html.Span(basis_label, style={
                "fontSize": "11px", "color": style["text"], "opacity": "0.75",
                "marginLeft": "5px", "verticalAlign": "middle",
            }) if basis_label else None,
        ], style={"marginBottom": "4px", "display": "flex", "alignItems": "center"}),
        dcc.Graph(
            figure=_sparkline_fig(series_id, style["border"]),
            config={"displayModeBar": False},
            style={"height": "48px", "marginBottom": "6px"},
        ),
        html.Div(style={"height": "1px", "backgroundColor": style["border"], "margin": "6px 0"}),
        html.Div([
            html.Span(style["label"], style={"fontWeight": "500"}),
            html.Div([
                html.Span(f"as of {as_of}", style={"opacity": "0.65"}),
                stale,
            ], style={"marginLeft": "auto", "display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "alignItems": "center",
            "fontSize": "11px", "color": style["text"],
        }),
    ], id=card_id, style={
        "backgroundColor": style["bg"],
        "border": f"1px solid {style['border']}",
        "borderLeft": f"4px solid {style['border']}",
        "borderRadius": "6px",
        "padding": "16px",
        "minHeight": "130px",
        "cursor": "help",
    })

    return html.Div([
        card,
        dbc.Tooltip(
            tooltip_text,
            target=card_id,
            placement="top",
            style={"fontSize": "12px", "maxWidth": "320px"},
        ),
    ])


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


def crisis_dim_card(name: str, dim: dict, md: int | None = None) -> dbc.Col:
    """One Crisis Watch or System Resilience dimension scorecard."""
    risk = dim["score"]
    style = RISK_STYLE[risk]
    components = dim["components"]

    comp_rows = [
        html.Div([
            html.Span(label, style={"fontSize": "11px", "color": C["slate"]}),
            html.Div([
                html.Div(display, style={
                    "fontSize": "11px", "fontWeight": "600",
                    "color": RISK_STYLE[r]["text"], "textAlign": "right",
                }),
                html.Div(f"as of {as_of}", style={
                    "fontSize": "9px", "color": C["slate"], "opacity": "0.6",
                    "textAlign": "right",
                }) if as_of else None,
            ], style={"marginLeft": "auto"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"})
        for label, r, display, as_of in components
    ]

    col_md = md if md is not None else (12 // DASH.n_crisis_dims)
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
    }), md=col_md, sm=6, xs=12)


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


def _narrative_item_web(risk: str, text: str) -> html.Div:
    rs = RISK_STYLE.get(risk, RISK_STYLE["neutral"])
    return html.Div(
        html.Span(text, style={"fontSize": "13px", "color": C["text"], "lineHeight": "1.75"}),
        style={
            "borderLeft": f"4px solid {rs['border']}",
            "backgroundColor": rs["bg"],
            "padding": "12px 16px",
            "marginBottom": "10px",
            "borderRadius": "0 6px 6px 0",
        },
    )


# ── Tab Content ───────────────────────────────────────────────────────────



def inflation_tab(lookback: int | None = 20) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "CPIAUCSL", yoy=True, lookback_years=lookback or 25,
                                 threshold_green=2.5, threshold_red=4.5,
                                 title="Headline CPI — Year-over-Year %")),
            ), md=6),
            dbc.Col(section_card(
                graph(area_chart(dl, "CPILFESL", yoy=True, lookback_years=lookback or 25,
                                 threshold_green=2.5, threshold_red=4.0,
                                 color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                                 title="Core CPI (ex Food & Energy) — Year-over-Year %")),
            ), md=6),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(multi_line_chart(dl,
                    [("CPILFESL", C["teal"]), ("PCEPILFE", C["blue"])],
                    title="Core CPI vs. Core PCE — YoY % (10 Years)",
                    lookback_years=min(lookback or 10, 10), yoy=True,
                )),
                prose(
                    "Core PCE (blue) is the Fed's primary inflation target at 2%. "
                    "It typically runs 0.3–0.5% below Core CPI (teal) due to different "
                    "weighting methodologies. The Fed watches the gap between the two — "
                    "when both are elevated together, the signal is more credible.",
                ),
            ), md=8),
            dbc.Col(section_card(
                graph(area_chart(dl, "PCEPILFE", yoy=True, lookback_years=lookback or 25,
                                 threshold_green=2.5, threshold_red=4.0,
                                 color=C["blue"], fill_color="rgba(43,108,176,0.10)",
                                 title="Core PCE — Year-over-Year %")),
                prose(
                    "The FOMC explicitly targets 2% Core PCE. Monthly, published ~4 weeks "
                    "after month-end by the BEA.",
                ),
            ), md=4),
        ], className="g-0"),
        html.Div("Money Supply", style=STYLE_SECTION_LABEL),
        dbc.Row([
            dbc.Col(section_card(
                graph(multi_line_chart(dl,
                    [("M1SL", C["blue"]), ("M2SL", C["teal"])],
                    title="M1 & M2 Money Supply — Level (Billions $)",
                    lookback_years=lookback or 20,
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
                graph(area_chart(dl, "M2SL", yoy=True, lookback_years=lookback or 30,
                                 title="M2 Growth — Year-over-Year %")),
            ), md=6),
            dbc.Col(section_card(
                graph(area_chart(dl, "M2REAL", yoy=True, lookback_years=lookback or 20,
                                 color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                                 title="Real M2 Growth — Year-over-Year %")),
            ), md=6),
        ], className="g-0"),
    ], style={"padding": "24px"})


def labor_tab(lookback: int | None = 20) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "UNRATE", lookback_years=lookback or 30,
                                 threshold_green=4.5, threshold_red=6.0,
                                 color=C["slate"], fill_color="rgba(74,85,104,0.08)",
                                 title="Unemployment Rate %")),
            ), md=6),
            dbc.Col(section_card(
                graph(bar_change_chart(dl, "PAYEMS", lookback_years=min(lookback or 5, 5),
                                       title="Nonfarm Payrolls — Monthly Change (Thousands)")),
            ), md=6),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(line_chart(dl, "CIVPART", lookback_years=lookback or 30, color=C["teal"],
                                 title="Labor Force Participation Rate %")),
            ), md=6),
            dbc.Col(section_card(
                graph(multi_line_chart(dl,
                    [("EMRATIO", C["blue"]), ("CIVPART", C["teal"])],
                    title="Employment-Population Ratio vs. Participation Rate %",
                    lookback_years=lookback or 30,
                )),
            ), md=6),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "ICSA", lookback_years=lookback or 10,
                                 ma_periods=4,
                                 color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                                 threshold_green=250, threshold_red=350,
                                 title="Initial Jobless Claims (Weekly, 4-Week MA)",
                                 recession_shading=True)),
                prose(
                    "Weekly initial unemployment insurance claims — one of the timeliest "
                    "labor market reads. The 4-week moving average (solid line) smooths "
                    "holiday and seasonal noise. Sustained readings above 300K signal "
                    "deteriorating labor conditions; above 400K indicate acute stress.",
                ),
            ), md=6),
            dbc.Col(section_card(
                graph(area_chart(dl, "PSAVERT", lookback_years=lookback or 20,
                                 color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                                 threshold_green=7, threshold_red=4,
                                 title="Personal Saving Rate %")),
                prose(
                    "Personal saving as a percent of disposable income. Low savings often "
                    "signal households drawing down buffers to maintain consumption — a "
                    "late-cycle vulnerability. Post-COVID savings normalization from the "
                    "excess-savings peak is now largely complete.",
                ),
            ), md=6),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "INDPRO", yoy=True, lookback_years=lookback or 20,
                                 color=C["blue"], fill_color="rgba(43,108,176,0.10)",
                                 threshold_green=1, threshold_red=-1,
                                 title="Industrial Production — Year-over-Year %",
                                 recession_shading=True)),
                prose(
                    "Industrial production index (manufacturing, mining, utilities) YoY change. "
                    "Negative readings often precede or coincide with recessions. "
                    "Manufacturing sub-indexes are released in the middle of the following month.",
                ),
            ), md=6),
            dbc.Col(section_card(
                graph(area_chart(dl, "USSLIND", lookback_years=lookback or 10,
                                 color=C["red"], fill_color="rgba(155,44,44,0.09)",
                                 threshold_green=0, threshold_red=-5,
                                 title="Conference Board Leading Economic Index (MoM %)")),
                prose(
                    "The LEI aggregates 10 forward-looking components (building permits, "
                    "jobless claims, yield curve, stock prices, etc.). Consecutive monthly "
                    "declines have historically signaled recession 6-12 months ahead.",
                ),
            ), md=6),
        ], className="g-0"),
    ], style={"padding": "24px"})


def markets_rates_tab(lookback: int | None = 20) -> html.Div:
    H = CHART.graph_height
    return html.Div([
        dbc.Row([
            dbc.Col(section_card(
                graph(dual_axis_chart(dl,
                    left_series=("VIXCLS", "#744210"),
                    right_series=("SP500_CAPE", C["amber"]),
                    title="VIX (left) vs CAPE (right) — Volatility vs Valuation",
                    lookback_years=lookback or 20, recession_shading=True), H),
                prose(
                    "Elevated CAPE with compressed VIX signals complacency — the market is "
                    "highly valued yet not pricing in risk. This combination has historically "
                    "preceded sharp corrections.",
                ),
            ), md=12),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(yield_spread_chart(dl, lookback_years=lookback or 20, recession_shading=True), H),
                prose(
                    "The 10Y-2Y spread is a widely monitored recession predictor. "
                    "Inversions (spread < 0, shaded red) have preceded every U.S. recession "
                    "in the past 50 years, typically with a 12–24 month lead time.",
                ),
            ), md=12),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "SP500_CAPE", lookback_years=None,
                                 threshold_green=20, threshold_red=30,
                                 color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                                 title="S&P 500 CAPE (Shiller P/E) — Full History",
                                 events=_EVENTS_LONG)),
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
                                 title="S&P 500 Trailing P/E (40 Years)",
                                 events=_EVENTS_LONG)),
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
                graph(area_chart(dl, "RECPROUSM156N", lookback_years=lookback or 20,
                    threshold_green=15, threshold_red=40,
                    color=C["red"], fill_color="rgba(155,44,44,0.09)",
                    title="NY Fed 12-Month Recession Probability %",
                    recession_shading=True), H),
                prose(
                    "NY Fed model probability of U.S. recession in the next 12 months, "
                    "derived from the Treasury yield curve. Readings above 40% have "
                    "historically signaled near-certain recession.",
                ),
            ), md=6),
            dbc.Col(section_card(
                graph(area_chart(dl, "DTWEXBGS", yoy=True,
                    title="U.S. Dollar Index — Year-over-Year % Change",
                    threshold_green=8, threshold_red=15,
                    color=C["blue"], fill_color="rgba(43,108,176,0.10)",
                    lookback_years=lookback or 20, events=_EVENTS_MED), H),
                prose(
                    "Nominal broad dollar index (goods-weighted). Large year-over-year "
                    "appreciation tightens global financial conditions — particularly for "
                    "emerging markets with dollar-denominated debt.",
                ),
            ), md=6),
        ], className="g-0"),
    ], style={"padding": "24px"})


def fiscal_tab(lookback: int | None = 20) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "GFDEGDQ188S", lookback_years=lookback or 40,
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
                graph(area_chart(dl, "GFDEBTN", lookback_years=lookback or 20,
                    color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                    title="Federal Debt Outstanding ($B, Monthly)")),
                prose(
                    "Monthly total public debt outstanding — more timely than the quarterly "
                    "Debt/GDP ratio. Tracks the dollar pace of accumulation independent of "
                    "GDP growth."
                ),
            ), md=8),
            dbc.Col(section_card(
                graph(area_chart(dl, "GFDEBTN", lookback_years=5,
                    color=C["red"], fill_color="rgba(155,44,44,0.09)",
                    title="Federal Debt Outstanding — Last 5 Years")),
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


def crisis_watch_tab(lookback: int | None = 20) -> html.Div:
    # ── Dimension scorecards — md=4 for up to 3 per row ──────────────────
    dim_cards = dbc.Row(
        [crisis_dim_card(name, dim, md=4) for name, dim in crisis_dims.items()],
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

        # ── Historical risk heatmap ────────────────────────────────────────
        section_card(
            graph(risk_heatmap_chart(re, lookback_years=10), height=380),
            prose(
                "Monthly risk classification of twelve key indicators over the past ten years. "
                "Green = normal; yellow = elevated; red = stressed. "
                "Clusters of color identify correlated deterioration across multiple indicators — "
                "the pattern that precedes systemic stress more reliably than any single reading."
            ),
            title="Indicator Risk Status — 10-Year History",
        ),

        # ── Dimension detail charts ────────────────────────────────────────
        section_card(
            dbc.Row([
                dbc.Col(graph(yield_spread_chart(dl, lookback_years=lookback or 20,
                    recession_shading=True), height=CHART.graph_height_small), md=6),
                dbc.Col(graph(area_chart(dl, "RECPROUSM156N", lookback_years=lookback or 20,
                    threshold_green=15, threshold_red=40,
                    color=C["red"], fill_color="rgba(155,44,44,0.09)",
                    title="NY Fed 12-Month Recession Probability %",
                    recession_shading=True), height=CHART.graph_height_small), md=6),
            ], className="g-2"),
            dbc.Row([
                dbc.Col(graph(area_chart(dl, "CPILFESL", yoy=True, lookback_years=lookback or 10,
                    color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                    threshold_green=2.5, threshold_red=4.0,
                    title="Core CPI YoY %"), height=CHART.graph_height_small), md=6),
                dbc.Col(graph(area_chart(dl, "UNRATE", lookback_years=lookback or 15,
                    threshold_green=4.5, threshold_red=6.0,
                    color=C["slate"], fill_color="rgba(74,85,104,0.08)",
                    title="Unemployment Rate %"), height=CHART.graph_height_small), md=6),
            ], className="g-2"),
            dbc.Row([
                dbc.Col(graph(area_chart(dl, "USSLIND", lookback_years=lookback or 10,
                    color=C["red"], fill_color="rgba(155,44,44,0.09)",
                    threshold_green=0, threshold_red=-5,
                    title="Conference Board Leading Economic Index (MoM %)",
                    recession_shading=True), height=CHART.graph_height_small), md=6),
                dbc.Col(graph(area_chart(dl, "ICSA", lookback_years=lookback or 10,
                    ma_periods=4,
                    color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                    threshold_green=250, threshold_red=350,
                    title="Initial Jobless Claims — 4-Week MA"), height=CHART.graph_height_small), md=6),
            ], className="g-2"),
            title="Key Indicator Detail"
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
                    "2006 — 16 months before the December 2007 recession began. CRE lending "
                    "standards tightened sharply through 2007. Both signals were visible and "
                    "actively debated; consensus remained constructive well into 2007.",
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
                        "Rising CRE lending standards + elevated market volatility + tightening "
                        "C&I standards → credit contraction → demand compression → "
                        "self-reinforcing slowdown. CRE SLOOS tightening is a leading "
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


# ── System Resilience Helpers ─────────────────────────────────────────────

def _absorption_capacity_table_web() -> html.Div:
    """HTML version of the System Absorption Capacity summary table."""
    capacity = re.system_absorption_capacity()

    header = html.Div([
        html.Div("Category",      style={"flex": "2", "fontWeight": "600", "fontSize": "11px",
                                         "textTransform": "uppercase", "letterSpacing": "0.06em",
                                         "color": C["muted"]}),
        html.Div("Status",        style={"flex": "1.2", "fontWeight": "600", "fontSize": "11px",
                                         "textTransform": "uppercase", "letterSpacing": "0.06em",
                                         "color": C["muted"]}),
        html.Div("Key Metric",    style={"flex": "2", "fontWeight": "600", "fontSize": "11px",
                                         "textTransform": "uppercase", "letterSpacing": "0.06em",
                                         "color": C["muted"]}),
        html.Div("Current Reading", style={"flex": "1.5", "fontWeight": "600", "fontSize": "11px",
                                           "textTransform": "uppercase", "letterSpacing": "0.06em",
                                           "color": C["muted"]}),
    ], style={"display": "flex", "padding": "8px 12px",
              "borderBottom": "2px solid #e2e8f0", "marginBottom": "4px"})

    rows = [header]
    for cat_name, info in capacity.items():
        risk = info["score"]
        rs = RISK_STYLE.get(risk, RISK_STYLE["neutral"])
        rows.append(html.Div([
            html.Div(cat_name, style={"flex": "2", "fontSize": "13px", "fontWeight": "500",
                                      "color": C["text"]}),
            html.Div(
                html.Span(rs["label"], style={
                    "backgroundColor": rs["bg"], "color": rs["text"],
                    "border": f"1px solid {rs['border']}",
                    "borderRadius": "4px", "padding": "2px 8px",
                    "fontSize": "11px", "fontWeight": "600",
                }),
                style={"flex": "1.2"},
            ),
            html.Div(info["key_metric"], style={"flex": "2", "fontSize": "12px",
                                                "color": C["muted"]}),
            html.Div(html.B(info["value"]), style={"flex": "1.5", "fontSize": "12px",
                                                   "color": C["text"]}),
        ], style={"display": "flex", "alignItems": "center",
                  "padding": "10px 12px", "borderBottom": "1px solid #edf2f7"}))

    return html.Div(rows, style={"border": "1px solid #e2e8f0", "borderRadius": "6px"})


def _refinancing_risk_box_web() -> html.Div:
    """HTML version of the Refinancing & Liquidity Risk warning box."""
    risk_level, triggers = re.refinancing_liquidity_risk()
    rs = RISK_STYLE.get(risk_level, RISK_STYLE["neutral"])

    summary = (
        "No multi-indicator stress conditions currently triggered."
        if not triggers else
        f"{len(triggers)} condition(s) triggered. "
        "Persistent combinations — not isolated spikes — are the primary concern."
    )

    trigger_items = [
        html.Li(t, style={"fontSize": "12px", "color": C["slate"], "marginBottom": "4px",
                           "lineHeight": "1.6"})
        for t in triggers
    ] if triggers else []

    return html.Div([
        html.Div([
            html.Div([
                html.Span("REFINANCING & LIQUIDITY RISK", style={
                    "fontWeight": "700", "fontSize": "11px", "letterSpacing": "0.07em",
                    "textTransform": "uppercase", "color": rs["text"],
                }),
                html.Span(f"  —  {rs['label'].upper()}", style={
                    "fontSize": "11px", "fontWeight": "600", "color": rs["text"], "opacity": "0.8",
                }),
            ], style={"marginBottom": "6px"}),
            html.P(summary, style={"fontSize": "12px", "color": rs["text"],
                                   "margin": "0", "lineHeight": "1.6"}),
        ], style={
            "backgroundColor": rs["bg"],
            "border": f"1px solid {rs['border']}",
            "borderLeft": f"4px solid {rs['border']}",
            "borderRadius": "6px",
            "padding": "14px 16px",
            "marginBottom": "12px" if triggers else "0",
        }),
        html.Ul(trigger_items, style={"paddingLeft": "20px", "marginBottom": "0"}) if triggers else None,
        html.P(
            "Note: this assessment weighs persistence and combinations. "
            "A single elevated indicator resolves quickly; structural deterioration does not.",
            style={"fontSize": "11px", "color": C["muted"], "marginTop": "10px",
                   "fontStyle": "italic", "marginBottom": "0"},
        ) if triggers else None,
    ])


# ── New Tab Content Functions ─────────────────────────────────────────────

def system_resilience_tab() -> html.Div:
    counts = {"red": 0, "yellow": 0, "green": 0, "neutral": 0}
    for dim in resilience_dims.values():
        counts[dim["score"]] += 1

    return html.Div([
        section_card(
            html.H5("System Resilience & Policy Dependency", style={
                "fontWeight": "700", "color": C["header_bg"], "marginBottom": "8px",
            }),
            html.P(
                "The modern financial system is increasingly dependent on liquidity provision, "
                "refinancing capacity, and policy intervention. Structural fragility often "
                "emerges not from isolated economic weakness, but from deterioration in the "
                "system's ability to absorb shocks without extraordinary stabilization measures.",
                style={"fontSize": "13px", "color": C["slate"], "lineHeight": "1.7",
                       "marginBottom": "8px"},
            ),
            html.Div(
                f"{counts['red']} dimension(s) stressed, {counts['yellow']} elevated, "
                f"{counts['green']} within normal range.",
                style={"fontSize": "12px", "fontWeight": "500", "color": C["muted"],
                       "borderTop": "1px solid #edf2f7", "paddingTop": "10px"},
            ),
        ),

        # Dimension cards — 3 per row (md=4 each)
        dbc.Row(
            [crisis_dim_card(name, dim, md=4) for name, dim in resilience_dims.items()],
            className="g-3 mb-4",
        ),

        # Absorption Capacity summary table
        section_card(
            _absorption_capacity_table_web(),
            title="System Absorption Capacity",
        ),

        # Refinancing & Liquidity Risk warning
        section_card(
            _refinancing_risk_box_web(),
            title="Refinancing & Liquidity Risk Assessment",
        ),

        # Structural framework
        section_card(
            dbc.Row([
                dbc.Col(prose(
                    "Structural Fragility — accumulates slowly and reduces the system's margin "
                    "of safety: elevated debt loads, persistent inflation, labor market "
                    "deterioration, demographic stagnation, and credit quality erosion. Often "
                    "masked by asset appreciation or credit availability until it is too late.",
                    header="Structural Fragility",
                ), md=6),
                dbc.Col(prose(
                    "Absorption Capacity — the system's active ability to withstand shocks "
                    "without progressively larger interventions: functioning credit markets, "
                    "liquidity provision, policy flexibility, anchored inflation expectations, "
                    "and stable funding markets. When this deteriorates, structural imbalances "
                    "become acute crises rather than manageable headwinds.",
                    header="Absorption Capacity",
                ), md=6),
            ], className="g-4"),
            prose(
                "Modern markets can tolerate elevated structural imbalances for extended periods "
                "if refinancing channels remain functional and policy credibility remains intact. "
                "Acute crises emerge when structural fragility converges with deterioration in "
                "liquidity, credit, or stabilization capacity — not from structural imbalances alone."
            ),
            title="Structural Stress vs System Absorption Capacity",
        ),

    ], style={"padding": "24px"})


def liquidity_funding_tab(lookback: int | None = 20) -> html.Div:
    H = CHART.graph_height
    return html.Div([
        section_card(prose(
            "This section monitors the financial plumbing: short-term funding markets, "
            "interbank credit conditions, and system-wide liquidity. Stress here typically "
            "appears before it surfaces in equity prices, unemployment, or GDP data — making "
            "these leading signals of systemic deterioration.",
            "Elevated readings in the Financial Stress Index (FSI) and National Financial "
            "Conditions Index (NFCI) indicate tighter-than-normal financial conditions. "
            "A widening CP-Treasury spread signals rising interbank credit cost. Rapid "
            "depletion of the Fed's reverse repo facility indicates reserves leaving the "
            "banking system.",
        )),

        dbc.Row([
            dbc.Col(section_card(
                graph(percentile_chart(dl, "STLFSI4",
                    title="St. Louis Fed Financial Stress Index",
                    color=C["red"], fill_color="rgba(155,44,44,0.09)",
                    lookback_years=lookback or 20, higher_is_bad=True, recession_shading=True), H),
            ), md=6),
            dbc.Col(section_card(
                graph(percentile_chart(dl, "NFCI",
                    title="Chicago Fed National Financial Conditions Index",
                    color=C["slate"], fill_color="rgba(74,85,104,0.08)",
                    lookback_years=lookback or 20, higher_is_bad=True, recession_shading=True), H),
            ), md=6),
        ], className="g-0"),

        dbc.Row([
            dbc.Col(section_card(
                graph(derived_spread_chart(dl, "DCPF3M", "DGS3MO",
                    title="3-Month CP minus Treasury Spread % (Modern TED Equivalent)",
                    color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                    threshold_green=0.5, threshold_red=1.0,
                    lookback_years=lookback or 25, recession_shading=True), H),
                prose("3-Month AA Financial Commercial Paper rate minus 3-Month Treasury — "
                      "the modern successor to the TED spread. Measures unsecured short-term "
                      "bank borrowing cost above the risk-free rate. Above 100bps signals "
                      "elevated interbank credit stress."),
            ), md=6),
            dbc.Col(section_card(
                graph(area_chart(dl, "SOFR",
                    title="Secured Overnight Financing Rate (SOFR) %",
                    color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                    lookback_years=8), H),
                prose("LIBOR replacement. In isolation tracks Fed Funds; "
                      "divergence signals repo market stress."),
            ), md=6),
        ], className="g-0"),

        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "RRPONTSYD",
                    title="Fed Overnight Reverse Repo Facility ($B)",
                    threshold_green=500, threshold_red=100,
                    color=C["blue"], fill_color="rgba(43,108,176,0.10)",
                    lookback_years=lookback or 15), H),
                prose("High usage = excess reserves in the system (accommodative). "
                      "Rapid drawdown toward zero signals reserves leaving faster "
                      "than the Fed can replace them."),
            ), md=6),
            dbc.Col(section_card(
                graph(percentile_chart(dl, "NFCICREDIT",
                    title="NFCI Credit Subindex — Credit Conditions Specifically",
                    color=C["red"], fill_color="rgba(155,44,44,0.09)",
                    lookback_years=lookback or 20, higher_is_bad=True, recession_shading=True), H),
                prose("Isolates the credit channel from the broader NFCI. "
                      "Early warning of tightening credit access before it shows in lending data."),
            ), md=6),
        ], className="g-0"),

    ], style={"padding": "24px"})


def credit_markets_tab(lookback: int | None = 20) -> html.Div:
    H = CHART.graph_height
    return html.Div([
        section_card(prose(
            "Most systemic crises emerge when refinancing capacity deteriorates and credit "
            "spreads widen persistently. Credit market functionality is a more reliable "
            "leading indicator of economic stress than equity market volatility — spreads "
            "reflect actual lending conditions, not just sentiment.",
            "Persistent spread widening matters more than temporary spikes. A widening that "
            "lasts 4-8 weeks or more indicates structural credit deterioration that directly "
            "impairs rollover financing for leveraged borrowers.",
        )),

        # ── Bank Lending Standards (SLOOS) ────────────────────────────────
        html.Div("Bank Lending Standards", style=STYLE_SECTION_LABEL),
        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "DRTSCILM",
                    title="Net % Banks Tightening C&I Loan Standards (Large & Medium Firms)",
                    threshold_green=10, threshold_red=40,
                    color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                    lookback_years=lookback or 25, recession_shading=True), H),
                prose(
                    "Quarterly Senior Loan Officer Opinion Survey (SLOOS). Positive values "
                    "indicate net tightening of credit standards; negative values indicate net "
                    "easing. Persistent tightening restricts business credit access, compressing "
                    "investment and hiring.",
                ),
            ), md=8),
            dbc.Col(section_card(
                prose(
                    "The SLOOS captures forward-looking credit availability from bank loan "
                    "officers — a leading indicator of business investment that typically "
                    "precedes GDP by 2–4 quarters.",
                    "Historical signals: net tightening above 30% preceded every recession "
                    "since 1990. The combination of rising spreads AND tightening standards "
                    "is particularly reliable — it indicates that both market pricing and "
                    "bank underwriting are simultaneously contracting.",
                    header="SLOOS as a Leading Indicator",
                ),
            ), md=4),
        ], className="g-0"),

        # ── Credit Spreads ────────────────────────────────────────────────
        html.Div("Credit Spreads", style=STYLE_SECTION_LABEL),
        dbc.Row([
            dbc.Col(section_card(
                graph(percentile_chart(dl, "BAMLH0A0HYM2",
                    title="High Yield OAS (ICE BofA) % — with Recession Periods",
                    color=C["red"], fill_color="rgba(155,44,44,0.09)",
                    lookback_years=lookback or 25, higher_is_bad=True, recession_shading=True),
                    CHART.graph_height + 40),
                prose("Percentile bands show where the current reading falls in the "
                      "25-year historical distribution. Green zone = below 25th percentile; "
                      "red zone = above 75th percentile (historically stressed)."),
            ), md=8),
            dbc.Col(section_card(
                graph(percentile_chart(dl, "BAMLC0A0CM",
                    title="Investment Grade OAS (ICE BofA) %",
                    color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                    lookback_years=lookback or 20, higher_is_bad=True, recession_shading=True),
                    CHART.graph_height + 40),
                prose("IG spreads widen when credit stress reaches investment-grade "
                      "issuers — a later but more systemic signal than HY alone."),
            ), md=4),
        ], className="g-0"),

        dbc.Row([
            dbc.Col(section_card(
                graph(dual_axis_chart(dl,
                    left_series=("BAMLH0A0HYM2", C["red"]),
                    right_series=("VIXCLS", C["amber"]),
                    title="HY Spread % (left) vs VIX (right) — Sentiment vs Credit",
                    lookback_years=lookback or 20, recession_shading=True), H),
                prose("Persistent HY spread widening without a corresponding VIX spike "
                      "suggests structural credit deterioration — more concerning for "
                      "refinancing capacity than sentiment-driven volatility spikes."),
            ), md=6),
            dbc.Col(section_card(
                graph(area_chart(dl, "SUBLPDRCSN",
                    title="CRE Lending Standards — Net % Tightening",
                    threshold_green=10, threshold_red=40,
                    color=C["red"], fill_color="rgba(155,44,44,0.09)",
                    lookback_years=lookback or 25, recession_shading=True), H),
                prose("SLOOS CRE-specific lending standards — a leading indicator "
                      "that tightens 6-12 months before delinquency rates rise. "
                      "Above 40% net tightening has preceded CRE credit contractions."),
            ), md=6),
        ], className="g-0"),

    ], style={"padding": "24px"})


def policy_constraints_tab(lookback: int | None = 20) -> html.Div:
    H = CHART.graph_height
    return html.Div([
        section_card(prose(
            "The critical issue is not absolute debt levels, but whether policymakers "
            "retain the flexibility to stabilize markets without destabilizing inflation, "
            "funding markets, or sovereign confidence.",
            "Three primary constraints bind policy maneuverability: fiscal space (how much "
            "of revenue is consumed by debt service), monetary space (whether inflation "
            "expectations allow rate cuts), and balance sheet capacity (whether the Fed "
            "retains room for further asset purchases).",
        )),

        dbc.Row([
            dbc.Col(section_card(
                graph(derived_ratio_chart(dl,
                    numerator_id="A091RC1Q027SBEA",
                    denominator_id="W006RC1Q027SBEA",
                    title="Federal Interest Payments / Current Receipts %",
                    color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                    lookback_years=lookback or 40,
                    threshold_green=12, threshold_red=20,
                    recession_shading=True,
                    overlay_id="GS10",
                    overlay_color=C["blue"],
                    overlay_label="10-Yr Treasury Yield %"), H),
                prose("Direct measure of fiscal leverage: what share of every dollar "
                      "collected in taxes must be paid as debt service. Green: <12%; "
                      "yellow: 12–20%; red: >20%. Dashed blue line: 10-year Treasury "
                      "yield — shows how rate normalization drives the rising debt service burden."),
            ), md=6),
            dbc.Col(section_card(
                graph(walcl_pct_gdp_chart(dl, lookback_years=25), H),
                prose("Fed balance sheet as % of nominal GDP. Shows the scale of prior "
                      "QE programs and the trajectory of QT. A large balance sheet limits "
                      "capacity for future emergency asset purchases."),
            ), md=6),
        ], className="g-0"),

        dbc.Row([
            dbc.Col(section_card(
                graph(real_rate_chart(dl, lookback_years=lookback or 30), H),
                prose("Real FF rate = FEDFUNDS minus Core CPI YoY. Deeply negative = "
                      "stimulative/inflationary; very high positive = demand-killing. "
                      "Zero is the neutral reference line."),
            ), md=6),
            dbc.Col(section_card(
                graph(multi_line_chart(dl,
                    [("FEDFUNDS", C["blue"]), ("CPILFESL", C["teal"])],
                    title="Federal Funds Rate vs Core CPI YoY %",
                    lookback_years=lookback or 20, yoy=False), H),
                prose("When FEDFUNDS is below Core CPI (real rate negative), monetary "
                      "policy is accommodative. The gap between these two series "
                      "determines the real policy stance."),
            ), md=6),
        ], className="g-0"),

        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "T5YIE",
                    title="5-Year Breakeven Inflation Expectations %",
                    threshold_green=2.5, threshold_red=3.2,
                    color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                    lookback_years=lookback or 20), H),
            ), md=6),
            dbc.Col(section_card(
                graph(area_chart(dl, "T10YIE",
                    title="10-Year Breakeven Inflation Expectations %",
                    threshold_green=2.5, threshold_red=3.0,
                    color=C["slate"], fill_color="rgba(74,85,104,0.08)",
                    lookback_years=lookback or 20), H),
                prose("Breakeven rates from the TIPS market. Persistent readings above "
                      "3% indicate unanchored inflation expectations — at which point the "
                      "Fed cannot cut rates in response to economic weakness without "
                      "risking a credibility crisis."),
            ), md=6),
        ], className="g-0"),

    ], style={"padding": "24px"})


def executive_summary_tab() -> html.Div:
    from _classes.series_registry import CATEGORY_ORDER

    overall = re.overall_stress_level()
    narratives = re.executive_narrative()
    rs_overall = RISK_STYLE.get(overall, RISK_STYLE["neutral"])

    available_ids = [sid for sid in REGISTRY
                     if sid in dl.available and REGISTRY[sid].get("show_in_summary", True)]
    kpi_rows = []
    for cat in CATEGORY_ORDER:
        ids = [s for s in available_ids if REGISTRY[s]["category"] == cat]
        if not ids:
            continue
        kpi_rows.append(html.Div(CATEGORY_LABELS.get(cat, cat), style=STYLE_SECTION_LABEL))
        kpi_rows.append(dbc.Row(
            [dbc.Col(kpi_card(sid), md=3, sm=6, xs=12) for sid in ids],
            className="g-3 mb-2",
        ))

    return html.Div([
        section_card(
            html.Div([
                html.H5("Economic Overview", style={
                    "fontWeight": "700", "color": C["header_bg"], "marginBottom": "8px",
                }),
                html.P(
                    "Synthesizes key indicators across six dimensions into an overall system "
                    "stress assessment. Color reflects the aggregate configuration — not a forecast.",
                    style={"fontSize": "13px", "color": C["slate"], "lineHeight": "1.7",
                           "marginBottom": "14px"},
                ),
                html.Div([
                    html.Span("OVERALL SYSTEM STRESS", style={
                        "fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.07em",
                        "textTransform": "uppercase", "color": rs_overall["text"],
                        "marginRight": "12px",
                    }),
                    html.Span(rs_overall["label"].upper(), style={
                        "fontSize": "18px", "fontWeight": "700", "color": rs_overall["text"],
                    }),
                ], style={
                    "backgroundColor": rs_overall["bg"],
                    "border": f"1px solid {rs_overall['border']}",
                    "borderLeft": f"4px solid {rs_overall['border']}",
                    "borderRadius": "6px",
                    "padding": "12px 16px",
                    "display": "inline-flex",
                    "alignItems": "center",
                }),
            ]),
        ),

        section_card(
            *[_narrative_item_web(risk, text) for risk, text in narratives],
            title="Key Risk Assessment",
        ),

        section_card(
            _absorption_capacity_table_web(),
            title="System Absorption Capacity",
        ),

        html.Div(kpi_rows),

    ], style={"padding": "24px"})


def housing_tab(lookback: int | None = 20) -> html.Div:
    H = CHART.graph_height
    return html.Div([
        section_card(prose(
            "The housing market provides a leading indicator of economic conditions. "
            "Mortgage rates directly affect affordability and buyer demand; housing starts "
            "reflect builder confidence and future inventory supply; home prices capture "
            "accumulated demand-supply imbalance.",
            "The combination of elevated rates, reduced starts, and stretched valuations "
            "signals fragility in this sector. Housing typically leads GDP by 12–18 months.",
        )),

        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "MORTGAGE30US",
                    title="30-Year Fixed Mortgage Rate %",
                    threshold_green=5.0, threshold_red=7.0,
                    color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                    lookback_years=lookback or 25, events=_EVENTS_MED), H),
                prose(
                    "Directly controls affordability. Rates above 7% significantly reduce "
                    "the pool of qualifying buyers and suppress transaction volume.",
                ),
            ), md=6),
            dbc.Col(section_card(
                graph(area_chart(dl, "HOUST",
                    title="Housing Starts — Total New Units (Thousands, SAAR)",
                    threshold_green=1300, threshold_red=900,
                    color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                    lookback_years=lookback or 25, events=_EVENTS_MED), H),
                prose(
                    "Monthly annualized rate of new privately-owned construction starts. "
                    "A persistent decline signals deteriorating builder confidence and "
                    "future supply constraints.",
                ),
            ), md=6),
        ], className="g-0"),

        dbc.Row([
            dbc.Col(section_card(
                graph(area_chart(dl, "CSUSHPINSA", yoy=True,
                    title="Case-Shiller National Home Price Index — Year-over-Year %",
                    threshold_green=8, threshold_red=15,
                    color=C["red"], fill_color="rgba(155,44,44,0.09)",
                    lookback_years=lookback or 25, recession_shading=True), H),
                prose(
                    "Annual home price appreciation. Sustained double-digit growth outpacing "
                    "income signals affordability deterioration. Negative readings indicate "
                    "nominal price declines — rare but historically significant.",
                ),
            ), md=8),
            dbc.Col(section_card(
                prose(
                    "Housing is typically a leading indicator: weakness appears in starts "
                    "and permits 12–18 months before GDP slowdown. The 2006–07 episode is "
                    "canonical — housing peaked in 2005, starts collapsed through 2006, "
                    "but recession began December 2007.",
                    "Post-2022 dynamic: Elevated mortgage rates have suppressed existing "
                    "home sales via the 'lock-in effect' — homeowners reluctant to trade "
                    "sub-3% mortgages for 7%+ loans. New construction is the primary "
                    "source of incremental supply.",
                    header="Housing as a Leading Indicator",
                ),
            ), md=4),
        ], className="g-0"),

    ], style={"padding": "24px"})


_GLOSSARY_SECTIONS = [
    ("Inflation", [
        ("CPI",       "Consumer Price Index — average price change for a fixed basket of urban household goods"),
        ("Core CPI",  "CPI excluding food & energy; strips volatile components to show underlying trend"),
        ("Core PCE",  "PCE Price Index ex food & energy — the FOMC's explicit 2% inflation target; published by BEA monthly"),
        ("CPI vs PCE","PCE typically runs 0.3–0.5% below CPI due to different category weights and substitution methodology"),
    ]),
    ("Money Supply", [
        ("M1",       "Narrow money — physical currency plus checking-account deposits"),
        ("M2",       "Broad money — M1 plus savings accounts, money-market funds, and small CDs"),
        ("Real M2",  "M2 adjusted for inflation; sustained contraction signals tightening conditions"),
    ]),
    ("Labor Market", [
        ("Unemployment Rate", "Share of the labor force actively seeking but unable to find work"),
        ("LFPR",             "Labor Force Participation Rate — working-age adults in the labor force (%)"),
        ("Emp/Pop Ratio",    "Share of all working-age adults employed; unaffected by participation shifts"),
        ("Nonfarm Payrolls", "Net monthly job additions across all non-agricultural sectors"),
    ]),
    ("Markets & Rates", [
        ("VIX",               "CBOE Volatility Index — market's 30-day implied volatility; the 'fear gauge'"),
        ("CAPE / Shiller P/E","S&P 500 price divided by 10-year inflation-adjusted average earnings"),
        ("Trailing P/E",      "Stock price divided by last 12 months of reported earnings"),
        ("10Y Treasury Yield","Benchmark long-term government rate; affects all asset prices"),
        ("Yield Curve (10Y-2Y)","Spread between 10-year and 2-year Treasury yields; inversion has preceded every US recession since 1970"),
        ("Inverted Curve",    "When short rates exceed long rates (spread < 0) — markets pricing near-term risk higher than long-run growth"),
        ("NY Fed Rec. Prob.", "Model-based 12-month recession probability using yield curve slope; above 30% has historically been a reliable signal"),
        ("Dollar Index (DTWEX)","Trade-weighted U.S. dollar index vs. 26 currencies; rapid appreciation tightens global financial conditions"),
    ]),
    ("Liquidity & Funding", [
        ("FSI",               "Financial Stress Index — 0 = normal; positive = above-average systemic stress"),
        ("NFCI",              "National Financial Conditions Index — Chicago Fed weekly gauge across 105 variables"),
        ("CP-Tsy Spread",     "3-Month AA Financial CP rate minus 3-Month Treasury — modern successor to the TED spread; measures unsecured bank borrowing premium"),
        ("SOFR",              "Secured Overnight Financing Rate — overnight repo benchmark; LIBOR replacement"),
        ("Reverse Repo (RRP)","Fed facility where banks park excess cash; high usage = ample reserves"),
        ("NFCI Credit",       "NFCI subindex isolating credit conditions specifically"),
    ]),
    ("Credit Markets", [
        ("HY Spread (OAS)", "Extra yield demanded above Treasuries for below-investment-grade bonds"),
        ("IG Spread (OAS)", "Extra yield demanded above Treasuries for investment-grade corporate bonds"),
        ("OAS",             "Option-Adjusted Spread — yield spread net of embedded call/put option value"),
        ("CRE",              "Commercial Real Estate — offices, retail, apartments, and industrial property"),
        ("CRE Lending Stds", "Net % banks tightening CRE loan standards (SLOOS); > 40% signals credit contraction ahead"),
        ("CC Delinquency",   "Share of credit card loans past due — broad consumer credit health signal"),
        ("SLOOS",           "Senior Loan Officer Opinion Survey — quarterly Fed survey on bank lending standards and demand"),
        ("Lending Standards","Net % of banks tightening loan conditions; tightening > 40% historically signals credit contraction ahead"),
    ]),
    ("Policy & Fiscal", [
        ("Fed Funds Rate",    "Fed's overnight policy rate set by the FOMC; the primary monetary tool"),
        ("Real FF Rate",      "Fed Funds Rate minus Core CPI YoY — the inflation-adjusted policy stance"),
        ("WALCL",             "Fed balance sheet total assets; expanded via QE, shrunk via QT"),
        ("QE / QT",           "Quantitative Easing (buying assets) / Tightening (shrinking balance sheet)"),
        ("T5YIE / T10YIE",   "5- and 10-year breakeven inflation rates from TIPS pricing"),
        ("TIPS",              "Treasury bonds whose principal adjusts with CPI; used to extract inflation expectations"),
        ("BEI / Breakeven",   "Market-implied inflation expectations from the gap between TIPS and nominal yields"),
        ("Debt / GDP",        "Federal debt as a percentage of gross domestic product"),
        ("Interest / Receipts","Federal interest payments as a share of total government revenues"),
        ("SAAR",              "Seasonally Adjusted Annual Rate — removes seasonal patterns, expressed at annual pace"),
    ]),
    ("Housing", [
        ("Housing Starts",     "New residential units started monthly (SAAR); a leading indicator for construction employment and materials"),
        ("Case-Shiller HPI",   "S&P/Case-Shiller repeat-sales home price index; published with ~2-month lag"),
        ("Mortgage Rate (30Y)","Freddie Mac Primary Mortgage Market Survey; directly affects housing affordability and purchase volume"),
    ]),
]


def _glossary_entry(term: str, defn: str) -> html.Div:
    return html.Div([
        html.Span(term, style={
            "fontWeight": "600", "fontSize": "12px", "color": C["text"],
            "minWidth": "160px", "display": "inline-block",
        }),
        html.Span(defn, style={
            "fontSize": "12px", "color": C["muted"], "lineHeight": "1.6",
        }),
    ], style={
        "padding": "7px 0",
        "borderBottom": f"1px solid #edf2f7",
        "display": "flex",
        "alignItems": "flex-start",
        "gap": "12px",
    })


def glossary_tab() -> html.Div:
    sections = []
    for section_name, entries in _GLOSSARY_SECTIONS:
        mid = (len(entries) + 1) // 2
        left = entries[:mid]
        right = entries[mid:]
        sections.append(
            section_card(
                dbc.Row([
                    dbc.Col([_glossary_entry(t, d) for t, d in left], md=6),
                    dbc.Col([_glossary_entry(t, d) for t, d in right], md=6),
                ], className="g-0"),
                title=section_name,
            )
        )
    return html.Div([
        section_card(prose(
            "Plain-English definitions for all acronyms and metrics used in this monitor. "
            "Grouped by theme."
        )),
        *sections,
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
            value="overview",
            children=[
                dcc.Tab(label="Overview",            value="overview",    style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Crisis Watch",        value="crisis",      style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="System Resilience",   value="resilience",  style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Liquidity & Funding", value="liquidity",   style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Credit Markets",      value="credit",      style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Policy Constraints",  value="policy",      style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Inflation",           value="inflation",   style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Labor Market",        value="labor",       style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Markets & Rates",     value="markets",     style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Housing",             value="housing",     style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Fiscal",              value="fiscal",      style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                dcc.Tab(label="Glossary",            value="glossary",    style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
            ],
            style={"backgroundColor": "#ffffff", "borderBottom": "1px solid #e2e8f0"},
        ),
        html.Div([
            html.Span("Time Range:", style={
                "fontSize": "11px", "color": "#718096", "marginRight": "10px", "fontWeight": "500",
            }),
            dcc.RadioItems(
                id="lookback-radio",
                options=[
                    {"label": "5Y",  "value": "5"},
                    {"label": "10Y", "value": "10"},
                    {"label": "20Y", "value": "20"},
                    {"label": "Max", "value": "0"},
                ],
                value="10",
                inline=True,
                inputStyle={"marginRight": "4px", "cursor": "pointer"},
                labelStyle={"marginRight": "14px", "fontSize": "12px", "cursor": "pointer", "color": "#4a5568"},
            ),
        ], style={
            "display": "flex", "alignItems": "center", "justifyContent": "flex-end",
            "padding": "6px 20px", "borderBottom": "1px solid #e2e8f0",
            "backgroundColor": "#f7fafc",
        }),
        html.Div(id="tab-content"),
    ], style={"backgroundColor": "#ffffff"}),

], style=STYLE_PAGE)


# ── Callback ──────────────────────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("lookback-radio", "value"),
)
def render_tab(tab: str, lookback_val: str) -> html.Div:
    lb = {"5": 5, "10": 10, "20": 20, "0": None}.get(lookback_val or "10", 10)
    dispatch = {
        "overview":   lambda: executive_summary_tab(),
        "crisis":     lambda: crisis_watch_tab(lb),
        "resilience": lambda: system_resilience_tab(),
        "liquidity":  lambda: liquidity_funding_tab(lb),
        "credit":     lambda: credit_markets_tab(lb),
        "policy":     lambda: policy_constraints_tab(lb),
        "inflation":  lambda: inflation_tab(lb),
        "labor":      lambda: labor_tab(lb),
        "markets":    lambda: markets_rates_tab(lb),
        "housing":    lambda: housing_tab(lb),
        "fiscal":     lambda: fiscal_tab(lb),
        "glossary":   lambda: glossary_tab(),
    }
    return dispatch.get(tab, lambda: executive_summary_tab())()


# ── Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loaded {len(dl.available)} series: {', '.join(sorted(dl.available))}")
    print(f"Starting dashboard at http://{DASH.host}:{DASH.port}")
    app.run(debug=False, host=DASH.host, port=DASH.port)
