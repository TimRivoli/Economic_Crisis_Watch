"""
FRED Economic Monitor — Institutional Dashboard
Run:  python FREDDashboard.py
Then open http://127.0.0.1:8050 in a browser.

Data is loaded from /data at startup and cached in memory.
To refresh data, run FREDDownloader.py and restart this script.
"""

import re as _re
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
	derived_spread_chart, bci_chart, bci_waterfall_chart, momentum_chart,
	backtest_signal_chart, recession_gauge_chart, recession_probability_chart,
	signal_decomposition_chart,
	# Labor analytics
	jolts_chart, wage_productivity_chart, u3_u6_chart,
	labor_deterioration_chart, claims_dashboard_chart,
	# Inflation analytics
	inflation_multi_chart, inflation_expectations_chart,
	shelter_decomposition_chart, sticky_flexible_chart,
	# Financial conditions & banking
	fci_composite_chart, hy_spread_fci_chart,
	delinquency_chart, bank_deposits_chart,
	# Global macro
	central_bank_rates_chart, commodity_chart, fx_chart,
	# Macro regime
	regime_timeline_chart, regime_scores_chart,
	# Structural macro
	output_gap_chart, productivity_chart, real_rates_chart,
	# Fiscal analytics
	debt_service_chart, primary_balance_chart,
	debt_trajectory_chart, fiscal_impulse_chart,
	r_g_chart,
	# Equity valuation
	cape_erp_chart, profit_margin_chart,
)
from _classes.risk_engine import RiskEngine
from _classes.leading_indicators import LeadingIndicatorEngine, BCI_COMPONENTS
from _classes.recession_probability import RecessionProbabilityEngine, HORIZONS as RPE_HORIZONS
from _classes.labor_analytics import LaborAnalyticsEngine
from _classes.inflation_analysis import InflationAnalysisEngine
from _classes.global_macro import GlobalMacroEngine
from _classes.regime_engine import RegimeEngine, REGIMES as MACRO_REGIMES
from _classes.structural_macro import StructuralMacroEngine
from _classes.fiscal_analytics import FiscalAnalyticsEngine
from _classes.equity_valuation import EquityValuationEngine
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
lie = LeadingIndicatorEngine(dl)
rpe = RecessionProbabilityEngine(dl)
rpe.train()
lae = LaborAnalyticsEngine(dl)
iae = InflationAnalysisEngine(dl)
gme = GlobalMacroEngine(dl)
rge = RegimeEngine(dl)
sme = StructuralMacroEngine(dl)
fae = FiscalAnalyticsEngine(dl)
eve = EquityValuationEngine(dl)
crisis_dims = re.crisis_dimensions()
resilience_dims = re.system_resilience_dimensions()
taxonomy_dims = re.risk_taxonomy()

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


def _parse_inline_html(text: str):
	"""Convert a string with <b>...</b> tags into a Dash children list."""
	if not isinstance(text, str) or '<b>' not in text:
		return text
	parts = _re.split(r'(<b>.*?</b>)', text)
	result = []
	for part in parts:
		m = _re.match(r'<b>(.*?)</b>', part, _re.DOTALL)
		if m:
			result.append(html.B(m.group(1)))
		elif part:
			result.append(part)
	return result or text


def prose(*paragraphs: str, header: str | None = None) -> html.Div:
	"""Styled prose block for educational content."""
	children = []
	if header:
		children.append(html.H6(header, style={
			"fontWeight": "600", "color": "#2d3748", "fontSize": "14px", "marginBottom": "10px",
		}))
	for p in paragraphs:
		children.append(html.P(_parse_inline_html(p), style={
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
	lb = lookback or 20
	dash_infl = iae.inflation_dashboard()
	regime_label, regime_color = iae.inflation_regime()
	cycl = iae.cyclical_vs_structural()

	def _fmt(v, fmt=".2f", suffix="%"):
		if v is None:
			return "N/A"
		return f"{v:{fmt}}{suffix}"

	return html.Div([
		# ── Regime Banner ──────────────────────────────────────────────────
		section_card(html.Div([
			html.H5("Inflation Assessment", style={"fontWeight": "700", "color": C["header_bg"], "marginBottom": "8px"}),
			html.Div([
				html.Div([
					html.Span("REGIME", style={"fontSize": "10px", "fontWeight": "700",
							  "letterSpacing": "0.07em", "color": regime_color, "display": "block"}),
					html.Span(regime_label, style={"fontSize": "24px", "fontWeight": "700", "color": regime_color}),
				], style={"marginRight": "40px"}),
				html.Div([
					html.Div([html.Span("Headline CPI: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(dash_infl["headline"]), style={"fontWeight": "600"})]),
					html.Div([html.Span("Core CPI: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(dash_infl["core_cpi"]), style={"fontWeight": "600"})]),
					html.Div([html.Span("Core PCE: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(dash_infl["core_pce"]), style={"fontWeight": "600"})]),
				], style={"marginRight": "40px"}),
				html.Div([
					html.Div([html.Span("Median CPI: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(dash_infl["median_cpi"]), style={"fontWeight": "600"})]),
					html.Div([html.Span("Trimmed Mean PCE: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(dash_infl["trimmed_mean_pce"]), style={"fontWeight": "600"})]),
					html.Div([html.Span("Supercore CPI: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(dash_infl["supercore"]), style={"fontWeight": "600"})]),
				], style={"marginRight": "40px"}),
				html.Div([
					html.Div([html.Span("Sticky CPI: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(dash_infl["sticky_cpi"]), style={"fontWeight": "600"})]),
					html.Div([html.Span("OER YoY: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(dash_infl["oer_yoy"]), style={"fontWeight": "600"})]),
					html.Div([html.Span("Mich Exp (1Y): ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(dash_infl["mich_exp"]), style={"fontWeight": "600"})]),
				]),
			], style={"display": "flex", "alignItems": "flex-start", "flexWrap": "wrap", "gap": "8px"}),
		])),

		# ── Headline & Core ───────────────────────────────────────────────
		html.Div("Headline & Core Inflation", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(area_chart(dl, "CPIAUCSL", yoy=True, lookback_years=lb,
								 threshold_green=2.5, threshold_red=4.5,
								 title="Headline CPI — YoY %")),
			), md=6),
			dbc.Col(section_card(
				graph(multi_line_chart(dl,
					[("CPILFESL", C["teal"]), ("PCEPILFE", C["blue"])],
					title="Core CPI vs. Core PCE — YoY %",
					lookback_years=min(lb, 15), yoy=True,
				)),
				prose("Core PCE (blue) is the Fed's 2% target. It runs ~0.3–0.5% below Core CPI "
					  "(teal) due to weighting differences. Both elevated together = credible signal."),
			), md=6),
		], className="g-0"),

		# ── Alternative Measures ──────────────────────────────────────────
		html.Div("Alternative Inflation Measures", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(inflation_multi_chart(dl, lookback_years=lb)),
				prose("Median CPI (Cleveland Fed) and Trimmed Mean PCE (Dallas Fed) strip extreme "
					  "price movements statistically — better predictors of future inflation than "
					  "Core CPI in academic literature (Stock & Watson, 2008)."),
			), md=8),
			dbc.Col(section_card(
				graph(sticky_flexible_chart(dl, lookback_years=lb)),
				prose("Sticky prices (rents, medical, education) change infrequently and signal "
					  "structural regime shifts. Sticky CPI persistently above 4% = entrenched "
					  "inflation requiring sustained policy response."),
			), md=4),
		], className="g-0"),

		# ── Shelter Decomposition ─────────────────────────────────────────
		html.Div("Shelter Decomposition", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(shelter_decomposition_chart(dl, lookback_years=min(lb, 8))),
				prose("OER (red) is the largest CPI component (~26% weight) and lags actual market "
					  "rents by 12–18 months. Supercore (teal, ex-shelter) shows services inflation "
					  "driven by wages rather than housing cost pass-through."),
			), md=12),
		], className="g-0"),

		# ── Expectations ─────────────────────────────────────────────────
		html.Div("Inflation Expectations", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(inflation_expectations_chart(dl, lookback_years=lb)),
				prose(
					f"Michigan 1-Yr survey: {_fmt(dash_infl['mich_exp'])} | "
					f"5-Yr breakeven: {_fmt(dash_infl['be_5y'])} | "
					f"10-Yr breakeven: {_fmt(dash_infl['be_10y'])}. "
					"Breakevens >2.5% or deanchoring above 4% on Michigan survey signal "
					"credibility erosion — historically requires more aggressive Fed response."),
			), md=12),
		], className="g-0"),

		# ── Money Supply ──────────────────────────────────────────────────
		html.Div("Money Supply", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(multi_line_chart(dl,
					[("M1SL", C["blue"]), ("M2SL", C["teal"])],
					title="M1 & M2 Money Supply — Level ($B)",
					lookback_years=lb)),
			), md=8),
			dbc.Col(section_card(
				graph(area_chart(dl, "M2REAL", yoy=True, lookback_years=lb,
								 color=C["teal"], fill_color="rgba(44,122,123,0.09)",
								 title="Real M2 Growth — YoY %")),
				prose("Real M2 contraction signals tightening monetary conditions. "
					  "Rapid M2 growth precedes inflation by 12–18 months."),
			), md=4),
		], className="g-0"),
	], style={"padding": "24px"})


def labor_tab(lookback: int | None = 20) -> html.Div:
	lb = lookback or 20
	ldi = lae.labor_deterioration_index()
	ldi_score = ldi.get("score")
	ldi_color = "#276749" if (ldi_score or 0) < 35 else ("#975a16" if (ldi_score or 0) < 65 else "#9b2c2c")
	ldi_label = "Healthy" if (ldi_score or 0) < 35 else ("Softening" if (ldi_score or 0) < 65 else "Stressed")

	jolts = lae.jolts_summary()
	wages = lae.wage_pressure()
	claims = lae.claims_summary()
	ugap = lae.unemployment_gap()

	def _fmt(v, fmt=".1f", suffix=""):
		if v is None:
			return "N/A"
		return f"{v:{fmt}}{suffix}"

	return html.Div([
		# ── Deterioration Index Banner ────────────────────────────────────
		section_card(html.Div([
			html.H5("Labor Market Assessment", style={"fontWeight": "700", "color": C["header_bg"], "marginBottom": "8px"}),
			html.Div([
				html.Div([
					html.Span("DETERIORATION INDEX", style={"fontSize": "10px", "fontWeight": "700",
							  "letterSpacing": "0.07em", "color": ldi_color, "display": "block"}),
					html.Span(f"{_fmt(ldi_score, '.0f')}/100", style={"fontSize": "28px",
							  "fontWeight": "700", "color": ldi_color}),
					html.Span(f" — {ldi_label}", style={"fontSize": "14px", "color": ldi_color}),
				], style={"marginRight": "40px"}),
				html.Div([
					html.Div([html.Span("U-3: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(ugap["u3"], ".1f", "%"), style={"fontWeight": "600", "fontSize": "14px"})]),
					html.Div([html.Span("U-6: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(ugap["u6"], ".1f", "%"), style={"fontWeight": "600", "fontSize": "14px"})]),
					html.Div([html.Span("U6-U3 Gap: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(ugap["gap"], ".1f", "pp"), style={"fontWeight": "600", "fontSize": "14px"})]),
				], style={"marginRight": "40px"}),
				html.Div([
					html.Div([html.Span("Quits Rate: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(jolts["quits_rate"], ".2f", "%"), style={"fontWeight": "600", "fontSize": "14px"})]),
					html.Div([html.Span("Layoffs Rate: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(jolts["layoffs_rate"], ".2f", "%"), style={"fontWeight": "600", "fontSize": "14px"})]),
					html.Div([html.Span("Beveridge: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(jolts["beveridge_diagnosis"][:40], style={"fontWeight": "600", "fontSize": "12px"})]),
				], style={"marginRight": "40px"}),
				html.Div([
					html.Div([html.Span("AHE YoY: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(wages["ahe_yoy"], ".1f", "%"), style={"fontWeight": "600", "fontSize": "14px"})]),
					html.Div([html.Span("Real Wage: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(wages["real_wage"], "+.1f", "%"), style={"fontWeight": "600", "fontSize": "14px"})]),
					html.Div([html.Span("Unit Labor Costs: ", style={"color": C["muted"], "fontSize": "12px"}),
							  html.Span(_fmt(wages["ulc_yoy"], ".1f", "% YoY"), style={"fontWeight": "600", "fontSize": "14px"})]),
				]),
			], style={"display": "flex", "alignItems": "flex-start", "flexWrap": "wrap", "gap": "8px"}),
		])),

		# ── Unemployment & Payrolls ───────────────────────────────────────
		html.Div("Unemployment & Employment", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(graph(u3_u6_chart(dl, lookback_years=lb))),  md=8),
			dbc.Col(section_card(
				graph(bar_change_chart(dl, "PAYEMS", lookback_years=min(lb, 5),
									   title="Nonfarm Payrolls — Monthly Change (Thousands)")),
				prose("Monthly job additions. Below 100K signals softening; negative signals contraction. "
					  "Subject to large revisions (±100K typical)."),
			), md=4),
		], className="g-0"),
		dbc.Row([
			dbc.Col(section_card(
				graph(multi_line_chart(dl, [("EMRATIO", C["blue"]), ("CIVPART", C["teal"])],
					title="Employment-Population Ratio vs. Participation Rate",
					lookback_years=lb)),
				prose("Participation rate captures labor supply; E/P ratio is the purer cyclical "
					  "signal (excludes structural demographic shifts from the denominator)."),
			), md=6),
			dbc.Col(section_card(
				graph(area_chart(dl, "INDPRO", yoy=True, lookback_years=lb,
								 color=C["blue"], fill_color="rgba(43,108,176,0.10)",
								 threshold_green=1, threshold_red=-1,
								 title="Industrial Production — YoY %", recession_shading=True)),
				prose("Manufacturing, mining, and utilities output. Consecutive negative monthly "
					  "readings historically precede recession. Released mid-month."),
			), md=6),
		], className="g-0"),

		# ── JOLTS & Claims ───────────────────────────────────────────────
		html.Div("JOLTS & Jobless Claims", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(jolts_chart(dl, lookback_years=lb)),
				prose("Top: Job Openings (thousands). Middle: Quits Rate — workers quit when "
					  "confident of finding better work; drops lead unemployment by 3–6 months. "
					  "Bottom: Layoffs Rate — rising signals employers cutting ahead of slowdown."),
			), md=8),
			dbc.Col(section_card(
				graph(claims_dashboard_chart(dl, lookback_years=min(lb, 8))),
				prose("Initial Claims (left): timeliest weekly labor read. Continued Claims (right): "
					  "absorption rate — how quickly laid-off workers are rehired. "
					  f"Absorption ratio (cont./init.): {_fmt(claims.get('absorption_ratio'), '.1f')}x."),
			), md=4),
		], className="g-0"),

		# ── Wages, Productivity & ULC ─────────────────────────────────────
		html.Div("Wages, Productivity & Unit Labor Costs", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(wage_productivity_chart(dl, lookback_years=lb)),
				prose("Wages above productivity growth = rising Unit Labor Costs (ULC). ULC "
					  "is the primary transmission channel from labor costs to goods prices. "
					  f"Productivity YoY: {_fmt(wages.get('prod_yoy'), '.1f', '%')}. "
					  f"ECI YoY: {_fmt(wages.get('eci_yoy'), '.1f', '%')}."),
			), md=8),
			dbc.Col(section_card(
				graph(labor_deterioration_chart(lae, lookback_years=lb)),
			), md=4),
		], className="g-0"),

		# ── Savings & Leading ─────────────────────────────────────────────
		html.Div("Household Resilience & Leading Signals", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(area_chart(dl, "PSAVERT", lookback_years=lb,
								 color=C["teal"], fill_color="rgba(44,122,123,0.09)",
								 threshold_green=7, threshold_red=4,
								 title="Personal Saving Rate %")),
				prose("Low savings signal households drawing down buffers to maintain consumption — "
					  "a late-cycle vulnerability. Post-COVID excess savings now largely exhausted."),
			), md=6),
			dbc.Col(section_card(
				graph(area_chart(dl, "USSLIND", lookback_years=lb,
								 color=C["red"], fill_color="rgba(155,44,44,0.09)",
								 threshold_green=0, threshold_red=-5,
								 title="Conference Board LEI (MoM %)")),
				prose("Aggregates 10 forward-looking components. Consecutive monthly declines have "
					  "historically signaled recession 6–12 months ahead."),
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
					"Elevated CAPE alongside compressed VIX indicates that valuations are "
					"stretched relative to near-term risk pricing. High CAPE is a long-run "
					"return predictor, not a short-term crash signal; low VIX reflects "
					"current market calm and may persist. The combination warrants attention "
					"but does not imply imminent correction.",
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
				graph(cape_erp_chart(eve, dl, lookback_years=lookback or 25)),
				prose(
					"CAPE in isolation is insufficient: a ratio of 30 is cheap when real "
					"yields are −1% (ERP ~4.3%) and extremely expensive when TIPS yields "
					"are +3% (ERP ~0.3%). The lower panel shows the equity risk premium — "
					"earnings yield minus 10Y TIPS yield. Negative ERP means equities price "
					"in returns below the risk-free real rate, the historical signal for "
					"poor long-run forward equity returns.",
				),
			), md=8),
			dbc.Col(section_card(
				graph(profit_margin_chart(eve, dl, lookback_years=lookback or 40)),
				prose(
					"Corporate profits as % of GDP. Above-average margins inflate CAPE "
					"numerically — the same earnings multiple looks more expensive if "
					"margins mean-revert to their long-run average. A CAPE of 30 built "
					"on 14%-of-GDP margins carries more valuation risk than one built on "
					"the historical ~7% average.",
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
			"A widening CP-Treasury spread signals rising interbank credit cost. "
			"Declining RRP balances reflect shifting reserve distribution — this can "
			"stem from reserve redistribution, Treasury bill issuance absorbing MMF "
			"demand, or genuine liquidity reduction; context determines which.",
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

		html.Div("Composite Financial Conditions", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(fci_composite_chart(dl, lookback_years=lookback or 20)),
				prose("NFCI (blue) and St. Louis FSI (red) both measure the tightness of "
					  "overall financial conditions. Values above zero indicate tighter-than-average "
					  "conditions. Red shading above zero marks stress territory. The two indices "
					  "use different methodologies and can diverge."),
			), md=12),
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

		# ── Credit Tightening Channel ─────────────────────────────────────
		html.Div("Credit Spreads & Lending Transmission", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(hy_spread_fci_chart(dl, lookback_years=lookback or 15)),
				prose("Top panel: HY (red) and IG (blue) OAS spreads showing risk premium across "
					  "the credit quality spectrum. Bottom: C&I lending standards (net % tightening) "
					  "— green bars = net easing, red bars = net tightening. "
					  "Rising spreads + tightening standards = dual contraction in credit access."),
			), md=12),
		], className="g-0"),

		# ── Banking Stress ────────────────────────────────────────────────
		html.Div("Banking & Credit Quality", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(delinquency_chart(dl, lookback_years=lookback or 15)),
				prose("Loan delinquency rates across all segments: all loans (blue), commercial "
					  "real estate (red), and residential mortgages (amber). Quarterly data from "
					  "FFIEC Call Reports. CRE delinquencies concentrated at regional banks — "
					  "office/retail vacancy surge is an ongoing risk. Alert threshold at 2.5%."),
			), md=8),
			dbc.Col(section_card(
				graph(bank_deposits_chart(dl, lookback_years=min(lookback or 10, 10))),
				prose("System-wide bank deposit levels and YoY growth. Rapid deposit outflows "
					  "indicate either tightening monetary conditions or bank stress. "
					  "SVB 2023 demonstrated that modern bank runs can be swift."),
			), md=4),
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


def leading_indicators_tab(lookback: int | None = 15) -> html.Div:
	bci_val, phase = lie.current_bci()
	bci_str = f"{bci_val:.2f}" if bci_val is not None else "N/A"

	_phase_style = {
		"Expansion":   ("#f0fff4", "#276749"),
		"Slowdown":    ("#fffff0", "#975a16"),
		"Contraction": ("#fff5f5", "#9b2c2c"),
		"Recovery":    ("#ebf8ff", "#2b6cb0"),
	}
	phase_bg, phase_text = _phase_style.get(phase, ("#f7fafc", "#718096"))

	mom_data = lie.all_momentum()

	def _mom_row(sid: str) -> html.Tr | None:
		m = mom_data.get(sid)
		if not m:
			return None
		meta = REGISTRY.get(sid, {})
		direction = m.get("direction", "—")
		dir_color = {"Improving": "#276749", "Deteriorating": "#9b2c2c"}.get(direction, "#718096")
		cell = {"fontSize": "12px", "padding": "6px 8px", "fontFamily": "monospace",
				"borderBottom": "1px solid #edf2f7"}
		return html.Tr([
			html.Td(meta.get("short_name", sid),
					style={**cell, "fontFamily": "inherit", "fontWeight": "500"}),
			html.Td(f"{m.get('current', 0):.2f}",       style={**cell, "textAlign": "right"}),
			html.Td(f"{m.get('trend_3m', 0):+.3f}",     style={**cell, "textAlign": "right"}),
			html.Td(f"{m.get('trend_6m', 0):+.3f}",     style={**cell, "textAlign": "right"}),
			html.Td(f"{m.get('acceleration', 0):+.3f}",  style={**cell, "textAlign": "right"}),
			html.Td(f"{m.get('z_vs_1y', 0):+.2f}σ", style={**cell, "textAlign": "right"}),
			html.Td(
				html.Span(direction, style={"color": dir_color, "fontWeight": "600",
											"fontSize": "11px"}),
				style={**cell, "fontFamily": "inherit"},
			),
		])

	_tbl_hdr_style = {
		"fontSize": "10px", "fontWeight": "600", "textTransform": "uppercase",
		"letterSpacing": "0.05em", "color": C["muted"], "padding": "6px 8px",
		"backgroundColor": "#f7fafc", "borderBottom": "2px solid #e2e8f0",
	}
	mom_header = html.Tr([
		html.Th(h, style=_tbl_hdr_style)
		for h in ["Series", "Current", "3M Trend", "6M Trend", "Accel.", "Z vs 1Y", "Direction"]
	])
	mom_rows = [r for sid in BCI_COMPONENTS for r in [_mom_row(sid)] if r]

	bt_data = lie.all_backtests()

	def _bt_row(sid: str) -> html.Tr | None:
		bt = bt_data.get(sid, {})
		if not bt or bt.get("n_recessions", 0) == 0:
			return None
		meta = REGISTRY.get(sid, {})
		hit  = bt.get("hit_rate")
		lead = bt.get("avg_lead_months")
		fp   = bt.get("false_pos_rate")
		hit_color = ("#276749" if hit and hit >= 80
					 else "#975a16" if hit and hit >= 60 else "#9b2c2c")
		cell = {"fontSize": "12px", "padding": "6px 8px",
				"borderBottom": "1px solid #edf2f7"}
		return html.Tr([
			html.Td(meta.get("short_name", sid),
					style={**cell, "fontWeight": "500"}),
			html.Td(
				html.Span(f"{hit:.0f}%" if hit is not None else "N/A",
						  style={"color": hit_color, "fontWeight": "600"}),
				style={**cell, "textAlign": "right"},
			),
			html.Td(f"{lead:.1f}" if lead is not None else "N/A",
					style={**cell, "textAlign": "right", "fontFamily": "monospace"}),
			html.Td(f"{fp:.0f}%" if fp is not None else "N/A",
					style={**cell, "textAlign": "right", "fontFamily": "monospace"}),
			html.Td(str(bt.get("n_recessions", 0)),
					style={**cell, "textAlign": "center"}),
		])

	bt_header = html.Tr([
		html.Th(h, style=_tbl_hdr_style)
		for h in ["Series", "Hit Rate", "Avg Lead (Mo.)", "False Pos. Rate", "Recessions"]
	])
	bt_rows = [r for sid in BCI_COMPONENTS for r in [_bt_row(sid)] if r]

	H = CHART.graph_height
	Hs = CHART.graph_height_small + 60

	return html.Div([
		# ── Header ────────────────────────────────────────────────────────
		section_card(
			html.Div([
				html.H5("Leading Indicators & Business Cycle Index", style={
					"fontWeight": "700", "color": C["header_bg"], "marginBottom": "8px",
				}),
				html.P(
					"The Composite Business Cycle Index (BCI) aggregates nine leading macro indicators "
					"into a single standardized score. Each component is z-scored against its 20-year "
					"calibration window, inverted where lower values signal deterioration, then combined "
					"as a weighted average and smoothed with a 3-month moving average. "
					"Phase classification uses BCI level × 3-month momentum direction.",
					style={"fontSize": "13px", "color": C["slate"], "lineHeight": "1.7",
						   "marginBottom": "14px"},
				),
				html.Div([
					html.Span("CURRENT PHASE", style={
						"fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.07em",
						"textTransform": "uppercase", "color": phase_text, "marginRight": "12px",
					}),
					html.Span(phase, style={
						"fontSize": "20px", "fontWeight": "700", "color": phase_text,
						"marginRight": "14px",
					}),
					html.Span(f"BCI: {bci_str}", style={
						"fontSize": "14px", "fontWeight": "500", "color": phase_text,
						"opacity": "0.85",
					}),
				], style={
					"backgroundColor": phase_bg,
					"border": f"1px solid {phase_text}44",
					"borderLeft": f"4px solid {phase_text}",
					"borderRadius": "6px",
					"padding": "12px 18px",
					"display": "inline-flex",
					"alignItems": "center",
				}),
			]),
		),

		# ── BCI chart (left) + component contributions (right) ────────────
		dbc.Row([
			dbc.Col(section_card(
				graph(bci_chart(lie, dl, lookback_years=lookback or 15), H + 40),
				prose(
					"Positive BCI (above zero) = above-trend conditions; negative = below-trend. "
					"Background shading encodes the current phase. Grey bands = NBER recessions. "
					"The dashed line at zero is the neutral threshold.",
				),
			), md=8),
			dbc.Col(section_card(
				graph(bci_waterfall_chart(lie, dl), H + 40),
				prose(
					"Current z-score × weight contribution of each component. "
					"Green bars pull the BCI positive (expansion); red bars pull it negative.",
				),
			), md=4),
		], className="g-0"),

		# ── Phase legend ──────────────────────────────────────────────────
		section_card(
			dbc.Row([
				dbc.Col(html.Div([
					html.Div("Expansion", style={"fontWeight": "700", "color": "#276749",
												  "fontSize": "12px", "marginBottom": "4px"}),
					html.P("BCI > 0, momentum positive. Above-trend growth accelerating. "
						   "Pro-cyclical positioning.",
						   style={"fontSize": "12px", "color": C["slate"],
								  "margin": "0", "lineHeight": "1.6"}),
				], style={"borderLeft": "4px solid #276749", "paddingLeft": "10px"}), md=3),
				dbc.Col(html.Div([
					html.Div("Slowdown", style={"fontWeight": "700", "color": "#975a16",
												 "fontSize": "12px", "marginBottom": "4px"}),
					html.P("BCI > 0, momentum negative. Above-trend but decelerating. "
						   "Policy pivot risk rising.",
						   style={"fontSize": "12px", "color": C["slate"],
								  "margin": "0", "lineHeight": "1.6"}),
				], style={"borderLeft": "4px solid #975a16", "paddingLeft": "10px"}), md=3),
				dbc.Col(html.Div([
					html.Div("Recovery", style={"fontWeight": "700", "color": "#2b6cb0",
												 "fontSize": "12px", "marginBottom": "4px"}),
					html.P("BCI ≤ 0, momentum positive. Below-trend but improving. "
						   "Early-cycle positioning.",
						   style={"fontSize": "12px", "color": C["slate"],
								  "margin": "0", "lineHeight": "1.6"}),
				], style={"borderLeft": "4px solid #2b6cb0", "paddingLeft": "10px"}), md=3),
				dbc.Col(html.Div([
					html.Div("Contraction", style={"fontWeight": "700", "color": "#9b2c2c",
													"fontSize": "12px", "marginBottom": "4px"}),
					html.P("BCI ≤ 0, momentum negative. Below-trend and worsening. "
						   "Defensive positioning. Recession risk elevated.",
						   style={"fontSize": "12px", "color": C["slate"],
								  "margin": "0", "lineHeight": "1.6"}),
				], style={"borderLeft": "4px solid #9b2c2c", "paddingLeft": "10px"}), md=3),
			], className="g-3"),
			title="Business Cycle Phase Framework",
		),

		# ── Trend & Momentum table ────────────────────────────────────────
		section_card(
			html.Table(
				[mom_header] + mom_rows,
				style={"width": "100%", "borderCollapse": "collapse"},
			),
			prose(
				"3M/6M trend slopes are per-month linear regression coefficients on the series' risk basis. "
				"Acceleration = trend_3m − trend_6m: positive means the series is accelerating in "
				"its current direction. Z vs 1Y: standard deviations from 12-month mean. "
				"Direction is assessed relative to each series' higher_is_bad orientation.",
			),
			title="Trend & Momentum Analytics — All BCI Components",
		),

		# ── Momentum charts: macro activity ───────────────────────────────
		html.Div("Leading Activity Indicators", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(momentum_chart(dl, "USSLIND",   lookback_years=lookback or 10), Hs),
			), md=6),
			dbc.Col(section_card(
				graph(momentum_chart(dl, "AMTMNO",    lookback_years=lookback or 10), Hs),
			), md=6),
		], className="g-0"),
		dbc.Row([
			dbc.Col(section_card(
				graph(momentum_chart(dl, "PERMIT",    lookback_years=lookback or 10), Hs),
			), md=6),
			dbc.Col(section_card(
				graph(momentum_chart(dl, "TEMPHELPS", lookback_years=lookback or 10), Hs),
			), md=6),
		], className="g-0"),

		# ── Momentum charts: financial/credit ─────────────────────────────
		html.Div("Financial & Credit Leading Indicators", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(momentum_chart(dl, "ICSA",         lookback_years=lookback or 10), Hs),
			), md=6),
			dbc.Col(section_card(
				graph(momentum_chart(dl, "BAMLH0A0HYM2", lookback_years=lookback or 10), Hs),
			), md=6),
		], className="g-0"),
		dbc.Row([
			dbc.Col(section_card(
				graph(momentum_chart(dl, "DRTSCILM", lookback_years=lookback or 10), Hs),
			), md=6),
			dbc.Col(section_card(
				graph(momentum_chart(dl, "T10Y2Y",   lookback_years=lookback or 10), Hs),
			), md=6),
		], className="g-0"),

		# ── Backtest signal charts ─────────────────────────────────────────
		html.Div("Historical Signal Validation", style=STYLE_SECTION_LABEL),
		dbc.Row([
			dbc.Col(section_card(
				graph(backtest_signal_chart(lie, dl, "USSLIND",   lookback_years=lookback or 20), H),
			), md=6),
			dbc.Col(section_card(
				graph(backtest_signal_chart(lie, dl, "ICSA",      lookback_years=lookback or 20), H),
			), md=6),
		], className="g-0"),
		dbc.Row([
			dbc.Col(section_card(
				graph(backtest_signal_chart(lie, dl, "BAMLH0A0HYM2", lookback_years=lookback or 20), H),
			), md=6),
			dbc.Col(section_card(
				graph(backtest_signal_chart(lie, dl, "T10Y2Y",    lookback_years=lookback or 20), H),
			), md=6),
		], className="g-0"),

		# ── Historical recession backtest table ───────────────────────────
		section_card(
			html.Table(
				[bt_header] + bt_rows,
				style={"width": "100%", "borderCollapse": "collapse"},
			),
			prose(
				"In-sample backtest vs. NBER recession dates. "
				"Hit rate: share of NBER recessions preceded by a stressed signal within 18 months. "
				"Avg lead: months of advance warning before recession onset. "
				"False positive rate: signals not followed by recession within 12 months. "
				"High hit rate + low false positive rate = reliable leading indicator.",
				"Note: statistics reflect post-hoc NBER dating. Initial data revisions "
				"may introduce mild lookahead bias. Use for relative comparison.",
			),
			title="Historical Recession Validation (Backtest)",
		),

	], style={"padding": "24px"})


def recession_probability_tab(lookback: int | None = 20) -> html.Div:
	probs = rpe.current_probabilities()
	H = CHART.graph_height

	def _prob_badge(h: int) -> html.Div:
		p = probs.get(h)
		pct = round(p * 100, 1) if p is not None else None
		label = f"{pct:.1f}%" if pct is not None else "N/A"
		if pct is None:   bg, txt, border = "#f7fafc", "#718096", "#cbd5e0"
		elif pct >= 50:   bg, txt, border = "#fff5f5", "#9b2c2c", "#fc8181"
		elif pct >= 20:   bg, txt, border = "#fffff0", "#975a16", "#f6ad55"
		else:             bg, txt, border = "#f0fff4", "#276749", "#68d391"
		horizon_label = {6: "6-Month", 12: "12-Month", 24: "24-Month"}.get(h, f"{h}M")
		return html.Div([
			html.Div(horizon_label, style={
				"fontSize": "10px", "fontWeight": "600", "textTransform": "uppercase",
				"letterSpacing": "0.06em", "color": txt, "marginBottom": "6px",
			}),
			html.Div(label, style={
				"fontSize": "32px", "fontWeight": "700", "color": txt, "lineHeight": "1",
				"marginBottom": "4px",
			}),
			html.Div("recession probability", style={
				"fontSize": "10px", "color": txt, "opacity": "0.7",
			}),
		], style={
			"backgroundColor": bg,
			"border": f"1px solid {border}",
			"borderLeft": f"4px solid {border}",
			"borderRadius": "6px",
			"padding": "18px 20px",
			"textAlign": "center",
			"flex": "1",
		})

	# Backtest metrics table
	bt_all = rpe.all_backtests()

	def _bt_row(h: int) -> html.Tr | None:
		bt = bt_all.get(h, {})
		if not bt:
			return None
		hit_color = ("#276749" if (bt.get("hit_rate") or 0) >= 70
					 else "#975a16" if (bt.get("hit_rate") or 0) >= 50
					 else "#9b2c2c")
		cell = {"fontSize": "12px", "padding": "6px 10px",
				"borderBottom": "1px solid #edf2f7"}
		return html.Tr([
			html.Td(f"{h}-Month", style={**cell, "fontWeight": "600"}),
			html.Td(
				html.Span(f"{bt.get('auc', 0):.3f}",
						  style={"fontFamily": "monospace", "fontWeight": "600",
								 "color": "#2b6cb0"}),
				style={**cell, "textAlign": "right"},
			),
			html.Td(
				html.Span(f"{bt.get('hit_rate', 0):.0f}%",
						  style={"color": hit_color, "fontWeight": "600"}),
				style={**cell, "textAlign": "right"},
			),
			html.Td(f"{bt.get('precision', 0):.0f}%",
					style={**cell, "textAlign": "right", "fontFamily": "monospace"}),
			html.Td(f"{bt.get('recall', 0):.0f}%",
					style={**cell, "textAlign": "right", "fontFamily": "monospace"}),
			html.Td(f"{bt.get('fp_rate', 0):.0f}%",
					style={**cell, "textAlign": "right", "fontFamily": "monospace"}),
			html.Td(f"{bt.get('n_signals', 0)} / {bt.get('n_test_obs', 0)}",
					style={**cell, "textAlign": "right", "fontFamily": "monospace"}),
		])

	_tbl_hdr = {"fontSize": "10px", "fontWeight": "600", "textTransform": "uppercase",
				"letterSpacing": "0.05em", "color": C["muted"], "padding": "6px 10px",
				"backgroundColor": "#f7fafc", "borderBottom": "2px solid #e2e8f0"}
	bt_header = html.Tr([html.Th(h, style=_tbl_hdr)
						 for h in ["Horizon", "AUC", "Hit Rate", "Precision",
								   "Recall", "FP Rate", "Signals / Obs."]])
	bt_rows = [r for h in RPE_HORIZONS for r in [_bt_row(h)] if r]

	return html.Div([
		# ── Header ────────────────────────────────────────────────────────
		section_card(
			html.H5("Recession Probability Model", style={
				"fontWeight": "700", "color": C["header_bg"], "marginBottom": "8px",
			}),
			html.P(
				"Logistic regression model trained on NBER recession dates using eight "
				"macro features: yield curve (10Y-2Y, 10Y-3M), Conference Board LEI momentum, "
				"initial jobless claims, ISM new orders, HY credit spreads, unemployment "
				"rate change, and payroll growth. Separate models are trained for 6M, 12M, "
				"and 24M horizons. L2 regularization prevents overfitting on the limited "
				"recession sample. Backtesting uses rolling 20-year training windows.",
				style={"fontSize": "13px", "color": C["slate"], "lineHeight": "1.7",
					   "marginBottom": "0"},
			),
		),

		# ── Probability badges ─────────────────────────────────────────────
		section_card(
			html.Div([_prob_badge(h) for h in RPE_HORIZONS],
					 style={"display": "flex", "gap": "16px"}),
			title="Current Recession Probability",
		),

		# ── Gauges ────────────────────────────────────────────────────────
		section_card(
			graph(recession_gauge_chart(probs), height=220),
		),

		# ── Rolling probability + signal decomposition ─────────────────────
		dbc.Row([
			dbc.Col(section_card(
				graph(recession_probability_chart(rpe, dl,
					  lookback_years=lookback or 20, with_bands=True), H + 20),
				prose(
					"All three horizon probabilities with NBER recession shading (grey). "
					"Dashed line = 35% classification threshold. Shaded band shows the "
					"90% bootstrap confidence interval for the 12-month model.",
				),
			), md=7),
			dbc.Col(section_card(
				graph(signal_decomposition_chart(rpe, horizon=12), H + 20),
				prose(
					"Log-odds contribution of each feature to the current 12-month "
					"probability. Red = pushes probability up; teal = pushes it down. "
					"Magnitude reflects both the standardized feature value and the "
					"learned regression weight.",
				),
			), md=5),
		], className="g-0"),

		# ── Model interpretation ──────────────────────────────────────────
		section_card(
			dbc.Row([
				dbc.Col(prose(
					"The model is trained on NBER post-hoc recession dates, which are "
					"published months after recessions begin. This creates an inherent "
					"asymmetry: the model is optimized to anticipate recessions before "
					"they are officially declared. The 35% threshold reflects the "
					"historical base rate of 'recession within 12 months' being roughly "
					"18% — the threshold is deliberately set above base rate to balance "
					"false positives against practical utility.",
					header="Model Interpretation",
				), md=6),
				dbc.Col(prose(
					"The logistic regression is linear in the feature space after "
					"standardization. It cannot capture interaction effects (e.g., yield "
					"curve inversion matters more when combined with credit spread widening) "
					"without explicit feature engineering. The signal decomposition chart "
					"shows each feature's standalone contribution — treat the model as one "
					"structured lens, not a definitive forecast.",
					header="Limitations",
				), md=6),
			], className="g-4"),
		),

		# ── Backtest metrics ───────────────────────────────────────────────
		section_card(
			html.Table(
				[bt_header] + bt_rows,
				style={"width": "100%", "borderCollapse": "collapse"},
			) if bt_rows else html.P(
				"Backtest requires NBER recession data and sufficient history. "
				"Run FREDDownloader.py to populate data.",
				style={"fontSize": "12px", "color": C["muted"], "padding": "12px"},
			),
			prose(
				"Rolling out-of-sample backtest: trained on 20 years of data prior to "
				"each test date, evaluated from 2000 onward. "
				"AUC (rank-sum): area under the ROC curve — 0.5 = random, 1.0 = perfect. "
				"Hit rate: recession months correctly flagged above threshold. "
				"Precision: of flagged periods, fraction that preceded recession. "
				"Recall: of recession-preceding months, fraction that were flagged. "
				"FP rate: flagged periods not followed by recession within the horizon.",
			),
			title="Out-of-Sample Backtest Performance (Rolling 20-Year Windows, Test from 2000)",
		),

	], style={"padding": "24px"})


def risk_scorecard_tab() -> html.Div:
	"""
	Four cleanly separated risk categories — no mixing of cyclical, financial,
	valuation, and fiscal risk signals.
	"""
	probs = rpe.current_probabilities()
	p12 = probs.get(12)
	p12_pct = round(p12 * 100, 1) if p12 is not None else None

	overall_tax = re.overall_stress_level()
	rs_overall = RISK_STYLE.get(overall_tax, RISK_STYLE["neutral"])

	def _category_card(name: str, dim: dict) -> dbc.Col:
		risk = dim["score"]
		rs = RISK_STYLE[risk]
		comp_rows = [
			html.Div([
				html.Span(label, style={"fontSize": "11px", "color": C["slate"]}),
				html.Div([
					html.Span(display, style={"fontSize": "11px", "fontWeight": "600",
											  "color": RISK_STYLE[r]["text"],
											  "textAlign": "right"}),
					html.Div(f"as of {as_of}" if as_of else "",
							 style={"fontSize": "9px", "color": C["muted"],
									"textAlign": "right"}) if as_of else None,
				], style={"marginLeft": "auto"}),
			], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"})
			for label, r, display, as_of in dim["components"]
		]
		return dbc.Col(html.Div([
			html.Div(name, style={
				"fontSize": "10px", "fontWeight": "700", "letterSpacing": "0.07em",
				"textTransform": "uppercase", "color": rs["text"], "marginBottom": "6px",
			}),
			html.Div(rs["label"], style={
				"fontSize": "22px", "fontWeight": "700",
				"color": rs["text"], "marginBottom": "10px",
			}),
			html.Div(comp_rows),
			html.Div(dim["description"], style={
				"fontSize": "11px", "color": rs["text"], "opacity": "0.75",
				"marginTop": "10px", "lineHeight": "1.5",
				"borderTop": f"1px solid {rs['border']}", "paddingTop": "8px",
			}),
		], style={
			"backgroundColor": rs["bg"],
			"border": f"1px solid {rs['border']}",
			"borderLeft": f"4px solid {rs['border']}",
			"borderRadius": "6px",
			"padding": "16px",
			"height": "100%",
		}), md=3, sm=6, xs=12)

	return html.Div([
		# ── Header ────────────────────────────────────────────────────────
		section_card(
			html.Div([
				html.H5("Risk Taxonomy Scorecard", style={
					"fontWeight": "700", "color": C["header_bg"], "marginBottom": "8px",
				}),
				html.P(
					"Four cleanly separated risk categories. Cyclical recession risk "
					"reflects leading macro signals. Financial stability risk reflects "
					"credit and funding stress, which can exist independently of the "
					"business cycle. Valuation risk is a long-run return predictor, not "
					"a recession trigger. Fiscal & policy risk measures the constraints "
					"on stabilization tools.",
					style={"fontSize": "13px", "color": C["slate"], "lineHeight": "1.7",
						   "marginBottom": "14px"},
				),
				html.Div([
					html.Span("OVERALL STRESS", style={
						"fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.07em",
						"textTransform": "uppercase", "color": rs_overall["text"],
						"marginRight": "12px",
					}),
					html.Span(rs_overall["label"].upper(), style={
						"fontSize": "18px", "fontWeight": "700", "color": rs_overall["text"],
						"marginRight": "20px",
					}),
					html.Span(
						f"Model Recession Probability (12M): {p12_pct:.1f}%" if p12_pct is not None else "",
						style={"fontSize": "12px", "color": rs_overall["text"], "opacity": "0.75"},
					),
				], style={
					"backgroundColor": rs_overall["bg"],
					"border": f"1px solid {rs_overall['border']}",
					"borderLeft": f"4px solid {rs_overall['border']}",
					"borderRadius": "6px", "padding": "12px 16px",
					"display": "inline-flex", "alignItems": "center",
				}),
			]),
		),

		# ── 4-category scorecards ─────────────────────────────────────────
		dbc.Row(
			[_category_card(name, dim) for name, dim in taxonomy_dims.items()],
			className="g-3 mb-4",
		),

		# ── Why the taxonomy matters ──────────────────────────────────────
		section_card(
			dbc.Row([
				dbc.Col(prose(
					"Mixing cyclical and structural risk signals in a single composite "
					"score produces a misleading reading. A low-yield-curve + wide-credit-spread "
					"environment can look 'elevated risk' on a combined index even when the "
					"signals reflect entirely different mechanisms (monetary tightening vs. "
					"credit deterioration). Separating them allows each to be acted upon independently.",
					header="Why Separate Risk Types?",
				), md=6),
				dbc.Col(prose(
					"The four categories have different lead times and resolution paths. "
					"Cyclical recession risk resolves when the business cycle turns. "
					"Financial stability risk can spike and resolve independently of the cycle. "
					"Valuation risk is a long-run structural headwind, not a near-term trigger. "
					"Fiscal risk builds slowly and constrains policy response — its impact is "
					"realized only when other risks crystallize simultaneously.",
					header="Different Lead Times, Different Policy Responses",
				), md=6),
			], className="g-4"),
			title="Risk Taxonomy Framework",
		),

	], style={"padding": "24px"})


_GLOSSARY_SECTIONS = [
	("Inflation — Headline & Core", [
		("CPI",             "Consumer Price Index — average price change for a fixed basket of urban household goods and services"),
		("Core CPI",        "CPI excluding food & energy; strips volatile components to reveal underlying price trend"),
		("Core PCE",        "PCE Price Index ex food & energy — the FOMC's explicit 2% inflation target; published monthly by the BEA"),
		("CPI vs PCE",      "PCE typically runs 0.3–0.5% below CPI due to different category weights and substitution methodology"),
		("PCE",             "Personal Consumption Expenditures price index — broader than CPI; covers all consumer spending, not just urban basket"),
		("Supercore CPI",   "CPI less food, energy, and shelter — isolates cyclical services inflation driven by wages and demand, not housing pass-through"),
	]),
	("Inflation — Alternative Measures", [
		("Median CPI",       "Cleveland Fed: inflation at the median price-change component — strips extreme movements; best predictor of future inflation (Stock & Watson)"),
		("Trimmed Mean PCE", "Dallas Fed: removes top and bottom 31% of price changes by weight each month; comparable in predictive power to Median CPI"),
		("Sticky Price CPI", "Atlanta Fed: components that change price infrequently (rents, medical, education); best predictor of persistent inflation regimes"),
		("Flexible CPI",     "Components that change price frequently (fuel, food, airfares); highly volatile and mean-reverting — transitory by nature"),
		("OER",              "Owners' Equivalent Rent — largest CPI component (~26% weight); imputes housing cost for owners; lags actual market rents by 12–18 months"),
		("Shelter CPI",      "OER plus rent of primary residence; together ~33% of CPI headline; the main driver of core CPI persistence above target"),
	]),
	("Inflation — Expectations & Regime", [
		("Breakeven Inflation","Market-implied inflation from the spread between TIPS and nominal Treasuries; 5-yr and 10-yr versions track different horizons"),
		("TIPS",              "Treasury Inflation-Protected Securities — principal adjusts with CPI; used to extract inflation expectations from bond pricing"),
		("Michigan Survey",   "University of Michigan 1-year consumer inflation expectations survey; volatile but tracks household psychology relevant to wage demands"),
		("Deanchoring",       "When long-run inflation expectations move persistently above 2% — signals Fed credibility erosion; historically requires more aggressive response"),
		("Cyclical Inflation", "Demand-driven price pressure linked to labor markets and output gaps — responds to monetary tightening within 12–24 months"),
		("Structural Inflation","Supply-side or cost-push price pressure (energy, supply chains, demographics) — less responsive to interest rates"),
		("Inflation Regime",  "At Target / Disinflation / Elevated / Entrenched — categorized by whether sticky prices, expectations, and core measures are all simultaneously elevated"),
	]),
	("Money Supply", [
		("M1",       "Narrow money — physical currency plus checking-account and demand deposits"),
		("M2",       "Broad money — M1 plus savings accounts, retail money-market funds, and small CDs"),
		("Real M2",  "M2 adjusted for inflation; sustained YoY contraction signals tightening monetary conditions"),
	]),
	("Labor Market — Unemployment", [
		("U-3 (Unemployment Rate)", "Official unemployment rate — share of labor force actively seeking but unable to find work"),
		("U-6",                     "Broadest BLS measure: U-3 plus marginally attached workers plus involuntary part-time employees"),
		("U6 – U3 Gap",             "Labor slack indicator; widening gap signals workers being excluded from headline measure — hidden underemployment"),
		("LFPR",                    "Labor Force Participation Rate — share of working-age adults in the labor force (%)"),
		("Emp/Pop Ratio",           "Share of all working-age adults employed; unaffected by people entering or leaving the labor force"),
		("Sahm Rule",               "When the 3-month average unemployment rate rises 0.5pp above its 12-month low — a near-perfect real-time recession indicator"),
	]),
	("Labor Market — Employment & Claims", [
		("Nonfarm Payrolls",    "Net monthly job additions across all non-agricultural sectors; subject to ±100K revisions"),
		("AHE",                 "Average Hourly Earnings — monthly wage gauge; YoY growth above ~4% is inconsistent with 2% inflation given ~2% productivity"),
		("ECI",                 "Employment Cost Index — quarterly BLS survey of employer compensation costs; less volatile than AHE and preferred by the Fed"),
		("Initial Claims",      "Weekly new unemployment insurance filings — one of the timeliest labor market reads; released weekly with a 4-day lag"),
		("Continued Claims",    "Number of workers currently receiving unemployment benefits; measures how quickly laid-off workers are rehired"),
		("Absorption Ratio",    "Continued claims ÷ initial claims — higher values signal slower re-employment; rises before the unemployment rate turns up"),
		("Temp Employment",     "Temporary Help Services payrolls; leads permanent hiring by 3–6 months — companies cut temps before cutting full-time staff"),
	]),
	("Labor Market — JOLTS & Wages", [
		("JOLTS",               "Job Openings and Labor Turnover Survey — BLS monthly survey covering openings, hires, quits, and layoffs"),
		("Job Openings",        "Total unfilled positions at month-end; leads payroll changes by 2–4 months; above 8M signals tight labor market"),
		("Quits Rate",          "Voluntary separations as % of employment — workers quit when confident of finding better work; drops 3–6 months before unemployment rises"),
		("Layoffs Rate",        "Involuntary separations as % of employment; rising layoffs rate leads initial claims by 1–3 months"),
		("Beveridge Curve",     "Relationship between job openings and unemployment; outward shift signals structural mismatch rather than cyclical slack"),
		("V/U Ratio",           "Job openings divided by unemployed persons — Fed's preferred labor tightness gauge; above 1 = more jobs than workers"),
		("Productivity",        "Output per hour of all persons, nonfarm business sector; quarterly BLS estimate subject to large revisions"),
		("ULC",                 "Unit Labor Costs = compensation ÷ productivity; primary transmission from wage growth to goods/services prices; ULC > 2.5% YoY is inflationary"),
		("Real Wage Growth",    "AHE YoY minus CPI YoY — negative real wages erode consumer purchasing power; important for consumption outlook"),
		("Wage-Productivity Gap","Wage growth above productivity growth = rising ULC; sustainable only if firms can pass through costs or compress margins"),
	]),
	("Leading Indicators & Business Cycle", [
		("LEI",           "Conference Board Leading Economic Index — composite of 10 forward-looking components including yield curve, claims, permits, and stock prices"),
		("ISM New Orders","ISM Manufacturing New Orders Index; above 50 = expansion, below 50 = contraction; leads GDP by 3–6 months"),
		("ISM PMI",       "Purchasing Managers' Index — monthly survey of manufacturing conditions; 50 is the neutral threshold between expansion and contraction"),
		("BCI",           "Business Cycle Index — composite z-scored weighted average of leading indicators; outputs Expansion/Slowdown/Recovery/Contraction phase"),
		("Building Permits","New housing units authorized; leads housing starts and construction employment by 1–3 months"),
		("NBER Recession", "National Bureau of Economic Research official recession dating — post-hoc determination of peak-to-trough contraction periods"),
		("Sahm Rule",      "3-month average unemployment rise of ≥0.5pp above 12-month low — near-perfect recession signal in real time"),
		("Yield Curve",    "Spread between long and short Treasury rates; inversion (spread < 0) has preceded every U.S. recession since 1970 with a 6–18 month lead"),
	]),
	("Markets & Rates", [
		("VIX",                "CBOE Volatility Index — 30-day implied volatility on S&P 500 options; the 'fear gauge'"),
		("CAPE / Shiller P/E", "S&P 500 price divided by 10-year inflation-adjusted average earnings; long-run valuation anchor"),
		("Trailing P/E",       "Stock price divided by last 12 months of reported earnings"),
		("10Y Treasury Yield", "Benchmark long-term government rate; affects all asset prices through discounting"),
		("Yield Curve (10Y-2Y)","Spread between 10-year and 2-year Treasury yields; inversion has preceded every U.S. recession since 1970"),
		("Inverted Curve",     "When short rates exceed long rates (spread < 0) — markets price near-term risk higher than long-run growth"),
		("NY Fed Rec. Prob.",  "Model-based 12-month recession probability using yield curve slope; above 30% has historically been a reliable signal"),
		("Dollar Index (DTWEX)","Trade-weighted U.S. dollar index vs. 26 currencies; rapid appreciation tightens global financial conditions"),
	]),
	("Financial Conditions", [
		("FCI",               "Financial Conditions Index — composite gauge of credit, equity, interest rate, and currency conditions"),
		("NFCI",              "National Financial Conditions Index (Chicago Fed) — 105-variable weekly gauge; positive = tighter than average"),
		("STLFSI",            "St. Louis Fed Financial Stress Index — 18-variable weekly composite; positive = above-average stress"),
		("FSI",               "Financial Stress Index (generic) — positive values indicate above-average systemic stress"),
		("CP-Tsy Spread",     "3-Month AA Financial CP rate minus 3-Month Treasury — modern TED spread; measures unsecured bank borrowing premium"),
		("SOFR",              "Secured Overnight Financing Rate — overnight repo benchmark; LIBOR replacement since 2023"),
		("Reverse Repo (RRP)","Fed facility where counterparties park cash overnight; high usage = ample reserves parked at Fed; drawdown reflects reserve redistribution, T-bill issuance absorbing MMF demand, or genuine liquidity reduction"),
		("NFCI Credit",       "NFCI credit subindex isolating credit conditions specifically — early warning of tightening access to credit"),
	]),
	("Credit Markets & Banking", [
		("HY Spread (OAS)",    "Extra yield demanded above Treasuries for below-investment-grade bonds; widens when default risk rises"),
		("IG Spread (OAS)",    "Extra yield demanded above Treasuries for investment-grade corporate bonds"),
		("OAS",                "Option-Adjusted Spread — yield spread net of embedded call/put option value; the 'clean' credit risk premium"),
		("CRE",                "Commercial Real Estate — offices, retail, apartments, and industrial property"),
		("SLOOS",              "Senior Loan Officer Opinion Survey — quarterly Fed survey of bank lending standards and demand"),
		("Lending Standards",  "Net % of banks tightening loan conditions; >40% net tightening historically precedes credit contraction by 6–12 months"),
		("CRE Lending Stds",   "SLOOS commercial real estate standards — leading indicator of CRE delinquencies by 6–12 months"),
		("CC Delinquency",     "Share of credit card loans past due; broad consumer credit health signal; rises before broader defaults"),
		("Delinquency Rate",   "Share of loans past due 30+ days; lags credit stress by 1–2 quarters but is a highly reliable confirmation signal"),
		("Bank Deposits",      "Total deposits at all commercial banks; rapid YoY contraction signals either tightening conditions or bank stress (cf. SVB 2023)"),
		("Charge-off Rate",    "Loans written off as uncollectible as % of total loans; lags delinquencies by 1–2 quarters"),
	]),
	("Policy & Fiscal", [
		("Fed Funds Rate",      "Fed's overnight policy rate set by the FOMC; the primary monetary policy tool"),
		("Real FF Rate",        "Fed Funds Rate minus Core CPI YoY — the inflation-adjusted policy stance; negative = accommodative, high positive = restrictive"),
		("WALCL",               "Fed balance sheet total assets; expanded via QE, reduced via QT"),
		("QE / QT",             "Quantitative Easing (asset purchases, expands balance sheet) / Quantitative Tightening (shrinking balance sheet)"),
		("T5YIE / T10YIE",     "5- and 10-year breakeven inflation rates implied by TIPS vs. nominal Treasury pricing"),
		("TIPS",                "Treasury Inflation-Protected Securities — principal adjusts with CPI; used to extract market inflation expectations"),
		("BEI / Breakeven",     "Market-implied inflation from the gap between TIPS and nominal yields at the same maturity"),
		("Debt / GDP",          "Federal debt as a percentage of GDP; above 120% historically associated with fiscal sustainability concerns"),
		("Interest / Receipts", "Federal interest payments as a share of government revenues; above 20% constrains fiscal stabilization capacity"),
		("SAAR",                "Seasonally Adjusted Annual Rate — removes seasonal patterns, expressed as if maintained for a full year"),
	]),
	("Housing", [
		("Housing Starts",      "New residential units started monthly (SAAR); leads construction employment and materials demand by 1–3 months"),
		("Building Permits",    "Housing units authorized before construction begins; leads starts by 1–2 months — the earliest housing market signal"),
		("Case-Shiller HPI",    "S&P/Case-Shiller repeat-sales home price index; published with a ~2-month lag"),
		("Mortgage Rate (30Y)", "Freddie Mac PMMS; directly affects housing affordability, purchase volume, and homeowner lock-in effect"),
		("Lock-in Effect",      "Homeowners with sub-4% mortgages are reluctant to sell and take on a higher-rate loan — suppresses existing home supply"),
	]),
	("Probabilistic Forecasting", [
		("Recession Probability","Model-estimated probability (0–100%) of NBER recession beginning within the forecast horizon (6M / 12M / 24M)"),
		("Logistic Regression",  "Statistical model mapping predictor variables to binary outcomes; here, predicting recession vs. expansion using macro features"),
		("Feature Contribution", "Log-odds contribution of each input variable to the model's probability estimate — positive = increases recession probability"),
		("Confidence Band",      "Bootstrap-estimated range of model uncertainty; wider band = less historical data or more disagreement across re-samples"),
		("OOS Backtest",         "Out-of-sample backtest — model is re-trained on past data only, then tested forward in time to assess real-world accuracy"),
		("Hit Rate",             "Fraction of NBER recessions preceded by a model signal within the forecast horizon"),
		("False Positive Rate",  "Fraction of signals not followed by a recession within the horizon — measures the cost of acting on the signal"),
		("Precision",            "Of all recession signals fired, the fraction that were correct — measures signal quality"),
		("Recall",               "Of all actual recessions, the fraction correctly flagged — measures signal completeness"),
	]),
	("Statistical Concepts", [
		("Z-Score",          "Number of standard deviations from the mean; used to normalize different-scale indicators for comparison"),
		("Percentile",       "Rank in the historical distribution; 75th percentile = higher than 75% of historical observations"),
		("YoY / MoM",        "Year-over-year / month-over-month percentage change"),
		("SAAR",             "Seasonally Adjusted Annual Rate — removes intra-year seasonality, expressed as annual pace"),
		("Rolling Window",   "Calculation over a moving fixed-length period (e.g. 20-year rolling average); avoids structural break bias"),
		("Calibration Window","Historical period used to set thresholds; longer windows reduce false signals from structural regime changes"),
		("Momentum",         "Rate of change in the rate of change — acceleration vs. deceleration in an indicator's trend"),
		("Diffusion Index",  "Fraction of sub-components improving minus fraction deteriorating; above 50 = net expansion"),
		("Revision Risk",    "Degree to which early data releases are subsequently revised; high revision risk reduces real-time signal reliability"),
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


# ── Global Macro Tab ──────────────────────────────────────────────────────────

def global_macro_tab(lookback: int | None = 15) -> html.Div:
	dash = gme.global_dashboard()
	comm = dash["commodities"]
	fx   = dash["fx"]
	cb   = dash["cb_divergence"]
	rr   = sme.real_rates()
	og   = sme.output_gap()
	demo = sme.demographic_pressure()
	glob = sme.globalization_metrics()

	def _fv(v, fmt=".2f", sfx=""):
		return f"{v:{fmt}}{sfx}" if v is not None else "N/A"

	cb_banner = (
		f"Fed: {_fv(cb['fed_rate'], '.2f', '%')}  |  "
		f"ECB: {_fv(cb['ecb_rate'], '.2f', '%')}  |  "
		f"BOJ: {_fv(cb['boj_rate'], '.2f', '%')}  |  "
		f"Fed–ECB Gap: {_fv(cb['fed_ecb_gap'], '+.2f', 'pp')}  |  "
		f"{cb['divergence_label']}"
	)
	comm_banner = (
		f"Brent: ${_fv(comm['brent_level'], '.0f')}/bbl  "
		f"({_fv(comm['brent_yoy'], '+.1f', '% YoY')})  |  "
		f"Gold: ${_fv(comm['gold_level'], '.0f')}/oz  "
		f"({_fv(comm['gold_yoy'], '+.1f', '% YoY')})  |  "
		f"PPI Commodities: {_fv(comm['ppi_commodities_yoy'], '+.1f', '% YoY')}  |  "
		f"{comm['commodity_regime']}"
	)
	struct_banner = (
		f"Output Gap: {_fv(og['gap_current'], '+.2f', '%')} ({og['label']})  |  "
		f"Real FF Rate: {_fv(rr['real_ff'], '+.2f', '%')}  |  "
		f"10Y TIPS: {_fv(rr['tips_10y'], '.2f', '%')}  |  "
		f"Policy Stance: {rr['policy_stance']}  |  "
		f"Demo: {demo['label']}"
	)

	return html.Div([
		section_card(prose(cb_banner)),
		dbc.Row([
			dbc.Col(section_card(
				html.H6("Central Bank Policy Rates", className="section-label"),
				graph(central_bank_rates_chart(dl, lookback_years=lookback or 25)),
			), md=6),
			dbc.Col(section_card(
				html.H6("Global Commodity Complex", className="section-label"),
				graph(commodity_chart(dl, lookback_years=lookback or 15)),
			), md=6),
		], className="g-0"),
		section_card(
			html.H6("FX Conditions — USD vs Major Currencies", className="section-label"),
			prose(
				f"DXY (Broad): {_fv(fx['dxy_level'], '.1f')} ({_fv(fx['dxy_yoy'], '+.1f', '% YoY')})  |  "
				f"EUR/USD: {_fv(fx['eur_usd'], '.4f')}  |  "
				f"JPY/USD: {_fv(fx['jpy_usd'], '.1f')}  |  "
				f"CNY/USD: {_fv(fx['cny_usd'], '.4f')}  |  "
				f"{fx['usd_regime']}"
			),
			graph(fx_chart(dl, lookback_years=lookback or 15)),
		),
		section_card(
			html.H6("Comm. Basket YoY %", className="section-label"),
			prose(comm_banner),
		),
		dbc.Row([
			dbc.Col(section_card(
				html.H6("Real Interest Rates & Policy Stance", className="section-label"),
				prose(struct_banner),
				graph(real_rates_chart(sme, dl, lookback_years=lookback or 20)),
			), md=6),
			dbc.Col(section_card(
				html.H6("Output Gap — Real vs Potential GDP", className="section-label"),
				graph(output_gap_chart(sme)),
			), md=6),
		], className="g-0"),
		dbc.Row([
			dbc.Col(section_card(
				html.H6("Nonfarm Productivity Trend", className="section-label"),
				graph(productivity_chart(sme, lookback_years=lookback or 20)),
			), md=6),
			dbc.Col(section_card(
				html.H6("Structural Context", className="section-label"),
				prose(
					f"<b>Demographics:</b> Working-age population growth "
					f"{_fv(demo['wap_yoy'], '+.2f', '% YoY')} — {demo['label']}.",
					"",
					f"<b>Globalization:</b> {glob['label']}. USD 5Y trend: "
					f"{_fv(glob['dxy_trend_5y_ann'], '+.1f', ' pts/yr ann.')}. "
					f"PPI commodities: {_fv(glob['ppi_commodities_yoy'], '+.1f', '% YoY')}.",
					"",
					f"<b>Real rate benchmark:</b> 10Y TIPS = {_fv(rr['tips_10y'], '.2f', '%')} "
					f"(long-run real rate; note: TIPS yield embeds a real term premium "
					f"~0.5–1.5% above true r*). "
					f"Real FF = {_fv(rr['real_ff'], '+.2f', '% ')} → "
					f"Stance vs. long-run real market rate: <b>{rr['policy_stance']}</b>.",
				),
			), md=6),
		], className="g-0"),
	], style={"padding": "24px"})


# ── Macro Regime Tab ──────────────────────────────────────────────────────────

def macro_regime_tab(lookback: int | None = 20) -> html.Div:
	regime_name, regime_color, regime_desc = rge.classify()
	dims = rge.dimension_scores()

	regime_card = html.Div([
		html.Div([
			html.Span("MACRO REGIME", style={
				"fontSize": "10px", "fontWeight": "700",
				"letterSpacing": "1px", "color": "rgba(255,255,255,0.75)",
			}),
			html.Div(regime_name, style={
				"fontSize": "22px", "fontWeight": "800",
				"color": "white", "marginTop": "4px",
			}),
			html.Div(regime_desc, style={
				"fontSize": "12px", "color": "rgba(255,255,255,0.85)",
				"marginTop": "8px", "lineHeight": "1.6",
			}),
		], style={
			"background": regime_color, "borderRadius": "8px",
			"padding": "20px 24px", "marginBottom": "16px",
		}),
	])

	regime_help = section_card(
		html.H6("Regime Definitions", className="section-label"),
		html.Div([
			html.Div([
				html.Span(name, style={
					"fontWeight": "600", "fontSize": "12px",
					"color": "white", "background": info["color"],
					"padding": "2px 10px", "borderRadius": "4px",
					"marginRight": "10px",
				}),
				html.Span({
					"Goldilocks":              "Above-trend growth + contained inflation + neutral FCI",
					"Reflation":               "Above-trend growth + rising inflation",
					"Stagflation":             "Below-trend growth + elevated inflation",
					"Disinflation":            "Softening growth + moderating inflation",
					"Liquidity Boom":          "Easy FCI + credit expansion + asset price inflation",
					"Tightening Cycle":        "Rising rates + tightening FCI constraining growth",
					"Balance-Sheet Recession": "Credit collapse + deflation risk + financial crisis",
					"Uncertain":               "Mixed signals across dimensions",
				}.get(name, ""), style={"fontSize": "12px", "color": C["muted"]}),
			], style={"padding": "6px 0", "borderBottom": "1px solid #edf2f7",
					  "display": "flex", "alignItems": "center"})
			for name, info in MACRO_REGIMES.items()
		]),
	)

	return html.Div([
		dbc.Row([
			dbc.Col([regime_card], md=5),
			dbc.Col(section_card(
				html.H6("Dimension Scores", className="section-label"),
				graph(regime_scores_chart(rge)),
			), md=7),
		], className="g-0"),
		section_card(
			html.H6("Regime History Timeline", className="section-label"),
			prose(
				"Colored bands show the simplified regime classification over time, "
				"derived from CPI, yield curve slope, and NFCI. NFCI overlay provides "
				"financial conditions context. Regime transitions are the key inflection "
				"points for asset allocation and risk positioning."
			),
			graph(regime_timeline_chart(rge, dl, lookback_years=lookback or 20)),
		),
		regime_help,
		section_card(
			html.H6("Regime-Aware Threshold Adjustments", className="section-label"),
			prose(
				"In the current regime, the following risk thresholds are dynamically adjusted "
				"relative to their static defaults. Indicators that are less informative in this "
				"environment receive adjusted thresholds to reduce false signals."
			),
			html.Div([
				html.Div([
					html.Span(k, style={"fontWeight": "600", "fontSize": "12px",
										"minWidth": "220px", "display": "inline-block"}),
					html.Span(f"→ {v}", style={"fontSize": "12px", "color": C["muted"]}),
				], style={"padding": "5px 0", "borderBottom": "1px solid #edf2f7"})
				for k, v in rge.get_threshold_adjustments().items()
			] or [prose("No adjustments in current regime.")]),
		),
	], style={"padding": "24px"})


# ── Fiscal Analytics Tab ──────────────────────────────────────────────────────

def fiscal_analytics_tab(lookback: int | None = 30) -> html.Div:
	fdash = fae.fiscal_dashboard()
	sus   = fdash["sustainability"]
	prim  = fdash["primary"]
	traj  = fdash["trajectory"]
	roll  = fdash["rollover"]
	rg    = fdash["r_g"]

	def _fv(v, fmt=".1f", sfx=""):
		return f"{v:{fmt}}{sfx}" if v is not None else "N/A"

	banner = (
		f"<b>Debt/GDP:</b> {_fv(sus['debt_gdp'], '.1f', '%')}  |  "
		f"<b>Interest/Receipts:</b> {_fv(sus['int_receipts'], '.1f', '%')} ({sus['label']})  |  "
		f"<b>Interest/GDP:</b> {_fv(sus['int_gdp'], '.2f', '%')}  |  "
		f"<b>Primary Balance:</b> {_fv(prim['primary_pct_gdp'], '+.1f', '% GDP')}  |  "
		f"<b>r−g:</b> {_fv(rg['nominal_rg'], '+.1f', 'pp')} ({rg['label']})  |  "
		f"<b>Eff. Rate:</b> {_fv(rg['effective_rate'], '.2f', '%')}  |  "
		f"<b>Trajectory:</b> {traj['label']}  |  "
		f"<b>Rollover:</b> {roll['label']}"
	)

	return html.Div([
		section_card(prose(banner)),
		dbc.Row([
			dbc.Col(section_card(
				html.H6("Federal Debt / GDP — Long-Run Trajectory", className="section-label"),
				graph(debt_trajectory_chart(dl, lookback_years=lookback or 40)),
			), md=8),
			dbc.Col(section_card(
				html.H6("Sustainability Snapshot", className="section-label"),
				prose(
					f"<b>Debt/GDP:</b> {_fv(sus['debt_gdp'], '.1f', '%')}",
					f"<b>Interest payments:</b> ${_fv(sus['int_pay_B'], ',.0f')}B (SAAR)",
					f"<b>Federal receipts:</b> ${_fv(sus['receipts_B'], ',.0f')}B (SAAR)",
					f"<b>Interest/Receipts:</b> {_fv(sus['int_receipts'], '.1f', '%')}",
					f"<b>Interest/GDP:</b> {_fv(sus['int_gdp'], '.2f', '%')}",
					"",
					f"<b>Status:</b> {sus['label']}",
					"",
					"The Interest/Receipts ratio is the most operationally meaningful "
					"measure — it shows how much of every tax dollar is consumed by "
					"debt service before any government service is funded.",
				),
			), md=4),
		], className="g-0"),
		section_card(
			html.H6("Debt Service Burden — Interest as % of Receipts and GDP", className="section-label"),
			prose(
				"Rising interest/receipts (top panel) is the signature of a debt burden "
				"compounding faster than revenue growth. When this ratio exceeds 20–25%, "
				"fiscal space for discretionary spending — including defense, infrastructure, "
				"and social programs — is severely constrained.",
			),
			graph(debt_service_chart(fae, lookback_years=lookback or 30)),
		),
		section_card(
			html.H6("r vs g — Debt Sustainability Dynamics", className="section-label"),
			prose(
				f"<b>10Y Treasury (r):</b> {_fv(rg['nominal_r'], '.2f', '%')}  |  "
				f"<b>Nominal GDP growth (g):</b> {_fv(rg['nominal_g'], '.1f', '%')}  |  "
				f"<b>r−g spread:</b> {_fv(rg['nominal_rg'], '+.2f', 'pp')}  |  "
				f"<b>Effective rate on debt:</b> {_fv(rg['effective_rate'], '.2f', '%')}  |  "
				f"<b>TIPS yield (real r):</b> {_fv(rg['real_r'], '.2f', '%')}  |  "
				f"<b>Real GDP growth (real g):</b> {_fv(rg['real_g'], '.1f', '%')}  |  "
				f"<b>Real r−g:</b> {_fv(rg['real_rg'], '+.2f', 'pp')}",
				"",
				f"<b>Assessment:</b> {rg['label']}. "
				"The Domar condition for debt stabilisation: the primary surplus must exceed "
				"(r − g) × Debt/GDP. When r > g, even a balanced primary budget allows "
				"debt/GDP to compound upward. The green-shaded fill marks periods when "
				"growth outpaced rates — historically the 2010s ZIRP era and pre-1980 "
				"inflation-eroded debt episodes.",
			),
			graph(r_g_chart(fae, dl, lookback_years=lookback or 40)),
		),
		dbc.Row([
			dbc.Col(section_card(
				html.H6("Primary vs Total Deficit/Surplus", className="section-label"),
				prose(
					"The primary balance (total − interest) is the key policy lever. "
					f"Current primary balance: {_fv(prim['primary_pct_gdp'], '+.1f', '% GDP')}. "
					"A primary surplus is necessary (though not sufficient) to stabilize "
					"debt/GDP when real interest rates exceed growth (r > g).",
				),
				graph(primary_balance_chart(fae, lookback_years=lookback or 40)),
			), md=6),
			dbc.Col(section_card(
				html.H6("Fiscal Impulse", className="section-label"),
				prose(
					"Fiscal impulse = year-over-year change in the deficit/GDP ratio. "
					"Positive bars = expansionary fiscal policy (stimulus); "
					"negative bars = fiscal drag (austerity or surplus). "
					"The impulse, not the level, drives near-term aggregate demand impact.",
				),
				graph(fiscal_impulse_chart(fae, lookback_years=lookback or 30)),
			), md=6),
		], className="g-0"),
		section_card(
			html.H6("Rollover Exposure & Trajectory", className="section-label"),
			prose(
				f"<b>Current 10Y yield:</b> {_fv(roll['current_10y_rate'], '.2f', '%')}  |  "
				f"<b>10Y yield 5 years ago:</b> {_fv(roll['rate_5y_ago'], '.2f', '%')}  |  "
				f"<b>Rate change:</b> {_fv(roll['rate_change_5y'], '+.2f', 'pp')}  |  "
				f"<b>Total debt:</b> ${_fv(roll['debt_total_B'], ',.0f')}B",
				"",
				f"<b>Estimated incremental rollover cost:</b> "
				f"${_fv(roll['est_incremental_cost_ann_B'], '+,.0f')}B per year (assuming ~30% "
				f"annual rollover at the full rate change vs. 5 years ago).",
				"",
				f"<b>Assessment:</b> {roll['label']}.",
				"",
				f"<b>Fiscal trajectory (5Y trend):</b> Debt/GDP changing at "
				f"{_fv(traj['debt_gdp_trend_ann'], '+.2f', 'pp/yr')}. "
				f"Interest acceleration: {_fv(traj['interest_acc_ann'], '+.1f', ' $B/yr')}. "
				f"Revenue growth: {_fv(traj['receipts_trend_ann'], '+.1f', ' $B/yr')}. "
				f"Interest absorbing {_fv(traj['acceleration_gap_pct'], '.0f', '% of revenue growth')}.",
				"",
				"Note: FRED does not publish Treasury maturity schedules. Rollover estimates "
				"assume ~6Y average maturity (~30% annual rollover) — actual exposure depends "
				"on the specific maturity distribution of outstanding debt.",
			),
		),
	], style={"padding": "24px"})


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
				dcc.Tab(label="Overview",            value="overview",         style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Crisis Watch",        value="crisis",           style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Leading Indicators",  value="leading",          style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Recession Probability", value="recession_prob", style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Risk Scorecard",      value="risk_scorecard",   style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Macro Regime",        value="macro_regime",     style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="System Resilience",   value="resilience",       style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Liquidity & Funding", value="liquidity",        style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Credit Markets",      value="credit",           style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Policy Constraints",  value="policy",           style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Global Macro",        value="global_macro",     style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Inflation",           value="inflation",        style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Labor Market",        value="labor",            style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Markets & Rates",     value="markets",          style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Housing",             value="housing",          style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Fiscal",              value="fiscal",           style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Fiscal Analytics",    value="fiscal_analytics", style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="All",            	 value="all",        	   style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
				dcc.Tab(label="Glossary",            value="glossary",         style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
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
		"leading":        lambda: leading_indicators_tab(lb),
		"recession_prob":  lambda: recession_probability_tab(lb),
		"risk_scorecard":  lambda: risk_scorecard_tab(),
		"macro_regime":    lambda: macro_regime_tab(lb),
		"resilience":      lambda: system_resilience_tab(),
		"liquidity":  lambda: liquidity_funding_tab(lb),
		"credit":     lambda: credit_markets_tab(lb),
		"policy":            lambda: policy_constraints_tab(lb),
		"global_macro":      lambda: global_macro_tab(lb),
		"inflation":         lambda: inflation_tab(lb),
		"labor":      lambda: labor_tab(lb),
		"markets":    lambda: markets_rates_tab(lb),
		"housing":    lambda: housing_tab(lb),
		"fiscal":            lambda: fiscal_tab(lb),
		"fiscal_analytics":  lambda: fiscal_analytics_tab(lb),
		"glossary":          lambda: glossary_tab(),
	}
	if tab == "all":
		return html.Div([
			dispatch.get(t, lambda: executive_summary_tab())()
			for t in dispatch
		])
	return dispatch.get(tab, lambda: executive_summary_tab())()

# ── Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
	print(f"Loaded {len(dl.available)} series: {', '.join(sorted(dl.available))}")
	print(f"Starting dashboard at http://{DASH.host}:{DASH.port}")
	app.run(debug=False, host=DASH.host, port=DASH.port)
