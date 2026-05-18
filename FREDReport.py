"""
FRED Economic Monitor — PDF Report Generator
Run:  python FREDReport.py [output_path]

Generates a static PDF mirroring all dashboard sections.
Requires:  pip install reportlab kaleido
"""

import sys
import os
import io
import argparse
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

try:
    import plotly.io as pio
except ImportError:
    sys.exit("Missing dependency: pip install plotly")

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image,
        Table, TableStyle, PageBreak, HRFlowable, KeepTogether, Flowable,
    )
except ImportError:
    sys.exit("Missing dependency: pip install reportlab")

from _classes.data_loader import DataLoader
from _classes.sql_storage import SQLStorage
from _classes.chart_factory import (
    line_chart, area_chart, multi_line_chart, bar_change_chart,
    percentile_chart, dual_axis_chart, derived_ratio_chart, derived_spread_chart,
    walcl_pct_gdp_chart, real_rate_chart, yield_spread_chart, risk_heatmap_chart,
    bci_chart, bci_waterfall_chart, momentum_chart, backtest_signal_chart,
    recession_gauge_chart, recession_probability_chart, signal_decomposition_chart,
    jolts_chart, wage_productivity_chart, u3_u6_chart,
    labor_deterioration_chart, claims_dashboard_chart,
    inflation_multi_chart, inflation_expectations_chart,
    shelter_decomposition_chart, sticky_flexible_chart,
    fci_composite_chart, hy_spread_fci_chart,
    delinquency_chart, bank_deposits_chart,
    central_bank_rates_chart, commodity_chart, fx_chart,
    regime_timeline_chart, regime_scores_chart,
    output_gap_chart, productivity_chart, real_rates_chart,
    debt_service_chart, primary_balance_chart,
    debt_trajectory_chart, fiscal_impulse_chart,
)
from _classes.leading_indicators import LeadingIndicatorEngine, BCI_COMPONENTS
from _classes.recession_probability import RecessionProbabilityEngine, HORIZONS as RPE_HORIZONS
from _classes.labor_analytics import LaborAnalyticsEngine
from _classes.inflation_analysis import InflationAnalysisEngine
from _classes.global_macro import GlobalMacroEngine
from _classes.regime_engine import RegimeEngine, REGIMES as MACRO_REGIMES
from _classes.structural_macro import StructuralMacroEngine
from _classes.fiscal_analytics import FiscalAnalyticsEngine
from _classes.risk_engine import RiskEngine
from _classes.series_registry import REGISTRY, CATEGORY_LABELS, CATEGORY_ORDER
from _classes.constants import PATHS, PALETTE as C, CHART, DASH, RISK_STYLE
from _classes.GoogleAPI import GoogleAPIUploadFile


# ── Page geometry ─────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = letter          # 612 x 792 pt
MARGIN     = 0.65 * inch
CONTENT_W  = PAGE_W - 2 * MARGIN          # ~7.2"
HALF_W     = (CONTENT_W - 0.12 * inch) / 2
QUARTER_W  = (CONTENT_W - 0.18 * inch) / 4
THIRD_W    = (CONTENT_W - 0.12 * inch) / 3

# Pixel sizes for Plotly → PNG (scale=1 renders at 72 dpi; we use larger px for quality)
FULL_W_PX, FULL_H_PX   = 920, 370
HALF_W_PX, HALF_H_PX   = 455, 330
SMALL_H_PX              = 290


# ── Section anchor IDs (used for internal PDF hyperlinks) ─────────────────────
CATEGORY_TO_ANCHOR = {
    "inflation":        "sec_inflation",
    "money_supply":     "sec_inflation",
    "labor":            "sec_labor",
    "markets":          "sec_markets",
    "rates":            "sec_markets",
    "valuation":        "sec_markets",
    "financial_stress": "sec_markets",
    "fiscal":           "sec_fiscal",
    "housing":          "sec_housing",
    "liquidity_stress": "sec_liquidity",
    "credit_markets":   "sec_credit",
    "policy_flex":      "sec_policy",
    "global_macro":     "sec_global_macro",
    "structural":       "sec_structural",
}

# ── Chart event annotations (used on long-history charts) ─────────────────
_EVENTS_LONG = [
    ("2001-09-11", "9/11"),
    ("2008-09-15", "GFC"),
    ("2020-03-23", "COVID"),
    ("2022-03-16", "Hikes"),
]
_EVENTS_MED = [
    ("2008-09-15", "GFC"),
    ("2020-03-23", "COVID"),
    ("2022-03-16", "Hikes"),
]
_EVENTS_SHORT = [
    ("2020-03-23", "COVID"),
    ("2022-03-16", "Hikes"),
]

# ── ReportLab style constants ──────────────────────────────────────────────────

HDR_BG    = colors.HexColor("#1a365d")
HDR_TEXT  = colors.white
BODY_CLR  = colors.HexColor("#2d3748")
MUTED_CLR = colors.HexColor("#718096")
RULE_CLR  = colors.HexColor("#e2e8f0")

def _ps(name, **kw):
    defaults = dict(fontName="Helvetica", fontSize=9, textColor=BODY_CLR, leading=13)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)

ST_TITLE   = _ps("title",   fontName="Helvetica-Bold", fontSize=18, textColor=HDR_TEXT)
ST_SUBTITLE= _ps("sub",     fontSize=10, textColor=colors.HexColor("#a0aec0"))
ST_SECTION = _ps("section", fontName="Helvetica-Bold", fontSize=13, spaceBefore=10, spaceAfter=4)
ST_SUBSECT = _ps("subsect", fontName="Helvetica-Bold", fontSize=11, spaceBefore=8,  spaceAfter=3)
ST_BODY    = _ps("body",    textColor=MUTED_CLR, leading=14, spaceBefore=2, spaceAfter=4)
ST_SMALL   = _ps("small",   fontSize=8, textColor=MUTED_CLR, leading=12)
ST_CAPTION = _ps("caption", fontSize=8, fontName="Helvetica-Oblique", textColor=MUTED_CLR,
                 spaceBefore=2, spaceAfter=6)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _risk_clrs(level: str):
    rs = RISK_STYLE.get(level, RISK_STYLE["neutral"])
    return (
        colors.HexColor(rs["bg"]),
        colors.HexColor(rs["text"]),
        colors.HexColor(rs["border"]),
    )


def _fig_to_image(fig, display_w: float, w_px: int, h_px: int) -> Image:
	try:
		png = pio.to_image(fig, format="png", width=w_px, height=h_px, scale=1)
	except Exception as exc:
		print(f"  [warn] chart render failed: {exc}")
		return Spacer(display_w, display_w * h_px / w_px)
	max_height = 5.5 * inch
	height = display_w * h_px / w_px				
	if height > max_height:
		print(f"Display size: {display_w} x {height}")
		scale = max_height / height
		display_w *= scale
		height = max_height
		print(f"Scaled to size: {display_w} x {height}")
	buf = io.BytesIO(png)
	return Image(buf, width=display_w, height=height)

def _full_chart(fig) -> Image:
    return _fig_to_image(fig, CONTENT_W, FULL_W_PX, FULL_H_PX)


def _half_chart(fig, small=False) -> Image:
    h = SMALL_H_PX if small else HALF_H_PX
    return _fig_to_image(fig, HALF_W, HALF_W_PX, h)


def _two_charts(fig_l, fig_r, small=False) -> Table:
    img_l = _half_chart(fig_l, small)
    img_r = _half_chart(fig_r, small)
    t = Table([[img_l, img_r]], colWidths=[HALF_W, HALF_W])
    t.setStyle(TableStyle([
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


class _BookmarkFlowable(Flowable):
    """Zero-height flowable that registers a PDF outline/bookmark entry."""
    def __init__(self, title: str, key: str, level: int = 0):
        super().__init__()
        self.title = title
        self.key = key
        self.level = level
        self.width = self.height = 0

    def draw(self):
        self.canv.bookmarkPage(self.key)
        self.canv.addOutlineEntry(self.title, self.key, level=self.level, closed=False)


class _ConfBar(Flowable):
    """
    Mini horizontal confidence-score bar for KPI cards.
    Draws a thin filled track + "conf XX/100" label using vector primitives.
    Works with any ReportLab font — no Unicode block characters required.
    """
    def __init__(self, score: float, avail_w: float,
                 fill: colors.Color, empty: colors.Color, text: colors.Color):
        super().__init__()
        self.score  = score
        self.width  = avail_w
        self.fill   = fill
        self.empty  = empty
        self.text   = text
        self.height = 8

    def draw(self):
        bar_w  = self.width * 0.40
        bar_h  = 3.0
        y      = (self.height - bar_h) / 2
        self.canv.setFillColor(self.empty)
        self.canv.rect(0, y, bar_w, bar_h, fill=1, stroke=0)
        fill_w = bar_w * min(1.0, max(0.0, self.score / 100))
        if fill_w > 0:
            self.canv.setFillColor(self.fill)
            self.canv.rect(0, y, fill_w, bar_h, fill=1, stroke=0)
        self.canv.setFont("Helvetica", 6.5)
        self.canv.setFillColor(self.text)
        self.canv.drawString(bar_w + 3, y - 0.5, f"conf  {self.score:.0f}/100")


def _section_rule(title: str, anchor_id: str | None = None, level: int = 0) -> list:
    items = []
    if anchor_id:
        # Named destination for internal hyperlinks (KPI cards → section)
        items.append(Paragraph(f'<a name="{anchor_id}"/>',
                               _ps("anc", fontSize=1, leading=1,
                                   spaceBefore=0, spaceAfter=0)))
        # PDF outline bookmark visible in viewer navigation panel
        items.append(_BookmarkFlowable(title, anchor_id, level=level))
    items += [
        Spacer(1, 0.12 * inch),
        Table(
            [[Paragraph(title.upper(), _ps("sr", fontName="Helvetica-Bold", fontSize=12,
                                           textColor=HDR_TEXT))]],
            colWidths=[CONTENT_W],
            rowHeights=[0.3 * inch],
            style=TableStyle([
                ("BACKGROUND",   (0, 0), (-1, -1), HDR_BG),
                ("TOPPADDING",   (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
                ("LEFTPADDING",  (0, 0), (-1, -1), 10),
            ]),
        ),
        Spacer(1, 0.08 * inch),
    ]
    return items


def _prose(text: str) -> Paragraph:
    return Paragraph(text, ST_BODY)


def _subsection(title: str) -> Paragraph:
    return Paragraph(title, ST_SUBSECT)



def _half_table(left_items, right_items):
    rows = []
    max_len = max(len(left_items), len(right_items))
    for i in range(max_len):
        left = left_items[i] if i < len(left_items) else Spacer(1,1)
        right = right_items[i] if i < len(right_items) else Spacer(1,1)
        rows.append([left, right])
    t = Table(rows, colWidths=[HALF_W, HALF_W], splitByRow=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ]))
    return t
	
# def _half_table(left_items: list, right_items: list) -> Table:
    # t = Table([[left_items, right_items]], colWidths=[HALF_W, HALF_W], splitByRow=1)
    # t.setStyle(TableStyle([
        # ("VALIGN", (0,0), (-1,-1), "TOP"),
        # ("LEFTPADDING", (0,0), (0,-1), 0),
        # ("RIGHTPADDING", (0,0), (0,-1), 6),
        # ("LEFTPADDING", (1,0), (1,-1), 6),
        # ("RIGHTPADDING", (1,0), (1,-1), 0),
    # ]))
    # return t

# ── KPI Card ──────────────────────────────────────────────────────────────────

def _kpi_card(series_id: str, re: RiskEngine, dl) -> Table:
    """Single KPI card with risk color coding, full name, and section hyperlink."""
    risk, display, _ = re.score(series_id)
    bg, txt, border = _risk_clrs(risk)
    rs = RISK_STYLE.get(risk, RISK_STYLE["neutral"])
    meta = REGISTRY.get(series_id, {})
    basis = meta.get("risk_basis", "level")
    basis_label = {"yoy": "YoY", "mom_change": "MoM"}.get(basis, "")
    _, as_of_date = dl.get_latest(series_id)
    as_of = as_of_date.strftime(DASH.date_display_fmt) if as_of_date else ""

    def _cp(name, **kw):
        kw.setdefault("textColor", txt)
        return _ps(name, **kw)

    short = meta.get("short_name", series_id).upper()
    anchor = CATEGORY_TO_ANCHOR.get(meta.get("category", ""), "")
    short_markup = (f'<link href="#{anchor}" color="#ffffff">{short} &uarr;</link>'
                    if anchor else short)

    full_name = meta.get("name", series_id)
    if len(full_name) > 52:
        full_name = full_name[:49] + "..."

    arrow = re.trend_arrow(series_id)
    basis_markup = f'  <font size="8">{basis_label}</font>' if basis_label else ""
    display_with_arrow = f'{display} <font size="10">{arrow}</font>{basis_markup}'

    conf_score = re.confidence_score(series_id)
    conf_bar = _ConfBar(
        conf_score,
        avail_w=QUARTER_W - 0.22 * inch - 14,
        fill=border,
        empty=colors.HexColor("#c8d0da"),
        text=txt,
    )

    rows = [
        [Paragraph(short_markup, _cp("cn", fontName="Helvetica-Bold", fontSize=9))],
        [Paragraph(full_name, _cp("cfn", fontName="Helvetica-Bold", fontSize=7, leading=10))],
        [Paragraph(display_with_arrow, _cp("cv", fontName="Helvetica-Bold", fontSize=17, leading=21))],
        [HRFlowable(width=QUARTER_W - 0.22 * inch, thickness=0.5, color=border)],
        [Paragraph(
            f"<b>{re.score_label(series_id, risk)}</b>  ·  as of {as_of}",
            _cp("cst", fontSize=7),
        )],
        [conf_bar],
    ]
    inner = Table(rows, colWidths=[QUARTER_W - 0.22 * inch])
    inner.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), bg),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    wrapper = Table([[inner]], colWidths=[QUARTER_W])
    wrapper.setStyle(TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), 2),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LINEAFTER",     (0, 0), (0, -1), 2, border),
    ]))
    return wrapper


def _kpi_row(ids: list, re: RiskEngine, dl) -> Table:
    """A row of up to 4 KPI cards."""
    cells = [_kpi_card(sid, re, dl) for sid in ids]
    # Pad to 4 columns
    while len(cells) < 4:
        cells.append(Spacer(QUARTER_W, 0.1))
    t = Table([cells], colWidths=[QUARTER_W] * 4)
    t.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    return t


# ── Crisis Dimension Card ─────────────────────────────────────────────────────

def _crisis_dim_card(name: str, dim: dict) -> Table:
    risk = dim["score"]
    bg, txt, border = _risk_clrs(risk)
    rs = RISK_STYLE.get(risk, RISK_STYLE["neutral"])

    def _cp(n, **kw):
        kw.setdefault("textColor", txt)
        return _ps(n, **kw)

    rows = [
        [Paragraph(name.upper(), _cp("dn", fontName="Helvetica-Bold", fontSize=8))],
        [Paragraph(rs["label"], _cp("ds", fontName="Helvetica-Bold", fontSize=15, leading=19))],
    ]
    # Available content width = card column width minus left+right padding (8+8=16pt)
    _cw = THIRD_W - 0.18 * inch - 16
    for label, r, display, as_of in dim["components"]:
        _, c_txt, _ = _risk_clrs(r)
        row_inner = Table([[
            Paragraph(label, _cp("dcl", fontSize=8)),
            Paragraph(
                f"<b>{display}</b>  <font size=7>{as_of}</font>",
                _ps("dcv", fontSize=8, textColor=c_txt),
            ),
        ]], colWidths=[_cw * 0.45, _cw * 0.55])
        row_inner.setStyle(TableStyle([
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING",   (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 1),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ]))
        rows.append([row_inner])

    card = Table(rows, colWidths=[THIRD_W - 0.18 * inch])
    card.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), bg),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LINEAFTER",    (0, 0), (0, -1), 2, border),
    ]))
    return card


# ── Section Builders ──────────────────────────────────────────────────────────

def _build_cover(story, dl):
    now = datetime.now().strftime("%B %d, %Y")
    cover = Table(
        [[Paragraph("FRED Economic Monitor", ST_TITLE)],
         [Paragraph("Federal Reserve Bank of St. Louis Data", ST_SUBTITLE)],
         [Spacer(1, 0.08 * inch)],
         [Paragraph(f"Report generated: {now}", ST_SUBTITLE)],
         [Paragraph(f"Data updated: {dl.last_updated()}", ST_SUBTITLE)],
         [Paragraph(f"{len(dl.available)} series loaded", ST_SUBTITLE)],
        ],
        colWidths=[CONTENT_W],
        style=TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), HDR_BG),
            ("LEFTPADDING",  (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("TOPPADDING",   (0, 0), (0, 0), 18),
            ("TOPPADDING",   (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING",(0, -1), (-1, -1), 18),
            ("BOTTOMPADDING",(0, 0), (-1, -2), 2),
        ]),
    )
    story.append(cover)
    story.append(Spacer(1, 0.2 * inch))


def _build_crisis(story, re: RiskEngine, dl, crisis_dims: dict):
    story += _section_rule("Crisis Watch", anchor_id="sec_crisis")
    story.append(_prose(
        "This section synthesizes multiple indicators into six structural dimensions. "
        "Color coding is descriptive, not predictive — elevated readings indicate conditions "
        "that have historically preceded stress, not guaranteed outcomes."
    ))

    counts = {"red": 0, "yellow": 0, "green": 0, "neutral": 0}
    for dim in crisis_dims.values():
        counts[dim["score"]] += 1
    story.append(_prose(
        f"<b>{counts['red']} dimension(s) stressed, {counts['yellow']} elevated, "
        f"{counts['green']} within normal range.</b>"
    ))
    story.append(Spacer(1, 0.1 * inch))

    # Dimension cards — 3 per row
    dim_items = list(crisis_dims.items())
    for row_start in range(0, len(dim_items), 3):
        chunk = dim_items[row_start:row_start + 3]
        cells = [_crisis_dim_card(name, dim) for name, dim in chunk]
        while len(cells) < 3:
            cells.append(Spacer(THIRD_W, 0.1))
        t = Table([cells], colWidths=[THIRD_W] * 3)
        t.setStyle(TableStyle([
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING",   (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ]))
        story.append(t)

    story.append(Spacer(1, 0.12 * inch))
    story.append(_subsection("Historical Indicator Risk Status"))
    story.append(_prose(
        "The heatmap below shows the risk classification of twelve key indicators over "
        "the past ten years (monthly). Green = normal; yellow = elevated; red = stressed. "
        "Clusters of color indicate correlated deterioration across multiple indicators."
    ))
    print("  Rendering crisis heatmap...")
    story.append(_full_chart(risk_heatmap_chart(re, lookback_years=10)))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_subsection("Indicator Detail"))

    print("  Rendering crisis charts...")
    story.append(_two_charts(
        area_chart(dl, "DGS10", title="10-Year Treasury Yield %", lookback_years=10),
        area_chart(dl, "SUBLPDRCSN", title="CRE Lending Standards — Net % Tightening",
                   color=C["red"], fill_color="rgba(155,44,44,0.09)",
                   threshold_green=10, threshold_red=40, lookback_years=20),
        small=True,
    ))
    story.append(_two_charts(
        area_chart(dl, "CPILFESL", yoy=True, title="Core CPI YoY %",
                   color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                   threshold_green=2.5, threshold_red=4.0, lookback_years=10),
        area_chart(dl, "UNRATE", title="Unemployment Rate %",
                   threshold_green=4.5, threshold_red=6.0,
                   color=C["slate"], fill_color="rgba(74,85,104,0.08)", lookback_years=20),
        small=True,
    ))
    story.append(_two_charts(
        area_chart(dl, "USSLIND", lookback_years=10,
                   color=C["red"], fill_color="rgba(155,44,44,0.09)",
                   threshold_green=0, threshold_red=-5,
                   title="Conference Board Leading Economic Index (MoM %)"),
        area_chart(dl, "ICSA", lookback_years=10,
                   ma_periods=4,
                   color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                   threshold_green=250, threshold_red=350,
                   title="Initial Jobless Claims — 4-Week MA"),
        small=True,
    ))

    story.append(Spacer(1, 0.12 * inch))
    story.append(_subsection("Structural Deterioration vs. Acute Crisis"))
    story.append(_half_table(
        [_prose(
            "Structural deterioration develops over years: rising debt loads, persistent "
            "above-target inflation, labor market erosion, or deteriorating credit quality "
            "in key sectors. These imbalances accumulate quietly, often masked by growth "
            "momentum or asset price appreciation, until they constrain policy options."
        )],
        [_prose(
            "The 2008 Global Financial Crisis illustrates the distinction: CRE and residential "
            "delinquencies climbed for 18+ months before Lehman's collapse compressed the acute "
            "phase into a single weekend. The structural fragility had been building; the "
            "trigger was catalytic, not causal."
        )],
    ))

    story.append(_subsection("Historically Dangerous Indicator Combinations"))

    combos = [
        ("Debt-Rate Spiral",
         "High government debt/GDP + rapidly rising interest rates → expanding debt service "
         "costs → crowding out productive investment. The critical metric is debt service "
         "as a share of revenue, not the absolute debt level."),
        ("Stagflation Trap",
         "Persistent above-target inflation + softening labor market → contradictory signals "
         "for central banks → elevated policy error risk. The 1970s remain the benchmark "
         "episode: dual mandates in conflict with no clean resolution available."),
        ("Credit Stress Cascade",
         "Rising loan delinquencies + elevated market volatility + tightening lending standards "
         "→ credit contraction → demand compression → self-reinforcing slowdown."),
        ("Confidence Breakdown",
         "When multiple dimensions show stress simultaneously, policy space narrows. Fiscal "
         "tools are constrained by debt levels. Monetary tools are constrained by inflation. "
         "Each instrument has fewer degrees of freedom."),
    ]
    combo_rows = []
    for i in range(0, len(combos), 2):
        row = []
        for name, desc in combos[i:i + 2]:
            row.append([_prose(f"<b>{name}</b>"), _prose(desc)])
        while len(row) < 2:
            row.append([Spacer(HALF_W, 0.1)])
        t = Table([row], colWidths=[HALF_W, HALF_W])
        t.setStyle(TableStyle([
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        combo_rows.append(t)
    story += combo_rows


_GLOSSARY = [
    # (term, plain-English definition) — grouped by theme

    # ── Inflation — Headline & Core ───────────────────────────────────────
    ("CPI",         "Consumer Price Index — average price change for a fixed basket of urban household goods"),
    ("Core CPI",    "CPI excluding food & energy; strips volatile components to show underlying trend"),
    ("Core PCE",    "PCE Price Index ex food & energy — the FOMC's explicit 2% inflation target; published by BEA monthly"),
    ("CPI vs PCE",  "PCE typically runs 0.3–0.5% below CPI due to different weights and substitution methodology"),
    ("Supercore CPI","CPI for services excluding food, energy, and shelter — closely tracks wage growth; the Fed's preferred cyclical inflation gauge"),
    ("Shelter CPI", "CPI component for housing costs (~35% of CPI); dominated by OER and lags real-time rents by 12–18 months"),

    # ── Inflation — Alternative Measures ─────────────────────────────────
    ("Median CPI",        "Cleveland Fed measure using the median price change across all CPI components; filters outliers better than core CPI"),
    ("Trimmed Mean PCE",  "Dallas Fed measure that removes top and bottom price-change outliers each month; smoothest real-time inflation signal"),
    ("Sticky CPI",        "Atlanta Fed CPI for goods/services with infrequent price changes (e.g., rents, insurance); best predictor of persistent inflation"),
    ("Flexible CPI",      "CPI for goods/services with frequent price resets (e.g., gasoline, airfare); leads overall CPI by several months"),
    ("OER",               "Owners' Equivalent Rent — imputed monthly rent a homeowner would pay; ~26% of CPI weight with a structural 12–18 month lag"),

    # ── Inflation — Expectations & Regime ────────────────────────────────
    ("BEI / Breakeven",   "Market-implied inflation expectations from the gap between TIPS and nominal yields"),
    ("T5YIE / T10YIE",   "5- and 10-year breakeven inflation rates from TIPS pricing"),
    ("TIPS",              "Treasury bonds whose principal adjusts with CPI; used to extract inflation expectations"),
    ("Michigan Survey",   "University of Michigan 1-year household inflation expectations survey; sensitive to gas prices and political sentiment"),
    ("Deanchoring",       "Loss of public confidence that inflation will return to target; self-fulfilling through wage and price setting behavior"),
    ("Cyclical Inflation","Demand-driven price pressure — tied to labor market tightness, services spending, wage growth; sensitive to monetary policy"),
    ("Structural Inflation","Supply-side or cost-push inflation — energy shocks, supply chain disruptions, demographic shifts; slower to respond to rate hikes"),
    ("Inflation Regime",  "Classification of current inflation environment: At Target (<2.5%), Disinflation, Elevated, or Entrenched (sticky >4%)"),

    # ── Money Supply ─────────────────────────────────────────────────────
    ("M1",      "Narrow money — physical currency plus checking-account deposits"),
    ("M2",      "Broad money — M1 plus savings accounts, money-market funds, and small CDs"),
    ("Real M2", "M2 adjusted for inflation; sustained contraction signals tightening conditions"),

    # ── Labor — Unemployment & Participation ─────────────────────────────
    ("U-3 (Unemployment)", "Official unemployment rate — share of the labor force actively seeking but unable to find work"),
    ("U-6 (Underemployment)","Broad labor underutilization: unemployed + marginally attached workers + part-time for economic reasons"),
    ("U6–U3 Gap",          "Difference between U-6 and U-3; widening gap signals rising hidden slack — workers underemployed, not counted as unemployed"),
    ("LFPR",               "Labor Force Participation Rate — working-age adults in the labor force (%)"),
    ("Emp/Pop Ratio",      "Share of all working-age adults employed; unaffected by participation shifts"),
    ("Sahm Rule",          "Recession indicator triggered when the 3-month avg unemployment rate rises 0.5pp above its 12-month low"),

    # ── Labor — Employment & Claims ──────────────────────────────────────
    ("Nonfarm Payrolls",   "Net monthly job additions across all non-agricultural sectors"),
    ("AHE",                "Average Hourly Earnings — monthly wage growth measure; YoY above 4% in a tight market signals wage-price pressure"),
    ("ECI",                "Employment Cost Index — quarterly measure of total compensation costs including benefits; less volatile than AHE"),
    ("Initial Claims",     "Weekly new unemployment insurance filings (ICSA); a timely leading indicator of layoff activity"),
    ("Continued Claims",   "Number of people currently receiving unemployment benefits (CCSA); reflects difficulty finding new jobs"),
    ("Absorption Ratio",   "Ratio of continued claims to initial claims; rising ratio means fired workers are taking longer to find new jobs"),
    ("Temp Employment",    "Temporary help services payrolls (TEMPHELPS); often turns negative 3–6 months before broader job losses"),

    # ── Labor — JOLTS, Wages & Productivity ──────────────────────────────
    ("JOLTS",              "Job Openings and Labor Turnover Survey — monthly BLS survey of open positions, hires, quits, and layoffs"),
    ("Job Openings",       "Total unfilled positions (JTSJOL); > 10M openings historically signals very tight labor conditions"),
    ("Quits Rate",         "Share of employed workers voluntarily quitting (JTSQUR); high rate = workers confident in finding better jobs"),
    ("Layoffs Rate",       "Share of employed workers involuntarily separated (JTSLDR); rising rate = employer-driven labor market deterioration"),
    ("Beveridge Curve",    "Relationship between job openings and unemployment; shift outward = matching inefficiency or structural mismatch"),
    ("V/U Ratio",          "Vacancies-to-Unemployed ratio; above 1.0 means more jobs than job seekers — historically rare and inflationary"),
    ("Nonfarm Productivity","Output per hour worked in the nonfarm business sector (OPHNFB); rising productivity allows wages to grow without inflation"),
    ("ULC",                "Unit Labor Costs — labor cost per unit of output (ULCNFB); sustained rise above 3% YoY signals inflation pressure"),
    ("Real Wage Growth",   "Nominal wage growth minus CPI; negative real wages erode purchasing power even when nominal wages rise"),
    ("LDI",                "Labor Deterioration Index — composite 0–100 score weighting unemployment, claims, JOLTS, and payrolls; >60 = significant stress"),

    # ── Leading Indicators & Business Cycle ──────────────────────────────
    ("LEI",                "Conference Board Leading Economic Index — composite of 10 forward-looking indicators; sustained decline signals recession"),
    ("BCI",                "Business Cycle Indicator — normalized composite of leading indicators scaled 0–100; <40 = contraction territory"),
    ("ISM PMI",            "ISM Manufacturing Purchasing Managers' Index; above 50 = expansion, below 50 = contraction"),
    ("ISM New Orders",     "ISM sub-index of new manufacturing orders; the most forward-looking component of the PMI"),
    ("Building Permits",   "Monthly new residential building permits (SAAR); a leading indicator for housing activity 3–6 months ahead"),
    ("NBER Recession",     "Official U.S. recession dates determined by the NBER Business Cycle Dating Committee; typically declared months after onset"),
    ("Diffusion Index",    "Share of indicators improving minus share deteriorating; above 50 = net expansion, below 50 = net contraction"),

    # ── Financial Conditions ──────────────────────────────────────────────
    ("FCI",                "Financial Conditions Index — composite of rates, spreads, equity prices, and volatility into a single tightness score"),
    ("NFCI",               "National Financial Conditions Index — Chicago Fed weekly gauge across 105 variables; negative = easy, positive = tight"),
    ("STLFSI",             "St. Louis Fed Financial Stress Index — weekly composite; values above 0 indicate above-average financial stress"),
    ("SOFR",               "Secured Overnight Financing Rate — overnight repo benchmark; LIBOR replacement since 2023"),
    ("Reverse Repo (RRP)", "Fed facility where counterparties park cash overnight; high usage = ample reserves parked at the Fed; drawdown reflects reserve redistribution, T-bill issuance absorbing MMF demand, or genuine liquidity reduction — context required"),
    ("NFCI Credit",        "NFCI subindex isolating credit conditions specifically; most sensitive component to bank lending tightness"),
    ("CP-Tsy Spread",      "Commercial paper rate minus T-bill rate — short-term funding stress indicator; spikes during financial crises"),

    # ── Credit Markets ────────────────────────────────────────────────────
    ("HY Spread (OAS)",    "Extra yield demanded above Treasuries for below-investment-grade bonds; > 600bps signals distress"),
    ("IG Spread (OAS)",    "Extra yield demanded above Treasuries for investment-grade corporate bonds"),
    ("OAS",                "Option-Adjusted Spread — yield spread net of embedded call/put option value"),
    ("SLOOS",              "Senior Loan Officer Opinion Survey — quarterly Fed survey on bank lending standards and demand"),
    ("Lending Standards",  "Net % of banks tightening loan conditions; tightening > 40% historically signals credit contraction ahead"),
    ("CRE",                "Commercial Real Estate — offices, retail, apartments, and industrial property"),
    ("CRE Lending Stds",  "Net % banks tightening CRE loan standards (SLOOS); > 40% signals credit contraction ahead"),
    ("CC Delinquency",     "Share of credit card loans past due — broad consumer credit health signal"),
    ("Delinquency Rate",   "Share of loans past due (all loans, CRE, or mortgages); rising rate signals deteriorating credit quality"),
    ("Bank Deposits",      "Total deposits at commercial banks (DPSACBM027SBOG); sharp decline signals deposit flight or tightening liquidity"),
    ("Charge-off Rate",    "Loans written off as unrecoverable as a % of total loans; lags delinquencies by 1–3 quarters"),

    # ── Markets & Valuation ───────────────────────────────────────────────
    ("VIX",                "CBOE Volatility Index — market's 30-day implied volatility; the 'fear gauge'; > 30 signals elevated stress"),
    ("CAPE / Shiller P/E", "S&P 500 price divided by 10-year inflation-adjusted average earnings; > 30 historically precedes lower future returns"),
    ("Trailing P/E",       "Stock price divided by last 12 months of reported earnings"),
    ("10Y Treasury Yield", "Benchmark long-term government rate; affects all asset prices via the risk-free rate"),

    # ── Policy & Fiscal ───────────────────────────────────────────────────
    ("Fed Funds Rate",     "Fed's overnight policy rate set by the FOMC; the primary monetary policy tool"),
    ("Real FF Rate",       "Fed Funds Rate minus Core CPI YoY — the inflation-adjusted policy stance; negative = accommodative"),
    ("QE / QT",            "Quantitative Easing (buying assets to inject reserves) / Tightening (shrinking the Fed balance sheet)"),
    ("WALCL",              "Fed balance sheet total assets; expanded via QE, shrunk via QT"),
    ("SAAR",               "Seasonally Adjusted Annual Rate — removes seasonal patterns, expressed at annual pace"),
    ("Debt / GDP",         "Federal debt as a percentage of gross domestic product; above 100% limits fiscal flexibility"),
    ("Interest / Receipts","Federal interest payments as a share of total government revenues; rising ratio crowds out discretionary spending"),

    # ── Yield Curve & Recession Indicators ───────────────────────────────
    ("Yield Curve (10Y-2Y)","Spread between 10-year and 2-year Treasury yields; inversion has preceded every US recession since 1970"),
    ("Inverted Curve",      "When short rates exceed long rates (spread < 0) — markets pricing near-term risk higher than long-run growth"),
    ("NY Fed Rec. Prob.",   "Model-based 12-month recession probability using yield curve slope; above 30% has historically been a reliable signal"),

    # ── Housing ───────────────────────────────────────────────────────────
    ("Housing Starts",     "New residential units started monthly (SAAR); a leading indicator for construction employment and materials"),
    ("Case-Shiller HPI",   "S&P/Case-Shiller repeat-sales home price index; published with ~2-month lag"),
    ("Mortgage Rate (30Y)","Freddie Mac Primary Mortgage Market Survey; directly affects housing affordability and purchase volume"),

    # ── Dollar ────────────────────────────────────────────────────────────
    ("Dollar Index (DTWEX)","Trade-weighted U.S. dollar index vs. 26 currencies; rapid appreciation tightens global financial conditions"),

    # ── Global Macro & FX ─────────────────────────────────────────────────
    ("Brent Crude",         "European benchmark for global oil pricing; closely tracked because it sets the reference price for ~2/3 of world oil contracts"),
    ("PPI Commodities",     "Producer Price Index for all commodities; a broad upstream inflation gauge that leads CPI by 2–3 months"),
    ("Gold Price",          "Safe-haven and inflation hedge; rising gold reflects dollar stress, elevated inflation fear, or geopolitical risk premium"),
    ("DXY / Broad USD",     "Trade-weighted U.S. dollar index; strong dollar tightens global financial conditions for dollar-denominated debt holders"),
    ("EUR/USD",             "Most-traded FX pair globally; reflects Fed-ECB policy divergence and Eurozone growth/risk premium vs. the U.S."),
    ("JPY/USD",             "Yen per dollar; rising JPY/USD = weaker yen = BOJ/Fed divergence. Yen is the world's primary carry-trade funding currency"),
    ("CNY/USD",             "Yuan per dollar; rapid depreciation signals China growth stress or capital outflows with global EM contagion risk"),
    ("ECB Rate",            "European Central Bank deposit facility rate; the benchmark monetary policy instrument for the 20-nation Euro Area"),
    ("BOJ Rate",            "Bank of Japan overnight call rate; BOJ normalization from near-zero is a major global tail risk via yen carry unwind"),
    ("CB Divergence",       "Gaps between major central bank policy rates; drives FX, capital flows, and global financial conditions asymmetries"),

    # ── Macro Regime ─────────────────────────────────────────────────────
    ("Macro Regime",        "Classification of the current macro environment based on growth, inflation, financial conditions, and credit dimensions"),
    ("Goldilocks",          "Above-trend growth + contained inflation + neutral FCI — the optimal backdrop for broad risk appetite"),
    ("Reflation",           "Above-trend growth + rising inflation — commodity and cyclical outperformance; central banks may be behind the curve"),
    ("Stagflation",         "Below-trend growth + elevated inflation — worst policy trade-off; rate cuts risk entrenching inflation"),
    ("Disinflation",        "Softening growth + moderating inflation — rate cuts become viable; duration assets typically outperform"),
    ("Liquidity Boom",      "Easy financial conditions + credit expansion + asset price inflation — watch for late-cycle excess"),
    ("Tightening Cycle",    "Rising rates + tightening FCI — overtightening risk (inversion, spread widening, claims acceleration)"),
    ("Balance-Sheet Recession","Credit collapse + deflation risk — the defining challenge is debt deleveraging, not monetary policy adjustment"),
    ("Dimension Score",     "Normalized 0-to-1 risk score for each macro dimension: 0 = healthy, 1 = severely stressed"),

    # ── Structural Macro ─────────────────────────────────────────────────
    ("Output Gap",          "(Real GDP − Potential GDP) / Potential GDP × 100%; positive = overheating, negative = economic slack"),
    ("Potential GDP",       "CBO's estimate of the economy's non-inflationary production capacity — sets the benchmark for the output gap"),
    ("r* (Neutral Rate)",   "The real interest rate consistent with full employment and stable inflation. Unobservable; estimated with state-space models (e.g., Holston-Laubach-Williams). Often proxied by the 10Y TIPS yield, though TIPS also embeds a real term premium (~0.5–1.5%) that causes it to overstate r*, especially during tightening cycles."),
    ("Real FF Rate",        "Nominal Fed Funds Rate minus core CPI YoY — policy stance indicator: above the 10Y real rate = restrictive; below = accommodative"),
    ("10Y TIPS Yield",      "Yield on 10Y inflation-protected Treasuries. A market-observable long-run real rate benchmark, but not equivalent to r*: the TIPS yield also embeds a real term premium and a TIPS liquidity discount, both absent from the neutral rate concept."),
    ("Working Age Pop.",    "Population aged 15–64 (FRED: LFWA64TTUSM647S); slow-moving driver of labor supply and long-run potential GDP"),
    ("Globalization Index", "Proxy from USD trend: dollar appreciation signals deglobalization; commodity supercycles track globalization phases"),

    # ── Fiscal Analytics ─────────────────────────────────────────────────
    ("Interest/Receipts",   "Federal interest payments as % of total tax receipts — the fraction of each tax dollar consumed by debt service"),
    ("Primary Balance",     "Budget balance excluding interest payments; the government's controllable fiscal position and key sustainability metric"),
    ("Primary Deficit",     "Primary balance below zero — government spending (ex-interest) exceeds revenues; the most concerning fiscal signal"),
    ("Fiscal Impulse",      "Year-over-year change in the deficit/GDP ratio; positive = fiscal stimulus (expanding deficit), negative = fiscal drag"),
    ("r > g Risk",          "When real interest rates exceed real GDP growth, debt/GDP rises without a primary surplus — the debt spiral condition"),
    ("Rollover Risk",       "Risk that maturing debt must be refinanced at materially higher rates, increasing the interest burden mechanically"),
    ("Debt Stabilization",  "Condition where debt/GDP stops rising; requires: primary surplus ≥ (r − g) × debt/GDP, or sufficiently high growth"),

    # ── Probabilistic Forecasting ─────────────────────────────────────────
    ("Recession Probability","Model-generated probability (0–100%) of recession onset within a specified horizon (6M / 12M / 24M)"),
    ("Logistic Regression", "Statistical model that maps macro indicator values to a probability between 0% and 100% via a sigmoid function"),
    ("Feature Contribution","Each indicator's additive contribution to the total recession probability score; identifies the dominant risk drivers"),
    ("Confidence Band",     "Bootstrap-estimated uncertainty range around the central probability; wider bands = less historical data or regime change"),
    ("OOS Backtest",        "Out-of-sample backtest — model trained on data before a given date, tested on data it never saw during training"),
    ("Hit Rate",            "Share of recessions correctly flagged by the model above a chosen probability threshold"),
    ("False Positive Rate", "Share of non-recession periods incorrectly flagged as recession risk; the trade-off against hit rate"),

    # ── Statistical Concepts ──────────────────────────────────────────────
    ("Z-Score",             "Standard deviations from the historical mean; ±2σ marks the outer 5% of a normal distribution"),
    ("Percentile",          "Rank of the current reading vs. historical distribution; 90th percentile = higher than 90% of all past readings"),
    ("YoY / MoM",           "Year-over-Year / Month-over-Month percent change — annualized and monthly rates of change"),
    ("Rolling Window",      "Historical lookback period for computing statistics; e.g., 20-year window updates each month as new data arrives"),
    ("Revision Risk",       "Magnitude of typical revisions to initial data releases; high for payrolls and GDP, low for daily market prices"),
]


def _build_methodology(story, re: RiskEngine, dl):
    """
    Methodology Appendix — transparent, reproducible scoring documentation.
    Covers: normalization methods, threshold system, confidence scoring,
    indicator registry table, limitations, and key references.
    """
    story += _section_rule("Methodology Appendix", anchor_id="sec_methodology")

    # ── 1. Framework overview ─────────────────────────────────────────────
    story.append(_prose(
        "<b>Overview.</b>  Every indicator in this report is scored using one of four "
        "normalization methods. All methods are computed at report-generation time from "
        "the same historical data used for the charts — no pre-computed lookup tables. "
        "The methodology is intentionally transparent so that any reader can reproduce "
        "a score given the raw FRED data and the parameters documented below."
    ))
    story.append(_prose(
        "Color codes (green / yellow / red) are <i>descriptive</i>, not predictive. "
        "They indicate where the current reading sits relative to the historical "
        "distribution or a literature-calibrated threshold — not whether a recession "
        "or crisis will occur. Combinations of elevated readings across multiple "
        "independent dimensions carry more information than any single indicator."
    ))
    story.append(Spacer(1, 0.08 * inch))

    # ── 2. Normalization methods table ────────────────────────────────────
    story.append(_subsection("Normalization Methods"))
    story.append(_prose(
        "Each indicator is assigned one of four scoring methods in the indicator registry. "
        "The method determines how the current value is translated into a risk level, and "
        "what label is displayed on the card."
    ))
    story.append(Spacer(1, 0.04 * inch))

    _MCW = [CONTENT_W * f for f in (0.18, 0.20, 0.15, 0.15, 0.32)]
    method_header = [
        Paragraph("<b>Method</b>",      _ps("mh", fontSize=7, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
        Paragraph("<b>Used For</b>",    _ps("mh", fontSize=7, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
        Paragraph("<b>Green Label</b>", _ps("mh", fontSize=7, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
        Paragraph("<b>Yellow Label</b>",_ps("mh", fontSize=7, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
        Paragraph("<b>Red Label</b>",   _ps("mh", fontSize=7, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
    ]
    method_rows = [
        ["Fixed Thresholds",
         "Policy targets, literature-calibrated breakpoints",
         "Within Range",
         "Elevated",
         "Stressed"],
        ["Percentile\n(higher=worse)",
         "Most market & macro indicators",
         "< 75th pctile",
         "75th–90th pctile",
         "> 90th pctile"],
        ["Percentile\n(lower=worse)",
         "Labor utilization, housing starts",
         "> 25th pctile",
         "10th–25th pctile",
         "< 10th pctile"],
        ["Z-Score",
         "Industrial production, structural series",
         "Within 1σ",
         "1–2σ from avg",
         "Beyond 2σ"],
        ["Standardized\nIndex",
         "NFCI, STLFSI (pre-standardized by issuer)",
         "Below Average",
         "Above Average",
         "High Stress"],
    ]
    _mst = _ps("mc", fontSize=7, textColor=BODY_CLR, leading=10)
    tbl_methods = Table(
        [method_header] + [[Paragraph(c, _mst) for c in row] for row in method_rows],
        colWidths=_MCW,
    )
    tbl_methods.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  HDR_BG),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#f7fafc")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f7fafc"), colors.white]),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW",     (0, 0), (-1, 0),  0.5, colors.white),
        ("LINEBELOW",     (0, 1), (-1, -1), 0.3, RULE_CLR),
        ("BOX",           (0, 0), (-1, -1), 0.5, RULE_CLR),
    ]))
    story.append(tbl_methods)
    story.append(Spacer(1, 0.06 * inch))
    story.append(_prose(
        "<b>Calibration window.</b>  Percentile and z-score methods use a rolling "
        "historical window (20 or 30 years, documented per indicator below) ending at "
        "the most recent observation. A minimum of 24 monthly observations is required; "
        "if insufficient history is available the method falls back to fixed thresholds. "
        "The window is chosen to capture at least one full economic cycle while "
        "remaining relevant to current structural conditions."
    ))
    story.append(Spacer(1, 0.10 * inch))

    # ── 3. Confidence scoring ─────────────────────────────────────────────
    story.append(_subsection("Confidence Scoring"))
    story.append(_prose(
        "Each indicator card displays a confidence score (0–100). This is a composite "
        "metric that answers: <i>how much should I trust the current signal?</i> "
        "It does not affect the color coding — it qualifies it. A red signal with "
        "confidence 40 warrants more caution than a red signal with confidence 85."
    ))
    story.append(Spacer(1, 0.04 * inch))

    _CCW = [CONTENT_W * f for f in (0.22, 0.10, 0.68)]
    conf_header = [
        Paragraph("<b>Component</b>",  _ps("ch", fontSize=7, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
        Paragraph("<b>Weight</b>",     _ps("ch", fontSize=7, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
        Paragraph("<b>Description</b>",_ps("ch", fontSize=7, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
    ]
    conf_rows = [
        ["Timeliness",            "25%",
         "Days since last observation relative to expected update frequency "
         "(daily=5d, weekly=14d, monthly=60d, quarterly=120d). "
         "Stale data scores 0."],
        ["Coverage",              "20%",
         "Years of history available, reaching full score at 20+ years. "
         "Insufficient history limits statistical scoring reliability."],
        ["Predictive Reliability","35%",
         "Literature-based signal reliability assigned per indicator (0.40–1.00). "
         "Draws on Estrella & Mishkin (1998), Conference Board LEI methodology, "
         "NBER recession research, and Fed SLOOS academic literature."],
        ["Revision Risk",         "20%",
         "Penalty for data subject to large initial-release revisions. "
         "Low=1.00 (daily market data, NBER), "
         "Medium=0.75 (CPI, PCE, LEI), "
         "High=0.50 (Nonfarm Payrolls, S&P earnings)."],
    ]
    _cst = _ps("cc", fontSize=7, textColor=BODY_CLR, leading=10)
    tbl_conf = Table(
        [conf_header] + [[Paragraph(c, _cst) for c in row] for row in conf_rows],
        colWidths=_CCW,
    )
    tbl_conf.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  HDR_BG),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f7fafc"), colors.white]),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW",     (0, 0), (-1, 0),  0.5, colors.white),
        ("LINEBELOW",     (0, 1), (-1, -1), 0.3, RULE_CLR),
        ("BOX",           (0, 0), (-1, -1), 0.5, RULE_CLR),
    ]))
    story.append(tbl_conf)
    story.append(Spacer(1, 0.10 * inch))

    # ── 4. Indicator registry table ────────────────────────────────────────
    story.append(_subsection("Indicator Registry"))
    story.append(_prose(
        "Complete registry of all scored indicators. "
        "<b>Method</b>: normalization approach. "
        "<b>Win</b>: calibration window in years. "
        "<b>Rel</b>: predictive reliability 0–1. "
        "<b>Rev</b>: revision risk (L=Low, M=Medium, H=High). "
        "Threshold Basis summarizes the literature or statistical source for the scoring threshold."
    ))
    story.append(Spacer(1, 0.04 * inch))

    _RCW = [
        CONTENT_W * 0.11,   # Short name
        CONTENT_W * 0.13,   # Method
        CONTENT_W * 0.05,   # Window
        CONTENT_W * 0.05,   # Reliability
        CONTENT_W * 0.04,   # Rev risk
        CONTENT_W * 0.62,   # Threshold rationale
    ]
    reg_header = [
        Paragraph("<b>Indicator</b>",    _ps("rh", fontSize=6.5, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
        Paragraph("<b>Method</b>",       _ps("rh", fontSize=6.5, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
        Paragraph("<b>Win</b>",          _ps("rh", fontSize=6.5, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
        Paragraph("<b>Rel</b>",          _ps("rh", fontSize=6.5, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
        Paragraph("<b>Rev</b>",          _ps("rh", fontSize=6.5, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
        Paragraph("<b>Threshold Basis</b>", _ps("rh", fontSize=6.5, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
    ]

    _method_abbrev = {
        "fixed_thresholds":   "Fixed",
        "percentile":         "Pctile",
        "zscore":             "Z-Score",
        "standardized_index": "Std. Index",
    }
    _rev_abbrev = {"low": "L", "medium": "M", "high": "H"}

    reg_rows = [reg_header]
    alt = False
    for cat in CATEGORY_ORDER:
        cat_ids = [sid for sid in REGISTRY
                   if REGISTRY[sid]["category"] == cat and sid in dl.available]
        if not cat_ids:
            continue
        cat_label = CATEGORY_LABELS.get(cat, cat).upper()
        reg_rows.append([
            Paragraph(cat_label,
                      _ps("rcat", fontSize=6, fontName="Helvetica-Bold",
                          textColor=colors.HexColor("#4a5568"))),
            Paragraph("", _ps("re", fontSize=6)),
            Paragraph("", _ps("re", fontSize=6)),
            Paragraph("", _ps("re", fontSize=6)),
            Paragraph("", _ps("re", fontSize=6)),
            Paragraph("", _ps("re", fontSize=6)),
        ])
        for sid in cat_ids:
            m = REGISTRY[sid]
            method_str  = _method_abbrev.get(m.get("normalization_method", "fixed_thresholds"), "Fixed")
            window_str  = str(m.get("calibration_window_years", 20)) + "y"
            rel_str     = f"{m.get('predictive_reliability', 0.5):.2f}"
            rev_str     = _rev_abbrev.get(m.get("revision_risk", "medium"), "M")
            rationale   = m.get("threshold_rationale", "")
            _rst = _ps("re", fontSize=6, textColor=BODY_CLR, leading=8.5)
            _rmt = _ps("rm", fontSize=6, textColor=MUTED_CLR, leading=8.5)
            reg_rows.append([
                Paragraph(m.get("short_name", sid), _rst),
                Paragraph(method_str,  _rmt),
                Paragraph(window_str,  _rmt),
                Paragraph(rel_str,     _rmt),
                Paragraph(rev_str,     _rmt),
                Paragraph(rationale,   _rmt),
            ])
            alt = not alt

    tbl_reg = Table(reg_rows, colWidths=_RCW)
    reg_style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  HDR_BG),
        ("FONTSIZE",      (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW",     (0, 0), (-1, 0),  0.5, colors.white),
        ("LINEBELOW",     (0, 1), (-1, -1), 0.3, RULE_CLR),
        ("BOX",           (0, 0), (-1, -1), 0.5, RULE_CLR),
    ]
    # Shade category-header rows differently
    for row_idx, row in enumerate(reg_rows):
        if row_idx == 0:
            continue
        cell_text = row[0].text if hasattr(row[0], "text") else ""
        if any(lbl.upper() in cell_text for lbl in CATEGORY_LABELS.values()):
            reg_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx),
                               colors.HexColor("#edf2f7")))
            reg_style.append(("SPAN", (0, row_idx), (-1, row_idx)))
        elif row_idx % 2 == 0:
            reg_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.white))
        else:
            reg_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx),
                               colors.HexColor("#f7fafc")))
    tbl_reg.setStyle(TableStyle(reg_style))
    story.append(tbl_reg)
    story.append(Spacer(1, 0.10 * inch))

    # ── 5. Limitations ─────────────────────────────────────────────────────
    story.append(_subsection("Limitations & Caveats"))
    limitations = [
        ("Survivorship in calibration windows",
         "The calibration window includes only the period for which data is available. "
         "Indicators with short histories (e.g., SOFR since 2018) cannot be reliably "
         "percentile-scored and fall back to fixed thresholds."),
        ("Regime shifts",
         "A 20-year rolling window may include structural breaks (e.g., post-2008 "
         "zero-rate environment, post-2020 M1 definitional change). Percentile scores "
         "calibrated against a structurally different regime may mis-classify current readings."),
        ("Revision risk",
         "Many indicators (Nonfarm Payrolls, GDP, PCE) are substantially revised "
         "after initial release. The confidence score penalizes high-revision series, "
         "but the scored value is always the latest available — which may itself be revised."),
        ("Composite vs. individual",
         "Crisis dimension scores use worst-of aggregation. This can be overly "
         "conservative: a single elevated indicator can lift an entire dimension "
         "even when the other components are benign."),
        ("No prediction",
         "All scores are backward-looking relative to the calibration window. "
         "A reading at the 95th percentile of history tells you it is historically "
         "unusual — it does not predict that a crisis will follow."),
    ]
    _lst = _ps("lim", fontSize=7, textColor=BODY_CLR, leading=10.5)
    _lmt = _ps("limd", fontSize=7, textColor=MUTED_CLR, leading=10.5)
    for title, desc in limitations:
        lim_row = Table(
            [[Paragraph(f"<b>{title}</b>", _lst), Paragraph(desc, _lmt)]],
            colWidths=[CONTENT_W * 0.26, CONTENT_W * 0.74],
        )
        lim_row.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (0, -1), 8),
            ("LEFTPADDING",   (1, 0), (1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW",     (0, 0), (-1, 0),  0.3, RULE_CLR),
        ]))
        story.append(lim_row)
    story.append(Spacer(1, 0.10 * inch))

    # ── 6. Key references ──────────────────────────────────────────────────
    story.append(_subsection("Key References"))
    refs = [
        ("Estrella & Mishkin (1998)",
         "Predicting U.S. Recessions: Financial Variables as Leading Indicators. "
         "Review of Economics and Statistics 80(1). "
         "Basis for NY Fed recession probability model (RECPROUSM156N) and yield-curve thresholds."),
        ("Conference Board LEI Methodology",
         "The Conference Board Leading Economic Index® for the U.S. — "
         "Methodological Guide (2022). "
         "Basis for USSLIND thresholds and recession-precursor interpretation."),
        ("Shiller, R. (2000)",
         "Irrational Exuberance. Princeton University Press. "
         "Basis for CAPE (SP500_CAPE) thresholds and historical calibration."),
        ("Bernanke & Lown (1991)",
         "The Credit Crunch. Brookings Papers on Economic Activity 1991(2). "
         "Basis for SLOOS C&I lending standards threshold (DRTSCILM ≥ 40%)."),
        ("IMF Fiscal Monitor",
         "Various editions. "
         "Basis for sovereign debt/GDP risk thresholds (GFDEGDQ188S)."),
        ("Sahm, C. (2019)",
         "Direct Stimulus Payments to Individuals. "
         "In Recession Ready, Hamilton Project. "
         "Context for unemployment rate trend interpretation (Sahm Rule)."),
        ("Chicago Fed NFCI Methodology",
         "Brave & Butters (2011), Monitoring Financial Stability: A Financial Conditions Index Approach. "
         "Economic Perspectives Q1 2011. "
         "Basis for NFCI index interpretation."),
    ]
    _rst2 = _ps("rft", fontSize=7, fontName="Helvetica-Bold", textColor=BODY_CLR, leading=10)
    _rdt  = _ps("rfd", fontSize=7, textColor=MUTED_CLR, leading=10)
    for author, desc in refs:
        ref_row = Table(
            [[Paragraph(author, _rst2), Paragraph(desc, _rdt)]],
            colWidths=[CONTENT_W * 0.26, CONTENT_W * 0.74],
        )
        ref_row.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (0, -1), 8),
            ("LEFTPADDING",   (1, 0), (1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW",     (0, 0), (-1, 0),  0.3, RULE_CLR),
        ]))
        story.append(ref_row)


def _build_global_macro(story, gme, rge, sme, dl):
    story += _section_rule("Global Macro & FX", anchor_id="sec_global_macro")
    print("  Rendering global macro charts...")

    dash  = gme.global_dashboard()
    comm  = dash["commodities"]
    fx    = dash["fx"]
    cb    = dash["cb_divergence"]
    rr    = sme.real_rates()
    og    = sme.output_gap()
    demo  = sme.demographic_pressure()

    def _fv(v, fmt=".2f", sfx=""):
        return f"{v:{fmt}}{sfx}" if v is not None else "N/A"

    story.append(_prose(
        f"<b>Central Bank Divergence:</b> "
        f"Fed {_fv(cb['fed_rate'], '.2f', '%')} | "
        f"ECB {_fv(cb['ecb_rate'], '.2f', '%')} | "
        f"BOJ {_fv(cb['boj_rate'], '.2f', '%')} | "
        f"Fed–ECB gap: {_fv(cb['fed_ecb_gap'], '+.2f', 'pp')} — {cb['divergence_label']}  ·  "
        f"<b>USD Broad:</b> {_fv(fx['dxy_yoy'], '+.1f', '% YoY')} ({fx['usd_regime']})  ·  "
        f"<b>Commodity Pressure:</b> {_fv(comm['commodity_pressure'], '+.1f', '% avg YoY')} — {comm['commodity_regime']}"
    ))
    story.append(Spacer(1, 0.08 * inch))

    story.append(_two_charts(
        central_bank_rates_chart(dl, lookback_years=25),
        fx_chart(dl, lookback_years=15),
    ))

    story.append(_subsection("Global Commodity Complex"))
    story.append(_prose(
        f"Brent crude: ${_fv(comm['brent_level'], '.0f')}/bbl ({_fv(comm['brent_yoy'], '+.1f', '% YoY')}). "
        f"Gold: ${_fv(comm['gold_level'], '.0f')}/oz ({_fv(comm['gold_yoy'], '+.1f', '% YoY')}). "
        f"PPI All Commodities: {_fv(comm['ppi_commodities_yoy'], '+.1f', '% YoY')}. "
        "Commodity surges lead consumer prices by 2–3 months via energy and food pass-through. "
        "Gold reflects safe-haven demand and long-run inflation expectations."
    ))
    story.append(_full_chart(commodity_chart(dl, lookback_years=15)))

    story.append(_subsection("Structural & Long-Run Context"))
    story.append(_prose(
        f"<b>Output gap:</b> {_fv(og['gap_current'], '+.2f', '%')} ({og['label']}). "
        f"<b>Real FF rate:</b> {_fv(rr['real_ff'], '+.2f', '%')} vs "
        f"10Y real rate (TIPS proxy) {_fv(rr['tips_10y'], '.2f', '%')} "
        f"(note: TIPS yield includes a real term premium above true r*) → "
        f"Stance: <b>{rr['policy_stance']}</b>. "
        f"<b>Demographics:</b> Working-age population growth "
        f"{_fv(demo['wap_yoy'], '+.2f', '% YoY')} — {demo['label']}. "
        "Demographic deceleration structurally constrains potential GDP growth and "
        "increases the pension/healthcare fiscal burden over the next decade."
    ))
    story.append(_two_charts(
        output_gap_chart(sme),
        real_rates_chart(sme, dl, lookback_years=20),
    ))
    story.append(_full_chart(productivity_chart(sme, lookback_years=20)))


def _build_macro_regime(story, rge, dl):
    story += _section_rule("Macro Regime Classification", anchor_id="sec_regime")
    print("  Rendering macro regime charts...")

    regime_name, regime_color, regime_desc = rge.classify()
    dims = rge.dimension_scores()

    def _fv(v, fmt=".2f", sfx=""):
        return f"{v:{fmt}}{sfx}" if v is not None else "N/A"

    story.append(_prose(
        f"<b>Current Regime: {regime_name}</b>  ·  {regime_desc}"
    ))
    story.append(Spacer(1, 0.06 * inch))

    # Dimension scores table
    dim_rows = [
        [Paragraph("<b>Dimension</b>", _ps("dh", fontSize=7, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
         Paragraph("<b>Score</b>",     _ps("dh", fontSize=7, fontName="Helvetica-Bold", textColor=HDR_TEXT)),
         Paragraph("<b>Label</b>",     _ps("dh", fontSize=7, fontName="Helvetica-Bold", textColor=HDR_TEXT))],
    ]
    for dim_key, dim_label in [("growth", "Growth"), ("inflation", "Inflation"),
                                ("financial", "Financial Conditions"), ("credit", "Credit")]:
        d = dims[dim_key]
        score = d["score"]
        lbl   = d["label"]
        clr   = colors.HexColor("#276749") if score < 0.35 else (
                colors.HexColor("#975a16") if score < 0.65 else colors.HexColor("#9b2c2c"))
        dim_rows.append([
            Paragraph(dim_label, _ps("dc", fontSize=7, textColor=BODY_CLR)),
            Paragraph(f"{score:.2f}", _ps("dc", fontSize=7, textColor=clr,
                                          fontName="Helvetica-Bold")),
            Paragraph(lbl, _ps("dc", fontSize=7, textColor=BODY_CLR)),
        ])

    tbl_dims = Table(dim_rows, colWidths=[THIRD_W, THIRD_W * 0.5, THIRD_W * 1.5])
    tbl_dims.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  HDR_BG),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f7fafc"), colors.white]),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("LINEBELOW",     (0, 0), (-1, -1), 0.3, RULE_CLR),
        ("BOX",           (0, 0), (-1, -1), 0.5, RULE_CLR),
    ]))
    story.append(tbl_dims)
    story.append(Spacer(1, 0.08 * inch))

    story.append(_two_charts(
        regime_scores_chart(rge),
        regime_timeline_chart(rge, dl, lookback_years=20),
    ))

    story.append(_subsection("Regime Definitions & Portfolio Implications"))
    for name, info in MACRO_REGIMES.items():
        if name == "Uncertain":
            continue
        implications = {
            "Goldilocks":              "Broad risk-on; equities, credit, cyclicals outperform. Duration neutral.",
            "Reflation":               "Commodities, energy, financials outperform. Underweight duration.",
            "Stagflation":             "Real assets, commodities, TIPS. Equities underperform; avoid duration.",
            "Disinflation":            "Duration outperforms. Quality equities. Avoid commodities.",
            "Liquidity Boom":          "Risk assets and credit outperform. Watch for late-cycle imbalances.",
            "Tightening Cycle":        "Underweight duration. Financials mixed. Quality over quantity in credit.",
            "Balance-Sheet Recession": "Cash and short-duration. Defensive equities. Avoid HY credit.",
        }.get(name, "Context-dependent positioning.")
        story.append(_prose(f"<b>{name}:</b> {implications}"))

    adj = rge.get_threshold_adjustments()
    if adj:
        story.append(Spacer(1, 0.06 * inch))
        story.append(_prose(
            "<b>Regime-adjusted thresholds:</b> In the current regime, the following static "
            "thresholds are overridden: " +
            "; ".join(f"{k} → {v}" for k, v in adj.items()) + "."
        ))


def _build_fiscal_analytics(story, fae, dl):
    story += _section_rule("Fiscal Analytics & Sustainability", anchor_id="sec_fiscal_analytics")
    print("  Rendering fiscal analytics charts...")

    fdash = fae.fiscal_dashboard()
    sus   = fdash["sustainability"]
    prim  = fdash["primary"]
    traj  = fdash["trajectory"]
    roll  = fdash["rollover"]

    def _fv(v, fmt=".2f", sfx=""):
        return f"{v:{fmt}}{sfx}" if v is not None else "N/A"

    story.append(_prose(
        f"<b>Status: {sus['label']}</b>  |  "
        f"Debt/GDP: {_fv(sus['debt_gdp'], '.1f', '%')}  |  "
        f"Interest/Receipts: {_fv(sus['int_receipts'], '.1f', '%')}  |  "
        f"Interest/GDP: {_fv(sus['int_gdp'], '.2f', '%')}  |  "
        f"Primary Balance: {_fv(prim['primary_pct_gdp'], '+.1f', '% GDP')}  |  "
        f"Trajectory: {traj['label']}"
    ))
    story.append(Spacer(1, 0.08 * inch))

    story.append(_prose(
        "The Interest/Receipts ratio — not the debt/GDP level — is the most operationally "
        "meaningful sustainability metric. It answers: 'Of every tax dollar collected, how "
        "many cents go directly to paying interest before any government service is funded?' "
        "This ratio is non-discretionary: unlike defense or welfare spending, interest cannot "
        "be cut. A rising Interest/Receipts ratio crowds out all other fiscal functions."
    ))
    story.append(Spacer(1, 0.06 * inch))

    story.append(_full_chart(debt_trajectory_chart(dl, lookback_years=40)))
    story.append(_full_chart(debt_service_chart(fae, lookback_years=30)))

    story.append(_subsection("Primary Balance & Fiscal Impulse"))
    story.append(_prose(
        "The primary balance (total balance + interest payments) is the government's "
        "controllable fiscal position. A primary surplus of (r − g) × debt/GDP is required "
        "to stabilize the debt ratio, where r = real interest rate and g = real GDP growth. "
        f"Current primary balance: {_fv(prim['primary_pct_gdp'], '+.1f', '% GDP')}. "
        f"Total deficit: {_fv(prim['deficit_pct_gdp'], '+.1f', '% GDP')} "
        f"(difference = interest burden of {_fv(prim['interest_B'], ',.0f', ' $B SAAR')})."
    ))
    story.append(_two_charts(
        primary_balance_chart(fae, lookback_years=40),
        fiscal_impulse_chart(fae, lookback_years=30),
        small=True,
    ))

    story.append(_subsection("Rollover Exposure & Trajectory"))
    story.append(_prose(
        f"10Y yield: {_fv(roll['current_10y_rate'], '.2f', '%')} vs "
        f"{_fv(roll['rate_5y_ago'], '.2f', '%')} five years ago "
        f"(change: {_fv(roll['rate_change_5y'], '+.2f', 'pp')}). "
        f"Total debt: ${_fv(roll['debt_total_B'], ',.0f')}B. "
        f"Estimated incremental annual rollover cost: "
        f"${_fv(roll['est_incremental_cost_ann_B'], '+,.0f')}B (30% annual rollover assumption). "
        f"Assessment: {roll['label']}. "
        f"5-year fiscal trajectory: debt/GDP changing at "
        f"{_fv(traj['debt_gdp_trend_ann'], '+.2f', 'pp/yr')}. "
        f"Interest acceleration: {_fv(traj['interest_acc_ann'], '+.1f', ' $B/yr ann.')}. "
        f"Revenue growth: {_fv(traj['receipts_trend_ann'], '+.1f', ' $B/yr ann.')}. "
        "The sustainability question is not whether debt/GDP is high, but whether it can "
        "be stabilized — which requires r < g, or a primary surplus, or both."
    ))


def _build_glossary(story):
    """Compact two-column term glossary for non-economist readers."""
    story += _section_rule("Indicator Glossary", anchor_id="sec_glossary")
    story.append(_prose(
        "Plain-English definitions for all acronyms and metrics used in this report. "
        "Click any indicator name in the Summary section above to jump to detailed charts."
    ))
    story.append(Spacer(1, 0.06 * inch))

    # Split into two balanced columns
    mid = (len(_GLOSSARY) + 1) // 2
    left_col = _GLOSSARY[:mid]
    right_col = _GLOSSARY[mid:]

    def _glossary_col(entries):
        rows = []
        for term, defn in entries:
            rows.append(Table(
                [[
                    Paragraph(f"<b>{term}</b>",
                              _ps("gt", fontSize=7, fontName="Helvetica-Bold",
                                  textColor=BODY_CLR, leading=10)),
                    Paragraph(defn,
                              _ps("gd", fontSize=7, textColor=MUTED_CLR, leading=10)),
                ]],
                colWidths=[HALF_W * 0.32, HALF_W * 0.68],
                style=TableStyle([
                    ("LEFTPADDING",  (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING",   (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
                    ("VALIGN",       (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW",    (0, 0), (-1, 0), 0.3, RULE_CLR),
                ]),
            ))
        return rows

    left_items  = _glossary_col(left_col)
    right_items = _glossary_col(right_col)
    max_len = max(len(left_items), len(right_items))
    glos_rows = []
    for i in range(max_len):
        l = left_items[i]  if i < len(left_items)  else Spacer(1, 1)
        r = right_items[i] if i < len(right_items) else Spacer(1, 1)
        glos_rows.append([l, r])
    layout = Table(glos_rows, colWidths=[HALF_W, HALF_W], splitByRow=1)
    layout.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (0, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, -1), 12),
        ("LEFTPADDING",  (1, 0), (1, -1), 12),
        ("RIGHTPADDING", (1, 0), (1, -1), 0),
    ]))
    story.append(layout)


def _build_summary(story, re: RiskEngine, dl):
    story += _section_rule("Summary — All Indicators", anchor_id="sec_summary")
    story.append(_prose(
        "Each card shows the current reading, scoring status, and confidence score. "
        "Click the indicator name to jump to the detailed section. "
        "Labels reflect the scoring method: percentile-scored indicators show "
        "pctile bands (&lt;75th / 75–90th / &gt;90th); z-score indicators show σ-distance; "
        "fixed-threshold indicators show Within Range / Elevated / Stressed. "
        "Context and combinations matter more than any single reading. "
        "See the Methodology Appendix for full scoring details."
    ))
    story.append(Spacer(1, 0.1 * inch))

    available_ids = [sid for sid in REGISTRY
                     if sid in dl.available and REGISTRY[sid].get("show_in_summary", True)]
    for cat in CATEGORY_ORDER:
        ids = [s for s in available_ids if REGISTRY[s]["category"] == cat]
        if not ids:
            continue
        story.append(Paragraph(CATEGORY_LABELS.get(cat, cat).upper(),
                               _ps("cat", fontName="Helvetica-Bold", fontSize=8,
                                   textColor=MUTED_CLR, spaceBefore=8, spaceAfter=4)))
        for row_start in range(0, len(ids), 4):
            story.append(_kpi_row(ids[row_start:row_start + 4], re, dl))
        story.append(Spacer(1, 0.06 * inch))


def _build_inflation(story, iae, dl):
    story += _section_rule("Inflation", anchor_id="sec_inflation")
    print("  Rendering inflation charts...")

    dash_infl = iae.inflation_dashboard()
    regime_label, _ = iae.inflation_regime()

    def _fv(v, fmt=".2f", sfx="%"):
        return f"{v:{fmt}}{sfx}" if v is not None else "N/A"

    story.append(_prose(
        f"<b>Regime: {regime_label}</b>  |  "
        f"Headline CPI: {_fv(dash_infl['headline'])}  |  "
        f"Core CPI: {_fv(dash_infl['core_cpi'])}  |  "
        f"Core PCE: {_fv(dash_infl['core_pce'])}  |  "
        f"Median CPI: {_fv(dash_infl['median_cpi'])}  |  "
        f"Trimmed Mean PCE: {_fv(dash_infl['trimmed_mean_pce'])}  |  "
        f"Supercore: {_fv(dash_infl['supercore'])}  |  "
        f"Sticky CPI: {_fv(dash_infl['sticky_cpi'])}  |  "
        f"OER: {_fv(dash_infl['oer_yoy'])}  |  "
        f"Mich Exp 1Y: {_fv(dash_infl['mich_exp'])}  |  "
        f"5Y Breakeven: {_fv(dash_infl['be_5y'])}  |  "
        f"10Y Breakeven: {_fv(dash_infl['be_10y'])}"
    ))
    story.append(Spacer(1, 0.08 * inch))

    # ── Headline & Core ───────────────────────────────────────────────
    story.append(_two_charts(
        area_chart(dl, "CPIAUCSL", yoy=True, lookback_years=20,
                   threshold_green=2.5, threshold_red=4.5,
                   title="Headline CPI — Year-over-Year %"),
        multi_line_chart(dl,
            [("CPILFESL", C["teal"]), ("PCEPILFE", C["blue"])],
            title="Core CPI vs. Core PCE — YoY %",
            lookback_years=15, yoy=True),
    ))

    # ── Alternative Measures ──────────────────────────────────────────
    story.append(_subsection("Alternative Inflation Measures"))
    story.append(_prose(
        "Median CPI (Cleveland Fed) and Trimmed Mean PCE (Dallas Fed) filter extreme price "
        "movements and are better predictors of future inflation than Core CPI. Sticky Price CPI "
        "captures components that change infrequently — the best predictor of persistent inflation "
        "and underlying regime shifts."
    ))
    story.append(_two_charts(
        inflation_multi_chart(dl, lookback_years=10),
        sticky_flexible_chart(dl, lookback_years=10),
        small=True,
    ))

    # ── Shelter Decomposition ─────────────────────────────────────────
    story.append(_subsection("Shelter Decomposition"))
    story.append(_prose(
        "Owners' Equivalent Rent (OER, ~26% of CPI) lags actual market rents by 12–18 months. "
        "Supercore CPI (Core ex-Shelter) shows services inflation driven by wages — "
        "the Fed's primary focus for determining when disinflation is structural vs. transitory."
    ))
    story.append(_full_chart(shelter_decomposition_chart(dl, lookback_years=8)))

    # ── Expectations ─────────────────────────────────────────────────
    story.append(_subsection("Inflation Expectations"))
    story.append(_prose(
        "Well-anchored expectations are essential to the Fed's credibility. "
        "Michigan 1-year survey persistently above 4% or 5/10-year breakevens above 2.5% "
        "signal deanchoring risk that historically requires more aggressive policy response."
    ))
    story.append(_full_chart(inflation_expectations_chart(dl, lookback_years=10)))
    story.append(Spacer(1, 0.10 * inch))

    # ── Money Supply ─────────────────────────────────────────────────
    story.append(_subsection("Money Supply"))
    print("  Rendering money supply charts...")
    story.append(_two_charts(
        area_chart(dl, "M2SL", yoy=True, lookback_years=20,
                   title="M2 Growth — Year-over-Year %"),
        area_chart(dl, "M2REAL", yoy=True, lookback_years=20,
                   color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                   title="Real M2 Growth — Year-over-Year %"),
    ))
    story.append(_prose(
        "Rapid M2 growth historically precedes inflation with a 12–18 month lag. "
        "Real M2 contraction signals tightening monetary conditions that typically "
        "suppress demand-driven price pressure within 2–4 quarters."
    ))


def _build_labor(story, lae, dl):
    story += _section_rule("Labor Market", anchor_id="sec_labor")
    print("  Rendering labor charts...")

    ldi = lae.labor_deterioration_index()
    wages = lae.wage_pressure()
    ugap = lae.unemployment_gap()
    jolts = lae.jolts_summary()

    def _fv(v, fmt=".1f", sfx=""):
        return f"{v:{fmt}}{sfx}" if v is not None else "N/A"

    story.append(_prose(
        f"<b>Labor Deterioration Index: {_fv(ldi.get('score'), '.0f')}/100</b>  |  "
        f"U-3: {_fv(ugap.get('u3'), '.1f', '%')}  |  "
        f"U-6: {_fv(ugap.get('u6'), '.1f', '%')}  |  "
        f"U6–U3 Gap: {_fv(ugap.get('gap'), '.1f', 'pp')}  |  "
        f"AHE YoY: {_fv(wages.get('ahe_yoy'), '.1f', '%')}  |  "
        f"Real Wage: {_fv(wages.get('real_wage'), '+.1f', '%')}  |  "
        f"ULC YoY: {_fv(wages.get('ulc_yoy'), '.1f', '%')}  |  "
        f"Quits: {_fv(jolts.get('quits_rate'), '.2f', '%')}  |  "
        f"Layoffs: {_fv(jolts.get('layoffs_rate'), '.2f', '%')}"
    ))
    story.append(Spacer(1, 0.08 * inch))

    # ── Unemployment & Payrolls ────────────────────────────────────────
    story.append(_two_charts(
        u3_u6_chart(dl, lookback_years=15),
        bar_change_chart(dl, "PAYEMS", lookback_years=5,
                         title="Nonfarm Payrolls — Monthly Change (Thousands)"),
    ))

    # ── JOLTS & Claims ────────────────────────────────────────────────
    story.append(_subsection("JOLTS & Jobless Claims"))
    story.append(_prose(
        "Job openings lead payroll changes by 2–4 months. The quits rate falls when workers "
        "lose confidence in job mobility — a reliable 3–6 month leading indicator of rising "
        "unemployment. Continued claims measure re-hiring absorption."
    ))
    story.append(_two_charts(
        jolts_chart(dl, lookback_years=10),
        claims_dashboard_chart(dl, lookback_years=8),
        small=True,
    ))

    # ── Wages, Productivity & ULC ─────────────────────────────────────
    story.append(_subsection("Wages, Productivity & Unit Labor Costs"))
    story.append(_prose(
        "Unit Labor Costs (wages ÷ productivity) is the primary transmission mechanism from "
        "labor costs to goods and services prices. Sustained ULC above 2.5% YoY is "
        "inconsistent with the Fed's 2% inflation target. "
        f"ECI YoY: {_fv(wages.get('eci_yoy'), '.1f', '%')} | "
        f"Productivity YoY: {_fv(wages.get('prod_yoy'), '.1f', '%')}."
    ))
    story.append(_two_charts(
        wage_productivity_chart(dl, lookback_years=10),
        labor_deterioration_chart(lae, lookback_years=10),
        small=True,
    ))

    # ── Participation & Industrial Production ─────────────────────────
    story.append(_subsection("Employment Structure & Industrial Activity"))
    story.append(_two_charts(
        multi_line_chart(dl,
            [("EMRATIO", C["blue"]), ("CIVPART", C["teal"])],
            title="Employment-Population Ratio vs. Participation Rate",
            lookback_years=20),
        area_chart(dl, "INDPRO", yoy=True, lookback_years=20,
                   color=C["blue"], fill_color="rgba(43,108,176,0.10)",
                   threshold_green=1, threshold_red=-1,
                   title="Industrial Production — Year-over-Year %",
                   recession_shading=True),
    ))
    story.append(_two_charts(
        area_chart(dl, "PSAVERT", lookback_years=20,
                   color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                   threshold_green=7, threshold_red=4,
                   title="Personal Saving Rate %"),
        area_chart(dl, "USSLIND", lookback_years=10,
                   color=C["red"], fill_color="rgba(155,44,44,0.09)",
                   threshold_green=0, threshold_red=-5,
                   title="Conference Board Leading Economic Index (MoM %)"),
    ))
    story.append(_prose(
        "Industrial production YoY contraction has coincided with every U.S. recession since 1960. "
        "The Conference Board LEI aggregates 10 forward-looking components; three consecutive "
        "monthly declines have preceded every NBER recession with a 6–12 month lead."
    ))


def _build_markets(story, dl):
    story += _section_rule("Markets & Rates", anchor_id="sec_markets")
    print("  Rendering markets/rates charts...")
    story.append(_full_chart(area_chart(dl, "RECPROUSM156N", lookback_years=30,
        threshold_green=15, threshold_red=40,
        color=C["red"], fill_color="rgba(155,44,44,0.09)",
        title="NY Fed 12-Month Recession Probability %",
        recession_shading=True)))
    story.append(_prose(
        "The NY Fed recession probability is derived from the yield curve (3-month vs 10-year spread). "
        "Readings above 30-40% have historically signaled recession within 12 months."
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_two_charts(
        multi_line_chart(dl,
            [("DGS10", C["blue"]), ("GS10", C["teal"])],
            title="10-Year Treasury Yield — Daily vs. Monthly Average %",
            lookback_years=20),
        area_chart(dl, "DTWEXBGS", lookback_years=20, yoy=True,
            color=C["slate"], fill_color="rgba(74,85,104,0.08)",
            title="Broad Dollar Index — YoY % Change",
            events=_EVENTS_SHORT),
    ))
    story.append(_prose(
        "A sharply rising dollar (>8% YoY) acts as a global tightening mechanism: it raises "
        "debt service costs for dollar-denominated EM borrowers, pressures commodity prices, "
        "and reduces earnings for US multinationals. Dollar strength in a late-cycle environment "
        "amplifies the restrictive effect of Fed policy beyond US borders."
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_full_chart(area_chart(dl, "SP500_CAPE", lookback_years=None,
        threshold_green=20, threshold_red=30,
        color=C["amber"], fill_color="rgba(183,121,31,0.09)",
        title="S&P 500 CAPE (Shiller P/E) — Full History",
        events=_EVENTS_LONG)))
    story.append(_prose(
        "The Cyclically Adjusted P/E ratio smooths earnings over a rolling 10-year real "
        "average, removing business cycle distortions. The historical mean is approximately 17. "
        "Readings above 30 — sustained through 1929, 2000, and 2021 — have historically "
        "preceded decade-long below-average real returns. CAPE does not predict timing."
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_two_charts(
        area_chart(dl, "SP500_PE", lookback_years=40,
                   threshold_green=20, threshold_red=27,
                   color=C["slate"], fill_color="rgba(74,85,104,0.08)",
                   title="S&P 500 Trailing P/E (40 Years)"),
        area_chart(dl, "SUBLPDRCSN", lookback_years=25,
                   threshold_green=10, threshold_red=40,
                   color=C["red"], fill_color="rgba(155,44,44,0.09)",
                   title="CRE Lending Standards — Net % Tightening"),
    ))
    story.append(_full_chart(
        dual_axis_chart(dl,
            left_series=("SP500_CAPE", C["amber"]),
            right_series=("VIXCLS", C["slate"]),
            title="S&P 500 CAPE (Shiller P/E, left) vs VIX Volatility Index (right)",
            lookback_years=30, recession_shading=True),
    ))
    story.append(_prose(
        "CAPE measures structural valuation risk (where valuations are relative to long-run "
        "earnings); VIX measures near-term sentiment and implied volatility. High CAPE is "
        "a long-run forward return predictor, not a short-term timing signal — elevated "
        "readings have coexisted with extended bull markets. Low VIX reflects current market "
        "calm and can persist. The combination most associated with historical drawdowns is "
        "high CAPE combined with a VIX spike above 30 (2001, 2008, 2022) — both elevated "
        "valuations and materialising stress, not valuations alone."
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_full_chart(yield_spread_chart(dl, lookback_years=30, recession_shading=True)))
    story.append(_prose(
        "The yield curve spread (10Y minus 2Y) is arguably the most reliable leading recession indicator: "
        "inversion has preceded every US recession since 1970, typically by 12-18 months. "
        "Red zones mark periods of inversion (spread < 0). Note that uninverting after a prolonged "
        "inversion can coincide with — rather than precede — economic contraction."
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_two_charts(
        line_chart(dl, "GS10", lookback_years=40, color=C["blue"],
                   title="10-Year Treasury Yield — Long History %"),
        line_chart(dl, "DGS10", lookback_years=10, color=C["blue"],
                   title="10-Year Treasury Yield — 10 Years"),
    ))
    story.append(_prose(
        "The secular decline in long-term yields from ~16% in 1981 to near 0% in 2020 "
        "supported rising asset valuations and reduced debt service costs. "
        "The reversal since 2022 has structural implications for real estate valuations, "
        "corporate refinancing, and government debt service."
    ))


def _build_fiscal(story, dl):
    story += _section_rule("Fiscal", anchor_id="sec_fiscal")
    print("  Rendering fiscal charts...")
    story.append(_full_chart(area_chart(dl, "GFDEGDQ188S", lookback_years=40,
        threshold_green=80, threshold_red=120,
        color=C["amber"], fill_color="rgba(183,121,31,0.09)",
        title="Federal Debt as % of GDP")))
    story.append(_prose(
        "Federal debt as a share of GDP rose sharply during World War II (peaking ~106%), "
        "declined through the postwar expansion, then began climbing again from the 1980s. "
        "The 2008 financial crisis and the 2020 pandemic triggered the largest peacetime "
        "increases on record. At elevated debt/GDP ratios, interest rate increases have a "
        "compounding effect: as debt rolls over at higher rates, the share of the budget "
        "consumed by interest payments expands, crowding out discretionary fiscal space."
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_two_charts(
        area_chart(dl, "GFDEBTN", lookback_years=20,
            color=C["amber"], fill_color="rgba(183,121,31,0.09)",
            title="Federal Debt Outstanding ($B, Monthly)"),
        area_chart(dl, "GFDEBTN", lookback_years=5,
            color=C["red"], fill_color="rgba(155,44,44,0.09)",
            title="Federal Debt Outstanding ($B) — Last 5 Years"),
    ))
    story.append(_prose(
        "Monthly total public debt outstanding provides a more timely read than the quarterly "
        "Debt/GDP ratio, which lags by a full quarter. The 5-year view highlights the pace of "
        "recent accumulation relative to the long-run trend."
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_full_chart(multi_line_chart(dl,
        [("M2SL", C["blue"]), ("GFDEGDQ188S", C["amber"])],
        title="M2 Money Supply (B$) & Federal Debt/GDP — Long-Run Trends",
        lookback_years=40,
    )))
    story.append(_prose(
        "Note: M2 is shown in billions of dollars; Debt/GDP is a percentage. "
        "The two series share a chart to illustrate the co-movement of monetary and "
        "fiscal expansion over the past four decades."
    ))


# ── System Resilience Helpers ─────────────────────────────────────────────────

def _absorption_capacity_table(re: RiskEngine) -> Table:
    """
    5-row summary table: System Absorption Capacity.
    Each row shows category, current status, key metric, and current reading.
    """
    capacity = re.system_absorption_capacity()

    header_style = _ps("ach", fontName="Helvetica-Bold", fontSize=8,
                        textColor=HDR_TEXT)
    hdr_row = [
        Paragraph("CATEGORY", header_style),
        Paragraph("STATUS", header_style),
        Paragraph("KEY METRIC", header_style),
        Paragraph("CURRENT READING", header_style),
    ]
    col_w = [CONTENT_W * 0.28, CONTENT_W * 0.18, CONTENT_W * 0.28, CONTENT_W * 0.26]

    rows = [hdr_row]
    style_cmds = [
        ("BACKGROUND",   (0, 0), (-1, 0), HDR_BG),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7fafc"), colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.5, RULE_CLR),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]

    for i, (cat_name, info) in enumerate(capacity.items(), start=1):
        risk = info["score"]
        bg, txt, _ = _risk_clrs(risk)
        rs = RISK_STYLE.get(risk, RISK_STYLE["neutral"])

        status_cell = Table(
            [[Paragraph(rs["label"], _ps("acl", fontName="Helvetica-Bold",
                                         fontSize=8, textColor=txt))]],
            colWidths=[col_w[1] - 0.18 * inch],
        )
        status_cell.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))

        rows.append([
            Paragraph(cat_name, _ps("acn", fontName="Helvetica-Bold", fontSize=9,
                                    textColor=BODY_CLR)),
            status_cell,
            Paragraph(info["key_metric"], _ps("acm", fontSize=8, textColor=MUTED_CLR)),
            Paragraph(f"<b>{info['value']}</b>",
                      _ps("acv", fontSize=8, textColor=BODY_CLR)),
        ])

    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle(style_cmds))
    return t


def _refinancing_risk_box(story, re: RiskEngine):
    """Refinancing & Liquidity Risk warning section with triggered conditions."""
    risk_level, triggers = re.refinancing_liquidity_risk()
    bg, txt, border = _risk_clrs(risk_level)
    rs = RISK_STYLE.get(risk_level, RISK_STYLE["neutral"])

    story.append(_subsection("Refinancing & Liquidity Risk Assessment"))

    if not triggers:
        summary = "No multi-indicator refinancing or liquidity stress conditions currently triggered."
    else:
        summary = (
            f"{len(triggers)} stress condition(s) triggered. "
            "Persistent combinations — not single spikes — are the primary concern."
        )

    header_inner = Table(
        [[Paragraph(
            f"<b>REFINANCING & LIQUIDITY RISK — {rs['label'].upper()}</b>  ·  {summary}",
            _ps("rrh", fontSize=9, textColor=txt),
        )]],
        colWidths=[CONTENT_W - 0.22 * inch],
    )
    header_inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    wrapper = Table([[header_inner]], colWidths=[CONTENT_W])
    wrapper.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("LINEAFTER",    (0, 0), (0, -1), 2, border),
    ]))
    story.append(wrapper)

    if triggers:
        story.append(Spacer(1, 0.06 * inch))
        for condition in triggers:
            story.append(_prose(f"  •  {condition}"))

    story.append(Spacer(1, 0.06 * inch))
    story.append(_prose(
        "<i>Note: This assessment emphasizes persistence and combinations. A single "
        "elevated indicator is less significant than two or more deteriorating in tandem. "
        "Temporary spikes resolve quickly; structural regime shifts do not.</i>"
    ))


# ── System Resilience Section Builders ───────────────────────────────────────

def _build_system_resilience(story, re: RiskEngine, dl):
    story += _section_rule("System Resilience & Policy Dependency", anchor_id="sec_resilience")
    story.append(_prose(
        "The modern financial system is increasingly dependent on liquidity provision, "
        "refinancing capacity, and policy intervention. Structural fragility often emerges "
        "not from isolated economic weakness, but from deterioration in the system's ability "
        "to absorb shocks without extraordinary stabilization measures."
    ))
    story.append(Spacer(1, 0.1 * inch))

    # Dimension cards (same style as Crisis Watch)
    resilience_dims = re.system_resilience_dimensions()
    story.append(_subsection("System Resilience Dimensions"))
    dim_items = list(resilience_dims.items())
    for row_start in range(0, len(dim_items), 3):
        chunk = dim_items[row_start:row_start + 3]
        cells = [_crisis_dim_card(name, dim) for name, dim in chunk]
        while len(cells) < 3:
            cells.append(Spacer(THIRD_W, 0.1))
        t = Table([cells], colWidths=[THIRD_W] * 3)
        t.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING",   (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ]))
        story.append(t)

    story.append(Spacer(1, 0.1 * inch))
    story.append(_subsection("System Absorption Capacity Summary"))
    story.append(_absorption_capacity_table(re))
    story.append(Spacer(1, 0.12 * inch))

    _refinancing_risk_box(story, re)
    story.append(Spacer(1, 0.12 * inch))

    # Structural framework explainer
    story.append(_subsection("Structural Stress vs System Absorption Capacity"))
    story.append(_half_table(
        [_prose(
            "<b>Structural Fragility</b> — conditions that accumulate slowly and reduce "
            "the system's margin of safety: elevated debt loads, persistent inflation, "
            "labor market deterioration, demographic stagnation, and tightening credit "
            "standards. These develop over years and are often masked by asset appreciation "
            "or credit availability."
        )],
        [_prose(
            "<b>Absorption Capacity</b> — the system's active ability to withstand shocks "
            "without requiring progressively larger interventions: functioning credit markets, "
            "liquidity provision, policy flexibility, anchored inflation expectations, and "
            "stable funding markets. When this deteriorates, structural imbalances become "
            "acute crises rather than manageable headwinds."
        )],
    ))
    story.append(Spacer(1, 0.06 * inch))
    story.append(_prose(
        "Modern markets can tolerate elevated structural imbalances for extended periods if "
        "refinancing channels remain functional and policy credibility remains intact. Acute "
        "crises emerge when structural fragility converges with deterioration in liquidity, "
        "credit, or stabilization capacity — not from structural imbalances alone."
    ))

    story.append(PageBreak())

    _build_liquidity_funding(story, dl)
    story.append(PageBreak())

    _build_credit_markets(story, dl)
    story.append(PageBreak())

    _build_policy_constraints(story, re, dl)


def _build_liquidity_funding(story, dl):
    story += _section_rule("Liquidity & Funding Stress", anchor_id="sec_liquidity", level=1)
    story.append(_prose(
        "This section monitors the financial plumbing that supports market functioning: "
        "short-term funding markets, interbank credit conditions, and the availability of "
        "safe-haven liquidity. Stress here typically appears before it surfaces in equity "
        "prices, unemployment, or GDP data — making these indicators leading signals of "
        "systemic deterioration."
    ))
    story.append(Spacer(1, 0.08 * inch))

    print("  Rendering liquidity/funding charts...")
    story.append(_subsection("Financial Stress Indices"))
    story.append(_prose(
        "The St. Louis Fed FSI and Chicago Fed NFCI both aggregate dozens of financial "
        "market variables into a single stress reading. Both are indexed around 0 (average "
        "conditions). Positive readings indicate tighter-than-normal conditions; readings "
        "above 1.0-1.5 have historically coincided with significant market dislocations."
    ))
    story.append(_two_charts(
        percentile_chart(dl, "STLFSI4",
            title="St. Louis Fed Financial Stress Index",
            color=C["red"], fill_color="rgba(155,44,44,0.09)",
            lookback_years=20, higher_is_bad=True, recession_shading=True),
        percentile_chart(dl, "NFCI",
            title="Chicago Fed National Financial Conditions Index",
            color=C["slate"], fill_color="rgba(74,85,104,0.08)",
            lookback_years=20, higher_is_bad=True, recession_shading=True),
    ))
    story.append(_prose(
        "Shaded vertical bands mark NBER-defined recession periods. "
        "Colored horizontal bands show the 25th/75th historical percentile ranges — "
        "providing context for whether current readings are historically extreme."
    ))
    story.append(Spacer(1, 0.08 * inch))

    story.append(_subsection("Interbank & Repo Funding Markets"))
    story.append(_prose(
        "The 3-Month CP-Treasury spread — AA Financial Commercial Paper rate minus the "
        "3-Month Treasury — is the modern successor to the discontinued TED spread. It "
        "measures the unsecured short-term credit premium banks pay above the risk-free "
        "rate. Readings above 100bps have historically signaled elevated interbank credit "
        "stress. SOFR tracks overnight secured repo conditions; divergence from Fed Funds "
        "signals repo market pressure."
    ))
    story.append(_two_charts(
        derived_spread_chart(dl, "DCPF3M", "DGS3MO",
            title="3-Month CP minus Treasury Spread % (Modern TED Equivalent)",
            color=C["amber"], fill_color="rgba(183,121,31,0.09)",
            threshold_green=0.5, threshold_red=1.0,
            lookback_years=25, recession_shading=True),
        area_chart(dl, "SOFR", lookback_years=8,
            title="Secured Overnight Financing Rate (SOFR) %",
            color=C["teal"], fill_color="rgba(44,122,123,0.09)"),
    ))
    story.append(_two_charts(
        area_chart(dl, "RRPONTSYD", lookback_years=15,
            threshold_green=500, threshold_red=100,
            color=C["blue"], fill_color="rgba(43,108,176,0.10)",
            title="Fed Overnight Reverse Repo Facility ($B)"),
        percentile_chart(dl, "NFCICREDIT",
            title="NFCI Credit Subindex (Chicago Fed)",
            color=C["red"], fill_color="rgba(155,44,44,0.09)",
            lookback_years=20, higher_is_bad=True, recession_shading=True),
    ))
    story.append(_prose(
        "Reverse repo usage peaked near $2.5 trillion in 2023 as excess reserves flooded "
        "the system. Rapid drawdown toward zero — if accompanied by rising funding rates — "
        "would signal reserves leaving the system faster than the Fed can replace them. "
        "The NFCI credit subindex isolates the credit channel from broader financial conditions, "
        "providing an early warning of tightening access to credit specifically."
    ))


def _build_credit_markets(story, dl):
    story += _section_rule("Credit Market Functionality", anchor_id="sec_credit", level=1)
    story.append(_prose(
        "Most systemic crises emerge when refinancing capacity deteriorates and credit spreads "
        "widen persistently, impairing liquidity and rollover financing. Credit market "
        "functionality is a more reliable leading indicator of economic stress than equity "
        "market volatility — spreads reflect actual lending conditions, not just sentiment."
    ))
    story.append(Spacer(1, 0.08 * inch))

    print("  Rendering credit market charts...")
    story.append(_subsection("Bank Lending Standards (SLOOS)"))
    story.append(_prose(
        "The Senior Loan Officer Opinion Survey asks banks whether they tightened or loosened "
        "credit standards on C&I (commercial and industrial) loans in the prior quarter. "
        "Net tightening above 40% has historically preceded credit contractions by 6-12 months — "
        "making it one of the most reliable leading indicators of future loan growth collapse. "
        "It captures supply-side credit rationing before it shows up in delinquencies."
    ))
    story.append(_two_charts(
        area_chart(dl, "DRTSCILM", lookback_years=30,
            threshold_green=10, threshold_red=40,
            color=C["amber"], fill_color="rgba(183,121,31,0.09)",
            title="Net % Banks Tightening C&I Loan Standards %",
            recession_shading=True, events=_EVENTS_MED),
        percentile_chart(dl, "DRTSCILM",
            title="Lending Standards — Historical Percentile",
            color=C["amber"], fill_color="rgba(183,121,31,0.09)",
            lookback_years=None, higher_is_bad=True, recession_shading=True),
    ))
    story.append(_prose(
        "Negative readings (net loosening) indicate accommodative lending conditions. "
        "The 2008 GFC and COVID episodes saw rapid swings from loosening to extreme tightening. "
        "The current trajectory matters as much as the level — rapidly tightening standards "
        "signal banks see rising credit risk ahead of the official data."
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_subsection("High Yield & Investment Grade Spreads"))
    story.append(_prose(
        "Credit spreads represent the additional yield investors demand over Treasuries to hold "
        "corporate bonds. High yield (HY) spreads are more sensitive to default risk and liquidity; "
        "investment grade (IG) spreads reflect credit conditions for larger, more creditworthy "
        "issuers. When both widen simultaneously, credit tightening is broad-based."
    ))
    story.append(_full_chart(
        percentile_chart(dl, "BAMLH0A0HYM2",
            title="High Yield OAS (ICE BofA) % — Recession-Period Context",
            color=C["red"], fill_color="rgba(155,44,44,0.09)",
            lookback_years=25, higher_is_bad=True, recession_shading=True,
            events=_EVENTS_MED),
    ))
    story.append(_two_charts(
        percentile_chart(dl, "BAMLC0A0CM",
            title="Investment Grade OAS (ICE BofA) %",
            color=C["amber"], fill_color="rgba(183,121,31,0.09)",
            lookback_years=20, higher_is_bad=True, recession_shading=True),
        dual_axis_chart(dl,
            left_series=("BAMLH0A0HYM2", C["red"]),
            right_series=("VIXCLS", C["amber"]),
            title="HY Spread % (left) vs VIX (right)",
            lookback_years=20, recession_shading=True),
    ))
    story.append(_prose(
        "HY spreads and VIX are often correlated in stress episodes but diverge in regime "
        "transitions. Persistent spread widening without a corresponding VIX spike suggests "
        "structural credit deterioration rather than temporary sentiment-driven volatility — "
        "the more concerning scenario for refinancing capacity."
    ))
    story.append(Spacer(1, 0.08 * inch))

    story.append(_subsection("CRE Lending Standards & Consumer Delinquencies"))
    story.append(_prose(
        "CRE lending standards (SLOOS) capture forward-looking bank risk appetite for "
        "commercial real estate — a leading indicator that tightens before delinquencies "
        "rise. Credit card delinquencies provide a near-real-time read on consumer balance "
        "sheet stress; both are reported quarterly with approximately 6 weeks of lag."
    ))
    story.append(_two_charts(
        area_chart(dl, "DRCCLACBS", lookback_years=20,
            threshold_green=3.0, threshold_red=5.0,
            color=C["amber"], fill_color="rgba(183,121,31,0.09)",
            title="Credit Card Loan Delinquency Rate % (20 Years)"),
        percentile_chart(dl, "DRCCLACBS",
            title="CC Delinquency — Historical Percentile Context",
            color=C["amber"], fill_color="rgba(183,121,31,0.09)",
            lookback_years=None, higher_is_bad=True, recession_shading=True),
    ))
    story.append(_prose(
        "Credit card delinquency is a broad consumer credit health signal. Post-pandemic "
        "normalization has pushed this rate above pre-2020 levels — a sign that the lowest-income "
        "households are increasingly stretched, even before any broader labor market deterioration."
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_two_charts(
        area_chart(dl, "SUBLPDRCSN", lookback_years=25,
            threshold_green=10, threshold_red=40,
            color=C["red"], fill_color="rgba(155,44,44,0.09)",
            title="CRE Lending Standards — Net % Tightening (25 Years)"),
        percentile_chart(dl, "SUBLPDRCSN",
            title="CRE Lending Standards — Historical Percentile Context",
            color=C["red"], fill_color="rgba(155,44,44,0.09)",
            lookback_years=None, higher_is_bad=True, recession_shading=True),
    ))
    story.append(_prose(
        "The SLOOS CRE lending standards series captures bankers' forward-looking assessment "
        "of commercial real estate credit risk — a leading indicator that typically tightens "
        "6-12 months before delinquency rates rise. Net tightening above 40% has preceded "
        "CRE credit contractions in every major stress episode since the 1990 S&L crisis."
    ))
    story.append(Spacer(1, 0.08 * inch))

    # ── Banking Stress ────────────────────────────────────────────────
    story.append(_subsection("Banking & Credit Quality"))
    story.append(_prose(
        "Loan delinquency rates across all segments provide a lagging but highly reliable "
        "signal of credit cycle stress. All-loan delinquency (blue) is the broadest gauge; "
        "CRE (red) is concentrated at regional/community banks; residential mortgages (amber) "
        "reflect consumer housing balance sheets. Deposit flows show system-wide funding stability."
    ))
    print("  Rendering banking stress charts...")
    story.append(_two_charts(
        delinquency_chart(dl, lookback_years=20),
        bank_deposits_chart(dl, lookback_years=10),
        small=True,
    ))
    story.append(_two_charts(
        fci_composite_chart(dl, lookback_years=15),
        hy_spread_fci_chart(dl, lookback_years=10),
        small=True,
    ))
    story.append(_prose(
        "The composite financial conditions comparison (NFCI + STLFSI4) shows when overall "
        "financial system stress is tightening or loosening. Credit spread overlay with lending "
        "standards shows the dual tightening channel — market pricing and bank underwriting "
        "simultaneously contracting."
    ))


def _build_policy_constraints(story, re: RiskEngine, dl):
    story += _section_rule("Policy Constraints & Flexibility", anchor_id="sec_policy", level=1)
    story.append(_prose(
        "The critical issue is not absolute debt levels, but whether policymakers retain "
        "the flexibility to stabilize markets without destabilizing inflation, funding markets, "
        "or sovereign confidence. This section tracks the three primary constraints on "
        "policy maneuverability: fiscal space (debt service burden), monetary space "
        "(inflation expectations), and balance sheet capacity (Fed asset size)."
    ))
    story.append(Spacer(1, 0.08 * inch))

    print("  Rendering policy constraints charts...")
    story.append(_subsection("Fiscal Maneuverability: Debt Service Burden"))
    story.append(_prose(
        "Federal interest payments as a share of current receipts measures how much of "
        "government revenue is consumed by debt service. Unlike debt/GDP (which depends on "
        "GDP growth assumptions), this ratio directly reflects the cash flow constraint on "
        "fiscal policy. Historically below 15%; the post-2022 rate normalization has driven "
        "this ratio materially higher as low-rate debt rolls over at current market rates."
    ))
    story.append(_two_charts(
        derived_ratio_chart(dl,
            numerator_id="A091RC1Q027SBEA",
            denominator_id="W006RC1Q027SBEA",
            title="Federal Interest Payments / Current Receipts %",
            color=C["amber"], fill_color="rgba(183,121,31,0.09)",
            lookback_years=40,
            threshold_green=12, threshold_red=20,
            recession_shading=True,
            overlay_id="GS10",
            overlay_color=C["blue"],
            overlay_label="10-Yr Treasury Yield %"),
        walcl_pct_gdp_chart(dl, lookback_years=25),
    ))
    story.append(_prose(
        "Left: interest/receipts ratio (amber) with 10-year Treasury yield overlaid (dashed blue, right axis). "
        "The yield overlay shows how the post-2022 rate normalization directly drove the "
        "step-change in debt service cost as low-rate debt rolled over at current market rates. "
        "Green band: below 12%; yellow: 12-20%; red: above 20%. "
        "Right: Fed balance sheet as % of GDP — scale of prior QE and trajectory of QT."
    ))
    story.append(Spacer(1, 0.08 * inch))

    story.append(_subsection("Monetary Constraints: Real Policy Rate & Inflation Expectations"))
    story.append(_prose(
        "The real Federal Funds Rate (nominal rate minus core CPI YoY) determines whether "
        "monetary policy is stimulative or restrictive in inflation-adjusted terms. Deeply "
        "negative real rates incentivize borrowing and asset speculation; very high positive "
        "real rates risk demand destruction and debt deflation. Inflation expectations are "
        "the forward-looking constraint: if markets expect inflation to remain elevated, "
        "the Fed cannot reduce rates without undermining its credibility."
    ))
    story.append(_two_charts(
        real_rate_chart(dl, lookback_years=30),
        multi_line_chart(dl,
            [("FEDFUNDS", C["blue"]), ("CPILFESL", C["teal"])],
            title="Federal Funds Rate vs Core CPI YoY % (20 Years)",
            lookback_years=20, yoy=False),
    ))
    story.append(_two_charts(
        area_chart(dl, "T5YIE", lookback_years=20,
            threshold_green=2.5, threshold_red=3.2,
            color=C["teal"], fill_color="rgba(44,122,123,0.09)",
            title="5-Year Breakeven Inflation Expectations %"),
        area_chart(dl, "T10YIE", lookback_years=20,
            threshold_green=2.5, threshold_red=3.0,
            color=C["slate"], fill_color="rgba(74,85,104,0.08)",
            title="10-Year Breakeven Inflation Expectations %"),
    ))
    story.append(_prose(
        "Breakeven rates derived from TIPS. The 2% Fed target implies a sustainable "
        "range of approximately 2-2.5% for long-run breakevens. Persistent readings above "
        "3% would indicate inflation expectations becoming unanchored — at which point the "
        "Fed would be unable to cut rates in response to economic weakness without risking "
        "a credibility crisis. This is the binding constraint on monetary policy flexibility."
    ))


def _build_toc(story):
    """Clickable Table of Contents — hyperlinks to section anchors, no page numbers."""
    story += _section_rule("Table of Contents")
    story.append(_prose(
        "Click any section title to jump directly to that page. "
        "KPI cards in the Summary link back to their detailed section."
    ))
    story.append(Spacer(1, 0.08 * inch))

    TOC_ENTRIES = [
        (False, "Executive Summary",                      "sec_executive"),
        (False, "Summary — All Indicators",               "sec_summary"),
        (False, "Indicator Glossary",                     "sec_glossary"),
        (False, "Crisis Watch",                           "sec_crisis"),
        (False, "System Resilience & Policy Dependency",  "sec_resilience"),
        (True,  "Liquidity & Funding Stress",             "sec_liquidity"),
        (True,  "Credit Market Functionality",            "sec_credit"),
        (True,  "Policy Constraints & Flexibility",       "sec_policy"),
        (False, "Inflation & Money Supply",               "sec_inflation"),
        (False, "Labor Market",                           "sec_labor"),
        (False, "Markets & Rates",                        "sec_markets"),
        (False, "Housing Market",                         "sec_housing"),
        (False, "Fiscal",                                 "sec_fiscal"),
        (False, "Leading Indicators & Business Cycle Index", "sec_leading"),
        (False, "Recession Probability Model",            "sec_recession_prob"),
        (False, "Risk Taxonomy Scorecard",                "sec_risk_scorecard"),
        (False, "Global Macro & FX",                     "sec_global_macro"),
        (False, "Macro Regime Classification",           "sec_regime"),
        (False, "Fiscal Analytics & Sustainability",     "sec_fiscal_analytics"),
    ]

    rows = []
    for indent, title, anchor in TOC_ENTRIES:
        fs = 8 if indent else 9
        left = 18 if indent else 0
        rows.append([
            Paragraph(
                f'<link href="#{anchor}" color="#2b6cb0">{title}</link>',
                _ps("toc", fontSize=fs, textColor=BODY_CLR, leftIndent=left, leading=15),
            ),
            Paragraph(
                f'<link href="#{anchor}" color="#2b6cb0">&#x25B8; Jump</link>',
                _ps("tocjump", fontSize=8, textColor=colors.HexColor("#2b6cb0"), alignment=2),
            ),
        ])

    toc_table = Table(rows, colWidths=[CONTENT_W * 0.85, CONTENT_W * 0.15])
    toc_table.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LINEBELOW",    (0, 0), (-1, -1), 0.3, RULE_CLR),
    ]))
    story.append(toc_table)


def _narrative_item(text: str, risk: str) -> Table:
    """Single executive narrative bullet with a full colored background and left accent strip."""
    bg, _, border = _risk_clrs(risk)
    t = Table(
        [[Spacer(5, 1), Paragraph(text, ST_BODY)]],
        colWidths=[5, CONTENT_W - 5],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), bg),
        ("BACKGROUND",   (0, 0), (0, -1), border),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (1, 0), (1, -1), 10),
        ("LEFTPADDING",  (1, 0), (1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


_HIST_COMPARE = [
    # (series_id, label, yoy_basis)
    ("CPILFESL",        "Core CPI",                 True),
    ("UNRATE",          "Unemployment Rate",         False),
    ("DGS10",           "10Y Treasury Yield",        False),
    ("BAMLH0A0HYM2",    "HY Credit Spread",          False),
    ("FEDFUNDS",        "Fed Funds Rate",            False),
    ("SP500_CAPE",      "S&P 500 CAPE",              False),
    ("GFDEGDQ188S",     "Federal Debt / GDP",        False),
    ("VIXCLS",          "VIX",                       False),
    ("SUBLPDRCSN",      "CRE Lending Stds (net %)",  False),
    ("T10YIE",          "10Y Inflation Expectation", False),
]

def _get_hist_val(dl, series_id: str, target_date: str | None, yoy: bool) -> str:
    df = dl.load(series_id)
    if df is None or df.empty:
        return "—"
    col = df.columns[0]
    if target_date is None:
        mask = df.index <= df.index[-1]
        rows = df
    else:
        target = pd.Timestamp(target_date)
        rows = df[df.index <= target]
    if rows.empty:
        return "—"
    val = float(rows.iloc[-1][col])

    if yoy:
        prior_date = (rows.index[-1] - pd.DateOffset(years=1))
        prior_rows = df[df.index <= prior_date]
        if prior_rows.empty:
            return "—"
        prior_val = float(prior_rows.iloc[-1][col])
        if prior_val == 0:
            return "—"
        val = (val - prior_val) / abs(prior_val) * 100
        return f"{val:.1f}%"

    meta = REGISTRY.get(series_id, {})
    units = meta.get("units", "")
    if "Percent" in units or units.endswith("%"):
        return f"{val:.1f}%"
    if "Billions" in units:
        return f"${val:,.0f}B"
    return f"{val:,.1f}"


def _hist_compare_table(dl, re: RiskEngine) -> Table:
    """Historical comparison table: current / 1Y ago / pre-COVID."""
    one_yr_ago = (datetime.now() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    col_labels = ["Indicator", "Current", "1 Year Ago", "Pre-COVID (Dec 2019)"]
    hdr_style = _ps("hch", fontName="Helvetica-Bold", fontSize=8, textColor=HDR_TEXT)
    hdr_row = [Paragraph(h, hdr_style) for h in col_labels]

    col_w = [CONTENT_W * 0.38, CONTENT_W * 0.20, CONTENT_W * 0.20, CONTENT_W * 0.22]
    rows = [hdr_row]
    style_cmds = [
        ("BACKGROUND",   (0, 0), (-1, 0), HDR_BG),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7fafc"), colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.4, RULE_CLR),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]

    for sid, label, yoy in _HIST_COMPARE:
        if sid not in dl.available:
            continue
        risk, _, _ = re.score(sid)
        bg, txt, _ = _risk_clrs(risk)
        cur  = _get_hist_val(dl, sid, None,         yoy)
        y1   = _get_hist_val(dl, sid, one_yr_ago,   yoy)
        pre  = _get_hist_val(dl, sid, "2019-12-01", yoy)
        rows.append([
            Paragraph(label, _ps("hcl", fontSize=8, fontName="Helvetica-Bold", textColor=BODY_CLR)),
            Paragraph(f"<b>{cur}</b>",  _ps("hcc", fontSize=8, textColor=txt, fontName="Helvetica-Bold")),
            Paragraph(y1,  _ps("hc1", fontSize=8, textColor=MUTED_CLR)),
            Paragraph(pre, _ps("hcp", fontSize=8, textColor=MUTED_CLR)),
        ])
        # Shade current value cell with risk color
        row_idx = len(rows) - 1
        style_cmds.append(("BACKGROUND", (1, row_idx), (1, row_idx), bg))

    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle(style_cmds))
    return t


def _build_executive_summary(story, re: RiskEngine, dl=None):
    """One-page plain-English synthesis of current conditions."""
    story += _section_rule("Executive Summary", anchor_id="sec_executive")
    story.append(_prose(
        "A synthesis of current economic conditions across six dimensions. "
        "Individual readings matter less than combinations and trends. "
        "Color indicates risk status: green = normal, yellow = elevated, red = stressed. "
        "Use the bookmarks panel in your PDF viewer to navigate between sections."
    ))
    story.append(Spacer(1, 0.08 * inch))

    # Overall stress badge
    overall = re.overall_stress_level()
    overall_bg, overall_txt, overall_border = _risk_clrs(overall)
    rs = RISK_STYLE.get(overall, RISK_STYLE["neutral"])
    label_map = {"green": "NORMAL CONDITIONS", "yellow": "MODERATE STRESS", "red": "HIGH STRESS"}
    overall_label = label_map.get(overall, "UNKNOWN")

    badge = Table(
        [[Paragraph(
            f"OVERALL ASSESSMENT: <b>{overall_label}</b>",
            _ps("ob", fontName="Helvetica-Bold", fontSize=11, textColor=overall_txt),
        )]],
        colWidths=[CONTENT_W - 0.22 * inch],
    )
    badge.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), overall_bg),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("LINEAFTER",    (0, 0), (0, -1), 3, overall_border),
    ]))
    wrapper = Table([[badge]], colWidths=[CONTENT_W])
    wrapper.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",(0, 0), (-1, -1), 0),
    ]))
    story.append(wrapper)
    story.append(Spacer(1, 0.12 * inch))

    narrative = re.executive_narrative()
    for risk, text in narrative:
        story.append(_narrative_item(text, risk))
        story.append(Spacer(1, 0.04 * inch))

    if dl is not None:
        story.append(Spacer(1, 0.12 * inch))
        story.append(_subsection("Historical Comparison — Current vs. 1 Year Ago vs. Pre-COVID"))
        story.append(_prose(
            "Key indicators compared across three time points. "
            "Current cell shading reflects the current risk status (green / yellow / red). "
            "Core CPI shown as year-over-year percent change; all others as reported level values."
        ))
        story.append(Spacer(1, 0.06 * inch))
        story.append(_hist_compare_table(dl, re))

    story.append(Spacer(1, 0.1 * inch))
    story.append(_prose(
        "<i>This summary is generated from threshold-based rules applied to current data. "
        "It is descriptive, not predictive. Readings that appear benign in isolation may be "
        "significant in combination — see Crisis Watch and System Resilience sections for "
        "multi-indicator composite analysis.</i>"
    ))


def _build_housing(story, dl):
    story += _section_rule("Housing Market", anchor_id="sec_housing")
    story.append(_prose(
        "Housing is the most interest-rate-sensitive sector in the economy and a critical "
        "transmission mechanism for monetary policy. Mortgage rates directly control "
        "affordability; starts lead construction employment by 3-6 months; and home prices "
        "affect household net worth, consumer confidence, and collateral values for lending."
    ))
    story.append(Spacer(1, 0.08 * inch))

    print("  Rendering housing charts...")
    story.append(_subsection("Mortgage Rates & Housing Starts"))
    story.append(_two_charts(
        area_chart(dl, "MORTGAGE30US", lookback_years=20,
            threshold_green=5.0, threshold_red=7.0,
            color=C["red"], fill_color="rgba(155,44,44,0.09)",
            title="30-Year Fixed Mortgage Rate %",
            events=_EVENTS_SHORT),
        area_chart(dl, "HOUST", lookback_years=20,
            threshold_green=1300, threshold_red=900,
            color=C["blue"], fill_color="rgba(43,108,176,0.10)",
            title="Housing Starts — Total (Thousands, SAAR)",
            recession_shading=True),
    ))
    story.append(_prose(
        "The 30-year mortgage rate rose from sub-3% in 2021 to over 7% in 2023 — "
        "the fastest rate increase in the modern mortgage market era. Housing starts "
        "respond with a lag of 3-6 months to rate changes, as pipeline projects run off. "
        "The secular housing undersupply (estimated 3-5M units nationally) provides a "
        "structural floor that limits the depth of a housing correction compared to 2008."
    ))
    story.append(Spacer(1, 0.08 * inch))

    story.append(_subsection("Home Price Appreciation"))
    story.append(_two_charts(
        area_chart(dl, "CSUSHPINSA", lookback_years=20, yoy=True,
            threshold_green=8, threshold_red=15,
            color=C["amber"], fill_color="rgba(183,121,31,0.09)",
            title="Case-Shiller Home Price Index — YoY %",
            recession_shading=True),
        percentile_chart(dl, "CSUSHPINSA",
            title="Case-Shiller HPI — Historical YoY Percentile",
            color=C["amber"], fill_color="rgba(183,121,31,0.09)",
            lookback_years=None, higher_is_bad=True, recession_shading=False),
    ))
    story.append(_prose(
        "Home price appreciation above 8% YoY outpaces wage growth and compresses "
        "affordability faster than income can adjust. The 2020-2022 surge — driven by "
        "pandemic-era demand shifts and 3% mortgages — has proved partially sticky "
        "even as rate headwinds increased. Elevated home prices combined with high "
        "mortgage rates create an 'affordability lock': would-be sellers are reluctant "
        "to trade their 3% mortgage for 7%, suppressing transaction volume."
    ))
    story.append(_two_charts(
        multi_line_chart(dl,
            [("MORTGAGE30US", C["red"]), ("HOUST", C["blue"])],
            title="Mortgage Rate % vs. Housing Starts (Thousands) — 20 Years",
            lookback_years=20),
        area_chart(dl, "MORTGAGE30US", lookback_years=40,
            threshold_green=5.0, threshold_red=7.0,
            color=C["red"], fill_color="rgba(155,44,44,0.09)",
            title="30-Year Mortgage Rate — 40-Year History %",
            events=_EVENTS_LONG),
    ))
    story.append(_prose(
        "The 40-year history chart provides crucial context: the 1980s featured mortgage "
        "rates above 15%. The post-2008 secular decline to sub-3% was a structural tailwind "
        "for home prices and homeowner net worth. The reversal since 2022 is large in "
        "absolute terms but still well below historical peaks."
    ))


# ── PDF Builder ───────────────────────────────────────────────────────────────

def _build_leading_indicators(story, lie, re, dl):
    """Leading Indicators & Business Cycle Index section."""
    story += _section_rule("Leading Indicators & Business Cycle Index",
                           anchor_id="sec_leading")
    print("  Computing BCI and leading signals...")

    bci_val, phase = lie.current_bci()
    bci_str = f"{bci_val:.2f}" if bci_val is not None else "N/A"

    _phase_colors = {
        "Expansion":   "#276749",
        "Slowdown":    "#975a16",
        "Contraction": "#9b2c2c",
        "Recovery":    "#2b6cb0",
        "No Data":     "#718096",
    }
    phase_color = _phase_colors.get(phase, "#718096")

    story.append(_prose(
        "The Composite Business Cycle Index (BCI) aggregates eight leading macro indicators "
        "into a single standardized score. Each component is z-scored against its 20-year "
        "calibration window, inverted when lower values are healthier, then combined as a "
        "weighted average (weights renormalized for missing data) and smoothed with a "
        "3-month moving average. "
        f"<b>Current BCI: {bci_str} &mdash; Phase: "
        f'<font color="{phase_color}">{phase}</font></b>'
    ))
    story.append(Spacer(1, 0.06 * inch))

    # BCI time series (left) + component waterfall (right)
    print("  Rendering BCI charts...")
    story.append(_two_charts(
        bci_chart(lie, dl, lookback_years=15),
        bci_waterfall_chart(lie, dl),
        small=True,
    ))
    story.append(Spacer(1, 0.06 * inch))

    # Phase legend table
    phase_rows = [
        ["Expansion",    "#276749", "BCI > 0, momentum positive — above-trend growth accelerating"],
        ["Slowdown",     "#975a16", "BCI > 0, momentum negative — above-trend but decelerating"],
        ["Recovery",     "#2b6cb0", "BCI ≤ 0, momentum positive — below-trend but improving"],
        ["Contraction",  "#9b2c2c", "BCI ≤ 0, momentum negative — below-trend and worsening"],
    ]
    phase_tbl_data = [
        [Paragraph(r[0], _ps("pl", fontName="Helvetica-Bold", fontSize=8,
                              textColor=colors.HexColor(r[1]))),
         Paragraph(r[2], ST_SMALL)]
        for r in phase_rows
    ]
    phase_tbl = Table(phase_tbl_data, colWidths=[CONTENT_W * 0.18, CONTENT_W * 0.82])
    phase_tbl.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1),
         [colors.HexColor("#f7f8fc"), colors.white]),
    ]))
    story.append(phase_tbl)
    story.append(Spacer(1, 0.12 * inch))

    # ── Momentum Analytics table ───────────────────────────────────────────
    story.append(_subsection("Trend & Momentum Analytics"))
    story.append(_prose(
        "Direction is based on the 3-month linear trend: <b>Deteriorating</b> means the "
        "series is trending toward worse conditions; <b>Improving</b> means improving. "
        "Z vs 1Y measures how far the current reading departs from its 12-month average "
        "in standard-deviation units."
    ))

    mom_data = lie.all_momentum()
    _mom_hdrs = ["Series", "Current", "3M Trend", "6M Trend", "Accel.", "Z vs 1Y", "Direction"]
    mom_rows_tbl = [
        [Paragraph(h, _ps("mh", fontName="Helvetica-Bold", fontSize=8,
                           textColor=colors.white))
         for h in _mom_hdrs]
    ]
    _dir_colors = {"Improving": "#276749", "Deteriorating": "#9b2c2c"}
    for sid in BCI_COMPONENTS:
        m = mom_data.get(sid, {})
        if not m:
            continue
        meta = REGISTRY.get(sid, {})
        direction = m.get("direction", "—")
        dc = _dir_colors.get(direction, "#718096")
        mom_rows_tbl.append([
            Paragraph(meta.get("short_name", sid), ST_SMALL),
            Paragraph(f"{m.get('current', 0):.2f}",        ST_SMALL),
            Paragraph(f"{m.get('trend_3m', 0):+.3f}",      ST_SMALL),
            Paragraph(f"{m.get('trend_6m', 0):+.3f}",      ST_SMALL),
            Paragraph(f"{m.get('acceleration', 0):+.3f}",  ST_SMALL),
            Paragraph(f"{m.get('z_vs_1y', 0):+.2f}σ", ST_SMALL),
            Paragraph(f'<font color="{dc}"><b>{direction}</b></font>', ST_SMALL),
        ])

    if len(mom_rows_tbl) > 1:
        mom_col_ws = [CONTENT_W * w for w in (0.18, 0.10, 0.12, 0.12, 0.10, 0.12, 0.26)]
        mom_tbl = Table(mom_rows_tbl, colWidths=mom_col_ws)
        mom_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), HDR_BG),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f7f8fc")]),
            ("GRID",          (0, 0), (-1, -1), 0.3, RULE_CLR),
        ]))
        story.append(mom_tbl)
    story.append(Spacer(1, 0.10 * inch))

    # ── Momentum charts for four key series ───────────────────────────────
    story.append(_subsection("Indicator Trend Details"))
    story.append(_prose(
        "Top panel: indicator level (faint) with 3-month (solid blue) and 6-month "
        "(dashed teal) rolling averages. Bottom panel: 3-month rate of change — "
        "green bars indicate improving momentum, red bars indicate deterioration."
    ))
    print("  Rendering momentum charts...")
    story.append(_two_charts(
        momentum_chart(dl, "USSLIND",      lookback_years=8),
        momentum_chart(dl, "BAMLH0A0HYM2", lookback_years=8),
        small=True,
    ))
    story.append(_two_charts(
        momentum_chart(dl, "T10Y2Y",    lookback_years=8),
        momentum_chart(dl, "ICSA",      lookback_years=8),
        small=True,
    ))
    story.append(_two_charts(
        momentum_chart(dl, "PERMIT",    lookback_years=8),
        momentum_chart(dl, "TEMPHELPS", lookback_years=8),
        small=True,
    ))
    story.append(Spacer(1, 0.10 * inch))

    # ── Historical backtest signal charts ──────────────────────────────────
    story.append(_subsection("Historical Signal Overlay Charts"))
    story.append(_prose(
        "Each chart shows the indicator series with NBER recession shading (grey), "
        "the signal threshold (dashed amber line at the 75th or 25th percentile of "
        "the full history), and orange dots marking where the signal fired. "
        "Visual validation of hit rates and false-positive rates."
    ))
    print("  Rendering backtest signal charts...")
    story.append(_two_charts(
        backtest_signal_chart(lie, dl, "USSLIND",      lookback_years=20),
        backtest_signal_chart(lie, dl, "ICSA",         lookback_years=20),
        small=True,
    ))
    story.append(_two_charts(
        backtest_signal_chart(lie, dl, "BAMLH0A0HYM2", lookback_years=20),
        backtest_signal_chart(lie, dl, "T10Y2Y",       lookback_years=20),
        small=True,
    ))
    story.append(Spacer(1, 0.10 * inch))

    # ── Recession backtest table ───────────────────────────────────────────
    story.append(_subsection("Historical Recession Validation"))
    story.append(_prose(
        "Hit rate: percentage of NBER recessions (within the indicator's data history) "
        "preceded by a stressed signal — series above the 75th percentile (higher_is_bad=True) "
        "or below the 25th percentile (higher_is_bad=False) — within 18 months before the "
        "recession start. Lead time: average months of first advance warning (first signal in "
        "the window). False positive rate: percentage of signal episodes not followed by a "
        "recession within 24 months. Only recessions within each indicator's data history are "
        "evaluated; indicators with insufficient history are omitted."
    ))

    bt_data = lie.all_backtests()
    _bt_hdrs = ["Series", "Hit Rate", "Avg Lead (Mo.)", "False Pos. Rate", "Recessions"]
    bt_rows_tbl = [
        [Paragraph(h, _ps("bh", fontName="Helvetica-Bold", fontSize=8,
                           textColor=colors.white))
         for h in _bt_hdrs]
    ]
    for sid in BCI_COMPONENTS:
        bt = bt_data.get(sid, {})
        if not bt or bt.get("n_recessions", 0) == 0:
            continue
        meta = REGISTRY.get(sid, {})
        hit  = bt.get("hit_rate")
        lead = bt.get("avg_lead_months")
        fp   = bt.get("false_pos_rate")
        bt_rows_tbl.append([
            Paragraph(meta.get("short_name", sid), ST_SMALL),
            Paragraph(f"{hit:.0f}%"  if hit  is not None else "N/A", ST_SMALL),
            Paragraph(f"{lead:.1f}"  if lead is not None else "N/A", ST_SMALL),
            Paragraph(f"{fp:.0f}%"   if fp   is not None else "N/A", ST_SMALL),
            Paragraph(str(bt.get("n_recessions", 0)), ST_SMALL),
        ])

    if len(bt_rows_tbl) > 1:
        bt_col_ws = [CONTENT_W * w for w in (0.28, 0.16, 0.20, 0.22, 0.14)]
        bt_tbl = Table(bt_rows_tbl, colWidths=bt_col_ws)
        bt_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), HDR_BG),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f7f8fc")]),
            ("GRID",          (0, 0), (-1, -1), 0.3, RULE_CLR),
        ]))
        story.append(bt_tbl)

    story.append(_prose(
        "<i>Note: Statistics reflect in-sample backtesting against NBER post-hoc recession "
        "dating and are limited to recessions within each indicator's available data history. "
        "ICE BofA HY/IG spread series (BAMLH0A0HYM2) are omitted because FRED's API "
        "returns only ~2 years of history due to licensing restrictions — insufficient for "
        "meaningful backtest evaluation. Initial jobless claims (ICSA) is a coincident "
        "indicator and not expected to signal recessions 18 months in advance. "
        "Data revisions may introduce mild lookahead bias. These figures should inform, "
        "not replace, judgement.</i>"
    ))


def _build_recession_probability(story, rpe: RecessionProbabilityEngine, dl):
    """Probabilistic Macro Forecasting section."""
    story += _section_rule("Recession Probability Model", anchor_id="sec_recession_prob")

    probs = rpe.current_probabilities()

    def _pct(h):
        v = probs.get(h)
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    def _level(h):
        v = probs.get(h)
        if v is None:
            return ("N/A", "#718096")
        if v >= 0.50:
            return ("High", "#9b2c2c")
        if v >= 0.20:
            return ("Elevated", "#975a16")
        return ("Low", "#276749")

    # Probability badges prose
    badge_parts = []
    for h in RPE_HORIZONS:
        lbl, clr = _level(h)
        badge_parts.append(
            f'<b>{h}M:</b> <font color="{clr}"><b>{_pct(h)} ({lbl})</b></font>'
        )
    story.append(_prose("  &nbsp;&nbsp;".join(badge_parts)))
    story.append(Spacer(1, 0.06 * inch))

    story.append(_prose(
        "Recession probabilities are estimated by a pure-numpy logistic regression model "
        "trained on NBER recession dates using eight macro features: yield curve (2Y and 3M "
        "spreads), Conference Board LEI (YoY), initial jobless claims (log, 4-week MA), ISM "
        "manufacturing new orders, high-yield credit spreads, unemployment rate change, and "
        "payroll growth. Separate models are trained per horizon (6M, 12M, 24M) with L2 "
        "regularisation. A 35% probability threshold is used for recession signal flags."
    ))
    story.append(Spacer(1, 0.08 * inch))

    # Gauge chart (3 side-by-side indicators)
    print("  Rendering recession probability gauges...")
    story.append(_full_chart(recession_gauge_chart(probs)))
    story.append(Spacer(1, 0.08 * inch))

    # Rolling probability chart + signal decomposition side-by-side
    print("  Rendering rolling probability and decomposition charts...")
    story.append(_two_charts(
        recession_probability_chart(rpe, dl, lookback_years=20, with_bands=False),
        signal_decomposition_chart(rpe, horizon=12),
        small=False,
    ))
    story.append(Spacer(1, 0.10 * inch))

    # Out-of-sample backtest table
    story.append(_subsection("Out-of-Sample Backtest Performance"))
    story.append(_prose(
        "Rolling out-of-sample backtest: the model is re-trained each month using only "
        "data available at that time (20-year window), then tested one step ahead. "
        "Precision = fraction of high-probability signals followed by a recession. "
        "Recall = fraction of recessions preceded by a high-probability signal. "
        "Hit rate = recessions with at least one signal fired within the horizon."
    ))

    try:
        all_bt = rpe.all_backtests()
    except Exception:
        all_bt = {}

    _bt_hdrs = ["Horizon", "Precision", "Recall", "Hit Rate", "False Pos. Rate", "Obs."]
    bt_rows = [
        [Paragraph(h, _ps("bh", fontName="Helvetica-Bold", fontSize=8,
                           textColor=colors.white))
         for h in _bt_hdrs]
    ]
    for h in RPE_HORIZONS:
        bt = all_bt.get(h, {})
        p   = bt.get("precision")
        r   = bt.get("recall")
        hr  = bt.get("hit_rate")
        fp  = bt.get("false_pos_rate")
        obs = bt.get("n_obs", 0)
        bt_rows.append([
            Paragraph(f"{h}M",             ST_SMALL),
            Paragraph(f"{p*100:.0f}%"  if p  is not None else "N/A", ST_SMALL),
            Paragraph(f"{r*100:.0f}%"  if r  is not None else "N/A", ST_SMALL),
            Paragraph(f"{hr*100:.0f}%" if hr is not None else "N/A", ST_SMALL),
            Paragraph(f"{fp*100:.0f}%" if fp is not None else "N/A", ST_SMALL),
            Paragraph(str(obs),            ST_SMALL),
        ])

    if len(bt_rows) > 1:
        bt_col_ws = [CONTENT_W * w for w in (0.12, 0.16, 0.16, 0.16, 0.22, 0.18)]
        bt_tbl = Table(bt_rows, colWidths=bt_col_ws)
        bt_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), HDR_BG),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f7f8fc")]),
            ("GRID",          (0, 0), (-1, -1), 0.3, RULE_CLR),
        ]))
        story.append(bt_tbl)
        story.append(Spacer(1, 0.06 * inch))

    story.append(_prose(
        "<i>Note: Logistic regression is a baseline model. It captures linear relationships "
        "between macro features and recession risk but does not account for structural breaks, "
        "policy regime changes, or non-linear interactions. Probabilities should be interpreted "
        "as a signal of elevated risk, not a precise forecast.</i>"
    ))


def _build_risk_scorecard(story, re: RiskEngine, dl):
    """Risk Taxonomy Scorecard section — four clean categories."""
    story += _section_rule("Risk Taxonomy Scorecard", anchor_id="sec_risk_scorecard")

    story.append(_prose(
        "Risk is organised into four conceptually distinct categories that can decouple "
        "from each other during different macro regimes. Mixing them produces misleading "
        "aggregate scores — a benign credit environment can mask cyclical deterioration, "
        "and fiscal imbalances can persist for years before affecting near-term growth."
    ))
    story.append(Spacer(1, 0.08 * inch))

    taxonomy = re.risk_taxonomy()
    # taxonomy keys: "Cyclical Recession Risk", "Financial Stability Risk",
    #                "Valuation & Sentiment Risk", "Fiscal & Policy Risk"
    # each value: {"score": str, "components": [(name, risk_level, display_val, as_of), ...],
    #              "description": str}

    _risk_colors = {
        "green":   "#276749",
        "yellow":  "#975a16",
        "red":     "#9b2c2c",
        "neutral": "#718096",
    }

    for cat_label, cat_data in taxonomy.items():
        overall_score = cat_data.get("score", "neutral")
        description   = cat_data.get("description", "")
        components    = cat_data.get("components", [])

        score_clr = _risk_colors.get(overall_score, "#718096")
        story.append(_subsection(
            f'{cat_label}  — '
            f'<font color="{score_clr}"><b>{overall_score.title()}</b></font>'
        ))
        story.append(_prose(description))
        story.append(Spacer(1, 0.04 * inch))

        if not components:
            story.append(_prose("<i>No data available for this category.</i>"))
            story.append(Spacer(1, 0.06 * inch))
            continue

        # Each component: (name, risk_level, display_value, as_of)
        _row_hdrs = ["Indicator", "Current Value", "As Of", "Risk Level"]
        tbl_rows = [
            [Paragraph(h, _ps("ch", fontName="Helvetica-Bold", fontSize=8,
                               textColor=colors.white))
             for h in _row_hdrs]
        ]
        for (name, level, display_val, as_of) in components:
            clr = _risk_colors.get(level, "#718096")
            tbl_rows.append([
                Paragraph(str(name)[:50],      ST_SMALL),
                Paragraph(str(display_val),    ST_SMALL),
                Paragraph(str(as_of),          ST_SMALL),
                Paragraph(f'<font color="{clr}"><b>{level.title()}</b></font>', ST_SMALL),
            ])

        if len(tbl_rows) > 1:
            col_ws = [CONTENT_W * w for w in (0.38, 0.20, 0.22, 0.20)]
            cat_tbl = Table(tbl_rows, colWidths=col_ws)
            cat_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), HDR_BG),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING",   (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f7f8fc")]),
                ("GRID",          (0, 0), (-1, -1), 0.3, RULE_CLR),
            ]))
            story.append(cat_tbl)
        story.append(Spacer(1, 0.10 * inch))

    story.append(_prose(
        "<i>Risk levels are assigned by threshold or percentile rules defined per series in "
        "series_registry.py. Green = below alert threshold; Yellow = moderate stress; "
        "Red = high stress. Overall category score is the maximum component risk level.</i>"
    ))


def build_pdf(output_path: str):
    print("Loading data...")
    sql = SQLStorage.from_config()
    dl = DataLoader(sql=sql)
    re = RiskEngine(dl)
    lie = LeadingIndicatorEngine(dl)
    crisis_dims = re.crisis_dimensions()
    print("Training recession probability model...")
    rpe = RecessionProbabilityEngine(dl)
    rpe.train()
    lae = LaborAnalyticsEngine(dl)
    iae = InflationAnalysisEngine(dl)
    gme = GlobalMacroEngine(dl)
    rge = RegimeEngine(dl)
    sme = StructuralMacroEngine(dl)
    fae = FiscalAnalyticsEngine(dl)

    print(f"Loaded {len(dl.available)} series.")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="FRED Economic Monitor",
        author="FRED Dashboard",
    )

    story = []

    _build_cover(story, dl)

    print("Building Executive Summary...")
    _build_executive_summary(story, re, dl)
    story.append(PageBreak())

    print("Building Summary section...")
    _build_summary(story, re, dl)
    story.append(PageBreak())

    print("Building Crisis Watch section...")
    _build_crisis(story, re, dl, crisis_dims)
    story.append(PageBreak())

    print("Building System Resilience & Policy Dependency section...")
    _build_system_resilience(story, re, dl)
    story.append(PageBreak())

    print("Building Inflation & Money Supply section...")
    _build_inflation(story, iae, dl)
    story.append(PageBreak())

    print("Building Labor Market section...")
    _build_labor(story, lae, dl)
    story.append(PageBreak())

    print("Building Markets & Rates section...")
    _build_markets(story, dl)
    story.append(PageBreak())

    print("Building Housing section...")
    _build_housing(story, dl)
    story.append(PageBreak())

    print("Building Fiscal section...")
    _build_fiscal(story, dl)
    story.append(PageBreak())

    print("Building Leading Indicators & BCI section...")
    _build_leading_indicators(story, lie, re, dl)
    story.append(PageBreak())

    print("Building Recession Probability section...")
    _build_recession_probability(story, rpe, dl)
    story.append(PageBreak())

    print("Building Risk Taxonomy Scorecard section...")
    _build_risk_scorecard(story, re, dl)
    story.append(PageBreak())

    print("Building Global Macro & FX section...")
    _build_global_macro(story, gme, rge, sme, dl)
    story.append(PageBreak())

    print("Building Macro Regime Classification section...")
    _build_macro_regime(story, rge, dl)
    story.append(PageBreak())

    print("Building Fiscal Analytics & Sustainability section...")
    _build_fiscal_analytics(story, fae, dl)
    story.append(PageBreak())

    print("Building Methodology Appendix...")
    _build_methodology(story, re, dl)
    story.append(PageBreak())

    print("Building Glossary section...")
    _build_glossary(story)

    def _page_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED_CLR)
        page_num = canvas.getPageNumber()
        canvas.drawRightString(
            PAGE_W - MARGIN, 0.38 * inch,
            f"Page {page_num}  ·  FRED Economic Monitor  ·  {datetime.now().strftime('%B %d, %Y')}",
        )
        canvas.restoreState()

    print(f"Writing PDF to {output_path} ...")
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    print("Done.")

    print("Uploading to Google Drive...")
    GoogleAPIUploadFile(output_path)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FRED Economic Monitor PDF report.")
    parser.add_argument(
        "output",
        nargs="?",
        default=os.path.join(PATHS.reports,
                             f"FRED_Report_{datetime.now().strftime('%Y%m%d')}.pdf"),
        help="Output PDF path (default: reports/FRED_Report_YYYYMMDD.pdf)",
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    build_pdf(args.output)
