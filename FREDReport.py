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
)
from _classes.risk_engine import RiskEngine
from _classes.series_registry import REGISTRY, CATEGORY_LABELS, CATEGORY_ORDER
from _classes.constants import PATHS, PALETTE as C, CHART, DASH, RISK_STYLE
#from _classes.GoogleAPI import GoogleAPIUploadFile


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
    buf = io.BytesIO(png)
    return Image(buf, width=display_w, height=display_w * h_px / w_px)


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


def _half_table(left_items: list, right_items: list) -> Table:
    """Two-column prose layout."""
    t = Table([[left_items, right_items]], colWidths=[HALF_W, HALF_W])
    t.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (0, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, -1), 6),
        ("LEFTPADDING",  (1, 0), (1, -1), 6),
        ("RIGHTPADDING", (1, 0), (1, -1), 0),
    ]))
    return t


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

    rows = [
        [Paragraph(short_markup, _cp("cn", fontName="Helvetica-Bold", fontSize=9))],
        [Paragraph(full_name, _cp("cfn", fontName="Helvetica-Bold", fontSize=7, leading=10))],
        [Paragraph(display_with_arrow, _cp("cv", fontName="Helvetica-Bold", fontSize=17, leading=21))],
        [HRFlowable(width=QUARTER_W - 0.22 * inch, thickness=0.5, color=border)],
        [Paragraph(
            f"<b>{rs['label']}</b>  ·  as of {as_of}",
            _cp("cst", fontSize=7),
        )],
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
    # Inflation
    ("CPI",      "Consumer Price Index — average price change for a fixed basket of urban household goods"),
    ("Core CPI", "CPI excluding food & energy; strips volatile components to show underlying trend"),
    ("Core PCE", "PCE Price Index ex food & energy — the FOMC's explicit 2% inflation target; published by BEA monthly"),
    ("CPI vs PCE","PCE typically runs 0.3–0.5% below CPI due to different weights and substitution methodology"),
    # Money
    ("M1", "Narrow money — physical currency plus checking-account deposits"),
    ("M2", "Broad money — M1 plus savings accounts, money-market funds, and small CDs"),
    ("Real M2", "M2 adjusted for inflation; sustained contraction signals tightening conditions"),
    # Labor
    ("Unemployment Rate", "Share of the labor force actively seeking but unable to find work"),
    ("LFPR", "Labor Force Participation Rate — working-age adults in the labor force (%)"),
    ("Emp/Pop Ratio", "Share of all working-age adults employed; unaffected by participation shifts"),
    ("Nonfarm Payrolls", "Net monthly job additions across all non-agricultural sectors"),
    # Markets
    ("VIX", "CBOE Volatility Index — market's 30-day implied volatility; the 'fear gauge'"),
    ("CAPE / Shiller P/E", "S&P 500 price divided by 10-year inflation-adjusted average earnings"),
    ("Trailing P/E", "Stock price divided by last 12 months of reported earnings"),
    ("10Y Treasury Yield", "Benchmark long-term government rate; affects all asset prices"),
    ("CRE", "Commercial Real Estate — offices, retail, apartments, and industrial property"),
    ("CRE Lending Stds", "Net % banks tightening CRE loan standards (SLOOS); > 40% signals credit contraction ahead"),
    ("CC Delinquency", "Share of credit card loans past due — broad consumer credit health signal"),
    # Liquidity
    ("FSI", "Financial Stress Index — 0 = normal; positive = above-average systemic stress"),
    ("NFCI", "National Financial Conditions Index — Chicago Fed weekly gauge across 105 variables"),
    ("TED Spread", "3-mo LIBOR minus 3-mo T-bill rate — interbank credit risk (discontinued 2023)"),
    ("SOFR", "Secured Overnight Financing Rate — overnight repo benchmark; LIBOR replacement"),
    ("Reverse Repo (RRP)", "Fed facility where banks park excess cash; high usage = ample reserves"),
    ("NFCI Credit", "NFCI subindex isolating credit conditions specifically"),
    # Credit
    ("HY Spread (OAS)", "Extra yield demanded above Treasuries for below-investment-grade bonds"),
    ("IG Spread (OAS)", "Extra yield demanded above Treasuries for investment-grade corporate bonds"),
    ("OAS", "Option-Adjusted Spread — yield spread net of embedded call/put option value"),
    # Policy
    ("Fed Funds Rate", "Fed's overnight policy rate set by the FOMC; the primary monetary tool"),
    ("WALCL", "Fed balance sheet total assets; expanded via QE, shrunk via QT"),
    ("T5YIE / T10YIE", "5- and 10-year breakeven inflation rates from TIPS pricing"),
    ("TIPS", "Treasury bonds whose principal adjusts with CPI; used to extract inflation expectations"),
    ("QE / QT", "Quantitative Easing (buying assets) / Tightening (shrinking balance sheet)"),
    ("BEI / Breakeven", "Market-implied inflation expectations from the gap between TIPS and nominal yields"),
    ("SAAR", "Seasonally Adjusted Annual Rate — removes seasonal patterns, expressed at annual pace"),
    ("Debt / GDP", "Federal debt as a percentage of gross domestic product"),
    ("Interest / Receipts", "Federal interest payments as a share of total government revenues"),
    ("Real FF Rate", "Fed Funds Rate minus Core CPI YoY — the inflation-adjusted policy stance"),
    # Yield curve / recession
    ("Yield Curve (10Y-2Y)", "Spread between 10-year and 2-year Treasury yields; inversion has preceded every US recession since 1970"),
    ("Inverted Curve", "When short rates exceed long rates (spread < 0) — markets pricing near-term risk higher than long-run growth"),
    ("NY Fed Rec. Prob.", "Model-based 12-month recession probability using yield curve slope; above 30% has historically been a reliable signal"),
    # Bank lending
    ("SLOOS", "Senior Loan Officer Opinion Survey — quarterly Fed survey on bank lending standards and demand"),
    ("Lending Standards", "Net % of banks tightening loan conditions; tightening > 40% historically signals credit contraction ahead"),
    # Housing
    ("Housing Starts", "New residential units started monthly (SAAR); a leading indicator for construction employment and materials"),
    ("Case-Shiller HPI", "S&P/Case-Shiller repeat-sales home price index; published with ~2-month lag"),
    ("Mortgage Rate (30Y)", "Freddie Mac Primary Mortgage Market Survey; directly affects housing affordability and purchase volume"),
    # Dollar
    ("Dollar Index (DTWEX)", "Trade-weighted U.S. dollar index vs. 26 currencies; rapid appreciation tightens global financial conditions"),
]


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

    layout = Table(
        [[_glossary_col(left_col), _glossary_col(right_col)]],
        colWidths=[HALF_W, HALF_W],
    )
    layout.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (0, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, -1), 12),
        ("LEFTPADDING",  (1, 0), (1, -1), 12),
        ("RIGHTPADDING", (1, 0), (1, -1), 0),
    ]))
    story.append(layout)


def _build_summary(story, re: RiskEngine, dl):
    story += _section_rule("Summary — All Indicators", anchor_id="sec_summary")
    story.append(_prose(
        "Each card shows the current reading, risk status, and full indicator name. "
        "Click the indicator name to jump to the detailed section. "
        "Normal (green) · Elevated (yellow) · Stressed (red). "
        "Context and combinations matter more than any single reading."
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


def _build_inflation(story, dl):
    story += _section_rule("Inflation", anchor_id="sec_inflation")
    print("  Rendering inflation charts...")
    story.append(_two_charts(
        area_chart(dl, "CPIAUCSL", yoy=True, lookback_years=25,
                   threshold_green=2.5, threshold_red=4.5,
                   title="Headline CPI — Year-over-Year %"),
        area_chart(dl, "CPILFESL", yoy=True, lookback_years=25,
                   threshold_green=2.5, threshold_red=4.0,
                   color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                   title="Core CPI (ex Food & Energy) — Year-over-Year %"),
    ))
    story.append(_two_charts(
        multi_line_chart(dl,
            [("CPILFESL", C["teal"]), ("PCEPILFE", C["blue"])],
            title="Core CPI vs. Core PCE — YoY % (10 Years)",
            lookback_years=10, yoy=True),
        area_chart(dl, "PCEPILFE", yoy=True, lookback_years=25,
                   threshold_green=2.5, threshold_red=4.0,
                   color=C["blue"], fill_color="rgba(43,108,176,0.10)",
                   title="Core PCE — Year-over-Year %"),
    ))
    story.append(_prose(
        "Core PCE (Personal Consumption Expenditures ex food & energy) is the FOMC's explicit "
        "2% inflation target, published monthly by the BEA with a ~4 week lag. It typically runs "
        "0.3–0.5% below Core CPI due to different category weights and substitution effects. "
        "When both measures are simultaneously elevated, the inflation signal is more credible "
        "and policy response more likely."
    ))
    story.append(Spacer(1, 0.10 * inch))
    story.append(_subsection("Money Supply"))
    print("  Rendering money supply charts...")
    story.append(_full_chart(multi_line_chart(dl,
        [("M1SL", C["blue"]), ("M2SL", C["teal"])],
        title="M1 & M2 Money Supply — Level (Billions $)",
        lookback_years=20,
    )))
    story.append(_prose(
        "M1 is the narrowest measure of money: currency and demand deposits. "
        "M2 adds savings accounts, retail money market funds, and small CDs. "
        "Rapid M2 growth often precedes inflation with a 12–18 month lag. "
        "Real M2 contraction — M2 growing slower than inflation — typically signals "
        "tightening monetary conditions."
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_two_charts(
        area_chart(dl, "M2SL", yoy=True, lookback_years=30,
                   title="M2 Growth — Year-over-Year %"),
        area_chart(dl, "M2REAL", yoy=True, lookback_years=20,
                   color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                   title="Real M2 Growth — Year-over-Year %"),
    ))


def _build_labor(story, dl):
    story += _section_rule("Labor Market", anchor_id="sec_labor")
    print("  Rendering labor charts...")
    story.append(_two_charts(
        area_chart(dl, "UNRATE", lookback_years=30,
                   threshold_green=4.5, threshold_red=6.0,
                   color=C["slate"], fill_color="rgba(74,85,104,0.08)",
                   title="Unemployment Rate %"),
        bar_change_chart(dl, "PAYEMS", lookback_years=5,
                         title="Nonfarm Payrolls — Monthly Change (Thousands)"),
    ))
    story.append(_two_charts(
        line_chart(dl, "CIVPART", lookback_years=30, color=C["teal"],
                   title="Labor Force Participation Rate %"),
        multi_line_chart(dl,
            [("EMRATIO", C["blue"]), ("CIVPART", C["teal"])],
            title="Employment-Population Ratio vs. Participation Rate %",
            lookback_years=30),
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_subsection("Initial Claims, Savings Rate & Leading Indicators"))
    story.append(_prose(
        "Initial jobless claims (4-week moving average) provide the most timely read on "
        "labor market health — released weekly with a 4-day lag. The personal saving rate "
        "tracks household buffer capacity: sustained lows signal stretched balance sheets."
    ))
    story.append(_two_charts(
        area_chart(dl, "ICSA", lookback_years=10,
                   ma_periods=4,
                   color=C["amber"], fill_color="rgba(183,121,31,0.09)",
                   threshold_green=250, threshold_red=350,
                   title="Initial Jobless Claims — 4-Week MA (Weekly)",
                   recession_shading=True),
        area_chart(dl, "PSAVERT", lookback_years=20,
                   color=C["teal"], fill_color="rgba(44,122,123,0.09)",
                   threshold_green=7, threshold_red=4,
                   title="Personal Saving Rate %"),
    ))
    story.append(_two_charts(
        area_chart(dl, "INDPRO", yoy=True, lookback_years=20,
                   color=C["blue"], fill_color="rgba(43,108,176,0.10)",
                   threshold_green=1, threshold_red=-1,
                   title="Industrial Production — Year-over-Year %",
                   recession_shading=True),
        area_chart(dl, "USSLIND", lookback_years=10,
                   color=C["red"], fill_color="rgba(155,44,44,0.09)",
                   threshold_green=0, threshold_red=-5,
                   title="Conference Board Leading Economic Index (MoM %)"),
    ))
    story.append(_prose(
        "Industrial production YoY contraction has coincided with every U.S. recession since 1960. "
        "The Conference Board LEI aggregates 10 forward-looking components; three consecutive "
        "monthly declines have preceded every NBER recession with a 6-12 month lead."
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
        "earnings); VIX measures near-term sentiment and implied volatility. Historically, "
        "high-CAPE regimes combined with VIX spikes mark the sharpest drawdown episodes — "
        "2001, 2008, and 2022. CAPE above 30 with a VIX spike above 30 has preceded "
        "significant multi-year return headwinds."
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

def build_pdf(output_path: str):
    print("Loading data...")
    sql = SQLStorage.from_config()
    dl = DataLoader(sql=sql)
    re = RiskEngine(dl)
    crisis_dims = re.crisis_dimensions()

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
    _build_inflation(story, dl)
    story.append(PageBreak())

    print("Building Labor Market section...")
    _build_labor(story, dl)
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
    #GoogleAPIUploadFile(output_path)


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
