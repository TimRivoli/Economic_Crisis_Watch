"""
Single source of truth for every configuration value, magic number, URL,
color token, and style constant used across the FRED dashboard package.

Import pattern:
    from _classes.constants import PATHS, URLS, API, CHART, TYPOGRAPHY, DASH
    from _classes.constants import PALETTE, RISK_STYLE
    from _classes.constants import STYLE_HEADER, STYLE_CARD, ...
"""

from __future__ import annotations
import os
from dataclasses import dataclass

# ── Package root (parent of this _classes/ directory) ─────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Paths ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Paths:
    root:    str = _ROOT
    data:    str = os.path.join(_ROOT, "data")
    reports: str = os.path.join(_ROOT, "reports")
    config:  str = os.path.join(_ROOT, "config.ini")

PATHS = _Paths()


# ── URLs ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Urls:
    fred_api:     str = "https://api.stlouisfed.org/fred/series/observations"
    shiller_xls:  str = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    multpl_cape:  str = "https://www.multpl.com/shiller-pe/table/by-month"
    multpl_pe:    str = "https://www.multpl.com/s-p-500-pe-ratio/table/by-month"
    google_fonts: str = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap"

URLS = _Urls()


# ── API / Network ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Api:
    timeout_sec:        int   = 30
    rate_limit_sec:     float = 0.25
    user_agent:         str   = "Mozilla/5.0 (compatible; research/data)"
    multpl_date_format: str   = "%b %d, %Y"

API = _Api()


# ── Typography ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Typography:
    font_family: str = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    size_body:   int = 12   # base chart / body text
    size_small:  int = 11   # axis ticks, labels, legends
    size_title:  int = 13   # chart titles
    size_value:  int = 28   # large KPI number on summary cards

TYPOGRAPHY = _Typography()


# ── Chart rendering defaults ───────────────────────────────────────────────

@dataclass(frozen=True)
class _Chart:
    default_lookback_years: int   = 20
    bar_lookback_years:     int   = 5
    line_width:             float = 2.0
    zero_line_width:        float = 0.8
    band_ceiling_mult:      float = 1.15   # y_max = data_max * this
    band_floor_mult:        float = 0.85   # y_min = data_min * this (when negative)
    graph_height:           int   = 320
    graph_height_small:     int   = 260
    inflation_target_pct:   float = 2.0    # reference line on YoY inflation charts
    margin_l:               int   = 52
    margin_r:               int   = 16
    margin_t:               int   = 44
    margin_b:               int   = 36
    legend_y:               float = 1.06
    legend_x:               float = 0.0

CHART = _Chart()


# ── Dashboard application settings ────────────────────────────────────────

@dataclass(frozen=True)
class _Dash:
    host:             str = "127.0.0.1"
    port:             int = 8050
    n_crisis_dims:    int = 6
    date_display_fmt: str = "%#d %b %Y"    # Windows strftime; use "%-d %b %Y" on Linux/Mac

DASH = _Dash()


# ── Color palette ─────────────────────────────────────────────────────────
# Used by chart_factory (as C) and FREDDashboard inline styles.

PALETTE: dict[str, str] = {
    # Chart line colors
    "blue":      "#2b6cb0",
    "teal":      "#2c7a7b",
    "slate":     "#4a5568",
    "green":     "#276749",
    "amber":     "#b7791f",
    "red":       "#9b2c2c",
    # UI structural colors
    "bg":        "#ffffff",
    "page_bg":   "#f7f8fc",
    "header_bg": "#1a365d",
    "grid":      "#edf2f7",
    "text":      "#2d3748",
    "muted":     "#718096",
    # Chart fill / band colors (semi-transparent)
    "fill_blue": "rgba(43,108,176,0.10)",
    "fill_red":  "rgba(155,44,44,0.08)",
    "fill_amb":  "rgba(183,121,31,0.07)",
    "fill_grn":  "rgba(39,103,73,0.07)",
}


# ── Risk level visual tokens ───────────────────────────────────────────────
# Each level provides bg/text/border colors and a human label.

RISK_STYLE: dict[str, dict[str, str]] = {
    "green":   {"bg": "#f0fff4", "text": "#276749", "border": "#68d391", "label": "Within Range"},
    "yellow":  {"bg": "#fffff0", "text": "#975a16", "border": "#f6ad55", "label": "Elevated"},
    "red":     {"bg": "#fff5f5", "text": "#9b2c2c", "border": "#fc8181", "label": "Stressed"},
    "neutral": {"bg": "#f7fafc", "text": "#4a5568", "border": "#cbd5e0", "label": "No Data"},
}


# ── Dashboard CSS style dicts ─────────────────────────────────────────────
# Consumed directly as the `style=` prop on Dash html.* elements.

STYLE_HEADER: dict = {
    "backgroundColor": PALETTE["header_bg"],
    "color":           "#ffffff",
    "padding":         "18px 32px",
    "display":         "flex",
    "alignItems":      "center",
    "justifyContent":  "space-between",
    "fontFamily":      "Inter, sans-serif",
    "boxShadow":       "0 2px 8px rgba(0,0,0,0.18)",
}

STYLE_PAGE: dict = {
    "backgroundColor": PALETTE["page_bg"],
    "minHeight":       "100vh",
    "fontFamily":      "Inter, sans-serif",
}

STYLE_TAB: dict = {
    "padding":         "10px 20px",
    "fontWeight":      "500",
    "fontSize":        f"{TYPOGRAPHY.size_small}px",
    "color":           PALETTE["slate"],
    "borderBottom":    "2px solid transparent",
    "backgroundColor": PALETTE["bg"],
}

STYLE_TAB_SELECTED: dict = {
    "padding":         "10px 20px",
    "fontWeight":      "600",
    "fontSize":        f"{TYPOGRAPHY.size_small}px",
    "color":           PALETTE["blue"],
    "borderBottom":    f"2px solid {PALETTE['blue']}",
    "backgroundColor": PALETTE["bg"],
}

STYLE_CARD: dict = {
    "backgroundColor": PALETTE["bg"],
    "borderRadius":    "6px",
    "padding":         "20px 24px",
    "boxShadow":       "0 1px 4px rgba(0,0,0,0.07)",
    "marginBottom":    "20px",
}

STYLE_SECTION_LABEL: dict = {
    "fontSize":      f"{TYPOGRAPHY.size_small}px",
    "fontWeight":    "600",
    "letterSpacing": "0.08em",
    "color":         PALETTE["muted"],
    "textTransform": "uppercase",
    "marginBottom":  "4px",
}
