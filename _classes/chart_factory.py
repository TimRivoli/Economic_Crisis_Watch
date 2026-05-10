"""
Reusable Plotly chart functions.  All functions accept a DataLoader instance
and return a go.Figure ready to embed in dcc.Graph.

Design: institutional palette, minimal chrome, no emoji, clean grid lines.
"""

import pandas as pd
import plotly.graph_objects as go
from .series_registry import REGISTRY
from .constants import PALETTE as C, TYPOGRAPHY, CHART

_BASE = dict(
    font=dict(family=TYPOGRAPHY.font_family, size=TYPOGRAPHY.size_body, color=C["text"]),
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["bg"],
    margin=dict(l=CHART.margin_l, r=CHART.margin_r, t=CHART.margin_t, b=CHART.margin_b),
    hovermode="x unified",
    legend=dict(orientation="h", y=CHART.legend_y, x=CHART.legend_x,
                font=dict(size=TYPOGRAPHY.size_small)),
    xaxis=dict(showgrid=True, gridcolor=C["grid"], linecolor=C["grid"],
               tickfont=dict(size=TYPOGRAPHY.size_small), zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=C["grid"], linecolor=C["grid"],
               tickfont=dict(size=TYPOGRAPHY.size_small), zeroline=False),
)


def _fig(title: str | None = None) -> go.Figure:
    fig = go.Figure()
    layout = dict(_BASE)
    if title:
        layout["title"] = dict(text=title, font=dict(size=TYPOGRAPHY.size_title, color=C["text"]), x=0, xanchor="left")
    fig.update_layout(**layout)
    return fig


def _trim(df: pd.DataFrame, years: int | None) -> pd.DataFrame:
    if years is None or df.empty:
        return df
    cutoff = df.index[-1] - pd.DateOffset(years=years)
    return df[df.index >= cutoff]


def _resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.resample("ME").last().dropna()
    except ValueError:
        return df.resample("M").last().dropna()


def _meta(series_id: str) -> dict:
    return REGISTRY.get(series_id, {})


# ── Public Chart Functions ─────────────────────────────────────────────────

def line_chart(
    dl,
    series_id: str,
    title: str | None = None,
    color: str | None = None,
    lookback_years: int | None = CHART.default_lookback_years,
    yoy: bool = False,
) -> go.Figure:
    """Standard time-series line chart, optionally showing YoY % change."""
    df = dl.load(series_id)
    meta = _meta(series_id)
    auto_title = title or meta.get("name", series_id)

    if df is None or df.empty:
        return _no_data(auto_title)

    plot = _trim(df, lookback_years)

    if yoy:
        monthly = _resample_monthly(plot)
        plot = (monthly.pct_change(12) * 100).dropna()
        auto_title = (title or meta.get("name", series_id)) + " — Year-over-Year %"

    col = plot.columns[0]
    line_color = color or C["blue"]

    fig = _fig(auto_title)
    fig.add_trace(go.Scatter(
        x=plot.index, y=plot[col],
        mode="lines",
        name=meta.get("short_name", series_id),
        line=dict(color=line_color, width=CHART.line_width),
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    if yoy:
        fig.add_hline(y=CHART.inflation_target_pct, line_color=C["grid"], line_dash="dot",
                      line_width=CHART.zero_line_width,
                      annotation_text="2% target", annotation_font_size=TYPOGRAPHY.size_small)
    return fig


def area_chart(
    dl,
    series_id: str,
    title: str | None = None,
    color: str | None = None,
    fill_color: str | None = None,
    lookback_years: int | None = CHART.default_lookback_years,
    yoy: bool = False,
    threshold_green: float | None = None,
    threshold_red: float | None = None,
) -> go.Figure:
    """Filled area chart with optional horizontal risk-band shading."""
    df = dl.load(series_id)
    meta = _meta(series_id)
    auto_title = title or meta.get("name", series_id)

    if df is None or df.empty:
        return _no_data(auto_title)

    plot = _trim(df, lookback_years)

    if yoy:
        monthly = _resample_monthly(plot)
        plot = (monthly.pct_change(12) * 100).dropna()
        auto_title = (title or meta.get("name", series_id)) + " — Year-over-Year %"

    col = plot.columns[0]
    line_color = color or C["blue"]
    area_fill = fill_color or C["fill_blue"]

    fig = _fig(auto_title)

    # Risk bands
    y_max = float(plot[col].max()) * CHART.band_ceiling_mult
    y_min = float(plot[col].min()) * CHART.band_floor_mult if plot[col].min() < 0 else 0
    if threshold_red is not None:
        fig.add_hrect(y0=threshold_red, y1=y_max, fillcolor=C["fill_red"], line_width=0)
    if threshold_green is not None and threshold_red is not None:
        fig.add_hrect(y0=threshold_green, y1=threshold_red, fillcolor=C["fill_amb"], line_width=0)

    fig.add_trace(go.Scatter(
        x=plot.index, y=plot[col],
        mode="lines",
        fill="tozeroy",
        fillcolor=area_fill,
        name=meta.get("short_name", series_id),
        line=dict(color=line_color, width=CHART.line_width),
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width, line_dash="dot")
    return fig


def multi_line_chart(
    dl,
    series: list[tuple[str, str]],   # [(series_id, color), ...]
    title: str | None = None,
    lookback_years: int | None = CHART.default_lookback_years,
    yoy: bool = False,
) -> go.Figure:
    """Overlay multiple series on one chart."""
    fig = _fig(title)
    for series_id, color in series:
        df = dl.load(series_id)
        if df is None or df.empty:
            continue
        meta = _meta(series_id)
        plot = _trim(df, lookback_years)
        if yoy:
            monthly = _resample_monthly(plot)
            plot = (monthly.pct_change(12) * 100).dropna()
        col = plot.columns[0]
        fig.add_trace(go.Scatter(
            x=plot.index, y=plot[col],
            mode="lines",
            name=meta.get("short_name", series_id),
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"{meta.get('short_name', series_id)}: %{{y:.2f}}<extra></extra>",
        ))
    return fig


def bar_change_chart(
    dl,
    series_id: str,
    title: str | None = None,
    lookback_years: int = CHART.bar_lookback_years,
) -> go.Figure:
    """Month-over-month absolute change as a green/red bar chart."""
    df = dl.load(series_id)
    meta = _meta(series_id)
    auto_title = title or (meta.get("name", series_id) + " — Monthly Change")

    if df is None or df.empty:
        return _no_data(auto_title)

    monthly = _resample_monthly(df)
    changes = monthly.diff().dropna()
    changes = _trim(changes, lookback_years)
    col = changes.columns[0]

    colors = [C["green"] if v >= 0 else C["red"] for v in changes[col]]

    fig = _fig(auto_title)
    fig.add_trace(go.Bar(
        x=changes.index, y=changes[col],
        marker_color=colors,
        name="MoM Change",
        hovertemplate="%{y:+,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width)
    return fig


def _no_data(title: str) -> go.Figure:
    fig = _fig(title)
    fig.add_annotation(
        text="Data not available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=TYPOGRAPHY.size_title, color=C["slate"]),
    )
    return fig
