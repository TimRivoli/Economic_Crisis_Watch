"""
Reusable Plotly chart functions.  All functions accept a DataLoader instance
and return a go.Figure ready to embed in dcc.Graph.

Design: institutional palette, minimal chrome, no emoji, clean grid lines.
"""

import pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def _or_empty(df) -> pd.DataFrame:
    return df if df is not None and not df.empty else pd.DataFrame()


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
    events: list | None = None,
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
        if not title:
            auto_title = meta.get("name", series_id) + " — Year-over-Year %"

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
    if events:
        _add_events(fig, plot, events)
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
    recession_shading: bool = False,
    events: list | None = None,
    ma_periods: int | None = None,
) -> go.Figure:
    """Filled area chart with optional horizontal risk-band shading.
    When ma_periods is set, overlays a rolling mean as a solid line over a faint raw area."""
    df = dl.load(series_id)
    meta = _meta(series_id)
    auto_title = title or meta.get("name", series_id)

    if df is None or df.empty:
        return _no_data(auto_title)

    plot = _trim(df, lookback_years)

    if yoy:
        monthly = _resample_monthly(plot)
        plot = (monthly.pct_change(12) * 100).dropna()
        if not title:
            auto_title = meta.get("name", series_id) + " — Year-over-Year %"

    col = plot.columns[0]
    line_color = color or C["blue"]
    area_fill = fill_color or C["fill_blue"]

    fig = _fig(auto_title)

    if recession_shading:
        _add_recession_bands(fig, dl, lookback_years)

    # Risk bands
    y_max = float(plot[col].max()) * CHART.band_ceiling_mult
    y_min = float(plot[col].min()) * CHART.band_floor_mult if plot[col].min() < 0 else 0
    if threshold_red is not None:
        fig.add_hrect(y0=threshold_red, y1=y_max, fillcolor=C["fill_red"], line_width=0, layer="below")
    if threshold_green is not None and threshold_red is not None:
        fig.add_hrect(y0=threshold_green, y1=threshold_red, fillcolor=C["fill_amb"], line_width=0, layer="below")

    short_name = meta.get("short_name", series_id)
    if ma_periods:
        fig.add_trace(go.Scatter(
            x=plot.index, y=plot[col],
            mode="lines",
            fill="tozeroy",
            fillcolor=area_fill,
            name=short_name,
            line=dict(color=line_color, width=0.8),
            opacity=0.35,
            hovertemplate="%{y:,.0f}<extra></extra>",
            showlegend=False,
        ))
        ma = plot[col].rolling(ma_periods).mean().dropna()
        fig.add_trace(go.Scatter(
            x=ma.index, y=ma,
            mode="lines",
            name=f"{ma_periods}-wk avg",
            line=dict(color=line_color, width=CHART.line_width),
            hovertemplate=f"{ma_periods}-wk avg: %{{y:,.0f}}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=plot.index, y=plot[col],
            mode="lines",
            fill="tozeroy",
            fillcolor=area_fill,
            name=short_name,
            line=dict(color=line_color, width=CHART.line_width),
            hovertemplate="%{y:.2f}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width, line_dash="dot")
    if events:
        _add_events(fig, plot, events)
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


def _add_events(fig: go.Figure, plot: pd.DataFrame, events: list):
    """Add vertical dashed annotation lines for historical events within the plot range."""
    if plot.empty or not events:
        return
    x_min, x_max = plot.index[0], plot.index[-1]
    for date_str, label in events:
        dt = pd.to_datetime(date_str)
        if x_min <= dt <= x_max:
            # Plotly requires a numeric ms-epoch value on datetime axes when
            # annotation_position is used (string x causes a type error internally)
            x_ms = int(dt.timestamp() * 1000)
            fig.add_vline(
                x=x_ms,
                line_dash="dot",
                line_color=C["slate"],
                line_width=1,
                annotation_text=label,
                annotation_font_size=TYPOGRAPHY.size_small - 1,
                annotation_textangle=-90,
                annotation_position="top left",
            )


def _no_data(title: str) -> go.Figure:
    fig = _fig(title)
    fig.add_annotation(
        text="Data not available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=TYPOGRAPHY.size_title, color=C["slate"]),
    )
    return fig


# ── Recession Shading ──────────────────────────────────────────────────────

def _add_recession_bands(fig: go.Figure, dl, lookback_years: int | None = None):
    """Add NBER recession shading (grey vertical bands) to an existing figure."""
    df = dl.load("USREC")
    if df is None or df.empty:
        return
    plot = _trim(df, lookback_years) if lookback_years else df
    col = plot.columns[0]
    in_rec = False
    start = None
    for date, val in plot[col].items():
        if val == 1 and not in_rec:
            in_rec = True
            start = date
        elif val == 0 and in_rec:
            in_rec = False
            fig.add_vrect(x0=start.isoformat(), x1=date.isoformat(),
                          fillcolor="rgba(180,180,180,0.18)", line_width=0, layer="below")
    if in_rec and start is not None:
        fig.add_vrect(x0=start.isoformat(), x1=plot.index[-1].isoformat(),
                      fillcolor="rgba(180,180,180,0.18)", line_width=0, layer="below")


# ── Extended Chart Functions ───────────────────────────────────────────────

def percentile_chart(
    dl,
    series_id: str,
    title: str | None = None,
    color: str | None = None,
    fill_color: str | None = None,
    lookback_years: int | None = 20,
    higher_is_bad: bool = True,
    recession_shading: bool = False,
    events: list | None = None,
) -> go.Figure:
    """
    Area chart with dynamic percentile bands computed from the full history.
    Bands mark the 25th and 75th historical percentiles; an annotation shows
    where the current reading falls in the long-run distribution.
    """
    df = dl.load(series_id)
    meta = _meta(series_id)
    auto_title = title or meta.get("name", series_id)

    if df is None or df.empty:
        return _no_data(auto_title)

    col = df.columns[0]
    all_vals = df[col].dropna()
    p25 = float(all_vals.quantile(0.25))
    p75 = float(all_vals.quantile(0.75))

    plot = _trim(df, lookback_years)
    current_val = float(plot[col].iloc[-1]) if not plot.empty else None
    pct_rank = float((all_vals < current_val).mean() * 100) if current_val is not None else None
    pct_label = f"  (current: {pct_rank:.0f}th percentile)" if pct_rank is not None else ""

    line_color = color or C["blue"]
    area_fill = fill_color or C["fill_blue"]
    y_max = float(all_vals.max()) * CHART.band_ceiling_mult
    y_min = float(all_vals.min()) * CHART.band_floor_mult if all_vals.min() < 0 else 0

    fig = _fig(auto_title + pct_label)

    if recession_shading:
        _add_recession_bands(fig, dl, lookback_years)

    if higher_is_bad:
        fig.add_hrect(y0=p75, y1=y_max, fillcolor=C["fill_red"], line_width=0, layer="below")
        fig.add_hrect(y0=p25, y1=p75, fillcolor=C["fill_amb"], line_width=0, layer="below")
    else:
        fig.add_hrect(y0=y_min, y1=p25, fillcolor=C["fill_red"], line_width=0, layer="below")
        fig.add_hrect(y0=p25, y1=p75, fillcolor=C["fill_amb"], line_width=0, layer="below")

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
    if events:
        _add_events(fig, plot, events)
    return fig


def dual_axis_chart(
    dl,
    left_series: tuple,
    right_series: tuple,
    title: str | None = None,
    lookback_years: int | None = 20,
    recession_shading: bool = False,
) -> go.Figure:
    """
    Two series plotted on independent Y-axes for scale-incompatible comparisons
    (e.g., HY spread vs VIX).  Left axis is solid; right axis is dashed.
    """
    left_id, left_color = left_series
    right_id, right_color = right_series
    left_meta = _meta(left_id)
    right_meta = _meta(right_id)
    auto_title = title or (f"{left_meta.get('short_name', left_id)} vs "
                           f"{right_meta.get('short_name', right_id)}")

    fig = _fig(auto_title)

    if recession_shading:
        _add_recession_bands(fig, dl, lookback_years)

    left_df = dl.load(left_id)
    if left_df is not None and not left_df.empty:
        lp = _trim(left_df, lookback_years)
        col = lp.columns[0]
        fig.add_trace(go.Scatter(
            x=lp.index, y=lp[col],
            mode="lines",
            name=left_meta.get("short_name", left_id),
            line=dict(color=left_color, width=CHART.line_width),
            yaxis="y1",
            hovertemplate=f"{left_meta.get('short_name', left_id)}: %{{y:.2f}}<extra></extra>",
        ))

    right_df = dl.load(right_id)
    if right_df is not None and not right_df.empty:
        rp = _trim(right_df, lookback_years)
        col = rp.columns[0]
        fig.add_trace(go.Scatter(
            x=rp.index, y=rp[col],
            mode="lines",
            name=right_meta.get("short_name", right_id),
            line=dict(color=right_color, width=CHART.line_width, dash="dash"),
            yaxis="y2",
            hovertemplate=f"{right_meta.get('short_name', right_id)}: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        yaxis=dict(
            title=dict(text=left_meta.get("short_name", left_id),
                       font=dict(size=TYPOGRAPHY.size_small, color=left_color)),
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            linecolor=C["grid"],
            tickfont=dict(size=TYPOGRAPHY.size_small),
            zeroline=False,
            title=dict(text=right_meta.get("short_name", right_id),
                       font=dict(size=TYPOGRAPHY.size_small, color=right_color)),
        ),
    )
    return fig


def derived_ratio_chart(
    dl,
    numerator_id: str,
    denominator_id: str,
    title: str | None = None,
    color: str | None = None,
    fill_color: str | None = None,
    lookback_years: int | None = None,
    threshold_green: float | None = None,
    threshold_red: float | None = None,
    scale: float = 100.0,
    y_label: str = "%",
    recession_shading: bool = False,
    overlay_id: str | None = None,
    overlay_color: str | None = None,
    overlay_label: str | None = None,
) -> go.Figure:
    """Chart showing numerator/denominator * scale over time (e.g. interest/receipts %).
    Optional overlay_id adds a second series on a right y-axis (e.g. 10Y Treasury yield)."""
    df_num = dl.load(numerator_id)
    df_den = dl.load(denominator_id)
    auto_title = title or f"{numerator_id} / {denominator_id} × {scale:.0f}"

    if df_num is None or df_den is None:
        return _no_data(auto_title)

    combined = pd.concat([df_num, df_den], axis=1).dropna()
    if combined.empty:
        return _no_data(auto_title)

    ratio = combined.iloc[:, 0] / combined.iloc[:, 1] * scale
    plot = _trim(ratio.to_frame(name="ratio"), lookback_years)
    col = plot.columns[0]
    line_color = color or C["amber"]
    area_fill = fill_color or C["fill_amb"]

    fig = _fig(auto_title)

    if recession_shading:
        _add_recession_bands(fig, dl, lookback_years)

    y_max = float(plot[col].max()) * CHART.band_ceiling_mult
    if threshold_red is not None:
        fig.add_hrect(y0=threshold_red, y1=y_max, fillcolor=C["fill_red"], line_width=0, layer="below")
    if threshold_green is not None and threshold_red is not None:
        fig.add_hrect(y0=threshold_green, y1=threshold_red, fillcolor=C["fill_amb"], line_width=0, layer="below")

    ratio_name = title.split("(")[0].strip() if title else "Ratio"
    fig.add_trace(go.Scatter(
        x=plot.index, y=plot[col],
        mode="lines",
        fill="tozeroy",
        fillcolor=area_fill,
        name=ratio_name,
        line=dict(color=line_color, width=CHART.line_width),
        yaxis="y1",
        hovertemplate=f"%{{y:.1f}}{y_label}<extra></extra>",
    ))

    if overlay_id is not None:
        df_ov = dl.load(overlay_id)
        if df_ov is not None and not df_ov.empty:
            ov_plot = _trim(df_ov, lookback_years)
            ov_col = ov_plot.columns[0]
            ov_color = overlay_color or C["blue"]
            ov_name = overlay_label or _meta(overlay_id).get("short_name", overlay_id)
            fig.add_trace(go.Scatter(
                x=ov_plot.index, y=ov_plot[ov_col],
                mode="lines",
                name=ov_name,
                line=dict(color=ov_color, width=CHART.line_width, dash="dash"),
                yaxis="y2",
                hovertemplate=f"{ov_name}: %{{y:.2f}}%<extra></extra>",
            ))
            fig.update_layout(
                margin=dict(r=52),
                yaxis2=dict(
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    linecolor=C["grid"],
                    tickfont=dict(size=TYPOGRAPHY.size_small, color=ov_color),
                    title=dict(text=ov_name, font=dict(size=TYPOGRAPHY.size_small, color=ov_color)),
                    zeroline=False,
                ),
            )

    return fig


def derived_spread_chart(
    dl,
    series_a: str,
    series_b: str,
    title: str | None = None,
    color: str | None = None,
    fill_color: str | None = None,
    lookback_years: int | None = None,
    threshold_green: float | None = None,
    threshold_red: float | None = None,
    recession_shading: bool = False,
    events: list | None = None,
) -> go.Figure:
    """Area chart of series_a minus series_b (e.g. CP-Treasury spread)."""
    df_a = dl.load(series_a)
    df_b = dl.load(series_b)
    auto_title = title or f"{series_a} minus {series_b}"

    if df_a is None or df_b is None:
        return _no_data(auto_title)

    combined = pd.concat([df_a, df_b], axis=1).dropna()
    if combined.empty:
        return _no_data(auto_title)

    spread = combined.iloc[:, 0] - combined.iloc[:, 1]
    plot = _trim(spread.to_frame(name="spread"), lookback_years)
    col = plot.columns[0]
    line_color = color or C["amber"]
    area_fill = fill_color or C["fill_amb"]

    fig = _fig(auto_title)

    if recession_shading:
        _add_recession_bands(fig, dl, lookback_years)

    y_max = float(plot[col].max()) * CHART.band_ceiling_mult
    if threshold_red is not None:
        fig.add_hrect(y0=threshold_red, y1=y_max, fillcolor=C["fill_red"], line_width=0, layer="below")
    if threshold_green is not None and threshold_red is not None:
        fig.add_hrect(y0=threshold_green, y1=threshold_red, fillcolor=C["fill_amb"], line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=plot.index, y=plot[col],
        mode="lines", fill="tozeroy",
        fillcolor=area_fill,
        name="Spread",
        line=dict(color=line_color, width=CHART.line_width),
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width)
    _add_events(fig, plot, events)
    return fig


def walcl_pct_gdp_chart(dl, lookback_years: int | None = 25) -> go.Figure:
    """Fed balance sheet as % of nominal GDP (WALCL millions / GDP billions * 100)."""
    df_walcl = dl.load("WALCL")
    df_gdp = dl.load("GDP")
    title = "Fed Balance Sheet as % of Nominal GDP"

    if df_walcl is None or df_gdp is None:
        return _no_data(title)

    try:
        walcl_q = df_walcl.resample("QE").last().dropna()
        gdp_q = df_gdp.resample("QE").last().dropna()
    except ValueError:
        walcl_q = df_walcl.resample("Q").last().dropna()
        gdp_q = df_gdp.resample("Q").last().dropna()

    combined = pd.concat([walcl_q, gdp_q], axis=1).dropna()
    if combined.empty:
        return _no_data(title)

    # WALCL in millions / (GDP in billions * 1000) * 100 → percent
    ratio = combined.iloc[:, 0] / (combined.iloc[:, 1] * 1000) * 100
    plot = _trim(ratio.to_frame(name="BS_pct_GDP"), lookback_years)

    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)
    fig.add_trace(go.Scatter(
        x=plot.index, y=plot["BS_pct_GDP"],
        mode="lines",
        fill="tozeroy",
        fillcolor=C["fill_blue"],
        name="Balance Sheet % GDP",
        line=dict(color=C["blue"], width=CHART.line_width),
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))
    return fig


def real_rate_chart(dl, lookback_years: int | None = 30) -> go.Figure:
    """Real Federal Funds Rate = FEDFUNDS minus Core CPI YoY %."""
    df_ff = dl.load("FEDFUNDS")
    df_cpi = dl.load("CPILFESL")
    title = "Real Federal Funds Rate (FEDFUNDS − Core CPI YoY) %"

    if df_ff is None or df_cpi is None:
        return _no_data(title)

    ff_m = _resample_monthly(df_ff)
    cpi_yoy = (_resample_monthly(df_cpi).pct_change(12) * 100).dropna()
    combined = pd.concat([ff_m, cpi_yoy], axis=1).dropna()
    if combined.empty:
        return _no_data(title)

    real = combined.iloc[:, 0] - combined.iloc[:, 1]
    plot = _trim(real.to_frame(name="Real FF"), lookback_years)
    col = plot.columns[0]

    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)
    fig.add_trace(go.Scatter(
        x=plot.index, y=plot[col],
        mode="lines",
        fill="tozeroy",
        fillcolor=C["fill_blue"],
        name="Real FF Rate",
        line=dict(color=C["blue"], width=CHART.line_width),
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width, line_dash="dot",
                  annotation_text="Neutral (0%)", annotation_font_size=TYPOGRAPHY.size_small)
    return fig


def yield_spread_chart(dl, lookback_years: int | None = 20,
                       recession_shading: bool = True) -> go.Figure:
    """
    10Y−2Y Treasury yield spread with inversion zones shaded red.
    Positive (normal) regions shaded teal; zero line marked prominently.
    """
    df = dl.load("T10Y2Y")
    title = "Yield Curve: 10-Year minus 2-Year Treasury Spread %"
    if df is None or df.empty:
        return _no_data(title)

    plot = _trim(df, lookback_years)
    col = plot.columns[0]

    fig = _fig(title)
    if recession_shading:
        _add_recession_bands(fig, dl, lookback_years)

    pos = plot[col].clip(lower=0)
    neg = plot[col].clip(upper=0)

    fig.add_trace(go.Scatter(
        x=plot.index, y=pos,
        mode="lines", fill="tozeroy",
        fillcolor="rgba(44,122,123,0.10)",
        line=dict(color="rgba(44,122,123,0)", width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=plot.index, y=neg,
        mode="lines", fill="tozeroy",
        fillcolor="rgba(155,44,44,0.15)",
        line=dict(color="rgba(155,44,44,0)", width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=plot.index, y=plot[col],
        mode="lines", name="10Y − 2Y Spread",
        line=dict(color=C["blue"], width=CHART.line_width),
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["red"], line_width=1.2, line_dash="dash",
                  annotation_text="Inversion", annotation_font_size=TYPOGRAPHY.size_small,
                  annotation_position="right")
    return fig


def risk_heatmap_chart(re, lookback_years: int = 10) -> go.Figure:
    """
    Historical risk-status heatmap: rows = key indicators, columns = monthly dates.
    Color encoding: 0=neutral(grey), 1=green, 2=yellow, 3=red.
    Accepts any object with a score_history(series_id, lookback_years) method.
    """
    _HEATMAP_SERIES = [
        ("CPILFESL",      "Core CPI"),
        ("UNRATE",        "Unemployment"),
        ("VIXCLS",        "VIX"),
        ("SP500_CAPE",    "CAPE"),
        ("T10Y2Y",        "Yield Curve"),
        ("RECPROUSM156N", "Rec. Prob."),
        ("BAMLH0A0HYM2",  "HY Spread"),
        ("BAMLC0A0CM",    "IG Spread"),
        ("STLFSI4",       "Stress Index"),
        ("NFCI",          "NFCI"),
        ("GFDEGDQ188S",   "Debt/GDP"),
        ("DRTSCILM",      "Lending Stds"),
    ]

    all_scores: dict[str, pd.Series] = {}
    for sid, label in _HEATMAP_SERIES:
        try:
            scores = re.score_history(sid, lookback_years)
            if scores is not None and not scores.empty:
                all_scores[label] = scores
        except Exception:
            pass

    title = f"Indicator Risk Status — {lookback_years}-Year History (Monthly)"
    if not all_scores:
        return _no_data(title)

    combined = pd.DataFrame(all_scores)
    try:
        combined = combined.resample("ME").last()
    except ValueError:
        combined = combined.resample("M").last()
    combined = combined.fillna(0)

    # Step-wise colorscale: discrete bands at 0/1/2/3
    colorscale = [
        [0.000, "rgb(225,225,225)"],
        [0.001, "rgb(225,225,225)"],
        [0.001, "#276749"],
        [0.332, "#276749"],
        [0.333, "#d69e2e"],
        [0.665, "#d69e2e"],
        [0.666, "#c53030"],
        [1.000, "#c53030"],
    ]

    fig = _fig(title)
    fig.add_trace(go.Heatmap(
        z=combined.T.values.tolist(),
        x=[d.isoformat() for d in combined.index],
        y=list(combined.columns),
        colorscale=colorscale,
        zmin=0, zmax=3,
        showscale=False,
        xgap=1, ygap=2,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(autorange="reversed", tickfont=dict(size=TYPOGRAPHY.size_small)),
        xaxis=dict(showgrid=False),
    )
    return fig


# ── Recession Probability Charts ──────────────────────────────────────────────

def recession_gauge_chart(probabilities: dict) -> go.Figure:
    """
    Three side-by-side gauge indicators for 6M / 12M / 24M recession probability.
    Color steps: green < 20%, yellow 20–50%, red ≥ 50%.
    """
    from plotly.subplots import make_subplots
    labels = {6: "6-Month", 12: "12-Month", 24: "24-Month"}
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}] * 3],
        horizontal_spacing=0.04,
    )
    for col_i, h in enumerate([6, 12, 24], start=1):
        val = probabilities.get(h)
        display_val = round(val * 100, 1) if val is not None else 0
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=display_val,
            number={"suffix": "%", "font": {"size": 22}},
            title={"text": f"{labels[h]}", "font": {"size": 12}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickfont": {"size": 9}},
                "bar": {"color": (
                    C["red"] if display_val >= 50
                    else C["amber"] if display_val >= 20
                    else C["green"]
                ), "thickness": 0.65},
                "steps": [
                    {"range": [0,  20], "color": "rgba(39,103,73,0.10)"},
                    {"range": [20, 50], "color": "rgba(183,121,31,0.10)"},
                    {"range": [50, 100], "color": "rgba(155,44,44,0.10)"},
                ],
                "threshold": {
                    "line": {"color": C["slate"], "width": 2},
                    "thickness": 0.75,
                    "value": 35,
                },
            },
        ), row=1, col=col_i)

    fig.update_layout(
        font=dict(family=TYPOGRAPHY.font_family, size=TYPOGRAPHY.size_body, color=C["text"]),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        margin=dict(l=20, r=20, t=40, b=10),
        title=dict(text="Recession Probability by Horizon",
                   font=dict(size=TYPOGRAPHY.size_title, color=C["text"]), x=0, xanchor="left"),
    )
    return fig


def recession_probability_chart(
    rpe, dl,
    lookback_years: int = 20,
    with_bands: bool = False,
) -> go.Figure:
    """
    Rolling recession probability for all three horizons with NBER recession shading.
    Optional bootstrap confidence band for the 12M horizon.
    """
    title = "Recession Probability — All Horizons"
    hist = rpe.probability_history(lookback_years=lookback_years)
    if hist.empty:
        return _no_data(title)

    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    _colors = {"6m": C["teal"], "12m": C["blue"], "24m": C["amber"]}
    _labels = {"6m": "6-Month", "12m": "12-Month", "24m": "24-Month"}

    if with_bands and "12m" in hist.columns:
        try:
            lo, med, hi = rpe.confidence_band(horizon=12, n_bootstrap=40,
                                              lookback_years=lookback_years)
            if not lo.empty:
                fig.add_trace(go.Scatter(
                    x=pd.concat([lo, hi[::-1]]).index,
                    y=pd.concat([lo * 100, hi[::-1] * 100]).values,
                    fill="toself",
                    fillcolor="rgba(43,108,176,0.12)",
                    line=dict(width=0),
                    showlegend=False,
                    name="90% CI",
                    hoverinfo="skip",
                ))
        except Exception:
            pass

    for col_name, color in _colors.items():
        if col_name not in hist.columns:
            continue
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=(hist[col_name] * 100).values,
            mode="lines",
            name=_labels[col_name],
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"{_labels[col_name]}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_hline(y=35, line_color=C["slate"], line_width=1.2, line_dash="dash",
                  annotation_text="35% threshold",
                  annotation_font_size=TYPOGRAPHY.size_small,
                  annotation_position="right")
    fig.update_yaxes(ticksuffix="%", range=[0, 105])
    return fig


def signal_decomposition_chart(rpe, horizon: int = 12) -> go.Figure:
    """
    Horizontal bar chart: log-odds contribution of each feature to the current
    12-month recession probability.  Green = pulls probability lower; red = higher.
    """
    from .recession_probability import FEATURES
    title = f"Feature Contributions — {horizon}-Month Recession Probability"
    contribs = rpe.feature_contributions(horizon=horizon)
    if not contribs:
        return _no_data(title)

    _labels = {
        "yc_2y":       "Yield Curve (10Y-2Y)",
        "yc_3m":       "Yield Curve (10Y-3M)",
        "lei_yoy":     "LEI Year-over-Year",
        "claims_log":  "Initial Claims (log)",
        "ism":         "ISM New Orders",
        "hy_spread":   "HY Credit Spread",
        "u3_chg":      "Unemployment Change",
        "payroll_yoy": "Payrolls Year-over-Year",
    }
    keys   = [k for k in FEATURES if k in contribs]
    vals   = [contribs[k] for k in keys]
    labels = [_labels.get(k, k) for k in keys]
    colors = [C["red"] if v > 0 else C["teal"] for v in vals]

    fig = _fig(title)
    fig.add_trace(go.Bar(
        y=labels, x=vals,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=C["slate"], line_width=CHART.zero_line_width)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.add_annotation(
        text="← Reduces recession probability   Increases recession probability →",
        xref="paper", yref="paper", x=0.5, y=-0.12,
        showarrow=False, font=dict(size=9, color=C["muted"]),
        xanchor="center",
    )
    return fig


def multi_horizon_probability_chart(rpe, dl, lookback_years: int = 20) -> go.Figure:
    """
    Stacked area chart showing 6M, 12M, 24M recession probabilities with
    NBER recession shading.  Uses recession_probability_chart internally.
    """
    return recession_probability_chart(rpe, dl, lookback_years=lookback_years,
                                       with_bands=False)


# ── Leading Indicator & BCI Charts ────────────────────────────────────────────

_PHASE_FILL = {
    "Expansion":   "rgba(39,103,73,0.07)",
    "Slowdown":    "rgba(183,121,31,0.08)",
    "Contraction": "rgba(155,44,44,0.09)",
    "Recovery":    "rgba(43,108,176,0.07)",
}


def bci_chart(lie, dl, lookback_years: int = 15) -> go.Figure:
    """
    Composite Business Cycle Index over time.
    Phase background bands (Expansion/Slowdown/Contraction/Recovery) plus
    NBER recession shading and a neutral zero line.
    """
    bci    = lie.composite_bci(lookback_years=lookback_years)
    phases = lie.phase_history(lookback_years=lookback_years)
    title  = "Composite Business Cycle Index (BCI)"

    if bci.empty:
        return _no_data(title)

    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    # Phase background segments
    if not phases.empty:
        seg_start = phases.index[0]
        seg_phase = phases.iloc[0]
        for i in range(1, len(phases)):
            if phases.iloc[i] != seg_phase:
                fig.add_vrect(
                    x0=seg_start.isoformat(),
                    x1=phases.index[i].isoformat(),
                    fillcolor=_PHASE_FILL.get(seg_phase, "rgba(0,0,0,0.04)"),
                    line_width=0, layer="below",
                )
                seg_start = phases.index[i]
                seg_phase = phases.iloc[i]
        fig.add_vrect(
            x0=seg_start.isoformat(),
            x1=phases.index[-1].isoformat(),
            fillcolor=_PHASE_FILL.get(seg_phase, "rgba(0,0,0,0.04)"),
            line_width=0, layer="below",
        )

    fig.add_trace(go.Scatter(
        x=bci.index, y=bci.values,
        mode="lines", name="BCI (3M smoothed)",
        line=dict(color=C["blue"], width=CHART.line_width),
        hovertemplate="BCI: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["slate"], line_width=1.2, line_dash="dash",
                  annotation_text="Neutral", annotation_font_size=TYPOGRAPHY.size_small,
                  annotation_position="right")
    return fig


def bci_waterfall_chart(lie, dl) -> go.Figure:
    """
    Current BCI component contributions as a horizontal bar chart.
    Green bars = positive (pro-expansion) contribution; red = negative.
    """
    from .series_registry import REGISTRY
    contribs = lie.component_contributions()
    title    = "BCI Component Contributions (Current)"

    valid = {sid: v for sid, v in contribs.items() if v is not None}
    if not valid:
        return _no_data(title)

    labels      = [REGISTRY.get(sid, {}).get("short_name", sid) for sid in valid]
    values      = list(valid.values())
    bar_colors  = [C["green"] if v >= 0 else C["red"] for v in values]

    fig = _fig(title)
    fig.add_trace(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker_color=bar_colors,
        name="Contribution",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=C["slate"], line_width=CHART.zero_line_width)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def backtest_signal_chart(lie, dl, series_id: str, lookback_years: int = 20) -> go.Figure:
    """
    Historical signal vs. recession chart for a BCI component.

    Shows the indicator series with NBER recession shading, a threshold line at
    the 75th/25th percentile of the full history, and markers where the signal
    fired. Helps visually validate hit rates and false-positive rates.
    """
    from .series_registry import REGISTRY
    from .leading_indicators import BCI_COMPONENTS
    meta = _meta(series_id)
    cfg  = BCI_COMPONENTS.get(series_id, {})
    basis = cfg.get("basis", meta.get("risk_basis", "level"))
    higher_is_bad = meta.get("higher_is_bad", True) if not cfg.get("invert") else not meta.get("higher_is_bad", True)
    title = f"{meta.get('short_name', series_id)} — Historical Signal vs. NBER Recessions"

    df = dl.load(series_id)
    if df is None or df.empty:
        return _no_data(title)

    try:
        monthly = df.resample("ME").last().dropna()
    except ValueError:
        monthly = df.resample("M").last().dropna()
    col = monthly.columns[0]
    if basis == "yoy":
        raw = (monthly[col].pct_change(12) * 100).dropna()
    else:
        raw = monthly[col].dropna()

    plot = _trim(raw.to_frame(name="val"), lookback_years).iloc[:, 0]
    if plot.empty:
        return _no_data(title)

    # Threshold: 75th pctile for higher_is_bad, 25th otherwise
    q = 0.75 if higher_is_bad else 0.25
    threshold = float(raw.quantile(q))
    signal = (raw >= threshold) if higher_is_bad else (raw <= threshold)
    signal_fired = signal[signal].index
    signal_in_window = signal_fired[signal_fired >= plot.index[0]]

    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    line_color = C["red"] if higher_is_bad else C["teal"]
    fill_color = "rgba(155,44,44,0.08)" if higher_is_bad else "rgba(44,122,123,0.08)"

    fig.add_trace(go.Scatter(
        x=plot.index, y=plot.values,
        mode="lines", name=meta.get("short_name", series_id),
        line=dict(color=line_color, width=CHART.line_width),
        fill="tozeroy", fillcolor=fill_color,
        hovertemplate="%{y:.2f}<extra></extra>",
    ))

    # Threshold reference line
    fig.add_hline(
        y=threshold, line_color=C["amber"], line_width=1.2, line_dash="dash",
        annotation_text=f"Signal ({q*100:.0f}th pct.)",
        annotation_font_size=TYPOGRAPHY.size_small,
        annotation_position="right",
    )

    # Signal-fired markers
    if len(signal_in_window) > 0:
        valid = signal_in_window[signal_in_window.isin(plot.index)]
        if len(valid) > 0:
            fig.add_trace(go.Scatter(
                x=valid,
                y=[float(plot.loc[d]) for d in valid],
                mode="markers", name="Signal Fired",
                marker=dict(color=C["amber"], size=5, symbol="circle"),
                hovertemplate="Signal: %{x|%b %Y}<extra></extra>",
            ))

    return fig


def momentum_chart(dl, series_id: str, lookback_years: int = 5) -> go.Figure:
    """
    Two-panel trend & momentum chart.

    Top panel: series level with 3-month and 6-month rolling averages.
    Bottom panel: 3-month momentum (difference) as green/red bars.
    """
    meta   = _meta(series_id)
    basis  = meta.get("risk_basis", "level")
    title  = f"{meta.get('name', series_id)} — Trend & Momentum"

    df = dl.load(series_id)
    if df is None or df.empty:
        return _no_data(title)

    try:
        monthly = df.resample("ME").last().dropna()
    except ValueError:
        monthly = df.resample("M").last().dropna()

    col    = monthly.columns[0]
    raw    = (monthly[col].pct_change(12) * 100).dropna() if basis == "yoy" else monthly[col].dropna()
    series = _trim(raw.to_frame(name="val"), lookback_years).iloc[:, 0]

    ma3 = series.rolling(3, min_periods=1).mean()
    ma6 = series.rolling(6, min_periods=1).mean()
    mom = series.diff(3).dropna()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.08,
        subplot_titles=(meta.get("short_name", series_id), "3-Month Momentum"),
    )

    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        name=meta.get("short_name", series_id),
        line=dict(color=C["slate"], width=0.8), opacity=0.45,
        hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ma3.index, y=ma3.values, mode="lines", name="3M Avg",
        line=dict(color=C["blue"], width=CHART.line_width),
        hovertemplate="3M: %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ma6.index, y=ma6.values, mode="lines", name="6M Avg",
        line=dict(color=C["teal"], width=CHART.line_width, dash="dash"),
        hovertemplate="6M: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    mom_colors = [C["green"] if v >= 0 else C["red"] for v in mom.values]
    fig.add_trace(go.Bar(
        x=mom.index, y=mom.values,
        marker_color=mom_colors, name="3M Momentum",
        hovertemplate="Momentum: %{y:+.2f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=TYPOGRAPHY.size_title, color=C["text"]),
            x=0, xanchor="left",
        ),
        font=dict(family=TYPOGRAPHY.font_family, size=TYPOGRAPHY.size_body, color=C["text"]),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        margin=dict(l=CHART.margin_l, r=CHART.margin_r,
                    t=CHART.margin_t, b=CHART.margin_b),
        hovermode="x unified",
        legend=dict(orientation="h", y=CHART.legend_y, x=CHART.legend_x,
                    font=dict(size=TYPOGRAPHY.size_small)),
    )
    for row_i in [1, 2]:
        fig.update_xaxes(
            showgrid=True, gridcolor=C["grid"], linecolor=C["grid"],
            tickfont=dict(size=TYPOGRAPHY.size_small), zeroline=False,
            row=row_i, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=C["grid"], linecolor=C["grid"],
            tickfont=dict(size=TYPOGRAPHY.size_small), zeroline=False,
            row=row_i, col=1,
        )

    return fig


# ── Labor Analytics Charts ────────────────────────────────────────────────────

def jolts_chart(dl, lookback_years: int = 10) -> go.Figure:
    """
    Three-panel JOLTS chart: job openings (thousands), quits rate, layoffs rate.
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.40, 0.30, 0.30],
        vertical_spacing=0.05,
    )
    _add_recession_bands(fig, dl, lookback_years)

    openings = _resample_monthly(_or_empty(dl.load("JTSJOL")))
    quits    = _resample_monthly(_or_empty(dl.load("JTSQUR")))
    layoffs  = _resample_monthly(_or_empty(dl.load("JTSLDR")))

    def _plot(df, row, color, name, fmt=",.0f"):
        if df.empty:
            return
        s = _trim(df, lookback_years).iloc[:, 0].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"%{{y:{fmt}}}<extra></extra>",
        ), row=row, col=1)

    _plot(openings, 1, C["blue"],  "Job Openings (k)", ".0f")
    _plot(quits,   2, C["teal"],  "Quits Rate %",     ".2f")
    _plot(layoffs, 3, C["amber"], "Layoffs Rate %",   ".2f")

    fig.update_layout(
        title=dict(text="JOLTS: Openings / Quits / Layoffs",
                   font=dict(size=TYPOGRAPHY.size_title, color=C["text"]),
                   x=0, xanchor="left"),
        **{k: v for k, v in _BASE.items() if k not in ("title", "xaxis", "yaxis", "legend")},
        showlegend=True,
        legend=dict(orientation="h", y=CHART.legend_y, x=CHART.legend_x,
                    font=dict(size=TYPOGRAPHY.size_small)),
    )
    for r in [1, 2, 3]:
        fig.update_xaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=TYPOGRAPHY.size_small), row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=TYPOGRAPHY.size_small), row=r, col=1)
    return fig


def wage_productivity_chart(dl, lookback_years: int = 10) -> go.Figure:
    """
    Overlay: Avg Hourly Earnings YoY%, Unit Labor Costs YoY%, Nonfarm Productivity YoY%.
    """
    title = "Wages vs. Productivity vs. Unit Labor Costs — YoY %"
    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    def _yoy_series(sid):
        df = dl.load(sid)
        if df is None or df.empty:
            return None
        s = _resample_monthly(df).iloc[:, 0].dropna()
        if len(s) < 13:
            return None
        s = (s.pct_change(12) * 100).dropna()
        return _trim(s.to_frame(), lookback_years).iloc[:, 0]

    ahe  = _yoy_series("CES0500000003")
    ulc  = _yoy_series("ULCNFB")
    prod = _yoy_series("OPHNFB")

    if ahe is not None:
        fig.add_trace(go.Scatter(x=ahe.index, y=ahe.values, mode="lines",
            name="Avg Hourly Earnings YoY",
            line=dict(color=C["blue"], width=CHART.line_width),
            hovertemplate="AHE: %{y:.1f}%<extra></extra>"))
    if ulc is not None:
        fig.add_trace(go.Scatter(x=ulc.index, y=ulc.values, mode="lines",
            name="Unit Labor Costs YoY",
            line=dict(color=C["red"], width=CHART.line_width, dash="dot"),
            hovertemplate="ULC: %{y:.1f}%<extra></extra>"))
    if prod is not None:
        fig.add_trace(go.Scatter(x=prod.index, y=prod.values, mode="lines",
            name="Productivity YoY",
            line=dict(color=C["teal"], width=CHART.line_width, dash="dash"),
            hovertemplate="Productivity: %{y:.1f}%<extra></extra>"))

    fig.add_hline(y=0, line_color=C["grid"], line_width=1)
    fig.add_hline(y=2, line_color=C["green"], line_width=1, line_dash="dot",
                  annotation_text="2% (Fed target zone)", annotation_font_size=TYPOGRAPHY.size_small)
    return fig


def u3_u6_chart(dl, lookback_years: int = 15) -> go.Figure:
    """
    U-3 vs U-6 unemployment with the gap (U6 - U3) as filled area below.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.60, 0.40], vertical_spacing=0.05)
    _add_recession_bands(fig, dl, lookback_years)

    def _s(sid):
        df = dl.load(sid)
        if df is None or df.empty:
            return None
        s = _resample_monthly(df).iloc[:, 0].dropna()
        return _trim(s.to_frame(), lookback_years).iloc[:, 0]

    u3 = _s("UNRATE")
    u6 = _s("U6RATE")

    if u3 is not None:
        fig.add_trace(go.Scatter(x=u3.index, y=u3.values, mode="lines",
            name="U-3 Unemployment",
            line=dict(color=C["slate"], width=CHART.line_width),
            hovertemplate="U3: %{y:.1f}%<extra></extra>"), row=1, col=1)
    if u6 is not None:
        fig.add_trace(go.Scatter(x=u6.index, y=u6.values, mode="lines",
            name="U-6 Unemployment",
            line=dict(color=C["red"], width=CHART.line_width, dash="dash"),
            hovertemplate="U6: %{y:.1f}%<extra></extra>"), row=1, col=1)

    if u3 is not None and u6 is not None:
        shared = u3.index.intersection(u6.index)
        gap = (u6.reindex(shared) - u3.reindex(shared)).dropna()
        fig.add_trace(go.Scatter(
            x=gap.index, y=gap.values, mode="lines",
            fill="tozeroy", fillcolor="rgba(155,44,44,0.10)",
            name="U6 - U3 Gap (Labor Slack)",
            line=dict(color=C["amber"], width=CHART.line_width),
            hovertemplate="Gap: %{y:.1f}pp<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text="Unemployment: U-3 vs U-6 (Top) | Slack Gap (Bottom)",
                   font=dict(size=TYPOGRAPHY.size_title, color=C["text"]),
                   x=0, xanchor="left"),
        **{k: v for k, v in _BASE.items() if k not in ("title", "xaxis", "yaxis")},
    )
    for r in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=TYPOGRAPHY.size_small), row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=TYPOGRAPHY.size_small), row=r, col=1)
    return fig


def labor_deterioration_chart(lae, lookback_years: int = 10) -> go.Figure:
    """
    Gauge showing current labor deterioration index (0–100) plus
    component bar breakdown.
    """
    from .series_registry import REGISTRY
    ldi = lae.labor_deterioration_index()
    score = ldi.get("score")
    components = ldi.get("components", {})

    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.45, 0.55],
                        specs=[[{"type": "indicator"}, {"type": "bar"}]])

    # Gauge
    gauge_color = C["green"] if (score or 0) < 35 else (C["amber"] if (score or 0) < 65 else C["red"])
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=score or 0,
        title={"text": "Labor Deterioration Index", "font": {"size": 13, "color": C["text"]}},
        number={"font": {"size": 28, "color": gauge_color}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": C["muted"]},
            "bar": {"color": gauge_color, "thickness": 0.25},
            "bgcolor": C["bg"],
            "steps": [
                {"range": [0, 35],   "color": "rgba(39,103,73,0.15)"},
                {"range": [35, 65],  "color": "rgba(183,121,31,0.15)"},
                {"range": [65, 100], "color": "rgba(155,44,44,0.15)"},
            ],
            "threshold": {"line": {"color": C["red"], "width": 2}, "value": 65},
        },
    ), row=1, col=1)

    # Component bars
    names, vals, colors = [], [], []
    for sid, comp in components.items():
        meta = REGISTRY.get(sid, {})
        names.append(meta.get("short_name", sid))
        v = comp.get("score") or 0
        vals.append(v)
        colors.append(C["green"] if v < 35 else (C["amber"] if v < 65 else C["red"]))

    if names:
        fig.add_trace(go.Bar(
            x=vals, y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}" for v in vals],
            textposition="inside",
            hovertemplate="%{y}: %{x:.0f}/100<extra></extra>",
            name="Components",
        ), row=1, col=2)
        fig.update_xaxes(range=[0, 100], row=1, col=2,
                         showgrid=True, gridcolor=C["grid"],
                         tickfont=dict(size=TYPOGRAPHY.size_small))
        fig.update_yaxes(row=1, col=2, tickfont=dict(size=TYPOGRAPHY.size_small))

    fig.update_layout(
        title=dict(text="Labor Deterioration Index — Current Reading & Components",
                   font=dict(size=TYPOGRAPHY.size_title, color=C["text"]),
                   x=0, xanchor="left"),
        **{k: v for k, v in _BASE.items() if k not in ("title", "xaxis", "yaxis")},
        showlegend=False,
    )
    return fig


def claims_dashboard_chart(dl, lookback_years: int = 8) -> go.Figure:
    """
    Side-by-side: Initial Claims (4-week MA) + Continued Claims.
    """
    fig = make_subplots(rows=1, cols=2, shared_xaxes=False,
                        subplot_titles=["Initial Claims (4-Week MA)", "Continued Claims"])

    def _plot_claims(sid, row, color, label):
        df = dl.load(sid)
        if df is None or df.empty:
            return
        s = _resample_monthly(df).iloc[:, 0].dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        ma = s.rolling(4).mean()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=f"{label} (raw)",
            line=dict(color=color, width=1, dash="dot"), opacity=0.4,
            hovertemplate=f"%{{y:,.0f}}<extra></extra>",
        ), row=1, col=row)
        fig.add_trace(go.Scatter(
            x=ma.index, y=ma.values, mode="lines", name=f"{label} MA4",
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"MA4: %{{y:,.0f}}<extra></extra>",
        ), row=1, col=row)

    _plot_claims("ICSA", 1, C["amber"], "Initial Claims")
    _plot_claims("CCSA", 2, C["red"],   "Continued Claims")
    _add_recession_bands(fig, dl, lookback_years)

    fig.update_layout(
        title=dict(text="Jobless Claims — Initial & Continued",
                   font=dict(size=TYPOGRAPHY.size_title, color=C["text"]),
                   x=0, xanchor="left"),
        **{k: v for k, v in _BASE.items() if k not in ("title", "xaxis", "yaxis")},
    )
    for c in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=TYPOGRAPHY.size_small), row=1, col=c)
        fig.update_yaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=TYPOGRAPHY.size_small), row=1, col=c)
    return fig


# ── Inflation Analytics Charts ────────────────────────────────────────────────

def inflation_multi_chart(dl, lookback_years: int = 10) -> go.Figure:
    """
    Four inflation measures on one chart:
    Core CPI, Core PCE, Median CPI, Trimmed Mean PCE.
    """
    title = "Inflation Gauges — Core CPI / PCE, Median CPI, Trimmed Mean PCE"
    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    # Pre-computed series (already YoY %) vs raw index series
    _pre = {"MEDCPIM158SFRBCLE", "PCETRIM12M159SFRBDAL", "CORESTICKM159SFRBATL"}

    specs = [
        ("CPILFESL",              C["blue"],  "Core CPI",          False),
        ("PCEPILFE",              C["teal"],  "Core PCE",          False),
        ("MEDCPIM158SFRBCLE",     C["amber"], "Median CPI",        True),
        ("PCETRIM12M159SFRBDAL",  C["slate"], "Trimmed Mean PCE",  True),
    ]
    for sid, color, name, pre in specs:
        df = dl.load(sid)
        if df is None or df.empty:
            continue
        s = _resample_monthly(df).iloc[:, 0].dropna()
        if not pre:
            if len(s) < 13:
                continue
            s = (s.pct_change(12) * 100).dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=2.0, line_color=C["green"], line_width=1, line_dash="dot",
                  annotation_text="2% Target", annotation_font_size=TYPOGRAPHY.size_small)
    return fig


def inflation_expectations_chart(dl, lookback_years: int = 10) -> go.Figure:
    """
    Inflation expectations: Michigan 1-yr survey + 5-yr + 10-yr breakevens.
    """
    title = "Inflation Expectations — Survey & Market Breakevens"
    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    specs = [
        ("MICH",   C["amber"], "Michigan 1-Yr Survey", True),
        ("T5YIE",  C["blue"],  "5-Yr Breakeven",       True),
        ("T10YIE", C["teal"],  "10-Yr Breakeven",      True),
    ]
    for sid, color, name, pre in specs:
        df = dl.load(sid)
        if df is None or df.empty:
            continue
        s = _resample_monthly(df).iloc[:, 0].dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=2.5, line_color=C["amber"], line_width=1, line_dash="dot",
                  annotation_text="Alert threshold", annotation_font_size=TYPOGRAPHY.size_small)
    return fig


def shelter_decomposition_chart(dl, lookback_years: int = 8) -> go.Figure:
    """
    Core CPI, Supercore CPI, and OER — shows shelter's drag on disinflation.
    """
    title = "Inflation Decomposition: Core CPI vs. Supercore vs. OER"
    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    specs = [
        ("CPILFESL",     C["blue"],  "Core CPI (ex F&E)", False),
        ("CPILFENS",     C["teal"],  "Supercore (ex Shelter)", False),
        ("CUSR0000SEHC", C["red"],   "Owners' Equiv. Rent", False),
    ]
    for sid, color, name, pre in specs:
        df = dl.load(sid)
        if df is None or df.empty:
            continue
        s = _resample_monthly(df).iloc[:, 0].dropna()
        if len(s) < 13:
            continue
        s = (s.pct_change(12) * 100).dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        dash = "dash" if "Supercore" in name else "solid"
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width, dash=dash),
            hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=2.0, line_color=C["green"], line_width=1, line_dash="dot",
                  annotation_text="2% Target", annotation_font_size=TYPOGRAPHY.size_small)
    return fig


def sticky_flexible_chart(dl, lookback_years: int = 10) -> go.Figure:
    """
    Sticky Price CPI vs Core CPI — persistent vs. transitory components.
    """
    title = "Sticky Price CPI vs. Core CPI — Persistent vs. Transitory"
    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    # Sticky CPI is pre-computed 12M%
    for sid, color, name, pre in [
        ("CORESTICKM159SFRBATL", C["red"],  "Sticky Price CPI", True),
        ("CPILFESL",             C["blue"], "Core CPI",         False),
    ]:
        df = dl.load(sid)
        if df is None or df.empty:
            continue
        s = _resample_monthly(df).iloc[:, 0].dropna()
        if not pre:
            if len(s) < 13:
                continue
            s = (s.pct_change(12) * 100).dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width,
                      dash="dash" if "Core" in name else "solid"),
            hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=2.0, line_color=C["green"], line_width=1, line_dash="dot",
                  annotation_text="2% Target", annotation_font_size=TYPOGRAPHY.size_small)
    return fig


# ── Financial Conditions Charts ───────────────────────────────────────────────

def fci_composite_chart(dl, lookback_years: int = 10) -> go.Figure:
    """
    Composite Financial Conditions chart: NFCI + STLFSI4 on same axis.
    Positive = tighter conditions; negative = looser.
    """
    title = "Financial Conditions — NFCI & St. Louis FSI"
    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    specs = [
        ("NFCI",   C["blue"],  "Chicago Fed NFCI"),
        ("STLFSI4",C["red"],   "St. Louis Stress Index"),
    ]
    for sid, color, name in specs:
        df = dl.load(sid)
        if df is None or df.empty:
            continue
        s = _resample_monthly(df).iloc[:, 0].dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>",
        ))

    # shade above 0 (tightening stress)
    fig.add_hline(y=0, line_color=C["muted"], line_width=1.5)
    fig.add_hrect(y0=0, y1=5, fillcolor="rgba(155,44,44,0.05)", layer="below", line_width=0)
    return fig


def hy_spread_fci_chart(dl, lookback_years: int = 10) -> go.Figure:
    """
    HY spread + IG spread + lending standards on one chart.
    Shows the financial tightening transmission channel.
    """
    title = "Credit Spreads & Lending Standards — Financial Tightening Channel"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.60, 0.40], vertical_spacing=0.05)
    _add_recession_bands(fig, dl, lookback_years)

    for sid, color, name in [
        ("BAMLH0A0HYM2", C["red"],  "HY OAS Spread (%)"),
        ("BAMLC0A0CM",   C["blue"], "IG OAS Spread (%)"),
    ]:
        df = dl.load(sid)
        if df is None or df.empty:
            continue
        s = _resample_monthly(df).iloc[:, 0].dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>",
        ), row=1, col=1)

    df_lend = dl.load("DRTSCILM")
    if df_lend is not None and not df_lend.empty:
        s = _resample_monthly(df_lend).iloc[:, 0].dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        bar_colors = [C["red"] if v > 0 else C["teal"] for v in s.values]
        fig.add_trace(go.Bar(
            x=s.index, y=s.values,
            marker_color=bar_colors, name="C&I Lending Standards (Net Tightening %)",
            hovertemplate="Lending Stds: %{y:.1f}%<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=TYPOGRAPHY.size_title, color=C["text"]),
                   x=0, xanchor="left"),
        **{k: v for k, v in _BASE.items() if k not in ("title", "xaxis", "yaxis")},
    )
    for r in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=TYPOGRAPHY.size_small), row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=TYPOGRAPHY.size_small), row=r, col=1)
    return fig


# ── Banking & Credit Stress Charts ───────────────────────────────────────────

def delinquency_chart(dl, lookback_years: int = 15) -> go.Figure:
    """
    Loan delinquency rates: all loans, CRE, and residential mortgages.
    """
    title = "Bank Loan Delinquency Rates — All Loans, CRE, Mortgage"
    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    specs = [
        ("DRALACBN",   C["blue"],  "All Loans"),
        ("DRCLACBS",   C["red"],   "CRE Loans"),
        ("DRSFRMACBS", C["amber"], "Residential Mortgages"),
    ]
    for sid, color, name in specs:
        df = dl.load(sid)
        if df is None or df.empty:
            continue
        s = _resample_monthly(df).iloc[:, 0].dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=2.5, line_color=C["amber"], line_width=1, line_dash="dot",
                  annotation_text="Alert (2.5%)", annotation_font_size=TYPOGRAPHY.size_small,
                  annotation_position="right")
    return fig


def bank_deposits_chart(dl, lookback_years: int = 10) -> go.Figure:
    """
    Bank deposits level (left) + YoY growth rate (right sub-panel).
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.05,
                        subplot_titles=["Deposits Level ($B)", "YoY Growth %"])
    _add_recession_bands(fig, dl, lookback_years)

    df = dl.load("DPSACBM027SBOG")
    if df is not None and not df.empty:
        s = _resample_monthly(df).iloc[:, 0].dropna()
        s_trim = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        fig.add_trace(go.Scatter(
            x=s_trim.index, y=s_trim.values, mode="lines",
            fill="tozeroy", fillcolor="rgba(43,108,176,0.10)",
            name="Bank Deposits ($B)",
            line=dict(color=C["blue"], width=CHART.line_width),
            hovertemplate="Deposits: $%{y:,.0f}B<extra></extra>",
        ), row=1, col=1)

        if len(s) >= 13:
            yoy = (s.pct_change(12) * 100).dropna()
            yoy_trim = _trim(yoy.to_frame(), lookback_years).iloc[:, 0]
            bar_colors = [C["teal"] if v >= 0 else C["red"] for v in yoy_trim.values]
            fig.add_trace(go.Bar(
                x=yoy_trim.index, y=yoy_trim.values,
                marker_color=bar_colors, name="YoY Growth %",
                hovertemplate="YoY: %{y:.1f}%<extra></extra>",
            ), row=2, col=1)

    fig.update_layout(
        title=dict(text="Bank Deposits — All Commercial Banks",
                   font=dict(size=TYPOGRAPHY.size_title, color=C["text"]),
                   x=0, xanchor="left"),
        **{k: v for k, v in _BASE.items() if k not in ("title", "xaxis", "yaxis")},
    )
    for r in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=TYPOGRAPHY.size_small), row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=TYPOGRAPHY.size_small), row=r, col=1)
    return fig


# ── Global Macro Charts ────────────────────────────────────────────────────


def central_bank_rates_chart(dl, lookback_years: int = 25) -> go.Figure:
    """Fed Funds Rate, ECB policy rate, and BOJ policy rate on one chart."""
    title = "Central Bank Policy Rates — Fed vs ECB vs BOJ"
    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    specs = [
        ("FEDFUNDS",        C["blue"],  "Fed Funds Rate"),
        ("IR3TIB01EZM156N", C["red"],   "ECB Policy Rate"),
        ("IRSTCB01JPM156N", C["teal"],  "BOJ Policy Rate"),
    ]
    for sid, color, name in specs:
        df = dl.load(sid)
        if df is None or df.empty:
            continue
        s = _resample_monthly(df).iloc[:, 0].dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width,
                  line_dash="dot")
    fig.update_layout(yaxis=dict(ticksuffix="%"))
    return fig


def commodity_chart(dl, lookback_years: int = 15) -> go.Figure:
    """
    Brent crude, gold, and PPI-All-Commodities normalized to 100
    at the start of the lookback period for cross-asset comparison.
    """
    title = "Global Commodity Complex — Indexed to 100 at Period Start"
    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    specs = [
        ("DCOILBRENTEU",      C["amber"], "Brent Crude ($/bbl)"),
        ("GOLDPMGBD228NLBMA", C["slate"], "Gold ($/troy oz)"),
        ("PPIACO",            C["teal"],  "PPI All Commodities"),
    ]
    for sid, color, name in specs:
        df = dl.load(sid)
        if df is None or df.empty:
            continue
        try:
            s = df.resample("ME").mean().iloc[:, 0].dropna()
        except ValueError:
            s = df.resample("M").mean().iloc[:, 0].dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0].dropna()
        if s.empty or s.iloc[0] == 0:
            continue
        s_idx = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=s_idx.index, y=s_idx.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>",
        ))

    fig.add_hline(y=100, line_color=C["slate"], line_width=CHART.zero_line_width,
                  line_dash="dot", annotation_text="Base = 100",
                  annotation_font_size=TYPOGRAPHY.size_small)
    return fig


def fx_chart(dl, lookback_years: int = 15) -> go.Figure:
    """
    Broad USD index plus EUR/USD, JPY/USD, CNY/USD normalized to 100.
    EUR/USD is inverted so all lines rise with USD strength.
    """
    title = "FX Conditions — USD Broad Index & Major Pairs (Indexed to 100)"
    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    specs = [
        ("DTWEXBGS", C["blue"],  "USD Broad Index"),
        ("DEXUSEU",  C["teal"],  "EUR/USD (inverted)"),
        ("DEXJPUS",  C["amber"], "JPY/USD"),
        ("DEXCHUS",  C["red"],   "CNY/USD"),
    ]
    for sid, color, name in specs:
        df = dl.load(sid)
        if df is None or df.empty:
            continue
        try:
            s = df.resample("ME").mean().iloc[:, 0].dropna()
        except ValueError:
            s = df.resample("M").mean().iloc[:, 0].dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0].dropna()
        if s.empty or s.iloc[0] == 0:
            continue
        if sid == "DEXUSEU":
            s = 1 / s  # invert so rising = stronger USD
        s_idx = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=s_idx.index, y=s_idx.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>",
        ))

    fig.add_hline(y=100, line_color=C["slate"], line_width=CHART.zero_line_width,
                  line_dash="dot", annotation_text="Base = 100",
                  annotation_font_size=TYPOGRAPHY.size_small)
    fig.add_annotation(
        text="Rising = stronger USD",
        xref="paper", yref="paper", x=0.01, y=0.96,
        showarrow=False, font=dict(size=TYPOGRAPHY.size_small, color=C["slate"]),
    )
    return fig


# ── Macro Regime Charts ────────────────────────────────────────────────────


def regime_timeline_chart(rge, dl, lookback_years: int = 20) -> go.Figure:
    """
    Color-coded timeline of macro regime periods with NFCI overlay.
    Regime periods are rendered as translucent vrects; NFCI is the overlay.
    """
    from .regime_engine import REGIMES

    title = "Macro Regime History"
    fig = _fig(title)

    regime_hist = rge.regime_history(lookback_years)

    if not regime_hist.empty:
        spans, cur_label, cur_start = [], None, None
        for dt, label in regime_hist.items():
            if label != cur_label:
                if cur_label is not None:
                    spans.append((cur_start, dt, cur_label))
                cur_label = label
                cur_start = dt
        if cur_label is not None and cur_start is not None:
            spans.append((cur_start, regime_hist.index[-1], cur_label))

        for start, end, label in spans:
            hex_c = REGIMES.get(label, REGIMES["Uncertain"])["color"]
            r, g, b = (int(hex_c.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            ann_text = label if (end - start).days > 210 else ""
            fig.add_vrect(
                x0=start.isoformat(), x1=end.isoformat(),
                fillcolor=f"rgba({r},{g},{b},0.18)", line_width=0, layer="below",
                annotation_text=ann_text,
                annotation_font_size=TYPOGRAPHY.size_small - 1,
                annotation_position="top left",
            )

    nfci_df = dl.load("NFCI")
    if nfci_df is not None and not nfci_df.empty:
        s = _resample_monthly(nfci_df).iloc[:, 0].dropna()
        s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            name="NFCI (Financial Conditions)",
            line=dict(color=C["blue"], width=CHART.line_width),
            hovertemplate="NFCI: %{y:.2f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width,
                      line_dash="dot")

    return fig


def regime_scores_chart(rge) -> go.Figure:
    """
    Horizontal bar chart of the four regime dimension scores (0–1 scale).
    Green < 0.35, amber 0.35–0.65, red > 0.65.
    """
    dims   = rge.dimension_scores()
    labels = ["Growth", "Inflation", "Financial Conditions", "Credit"]
    keys   = ["growth", "inflation", "financial", "credit"]
    scores = [dims[k]["score"] for k in keys]
    subs   = [dims[k]["label"] for k in keys]

    colors = [C["green"] if s < 0.35 else C["amber"] if s < 0.65 else C["red"]
              for s in scores]

    fig = _fig("Regime Dimension Scores — 0 = Healthy, 1 = Stressed")
    fig.add_trace(go.Bar(
        x=scores,
        y=[f"{lbl}<br><i>{sub}</i>" for lbl, sub in zip(labels, subs)],
        orientation="h",
        marker_color=colors,
        text=[f"{s:.2f}" for s in scores],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0.35, line_color=C["amber"], line_width=1, line_dash="dot",
                  annotation_text="Elevated",
                  annotation_font_size=TYPOGRAPHY.size_small,
                  annotation_position="top right")
    fig.add_vline(x=0.65, line_color=C["red"], line_width=1, line_dash="dot",
                  annotation_text="Stressed",
                  annotation_font_size=TYPOGRAPHY.size_small,
                  annotation_position="top right")
    fig.update_layout(xaxis=dict(range=[0, 1.15]), bargap=0.35)
    return fig


# ── Structural Macro Charts ────────────────────────────────────────────────


def output_gap_chart(sme) -> go.Figure:
    """
    Two-panel: real GDP vs potential GDP (rebased to 100), then gap as %.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.45], vertical_spacing=0.06,
        subplot_titles=["Real vs Potential GDP (Indexed to 100)",
                        "Output Gap (% of Potential GDP)"],
    )

    hist = sme.output_gap_history(lookback_years=25)
    gdp  = hist.get("GDPC1")
    pot  = hist.get("GDPPOT")
    gap  = hist.get("OUTPUT_GAP")

    if gdp is not None and not gdp.empty:
        g_idx = gdp / gdp.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=g_idx.index, y=g_idx.values, mode="lines",
            name="Real GDP", line=dict(color=C["blue"], width=CHART.line_width),
            hovertemplate="Real GDP: %{y:.1f}<extra></extra>",
        ), row=1, col=1)

    if pot is not None and not pot.empty:
        p_idx = pot / pot.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=p_idx.index, y=p_idx.values, mode="lines",
            name="Potential GDP (CBO)", line=dict(color=C["amber"], width=CHART.line_width, dash="dash"),
            hovertemplate="Potential: %{y:.1f}<extra></extra>",
        ), row=1, col=1)

    if gap is not None and not gap.empty:
        gap_colors = [C["red"] if v > 1.5 else C["teal"] if v >= 0 else C["blue"]
                      for v in gap.values]
        fig.add_trace(go.Bar(
            x=gap.index, y=gap.values, name="Output Gap %",
            marker_color=gap_colors,
            hovertemplate="Gap: %{y:.2f}%<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width,
                      row=2, col=1)

    fig.update_layout(
        title=dict(text="Output Gap — Real GDP vs CBO Potential",
                   font=dict(size=TYPOGRAPHY.size_title, color=C["text"]),
                   x=0, xanchor="left"),
        **{k: v for k, v in _BASE.items() if k not in ("title", "xaxis", "yaxis")},
    )
    for r in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor=C["grid"],
                         tickfont=dict(size=TYPOGRAPHY.size_small), row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=C["grid"],
                         tickfont=dict(size=TYPOGRAPHY.size_small), row=r, col=1)
    return fig


def productivity_chart(sme, lookback_years: int = 20) -> go.Figure:
    """Nonfarm productivity YoY (%) with rolling 2-year average."""
    title = "Nonfarm Business Sector Productivity — YoY % Change"
    yoy   = sme.productivity_history(lookback_years=lookback_years)

    if yoy is None or yoy.empty:
        return _no_data(title)

    fig = _fig(title)
    colors = [C["teal"] if v >= 0 else C["red"] for v in yoy.values]
    fig.add_trace(go.Bar(
        x=yoy.index, y=yoy.values, name="Productivity YoY %",
        marker_color=colors,
        hovertemplate="Productivity: %{y:.2f}%<extra></extra>",
    ))
    roll = yoy.rolling(8, min_periods=4).mean()
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll.values, mode="lines",
        name="8Q Rolling Avg",
        line=dict(color=C["blue"], width=2, dash="dash"),
        hovertemplate="8Q Avg: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width)
    fig.update_layout(yaxis=dict(ticksuffix="%"))
    return fig


def real_rates_chart(sme, dl, lookback_years: int = 20) -> go.Figure:
    """
    10Y TIPS yield and real Fed Funds rate.
    Note: 10Y TIPS yield ≠ r* — it includes a real term premium (~0.5–1.5%)
    above the neutral rate. Chart shows long-run real market rates and policy stance.
    """
    title = "Real Interest Rates — TIPS 10Y & Real Fed Funds Rate"
    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    hist = sme.real_rates_history(lookback_years=lookback_years)
    specs = [
        ("DFII10",  C["blue"],  "10Y TIPS Yield (long-run real rate)"),
        ("REAL_FF", C["amber"], "Real Fed Funds Rate"),
    ]
    for key, color, name in specs:
        s = hist.get(key)
        if s is None or s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=CHART.line_width),
            hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width,
                  annotation_text="Zero real rate",
                  annotation_font_size=TYPOGRAPHY.size_small,
                  annotation_position="right")
    fig.update_layout(yaxis=dict(ticksuffix="%"))
    return fig


# ── Fiscal Analytics Charts ────────────────────────────────────────────────


def debt_service_chart(fae, lookback_years: int = 30) -> go.Figure:
    """
    Interest/receipts and interest/GDP on dual-panel chart.
    Interest/receipts is the key operational sustainability metric.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.45], vertical_spacing=0.06,
        subplot_titles=["Interest / Federal Receipts (%)", "Interest / GDP (%)"],
    )

    hist = fae.debt_service_history(lookback_years=lookback_years)
    ir   = hist.get("INT_RECEIPTS")
    ig   = hist.get("INT_GDP")

    if ir is not None and not ir.empty:
        fig.add_trace(go.Scatter(
            x=ir.index, y=ir.values, mode="lines", name="Interest / Receipts %",
            fill="tozeroy", fillcolor="rgba(155,44,44,0.09)",
            line=dict(color=C["red"], width=CHART.line_width),
            hovertemplate="Int/Receipts: %{y:.1f}%<extra></extra>",
        ), row=1, col=1)
        for thresh, lbl in [(15, "Elevated (15%)"), (25, "Stressed (25%)")]:
            fig.add_hline(y=thresh, line_color=C["amber"] if thresh == 15 else C["red"],
                          line_width=1, line_dash="dot",
                          annotation_text=lbl, annotation_font_size=TYPOGRAPHY.size_small,
                          row=1, col=1)

    if ig is not None and not ig.empty:
        fig.add_trace(go.Scatter(
            x=ig.index, y=ig.values, mode="lines", name="Interest / GDP %",
            fill="tozeroy", fillcolor="rgba(183,121,31,0.09)",
            line=dict(color=C["amber"], width=CHART.line_width),
            hovertemplate="Int/GDP: %{y:.2f}%<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text="Federal Debt Service Burden",
                   font=dict(size=TYPOGRAPHY.size_title, color=C["text"]),
                   x=0, xanchor="left"),
        **{k: v for k, v in _BASE.items() if k not in ("title", "xaxis", "yaxis")},
    )
    for r in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor=C["grid"],
                         tickfont=dict(size=TYPOGRAPHY.size_small), row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=C["grid"],
                         tickfont=dict(size=TYPOGRAPHY.size_small),
                         ticksuffix="%", row=r, col=1)
    return fig


def primary_balance_chart(fae, lookback_years: int = 40) -> go.Figure:
    """
    Annual total deficit vs primary balance as % of GDP.
    Gap between lines = interest payments as % of GDP.
    """
    title = "Total vs Primary Deficit/Surplus — % of GDP (Annual FY)"
    fig = _fig(title)

    hist  = fae.primary_balance_history(lookback_years=lookback_years)
    total = hist.get("FYFSGDA188S")
    prim  = hist.get("PRIMARY_PCT_GDP")

    if total is not None and not total.empty:
        fig.add_trace(go.Bar(
            x=total.index, y=total.values, name="Total Deficit/Surplus % GDP",
            marker_color=[C["red"] if v < 0 else C["teal"] for v in total.values],
            hovertemplate="Total: %{y:.1f}% GDP<extra></extra>", opacity=0.70,
        ))

    if prim is not None and not prim.empty:
        fig.add_trace(go.Scatter(
            x=prim.index, y=prim.values, mode="lines+markers",
            name="Primary Balance % GDP",
            line=dict(color=C["blue"], width=2),
            marker=dict(size=5),
            hovertemplate="Primary: %{y:.1f}% GDP<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width,
                  annotation_text="Balance",
                  annotation_font_size=TYPOGRAPHY.size_small)
    fig.update_layout(yaxis=dict(ticksuffix="%"), bargap=0.2)
    return fig


def debt_trajectory_chart(dl, lookback_years: int = 40) -> go.Figure:
    """Federal Debt % of GDP with long-run trend overlay."""
    title = "Federal Debt / GDP — Long-Run Trajectory"
    df = dl.load("GFDEGDQ188S")
    if df is None or df.empty:
        return _no_data(title)

    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    s = _resample_monthly(df).iloc[:, 0].dropna()
    s = _trim(s.to_frame(), lookback_years).iloc[:, 0]
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines", name="Debt / GDP",
        fill="tozeroy", fillcolor="rgba(183,121,31,0.09)",
        line=dict(color=C["amber"], width=CHART.line_width),
        hovertemplate="Debt/GDP: %{y:.1f}%<extra></extra>",
    ))

    if len(s) >= 40:
        vals = s.values
        x    = np.arange(len(vals), dtype=float)
        m, b = np.polyfit(x, vals, 1)
        fig.add_trace(go.Scatter(
            x=s.index, y=m * x + b, mode="lines",
            name="Long-Run Trend",
            line=dict(color=C["red"], width=1.5, dash="dash"),
            hoverinfo="skip",
        ))

    for thresh, lbl in [(80, "80%"), (100, "100%"), (120, "120%")]:
        fig.add_hline(y=thresh, line_color=C["slate"], line_width=0.8, line_dash="dot",
                      annotation_text=lbl, annotation_font_size=TYPOGRAPHY.size_small,
                      annotation_position="right")
    fig.update_layout(yaxis=dict(ticksuffix="%"))
    return fig


def fiscal_impulse_chart(fae, lookback_years: int = 30) -> go.Figure:
    """
    Fiscal impulse = annual change in deficit/GDP.
    Positive bars = fiscal stimulus; negative = fiscal drag.
    """
    title = "Fiscal Impulse — Annual Change in Deficit/GDP (pp)"
    impulse = fae.fiscal_impulse_history(lookback_years=lookback_years)

    if impulse is None or impulse.empty:
        return _no_data(title)

    fig = _fig(title)
    colors = [C["teal"] if v >= 0 else C["amber"] for v in impulse.values]
    fig.add_trace(go.Bar(
        x=impulse.index, y=impulse.values, name="Fiscal Impulse",
        marker_color=colors,
        hovertemplate="Impulse: %{y:.2f}pp<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["slate"], line_width=CHART.zero_line_width)
    fig.add_annotation(
        text="Positive = stimulus  ·  Negative = fiscal drag",
        xref="paper", yref="paper", x=0.01, y=0.96,
        showarrow=False, font=dict(size=TYPOGRAPHY.size_small, color=C["slate"]),
    )
    fig.update_layout(yaxis=dict(ticksuffix="pp"), bargap=0.2)
    return fig


# ── r-g Dynamics ──────────────────────────────────────────────────────────────

def r_g_chart(fae, dl, lookback_years: int = 40) -> go.Figure:
    """
    r-g spread: 10Y Treasury yield vs nominal GDP growth, with effective
    interest rate overlay and spread panel below.

    When r > g (spread > 0), debt/GDP expands automatically even with a
    balanced primary budget.  The shaded fill turns red when r > g and
    green when g > r to make the regime visually immediate.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.40], vertical_spacing=0.06,
        subplot_titles=["10Y Treasury Yield vs Nominal GDP Growth (%)",
                        "r − g Spread (pp) — positive = debt-expanding"],
    )

    hist = fae.r_g_history(lookback_years=lookback_years)
    gs10  = hist.get("GS10")
    ng    = hist.get("NGDP_GROWTH")
    rg    = hist.get("RG_NOMINAL")
    eff   = hist.get("EFFECTIVE_RATE")

    _add_recession_bands(fig, dl, lookback_years)

    if gs10 is not None and ng is not None and not gs10.empty and not ng.empty:
        common = gs10.index.intersection(ng.index)
        if len(common) > 0:
            r_vals = gs10.loc[common].values
            g_vals = ng.loc[common].values
            # Build fill colour arrays: red when r > g, green otherwise
            fill_pos = np.where(r_vals > g_vals, r_vals, g_vals)
            fill_neg = np.where(r_vals < g_vals, r_vals, g_vals)

            # Green fill: g > r (safe zone)
            fig.add_trace(go.Scatter(
                x=list(common) + list(common[::-1]),
                y=list(np.maximum(r_vals, g_vals)) + list(np.minimum(r_vals, g_vals)[::-1]),
                fill="toself", fillcolor="rgba(47,133,90,0.12)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=common, y=gs10.loc[common].values, mode="lines",
                name="10Y Treasury (r)", line=dict(color=C["red"], width=CHART.line_width),
                hovertemplate="10Y yield: %{y:.2f}%<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=common, y=ng.loc[common].values, mode="lines",
                name="Nominal GDP Growth (g)", line=dict(color=C["teal"], width=CHART.line_width),
                hovertemplate="GDP growth: %{y:.1f}%<extra></extra>",
            ), row=1, col=1)

    if eff is not None and not eff.empty:
        eff_trim = eff[eff.index >= (pd.Timestamp.now() - pd.DateOffset(years=lookback_years))]
        fig.add_trace(go.Scatter(
            x=eff_trim.index, y=eff_trim.values, mode="lines",
            name="Effective Interest Rate on Debt",
            line=dict(color=C["amber"], width=1.5, dash="dot"),
            hovertemplate="Effective rate: %{y:.2f}%<extra></extra>",
        ), row=1, col=1)

    if rg is not None and not rg.empty:
        pos = rg.clip(lower=0)
        neg = rg.clip(upper=0)
        fig.add_trace(go.Scatter(
            x=rg.index, y=pos.values, mode="lines",
            name="r−g > 0 (debt-expanding)",
            fill="tozeroy", fillcolor="rgba(155,44,44,0.20)",
            line=dict(color=C["red"], width=1),
            hovertemplate="r−g: %{y:+.2f}pp<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=rg.index, y=neg.values, mode="lines",
            name="r−g < 0 (debt-stabilizing)",
            fill="tozeroy", fillcolor="rgba(47,133,90,0.20)",
            line=dict(color=C["teal"], width=1),
            hoverinfo="skip",
        ), row=2, col=1)
        fig.add_hline(y=0, line_color=C["slate"], line_width=1,
                      annotation_text="Breakeven (r=g)",
                      annotation_font_size=TYPOGRAPHY.size_small, row=2, col=1)

    fig.update_layout(
        title=dict(text="r vs g — Debt Sustainability Dynamics",
                   font=dict(size=TYPOGRAPHY.size_title, color=C["text"]),
                   x=0, xanchor="left"),
        **{k: v for k, v in _BASE.items() if k not in ("title", "xaxis", "yaxis")},
    )
    for row in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor=C["grid"],
                         tickfont=dict(size=TYPOGRAPHY.size_small), row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=C["grid"],
                         tickfont=dict(size=TYPOGRAPHY.size_small),
                         ticksuffix="%" if row == 1 else "pp", row=row, col=1)
    return fig


# ── CAPE + Equity Risk Premium ─────────────────────────────────────────────────

def cape_erp_chart(ev, dl, lookback_years: int = 25) -> go.Figure:
    """
    CAPE (panel 1) alongside the Equity Risk Premium vs TIPS (panel 2).

    ERP = earnings yield (100/CAPE) minus 10Y TIPS yield.
    Positive ERP = stocks yield more in real terms than bonds.
    Negative ERP = stocks are pricing in returns below the risk-free real rate.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.50], vertical_spacing=0.06,
        subplot_titles=["CAPE (Shiller P/E) — Cyclically Adjusted Valuation",
                        "Equity Risk Premium vs 10Y TIPS (earnings yield − TIPS yield, pp)"],
    )

    hist = ev.erp_history(lookback_years=lookback_years)
    cape = hist.get("CAPE")
    erp  = hist.get("ERP_TIPS")
    ey   = hist.get("EARNINGS_YIELD")
    tips = hist.get("DFII10")

    _add_recession_bands(fig, dl, lookback_years)

    # Panel 1: CAPE with zone fills
    if cape is not None and not cape.empty:
        for lo, hi, col in [(0, 20, "rgba(47,133,90,0.08)"),
                            (20, 28, "rgba(183,121,31,0.08)"),
                            (28, 60, "rgba(155,44,44,0.08)")]:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0,
                          row=1, col=1)
        fig.add_trace(go.Scatter(
            x=cape.index, y=cape.values, mode="lines", name="CAPE",
            line=dict(color=C["amber"], width=CHART.line_width),
            hovertemplate="CAPE: %{y:.1f}x<extra></extra>",
        ), row=1, col=1)
        for thresh, lbl in [(17, "Hist. Avg ~17"), (28, "Elevated"), (35, "Extreme")]:
            fig.add_hline(y=thresh, line_color=C["slate"], line_width=0.8, line_dash="dot",
                          annotation_text=lbl, annotation_font_size=TYPOGRAPHY.size_small,
                          row=1, col=1)

    # Panel 2: ERP with zero line
    if erp is not None and not erp.empty:
        pos = erp.clip(lower=0)
        neg = erp.clip(upper=0)
        fig.add_trace(go.Scatter(
            x=erp.index, y=pos.values, mode="lines",
            name="ERP > 0 (stocks cheaper than bonds)",
            fill="tozeroy", fillcolor="rgba(47,133,90,0.18)",
            line=dict(color=C["teal"], width=1.5),
            hovertemplate="ERP: %{y:+.2f}pp<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=erp.index, y=neg.values, mode="lines",
            name="ERP < 0 (stocks expensive vs bonds)",
            fill="tozeroy", fillcolor="rgba(155,44,44,0.18)",
            line=dict(color=C["red"], width=1.5),
            hoverinfo="skip",
        ), row=2, col=1)
        fig.add_hline(y=0, line_color=C["slate"], line_width=1,
                      annotation_text="ERP = 0",
                      annotation_font_size=TYPOGRAPHY.size_small, row=2, col=1)
        for lvl, lbl in [(2.0, "Fair (2pp)"), (3.5, "Attractive (3.5pp)")]:
            fig.add_hline(y=lvl, line_color=C["teal"], line_width=0.8, line_dash="dot",
                          annotation_text=lbl, annotation_font_size=TYPOGRAPHY.size_small,
                          row=2, col=1)

    fig.update_layout(
        title=dict(text="CAPE & Equity Risk Premium — Full Valuation Context",
                   font=dict(size=TYPOGRAPHY.size_title, color=C["text"]),
                   x=0, xanchor="left"),
        **{k: v for k, v in _BASE.items() if k not in ("title", "xaxis", "yaxis")},
    )
    fig.update_yaxes(showgrid=True, gridcolor=C["grid"],
                     tickfont=dict(size=TYPOGRAPHY.size_small), row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=C["grid"],
                     tickfont=dict(size=TYPOGRAPHY.size_small),
                     ticksuffix="pp", row=2, col=1)
    for row in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor=C["grid"],
                         tickfont=dict(size=TYPOGRAPHY.size_small), row=row, col=1)
    return fig


# ── Profit Margin Context ──────────────────────────────────────────────────────

def profit_margin_chart(ev, dl, lookback_years: int = 40) -> go.Figure:
    """
    Corporate profits as % of nominal GDP — shows how much of the CAPE elevation
    is driven by structurally high margins versus genuinely low required returns.

    If margins mean-revert, forward earnings will decline, making current CAPE
    understated relative to what it would be at historical average margins.
    """
    title = "Corporate Profit Margins — Profits as % of GDP"
    pm = ev.profit_margin_history(lookback_years=lookback_years)
    if pm is None or pm.empty:
        return _no_data(title)

    fig = _fig(title)
    _add_recession_bands(fig, dl, lookback_years)

    historical_avg = float(pm.mean())
    fig.add_trace(go.Scatter(
        x=pm.index, y=pm.values, mode="lines", name="Profit Margin % GDP",
        fill="tozeroy", fillcolor="rgba(43,108,176,0.09)",
        line=dict(color=C["blue"], width=CHART.line_width),
        hovertemplate="Corp. Profits / GDP: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=historical_avg, line_color=C["amber"], line_width=1.5, line_dash="dash",
                  annotation_text=f"Hist. Avg {historical_avg:.1f}%",
                  annotation_font_size=TYPOGRAPHY.size_small)
    fig.update_layout(
        yaxis=dict(ticksuffix="%"),
        annotations=[dict(
            x=0.01, y=0.96, xref="paper", yref="paper",
            text=(f"<b>Current: {pm.iloc[-1]:.1f}%</b> | "
                  f"Hist. avg: {historical_avg:.1f}% | "
                  "Above-avg margins inflate CAPE numerically"),
            showarrow=False,
            font=dict(size=TYPOGRAPHY.size_small, color=C["slate"]),
            bgcolor=C["bg"], borderpad=4,
        )],
    )
    return fig
