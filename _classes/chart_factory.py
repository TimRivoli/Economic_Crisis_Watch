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
