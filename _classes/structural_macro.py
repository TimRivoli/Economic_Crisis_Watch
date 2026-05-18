"""
Structural Macro Engine
Tracks slow-moving structural forces: productivity, demographic pressures,
output gap (real vs potential GDP), real interest rates / r*, and
globalization metrics derived from FRED-available series.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class StructuralMacroEngine:
    """Structural macro analytics from FRED data."""

    def __init__(self, dl):
        self._dl = dl

    # ── Internal helpers ───────────────────────────────────────────────────

    def _monthly(self, sid: str) -> pd.Series | None:
        df = self._dl.load(sid)
        if df is None or df.empty:
            return None
        col = df.columns[0]
        try:
            m = df.resample("ME").last()
        except ValueError:
            m = df.resample("M").last()
        return m[col].dropna()

    def _quarterly(self, sid: str) -> pd.Series | None:
        df = self._dl.load(sid)
        if df is None or df.empty:
            return None
        col = df.columns[0]
        try:
            q = df.resample("QE").last()
        except ValueError:
            q = df.resample("Q").last()
        return q[col].dropna()

    def _level_last(self, sid: str, freq: str = "monthly") -> float | None:
        s = self._monthly(sid) if freq == "monthly" else self._quarterly(sid)
        return float(s.iloc[-1]) if (s is not None and not s.empty) else None

    def _slope_ann(self, s: pd.Series, periods: int, period_per_year: int) -> float | None:
        """Annualized linear trend over the last N periods."""
        if s is None or len(s) < periods:
            return None
        vals = s.iloc[-periods:].values
        if np.any(np.isnan(vals)):
            return None
        x = np.arange(len(vals), dtype=float)
        return float(np.polyfit(x, vals, 1)[0] * period_per_year)

    # ── Output Gap ────────────────────────────────────────────────────────

    def output_gap(self) -> dict:
        """
        (Real GDP − Potential GDP) / Potential GDP × 100.
        Positive = overheating; negative = economic slack.
        Both series are CBO real (chained 2017$, quarterly).
        """
        gdp = self._quarterly("GDPC1")
        pot = self._quarterly("GDPPOT")

        gap_series  = None
        gap_current = None
        trend_2y    = None

        if gdp is not None and pot is not None:
            common = gdp.index.intersection(pot.index)
            if len(common) > 0:
                gap_series = ((gdp.loc[common] - pot.loc[common])
                              / pot.loc[common] * 100).dropna()
                if not gap_series.empty:
                    gap_current = float(gap_series.iloc[-1])
                    trend_2y = self._slope_ann(gap_series, min(8, len(gap_series)), 4)

        label = "Unknown"
        if gap_current is not None:
            if gap_current > 1.5:
                label = "Overheating"
            elif gap_current > 0.3:
                label = "Above Potential"
            elif gap_current > -1.5:
                label = "Near Potential"
            else:
                label = "Significant Slack"

        return {
            "gap_current": gap_current,
            "trend_2y":    trend_2y,
            "label":       label,
            "gap_series":  gap_series,
        }

    # ── Productivity ──────────────────────────────────────────────────────

    def productivity_trend(self, lookback_years: int = 15) -> dict:
        """
        Nonfarm business sector productivity (OPHNFB) YoY trend analysis.
        Compares recent average to long-run average to detect structural
        acceleration or deceleration.
        """
        s = self._quarterly("OPHNFB")
        if s is None or len(s) < 8:
            return {
                "trend_10y_avg": None, "recent_avg": None,
                "acceleration": None, "label": "Insufficient Data",
                "yoy_series": None,
            }

        yoy = (s.pct_change(4) * 100).dropna()
        hist_w    = min(40, len(yoy))   # up to 10 years of quarters
        long_avg  = float(yoy.iloc[-hist_w:].mean())
        recent_w  = min(4, len(yoy))
        recent_avg = float(yoy.iloc[-recent_w:].mean())
        acceleration = recent_avg - long_avg

        label = ("Accelerating" if acceleration > 0.5 else
                 "Trend Pace"   if acceleration > -0.3 else "Decelerating")

        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        return {
            "trend_10y_avg": long_avg,
            "recent_avg":    recent_avg,
            "acceleration":  acceleration,
            "label":         label,
            "yoy_series":    yoy[yoy.index >= cutoff],
        }

    # ── Real Interest Rates & Policy Stance ──────────────────────────────

    def real_rates(self) -> dict:
        """
        Real rate landscape.
        - 10Y TIPS yield (DFII10): long-run real rate from the bond market.
        - Real Fed Funds Rate: nominal FF − core CPI YoY.
        - Policy stance: is real FF above or below long-run real rate expectations?

        METHODOLOGY NOTE — TIPS yield ≠ r*:
        The true neutral rate (r*) is unobservable. The 10Y TIPS yield is used
        as a practical long-run real rate benchmark, but it is not r*. It
        conflates three distinct components:
          (a) Expected average short-term real rate over 10 years — the component
              closest to r*;
          (b) Real term premium — compensation for holding long-duration real bonds,
              which empirical estimates place at roughly +0.5 to +1.5%;
          (c) TIPS liquidity premium — TIPS are less liquid than nominal Treasuries,
              which adds a small positive bias to their yield.
        As a result, the 10Y TIPS yield systematically overstates r* during
        tightening cycles (elevated term premium) and can understate it during
        QE-compressed periods. The policy stance comparison below is best read
        as "real FF vs. long-term real market rate expectations," not as
        "real FF vs. neutral rate."
        """
        tips   = self._monthly("DFII10")
        ff     = self._monthly("FEDFUNDS")
        core   = self._monthly("CPILFESL")

        tips_current    = None
        real_ff         = None
        real_rate_proxy = None  # 10Y TIPS — long-run real rate benchmark, not r*
        policy_stance   = "Unknown"

        if tips is not None and not tips.empty:
            tips_current    = float(tips.iloc[-1])
            real_rate_proxy = tips_current

        if ff is not None and core is not None and len(core) > 12:
            yoy = (core.pct_change(12) * 100).dropna()
            ff_a = ff.reindex(yoy.index, method="ffill").dropna()
            real_ff_s = (ff_a - yoy).dropna()
            if not real_ff_s.empty:
                real_ff = float(real_ff_s.iloc[-1])

        if real_ff is not None and real_rate_proxy is not None:
            gap = real_ff - real_rate_proxy
            policy_stance = ("Restrictive"        if gap > 1.0  else
                             "Mildly Restrictive"  if gap > 0    else
                             "Near Neutral"        if gap > -0.5 else
                             "Accommodative")

        return {
            "tips_10y":      tips_current,
            "real_ff":       real_ff,
            "real_rate_proxy": real_rate_proxy,
            "policy_stance": policy_stance,
        }

    # ── Demographics ──────────────────────────────────────────────────────

    def demographic_pressure(self) -> dict:
        """
        Working-age population growth (LFWA64TTUSM647S), LFPR trend,
        and employment-population ratio slope — structural labor supply signals.
        """
        wap  = self._monthly("LFWA64TTUSM647S")
        lfpr = self._monthly("CIVPART")
        ep   = self._monthly("EMRATIO")

        wap_yoy     = None
        wap_current = None
        lfpr_slope  = None
        ep_slope    = None

        if wap is not None and len(wap) > 12:
            pct = (wap.pct_change(12) * 100).dropna()
            if not pct.empty:
                wap_yoy     = float(pct.iloc[-1])
                wap_current = float(wap.iloc[-1])

        # 3-year annualized slope
        lfpr_slope = self._slope_ann(lfpr, 36, 12) if lfpr is not None else None
        ep_slope   = self._slope_ann(ep,   36, 12) if ep   is not None else None

        label = "Neutral"
        if wap_yoy is not None:
            if wap_yoy < 0.2:
                label = "Strong Headwind — Workforce Aging"
            elif wap_yoy < 0.5:
                label = "Moderate Headwind"
            elif wap_yoy < 0.8:
                label = "Modest Support"
            else:
                label = "Favorable Demographics"

        return {
            "wap_yoy":         wap_yoy,
            "wap_current_k":   wap_current,
            "lfpr_slope_ann":  lfpr_slope,
            "ep_slope_ann":    ep_slope,
            "label":           label,
        }

    # ── Globalization Proxy ───────────────────────────────────────────────

    def globalization_metrics(self) -> dict:
        """
        Dollar strength trend and PPI commodity cycle as proxies for
        globalization/deglobalization cycles and trade conditions.
        Dollar appreciation is associated with financial fragmentation;
        commodity supercycles correspond to globalization phases.
        """
        dxy = self._monthly("DTWEXBGS")
        ppi = self._monthly("PPIACO")

        dxy_trend_5y = self._slope_ann(dxy, 60, 12) if dxy is not None else None
        ppi_yoy      = None
        if ppi is not None and len(ppi) > 12:
            pct = (ppi.pct_change(12) * 100).dropna()
            ppi_yoy = float(pct.iloc[-1]) if not pct.empty else None

        label = "Uncertain"
        if dxy_trend_5y is not None:
            if dxy_trend_5y > 2.5:
                label = "Deglobalization Pressure — USD Strengthening"
            elif dxy_trend_5y < -2.5:
                label = "Globalization Phase — USD Softening"
            else:
                label = "Neutral"

        return {
            "dxy_trend_5y_ann":    dxy_trend_5y,
            "ppi_commodities_yoy": ppi_yoy,
            "label":               label,
        }

    # ── History for Charts ────────────────────────────────────────────────

    def real_rates_history(self, lookback_years: int = 20) -> dict[str, pd.Series]:
        """DFII10 and real FF rate as monthly series."""
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        out = {}

        tips = self._monthly("DFII10")
        if tips is not None and not tips.empty:
            out["DFII10"] = tips[tips.index >= cutoff]

        ff   = self._monthly("FEDFUNDS")
        core = self._monthly("CPILFESL")
        if ff is not None and core is not None and len(core) > 12:
            yoy  = (core.pct_change(12) * 100).dropna()
            ff_a = ff.reindex(yoy.index, method="ffill").dropna()
            real_ff = (ff_a - yoy).dropna()
            out["REAL_FF"] = real_ff[real_ff.index >= cutoff]

        return out

    def output_gap_history(self, lookback_years: int = 20) -> dict[str, pd.Series]:
        """Real GDP, Potential GDP, and gap series for charting."""
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        out = {}

        gdp = self._quarterly("GDPC1")
        pot = self._quarterly("GDPPOT")

        if gdp is not None:
            out["GDPC1"]  = gdp[gdp.index >= cutoff]
        if pot is not None:
            out["GDPPOT"] = pot[pot.index >= cutoff]

        gap_data = self.output_gap()
        gs = gap_data.get("gap_series")
        if gs is not None and not gs.empty:
            out["OUTPUT_GAP"] = gs[gs.index >= cutoff]

        return out

    def productivity_history(self, lookback_years: int = 20) -> pd.Series | None:
        """Quarterly nonfarm productivity YoY series for charting."""
        s = self._quarterly("OPHNFB")
        if s is None or len(s) < 5:
            return None
        yoy    = (s.pct_change(4) * 100).dropna()
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        return yoy[yoy.index >= cutoff]
