"""
Equity Valuation Engine
Contextualizes CAPE and market valuation with:
  - Equity Risk Premium (ERP = earnings yield minus real yield)
  - Corporate profit margins (profits as % of GDP)
  - Real yield backdrop (10Y TIPS)
  - Nominal earnings yield

Motivation: CAPE in isolation is insufficient.  A CAPE of 30 means very
different things when TIPS yields are −1% (ERP ~4.3%, cheap vs bonds)
versus +3% (ERP ~0.3%, expensive vs bonds). Similarly, today's elevated CAPE
partly reflects structurally higher profit margins — knowing margin history
helps assess whether reversion risk is embedded in the CAPE level.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class EquityValuationEngine:
    """Equity valuation context from FRED-available market and macro data."""

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

    # ── CAPE Dashboard ────────────────────────────────────────────────────

    def cape_dashboard(self) -> dict:
        """
        CAPE with full valuation context.

        ERP (vs TIPS) = earnings yield − 10Y TIPS yield.
        Positive ERP = stocks yield more in real terms than bonds.
        Near-zero or negative ERP = historically poor forward equity returns.

        Profit margin (Corp. profits / GDP) shows how much of an elevated CAPE
        is driven by structurally high margins (a level that may not persist)
        vs. genuinely low required returns.
        """
        cape   = self._monthly("SP500_CAPE")
        tips   = self._monthly("DFII10")
        gs10   = self._monthly("GS10")
        cp     = self._quarterly("CP")
        gdp    = self._quarterly("GDP")

        cape_level = float(cape.iloc[-1])  if (cape  is not None and not cape.empty)  else None
        tips_yield = float(tips.iloc[-1])  if (tips  is not None and not tips.empty)  else None
        gs10_yield = float(gs10.iloc[-1])  if (gs10  is not None and not gs10.empty)  else None

        earnings_yield   = (100.0 / cape_level)  if (cape_level and cape_level > 0) else None
        erp_vs_tips      = (earnings_yield - tips_yield)  if (earnings_yield is not None and tips_yield  is not None) else None
        erp_vs_nominal   = (earnings_yield - gs10_yield)  if (earnings_yield is not None and gs10_yield is not None) else None

        # Profit margin: corporate profits as % of nominal GDP
        profit_margin_pct = None
        if cp is not None and gdp is not None:
            common = cp.index.intersection(gdp.index)
            if len(common) > 0:
                pm = (cp.loc[common] / gdp.loc[common] * 100).dropna()
                profit_margin_pct = float(pm.iloc[-1]) if not pm.empty else None

        # Historical profit margin context (postwar average ~7%)
        pm_historical_avg = None
        if cp is not None and gdp is not None:
            common = cp.index.intersection(gdp.index)
            if len(common) > 20:
                pm_all = (cp.loc[common] / gdp.loc[common] * 100).dropna()
                pm_historical_avg = float(pm_all.mean()) if not pm_all.empty else None

        # CAPE regime
        cape_regime = "Unknown"
        if cape_level is not None:
            if cape_level > 38:
                cape_regime = "Extreme (>38) — 1929/2000 territory"
            elif cape_level > 30:
                cape_regime = "Elevated (>30) — historically poor 10Y returns"
            elif cape_level > 22:
                cape_regime = "Above-Average — modest return headwinds"
            elif cape_level > 14:
                cape_regime = "Fair Value Range (14–22)"
            else:
                cape_regime = "Below Average — historically strong forward returns"

        # ERP regime (vs TIPS — the cleanest real-vs-real comparison)
        erp_regime = "Unknown"
        if erp_vs_tips is not None:
            if erp_vs_tips < 0:
                erp_regime = "Stocks Expensive vs Bonds — ERP Negative"
            elif erp_vs_tips < 1.0:
                erp_regime = "Compressed Premium — Low Margin of Safety"
            elif erp_vs_tips < 2.5:
                erp_regime = "Moderate Premium"
            elif erp_vs_tips < 4.0:
                erp_regime = "Fair Risk Compensation"
            else:
                erp_regime = "Attractive Real Risk Premium"

        return {
            "cape":                  cape_level,
            "earnings_yield":        earnings_yield,
            "tips_yield":            tips_yield,
            "nominal_yield":         gs10_yield,
            "erp_vs_tips":           erp_vs_tips,
            "erp_vs_nominal":        erp_vs_nominal,
            "profit_margin_pct_gdp": profit_margin_pct,
            "pm_historical_avg":     pm_historical_avg,
            "cape_regime":           cape_regime,
            "erp_regime":            erp_regime,
        }

    # ── History for Charts ────────────────────────────────────────────────

    def erp_history(self, lookback_years: int = 25) -> dict[str, pd.Series]:
        """
        Historical ERP and its components for charting.
        Returns: CAPE, EARNINGS_YIELD, DFII10, ERP_TIPS, GS10, ERP_NOMINAL
        """
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        out: dict[str, pd.Series] = {}

        cape = self._monthly("SP500_CAPE")
        tips = self._monthly("DFII10")
        gs10 = self._monthly("GS10")

        if cape is not None and not cape.empty:
            ey = (100.0 / cape).replace([np.inf, -np.inf], np.nan).dropna()
            cape_trim = cape[cape.index >= cutoff]
            ey_trim   = ey[ey.index >= cutoff]
            if not cape_trim.empty:
                out["CAPE"]          = cape_trim
            if not ey_trim.empty:
                out["EARNINGS_YIELD"] = ey_trim

        if tips is not None and not tips.empty and "EARNINGS_YIELD" in out:
            ey = out["EARNINGS_YIELD"]
            common = ey.index.intersection(tips.index)
            common = common[common >= cutoff]
            if len(common) > 0:
                out["DFII10"]   = tips.loc[common]
                out["ERP_TIPS"] = (ey.loc[common] - tips.loc[common]).dropna()

        if gs10 is not None and not gs10.empty and "EARNINGS_YIELD" in out:
            ey = out["EARNINGS_YIELD"]
            common = ey.index.intersection(gs10.index)
            common = common[common >= cutoff]
            if len(common) > 0:
                out["GS10"]        = gs10.loc[common]
                out["ERP_NOMINAL"] = (ey.loc[common] - gs10.loc[common]).dropna()

        return out

    def profit_margin_history(self, lookback_years: int = 40) -> pd.Series | None:
        """Corporate profits as % of nominal GDP."""
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        cp  = self._quarterly("CP")
        gdp = self._quarterly("GDP")
        if cp is None or gdp is None:
            return None
        common = cp.index.intersection(gdp.index)
        common = common[common >= cutoff]
        if len(common) == 0:
            return None
        pm = (cp.loc[common] / gdp.loc[common] * 100).dropna()
        return pm if not pm.empty else None
