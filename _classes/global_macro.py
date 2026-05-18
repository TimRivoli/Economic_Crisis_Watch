"""
Global Macro Engine
Tracks international indicators: commodity complex, FX conditions,
major central bank policy divergence, and global growth pulse.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class GlobalMacroEngine:
    """Global macro dashboard from FRED-available international data."""

    def __init__(self, dl):
        self._dl = dl

    # ── Internal helpers ───────────────────────────────────────────────────

    def _monthly(self, series_id: str, agg: str = "mean") -> pd.Series | None:
        df = self._dl.load(series_id)
        if df is None or df.empty:
            return None
        col = df.columns[0]
        fn = df.resample("ME").mean if agg == "mean" else df.resample("ME").last
        try:
            m = fn()
        except ValueError:
            fn2 = df.resample("M").mean if agg == "mean" else df.resample("M").last
            m = fn2()
        return m[col].dropna()

    def _level_last(self, sid: str) -> float | None:
        s = self._monthly(sid)
        return float(s.iloc[-1]) if (s is not None and not s.empty) else None

    def _yoy_last(self, sid: str, periods: int = 12) -> float | None:
        s = self._monthly(sid)
        if s is None or len(s) < periods + 1:
            return None
        pct = (s.pct_change(periods) * 100).dropna()
        return float(pct.iloc[-1]) if not pct.empty else None

    def _trend_ann(self, sid: str, months: int = 36) -> float | None:
        """Annualized linear trend over the last N months."""
        s = self._monthly(sid)
        if s is None or len(s) < months:
            return None
        vals = s.iloc[-months:].values
        x = np.arange(len(vals), dtype=float)
        return float(np.polyfit(x, vals, 1)[0] * 12)

    # ── Commodity Complex ──────────────────────────────────────────────────

    def commodity_complex(self) -> dict:
        """Brent crude, gold, and PPI-All-Commodities — levels and YoY change."""
        brent_lvl = self._level_last("DCOILBRENTEU")
        brent_yoy = self._yoy_last("DCOILBRENTEU")
        gold_lvl  = self._level_last("GOLDPMGBD228NLBMA")
        gold_yoy  = self._yoy_last("GOLDPMGBD228NLBMA")
        ppi_yoy   = self._yoy_last("PPIACO")

        vals = [v for v in [brent_yoy, gold_yoy, ppi_yoy] if v is not None]
        commodity_pressure = float(np.mean(vals)) if vals else None

        # Commodity regime: rising commodities = inflationary impulse
        regime = "Neutral"
        if commodity_pressure is not None:
            if commodity_pressure > 15:
                regime = "Commodity Surge (Inflationary)"
            elif commodity_pressure > 5:
                regime = "Rising Commodities"
            elif commodity_pressure < -10:
                regime = "Commodity Decline (Deflationary Impulse)"
            elif commodity_pressure < -2:
                regime = "Falling Commodities"

        return {
            "brent_level":         brent_lvl,
            "brent_yoy":           brent_yoy,
            "gold_level":          gold_lvl,
            "gold_yoy":            gold_yoy,
            "ppi_commodities_yoy": ppi_yoy,
            "commodity_pressure":  commodity_pressure,
            "commodity_regime":    regime,
        }

    # ── FX Conditions ─────────────────────────────────────────────────────

    def fx_conditions(self) -> dict:
        """Broad USD, EUR/USD, JPY/USD, CNY/USD — levels and 12M change."""
        dxy_lvl  = self._level_last("DTWEXBGS")
        dxy_yoy  = self._yoy_last("DTWEXBGS")
        eur_lvl  = self._level_last("DEXUSEU")
        eur_yoy  = self._yoy_last("DEXUSEU")
        jpy_lvl  = self._level_last("DEXJPUS")
        jpy_yoy  = self._yoy_last("DEXJPUS")
        cny_lvl  = self._level_last("DEXCHUS")
        cny_yoy  = self._yoy_last("DEXCHUS")

        usd_regime = "Neutral"
        if dxy_yoy is not None:
            if dxy_yoy > 8:
                usd_regime = "Strong Dollar — Tightening Global Conditions"
            elif dxy_yoy > 4:
                usd_regime = "Dollar Appreciating"
            elif dxy_yoy < -8:
                usd_regime = "Weak Dollar — Accommodative Global Impulse"
            elif dxy_yoy < -4:
                usd_regime = "Dollar Depreciating"

        return {
            "dxy_level":  dxy_lvl,
            "dxy_yoy":    dxy_yoy,
            "eur_usd":    eur_lvl,
            "eur_yoy":    eur_yoy,
            "jpy_usd":    jpy_lvl,
            "jpy_yoy":    jpy_yoy,
            "cny_usd":    cny_lvl,
            "cny_yoy":    cny_yoy,
            "usd_regime": usd_regime,
        }

    # ── Central Bank Divergence ────────────────────────────────────────────

    def central_bank_divergence(self) -> dict:
        """Fed vs ECB vs BOJ policy rates — gaps and divergence label."""
        fed = self._level_last("FEDFUNDS")
        ecb = self._level_last("IR3TIB01EZM156N")
        boj = self._level_last("IRSTCB01JPM156N")

        fed_ecb_gap = (fed - ecb) if (fed is not None and ecb is not None) else None
        fed_boj_gap = (fed - boj) if (fed is not None and boj is not None) else None

        divergence_label = "Unknown"
        if fed_ecb_gap is not None:
            if fed_ecb_gap > 2.5:
                divergence_label = "Wide US Premium — USD Supportive"
            elif fed_ecb_gap > 1.0:
                divergence_label = "US Rate Advantage"
            elif fed_ecb_gap < -1.0:
                divergence_label = "ECB Leading — EUR Supportive"
            else:
                divergence_label = "Moderate Divergence"

        return {
            "fed_rate":         fed,
            "ecb_rate":         ecb,
            "boj_rate":         boj,
            "fed_ecb_gap":      fed_ecb_gap,
            "fed_boj_gap":      fed_boj_gap,
            "divergence_label": divergence_label,
        }

    # ── Consolidated Dashboard ────────────────────────────────────────────

    def global_dashboard(self) -> dict:
        return {
            "commodities":   self.commodity_complex(),
            "fx":            self.fx_conditions(),
            "cb_divergence": self.central_bank_divergence(),
        }

    # ── History for Charts ────────────────────────────────────────────────

    def commodity_history(self, lookback_years: int = 15) -> dict[str, pd.Series]:
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        out = {}
        for sid in ["DCOILBRENTEU", "GOLDPMGBD228NLBMA", "PPIACO"]:
            s = self._monthly(sid)
            if s is not None and not s.empty:
                out[sid] = s[s.index >= cutoff]
        return out

    def fx_history(self, lookback_years: int = 15) -> dict[str, pd.Series]:
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        out = {}
        for sid in ["DTWEXBGS", "DEXUSEU", "DEXJPUS", "DEXCHUS"]:
            s = self._monthly(sid)
            if s is not None and not s.empty:
                out[sid] = s[s.index >= cutoff]
        return out

    def cb_rates_history(self, lookback_years: int = 25) -> dict[str, pd.Series]:
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        out = {}
        for sid in ["FEDFUNDS", "IR3TIB01EZM156N", "IRSTCB01JPM156N"]:
            s = self._monthly(sid, agg="last")
            if s is not None and not s.empty:
                out[sid] = s[s.index >= cutoff]
        return out
