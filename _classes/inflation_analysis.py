"""
Inflation Analysis Engine
Decomposes inflation into cyclical vs. structural components, tracks
alternative measures, shelter lag, and expectations anchoring.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class InflationAnalysisEngine:
    """Inflation decomposition and regime classification from a DataLoader."""

    def __init__(self, dl):
        self._dl = dl

    # ── Internal helpers ───────────────────────────────────────────────────

    def _monthly(self, series_id: str) -> pd.Series | None:
        df = self._dl.load(series_id)
        if df is None or df.empty:
            return None
        col = df.columns[0]
        try:
            monthly = df.resample("ME").last()
        except ValueError:
            monthly = df.resample("M").last()
        return monthly[col].dropna()

    def _yoy_last(self, series_id: str, periods: int = 12) -> float | None:
        s = self._monthly(series_id)
        if s is None or len(s) < periods + 1:
            return None
        pct = (s.pct_change(periods) * 100).dropna()
        return float(pct.iloc[-1]) if not pct.empty else None

    def _level_last(self, series_id: str) -> float | None:
        s = self._monthly(series_id)
        if s is None or s.empty:
            return None
        return float(s.iloc[-1])

    # ── Inflation Dashboard ────────────────────────────────────────────────

    def inflation_dashboard(self) -> dict:
        """
        All key inflation readings in one call.
        Returns dict with keys:
          headline, core_cpi, core_pce, median_cpi, trimmed_mean_pce,
          supercore, sticky_cpi, oer_yoy, mich_exp, be_5y, be_10y
        """
        return {
            "headline":        self._yoy_last("CPIAUCSL"),
            "core_cpi":        self._yoy_last("CPILFESL"),
            "core_pce":        self._yoy_last("PCEPILFE"),
            "median_cpi":      self._level_last("MEDCPIM158SFRBCLE"),    # pre-computed YoY
            "trimmed_mean_pce":self._level_last("PCETRIM12M159SFRBDAL"), # pre-computed YoY
            "supercore":       self._yoy_last("CPILFENS"),
            "sticky_cpi":      self._level_last("CORESTICKM159SFRBATL"), # pre-computed YoY
            "oer_yoy":         self._yoy_last("CUSR0000SEHC"),
            "mich_exp":        self._level_last("MICH"),
            "be_5y":           self._level_last("T5YIE"),
            "be_10y":          self._level_last("T10YIE"),
        }

    # ── Cyclical vs. Structural Decomposition ─────────────────────────────

    def cyclical_vs_structural(self) -> dict:
        """
        Separates inflation into three components:

        Cyclical:  demand-driven, labour-cost-linked services — supercore CPI,
                   sticky prices, ECI-linked wage inflation.

        Structural: supply/cost-push — energy pass-through, goods inflation
                    (core CPI minus supercore), global commodity.

        Expectations: forward-looking anchoring risk — MICH, breakevens.

        Returns dict with component readings and regime classification.
        """
        dash = self.inflation_dashboard()

        cyclical_series = [dash["supercore"], dash["sticky_cpi"]]
        cyclical_vals   = [v for v in cyclical_series if v is not None]
        cyclical_avg    = float(np.mean(cyclical_vals)) if cyclical_vals else None

        # Structural proxy: core CPI minus supercore = shelter + goods contribution
        structural_proxy = None
        if dash["core_cpi"] is not None and dash["supercore"] is not None:
            structural_proxy = dash["core_cpi"] - dash["supercore"]

        # Expectations anchoring score (0 = well anchored, 100 = unanchored)
        exp_score = None
        exp_components = []
        if dash["mich_exp"] is not None:
            # MICH deviations above 3% are concerning
            exp_components.append(max(0, dash["mich_exp"] - 3.0) * 20)
        if dash["be_5y"] is not None:
            exp_components.append(max(0, dash["be_5y"] - 2.5) * 25)
        if dash["be_10y"] is not None:
            exp_components.append(max(0, dash["be_10y"] - 2.5) * 20)
        if exp_components:
            exp_score = float(np.clip(np.mean(exp_components), 0, 100))

        return {
            "cyclical_avg":      cyclical_avg,
            "structural_proxy":  structural_proxy,
            "expectations_score": exp_score,
            "components": {
                "supercore":         dash["supercore"],
                "sticky_cpi":        dash["sticky_cpi"],
                "shelter_oer":       dash["oer_yoy"],
                "core_goods_proxy":  structural_proxy,
            },
        }

    # ── Shelter Lag Analysis ───────────────────────────────────────────────

    def shelter_decomposition(self, lookback_years: int = 5) -> dict:
        """
        Compares OER (lagged market rents) vs. overall shelter CPI vs. core CPI.
        Returns time series of each for charting.
        """
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)

        def _get_yoy(sid, pre_computed=False):
            s = self._monthly(sid)
            if s is None or s.empty:
                return None
            s = s[s.index >= cutoff]
            if pre_computed:
                return s  # already a % change series
            if len(s) < 13:
                return None
            return (s.pct_change(12) * 100).dropna()

        return {
            "core_cpi": _get_yoy("CPILFESL"),
            "oer":      _get_yoy("CUSR0000SEHC"),
            "supercore":_get_yoy("CPILFENS"),
        }

    # ── Inflation Regime Classification ───────────────────────────────────

    def inflation_regime(self) -> tuple[str, str]:
        """
        Returns (regime_label, color) based on current readings.
        Regimes: 'At Target', 'Disinflation', 'Elevated', 'Entrenched'
        """
        dash = self.inflation_dashboard()

        headline  = dash["headline"]
        core      = dash["core_cpi"]
        sticky    = dash["sticky_cpi"]
        mich      = dash["mich_exp"]

        # Disinflation: headline falling, core near target
        # Entrenched: sticky > 4%, expectations rising
        # Elevated: above target but not entrenched
        # At Target: all measures near 2%

        anchored = (mich is None or mich < 3.5)

        at_target  = (headline is not None and headline < 2.5 and
                      core is not None     and core < 2.5)
        disinflation = (headline is not None and headline < 3.5 and
                        core is not None     and core < 3.5 and
                        (sticky is None      or sticky < 4.0))
        entrenched = (sticky is not None and sticky > 4.0 and not anchored)
        elevated   = (core is not None and core >= 2.5 and not entrenched)

        if at_target and anchored:
            return "At Target", "#276749"
        if entrenched:
            return "Entrenched", "#9b2c2c"
        if disinflation:
            return "Disinflation", "#2b6cb0"
        if elevated:
            return "Elevated", "#975a16"
        return "Uncertain", "#718096"

    # ── Rolling History for Charts ─────────────────────────────────────────

    def inflation_history(
        self, series_ids: list[str], lookback_years: int = 10,
        pre_computed: list[str] | None = None
    ) -> dict[str, pd.Series]:
        """
        Returns YoY time series for each series_id trimmed to lookback_years.
        pre_computed: list of IDs that are already percent-change series (no pct_change).
        """
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        pre_computed = pre_computed or []
        out = {}
        for sid in series_ids:
            s = self._monthly(sid)
            if s is None or s.empty:
                continue
            if sid in pre_computed:
                s = s[s.index >= cutoff]
                out[sid] = s
            else:
                if len(s) < 13:
                    continue
                yoy = (s.pct_change(12) * 100).dropna()
                out[sid] = yoy[yoy.index >= cutoff]
        return out
