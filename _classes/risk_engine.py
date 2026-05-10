"""
RiskEngine: scores individual series and synthesizes composite scores
for the five Crisis Watch dimensions.

Scoring is based on thresholds defined in series_registry.py.
All scores are probabilistic/descriptive — not predictions.
"""

from __future__ import annotations
import pandas as pd
from .series_registry import REGISTRY
from .constants import RISK_STYLE

_RISK_ORDER = {"red": 3, "yellow": 2, "green": 1, "neutral": 0}
_RISK_FROM_INT = {3: "red", 2: "yellow", 1: "green", 0: "neutral"}


def _in_range(lo, hi, v: float) -> bool:
    return (lo is None or v >= lo) and (hi is None or v < hi)


class RiskEngine:
    def __init__(self, data_loader):
        self.dl = data_loader

    # ── Core Scoring ───────────────────────────────────────────────────────

    def score_value(self, series_id: str, value: float | None) -> str:
        """Return 'green', 'yellow', 'red', or 'neutral' for an explicit value."""
        if series_id not in REGISTRY or value is None or pd.isna(value):
            return "neutral"
        thresholds = REGISTRY[series_id]["risk_thresholds"]
        for level in ("green", "yellow", "red"):
            if level in thresholds and _in_range(*thresholds[level], value):
                return level
        return "neutral"

    def score(self, series_id: str) -> tuple[str, str, float | None]:
        """
        Return (risk_level, formatted_display_string, raw_value) for a series.
        The risk_basis in the registry determines which value is scored.
        """
        meta = REGISTRY.get(series_id)
        if not meta:
            return "neutral", "N/A", None

        basis = meta.get("risk_basis", "level")

        if basis == "yoy":
            value = self.dl.get_yoy(series_id)
        elif basis == "mom_change":
            value = self.dl.get_mom(series_id)
        else:
            value, _ = self.dl.get_latest(series_id)

        risk = self.score_value(series_id, value)
        display = self._format(series_id, value, basis)
        return risk, display, value

    # ── Crisis Dimension Synthesis ─────────────────────────────────────────

    def crisis_dimensions(self) -> dict[str, dict]:
        """
        Return composite risk scores for the five Crisis Watch dimensions.
        Each dimension aggregates two or more indicators, taking the worst score.
        Returns a dict keyed by dimension name with 'score', 'components', 'description'.
        """
        def worst(*ids_and_bases):
            scores = []
            for sid in ids_and_bases:
                risk, display, _ = self.score(sid)
                scores.append((risk, sid, display))
            worst_level = _RISK_FROM_INT[max(_RISK_ORDER.get(s, 0) for s, _, _ in scores)]
            return worst_level, scores

        y10_risk, y10_display, _ = self.score("DGS10")
        cre_risk, cre_display, _ = self.score("DRCRELEXFACBS")
        core_risk, core_display, _ = self.score("CPILFESL")
        cpi_risk, cpi_display, _ = self.score("CPIAUCSL")
        un_risk, un_display, _ = self.score("UNRATE")
        pay_risk, pay_display, _ = self.score("PAYEMS")
        vix_risk, vix_display, _ = self.score("VIXCLS")
        debt_risk, debt_display, _ = self.score("GFDEGDQ188S")
        cape_risk, cape_display, _ = self.score("SP500_CAPE")
        pe_risk, pe_display, _ = self.score("SP500_PE")

        def composite(*levels):
            return _RISK_FROM_INT[max(_RISK_ORDER.get(lv, 0) for lv in levels)]

        return {
            "Treasury Stress": {
                "score": composite(y10_risk, cre_risk),
                "components": [
                    ("10Y Yield", y10_risk, y10_display),
                    ("CRE Delinquency", cre_risk, cre_display),
                ],
                "description": (
                    "Combines the 10-year Treasury yield level with commercial real estate "
                    "loan delinquencies. High yields amplify debt service burdens; rising CRE "
                    "delinquencies signal credit stress in a rate-sensitive sector."
                ),
            },
            "Inflation Persistence": {
                "score": composite(core_risk, cpi_risk),
                "components": [
                    ("Core CPI YoY", core_risk, core_display),
                    ("Headline CPI YoY", cpi_risk, cpi_display),
                ],
                "description": (
                    "Tracks whether inflation is above the Fed's 2% target and whether "
                    "it is broad-based (headline) or structural (core, ex food & energy). "
                    "Persistent core inflation is more concerning than headline spikes."
                ),
            },
            "Labor Weakness": {
                "score": composite(un_risk, pay_risk),
                "components": [
                    ("Unemployment Rate", un_risk, un_display),
                    ("Payrolls MoM", pay_risk, pay_display),
                ],
                "description": (
                    "Combines the unemployment rate level with monthly payroll growth. "
                    "Rising unemployment alongside slowing payrolls indicates genuine "
                    "demand destruction rather than structural labor supply shifts."
                ),
            },
            "Financial Stress": {
                "score": composite(vix_risk, cre_risk),
                "components": [
                    ("VIX", vix_risk, vix_display),
                    ("CRE Delinquency", cre_risk, cre_display),
                ],
                "description": (
                    "Equity market volatility (VIX) captures near-term market sentiment; "
                    "CRE delinquencies reflect credit quality in real economy lending. "
                    "Simultaneous stress in both suggests broad financial tightening."
                ),
            },
            "Debt Sustainability": {
                "score": composite(debt_risk, y10_risk),
                "components": [
                    ("Federal Debt / GDP", debt_risk, debt_display),
                    ("10Y Yield", y10_risk, y10_display),
                ],
                "description": (
                    "At elevated debt/GDP levels, rising interest rates compound the fiscal "
                    "burden non-linearly. This dimension tracks whether the debt load and "
                    "borrowing cost together constrain policy flexibility."
                ),
            },
            "Market Valuation": {
                "score": composite(cape_risk, pe_risk),
                "components": [
                    ("CAPE (Shiller P/E)", cape_risk, cape_display),
                    ("Trailing P/E", pe_risk, pe_display),
                ],
                "description": (
                    "Elevated valuations do not predict the timing of corrections, but "
                    "they reduce the margin of safety: the higher the starting CAPE, the "
                    "lower the historical probability of strong 10-year forward real returns. "
                    "CAPE above 30 has historically been followed by decade-long below-average returns."
                ),
            },
        }

    # ── Formatting ─────────────────────────────────────────────────────────

    @staticmethod
    def _format(series_id: str, value: float | None, basis: str) -> str:
        if value is None:
            return "N/A"
        meta = REGISTRY[series_id]
        units = meta.get("units", "")
        if basis == "yoy":
            return f"{value:.1f}%"
        if basis == "mom_change":
            return f"{value:+,.0f}K"
        if "Percent" in units or units.endswith("%"):
            return f"{value:.1f}%"
        if "Billions" in units:
            return f"${value:,.0f}B"
        return f"{value:,.1f}"
