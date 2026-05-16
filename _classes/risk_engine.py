"""
RiskEngine: scores individual series and synthesizes composite scores
for the five Crisis Watch dimensions.

Scoring is based on thresholds defined in series_registry.py.
All scores are probabilistic/descriptive — not predictions.
"""

from __future__ import annotations
import pandas as pd
from .series_registry import REGISTRY
from .constants import RISK_STYLE, DASH

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

    def _as_of(self, series_id: str) -> str:
        """Formatted date of the latest observation for a series."""
        _, date = self.dl.get_latest(series_id)
        return date.strftime(DASH.date_display_fmt) if date is not None else ""

    def cp_treasury_spread(self) -> tuple[str, str, str]:
        """
        Compute 3-Month CP minus 3-Month Treasury spread and return
        (risk, display, as_of).  Thresholds mirror the old TED spread:
        green < 0.50%, yellow 0.50–1.00%, red >= 1.00%.
        """
        cp_val, cp_date = self.dl.get_latest("DCPF3M")
        tsy_val, tsy_date = self.dl.get_latest("DGS3MO")
        if cp_val is None or tsy_val is None:
            return "neutral", "N/A", ""
        spread = cp_val - tsy_val
        display = f"{spread:.2f}%"
        if spread < 0.50:
            risk = "green"
        elif spread < 1.00:
            risk = "yellow"
        else:
            risk = "red"
        # Use the earlier of the two dates as the "as of"
        date = min(d for d in (cp_date, tsy_date) if d is not None)
        as_of = date.strftime(DASH.date_display_fmt) if date is not None else ""
        return risk, display, as_of

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
        y10_as_of = self._as_of("DGS10")
        yc_risk, yc_display, _ = self.score("T10Y2Y")
        yc_as_of = self._as_of("T10Y2Y")
        rec_risk, rec_display, _ = self.score("RECPROUSM156N")
        rec_as_of = self._as_of("RECPROUSM156N")
        cre_risk, cre_display, _ = self.score("SUBLPDRCSN")
        cre_as_of = self._as_of("SUBLPDRCSN")
        core_risk, core_display, _ = self.score("CPILFESL")
        core_as_of = self._as_of("CPILFESL")
        cpi_risk, cpi_display, _ = self.score("CPIAUCSL")
        cpi_as_of = self._as_of("CPIAUCSL")
        un_risk, un_display, _ = self.score("UNRATE")
        un_as_of = self._as_of("UNRATE")
        pay_risk, pay_display, _ = self.score("PAYEMS")
        pay_as_of = self._as_of("PAYEMS")
        usslind_risk, usslind_display, _ = self.score("USSLIND")
        usslind_as_of = self._as_of("USSLIND")
        vix_risk, vix_display, _ = self.score("VIXCLS")
        vix_as_of = self._as_of("VIXCLS")
        debt_risk, debt_display, _ = self.score("GFDEGDQ188S")
        debt_as_of = self._as_of("GFDEGDQ188S")
        cape_risk, cape_display, _ = self.score("SP500_CAPE")
        cape_as_of = self._as_of("SP500_CAPE")
        pe_risk, pe_display, _ = self.score("SP500_PE")
        pe_as_of = self._as_of("SP500_PE")

        def composite(*levels):
            return _RISK_FROM_INT[max(_RISK_ORDER.get(lv, 0) for lv in levels)]

        return {
            "Treasury Stress": {
                "score": composite(y10_risk, yc_risk, cre_risk),
                "components": [
                    ("10Y Yield", y10_risk, y10_display, y10_as_of),
                    ("Yield Curve", yc_risk, yc_display, yc_as_of),
                    ("CRE Lending Stds", cre_risk, cre_display, cre_as_of),
                ],
                "description": (
                    "Combines the 10-year Treasury yield level, yield curve slope (inverted "
                    "curve has preceded every recession since 1970), and CRE lending standards. "
                    "High yields amplify debt service burdens; curve inversion signals "
                    "tighter credit conditions ahead."
                ),
            },
            "Recession Risk": {
                "score": composite(rec_risk, yc_risk),
                "components": [
                    ("NY Fed Rec. Prob.", rec_risk, rec_display, rec_as_of),
                    ("Yield Curve", yc_risk, yc_display, yc_as_of),
                ],
                "description": (
                    "NY Fed recession probability model (yield-curve based) combined with "
                    "the current curve slope. Above 30-40% probability has historically "
                    "been a reliable 12-month leading indicator of NBER recession."
                ),
            },
            "Inflation Persistence": {
                "score": composite(core_risk, cpi_risk),
                "components": [
                    ("Core CPI YoY", core_risk, core_display, core_as_of),
                    ("Headline CPI YoY", cpi_risk, cpi_display, cpi_as_of),
                ],
                "description": (
                    "Tracks whether inflation is above the Fed's 2% target and whether "
                    "it is broad-based (headline) or structural (core, ex food & energy). "
                    "Persistent core inflation is more concerning than headline spikes."
                ),
            },
            "Labor Weakness": {
                "score": composite(un_risk, pay_risk, usslind_risk),
                "components": [
                    ("Unemployment Rate", un_risk, un_display, un_as_of),
                    ("Payrolls MoM", pay_risk, pay_display, pay_as_of),
                    ("Leading Econ Index", usslind_risk, usslind_display, usslind_as_of),
                ],
                "description": (
                    "Combines the unemployment rate and monthly payroll growth with the "
                    "Conference Board Leading Economic Index (MoM change). The LEI provides "
                    "a forward-looking composite signal that typically leads labor deterioration "
                    "by 3-6 months."
                ),
            },
            "Financial Stress": {
                "score": composite(vix_risk, cre_risk),
                "components": [
                    ("VIX", vix_risk, vix_display, vix_as_of),
                    ("CRE Lending Stds", cre_risk, cre_display, cre_as_of),
                ],
                "description": (
                    "Equity market volatility (VIX) captures near-term market sentiment; "
                    "CRE lending standards (SLOOS) reflect forward-looking credit tightening "
                    "in commercial real estate. Simultaneous stress in both signals broad "
                    "financial deterioration."
                ),
            },
            "Debt Sustainability": {
                "score": composite(debt_risk, y10_risk),
                "components": [
                    ("Federal Debt / GDP", debt_risk, debt_display, debt_as_of),
                    ("10Y Yield", y10_risk, y10_display, y10_as_of),
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
                    ("CAPE (Shiller P/E)", cape_risk, cape_display, cape_as_of),
                    ("Trailing P/E", pe_risk, pe_display, pe_as_of),
                ],
                "description": (
                    "Elevated valuations do not predict the timing of corrections, but "
                    "they reduce the margin of safety: the higher the starting CAPE, the "
                    "lower the historical probability of strong 10-year forward real returns. "
                    "CAPE above 30 has historically been followed by decade-long below-average returns."
                ),
            },
        }

    # ── Trend & History ────────────────────────────────────────────────────

    def trend_arrow(self, series_id: str) -> str:
        """Return ↑, ↓, or → based on 3-month directional trend."""
        direction = self.dl.get_trend(series_id, months=3)
        return {"rising": "↑", "falling": "↓"}.get(direction, "→")

    def trend_adjusted_score(self, series_id: str) -> tuple[str, str, float | None]:
        """
        Like score() but applies trend weighting:
          - yellow + worsening trend → red
          - red + improving trend   → yellow
        'Worsening' means rising if higher_is_bad=True, falling otherwise.
        """
        risk, display, value = self.score(series_id)
        if risk in ("neutral", "green"):
            return risk, display, value
        meta = REGISTRY.get(series_id, {})
        higher_is_bad = meta.get("higher_is_bad", True)
        trend = self.dl.get_trend(series_id, months=3)
        worsening = (higher_is_bad and trend == "rising") or (not higher_is_bad and trend == "falling")
        improving  = (higher_is_bad and trend == "falling") or (not higher_is_bad and trend == "rising")
        if risk == "yellow" and worsening:
            return "red", display, value
        if risk == "red" and improving:
            return "yellow", display, value
        return risk, display, value

    def score_history(self, series_id: str, lookback_years: int = 10) -> pd.Series:
        """Monthly time series of risk scores: 3=red, 2=yellow, 1=green, 0=neutral."""
        meta = REGISTRY.get(series_id)
        if not meta:
            return pd.Series(dtype=float)
        df = self.dl.load(series_id)
        if df is None or df.empty:
            return pd.Series(dtype=float)

        basis = meta.get("risk_basis", "level")
        try:
            monthly = df.resample("ME").last().dropna()
        except ValueError:
            monthly = df.resample("M").last().dropna()

        if basis == "yoy":
            values = (monthly.pct_change(12) * 100).dropna()
        elif basis == "mom_change":
            values = monthly.diff().dropna()
        else:
            values = monthly

        if values.empty:
            return pd.Series(dtype=float)

        cutoff = values.index[-1] - pd.DateOffset(years=lookback_years)
        values = values[values.index >= cutoff]
        col = values.columns[0]
        return values[col].apply(
            lambda v: _RISK_ORDER.get(self.score_value(series_id, v), 0)
        )

    # ── Derived Metric Helpers ─────────────────────────────────────────────

    def get_real_fed_funds_rate(self) -> float | None:
        """Real Fed Funds Rate = FEDFUNDS level minus Core CPI YoY."""
        ff, _ = self.dl.get_latest("FEDFUNDS")
        core_cpi = self.dl.get_yoy("CPILFESL")
        if ff is None or core_cpi is None:
            return None
        return ff - core_cpi

    def get_interest_receipts_ratio(self) -> tuple[float | None, object]:
        """Federal interest payments / current receipts * 100 (%)."""
        df_int = self.dl.load("A091RC1Q027SBEA")
        df_rec = self.dl.load("W006RC1Q027SBEA")
        if df_int is None or df_rec is None:
            return None, None
        combined = pd.concat([df_int, df_rec], axis=1).dropna()
        if combined.empty:
            return None, None
        latest = combined.iloc[-1]
        ratio = float(latest.iloc[0] / latest.iloc[1] * 100)
        return ratio, combined.index[-1]

    def get_walcl_pct_gdp(self) -> float | None:
        """Fed balance sheet (WALCL, millions) as % of nominal GDP (billions)."""
        df_walcl = self.dl.load("WALCL")
        df_gdp = self.dl.load("GDP")
        if df_walcl is None or df_gdp is None:
            return None
        try:
            walcl_q = df_walcl.resample("QE").last().dropna()
            gdp_q = df_gdp.resample("QE").last().dropna()
        except ValueError:
            walcl_q = df_walcl.resample("Q").last().dropna()
            gdp_q = df_gdp.resample("Q").last().dropna()
        combined = pd.concat([walcl_q, gdp_q], axis=1).dropna()
        if combined.empty:
            return None
        latest = combined.iloc[-1]
        return float(latest.iloc[0] / (latest.iloc[1] * 1000) * 100)

    # ── System Resilience Dimensions ───────────────────────────────────────

    def system_resilience_dimensions(self) -> dict[str, dict]:
        """
        Composite risk scores for the four System Resilience dimensions:
        Liquidity & Funding, Credit Functionality, Policy Flexibility, and
        Inflation Anchoring.  Returns same structure as crisis_dimensions().
        """
        def composite(*levels):
            return _RISK_FROM_INT[max(_RISK_ORDER.get(lv, 0) for lv in levels)]

        # Liquidity & Funding
        fsi_risk, fsi_disp, _ = self.score("STLFSI4")
        fsi_as_of = self._as_of("STLFSI4")
        cp_tsy_risk, cp_tsy_disp, cp_tsy_as_of = self.cp_treasury_spread()
        rrp_risk, rrp_disp, _ = self.score("RRPONTSYD")
        rrp_as_of = self._as_of("RRPONTSYD")
        nfci_risk, nfci_disp, _ = self.score("NFCI")
        nfci_as_of = self._as_of("NFCI")

        # Credit Functionality
        hy_risk, hy_disp, _ = self.score("BAMLH0A0HYM2")
        hy_as_of = self._as_of("BAMLH0A0HYM2")
        ig_risk, ig_disp, _ = self.score("BAMLC0A0CM")
        ig_as_of = self._as_of("BAMLC0A0CM")
        nfci_cr_risk, nfci_cr_disp, _ = self.score("NFCICREDIT")
        nfci_cr_as_of = self._as_of("NFCICREDIT")
        cre_risk, cre_disp, _ = self.score("SUBLPDRCSN")
        cre_as_of = self._as_of("SUBLPDRCSN")
        lend_risk, lend_disp, _ = self.score("DRTSCILM")
        lend_as_of = self._as_of("DRTSCILM")

        # Inflation Anchoring / Policy Flexibility
        t5y_risk, t5y_disp, _ = self.score("T5YIE")
        t5y_as_of = self._as_of("T5YIE")
        t10y_risk, t10y_disp, _ = self.score("T10YIE")
        t10y_as_of = self._as_of("T10YIE")

        # Interest / Receipts derived ratio
        int_rec, int_rec_date = self.get_interest_receipts_ratio()
        int_rec_disp = f"{int_rec:.1f}%" if int_rec is not None else "N/A"
        int_rec_as_of = (int_rec_date.strftime(DASH.date_display_fmt)
                         if int_rec_date is not None else "")
        if int_rec is None:
            int_rec_risk = "neutral"
        elif int_rec < 12:
            int_rec_risk = "green"
        elif int_rec < 20:
            int_rec_risk = "yellow"
        else:
            int_rec_risk = "red"

        # Real Fed Funds Rate
        real_ff = self.get_real_fed_funds_rate()
        real_ff_disp = f"{real_ff:.1f}%" if real_ff is not None else "N/A"
        ff_as_of = self._as_of("FEDFUNDS")
        if real_ff is None:
            real_ff_risk = "neutral"
        elif real_ff < -1:
            real_ff_risk = "yellow"   # deeply negative = stimulative/inflationary
        elif real_ff <= 4:
            real_ff_risk = "green"
        else:
            real_ff_risk = "yellow"   # very high real rate = demand-killing

        return {
            "Liquidity & Funding": {
                "score": composite(fsi_risk, cp_tsy_risk, nfci_risk, rrp_risk),
                "components": [
                    ("Financial Stress Index", fsi_risk, fsi_disp, fsi_as_of),
                    ("Nat'l Financial Conditions", nfci_risk, nfci_disp, nfci_as_of),
                    ("CP-Treasury Spread", cp_tsy_risk, cp_tsy_disp, cp_tsy_as_of),
                    ("Fed Reverse Repo ($B)", rrp_risk, rrp_disp, rrp_as_of),
                ],
                "description": (
                    "Aggregates the St. Louis Financial Stress Index, Chicago National "
                    "Financial Conditions Index, 3-Month CP-Treasury spread, and Fed "
                    "reverse repo usage. Deterioration here appears before stress reaches "
                    "equity markets or unemployment data."
                ),
            },
            "Credit Functionality": {
                "score": composite(hy_risk, ig_risk, nfci_cr_risk, lend_risk, cre_risk),
                "components": [
                    ("High Yield Spread", hy_risk, hy_disp, hy_as_of),
                    ("Investment Grade Spread", ig_risk, ig_disp, ig_as_of),
                    ("C&I Loan Standards", lend_risk, lend_disp, lend_as_of),
                    ("CRE Loan Standards", cre_risk, cre_disp, cre_as_of),
                    ("Financial Conditions: Credit", nfci_cr_risk, nfci_cr_disp, nfci_cr_as_of),
                ],
                "description": (
                    "Tracks refinancing capacity via high yield and investment grade spreads, "
                    "SLOOS C&I and CRE loan standards, and the National Financial Conditions "
                    "credit subindex. Bank tightening is a leading indicator of credit "
                    "contraction 6-12 months ahead."
                ),
            },
            "Policy Flexibility": {
                "score": composite(t5y_risk, t10y_risk, int_rec_risk),
                "components": [
                    ("5-Yr Inflation Expectations", t5y_risk, t5y_disp, t5y_as_of),
                    ("10-Yr Inflation Expectations", t10y_risk, t10y_disp, t10y_as_of),
                    ("Interest / Govt Receipts", int_rec_risk, int_rec_disp, int_rec_as_of),
                    ("Real Fed Funds Rate", real_ff_risk, real_ff_disp, ff_as_of),
                ],
                "description": (
                    "Measures whether policymakers retain stabilization flexibility. "
                    "Unanchored inflation expectations limit monetary easing. A rising "
                    "interest-to-receipts ratio constrains fiscal space regardless of debt "
                    "levels."
                ),
            },
        }

    # ── System Absorption Capacity ─────────────────────────────────────────

    def system_absorption_capacity(self) -> dict[str, dict]:
        """
        Five-category summary of the system's shock-absorption capacity.
        Returns dict keyed by category name with 'score', 'key_metric', 'value'.
        """
        def composite(*levels):
            return _RISK_FROM_INT[max(_RISK_ORDER.get(lv, 0) for lv in levels)]

        fsi_risk, fsi_disp, _ = self.score("STLFSI4")
        nfci_risk, nfci_disp, _ = self.score("NFCI")
        cp_tsy_risk, cp_tsy_disp, _ = self.cp_treasury_spread()
        rrp_risk, rrp_disp, _ = self.score("RRPONTSYD")
        hy_risk, hy_disp, _ = self.score("BAMLH0A0HYM2")
        ig_risk, ig_disp, _ = self.score("BAMLC0A0CM")
        nfci_cr_risk, nfci_cr_disp, _ = self.score("NFCICREDIT")
        cre_lend_risk, cre_lend_disp, _ = self.score("SUBLPDRCSN")
        t5y_risk, t5y_disp, _ = self.score("T5YIE")
        t10y_risk, t10y_disp, _ = self.score("T10YIE")

        int_rec, _ = self.get_interest_receipts_ratio()
        int_rec_disp = f"{int_rec:.1f}%" if int_rec is not None else "N/A"
        if int_rec is None:
            debt_service_risk = "neutral"
        elif int_rec < 12:
            debt_service_risk = "green"
        elif int_rec < 20:
            debt_service_risk = "yellow"
        else:
            debt_service_risk = "red"

        vix_risk, vix_disp, _ = self.score("VIXCLS")
        debt_risk, debt_disp, _ = self.score("GFDEGDQ188S")

        return {
            "Liquidity": {
                "score": composite(fsi_risk, nfci_risk),
                "key_metric": "Financial Stress / Nat'l Financial Conditions",
                "value": f"{fsi_disp} / {nfci_disp}",
            },
            "Funding Stability": {
                "score": composite(cp_tsy_risk, rrp_risk),
                "key_metric": "CP-Treasury Spread / Reverse Repo",
                "value": f"{cp_tsy_disp} / {rrp_disp}",
            },
            "Policy Flexibility": {
                "score": composite(debt_service_risk, debt_risk),
                "key_metric": "Interest / Govt Receipts",
                "value": int_rec_disp,
            },
            "Credit Functionality": {
                "score": composite(hy_risk, ig_risk, nfci_cr_risk, cre_lend_risk),
                "key_metric": "High Yield Spread / CRE Loan Standards",
                "value": f"{hy_disp} / {cre_lend_disp}",
            },
            "Inflation Anchoring": {
                "score": composite(t5y_risk, t10y_risk),
                "key_metric": "5Y / 10Y Breakeven Inflation",
                "value": f"{t5y_disp} / {t10y_disp}",
            },
        }

    # ── Refinancing & Liquidity Risk Warning ───────────────────────────────

    def refinancing_liquidity_risk(self) -> tuple[str, list[str]]:
        """
        Multi-indicator assessment of combined refinancing and liquidity risk.
        Returns (risk_level: str, triggers: list[str]).
        Persistent combinations matter more than single-indicator spikes.
        """
        triggers = []

        hy_risk, _, hy_val = self.score("BAMLH0A0HYM2")
        hy_trend = self.dl.get_trend("BAMLH0A0HYM2", months=3)
        if hy_risk == "red":
            triggers.append("HY spreads at stressed levels (> 700 bps) — refinancing access impaired")
        elif hy_risk == "yellow" and hy_trend == "rising":
            triggers.append("HY spreads elevated and widening — credit conditions tightening")

        fsi_risk, _, _ = self.score("STLFSI4")
        if fsi_risk == "red":
            triggers.append("St. Louis Financial Stress Index significantly elevated")
        elif fsi_risk == "yellow":
            fsi_trend = self.dl.get_trend("STLFSI4", months=4)
            if fsi_trend == "rising":
                triggers.append("Financial Stress Index elevated and rising — systemic pressure building")

        nfci_risk, _, _ = self.score("NFCI")
        if nfci_risk in ("yellow", "red"):
            triggers.append("National Financial Conditions Index showing tighter-than-average conditions")

        vix_risk, _, vix_val = self.score("VIXCLS")
        if vix_risk == "red":
            vix_trend = self.dl.get_trend("VIXCLS", months=4)
            if vix_trend != "falling":
                triggers.append("VIX at stressed levels (> 30) with no clear abatement — market confidence impaired")

        y10_trend = self.dl.get_trend("DGS10", months=6)
        hy_trend_6m = self.dl.get_trend("BAMLH0A0HYM2", months=6)
        if y10_trend == "rising" and hy_trend_6m == "rising":
            triggers.append("Dual tightening: rising Treasury yields + widening credit spreads")

        t10y_risk, _, _ = self.score("T10YIE")
        if t10y_risk == "red":
            triggers.append("10-year inflation expectations above 3% — monetary policy space severely constrained")
        elif t10y_risk == "yellow":
            triggers.append("Long-run inflation expectations elevated — limits Fed's ability to stabilize markets")

        rrp_risk, _, _ = self.score("RRPONTSYD")
        if rrp_risk == "red":
            rrp_trend = self.dl.get_trend("RRPONTSYD", months=6)
            if rrp_trend == "falling":
                triggers.append("Fed reverse repo near depletion and declining — systemic liquidity cushion thinning")

        int_rec, _ = self.get_interest_receipts_ratio()
        if int_rec is not None:
            if int_rec > 20:
                triggers.append(
                    f"Federal debt service at {int_rec:.1f}% of receipts — "
                    "fiscal stabilization capacity materially constrained"
                )
            elif int_rec > 15:
                triggers.append(f"Federal interest / receipts at {int_rec:.1f}% — elevated and rising trajectory")

        n = len(triggers)
        if n == 0:
            level = "green"
        elif n <= 2:
            level = "yellow"
        else:
            level = "red"

        return level, triggers

    # ── Executive Narrative ────────────────────────────────────────────────

    def executive_narrative(self) -> list[tuple[str, str]]:
        """
        Returns list of (risk_level, sentence) tuples for the executive summary page.
        Covers six dimensions: inflation, labor, credit, valuation, fiscal/policy, liquidity.
        """
        results = []

        # Inflation
        core_risk, core_disp, _ = self.score("CPILFESL")
        if core_risk == "green":
            results.append(("green",
                f"Core inflation ({core_disp} YoY) has returned within the Fed's 2% target range — "
                "monetary policy has room to respond to economic weakness if needed."))
        elif core_risk == "yellow":
            results.append(("yellow",
                f"Core inflation ({core_disp} YoY) remains above target, limiting policy flexibility. "
                "The Fed cannot ease without risking re-acceleration."))
        else:
            results.append(("red",
                f"Core inflation ({core_disp} YoY) is significantly elevated — the primary binding "
                "constraint on monetary policy. Rate cuts would risk entrenching inflation."))

        # Labor
        un_risk, un_disp, _ = self.score("UNRATE")
        pay_risk, pay_disp, _ = self.score("PAYEMS")
        labor_worst = _RISK_FROM_INT[max(_RISK_ORDER.get(un_risk, 0), _RISK_ORDER.get(pay_risk, 0))]
        if labor_worst == "green":
            results.append(("green",
                f"Labor market solid: unemployment at {un_disp}, payrolls adding {pay_disp}/month. "
                "No demand-destruction signal in employment data."))
        elif labor_worst == "red":
            results.append(("red",
                f"Labor market deteriorating: unemployment {un_disp}, payrolls {pay_disp}/month. "
                "Demand destruction in employment is consistent with late-cycle contraction."))
        else:
            results.append(("yellow",
                f"Labor market showing mixed signals: unemployment {un_disp}, payrolls {pay_disp}/month. "
                "Monitor for softening trend rather than current levels."))

        # Yield curve / recession risk
        yc_risk, yc_disp, _ = self.score("T10Y2Y")
        rec_risk, rec_disp, _ = self.score("RECPROUSM156N")
        yc_worst = _RISK_FROM_INT[max(_RISK_ORDER.get(yc_risk, 0), _RISK_ORDER.get(rec_risk, 0))]
        if yc_worst == "green":
            results.append(("green",
                f"Yield curve positively sloped ({yc_disp}) with low recession probability ({rec_disp}%). "
                "Historical leading indicators not signaling near-term contraction."))
        elif yc_worst == "red":
            results.append(("red",
                f"Yield curve inverted/flat ({yc_disp}); NY Fed recession probability elevated ({rec_disp}%). "
                "Leading indicators consistent with elevated recession risk in the next 12 months."))
        else:
            results.append(("yellow",
                f"Yield curve flat or partially inverted ({yc_disp}); recession probability {rec_disp}%. "
                "Historical signal warrants monitoring — inversion has preceded every post-1970 recession."))

        # Credit spreads
        hy_risk, hy_disp, _ = self.score("BAMLH0A0HYM2")
        ig_risk, ig_disp, _ = self.score("BAMLC0A0CM")
        credit_worst = _RISK_FROM_INT[max(_RISK_ORDER.get(hy_risk, 0), _RISK_ORDER.get(ig_risk, 0))]
        if credit_worst == "green":
            results.append(("green",
                f"Credit markets functional: HY spreads {hy_disp}, IG spreads {ig_disp}. "
                "Refinancing conditions normal — no systemic funding stress."))
        elif credit_worst == "red":
            results.append(("red",
                f"Credit stress: HY spreads {hy_disp}, IG spreads {ig_disp}. "
                "Elevated spreads impair rollover financing and signal genuine default risk pricing."))
        else:
            results.append(("yellow",
                f"Credit spreads modestly elevated: HY {hy_disp}, IG {ig_disp}. "
                "Not alarming in isolation — becomes significant if sustained or widening."))

        # CRE lending standards
        cre_risk, cre_disp, _ = self.score("SUBLPDRCSN")
        if cre_risk == "green":
            results.append(("green",
                f"CRE lending standards easing or stable ({cre_disp} net tightening). "
                "Banks not restricting commercial real estate credit — no forward stress signal."))
        elif cre_risk == "red":
            results.append(("red",
                f"CRE lending standards significantly tightened ({cre_disp} net tightening). "
                "Banks sharply restricting CRE credit — a leading signal that typically precedes "
                "delinquency rises and CRE value declines by 6-12 months."))
        else:
            results.append(("yellow",
                f"CRE lending standards moderately tightened ({cre_disp} net tightening). "
                "Banks showing elevated caution on commercial real estate — monitor trajectory."))

        # Valuation
        cape_risk, cape_disp, _ = self.score("SP500_CAPE")
        if cape_risk == "green":
            results.append(("green",
                f"Equity valuations within normal historical range (CAPE {cape_disp}). "
                "Starting valuations consistent with acceptable forward return expectations."))
        elif cape_risk == "red":
            results.append(("red",
                f"Equity valuations at historically stressed levels (CAPE {cape_disp}). "
                "Among the highest readings of the past century — historically followed by "
                "decade-long below-average real returns. CAPE does not predict timing."))
        else:
            results.append(("yellow",
                f"Equity valuations elevated (CAPE {cape_disp}). Above long-run average of ~17; "
                "historically associated with below-average 10-year forward returns."))

        # Policy flexibility (fiscal + monetary)
        int_rec, _ = self.get_interest_receipts_ratio()
        t10y_risk, t10y_disp, _ = self.score("T10YIE")
        if int_rec is not None:
            if int_rec < 12 and t10y_risk == "green":
                results.append(("green",
                    f"Policy flexibility intact: debt service at {int_rec:.1f}% of receipts; "
                    f"10-year inflation expectations anchored at {t10y_disp}. "
                    "Both monetary and fiscal tools remain available."))
            elif int_rec > 20 or t10y_risk == "red":
                results.append(("red",
                    f"Policy flexibility severely constrained: debt service {int_rec:.1f}% of receipts; "
                    f"inflation expectations {t10y_disp}. Fiscal and monetary stabilization tools "
                    "are simultaneously impaired."))
            else:
                results.append(("yellow",
                    f"Policy flexibility moderately constrained: debt service {int_rec:.1f}% of receipts; "
                    f"inflation expectations {t10y_disp}. Room exists, but margin is narrowing."))

        return results

    def overall_stress_level(self) -> str:
        """
        Aggregate risk level across all crisis dimensions.
        Applies trend-weighting: yellow+worsening series vote as red for the aggregate.
        """
        dims = self.crisis_dimensions()
        counts = {"red": 0, "yellow": 0, "green": 0, "neutral": 0}
        for d in dims.values():
            counts[d["score"]] = counts.get(d["score"], 0) + 1

        # Trend-weight key series: each yellow-worsening adds a fractional upgrade
        trend_upgrades = sum(
            1 for sid in ("UNRATE", "BAMLH0A0HYM2", "CPILFESL", "DGS10", "STLFSI4")
            if self.trend_adjusted_score(sid)[0] == "red" and self.score(sid)[0] == "yellow"
        )
        if counts["red"] >= 2 or (counts["red"] >= 1 and trend_upgrades >= 1):
            return "red"
        if counts["red"] >= 1 or counts["yellow"] >= 3 or trend_upgrades >= 2:
            return "yellow"
        if counts["yellow"] >= 1:
            return "yellow"
        return "green"

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
