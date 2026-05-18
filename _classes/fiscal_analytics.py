"""
Fiscal Analytics Engine
Advanced fiscal sustainability metrics: debt service burden, primary
balance, fiscal trajectory, rollover exposure, and impulse analysis.

Avoids simplistic debt-level interpretations: the key sustainability
questions are the interest burden relative to revenues, the primary
balance trajectory, and the refinancing cost environment.

Data sources (all FRED):
- A091RC1Q027SBEA — Federal Interest Payments (Quarterly SAAR, $B)
- W006RC1Q027SBEA — Federal Current Receipts (Quarterly SAAR, $B)
- GDP             — Nominal GDP (Quarterly SAAR, $B)
- GFDEGDQ188S     — Federal Debt % of GDP (Quarterly)
- GFDEBTN         — Federal Debt Outstanding ($B, Monthly)
- FYFSD           — Federal Surplus/Deficit (Annual FY, $B)
- FYFSGDA188S          — Federal Surplus/Deficit as % GDP (Annual FY)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class FiscalAnalyticsEngine:
    """Advanced fiscal sustainability analytics from BEA/Treasury FRED data."""

    def __init__(self, dl):
        self._dl = dl

    # ── Internal helpers ───────────────────────────────────────────────────

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

    def _annual(self, sid: str) -> pd.Series | None:
        df = self._dl.load(sid)
        if df is None or df.empty:
            return None
        col = df.columns[0]
        try:
            a = df.resample("YE").last()
        except ValueError:
            a = df.resample("Y").last()
        return a[col].dropna()

    def _slope_ann(self, s: pd.Series, quarters: int = 20) -> float | None:
        """Annualized linear trend over the last N quarters."""
        if s is None or len(s) < quarters:
            return None
        vals = s.iloc[-quarters:].values
        if np.any(np.isnan(vals)):
            return None
        x = np.arange(len(vals), dtype=float)
        return float(np.polyfit(x, vals, 1)[0] * 4)  # quarterly → annual rate

    # ── Debt Sustainability ───────────────────────────────────────────────

    def debt_sustainability(self) -> dict:
        """
        Core sustainability metrics.

        Interest/Receipts is the most operationally meaningful ratio:
        it measures what fraction of government revenues is consumed by
        debt service — the margin left for all other government functions.
        Above 20% historically precedes fiscal stress; above 30% is critical.

        Interest/GDP situates the burden in economic context.
        Debt/GDP is a stock measure and should not be read in isolation.
        """
        debt_gdp_q  = self._quarterly("GFDEGDQ188S")
        int_pay_q   = self._quarterly("A091RC1Q027SBEA")
        receipts_q  = self._quarterly("W006RC1Q027SBEA")
        gdp_q       = self._quarterly("GDP")

        debt_gdp       = float(debt_gdp_q.iloc[-1]) if (debt_gdp_q is not None and not debt_gdp_q.empty) else None
        int_pay        = float(int_pay_q.iloc[-1])  if (int_pay_q  is not None and not int_pay_q.empty)  else None
        receipts       = float(receipts_q.iloc[-1]) if (receipts_q is not None and not receipts_q.empty) else None
        gdp            = float(gdp_q.iloc[-1])      if (gdp_q      is not None and not gdp_q.empty)      else None

        int_gdp       = (int_pay / gdp * 100)       if (int_pay and gdp)      else None
        int_receipts  = (int_pay / receipts * 100)  if (int_pay and receipts) else None

        label = "Unknown"
        if int_receipts is not None:
            if int_receipts > 25:
                label = "Critical — Debt Service Crowding Out"
            elif int_receipts > 18:
                label = "Stressed"
            elif int_receipts > 12:
                label = "Elevated"
            else:
                label = "Manageable"

        return {
            "debt_gdp":           debt_gdp,
            "int_gdp":            int_gdp,
            "int_receipts":       int_receipts,
            "int_pay_B":          int_pay,
            "receipts_B":         receipts,
            "gdp_B":              gdp,
            "label":              label,
        }

    # ── Primary Balance ───────────────────────────────────────────────────

    def primary_balance(self) -> dict:
        """
        Primary balance = total budget balance + interest payments.
        A positive primary balance means the government covers its non-interest
        spending from revenues — a necessary (not sufficient) condition for
        long-run debt stabilization.

        Debt/GDP stabilizes when:
          primary surplus ≥ (r − g) × debt/GDP
        where r = real interest rate, g = real GDP growth rate.
        """
        deficit_a   = self._annual("FYFSD")       # total deficit $B (negative = deficit)
        deficit_pct = self._annual("FYFSGDA188S")       # total deficit % GDP (negative = deficit)
        int_pay_q   = self._quarterly("A091RC1Q027SBEA")
        gdp_q       = self._quarterly("GDP")

        total_deficit   = float(deficit_a.iloc[-1])   if (deficit_a   is not None and not deficit_a.empty)   else None
        deficit_pct_cur = float(deficit_pct.iloc[-1]) if (deficit_pct is not None and not deficit_pct.empty) else None

        # Annualize quarterly interest: SAAR series, so last observation ≈ annual rate
        int_annual = float(int_pay_q.iloc[-1]) if (int_pay_q is not None and not int_pay_q.empty) else None

        primary_B   = None
        primary_pct = None
        if total_deficit is not None and int_annual is not None:
            # Primary deficit = total deficit + interest payments
            # (deficit is negative, so adding interest reduces the magnitude)
            primary_B = total_deficit + int_annual
            gdp_cur = float(gdp_q.iloc[-1]) if (gdp_q is not None and not gdp_q.empty) else None
            if gdp_cur:
                primary_pct = primary_B / gdp_cur * 100

        return {
            "total_deficit_B":   total_deficit,
            "deficit_pct_gdp":   deficit_pct_cur,
            "interest_B":        int_annual,
            "primary_balance_B": primary_B,
            "primary_pct_gdp":   primary_pct,
        }

    # ── Fiscal Trajectory ─────────────────────────────────────────────────

    def fiscal_trajectory(self) -> dict:
        """
        5-year trend in key fiscal metrics.
        The critical question is not the level but the direction:
        is the interest burden growing faster than revenues?
        If interest acceleration > revenue growth, the debt spiral tightens.
        """
        debt_gdp_q = self._quarterly("GFDEGDQ188S")
        int_pay_q  = self._quarterly("A091RC1Q027SBEA")
        receipts_q = self._quarterly("W006RC1Q027SBEA")

        debt_trend   = self._slope_ann(debt_gdp_q, 20)
        int_acc      = self._slope_ann(int_pay_q,  20)
        rec_trend    = self._slope_ann(receipts_q, 20)

        # Interest acceleration relative to receipts growth
        accel_gap = None
        if int_acc is not None and rec_trend is not None and rec_trend > 0:
            accel_gap = int_acc / rec_trend * 100  # % of revenue growth captured by interest

        label = "Unknown"
        if debt_trend is not None:
            label = ("Rapidly Deteriorating"  if debt_trend > 3  else
                     "Gradually Deteriorating" if debt_trend > 1  else
                     "Broadly Stable"          if debt_trend > -1 else
                     "Improving")

        return {
            "debt_gdp_trend_ann":   debt_trend,
            "interest_acc_ann":     int_acc,
            "receipts_trend_ann":   rec_trend,
            "acceleration_gap_pct": accel_gap,
            "label":                label,
        }

    # ── Rollover Exposure ─────────────────────────────────────────────────

    def rollover_exposure(self) -> dict:
        """
        Estimates refinancing cost pressure from the rate environment.
        FRED does not publish Treasury maturity schedules, so we proxy via:
        (1) 10Y yield change from 5 years ago — cost of rolling maturing debt.
        (2) Total debt outstanding × rate change × assumed 30% annual rollover
            (consistent with ~6Y average maturity of the Federal debt).

        Higher rate environments = more expensive rollovers = rising interest burden
        even without new net borrowing.
        """
        dgs10  = self._monthly("DGS10")
        debt_q = self._quarterly("GFDEBTN")

        current_rate = None
        rate_5y_ago  = None
        rate_change  = None
        debt_current = None
        rollover_est = None

        if dgs10 is not None and not dgs10.empty:
            current_rate = float(dgs10.iloc[-1])
            if len(dgs10) > 60:
                rate_5y_ago = float(dgs10.iloc[-61])
                rate_change = current_rate - rate_5y_ago

        if debt_q is not None and not debt_q.empty:
            debt_current = float(debt_q.iloc[-1])

        if rate_change is not None and debt_current is not None:
            rollover_est = rate_change * 0.30 * debt_current / 100  # $B annual incremental cost

        label = "Unknown"
        if rate_change is not None:
            label = ("High Rollover Cost Pressure"    if rate_change > 2.5  else
                     "Moderate Rollover Pressure"     if rate_change > 1.0  else
                     "Manageable"                     if rate_change > -0.5 else
                     "Favorable — Rate Decline")

        return {
            "current_10y_rate":          current_rate,
            "rate_5y_ago":               rate_5y_ago,
            "rate_change_5y":            rate_change,
            "debt_total_B":              debt_current,
            "est_incremental_cost_ann_B": rollover_est,
            "label":                     label,
        }

    # ── Fiscal Impulse ────────────────────────────────────────────────────

    def fiscal_impulse(self) -> dict:
        """
        Fiscal impulse = annual change in the deficit/GDP ratio (sign-flipped).
        Positive impulse = government is adding stimulus (deficit expanding).
        Negative impulse = fiscal drag (deficit shrinking or surplus growing).
        """
        deficit_pct = self._annual("FYFSGDA188S")
        if deficit_pct is None or len(deficit_pct) < 2:
            return {"impulse_current": None, "impulse_series": None}

        impulse_s = deficit_pct.diff()  # YoY change in deficit % GDP
        # Sign convention: expanding deficit = positive impulse (stimulus)
        impulse_current = float(impulse_s.iloc[-1]) if not impulse_s.dropna().empty else None
        return {
            "impulse_current": impulse_current,
            "impulse_series":  impulse_s.dropna(),
        }

    # ── r-g Dynamics ─────────────────────────────────────────────────────

    def r_g_dynamics(self) -> dict:
        """
        r-g spread: nominal interest rate (10Y Treasury) minus nominal GDP growth.

        When r > g, debt/GDP automatically expands even if the primary budget is
        balanced, because interest compounds faster than the economy grows.
        The Domar condition for debt stabilisation:
          primary surplus ≥ (r − g) × debt/GDP

        Also computes real r-g (TIPS yield minus real GDP growth) and the
        effective interest rate the Treasury actually pays on outstanding debt.
        """
        gs10    = self._monthly("GS10")
        gdp     = self._quarterly("GDP")
        dfii10  = self._monthly("DFII10")
        gdpc1   = self._quarterly("GDPC1")
        int_pay = self._quarterly("A091RC1Q027SBEA")
        debt_m  = self._monthly("GFDEBTN")

        # Nominal r
        nom_r = float(gs10.iloc[-1]) if (gs10 is not None and not gs10.empty) else None

        # Nominal g (quarterly YoY → take most recent quarter)
        nom_g = None
        if gdp is not None and len(gdp) >= 5:
            yoy = (gdp.pct_change(4) * 100).dropna()
            nom_g = float(yoy.iloc[-1]) if not yoy.empty else None

        # Real r (TIPS)
        real_r = float(dfii10.iloc[-1]) if (dfii10 is not None and not dfii10.empty) else None

        # Real g
        real_g = None
        if gdpc1 is not None and len(gdpc1) >= 5:
            yoy_r = (gdpc1.pct_change(4) * 100).dropna()
            real_g = float(yoy_r.iloc[-1]) if not yoy_r.empty else None

        nom_rg  = (nom_r  - nom_g)  if (nom_r  is not None and nom_g  is not None) else None
        real_rg = (real_r - real_g) if (real_r is not None and real_g is not None) else None

        # Effective interest rate: interest payments (SAAR) / debt outstanding × 100
        effective_rate = None
        if int_pay is not None and not int_pay.empty and debt_m is not None and not debt_m.empty:
            try:
                try:
                    debt_q = debt_m.resample("QE").last()
                except ValueError:
                    debt_q = debt_m.resample("Q").last()
                common = int_pay.index.intersection(debt_q.index)
                if len(common) > 0:
                    eff = (int_pay.loc[common] / debt_q.loc[common] * 100).dropna()
                    effective_rate = float(eff.iloc[-1]) if not eff.empty else None
            except Exception:
                pass

        label = "Unknown"
        if nom_rg is not None:
            if nom_rg > 3:
                label = "Debt-Destabilizing — r ≫ g"
            elif nom_rg > 1:
                label = "Debt-Expanding — r > g"
            elif nom_rg > -1:
                label = "Near-Neutral"
            else:
                label = "Debt-Stabilizing — g > r"

        return {
            "nominal_r":      nom_r,
            "nominal_g":      nom_g,
            "nominal_rg":     nom_rg,
            "real_r":         real_r,
            "real_g":         real_g,
            "real_rg":        real_rg,
            "effective_rate": effective_rate,
            "label":          label,
        }

    def r_g_history(self, lookback_years: int = 40) -> dict[str, pd.Series]:
        """
        Historical r-g components for charting.

        Returns series keyed by:
          GS10          — 10Y Treasury yield (monthly)
          NGDP_GROWTH   — Nominal GDP YoY growth (quarterly, ffill to monthly)
          RG_NOMINAL    — Nominal r-g spread (monthly)
          DFII10        — 10Y TIPS yield (monthly, from 2003)
          REAL_G        — Real GDP YoY growth (ffill to monthly, from 1948)
          RG_REAL       — Real r-g spread (monthly, from 2003)
          EFFECTIVE_RATE — Effective interest rate on the debt (quarterly)
        """
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        out: dict[str, pd.Series] = {}

        gs10   = self._monthly("GS10")
        gdp    = self._quarterly("GDP")
        dfii10 = self._monthly("DFII10")
        gdpc1  = self._quarterly("GDPC1")
        int_pay = self._quarterly("A091RC1Q027SBEA")
        debt_m  = self._monthly("GFDEBTN")

        def _ffill_to_monthly(q_series: pd.Series, freq: str = "ME") -> pd.Series:
            try:
                return q_series.resample(freq).last().ffill()
            except ValueError:
                return q_series.resample("M").last().ffill()

        # Nominal r and g
        if gs10 is not None and gdp is not None:
            try:
                ngdp_yoy_q = (gdp.pct_change(4) * 100).dropna()
                ngdp_yoy_m = _ffill_to_monthly(ngdp_yoy_q)
                common = gs10.index.intersection(ngdp_yoy_m.index)
                common = common[common >= cutoff]
                if len(common) > 0:
                    out["GS10"]        = gs10.loc[common]
                    out["NGDP_GROWTH"] = ngdp_yoy_m.loc[common]
                    out["RG_NOMINAL"]  = (gs10.loc[common] - ngdp_yoy_m.loc[common]).dropna()
            except Exception:
                pass

        # Real r and g
        if dfii10 is not None and gdpc1 is not None:
            try:
                real_yoy_q = (gdpc1.pct_change(4) * 100).dropna()
                real_yoy_m = _ffill_to_monthly(real_yoy_q)
                common = dfii10.index.intersection(real_yoy_m.index)
                common = common[common >= cutoff]
                if len(common) > 0:
                    out["DFII10"]  = dfii10.loc[common]
                    out["REAL_G"]  = real_yoy_m.loc[common]
                    out["RG_REAL"] = (dfii10.loc[common] - real_yoy_m.loc[common]).dropna()
            except Exception:
                pass

        # Effective interest rate
        if int_pay is not None and debt_m is not None:
            try:
                try:
                    debt_q = debt_m.resample("QE").last()
                except ValueError:
                    debt_q = debt_m.resample("Q").last()
                common = int_pay.index.intersection(debt_q.index)
                common = common[common >= cutoff]
                if len(common) > 0:
                    eff = (int_pay.loc[common] / debt_q.loc[common] * 100).dropna()
                    out["EFFECTIVE_RATE"] = eff
            except Exception:
                pass

        return out

    # ── Consolidated Dashboard ────────────────────────────────────────────

    def fiscal_dashboard(self) -> dict:
        return {
            "sustainability": self.debt_sustainability(),
            "primary":        self.primary_balance(),
            "trajectory":     self.fiscal_trajectory(),
            "rollover":       self.rollover_exposure(),
            "impulse":        self.fiscal_impulse(),
            "r_g":            self.r_g_dynamics(),
        }

    # ── History for Charts ────────────────────────────────────────────────

    def debt_service_history(self, lookback_years: int = 30) -> dict[str, pd.Series]:
        """Interest/receipts and interest/GDP ratios as quarterly series."""
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        out = {}

        int_pay  = self._quarterly("A091RC1Q027SBEA")
        receipts = self._quarterly("W006RC1Q027SBEA")
        gdp_q    = self._quarterly("GDP")

        if int_pay is not None and receipts is not None:
            common = int_pay.index.intersection(receipts.index)
            common = common[common >= cutoff]
            if len(common) > 0:
                out["INT_RECEIPTS"] = (int_pay.loc[common] / receipts.loc[common] * 100).dropna()

        if int_pay is not None and gdp_q is not None:
            common = int_pay.index.intersection(gdp_q.index)
            common = common[common >= cutoff]
            if len(common) > 0:
                out["INT_GDP"] = (int_pay.loc[common] / gdp_q.loc[common] * 100).dropna()

        return out

    def primary_balance_history(self, lookback_years: int = 40) -> dict[str, pd.Series]:
        """Annual primary balance and total deficit/surplus as % of GDP."""
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        out = {}

        deficit_pct = self._annual("FYFSGDA188S")
        if deficit_pct is not None and not deficit_pct.empty:
            out["FYFSGDA188S"] = deficit_pct[deficit_pct.index >= cutoff]

        # Primary balance = deficit + interest payments (annualized)
        deficit_B = self._annual("FYFSD")
        int_q     = self._quarterly("A091RC1Q027SBEA")
        gdp_q     = self._quarterly("GDP")

        if deficit_B is not None and int_q is not None and gdp_q is not None:
            try:
                int_ann = int_q.resample("YE").mean()
                gdp_ann = gdp_q.resample("YE").mean()
            except ValueError:
                int_ann = int_q.resample("Y").mean()
                gdp_ann = gdp_q.resample("Y").mean()

            common = deficit_B.index.intersection(int_ann.index).intersection(gdp_ann.index)
            common = common[common >= cutoff]
            if len(common) > 0:
                prim = ((deficit_B.loc[common] + int_ann.loc[common])
                        / gdp_ann.loc[common] * 100).dropna()
                out["PRIMARY_PCT_GDP"] = prim

        return out

    def fiscal_impulse_history(self, lookback_years: int = 30) -> pd.Series | None:
        """Annual fiscal impulse (change in deficit % GDP) for charting."""
        imp = self.fiscal_impulse()
        s   = imp.get("impulse_series")
        if s is None or s.empty:
            return None
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        return s[s.index >= cutoff]
