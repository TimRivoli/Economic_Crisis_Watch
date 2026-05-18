"""
Labor Analytics Engine
Computes labor deterioration indexes, JOLTS summaries, wage/productivity
diagnostics, and composite labor health scores.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# Weights for the Labor Deterioration Index (0–100, higher = more stress)
_LDI_COMPONENTS: dict[str, dict] = {
    "UNRATE":  {"weight": 0.20, "invert": False, "basis": "level",     "calibration_years": 20},
    "U6RATE":  {"weight": 0.15, "invert": False, "basis": "level",     "calibration_years": 20},
    "ICSA":    {"weight": 0.20, "invert": False, "basis": "level",     "calibration_years": 20},
    "CCSA":    {"weight": 0.10, "invert": False, "basis": "level",     "calibration_years": 20},
    "JTSQUR":  {"weight": 0.15, "invert": True,  "basis": "level",     "calibration_years": 15},
    "JTSLDR":  {"weight": 0.10, "invert": False, "basis": "level",     "calibration_years": 15},
    "PAYEMS":  {"weight": 0.10, "invert": True,  "basis": "yoy",       "calibration_years": 20},
}


class LaborAnalyticsEngine:
    """Computes composite labor health metrics from a DataLoader instance."""

    def __init__(self, dl):
        self._dl = dl
        self._cache: dict = {}

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

    def _zscore(self, s: pd.Series, window_years: int) -> pd.Series:
        """Rolling z-score normalisation over the trailing calibration window."""
        n = window_years * 12
        mu = s.rolling(n, min_periods=max(24, n // 4)).mean()
        sd = s.rolling(n, min_periods=max(24, n // 4)).std(ddof=1)
        return (s - mu) / sd.replace(0, np.nan)

    def _to_pct_0_100(self, z: float | None) -> float | None:
        """Map a z-score to a 0–100 stress percentile via normal CDF approximation."""
        if z is None or np.isnan(z):
            return None
        # sigmoid-style clamp
        pct = 100 / (1 + np.exp(-z * 0.8))
        return float(np.clip(pct, 0, 100))

    # ── Labor Deterioration Index ──────────────────────────────────────────

    def labor_deterioration_index(self) -> dict:
        """
        Composite 0–100 labor stress index.
        0 = pristine conditions, 100 = acute deterioration.
        Returns {score, components, trend_3m}.
        """
        if "ldi" in self._cache:
            return self._cache["ldi"]

        component_scores: dict[str, float | None] = {}
        component_weights: dict[str, float] = {}

        for sid, cfg in _LDI_COMPONENTS.items():
            raw = self._monthly(sid)
            if raw is None or raw.empty:
                continue
            if cfg["basis"] == "yoy":
                raw = raw.pct_change(12) * 100
                raw = raw.dropna()
            if raw.empty or len(raw) < 24:
                continue
            z = self._zscore(raw, cfg["calibration_years"])
            if cfg["invert"]:
                z = -z
            last_z = float(z.iloc[-1]) if not z.empty and not np.isnan(z.iloc[-1]) else None
            component_scores[sid] = self._to_pct_0_100(last_z)
            component_weights[sid] = cfg["weight"]

        if not component_scores:
            return {"score": None, "components": {}, "trend_3m": None}

        total_w = sum(component_weights[s] for s in component_scores)
        composite = sum(
            component_scores[s] * component_weights[s] / total_w
            for s in component_scores
            if component_scores[s] is not None
        )

        result = {
            "score": float(composite),
            "components": {
                sid: {
                    "score": component_scores[sid],
                    "weight": component_weights[sid],
                }
                for sid in component_scores
            },
            "trend_3m": self._ldi_trend_3m(),
        }
        self._cache["ldi"] = result
        return result

    def _ldi_trend_3m(self) -> float | None:
        """3-month change in ICSA z-score as a proxy for LDI momentum."""
        raw = self._monthly("ICSA")
        if raw is None or len(raw) < 36:
            return None
        z = self._zscore(raw, 20)
        if len(z) < 3:
            return None
        return float(z.iloc[-1] - z.iloc[-3])

    # ── JOLTS Summary ──────────────────────────────────────────────────────

    def jolts_summary(self) -> dict:
        """Current JOLTS readings and Beveridge-curve diagnosis."""
        openings = self._monthly("JTSJOL")
        quits    = self._monthly("JTSQUR")
        layoffs  = self._monthly("JTSLDR")
        unemployed = self._monthly("UNRATE")

        def _last(s):
            if s is None or s.empty:
                return None
            v = s.iloc[-1]
            return float(v) if not np.isnan(v) else None

        openings_val = _last(openings)
        quits_val    = _last(quits)
        layoffs_val  = _last(layoffs)

        # V/U ratio: openings (thousands) / unemployed persons
        unemployed_level = self._monthly("PAYEMS")  # fallback — won't be right
        vu_ratio = None
        if openings is not None and unemployed is not None:
            # UNRATE is %, need to approximate count
            # V/U = openings_level / (UNRATE * civilian_labor_force) — use ratio directly
            # Just use openings-per-percent-unemployed as proxy
            last_openings = openings.reindex(unemployed.index, method="ffill")
            vu_s = last_openings / (unemployed * 100)  # rough ratio
            if not vu_s.empty:
                vu_ratio = float(vu_s.dropna().iloc[-1]) if not vu_s.dropna().empty else None

        # Beveridge diagnosis
        if quits_val is not None and layoffs_val is not None:
            if quits_val >= 2.5 and layoffs_val <= 1.2:
                beveridge = "Tight — high worker confidence, low involuntary separations"
            elif quits_val < 2.0 and layoffs_val > 1.5:
                beveridge = "Loosening — workers losing job mobility, layoffs rising"
            elif quits_val < 1.8 and layoffs_val > 1.8:
                beveridge = "Stressed — labor market deteriorating"
            else:
                beveridge = "Transitional"
        else:
            beveridge = "Insufficient data"

        return {
            "openings_k": openings_val,
            "quits_rate": quits_val,
            "layoffs_rate": layoffs_val,
            "beveridge_diagnosis": beveridge,
        }

    # ── Wage & Productivity Diagnostics ───────────────────────────────────

    def wage_pressure(self) -> dict:
        """
        Current wage growth, productivity, and unit labor cost readings.
        Returns dict with YoY rates and real wage growth.
        """
        ahe   = self._monthly("CES0500000003")
        eci   = self._monthly("ECIALLCIV")
        prod  = self._monthly("OPHNFB")
        ulc   = self._monthly("ULCNFB")
        cpi   = self._monthly("CPIAUCSL")

        def _yoy_last(s, periods=12):
            if s is None or len(s) < periods + 1:
                return None
            pct = s.pct_change(periods) * 100
            v = pct.dropna()
            return float(v.iloc[-1]) if not v.empty else None

        def _qoq_ann(s):
            if s is None or len(s) < 2:
                return None
            # quarterly series — 4-quarter annualised
            pct4 = s.pct_change(4) * 100
            v = pct4.dropna()
            return float(v.iloc[-1]) if not v.empty else None

        ahe_yoy   = _yoy_last(ahe, 12)
        eci_yoy   = _qoq_ann(eci)   # quarterly series so use 4-period YoY
        prod_yoy  = _qoq_ann(prod)  # quarterly
        ulc_yoy   = _qoq_ann(ulc)   # quarterly
        cpi_yoy   = _yoy_last(cpi, 12)

        real_wage = (ahe_yoy - cpi_yoy) if (ahe_yoy is not None and cpi_yoy is not None) else None

        # Wage-productivity gap: wage growth minus productivity growth (quarterly basis)
        wp_gap = (ulc_yoy) if ulc_yoy is not None else None  # ULC *is* wages / productivity

        return {
            "ahe_yoy":    ahe_yoy,
            "eci_yoy":    eci_yoy,
            "prod_yoy":   prod_yoy,
            "ulc_yoy":    ulc_yoy,
            "real_wage":  real_wage,
            "wp_gap":     wp_gap,
        }

    # ── Continuing Claims Absorption Rate ─────────────────────────────────

    def claims_summary(self) -> dict:
        """Initial and continuing claims levels and trends."""
        icsa = self._monthly("ICSA")
        ccsa = self._monthly("CCSA")

        def _last_and_trend(s, window=4):
            if s is None or s.empty:
                return None, None
            v = float(s.iloc[-1])
            if len(s) >= window + 1:
                trend = float(s.iloc[-1] - s.iloc[-(window + 1)])
            else:
                trend = None
            return v, trend

        icsa_val, icsa_trend = _last_and_trend(icsa, 4)
        ccsa_val, ccsa_trend = _last_and_trend(ccsa, 4)

        # Absorption ratio: continuing / initial claims (higher = longer duration)
        absorption = None
        if icsa_val and ccsa_val and icsa_val > 0:
            absorption = ccsa_val / icsa_val

        return {
            "initial_claims": icsa_val,
            "initial_trend_4w": icsa_trend,
            "continued_claims": ccsa_val,
            "continued_trend_4w": ccsa_trend,
            "absorption_ratio": absorption,
        }

    # ── U6 Gap (Labor Slack Indicator) ────────────────────────────────────

    def unemployment_gap(self) -> dict:
        """U6 minus U3 spread — measures hidden labor slack."""
        u3 = self._monthly("UNRATE")
        u6 = self._monthly("U6RATE")
        if u3 is None or u6 is None:
            return {"u3": None, "u6": None, "gap": None, "gap_trend_6m": None}

        shared = u3.index.intersection(u6.index)
        if shared.empty:
            return {"u3": None, "u6": None, "gap": None, "gap_trend_6m": None}

        u3_s = u3.reindex(shared)
        u6_s = u6.reindex(shared)
        gap  = (u6_s - u3_s).dropna()

        u3_now  = float(u3_s.iloc[-1]) if not u3_s.empty else None
        u6_now  = float(u6_s.iloc[-1]) if not u6_s.empty else None
        gap_now = float(gap.iloc[-1])  if not gap.empty  else None
        gap_trend = float(gap.iloc[-1] - gap.iloc[-7]) if len(gap) >= 7 else None

        return {
            "u3": u3_now,
            "u6": u6_now,
            "gap": gap_now,
            "gap_trend_6m": gap_trend,
        }
