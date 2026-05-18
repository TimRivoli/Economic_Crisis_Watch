"""
LeadingIndicatorEngine: Composite Business Cycle Index (BCI) and momentum analytics.

Methodology
-----------
BCI:
  Each component is z-scored against its 20-year calibration window
  (mean/std computed from the last 20 years).  Components where a higher
  value signals deterioration are inverted.  The weighted average of
  available z-scores is computed at each date (weights renormalized when
  components are missing) and smoothed with a 3-month moving average.

Phase classification (2×2 matrix):
  BCI > 0, 3m momentum > 0  →  Expansion
  BCI > 0, 3m momentum ≤ 0  →  Slowdown
  BCI ≤ 0, 3m momentum > 0  →  Recovery
  BCI ≤ 0, 3m momentum ≤ 0  →  Contraction

Momentum analytics:
  3m/6m linear trend slope, acceleration (3m − 6m slope), z-score
  relative to the 12-month mean, and 3m/6m rolling averages.

Recession backtest:
  Compares each component against NBER recession start dates.
  A 'signal' fires when the series crosses the 75th percentile of its
  full history (25th for higher_is_bad=False).  Hit rate, average lead
  time in months, and false-positive rate are reported.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ── BCI component weights and metadata ────────────────────────────────────────

BCI_COMPONENTS: dict[str, dict] = {
    "USSLIND":      {"weight": 0.18, "invert": False, "basis": "level"},
    "T10Y2Y":       {"weight": 0.09, "invert": False, "basis": "level"},
    "T10Y3M":       {"weight": 0.09, "invert": False, "basis": "level"},
    "ICSA":         {"weight": 0.14, "invert": True,  "basis": "level"},
    "TEMPHELPS":    {"weight": 0.09, "invert": False, "basis": "yoy"},
    "BAMLH0A0HYM2": {"weight": 0.14, "invert": True,  "basis": "level"},
    "PERMIT":       {"weight": 0.09, "invert": False, "basis": "yoy"},
    "AMTMNO":       {"weight": 0.09, "invert": False, "basis": "yoy"},
    "DRTSCILM":     {"weight": 0.09, "invert": True,  "basis": "level"},
}

_CALIBRATION_YEARS = 20


def _classify_phase(bci_level: float, momentum: float) -> str:
    """2×2 phase matrix: BCI level sign × 3-month momentum sign."""
    if bci_level > 0 and momentum > 0:
        return "Expansion"
    if bci_level > 0 and momentum <= 0:
        return "Slowdown"
    if bci_level <= 0 and momentum > 0:
        return "Recovery"
    return "Contraction"


class LeadingIndicatorEngine:
    def __init__(self, data_loader):
        self.dl = data_loader

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _monthly_series(self, series_id: str, basis: str) -> pd.Series:
        """Return monthly series on the declared basis (level or yoy %)."""
        df = self.dl.load(series_id)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        try:
            monthly = df.resample("ME").last().dropna()
        except ValueError:
            monthly = df.resample("M").last().dropna()
        if monthly.empty:
            return pd.Series(dtype=float)
        col = monthly.columns[0]
        if basis == "yoy":
            return (monthly[col].pct_change(12) * 100).replace([np.inf, -np.inf], np.nan).dropna()
        return monthly[col].dropna()

    def _standardize_series(self, series: pd.Series,
                            window_years: int = _CALIBRATION_YEARS) -> pd.Series:
        """
        Z-score the full series using the mean/std from the calibration window
        (last window_years of data).  Returns empty if fewer than 12 obs in window.
        """
        if series.empty:
            return pd.Series(dtype=float)
        cutoff = series.index[-1] - pd.DateOffset(years=window_years)
        hist = series[series.index >= cutoff]
        if len(hist) < 12:
            return pd.Series(dtype=float)
        mean = float(hist.mean())
        std = float(hist.std())
        if std == 0:
            return pd.Series(dtype=float)
        return ((series - mean) / std).rename(series.name)

    # ── BCI Computation ────────────────────────────────────────────────────────

    def composite_bci(self, lookback_years: int = 20) -> pd.Series:
        """
        Monthly Composite Business Cycle Index (3-month smoothed).

        Loads each BCI component, standardizes it against its 20-year calibration
        window, inverts the sign when lower is healthier, and takes the weighted
        average across whatever components have data at each date.  Weights are
        renormalized when a component is missing.  The result is smoothed with a
        3-month rolling mean.
        """
        component_data: list[tuple[pd.Series, float]] = []
        for series_id, cfg in BCI_COMPONENTS.items():
            raw = self._monthly_series(series_id, cfg["basis"])
            if raw.empty:
                continue
            z = self._standardize_series(raw)
            if z.empty:
                continue
            if cfg["invert"]:
                z = -z
            component_data.append((z, cfg["weight"]))

        if not component_data:
            return pd.Series(dtype=float, name="BCI")

        # Align on common monthly index
        df = pd.concat(
            [s.rename(f"c{i}") for i, (s, _) in enumerate(component_data)],
            axis=1,
        )
        weights = np.array([w for _, w in component_data])
        min_comps = max(1, len(component_data) // 2)

        bci_vals: dict = {}
        for date, row in df.iterrows():
            available = row.notna().values
            if available.sum() < min_comps:
                continue
            w = weights[available]
            v = row.values[available].astype(float)
            bci_vals[date] = float(np.dot(v, w / w.sum()))

        if not bci_vals:
            return pd.Series(dtype=float, name="BCI")

        bci = pd.Series(bci_vals, name="BCI").sort_index()

        if lookback_years:
            cutoff = bci.index[-1] - pd.DateOffset(years=lookback_years)
            bci = bci[bci.index >= cutoff]

        return bci.rolling(3, min_periods=1).mean()

    def current_bci(self) -> tuple[float | None, str]:
        """Return (current BCI value, phase label)."""
        bci = self.composite_bci()
        if bci.empty:
            return None, "No Data"
        level = float(bci.iloc[-1])
        mom = float(bci.iloc[-1] - bci.iloc[-3]) if len(bci) >= 3 else 0.0
        return level, _classify_phase(level, mom)

    def phase_history(self, lookback_years: int = 15) -> pd.Series:
        """Monthly time series of phase labels."""
        bci = self.composite_bci(lookback_years=lookback_years)
        if bci.empty:
            return pd.Series(dtype=object)
        phases = []
        for i in range(len(bci)):
            level = float(bci.iloc[i])
            mom = float(bci.iloc[i] - bci.iloc[i - 2]) if i >= 2 else 0.0
            phases.append(_classify_phase(level, mom))
        return pd.Series(phases, index=bci.index, name="phase")

    def component_contributions(self) -> dict[str, float | None]:
        """Current z-score contribution (z × weight) for each BCI component."""
        result: dict[str, float | None] = {}
        for series_id, cfg in BCI_COMPONENTS.items():
            raw = self._monthly_series(series_id, cfg["basis"])
            if raw.empty:
                result[series_id] = None
                continue
            z = self._standardize_series(raw)
            if z.empty:
                result[series_id] = None
                continue
            current_z = float(z.iloc[-1])
            if cfg["invert"]:
                current_z = -current_z
            result[series_id] = round(current_z * cfg["weight"], 4)
        return result

    # ── Momentum Analytics ─────────────────────────────────────────────────────

    def momentum_analytics(self, series_id: str) -> dict:
        """
        Trend and momentum analytics for a single series.

        current      — latest value (on the series' risk_basis)
        trend_3m     — linear slope (per month) over last 3 months
        trend_6m     — linear slope (per month) over last 6 months
        acceleration — trend_3m minus trend_6m (positive = accelerating)
        z_vs_1y      — (current − 12m mean) / 12m std
        rolling_3m   — 3-month rolling mean
        rolling_6m   — 6-month rolling mean
        direction    — Improving | Deteriorating | Stable
        """
        from .series_registry import REGISTRY
        meta = REGISTRY.get(series_id, {})
        basis = meta.get("risk_basis", "level")
        higher_is_bad = meta.get("higher_is_bad", True)

        raw = self._monthly_series(series_id, basis)
        if raw.empty or len(raw) < 6:
            return {}

        current = float(raw.iloc[-1])
        rolling_3m = float(raw.iloc[-3:].mean())
        rolling_6m = float(raw.iloc[-6:].mean())

        def _slope(vals: pd.Series) -> float:
            n = len(vals)
            if n < 2:
                return 0.0
            x = np.arange(n, dtype=float)
            return float(np.polyfit(x, vals.values.astype(float), 1)[0])

        trend_3m = _slope(raw.iloc[-3:])
        trend_6m = _slope(raw.iloc[-6:])
        acceleration = trend_3m - trend_6m

        recent = raw.iloc[-12:] if len(raw) >= 12 else raw
        mean_1y = float(recent.mean())
        std_1y = float(recent.std())
        z_vs_1y = (current - mean_1y) / std_1y if std_1y > 0 else 0.0

        scale = abs(current) if abs(current) > 0.01 else 1.0
        if abs(trend_3m) < 0.01 * scale:
            direction = "Stable"
        elif (higher_is_bad and trend_3m > 0) or (not higher_is_bad and trend_3m < 0):
            direction = "Deteriorating"
        else:
            direction = "Improving"

        return {
            "current":      round(current, 3),
            "trend_3m":     round(trend_3m, 4),
            "trend_6m":     round(trend_6m, 4),
            "acceleration": round(acceleration, 4),
            "z_vs_1y":      round(z_vs_1y, 3),
            "rolling_3m":   round(rolling_3m, 3),
            "rolling_6m":   round(rolling_6m, 3),
            "direction":    direction,
        }

    def all_momentum(self) -> dict[str, dict]:
        """Momentum analytics for all BCI components."""
        return {sid: self.momentum_analytics(sid) for sid in BCI_COMPONENTS}

    # ── Recession Backtest ─────────────────────────────────────────────────────

    def recession_backtest(self, series_id: str,
                           threshold_pct: float = 75.0) -> dict:
        """
        Backtest a leading indicator against NBER recession start dates.

        A 'signal' fires when the series crosses threshold_pct of its full
        distribution (75th for higher_is_bad=True, 25th for False).

        Returns: hit_rate (%), avg_lead_months, false_pos_rate (%), n_recessions.
        """
        from .series_registry import REGISTRY
        meta = REGISTRY.get(series_id, {})
        basis = meta.get("risk_basis", "level")
        higher_is_bad = meta.get("higher_is_bad", True)

        raw = self._monthly_series(series_id, basis)
        rec_df = self.dl.load("USREC")
        if raw.empty or rec_df is None or rec_df.empty:
            return {"hit_rate": None, "avg_lead_months": None,
                    "false_pos_rate": None, "n_recessions": 0}

        try:
            rec = rec_df.resample("ME").last().dropna()
        except ValueError:
            rec = rec_df.resample("M").last().dropna()
        rec_col = rec.iloc[:, 0]

        q = threshold_pct / 100 if higher_is_bad else (1.0 - threshold_pct / 100)
        threshold = float(raw.quantile(q))
        signal = (raw >= threshold) if higher_is_bad else (raw <= threshold)

        # NBER recession start dates (0→1 transitions)
        rec_starts = []
        prev = 0
        for date, val in rec_col.items():
            v = int(val)
            if v == 1 and prev == 0:
                rec_starts.append(date)
            prev = v

        if not rec_starts:
            return {"hit_rate": 0.0, "avg_lead_months": None,
                    "false_pos_rate": 0.0, "n_recessions": 0}

        # Only evaluate recessions that fall within the signal data's history.
        # Pre-data recessions have no signal observations and would always be
        # counted as misses, artificially deflating the hit rate.
        data_start = raw.index[0]
        rec_starts = [r for r in rec_starts if r > data_start]
        if not rec_starts:
            return {"hit_rate": 0.0, "avg_lead_months": None,
                    "false_pos_rate": 0.0, "n_recessions": 0}

        hits, lead_times = 0, []
        for rec_start in rec_starts:
            window = signal[
                (signal.index >= rec_start - pd.DateOffset(months=18)) &
                (signal.index < rec_start)
            ]
            active = window[window]
            if not active.empty:
                hits += 1
                # Use the FIRST signal in the window = maximum advance warning
                lead_times.append((rec_start - active.index[0]).days / 30.44)

        hit_rate = hits / len(rec_starts) * 100

        # FP rate: fraction of signal EPISODES not followed by a recession within
        # 24 months.  Counting per-episode (not per-date) avoids inflating the
        # rate when the signal stays elevated for many consecutive months.
        episode_starts: list = []
        in_ep = False
        for date, val in signal.items():
            if val and not in_ep:
                episode_starts.append(date)
                in_ep = True
            elif not val:
                in_ep = False

        fp_episodes = sum(
            1 for ep in episode_starts
            if not any(ep < r <= ep + pd.DateOffset(months=24) for r in rec_starts)
        )
        fp_rate = fp_episodes / max(len(episode_starts), 1) * 100

        return {
            "hit_rate":        round(hit_rate, 1),
            "avg_lead_months": round(float(np.mean(lead_times)), 1) if lead_times else None,
            "false_pos_rate":  round(fp_rate, 1),
            "n_recessions":    len(rec_starts),
        }

    def all_backtests(self) -> dict[str, dict]:
        """Recession backtests for all BCI components."""
        return {sid: self.recession_backtest(sid) for sid in BCI_COMPONENTS}

    # ── Summary ────────────────────────────────────────────────────────────────

    def leading_signals(self) -> list[dict]:
        """Current signal summary for all BCI components."""
        from .series_registry import REGISTRY
        signals = []
        for sid, cfg in BCI_COMPONENTS.items():
            mom = self.momentum_analytics(sid)
            if not mom:
                continue
            meta = REGISTRY.get(sid, {})
            signals.append({
                "series_id": sid,
                "name":      meta.get("short_name", sid),
                "value":     mom.get("current"),
                "direction": mom.get("direction"),
                "z_vs_1y":   mom.get("z_vs_1y"),
                "weight":    cfg["weight"],
            })
        return signals
