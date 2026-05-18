"""
Recession Probability Engine
=============================
Pure numpy/pandas logistic regression for multi-horizon recession forecasting.

Methodology
-----------
Eight macro features drawn from FRED data are standardized (zero mean, unit
variance over the training window) and fed into a logistic regression model
trained with L2-regularised gradient descent.  Three separate models are
trained — one per forecast horizon (6, 12, and 24 months ahead).

The binary target for horizon h is 1 if a NBER recession (USREC == 1) occurs
in *any* of the h months following the observation date, and 0 otherwise.
This "forward-window" encoding means high probability at horizon 12 does not
necessarily imply high probability at horizon 6.

Features
--------
yc_2y       T10Y2Y level         — classic yield-curve recession signal
yc_3m       T10Y3M level         — near-term credit stress
lei_yoy     CFNAI level          — Chicago Fed composite activity (USSLIND discontinued)
claims_log  log(ICSA 4-wk MA)    — labour-market stress
ism         AMTMNO yoy           — manufacturing orders momentum
hy_spread   BAA10Y level         — Moody's Baa credit spread (replaces BAMLH0A0HYM2 API-restricted)
u3_chg      UNRATE 3-month change — unemployment turning-point
payroll_yoy PAYEMS year-over-year — broad employment trend

Gradient descent
----------------
  cost = -[y*log(p) + (1-y)*log(1-p)] + (lambda/2) * sum(w[1:]^2)
  grad_w[0]   = mean(p - y)                        # bias — no L2
  grad_w[1:]  = mean((p - y) * X, axis=0) + l2 * w[1:]

Hyperparameters: 1 500 epochs, lr = 0.05, l2 = 0.02.

AUC
---
Computed via Wilcoxon rank-sum approximation.  scipy.stats.rankdata is used
when available; otherwise a manual O(n log n) implementation is used.

Usage
-----
    from _classes.data_loader import DataLoader
    from _classes.recession_probability import RecessionProbabilityEngine

    dl = DataLoader()
    engine = RecessionProbabilityEngine(dl)
    engine.train()
    print(engine.current_probabilities())
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .data_loader import DataLoader

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

FEATURES: list[str] = [
    "yc_2y",
    "yc_3m",
    "lei_yoy",
    "claims_log",
    "ism",
    "hy_spread",
    "u3_chg",
    "payroll_yoy",
]

HORIZONS: list[int] = [6, 12, 24]

PROB_THRESHOLD: float = 0.35  # classification cutoff for precision / recall

# Maps feature name -> (FRED series ID, transform type)
_FEATURE_SERIES: dict[str, tuple[str, str]] = {
    "yc_2y":       ("T10Y2Y",  "level"),
    "yc_3m":       ("T10Y3M",  "level"),
    "lei_yoy":     ("CFNAI",   "level"),   # USSLIND discontinued on FRED; CFNAI available 1967-present
    "claims_log":  ("ICSA",    "log_ma4"),
    "ism":         ("AMTMNO",  "yoy"),
    "hy_spread":   ("BAA10Y",  "level"),   # BAMLH0A0HYM2 API-restricted to ~3 years; BAA10Y available 1986-present
    "u3_chg":      ("UNRATE",  "diff3"),
    "payroll_yoy": ("PAYEMS",  "yoy"),
}

_RECESSION_SERIES = "USREC"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: clips z to [-500, 500] before exp."""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logistic_train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1500,
    lr: float = 0.05,
    l2: float = 0.02,
) -> np.ndarray:
    """
    Train logistic regression via gradient descent.

    Parameters
    ----------
    X : (n_samples, n_features)  — already standardised, NO bias column added
    y : (n_samples,)             — binary {0, 1}

    Returns
    -------
    w : (1 + n_features,)        — [bias, w_1, ..., w_k]
    """
    X = X.astype(float)
    y = y.astype(float)
    n, k = X.shape
    # prepend bias column of ones
    Xb = np.hstack([np.ones((n, 1), dtype=float), X])  # (n, k+1)
    w = np.zeros(k + 1, dtype=float)

    for _ in range(epochs):
        p = _sigmoid(Xb @ w)           # (n,)
        err = p - y                     # (n,)
        grad = (Xb.T @ err) / n        # (k+1,)
        # L2 on weights only, not on bias (index 0)
        grad[1:] += l2 * w[1:]
        w -= lr * grad

    return w


def _standardise(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardise X column-wise.  If mean/std are provided they are used
    directly (inference mode); otherwise they are computed from X.

    Returns (X_scaled, mean, std).
    """
    X = X.astype(float)
    if mean is None:
        mean = np.nanmean(X, axis=0)
    if std is None:
        std = np.nanstd(X, axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def _auc_rank_sum(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Approximate AUC via Wilcoxon rank-sum.
    Uses scipy.stats.rankdata when available; falls back to a manual sort.
    """
    y_true = y_true.astype(float)
    y_score = y_score.astype(float)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    try:
        from scipy.stats import rankdata  # type: ignore[import]
        ranks = rankdata(y_score)
    except Exception:
        # Manual rank computation using argsort
        order = np.argsort(y_score)
        ranks = np.empty(len(y_score), dtype=float)
        # handle ties: average rank
        i = 0
        while i < len(order):
            j = i
            while j < len(order) - 1 and y_score[order[j + 1]] == y_score[order[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1

    rank_sum_pos = ranks[y_true == 1].sum()
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a DataFrame to month-start frequency, taking last observation.

    Using "MS" (MonthStart) is consistent across all pandas versions and ensures
    that daily series (e.g. BAMLH0A0HYM2) and native-monthly series (e.g. PAYEMS
    with first-of-month dates) both land on the same timestamp for each month.
    """
    return df.resample("MS").last()


def _series_to_monthly(raw: pd.DataFrame, sid: str) -> pd.Series:
    """Extract the value column from a DataLoader result as a monthly Series."""
    s = raw.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return _resample_monthly(s.to_frame()).iloc[:, 0].rename(sid)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class RecessionProbabilityEngine:
    """
    Multi-horizon logistic regression recession probability model.

    Parameters
    ----------
    data_loader : DataLoader
        A configured DataLoader instance used to fetch FRED series.
    """

    def __init__(self, data_loader: "DataLoader") -> None:
        self.dl = data_loader
        self._models: dict[int, np.ndarray] = {}          # horizon -> weight vector
        self._scalers: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # horizon -> (mean, std)
        self._feature_cache: pd.DataFrame | None = None
        self._prob_cache: dict[int, pd.Series] = {}

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(self) -> pd.DataFrame:
        """
        Load each FRED series via data_loader, apply transforms, and return a
        monthly DataFrame indexed by date with columns = FEATURES (subset of
        available series only).
        """
        if self._feature_cache is not None:
            return self._feature_cache

        columns: dict[str, pd.Series] = {}


        for feat, (sid, transform) in _FEATURE_SERIES.items():
            try:
                raw = self.dl.load(sid)
                if raw is None or raw.empty:
                    print(f"[RecessionProbability]   {feat} ({sid}): NOT FOUND in data store")
                    log.debug("recession_probability: series %s unavailable, skipping %s", sid, feat)
                    continue
                monthly = _series_to_monthly(raw, sid).dropna()
                if monthly.empty:
                    print(f"[RecessionProbability]   {feat} ({sid}): loaded but monthly series is empty")
                    continue
                if transform == "level":
                    columns[feat] = monthly
                elif transform == "yoy":
                    yoy = monthly.pct_change(12) * 100.0
                    columns[feat] = yoy
                elif transform == "log_ma4":
                    ma4 = monthly.rolling(4, min_periods=1).mean()
                    log_ma4 = np.log(ma4.where(ma4 > 0))
                    columns[feat] = log_ma4
                elif transform == "diff3":
                    columns[feat] = monthly.diff(3)
                #print(columns[feat])
            except Exception as exc:
                print(f"[RecessionProbability]   {feat} ({sid}): EXCEPTION — {type(exc).__name__}: {exc}")
                log.warning("recession_probability: error processing %s (%s): %s", feat, sid, exc)
                continue

        if not columns:
            print("[RecessionProbability] columns dict is empty — all features failed to load or were skipped.")
            return pd.DataFrame()
        df = pd.concat(columns, axis=1).dropna()
        self._feature_cache = df
        return df

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    def _build_target(self, features: pd.DataFrame, horizon: int) -> pd.Series:
        """
        Build binary recession target: 1 if a recession occurs in any of the
        next `horizon` months from each observation date.

        Returns a Series aligned to features.index (with NaN rows removed).
        """
        raw = self.dl.load(_RECESSION_SERIES)
        if raw is None or raw.empty:
            return pd.Series(dtype=float)

        usrec = _series_to_monthly(raw, _RECESSION_SERIES).reindex(features.index)

        # Forward-shift: at each date t, look h months ahead
        # rolling max over the next `horizon` months
        # shift(-horizon) aligns future values to current index
        # We build a forward window manually
        future_max = (
            usrec
            .shift(-1)               # start looking one month ahead
            .rolling(window=horizon, min_periods=1)
            .max()
            .shift(-(horizon - 1))   # align result back
        )

        # Alternative cleaner approach: use a reverse rolling max
        # Reverse the series, rolling max of `horizon`, reverse back
        rev = usrec.iloc[::-1]
        fwd_max = rev.rolling(window=horizon, min_periods=1).max().iloc[::-1]
        # fwd_max.iloc[i] = max(usrec[i], usrec[i+1], ..., usrec[i+horizon-1])
        # We want strictly future: shift by 1 so we don't include current month
        target = fwd_max.shift(-1)  # 1 if recession in next `horizon` months

        target = target.reindex(features.index).dropna()
        return target.astype(float)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, min_obs: int = 60) -> None:
        """
        Train logistic regression models for all horizons.

        Skips a horizon if fewer than `min_obs` aligned observations exist
        or if the target has fewer than 3 positive examples.
        """
        features = self._build_features()
        if features.empty:
            print("[RecessionProbability] Training aborted: no features could be loaded.")
            log.warning("recession_probability: no features available, training aborted.")
            return

        print(f"[RecessionProbability] Feature matrix: {len(features)} obs × {len(features.columns)} features "
              f"({features.index.min().date()} – {features.index.max().date()})")

        for horizon in HORIZONS:
            try:
                target = self._build_target(features, horizon)
                if target.empty:
                    print(f"[RecessionProbability] h={horizon}: no target data, skipping.")
                    log.info("recession_probability: no target for h=%d, skipping.", horizon)
                    continue

                # Align features and target on common index
                common = features.index.intersection(target.index)
                X_df = features.loc[common]
                y = target.loc[common].values.astype(float)

                if len(common) < min_obs:
                    print(f"[RecessionProbability] h={horizon}: only {len(common)} obs (need {min_obs}), skipping.")
                    log.info(
                        "recession_probability: h=%d has only %d obs (need %d), skipping.",
                        horizon, len(common), min_obs,
                    )
                    continue

                n_pos = int(y.sum())
                if n_pos < 3:
                    print(f"[RecessionProbability] h={horizon}: only {n_pos} positive examples, skipping.")
                    log.info(
                        "recession_probability: h=%d has only %d positive examples, skipping.",
                        horizon, n_pos,
                    )
                    continue

                X = X_df.values.astype(float)
                X_scaled, mean_vec, std_vec = _standardise(X)

                w = _logistic_train(X_scaled, y)

                self._models[horizon] = w
                self._scalers[horizon] = (mean_vec, std_vec)
                # Invalidate probability cache for this horizon
                self._prob_cache.pop(horizon, None)

                print(f"[RecessionProbability] h={horizon}: trained on {len(common)} obs ({n_pos} positive). OK")
                log.info(
                    "recession_probability: trained h=%d on %d obs (%d positive).",
                    horizon, len(common), n_pos,
                )

            except Exception as exc:
                print(f"[RecessionProbability] h={horizon}: ERROR — {exc}")
                log.error("recession_probability: error training h=%d: %s", horizon, exc)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_raw(self, horizon: int) -> pd.Series:
        """
        Generate probability series over all available feature dates using the
        trained model for `horizon`.  Results are cached.
        """
        if horizon in self._prob_cache:
            return self._prob_cache[horizon]

        if horizon not in self._models:
            return pd.Series(dtype=float)

        features = self._build_features()
        if features.empty:
            return pd.Series(dtype=float)

        w = self._models[horizon]
        mean_vec, std_vec = self._scalers[horizon]

        # The model was trained on len(mean_vec) features; use only that many
        n_feat = len(mean_vec)
        available_cols = features.columns.tolist()

        if len(available_cols) < n_feat:
            log.warning(
                "recession_probability: h=%d model needs %d features, only %d available.",
                horizon, n_feat, len(available_cols),
            )
            return pd.Series(dtype=float)

        X = features.iloc[:, :n_feat].values.astype(float)
        X_scaled = (X - mean_vec) / np.where(std_vec == 0, 1.0, std_vec)

        # Add bias column
        Xb = np.hstack([np.ones((len(X_scaled), 1), dtype=float), X_scaled])
        probs = _sigmoid(Xb @ w)

        result = pd.Series(probs, index=features.index, name=f"p_rec_{horizon}m")
        self._prob_cache[horizon] = result
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_probabilities(self) -> dict[int, float | None]:
        """
        Return the latest recession probability for each horizon.

        Returns
        -------
        {6: p6, 12: p12, 24: p24}  — None if model not available for that horizon.
        """
        result: dict[int, float | None] = {}
        for h in HORIZONS:
            series = self._predict_raw(h)
            if series.empty:
                result[h] = None
            else:
                val = series.dropna()
                result[h] = float(val.iloc[-1]) if not val.empty else None
        return result

    def probability_history(self, lookback_years: int = 20) -> pd.DataFrame:
        """
        Return a DataFrame of probability histories with columns "6m", "12m", "24m".

        Parameters
        ----------
        lookback_years : int
            How many years of history to return.
        """
        cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=lookback_years)
        cols: dict[str, pd.Series] = {}
        for h in HORIZONS:
            s = self._predict_raw(h).dropna()
            s.index = pd.to_datetime(s.index)
            cols[f"{h}m"] = s[s.index >= cutoff]

        if not cols:
            return pd.DataFrame(columns=["6m", "12m", "24m"])

        df = pd.concat(cols, axis=1)
        df.index = pd.to_datetime(df.index)
        return df

    def feature_contributions(self, horizon: int = 12) -> dict[str, float]:
        """
        Log-odds contribution of each feature to the current (latest) forecast.

        contrib_i = w_i * x_scaled_i

        The bias term is excluded.  Returns an empty dict if the model is not
        trained for `horizon`.
        """
        if horizon not in self._models:
            return {}

        features = self._build_features()
        if features.empty:
            return {}

        w = self._models[horizon]
        mean_vec, std_vec = self._scalers[horizon]
        n_feat = len(mean_vec)

        available_cols = features.columns.tolist()
        if len(available_cols) < n_feat:
            return {}

        latest = features.iloc[-1, :n_feat].values.astype(float)
        x_scaled = (latest - mean_vec) / np.where(std_vec == 0, 1.0, std_vec)

        # w[0] is bias; w[1:] are feature weights
        feature_names = available_cols[:n_feat]
        contribs = {
            name: float(w[i + 1] * x_scaled[i])
            for i, name in enumerate(feature_names)
        }
        return contribs

    def confidence_band(
        self,
        horizon: int = 12,
        n_bootstrap: int = 60,
        lookback_years: int = 20,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bootstrap confidence band around the probability history.

        Parameters
        ----------
        horizon        : forecast horizon in months
        n_bootstrap    : number of bootstrap resamples
        lookback_years : years of history to return

        Returns
        -------
        (lower, median, upper)  — 5th, 50th, 95th percentile at each date,
        each trimmed to `lookback_years`.  On failure returns three copies of
        the main series.
        """
        main = self._predict_raw(horizon).dropna()
        cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=lookback_years)
        main_trimmed = main[main.index >= cutoff]

        fallback = (main_trimmed, main_trimmed, main_trimmed)

        if horizon not in self._models:
            return fallback

        features = self._build_features()
        if features.empty:
            return fallback

        try:
            target = self._build_target(features, horizon)
            if target.empty:
                return fallback

            common = features.index.intersection(target.index)
            if len(common) < 20:
                return fallback

            X_df = features.loc[common]
            y = target.loc[common].values.astype(float)
            n_feat = len(self._scalers[horizon][0])
            X = X_df.iloc[:, :n_feat].values.astype(float)

            # All feature dates for out-of-sample probability matrix
            X_all = features.iloc[:, :n_feat].values.astype(float)
            n_all = len(X_all)

            rng = np.random.default_rng(42)
            boot_probs = np.full((n_bootstrap, n_all), np.nan)

            for b in range(n_bootstrap):
                idx = rng.integers(0, len(X), size=len(X))
                Xb_train = X[idx]
                yb_train = y[idx]

                if yb_train.sum() < 2:
                    continue

                X_scaled, mean_b, std_b = _standardise(Xb_train)
                w_b = _logistic_train(X_scaled, yb_train)

                std_b_safe = np.where(std_b == 0, 1.0, std_b)
                X_all_scaled = (X_all - mean_b) / std_b_safe
                Xb_all = np.hstack([np.ones((n_all, 1), dtype=float), X_all_scaled])
                boot_probs[b] = _sigmoid(Xb_all @ w_b)

            # Compute percentiles (ignore NaN rows)
            with np.errstate(all="ignore"):
                lower_arr = np.nanpercentile(boot_probs, 5, axis=0)
                median_arr = np.nanpercentile(boot_probs, 50, axis=0)
                upper_arr = np.nanpercentile(boot_probs, 95, axis=0)

            idx = features.index
            lower = pd.Series(lower_arr, index=idx)[lambda s: s.index >= cutoff]
            median = pd.Series(median_arr, index=idx)[lambda s: s.index >= cutoff]
            upper = pd.Series(upper_arr, index=idx)[lambda s: s.index >= cutoff]

            return lower, median, upper

        except Exception as exc:
            log.warning("recession_probability: confidence_band failed (%s), using fallback.", exc)
            return fallback

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def rolling_backtest(
        self,
        horizon: int = 12,
        train_years: int = 20,
        test_start: str = "2000-01-01",
    ) -> dict:
        """
        Walk-forward backtest: for each month t >= test_start, train on the
        prior `train_years` of data and forecast at t.

        Returns
        -------
        dict with keys: precision, recall, hit_rate, fp_rate, n_signals,
        n_test_obs, auc.  Returns {} if insufficient data.
        """
        features = self._build_features()
        if features.empty:
            return {}

        target_full = self._build_target(features, horizon)
        if target_full.empty:
            return {}

        common = features.index.intersection(target_full.index)
        if len(common) < 30:
            return {}

        X_df = features.loc[common]
        y_s = target_full.loc[common]
        n_feat_expected = len(X_df.columns)

        test_start_ts = pd.Timestamp(test_start)
        train_delta = pd.DateOffset(years=train_years)

        test_dates = common[common >= test_start_ts]
        if len(test_dates) == 0:
            return {}

        y_true_list: list[float] = []
        y_score_list: list[float] = []

        for t in test_dates:
            train_end = t
            train_start = t - train_delta
            train_idx = common[(common >= train_start) & (common < train_end)]

            if len(train_idx) < 60:
                continue

            X_train = X_df.loc[train_idx].values.astype(float)
            y_train = y_s.loc[train_idx].values.astype(float)

            if y_train.sum() < 3:
                continue

            X_scaled, mean_t, std_t = _standardise(X_train)
            w_t = _logistic_train(X_scaled, y_train)

            x_t = X_df.loc[[t]].values.astype(float)
            std_safe = np.where(std_t == 0, 1.0, std_t)
            x_t_scaled = (x_t - mean_t) / std_safe
            xb = np.hstack([np.ones((1, 1), dtype=float), x_t_scaled])
            p_t = float(_sigmoid(xb @ w_t)[0])

            y_true_list.append(float(y_s.loc[t]))
            y_score_list.append(p_t)

        if len(y_true_list) == 0:
            return {}

        y_true = np.array(y_true_list, dtype=float)
        y_score = np.array(y_score_list, dtype=float)

        y_pred = (y_score >= PROB_THRESHOLD).astype(float)

        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        tn = float(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        hit_rate  = (tp + tn) / len(y_true) if len(y_true) > 0 else float("nan")
        fp_rate   = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        n_signals = int(y_pred.sum())

        auc = _auc_rank_sum(y_true, y_score)

        return {
            "precision":   precision,
            "recall":      recall,
            "hit_rate":    hit_rate,
            "fp_rate":     fp_rate,
            "n_signals":   n_signals,
            "n_test_obs":  len(y_true_list),
            "auc":         auc,
        }

    def all_backtests(self) -> dict[int, dict]:
        """
        Run rolling_backtest for all HORIZONS.

        Returns
        -------
        {horizon: backtest_result_dict, ...}
        """
        return {h: self.rolling_backtest(h) for h in HORIZONS}
