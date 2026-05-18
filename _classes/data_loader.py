"""
DataLoader: scans the /data directory at startup, maps CSV files to FRED series IDs,
loads data on demand, and caches in memory.  Unknown files (not in the registry) are
loaded with a best-effort series ID derived from their filename.

When a SQLStorage instance is supplied, SQL takes precedence over CSV for both
discovery and loading.
"""

from __future__ import annotations

import os
import pandas as pd
import datetime
from .series_registry import REGISTRY
from .constants import PATHS

# Reverse map: filename -> series_id (built once from registry)
_FILENAME_TO_ID = {meta["filename"]: sid for sid, meta in REGISTRY.items()}

_SQL_SENTINEL = "sql"   # value stored in self.available when source is SQL


class DataLoader:
    def __init__(self, data_dir: str = PATHS.data, sql=None):
        self.data_dir = data_dir
        self._sql = sql           # SQLStorage | None
        self._cache: dict[str, pd.DataFrame] = {}
        self.available: dict[str, str] = {}   # series_id -> file path or _SQL_SENTINEL
        self._scan()

    # ── Discovery ──────────────────────────────────────────────────────────

    def _scan(self):
        """Identify available series from SQL (preferred) or CSV files.

        In SQL mode, CSV files in the data directory act as a fallback for any
        series not yet present in the database (e.g. newly added series).
        """
        if self._sql is not None:
            for sid in self._sql.list_series():
                self.available[sid] = _SQL_SENTINEL

        if not os.path.isdir(self.data_dir):
            return
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".csv"):
                continue
            path = os.path.join(self.data_dir, fname)
            if fname in _FILENAME_TO_ID:
                sid = _FILENAME_TO_ID[fname]
            else:
                sid = fname.replace(".csv", "").upper().replace("-", "_")
            if sid not in self.available:   # SQL entry takes precedence
                self.available[sid] = path

    # ── Loading & Caching ──────────────────────────────────────────────────

    def load(self, series_id: str) -> pd.DataFrame | None:
        """Return a single-column DataFrame indexed by date, or None if unavailable."""
        if series_id in self._cache:
            return self._cache[series_id]
        if series_id not in self.available:
            return None

        if self.available[series_id] == _SQL_SENTINEL:
            df = self._sql.read_series(series_id)
            if df is not None:
                self._cache[series_id] = df
            return df

        try:
            df = pd.read_csv(self.available[series_id], index_col=0, parse_dates=True)
            df = df.iloc[:, :1]               # keep first value column only
            df.columns = [series_id]
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df = df.dropna()
            self._cache[series_id] = df
            return df
        except Exception:
            return None

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load every available series and return as a dict."""
        return {sid: self.load(sid) for sid in self.available}

    # ── Derived Values ─────────────────────────────────────────────────────

    def get_latest(self, series_id: str) -> tuple[float | None, pd.Timestamp | None]:
        """Most recent non-null value and its date."""
        df = self.load(series_id)
        if df is None or df.empty:
            return None, None
        return float(df.iloc[-1, 0]), df.index[-1]

    def get_yoy(self, series_id: str) -> float | None:
        """Year-over-year % change at the most recent month-end observation."""
        df = self.load(series_id)
        if df is None or df.empty:
            return None
        try:
            monthly = df.resample("ME").last().dropna()
        except ValueError:
            monthly = df.resample("M").last().dropna()
        if len(monthly) < 13:
            return None
        latest = monthly.iloc[-1, 0]
        year_ago = monthly.iloc[-13, 0]
        if year_ago == 0 or pd.isna(year_ago):
            return None
        return (latest - year_ago) / abs(year_ago) * 100

    def get_mom(self, series_id: str) -> float | None:
        """Absolute month-over-month change at the most recent observation."""
        df = self.load(series_id)
        if df is None or df.empty:
            return None
        try:
            monthly = df.resample("ME").last().dropna()
        except ValueError:
            monthly = df.resample("M").last().dropna()
        if len(monthly) < 2:
            return None
        return float(monthly.iloc[-1, 0] - monthly.iloc[-2, 0])

    def get_trend(self, series_id: str, months: int = 6) -> str:
        """Directional trend over the last N months: 'rising', 'falling', 'stable'."""
        df = self.load(series_id)
        if df is None or df.empty:
            return "unknown"
        try:
            monthly = df.resample("ME").last().dropna()
        except ValueError:
            monthly = df.resample("M").last().dropna()
        if len(monthly) < months + 1:
            return "unknown"
        window = monthly.iloc[-(months + 1):]
        pct = (window.iloc[-1, 0] - window.iloc[0, 0]) / abs(window.iloc[0, 0]) * 100
        if pct > 3:
            return "rising"
        if pct < -3:
            return "falling"
        return "stable"

    # ── Metadata Helpers ───────────────────────────────────────────────────

    def series_name(self, series_id: str) -> str:
        if series_id in REGISTRY:
            return REGISTRY[series_id]["name"]
        return series_id.replace("_", " ").title()

    def last_updated(self) -> str:
        if self._sql is not None:
            return self._sql.last_updated()
        mtimes = []
        for path in self.available.values():
            try:
                mtimes.append(os.path.getmtime(path))
            except OSError:
                pass
        if not mtimes:
            return "Unknown"
        return datetime.datetime.fromtimestamp(max(mtimes)).strftime("%B %d, %Y %H:%M")
