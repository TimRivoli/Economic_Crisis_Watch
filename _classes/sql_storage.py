"""
SQLStorage: SQLAlchemy-backed persistence for FRED time-series data.

When the [Database] section of config.ini is populated and usesqldriver=True,
FREDDownloader writes here instead of CSV and DataLoader reads from here.

Table layout (long format — one row per series × date):
    fred_series_data     (series_id, date PK) → value
    fred_series_metadata (series_id PK)       → name, category, row_count, last_updated
"""

from __future__ import annotations

import configparser
import datetime
import os
from urllib.parse import quote_plus

import pandas as pd
import pyodbc
from sqlalchemy import (
    Column, DateTime, Date, Float, Integer, MetaData, String,
    Table, create_engine, inspect, text,
)
from sqlalchemy.exc import SQLAlchemyError

from .constants import PATHS
from .series_registry import REGISTRY

# ── Table definitions ──────────────────────────────────────────────────────

_metadata = MetaData()

_DATA_TABLE = Table(
    "fred_series_data", _metadata,
    Column("series_id", String(50),  nullable=False, primary_key=True),
    Column("date",      Date,         nullable=False, primary_key=True),
    Column("value",     Float,        nullable=True),
)

_META_TABLE = Table(
    "fred_series_metadata", _metadata,
    Column("series_id",    String(50),  nullable=False, primary_key=True),
    Column("name",         String(200), nullable=True),
    Column("category",     String(50),  nullable=True),
    Column("source",       String(100), nullable=True),
    Column("row_count",    Integer,     nullable=True),
    Column("last_updated", DateTime,    nullable=True),
)


class SQLStorage:
    """Read/write FRED series data via SQLAlchemy against SQL Server."""

    def __init__(self, server: str, database: str,
                 trusted: bool = True, username: str = "", password: str = ""):
        self.server   = server
        self.database = database
        self.engine   = self._build_engine(server, database, trusted, username, password)
        self._ensure_tables()

    # ── Construction ───────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str = PATHS.config) -> "SQLStorage | None":
        """
        Read config.ini and return a SQLStorage instance, or None when:
          - [Database] section is absent
          - usesqldriver is False or missing
          - databaseserver or databasename are empty
        """
        cfg = configparser.ConfigParser()
        cfg.read(config_path)

        if not cfg.has_section("Database"):
            return None
        if not cfg.getboolean("Database", "usesqldriver", fallback=False):
            return None

        server   = cfg.get("Database", "databaseserver",   fallback="").strip().strip("'\"")
        database = cfg.get("Database", "databasename",     fallback="").strip().strip("'\"")
        if not server or not database:
            return None

        trusted  = cfg.getboolean("Database", "usetrustedconnection", fallback=True)
        username = cfg.get("Database", "databaseusername", fallback="").strip().strip("'\"")
        password = cfg.get("Database", "databasepassword", fallback="").strip().strip("'\"")

        try:
            return cls(server, database, trusted, username, password)
        except SQLAlchemyError as exc:
            print(f"[SQLStorage] Connection failed — falling back to CSV: {exc}")
            return None

    @staticmethod
    def _build_engine(server, database, trusted, username, password):
        # Use the highest-versioned SQL Server ODBC driver available
        drivers = sorted(d for d in pyodbc.drivers() if "SQL Server" in d)
        driver  = drivers[-1] if drivers else "ODBC Driver 17 for SQL Server"

        if trusted:
            odbc = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"Trusted_Connection=yes;"
            )
        else:
            odbc = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password};"
            )

        url = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"
        return create_engine(url, fast_executemany=True)

    def _ensure_tables(self):
        """Create fred_series_data and fred_series_metadata if they don't exist."""
        inspector = inspect(self.engine)
        existing  = set(inspector.get_table_names())

        with self.engine.begin() as conn:
            if "fred_series_data" not in existing:
                _DATA_TABLE.create(conn)
            if "fred_series_metadata" not in existing:
                _META_TABLE.create(conn)

    # ── Write ──────────────────────────────────────────────────────────────

    def write_series(self, series_id: str, df: pd.DataFrame) -> int:
        """
        Replace all stored rows for series_id with the contents of df
        (a date-indexed single-column DataFrame) and refresh the metadata row.
        Returns the number of rows written.
        """
        if df is None or df.empty:
            return 0

        # Deduplicate by date — keep last value if the source has duplicates
        # (e.g. daily data with two readings in the same month after resampling)
        df = df[~df.index.duplicated(keep="last")]

        records = [
            {"series_id": series_id,
             "date":      idx.date() if hasattr(idx, "date") else idx,
             "value":     None if pd.isna(row.iloc[0]) else float(row.iloc[0])}
            for idx, row in df.iterrows()
        ]

        meta_row = {
            "series_id":    series_id,
            "name":         REGISTRY.get(series_id, {}).get("name", series_id),
            "category":     REGISTRY.get(series_id, {}).get("category", ""),
            "source":       REGISTRY.get(series_id, {}).get("filename", "fredapi"),
            "row_count":    len(records),
            "last_updated": datetime.datetime.now(),
        }

        with self.engine.begin() as conn:
            conn.execute(_DATA_TABLE.delete().where(_DATA_TABLE.c.series_id == series_id))
            if records:
                conn.execute(_DATA_TABLE.insert(), records)

            conn.execute(_META_TABLE.delete().where(_META_TABLE.c.series_id == series_id))
            conn.execute(_META_TABLE.insert(), [meta_row])

        return len(records)

    # ── Read ───────────────────────────────────────────────────────────────

    def read_series(self, series_id: str) -> pd.DataFrame | None:
        """
        Return a date-indexed DataFrame with a single column named series_id,
        or None if the series is not stored.
        """
        query = (
            _DATA_TABLE.select()
            .where(_DATA_TABLE.c.series_id == series_id)
            .order_by(_DATA_TABLE.c.date)
        )
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query)
                rows   = result.fetchall()

            if not rows:
                return None

            df = pd.DataFrame(rows, columns=["series_id", "date", "value"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["value"]]
            df.columns = [series_id]
            return df

        except SQLAlchemyError:
            return None

    def list_series(self) -> list[str]:
        """Return all series_ids currently stored in the metadata table."""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    _META_TABLE.select().order_by(_META_TABLE.c.series_id)
                ).fetchall()
            return [r[0] for r in rows]
        except SQLAlchemyError:
            return []

    def last_updated(self) -> str:
        """Formatted timestamp of the most recent write across all series."""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text("SELECT MAX(last_updated) FROM fred_series_metadata")
                ).fetchone()
            if row and row[0]:
                ts = row[0]
                if isinstance(ts, str):
                    ts = datetime.datetime.fromisoformat(ts)
                return ts.strftime("%B %d, %Y %H:%M")
        except SQLAlchemyError:
            pass
        return "Unknown"

    def connection_info(self) -> str:
        return f"{self.server} / {self.database}"
