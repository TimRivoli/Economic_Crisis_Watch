import configparser
import os
import sys
import time
import requests
import pandas as pd
from io import BytesIO

sys.path.insert(0, os.path.dirname(__file__))
from _classes.constants import PATHS, URLS, API
from _classes.sql_storage import SQLStorage

_HEADERS = {"User-Agent": API.user_agent}

# Series to download: (series_id, filename, description)
SERIES = [
    # CPI & PCE
    ("CPIAUCSL",         "cpi_all_items.csv",            "CPI All Urban Consumers: All Items"),
    ("CPILFESL",         "cpi_core.csv",                 "CPI All Urban Consumers: Less Food & Energy"),
    ("PCEPILFE",         "pce_core.csv",                 "PCE Price Index Excluding Food & Energy (Core PCE — Fed's preferred measure)"),
    # Money Supply
    ("M1SL",             "money_supply_m1.csv",          "M1 Money Stock"),
    ("M2SL",             "money_supply_m2.csv",          "M2 Money Stock"),
    ("M2REAL",           "money_supply_m2_real.csv",     "Real M2 Money Stock"),
    # Volatility & Interest Rates
    ("VIXCLS",           "vix_volatility.csv",           "CBOE Volatility Index (VIX)"),
    ("DGS10",            "treasury_yield_10y_daily.csv", "10-Year Treasury Constant Maturity Rate (Daily)"),
    ("GS10",             "treasury_yield_10y_monthly.csv","Market Yield on 10-Year Treasury Securities (Monthly)"),
    # Real Estate / Lending Standards
    ("SUBLPDRCSN",          "cre_lending_standards.csv",    "Net % Domestic Banks Tightening Standards for CRE Loans — Nonfarm Nonresidential (SLOOS)"),
    # Fiscal
    ("GFDEGDQ188S",      "federal_debt_pct_gdp.csv",     "Federal Debt: Total Public Debt as Percent of GDP"),
    ("GFDEBTN",          "federal_debt_total.csv",       "Federal Debt: Total Public Debt (Billions of Dollars, Monthly)"),
    # Employment / Labor
    ("UNRATE",           "unemployment_rate.csv",        "Unemployment Rate"),
    ("ICSA",             "initial_jobless_claims.csv",   "Initial Claims for Unemployment Insurance (Weekly)"),
    ("PSAVERT",          "personal_savings_rate.csv",    "Personal Saving Rate (%)"),
    ("INDPRO",           "industrial_production.csv",    "Industrial Production Index"),
    ("EMRATIO",          "employment_population_ratio.csv", "Employment-Population Ratio"),
    ("CIVPART",          "labor_force_participation.csv","Civilian Labor Force Participation Rate"),
    ("PAYEMS",           "nonfarm_payrolls.csv",         "All Employees: Total Nonfarm Payrolls"),
    # ── Liquidity & Funding Stress ────────────────────────────────────────
    ("STLFSI4",          "stlfsi4.csv",                  "St. Louis Fed Financial Stress Index"),
    ("DCPF3M",           "cp_3m_financial.csv",          "3-Month AA Financial Commercial Paper Rate (Discount Basis)"),
    ("DGS3MO",           "treasury_3m.csv",              "3-Month Treasury Constant Maturity Rate"),
    ("SOFR",             "sofr_rate.csv",                "Secured Overnight Financing Rate"),
    ("RRPONTSYD",        "rrp_usage.csv",                "Overnight Reverse Repurchase Agreements: Treasury Securities Sold by Fed"),
    # ── Credit Markets ────────────────────────────────────────────────────
    ("BAMLH0A0HYM2",     "hy_spread.csv",                "ICE BofA US High Yield Index Option-Adjusted Spread"),
    ("BAMLC0A0CM",       "ig_spread.csv",                "ICE BofA US Corporate Master Option-Adjusted Spread (IG)"),
    ("DRCCLACBS",        "cc_delinquency_rate.csv",      "Delinquency Rate on Credit Card Loans, All Commercial Banks"),
    # ── Policy Constraints ────────────────────────────────────────────────
    ("WALCL",            "fed_balance_sheet.csv",        "Federal Reserve: Total Assets (Balance Sheet)"),
    ("FEDFUNDS",         "fed_funds_rate.csv",           "Federal Funds Effective Rate"),
    ("T5YIE",            "inflation_exp_5y.csv",         "5-Year Breakeven Inflation Rate"),
    ("T10YIE",           "inflation_exp_10y.csv",        "10-Year Breakeven Inflation Rate"),
    # ── Fiscal Maneuverability ────────────────────────────────────────────
    ("A091RC1Q027SBEA",  "federal_interest_payments.csv","Federal Government: Interest Payments (Quarterly SAAR)"),
    ("W006RC1Q027SBEA",  "federal_current_receipts.csv", "Federal Government: Current Receipts (Quarterly SAAR)"),
    ("GDP",              "nominal_gdp.csv",              "Gross Domestic Product, Nominal (Quarterly SAAR)"),
    # ── Financial Conditions & Regime ─────────────────────────────────────
    ("NFCI",             "nfci.csv",                     "Chicago Fed National Financial Conditions Index"),
    ("NFCICREDIT",       "nfci_credit.csv",              "Chicago Fed NFCI: Credit Subindex"),
    # ── Recession Reference ───────────────────────────────────────────────
    ("USREC",            "us_recession.csv",             "NBER Recession Indicator (1=Recession, 0=Expansion)"),
    # ── Yield Curve ───────────────────────────────────────────────────────
    ("T10Y2Y",           "yield_curve_spread.csv",       "10-Year minus 2-Year Treasury Yield Spread"),
    # ── Bank Lending Standards ────────────────────────────────────────────
    ("DRTSCILM",         "bank_lending_standards.csv",   "Net % Banks Tightening C&I Loan Standards (Large & Medium Firms)"),
    # ── Leading Indicators ────────────────────────────────────────────────────
    ("USSLIND",          "leading_economic_index.csv",   "Conference Board Leading Economic Index (MoM % Change)"),
    # ── Recession Probability ─────────────────────────────────────────────
    ("RECPROUSM156N",    "recession_probability.csv",    "NY Fed Recession Probability: 12-Month Ahead"),
    # ── Housing ───────────────────────────────────────────────────────────
    ("MORTGAGE30US",     "mortgage_rate_30y.csv",        "30-Year Fixed Rate Mortgage Average"),
    ("HOUST",            "housing_starts.csv",           "Housing Starts: Total New Privately Owned Units (Thousands, SAAR)"),
    ("CSUSHPINSA",       "case_shiller_hpi.csv",         "S&P/Case-Shiller U.S. National Home Price Index"),
    # ── Dollar ────────────────────────────────────────────────────────────
    ("DTWEXBGS",         "dollar_index.csv",             "Nominal Broad U.S. Dollar Index (Trade Weighted, Goods)"),
]


def load_api_key():
    cfg = configparser.ConfigParser()
    cfg.read(PATHS.config)
    return cfg["fred"]["api_key"].strip()


def fetch_series(series_id, api_key):
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
    }
    resp = requests.get(URLS.fred_api, params=params, timeout=API.timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    observations = data.get("observations", [])
    if not observations:
        return None
    df = pd.DataFrame(observations)[["date", "value"]]
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    # FRED uses "." for missing values
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna(subset=[series_id])
    df = df.set_index("date")
    return df


def fetch_shiller_data():
    """
    Download Robert Shiller's S&P 500 dataset from Yale and return two DataFrames:
      - pe_df:   trailing P/E ratio (Price / Earnings)
      - cape_df: Cyclically Adjusted P/E (10-year real earnings average)
    Source: http://www.econ.yale.edu/~shiller/data/ie_data.xls
    """
    resp = requests.get(URLS.shiller_xls, timeout=API.timeout_sec)
    resp.raise_for_status()

    raw = pd.read_excel(BytesIO(resp.content), sheet_name="Data", header=7, engine="xlrd")

    def parse_date(val):
        try:
            f = float(str(val).strip())
            year = int(f)
            month = round((f - year) * 100)
            return pd.Timestamp(year=year, month=max(1, min(12, month or 1)), day=1)
        except Exception:
            return pd.NaT

    dates = raw.iloc[:, 0].apply(parse_date)
    valid = dates.notna()

    price    = pd.to_numeric(raw["P"],    errors="coerce")
    earnings = pd.to_numeric(raw["E"],    errors="coerce")
    cape     = pd.to_numeric(raw["CAPE"], errors="coerce")

    pe = (price / earnings).replace([float("inf"), float("-inf")], pd.NA)

    idx = valid.values
    pe_df   = pd.DataFrame({"SP500_PE":   pe.values[idx]},   index=dates.values[idx]).dropna()
    cape_df = pd.DataFrame({"SP500_CAPE": cape.values[idx]},  index=dates.values[idx]).dropna()

    return pe_df, cape_df


def _multpl_table(url: str, col_name: str) -> pd.DataFrame:
    """
    Fetch a multpl.com monthly data table and return a clean DataFrame.
    Values may be prefixed with a trend arrow glyph — strip with regex.
    """
    from io import StringIO
    resp = requests.get(url, headers=_HEADERS, timeout=API.timeout_sec)
    resp.raise_for_status()
    raw = pd.read_html(StringIO(resp.text))[0]
    raw.columns = ["date", col_name]
    # Explicit format required: pandas 2.x format-inference breaks on this table.
    # multpl.com uses 'Apr 1, 2026' style throughout.
    raw["date"] = pd.to_datetime(raw["date"], format=API.multpl_date_format, errors="coerce")
    # Strip leading non-numeric characters (trend arrow glyphs on PE table)
    raw[col_name] = pd.to_numeric(
        raw[col_name].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce"
    )
    raw = raw.dropna().set_index("date").sort_index()
    raw.index = raw.index.to_period("M").to_timestamp()   # normalize to month-start
    return raw


def extend_shiller(pe_df: pd.DataFrame, cape_df: pd.DataFrame):
    """
    Extend Shiller CSVs with multpl.com data for any months after the Yale
    Excel cutoff.  Returns (pe_df, cape_df) with the gap filled.
    """
    for df, url, col in [
        (pe_df,   URLS.multpl_pe,   "SP500_PE"),
        (cape_df, URLS.multpl_cape, "SP500_CAPE"),
    ]:
        cutoff = df.index.max()
        recent = _multpl_table(url, col)
        new_rows = recent[recent.index > cutoff]
        if not new_rows.empty:
            merged = pd.concat([df, new_rows]).sort_index()
            if col == "SP500_PE":
                pe_df = merged
            else:
                cape_df = merged

    return pe_df, cape_df


def _save(sql, series_id: str, filename: str, df: pd.DataFrame):
    """Write df to SQL (preferred) and/or CSV."""
    if sql is not None:
        rows = sql.write_series(series_id, df)
        print(f"saved {rows} rows -> SQL [{series_id}]")
    else:
        os.makedirs(PATHS.data, exist_ok=True)
        df.to_csv(os.path.join(PATHS.data, filename))
        print(f"saved {len(df)} rows -> {filename}")


def main():
    os.makedirs(PATHS.data, exist_ok=True)
    api_key = load_api_key()

    sql = SQLStorage.from_config()
    if sql:
        print(f"SQL mode: {sql.connection_info()}\n")
    else:
        print(f"CSV mode: {PATHS.data}\n")

    for series_id, filename, description in SERIES:
        print(f"Downloading {series_id}: {description} ...", end=" ", flush=True)
        try:
            df = fetch_series(series_id, api_key)
            if df is None or df.empty:
                print("no data returned, skipping.")
                continue
            _save(sql, series_id, filename, df)
        except requests.HTTPError as e:
            print(f"HTTP error: {e}")
        except Exception as e:
            print(f"error: {e}")
        time.sleep(API.rate_limit_sec)

    # ── Shiller data (not on FRED; sourced directly from Yale) ───────────
    print("Downloading Shiller data (S&P 500 P/E & CAPE) from Yale ...", end=" ", flush=True)
    try:
        pe_df, cape_df = fetch_shiller_data()
        print(f"Yale cutoff: {cape_df.index.max().date()}", end="  ")

        print("extending with multpl.com ...", end=" ", flush=True)
        pe_df, cape_df = extend_shiller(pe_df, cape_df)

        for df, sid, fname in [
            (pe_df,   "SP500_PE",   "sp500_pe_ratio.csv"),
            (cape_df, "SP500_CAPE", "sp500_cape.csv"),
        ]:
            if df is not None and not df.empty:
                _save(sql, sid, fname, df)
                print(f"  (through {df.index.max().date()})", end="  ")
        print()
    except Exception as e:
        print(f"error: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
